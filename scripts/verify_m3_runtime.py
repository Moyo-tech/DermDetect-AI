"""Evaluator launcher: stage synthetic code only and verify the new CPU runtime.

No real data/predictions loaded. Each invocation uses a fresh evidence directory.
The M2 production seal/profile/runtime are never edited.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def protected_source_root():
    # Explicit local source routing; absent configuration fails closed.
    return Path(os.environ['DERMDETECT_PROTECTED_SOURCE_ROOT']).resolve(strict=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dermdetect.data.access_boundary import development_profile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+ '\n')


def inventory(roots):
    files = {}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_symlink(): files[str(path)] = {'symlink':os.readlink(path),'resolved':str(path.resolve(strict=True))}
            elif path.is_file(): files[str(path)] = {'sha256':sha(path),'bytes':path.stat().st_size}
    return files


def main(destination):
    destination.mkdir()
    bundle = destination / 'bundle'; output = destination / 'worker-output'
    bundle.mkdir(); output.mkdir()
    for relative in ['dermdetect/__init__.py','dermdetect/data/__init__.py','dermdetect/data/numerical_policy.py',
                     *[p.relative_to(ROOT/'src').as_posix() for p in sorted((ROOT/'src/dermdetect/models').glob('*.py'))]]:
        target = bundle/'src'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/'src'/relative,target)
    shutil.copyfile(ROOT/'tests/test_model_development.py',bundle/'test_model_development.py')
    # Probe code opens protected paths but never reads their contents, even on a failed denial.
    worker = '''import importlib.metadata, json, os, pathlib, socket, subprocess, sys, unittest
sys.path.insert(0,str(pathlib.Path(__file__).parent/'src'))
from PIL import Image
import numpy as np
checks={}
config=json.loads((pathlib.Path(__file__).parent/'probes.json').read_text())
for name,path in config['denied'].items():
    try: fd=os.open(path,os.O_RDONLY)
    except PermissionError: checks[name]=True
    except Exception: checks[name]=False
    else: os.close(fd); checks[name]=False
for name,item in config['loaders'].items():
    import torch
    try:
        if item['kind']=='pil': Image.open(item['path'])
        elif item['kind']=='numpy': np.load(item['path'],allow_pickle=False)
        else: torch.load(item['path'],map_location='cpu',weights_only=True)
    except PermissionError: checks[name]=True
    except Exception: checks[name]=False
    else: checks[name]=False
for name,path in config['metadata'].items():
    try: os.stat(path)
    except PermissionError: checks[name]=True
    except Exception: checks[name]=False
    else: checks[name]=False
for name,path in config['zip'].items():
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive: archive.infolist()
    except PermissionError: checks[name]=True
    except Exception: checks[name]=False
    else: checks[name]=False
try:
    with socket.socket() as connection: connection.connect(('127.0.0.1',9))
except PermissionError: checks['network_denied']=True
except Exception: checks['network_denied']=False
else: checks['network_denied']=False
try: subprocess.run(['/bin/echo','synthetic-helper'],check=True,capture_output=True)
except PermissionError: checks['helper_denied']=True
except Exception: checks['helper_denied']=False
else: checks['helper_denied']=False
try: fd=os.open(config['write_protected'],os.O_WRONLY|os.O_APPEND)
except PermissionError: checks['immutable_code_write_denied']=True
except Exception: checks['immutable_code_write_denied']=False
else: os.close(fd); checks['immutable_code_write_denied']=False
if not all(checks.values()): raise RuntimeError('Boundary failures: '+repr([k for k,v in checks.items() if not v]))
# Exercise the image-loader route using an invented PNG only.
image=pathlib.Path(config['output'])/'invented.png'; Image.new('RGB',(5,7),(17,93,201)).save(image)
with Image.open(image) as img: assert np.asarray(img).shape==(7,5,3)
checks['synthetic_image_loader_allowed']=True
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import test_model_development
suite=unittest.defaultTestLoader.loadTestsFromModule(test_model_development)
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped: raise RuntimeError('Synthetic model suite failed or skipped')
record={'status':'synthetic_cpu_runtime_verified','boundary_checks':checks,'tests_run':result.testsRun,
        'packages':{p:importlib.metadata.version(p) for p in ('torch','torchvision','numpy','scipy','scikit-learn','Pillow','transformers','huggingface-hub','safetensors')},
        'device':'cpu','real_inputs_opened':False,'model_weights_downloaded':False,'held_out_outcomes_accessed':False}
(pathlib.Path(config['output'])/'verification.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\\n')
print(json.dumps({'status':record['status'],'tests':result.testsRun,'boundary_checks':len(checks)}))
'''
    (bundle/'worker.py').write_text(worker)
    # Select exact Python/native runtime roots; do not allow the whole Homebrew tree.
    runtime_env = ROOT/'env/m3-model-dev'
    executable = runtime_env/'bin/python'
    resolved = executable.resolve(strict=True)
    base = Path(sys.base_prefix).resolve(strict=True)
    python_root = base.parents[3]
    roots = {runtime_env,python_root}
    todo = [p for root in roots for p in root.rglob('*') if p.is_file() and (p.suffix in ('.so','.dylib') or p.resolve()==resolved)]
    visited = set(); linked = set(); install_names = set()
    while todo:
        chunk = [p for p in todo if p not in visited]; todo=[]
        if not chunk: break
        visited.update(chunk)
        for start in range(0,len(chunk),80):
            result=subprocess.run(['/usr/bin/otool','-L',*[str(p) for p in chunk[start:start+80]]],check=True,capture_output=True,text=True)
            current_binary = None; first_entry = False
            for line in result.stdout.splitlines():
                if not line.startswith('\t'):
                    if line.endswith(':'): current_binary = Path(line[:-1]); first_entry = True
                    continue
                name=line.strip().split(' (',1)[0]
                if first_entry and current_binary and current_binary.suffix == '.dylib' and Path(name).name == current_binary.name:
                    install_names.add(name); first_entry = False; continue
                first_entry = False; linked.add(name)
                if name.startswith('/opt/homebrew/'):
                    dependency=Path(name).resolve(strict=True)
                    if not any(dependency.is_relative_to(root) for root in roots): roots.add(dependency.parent)
                    if dependency not in visited: todo.append(dependency)
    protected = [ROOT/'outputs',ROOT/'configs',ROOT/'docs',ROOT/'src',ROOT/'scripts',ROOT/'tests',ROOT/'.git',
                 ROOT/'env/m2-integrity',ROOT/'env/m2-production',protected_source_root()]
    # destination is under outputs; allow synthetic bundle/output and deny other evaluator evidence.
    protected = [p for p in protected if p != ROOT/'outputs']
    protected.extend(p for p in (ROOT/'outputs').iterdir() if p != destination)
    denied = {}; zip_routes={}; metadata={}; aliases=[]
    # Probe all existing evaluator files directly and through output symlinks.
    evaluator = sorted(p for root in protected if root.is_dir() and root != protected_source_root()
                       for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    # Exclude M2 exported development payload enumeration: its tree is denied as a whole,
    # and a representative route is enough here; all evaluator/audit outcomes are exhaustive.
    evaluator = [p for p in evaluator if not p.is_relative_to(ROOT/'env/m2-production') and not p.is_relative_to(ROOT/'env/m2-integrity')]
    config=json.loads((ROOT/'configs/local/ham_archive_paths.json').read_text())
    sources=[Path(config['metadata_path']),*[Path(p) for p in config['image_archives'].values()]]
    for i,path in enumerate([*sources,*evaluator,ROOT/'env/m2-production/inputs/bundle_manifest.json']):
        denied[f'direct_{i}']=str(path)
        alias=output/f'protected_alias_{i}';alias.symlink_to(path);aliases.append(alias)
        denied[f'alias_{i}']=str(alias)
    for i,path in enumerate(sources[1:]):
        zip_routes[f'zip_direct_{i}']=str(path)
        alias=output/f'zip_alias_{i}';alias.symlink_to(path);aliases.append(alias)
        zip_routes[f'zip_alias_{i}']=str(alias)
    for i,root in enumerate(protected): metadata[f'protected_root_{i}']=str(root)
    # Future outcomes are represented by invented sentinels, never actual outcomes.
    for name in ('future_test_image','future_test_label','future_ddi_outcome'):
        sentinel=destination/(name+'.txt');sentinel.write_text('invented sealed sentinel\n')
        denied[name]=str(sentinel);protected.append(sentinel)
        alias=output/(name+'_alias');alias.symlink_to(sentinel);aliases.append(alias)
        denied[name+'_alias']=str(alias)
    import numpy as np
    import torch
    from PIL import Image
    loaders={}
    for kind,suffix in [('pil','.png'),('numpy','.npy'),('torch','.pt')]:
        fixture=destination/('invented_protected_'+kind+suffix)
        if kind=='pil': Image.new('RGB',(3,5),(1,2,3)).save(fixture)
        elif kind=='numpy': np.save(fixture,np.arange(4),allow_pickle=False)
        else: torch.save(torch.arange(4),fixture)
        protected.append(fixture)
        loaders[kind+'_loader_denied']={'kind':kind,'path':str(fixture)}
        alias=output/('loader_alias_'+kind+suffix);alias.symlink_to(fixture);aliases.append(alias)
        loaders[kind+'_alias_loader_denied']={'kind':kind,'path':str(alias)}
    write(bundle/'probes.json',{'denied':denied,'zip':zip_routes,'metadata':metadata,'loaders':loaders,
          'write_protected':str(bundle/'worker.py'),'output':str(output)})
    profile=development_profile(bundle=bundle,output=output,runtime_roots=sorted(roots),
        interpreter=resolved,protected_roots=protected)
    (destination/'boundary.sb').write_text(profile)
    write(destination/'runtime_inventory.json',{'runtime_roots':[str(p) for p in sorted(roots)],
        'files':inventory(roots),'linked_libraries':sorted(linked),'dylib_install_names':sorted(install_names),'python':sys.version,
        'macos_build':subprocess.run(['/usr/bin/sw_vers'],capture_output=True,text=True,check=True).stdout})
    code={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()}
    write(destination/'code_inventory.json',code)
    command=['/usr/bin/sandbox-exec','-f',str(destination/'boundary.sb'),str(executable),'-B','-I',str(bundle/'worker.py')]
    env={'PATH':'/usr/bin:/bin','LC_ALL':'C','TMPDIR':str(output),'HOME':str(output),
        'HF_HOME':str(output/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result=subprocess.run(command,cwd=bundle,env=env,stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True)
    (destination/'worker.stdout.txt').write_text(result.stdout);(destination/'worker.stderr.txt').write_text(result.stderr)
    for alias in aliases: alias.unlink()
    runtime_after=inventory(roots)
    runtime_before=json.loads((destination/'runtime_inventory.json').read_text())['files']
    record={'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'command':command,'returncode':result.returncode,'runtime_unchanged':runtime_before==runtime_after,
        'code_unchanged':code=={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()},
        'profile_sha256':sha(destination/'boundary.sb'),'runtime_inventory_sha256':sha(destination/'runtime_inventory.json'),
        'code_inventory_sha256':sha(destination/'code_inventory.json'),'inherited_source_descriptors':0,
        'scope':'fresh CPU process; synthetic inputs only; no production model/accelerator verification',
        'limits':'Process boundary can be bypassed by machine owner outside launcher; OS shared-cache libraries bound by build, not full hashes. CPU tests do not verify accelerator/provider isolation.'}
    write(destination/'run_record.json',record)
    print(json.dumps(record,sort_keys=True))
    if result.returncode or not record['runtime_unchanged'] or not record['code_unchanged']: raise SystemExit(2)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/m3-runtime-verification')
    args=parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT/'env/m3-model-dev').resolve():
        raise SystemExit('Use the separate M3 environment')
    main(args.output.resolve())
