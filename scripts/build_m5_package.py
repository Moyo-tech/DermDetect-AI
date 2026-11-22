#!/usr/bin/env python3
"""Stage an exact synthetic-checkable CPU package; never open scientific inputs."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
FILES=['src/dermdetect/__init__.py','src/dermdetect/models/__init__.py',
       'src/dermdetect/models/m5_run.py','src/dermdetect/models/adaptation.py',
       'src/dermdetect/models/densenet.py','src/dermdetect/models/embeddings.py',
       'src/dermdetect/data/__init__.py','src/dermdetect/data/manifests.py',
       'src/dermdetect/data/schemas.py','src/dermdetect/data/numerical_policy.py','scripts/m5_cpu_worker.py',
       'scripts/supervise_m5_local.py','scripts/prepare_m5_inputs.py','tests/test_m5_run.py','tests/test_model_development.py']
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    B=a.output.resolve()/'bundle';B.mkdir(parents=True,exist_ok=False); hashes={}
    for n in FILES:
        target=B/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
        hashes[n]=hashlib.sha256(target.read_bytes()).hexdigest()
    entry="""from pathlib import Path
import importlib.util,sys,unittest
B=Path(__file__).parent;sys.path.insert(0,str(B/'src'))
for name in ['m5_cpu_worker','supervise_m5_local','prepare_m5_inputs']:
 p=B/'scripts'/(name+'.py');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
suite=unittest.TestSuite()
for name in ['test_m5_run.py','test_model_development.py']:suite.addTests(unittest.defaultTestLoader.discover(str(B/'tests'),pattern=name))
r=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not r.wasSuccessful())
"""
    (B/'synthetic_entry.py').write_text(entry);hashes['synthetic_entry.py']=hashlib.sha256((B/'synthetic_entry.py').read_bytes()).hexdigest()
    (a.output/'source_manifest.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    W=a.output.resolve()/'worker-output';W.mkdir();old=ROOT/'outputs/m3-runtime-verification-final'
    # No new scientific allowlist paths are added to this synthetic package.
    profile=(old/'boundary.sb').read_text().replace(str(old/'bundle'),str(B)).replace(str(old/'worker-output'),str(W))
    (a.output/'boundary.sb').write_text(profile)
    probe="""from pathlib import Path
import errno,json,os,socket,subprocess,sys
B=Path(__file__).parent;O=Path(os.environ['TMPDIR']);results={}
def denied(name,call):
 try:
  v=call()
  if isinstance(v,int):os.close(v)
 except OSError as e:
  assert e.errno in (errno.EPERM,errno.EACCES),(name,e.errno);results[name]=True;return
 raise AssertionError('Boundary unexpectedly allowed '+name)
denied('protected_metadata',lambda:os.open(PROTECTED,os.O_RDONLY))
denied('staged_source_write',lambda:os.open(str(B/'src/dermdetect/models/m5_run.py'),os.O_WRONLY))
denied('runtime_write',lambda:os.open(str(Path(sys.prefix)/'.m5_write_probe'),os.O_WRONLY|os.O_CREAT,0o600))
s=socket.socket();denied('network',lambda:s.connect(('127.0.0.1',9)));s.close()
denied('helper',lambda:subprocess.run(['/usr/bin/true'],check=True))
alias=O/'protected_alias';alias.symlink_to(PROTECTED);denied('output_symlink_alias',lambda:os.open(str(alias),os.O_RDONLY))
(O/'allowed_probe.txt').write_text('invented allowed output')
assert (O/'allowed_probe.txt').read_text()=='invented allowed output';results['allowed_output']=True
print(json.dumps(results))
""".replace('PROTECTED',repr(str(ROOT/'outputs/m2-production-closeout/production_seal.json')))
    (B/'boundary_probe.py').write_text(probe);hashes['boundary_probe.py']=hashlib.sha256((B/'boundary_probe.py').read_bytes()).hexdigest()
    (a.output/'source_manifest.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'bundle':str(B),'files':len(hashes),'scientific_inputs_staged':False}))
if __name__=='__main__':main()
