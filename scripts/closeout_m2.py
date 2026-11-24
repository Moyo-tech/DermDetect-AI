"""Evaluator-only M2 preparation, enforced CPU verification and conditional seal.

No model, prediction, training or held-out outcome operations. Every phase writes
fresh evidence; the original proposal and submitted review exports stay intact.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile


def protected_source_root():
    # Explicit local source routing; absent configuration fails closed.
    return Path(os.environ['DERMDETECT_PROTECTED_SOURCE_ROOT']).resolve(strict=True)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dermdetect.data.access_boundary import development_profile
from dermdetect.data.development_bundle import BundleApproval, bundle_input_hashes, verify_staged_bundle
from dermdetect.data.manifests import canonical_json_bytes, file_sha256
from dermdetect.data.proposal_io import load_proposed_split
from dermdetect.data.production_seal import assemble_seal, REQUIRED_BINDINGS
from dermdetect.data.streaming_bundle import stage_streamed_bundle

EVIDENCE=ROOT/'outputs/m2-production-closeout'
PROPOSAL=ROOT/'outputs/m2-proposed-constrained-split'
PROPOSAL_HASH='7c9abfe0f0cd9b2657b37a79c913a9366b1d669a232f9e21ed3b39f40e905726'
PACKAGE=ROOT/'docs/m2_final_approval_package.md'
PACKAGE_HASH='a324416bf591769b97c5c3a9f360ca4afde8be55014b33bb639e7ac99d075ad2'
DEVELOPMENT=ROOT/'env/m2-production'
BUNDLE=DEVELOPMENT/'inputs'
OUTPUT=DEVELOPMENT/'worker-output'


def read(path):return json.loads(path.read_text())
def write(path,value):
    with path.open('xb') as handle:handle.write(canonical_json_bytes(value)+b'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def verify_approval():
    approval=read(EVIDENCE/'approval.json')
    if (file_sha256(PACKAGE)!=PACKAGE_HASH or file_sha256(PROPOSAL/'proposal_report.json')!=PROPOSAL_HASH
            or approval.get('approved_package_sha256')!=PACKAGE_HASH or approval.get('proposal_report_sha256')!=PROPOSAL_HASH):
        raise ValueError('Approved package/proposal bytes changed')
    for flag in ('accept_exact_assignment','approve_numerical_policy','authorize_train_validation_export',
                 'authorize_cpu_boundary_verification','authorize_freeze_only_after_all_checks_pass'):
        if approval.get(flag) is not True:raise ValueError('Missing explicit approval: '+flag)
    return approval


def code_inventory():
    development={'worker.py':ROOT/'scripts/m2_development_worker.py',
                 'numerical_policy.py':ROOT/'src/dermdetect/data/numerical_policy.py'}
    evaluator=[ROOT/'scripts/closeout_m2.py',*sorted((ROOT/'src/dermdetect/data').glob('*.py'))]
    # This inspection forbids third-party/model/decoder imports and embedded source routes.
    import ast
    allowed={'argparse','collections','dataclasses','fractions','hashlib','json','math','os','pathlib',
             'random','re','socket','subprocess','sys','zipfile','numerical_policy'}
    for path in development.values():
        source=path.read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            names=([alias.name.split('.')[0] for alias in node.names] if isinstance(node,ast.Import)
                   else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            if any(name not in allowed for name in names):raise ValueError('Uninspected development dependency')
        if any(token in source for token in ('/Users/','dataverse_files','ISIC_00','HAM_000','torch','PIL.')):
            raise ValueError('Embedded protected source identifiers or model/decoder code')
    manifest={'development':{name:file_sha256(path) for name,path in development.items()},
        'evaluator':{path.relative_to(ROOT).as_posix():file_sha256(path) for path in evaluator},
        'inspection':'Agent-inspected after user scope approval; no claim the user personally inspected these hashes.',
        'development_third_party_dependencies':[],'cpu_only':True}
    return development,manifest


def runtime_inventory():
    interpreter=Path(sys.executable).resolve(strict=True)
    base=Path(sys.base_prefix).resolve(strict=True)
    python_root=base.parents[3]
    roots={python_root}
    native=[p for p in python_root.rglob('*') if p.is_file() and (p.suffix in ('.so','.dylib') or p==interpreter or p.name=='Python')]
    linked=set();todo=list(native);visited=set()
    while todo:
        chunk=[p for p in todo if p not in visited];todo=[]
        if not chunk:break
        visited.update(chunk)
        result=subprocess.run(['/usr/bin/otool','-L',*[str(p) for p in chunk]],check=True,capture_output=True,text=True)
        for line in result.stdout.splitlines():
            if not line.startswith('\t'):continue
            name=line.strip().split(' (',1)[0];linked.add(name)
            if name.startswith('/opt/homebrew/'):
                dependency=Path(name).resolve(strict=True)
                if not dependency.is_relative_to(python_root):roots.add(dependency.parent)
                if dependency not in visited:todo.append(dependency)
    entries={}
    for root in sorted(roots):
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                entries[str(path)]={'symlink_target':os.readlink(path),'resolved':str(path.resolve(strict=True))}
            elif path.is_file():entries[str(path)]={'sha256':file_sha256(path),'bytes':path.stat().st_size}
            elif not path.is_dir():raise ValueError('Special runtime file')
    platform=subprocess.run(['/usr/bin/sw_vers'],check=True,capture_output=True,text=True).stdout
    return {'interpreter':str(interpreter),'python_version':sys.version,'runtime_roots':[str(p) for p in sorted(roots)],
        'launch_flags':['-B','-I','-S'],'site_packages_enabled':False,'environment':'minimal fixed environment; source/model path variables excluded',
        'files':entries,'linked_libraries':sorted(linked),'macos_build':platform,
        'platform_runtime_roots':['/System/Library','/usr/lib','/usr/share','/Library/Apple','/System/Volumes/Preboot','/private/var/db/timezone'],
        'platform_limit':'System libraries may reside in the OS dyld shared cache; platform build and linked dependencies are recorded, not every OS file hash.',
        'new_dependency_installation':False}


def verify_runtime(inventory):
    if runtime_inventory()!=inventory:raise ValueError('Inspected runtime or OS build changed')


def preservation(grouping,partitions,report,*,verify_sources=True):
    accepted=read(ROOT/'outputs/m2-audit-acceptance/acceptance.json')
    for key,path in accepted['input_paths'].items():
        if file_sha256(ROOT/path)!=accepted['input_sha256'][key]:raise ValueError('Accepted review/input changed')
    original=ROOT/'outputs/m2-human-audit-import/m2-human-annotations.draft.json'
    if file_sha256(original)!='7bbd33b66c815cf8536274b4da3b03871dee4a04031dccf757b27fbe5f74b9ff':raise ValueError('Original354 export changed')
    for name,checksum in report['artifact_sha256'].items():
        if file_sha256(PROPOSAL/name)!=checksum:raise ValueError('Approved proposal payload changed')
    config=read(ROOT/'configs/local/ham_archive_paths.json')
    inventory=read(ROOT/'outputs/m2-ham-archive-inventory/integrity_report.json')
    source_hashes={config['metadata_path']:inventory['metadata_sha256'],
                  **{config['image_archives'][a['alias']]:a['sha256'] for a in inventory['archives']}}
    if verify_sources:
        for name,checksum in source_hashes.items():
            if file_sha256(Path(name))!=checksum:raise ValueError('Original HAM source changed')
    owners={r.original.image_id:p for p,members in partitions.items() for r in members}
    if not all(owners[u.left]==owners[u.right] for u in grouping.uncertainties):raise ValueError('Unclear link crosses split')
    return {'status':'accepted_sources_review_and_proposal_preserved','exact_approved_membership_replayed':True,
        'zero_overlap_dimensions':len(report['allocation_verification']['overlap_dimensions']),
        'all15_unclear_retained_and_colocated':len(grouping.uncertainties)==15,
        'original_source_hashes_unchanged':True,'raw_exports_unchanged':True,'source_sha256':source_hashes,
        'proposal_report_sha256':PROPOSAL_HASH,'source_labels_and_original_fields_unchanged':True}


def prepare():
    approval=verify_approval()
    grouping,partitions,report=load_proposed_split(PROPOSAL,expected_report_sha256=PROPOSAL_HASH)
    preserved=preservation(grouping,partitions,report)
    paths,code=code_inventory();runtime=runtime_inventory()
    if DEVELOPMENT.exists():raise ValueError('No overwrite of an existing production development tree')
    DEVELOPMENT.mkdir();OUTPUT.mkdir()
    write(EVIDENCE/'code_manifest.json',code);write(EVIDENCE/'runtime_inventory.json',runtime)
    write(EVIDENCE/'preservation_before_export.json',preserved)
    code_files={name:path.read_bytes() for name,path in paths.items()}
    protected_files=sorted(p for p in (ROOT/'outputs').rglob('*') if p.is_file() and not p.is_symlink())
    fingerprints={file_sha256(p) for p in protected_files}|set(preserved['source_sha256'].values())
    write(EVIDENCE/'protected_fingerprints.json',{'files':{p.relative_to(ROOT).as_posix():file_sha256(p) for p in protected_files},
        'complete_source_sha256':preserved['source_sha256']})
    roots=(ROOT/'outputs',ROOT/'configs/local',protected_source_root())
    hashes=bundle_input_hashes(records=grouping.records,partitions=partitions,grouping=grouping,
        code_files=code_files,protected_content_sha256=fingerprints)
    approvals=[BundleApproval(scope,checksum,'user scope approval; agent final code/composition inspection',
        'outputs/m2-production-closeout/approval.json and code_manifest.json; no fabricated personal code-review attestation',True)
        for scope,checksum in hashes.items()]
    write(EVIDENCE/'bundle_input_approvals.json',{'scope_authority_sha256':file_sha256(EVIDENCE/'approval.json'),
        'input_hashes':hashes,'reviewer_attribution':'User authorized approved specification; agent inspected and hash-bound final implementation'})
    config=read(ROOT/'configs/local/ham_archive_paths.json')
    archives={alias:zipfile.ZipFile(path) for alias,path in config['image_archives'].items()}
    opened=[]
    def open_payload(row):
        alias,member=row.original.path.split('/',1)
        if member!=row.original.image_id+'.jpg' or alias not in archives:raise ValueError('Noninventoried HAM member route')
        opened.append(row.original.image_id)
        return archives[alias].open(member)
    try:
        staged=stage_streamed_bundle(destination=BUNDLE,provenance_path=EVIDENCE/'evaluator_provenance.json',
            records=grouping.records,partitions=partitions,grouping=grouping,
            allocation_report=report['allocation_verification'],code_files=code_files,open_payload=open_payload,
            approvals=approvals,protected_roots=roots,protected_content_sha256=fingerprints,
            evidence_context='Final approved HAM-only source/audit/proposal; full provenance remains evaluator-only; source thread '+approval['source_thread_id'])
    finally:
        for archive in archives.values():archive.close()
    expected={r.original.image_id for p in ('train','validation') for r in partitions[p] if r.active}
    if set(opened)!=expected or len(opened)!=len(expected):raise ValueError('Unexpected payload reads')
    verified=verify_staged_bundle(destination=BUNDLE,expected_manifest_sha256=staged['bundle_manifest_sha256'])
    manifest=read(BUNDLE/'bundle_manifest.json')
    test_ids={r.original.image_id for r in partitions['internal_test']}
    test_hashes={r.original.sha256 for r in partitions['internal_test']}
    for p in ('train','validation'):
        rows=read(BUNDLE/'manifests'/f'{p}.json')
        expected_rows=[r for r in partitions[p] if r.active]
        if {r['image_id'] for r in rows}!={r.original.image_id for r in expected_rows}:raise ValueError('Composition membership changed')
        if any(r['image_id'] in test_ids or r['sha256'] in test_hashes for r in rows):raise ValueError('Protected test membership/payload in bundle')
        expected_map={r.original.image_id:r for r in expected_rows}
        for row in rows:
            anchor=expected_map[row['image_id']]
            if (row['dx'],row['analysis_lesion_id'],row['split_group_id'])!=(anchor.original.dx,anchor.analysis_lesion_id,anchor.split_group_id):
                raise ValueError('Redacted row changed label or identity')
    metadata=b''.join((BUNDLE/name).read_bytes() for name in manifest['files'] if not name.startswith('data/'))
    if any(identity.encode() in metadata for identity in test_ids):raise ValueError('Embedded future-test identifier in metadata/code')
    composition={'status':'production_composition_verified','active_development_payloads':len(opened),
        'no_protected_membership_or_payloads':True,'all_payload_sha256_passed':True,'code_semantic_inspection_passed':True,
        'bundle_manifest_sha256':staged['bundle_manifest_sha256'],'staging_verification':verified,
        'code_files':list(code_files),'inactive_copies_exported':0,'image_decode_or_model_run':False,
        'stream_chunk_bytes':staged['stream_chunk_bytes'],'production_seal_validated':False}
    write(EVIDENCE/'composition.json',composition)
    # Final profile denies the complete project evaluator/source trees; the exported
    # worker has no dependency on source modules or the Pillow virtual environment.
    protected=[ROOT/'outputs',ROOT/'configs',ROOT/'docs',ROOT/'src',ROOT/'scripts',ROOT/'tests',ROOT/'.git',
               ROOT/'env/m2-integrity',protected_source_root()]
    profile=development_profile(bundle=BUNDLE,output=OUTPUT,runtime_roots=[Path(p) for p in runtime['runtime_roots']],
        interpreter=Path(runtime['interpreter']),protected_roots=protected)
    (EVIDENCE/'boundary.sb').write_text(profile)
    write(EVIDENCE/'stage_record.json',{'status':'stage_ready_for_production_boundary','staged':staged,
        'profile_sha256':file_sha256(EVIDENCE/'boundary.sb'),'runtime_inventory_sha256':file_sha256(EVIDENCE/'runtime_inventory.json'),
        'code_manifest_sha256':file_sha256(EVIDENCE/'code_manifest.json'),'recorded_utc':now(),'production_freeze':False})
    print(json.dumps({'status':composition['status'],'active_payloads':len(opened),'runtime_files':len(runtime['files'])}))


def run_tests():
    record=EVIDENCE/'test_run.json'
    if record.exists():raise ValueError('No overwrite of test evidence')
    command=[str(ROOT/'env/m2-integrity/bin/python'),'-B','-m','unittest','discover','-s','tests','-v']
    started=now();result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (EVIDENCE/'tests.stdout.txt').write_text(result.stdout);(EVIDENCE/'tests.stderr.txt').write_text(result.stderr)
    match=re.search(r'Ran (\d+) tests',result.stderr)
    test={'command':command,'start_utc':started,'end_utc':now(),'returncode':result.returncode,
        'passed':int(match.group(1)) if match and result.returncode==0 and result.stderr.rstrip().endswith('OK') else 0,
        'input_scope':'invented fixtures only'}
    write(record,test)
    print(json.dumps(test))
    if result.returncode:raise ValueError('Required tests failed')


def boundary():
    verify_approval();stage=read(EVIDENCE/'stage_record.json');runtime=read(EVIDENCE/'runtime_inventory.json')
    verify_runtime(runtime);_,current_code=code_inventory()
    if current_code!=read(EVIDENCE/'code_manifest.json'):raise ValueError('Inspected code changed')
    verify_staged_bundle(destination=BUNDLE,expected_manifest_sha256=stage['staged']['bundle_manifest_sha256'])
    profile=EVIDENCE/'boundary.sb'
    if file_sha256(profile)!=stage['profile_sha256']:raise ValueError('Inspected profile changed')
    if (EVIDENCE/'boundary_run.json').exists():raise ValueError('Passing boundary evidence already exists')
    attempt=len(list(EVIDENCE.glob('boundary_attempt_*.json')))+1
    denied={};zip_denied={};meta={};aliases=[]
    config=read(ROOT/'configs/local/ham_archive_paths.json')
    source_paths=[Path(config['metadata_path']),*[Path(p) for p in config['image_archives'].values()]]
    # Every saved evaluator artifact existing at this checkpoint, including all
    # original/revised review artifacts and full-resolution views, is probed.
    evidence_files=sorted(p for p in (ROOT/'outputs').rglob('*') if p.is_file() and not p.is_symlink())
    for index,path in enumerate([*source_paths,*evidence_files,ROOT/'configs/local/ham_archive_paths.json']):
        denied[f'protected_direct_{index}']=str(path)
        alias=OUTPUT/f'protected_alias_{index}';alias.symlink_to(path);aliases.append(alias)
        denied[f'protected_alias_{index}']=str(alias)
    source_alias=OUTPUT/'source_alias';source_alias.symlink_to(source_paths[0]);aliases.append(source_alias)
    denied['source_alias_denied']=str(source_alias)
    test_alias=OUTPUT/'test_membership_alias';test_alias.symlink_to(PROPOSAL/'internal_test.csv');aliases.append(test_alias)
    denied['test_membership_alias_denied']=str(test_alias)
    for alias,path in config['image_archives'].items():
        zip_denied['source_zip_loader_'+alias]=path
        proxy=OUTPUT/('zip_alias_'+alias);proxy.symlink_to(path);aliases.append(proxy)
        zip_denied['source_zip_alias_loader_'+alias]=str(proxy)
    for name in ('future_outcome_sentinel','future_image_sentinel','future_test_label_sentinel'):
        sentinel=EVIDENCE/(name+'.txt')
        if not sentinel.exists():sentinel.write_text('INVENTED SENTINEL; no real held-out contents\n')
        denied[name+'_denied']=str(sentinel)
    for index,path in enumerate([ROOT/'outputs',ROOT/'configs',protected_source_root()]):
        meta[f'protected_root_metadata_{index}']=str(path)
    runtime_file=next(Path(p) for p,value in runtime['files'].items() if 'sha256' in value and p.endswith('/json/__init__.py'))
    probe={'denied_open':denied,'denied_zip':zip_denied,'denied_metadata':meta,
        'denied_write':{'write_code_denied':str(BUNDLE/'code/worker.py'),'write_runtime_denied':str(runtime_file),
                        'write_evaluator_denied':str(EVIDENCE/'approval.json')}}
    config_record=EVIDENCE/f'boundary_config_attempt_{attempt}.json';write(config_record,probe)
    probe_path=OUTPUT/'boundary_config.json';probe_path.write_bytes(config_record.read_bytes())
    command=['/usr/bin/sandbox-exec','-f',str(profile),runtime['interpreter'],'-B','-I','-S',str(BUNDLE/'code/worker.py'),
        '--bundle',str(BUNDLE),'--expected-manifest-sha256',stage['staged']['bundle_manifest_sha256'],'--output',str(OUTPUT),
        '--probe-config',str(probe_path),'--probe-sha256',file_sha256(probe_path)]
    started=now()
    result=subprocess.run(command,cwd=BUNDLE,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'},
        stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True)
    for alias in aliases:alias.unlink()
    probe_path.unlink()
    (EVIDENCE/f'boundary_attempt_{attempt}.stdout.txt').write_text(result.stdout)
    (EVIDENCE/f'boundary_attempt_{attempt}.stderr.txt').write_text(result.stderr)
    run={'command':command,'start_utc':started,'end_utc':now(),'returncode':result.returncode,
        'fresh_worker_launch':True,'inherited_source_descriptors':0,'production_seal_validated':False}
    if result.returncode:
        run['status']='production_cpu_boundary_failed';write(EVIDENCE/f'boundary_attempt_{attempt}.json',run)
        print(json.dumps(run));raise ValueError('Actual production boundary failed: '+result.stderr.strip())
    verify_staged_bundle(destination=BUNDLE,expected_manifest_sha256=stage['staged']['bundle_manifest_sha256'])
    verify_runtime(runtime)
    worker=read(OUTPUT/'worker_verification.json')
    if not all(worker['boundary_checks'].values()):raise ValueError('Incomplete boundary verification')
    write(EVIDENCE/'worker_result.json',worker)
    run.update(status='production_cpu_boundary_passed',bundle_runtime_unchanged=True,
        boundary_checks=len(worker['boundary_checks']),worker_result_sha256=file_sha256(EVIDENCE/'worker_result.json'))
    write(EVIDENCE/'boundary_config.json',probe)
    write(EVIDENCE/f'boundary_attempt_{attempt}.json',run);write(EVIDENCE/'boundary_run.json',run)
    print(json.dumps({'status':run['status'],'checks':len(worker['boundary_checks'])}))


def seal():
    approval=verify_approval();grouping,partitions,proposal=load_proposed_split(PROPOSAL,expected_report_sha256=PROPOSAL_HASH)
    preserved=preservation(grouping,partitions,proposal);write(EVIDENCE/'preservation_final.json',preserved)
    stage=read(EVIDENCE/'stage_record.json');runtime=read(EVIDENCE/'runtime_inventory.json');verify_runtime(runtime)
    _,code=code_inventory()
    if code!=read(EVIDENCE/'code_manifest.json'):raise ValueError('Final inspected code changed')
    verify_staged_bundle(destination=BUNDLE,expected_manifest_sha256=stage['staged']['bundle_manifest_sha256'])
    files={'approval':EVIDENCE/'approval.json','approved_package':PACKAGE,'proposal_report':PROPOSAL/'proposal_report.json',
        'numerical_policy':ROOT/'src/dermdetect/data/numerical_policy.py','code_manifest':EVIDENCE/'code_manifest.json',
        'runtime_inventory':EVIDENCE/'runtime_inventory.json','bundle_manifest':BUNDLE/'bundle_manifest.json',
        'evaluator_provenance':EVIDENCE/'evaluator_provenance.json','boundary_profile':EVIDENCE/'boundary.sb',
        'boundary_config':EVIDENCE/'boundary_config.json','worker_result':EVIDENCE/'worker_result.json',
        'boundary_run':EVIDENCE/'boundary_run.json','composition':EVIDENCE/'composition.json',
        'preservation':EVIDENCE/'preservation_final.json','tests':EVIDENCE/'test_run.json'}
    if set(files)!=set(REQUIRED_BINDINGS):raise ValueError('Incomplete seal evidence')
    if file_sha256(files['boundary_profile'])!=stage['profile_sha256']:raise ValueError('Boundary profile changed')
    result=assemble_seal(approval=approval,proposal=proposal,tests=read(files['tests']),composition=read(files['composition']),
        worker=read(files['worker_result']),boundary=read(files['boundary_run']),preservation=preserved,
        bindings={k:file_sha256(p) for k,p in files.items()},paths={k:str(p) for k,p in files.items()},recorded_utc=now())
    write(EVIDENCE/'production_seal.json',result)
    write(EVIDENCE/'seal_checksum.json',{'production_seal_sha256':file_sha256(EVIDENCE/'production_seal.json')})
    print(json.dumps({'status':result['status'],'production_freeze':True,'m2_complete':True}))


def launch():
    seal_path=EVIDENCE/'production_seal.json';sealed=read(seal_path)
    if file_sha256(seal_path)!=read(EVIDENCE/'seal_checksum.json')['production_seal_sha256']:raise ValueError('Seal changed')
    if sealed.get('production_seal_validated') is not True:raise ValueError('No production seal')
    for key,path in sealed['paths'].items():
        if file_sha256(Path(path))!=sealed['evidence_sha256'][key]:raise ValueError('Frozen evidence changed: '+key)
    _,code=code_inventory()
    if code!=read(EVIDENCE/'code_manifest.json'):raise ValueError('Frozen code changed')
    runtime=read(EVIDENCE/'runtime_inventory.json');verify_runtime(runtime)
    checksum=sealed['evidence_sha256']['bundle_manifest']
    verify_staged_bundle(destination=BUNDLE,expected_manifest_sha256=checksum)
    command=['/usr/bin/sandbox-exec','-f',str(EVIDENCE/'boundary.sb'),runtime['interpreter'],'-B','-I','-S',str(BUNDLE/'code/worker.py'),
        '--bundle',str(BUNDLE),'--expected-manifest-sha256',checksum,'--output',str(OUTPUT)]
    result=subprocess.run(command,cwd=BUNDLE,env={'PATH':'/usr/bin:/bin','LC_ALL':'C','PYTHONDONTWRITEBYTECODE':'1'},
        stdin=subprocess.DEVNULL,close_fds=True,capture_output=True,text=True)
    print(result.stdout,end='');print(result.stderr,end='',file=sys.stderr)
    if result.returncode:raise ValueError('Frozen worker failed')


def run_phase(name):
    actions={'prepare':prepare,'tests':run_tests,'boundary':boundary,'seal':seal,'launch':launch}
    if name not in actions:raise ValueError('Unknown closeout phase')
    return actions[name]()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','tests','boundary','seal','launch'))
    args=parser.parse_args()
    try:run_phase(args.phase)
    except Exception as exc:
        print(json.dumps({'status':'failed','phase':args.phase,'error':type(exc).__name__,'message':str(exc)}),file=sys.stderr)
        raise SystemExit(2)
