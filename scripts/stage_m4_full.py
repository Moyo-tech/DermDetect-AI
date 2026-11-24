"""Approval-gated streaming full-M4 development upload using existing SSH only.
No provider wake, credentials/config changes, raw ZIPs or held-out routes.
"""
import argparse,hashlib,json,subprocess,sys,tarfile,time,shlex
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dermdetect.models.full_run_contracts import validate_execution_approval,sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--CPU-start-utc',required=True);p.add_argument('--CPU-window-seconds',type=int,required=True);p.add_argument('--ssh-destination',required=True);p.add_argument('--remote-root',required=True);a=p.parse_args()
    config_path=a.package/'full_config.json';config,approval=validate_execution_approval(config_path,a.approval,stage='training')
    if approval.get('development_upload_approved') is not True or approval.get('transfer_manifest_sha256')!=sha256_file(a.package/'transfer_manifest.json'):raise PermissionError('Exact development transfer approval required')
    if approval.get('CPU_transfer_window_seconds')!=a.CPU_window_seconds or not 0<a.CPU_window_seconds<=3600:raise PermissionError('Explicit finite CPU transfer window required')
    import datetime
    ACT=datetime.datetime.fromisoformat(a.CPU_start_utc.replace('Z','+00:00')).timestamp();deadline=ACT+a.CPU_window_seconds
    if not 0<=time.time()-ACT<a.CPU_window_seconds-60:raise TimeoutError('Late CPU preparation; Sleep now')
    transfer=json.loads((a.package/'transfer_manifest.json').read_text());entries=transfer['entries']
    if len([e for e in entries if e['kind']=='development_image'])!=8505:raise ValueError('Exact8505 development image scope required')
    if sha256_file(a.code/'full_code_manifest.json')!=config['full_code_manifest_sha256']:raise ValueError('Approved curated source manifest changed')
    source_manifest=json.loads((a.code/'full_code_manifest.json').read_text())['files']
    files=[]
    for n,h in source_manifest.items():
        q=a.code/n
        if sha256_file(q)!=h:raise ValueError('Source changed')
        files.append((q,'full_code/'+n))
    files.append((a.code/'full_code_manifest.json','full_code/full_code_manifest.json'))
    for n in ['full_config.json','train.json','validation.json','development_images.json','transfer_manifest.json']:
        if n in config['manifest_hashes'] and sha256_file(a.package/n)!=config['manifest_hashes'][n]:raise ValueError('Development manifest changed')
        files.append((a.package/n,'full_package/'+n))
    files.append((a.approval,'full_package/execution_approval.json'))
    for e in entries:
        if e['kind']=='development_image' and e['action']=='upload_after_approval':
            q=Path(e['source'])
            if q.is_symlink() or q.stat().st_size!=e['bytes'] or sha256_file(q)!=e['sha256']:raise ValueError('Bound development image changed')
            if not e['destination'].startswith(('data/train/','data/validation/')) or '..' in Path(e['destination']).parts:raise ValueError('Forbidden upload destination')
            files.append((q,'full_payload/'+e['destination']))
    if time.time()>deadline-60:raise TimeoutError('CPU hash preparation exhausted allocation')
    R=a.remote_root
    if approval.get('ssh_destination')!=a.ssh_destination or approval.get('remote_root')!=R:raise PermissionError('Exact approved destination required')
    # Exact reuse list is numeric/controlled metadata; normal client handles auth privately.
    remote='''from pathlib import Path
import json,hashlib,os
R=Path(ROOT);T=json.loads((R/'full_package/transfer_manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for e in T['entries']:
 if e['kind']=='pinned_checkpoint':
  p=R/'payload'/e['destination'];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
 else:
  p=R/'full_payload'/e['destination']
  if e['action']=='verify_existing_and_link':
   original=R/'payload'/e['existing_remote_source'];assert sha(original)==e['sha256'];p.parent.mkdir(parents=True,exist_ok=True);os.link(original,p)
  assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
print(json.dumps({'status':'exact_development_transfer_verified','images':8505,'checkpoints':5,'no_model_calls':True}))
'''.replace('ROOT',repr(R))
    cmd='test ! -e /dev/nvidia0 && test ! -e '+shlex.quote(R+'/full_code')+' && test ! -e '+shlex.quote(R+'/full_payload')+' && tar -xf - -C '+shlex.quote(R)+' && '+shlex.quote(R+'/env/bin/python')+' -I -B -c '+shlex.quote(remote)
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/dev/null',a.ssh_destination,cmd]
    # File-backed transport logs avoid PIPE deadlock; never print SSH diagnostics/raw stderr.
    import tempfile
    with tempfile.TemporaryFile() as stdout,tempfile.TemporaryFile() as stderr:
        child=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr)
        import threading
        cutoff=threading.Timer(max(1,deadline-30-time.time()),child.kill);cutoff.start()
        try:
            with tarfile.open(fileobj=child.stdin,mode='w|') as archive:
                for q,n in files:
                    if time.time()>deadline-60:raise TimeoutError('CPU transfer cutoff')
                    archive.add(q,arcname=n,recursive=False)
            child.stdin.close();rc=child.wait(timeout=max(1,deadline-30-time.time()));stdout.seek(0)
            record={'returncode':rc,'result':json.loads(stdout.read()) if rc==0 else None,'Sleep_on_failure':rc!=0}
        finally:
            cutoff.cancel()
            if child.poll() is None:child.kill();child.wait()
    (a.package/'transfer_run_record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return record['returncode']
if __name__=='__main__':raise SystemExit(main())
