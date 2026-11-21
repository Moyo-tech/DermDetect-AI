#!/usr/bin/env python3
"""Mac local CPU stage guardian; no scientific input/model reads or cloud access."""
from pathlib import Path
import argparse,hashlib,json,os,re,signal,subprocess,time

CAPS={'benchmark':120,'features':4200,'fitting':2700}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def memory_sample(pid):
    ps=subprocess.run(['/bin/ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,check=False)
    rss=int(ps.stdout.strip() or '0')*1024
    vm=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True).stdout
    page=int(re.search(r'page size of (\d+) bytes',vm).group(1));pages={}
    for line in vm.splitlines()[1:]:
        if ':' in line:
            k,v=line.split(':',1)
            if v.strip().rstrip('.').isdigit():pages[k]=int(v.strip().rstrip('.'))
    available=page*sum(pages.get(k,0) for k in ['Pages free','Pages inactive','Pages speculative'])
    swap=subprocess.run(['/usr/sbin/sysctl','-n','vm.swapusage'],capture_output=True,text=True,check=True).stdout
    match=re.search(r'used = ([\d.]+)([MG])',swap)
    if not match:raise ValueError('Cannot verify swap monitor')
    used=float(match.group(1))*(1024**2 if match.group(2)=='M' else 1024**3)
    return dict(RSS_bytes=rss,available_estimate_bytes=available,swap_used_bytes=int(used))
GiB=1024**3;MiB=1024**2
# Stage-specific guards. Image stages keep the original limits. Fitting reads only
# ~75 MB of cached features; a full-scale synthetic M5 fitting run peaked near 1 GiB
# RSS, so 2 GiB/1.5 GiB still stops well before an 8 GiB host is stressed.
GUARDS={'benchmark':dict(RSS_max=6*GiB,available_min=4*GiB,swap_growth_max=256*MiB,free_disk_min=8*GiB),
        'features':dict(RSS_max=6*GiB,available_min=4*GiB,swap_growth_max=256*MiB,free_disk_min=8*GiB),
        'fitting':dict(RSS_max=2*GiB,available_min=3*GiB//2,swap_growth_max=256*MiB,free_disk_min=8*GiB)}
def _label(prefix,value):
    return prefix+(f'{value/GiB:g}GiB' if value>=GiB else f'{value/MiB:g}MiB')
def reason(sample,baseline,free_disk,guard):
    if sample['RSS_bytes']>guard['RSS_max']:return _label('RSS',guard['RSS_max'])
    if sample['available_estimate_bytes']<guard['available_min']:return _label('available_memory',guard['available_min'])
    if sample['swap_used_bytes']-baseline>guard['swap_growth_max']:return _label('swap_growth',guard['swap_growth_max'])
    if free_disk<guard['free_disk_min']:return _label('free_disk',guard['free_disk_min'])
    return None
def stage_profile(base,write_root,stage_root):
    old='(deny file-write* (require-not (subpath "'+str(write_root)+'")))'
    new='(deny file-write* (require-not (subpath "'+str(stage_root)+'")))'
    if base.count(old)!=1:raise ValueError('Cannot verify exact stage write boundary')
    return base.replace(old,new)+'\n(deny file-write* (literal "'+str(stage_root/'worker_boundary.sb')+'"))\n'
def main():
    p=argparse.ArgumentParser()
    for n in ['configuration','approval','stage','boundary','manifest','output','ledger','python']:p.add_argument('--'+n,required=True)
    a=p.parse_args(); config=Path(a.configuration);c=json.loads(config.read_text());approval=json.loads(Path(a.approval).read_text())
    if (a.stage not in CAPS or c['budgets_seconds']!=dict(CAPS,total=7200) or approval.get('status')!='user_approved' or
        approval.get('configuration_SHA')!=sha(config) or approval.get('code_manifest_SHA')!=sha(a.manifest) or
        approval.get('boundary_SHA')!=sha(a.boundary) or not approval.get('user_reference') or
        a.stage not in approval.get('stages',[]) or approval.get('held_out_access') is not False or approval.get('cloud_or_GPU') is not False):
        raise ValueError('Exact stage/source/profile/budget approval required')
    B=Path(__file__).resolve().parents[1]
    for relative,h in json.loads(Path(a.manifest).read_text()).items():
        if sha(B/relative)!=h:raise ValueError('Staged source changed')
    ledger=Path(a.ledger);history=json.loads(ledger.read_text()) if ledger.exists() else []
    if any(v['configuration_SHA']!=sha(config) for v in history) or any(v['stage']==a.stage for v in history):
        raise ValueError('Config changed or automatic stage retry forbidden')
    staging=json.loads((Path(c['staged_input_root'])/'staging_record.json').read_text())
    if staging['configuration_SHA']!=sha(config) or not staging['verified'] or staging['wall_seconds']>180:
        raise ValueError('Verified bounded staging receipt required')
    charged=sum(v['elapsed_seconds'] for v in history)+staging['wall_seconds'];remaining=7200-charged
    if remaining<CAPS[a.stage]:raise ValueError('Insufficient cumulative approved CPU budget')
    O=Path(a.output);O.mkdir(); env={'PATH':'/usr/bin:/bin','LC_ALL':'C','HOME':str(O),'TMPDIR':str(O),
         'HF_HOME':str(O/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'',
         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','VECLIB_MAXIMUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    initial=memory_sample(os.getpid());st=os.statvfs(O);guard=GUARDS[a.stage];why=reason(initial,initial['swap_used_bytes'],st.f_bavail*st.f_frsize,guard)
    if why:raise ValueError('Prelaunch resource guard '+why)
    effective=O/'worker_boundary.sb'
    effective.write_text(stage_profile(Path(a.boundary).read_text(),Path(a.boundary).resolve().parent/'worker-output',O.resolve()))
    # Worker output is a child of the writable guardian root; scientific inputs are profile read-only.
    cmd=['/usr/bin/sandbox-exec','-f',str(effective),a.python,'-B','-I',str(B/'scripts/m5_cpu_worker.py'),
         '--configuration',a.configuration,'--approval',a.approval,'--stage',a.stage,'--output',str(O/'artifacts')]
    start=time.monotonic();record={'configuration_SHA':sha(config),'stage':a.stage,'elapsed_seconds':0,'status':'started','resource_guard':guard}
    history.append(record);ledger.write_text(json.dumps(history,indent=2)+'\n')
    with (O/'stdout.txt').open('x') as stdout,(O/'stderr.txt').open('x') as stderr,(O/'samples.jsonl').open('x') as samples:
        child=subprocess.Popen(cmd,env=env,cwd=B,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,close_fds=True)
        try:
            while child.poll() is None:
                sample=memory_sample(child.pid);st=os.statvfs(O);why=reason(sample,initial['swap_used_bytes'],st.f_bavail*st.f_frsize,guard)
                if time.monotonic()-start>=CAPS[a.stage]:why='CPU_wall_deadline'
                samples.write(json.dumps(dict(sample,elapsed_seconds=time.monotonic()-start,stop_reason=why))+'\n');samples.flush()
                if why:os.killpg(child.pid,signal.SIGTERM);break
                time.sleep(1)
            try:rc=child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait(timeout=5)
        finally:
            if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            source_ok=all(sha(B/n)==h for n,h in json.loads(Path(a.manifest).read_text()).items())
            record.update(elapsed_seconds=time.monotonic()-start,returncode=child.returncode,stop_reason=why,
                          status='passed' if child.returncode==0 and not why and source_ok else 'stopped',worker_reaped=True,
                          source_integrity_after=source_ok,stage_write_profile_SHA=sha(effective))
            ledger.write_text(json.dumps(history,indent=2)+'\n');(O/'run_record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if record['status']!='passed':raise SystemExit(2)
if __name__=='__main__':main()
