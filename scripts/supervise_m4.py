"""Full M4 supervision inside the approved Studio. Never starts/stops provider.
Fresh controls and exact execution approval required; manual Sleep remains mandatory.
"""
import argparse,datetime,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dermdetect.models.full_run_contracts import validate_execution_approval,sha256_file

def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)

def host(pid=0):
    memory={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if len(l.split())>2 and l.split()[2]=='kB'}
    rss=0
    if pid:
        try:rss=int(next(l.split()[1] for l in Path('/proc',str(pid),'status').read_text().splitlines() if l.startswith('VmRSS:')))*1024
        except (OSError,StopIteration):pass
    return {'total':memory['MemTotal'],'available':memory['MemAvailable'],'swap':memory['SwapTotal']-memory['SwapFree'],'RSS':rss}

def gpu():
    r=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True)
    fields=[v.strip() for v in r.stdout.strip().split(',')]
    if len(fields)!=3 or fields[0]!='NVIDIA L4' or int(fields[1])<23000:raise ValueError('Expected one L4')
    return {'name':fields[0],'total_MiB':int(fields[1]),'used_bytes':int(fields[2])*2**20}

def check_provider(control,approval,stage):
    required=['same_GCP','one_L4','non_interruptible','no_move_fee','manual_Sleep_available','no_topup']
    if not all(control.get(k) is True for k in required):raise PermissionError('Fresh provider/stop control incomplete')
    rate=control.get('credits_per_hour');balance=control.get('current_balance')
    if type(rate) not in (int,float) or not math.isfinite(rate) or not 0<rate<=.79:raise ValueError('Fresh bounded rate required')
    if type(balance) not in (int,float) or not math.isfinite(balance) or balance<approval['additional_existing_credit_cap']:raise ValueError('Fresh sufficient existing balance required')
    stamp=datetime.datetime.fromisoformat(control['verified_UTC'].replace('Z','+00:00')).timestamp()
    if not 0<=time.time()-stamp<900:raise ValueError('Provider verification stale')
    used=control.get('persistent_used_bytes');reserve=control.get('storage_and_transfer_reserve_credits')
    if type(used) is not int or used<0 or used+2*10**9>approval['persistent_storage_max_bytes']:
        raise ValueError('Fresh persistent usage plus2GB checkpoint reserve exceeds42GB policy')
    if type(reserve) not in (int,float) or not math.isfinite(reserve) or not 0<=reserve<=.10:
        raise ValueError('Storage/ordinary transfer estimate must fit reserved0.10credits')
    allocation=approval['allocation_seconds'][stage]
    estimated=sum(approval['allocation_seconds'].values())*rate/3600+.10+.04
    if estimated>approval['additional_existing_credit_cap']+1e-10:raise ValueError('Compute/storage reserve exceeds approval')
    return allocation

def own_output_bytes(root):
    # Metadata only in the authorized task output; hard links count once.
    seen=set();total=0
    for current,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=[n for n in dirs if not (Path(current)/n).is_symlink()]
        for name in files:
            p=Path(current)/name
            try:
                if p.is_symlink():continue
                row=p.stat();key=(row.st_dev,row.st_ino)
                if key not in seen:seen.add(key);total+=row.st_size
            except FileNotFoundError:pass
    return total

def monitor_reason(sample,baseline,guards,elapsed,remaining):
    tests=[(remaining<=300,'allocation_checkpoint_reserve'),(sample['host']['available']<guards['available_RAM_min_bytes'],'RAM_available'),(sample['host']['RSS']>guards['RSS_max_fraction']*sample['host']['total'],'RSS'),(sample['host']['swap']-baseline['swap']>guards['swap_growth_max_bytes'],'swap_growth'),(sample['GPU']['used_bytes']>guards['GPU_used_max_bytes'],'GPU_memory'),(sample['disk_free']<guards['free_disk_min_bytes'],'disk'),(sample['persistent_used_estimate']>guards['persistent_storage_max_bytes'],'persistent_storage')]
    return next((reason for failure,reason in tests if failure),None)

def reconcile_allocation(ledger,control,approval,stage,output,*,resume):
    limit=approval['allocation_seconds'][stage]
    prior=[r for r in ledger['stages'] if r['stage']==stage]
    if not prior:
        if resume:raise PermissionError('No prior authorized trajectory')
        return limit,None
    if not resume:raise PermissionError('Stage already reserved; explicit resume required')
    last=prior[-1];proof=control.get('resume_closure',{})
    if (proof.get('user_confirmed_Sleeping') is not True or not proof.get('user_authority') or
        proof.get('previous_start_UTC')!=last['start_UTC'] or proof.get('previous_output')!=output or
        last['output']!=output or not last['Sleeping_pending']):
        raise PermissionError('Exact prior Sleep/trajectory reconciliation required')
    began=datetime.datetime.fromisoformat(last['start_UTC'].replace('Z','+00:00')).timestamp()
    ended=datetime.datetime.fromisoformat(proof['Sleeping_UTC'].replace('Z','+00:00')).timestamp()
    if not began<=ended<=time.time():raise ValueError('Invalid user-attested allocation interval')
    charge=math.ceil(ended-began)
    if charge>last['allocation_seconds']:raise PermissionError('Prior allocation exceeded cap; review required')
    carried=sum(r.get('conservative_paid_seconds',0) for r in prior[:-1])+charge
    remaining=limit-carried
    if remaining<=600:raise PermissionError('Insufficient remaining allocation; no automatic extension')
    last['conservative_paid_seconds']=charge;last['user_attested_Sleeping_UTC']=proof['Sleeping_UTC']
    last['user_authority']=proof['user_authority'];last['billing_release_verified']=False
    return remaining,last

def main():
    p=argparse.ArgumentParser()
    for name in ['stage','config','approval','provider-confirmation','code-root','payload-root','output','start-utc']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--checkpoint');p.add_argument('--snapshot');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if sys.platform!='linux':raise RuntimeError('Actual supervised run requires Linux')
    config,approval=validate_execution_approval(a.config,a.approval,stage=a.stage)
    control=json.loads(Path(a.provider_confirmation).read_text());allocation=check_provider(control,approval,a.stage)
    ACT=datetime.datetime.fromisoformat(a.start_utc.replace('Z','+00:00')).timestamp()
    if not 0<=time.time()-ACT<480:raise TimeoutError('Late readiness; Sleep now')
    C=Path(a.code_root).resolve();R=C.parent;O=Path(a.output).resolve()
    if R.name!='dermdetect_bounded_m4_20261003' or not O.is_relative_to(R) or not Path(a.payload_root).resolve().is_relative_to(R):raise ValueError('Exact private task destination required')
    if a.stage=='training' and (not a.checkpoint or Path(a.checkpoint).resolve()!=R/'payload/weights/densenet/densenet121-a639ec97.pth'):
        raise ValueError('Exact previously approved DenseNet checkpoint route required')
    if a.stage=='extraction' and (not a.snapshot or Path(a.snapshot).resolve()!=R/'payload/weights/medsiglip'):
        raise ValueError('Exact previously approved MedSigLIP snapshot route required')
    O.mkdir(exist_ok=a.resume)
    S=O/('supervision_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S'));S.mkdir(exist_ok=False)
    lock=(R/'full_M4_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Conservatively reserve full allocation; a retry requires reviewed reconciliation.
    ledger_path=R/'full_M4_allocation_ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'config_sha256':sha256_file(a.config),'reserved_seconds':{},'stages':[]}
    if ledger['config_sha256']!=sha256_file(a.config):raise PermissionError('Configuration changed; no trajectory resume')
    allocation,closed=reconcile_allocation(ledger,control,approval,a.stage,str(O),resume=a.resume)
    if allocation-(time.time()-ACT)<=300:raise TimeoutError('Remaining allocation cannot cover shutdown reserve')
    ledger['reserved_seconds'][a.stage]=allocation
    if closed:closed['Sleeping_pending']=False
    ledger['stages'].append({'stage':a.stage,'start_UTC':a.start_utc,'output':str(O),'allocation_seconds':allocation,'Sleeping_pending':True});write(ledger_path,ledger)
    if sha256_file(C/'full_code_manifest.json')!=config['full_code_manifest_sha256']:raise ValueError('Approved source manifest changed')
    integrity=json.loads((C/'full_code_manifest.json').read_text())
    def sources():
        current={n:sha256_file(C/n) for n in integrity['files']}
        if current!=integrity['files']:raise ValueError('Source package changed')
        return current
    before=sources();baseline=host();device=gpu();output_baseline_bytes=own_output_bytes(O)
    import shutil
    if baseline['available']<4*2**30 or shutil.disk_usage(R).free<8*2**30:raise ValueError('Initial resource guard')
    stop=O/'stop_requested.json'
    if a.resume and stop.exists():stop.unlink()
    args=[sys.executable,'-I','-B',str(C/'Full_worker_entry.py'),'--stage',a.stage,'--config',a.config,'--approval',a.approval,'--payload-root',a.payload_root,'--output',str(O),'--stop-file',str(stop),'--max-seconds',str(max(1,allocation-300-(time.time()-ACT)))]
    if a.checkpoint:args+=['--checkpoint',a.checkpoint]
    if a.snapshot:args+=['--snapshot',a.snapshot]
    if a.resume and a.stage=='training':args+=['--resume',str(O/'training/recovery.pt')]
    if (O/'worker_result.json').exists():
        import shutil
        shutil.copyfile(O/'worker_result.json',S/'previous_worker_result.json')
    env={'PATH':'/nonexistent','HOME':str(O),'TMPDIR':str(O),'HF_HOME':str(O/'hf'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','LC_CTYPE':'C.UTF-8','PWD':str(C),'LD_LIBRARY_PATH':'/usr/local/nvidia/lib64:/usr/local/nvidia/lib','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    begin=time.monotonic();reason=None;child=None
    try:
        with (S/'stdout.txt').open('w') as stdout,(S/'stderr.txt').open('w') as stderr:
            child=subprocess.Popen(args,cwd=C,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,start_new_session=True)
            while child.poll() is None:
                elapsed=time.monotonic()-begin
                try:
                    sample={'host':host(child.pid),'GPU':gpu(),'disk_free':shutil.disk_usage(R).free,'worker_seconds':elapsed,'activation_seconds':time.time()-ACT,'persistent_used_estimate':control['persistent_used_bytes']+max(0,own_output_bytes(O)-output_baseline_bytes)}
                    reason=monitor_reason(sample,baseline,config['resource_guards'],elapsed,allocation-(time.time()-ACT))
                    with (S/'samples.jsonl').open('a') as h:h.write(json.dumps(sample)+'\n')
                except Exception:reason='resource_monitor_failed'
                if reason:break
                time.sleep(.5)
    finally:
        if child and child.poll() is None:
            write(stop,{'reason':reason or 'supervisor_failure'})
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    source_after=False
    try:source_after=sources()==before
    except Exception:reason=reason or 'source_integrity_failed'
    worker_result=json.loads((O/'worker_result.json').read_text()) if (O/'worker_result.json').exists() else None
    record={'status':'passed' if child and child.returncode==0 and worker_result and worker_result.get('complete') and not reason and source_after else 'stopped_incomplete_or_failure','stage':a.stage,'reason':reason,'returncode':child.returncode if child else None,'worker_seconds':time.monotonic()-begin,'activation_elapsed_seconds':time.time()-ACT,'source_integrity_after':source_after,'result':worker_result,'GPU':device,'Sleep_now':True,'billing_release_verified':False}
    write(S/'run_record.json',record);print(json.dumps(record));return 0 if record['status']=='passed' else 2

if __name__=='__main__':raise SystemExit(main())
