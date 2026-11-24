"""Pending execution path: explicit cloud HAM approval and joint lock required.

Trusted offline model APIs and clean environment, NOT a kernel network/filesystem
sandbox. Never provision, upload, lock, unseal implicitly, fit, or auto-retry.
"""
import argparse,json,os,signal,subprocess,sys,time,shutil,math,importlib.metadata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dermdetect.evaluation.authorization import checked_json,verify_grant,file_sha,route,input_root
from dermdetect.evaluation.cloud_ham import validate_contract,clean_environment,sample
from dermdetect.evaluation.supervision import resource_reason,watch


def grant_for(binding,execution_context="cloud_cpu"):
    a=checked_json(ROOT,binding)
    if execution_context=='parity_gpu':
        from dermdetect.evaluation.gpu_parity import verify_request
        return verify_request(ROOT,binding)
    # Reject cloud scope before verify_grant can reverify/open dataset artifacts.
    validate_contract(a,execution_context)
    if execution_context=='cloud_gpu':
        from dermdetect.evaluation.gpu_parity import verify_receipt_before_unseal
        verify_receipt_before_unseal(ROOT,a)
    return verify_grant(ROOT,run_approval=binding,system_manifest=a['system_manifest'],
        evaluator_manifest=a['evaluator_manifest'],acceptance=a['metric_acceptance'],
        lock=a.get('lock'),lock_approval=a.get('lock_approval'),execution_context=execution_context)


def output_path(grant):
    dataset=grant.dataset if hasattr(grant,'dataset') else {'manifest':grant.approval['TRAIN_subset'],'payload_mode':'files','payload_root':grant.approval['payload_root']}
    out=Path(grant.approval['output'])
    if not out.is_absolute() or '..' in out.parts or any(p.is_symlink() for p in (out,*out.parents)):
        raise PermissionError('Canonical absolute output required')
    if out.exists() or out.with_name(out.name+'.supervision').exists():raise PermissionError('No overwrite or automatic retry')
    # Never place writable output inside a source/model/data directory.
    protected=[route(grant.root,b['path']) for b in grant.evaluator['files']]
    protected += [route(grant.root,dataset['manifest']['path'])]
    for p in protected:
        if p==out or out in p.parents:raise PermissionError('Output overlaps approved input')
    protected += [route(grant.root,grant.systems['M5_source_configuration']['path'])]
    for system in grant.systems['systems'].values():
        weights=system['selected_backbone']
        for b in weights if isinstance(weights,list) else [weights]:protected.append(route(grant.root,b['path']))
        for key in ('probe','probe_metadata','temperature_record'):
            if key in system:protected.append(route(grant.root,system[key]['path']))
    roots={p.parent for p in protected}
    roots.update((Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()))
    if dataset['payload_mode']=='files':roots.add(input_root(grant.root,dataset['payload_root']))
    else:roots.update(input_root(grant.root,b['path']).parent for b in dataset['sources'].values())
    for dest in (out,out.with_name(out.name+'.supervision')):
        # ROOT itself contains both reviewed files and outputs: protect concrete
        # source/model/data directories, not the entire curated work tree.
        if any(p!=grant.root and (dest==p or p in dest.parents or dest in p.parents) for p in roots):
            raise PermissionError('Output/control overlaps source, model, payload or runtime directory')
    return out


def main(execution_context="cloud_cpu"):
    gpu=execution_context in ('cloud_gpu','parity_gpu')
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--request-sha',required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--deadline',type=float);args=p.parse_args()
    started=time.monotonic()
    if sys.platform!='linux':raise PermissionError('Linux-only entry point')
    binding={'path':args.request,'SHA':args.request_sha};grant=grant_for(binding,execution_context);a=grant.approval
    expected=grant.evaluator.get('expected_distribution_versions',{})
    if len(expected)!=47 or {n:importlib.metadata.version(n) for n in expected}!=expected:
        raise PermissionError('Exact reviewed all47 dependency versions required')
    out=Path(a['output']);control=out.with_name(out.name+'.supervision')
    if args.worker:
        if args.deadline is None or not math.isfinite(args.deadline) or args.deadline>a['provider_work_deadline_unix']-10:
            raise PermissionError('Bounded worker deadline required')
        launch=json.loads((control/'launch.json').read_text())
        if launch!={'request_SHA':args.request_sha,'deadline':args.deadline} or dict(os.environ)!=clean_environment(control,a['device']):
            raise PermissionError('Reviewed parent receipt and clean environment required')
        def check():
            if time.time()>=args.deadline:raise TimeoutError('Cloud HAM deadline')
        from dermdetect.evaluation.torch_backend import TorchCPUBackend
        from dermdetect.evaluation.pipeline import run_ham
        if execution_context=='parity_gpu':
            from dermdetect.evaluation.torch_gpu_backend import TorchGPUBackend
            from dermdetect.evaluation.gpu_parity import run
            print(json.dumps(run(grant,TorchCPUBackend,TorchGPUBackend,out,check),sort_keys=True));return
        if gpu:
            from dermdetect.evaluation.torch_gpu_backend import TorchGPUBackend
            backend=TorchGPUBackend(grant,check)
        else:backend=TorchCPUBackend(grant,check)
        print(json.dumps(run_ham(grant,backend,out,check=check),sort_keys=True));return
    source_names={v['path'] for v in grant.evaluator['files']}
    required={'src/dermdetect/__init__.py','scripts/run_cloud_ham_evaluation.py','src/dermdetect/evaluation/cloud_ham.py','src/dermdetect/evaluation/authorization.py',
              'src/dermdetect/evaluation/supervision.py','src/dermdetect/evaluation/pipeline.py','src/dermdetect/evaluation/torch_backend.py'}
    if gpu:required.update(('scripts/run_cloud_gpu_ham_evaluation.py','scripts/run_cloud_gpu_parity.py'))
    required.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/dermdetect/evaluation').glob('*.py'))
    required.update('src/dermdetect/models/'+n for n in ('__init__.py','densenet.py','embeddings.py','tf_source_resize.py'))
    if not required <= source_names:raise PermissionError('Runner and scientific sources must be review-bound')
    out=output_path(grant);control.mkdir(parents=False)
    deadline=min(time.time()+max(0,a['max_wall_seconds']-(time.monotonic()-started)),a['provider_work_deadline_unix']-10)
    result={'status':'failed','worker_reaped':True,'source_integrity_after':False,'request_SHA':args.request_sha,
            'provider_stop_observed':False,'OS_isolation':False,'device':a['device']}
    try:
        memory_scope=Path('/proc/self/cgroup').read_text()
        initial=sample(os.getpid(),a,memory_scope)
        reason=resource_reason(initial,initial['swap_used_bytes'],shutil.disk_usage(control).free)
        if reason or deadline<=time.time():raise PermissionError('Prelaunch resource/deadline guard: '+str(reason))
        (control/'launch.json').write_text(json.dumps({'request_SHA':args.request_sha,'deadline':deadline}))
        command=[sys.executable,'-I','-B',str(ROOT/'scripts/run_cloud_gpu_parity.py' if execution_context=='parity_gpu' else ROOT/'scripts/run_cloud_gpu_ham_evaluation.py' if gpu else Path(__file__).resolve()),'--request',args.request,'--request-sha',args.request_sha,'--worker','--deadline',str(deadline)]
        with (control/'stdout.log').open('x') as stdout,(control/'stderr.log').open('x') as stderr,(control/'samples.jsonl').open('x') as samples:
            child=subprocess.Popen(command,cwd=ROOT,env=clean_environment(control,a['device']),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,close_fds=True,start_new_session=True)
            result['worker_reaped']=False
            def record(v):samples.write(json.dumps(v)+'\n');samples.flush()
            def measure(pid):
                if sum(x.stat().st_size for x in control.glob('*.log'))>64*2**20:raise RuntimeError('Log cap')
                try:return sample(pid,a,memory_scope)
                except (FileNotFoundError,KeyError):
                    if child.poll() is None:raise
                    return {'RSS_bytes':0,'available_estimate_bytes':initial['available_estimate_bytes'],'swap_used_bytes':initial['swap_used_bytes']}
            try:
                result.update(watch(child,sample=measure,baseline=initial['swap_used_bytes'],disk_free=lambda:shutil.disk_usage(control).free,
                                    deadline=time.monotonic()+max(0,deadline-time.time()),record_sample=record))
            finally:
                # watch guarantees reap even when sampling fails; preserve that
                # actual state in the failure record rather than losing it.
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=5)
                result.update(worker_reaped=child.poll() is not None,returncode=child.returncode)
                result['process_group_kill_requested']=True
                # Trusted code must not detach children into new sessions;
                # without an OS sandbox this is not hostile descendant containment.
        if result['returncode'] or result['stop_reason']:raise RuntimeError('Worker did not finish cleanly')
        grant.reverify();result['source_integrity_after']=True
        receipt=out/'receipt.json';r=json.loads(receipt.read_text())
        if r.get('status')!='complete':raise PermissionError('Complete scientific receipt required')
        result.update(status='passed',receipt_SHA=file_sha(receipt))
    finally:
        result['end_to_end_wall_seconds']=time.monotonic()-started
        (control/'run_record.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
