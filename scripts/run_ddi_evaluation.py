#!/usr/bin/env python3
"""Locked DDI external evaluation runner (Linux cloud CPU, trusted process, no OS sandbox).

Parent: verifies the scoped DDI grant (HAM joint lock + DDI add-on lock + passed
integrity record) before any DDI row is read, then supervises one worker process
group under the unchanged resource guards. Worker: runs dermdetect.external.ddi.run_ddi
with the locked CPU backend. Never provisions, uploads, locks, fits or retries.
"""
import argparse, importlib.metadata, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dermdetect.evaluation.authorization import checked_json, file_sha
from dermdetect.evaluation.cloud_ham import clean_environment, sample
from dermdetect.evaluation.supervision import LIMITS, resource_reason, watch
from dermdetect.external import ddi

REQUIRED_TRUE = ('payload_authorized', 'trusted_process_no_OS_isolation_acknowledged', 'resource_allocation_verified',
                 'no_credentials_in_workload_verified', 'provider_stop_supervision_verified')


def validate_cloud_contract(a):
    if (a.get('cloud') is not True or a.get('device') != 'cpu' or a.get('resource_limits') != LIMITS
            or any(a.get(k) is not True for k in REQUIRED_TRUE)):
        raise PermissionError('Approved DDI cloud CPU trusted-process contract required')
    for k in ('destination_id', 'provider', 'transfer_approval_reference'):
        if not isinstance(a.get(k), str) or not a[k].strip(): raise PermissionError('Exact destination/transfer evidence required')
    for k in ('max_wall_seconds', 'provider_work_deadline_unix', 'max_total_usd', 'dedicated_memory_bytes'):
        if type(a.get(k)) not in (int, float) or not math.isfinite(a[k]) or a[k] <= 0: raise PermissionError('Finite positive envelope required')
    if a['max_wall_seconds'] > 86400 or a['dedicated_memory_bytes'] < 14*2**30 or a.get('currency') != 'USD':
        raise PermissionError('Reviewed resource/budget bounds required')
    if a.get('memory_accounting') not in ('cgroup_v2', 'cgroup_v1'): raise PermissionError('Container memory accounting required')
    if time.time()+15 >= a['provider_work_deadline_unix']: raise PermissionError('Provider work deadline expired')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--request', required=True); p.add_argument('--request-sha', required=True)
    p.add_argument('--worker', action='store_true'); p.add_argument('--deadline', type=float); args = p.parse_args()
    started = time.monotonic()
    if sys.platform != 'linux': raise PermissionError('Linux-only entry point')
    binding = {'path': args.request, 'SHA': args.request_sha}
    validate_cloud_contract(checked_json(ROOT, binding))
    grant, inventory = ddi.verify_ddi_grant(ROOT, binding); a = grant.approval
    expected = grant.evaluator.get('expected_distribution_versions', {})
    if len(expected) != 47 or {n: importlib.metadata.version(n) for n in expected} != expected:
        raise PermissionError('Exact reviewed all47 dependency versions required')
    bound = {b['path'] for b in grant.evaluator['files']}
    required = {'scripts/run_ddi_evaluation.py', 'src/dermdetect/external/__init__.py', 'src/dermdetect/external/ddi.py'}
    required.update(str(q.relative_to(ROOT)) for q in (ROOT/'src/dermdetect/evaluation').glob('*.py'))
    required.update('src/dermdetect/models/'+n for n in ('__init__.py', 'densenet.py', 'embeddings.py', 'tf_source_resize.py'))
    if not required <= bound: raise PermissionError('Runner, adapter and scientific sources must be add-on-lock bound')
    out = Path(a['output']); control = out.with_name(out.name+'.supervision')
    if args.worker:
        if args.deadline is None or not math.isfinite(args.deadline) or args.deadline > a['provider_work_deadline_unix']-10:
            raise PermissionError('Bounded worker deadline required')
        if json.loads((control/'launch.json').read_text()) != {'request_SHA': args.request_sha, 'deadline': args.deadline} or dict(os.environ) != clean_environment(control, 'cpu'):
            raise PermissionError('Reviewed parent receipt and clean environment required')
        def check():
            if time.time() >= args.deadline: raise TimeoutError('DDI deadline')
        from dermdetect.evaluation.torch_backend import TorchCPUBackend
        backend = TorchCPUBackend(grant, check)
        print(json.dumps(ddi.run_ddi(grant, inventory, backend, out, check=check), sort_keys=True)); return
    if not out.is_absolute() or out.exists() or control.exists(): raise PermissionError('Fresh absolute output; no overwrite or retry')
    if any(q == out or out in q.parents for q in (ROOT, grant.dataset_dir())): raise PermissionError('Output overlaps code or data')
    control.mkdir(parents=False)
    deadline = min(time.time()+max(0, a['max_wall_seconds']-(time.monotonic()-started)), a['provider_work_deadline_unix']-10)
    result = {'status': 'failed', 'worker_reaped': True, 'source_integrity_after': False, 'request_SHA': args.request_sha, 'OS_isolation': False, 'device': 'cpu'}
    try:
        scope = Path('/proc/self/cgroup').read_text(); initial = sample(os.getpid(), a, scope)
        reason = resource_reason(initial, initial['swap_used_bytes'], shutil.disk_usage(control).free)
        if reason or deadline <= time.time(): raise PermissionError('Prelaunch resource/deadline guard: '+str(reason))
        (control/'launch.json').write_text(json.dumps({'request_SHA': args.request_sha, 'deadline': deadline}))
        command = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--request', args.request, '--request-sha', args.request_sha, '--worker', '--deadline', repr(deadline)]
        with (control/'stdout.log').open('x') as so, (control/'stderr.log').open('x') as se, (control/'samples.jsonl').open('x') as samples:
            child = subprocess.Popen(command, cwd=ROOT, env=clean_environment(control, 'cpu'), stdin=subprocess.DEVNULL, stdout=so, stderr=se, close_fds=True, start_new_session=True)
            result['worker_reaped'] = False
            def record(v): samples.write(json.dumps(v)+'\n'); samples.flush()
            def measure(pid):
                try: return sample(pid, a, scope)
                except (FileNotFoundError, KeyError):
                    if child.poll() is None: raise
                    return {'RSS_bytes': 0, 'available_estimate_bytes': initial['available_estimate_bytes'], 'swap_used_bytes': initial['swap_used_bytes']}
            try:
                result.update(watch(child, sample=measure, baseline=initial['swap_used_bytes'], disk_free=lambda: shutil.disk_usage(control).free,
                                    deadline=time.monotonic()+max(0, deadline-time.time()), record_sample=record))
            finally:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait(timeout=5); result.update(worker_reaped=child.poll() is not None, returncode=child.returncode, process_group_kill_requested=True)
        if result['returncode'] or result.get('stop_reason'): raise RuntimeError('Worker did not finish cleanly')
        grant.reverify(); result['source_integrity_after'] = True
        r = json.loads((out/'receipt.json').read_text())
        if r.get('status') != 'complete': raise PermissionError('Complete DDI receipt required')
        result.update(status='passed', receipt_SHA=file_sha(out/'receipt.json'))
    finally:
        result['end_to_end_wall_seconds'] = time.monotonic()-started
        (control/'run_record.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
