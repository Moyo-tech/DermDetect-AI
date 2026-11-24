"""Local CPU admission and deny-network/read-allowlist profile for HAM inference."""
import json
from pathlib import Path
from .authorization import checked_file, checked_runtime_file, route, input_root, checked_input_file

LIMITS={'RSS_bytes':6*1024**3,'available_bytes':4*1024**3,'swap_growth_bytes':256*1024**2,'disk_reserve_bytes':8*1024**3,'threads':2}

def resource_reason(sample,baseline,free_disk):
    if sample['RSS_bytes']>LIMITS['RSS_bytes']:return 'RSS6GiB'
    if sample['available_estimate_bytes']<LIMITS['available_bytes']:return 'available_memory4GiB'
    if sample['swap_used_bytes']-baseline>LIMITS['swap_growth_bytes']:return 'swap_growth256MiB'
    if free_disk<LIMITS['disk_reserve_bytes']:return 'free_disk8GiB'
    return None

def admission(grant):
    a=grant.approval
    if a.get('device')!='cpu' or a.get('resource_limits')!=LIMITS or type(a.get('max_wall_seconds')) not in (int,float) or not 0<a['max_wall_seconds']<=7200:
        raise PermissionError('Explicit bounded local CPU approval with unchanged resource guards required')
    if not a.get('boundary_review_reference'):
        raise PermissionError('Inference boundary review required')
    output=Path(a['output'])
    if not output.is_absolute() or '..' in output.parts or any(p.is_symlink() for p in (output,*output.parents)):
        raise PermissionError('Approved absolute nonsymlink output required')
    if output.exists() or output.with_name(output.name+'.supervision').exists():
        raise PermissionError('No automatic repeat or existing-run overwrite')
    return output


def profile(grant,output,control,interpreter,runtime_roots,probe_paths):
    """Read individual approved files; read model snapshot directories for listing.

    Raw ZIP access, if separately approved, is to the named source archives only;
    the trusted reader opens solely the approved image members. It is not a
    hostile-code or machine-owner security boundary.
    """
    output=Path(output).resolve();control=Path(control).resolve(strict=True)
    files=[checked_file(grant.root,b) for b in grant.evaluator['files']]
    files += [checked_file(grant.root,grant.systems['M5_source_configuration']),checked_file(grant.root,grant.dataset['manifest'])]
    for s in grant.systems['systems'].values():
        for key in ('probe','probe_metadata','temperature_record'):
            if key in s:files.append(checked_file(grant.root,s[key]))
        b=s['selected_backbone']
        files += [checked_file(grant.root,v) for v in b] if isinstance(b,list) else [checked_file(grant.root,b)]
    if grant.dataset['payload_mode']=='zip':
        files += [checked_input_file(grant.root,b) for b in grant.dataset['sources'].values()]
    else:
        from .pipeline import load_rows
        base=input_root(grant.root,grant.dataset['payload_root'])
        files += [route(base,r['path']) for r in load_rows(grant)]
    files += [Path(p).resolve(strict=True) for p in probe_paths]
    # The authority request lists every consent/manifest file required by worker.
    files += [checked_file(grant.root,b) for b in grant.authority_files]
    for binding in grant.evaluator.get('runtime_files',[]):files.extend(checked_runtime_file(binding))
    literal=lambda p:'(literal '+json.dumps(str(p))+')'
    sub=lambda p:'(subpath '+json.dumps(str(p))+')'
    for p in files:
        if p==output or output in p.parents or p in output.parents:
            raise PermissionError('Read-only source and writable output overlap')
    runtimes=set()
    for p in runtime_roots:
        p=Path(p).absolute()
        if p==Path('/') or p==output or p in output.parents or output in p.parents:
            raise PermissionError('Unsafe runtime root')
        runtimes.update((p,p.resolve(strict=True)))
    dirs={p.parent for p in files}
    dirs.update(q for p in files for q in p.parents)
    system=['/System/Library','/usr/lib','/usr/share','/Library/Apple','/System/Volumes/Preboot','/private/var/db/timezone']
    reads=[literal(p) for p in set(files)|dirs]+[sub(p) for p in runtimes]+[sub(p) for p in system]
    reads += [sub(output),sub(control/'probe_output'),literal('/dev/null'),literal('/dev/urandom'),literal('/dev/random'),sub('/dev/fd'),literal('/private/etc/passwd'),literal('/private/etc/localtime')]
    exe=Path(interpreter).resolve(strict=True);executables=[exe]
    companion=exe.parent.parent/'Resources/Python.app/Contents/MacOS/Python'
    if companion.is_file():executables.append(companion.resolve(strict=True))
    return '(version 1)\n(allow default)\n(deny network*)\n(deny file-read-data (require-not (require-any '+ ' '.join(reads)+')))\n(deny file-write* (require-not (require-any '+sub(output)+' '+sub(control/'probe_output')+')))\n(deny process-exec (require-not (require-any '+' '.join(literal(p) for p in executables)+')))\n'


def watch(child, *, sample, baseline, disk_free, deadline, record_sample, clock=None, pause=None):
    """Supervise/reap one process group; every exit includes observed termination."""
    import os
    import signal
    import time
    clock=clock or time.monotonic;pause=pause or time.sleep;reason=None
    try:
        while child.poll() is None:
            values=sample(child.pid);reason=resource_reason(values,baseline,disk_free())
            if clock()>=deadline:reason='CPU_wall_deadline'
            record_sample(dict(values,stop_reason=reason))
            if reason:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                break
            pause(1)
        try:child.wait(timeout=5)
        except Exception:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait(timeout=5)
    finally:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait(timeout=5)
    return {'returncode':child.returncode,'worker_reaped':True,'stop_reason':reason}
