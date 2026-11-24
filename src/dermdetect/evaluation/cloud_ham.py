"""Explicitly opted-in trusted-process Linux HAM execution, not OS containment.

No provider calls, credential handling, dataset acquisition or automatic retries.
Scientific pipeline/backend remain unchanged. Operator approvals are not signatures.
"""
import math
import os
from pathlib import Path
import time
from .supervision import LIMITS


def validate_contract(a, execution_context="cloud_cpu"):
    gpu=execution_context=="cloud_gpu"
    if execution_context not in ("cloud_cpu","cloud_gpu"):raise PermissionError("Unknown execution context")
    required = ('compute_authorized','payload_authorized','trusted_process_no_OS_isolation_acknowledged',
                'resource_allocation_verified','no_credentials_in_workload_verified',
                'provider_stop_supervision_verified')
    if (a.get('scope') != ('HAM_CLOUD_GPU_EVALUATION_V1' if gpu else 'HAM_CLOUD_CPU_EVALUATION_V1') or a.get('cloud') is not True
            or a.get('device') != ('cuda:0' if gpu else 'cpu') or any(a.get(k) is not True for k in required)
            or a.get('resource_limits') != LIMITS):
        raise PermissionError('Distinct approved cloud HAM trusted-process contract required')
    if gpu and (a.get('GPU_authorized') is not True or a.get('CPU_GPU_parity_required') is not True
                or a.get('max_GPU_allocated_bytes')!=20*2**30):
        raise PermissionError('Separate GPU authorization and parity/20GiB guard required')
    for k in ('destination_id','provider','resource_verification_reference','transfer_approval_reference','user_reference'):
        if not isinstance(a.get(k),str) or not a[k].strip():raise PermissionError('Exact destination and operator evidence required')
    for k in ('max_wall_seconds','provider_work_deadline_unix','max_total_usd','dedicated_memory_bytes'):
        if type(a.get(k)) not in (int,float) or not math.isfinite(a[k]) or a[k]<=0:raise PermissionError('Finite positive execution envelope required')
    if a['max_wall_seconds']>86400 or a['dedicated_memory_bytes']<14*2**30 or a.get('currency')!='USD':
        raise PermissionError('Reviewed CPU resource/budget bounds required')
    if a.get('memory_accounting') not in ('cgroup_v2','cgroup_v1','dedicated_host'):
        raise PermissionError('Explicit supported memory accounting required')
    if a['memory_accounting']=='dedicated_host' and a.get('whole_host_exclusive_verified') is not True:
        raise PermissionError('Host counters require verified exclusive whole host, not a container allocation')
    # This path never starts/stops a provider. Its cost cap is operator-supervised.
    if time.time()+15>=a['provider_work_deadline_unix']:raise PermissionError('Provider work deadline expired')


def clean_environment(output, device="cpu"):
    env={'PATH':'/usr/bin:/bin','LC_ALL':'C','HOME':str(output),'TMPDIR':str(output),
            'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0' if device=='cuda:0' else '',
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
            'TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1'}
    if device=='cuda:0':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    return env


def cgroup_memory(mode):
    """Read this process' actual memory group via kernel mount/group metadata."""
    groups=Path('/proc/self/cgroup').read_text().splitlines()
    mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
    for row in mounts:
        sep=row.index('-');kind=row[sep+1];opts=row[sep+3].split(',')
        if not ((mode=='cgroup_v2' and kind=='cgroup2') or (mode=='cgroup_v1' and kind=='cgroup' and 'memory' in opts)):continue
        for g in groups:
            _,controllers,group=g.split(':',2)
            if not ((mode=='cgroup_v2' and not controllers) or (mode=='cgroup_v1' and 'memory' in controllers.split(','))):continue
            mount_root=Path(row[3]);group_path=Path(group)
            if not group_path.is_relative_to(mount_root):continue
            directory=Path(row[4])/group_path.relative_to(mount_root)
            observations=[]
            mountpoint=Path(row[4])
            while True:
                def number(name, unlimited=False):
                    value=(directory/name).read_text().strip()
                    if unlimited and value=='max':return None
                    if not value.isdigit():raise PermissionError('Numeric cgroup memory counter required')
                    return int(value)
                try:
                    if mode=='cgroup_v2':
                        limit=number('memory.max',True);used=number('memory.current');swap=number('memory.swap.current')
                    else:
                        limit=number('memory.limit_in_bytes');used=number('memory.usage_in_bytes')
                        swap=number('memory.memsw.usage_in_bytes')-used
                except FileNotFoundError:
                    # A real controller root may have no limit files. Never
                    # silently skip a leaf/intermediate group's accounting.
                    if directory!=mountpoint or not observations:raise
                else:
                    if (limit is not None and limit<=0) or used<0 or swap<0:raise PermissionError('Invalid cgroup memory counters')
                    observations.append((limit,used,swap))
                if directory==mountpoint:break
                directory=directory.parent
            finite=[(limit,used) for limit,used,_ in observations if limit is not None]
            if not finite:raise PermissionError('Finite visible allocation required')
            limit=min(n for n,_ in finite);headroom=min(n-u for n,u in finite)
            # Track leaf swap growth consistently; sibling reclaim in an
            # ancestor must not hide this workload's swap increase.
            return limit,limit-headroom,observations[0][2]
    raise PermissionError('Cannot identify approved memory controller; do not assume host headroom')


def sample(pid, approval, expected_scope=None):
    if expected_scope is not None:
        if Path('/proc/self/cgroup').read_text()!=expected_scope or Path(f'/proc/{pid}/cgroup').read_text()!=expected_scope:
            raise PermissionError('Worker/controller memory group changed')
    mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
    status={k:v.strip() for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)}
    rss=int(status['VmRSS'].split()[0])*1024
    mode=approval['memory_accounting']
    if mode=='dedicated_host':
        if approval.get('whole_host_exclusive_verified') is not True:raise PermissionError('Exclusive host required')
        limit=mem['MemTotal'];available=mem['MemAvailable'];swap=mem['SwapTotal']-mem['SwapFree']
    else:
        limit,used,swap=cgroup_memory(mode);available=min(mem['MemAvailable'],limit-used)
    if limit<approval['dedicated_memory_bytes']:raise PermissionError('Actual memory allocation below approved quote')
    return {'RSS_bytes':rss,'available_estimate_bytes':available,'swap_used_bytes':swap}
