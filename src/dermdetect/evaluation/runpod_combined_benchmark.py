# Separate 60-minute provider-window proposal; NOT authorization.
# Original 30-minute adapter remains unchanged; science stays capped at 600s.
"""Runpod-specific TRAIN benchmark admission, not a remote execution entry point.

No Runpod SDK/network, installations, lifecycle calls, lock or held-out grants.
Operator receipts are external hash anchors, not signatures or proof of provider
state. Existing numerical methods and TRAIN benchmark core stay unchanged.
"""
from pathlib import Path
import math
from . import linux_benchmark as core
from .authorization import checked_file, checked_json, runtime_specs
from .engine import digest
from .metrics import NUMERICAL_POLICY
from .supervision import LIMITS

CAPS = {k: v for k, v in core.CAPS.items()
        if k not in ('credit_subcap', 'Sleep_request_seconds')}
CAPS.update(activation_seconds=3600, work_stop_seconds=3300, stop_request_seconds=3480)
WORKROOT = '/workspace/dermdetect_runpod/benchmark'
RUNTIME_ROOT = '/workspace/dermdetect_runpod/runtime'
IDENTITY_KEYS = {'provider', 'pod_id', 'datacenter_id', 'cloud_type', 'machine_type'}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def exact_identity(value):
    if (not isinstance(value, dict) or set(value) != IDENTITY_KEYS
            or value.get('provider') != 'runpod'
            or any(not isinstance(x, str) or not x.strip() for x in value.values())):
        raise PermissionError('Exact provider, Pod, datacenter, cloud and machine identity required')
    return value


def verify_request(root, binding):
    """Reject missing approval before scientific-artifact or TRAIN reads."""
    a = checked_json(root, binding)
    if (a.get('status') != 'user_approved' or not a.get('user_reference')
            or a.get('scope') != 'M6_RUNPOD_60MIN_TRAIN_CPU_BENCHMARK'
            or a.get('compute_authorized') is not True
            or a.get('payload_authorized') is not True
            or a.get('cloud') is not True or a.get('GPU') is not False
            or a.get('held_out_access') is not False
            or a.get('actual_lock_authorized') is not False
            or a.get('remote_install_authorized') is not False
            or a.get('device') != 'cpu' or a.get('caps') != CAPS
            or a.get('resource_limits') != LIMITS
            or a.get('trusted_worker_local_IPC_acknowledged') is not True):
        raise PermissionError('Fresh Runpod CPU benchmark approval required; no GPU/install/lock/unseal')
    exact_identity(a.get('destination_identity'))
    if a.get('workroot') != WORKROOT or a.get('runtime_root') != RUNTIME_ROOT:
        raise PermissionError('Exact separate Runpod work/runtime roots required')
    if a.get('currency') != 'USD' or not finite(a.get('max_total_usd')) or a['max_total_usd'] <= 0:
        raise PermissionError('Explicit positive all-in USD spending limit required')
    if not isinstance(a.get('retention_deadline_UTC'), str):
        raise PermissionError('Explicit retention deadline required')
    core.timestamp(a['retention_deadline_UTC'])
    if a.get('source_TRAIN_manifest_SHA') != core.TRAIN_MANIFEST_SHA:
        raise PermissionError('Original TRAIN lineage required')
    systems = checked_json(root, a['system_manifest'])
    evaluator = checked_json(root, a['evaluator_manifest'])
    accepted = checked_json(root, a['metric_acceptance'])
    if digest(systems.get('systems')) != core.SCIENTIFIC_SYSTEMS_SHA:
        raise PermissionError('Approved scientific systems changed')
    runtime_specs(systems)
    if (accepted.get('status') != 'user_accepted'
            or accepted.get('accepted_policy_SHA') != digest(NUMERICAL_POLICY)
            or evaluator.get('numerical_policy_SHA') != digest(NUMERICAL_POLICY)):
        raise PermissionError('Unchanged accepted metric policy required')
    if a['Linux_lock']['SHA'] != core.LINUX_LOCK_SHA:
        raise PermissionError('Original Linux hash lock required')
    checked_file(root, a['Linux_lock'])
    if a['distribution_plan']['SHA'] != core.DISTRIBUTION_PLAN_SHA:
        raise PermissionError('Original 47-version plan required')
    expected = checked_json(root, a['distribution_plan'])['expected_distribution_versions']
    if len(expected) != 47 or evaluator.get('expected_distribution_versions') != expected:
        raise PermissionError('All 47 distribution versions must match')
    # Reuse the exact TRAIN-only grant and scientific core, not a fake Studio receipt.
    grant = core.BenchmarkGrant(core._TOKEN, root, a, systems, evaluator,
        [binding, a['system_manifest'], a['evaluator_manifest'], a['metric_acceptance'],
         a['Linux_lock'], a['distribution_plan'], a['TRAIN_subset']])
    grant.reverify()
    return grant


def quote_bound(request, receipt, *, start, now):
    """Conservative all-in bound from operator-observed quotes, not an API quote.

Rates must be tax/fee-inclusive. Retained volume is charged for the entire
approved retention window, including running time, so it cannot be omitted at
stop. Fixed/setup/transfer/tax/rounding/lag reserves are independently explicit.
    """
    end = core.timestamp(request['retention_deadline_UTC'])
    if not start + CAPS['activation_seconds'] <= end <= start + 86400:
        raise PermissionError('Retention must cover run window and end within 24 hours of Start')
    keys = ['compute_usd_per_hour', 'retained_storage_usd_per_hour',
            'fixed_setup_transfer_fees_usd', 'billing_rounding_reserve_usd',
            'pending_charge_reserve_usd', 'existing_window_spend_usd',
            'available_balance_usd', 'remaining_project_budget_usd']
    if any(not finite(receipt.get(k)) for k in keys):
        raise PermissionError('Finite nonnegative complete USD quote and reserves required')
    if receipt.get('all_taxes_mandatory_fees_included') is not True:
        raise PermissionError('Complete provider quote required')
    # Whole activation allocation + whole retained-storage window; existing
    # partial spend is added conservatively rather than resetting the budget.
    bound = (receipt['existing_window_spend_usd']
             + receipt['compute_usd_per_hour'] * CAPS['activation_seconds'] / 3600
             + receipt['retained_storage_usd_per_hour'] * (end-start) / 3600
             + receipt['fixed_setup_transfer_fees_usd']
             + receipt['billing_rounding_reserve_usd']
             + receipt['pending_charge_reserve_usd'])
    cap = request['max_total_usd']
    if (bound > cap or receipt['available_balance_usd'] < bound
            or receipt['remaining_project_budget_usd'] < bound):
        raise PermissionError('All-in bound exceeds approved cap, balance or remaining project budget')
    return bound


def verify_provider(grant, binding, *, start_UTC, now):
    receipt = checked_json(grant.root, binding)
    request = grant.approval
    if request.get('scope') != 'M6_RUNPOD_60MIN_TRAIN_CPU_BENCHMARK':
        raise PermissionError('A Lightning grant cannot admit Runpod')
    identity = exact_identity(receipt.get('destination_identity'))
    if identity != request['destination_identity']:
        raise PermissionError('Live Pod differs from approved destination')
    required = ['private_access_verified', 'cpu_only_verified',
                'stop_control_verified', 'user_supervising',
                'persistent_volume_behavior_verified', 'storage_quote_verified',
                'retention_permission', 'no_topup', 'no_subscription',
                'no_other_resources_created', 'prices_unchanged_since_approval']
    if (any(receipt.get(k) is not True for k in required)
            or not receipt.get('user_reference')):
        raise PermissionError('Current access, stop, storage and scope evidence required')
    if not finite(now):
        raise PermissionError('Finite current time required')
    start = core.timestamp(start_UTC)
    verified = core.timestamp(receipt['verified_UTC'])
    if not 0 <= now-verified < 600 or not start <= verified <= now:
        raise PermissionError('Fresh post-Start provider evidence required')
    if not 0 <= now-start < CAPS['work_stop_seconds']:
        raise PermissionError('Original activation work window expired')
    if receipt.get('actual_Start_UTC') != start_UTC:
        raise PermissionError('Original Start mismatch; no clock reset')
    if (type(receipt.get('persistent_used_bytes')) is not int
            or not 0 <= receipt['persistent_used_bytes'] <= CAPS['persistent_bytes']):
        raise PermissionError('Persistent storage cap missing/exceeded')
    bound = quote_bound(request, receipt, start=start, now=now)
    return {'provider': 'runpod', 'start_unix': start,
            'hard_stop_unix': start+CAPS['activation_seconds'],
            'work_stop_unix': start+CAPS['work_stop_seconds'],
            'stop_request_unix': start+CAPS['stop_request_seconds'],
            'all_in_quote_bound_usd': bound,
            'provider_stop_observed': False,
            'retention_deadline_UTC': request['retention_deadline_UTC'],
            'persistent_used_bytes': receipt['persistent_used_bytes']}


def validate_runtime_observation(obs, request):
    """Pure validation only; actual controller must collect trusted live values."""
    if (obs.get('system') != 'Linux' or obs.get('architecture') != 'x86_64'
            or obs.get('python') != '3.11.13'
            or obs.get('GPU_present') is not False
            or obs.get('effective_cpu_count') != 4
            or obs.get('actual_kernel_enforcement_passed') is not True
            or obs.get('all_47_pins_match') is not True):
        raise PermissionError('Exact CPU runtime and actual enforcement required')
    if (obs.get('workroot') != request['workroot']
            or obs.get('runtime_root') != request['runtime_root']):
        raise PermissionError('Runtime location differs from approved roots')
    for key, minimum in [('total_bytes',14*1024**3),
                         ('available_bytes',4*1024**3),
                         ('disk_free_bytes',8*1024**3)]:
        if type(obs.get(key)) is not int or obs[key] < minimum:
            raise PermissionError('Original resource guard failed: '+key)
    return {'CPU_runtime_admitted':True, 'held_out_permission':False,
            'actual_lock_permission':False, 'GPU_permission':False}

# Explicit reuse, no mutation of the original scientific/supervision modules.
def scientific_clock(*,mode,start_UTC,request_SHA,now,saved=None):
    start=core.timestamp(start_UTC)
    if mode=='readiness':
        if saved is not None:raise PermissionError('Readiness cannot repeat/reset')
        clock={'started_unix':now,'deadline_unix':now+CAPS['benchmark_seconds'],'Start_UTC':start_UTC,'request_SHA':request_SHA}
    elif mode=='run':
        clock=dict(saved or {})
        if clock.get('Start_UTC')!=start_UTC or clock.get('request_SHA')!=request_SHA:raise PermissionError('Original clock binding mismatch')
    else:raise PermissionError('Only readiness/run modes admitted')
    for key in ('started_unix','deadline_unix'):
        if type(clock.get(key)) not in (int,float) or not math.isfinite(clock[key]):raise PermissionError('Finite clock required')
    if clock['deadline_unix']-clock['started_unix']!=CAPS['benchmark_seconds'] or not start<=clock['started_unix']<=now:
        raise PermissionError('Cumulative600s clock cannot reset or expand')
    deadline=min(clock['deadline_unix'],start+CAPS['work_stop_seconds'])
    if now>=deadline:raise TimeoutError('Cumulative benchmark/activation deadline')
    return clock,deadline
benchmark_core = core.benchmark_core
synthetic_metric_timing = core.synthetic_metric_timing
size = core.size
storage_reason = core.storage_reason


def host(pid):
    """Enforce all visible container limits, not host memory/CPU alone."""
    import os
    values = core.host(pid)
    rows = [s[3:] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::')]
    if len(rows) != 1 or not rows[0].startswith('/') or '..' in rows[0].split('/'):
        raise PermissionError('Safely resolvable cgroup v2 required')
    child_rows = [s[3:] for s in Path('/proc',str(pid),'cgroup').read_text().splitlines() if s.startswith('0::')]
    if child_rows != rows:
        raise PermissionError('Monitored process left the controller cgroup')
    root = Path('/sys/fs/cgroup');current = root / rows[0].lstrip('/')
    values['host_swap_used_bytes'] = values['swap_used_bytes']
    values['swap_used_bytes'] = int((current/'memory.swap.current').read_text())
    if values['swap_used_bytes'] < 0:raise ValueError('Negative container swap usage')
    values['cgroup_scope'] = rows[0]
    cpu = float(len(os.sched_getaffinity(0)))
    while True:
        text = (current/'memory.max').read_text().strip()
        used = int((current/'memory.current').read_text())
        if used < 0:raise ValueError('Negative cgroup usage')
        if text != 'max':
            limit = int(text)
            if limit < 0:raise ValueError('Negative cgroup limit')
            values['total_bytes'] = min(values['total_bytes'],limit)
            values['available_estimate_bytes'] = min(values['available_estimate_bytes'],max(0,limit-used))
        quota,period = (current/'cpu.max').read_text().split();period = int(period)
        if period <= 0:raise ValueError('Invalid cgroup CPU period')
        if quota != 'max':
            quota = int(quota)
            if quota <= 0:raise ValueError('Invalid cgroup CPU quota')
            cpu = min(cpu,quota/period)
        if current == root:break
        current = current.parent
    if cpu != 4:raise PermissionError('Exactly four effective CPU capacity required')
    values['effective_cpu_count'] = cpu;values['container_resources_verified'] = True
    return values


def child_host(child):
    import os
    def released():
        result = host(os.getpid());result['RSS_bytes'] = 0;result['child_exit_observed'] = True
        return result
    if child.poll() is not None:return released()
    try:return host(child.pid)
    except (FileNotFoundError,ValueError):
        if child.poll() is not None:return released()
        raise
