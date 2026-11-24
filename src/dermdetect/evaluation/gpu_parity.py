"""Exactly two frozen TRAIN views: CPU/GPU + reload numerical comparison.

No validation/test evaluation, fitting, selection or lock creation. A new explicit
GPU TRAIN contract is mandatory; old CPU benchmark permission does not suffice.
"""
import importlib.metadata
import math
import time
from pathlib import Path
from . import linux_benchmark as b
from .authorization import checked_json,checked_file,runtime_specs
from .cloud_ham import validate_contract
from .engine import digest
from .metrics import NUMERICAL_POLICY,probabilities
from .pipeline import write_once


def verify_request(root,binding):
    a=checked_json(root,binding)
    if (a.get('status')!='user_approved' or a.get('scope')!='TRAIN_CPU_GPU_PARITY_V1'
            or a.get('held_out_access') is not False or a.get('actual_lock_authorized') is not False
            or a.get('source_TRAIN_manifest_SHA')!=b.TRAIN_MANIFEST_SHA):
        raise PermissionError('Fresh exact two-TRAIN CPU/GPU permission required')
    # Same explicit resource/financial/trust envelope, not a HAM/unseal grant.
    envelope=dict(a,scope='HAM_CLOUD_GPU_EVALUATION_V1');validate_contract(envelope,'cloud_gpu')
    systems=checked_json(root,a['system_manifest']);evaluator=checked_json(root,a['evaluator_manifest'])
    accepted=checked_json(root,a['metric_acceptance'])
    if digest(systems.get('systems'))!=b.SCIENTIFIC_SYSTEMS_SHA:raise PermissionError('Fixed scientific systems changed')
    runtime_specs(systems)
    if accepted.get('status')!='user_accepted' or accepted.get('accepted_policy_SHA')!=digest(NUMERICAL_POLICY) or evaluator.get('numerical_policy_SHA')!=digest(NUMERICAL_POLICY):
        raise PermissionError('Accepted unchanged metric policy required')
    if a['Linux_lock']['SHA']!=b.LINUX_LOCK_SHA or a['distribution_plan']['SHA']!=b.DISTRIBUTION_PLAN_SHA:
        raise PermissionError('Original pinned Linux runtime required')
    checked_file(root,a['Linux_lock']);expected=checked_json(root,a['distribution_plan'])['expected_distribution_versions']
    if len(expected)!=47 or expected!=evaluator.get('expected_distribution_versions') or {n:importlib.metadata.version(n) for n in expected}!=expected:
        raise PermissionError('Exact original47 distributions required')
    # Reuse original exact curated-composition, two-image identity/byte/hash and
    # immutable artifact reader. Private token is a Python marker, not security.
    grant=b.BenchmarkGrant(b._TOKEN,root,a,systems,evaluator,[binding,a['system_manifest'],a['evaluator_manifest'],a['metric_acceptance'],a['Linux_lock'],a['distribution_plan'],a['TRAIN_subset']])
    grant.reverify();return grant


def compare_references(cpu,gpu,specs,expected_count=2):
    if expected_count not in (1,2) or len(cpu)!=expected_count or len(gpu)!=expected_count:raise ValueError('Exact two references required')
    maximum=0.
    for left,right in zip(cpu,gpu):
        if left['image_id']!=right['image_id']:raise ValueError('TRAIN identities differ')
        for key in ('dense_features','med_features'):
            maximum=max(maximum,b.compare(left[key],right[key],atol=1e-4,rtol=1e-4))
        for name,s in specs.items():
            maximum=max(maximum,b.compare(left['logits'][name],right['logits'][name],atol=1e-4,rtol=1e-4))
            p=probabilities([left['logits'][name]],s['temperature'])[0][0]
            q=probabilities([right['logits'][name]],s['temperature'])[0][0]
            b.compare(p,q,atol=1e-4,rtol=1e-4)
            predict=lambda x:(int(x[1]>=.5) if len(x)==2 else max(range(len(x)),key=lambda i:x[i]))
            if predict(p)!=predict(q) or any((max(p)>=c)!=(max(q)>=c) for c in s['cutoffs']):
                raise ValueError('Fixed TRAIN classification/coverage decision changed')
    return maximum


def run(grant,cpu_factory,gpu_factory,output,check=lambda:None):
    if not isinstance(grant,b.BenchmarkGrant) or grant.approval.get('scope')!='TRAIN_CPU_GPU_PARITY_V1':raise PermissionError('Verified TRAIN parity grant required')
    output=Path(output);output.mkdir()
    specs=runtime_specs(grant.systems)
    reload_check=lambda left,right:compare_references([left],[right],specs,expected_count=1)
    cpu=b.benchmark_core(grant,cpu_factory,output/'cpu',check=check,reload_check=reload_check)
    gpu=b.benchmark_core(grant,gpu_factory,output/'gpu',check=check,reporting_device='cuda:0',reload_check=reload_check)
    difference=compare_references(cpu['reference'],gpu['reference'],runtime_specs(grant.systems))
    grant.reverify();check()
    timings=b.synthetic_metric_timing(check)
    # Deliberately conservative engineering projection, not a statistical bound.
    forwards=[r for r in gpu['stages'] if 'dense_seconds' in r]
    per_view=max(r['dense_seconds'] for r in forwards)+max(r['med_seconds'] for r in forwards)
    setup=max(r['setup_seconds'] for r in gpu['stages'] if 'setup_seconds' in r)
    projection=setup+1.5*1508*per_view+timings['seven_class_pair_seconds']+timings['binary_pair_seconds']+120
    result={'status':'complete','parity_passed':True,'scope':'TRAIN_CPU_GPU_PARITY_V1',
       'system_manifest':grant.approval['system_manifest'],'evaluator_manifest':grant.approval['evaluator_manifest'],
       'destination_id':grant.approval['destination_id'],'scientific_systems_SHA':digest(grant.systems['systems']),
       'maximum_feature_or_logit_absolute_difference':difference,'atol':1e-4,'rtol':1e-4,
       'CPU_forwards_per_model':3,'GPU_forwards_per_model':3,'held_out_access':False,'lock_written':False,
       'synthetic_metric_timing':timings,'HAM_projected_seconds':projection,
       'projection_basis':'setup + 1.5*1508*sum(max measured per-model view times) + invented bootstrap +120s reserve; not guarantee',
       'provider_stop_observed':False}
    write_once(output/'receipt.json',result);return result


def verify_receipt_before_unseal(root,a):
    receipt=checked_json(root,a['CPU_GPU_parity_receipt'])
    supervision=checked_json(root,a['CPU_GPU_parity_supervision'])
    if (receipt.get('status')!='complete' or receipt.get('parity_passed') is not True
        or receipt.get('scope')!='TRAIN_CPU_GPU_PARITY_V1' or receipt.get('held_out_access') is not False
        or any(receipt.get(k)!=a.get(k) for k in ('system_manifest','evaluator_manifest','destination_id'))
        or supervision.get('status')!='passed' or supervision.get('worker_reaped') is not True
        or supervision.get('source_integrity_after') is not True or supervision.get('receipt_SHA')!=a['CPU_GPU_parity_receipt']['SHA']):
        raise PermissionError('Same-source/destination supervised TRAIN CPU/GPU parity required before unseal')
    forecast=receipt.get('HAM_projected_seconds')
    if type(forecast) not in (int,float) or not math.isfinite(forecast) or not 0<forecast<=min(a['max_wall_seconds'],a['provider_work_deadline_unix']-time.time()-15):
        raise PermissionError('Measured HAM forecast exceeds approved run budget')
