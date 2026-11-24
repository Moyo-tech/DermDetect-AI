"""Linux TRAIN-only CPU benchmark admission/core. No final evaluation capability.

Approval/provider records are trusted operator SHA anchors, not signatures. This
module cannot create a joint lock or a HAM/DDI outcome-unseal grant.
"""
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from .authorization import checked_file,checked_json,file_sha,route,runtime_specs
from .engine import digest,SYSTEMS,paired_intervals
from .metrics import NUMERICAL_POLICY
from .pipeline import normalize_rows,write_once
from .supervision import LIMITS,resource_reason

WORKROOT='/teamspace/studios/this_studio/dermdetect_m6_m7_20261101'
RETAINED_ROOT='/teamspace/studios/this_studio/dermdetect_bounded_m4_20261003'
DISTRIBUTION_PLAN_SHA='959dc3f9088b0271b19cb0e6fc625e48e67448234db111a3f3561a426409a549'
LINUX_LOCK_SHA='528662fbc977fa44c7faa2468dfe43c53101c2568ef8e0f2f3c5094eccb8954c'
SCIENTIFIC_SYSTEMS_SHA='3398a83c941cf2202674280649595c04aa4e69d6bdd1c19a72e9293a25cc88c2'
TRAIN_MANIFEST_SHA='30d6816e400fb92d01820d9c97eb46579837a0e0e627c0bff0e5e949ae6bf80f'
EXPECTED_TRAIN=(('ISIC_0024306','200ad6e2c9d8a56595f35f715d8c3af7ea45b1ad7b7e5133fe6a13b809abb4b4',279172),
                ('ISIC_0024307','33837415d2b284c923a8fb550a281857f6b59fa551d17585fb4e57da3602894b',274405))
CAPS={'activation_seconds':1800,'work_stop_seconds':1500,'Sleep_request_seconds':1680,'benchmark_seconds':600,
      'readiness_seconds':120,'credit_subcap':.25,'persistent_bytes':42_000_000_000,'growth_bytes':5*1024**3,
      'payload_code_config_bytes':4*1024**2,'selected_parameters_bytes':256*1024,'outputs_bytes':64*1024**2,
      'forwards_per_model':3,'batch_size':1,'threads':2}
_TOKEN=object()

def timestamp(value):
    dt=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise PermissionError('Timezone-aware timestamp required')
    return dt.timestamp()

def strict_package_files(root):
    """Relative file paths under root, excluding top-level benchmark_control.

    Fails closed: an unlistable directory raises instead of being skipped
    (pathlib.rglob silently skips it on Python 3.11, which made an in-sandbox
    check undercount the package on 2026-11-22), and symlinks are refused.
    """
    root=Path(root)
    def fail(error):raise PermissionError('Curated package directory not listable: '+str(error.filename))
    files=set()
    for current,dirs,names in os.walk(root,onerror=fail,followlinks=False):
        here=Path(current)
        if here==root:dirs[:]=[d for d in dirs if d!='benchmark_control']
        for name in dirs+names:
            if (here/name).is_symlink():raise PermissionError('Curated package symlink forbidden')
        files.update(str((here/name).relative_to(root)) for name in names if (here/name).is_file())
        if any(not (here/name).is_file() for name in names):raise PermissionError('Curated package special file forbidden')
    return files

class BenchmarkGrant:
    def __init__(self,token,root,approval,systems,evaluator,authority):
        if token is not _TOKEN:raise PermissionError('Exact benchmark approval verifier required')
        self.root=Path(root).resolve(strict=True)
        self._approval=json.dumps(approval,sort_keys=True,allow_nan=False)
        self._systems=json.dumps(systems,sort_keys=True,allow_nan=False)
        self._evaluator=json.dumps(evaluator,sort_keys=True,allow_nan=False)
        self._authority=json.dumps(authority,sort_keys=True,allow_nan=False)
    @property
    def approval(self):return json.loads(self._approval)
    @property
    def systems(self):return json.loads(self._systems)
    @property
    def evaluator(self):return json.loads(self._evaluator)
    @property
    def authority_files(self):return json.loads(self._authority)
    def rows(self):
        a=self.approval
        raw=checked_json(self.root,a['TRAIN_subset'])
        rows=normalize_rows(raw,'train')
        actual=tuple((r['image_id'],r['sha256'],r['bytes']) for r in raw)
        if actual!=EXPECTED_TRAIN or tuple(r['image_id'] for r in rows)!=tuple(x[0] for x in EXPECTED_TRAIN):
            raise PermissionError('Only exact two prespecified TRAIN views admitted')
        return rows
    def read(self,row):
        if row not in self.rows():raise PermissionError('Unapproved TRAIN row')
        base=route(self.root,self.approval['payload_root']);p=route(base,row['path'])
        if p.stat().st_size!=next(x[2] for x in EXPECTED_TRAIN if x[0]==row['image_id']):raise ValueError('Exact TRAIN byte count required')
        data=p.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('TRAIN image checksum mismatch')
        return data
    def artifact_bindings(self):
        out=[self.systems['M5_source_configuration'],*self.authority_files,*self.evaluator['files']]
        for s in self.systems['systems'].values():
            for key in ('probe','probe_metadata','temperature_record'):
                if key in s:out.append(s[key])
            b=s['selected_backbone'];out.extend(b if isinstance(b,list) else [b])
        return list({b['path']:b for b in out}.values())
    def reverify(self,*,composition=True):
        """Re-hash every bound file; by default also prove exact package composition.

        composition=False is only for code already running under the worker's
        filesystem sandbox, which grants directory listing to few directories.
        It is refused unless this same process passed a full composition check
        first (verify_request runs one before the sandbox is enforced). Bound
        files are still re-hashed, the sandbox prevents worker writes outside
        its output, and parent controllers repeat the full check after the run.
        """
        bindings=self.artifact_bindings()
        for binding in bindings:checked_file(self.root,binding)
        rows=self.rows()
        for row in rows:self.read(row)
        expected={str(route(self.root,x['path']).relative_to(self.root)) for x in bindings}
        base=route(self.root,self.approval['payload_root'])
        expected.update(str(route(base,r['path']).relative_to(self.root)) for r in rows)
        if composition:
            if strict_package_files(self.root)!=expected:raise PermissionError('Exact curated package composition required')
            self._composition_verified_pid=os.getpid()
        elif getattr(self,'_composition_verified_pid',None)!=os.getpid():
            raise PermissionError('Full pre-sandbox composition check required in this process')
        weights=set();parameters=set()
        for system in self.systems['systems'].values():
            bb=system['selected_backbone']
            if isinstance(bb,list):weights.update(v['path'] for v in bb)
            else:weights.add(bb['path'])
            parameters.update(system[key]['path'] for key in ('probe','probe_metadata','temperature_record') if key in system)
        images={str(route(base,r['path']).relative_to(self.root)) for r in rows}
        code=expected-weights-parameters-images
        if sum((self.root/x).stat().st_size for x in code)>CAPS['payload_code_config_bytes']:
            raise PermissionError('Curated code/config4MiB cap')
        if sum((self.root/x).stat().st_size for x in parameters)>CAPS['selected_parameters_bytes']:
            raise PermissionError('Selected parameters256KiB cap')



def verify_request(root,binding):
    a=checked_json(root,binding)
    if (a.get('status')!='user_approved' or not a.get('user_reference') or a.get('scope')!='M6_TRAIN_CPU_BENCHMARK'
            or a.get('compute_authorized') is not True or a.get('payload_authorized') is not True
            or a.get('cloud') is not True or a.get('GPU') is not False or a.get('held_out_access') is not False
            or a.get('actual_lock_authorized') is not False or a.get('remote_install_authorized') is not False
            or a.get('device')!='cpu' or a.get('caps')!=CAPS or a.get('resource_limits')!=LIMITS
            or a.get('trusted_worker_local_IPC_acknowledged') is not True):
        raise PermissionError('Fresh exact TRAIN-only CPU approval required; no GPU/unseal/install/lock')
    identity=a.get('destination_identity')
    if not isinstance(identity,dict) or set(identity)!={'org','teamspace','studio_id','studio_name'} or any(not isinstance(v,str) or not v.strip() for v in identity.values()):raise PermissionError('Resolved approved destination identity required')
    if a.get('source_TRAIN_manifest_SHA')!=TRAIN_MANIFEST_SHA:raise PermissionError('Original TRAIN lineage required')
    systems=checked_json(root,a['system_manifest']);evaluator=checked_json(root,a['evaluator_manifest'])
    accepted=checked_json(root,a['metric_acceptance'])
    if digest(systems.get('systems'))!=SCIENTIFIC_SYSTEMS_SHA:raise PermissionError('Approved scientific systems changed')
    runtime_specs(systems)
    if (accepted.get('status')!='user_accepted' or accepted.get('accepted_policy_SHA')!=digest(NUMERICAL_POLICY)
            or evaluator.get('numerical_policy_SHA')!=digest(NUMERICAL_POLICY)):
        raise PermissionError('Accepted unchanged metric policy required')
    if a['Linux_lock']['SHA']!=LINUX_LOCK_SHA:raise PermissionError('Original Linux hash lock required')
    checked_file(root,a['Linux_lock'])
    if a['distribution_plan']['SHA']!=DISTRIBUTION_PLAN_SHA:raise PermissionError('Original47-version plan changed')
    expected=checked_json(root,a['distribution_plan'])['expected_distribution_versions']
    if len(expected)!=47 or evaluator.get('expected_distribution_versions')!=expected:
        raise PermissionError('Original all47 distribution contract required')
    # No native model construction or evaluation data access occurs in verifier.
    g=BenchmarkGrant(_TOKEN,root,a,systems,evaluator,[binding,a['system_manifest'],a['evaluator_manifest'],a['metric_acceptance'],a['Linux_lock'],a['distribution_plan'],a['TRAIN_subset']])
    g.reverify();return g


def verify_provider(grant,binding,*,start_UTC,now):
    v=checked_json(grant.root,binding);start=timestamp(start_UTC);a=grant.approval
    required=['same_private_Studio','Free4CPU','manual_Sleep_available','user_supervising','no_topup','no_move_fee','storage_transfer_quote_verified','retention_permission','prior_project_cap_reconciled']
    if v.get('destination_identity')!=a.get('destination_identity'):raise PermissionError('Live provider identity differs from approved destination')
    if any(v.get(k) is not True for k in required) or not v.get('user_reference'):
        raise PermissionError('Current private Studio/FreeCPU/stop/retention evidence required')
    if not 0<=now-timestamp(v['verified_UTC'])<600:raise PermissionError('Provider evidence stale/future')
    if not 0<=now-start<CAPS['activation_seconds']:raise PermissionError('Original Start clock expired')
    for key in ['credits_per_hour','current_balance','project_credit_headroom','storage_reserve_credits','pending_charges_reserve_credits']:
        if type(v.get(key)) not in (int,float) or not math.isfinite(v[key]):raise PermissionError('Finite provider quote required')
    if (v['credits_per_hour']!=0 or v['current_balance']<.25 or v['project_credit_headroom']<.25 or v['storage_reserve_credits']<0
            or v['pending_charges_reserve_credits']<0 or v['storage_reserve_credits']+v['pending_charges_reserve_credits']>.25):
        raise PermissionError('FreeCPU or0.25all-in subcap not met')
    if type(v.get('persistent_used_bytes')) is not int or not 0<=v['persistent_used_bytes']<=CAPS['persistent_bytes']:
        raise PermissionError('Storage budget missing/exceeded')
    retention=min(start+86400,timestamp('2026-11-02T12:00:00Z'))
    if timestamp(a['retention_deadline_UTC'])!=retention or start+1800>retention:
        raise PermissionError('Exact renewed retention/window required; no deadline reset')
    if v.get('actual_Start_UTC')!=start_UTC:raise PermissionError('Provider Start mismatch')
    return v,start


def scientific_clock(*,mode,start_UTC,request_SHA,now,saved=None):
    start=timestamp(start_UTC)
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


def parse_host(meminfo,status):
    m={}
    for line in meminfo.splitlines():
        parts=line.split()
        if len(parts)==3 and parts[2]=='kB':m[parts[0].rstrip(':')]=int(parts[1])*1024
    keys=['MemTotal','MemAvailable','SwapTotal','SwapFree']
    if any(k not in m or m[k]<0 for k in keys) or m['SwapFree']>m['SwapTotal']:raise ValueError('Invalid Linux memory evidence')
    rss=[int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')]
    if len(rss)!=1:raise ValueError('Linux RSS evidence absent')
    return {'total_bytes':m['MemTotal'],'available_estimate_bytes':m['MemAvailable'],'swap_used_bytes':m['SwapTotal']-m['SwapFree'],'RSS_bytes':rss[0]}

def host(pid):
    return parse_host(Path('/proc/meminfo').read_text(),Path('/proc',str(pid),'status').read_text())

def child_host(child):
    def released():
        values=host(os.getpid());values['RSS_bytes']=0;values['child_exit_observed']=True;return values
    if child.poll() is not None:return released()
    try:return host(child.pid)
    except (FileNotFoundError,ValueError):
        if child.poll() is not None:return released()
        raise # A live worker with missing evidence must not become RSS=0.


def size(root):
    seen=set();total=0
    for p in Path(root).rglob('*'):
        if p.is_symlink():raise PermissionError('Source/output symlink forbidden')
        if p.is_file():
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:total+=st.st_size;seen.add(key)
    return total

def storage_reason(growth,persistent,outputs):
    if growth>CAPS['growth_bytes']:return 'new_growth5GiB'
    if persistent+growth>CAPS['persistent_bytes']:return 'persistent42GB_decimal'
    if outputs>CAPS['outputs_bytes']:return 'outputs64MiB'
    return None


def check_vector(x,n):
    if len(x)!=n or any(not math.isfinite(v) for v in x):raise ValueError('Finite original feature/logit dimension required')

def compare(x,y,*,atol,rtol):
    if len(x)!=len(y) or any(not math.isclose(a,b,abs_tol=atol,rel_tol=rtol) for a,b in zip(x,y)):
        raise ValueError('Original reload/parity tolerance failed')
    return max(abs(a-b) for a,b in zip(x,y))


def benchmark_core(grant,backend_factory,output,*,check=lambda:None,reporting_device="cpu",reload_check=None):
    """Exactly two TRAIN forwards plus first-view reload, no fitting/scoring."""
    if reporting_device not in ('cpu','cuda:0'):raise PermissionError('Explicit benchmark reporting device required')
    if reporting_device=='cuda:0' and grant.approval.get('scope')!='TRAIN_CPU_GPU_PARITY_V1':raise PermissionError('Separate GPU parity grant required')
    if not isinstance(grant,BenchmarkGrant):raise PermissionError('Verified benchmark grant required')
    # Runs under the worker sandbox: composition was proven before enforcement.
    grant.reverify(composition=False);rows=grant.rows();output=Path(output).resolve()
    if not output.is_relative_to(grant.root/'benchmark_control'):raise PermissionError('Only benchmark-control output destination allowed')
    output.mkdir();began=time.monotonic()
    from .arrays import portable_probe,probe_logits
    reference=[];stages=[];counts={'dense':0,'med':0};backend=None
    try:
        for generation in range(2):
            probes={name:portable_probe(checked_file(grant.root,s['probe'])) for name,s in grant.systems['systems'].items() if name!='dense7'}
            check();started=time.monotonic();backend=backend_factory(grant,check);setup=time.monotonic()-started
            for row in rows if generation==0 else rows[:1]:
                check();payload=grant.read(row);counts['dense']+=1
                if counts['dense']>3:raise RuntimeError('Dense forward cap violated before call')
                started=time.monotonic();z,dense=backend.dense(payload);dense_time=time.monotonic()-started
                check();counts['med']+=1
                if counts['med']>3:raise RuntimeError('Med forward cap violated before call')
                started=time.monotonic();med=backend.med(payload);med_time=time.monotonic()-started
                if counts['dense']>3 or counts['med']>3:raise RuntimeError('Forward cap violated')
                check_vector(z,7);check_vector(dense,1024);check_vector(med,1152)
                if not math.isclose(math.fsum(x*x for x in med),1.,abs_tol=1e-5,rel_tol=1e-5):raise ValueError('Original L2 feature contract failed')
                logits={'dense7':z,'med7':probe_logits(probes['med7'],med),'dense2':probe_logits(probes['dense2'],dense),'med2':probe_logits(probes['med2'],med)}
                record={'image_id':row['image_id'],'dense_features':dense,'med_features':med,'logits':logits}
                if generation==0:reference.append(record)
                else:
                    compare(dense,reference[0]['dense_features'],atol=1e-4,rtol=1e-4);compare(med,reference[0]['med_features'],atol=1e-4,rtol=1e-4)
                    compare(z,reference[0]['logits']['dense7'],atol=1e-4,rtol=1e-4)
                    for name in ('dense2','med7','med2'):
                        compare(logits[name],reference[0]['logits'][name],atol=1e-4,rtol=1e-4)
                        fixed=reference[0]['dense_features'] if name=='dense2' else reference[0]['med_features']
                        compare(probe_logits(probes[name],fixed),reference[0]['logits'][name],atol=1e-12,rtol=1e-12)
                    if reload_check is not None:reload_check(reference[0],record)
                stages.append({'generation':generation,'image_id':row['image_id'],'dense_seconds':dense_time,'med_seconds':med_time})
            state=backend.finish();check()
            if state.get('frozen_state_unchanged') is not True or state.get('RNG_unchanged') is not True:raise ValueError('Frozen model/RNG changed')
            stages.append({'generation':generation,'setup_seconds':setup,'state':state});backend.close();backend=None
        grant.reverify(composition=False);check()
        result={'status':'passed','purpose':'TRAIN-only compatibility/reload/throughput; no accuracy/calibration estimates','forward_counts':counts,'reference':reference,'stages':stages,'wall_seconds':time.monotonic()-began,'GPU':reporting_device=='cuda:0','device':reporting_device,'held_out_access':False,'lock_written':False,'provider_Sleep_verified':False}
        write_once(output/'benchmark.json',result);return result
    except BaseException as exc:
        write_once(output/'failure.json',{'status':'failed','error_type':type(exc).__name__,'message':str(exc),'forward_counts':counts,'no_retry_authorized':True});raise
    finally:
        if backend is not None:backend.close()


def synthetic_metric_timing(check=lambda:None):
    """Invented population only, separate from TRAIN outputs and real outcomes."""
    ids=tuple('invented-'+str(i) for i in range(1117));labels=tuple(i%7 for i in range(1117));z=tuple(tuple(2. if c==y else 0. for c in range(7)) for y in labels)
    a={'ids':ids,'labels':labels,'clusters':ids,'logits':z,'temperature':1.}
    started=time.monotonic();paired_intervals(a,a,partition='synthetic',check=check)
    seven=time.monotonic()-started
    binary=dict(a,labels=tuple(i%2 for i in range(1117)),logits=tuple((2.,0.) if i%2==0 else (0.,2.) for i in range(1117)))
    started=time.monotonic();paired_intervals(binary,binary,partition='synthetic',check=check)
    return {'physical_units':1117,'replicates':1000,'seven_class_pair_seconds':seven,'binary_pair_seconds':time.monotonic()-started,'invented_data_only':True}
