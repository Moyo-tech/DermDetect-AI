"""Invented contracts/data only. No actual image/model/held-out source access."""
import copy,importlib.util,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from dermdetect.evaluation import authorization as a,cloud_ham as c
from test_m6_integration import Fixture
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cloudrunner',ROOT/'scripts/run_cloud_ham_evaluation.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def contract(f):
 f.approval.update(scope='HAM_CLOUD_CPU_EVALUATION_V1',cloud=True,payload_authorized=True,
    trusted_process_no_OS_isolation_acknowledged=True,resource_allocation_verified=True,
    no_credentials_in_workload_verified=True,provider_stop_supervision_verified=True,
    memory_accounting='dedicated_host',whole_host_exclusive_verified=True,destination_id='INVENTED',provider='INVENTED',resource_verification_reference='INVENTED',
    transfer_approval_reference='INVENTED',provider_work_deadline_unix=time.time()+300,
    max_total_usd=.25,currency='USD',dedicated_memory_bytes=16*2**30)
 f.request=f.put('request.json',f.approval)

def grant(f):
 return a.verify_grant(f.root,run_approval=f.request,system_manifest=f.system,evaluator_manifest=f.eval,
    acceptance=f.accept,lock=f.lock if f.partition=='internal_test' else None,
    lock_approval=f.lockapproval if f.partition=='internal_test' else None,execution_context='cloud_cpu')

class CloudContracts(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(Path(self.tmp.name).resolve());contract(self.f)
 def test_cloud_grant_opt_in(self):self.assertEqual(grant(self.f).partition,'validation')
 def test_existing_local_entry_refuses_cloud(self):
  with self.assertRaises(PermissionError):self.f.grant()
 def test_each_required_attestation(self):
  for k in ['payload_authorized','trusted_process_no_OS_isolation_acknowledged','resource_allocation_verified','no_credentials_in_workload_verified','provider_stop_supervision_verified']:
   v=copy.deepcopy(self.f.approval);v[k]=False
   with self.subTest(k=k),self.assertRaises(PermissionError):c.validate_contract(v)
 def test_bad_scope_device_deadline_and_money(self):
  for k,v in [('scope','RUNPOD_TRAIN_BENCHMARK'),('device','cuda'),('provider_work_deadline_unix',0),('max_total_usd',float('nan')),('max_wall_seconds',86401),('dedicated_memory_bytes',8*2**30)]:
   d=copy.deepcopy(self.f.approval);d[k]=v
   with self.subTest(k=k),self.assertRaises(PermissionError):c.validate_contract(d)
 def test_clean_environment_excludes_credentials(self):
  e=c.clean_environment(Path('/invented'));self.assertEqual(e['HF_HUB_OFFLINE'],'1');self.assertEqual(e['CUDA_VISIBLE_DEVICES'],'');self.assertNotIn('HF_TOKEN',e);self.assertNotIn('HTTPS_PROXY',e)
 def test_pending_refused_before_dataset(self):
  self.f.changed('status','pending_NOT_AUTHORIZED');self.f.root.joinpath('dataset.json').unlink()
  with self.assertRaisesRegex(PermissionError,'Explicit run approval'):grant(self.f)
 def test_heldout_no_lock_refused_before_dataset(self):
  self.f.changed('partition','internal_test');self.f.changed('held_out_access',True);self.f.root.joinpath('dataset.json').unlink()
  with self.assertRaisesRegex(PermissionError,'Separate scoped unseal'):grant(self.f)
 def test_invented_heldout_requires_real_bound_lock(self):
  with tempfile.TemporaryDirectory() as tmp:
   f=Fixture(Path(tmp).resolve(),'internal_test');contract(f);self.assertEqual(grant(f).lock_SHA,f.lock['SHA'])
 def test_output_no_retry(self):
  g=grant(self.f);out=runner.output_path(g);out.mkdir()
  with self.assertRaises(PermissionError):runner.output_path(g)
 def test_source_tamper_rejected(self):
  self.f.root.joinpath('source.py').write_text('CHANGED')
  with self.assertRaisesRegex(PermissionError,'integrity'):grant(self.f)
 def test_readonly_memory_sampler(self):
  def text(p):
   return 'MemTotal: 18000000 kB\nMemAvailable: 9000000 kB\nSwapTotal: 1000 kB\nSwapFree: 800 kB\n' if str(p)=='/proc/meminfo' else 'VmRSS: 2000000 kB\n'
  with patch.object(Path,'read_text',text):r=c.sample(123,self.f.approval)
  self.assertEqual(r['RSS_bytes'],2048000000);self.assertEqual(r['available_estimate_bytes'],9000000*1024);self.assertEqual(r['swap_used_bytes'],204800)

class MemoryControllers(unittest.TestCase):
 def exercise(self,mode,files):
  def text(p):
   if str(p) not in files:raise FileNotFoundError(str(p))
   return files[str(p)]
  with patch.object(Path,'read_text',text):return c.cgroup_memory(mode)
 def v2(self):return {'/proc/self/cgroup':'0::/job\n','/proc/self/mountinfo':'12 1 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n','/sys/fs/cgroup/job/memory.max':str(16*2**30),'/sys/fs/cgroup/job/memory.current':'1000','/sys/fs/cgroup/job/memory.swap.current':'200'}
 def test_v2(self):self.assertEqual(self.exercise('cgroup_v2',self.v2()),(16*2**30,1000,200))
 def test_v1(self):
  f={'/proc/self/cgroup':'5:memory:/job\n','/proc/self/mountinfo':'12 1 0:1 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n','/sys/fs/cgroup/memory/job/memory.limit_in_bytes':str(16*2**30),'/sys/fs/cgroup/memory/job/memory.usage_in_bytes':'1000','/sys/fs/cgroup/memory/job/memory.memsw.usage_in_bytes':'1200'}
  self.assertEqual(self.exercise('cgroup_v1',f),(16*2**30,1000,200))
 def test_unlimited_refused(self):
  f=self.v2();f['/sys/fs/cgroup/job/memory.max']='max'
  with self.assertRaises(PermissionError):self.exercise('cgroup_v2',f)
 def test_missing_swap_refused(self):
  f=self.v2();del f['/sys/fs/cgroup/job/memory.swap.current']
  with self.assertRaises(FileNotFoundError):self.exercise('cgroup_v2',f)
 def test_nested_mount(self):
  f=self.v2();f['/proc/self/mountinfo']='12 1 0:1 /job /sys/fs/cgroup rw - cgroup2 cgroup rw\n'
  for k in list(f):
   if '/job/memory.' in k:f[k.replace('/job/','/')]=f.pop(k)
  self.assertEqual(self.exercise('cgroup_v2',f),(16*2**30,1000,200))
 def test_tighter_parent_headroom(self):
  f=self.v2();f.update({'/sys/fs/cgroup/memory.max':str(8*2**30),'/sys/fs/cgroup/memory.current':str(7*2**30),'/sys/fs/cgroup/memory.swap.current':'400'})
  self.assertEqual(self.exercise('cgroup_v2',f),(8*2**30,7*2**30,200))
 def test_no_matching_mount_refused(self):
  f=self.v2();f['/proc/self/mountinfo']=''
  with self.assertRaises(PermissionError):self.exercise('cgroup_v2',f)

class CloudOutputPaths(unittest.TestCase):
 def test_model_and_payload_subdirectories_refused(self):
  for name in ['med7/new_result','payload/new_result']:
   with tempfile.TemporaryDirectory() as tmp:
    f=Fixture(Path(tmp).resolve());contract(f);f.changed('output',str(f.root/name))
    with self.assertRaises(PermissionError):runner.output_path(grant(f))
 def test_host_accounting_requires_whole_host(self):
  with tempfile.TemporaryDirectory() as tmp:
   f=Fixture(Path(tmp).resolve());contract(f);f.approval['whole_host_exclusive_verified']=False
   with self.assertRaises(PermissionError):c.validate_contract(f.approval)

if __name__=='__main__':unittest.main()
