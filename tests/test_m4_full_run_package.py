"""Synthetic configuration, approval and controller guards; no GPU/network calls."""
import importlib.util,json,tempfile,time,unittest,datetime,sys,ast
from pathlib import Path
from dermdetect.models.full_run_contracts import validate_execution_approval,sha256_file,rooted_file
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supervisor',ROOT/'scripts/supervise_m4.py');controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
class FullPackageTests(unittest.TestCase):
 def fixture(self,tmp):
  c=json.loads((ROOT/'tests/fixtures/synthetic_full_run.json').read_text());c['configuration_decisions_accepted']=True
  cp=Path(tmp)/'config.json';cp.write_text(json.dumps(c))
  a={'full_execution_approved':True,'config_sha256':sha256_file(cp),'user_authority':'synthetic_only_no_execution','stages':['training','extraction'],'allocation_seconds':{'training':10800,'extraction':3600},'additional_existing_credit_cap':3.30,'same_private_Studio':True,'manual_Sleep_required':True,'local_IPC_residual_risk_accepted':True,'held_out_authorized':False,'persistent_storage_max_bytes':42*10**9}
  ap=Path(tmp)/'approval.json';ap.write_text(json.dumps(a));return cp,ap,c,a
 def remote_stage_body(self):
  tree=ast.parse((ROOT/'scripts/stage_m4_full.py').read_text())
  bodies=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('from pathlib import Path') and "for e in T['entries']" in n.value]
  self.assertEqual(len(bodies),1)
  return bodies[0]
 def test_actual_generated_remote_body_and_all_new_scripts_compile(self):
  for n in ['train_m4.py','extract_m4.py','supervise_m4.py','stage_m4_full.py','prepare_m4_full_package.py','build_m4_full_code.py']:
   compile((ROOT/'scripts'/n).read_text(),n,'exec')
  compile(self.remote_stage_body().replace('ROOT',repr('/invented/task')),'actual_remote_stage_wrapper','exec')
 def test_actual_remote_reuse_creates_hard_link_without_duplicate_bytes(self):
  import hashlib,io,contextlib
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);(r/'payload/data').mkdir(parents=True);(r/'full_package').mkdir()
   source=r/'payload/data/invented.payload';source.write_bytes(b'invented-not-image')
   e={'kind':'development_image','action':'verify_existing_and_link','existing_remote_source':'data/invented.payload','destination':'data/train/invented.payload','bytes':source.stat().st_size,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
   (r/'full_package/transfer_manifest.json').write_text(json.dumps({'entries':[e]}))
   with contextlib.redirect_stdout(io.StringIO()):exec(compile(self.remote_stage_body().replace('ROOT',repr(str(r))),'remote_stage','exec'),{})
   dest=r/'full_payload/data/train/invented.payload';self.assertEqual(source.stat().st_ino,dest.stat().st_ino);self.assertEqual(controller.own_output_bytes(r/'full_payload'),source.stat().st_size)
   import os
   os.link(dest,r/'full_payload/second-alias.payload');self.assertEqual(controller.own_output_bytes(r/'full_payload'),source.stat().st_size)
 def test_storage_ceiling_and_quote_fail_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   cp,ap,c,a=self.fixture(tmp);a['persistent_storage_max_bytes']=43*10**9;ap.write_text(json.dumps(a))
   with self.assertRaises(ValueError):validate_execution_approval(cp,ap,stage='training')
  g={'available_RAM_min_bytes':4,'GPU_used_max_bytes':20,'swap_growth_max_bytes':128,'RSS_max_fraction':.75,'free_disk_min_bytes':8,'persistent_storage_max_bytes':42}
  sample={'host':{'available':8,'total':32,'RSS':4,'swap':0},'GPU':{'used_bytes':5},'disk_free':16,'persistent_used_estimate':43}
  self.assertEqual(controller.monitor_reason(sample,{'swap':0},g,1,600),'persistent_storage')
 def test_exact_config_and_stage_binding(self):
  with tempfile.TemporaryDirectory() as tmp:
   cp,ap,c,a=self.fixture(tmp);self.assertEqual(validate_execution_approval(cp,ap,stage='training')[0],c)
   cp.write_text(cp.read_text()+' ')
   with self.assertRaises(PermissionError):validate_execution_approval(cp,ap,stage='training')
 def test_unaccepted_decisions_and_scope_fail_closed(self):
  with tempfile.TemporaryDirectory() as tmp:
   cp,ap,c,a=self.fixture(tmp);c['configuration_decisions_accepted']=False;cp.write_text(json.dumps(c));a['config_sha256']=sha256_file(cp);ap.write_text(json.dumps(a))
   with self.assertRaises(PermissionError):validate_execution_approval(cp,ap,stage='extraction')
   c['configuration_decisions_accepted']=True;c['training']['batch_size']=32;cp.write_text(json.dumps(c));a['config_sha256']=sha256_file(cp);ap.write_text(json.dumps(a))
   with self.assertRaises(ValueError):validate_execution_approval(cp,ap,stage='training')
 def test_relative_file_rejects_escape_and_symlink(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);(r/'inside').write_text('invented');self.assertEqual(rooted_file(r,'inside'),r.resolve()/'inside')
   (r/'alias').symlink_to(r/'inside')
   for n in ['../outside','alias','/tmp/anything']:
    with self.assertRaises(ValueError):rooted_file(r,n)
 def test_resource_and_shutdown_guards(self):
  g={'available_RAM_min_bytes':4,'GPU_used_max_bytes':20,'swap_growth_max_bytes':128,'RSS_max_fraction':.75,'free_disk_min_bytes':8,'persistent_storage_max_bytes':42}
  s={'host':{'available':8,'total':32,'RSS':4,'swap':0},'GPU':{'used_bytes':5},'disk_free':16,'persistent_used_estimate':17};base={'swap':0}
  self.assertIsNone(controller.monitor_reason(s,base,g,1,600));self.assertEqual(controller.monitor_reason(s,base,g,1,300),'allocation_checkpoint_reserve')
  s['host']['swap']=129;self.assertEqual(controller.monitor_reason(s,base,g,1,600),'swap_growth')
 def test_resume_preserves_paid_history_and_requires_Sleep_proof(self):
  stamp=datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(hours=2)
  start=stamp.isoformat();end=(stamp+datetime.timedelta(hours=1)).isoformat();out='/invented/task'
  ledger={'stages':[{'stage':'training','start_UTC':start,'output':out,'allocation_seconds':10800,'Sleeping_pending':True}],'reserved_seconds':{'training':10800}}
  a={'allocation_seconds':{'training':10800}}
  with self.assertRaises(PermissionError):controller.reconcile_allocation(ledger,{},a,'training',out,resume=True)
  c={'resume_closure':{'user_confirmed_Sleeping':True,'user_authority':'synthetic_attestation','previous_start_UTC':start,'previous_output':out,'Sleeping_UTC':end}}
  remaining,closed=controller.reconcile_allocation(ledger,c,a,'training',out,resume=True);self.assertEqual(remaining,7200);self.assertEqual(closed['conservative_paid_seconds'],3600);self.assertFalse(closed['billing_release_verified'])
 def test_budget_rate_and_manual_stop_gates(self):
  with tempfile.TemporaryDirectory() as tmp:
   _,_,_,a=self.fixture(tmp)
   c={k:True for k in ['same_GCP','one_L4','non_interruptible','no_move_fee','manual_Sleep_available','no_topup']};c.update(credits_per_hour=.79,current_balance=10,persistent_used_bytes=17*10**9,storage_and_transfer_reserve_credits=.10,verified_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
   self.assertEqual(controller.check_provider(c,a,'training'),10800)
   c['persistent_used_bytes']=41*10**9
   with self.assertRaises(ValueError):controller.check_provider(c,a,'training')
   c['persistent_used_bytes']=17*10**9;c['storage_and_transfer_reserve_credits']=.11
   with self.assertRaises(ValueError):controller.check_provider(c,a,'training')
   c['storage_and_transfer_reserve_credits']=.10
   c['manual_Sleep_available']=False
   with self.assertRaises(PermissionError):controller.check_provider(c,a,'training')
if __name__=='__main__':unittest.main()
