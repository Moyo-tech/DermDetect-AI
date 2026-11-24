"""Invented vectors/receipts only, no scientific images, models or GPU."""
import copy,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from dermdetect.evaluation import gpu_parity as g,cloud_ham as c,authorization as a
from test_m6_integration import Fixture
from test_cloud_ham import contract


def refs():
 return [{'image_id':'invented'+str(i),'dense_features':[.2,.8],'med_features':[.6,.8],
          'logits':{'dense7':[1.,0.],'med7':[1.,0.],'dense2':[1.,0.],'med2':[1.,0.]}} for i in range(2)]
def specs():return {n:{'temperature':1.,'cutoffs':[.8,.6]} for n in refs()[0]['logits']}
class ParityTests(unittest.TestCase):
 def test_identical(self):self.assertEqual(g.compare_references(refs(),refs(),specs()),0.)
 def test_identity_change(self):
  r=refs();r[0]['image_id']='other'
  with self.assertRaises(ValueError):g.compare_references(refs(),r,specs())
 def test_numeric_difference(self):
  r=refs();r[0]['med_features'][0]+=.1
  with self.assertRaises(ValueError):g.compare_references(refs(),r,specs())
 def test_classification_flip_inside_tolerance_rejected(self):
  x=refs();y=refs();x[0]['logits']['dense2']=[0.,0.];y[0]['logits']['dense2']=[.00001,0.]
  with self.assertRaisesRegex(ValueError,'decision'):g.compare_references(x,y,specs())
 def test_exact_two_only(self):
  with self.assertRaises(ValueError):g.compare_references(refs()[:1],refs()[:1],specs())
 def test_cpu_approval_does_not_authorize_GPU(self):
  with tempfile.TemporaryDirectory() as tmp:
   f=Fixture(Path(tmp).resolve());contract(f)
   with self.assertRaises(PermissionError):c.validate_contract(f.approval,'cloud_gpu')
 def test_gpu_clean_environment(self):
  env=c.clean_environment('/invented','cuda:0');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0');self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8');self.assertNotIn('HF_TOKEN',env)
 def test_pending_parity_refused_before_model_or_data(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'request.json';p.write_text(json.dumps({'status':'pending'}));binding={'path':'request.json','SHA':a.file_sha(p)}
   with self.assertRaises(PermissionError):g.verify_request(root,binding)
 def test_direct_GPU_grant_requires_parity_before_dataset(self):
  with tempfile.TemporaryDirectory() as tmp:
   f=Fixture(Path(tmp).resolve(),'internal_test');contract(f)
   f.approval.update(scope='HAM_CLOUD_GPU_EVALUATION_V1',device='cuda:0',GPU_authorized=True,CPU_GPU_parity_required=True,max_GPU_allocated_bytes=20*2**30,
      CPU_GPU_parity_receipt={'path':'missing_parity.json','SHA':'0'*64},CPU_GPU_parity_supervision={'path':'missing_supervision.json','SHA':'0'*64})
   f.request=f.put('request.json',f.approval);f.root.joinpath('dataset.json').unlink()
   with self.assertRaises(PermissionError):
    a.verify_grant(f.root,run_approval=f.request,system_manifest=f.system,evaluator_manifest=f.eval,acceptance=f.accept,lock=f.lock,lock_approval=f.lockapproval,execution_context='cloud_gpu')
 def test_reload_decision_flip_rejected(self):
  x=refs()[:1];y=refs()[:1];x[0]['logits']['dense2']=[0.,0.];y[0]['logits']['dense2']=[.00001,0.]
  with self.assertRaisesRegex(ValueError,'decision'):g.compare_references(x,y,specs(),expected_count=1)
 def test_receipt_bound_to_same_source_destination(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);request={'system_manifest':{'path':'system','SHA':'s'},'evaluator_manifest':{'path':'eval','SHA':'e'},'destination_id':'pod-A','max_wall_seconds':100,'provider_work_deadline_unix':time.time()+300}
   result=dict(request,status='complete',parity_passed=True,scope='TRAIN_CPU_GPU_PARITY_V1',held_out_access=False,HAM_projected_seconds=50)
   p=root/'parity.json';p.write_text(json.dumps(result));request['CPU_GPU_parity_receipt']={'path':'parity.json','SHA':a.file_sha(p)}
   s=root/'supervision.json';s.write_text(json.dumps({'status':'passed','worker_reaped':True,'source_integrity_after':True,'receipt_SHA':a.file_sha(p)}));request['CPU_GPU_parity_supervision']={'path':'supervision.json','SHA':a.file_sha(s)}
   g.verify_receipt_before_unseal(root,request)
   for k,v in [('destination_id','pod-B'),('max_wall_seconds',40),('evaluator_manifest',{}),('provider_work_deadline_unix',time.time()+30)]:
    bad=copy.deepcopy(request);bad[k]=v
    with self.subTest(k=k),self.assertRaises(PermissionError):g.verify_receipt_before_unseal(root,bad)

if __name__=='__main__':unittest.main()
