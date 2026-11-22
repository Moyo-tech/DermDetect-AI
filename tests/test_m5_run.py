"""Invented metadata, tensors and pixels only; no research artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import sys
import numpy as np
import torch
from torch import nn
from sklearn.exceptions import ConvergenceWarning
from dermdetect.models.m5_run import POLICY, membership, run, fitted, candidate_task
from dermdetect.models.m5_run import calibration_scores


def fixture():
    rng=np.random.default_rng(20260729); rows=[]; x=[]
    for split in ['train','validation']:
        for y in range(7):
            for j in range(4):
                rows.append(dict(image_id=f'{split}-{y}-{j}',physical_id=f'{split}-{y}-{j}',target=y,split=split))
                # Deliberately noisy validation prevents degenerate calibration.
                z=np.eye(7)[y if split=='train' or j<2 else (y+1)%7]+rng.normal(0,.2,7)
                x.append(z)
    x=np.asarray(x); vr=[r for r in rows if r['split']=='validation']
    ref={'physical_ids':sorted(r['physical_id'] for r in vr),'logits':(x[len(x)//2:]*2).tolist()}
    return rows,x,ref


class M5Tests(unittest.TestCase):
    def test_stage_profile_only_narrows_output_write_permission(self):
        m=self.script('supervise_m5_local.py');base='(deny network*)\n(deny file-write* (require-not (subpath "/tmp/output")))\n'
        value=m.stage_profile(base,Path('/tmp/output'),Path('/tmp/output/fitting'))
        self.assertIn('(deny network*)',value);self.assertIn('(subpath "/tmp/output/fitting")',value)
        self.assertNotIn('(subpath "/tmp/output")',value)
        with self.assertRaises(ValueError):m.stage_profile(base,Path('/wrong'),Path('/tmp/output/fitting'))
    def test_staging_requires_authority_before_any_payload(self):
        m=self.script('prepare_m5_inputs.py')
        with tempfile.TemporaryDirectory() as d:
            D=Path(d);(D/'c.json').write_text('{}');(D/'a.json').write_text('{}')
            with patch.object(sys,'argv',['stage','--configuration',str(D/'c.json'),'--approval',str(D/'a.json')]):
                with self.assertRaisesRegex(ValueError,'approval'):m.main()
    def test_calibration_metrics_match_equal_unit_arithmetic(self):
        x=np.array([[0.,0.],[0.,0.]]);r=calibration_scores(x,[0,1],1.)
        self.assertAlmostEqual(r['NLL'],np.log(2));self.assertAlmostEqual(r['Brier'],.5)
        self.assertEqual(sum(b['count'] for b in r['ECE_bins']),2);self.assertAlmostEqual(r['ECE'],0.)
    def script(self,name):
        p=Path(__file__).resolve().parents[1]/'scripts'/name
        spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_generated_entry_fitting_invented_inputs(self):
        from dermdetect.models.embeddings import save_cache
        from dermdetect.models.m5_run import write
        m=self.script('m5_cpu_worker.py');rows,x,ref=fixture();order=np.argsort([r['image_id'] for r in rows]);rows=[rows[i] for i in order];x=x[order]
        with tempfile.TemporaryDirectory() as d:
            D=Path(d);inputs={};hashes={}
            for s in ['train','validation']:
                p=D/(s+'.json');write(p,[r for r in rows if r['split']==s]);inputs[s]=str(p);hashes[s]=m.file_sha256(p)
            ids=[r['image_id'] for r in rows];key={'image_sha256':{i:'0'*64 for i in ids},'dimension':7}
            normalized=x/np.linalg.norm(x,axis=1)[:,None];hashes['med_metadata']=save_cache(D/'med',ids,normalized,key)
            inputs['med_cache']=str(D/'med');inputs['checkpoint']=str(D/'selected.pt')
            val=[r for r in rows if r['split']=='validation'];z=x[[i for i,r in enumerate(rows) if r['split']=='validation']]*2
            reference={'physical_ids':[r['physical_id'] for r in val],'logits':z.tolist()}
            torch.save(dict(configuration={},binding={},membership={s:[r for r in rows if r['split']==s] for s in ['train','validation']},
                            state={'best_checkpoint':'best_epoch_021.pt','best_key':[0,0,21]},validation_reference=reference),inputs['checkpoint'])
            hashes['checkpoint']=m.file_sha256(Path(inputs['checkpoint']));dense=D/'dense';dense.mkdir();inputs['dense_cache']=str(dense)
            c={'policy':POLICY,'inputs':inputs,'hashes':hashes,'software':{},'Dense_training_configuration':{},'Dense_binding':{},
               'Med_binding':key,'Dense_feature_binding':{'invented':True},'budgets_seconds':{'fitting':30}}
            cp=D/'configuration.json';write(cp,c);cfgsha=m.file_sha256(cp)
            dx=np.tile(x,(1,147))[:,:1024].astype(np.float32);np.save(dense/'features.npy',dx,allow_pickle=False)
            write(dense/'cache.json',dict(binding=c['Dense_feature_binding'],ids=ids,features_SHA=m.file_sha256(dense/'features.npy')))
            write(dense/'cache_receipt.json',dict(source_configuration_SHA=cfgsha,metadata_SHA=m.file_sha256(dense/'cache.json')))
            ap=D/'approval.json';write(ap,dict(status='user_approved',user_reference='invented_test_authority',configuration_SHA=cfgsha,
                                             stages=['fitting'],held_out_access=False,cloud_or_GPU=False))
            with patch.object(sys,'argv',['m5_cpu_worker','--configuration',str(cp),'--approval',str(ap),'--stage','fitting','--output',str(D/'run')]):m.main()
            self.assertEqual(json.loads((D/'run/result.json').read_text())['status'],'passed')

    def test_guardian_resource_limits_and_cap(self):
        m=self.script('supervise_m5_local.py');base={'RSS_bytes':1,'available_estimate_bytes':8*1024**3,'swap_used_bytes':0}
        self.assertIsNone(m.reason(base,0,9*1024**3));self.assertEqual(sum(m.CAPS.values()),7020)
        for k,v in [('RSS_bytes',7*1024**3),('available_estimate_bytes',3*1024**3),('swap_used_bytes',257*1024**2)]:
            s=dict(base);s[k]=v;self.assertIsNotNone(m.reason(s,0,9*1024**3))
        self.assertEqual(m.reason(base,0,7*1024**3),'free_disk8GiB')

    def test_guardian_rejects_missing_authority_before_launch(self):
        m=self.script('supervise_m5_local.py')
        with tempfile.TemporaryDirectory() as d:
            D=Path(d);(D/'c.json').write_text(json.dumps({'budgets_seconds':dict(m.CAPS,total=7200)}));(D/'a.json').write_text('{}')
            argv=['guardian']
            for k,v in dict(configuration=D/'c.json',approval=D/'a.json',stage='benchmark',boundary='absent',manifest='absent',output=D/'out',ledger=D/'ledger',python=sys.executable).items():argv+=['--'+k,str(v)]
            with patch.object(sys,'argv',argv),patch('subprocess.Popen') as proc:
                with self.assertRaisesRegex(ValueError,'approval'):m.main()
                proc.assert_not_called()

    def test_complete_finite_pipeline_and_reload(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            result=run(rows,ids,x,ids,x[:,:6],ref,Path(d)/'run',policy=POLICY,max_seconds=30)
            self.assertEqual(result['primary_candidate_fits'],12);self.assertEqual(result['diagnostic_fits'],3)
            self.assertEqual(set(result['decisions']),{'dense7','med7','dense2','med2'})
            for name in ['med7','med2','dense2']:
                trials=json.loads((Path(d)/'run'/name/'candidates.json').read_text())
                self.assertEqual([r['C'] for r in trials],POLICY['C_grid'])
                self.assertTrue(all(r['reload_max_abs_difference']==0 for r in trials))
            self.assertEqual(result['decisions']['dense2']['binary_classification']['equality'],'malignant')

    def test_membership_rejects_heldout_duplicate_and_cross_physical(self):
        rows,_,_=fixture()
        for kind in ['heldout','duplicate','cross','group','cluster','label']:
            altered=copy.deepcopy(rows)
            if kind=='heldout':altered[0]['split']='internal_test'
            if kind=='duplicate':altered[1]['image_id']=altered[0]['image_id']
            if kind=='cross':altered[-1]['physical_id']=altered[0]['physical_id']
            if kind=='group':altered[-1]['split_group_id']='g';altered[0]['split_group_id']='g'
            if kind=='cluster':altered[-1]['cluster_id']='g';altered[0]['cluster_id']='g'
            if kind=='label':altered[1]['physical_id']=altered[0]['physical_id'];altered[1]['target']=1
            with self.assertRaises(ValueError,msg=kind):membership(altered)

    def test_cache_order_and_policy_fail_before_output(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            for bad_ids,p in [(ids[::-1],POLICY),(ids,dict(POLICY,C_grid=[1]))]:
                with self.assertRaises(ValueError):run(rows,bad_ids,x,ids,x,ref,Path(d)/'out',policy=p,max_seconds=30)
            self.assertFalse((Path(d)/'out').exists())

    def test_reference_ids_reject_and_preserve_failure(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows];ref['physical_ids'].reverse()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):run(rows,ids,x,ids,x,ref,Path(d)/'out',policy=POLICY,max_seconds=30)
            self.assertTrue((Path(d)/'out/failure.json').exists())

    def test_convergence_warning_stops_without_retry(self):
        def bad(*args,**kwargs):warnings.warn('invented nonconvergence',ConvergenceWarning)
        with patch('dermdetect.models.m5_run.Probe.fit',bad):
            with self.assertRaisesRegex(ValueError,'convergence'):fitted(1,np.eye(2),[0,1],['u','v'],2)

    def test_deadline_failure_preserved(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TimeoutError):run(rows,ids,x,ids,x,ref,Path(d)/'out',policy=POLICY,max_seconds=1e-12)
            self.assertTrue((Path(d)/'out/failure.json').exists())

    def test_temperature_bound_failure_retains_candidates(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d,patch('dermdetect.models.m5_run.fit_temperature',return_value={'at_bound':True}):
            with self.assertRaisesRegex(ValueError,'bound'):run(rows,ids,x,ids,x,ref,Path(d)/'out',policy=POLICY,max_seconds=30)
            self.assertTrue((Path(d)/'out/med7/candidates.json').exists())

    def test_missing_class_and_degenerate_feature_fail(self):
        rows,x,ref=fixture();ids=[r['image_id'] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):run(rows,ids,np.zeros_like(x),ids,x,ref,Path(d)/'out',policy=POLICY,max_seconds=30)
            fewer=[r for r in rows if r['target']!=6]
            with self.assertRaises(ValueError):candidate_task('small',fewer,x[:len(fewer)],7,Path(d)/'small',lambda:None)

    def test_worker_approval_before_scientific_path(self):
        script=Path(__file__).resolve().parents[1]/'scripts/m5_cpu_worker.py'
        spec=importlib.util.spec_from_file_location('m5_entry',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        with tempfile.TemporaryDirectory() as d:
            c=Path(d)/'configuration.json';a=Path(d)/'approval.json'
            c.write_text(json.dumps({'policy':POLICY,'inputs':{'checkpoint':'NO_SUCH_SCIENTIFIC_INPUT'}}));a.write_text('{}')
            with self.assertRaisesRegex(ValueError,'approval'):m.authorized(c,a,'fitting')

    def test_frozen_feature_extraction_invented_pixels_and_hash_stop(self):
        script=Path(__file__).resolve().parents[1]/'scripts/m5_cpu_worker.py';spec=importlib.util.spec_from_file_location('m5_features',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        from PIL import Image
        class Tiny(nn.Module):
            def __init__(self):super().__init__();self.bn=nn.BatchNorm2d(3);self.drop=nn.Dropout(.2)
            def features(self,x):return self.drop(self.bn(x)).mean((2,3)).repeat(1,342)[:,:1024]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invented.png';Image.new('RGB',(12,12),(100,80,50)).save(p)
            rows=[dict(path=p.name,sha256=m.file_sha256(p))];z=m.extract(Tiny(),rows,Path(d),lambda:None)
            self.assertEqual(z.shape,(1,1024));self.assertEqual(z.dtype,np.float32)
            rows[0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'hash'):m.extract(Tiny(),rows,Path(d),lambda:None)

if __name__=='__main__':unittest.main()
