"""Invented files/weights/labels/consent only: never read real held-out artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from dermdetect.evaluation import authorization as a, engine as e, metrics as m, pipeline as p, supervision as s
from test_m6_evaluation import npy


class Fixture:
    def __init__(self,root,partition='validation'):
        self.root=Path(root);self.partition=partition
        self.code=self.put('source.py',b'# INVENTED SOURCE\n')
        self.config=self.put('config.json',{})
        self.rows=[];self.root.joinpath('payload').mkdir()
        for i,label in enumerate(e.CLASSES):
            data=('INVENTED IMAGE BYTES '+str(i)).encode();self.put('payload/'+str(i),data)
            self.rows.append(dict(image_id='i'+str(i),physical_id='p'+str(i),cluster_id='c'+str(i),label=label,sha256=hashlib.sha256(data).hexdigest(),path=str(i),partition=partition,active=True))
        self.dataset=self.put('dataset.json',self.rows)
        systems={}
        for name in e.SYSTEMS:
            k=7 if name.endswith('7') else 2;backbone=self.put(name+'/backbone',b'INVENTED WEIGHTS')
            system=dict(classes=list(e.CLASSES) if k==7 else ['benign','malignant'],source_binary_mapping=e.BINARY if k==2 else None,
                        binary_classification={'equality':'malignant','malignant_probability':.5} if k==2 else None,
                        temperature={'temperature':1.},confidence_rules=[{'target_coverage':q,'cutoff':c,'tie_policy':'include_all_ties'} for q,c in [(.9,.5),(.8,.8)]],
                        selected_backbone=backbone if name.startswith('dense') else [backbone],temperature_record=self.put(name+'/temperature.json',{'temperature':1.}))
            if name!='dense7':
                state={'mean':([0,0],(2,)),'scale':([1,1],(2,)),'variance':([1,1],(2,)),
                       'coefficients':(([1.,0.] if k==2 else [float(c==j) for c in range(7) for j in range(2)]),(1,2) if k==2 else (7,2)),
                       'intercept':([0.]*(1 if k==2 else k),(1,) if k==2 else (k,)),'classes':(list(range(k)),(k,))}
                path=self.root/name/'probe.npz'
                with zipfile.ZipFile(path,'w') as z:
                    for key,(values,shape) in state.items():z.writestr(key+'.npy',npy(values,shape,'<i8' if key=='classes' else '<f8'))
                system['probe']=self.bind(path);system['probe_metadata']=self.put(name+'/probe.json',{'sha256':a.file_sha(path)})
            systems[name]=system
        self.system=self.put('systems.json',{'systems':systems,'M5_source_configuration':self.config})
        self.eval=self.put('evaluator.json',{'files':[self.code],'runtime':{'invented':True},'numerical_policy_SHA':e.digest(m.NUMERICAL_POLICY)})
        self.accept=self.put('acceptance.json',{'status':'user_accepted','user_reference':'INVENTED','accepted_policy_SHA':e.digest(m.NUMERICAL_POLICY),'accepted_conventions':['AP','AURC','binary_Brier','undefined']})
        self.lockapproval=self.put('fake_lock_approval.json',{'status':'user_approved','lock_authorized':True,'user_reference':'INVENTED','system_manifest':self.system,'evaluator_manifest':self.eval,'metric_acceptance':self.accept})
        self.lock=self.put('fake_lock.json',{'status':'joint_locked','systems':list(e.SYSTEMS),'system_manifest':self.system,'evaluator_manifest':self.eval,'metric_acceptance':self.accept,'lock_approval':self.lockapproval})
        self.approval={'status':'user_approved','user_reference':'INVENTED','partition':partition,'held_out_access':partition=='internal_test','compute_authorized':True,'cloud':False,
                       'system_manifest':self.system,'evaluator_manifest':self.eval,'metric_acceptance':self.accept,'device':'cpu','resource_limits':s.LIMITS,'max_wall_seconds':120,
                       'boundary_review_reference':'INVENTED','dataset':{'manifest':self.dataset,'format':'m4_development_json','payload_mode':'files','payload_root':'payload','canonical_rows_SHA':e.digest(self.rows)},'output':str(self.root/'result')}
        if partition=='internal_test':self.approval['lock']=self.lock
        self.request=self.put('request.json',self.approval)
    def bind(self,path):
        return {'path':str(path.relative_to(self.root)),'SHA':a.file_sha(path),'bytes':path.stat().st_size}
    def put(self,path,data):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(data if isinstance(data,bytes) else (json.dumps(data,sort_keys=True)+'\n').encode());return self.bind(p)
    def grant(self):
        return a.verify_grant(self.root,run_approval=self.request,system_manifest=self.system,evaluator_manifest=self.eval,acceptance=self.accept,
                              lock=self.lock if self.partition=='internal_test' else None,lock_approval=self.lockapproval if self.partition=='internal_test' else None)
    def changed(self,key,value):
        self.approval[key]=value;self.request=self.put('request.json',self.approval)


class FakeBackend:
    def __init__(self):self.calls=[];self.closed=False
    def dense(self,b):self.calls.append(('dense',b));return (1.,0.,0.,0.,0.,0.,0.),(1.,2.)
    def med(self,b):self.calls.append(('med',b));return (2.,1.)
    def finish(self):return {'frozen_state_unchanged':True,'RNG_unchanged':True,'device':'invented_CPU'}
    def close(self):self.closed=True


class Authorization(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def test_development_grant_and_immutable_properties(self):
        g=self.f.grant();self.assertEqual(g.partition,'validation');c=g.systems;c['systems'].clear();self.assertEqual(set(g.systems['systems']),set(e.SYSTEMS))
    def test_fake_internal_authority_chain(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t,'internal_test');g=f.grant();e.authority('internal_test',grant=g);self.assertEqual(g.lock_SHA,f.lock['SHA'])
            with self.assertRaises(PermissionError):g.check('DDI')
    def test_no_caller_dict_or_public_constructor_grant(self):
        with self.assertRaises(PermissionError):e.authority('internal_test',grant={'approved':True})
        with self.assertRaises(PermissionError):a.RunGrant(None,root=None,partition=None,dataset=None,systems=None,evaluator=None,approval=None,lock_SHA=None,authority_files=[])
    def test_internal_requires_separate_lock_and_unseal(self):
        self.f.changed('partition','internal_test');self.f.changed('held_out_access',True)
        with self.assertRaises(PermissionError):self.f.grant()
    def test_denied_before_dataset_open(self):
        self.f.changed('compute_authorized',False);(self.f.root/'dataset.json').unlink()
        with self.assertRaises(PermissionError):self.f.grant()
    def test_rule_substitution(self):
        g=self.f.grant();specs=a.runtime_specs(g.systems);specs['med7']['temperature']=2
        with self.assertRaises(PermissionError):g.check('validation',specs)
    def test_source_mutation_refused(self):
        g=self.f.grant();(self.f.root/'source.py').write_text('CHANGED')
        with self.assertRaises(PermissionError):g.reverify()
    def test_symlink_and_escape_refused(self):
        (self.f.root/'alias').symlink_to(self.f.root/'source.py')
        for path in ['alias','../outside','/absolute']:
            with self.subTest(path=path),self.assertRaises(PermissionError):a.route(self.f.root,path)
    def test_acceptance_mismatch_refused(self):
        d=json.loads((self.f.root/'acceptance.json').read_text());d['accepted_policy_SHA']='0'*64;self.f.accept=self.f.put('acceptance.json',d);self.f.changed('metric_acceptance',self.f.accept)
        with self.assertRaises(PermissionError):self.f.grant()
    def test_DDI_never_admitted_by_HAM_adapter(self):
        self.f.changed('partition','DDI')
        with self.assertRaises(PermissionError):self.f.grant()
    def test_wrong_anchor_and_probe_bytes_refused(self):
        self.f.request['SHA']='0'*64
        with self.assertRaises(PermissionError):self.f.grant()
        self.f.request=self.f.bind(self.f.root/'request.json');(self.f.root/'med2/probe.npz').write_bytes(b'CHANGED')
        with self.assertRaises(PermissionError):self.f.grant()


class Pipeline(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.f=Fixture(self.temp.name)
    def test_end_to_end_invented_validation_record_and_review(self):
        g=self.f.grant();backend=FakeBackend();result=p.run_ham(g,backend,self.f.root/'result')
        self.assertEqual(result['status'],'complete');self.assertTrue(backend.closed);self.assertEqual(len(backend.calls),14)
        with self.assertRaises(PermissionError):p.permit_recorded_review(self.f.root/'result',result['receipt_SHA'])
        supervisor=self.f.put('supervision.json',{'status':'passed','worker_reaped':True,'source_integrity_after':True,'receipt_SHA':result['receipt_SHA']})
        self.assertTrue(p.permit_recorded_review(self.f.root/'result',result['receipt_SHA'],supervision_record=self.f.root/'supervision.json',expected_supervision_SHA=supervisor['SHA']))
        pred=json.loads((self.f.root/'result/predictions.json').read_text());self.assertEqual(len(pred['physical']['med7']),7);self.assertEqual(len(pred['physical']['med2']),6)
        self.assertEqual(json.loads((self.f.root/'result/metrics.json').read_text())['result']['paired_intervals']['7']['replicates'],1000)
        with self.assertRaises(FileExistsError):p.run_ham(g,FakeBackend(),self.f.root/'result')
    def test_invented_internal_grant_runs_no_real_heldout_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t,'internal_test');result=p.run_ham(f.grant(),FakeBackend(),f.root/'result');self.assertEqual(result['lock_SHA'],f.lock['SHA'])
    def test_hash_failure_before_encoder_and_partial_record_not_final(self):
        g=self.f.grant();(self.f.root/'payload/0').write_text('CHANGED');backend=FakeBackend()
        with self.assertRaises(ValueError):p.run_ham(g,backend,self.f.root/'result')
        self.assertEqual(backend.calls,[]);self.assertTrue((self.f.root/'result/failure.json').exists());self.assertFalse((self.f.root/'result/receipt.json').exists())
    def test_frozen_state_failure_and_close(self):
        backend=FakeBackend();backend.finish=lambda:{'frozen_state_unchanged':False,'RNG_unchanged':True}
        with self.assertRaises(ValueError):p.run_ham(self.f.grant(),backend,self.f.root/'result')
        self.assertTrue(backend.closed);self.assertFalse((self.f.root/'result/receipt.json').exists())
    def test_timeout_checks_and_row_substitution(self):
        g=self.f.grant()
        with self.assertRaises(TimeoutError):p.run_ham(g,FakeBackend(),self.f.root/'result',check=lambda:(_ for _ in ()).throw(TimeoutError()))
        rows=list(p.load_rows(g));rows[0]=dict(rows[0],label='nv')
        with self.assertRaises(PermissionError):g.check_rows(rows)
    def test_complete_record_tamper_refuses_review(self):
        result=p.run_ham(self.f.grant(),FakeBackend(),self.f.root/'result');(self.f.root/'result/predictions.json').write_text('{}')
        supervisor=self.f.put('tamper_supervision.json',{'status':'passed','worker_reaped':True,'source_integrity_after':True,'receipt_SHA':result['receipt_SHA']})
        with self.assertRaises(PermissionError):p.permit_recorded_review(self.f.root/'result',result['receipt_SHA'],supervision_record=self.f.root/'tamper_supervision.json',expected_supervision_SHA=supervisor['SHA'])
    def test_M2_CSV_normalization_active_and_cross_partition(self):
        rows=[dict(image_id='invented',analysis_lesion_id='p',split_group_id='c',dx='nv',sha256='1'*64,path='a/invented.jpg',active='true')]
        out=p.normalize_rows(rows,'internal_test',csv_format=True);self.assertEqual(out[0]['physical_id'],'p');self.assertEqual(out[0]['label'],'nv')
        rows[0]['split']='train'
        with self.assertRaises(ValueError):p.normalize_rows(rows,'internal_test',csv_format=True)
    def test_invalid_row_flags_targets_paths(self):
        for change in [dict(active=1),dict(target=9),dict(path='../secret')]:
            rows=[dict(self.f.rows[0],**change)]
            with self.subTest(change=change),self.assertRaises(ValueError):p.normalize_rows(rows,'validation')
    def test_unapproved_reader_row(self):
        reader=p.PayloadReader(self.f.grant());self.addCleanup(reader.close)
        with self.assertRaises(PermissionError):reader.read(dict(self.f.rows[0],path='1'))
    def test_zip_exact_members(self):
        f=self.f
        with zipfile.ZipFile(f.root/'images.zip','w') as z:
            for i,r in enumerate(f.rows):z.writestr(r['image_id']+'.jpg',(f.root/'payload'/str(i)).read_bytes());r['path']='ham/'+r['image_id']+'.jpg'
        f.dataset=f.put('dataset.json',f.rows);d=dict(f.approval['dataset'],manifest=f.dataset,canonical_rows_SHA=e.digest(f.rows),payload_mode='zip',sources={'ham':f.bind(f.root/'images.zip')});f.changed('dataset',d)
        reader=p.PayloadReader(f.grant());self.addCleanup(reader.close);self.assertIn(b'INVENTED',reader.read(f.rows[0]))
    def test_binary_probability_and_tie_policies_recorded(self):
        result=p.run_ham(self.f.grant(),FakeBackend(),self.f.root/'result');d=json.loads((self.f.root/'result/predictions.json').read_text())
        self.assertTrue(all(len(r['accepted'])==2 and len(r['probabilities'])==2 for r in d['physical']['dense2']))


class Supervision(unittest.TestCase):
    def test_guards_are_unchanged_and_cannot_be_weakened(self):
        sample={'RSS_bytes':1,'available_estimate_bytes':4*1024**3,'swap_used_bytes':0};self.assertIsNone(s.resource_reason(sample,0,8*1024**3))
        for change in [dict(RSS_bytes=6*1024**3+1),dict(available_estimate_bytes=4*1024**3-1),dict(swap_used_bytes=256*1024**2+1)]:self.assertIsNotNone(s.resource_reason(dict(sample,**change),0,8*1024**3))
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);f.changed('resource_limits',dict(s.LIMITS,available_bytes=1))
            with self.assertRaises(PermissionError):s.admission(f.grant())
    def test_profile_separates_source_output_and_denies_network_helper(self):
        import sys
        with tempfile.TemporaryDirectory() as t:
            f=Fixture(t);control=f.root/'control';control.mkdir();probe=control/'probe.py';probe.write_text('# invented')
            text=s.profile(f.grant(),f.root/'result',control,sys.executable,[Path(sys.prefix),Path(sys.base_prefix)],[probe]);self.assertIn('(deny network*)',text);self.assertIn('(deny process-exec',text);self.assertIn('(deny file-write*',text)
            with self.assertRaises(PermissionError):s.profile(f.grant(),f.root,control,sys.executable,[Path(sys.prefix)],[probe])


class BackendAdapters(unittest.TestCase):
    def test_tiny_torch_backend_uses_one_dense_pass_and_pinned_med_route(self):
        # Tiny invented modules exercise the production adapter methods. No
        # actual DenseNet/MedSigLIP checkpoint is constructed or loaded.
        import io
        import torch
        from PIL import Image
        from types import SimpleNamespace
        from dermdetect.evaluation.torch_backend import TorchCPUBackend,state_sha
        class TinyDense(torch.nn.Module):
            def __init__(self):
                super().__init__();self.backbone=torch.nn.Module();self.backbone.classifier=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(1024,7));self.calls=0
            def features(self,x):self.calls+=1;return torch.ones((1,1024),dtype=torch.float32)
        class TinyMed(torch.nn.Module):
            def get_image_features(self,pixel_values):return torch.arange(1,1153,dtype=torch.float32).reshape(1,1152)
        backend=TorchCPUBackend.__new__(TorchCPUBackend);backend.check=lambda:None;backend.torch=torch
        backend.dense_model=TinyDense().eval().requires_grad_(False);backend.med_model=TinyMed().eval();seen=[]
        def processor(**kw):seen.append(kw);return {'pixel_values':torch.zeros((1,3,448,448),dtype=torch.float32)}
        backend.processor=processor;backend.initial=(state_sha(backend.dense_model),state_sha(backend.med_model));backend.rng=torch.get_rng_state().clone()
        backend.dense_load_seconds=0.;backend.med_load_seconds=0.;backend.setup_seconds=0.
        buffer=io.BytesIO();Image.new('RGB',(8,8),(32,64,128)).save(buffer,format='PNG');payload=buffer.getvalue()
        import importlib
        dense_module=importlib.import_module('dermdetect.models.densenet')
        with patch.object(dense_module,'transform',return_value=torch.ones((3,4,4))) as transform:
            logits,features=backend.dense(payload);self.assertEqual(backend.dense_model.calls,1);self.assertEqual(len(logits),7);self.assertEqual(len(features),1024);self.assertFalse(transform.call_args.kwargs['training'])
        med=backend.med(payload);self.assertEqual(len(med),1152);self.assertLessEqual(abs(sum(x*x for x in med)-1.),1e-5)
        self.assertFalse(seen[0]['do_resize']);self.assertEqual(seen[0]['images'].size,(448,448))
        result=backend.finish();self.assertTrue(result['frozen_state_unchanged']);self.assertTrue(result['RNG_unchanged'])


class ProcessSupervision(unittest.TestCase):
    def test_actual_invented_child_reaped_on_memory_guard(self):
        import subprocess,sys,time
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],start_new_session=True)
        rows=[]
        r=s.watch(child,sample=lambda pid:{'RSS_bytes':1,'available_estimate_bytes':1,'swap_used_bytes':0},baseline=0,disk_free=lambda:10*1024**3,deadline=time.monotonic()+3,record_sample=rows.append)
        self.assertTrue(r['worker_reaped']);self.assertIsNotNone(child.poll());self.assertEqual(r['stop_reason'],'available_memory4GiB')
    def test_actual_invented_child_reaped_if_sampler_fails(self):
        import subprocess,sys,time
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],start_new_session=True)
        def fail(pid):raise RuntimeError('INVENTED SAMPLER FAILURE')
        with self.assertRaises(RuntimeError):s.watch(child,sample=fail,baseline=0,disk_free=lambda:10*1024**3,deadline=time.monotonic()+3,record_sample=lambda row:None)
        self.assertIsNotNone(child.poll())


class NativeRuntime(unittest.TestCase):
    def test_named_runtime_file_hash_and_unrelated_route_refusal(self):
        from dermdetect.evaluation.torch_backend import native_runtime_files
        binding=native_runtime_files()[0];alias,resolved=a.checked_runtime_file(binding);self.assertTrue(resolved.is_file())
        with self.assertRaises(PermissionError):a.checked_runtime_file(dict(binding,SHA='0'*64))
        with self.assertRaises(PermissionError):a.checked_runtime_file({'path':'/opt/example/not_a_runtime.dylib','SHA':'0'*64})


class RuntimeImports(unittest.TestCase):
    def test_offline_encoder_adapter_dependencies_import_no_models(self):
        import ssl
        from transformers import AutoModel,AutoImageProcessor
        from dermdetect.models.densenet import DenseNetSystem
        self.assertTrue(callable(AutoModel.from_pretrained));self.assertTrue(callable(AutoImageProcessor.from_pretrained));self.assertTrue(callable(DenseNetSystem));self.assertTrue(ssl.OPENSSL_VERSION)
    def test_invented_child_wall_deadline_is_enforced_and_reaped(self):
        import subprocess,sys,time
        child=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(20)'],start_new_session=True)
        r=s.watch(child,sample=lambda pid:{'RSS_bytes':1,'available_estimate_bytes':4*1024**3,'swap_used_bytes':0},baseline=0,disk_free=lambda:10*1024**3,deadline=time.monotonic()-1,record_sample=lambda row:None)
        self.assertEqual(r['stop_reason'],'CPU_wall_deadline');self.assertTrue(r['worker_reaped']);self.assertIsNotNone(child.poll())


class ExternalInputRoutes(unittest.TestCase):
    def test_named_external_source_is_hash_bound_without_copy_or_symlink(self):
        with tempfile.TemporaryDirectory() as t, tempfile.TemporaryDirectory() as external:
            f=Fixture(t);q=Path(external).resolve()/'source.zip';q.write_bytes(b'INVENTED EXTERNAL INPUT')
            binding={'path':str(q),'SHA':a.file_sha(q),'bytes':q.stat().st_size}
            self.assertEqual(a.checked_input_file(f.root,binding),q)
            with self.assertRaises(PermissionError):a.checked_input_file(f.root,dict(binding,SHA='0'*64))
            with self.assertRaises(PermissionError):a.checked_file(f.root,binding)
            alias=Path(t)/'external_alias';alias.symlink_to(q)
            with self.assertRaises(PermissionError):a.checked_input_file(f.root,{'path':str(alias),'SHA':a.file_sha(q)})
