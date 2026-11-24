"""CPU loader tested with invented text payloads and no dataset content."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/dermdetect/data'))
spec=importlib.util.spec_from_file_location('m2_worker_fixture',ROOT/'scripts/m2_development_worker.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.bundle=self.root/'bundle';self.bundle.mkdir()
        self.output=self.root/'output';self.output.mkdir();self.files={}
        for partition in ('train','validation'):
            rows=[]
            for c in worker.CLASS_ORDER:
                image=partition+'_'+c;reference=f'data/{partition}/{image}.payload'
                payload=('INVENTED '+image).encode();self.files[reference]=payload
                rows.append({'image_id':image,'dx':c,'analysis_lesion_id':image,
                             'split_group_id':image,'path':reference,'sha256':hashlib.sha256(payload).hexdigest()})
            self.files[f'manifests/{partition}.json']=worker.canonical(rows)
        self.files['code/worker.py']=b'INVENTED inspected CPU code'
        self.write()

    def write(self):
        for name,payload in self.files.items():
            path=self.bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        manifest={'policy':'staged-development-bundle-v1','status':'staged_only','production_seal_validated':False,
                  'files':{name:hashlib.sha256(payload).hexdigest() for name,payload in self.files.items()}}
        (self.bundle/'bundle_manifest.json').write_bytes(worker.canonical(manifest))
        self.digest=worker.digest(self.bundle/'bundle_manifest.json')

    def test_exact_cpu_verification_weights_and_paired_draw_summary(self):
        result=worker.verify_and_report(self.bundle,self.digest,self.output)
        self.assertEqual(result['weight_sum'],'7')
        self.assertEqual(result['validation_paired_draws'],1000)
        self.assertEqual(set(result['class_loss_mass'].values()),{'1/7'})
        self.assertFalse(result['predictions_or_outcomes_read'])
        self.assertFalse(result['production_seal_validated'])

    def test_added_file_symlink_and_payload_tampering_refused(self):
        extra=self.bundle/'full_metadata.csv';extra.write_text('INVENTED')
        with self.assertRaises(ValueError):worker.verify_bundle(self.bundle,self.digest)
        extra.unlink();extra.symlink_to(self.output)
        with self.assertRaises(ValueError):worker.verify_bundle(self.bundle,self.digest)
        extra.unlink();(self.bundle/'code/worker.py').write_bytes(b'CHANGED')
        with self.assertRaises(ValueError):worker.verify_bundle(self.bundle,self.digest)

    def test_rebound_illegal_metadata_and_cross_split_group_refused(self):
        for change in ('source','path','cross'):
            name='manifests/validation.json';original=self.files[name];rows=json.loads(original)
            if change=='source':rows[0]['original_source_path']='INVENTED protected path'
            elif change=='path':rows[0]['path']='../evaluator/test.payload'
            else:rows[0]['split_group_id']='train_akiec'
            self.files[name]=worker.canonical(rows);self.write()
            with self.assertRaises(ValueError):worker.verify_bundle(self.bundle,self.digest)
            self.files[name]=original

    def test_duplicate_json_keys_and_rebound_unlisted_payload_refused(self):
        with self.assertRaises(ValueError):
            p=self.root/'duplicate.json';p.write_text('{"a":1,"a":2}');worker.read_json(p)
        self.files['data/train/unlisted.payload']=b'INVENTED';self.write()
        with self.assertRaises(ValueError):worker.verify_bundle(self.bundle,self.digest)


if __name__=='__main__':unittest.main()
