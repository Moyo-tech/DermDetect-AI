import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dermdetect.data.analysis import resolve_analysis
from dermdetect.data.conservative_groups import build_conservative_groups
from dermdetect.data.constrained_allocation import allocate_constrained_groups,verify_constrained_allocation
from dermdetect.data.development_bundle import BundleApproval,bundle_input_hashes,verify_staged_bundle
from dermdetect.data.schemas import CLASS_ORDER,ManifestRecord,ContractError
from dermdetect.data.streaming_bundle import stage_streamed_bundle


class StreamingBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        protected=self.root/'evaluator';protected.mkdir();self.payloads={};records=[]
        for c in CLASS_ORDER:
            for i in range(10):
                image=f'{c}_{i}';payload=('INVENTED '+image).encode();self.payloads[image]=payload
                records.append(ManifestRecord(image,image,c,'invented','fixture-v1',image+'.txt',hashlib.sha256(payload).hexdigest()))
        base=next(r for r in records if r.image_id=='nv_0')
        records.append(replace(base,image_id='nv_0_copy',path='nv_0_copy.txt'));self.payloads['nv_0_copy']=self.payloads['nv_0']
        group=build_conservative_groups(resolve_analysis(records,()),());partitions=allocate_constrained_groups(group)
        self.opened=[]
        owner=self
        class Bounded(io.BytesIO):
            def read(self,size=-1):
                owner.assertGreater(size,0);owner.assertLessEqual(size,1024*1024)
                return super().read(size)
        def open_payload(row):self.opened.append(row.original.image_id);return Bounded(self.payloads[row.original.image_id])
        self.args=dict(destination=self.root/'bundle',provenance_path=protected/'provenance.json',records=group.records,
            partitions=partitions,allocation_report=verify_constrained_allocation(group,partitions),
            code_files={'worker.py':b'INVENTED inspected worker'},open_payload=open_payload,protected_roots=[protected],
            protected_content_sha256=[hashlib.sha256(b'INVENTED complete source').hexdigest()],
            evidence_context='INVENTED evaluator context',grouping=group)
        hashes=bundle_input_hashes(**{k:self.args[k] for k in ('records','partitions','code_files','protected_content_sha256','grouping')})
        self.args['approvals']=[BundleApproval(k,v,'invented approver','invented scope evidence',True) for k,v in hashes.items()]

    def test_bounded_stream_only_exact_active_development_members(self):
        result=stage_streamed_bundle(**self.args)
        expected={r.original.image_id for p in ('train','validation') for r in self.args['partitions'][p] if r.active}
        self.assertEqual(set(self.opened),expected);self.assertEqual(len(self.opened),len(expected))
        self.assertNotIn('nv_0_copy',self.opened)
        verified=verify_staged_bundle(destination=self.args['destination'],expected_manifest_sha256=result['bundle_manifest_sha256'])
        self.assertEqual(verified['status'],'staged_composition_verified')
        self.assertFalse(result['production_seal_validated'])

    def test_bad_stream_rolls_back_bundle_without_provenance(self):
        self.args['open_payload']=lambda row:io.BytesIO(b'CHANGED INVENTED BYTES')
        with self.assertRaises(ContractError):stage_streamed_bundle(**self.args)
        self.assertFalse(self.args['destination'].exists());self.assertFalse(self.args['provenance_path'].exists())

    def test_serialized_report_preserves_contract_but_changed_counts_stop(self):
        saved=json.loads(json.dumps(self.args['allocation_report']))
        self.args['allocation_report']=saved
        result=stage_streamed_bundle(**self.args)
        self.assertEqual(result['exported_active_images'],len(self.opened))
        self.args['destination']=self.root/'other_bundle'
        self.args['provenance_path']=self.root/'evaluator/other_provenance.json'
        saved['physical_units']+=1
        with self.assertRaises(ContractError):stage_streamed_bundle(**self.args)
        self.assertFalse(self.args['destination'].exists())

    def test_missing_approval_prevents_any_read_or_output(self):
        self.args['approvals']=[]
        with self.assertRaises(ContractError):stage_streamed_bundle(**self.args)
        self.assertEqual(self.opened,[]);self.assertFalse(self.args['destination'].exists())
