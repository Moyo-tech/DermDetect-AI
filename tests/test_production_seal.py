"""Invented seal evidence; no real assignment or access boundary is exercised."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dermdetect.data.production_seal import assemble_seal,AUTHORIZATIONS,REQUIRED_BINDINGS


class SealTests(unittest.TestCase):
    def fixture(self):
        bindings={k:'a'*64 for k in REQUIRED_BINDINGS}
        approval={**dict.fromkeys(AUTHORIZATIONS,True),'training_authorized':False,'outcome_access_authorized':False,
            'commit_push_authorized':False,'proposal_report_sha256':'a'*64,'approved_package_sha256':'a'*64}
        counts={c:{'physical_units':1} for c in ('akiec','bcc','bkl','df','mel','nv','vasc')}
        proposal={'production_freeze':False,'source_labels_and_original_records_unchanged':True,
            'deterministic_membership_replay':True,'unclear_pairs_retained':15,'analysis_lesions':7449,
            'active_views':10013,'original_records':10015,'artifact_sha256':{'invented':'a'*64},
            'allocation_verification':{'counts':{'train':counts,'validation':counts}}}
        partitions={s:{'active_views':a,'physical_units':p,'conservative_groups':g,
            'class_physical_counts':{c:1 for c in counts}} for s,a,p,g in (('train',7000,5214,5203),('validation',1505,1118,1117))}
        required={'network_denied','helper_exec_denied','write_code_denied','write_runtime_denied',
            'write_evaluator_denied','source_alias_denied','test_membership_alias_denied',
            'future_outcome_sentinel_denied','future_image_sentinel_denied'}
        return dict(approval=approval,proposal=proposal,tests={'returncode':0,'passed':218,'input_scope':'invented fixtures only'},
            composition={'status':'production_composition_verified','active_development_payloads':8505,
                'no_protected_membership_or_payloads':True,'all_payload_sha256_passed':True},
            worker={'status':'development_cpu_contracts_verified','partitions':partitions,'weight_sum':'7000',
                'class_loss_mass':dict.fromkeys(counts,'1/7'),'validation_paired_draws':1000,
                'predictions_or_outcomes_read':False,'output_write_allowed':True,'boundary_checks':dict.fromkeys(required,True)},
            boundary={'status':'production_cpu_boundary_passed','returncode':0,'bundle_runtime_unchanged':True,
                'fresh_worker_launch':True,'inherited_source_descriptors':0},
            preservation={'status':'accepted_sources_review_and_proposal_preserved','exact_approved_membership_replayed':True,
                'zero_overlap_dimensions':5,'all15_unclear_retained_and_colocated':True,
                'original_source_hashes_unchanged':True,'raw_exports_unchanged':True},
            bindings=bindings,paths={'invented':'invented/evaluator/only'},recorded_utc='2026-10-02T00:00:00Z')

    def test_complete_evidence_assembles_separate_seal_without_mutation(self):
        args=self.fixture();original=deepcopy(args);result=assemble_seal(**args)
        self.assertEqual(args,original);self.assertTrue(result['production_freeze'])
        self.assertFalse(result['training_authorized']);self.assertFalse(result['outcome_access_authorized'])

    def test_each_missing_authorization_and_binding_refused(self):
        for flag in AUTHORIZATIONS:
            args=self.fixture();args['approval'][flag]=False
            with self.assertRaises(ValueError):assemble_seal(**args)
        args=self.fixture();args['bindings']['proposal_report']='b'*64
        with self.assertRaises(ValueError):assemble_seal(**args)

    def test_failed_stale_or_missing_evidence_refused(self):
        mutations=[('tests','returncode',1),('composition','all_payload_sha256_passed',False),
            ('boundary','bundle_runtime_unchanged',False),('boundary','inherited_source_descriptors',1),
            ('preservation','raw_exports_unchanged',False),('proposal','unclear_pairs_retained',14),
            ('worker','weight_sum','8505')]
        for section,key,value in mutations:
            args=self.fixture();args[section][key]=value
            with self.assertRaises(ValueError):assemble_seal(**args)
        args=self.fixture();args['worker']['boundary_checks']['network_denied']=False
        with self.assertRaises(ValueError):assemble_seal(**args)
        args=self.fixture();del args['worker']['boundary_checks']['write_runtime_denied']
        with self.assertRaises(ValueError):assemble_seal(**args)
