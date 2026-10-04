"""File-adapter acceptance and proposed split tests using invented text only."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_ham_constrained_split as runner
from dermdetect.data.manifests import canonical_json_bytes,file_sha256,manifest_csv_bytes,sha256_bytes
from dermdetect.data.schemas import CLASS_ORDER,ContractError,ManifestRecord
from dermdetect.data.review_policy import RANKED_PROCEDURE,RANKED_STATUS,APPROVAL_REFERENCE
from resolve_ham_analysis import QUEUE_FIELDS,VIEW_TRANSFORM


class AcceptedConstrainedSplitTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='invented-accepted-split-');self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve();self.output=self.root/'outputs/proposed';(self.root/'outputs').mkdir()
        self.paths={name:self.root/(name+'.json') for name in runner.INPUTS}
        self.paths['manifest']=self.root/'manifest.csv';self.paths['candidate_queue']=self.root/'queue.csv'
        records=[ManifestRecord(f'{dx}_{i:02}',f'{dx}_{i:02}',dx,'invented','v1',f'{dx}_{i:02}.txt',sha256_bytes(f'{dx}_{i:02}'.encode()))
                 for dx in CLASS_ORDER for i in range(10)]
        self.paths['manifest'].write_bytes(manifest_csv_bytes(records));by_id={r.image_id:r for r in records}
        pairs=[('bcc_00','nv_00','uncertain','pending'),('nv_01','nv_02','same_lesion','different_lesions'),
               ('mel_00','mel_01','different_lesions',None)]
        queue=[];views=[];observations=[];humans=[]
        for number,(a,b,recommendation,decision) in enumerate(pairs):
            payloads={i:by_id[i].sha256 for i in (a,b)};view=f'invented_view_{number}.png'
            (self.root/view).write_bytes(('INVENTED TEXT NOT PIXELS '+view).encode());digest=file_sha256(self.root/view)
            queue.append(dict(zip(QUEUE_FIELDS,(a,b,'10','independent_published_top1000_cross_id','pending','','',''))))
            views.append({'image_ids':[a,b],'source_payload_sha256':[payloads[a],payloads[b]],'view_file':view,
                          'view_sha256':digest,'view_transform':VIEW_TRANSFORM})
            observations.append({'left':a,'right':b,'agent_recommendation':recommendation,'agent_reviewer':'Codex invented fixture',
                'agent_rationale':'INVENTED landmarks','review_date':'2026-10-02','full_resolution_viewed':True,
                'human_review_status':'not_performed','view_file':view,'view_sha256':digest,'source_payload_sha256':payloads})
            if decision:
                humans.append({'left':a,'right':b,'decision':decision,'reviewer':'INVENTED reviewer','review_date':'2026-10-02',
                    'rationale':'INVENTED inspection not actual human review','full_view_inspected':True,'evidence':view,
                    'view_sha256':digest,'source_payload_sha256':payloads,
                    'source_applicability':{i:{'source':'invented','source_version':'v1'} for i in (a,b)},'view_repetition':'byte_distinct_views'})
        with self.paths['candidate_queue'].open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=QUEUE_FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(queue)
        base={'manifest_sha256':file_sha256(self.paths['manifest']),'candidate_queue_sha256':file_sha256(self.paths['candidate_queue'])}
        self.write('visual_index',{'manifest_sha256':base['manifest_sha256'],'status':'visual_evidence_prepared_not_adjudicated','pairs':views})
        base['visual_index_sha256']=file_sha256(self.paths['visual_index'])
        self.write('audit_plan',{**base,'policy':'m2-ai-assisted-audit-v1','audit_seed':20260729})
        base['audit_plan_sha256']=file_sha256(self.paths['audit_plan'])
        self.enterContext(patch.object(runner,'AUDIT_PLAN_SHA256',base['audit_plan_sha256']))
        self.write('agent_aggregate',{**base,'pairs':observations,'production_freeze':False})
        aggregate_hash=file_sha256(self.paths['agent_aggregate'])
        human={**{k:v for k,v in base.items() if k!='candidate_queue_sha256'},'full_candidate_queue_sha256':base['candidate_queue_sha256'],
            'agent_aggregate_sha256':aggregate_hash,'candidate_queue_sha256':'INVENTED focused queue','draft':True,
            'human_acceptance':{'accept_procedure':False,'accept_identity_and_participation':False},'decisions':humans}
        self.write('human_audit',human);self.write('human_baseline',human)
        self.write('audit_report',{**base,'agent_aggregate_sha256':aggregate_hash,'human_audit_complete':True,
            'human_identity_consistent':True,'production_freeze':False,'split_ready':False,'human_inspected_unclear':1,
            'human_required_pairs':2,'human_audit_rows_recorded_as_attestations':2,
            'required_pairs':[{'left':r['left'],'right':r['right']} for r in humans]})
        self.write('screen_report',{'manifest_sha256':base['manifest_sha256'],'candidates_sha256':base['candidate_queue_sha256'],
            'procedure':RANKED_PROCEDURE,'status':RANKED_STATUS,'procedure_approval':'approved_candidate_selection_only',
            'approval_reference':APPROVAL_REFERENCE,'human_decisions_recorded':0,'broad_outside_status':'unadjudicated',
            'production_freeze':False,'split_ready':False,'review_candidates':3,'broad_pairs_outside_approved_review':0})
        archive_hashes={alias:sha256_bytes(('INVENTED ARCHIVE '+alias).encode()) for alias in ('part1','part2')}
        (self.root/'integrity_report.json').write_bytes(canonical_json_bytes({'manifest_sha256':base['manifest_sha256'],
            'archive_crc':'passed_for_all_referenced_members','archives':[{'alias':a,'sha256':h} for a,h in archive_hashes.items()],
            'metadata_sha256':sha256_bytes(b'INVENTED METADATA'),'data_kind':'invented_fixture_not_real_inventory'}))
        decode=self.root/'outputs/m2-image-integrity';decode.mkdir()
        (decode/'decode_checkpoint.json').write_bytes(canonical_json_bytes({'manifest_sha256':base['manifest_sha256'],
            'archive_sha256':archive_hashes,'records_decoded':70,'status':'all_listed_jpegs_decoded_candidate_screen_pending',
            'data_kind':'invented_schema_fixture_no_real_decoding'}))
        self.acceptance=self.root/'acceptance.json'
        self.approval={'policy':runner.ACCEPTANCE_POLICY,'accept_audit_decisions_and_revisions':True,'retain_all15_unclear':True,
            'authorize_proposed_analysis_and_assignment':True,'authorize_assignment_freeze':False,
            'authorize_development_payload_export':False,'approval_reference':'INVENTED approval for synthetic test',
            'approval_date':'2026-10-02','approver':'INVENTED reviewer','user_quote':'INVENTED acceptance, no actual approval',
            'input_paths':{k:p.relative_to(self.root).as_posix() for k,p in self.paths.items()},
            'input_sha256':{k:file_sha256(p) for k,p in self.paths.items()}}
        self.acceptance.write_bytes(canonical_json_bytes(self.approval))

    def write(self,name,value):self.paths[name].write_bytes(canonical_json_bytes(value))
    def invoke(self):return runner.prepare(self.acceptance,self.output,evidence_root=self.root)

    def test_accepted_audit_produces_whole_mixed_group_proposal_without_export_or_freeze(self):
        original=self.paths['human_audit'].read_bytes();result=self.invoke()
        self.assertEqual(result['original_records'],70);self.assertEqual(result['analysis_lesions'],70)
        self.assertEqual(result['unclear_pairs_retained'],1)
        self.assertEqual(result['effective_attribution_counts'],{'human':2,'agent':1})
        self.assertEqual(len(result['allocation_verification']['mixed_label_groups']),1)
        self.assertFalse(result['production_freeze']);self.assertFalse(result['final_assignment_accepted'])
        self.assertFalse(result['bundle_adapter_preview']['image_payloads_exported'])
        self.assertFalse(result['bundle_adapter_preview']['export_approval_present'])
        self.assertEqual(original,self.paths['human_audit'].read_bytes())
        for name,digest in result['artifact_sha256'].items():self.assertEqual(file_sha256(self.output/name),digest)
        owners={}
        for split in ('train','validation','internal_test'):
            with (self.output/(split+'.csv')).open() as handle:
                owners.update({r['image_id']:split for r in csv.DictReader(handle)})
        self.assertEqual(owners['bcc_00'],owners['nv_00'])
        with self.assertRaises(ContractError):self.invoke()

    def test_stale_denied_or_implicit_acceptance_stops_before_any_outputs(self):
        for change in ('denied','freeze','stale','missing'):
            with self.subTest(change=change):
                approval=json.loads(json.dumps(self.approval))
                if change=='denied':approval['accept_audit_decisions_and_revisions']=False
                elif change=='freeze':approval['authorize_assignment_freeze']=True
                elif change=='stale':approval['input_sha256']['human_audit']='0'*64
                else:del approval['user_quote']
                self.acceptance.write_bytes(canonical_json_bytes(approval))
                with self.assertRaises(ContractError):self.invoke()
                self.assertFalse(self.output.exists())

    def test_changed_immutable_binding_cannot_be_accepted_by_hash_refresh(self):
        human=json.loads(self.paths['human_audit'].read_text());human['decisions'][0]['view_sha256']='0'*64
        self.write('human_audit',human)
        self.approval['input_sha256']['human_audit']=file_sha256(self.paths['human_audit'])
        self.acceptance.write_bytes(canonical_json_bytes(self.approval))
        with self.assertRaises(ContractError) as caught:self.invoke()
        self.assertEqual(caught.exception.code,'visual_evidence_mismatch');self.assertFalse(self.output.exists())


if __name__=='__main__':unittest.main()
