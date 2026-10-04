"""Human/agent attribution and unresolved physical identities on invented data."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dermdetect.data.audited_analysis import resolve_audited_analysis
from dermdetect.data.schemas import ContractError, ManifestRecord


class AuditedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows=[ManifestRecord(i,'lesion_'+i,'nv','invented','v1',i+'.txt',hashlib.sha256(i.encode()).hexdigest())
                   for i in ('a','b','c','d')]
        by_id={r.image_id:r for r in self.rows}
        self.observations=[];self.human=[]
        for left,right,recommendation,decision in [('a','b','same_lesion','different_lesions'),
                                                  ('b','c','uncertain','pending'),
                                                  ('c','d','different_lesions',None)]:
            view='invented_'+left+right+'.png';view_hash=hashlib.sha256(view.encode()).hexdigest()
            payloads={i:by_id[i].sha256 for i in (left,right)}
            obs={'left':left,'right':right,'agent_recommendation':recommendation,
                 'agent_reviewer':'Codex invented fixture','agent_rationale':'invented agent comparison',
                 'review_date':'2026-10-02','full_resolution_viewed':True,'human_review_status':'not_performed',
                 'view_file':view,'view_sha256':view_hash,'source_payload_sha256':payloads}
            self.observations.append(obs)
            if decision:
                self.human.append({'left':left,'right':right,'decision':decision,'reviewer':'invented human',
                    'rationale':'invented inspected uncertainty or override','review_date':'2026-10-02','full_view_inspected':True,
                    'evidence':view,'view_sha256':view_hash,'source_payload_sha256':payloads,
                    'source_applicability':{i:{'source':'invented','source_version':'v1'} for i in (left,right)}})
        self.args={'candidate_pairs':self.observations,'agent_observations':self.observations,
                   'human_decisions':self.human,'required_human_pairs':self.human}

    def test_source_override_uncertainty_and_retained_negative_have_distinct_attribution(self):
        result=resolve_audited_analysis(self.rows,**self.args)
        effective={(r['left'],r['right']):r for r in result.effective_decisions}
        self.assertEqual(effective['a','b']['decision'],'different_lesions')
        self.assertEqual(effective['a','b']['decision_attribution'],'human')
        self.assertEqual(effective['c','d']['decision_attribution'],'agent')
        self.assertNotIn('human_reviewer',effective['c','d'])
        rows={r.original.image_id:r for r in result.grouping.records}
        self.assertNotEqual(rows['b'].analysis_lesion_id,rows['c'].analysis_lesion_id)
        self.assertEqual(rows['b'].split_group_id,rows['c'].split_group_id)
        self.assertEqual(len(result.grouping.uncertainties),1)
        self.assertEqual(len(result.effective_decisions),3)

    def test_uninspected_uncertain_or_same_cannot_be_retained_as_a_negative(self):
        for recommendation in ('uncertain','same_lesion'):
            observations=[dict(r) for r in self.observations];observations[-1]['agent_recommendation']=recommendation
            with self.assertRaises(ContractError) as caught:
                resolve_audited_analysis(self.rows,**{**self.args,'agent_observations':observations})
            self.assertEqual(caught.exception.code,'unreviewed_identity_proposal')

    def test_missing_duplicate_uninspected_and_stale_human_rows_stop(self):
        cases=[self.human[:-1],self.human+[self.human[0]],
               [dict(r,full_view_inspected=False) for r in self.human],
               [dict(r,view_sha256='0'*64) for r in self.human],
               [dict(r,source_payload_sha256={}) for r in self.human]]
        for human in cases:
            with self.subTest(human=human),self.assertRaises(ContractError):
                resolve_audited_analysis(self.rows,**{**self.args,'human_decisions':human})

    def test_human_same_different_transitive_conflict_cannot_pass_adapter(self):
        observations=[dict(r) for r in self.observations]
        observations[2].update(left='a',right='c',source_payload_sha256={i:self.rows[['a','b','c','d'].index(i)].sha256 for i in ('a','c')})
        human=[dict(r,decision='same_lesion') for r in self.human]
        with self.assertRaises(ContractError) as caught:
            resolve_audited_analysis(self.rows,candidate_pairs=observations,agent_observations=observations,
                                    human_decisions=human,required_human_pairs=human)
        self.assertEqual(caught.exception.code,'contradictory_review')


if __name__=='__main__':unittest.main()
