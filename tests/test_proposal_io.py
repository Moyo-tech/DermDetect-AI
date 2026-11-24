"""Saved-proposal adapter replay/refusal tests on invented text metadata only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import test_accepted_constrained_split as fixture_module
from dermdetect.data.manifests import canonical_json_bytes,file_sha256
from dermdetect.data.proposal_io import load_proposed_split
from dermdetect.data.schemas import ContractError


class ProposalIOTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.AcceptedConstrainedSplitTests(methodName='test_accepted_audit_produces_whole_mixed_group_proposal_without_export_or_freeze')
        self.addCleanup(self.fixture.doCleanups);self.fixture.setUp();self.fixture.invoke()
        self.root=self.fixture.output;self.report_hash=file_sha256(self.root/'proposal_report.json')

    def test_saved_proposal_reconstructs_complete_grouping_and_bundle_inputs(self):
        grouping,partitions,report=load_proposed_split(self.root,expected_report_sha256=self.report_hash)
        self.assertEqual(len(grouping.records),70)
        self.assertEqual(sum(len(rows) for rows in partitions.values()),70)
        by_id={r.original.image_id:r for r in grouping.records}
        self.assertEqual(by_id['bcc_00'].split_group_id,by_id['nv_00'].split_group_id)
        self.assertNotEqual(by_id['bcc_00'].analysis_lesion_id,by_id['nv_00'].analysis_lesion_id)
        self.assertFalse(report['final_assignment_accepted'])

    def test_wrong_report_hash_changed_payload_and_unlisted_files_are_rejected(self):
        with self.assertRaises(ContractError) as caught:load_proposed_split(self.root,expected_report_sha256='0'*64)
        self.assertEqual(caught.exception.code,'proposal_report_mismatch')
        p=self.root/'analysis_manifest.csv';original=p.read_bytes();p.write_bytes(original+b'\n')
        with self.assertRaises(ContractError) as caught:load_proposed_split(self.root,expected_report_sha256=self.report_hash)
        self.assertEqual(caught.exception.code,'proposal_payload_mismatch');p.write_bytes(original)
        (self.root/'unlisted.payload').write_bytes(b'INVENTED extra protected bytes')
        with self.assertRaises(ContractError) as caught:load_proposed_split(self.root,expected_report_sha256=self.report_hash)
        self.assertEqual(caught.exception.code,'unexpected_proposal_files')

    def test_report_hash_refresh_does_not_allow_serialized_identity_or_label_changes(self):
        p=self.root/'analysis_manifest.csv';lines=p.read_text().splitlines()
        fields=lines[1].split(',');fields[2]='nv' if fields[2]!='nv' else 'bcc';lines[1]=','.join(fields)
        p.write_text('\n'.join(lines)+'\n')
        report=json.loads((self.root/'proposal_report.json').read_text())
        report['artifact_sha256'][p.name]=file_sha256(p)
        (self.root/'proposal_report.json').write_bytes(canonical_json_bytes(report))
        with self.assertRaises(ContractError) as caught:
            load_proposed_split(self.root,expected_report_sha256=file_sha256(self.root/'proposal_report.json'))
        self.assertEqual(caught.exception.code,'invalid_conservative_grouping')


if __name__=='__main__':unittest.main()
