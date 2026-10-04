"""Prospective review membership remains pinned, complete and unapproved."""
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data.manifests import file_sha256
from dermdetect.data.schemas import ContractError
from complete_ham_candidate_screen import prepare as screen
import prepare_ham_ranked_proposal as proposal
import test_streamed_candidate_review as fixtures


class RankedProposalTests(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.StreamedReviewTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.args = self.case.fixture()
        screen(*self.args)
        self.root = self.case.root
        self.rank = self.root / "rank.csv"
        self.rank.write_text("similarity,from_img,to_img\n0.99,invented_a.jpg,invented_c.jpg\n0.95,invented_a.jpg,invented_b.jpg\n0.90,invented_b.jpg,invented_c.jpg\n")
        self.enterContext(patch.object(proposal, "PUBLISHED_SHA256", file_sha256(self.args[4])))
        self.enterContext(patch.object(proposal, "RANK_COUNT", 3))

    def invoke(self):
        return proposal.prepare(self.args[0], self.args[-1], self.rank, self.args[4], self.root / "proposal.json")

    def test_independent_candidates_join_without_any_identity_or_split_approval(self):
        with patch.object(proposal, "RANK_SHA256", file_sha256(self.rank)):
            result = self.invoke()
        self.assertEqual(result["proposed_review_pairs"], 3)
        self.assertEqual(result["independent_ranked_pairs_absent_from_broad_queue"], 1)
        self.assertEqual(result["status"], "proposed_review_set_not_approved")
        self.assertFalse(result["production_freeze"])
        self.assertFalse(result["split_ready"])
        self.assertTrue(all(c["decision"] == "pending" and c["reviewer"] == "" for c in result["candidates"]))
        self.assertEqual(result["human_decisions_recorded"], 0)

    def test_changed_pins_or_invalid_order_fail_without_proposal(self):
        with patch.object(proposal, "RANK_SHA256", "0" * 64), self.assertRaises(ContractError):
            self.invoke()
        self.assertFalse((self.root / "proposal.json").exists())
        lines = self.rank.read_text().splitlines()
        self.rank.write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
        with patch.object(proposal, "RANK_SHA256", file_sha256(self.rank)), self.assertRaises(ContractError):
            self.invoke()
        self.assertFalse((self.root / "proposal.json").exists())


if __name__ == "__main__":
    unittest.main()
