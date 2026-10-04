"""Approved selection contracts, using invented text and fingerprints only."""
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_streamed_candidate_review as fixtures
import complete_ham_candidate_screen as complete
import prepare_ham_candidate_review as public
import prepare_ham_ranked_proposal as proposal
from prepare_ham_ranked_review import prepare
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, sha256_bytes, read_manifest, manifest_csv_bytes
from dermdetect.data.review_policy import APPROVAL_REFERENCE, RANKED_PROCEDURE, RANKED_STATUS, validate_review_screen
from dermdetect.data.schemas import ContractError


class RankedReviewTests(unittest.TestCase):
    def setUp(self):
        case = fixtures.StreamedReviewTests("runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.root = case.root
        self.args = case.fixture({"invented_a":0, "invented_b":1, "invented_c":2**64-1, "invented_d":0})
        manifest, inventory, checkpoint, checkpoint_digest, published, screen = self.args
        # Cross-label candidates must survive without class-based filtering.
        rows = [replace(row, dx="bcc") if row.image_id == "invented_b" else row for row in read_manifest(manifest)]
        manifest.write_bytes(manifest_csv_bytes(rows))
        cached = json.loads(checkpoint.read_text());cached["manifest_sha256"] = file_sha256(manifest)
        checkpoint.write_bytes(canonical_json_bytes(cached));checkpoint_digest = file_sha256(checkpoint)
        self.args = (manifest, inventory, checkpoint, checkpoint_digest, published, screen)
        source_dir = self.root / "sources"
        source_dir.mkdir()
        content = b"similarity,from_img,to_img\n0.99,invented_a.jpg,invented_c.jpg\n"
        (source_dir / "invented.csv").write_bytes(content)
        (source_dir / "source.json").write_bytes(canonical_json_bytes({"commit":"invented-commit"}))
        self.enterContext(patch.object(public, "COMMIT", "invented-commit"))
        self.enterContext(patch.object(public, "SOURCES", {"invented.csv":("reported_missed_duplicate",1,sha256_bytes(content))}))
        published.write_bytes(canonical_json_bytes(public.prepare(manifest, source_dir)))
        complete.prepare(*self.args, published_sha256=file_sha256(published), published_source_dir=source_dir)
        (self.root / "integrity_report.json").write_bytes(inventory.read_bytes())
        ranking = self.root / "ranking.csv"
        rank_bytes = (b"similarity,from_img,to_img\n0.99,invented_a.jpg,invented_c.jpg\n"
                      b"0.98,invented_b.jpg,invented_c.jpg\n0.97,invented_c.jpg,invented_d.jpg\n")
        ranking.write_bytes(rank_bytes)
        ranking_source = self.root / "ranking_source.json"
        ranking_source.write_bytes(canonical_json_bytes({"commit":"invented-commit", "sha256":file_sha256(ranking),
                                                      "bytes":len(rank_bytes)}))
        for name,value in (("COMMIT","invented-commit"), ("RANK_SHA256",file_sha256(ranking)),
                           ("PUBLISHED_SHA256",file_sha256(published)), ("RANK_COUNT",3)):
            self.enterContext(patch.object(proposal,name,value))
        self.run_args = (manifest, screen, checkpoint, checkpoint_digest, ranking, published,
                         ranking_source, source_dir, self.root / "active")

    def invoke(self):
        return prepare(*self.run_args, approval_reference=APPROVAL_REFERENCE)

    def blocked(self, callback=None):
        with self.assertRaises(ContractError):
            (callback or self.invoke)()
        self.assertFalse(self.run_args[-1].exists())

    def test_selected_union_includes_independent_misses_and_preserves_unadjudicated_outside(self):
        report = self.invoke()
        self.assertEqual(report["review_candidates"],4)
        self.assertEqual(report["independent_ranked_pairs_absent_from_broad_queue"],2)
        self.assertEqual(report["broad_pairs_outside_approved_review"],2)
        self.assertEqual(report["status"], RANKED_STATUS)
        self.assertEqual(validate_review_screen(report, report["manifest_sha256"],report["candidates_sha256"]),RANKED_PROCEDURE)
        output = self.run_args[-1]
        with (output / "candidates.csv").open(newline="") as handle:
            selected = list(csv.DictReader(handle))
        self.assertEqual({(r["left"],r["right"]) for r in selected}, {
            ("invented_a","invented_c"), ("invented_b","invented_c"),
            ("invented_c","invented_d"), ("invented_a","invented_d")})
        self.assertTrue(all(r["decision"]=="pending" for r in selected))
        with (output / "broad_outside.csv").open(newline="") as handle:
            outside = list(csv.DictReader(handle))
        self.assertTrue(all(r["decision"]=="unadjudicated" for r in outside))
        views = json.loads((output / "view_queue.json").read_text())
        self.assertEqual(len(views["candidates"]),4)
        self.assertEqual(report["view_queue_sha256"],file_sha256(output / "view_queue.json"))
        for dirname,key in (("input_snapshot","input_snapshot_sha256"),("source_snapshot","source_snapshot_sha256")):
            for name,digest in report[key].items():
                self.assertEqual(file_sha256(output / dirname / name),digest)
        self.assertFalse(report["production_freeze"])

    def test_missing_or_wrong_selection_approval_is_rejected(self):
        for reference in ("", "unrelated approval"):
            self.blocked(lambda:prepare(*self.run_args,approval_reference=reference))

    def test_ranking_bytes_or_source_record_tampering_fail(self):
        ranking=self.run_args[4]
        ranking.write_bytes(ranking.read_bytes()+b" ")
        self.blocked()

    def test_missing_selected_lookup_in_checkpoint_fails_even_with_updated_digest(self):
        checkpoint=self.run_args[2]
        value=json.loads(checkpoint.read_text())
        del value["fingerprints"]["invented_c"]
        checkpoint.write_bytes(canonical_json_bytes(value))
        new_digest=file_sha256(checkpoint)
        self.run_args=(*self.run_args[:3],new_digest,*self.run_args[4:])
        rp=self.run_args[1]/"screen_report.json"
        report=json.loads(rp.read_text());report["checkpoint_sha256"]=new_digest
        rp.write_bytes(canonical_json_bytes(report))
        self.blocked()

    def test_checkpoint_stale_and_source_metadata_mismatch_stop(self):
        checkpoint=self.run_args[2]
        checkpoint.write_bytes(checkpoint.read_bytes()+b" ")
        self.blocked()

    def test_ranking_source_revision_mismatch_stops(self):
        source=self.run_args[6]
        value=json.loads(source.read_text());value["commit"]="different-commit"
        source.write_bytes(canonical_json_bytes(value))
        self.blocked()

    def test_broad_row_deletion_cannot_be_rehashed_into_complete_evidence(self):
        queue=self.run_args[1]/"candidates.csv"
        with queue.open(newline="") as handle:
            values=list(csv.DictReader(handle))
        values.pop()
        from prepare_ham_ranked_review import csv_bytes
        queue.write_bytes(csv_bytes(values))
        rp=self.run_args[1]/"screen_report.json"
        report=json.loads(rp.read_text())
        report["candidates_sha256"]=file_sha256(queue);report["review_candidates"]=len(values)
        rp.write_bytes(canonical_json_bytes(report))
        self.blocked()

    def test_missing_inventory_binding_and_overwrite_fail(self):
        (self.root / "integrity_report.json").write_text("changed inventory")
        self.blocked()


if __name__ == "__main__":
    unittest.main()
