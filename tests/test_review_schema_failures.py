"""Malformed invented review artifacts fail structurally before any output."""
import csv
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_streamed_candidate_review as fixtures
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, sha256_bytes
from dermdetect.data.schemas import ContractError
from dermdetect.data.review_policy import APPROVAL_REFERENCE, RANKED_PROCEDURE, RANKED_STATUS
from resolve_ham_analysis import QUEUE_FIELDS, resolve


class ReviewSchemaFailureTests(unittest.TestCase):
    def setUp(self):
        self.fixture_owner = fixtures.StreamedReviewTests()
        self.fixture_owner.setUp()
        self.addCleanup(self.fixture_owner.doCleanups)
        self.args, self.review_path, self.review = self.fixture_owner.reviewed_fixture()
        self.root = self.fixture_owner.root
        self.screen = self.args[-1]
        self.queue_path = self.screen / "candidates.csv"
        self.original_queue = self.queue_path.read_bytes()
        self.report_path = self.screen / "screen_report.json"
        self.original_report = self.report_path.read_bytes()
        self.index_path = self.root / "visual_index.json"
        self.original_index = self.index_path.read_bytes()

    def save_review(self, value):
        self.review_path.write_bytes(canonical_json_bytes(value))

    def reject(self, code):
        output = self.root / "blocked"
        with self.assertRaises(ContractError) as caught:
            resolve(self.args[0], self.screen, self.review_path, output)
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(output.exists())

    def rebind_queue(self, payload):
        self.queue_path.write_bytes(payload)
        report = json.loads(self.original_report)
        report["candidates_sha256"] = file_sha256(self.queue_path)
        self.report_path.write_bytes(canonical_json_bytes(report))
        review = json.loads(json.dumps(self.review))
        review["candidate_queue_sha256"] = report["candidates_sha256"]
        review["screen_report_sha256"] = file_sha256(self.report_path)
        self.save_review(review)

    def test_nonobject_and_invalid_json_roots_fail_cleanly(self):
        for value in ([], None, "not an object", 7):
            with self.subTest(value=value):
                self.save_review(value)
                self.reject("invalid_review_artifact")
        self.review_path.write_bytes(b"{invalid JSON")
        self.reject("invalid_review_artifact")
        self.save_review(self.review)
        self.report_path.write_bytes(canonical_json_bytes([]))
        self.reject("invalid_review_artifact")

    def test_decision_rows_and_ids_are_structurally_validated(self):
        for entry in ([], None, "a decision", 7):
            value = json.loads(json.dumps(self.review))
            value["decisions"][0] = entry
            self.save_review(value)
            self.reject("invalid_review")
        for field in ("left", "right"):
            for identity in (None, [], {}, 7, "", " invented_a", "unknown"):
                value = json.loads(json.dumps(self.review))
                value["decisions"][0][field] = identity
                self.save_review(value)
                self.reject("invalid_review")

    def test_acceptance_nonobjects_fail_cleanly(self):
        for acceptance in (None, [], "approved", True):
            value = json.loads(json.dumps(self.review))
            value["human_acceptance"] = acceptance
            self.save_review(value)
            self.reject("human_acceptance_pending")

    def test_exact_csv_header_width_distance_reason_and_ids_are_required(self):
        rows = list(csv.reader(io.StringIO(self.original_queue.decode())))
        variants = []
        variants.append([list(QUEUE_FIELDS[:-1]), rows[1][:-1]])
        variants.append([list(QUEUE_FIELDS) + ["extra"], rows[1] + ["extra"]])
        variants.append([list(QUEUE_FIELDS), rows[1][:-1]])
        variants.append([list(QUEUE_FIELDS), rows[1] + ["extra"]])
        for column, value in ((0, "unknown"), (0, " invented_a"), (1, rows[1][0]),
                              (2, "65"), (2, "-1"), (2, "four"), (3, "")):
            changed = list(rows[1])
            changed[column] = value
            variants.append([list(QUEUE_FIELDS), changed])
        for variant in variants:
            stream = io.StringIO(newline="")
            csv.writer(stream, lineterminator="\n").writerows(variant)
            self.rebind_queue(stream.getvalue().encode())
            self.reject("invalid_candidate_queue")

    def test_visual_index_objects_ids_hashes_and_routes_are_required(self):
        for value in ([], None, 7):
            self.index_path.write_bytes(canonical_json_bytes(value))
            review = json.loads(json.dumps(self.review))
            review["visual_index_sha256"] = file_sha256(self.index_path)
            self.save_review(review)
            self.reject("invalid_review_artifact")
        for entry in ([], None, 7):
            index = json.loads(self.original_index)
            index["pairs"][0] = entry
            self.index_path.write_bytes(canonical_json_bytes(index))
            review = json.loads(json.dumps(self.review))
            review["visual_index_sha256"] = file_sha256(self.index_path)
            self.save_review(review)
            self.reject("visual_evidence_mismatch")
        for ids in ([[], "invented_a"], [None, "invented_a"], ["invented_a", "invented_a"]):
            index = json.loads(self.original_index)
            index["pairs"][0]["image_ids"] = ids
            self.index_path.write_bytes(canonical_json_bytes(index))
            review = json.loads(json.dumps(self.review))
            review["visual_index_sha256"] = file_sha256(self.index_path)
            self.save_review(review)
            self.reject("visual_evidence_mismatch")
        index = json.loads(self.original_index)
        index["pairs"][1]["view_file"] = index["pairs"][0]["view_file"]
        index["pairs"][1]["view_sha256"] = index["pairs"][0]["view_sha256"]
        self.index_path.write_bytes(canonical_json_bytes(index))
        review = json.loads(json.dumps(self.review))
        review["visual_index_sha256"] = file_sha256(self.index_path)
        self.save_review(review)
        self.reject("visual_evidence_mismatch")

    def test_reversed_repeated_decision_and_selfpair_stop(self):
        value = json.loads(json.dumps(self.review))
        item = dict(value["decisions"][0])
        item["left"], item["right"] = item["right"], item["left"]
        value["decisions"].append(item)
        self.save_review(value)
        self.reject("candidate_membership_mismatch")
        value = json.loads(json.dumps(self.review))
        value["decisions"][0]["right"] = value["decisions"][0]["left"]
        self.save_review(value)
        self.reject("invalid_review")

    def new_reviewed_fixture(self, hashes):
        owner = fixtures.StreamedReviewTests()
        owner.setUp()
        self.addCleanup(owner.doCleanups)
        owner.fixture = lambda: fixtures.StreamedReviewTests.fixture(owner, hashes=hashes)
        return owner, owner.reviewed_fixture()

    def test_665_invented_decisions_supported_with_ranked_policy_and_no_assignment(self):
        # Invented metadata/evidence only: exercise the actual queue size and
        # policy schema without claiming these are human or real-data decisions.
        owner, (args, review_path, review) = self.new_reviewed_fixture(
            {f"invented_{i:03}": 0 for i in range(38)})
        queue_path = args[-1] / "candidates.csv"
        with queue_path.open(newline="") as handle:
            queue = list(csv.DictReader(handle))[:665]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=QUEUE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(queue)
        queue_path.write_text(stream.getvalue())
        report_path = args[-1] / "screen_report.json"
        report = json.loads(report_path.read_text())
        report.update({"status": RANKED_STATUS, "procedure": RANKED_PROCEDURE,
            "procedure_approval": "approved_candidate_selection_only",
            "approval_reference": APPROVAL_REFERENCE, "human_decisions_recorded": 0,
            "broad_outside_status": "unadjudicated", "review_candidates": 665,
            "pending_decisions": 665, "candidates_sha256": file_sha256(queue_path)})
        report_path.write_bytes(canonical_json_bytes(report))
        index_path = owner.root / "visual_index.json"
        index = json.loads(index_path.read_text())
        index["pairs"] = index["pairs"][:665]
        index_path.write_bytes(canonical_json_bytes(index))
        review.update({"procedure": RANKED_PROCEDURE,
            "candidate_queue_sha256": file_sha256(queue_path),
            "screen_report_sha256": file_sha256(report_path),
            "visual_index_sha256": file_sha256(index_path), "decisions": review["decisions"][:665]})
        for value in (None, False, "true", 1):
            review["decisions"][0]["full_view_inspected"] = value
            review_path.write_bytes(canonical_json_bytes(review))
            with self.assertRaises(ContractError) as caught:
                resolve(args[0], args[-1], review_path, owner.root / "analysis")
            self.assertEqual(caught.exception.code, "missing_full_view_attestation")
            self.assertFalse((owner.root / "analysis").exists())
        for item in review["decisions"]:
            item["full_view_inspected"] = True
        review["human_acceptance"]["decision_sha256"] = sha256_bytes(canonical_json_bytes(review["decisions"]))
        review_path.write_bytes(canonical_json_bytes(review))
        result = resolve(args[0], args[-1], review_path, owner.root / "analysis")
        self.assertEqual(result["decision_count"], 665)
        self.assertEqual(result["original_records"], 38)
        self.assertFalse(result["production_freeze"])
        self.assertEqual(result["status"], "analysis_resolved_no_assignment_or_freeze")

    def test_hash_bound_full_visual_review_still_rejects_transitive_contradiction(self):
        owner, (args, review_path, review) = self.new_reviewed_fixture(
            {"invented_a": 0, "invented_b": 1, "invented_c": 2})
        for item in review["decisions"]:
            item["decision"] = "different_lesions" if "invented_a" not in (item["left"], item["right"]) else "same_lesion"
        review["human_acceptance"]["decision_sha256"] = sha256_bytes(canonical_json_bytes(review["decisions"]))
        review_path.write_bytes(canonical_json_bytes(review))
        with self.assertRaises(ContractError) as caught:
            resolve(args[0], args[-1], review_path, owner.root / "analysis")
        self.assertEqual(caught.exception.code, "contradictory_review")
        self.assertFalse((owner.root / "analysis").exists())


if __name__ == "__main__":
    unittest.main()
