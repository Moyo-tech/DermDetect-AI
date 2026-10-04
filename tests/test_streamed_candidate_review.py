"""Exhaustiveness, bound failures, cached evidence and final review gates."""
import csv
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data import image_integrity
from dermdetect.data.candidate_screen import iter_hash_pairs, PROCEDURE
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, manifest_csv_bytes, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError, ManifestRecord
from complete_ham_candidate_screen import prepare
from resolve_ham_analysis import resolve, VIEW_TRANSFORM


class StreamedReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dermdetect-stream-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, hashes=None, same_original=False):
        hashes = hashes or {"invented_a": 0, "invented_b": 1, "invented_c": 2**64 - 1}
        manifest = self.root / "manifest.csv"
        rows = [ManifestRecord(key, "invented_lesion" if same_original else key, "nv", "invented", "v1",
                               "invented/" + key + ".jpg", sha256_bytes(key.encode())) for key in hashes]
        manifest.write_bytes(manifest_csv_bytes(rows))
        inventory = self.root / "inventory.json"
        inventory.write_bytes(canonical_json_bytes({"local_config_sha256": "a" * 64,
                             "archives": [{"alias": "invented", "sha256": "b" * 64}]}))
        checkpoint = self.root / "checkpoint.json"
        checkpoint.write_bytes(canonical_json_bytes({"manifest_sha256": file_sha256(manifest),
            "local_config_sha256": "a" * 64, "archive_sha256": {"invented": "b" * 64},
            "decoder_source_sha256": file_sha256(Path(image_integrity.__file__)),
            "status": "all_listed_jpegs_decoded_candidate_screen_pending", "records_decoded": len(rows),
            "fingerprints": {key: {"width": 2, "height": 2, "dhash64": value,
                                    "rgb_sha256": sha256_bytes(key.encode())} for key, value in hashes.items()}}))
        published = self.root / "published.json"
        # Deliberately mandatory despite a hash miss, when available.
        published.write_bytes(canonical_json_bytes({"manifest_sha256": file_sha256(manifest), "candidates":
            [{"left": "invented_a", "right": "invented_c", "kind": "invented_known_pair"}]
            if "invented_c" in hashes else []}))
        return manifest, inventory, checkpoint, file_sha256(checkpoint), published, self.root / "screen"

    def test_index_matches_bruteforce_at_boundary_and_is_input_order_invariant(self):
        rng = random.Random(17)
        hashes = {f"invented_{i:03}": rng.getrandbits(64) for i in range(120)}
        base = hashes["invented_000"]
        hashes.update({"four_bands": base ^ sum(1 << k for k in (0, 13, 26, 39)),
                       "five_bands": base ^ sum(1 << k for k in (0, 13, 26, 39, 52)), "equal": base})
        expected = {(a, b, (hashes[a] ^ hashes[b]).bit_count()) for a, b in itertools.combinations(sorted(hashes), 2)
                    if (hashes[a] ^ hashes[b]).bit_count() <= 4}
        actual = tuple(iter_hash_pairs(hashes))
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(actual, tuple(iter_hash_pairs(dict(reversed(list(hashes.items()))))))

    def test_full_output_crosses_old_cap_without_truncation_or_decisions(self):
        args = self.fixture({f"invented_{i:03}": 0 for i in range(201)})
        result = prepare(*args)
        self.assertEqual(result["all_hash_matches"], 20100)
        self.assertEqual(result["review_candidates"], 20100)
        with (args[-1] / "candidates.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20100)
        self.assertEqual(len({(r["left"], r["right"]) for r in rows}), 20100)
        self.assertTrue(all(r["decision"] == "pending" for r in rows))
        self.assertFalse(result["split_ready"])
        self.assertFalse(result["production_freeze"])

    def test_mandatory_miss_retained_and_same_id_hash_matches_audited(self):
        args = self.fixture(same_original=True)
        result = prepare(*args)
        self.assertEqual(result["within_original_id_hash_matches"], 1)
        self.assertEqual(result["cross_id_hash_matches"], 0)
        with (args[-1] / "candidates.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reasons"], "invented_known_pair")
        self.assertEqual(rows[0]["dhash_distance"], "64")

    def test_checkpoint_tampering_and_incomplete_membership_fail_before_output(self):
        for change in ("binding", "membership", "bool_hash", "rgb_hash"):
            with self.subTest(change=change):
                args = self.fixture()
                value = json.loads(args[2].read_text())
                if change == "binding": value["manifest_sha256"] = "0" * 64
                if change == "membership": del value["fingerprints"]["invented_a"]
                if change == "bool_hash": value["fingerprints"]["invented_a"]["dhash64"] = True
                if change == "rgb_hash": value["fingerprints"]["invented_a"]["rgb_sha256"] = "wrong"
                args[2].write_bytes(canonical_json_bytes(value))
                changed_args = (*args[:3], file_sha256(args[2]), *args[4:])
                with self.assertRaises(ContractError): prepare(*changed_args)
                self.assertFalse(args[-1].exists())
        args = self.fixture()
        args[2].write_bytes(args[2].read_bytes() + b" ")
        with self.assertRaises(ContractError): prepare(*args)
        self.assertFalse(args[-1].exists())

    def test_storage_failure_keeps_partial_and_has_no_complete_report(self):
        args = self.fixture()
        with patch("complete_ham_candidate_screen.MAX_OUTPUT_BYTES", 160):
            with self.assertRaises(ContractError) as caught: prepare(*args)
        self.assertEqual(caught.exception.code, "screen_storage_bound")
        self.assertTrue((args[-1] / "failure.json").exists())
        self.assertFalse((args[-1] / "screen_report.json").exists())
        self.assertFalse((args[-1] / "candidates.csv").exists())

    def reviewed_fixture(self):
        args = self.fixture()
        report = prepare(*args)
        with (args[-1] / "candidates.csv").open(newline="") as handle:
            pairs = list(csv.DictReader(handle))
        decisions = [{"left": c["left"], "right": c["right"], "decision": "different_lesions",
                      "review_date": "2026-10-02", "reviewer": "invented reviewer",
                      "rationale": "invented distinct structures", "evidence": "invented visual evidence"} for c in pairs]
        # Invented evidence bytes exercise binding only; these are not images
        # or a claim that any real human review has occurred.
        manifest_rows = {r.image_id: r for r in read_manifest(args[0])}
        views = []
        for number, item in enumerate(decisions):
            ids = [item["left"], item["right"]]
            view = self.root / f"invented_view_{number}.txt"
            view.write_text("INVENTED EVIDENCE ONLY: " + ",".join(ids))
            item.update({"evidence": view.name, "view_sha256": file_sha256(view),
                         "source_payload_sha256": {i: manifest_rows[i].sha256 for i in ids},
                         "source_applicability": {i: {"source": manifest_rows[i].source, "source_version": manifest_rows[i].source_version} for i in ids},
                         "view_repetition": "byte_distinct_views"})
            views.append({"image_ids": ids, "source_payload_sha256": [manifest_rows[i].sha256 for i in ids],
                          "view_file": view.name, "view_sha256": file_sha256(view), "view_transform": VIEW_TRANSFORM})
        visual_index = self.root / "visual_index.json"
        visual_index.write_bytes(canonical_json_bytes({"manifest_sha256": report["manifest_sha256"],
                                 "status": "visual_evidence_prepared_not_adjudicated", "pairs": views}))
        value = {"manifest_sha256": report["manifest_sha256"], "procedure": PROCEDURE,
                 "candidate_queue_sha256": report["candidates_sha256"],
                 "screen_report_sha256": file_sha256(args[-1] / "screen_report.json"), "draft": False,
                 "visual_index": visual_index.name, "visual_index_sha256": file_sha256(visual_index),
                 "decisions": decisions, "human_acceptance": {"reviewer": "invented human", "date": "2026-10-02",
                     "approval_reference": "invented authorization, not real", "accept_procedure": True,
                     "accept_identity_and_participation": True,
                     "decision_sha256": sha256_bytes(canonical_json_bytes(decisions))}}
        review = self.root / "review.json"
        review.write_bytes(canonical_json_bytes(value))
        return args, review, value

    def test_complete_invented_review_produces_all_rows_without_split(self):
        args, review, _ = self.reviewed_fixture()
        result = resolve(args[0], args[-1], review, self.root / "analysis")
        self.assertEqual(result["original_records"], 3)
        self.assertEqual(result["analysis_lesions"], 3)
        self.assertFalse(result["production_freeze"])
        self.assertEqual({p.name for p in (self.root / "analysis").iterdir()},
                         {"analysis_manifest.csv", "analysis_report.json"})

    def test_missing_pending_unaccepted_and_forged_reviews_stop_without_output(self):
        args, review, original = self.reviewed_fixture()
        for change, code in (("missing", "candidate_membership_mismatch"), ("pending", "unresolved_candidate"),
                             ("draft", "human_acceptance_pending"), ("hash", "review_binding_mismatch"),
                             ("date", "missing_review_date"), ("approval", "human_acceptance_pending")):
            with self.subTest(change=change):
                value = json.loads(json.dumps(original))
                if change == "missing": value["decisions"].pop()
                if change == "pending": value["decisions"][0]["decision"] = "pending"
                if change == "draft": value["draft"] = True
                if change == "hash": value["candidate_queue_sha256"] = "0" * 64
                if change == "date": value["decisions"][0]["review_date"] = "2026-1-1"
                if change == "approval": value["human_acceptance"]["accept_procedure"] = False
                review.write_bytes(canonical_json_bytes(value))
                with self.assertRaises(ContractError) as caught:
                    resolve(args[0], args[-1], review, self.root / "blocked")
                self.assertEqual(caught.exception.code, code)
                self.assertFalse((self.root / "blocked").exists())

    def test_missing_changed_unrooted_or_unbound_visual_evidence_stops_resolution(self):
        args, review, original = self.reviewed_fixture()
        for change in ("missing_index", "index_hash", "payload_hash", "source_version", "transform", "repetition", "view_bytes", "escape"):
            with self.subTest(change=change):
                value = json.loads(json.dumps(original))
                index_path = self.root / "visual_index.json"
                saved_index = index_path.read_bytes()
                view_path = self.root / value["decisions"][0]["evidence"]
                saved_view = view_path.read_bytes()
                if change == "missing_index": value.pop("visual_index")
                if change == "index_hash": value["visual_index_sha256"] = "0" * 64
                if change == "payload_hash": value["decisions"][0]["source_payload_sha256"]["invented_a"] = "0" * 64
                if change == "source_version": value["decisions"][0]["source_applicability"]["invented_a"]["source_version"] = "wrong"
                if change == "repetition": value["decisions"][0]["view_repetition"] = "exact_content_repeat"
                if change == "transform":
                    index = json.loads(saved_index)
                    index["pairs"][0]["view_transform"] = "cropped thumbnail"
                    index_path.write_bytes(canonical_json_bytes(index))
                    value["visual_index_sha256"] = file_sha256(index_path)
                if change == "view_bytes": view_path.write_bytes(b"CHANGED INVENTED EVIDENCE")
                if change == "escape": value["visual_index"] = "../visual_index.json"
                value["human_acceptance"]["decision_sha256"] = sha256_bytes(canonical_json_bytes(value["decisions"]))
                review.write_bytes(canonical_json_bytes(value))
                with self.assertRaises(ContractError):
                    resolve(args[0], args[-1], review, self.root / "blocked_visual")
                self.assertFalse((self.root / "blocked_visual").exists())
                index_path.write_bytes(saved_index)
                view_path.write_bytes(saved_view)


if __name__ == "__main__":
    unittest.main()
