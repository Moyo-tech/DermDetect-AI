"""Invented view bytes only: preservation, membership, provenance and safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, manifest_csv_bytes, sha256_bytes
from dermdetect.data.review_policy import APPROVAL_REFERENCE, RANKED_PROCEDURE, RANKED_STATUS
from dermdetect.data.schemas import ContractError, ManifestRecord
from prepare_ham_review_views import VIEW_STATUS, merge, remaining
from resolve_ham_analysis import VIEW_TRANSFORM


class ReviewViewsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.rows = {key: ManifestRecord(key, "lesion_" + key, "nv", "invented", "v1",
            f"invented/{key}.txt", sha256_bytes(key.encode())) for key in ("a", "b", "c", "d")}
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_bytes(manifest_csv_bytes(self.rows.values()))
        self.manifest_digest = file_sha256(self.manifest)
        self.pairs = [("a", "b"), ("a", "c"), ("c", "d")]
        self.active = self.root / "active.json"
        self.active.write_bytes(canonical_json_bytes({"manifest_sha256": self.manifest_digest,
            "procedure": RANKED_PROCEDURE, "status": RANKED_STATUS,
            "approval_reference": APPROVAL_REFERENCE, "production_freeze": False,
            "candidates": [{"left": a, "right": b, "kind": "invented_candidate"} for a, b in self.pairs]}))
        self.old = self.make_index("old", self.pairs[:1], "e" * 64)
        self.old_bytes = (self.old.parent / "pair_01.png").read_bytes()
        self.remaining = self.root / "remaining.json"
        self.output = self.root / "merged_index.json"

    def make_index(self, directory, pairs, queue_sha256):
        folder = self.root / directory
        folder.mkdir()
        views = []
        for number, ids in enumerate(pairs, 1):
            path = folder / f"pair_{number:02}.png"
            # Text suffices to verify artifact binding; never actual images.
            path.write_bytes(("INVENTED FULL VIEW BYTES: " + ",".join(ids)).encode())
            views.append({"pair_number": number, "image_ids": list(ids),
                "original_lesion_ids": [self.rows[i].lesion_id for i in ids],
                "source_payload_sha256": [self.rows[i].sha256 for i in ids],
                "view_file": path.name, "view_sha256": file_sha256(path),
                "view_transform": VIEW_TRANSFORM, "decision": "pending", "reviewer": "", "rationale": ""})
        index = folder / "index.json"
        index.write_bytes(canonical_json_bytes({"manifest_sha256": self.manifest_digest,
            "status": VIEW_STATUS, "production_freeze": False, "queue_sha256": queue_sha256, "pairs": views}))
        return index

    def prepare_new(self):
        result = remaining(self.manifest, self.active, self.old, self.remaining)
        self.assertEqual(result["remaining_views"], 2)
        return self.make_index("new", self.pairs[1:], file_sha256(self.remaining))

    def assert_reject(self, code, call, output):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(output.exists())

    def test_remaining_derivation_reuses_verified_views_without_rewriting_them(self):
        result = remaining(self.manifest, self.active, self.old, self.remaining)
        self.assertEqual((result["active_candidates"], result["preserved_views"], result["remaining_views"]), (3, 1, 2))
        self.assertEqual([(item["left"], item["right"]) for item in result["candidates"]], self.pairs[1:])
        self.assertEqual(result["preserved_visual_index_sha256"], file_sha256(self.old))
        self.assertEqual(result["active_view_queue_sha256"], file_sha256(self.active))
        self.assertEqual((self.old.parent / "pair_01.png").read_bytes(), self.old_bytes)

    def test_merge_exact_coverage_prefixes_routes_and_preserves_old_bytes_hashes(self):
        new = self.prepare_new()
        result = merge(self.manifest, self.active, self.old, self.remaining, new, self.output)
        self.assertEqual((result["preserved_views"], result["new_views"], result["human_decisions_recorded"]), (1, 2, 0))
        self.assertFalse(result["production_freeze"])
        self.assertEqual(result["queue_sha256"], file_sha256(self.active))
        self.assertEqual([tuple(v["image_ids"]) for v in result["pairs"]], self.pairs)
        self.assertEqual(result["pairs"][0]["view_file"], "old/pair_01.png")
        self.assertEqual(result["pairs"][0]["view_sha256"], sha256_bytes(self.old_bytes))
        for view in result["pairs"]:
            self.assertNotIn("..", Path(view["view_file"]).parts)
            self.assertEqual(file_sha256(self.root / view["view_file"]), view["view_sha256"])
        self.assertEqual(result["source_indices"][0]["sha256"], file_sha256(self.old))
        self.assertEqual((self.old.parent / "pair_01.png").read_bytes(), self.old_bytes)

    def test_changed_old_payload_transform_source_or_pending_state_stops_derivation(self):
        original = self.old.read_bytes()
        for field, value in (("view_transform", "thumbnail"), ("source_payload_sha256", ["0" * 64] * 2),
                             ("decision", "same_lesion"), ("reviewer", "invented reviewer")):
            index = json.loads(original)
            index["pairs"][0][field] = value
            self.old.write_bytes(canonical_json_bytes(index))
            self.assert_reject("visual_evidence_mismatch",
                lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)
        self.old.write_bytes(original)
        (self.old.parent / "pair_01.png").write_bytes(b"CHANGED SAVED VIEW")
        self.assert_reject("visual_evidence_mismatch",
            lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)

    def test_remaining_queue_tamper_and_missing_overlap_or_wrong_source_new_views_stop_merge(self):
        new = self.prepare_new()
        original_remaining, original_new = self.remaining.read_bytes(), new.read_bytes()
        changed = json.loads(original_remaining)
        changed["candidates"].pop()
        self.remaining.write_bytes(canonical_json_bytes(changed))
        self.assert_reject("review_binding_mismatch",
            lambda: merge(self.manifest, self.active, self.old, self.remaining, new, self.output), self.output)
        self.remaining.write_bytes(original_remaining)
        for mutation in ("missing", "overlap", "queue_hash"):
            index = json.loads(original_new)
            if mutation == "missing": index["pairs"].pop()
            if mutation == "overlap":
                index["pairs"][0]["image_ids"] = ["a", "b"]
                index["pairs"][0]["original_lesion_ids"] = [self.rows[i].lesion_id for i in ("a", "b")]
                index["pairs"][0]["source_payload_sha256"] = [self.rows[i].sha256 for i in ("a", "b")]
            if mutation == "queue_hash": index["queue_sha256"] = "0" * 64
            new.write_bytes(canonical_json_bytes(index))
            self.assert_reject("candidate_membership_mismatch",
                lambda: merge(self.manifest, self.active, self.old, self.remaining, new, self.output), self.output)

    def test_traversal_symlink_and_index_placement_are_refused(self):
        original = self.old.read_bytes()
        index = json.loads(original)
        index["pairs"][0]["view_file"] = "../old/pair_01.png"
        self.old.write_bytes(canonical_json_bytes(index))
        self.assert_reject("unsafe_path", lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)
        self.old.write_bytes(original)
        alias = self.old.parent / "alias.png"
        alias.symlink_to(self.old.parent / "pair_01.png")
        index["pairs"][0]["view_file"] = "alias.png"
        self.old.write_bytes(canonical_json_bytes(index))
        self.assert_reject("unsafe_path", lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)
        self.old.write_bytes(original)
        new = self.prepare_new()
        nested_output = self.old.parent / "merged_index.json"
        self.assert_reject("unsafe_path",
            lambda: merge(self.manifest, self.active, self.old, self.remaining, new, nested_output), nested_output)

    def test_duplicate_unknown_old_pairs_and_overwrite_are_refused(self):
        original = self.old.read_bytes()
        index = json.loads(original)
        index["pairs"].append(index["pairs"][0])
        self.old.write_bytes(canonical_json_bytes(index))
        self.assert_reject("visual_evidence_mismatch", lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)
        self.old.write_bytes(original)
        queue = json.loads(self.active.read_bytes())
        queue["candidates"] = queue["candidates"][1:]
        self.active.write_bytes(canonical_json_bytes(queue))
        self.assert_reject("candidate_membership_mismatch", lambda: remaining(self.manifest, self.active, self.old, self.remaining), self.remaining)


if __name__ == "__main__":
    unittest.main()
