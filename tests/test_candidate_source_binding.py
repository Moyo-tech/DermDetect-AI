"""Source completeness checks with invented CSV text and cached fingerprints."""
import json
import csv
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import tracemalloc
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import test_streamed_candidate_review as fixtures
import prepare_ham_candidate_review as published_review
from complete_ham_candidate_screen import prepare
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, sha256_bytes, read_manifest, manifest_csv_bytes
from dermdetect.data.schemas import ContractError


class CandidateSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture_case = fixtures.StreamedReviewTests("runTest")
        self.fixture_case.setUp()
        self.addCleanup(self.fixture_case.doCleanups)
        self.args = self.fixture_case.fixture()
        self.root = self.fixture_case.root
        inventory = json.loads(self.args[1].read_text())
        inventory["data_kind"] = "real"  # Contract test; every payload/ID is invented.
        self.args[1].write_bytes(canonical_json_bytes(inventory))
        self.source_dir = self.root / "sources"
        self.source_dir.mkdir()
        content = b"similarity,from_img,to_img\n0.99,invented_a.jpg,invented_c.jpg\n"
        (self.source_dir / "invented.csv").write_bytes(content)
        (self.source_dir / "source.json").write_bytes(canonical_json_bytes({"commit": "invented-commit"}))
        self.enterContext(patch.object(published_review, "COMMIT", "invented-commit"))
        self.enterContext(patch.object(published_review, "SOURCES", {
            "invented.csv": ("invented_known_pair", 1, sha256_bytes(content))}))
        self.expected = published_review.prepare(self.args[0], self.source_dir)
        self.args[4].write_bytes(canonical_json_bytes(self.expected))

    def invoke(self, **kwargs):
        return prepare(*self.args, published_sha256=file_sha256(self.args[4]),
                       published_source_dir=self.source_dir, **kwargs)

    def assert_blocked(self, code, callback):
        with self.assertRaises(ContractError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        self.assertFalse(self.args[-1].exists())

    def test_real_evidence_requires_explicit_queue_pin_and_source_directory(self):
        for kwargs in ({}, {"published_sha256": file_sha256(self.args[4])},
                       {"published_source_dir": self.source_dir}):
            with self.subTest(kwargs=kwargs):
                self.assert_blocked("candidate_source_binding_required", lambda: prepare(*self.args, **kwargs))

    def test_complete_pinned_text_sources_bind_report_and_retain_hash_miss(self):
        report = self.invoke()
        self.assertEqual(report["mandatory_pairs"], 1)
        self.assertEqual(report["review_candidates"], 2)
        self.assertEqual(report["published_queue_binding_kind"], "reconstructed_from_pinned_textual_sources")
        self.assertEqual(report["expected_published_queue_sha256"], file_sha256(self.args[4]))
        self.assertEqual(report["published_source_files_sha256"]["invented.csv"],
                         file_sha256(self.source_dir / "invented.csv"))
        self.assertEqual(report["published_preparation_source_sha256"], file_sha256(Path(published_review.__file__)))
        self.assertFalse(report["production_freeze"])

    def test_removed_or_changed_pairs_fail_even_with_new_matching_queue_hash(self):
        for change in ("remove", "replace", "source_hash", "payload_hash", "commit"):
            with self.subTest(change=change):
                value = json.loads(json.dumps(self.expected))
                if change == "remove": value["candidates"] = []
                if change == "replace": value["candidates"][0]["right"] = "invented_b"
                if change == "source_hash": value["candidates"][0]["source_file_sha256"] = "0" * 64
                if change == "payload_hash": value["candidates"][0]["payload_sha256"][0] = "0" * 64
                if change == "commit": value["public_candidate_commit"] = "changed"
                self.args[4].write_bytes(canonical_json_bytes(value))
                self.assert_blocked("candidate_source_mismatch", self.invoke)

    def test_source_bytes_or_revision_change_fails_before_output(self):
        (self.source_dir / "invented.csv").write_bytes(b"changed invented source")
        self.assert_blocked("candidate_source_mismatch", self.invoke)

    def test_explicit_recorded_queue_hash_mismatch_fails(self):
        self.assert_blocked("candidate_source_mismatch", lambda: prepare(*self.args,
            published_sha256="0" * 64, published_source_dir=self.source_dir))

    def equality_fixture(self, count=2, same_bytes=False):
        args = self.fixture_case.fixture({f"invented_{i:03}": 0 for i in range(count)})
        checkpoint = json.loads(args[2].read_text())
        for value in checkpoint["fingerprints"].values():
            value["rgb_sha256"] = "d" * 64
        if same_bytes:
            rows = [replace(row, sha256="e" * 64) for row in read_manifest(args[0])]
            args[0].write_bytes(manifest_csv_bytes(rows))
            checkpoint["manifest_sha256"] = file_sha256(args[0])
            published = json.loads(args[4].read_text())
            published["manifest_sha256"] = file_sha256(args[0])
            args[4].write_bytes(canonical_json_bytes(published))
        args[2].write_bytes(canonical_json_bytes(checkpoint))
        return (*args[:3], file_sha256(args[2]), *args[4:])

    def test_equal_rgb_and_bytes_stream_past_old_cap_with_bounded_memory(self):
        args = self.equality_fixture(count=201, same_bytes=True)
        tracemalloc.start()
        try:
            report = prepare(*args)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(report["review_candidates"], 20100)
        self.assertLess(peak, 4 * 1024 * 1024)
        with (args[-1] / "candidates.csv").open(newline="") as handle:
            count = 0
            for row in csv.DictReader(handle):
                count += 1
                self.assertEqual(set(row["reasons"].split("|")), {
                    "dhash64_hamming_at_most_4", "identical_oriented_decoded_rgb", "identical_payload_sha256"})
            self.assertEqual(count, 20100)

    def test_inconsistent_rgb_dimensions_or_dhash_fail_before_output(self):
        for change in ("width", "height", "dhash64"):
            with self.subTest(change=change):
                args = self.equality_fixture()
                checkpoint = json.loads(args[2].read_text())
                checkpoint["fingerprints"]["invented_001"][change] += 1
                args[2].write_bytes(canonical_json_bytes(checkpoint))
                changed = (*args[:3], file_sha256(args[2]), *args[4:])
                with self.assertRaises(ContractError) as caught:
                    prepare(*changed)
                self.assertEqual(caught.exception.code, "inconsistent_exact_fingerprint")
                self.assertFalse(args[-1].exists())

    def test_identical_bytes_require_complete_equal_fingerprints(self):
        args = self.equality_fixture(same_bytes=True)
        checkpoint = json.loads(args[2].read_text())
        # Same pixels/hash remain consistent; a backend provenance difference
        # must still stop reuse of purportedly identically decoded byte copies.
        checkpoint["fingerprints"]["invented_001"]["jpeg_backend_version"] = "invented-different-backend"
        args[2].write_bytes(canonical_json_bytes(checkpoint))
        changed = (*args[:3], file_sha256(args[2]), *args[4:])
        with self.assertRaises(ContractError) as caught:
            prepare(*changed)
        self.assertEqual(caught.exception.code, "inconsistent_exact_fingerprint")
        self.assertFalse(args[-1].exists())


if __name__ == "__main__":
    unittest.main()
