"""Review-queue and boundary configuration checks using invented inputs."""
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data.access_boundary import development_profile
from dermdetect.data.image_integrity import decode_fingerprint
from dermdetect.data.manifests import manifest_csv_bytes, sha256_bytes
from dermdetect.data.schemas import ContractError, ManifestRecord
from prepare_ham_candidate_review import prepare
from check_ham_image_integrity import scan


class ReviewPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dermdetect-review-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def candidates(self):
        rows = [ManifestRecord(i, i, "nv", "invented", "v1", f"invented/{i}.jpg", sha256_bytes(i.encode()))
                for i in ("invented_a", "invented_b")]
        manifest = self.root / "manifest.csv"
        manifest.write_bytes(manifest_csv_bytes(rows))
        content = b"similarity,from_img,to_img\n0.99,invented_a.jpg,invented_b.jpg\n"
        (self.root / "invented_candidates.csv").write_bytes(content)
        (self.root / "source.json").write_text(json.dumps({"commit": "invented-commit"}))
        sources = {"invented_candidates.csv": ("reported_missed_duplicate", 1, sha256_bytes(content))}
        return manifest, sources

    def test_preparation_never_inherits_published_identity_conclusion(self):
        manifest, sources = self.candidates()
        with patch("prepare_ham_candidate_review.SOURCES", sources), patch("prepare_ham_candidate_review.COMMIT", "invented-commit"):
            report = prepare(manifest, self.root)
        self.assertTrue(report["identifier_correspondence_checked"])
        self.assertFalse(report["local_visual_identity_confirmed"])
        self.assertFalse(report["production_freeze"])
        self.assertEqual(report["candidates"][0]["decision"], "pending")

    def test_pinned_candidate_bytes_and_missing_correspondence_fail(self):
        manifest, sources = self.candidates()
        with patch("prepare_ham_candidate_review.SOURCES", sources), patch("prepare_ham_candidate_review.COMMIT", "invented-commit"):
            (self.root / "invented_candidates.csv").write_bytes(b"changed")
            with self.assertRaises(ContractError) as caught:
                prepare(manifest, self.root)
            self.assertEqual(caught.exception.code, "candidate_source_mismatch")
        manifest, sources = self.candidates()
        from dermdetect.data.manifests import read_manifest
        manifest.write_bytes(manifest_csv_bytes(read_manifest(manifest)[:1]))
        with patch("prepare_ham_candidate_review.SOURCES", sources), patch("prepare_ham_candidate_review.COMMIT", "invented-commit"):
            with self.assertRaises(ContractError) as caught:
                prepare(manifest, self.root)
            self.assertEqual(caught.exception.code, "candidate_correspondence_failure")

    def test_missing_and_wrong_decoder_fail_without_loading_images(self):
        import importlib.metadata
        for version, code in ((None, "decoder_unavailable"), ("0.0", "decoder_version_mismatch")):
            options = {"side_effect": importlib.metadata.PackageNotFoundError()} if version is None else {"return_value": version}
            with patch("dermdetect.data.image_integrity.importlib.metadata.version", **options):
                with self.assertRaises(ContractError) as caught:
                    decode_fingerprint(b"INVENTED TEXT; NOT AN IMAGE")
                self.assertEqual(caught.exception.code, code)

    def test_boundary_rejects_protected_root_in_read_allowlist(self):
        for name in ("bundle", "out", "protected", "runtime"):
            (self.root / name).mkdir()
        exe = self.root / "runtime" / "invented_python"
        exe.write_text("INVENTED; NOT AN EXECUTABLE")
        arguments = dict(bundle=self.root / "bundle", output=self.root / "out", interpreter=exe,
                         runtime_roots=[self.root / "runtime"], protected_roots=[self.root / "protected"])
        profile = development_profile(**arguments)
        self.assertIn("deny network*", profile)
        self.assertIn("deny file-read-data", profile)
        self.assertIn("deny process-exec", profile)
        for unsafe in (self.root, self.root / "protected", Path("/")):
            with self.assertRaises(ContractError):
                development_profile(**{**arguments, "runtime_roots": [unsafe]})

    def test_writable_boundary_cannot_overlap_frozen_bundle_or_runtime(self):
        for name in ("bundle", "out", "protected", "runtime"):
            (self.root / name).mkdir()
        exe = self.root / "runtime" / "invented_python"
        exe.write_text("INVENTED; NOT AN EXECUTABLE")
        arguments = dict(bundle=self.root / "bundle", output=self.root / "out", interpreter=exe,
                         runtime_roots=[self.root / "runtime"], protected_roots=[self.root / "protected"])
        for unsafe in (self.root, self.root / "bundle", self.root / "runtime"):
            with self.subTest(output=unsafe), self.assertRaises(ContractError):
                development_profile(**{**arguments, "output": unsafe})
        nested = self.root / "bundle" / "nested-output"
        nested.mkdir()
        with self.assertRaises(ContractError):
            development_profile(**{**arguments, "output": nested})

    def scan_fixture(self):
        import zipfile
        source = self.root / "public_source.json"
        source.write_text("INVENTED SOURCE EVIDENCE")
        archive = self.root / "invented.zip"
        payloads = {f"invented_{i}": f"INVENTED TEXT {i}; NOT AN IMAGE".encode() for i in range(2)}
        with zipfile.ZipFile(archive, "w") as handle:
            for key, value in payloads.items():
                handle.writestr(key + ".jpg", value)
        config = self.root / "config.json"
        config.write_text(json.dumps({"source_review_path": str(source),
            "source_review_sha256": sha256_bytes(source.read_bytes()),
            "image_archives": {"invented": str(archive)}}))
        inventory = self.root / "inventory"
        inventory.mkdir()
        rows = [ManifestRecord(key, key, "nv", "invented", "v1", f"invented/{key}.jpg", sha256_bytes(value))
                for key, value in payloads.items()]
        (inventory / "manifest.csv").write_bytes(manifest_csv_bytes(rows))
        (inventory / "integrity_report.json").write_text(json.dumps({"local_config_sha256": sha256_bytes(config.read_bytes()),
            "archives": [{"alias": "invented", "sha256": sha256_bytes(archive.read_bytes())}]}))
        queue = self.root / "queue.json"
        queue.write_text(json.dumps({"manifest_sha256": sha256_bytes((inventory / "manifest.csv").read_bytes()),
            "candidates": [{"left": "invented_0", "right": "invented_1", "kind": "invented_published_candidate"}]}))
        return config, inventory, queue, self.root / "scan_output", archive

    def test_scan_orchestration_preserves_known_candidate_even_when_hash_screen_misses(self):
        config, inventory, queue, output, _ = self.scan_fixture()
        # This verifies orchestration only: text bytes and a fake fingerprint
        # never stand in for a successful Pillow/JPEG validation.
        values = [{"width": 1, "height": 1, "rgb_sha256": str(i), "dhash64": i * (2**64 - 1)} for i in range(2)]
        with patch("check_ham_image_integrity.decode_fingerprint", side_effect=values):
            report = scan(config, inventory, queue, output)
        self.assertEqual(report["records_decoded"], 2)
        self.assertFalse(report["production_freeze"])
        self.assertEqual(report["candidates"][0]["reasons"], ["invented_published_candidate"])
        self.assertEqual(report["candidates"][0]["decision"], "pending")
        self.assertEqual(set(p.name for p in output.iterdir()), {"image_integrity.json", "decode_checkpoint.json"})

    def test_changed_archive_fails_before_decoder_or_output(self):
        config, inventory, queue, output, archive = self.scan_fixture()
        archive.write_bytes(b"CHANGED INVENTED ARCHIVE")
        with patch("check_ham_image_integrity.decode_fingerprint") as decoder:
            with self.assertRaises(ContractError) as caught:
                scan(config, inventory, queue, output)
            decoder.assert_not_called()
        self.assertEqual(caught.exception.code, "archive_checksum_mismatch")
        self.assertFalse(output.exists())

    def test_decode_failure_stops_without_exclusion_or_success_artifact(self):
        config, inventory, queue, output, _ = self.scan_fixture()
        with patch("check_ham_image_integrity.decode_fingerprint", side_effect=ContractError("unreadable_image", "invented failure")):
            with self.assertRaises(ContractError) as caught:
                scan(config, inventory, queue, output)
        self.assertEqual(caught.exception.code, "unreadable_image")
        self.assertEqual(caught.exception.details["image_id"], "invented_0")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
