"""M2 preparation tests: invented metadata and text files with JPG suffixes.

No medical images, real dataset paths or held-out outcomes are used.
"""
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from dermdetect.data.ham import ingest_ham, parse_ham_metadata, verify_ham_files
from dermdetect.data.manifests import manifest_csv_bytes, read_manifest, sha256_bytes
from dermdetect.data.schemas import CLASS_ORDER, ContractError
from run_synthetic_ham_ingestion import run_demo


def csv_bytes(rows, header=("image_id", "lesion_id", "dx", "dx_type")):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class HamPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dermdetect-ham-preparation-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.roots = {name: self.base / name for name in ("part1", "part2")}
        for root in self.roots.values():
            root.mkdir()
        self.metadata = []
        for i, dx in enumerate(CLASS_ORDER):
            for j in range(2):
                image_id = f"invented_{dx}_{j}"
                self.metadata.append([image_id, f"invented_lesion_{dx}", dx, "invented"])
                root = self.roots[f"part{1 + (i + j) % 2}"]
                (root / f"{image_id}.jpg").write_text(
                    f"SYNTHETIC TEXT; NOT AN IMAGE: {image_id}\n", encoding="utf-8")
        self.path = self.base / "invented_metadata.csv"
        self.save()

    def save(self):
        self.path.write_bytes(csv_bytes(self.metadata))

    def ingest(self, **kwargs):
        options = dict(source_version="invented-v1", provenance_reference="invented-test-fixture",
                       permitted_use_reference="invented-text-only", data_kind="synthetic")
        options.update(kwargs)
        return ingest_ham(self.path, self.roots, **options)

    def assert_code(self, code, callback):
        with self.assertRaises(ContractError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_known_counts_source_hash_aliases_and_pending_review(self):
        result = self.ingest()
        self.assertEqual(len(result.records), 14)
        self.assertEqual(len({row.lesion_id for row in result.records}), 7)
        self.assertEqual(result.metadata_sha256, sha256_bytes(self.path.read_bytes()))
        self.assertEqual({row.path.split("/")[0] for row in result.records}, set(self.roots))
        report = result.integrity_report()
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertFalse(report["split_ready"])
        self.assertEqual(report["exact_duplicate_candidates"], [])
        self.assertEqual(report["missing_classes"], [])
        self.assertEqual(report["image_decoding"], "not_checked")
        verify_ham_files(result.records, self.roots)

    def test_reordered_csv_preserves_manifest_but_changes_source_hash(self):
        original = self.ingest()
        self.metadata.reverse()
        self.save()
        reordered = self.ingest()
        self.assertEqual(manifest_csv_bytes(original.records), manifest_csv_bytes(reordered.records))
        self.assertNotEqual(original.metadata_sha256, reordered.metadata_sha256)

    def test_manifest_reload_and_checksum_mutation(self):
        result = self.ingest()
        manifest = self.base / "manifest.csv"
        manifest.write_bytes(manifest_csv_bytes(result.records))
        reloaded = read_manifest(manifest)
        self.assertEqual(reloaded, result.records)
        verify_ham_files(reloaded, self.roots)
        row = reloaded[0]
        alias, filename = row.path.split("/")
        (self.roots[alias] / filename).write_text("SYNTHETIC altered bytes\n")
        self.assert_code("checksum_mismatch", lambda: verify_ham_files(reloaded, self.roots))

    def test_duplicate_candidates_retained_without_splitting_or_exclusion(self):
        original = self.ingest().records
        def payload(row):
            alias, filename = row.path.split("/")
            return self.roots[alias] / filename
        payload(original[1]).write_bytes(payload(original[0]).read_bytes())
        result = self.ingest()
        report = result.integrity_report()
        self.assertEqual(len(result.records), 14)
        self.assertEqual(report["exact_duplicate_candidates"][0]["image_ids"],
                         [original[0].image_id, original[1].image_id])
        self.assertFalse(report["split_ready"])
        self.assertEqual(report["split"], "not_created")

    def test_all_metadata_validated_before_payload_hashing(self):
        cases = [("duplicate_image_id", lambda: self.metadata.append(self.metadata[0])),
                 ("lesion_label_conflict", lambda: self.metadata[1].__setitem__(2, "nv")),
                 ("missing_field", lambda: self.metadata[-1].__setitem__(1, "")),
                 ("unknown_class", lambda: self.metadata[-1].__setitem__(2, "unknown"))]
        saved = [row[:] for row in self.metadata]
        for code, mutate in cases:
            with self.subTest(code=code):
                self.metadata = [row[:] for row in saved]
                mutate()
                self.save()
                with patch("dermdetect.data.ham.file_sha256") as hasher:
                    self.assert_code(code, self.ingest)
                    hasher.assert_not_called()

    def test_unsafe_image_ids(self):
        for value in ("../escape", "/absolute", "bad/id", "bad\\id", "foo.jpg", ".."):
            with self.subTest(value=value):
                self.metadata[0][0] = value
                self.save()
                self.assert_code("unsafe_image_id", self.ingest)

    def test_bad_headers_rows_encoding_and_empty_manifest(self):
        cases = [
            (b"image_id,dx\nx,nv\n", "invalid_ham_header"),
            (b"image_id,lesion_id,dx,dx\nx,l,nv,nv\n", "invalid_ham_header"),
            (b"image_id,lesion_id,dx\nx,l,nv,extra\n", "invalid_csv_row"),
            (b"image_id,lesion_id,dx\nx,l\n", "invalid_csv_row"),
            (b"image_id,lesion_id,dx\n", "empty_manifest"),
            (b"\xff", "invalid_ham_csv"),
            (b'image_id,lesion_id,dx\n"unterminated', "invalid_ham_csv"),
        ]
        for content, code in cases:
            with self.subTest(code=code, content=content):
                self.assert_code(code, lambda: parse_ham_metadata(content, "invented-v1"))

    def test_missing_and_ambiguous_paths_fail_before_payload_hashing(self):
        row = self.ingest().records[0]
        alias, filename = row.path.split("/")
        path = self.roots[alias] / filename
        original = path.read_bytes()
        path.unlink()
        with patch("dermdetect.data.ham.file_sha256") as hasher:
            self.assert_code("missing_image", self.ingest)
            hasher.assert_not_called()
        for root in self.roots.values():
            (root / filename).write_bytes(original)
        with patch("dermdetect.data.ham.file_sha256") as hasher:
            self.assert_code("ambiguous_image", self.ingest)
            hasher.assert_not_called()

    def test_symlink_escape_and_nonfile_payload(self):
        row = self.ingest().records[0]
        alias, filename = row.path.split("/")
        path = self.roots[alias] / filename
        path.unlink()
        outside = self.base / "invented_outside.txt"
        outside.write_text("SYNTHETIC outside allowed root\n")
        path.symlink_to(outside)
        self.assert_code("unsafe_path", self.ingest)
        path.unlink()
        path.mkdir()
        self.assert_code("unreadable_file", self.ingest)

    def test_explicit_source_evidence_and_roots_required(self):
        for key, value in (("source_version", ""), ("provenance_reference", ""),
                           ("permitted_use_reference", " "), ("data_kind", "unknown")):
            with self.subTest(key=key):
                self.assert_code("invalid_source_record", lambda: self.ingest(**{key: value}))
        for roots in ({}, {"../unsafe": self.base}, {"a": self.base, "b": self.roots["part1"]},
                      {"a": self.roots["part1"], "b": self.roots["part1"]},
                      {"a": self.base / "missing"}):
            with self.subTest(roots=roots):
                self.roots_saved = self.roots
                self.roots = roots
                self.assert_code("invalid_image_roots", self.ingest)
                self.roots = self.roots_saved

    def test_reloaded_alias_path_must_match_id(self):
        rows = self.ingest().records
        for reference in ("part1/../escape.jpg", "unknown/foo.jpg", "part1/wrong.jpg"):
            altered = (replace(rows[0], path=reference),) + rows[1:]
            self.assert_code("unsafe_path", lambda: verify_ham_files(altered, self.roots))

    def test_unlisted_files_are_not_opened_or_included(self):
        (self.roots["part1"] / "unlisted.jpg").symlink_to(self.base / "missing-outside")
        self.assertEqual(len(self.ingest().records), 14)

    def test_demo_repeats_byte_identically_and_creates_no_split(self):
        first, second = self.base / "demo1", self.base / "demo2"
        run_demo(first)
        run_demo(second)
        files = {p.relative_to(first): p.read_bytes() for p in first.rglob("*") if p.is_file()}
        self.assertEqual(files, {p.relative_to(second): p.read_bytes()
                                 for p in second.rglob("*") if p.is_file()})
        self.assertEqual(len(files), 17)  # 14 payloads, metadata, manifest and report.
        self.assertFalse(any((first / f"{name}.csv").exists()
                             for name in ("train", "validation", "internal_test")))
        report = json.loads(files[Path("integrity_report.json")])
        self.assertEqual(report["metadata_sha256"], sha256_bytes(files[Path("invented_metadata.csv")]))
        self.assertEqual(report["manifest_sha256"], sha256_bytes(files[Path("manifest.csv")]))

    def test_demo_cli_refuses_existing_outputs_and_real_input_switches(self):
        output = self.base / "demo"
        run_demo(output)
        original = (output / "manifest.csv").read_bytes()
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_synthetic_ham_ingestion.py"
        result = subprocess.run([sys.executable, "-B", str(script), "--output-dir", str(output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["error"]["code"], "output_exists")
        self.assertEqual(original, (output / "manifest.csv").read_bytes())
        result = subprocess.run([sys.executable, "-B", str(script), "--metadata", str(self.path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
