"""Archive ingestion contracts on invented CSV and text bytes only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data.ham_archives import ingest_ham_archives
from dermdetect.data.manifests import manifest_csv_bytes, sha256_bytes
from dermdetect.data.schemas import CLASS_ORDER, ContractError
from build_ham_archive_inventory import run


class HamArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dermdetect-archive-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = self.root / "invented_metadata"
        self.rows = [[f"invented_{dx}_{i}", f"lesion_{dx}", dx]
                     for dx in CLASS_ORDER for i in (1, 2)]
        self.archives = {f"part{i}": self.root / f"part{i}.zip" for i in (1, 2)}
        self.members = {alias: [(f"{image}.jpg", f"SYNTHETIC TEXT; NOT AN IMAGE: {image}\n".encode())
                               for image, lesion, dx in self.rows if image.endswith(alias[-1])]
                        for alias in self.archives}
        self.save_metadata()
        self.save_archives()

    def save_metadata(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("image_id", "lesion_id", "dx"))
        writer.writerows(self.rows)
        self.metadata.write_text(stream.getvalue())

    def save_archives(self, compression=zipfile.ZIP_STORED):
        for alias, path in self.archives.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(path, "w", compression=compression) as archive:
                    for name, content in self.members[alias]:
                        archive.writestr(name, content)

    def ingest(self, **kwargs):
        options = dict(source_version="invented-v1", provenance_reference="invented-fixture",
                       permitted_use_reference="invented-text-only", data_kind="synthetic")
        options.update(kwargs)
        return ingest_ham_archives(self.metadata, self.archives, **options)

    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_complete_membership_hashes_crc_without_extraction(self):
        expected = {alias: hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
                    for alias, path in self.archives.items()}
        result = self.ingest(expected_archive_md5=expected)
        self.assertEqual(len(result.records), 14)
        self.assertEqual(len({r.lesion_id for r in result.records}), 7)
        for row in result.records:
            alias, name = row.path.split("/")
            self.assertEqual(row.sha256, sha256_bytes(dict(self.members[alias])[name]))
        report = result.integrity_report()
        self.assertEqual(report["data_kind"], "synthetic")
        self.assertFalse(report["split_ready"])
        self.assertTrue(all(item["publisher_md5_match"] for item in report["archives"]))
        self.assertEqual(report["archive_crc"], "passed_for_all_referenced_members")
        self.assertEqual(report["image_decoding"], "not_checked")
        self.assertEqual(set(p.name for p in self.root.iterdir()), {"invented_metadata", "part1.zip", "part2.zip"})

    def test_input_order_does_not_change_manifest(self):
        original = self.ingest()
        self.rows.reverse()
        self.save_metadata()
        for members in self.members.values():
            members.reverse()
        self.save_archives()
        changed = self.ingest()
        self.assertEqual(manifest_csv_bytes(original.records), manifest_csv_bytes(changed.records))
        self.assertNotEqual(original.metadata_sha256, changed.metadata_sha256)
        self.assertNotEqual(original.archive_inventory[0]["sha256"], changed.archive_inventory[0]["sha256"])

    def test_deflated_and_stored_payloads_have_identical_manifest(self):
        original = self.ingest()
        self.save_archives(compression=zipfile.ZIP_DEFLATED)
        self.assertEqual(manifest_csv_bytes(original.records), manifest_csv_bytes(self.ingest().records))

    def test_invalid_metadata_precedes_archive_open(self):
        self.rows.append(self.rows[0])
        self.save_metadata()
        with patch("dermdetect.data.ham_archives.zipfile.ZipFile") as opener:
            self.assert_code("duplicate_image_id", self.ingest)
            opener.assert_not_called()

    def test_membership_mismatch_precedes_member_payload_open(self):
        self.members["part1"].pop()
        self.save_archives()
        with patch("zipfile.ZipFile.open") as opener:
            self.assert_code("archive_membership_mismatch", self.ingest)
            opener.assert_not_called()

    def test_unlisted_and_cross_archive_duplicate_members_stop(self):
        self.members["part1"].append(("unlisted.jpg", b"SYNTHETIC unlisted"))
        self.save_archives()
        self.assert_code("archive_membership_mismatch", self.ingest)
        self.members["part1"].pop()
        self.members["part2"].append(self.members["part1"][0])
        self.save_archives()
        self.assert_code("ambiguous_image", self.ingest)

    def test_repeated_and_unsafe_members_stop(self):
        self.members["part1"].append(self.members["part1"][0])
        self.save_archives()
        self.assert_code("duplicate_archive_member", self.ingest)
        self.members["part1"].pop()
        for name in ("../escape.jpg", "/absolute.jpg", "nested/a.jpg", "not-a-jpg.txt"):
            with self.subTest(name=name):
                self.members["part1"].append((name, b"SYNTHETIC"))
                self.save_archives()
                self.assert_code("unsafe_archive_member", self.ingest)
                self.members["part1"].pop()
        self.save_archives()
        with zipfile.ZipFile(self.archives["part1"], "a") as archive:
            link = zipfile.ZipInfo("symlink.jpg")
            link.create_system = 3
            link.external_attr = 0o120777 << 16
            archive.writestr(link, "SYNTHETIC target")
        self.assert_code("unsafe_archive_member", self.ingest)

    def test_resource_and_source_checksum_limits(self):
        self.assert_code("archive_limit", lambda: self.ingest(max_member_bytes=1))
        self.assert_code("archive_limit", lambda: self.ingest(max_total_bytes=1))
        self.assert_code("invalid_archive_limits", lambda: self.ingest(max_total_bytes=True))
        expected = {alias: "0" * 32 for alias in self.archives}
        with patch("zipfile.ZipFile.open") as opener:
            self.assert_code("archive_source_mismatch", lambda: self.ingest(expected_archive_md5=expected))
            opener.assert_not_called()
        self.assert_code("invalid_source_record", lambda: self.ingest(expected_archive_md5={}))

    def test_crc_corruption_is_a_structured_failure(self):
        path = self.archives["part1"]
        content = bytearray(path.read_bytes())
        offset = content.index(b"SYNTHETIC TEXT")
        content[offset] ^= 1
        path.write_bytes(content)
        self.assert_code("unreadable_archive_input", self.ingest)

    def test_content_duplicates_retained_for_review(self):
        name, _ = self.members["part1"][1]
        self.members["part1"][1] = (name, self.members["part1"][0][1])
        self.save_archives()
        result = self.ingest()
        self.assertEqual(len(result.records), 14)
        self.assertEqual(len(result.integrity_report()["exact_duplicate_candidates"]), 1)
        self.assertFalse(result.integrity_report()["split_ready"])

    def test_inventory_runner_persists_only_derived_files_and_source_failure_writes_nothing(self):
        review = self.root / "invented_source_review.json"
        review.write_text('{"data_kind":"synthetic"}\n')
        config = dict(schema_version=1, metadata_path=str(self.metadata),
                      image_archives={alias: str(path) for alias, path in self.archives.items()},
                      source_version="invented-v1", provenance_reference="invented-fixture",
                      permitted_use_reference="invented-text-only", data_kind="synthetic",
                      expected_archive_md5={alias: hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
                                            for alias, path in self.archives.items()},
                      source_review_path=str(review), source_review_sha256=sha256_bytes(review.read_bytes()))
        config_path = self.root / "invented_config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "inventory"
        report = run(config_path, output)
        self.assertEqual(report["records"], 14)
        self.assertEqual(set(p.name for p in output.iterdir()), {"manifest.csv", "integrity_report.json"})
        self.assertEqual(report["manifest_sha256"], sha256_bytes((output / "manifest.csv").read_bytes()))
        self.assert_code("output_exists", lambda: run(config_path, output))
        review.write_text('{"data_kind":"synthetic","changed":true}\n')
        failed = self.root / "failed_inventory"
        self.assert_code("source_review_mismatch", lambda: run(config_path, failed))
        self.assertFalse(failed.exists())


if __name__ == "__main__":
    unittest.main()
