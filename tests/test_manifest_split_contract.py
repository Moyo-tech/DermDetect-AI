"""M1 invariants tested against invented metadata and text payloads only."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dermdetect.data.manifests import (build_manifest, canonical_json_bytes, manifest_csv_bytes,
                                      read_manifest, rooted_file, sha256_bytes, verify_files)
from dermdetect.data.overlap import exact_duplicates, overlap_report, require_unique_content
from dermdetect.data.schemas import CLASS_ORDER, ContractError, MANIFEST_FIELDS, ManifestRecord, validate_manifest
from dermdetect.data.splits import (SPLIT_ORDER, SplitConfig, lesion_quotas, partition_counts,
                                   split_manifest, verify_partition_membership)
from run_synthetic_contract import load_fixture, materialize_fixture, run_demo


class SyntheticContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dermdetect-contract-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = load_fixture()
        self.metadata = materialize_fixture(self.root, self.spec)
        self.rows = build_manifest(self.metadata, self.root)
        self.config = SplitConfig.from_mapping(self.spec["split_config"])

    def assert_code(self, code, callback):
        with self.assertRaises(ContractError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        return caught.exception

    def test_known_fixture_counts_and_taxonomy(self):
        self.assertEqual(CLASS_ORDER, ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))
        self.assertEqual(len(self.rows), 63)
        self.assertEqual(len({row.lesion_id for row in self.rows}), 35)
        for dx in CLASS_ORDER:
            self.assertEqual(sum(row.dx == dx for row in self.rows), 9)
            self.assertEqual(len({row.lesion_id for row in self.rows if row.dx == dx}), 5)
        self.assertEqual({row.source for row in self.rows}, {"invented-fixture"})
        payload = (self.root / self.rows[0].path).read_bytes()
        self.assertIn(b"SYNTHETIC TEXT; NOT AN IMAGE", payload)
        self.assertEqual(self.rows[0].sha256, sha256_bytes(payload))

    def test_all_required_fields_reject_missing_or_blank(self):
        for field in MANIFEST_FIELDS:
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    row = self.rows[0].to_dict()
                    if missing:
                        row.pop(field)
                    else:
                        row[field] = " "
                    self.assert_code("missing_field", lambda: ManifestRecord.from_mapping(row))

    def test_unknown_class_and_invalid_checksum(self):
        for field, value, code in [("dx", "unknown", "unknown_class"),
                                   ("sha256", "a" * 63, "invalid_checksum"),
                                   ("sha256", "G" * 64, "invalid_checksum")]:
            with self.subTest(field=field, value=value):
                row = {**self.rows[0].to_dict(), field: value}
                self.assert_code(code, lambda: ManifestRecord.from_mapping(row))

    def test_surrounding_whitespace_is_not_silently_normalized(self):
        row = {**self.rows[0].to_dict(), "lesion_id": " " + self.rows[0].lesion_id}
        self.assert_code("noncanonical_field", lambda: ManifestRecord.from_mapping(row))

    def test_duplicate_id_rejected_without_dropping_rows(self):
        self.assert_code("duplicate_image_id", lambda: validate_manifest(self.rows + (self.rows[0],)))

    def test_lesion_label_conflict_rejected(self):
        member = next(row for row in self.rows if row.lesion_id == "toy_akiec_L02")
        altered = tuple(replace(row, dx="bcc") if row.image_id == member.image_id else row for row in self.rows)
        self.assert_code("lesion_label_conflict", lambda: validate_manifest(altered))

    def test_empty_manifest_rejected(self):
        self.assert_code("empty_manifest", lambda: validate_manifest([]))

    def test_exact_content_duplicates_have_explicit_ids(self):
        altered = (self.rows[0], replace(self.rows[1], sha256=self.rows[0].sha256))
        report = exact_duplicates(altered)
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]["image_ids"], sorted(row.image_id for row in altered))
        self.assert_code("duplicate_content", lambda: require_unique_content(altered))

    def test_duplicate_content_prevents_splitting(self):
        altered = (self.rows[0], replace(self.rows[1], sha256=self.rows[0].sha256)) + self.rows[2:]
        self.assert_code("duplicate_content", lambda: split_manifest(altered, self.config))

    def test_changed_payload_fails_checksum_verification(self):
        (self.root / self.rows[0].path).write_text("invented changed text\n", encoding="utf-8")
        self.assert_code("checksum_mismatch", lambda: verify_files(self.rows, self.root))

    def test_missing_file_and_traversal_rejected_before_read(self):
        self.assert_code("unreadable_file", lambda: rooted_file(self.root, "payloads/missing.txt"))
        for reference in ("../outside.txt", "/tmp/file", "payloads/../outside.txt", "payloads\\file", "./payloads/file"):
            with self.subTest(reference=reference):
                self.assert_code("unsafe_path", lambda: rooted_file(self.root, reference))

    def test_symlink_escape_rejected(self):
        outside = self.root / "invented-outside.txt"
        outside.write_text("invented text\n", encoding="utf-8")
        bounded = self.root / "bounded"
        bounded.mkdir()
        (bounded / "alias.txt").symlink_to(outside)
        self.assert_code("unsafe_path", lambda: rooted_file(bounded, "alias.txt"))

    def test_invalid_metadata_is_checked_before_payload_open(self):
        row = {**self.metadata[0], "dx": "unknown"}
        with patch("dermdetect.data.manifests.file_sha256") as reader:
            self.assert_code("unknown_class", lambda: build_manifest([row], self.root))
            reader.assert_not_called()

    def test_csv_roundtrip_is_canonical_under_permutation(self):
        shuffled = list(self.rows)
        random.Random(99).shuffle(shuffled)
        expected = manifest_csv_bytes(self.rows)
        self.assertEqual(expected, manifest_csv_bytes(shuffled))
        self.assertEqual(sha256_bytes(expected), sha256_bytes(manifest_csv_bytes(reversed(self.rows))))
        path = self.root / "manifest.csv"
        path.write_bytes(expected)
        self.assertEqual(read_manifest(path), self.rows)

    def test_csv_rejects_wrong_header_and_ragged_rows(self):
        path = self.root / "malformed.csv"
        path.write_text("image_id,dx\na,mel\n", encoding="utf-8")
        self.assert_code("invalid_header", lambda: read_manifest(path))
        path.write_text(",".join(MANIFEST_FIELDS) + "\na,lesion,mel,source,v,p," + "a" * 64 + ",extra\n", encoding="utf-8")
        self.assert_code("invalid_csv_row", lambda: read_manifest(path))

    def test_explicit_config_validation(self):
        for seed, proportions in [(True, (.6, .2, .2)), (1.5, (.6, .2, .2)),
                                   (17, (.6, .2, .3)), (17, (1, 0, 0)),
                                   (17, (float("nan"), .2, .2)), (17, (.5, .5)),
                                   (17, [.6, .2, .2]), (17, None)]:
            with self.subTest(seed=seed, proportions=proportions):
                self.assert_code("invalid_config", lambda: SplitConfig(seed, proportions))
        self.assert_code("invalid_config", lambda: SplitConfig.from_mapping({"seed": 17}))

    def test_config_roundtrip_preserves_and_checks_version(self):
        self.assertEqual(SplitConfig.from_mapping(self.config.to_dict()), self.config)
        for field in ("algorithm", "status"):
            altered = {**self.config.to_dict(), field: "unsupported"}
            self.assert_code("invalid_config", lambda: SplitConfig.from_mapping(altered))

    def test_quotas_have_hand_calculated_counts(self):
        self.assertEqual(lesion_quotas(5, (.6, .2, .2)), (3, 1, 1))
        self.assertEqual(lesion_quotas(10, (.6, .2, .2)), (6, 2, 2))
        self.assertEqual(lesion_quotas(3, (.6, .2, .2)), (1, 1, 1))
        self.assert_code("insufficient_groups", lambda: lesion_quotas(2, (.6, .2, .2)))

    def test_split_is_exhaustive_grouped_and_class_representative(self):
        partitions = split_manifest(self.rows, self.config)
        assigned_ids = [row.image_id for name in SPLIT_ORDER for row in partitions[name]]
        self.assertEqual(len(assigned_ids), 63)
        self.assertEqual(set(assigned_ids), {row.image_id for row in self.rows})
        owners = {}
        for name in SPLIT_ORDER:
            self.assertEqual({row.dx for row in partitions[name]}, set(CLASS_ORDER))
            for row in partitions[name]:
                owners.setdefault(row.lesion_id, set()).add(name)
        self.assertEqual(len(owners), 35)
        self.assertTrue(all(len(names) == 1 for names in owners.values()))
        self.assertTrue(verify_partition_membership(self.rows, partitions)["zero_overlap"])
        counts = partition_counts(partitions, self.config)
        for name, expected in (("train", 21), ("validation", 7), ("internal_test", 7)):
            self.assertEqual(counts[name]["lesions"], expected)
            for dx in CLASS_ORDER:
                self.assertEqual(counts[name]["by_class"][dx]["lesions"], 3 if name == "train" else 1)
        self.assertEqual(sum(counts[name]["images"] for name in SPLIT_ORDER), 63)

    def test_split_bytes_and_counts_ignore_input_order(self):
        expected = split_manifest(self.rows, self.config)
        for candidate in (list(reversed(self.rows)), random.Random(41).sample(list(self.rows), len(self.rows))):
            actual = split_manifest(candidate, self.config)
            self.assertEqual(canonical_json_bytes(partition_counts(actual, self.config)),
                             canonical_json_bytes(partition_counts(expected, self.config)))
            for name in SPLIT_ORDER:
                self.assertEqual(manifest_csv_bytes(actual[name]), manifest_csv_bytes(expected[name]))

    def test_seed_is_effective_and_repeatable(self):
        first = split_manifest(self.rows, self.config)
        self.assertEqual(first, split_manifest(self.rows, self.config))
        self.assertNotEqual(first, split_manifest(self.rows, SplitConfig(18, self.config.proportions)))

    def test_missing_class_and_too_few_lesions_fail(self):
        missing = tuple(row for row in self.rows if row.dx != "vasc")
        self.assert_code("missing_class", lambda: split_manifest(missing, self.config))
        scarce = tuple(row for row in self.rows if row.dx != "df" or row.lesion_id in {"toy_df_L01", "toy_df_L02"})
        self.assert_code("insufficient_groups", lambda: split_manifest(scarce, self.config))

    def test_overlap_dimensions_are_detected_independently(self):
        for field in ("image_id", "lesion_id", "sha256"):
            with self.subTest(field=field):
                left = self.rows[0]
                right = replace(self.rows[1], **{field: getattr(left, field)})
                report = overlap_report({"train": (left,), "validation": (right,)})
                self.assertFalse(report["zero_overlap"])
                self.assertEqual(report["pairs"][0][field], [getattr(left, field)])

    def test_membership_check_rejects_overlap_omission_and_alteration(self):
        partitions = split_manifest(self.rows, self.config)
        overlap = {**partitions, "validation": partitions["validation"] + (partitions["train"][0],)}
        self.assert_code("cross_split_overlap", lambda: verify_partition_membership(self.rows, overlap))
        omitted = {**partitions, "train": partitions["train"][1:]}
        self.assert_code("membership_mismatch", lambda: verify_partition_membership(self.rows, omitted))
        changed = {**partitions, "train": (replace(partitions["train"][0], source_version="changed"),) + partitions["train"][1:]}
        self.assert_code("membership_mismatch", lambda: verify_partition_membership(self.rows, changed))
        self.assert_code("invalid_partitions", lambda: verify_partition_membership(self.rows, {"train": self.rows}))

    def test_membership_verifier_rejects_duplicates_within_one_partition(self):
        partitions = split_manifest(self.rows, self.config)
        first, second = partitions["train"][:2]
        altered = replace(second, sha256=first.sha256)
        rows = tuple(altered if row.image_id == second.image_id else row for row in self.rows)
        assigned = {**partitions, "train": (first, altered) + partitions["train"][2:]}
        self.assert_code("duplicate_content", lambda: verify_partition_membership(rows, assigned))

    def test_persisted_demo_reloads_and_reruns_byte_identically(self):
        a, b = self.root / "out-a", self.root / "out-b"
        self.assertEqual(run_demo(a), run_demo(b))
        files = sorted(path.relative_to(a) for path in a.rglob("*") if path.is_file())
        self.assertEqual(files, sorted(path.relative_to(b) for path in b.rglob("*") if path.is_file()))
        for relative in files:
            self.assertEqual((a / relative).read_bytes(), (b / relative).read_bytes())
        verify_files(read_manifest(a / "manifest.csv"), a)
        self.assert_code("output_exists", lambda: run_demo(a))

    def test_all_cli_failures_are_structured_and_leave_no_success_artifacts(self):
        cases = {"duplicate-id": "duplicate_image_id", "missing-field": "missing_field",
                 "unknown-class": "unknown_class", "label-conflict": "lesion_label_conflict",
                 "duplicate-content": "duplicate_content", "checksum-mismatch": "checksum_mismatch",
                 "overlap": "cross_split_overlap"}
        for case, expected in cases.items():
            with self.subTest(case=case):
                output = self.root / ("failure-" + case)
                result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_synthetic_contract.py"),
                                         "--case", case, "--output-dir", str(output)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                report = json.loads(result.stderr)
                self.assertEqual(report["data_kind"], "synthetic")
                self.assertEqual(report["error"]["code"], expected)
                self.assertFalse(output.exists())

    def test_cli_accepts_no_real_data_input_switch(self):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_synthetic_contract.py"),
                                 "--manifest", "not-a-real-path.csv"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)


if __name__ == "__main__":
    unittest.main()
