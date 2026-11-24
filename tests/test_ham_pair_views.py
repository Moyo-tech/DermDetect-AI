"""Invented JPEGs verify unscaled evidence, cache reuse and fail-closed batches."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, manifest_csv_bytes, sha256_bytes
from dermdetect.data.schemas import ContractError, ManifestRecord
import prepare_ham_pair_views as views


class PairViewTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        temporary = tempfile.TemporaryDirectory(prefix="dermdetect-invented-views-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.payloads, self.rows = {}, []
        for key, colour in zip(("invented_a", "invented_b", "invented_c"), ("red", "green", "blue")):
            stream = io.BytesIO()
            with Image.new("RGB", (600, 450), colour) as image:
                image.save(stream, format="JPEG")
            self.payloads[key] = stream.getvalue()
            self.rows.append(ManifestRecord(key, key, "nv", "invented", "v1", "part1/" + key + ".jpg", sha256_bytes(stream.getvalue())))
        archive = self.root / "invented.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for key, payload in self.payloads.items():
                handle.writestr(key + ".jpg", payload)
        self.config, self.manifest, self.queue = (self.root / name for name in ("config.json", "manifest.csv", "queue.json"))
        self.config.write_bytes(canonical_json_bytes({"image_archives": {"part1": str(archive)}}))
        self.manifest.write_bytes(manifest_csv_bytes(self.rows))
        self.value = {"manifest_sha256": file_sha256(self.manifest), "candidates": [
            {"left": "invented_a", "right": "invented_b", "kind": "invented"},
            {"left": "invented_a", "right": "invented_c", "kind": "invented"}]}
        self.queue.write_bytes(canonical_json_bytes(self.value))
        self.output = self.root / "views"

    def run_views(self):
        return views.prepare(self.config, self.manifest, self.queue, self.output)

    def test_full_resolution_pixels_hashes_and_repeated_decode_cache(self):
        from PIL import Image
        with patch.object(views, "decode_fingerprint", wraps=views.decode_fingerprint) as decoder:
            report = self.run_views()
        self.assertEqual(decoder.call_count, 3)
        self.assertEqual(report["bounded_image_cache"], 32)
        self.assertFalse(report["production_freeze"])
        for pair in report["pairs"]:
            path = self.output / pair["view_file"]
            self.assertEqual(pair["view_sha256"], file_sha256(path))
            self.assertEqual(pair["source_payload_sha256"], [sha256_bytes(self.payloads[i]) for i in pair["image_ids"]])
            with Image.open(path) as sheet, Image.open(io.BytesIO(self.payloads[pair["image_ids"][0]])) as left:
                self.assertEqual(sheet.size, (1200, 526))
                self.assertEqual(sheet.crop((0, 76, 600, 526)).tobytes(), left.convert("RGB").tobytes())
        self.assertEqual(report["output_png_bytes"], sum((self.output / p["view_file"]).stat().st_size for p in report["pairs"]))

    def test_payload_tampering_never_produces_complete_index(self):
        archive = Path(json.loads(self.config.read_text())["image_archives"]["part1"])
        with zipfile.ZipFile(archive, "w") as handle:
            for key, payload in self.payloads.items():
                handle.writestr(key + ".jpg", payload + b"tampered")
        with self.assertRaises(ContractError) as caught:
            self.run_views()
        self.assertEqual(caught.exception.code, "image_checksum_mismatch")
        self.assertFalse((self.output / "index.json").exists())

    def test_duplicate_and_oversized_batches_fail_before_image_read(self):
        for candidates in ([self.value["candidates"][0]] * 2, [self.value["candidates"][0]] * 1001):
            self.queue.write_bytes(canonical_json_bytes({**self.value, "candidates": candidates}))
            with patch.object(views.zipfile, "ZipFile") as opener, self.assertRaises(ContractError):
                self.run_views()
            opener.assert_not_called()
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
