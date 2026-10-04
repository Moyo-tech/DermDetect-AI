"""Real pinned-decoder checks on invented JPEGs, never dataset images."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dermdetect.data.image_integrity import decode_fingerprint
from dermdetect.data.manifests import manifest_csv_bytes, sha256_bytes
from dermdetect.data.schemas import ContractError, ManifestRecord
from check_ham_image_integrity import scan

try:
    from PIL import Image, ImageFile
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Run with the approved isolated Pillow environment.")
class PinnedDecoderTests(unittest.TestCase):
    def jpeg(self, inverse=False):
        with Image.new("RGB", (19, 17)) as image:
            image.putdata([(255 - x * 12 if inverse else x * 12,) * 3
                           for y in range(17) for x in range(19)])
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=95)
            return stream.getvalue()

    def fixture(self, root, payloads):
        source = root / "invented_source.json"
        source.write_text("INVENTED SOURCE EVIDENCE")
        archive = root / "invented.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for key, payload in payloads.items():
                handle.writestr(key + ".jpg", payload)
        config = root / "config.json"
        config.write_text(json.dumps({"source_review_path": str(source),
            "source_review_sha256": sha256_bytes(source.read_bytes()),
            "image_archives": {"invented": str(archive)}}))
        inventory = root / "inventory"
        inventory.mkdir()
        rows = [ManifestRecord(key, key, "nv", "invented", "v1", f"invented/{key}.jpg",
                               sha256_bytes(payload)) for key, payload in payloads.items()]
        (inventory / "manifest.csv").write_bytes(manifest_csv_bytes(rows))
        (inventory / "integrity_report.json").write_text(json.dumps({
            "local_config_sha256": sha256_bytes(config.read_bytes()),
            "archives": [{"alias": "invented", "sha256": sha256_bytes(archive.read_bytes())}]}))
        queue = root / "queue.json"
        queue.write_text(json.dumps({"manifest_sha256": sha256_bytes((inventory / "manifest.csv").read_bytes()),
            "candidates": [{"left": "invented_a", "right": "invented_b", "kind": "invented_known_pair"}]}))
        return config, inventory, queue

    def test_gradient_fingerprints_match_known_direction_and_pixel_hash(self):
        payload = self.jpeg()
        result = decode_fingerprint(payload)
        self.assertEqual((result["width"], result["height"], result["dhash64"]), (19, 17, 0))
        self.assertEqual(decode_fingerprint(self.jpeg(True))["dhash64"], 2**64 - 1)
        with Image.open(io.BytesIO(payload)) as image:
            self.assertEqual(result["rgb_sha256"], hashlib.sha256(b"RGB:19:17:" + image.tobytes()).hexdigest())

    def test_truncation_tolerance_and_pixel_bound_stop(self):
        with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True):
            with self.assertRaises(ContractError) as caught:
                decode_fingerprint(self.jpeg())
            self.assertEqual(caught.exception.code, "unsafe_decoder_settings")
        with patch("dermdetect.data.image_integrity.MAX_PIXELS", 10):
            with self.assertRaises(ContractError) as caught:
                decode_fingerprint(self.jpeg())
            self.assertEqual(caught.exception.code, "unsupported_image")

    def test_full_archive_scan_keeps_known_pair_even_when_hash_distance_is_64(self):
        with tempfile.TemporaryDirectory(prefix="dermdetect-jpeg-test-") as temporary:
            root = Path(temporary)
            config, inventory, queue = self.fixture(root, {"invented_a": self.jpeg(), "invented_b": self.jpeg(True)})
            output = root / "scan"
            result = scan(config, inventory, queue, output)
            self.assertEqual(result["records_decoded"], 2)
            self.assertFalse(result["production_freeze"])
            self.assertEqual(result["candidates"][0]["reasons"], ["invented_known_pair"])
            self.assertEqual(json.loads((output / "image_integrity.json").read_text()), result)
            self.assertEqual(set(p.name for p in output.iterdir()), {"image_integrity.json", "decode_checkpoint.json"})

    def test_candidate_failure_preserves_decode_checkpoint_without_truncating(self):
        with tempfile.TemporaryDirectory(prefix="dermdetect-jpeg-test-") as temporary:
            root = Path(temporary)
            config, inventory, queue = self.fixture(root, {"invented_a": self.jpeg(), "invented_b": self.jpeg()})
            output = root / "scan"
            with patch("dermdetect.data.image_integrity.MAX_CANDIDATES", 0):
                with self.assertRaises(ContractError) as caught:
                    scan(config, inventory, queue, output)
            self.assertEqual(caught.exception.code, "candidate_limit_exceeded")
            self.assertFalse((output / "image_integrity.json").exists())
            checkpoint = json.loads((output / "decode_checkpoint.json").read_text())
            self.assertEqual(checkpoint["records_decoded"], 2)
            failure = json.loads((output / "candidate_failure.json").read_text())
            self.assertEqual(failure["status"], "candidate_screen_failed")
            self.assertFalse(failure["production_freeze"])

    def test_hash_bound_checkpoint_reuse_avoids_redecode_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="dermdetect-jpeg-test-") as temporary:
            root = Path(temporary)
            config, inventory, queue = self.fixture(root, {"invented_a": self.jpeg(), "invented_b": self.jpeg(True)})
            first = scan(config, inventory, queue, root / "first")
            checkpoint = root / "first" / "decode_checkpoint.json"
            digest = sha256_bytes(checkpoint.read_bytes())
            with patch("check_ham_image_integrity.decode_fingerprint") as decoder:
                reused = scan(config, inventory, queue, root / "reused", checkpoint, digest)
                decoder.assert_not_called()
            self.assertEqual(first, reused)
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
            with self.assertRaises(ContractError) as caught:
                scan(config, inventory, queue, root / "changed", checkpoint, digest)
            self.assertEqual(caught.exception.code, "decode_checkpoint_mismatch")

    def test_real_decode_failure_preserves_source_and_writes_no_success_output(self):
        with tempfile.TemporaryDirectory(prefix="dermdetect-jpeg-test-") as temporary:
            root = Path(temporary)
            config, inventory, queue = self.fixture(root, {"invented_a": self.jpeg(), "invented_b": b"INVENTED NOT A JPEG"})
            before = (root / "invented.zip").read_bytes()
            output = root / "scan"
            with self.assertRaises(ContractError) as caught:
                scan(config, inventory, queue, output)
            self.assertEqual(caught.exception.code, "unreadable_image")
            self.assertEqual(caught.exception.details["image_id"], "invented_b")
            self.assertFalse(output.exists())
            self.assertEqual((root / "invented.zip").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
