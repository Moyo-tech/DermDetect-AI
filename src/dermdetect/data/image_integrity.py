"""Pinned decoder proposal and bounded candidate screening. Real use gated.

Pillow is deliberately imported only at decode time; it is not installed by us.
Hashes screen for review and never establish lesion identity.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import io
import warnings
from typing import Mapping

from .schemas import ContractError

PILLOW_VERSION = "12.3.0"
MAX_PIXELS = 16_000_000
MAX_BYTES = 32 * 1024 * 1024
CANDIDATE_DISTANCE = 4
MAX_CANDIDATES = 20_000


def dhash_from_luminance(values: bytes) -> int:
    """64 row-major bits: left > right on a 9-by-8 luminance grid."""
    if not isinstance(values, bytes) or len(values) != 72:
        raise ContractError("invalid_hash_grid", "dHash requires exactly 72 luminance bytes.")
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | (values[row * 9 + col] > values[row * 9 + col + 1])
    return result


def decode_fingerprint(payload: bytes) -> dict:
    """Verify, reopen and fully decode a bounded JPEG; fail on every warning.

    No truncation tolerance or exclusion. EXIF orientation is applied for the
    integrity fingerprint only; this does not define either model's transform.
    """
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_BYTES:
        raise ContractError("invalid_image_payload", "JPEG payload exceeds the prespecified bound.")
    try:
        version = importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError("decoder_unavailable", "Pinned Pillow must be separately approved and installed.") from exc
    if version != PILLOW_VERSION:
        raise ContractError("decoder_version_mismatch", "Decoder must match the proposed pin.")
    from PIL import Image, ImageFile, ImageOps, features
    if ImageFile.LOAD_TRUNCATED_IMAGES or not features.check("jpg"):
        raise ContractError("unsafe_decoder_settings", "Strict JPEG decoding is required.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(io.BytesIO(payload), formats=["JPEG"]) as image:
                if image.width * image.height > MAX_PIXELS or image.mode not in {"RGB", "L"}:
                    raise ContractError("unsupported_image", "Image dimensions/mode require explicit review.")
                image.verify()
            with Image.open(io.BytesIO(payload), formats=["JPEG"]) as image:
                image.load()
                with ImageOps.exif_transpose(image) as oriented:
                    with oriented.convert("RGB") as rgb:
                        header = f"RGB:{rgb.width}:{rgb.height}:".encode()
                        pixels = hashlib.sha256(header + rgb.tobytes()).hexdigest()
                        with rgb.convert("L") as luminance:
                            with luminance.resize((9, 8), Image.Resampling.BILINEAR) as small:
                                value = dhash_from_luminance(small.tobytes())
                        return {"width": rgb.width, "height": rgb.height,
                                "rgb_sha256": pixels, "dhash64": value,
                                "pillow_version": version,
                                "jpeg_backend_version": features.version("jpg")}
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError("unreadable_image", "Full JPEG integrity decode failed; no row may be silently excluded.") from exc


def near_duplicate_pairs(hashes: Mapping[str, int]) -> tuple[tuple[str, str], ...]:
    """All <=4-bit pairs, using five exact bands, with a hard review-queue cap.

    At most four altered bits leave at least one band unchanged. Filtering on
    complete Hamming distance removes band collisions. This is not a guarantee
    that all visually related lesions have similar dHashes.
    """
    buckets = defaultdict(list)
    pairs = set()
    for image_id, value in sorted(hashes.items()):
        if not isinstance(image_id, str) or not image_id or type(value) is not int or not 0 <= value < 2**64:
            raise ContractError("invalid_fingerprint", "IDs and unsigned 64-bit fingerprints are required.")
        keys = [(band, (value >> offset) & ((1 << width) - 1))
                for band, (offset, width) in enumerate(((0, 13), (13, 13), (26, 13), (39, 13), (52, 12)))]
        previous = {other for key in keys for other in buckets[key]}
        for other in previous:
            if (value ^ hashes[other]).bit_count() <= CANDIDATE_DISTANCE:
                pairs.add((other, image_id))
                if len(pairs) > MAX_CANDIDATES:
                    raise ContractError("candidate_limit_exceeded", "Review queue too large; amend procedure, never truncate.")
        for key in keys:
            buckets[key].append(image_id)
    return tuple(sorted(pairs))
