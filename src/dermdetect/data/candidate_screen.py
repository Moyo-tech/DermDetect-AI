"""Exhaustive cached-fingerprint enumeration; no identity decisions or images.

This proposed storage amendment preserves dHash <=4. It changes how matches
are retained, not which matches qualify. Review and freeze are separate gates.
"""
from collections import defaultdict
from typing import Iterator, Mapping

from .schemas import ContractError

PROCEDURE = "dhash64-distance4-streamed-review-v2"
MAX_RECORDS = 12_000
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
BANDS = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))


def iter_hash_pairs(hashes: Mapping[str, int]) -> Iterator[tuple[str, str, int]]:
    """Yield every qualifying unordered pair exactly once, deterministically.

    Five disjoint bands ensure that <=4 changed bits leave a matching band.
    Only per-image previous IDs are materialized, rather than the whole queue.
    Validate the entire input before yielding, including bool/int distinctions.
    """
    if not hashes or len(hashes) > MAX_RECORDS:
        raise ContractError("screen_record_bound", "Screen requires 1..12,000 fingerprints.")
    for image_id, value in hashes.items():
        if (not isinstance(image_id, str) or not image_id or type(value) is not int
                or not 0 <= value < 2**64):
            raise ContractError("invalid_fingerprint", "IDs and unsigned 64-bit hashes are required.")
    buckets = defaultdict(list)
    for right in sorted(hashes):
        value = hashes[right]
        keys = [(band, (value >> offset) & ((1 << width) - 1))
                for band, (offset, width) in enumerate(BANDS)]
        previous = {left for key in keys for left in buckets[key]}
        for left in sorted(previous):
            distance = (value ^ hashes[left]).bit_count()
            if distance <= 4:
                yield left, right, distance
        for key in keys:
            buckets[key].append(right)
