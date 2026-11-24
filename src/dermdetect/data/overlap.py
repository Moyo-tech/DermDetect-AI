"""Exact duplication and cross-partition membership audits; no perceptual claims."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping

from .schemas import ContractError, ManifestRecord, validate_manifest


def exact_duplicates(records: Iterable[ManifestRecord]) -> list[dict]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in validate_manifest(records):
        by_hash[row.sha256].append(row.image_id)
    return [{"sha256": digest, "image_ids": sorted(ids)}
            for digest, ids in sorted(by_hash.items()) if len(ids) > 1]


def require_unique_content(records: Iterable[ManifestRecord]) -> None:
    duplicates = exact_duplicates(records)
    if duplicates:
        raise ContractError("duplicate_content", "Exact duplicate payloads require explicit resolution.",
                            duplicates=duplicates)


def overlap_report(partitions: Mapping[str, Iterable[ManifestRecord]]) -> dict:
    """Detect every cross-partition overlap without hiding duplicate ID conflicts."""
    rows = {name: tuple(records) for name, records in partitions.items()}
    pairs = []
    for left, right in combinations(sorted(rows), 2):
        values = {}
        for field in ("image_id", "lesion_id", "sha256"):
            values[field] = sorted({getattr(row, field) for row in rows[left]}
                                   & {getattr(row, field) for row in rows[right]})
        pairs.append({"left": left, "right": right, **values})
    return {"zero_overlap": not any(pair[field] for pair in pairs
                                    for field in ("image_id", "lesion_id", "sha256")),
            "pairs": pairs}


def require_zero_overlap(partitions: Mapping[str, Iterable[ManifestRecord]]) -> dict:
    report = overlap_report(partitions)
    if not report["zero_overlap"]:
        raise ContractError("cross_split_overlap", "Partition IDs, lesions or hashes overlap.",
                            report=report)
    return report
