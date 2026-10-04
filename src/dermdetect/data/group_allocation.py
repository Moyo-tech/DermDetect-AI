"""Production-policy proposal, exercised on synthetic data only; no file I/O."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from typing import Iterable, Mapping

from .analysis import AnalysisRecord, validate_analysis
from .schemas import CLASS_ORDER, ContractError

ALGORITHM = "class-group-largest-remainder-sha256-v1"
SPLITS = ("train", "validation", "internal_test")
PERCENTAGES = (70, 15, 15)
SEED = 20260729


def group_quotas(count: int) -> tuple[int, ...]:
    if type(count) is not int or count <= 0:
        raise ContractError("invalid_group_count", "Group count must be a positive integer.")
    quotas = [count * p // 100 for p in PERCENTAGES]
    order = sorted(range(3), key=lambda i: (-(count * PERCENTAGES[i] % 100), i))
    for i in order[:count - sum(quotas)]:
        quotas[i] += 1
    if min(quotas) < 1:
        raise ContractError("missing_partition_class", "Rounded quotas must contain every class in every split.")
    return tuple(quotas)


def allocate_groups(records: Iterable[AnalysisRecord]) -> dict[str, tuple[AnalysisRecord, ...]]:
    rows = validate_analysis(records)
    groups: dict[str, list[AnalysisRecord]] = defaultdict(list)
    by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row.split_group_id].append(row)
    for identity, members in groups.items():
        by_class[members[0].original.dx].append(identity)
    if set(by_class) != set(CLASS_ORDER):
        raise ContractError("missing_class", "Allocation requires all seven classes.")
    assigned: dict[str, list[AnalysisRecord]] = {name: [] for name in SPLITS}
    for dx in CLASS_ORDER:
        quotas = group_quotas(len(by_class[dx]))
        def key(identity: str) -> tuple[str, str]:
            raw = json.dumps([ALGORITHM, SEED, dx, identity], separators=(",", ":"))
            return hashlib.sha256(raw.encode()).hexdigest(), identity
        ordered = sorted(by_class[dx], key=key)
        offset = 0
        for name, quota in zip(SPLITS, quotas):
            for identity in ordered[offset:offset + quota]:
                assigned[name].extend(groups[identity])
            offset += quota
    result = {name: tuple(sorted(members, key=lambda row: row.original.image_id))
              for name, members in assigned.items()}
    verify_allocation(rows, result)
    return result


def verify_allocation(records: Iterable[AnalysisRecord],
                      partitions: Mapping[str, Iterable[AnalysisRecord]]) -> dict:
    rows = validate_analysis(records)
    if set(partitions) != set(SPLITS):
        raise ContractError("invalid_partitions", "Exactly the three prescribed partitions are required.")
    assigned = {name: tuple(partitions[name]) for name in SPLITS}
    combined = tuple(sorted((row for members in assigned.values() for row in members),
                            key=lambda row: row.original.image_id))
    if combined != rows:
        raise ContractError("membership_mismatch", "Every original row must be assigned once, unmodified.")
    owner = {}
    counts = {}
    for name, members in assigned.items():
        for row in members:
            for field, value in (("image", row.original.image_id), ("lesion", row.original.lesion_id),
                                 ("hash", row.original.sha256), ("analysis", row.analysis_lesion_id),
                                 ("group", row.split_group_id)):
                if (field, value) in owner and owner[field, value] != name:
                    raise ContractError("cross_split_overlap", "An identity/view crosses partitions.")
                owner[field, value] = name
        counts[name] = {dx: {"groups": len({r.split_group_id for r in members if r.original.dx == dx}),
                            "original_records": sum(r.original.dx == dx for r in members),
                            "active_views": sum(r.active and r.original.dx == dx for r in members)}
                        for dx in CLASS_ORDER}
    for dx in CLASS_ORDER:
        count = len({row.split_group_id for row in rows if row.original.dx == dx})
        quotas = group_quotas(count)
        if tuple(counts[name][dx]["groups"] for name in SPLITS) != quotas:
            raise ContractError("quota_mismatch", "Allocation must use the fixed class-wise quotas.")
    return {"algorithm": ALGORITHM, "seed": SEED, "percentages": PERCENTAGES,
            "zero_overlap": True, "counts": counts,
            "interpretation": "descriptive_comparison_inconclusive_allowed",
            "rare_class_adequacy_claim": False,
            "production_freeze": False}
