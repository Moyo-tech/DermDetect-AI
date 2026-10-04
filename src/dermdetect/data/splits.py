"""Toy class-stratified lesion allocator, not an approved production split policy."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

from .overlap import require_unique_content, require_zero_overlap
from .schemas import CLASS_ORDER, ContractError, ManifestRecord, validate_manifest

SPLIT_ORDER = ("train", "validation", "internal_test")
ALGORITHM = "toy-lesion-quota-sha256-v1"
CONFIG_STATUS = "synthetic_demonstration_only"


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    proportions: tuple[float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("invalid_config", "Seed must be an explicit integer.")
        if (not isinstance(self.proportions, tuple) or len(self.proportions) != 3
                or any(isinstance(p, bool) or not isinstance(p, (int, float))
                       or not math.isfinite(p) or p <= 0 for p in self.proportions)
                or not math.isclose(sum(self.proportions), 1.0, rel_tol=0, abs_tol=1e-12)):
            raise ContractError("invalid_config", "Three positive finite proportions must sum to one.")

    @classmethod
    def from_mapping(cls, raw: Mapping) -> SplitConfig:
        keys = set(raw)
        if keys == {"seed", "proportions", "algorithm", "status"}:
            if raw["algorithm"] != ALGORITHM or raw["status"] != CONFIG_STATUS:
                raise ContractError("invalid_config", "Unsupported algorithm or demonstration status.")
        elif keys != {"seed", "proportions"}:
            raise ContractError("invalid_config", "Config requires seed/proportions and optional version tags.")
        if not isinstance(raw["proportions"], dict) or set(raw["proportions"]) != set(SPLIT_ORDER):
            raise ContractError("invalid_config", "All three named proportions are required.")
        return cls(raw["seed"], tuple(raw["proportions"][name] for name in SPLIT_ORDER))

    def to_dict(self) -> dict:
        return {"seed": self.seed, "proportions": dict(zip(SPLIT_ORDER, self.proportions)),
                "algorithm": ALGORITHM, "status": CONFIG_STATUS}


def lesion_quotas(count: int, proportions: tuple[float, float, float]) -> tuple[int, ...]:
    """Integer quotas with minimum one/class/split; fixed-order tie breaking."""
    if count < 3:
        raise ContractError("insufficient_groups", "At least three lesions per class are required.",
                            lesion_count=count)
    quotas = [1, 1, 1]
    targets = [count * p for p in proportions]
    for _ in range(count - 3):
        # Each step minimizes the increment in squared lesion-count deviation.
        index = max(range(3), key=lambda i: (targets[i] - quotas[i], -i))
        quotas[index] += 1
    return tuple(quotas)


def split_manifest(records: Iterable[ManifestRecord], config: SplitConfig) -> dict[str, tuple[ManifestRecord, ...]]:
    rows = validate_manifest(records)
    require_unique_content(rows)
    groups: dict[str, list[ManifestRecord]] = defaultdict(list)
    for row in rows:
        groups[row.lesion_id].append(row)
    by_class: dict[str, list[str]] = defaultdict(list)
    for lesion_id, group in groups.items():
        by_class[group[0].dx].append(lesion_id)
    missing = [dx for dx in CLASS_ORDER if dx not in by_class]
    if missing:
        raise ContractError("missing_class", "Seven-class splitting requires every class.", classes=missing)
    partitions: dict[str, list[ManifestRecord]] = {name: [] for name in SPLIT_ORDER}
    for dx in CLASS_ORDER:
        quotas = lesion_quotas(len(by_class[dx]), config.proportions)
        def order_key(lesion_id: str) -> tuple[str, str]:
            key = json.dumps([ALGORITHM, config.seed, dx, lesion_id], separators=(",", ":"))
            return hashlib.sha256(key.encode("utf-8")).hexdigest(), lesion_id
        ordered = sorted(by_class[dx], key=order_key)
        offset = 0
        for name, quota in zip(SPLIT_ORDER, quotas):
            for lesion_id in ordered[offset:offset + quota]:
                partitions[name].extend(groups[lesion_id])
            offset += quota
    result = {name: tuple(sorted(partitions[name], key=lambda row: row.image_id))
              for name in SPLIT_ORDER}
    verify_partition_membership(rows, result)
    return result


def verify_partition_membership(records: Iterable[ManifestRecord], partitions: Mapping[str, Iterable[ManifestRecord]]) -> dict:
    """Require exhaustive, unmodified, exactly-once membership plus zero overlap."""
    rows = validate_manifest(records)
    require_unique_content(rows)
    if set(partitions) != set(SPLIT_ORDER):
        raise ContractError("invalid_partitions", "Exactly three named partitions are required.")
    assigned = {name: tuple(partitions[name]) for name in SPLIT_ORDER}
    report = require_zero_overlap(assigned)
    combined = validate_manifest(row for name in SPLIT_ORDER for row in assigned[name])
    if combined != rows:
        raise ContractError("membership_mismatch", "Partitions omit, add or alter manifest records.")
    for name, group in assigned.items():
        if set(row.dx for row in group) != set(CLASS_ORDER):
            raise ContractError("missing_partition_class", "Every partition must contain all seven classes.",
                                partition=name)
    return report


def partition_counts(partitions: Mapping[str, Iterable[ManifestRecord]], config: SplitConfig) -> dict:
    materialized = {name: tuple(rows) for name, rows in partitions.items()}
    totals = {dx: {"images": sum(row.dx == dx for rows in materialized.values() for row in rows),
                   "lesions": len({row.lesion_id for rows in materialized.values() for row in rows if row.dx == dx})}
              for dx in CLASS_ORDER}
    result = {}
    for name, proportion in zip(SPLIT_ORDER, config.proportions):
        rows = materialized[name]
        by_class = {}
        for dx in CLASS_ORDER:
            images = sum(row.dx == dx for row in rows)
            lesions = len({row.lesion_id for row in rows if row.dx == dx})
            by_class[dx] = {"images": images, "lesions": lesions,
                            "target_images": totals[dx]["images"] * proportion,
                            "target_lesions": totals[dx]["lesions"] * proportion,
                            "image_deviation": images - totals[dx]["images"] * proportion,
                            "lesion_deviation": lesions - totals[dx]["lesions"] * proportion}
        result[name] = {"images": len(rows), "lesions": len({row.lesion_id for row in rows}),
                        "by_class": by_class}
    return result
