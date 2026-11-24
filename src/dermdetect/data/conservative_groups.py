"""Synthetic split-only co-location contracts; no I/O or physical identity decisions.

Evidence strings and inspection attestations are structural provenance, not reviewer
authentication. The retained physical input is the completeness anchor; callers
must separately bind that input to their approved inventory/review artifacts.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
import hashlib
import json
from typing import Iterable, Mapping

from .analysis import AnalysisRecord, validate_analysis
from .group_allocation import ALGORITHM, PERCENTAGES, SEED, SPLITS, allocate_groups, group_quotas
from .schemas import CLASS_ORDER, ContractError

POLICY = "reviewed-uncertainty-split-only-v1"


@dataclass(frozen=True)
class ReviewedUncertainty:
    left: str
    right: str
    evidence: str
    reviewer: str
    rationale: str
    review_date: str
    full_view_inspected: bool = False


@dataclass(frozen=True)
class ConservativeGrouping:
    records: tuple[AnalysisRecord, ...]
    uncertainties: tuple[ReviewedUncertainty, ...]
    policy: str
    physical_records: tuple[AnalysisRecord, ...]


def _physical(records: Iterable[AnalysisRecord]) -> tuple[AnalysisRecord, ...]:
    try:
        rows = tuple(records)
    except TypeError as exc:
        raise ContractError("invalid_physical_records", "An iterable of physical rows is required.") from exc
    if any(not isinstance(r, AnalysisRecord) for r in rows):
        raise ContractError("invalid_physical_records", "Physical rows must be AnalysisRecord objects.")
    try:
        return validate_analysis(rows)
    except (AttributeError, TypeError) as exc:
        raise ContractError("invalid_physical_records", "Malformed physical rows.") from exc


def _construct(physical_records, uncertainties) -> ConservativeGrouping:
    rows = _physical(physical_records)
    by_image = {r.original.image_id: r for r in rows}
    parent = {r.analysis_lesion_id: r.analysis_lesion_id for r in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    try:
        uncertainties = tuple(uncertainties)
    except TypeError as exc:
        raise ContractError("invalid_uncertainty", "An iterable of reviewed links is required.") from exc
    reviewed = []
    seen = set()
    for item in uncertainties:
        if not isinstance(item, ReviewedUncertainty):
            raise ContractError("invalid_uncertainty", "A reviewed uncertainty object is required.")
        fields = (item.left, item.right, item.evidence, item.reviewer,
                  item.rationale, item.review_date)
        if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in fields):
            raise ContractError("missing_uncertainty_evidence", "Canonical IDs, evidence, reviewer, rationale and date are required.")
        if item.full_view_inspected is not True:
            raise ContractError("uninspected_uncertainty", "Untouched or uninspected candidates cannot drive co-location.")
        try:
            parsed = date.fromisoformat(item.review_date)
        except ValueError as exc:
            raise ContractError("invalid_review_date", "Use an ISO calendar date.") from exc
        if parsed.isoformat() != item.review_date:
            raise ContractError("invalid_review_date", "Use YYYY-MM-DD.")
        if item.left == item.right or item.left not in by_image or item.right not in by_image:
            raise ContractError("invalid_uncertainty", "Link must name two distinct physical inventory images.")
        key = tuple(sorted((item.left, item.right)))
        a = by_image[item.left].analysis_lesion_id
        b = by_image[item.right].analysis_lesion_id
        if a == b:
            raise ContractError("invalid_uncertainty", "Uncertainty must link distinct physical components.")
        if key in seen:
            raise ContractError("repeated_uncertainty", "Duplicate image-pair review entries are ambiguous.")
        seen.add(key)
        reviewed.append(replace(item, left=key[0], right=key[1]))
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)
    components = defaultdict(set)
    for identity in parent:
        components[root(identity)].add(identity)
    identities = {}
    for members in components.values():
        raw = json.dumps([POLICY, sorted(members)], separators=(",", ":"))
        identity = "conservative-split-" + hashlib.sha256(raw.encode()).hexdigest()
        identities.update({member: identity for member in members})
    result = tuple(replace(row, split_group_id=identities[row.analysis_lesion_id]) for row in rows)
    return ConservativeGrouping(result, tuple(sorted(reviewed, key=lambda x: (x.left, x.right))), POLICY, rows)


def build_conservative_groups(physical_records: Iterable[AnalysisRecord],
                              uncertainties: Iterable[ReviewedUncertainty]) -> ConservativeGrouping:
    """Co-locate whole reviewed physical components; preserve all other fields."""
    return _construct(physical_records, uncertainties)


def validate_conservative_groups(grouping: ConservativeGrouping) -> ConservativeGrouping:
    """Reconstruct canonical membership, including the independent source anchor."""
    if (not isinstance(grouping, ConservativeGrouping) or grouping.policy != POLICY
            or any(type(v) is not tuple for v in
                   (grouping.records, grouping.uncertainties, grouping.physical_records))):
        raise ContractError("invalid_conservative_grouping", "Use the immutable policy-bound grouping wrapper.")
    expected = _construct(grouping.physical_records, grouping.uncertainties)
    if grouping != expected:
        raise ContractError("invalid_conservative_grouping", "Records, physical anchor, links and canonical split IDs must agree exactly.")
    return grouping


def conservative_group_report(grouping: ConservativeGrouping) -> dict:
    """Report feasibility only under the unchanged fixed physical assignment.

    A crossing group is not proof that every alternative allocator is infeasible;
    it requires an explicitly reviewed policy rather than an automatic repair.
    """
    grouping = validate_conservative_groups(grouping)
    groups = defaultdict(list)
    for row in grouping.records:
        groups[row.split_group_id].append(row)
    mixed = {key: tuple(sorted({r.original.dx for r in members}))
             for key, members in sorted(groups.items()) if len({r.original.dx for r in members}) > 1}
    nontrivial = tuple(key for key, members in sorted(groups.items())
                       if len({r.analysis_lesion_id for r in members}) > 1)
    quotas, issues = {}, []
    if mixed:
        issues.append("mixed_label_split_groups")
    else:
        for dx in CLASS_ORDER:
            count = len({r.analysis_lesion_id for r in grouping.records if r.original.dx == dx})
            if not count:
                issues.append("missing_class:" + dx)
                continue
            try:
                quotas[dx] = group_quotas(count)
            except ContractError as exc:
                issues.append(exc.code + ":" + dx)
    crossing = ()
    if not issues:
        assigned = allocate_groups(grouping.physical_records)
        owner = {r.analysis_lesion_id: name for name, members in assigned.items() for r in members}
        crossing = tuple(key for key, members in sorted(groups.items())
                         if len({owner[r.analysis_lesion_id] for r in members}) > 1)
    return {"policy": POLICY, "physical_units": len({r.analysis_lesion_id for r in grouping.records}),
            "split_groups": len(groups), "original_records": len(grouping.records),
            "active_views": sum(r.active for r in grouping.records),
            "mixed_label_groups": mixed, "same_label_groups": not bool(mixed),
            "quota_counts_feasible": not issues,
            "allocation_feasible": not issues and not crossing,
            "feasibility_scope": "unchanged_fixed_physical_allocation",
            "infeasible_under_fixed_allocation": crossing,
            "nontrivial_split_groups": nontrivial,
            "allocation_policy_required": bool(issues or crossing),
            "feasibility_issues": tuple(issues),
            "quota_unit": "physical_analysis_lesion", "class_physical_quotas": quotas,
            "production_freeze": False}


def _allocation_inputs(grouping):
    grouping = validate_conservative_groups(grouping)
    report = conservative_group_report(grouping)
    if report["mixed_label_groups"]:
        raise ContractError("mixed_label_split_group", "Mixed-label split groups require a separate allocation policy; no exclusions or relabeling.",
                            groups=report["mixed_label_groups"])
    if report["feasibility_issues"]:
        raise ContractError("infeasible_conservative_quotas", "All seven classes need nonzero fixed quotas in every partition.",
                            issues=report["feasibility_issues"])
    groups = defaultdict(list)
    by_class = defaultdict(list)
    for row in grouping.records:
        groups[row.split_group_id].append(row)
    for key, members in groups.items():
        by_class[members[0].original.dx].append(key)
    return grouping, report, groups, by_class


def allocate_conservative_groups(grouping: ConservativeGrouping) -> dict[str, tuple[AnalysisRecord, ...]]:
    """Pure deterministic synthetic allocator; no acceptance or freeze implied."""
    grouping, report, _, _ = _allocation_inputs(grouping)
    if report["allocation_policy_required"]:
        raise ContractError("allocation_policy_required", "Co-location crosses the unchanged fixed physical assignment; no swaps, new seed or cluster quotas inferred.",
                            reason="infeasible_under_fixed_allocation", groups=report["infeasible_under_fixed_allocation"])
    physical_assignment = allocate_groups(grouping.physical_records)
    by_image = {row.original.image_id: row for row in grouping.records}
    result = {name: tuple(by_image[row.original.image_id] for row in members)
              for name, members in physical_assignment.items()}
    verify_conservative_allocation(grouping, result)
    return result


def verify_conservative_allocation(grouping: ConservativeGrouping,
                                   partitions: Mapping[str, Iterable[AnalysisRecord]]) -> dict:
    grouping, report, _, _ = _allocation_inputs(grouping)
    if not isinstance(partitions, Mapping) or set(partitions) != set(SPLITS):
        raise ContractError("invalid_partitions", "Exactly three prescribed partitions are required.")
    try:
        assigned = {name: tuple(partitions[name]) for name in SPLITS}
    except TypeError as exc:
        raise ContractError("invalid_partitions", "Partition values must be row iterables.") from exc
    if any(not isinstance(row, AnalysisRecord) for members in assigned.values() for row in members):
        raise ContractError("membership_mismatch", "Every original row must be assigned once, unmodified.")
    combined = tuple(sorted((r for members in assigned.values() for r in members), key=lambda r: r.original.image_id))
    if combined != grouping.records:
        raise ContractError("membership_mismatch", "Every original row must be assigned once, unmodified.")
    owners, counts = {}, {}
    for name, members in assigned.items():
        for row in members:
            for field, value in (("image", row.original.image_id), ("lesion", row.original.lesion_id),
                                 ("analysis", row.analysis_lesion_id), ("group", row.split_group_id),
                                 ("hash", row.original.sha256)):
                if (field, value) in owners and owners[field, value] != name:
                    raise ContractError("cross_split_overlap", "An original image, lesion, physical unit, split group or hash crosses partitions.", field=field, value=value)
                owners[field, value] = name
        counts[name] = {dx: {"physical_units": len({r.analysis_lesion_id for r in members if r.original.dx == dx}),
                            "split_groups": len({r.split_group_id for r in members if r.original.dx == dx}),
                            "original_records": sum(r.original.dx == dx for r in members),
                            "active_views": sum(r.active and r.original.dx == dx for r in members)} for dx in CLASS_ORDER}
    for dx in CLASS_ORDER:
        if tuple(counts[name][dx]["physical_units"] for name in SPLITS) != report["class_physical_quotas"][dx]:
            raise ContractError("quota_mismatch", "Use fixed class-wise physical analysis-unit quotas.", dx=dx)
    fixed = allocate_groups(grouping.physical_records)
    fixed_owner = {row.original.image_id: name for name, members in fixed.items() for row in members}
    matches_fixed = all(fixed_owner[row.original.image_id] == name
                        for name, members in assigned.items() for row in members)
    return {**report, "algorithm": ALGORITHM, "seed": SEED, "percentages": PERCENTAGES,
            "matches_fixed_physical_allocation": matches_fixed,
            "verification_scope": "membership_overlap_physical_quotas",
            "zero_overlap": True, "counts": counts, "rare_class_adequacy_claim": False,
            "interpretation": "descriptive_comparison_inconclusive_allowed"}
