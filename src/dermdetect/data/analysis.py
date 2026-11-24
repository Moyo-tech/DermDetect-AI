"""Reviewed lesion identity and distinct-view participation; no payload access."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from .schemas import ContractError, ManifestRecord, validate_manifest

POLICY = "reviewed-lesion-distinct-view-v1"


@dataclass(frozen=True)
class CandidateDecision:
    left: str
    right: str
    evidence: str
    decision: str = "pending"
    reviewer: str = ""
    rationale: str = ""


@dataclass(frozen=True)
class AnalysisRecord:
    original: ManifestRecord
    analysis_lesion_id: str
    split_group_id: str
    active: bool
    representative_image_id: str
    inactive_reason: str

    def to_dict(self) -> dict:
        return {**self.original.to_dict(),
                "analysis_lesion_id": self.analysis_lesion_id,
                "split_group_id": self.split_group_id, "active": self.active,
                "representative_image_id": self.representative_image_id,
                "inactive_reason": self.inactive_reason}


def _group_id(lesion_ids: Iterable[str]) -> str:
    value = json.dumps([POLICY, sorted(lesion_ids)], separators=(",", ":"))
    return "lesion-component-" + hashlib.sha256(value.encode()).hexdigest()


def resolve_analysis(records: Iterable[ManifestRecord],
                     decisions: Iterable[CandidateDecision]) -> tuple[AnalysisRecord, ...]:
    """Require all supplied candidates reviewed; preserve every original row.

    Published conclusions are evidence, never automatic identity decisions.
    A confirmed same-lesion edge joins entire original-lesion components.
    Cross-ID exact matches must also be supplied as reviewed candidate edges.
    """
    rows = validate_manifest(records)
    by_id = {row.image_id: row for row in rows}
    parent = {row.lesion_id: row.lesion_id for row in rows}

    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen = set()
    distinct_pairs = []
    for item in decisions:
        if (not isinstance(item, CandidateDecision) or item.left == item.right
                or item.left not in by_id or item.right not in by_id):
            raise ContractError("invalid_candidate", "Candidate must name two inventory images.")
        key = tuple(sorted((item.left, item.right)))
        if key in seen:
            raise ContractError("repeated_candidate", "Duplicate pair decisions are ambiguous.")
        seen.add(key)
        if item.decision not in {"same_lesion", "different_lesions"}:
            raise ContractError("unresolved_candidate", "Every candidate needs final adjudication.",
                                image_ids=key)
        if any(not isinstance(value, str) or not value.strip()
               for value in (item.evidence, item.reviewer, item.rationale)):
            raise ContractError("missing_review_evidence", "Evidence, reviewer and rationale are required.")
        left, right = by_id[item.left], by_id[item.right]
        if item.decision == "different_lesions":
            if left.lesion_id == right.lesion_id or left.sha256 == right.sha256:
                raise ContractError("contradictory_review", "Distinct-lesion decision conflicts with inventory.")
            distinct_pairs.append((left.lesion_id, right.lesion_id))
            continue
        if left.dx != right.dx:
            raise ContractError("component_label_conflict", "Shared-lesion candidates disagree on label.")
        a, b = root(left.lesion_id), root(right.lesion_id)
        parent[max(a, b)] = min(a, b)

    if any(root(left) == root(right) for left, right in distinct_pairs):
        raise ContractError("contradictory_review", "Distinct-lesion decision conflicts with connected identity evidence.")

    components: dict[str, set[str]] = defaultdict(set)
    for lesion in parent:
        components[root(lesion)].add(lesion)
    identities = {lesion: _group_id(members)
                  for members in components.values() for lesion in members}
    hashes: dict[str, set[str]] = defaultdict(set)
    views: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        identity = identities[row.lesion_id]
        hashes[row.sha256].add(identity)
        views[identity, row.sha256].append(row.image_id)
    if any(len(values) > 1 for values in hashes.values()):
        raise ContractError("unresolved_exact_identity", "Cross-lesion exact matches need identity review.")
    result = []
    for row in rows:
        identity = identities[row.lesion_id]
        representative = min(views[identity, row.sha256])
        active = row.image_id == representative
        result.append(AnalysisRecord(row, identity, identity, active, representative,
                                     "" if active else "repeated_exact_view"))
    return validate_analysis(result)


def validate_analysis(records: Iterable[AnalysisRecord]) -> tuple[AnalysisRecord, ...]:
    """Reject omitted/overweighted views, broken original groups and forged units."""
    rows = tuple(records)
    validate_manifest(row.original for row in rows)
    lesion_units: dict[str, set[str]] = defaultdict(set)
    groups: dict[str, list[AnalysisRecord]] = defaultdict(list)
    views: dict[tuple[str, str], list[AnalysisRecord]] = defaultdict(list)
    hash_units: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.split_group_id != row.analysis_lesion_id or type(row.active) is not bool:
            raise ContractError("invalid_analysis_group", "This policy uses one physical-lesion component.")
        lesion_units[row.original.lesion_id].add(row.analysis_lesion_id)
        groups[row.analysis_lesion_id].append(row)
        views[row.analysis_lesion_id, row.original.sha256].append(row)
        hash_units[row.original.sha256].add(row.analysis_lesion_id)
    if any(len(values) != 1 for values in lesion_units.values()) or any(
            len(values) != 1 for values in hash_units.values()):
        raise ContractError("invalid_analysis_group", "Original lesions and exact views cannot span units.")
    for identity, members in groups.items():
        if identity != _group_id({row.original.lesion_id for row in members}):
            raise ContractError("invalid_analysis_group", "Component ID does not match original lesion membership.")
        if len({row.original.dx for row in members}) != 1:
            raise ContractError("component_label_conflict", "Component labels must agree.")
    for members in views.values():
        representative = min(row.original.image_id for row in members)
        for row in members:
            active = row.original.image_id == representative
            if (row.active != active or row.representative_image_id != representative
                    or row.inactive_reason != ("" if active else "repeated_exact_view")):
                raise ContractError("invalid_participation", "Use exactly the canonical distinct-view representative.")
    return tuple(sorted(rows, key=lambda row: row.original.image_id))
