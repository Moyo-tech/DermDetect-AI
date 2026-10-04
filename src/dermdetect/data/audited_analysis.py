"""Pure accepted-audit identity/uncertainty adapter; no I/O or approval grant.

The file adapter must bind exact source, audit, observation and acceptance
bytes before calling this module. Human annotations override recommendations;
uninspected retained recommendations may only be agent-different. Attribution
is preserved. Pending identity becomes a reviewed split-only link, never a
physical merge or an omitted candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable

from .analysis import CandidateDecision, resolve_analysis
from .conservative_groups import ConservativeGrouping, ReviewedUncertainty, build_conservative_groups
from .schemas import ContractError, ManifestRecord, validate_manifest

POLICY = "accepted-audit-physical-uncertainty-split-v1"


@dataclass(frozen=True)
class AuditedAnalysis:
    grouping: ConservativeGrouping
    effective_decisions: tuple[dict, ...]
    policy: str = POLICY


def _key(item, by_id):
    if not isinstance(item, dict):
        raise ContractError("invalid_audited_decision", "Pair evidence must be an object.")
    a, b = item.get("left"), item.get("right")
    if (any(not isinstance(i, str) or not i or i != i.strip() for i in (a, b))
            or a == b or a not in by_id or b not in by_id):
        raise ContractError("invalid_audited_decision", "Canonical inventory image pairs are required.")
    return tuple(sorted((a, b)))


def _date(value):
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except (TypeError, ValueError):
        raise ContractError("invalid_review_date", "Inspection dates must be ISO calendar dates.")


def resolve_audited_analysis(records: Iterable[ManifestRecord], *, candidate_pairs,
                            agent_observations, human_decisions, required_human_pairs) -> AuditedAnalysis:
    rows = validate_manifest(records)
    by_id = {r.image_id: r for r in rows}
    candidate_keys = [_key(r, by_id) for r in candidate_pairs]
    if not candidate_keys or len(candidate_keys) != len(set(candidate_keys)):
        raise ContractError("candidate_membership_mismatch", "Full candidate membership must be nonempty and unique.")
    required_keys = [_key(r, by_id) for r in required_human_pairs]
    if len(required_keys) != len(set(required_keys)) or not set(required_keys) <= set(candidate_keys):
        raise ContractError("candidate_membership_mismatch", "Required human pairs must be unique full-queue members.")
    observed, human = {}, {}
    for item in agent_observations:
        key = _key(item, by_id)
        if key in observed or key not in candidate_keys:
            raise ContractError("candidate_membership_mismatch", "Agent observations must cover the full queue once.")
        if (item.get("agent_recommendation") not in {"same_lesion", "different_lesions", "uncertain"}
                or item.get("full_resolution_viewed") is not True
                or item.get("human_review_status") != "not_performed"
                or any(not isinstance(item.get(f), str) or not item[f].strip()
                       for f in ("agent_reviewer", "agent_rationale", "view_file"))
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("view_sha256", "")))
                or item.get("source_payload_sha256") != {i: by_id[i].sha256 for i in key}):
            raise ContractError("invalid_audited_observation", "Attributed, full-view, source-hash-bound recommendations are required.")
        _date(item.get("review_date"))
        observed[key] = item
    if set(observed) != set(candidate_keys):
        raise ContractError("candidate_membership_mismatch", "Agent observations must cover every full-queue candidate.")
    for item in human_decisions:
        key = _key(item, by_id)
        if key in human or key not in required_keys:
            raise ContractError("candidate_membership_mismatch", "Human decisions must cover the prescribed audit once.")
        source = {i: {"source": by_id[i].source, "source_version": by_id[i].source_version} for i in key}
        if (item.get("decision") not in {"same_lesion", "different_lesions", "pending"}
                or item.get("full_view_inspected") is not True
                or any(not isinstance(item.get(f), str) or not item[f].strip() for f in ("reviewer", "rationale"))
                or item.get("source_applicability") != source
                or item.get("source_payload_sha256") != observed[key]["source_payload_sha256"]
                or item.get("view_sha256") != observed[key]["view_sha256"]
                or item.get("evidence") != observed[key]["view_file"]):
            raise ContractError("invalid_audited_decision", "Human decisions must retain exact source/view bindings and inspection evidence.")
        _date(item.get("review_date"))
        human[key] = item
    if set(human) != set(required_keys):
        raise ContractError("human_audit_incomplete", "Every required human pair needs inspection, including unclear pairs.")
    effective, resolved, uncertain = [], [], []
    for key in sorted(candidate_keys):
        observation = observed[key]
        if key in human:
            row = human[key]
            decision, reviewer, rationale, reviewed_date = (row[f] for f in ("decision", "reviewer", "rationale", "review_date"))
            origin = "human"
        else:
            if observation["agent_recommendation"] != "different_lesions":
                raise ContractError("unreviewed_identity_proposal", "Retained uninspected evidence may only be audited agent-different recommendations.")
            decision, reviewer, rationale, reviewed_date = (observation[f] for f in
                ("agent_recommendation", "agent_reviewer", "agent_rationale", "review_date"))
            origin = "agent"
        evidence = observation["view_file"]
        effective.append({"left": key[0], "right": key[1], "decision": decision,
                          "decision_attribution": origin, "reviewer": reviewer, "review_date": reviewed_date,
                          "rationale": rationale, "evidence": evidence,
                          "view_sha256": observation["view_sha256"],
                          "source_payload_sha256": observation["source_payload_sha256"],
                          "original_agent_recommendation": observation["agent_recommendation"]})
        if decision == "pending":
            uncertain.append(ReviewedUncertainty(*key, evidence, reviewer.strip(), rationale.strip(),
                                                 reviewed_date, full_view_inspected=True))
        else:
            resolved.append(CandidateDecision(*key, evidence, decision, reviewer, rationale))
    physical = resolve_analysis(rows, resolved)
    grouped = build_conservative_groups(physical, uncertain)
    return AuditedAnalysis(grouped, tuple(effective))
