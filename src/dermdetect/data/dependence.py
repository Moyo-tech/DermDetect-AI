"""Pure physical-unit membership and candidate paired resampling contracts.

No predictions, outcomes, weights, scores or confidence intervals are accepted.
Conservative clusters describe recorded uncertainty, not verified independent
lesions or patients. The numerical blueprint is synthetic-tested, not an approved
production policy. Its variable physical-unit sample size for unequal clusters
does not choose an estimand or cluster-equal weighting.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Iterable, Mapping

from .analysis import AnalysisRecord
from .conservative_groups import ConservativeGrouping, validate_conservative_groups
from .schemas import CLASS_ORDER, ContractError

POLICY = "physical-membership-dependence-plan-v1"
PHYSICAL_UNITS = "physical-analysis-units"
CONSERVATIVE_CLUSTERS = "conservative-split-clusters"
BLUEPRINT = "class-stratified-cluster-blueprint-v1"


def _items(value, code, message):
    if isinstance(value, (str, bytes)):
        raise ContractError(code, message)
    try:
        return tuple(value)
    except TypeError as exc:
        raise ContractError(code, message) from exc


@dataclass(frozen=True)
class PhysicalUnit:
    physical_id: str
    cluster_id: str
    label: str
    original_image_ids: tuple[str, ...]
    active_image_ids: tuple[str, ...]


@dataclass(frozen=True)
class DependenceCluster:
    cluster_id: str
    physical_ids: tuple[str, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class DependencePlan:
    grouping: ConservativeGrouping
    resampling_unit_policy: str
    physical_units: tuple[PhysicalUnit, ...]
    clusters: tuple[DependenceCluster, ...]
    # (partition name, canonical image IDs). Empty means no allocation supplied.
    partitions: tuple[tuple[str, tuple[str, ...]], ...]
    membership_sha256: str
    policy: str = POLICY


@dataclass(frozen=True)
class PairedDraw:
    selected_cluster_ids: tuple[str, ...]
    # Includes zero multiplicities, in canonical physical-ID order.
    physical_multiplicity: tuple[tuple[str, int], ...]
    indices_a: tuple[int, ...]
    indices_b: tuple[int, ...]


@dataclass(frozen=True)
class PairedBlueprint:
    membership_sha256: str
    resampling_unit_policy: str
    seed: int
    replicate_count: int
    physical_ids: tuple[str, ...]
    system_a_ids: tuple[str, ...]
    system_b_ids: tuple[str, ...]
    draws: tuple[PairedDraw, ...]
    partition: str | None
    scope: str
    observed_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    seven_class_membership_complete: bool
    policy: str = BLUEPRINT
    approved_production_policy: bool = False
    production_freeze: bool = False
    physical_class_counts_fixed: bool = False


def _partitions(grouping, partitions):
    if partitions is None:
        return ()
    if not isinstance(partitions, Mapping) or not partitions:
        raise ContractError("invalid_partitions", "Supply named complete partitions or None.")
    by_image = {r.original.image_id: r for r in grouping.records}
    seen, owners, result = set(), {}, []
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in partitions):
        raise ContractError("invalid_partitions", "Partition names must be canonical nonempty strings.")
    for name, members in sorted(partitions.items()):
        ids = []
        for row in _items(members, "invalid_partitions", "Partition membership must be iterable."):
            if (not isinstance(row, AnalysisRecord) or by_image.get(row.original.image_id) != row
                    or row.original.image_id in seen):
                raise ContractError("membership_mismatch", "Each unmodified original row must occur exactly once.")
            seen.add(row.original.image_id)
            ids.append(row.original.image_id)
            if row.split_group_id in owners and owners[row.split_group_id] != name:
                raise ContractError("cross_split_overlap", "A conservative cluster cannot cross partitions.")
            owners[row.split_group_id] = name
        result.append((name, tuple(sorted(ids))))
    if seen != set(by_image):
        raise ContractError("membership_mismatch", "Partitions must cover every original row exactly once.")
    return tuple(result)


def build_dependence_plan(grouping: ConservativeGrouping, *,
                          resampling_unit_policy: str,
                          partitions: Mapping[str, Iterable[AnalysisRecord]] | None = None) -> DependencePlan:
    """Keep aggregation/calibration/evaluation membership at physical identities.

    An explicit cluster policy permits membership planning, not independent-unit
    or patient independence claims, allocation approval, or a weighting choice.
    """
    grouping = validate_conservative_groups(grouping)
    if resampling_unit_policy not in (PHYSICAL_UNITS, CONSERVATIVE_CLUSTERS):
        raise ContractError("resampling_policy_required", "Select an explicit supported resampling-unit policy.")
    members, cluster_members = defaultdict(list), defaultdict(set)
    for row in grouping.records:
        members[row.analysis_lesion_id].append(row)
        cluster_members[row.split_group_id].add(row.analysis_lesion_id)
    units = tuple(PhysicalUnit(identity, rows[0].split_group_id, rows[0].original.dx,
                              tuple(r.original.image_id for r in rows),
                              tuple(r.original.image_id for r in rows if r.active))
                  for identity, rows in sorted(members.items()))
    if any(not unit.active_image_ids for unit in units):
        raise ContractError("absent_active_unit", "Every physical unit needs active distinct views.")
    by_id = {unit.physical_id: unit for unit in units}
    clusters = tuple(DependenceCluster(identity, tuple(sorted(ids)),
                                       tuple(sorted({by_id[key].label for key in ids})))
                     for identity, ids in sorted(cluster_members.items()))
    if resampling_unit_policy == PHYSICAL_UNITS and any(len(c.physical_ids) > 1 for c in clusters):
        raise ContractError("unresolved_dependence", "Independent physical-unit resampling cannot separate uncertainty-linked members.")
    assigned = _partitions(grouping, partitions)
    payload = {"policy": POLICY, "resampling_unit_policy": resampling_unit_policy,
               "grouping": asdict(grouping), "partitions": assigned}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    return DependencePlan(grouping, resampling_unit_policy, units, clusters, assigned, digest)


def validate_dependence_plan(plan: DependencePlan) -> DependencePlan:
    """Rebuild memberships/digest; reject altered units, labels, counts or splits."""
    if not isinstance(plan, DependencePlan) or plan.policy != POLICY:
        raise ContractError("invalid_dependence_plan", "Use a policy-bound dependence plan.")
    validate_conservative_groups(plan.grouping)
    if type(plan.partitions) is not tuple:
        raise ContractError("invalid_dependence_plan", "Partition membership must be an immutable tuple.")
    partitions = None
    if plan.partitions:
        by_id = {r.original.image_id: r for r in plan.grouping.records}
        try:
            partitions = {name: tuple(by_id[key] for key in ids) for name, ids in plan.partitions}
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid_dependence_plan", "Malformed partition membership.") from exc
    expected = build_dependence_plan(plan.grouping, resampling_unit_policy=plan.resampling_unit_policy,
                                     partitions=partitions)
    if expected != plan:
        raise ContractError("invalid_dependence_plan", "All plan fields must match anchored membership exactly.")
    return plan


def _scope(plan, partition):
    plan = validate_dependence_plan(plan)
    if partition is None:
        if plan.partitions:
            raise ContractError("partition_scope_required", "Select the target partition explicitly; never pool assigned partitions.")
        return plan.physical_units, plan.clusters
    assigned = dict(plan.partitions)
    if not isinstance(partition, str) or partition not in assigned:
        raise ContractError("invalid_partition_scope", "Requested partition must exist in the validated plan.")
    ids = set(assigned[partition])
    units = tuple(u for u in plan.physical_units if set(u.original_image_ids) <= ids)
    if not units:
        raise ContractError("empty_resampling_scope", "An empty partition cannot be resampled.")
    physical_ids = {u.physical_id for u in units}
    return units, tuple(c for c in plan.clusters if set(c.physical_ids) <= physical_ids)


def dependence_report(plan: DependencePlan) -> dict:
    plan = validate_dependence_plan(plan)
    return {"policy": POLICY, "membership_sha256": plan.membership_sha256,
            "resampling_unit_policy": plan.resampling_unit_policy,
            "aggregation_unit": "physical-analysis-identity",
            "calibration_unit": "physical-analysis-identity",
            "evaluation_unit": "physical-analysis-identity",
            "physical_units": len(plan.physical_units), "clusters": len(plan.clusters),
            "original_records": len(plan.grouping.records),
            "active_distinct_views": sum(len(u.active_image_ids) for u in plan.physical_units),
            "class_counts": {dx: {"physical_units": sum(u.label == dx for u in plan.physical_units),
                                   "active_distinct_views": sum(len(u.active_image_ids) for u in plan.physical_units if u.label == dx),
                                   "homogeneous_clusters": sum(c.labels == (dx,) for c in plan.clusters)} for dx in CLASS_ORDER},
            "cluster_sizes": {c.cluster_id: len(c.physical_ids) for c in plan.clusters},
            "cluster_labels": {c.cluster_id: c.labels for c in plan.clusters},
            "mixed_label_clusters": {c.cluster_id: c.labels for c in plan.clusters if len(c.labels) > 1},
            "physical_independence_claim": False, "patient_independence_claim": False,
            "weighting_policy_selected": False, "approved_production_policy": False,
            "production_freeze": False}


def validate_aggregation_membership(plan: DependencePlan,
                                    systems: Mapping[str, Mapping[str, Iterable[str]]], *,
                                    partition: str | None = None) -> dict:
    """Require each system to use every physical unit's active distinct views once.

    Only membership is accepted; the function does not average predictions or
    choose a within-unit reduction, calibration method or loss/metric weight.
    """
    units, _ = _scope(plan, partition)
    expected = {u.physical_id: u.active_image_ids for u in units}
    if not isinstance(systems, Mapping) or not systems:
        raise ContractError("invalid_system_membership", "Supply named system memberships.")
    for name, membership in systems.items():
        if (not isinstance(name, str) or not name.strip() or name != name.strip()
                or not isinstance(membership, Mapping) or set(membership) != set(expected)):
            raise ContractError("aggregation_membership_mismatch", "Every system must cover the exact physical identities.")
        for identity, image_ids in membership.items():
            actual = _items(image_ids, "aggregation_membership_mismatch", "Active-view membership must be iterable.")
            if (any(not isinstance(key, str) for key in actual)
                    or len(set(actual)) != len(actual) or set(actual) != set(expected[identity])):
                raise ContractError("aggregation_membership_mismatch", "Use each active distinct view once in its anchored physical unit.")
    return {"membership_sha256": plan.membership_sha256, "physical_units": len(expected),
            "active_distinct_views": sum(map(len, expected.values())),
            "systems": tuple(sorted(systems)), "aggregation_unit": "physical-analysis-identity",
            "weighting_policy_selected": False, "production_freeze": False}


def paired_class_stratified_cluster_blueprint(plan: DependencePlan,
                                              system_a_ids: Iterable[str], system_b_ids: Iterable[str], *,
                                              seed: int, replicate_count: int,
                                              partition: str | None = None) -> PairedBlueprint:
    """Sample each observed class's cluster count with replacement, jointly.

    Seed and replicate count have no defaults. Mixed-label clusters require a
    separately reviewed alternative; they are neither split nor majority-labeled.
    Inputs must cover the same complete physical-ID scope exactly once. Class
    coverage must include all seven prescribed classes. Presence alone does not
    establish scientific adequacy, power, or metric computability. Draws
    may omit IDs and repeat IDs; every member of a selected cluster has identical
    multiplicity, applied to both systems despite different input row orders.
    """
    units, clusters = _scope(plan, partition)
    if type(seed) is not int or seed < 0 or type(replicate_count) is not int or replicate_count <= 0:
        raise ContractError("invalid_resampling_configuration", "Caller seed must be a nonnegative integer and replicate count a positive integer.")
    if any(len(c.labels) != 1 for c in clusters):
        raise ContractError("mixed_label_cluster_stratification", "Mixed-label clusters need a separately reviewed resampling policy.")
    physical_ids = tuple(u.physical_id for u in units)
    a = _items(system_a_ids, "paired_coverage_mismatch", "System physical IDs must be iterable.")
    b = _items(system_b_ids, "paired_coverage_mismatch", "System physical IDs must be iterable.")
    for ids in (a, b):
        if (any(not isinstance(key, str) for key in ids) or len(ids) != len(set(ids))
                or set(ids) != set(physical_ids)):
            raise ContractError("paired_coverage_mismatch", "Both systems must cover each scoped physical identity exactly once.")
    position_a, position_b = {key: i for i, key in enumerate(a)}, {key: i for i, key in enumerate(b)}
    by_class = {dx: tuple(c for c in clusters if c.labels == (dx,)) for dx in CLASS_ORDER}
    observed = tuple(dx for dx in CLASS_ORDER if by_class[dx])
    missing = tuple(dx for dx in CLASS_ORDER if not by_class[dx])
    if missing:
        raise ContractError("missing_class_strata", "The primary seven-class blueprint requires every prescribed class in scope.", missing_classes=missing)
    rng, draws = random.Random(seed), []
    for _ in range(replicate_count):
        selected, multiplicity = [], dict.fromkeys(physical_ids, 0)
        for dx in CLASS_ORDER:
            bucket = by_class[dx]
            for _ in bucket:
                cluster = bucket[rng.randrange(len(bucket))]
                selected.append(cluster.cluster_id)
                for identity in cluster.physical_ids:
                    multiplicity[identity] += 1
        repeated = tuple(key for key in physical_ids for _ in range(multiplicity[key]))
        draws.append(PairedDraw(tuple(selected), tuple(multiplicity.items()),
                                tuple(position_a[key] for key in repeated),
                                tuple(position_b[key] for key in repeated)))
    return PairedBlueprint(plan.membership_sha256, plan.resampling_unit_policy, seed,
                           replicate_count, physical_ids, a, b, tuple(draws), partition,
                           "partition:" + partition if partition is not None else "unpartitioned-synthetic-contract-population",
                           observed, missing, not bool(missing))
