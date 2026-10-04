"""Deterministic whole-component allocation with exact physical-unit quotas.

No I/O, outcomes, relabeling, exclusions, seed changes or production freeze.
Bounded exhaustive search assigns nontrivial components; remaining singleton
physical units fill the exact residual class quotas. Search exhaustion and a
proved infeasible problem are distinct errors. Baseline preferences reduce
disruption without claiming a globally minimum-movement assignment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Iterable, Mapping

from .analysis import AnalysisRecord
from .conservative_groups import ConservativeGrouping, validate_conservative_groups
from .group_allocation import PERCENTAGES, SEED, SPLITS, allocate_groups, group_quotas
from .schemas import CLASS_ORDER, ContractError

ALGORITHM = "physical-quota-whole-component-dfs-sha256-v1"
MAX_SEARCH_NODES = 100_000
MAX_NONTRIVIAL_COMPONENTS = 256


def _key(*values):
    raw = json.dumps([ALGORITHM, SEED, *values], separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest(), values


def configuration():
    return {"algorithm": ALGORITHM, "seed": SEED, "percentages": PERCENTAGES,
            "quota_unit": "physical_analysis_lesion", "search_node_limit": MAX_SEARCH_NODES,
            "nontrivial_component_limit": MAX_NONTRIVIAL_COMPONENTS,
            "search": "complete bounded DFS, failed-state memoization, capacity/whole-component forward pruning",
            "order": "decreasing physical component size, then fixed-seed SHA-256",
            "partition_preference": "fewest component members moved from fixed physical baseline, then fixed-seed SHA-256",
            "singleton_fill": "retain baseline ownership within residual capacities; fixed-seed overflow order fills train/validation/internal_test",
            "movement_optimality_claim": False, "production_freeze": False}


def _inputs(grouping):
    grouping = validate_conservative_groups(grouping)
    groups, physical = defaultdict(list), {}
    for row in grouping.records:
        groups[row.split_group_id].append(row)
        physical[row.analysis_lesion_id] = row.original.dx
    counts = Counter(physical.values())
    if set(counts) != set(CLASS_ORDER):
        raise ContractError("missing_class", "All seven source classes are required.")
    quotas = {dx: group_quotas(counts[dx]) for dx in CLASS_ORDER}
    members = {g: frozenset(r.analysis_lesion_id for r in rows) for g, rows in groups.items()}
    vectors = {g: tuple(sum(physical[p] == dx for p in ids) for dx in CLASS_ORDER)
               for g, ids in members.items()}
    baseline = allocate_groups(grouping.physical_records)
    owner = {r.analysis_lesion_id: name for name, rows in baseline.items() for r in rows}
    return grouping, groups, physical, members, vectors, quotas, owner


def _solve(grouping, search_node_limit):
    if type(search_node_limit) is not int or search_node_limit <= 0:
        raise ContractError("invalid_search_limit", "A positive integer search budget is required.")
    grouping, groups, physical, members, vectors, quotas, baseline = _inputs(grouping)
    large = sorted((g for g, ids in members.items() if len(ids) > 1),
                   key=lambda g: (-len(members[g]), _key("component", g)))
    if len(large) > MAX_NONTRIVIAL_COMPONENTS:
        raise ContractError("allocation_search_limit", "Component depth limit reached; infeasibility is not established.")
    capacity = [[quotas[dx][s] for dx in CLASS_ORDER] for s in range(len(SPLITS))]
    failed, placement = set(), {}
    nodes = 0

    def fits(group, split):
        return all(v <= capacity[split][c] for c, v in enumerate(vectors[group]))

    def visit(depth):
        nonlocal nodes
        nodes += 1
        if nodes > search_node_limit:
            raise ContractError("allocation_search_limit", "Search budget reached; infeasibility is not established.",
                                search_node_limit=search_node_limit)
        if depth == len(large):
            return True
        state = depth, tuple(v for split in capacity for v in split)
        if state in failed:
            return False
        if any(not any(fits(g, s) for s in range(len(SPLITS))) for g in large[depth:]):
            failed.add(state)
            return False
        group = large[depth]
        choices = sorted(range(len(SPLITS)), key=lambda s: (
            sum(baseline[p] != SPLITS[s] for p in members[group]), _key("partition", group, SPLITS[s])))
        for split in choices:
            if not fits(group, split):
                continue
            for c, value in enumerate(vectors[group]):
                capacity[split][c] -= value
            placement[group] = SPLITS[split]
            if visit(depth + 1):
                return True
            del placement[group]
            for c, value in enumerate(vectors[group]):
                capacity[split][c] += value
        failed.add(state)
        return False

    if not visit(0):
        raise ContractError("infeasible_constrained_allocation", "No whole-component assignment satisfies the unchanged physical-unit quotas.",
                            search_nodes=nodes, failed_states=len(failed), class_quotas=quotas)
    # Each remaining group contains exactly one physical unit. Nonnegative
    # residual capacities sum to their class counts, so completion is exact.
    singletons = [g for g, ids in members.items() if len(ids) == 1]
    for c, dx in enumerate(CLASS_ORDER):
        candidates = [g for g in singletons if vectors[g][c] == 1]
        overflow = []
        for split, name in enumerate(SPLITS):
            own = sorted((g for g in candidates if baseline[next(iter(members[g]))] == name),
                         key=lambda g: _key("singleton", dx, g))
            keep = min(len(own), capacity[split][c])
            for group in own[:keep]:
                placement[group] = name
            overflow.extend(own[keep:])
            capacity[split][c] -= keep
        overflow.sort(key=lambda g: _key("singleton", dx, g))
        offset = 0
        for split, name in enumerate(SPLITS):
            count = capacity[split][c]
            for group in overflow[offset:offset + count]:
                placement[group] = name
            offset += count
            capacity[split][c] = 0
        if offset != len(overflow):
            raise ContractError("allocation_internal_error", "Residual singleton counts do not match capacities.")
    assigned = {name: tuple(sorted((r for g, rows in groups.items() if placement[g] == name for r in rows),
                                  key=lambda r: r.original.image_id)) for name in SPLITS}
    return assigned, {"search_nodes": nodes, "failed_states": len(failed),
                      "nontrivial_components": len(large), "search_node_limit": search_node_limit,
                      "infeasibility_proved": False, "movement_optimality_claim": False}


def allocate_constrained_groups(grouping: ConservativeGrouping, *, search_node_limit=MAX_SEARCH_NODES):
    assigned, _ = _solve(grouping, search_node_limit)
    verify_constrained_allocation(grouping, assigned)
    return assigned


def propose_constrained_groups(grouping: ConservativeGrouping):
    assigned, search = _solve(grouping, MAX_SEARCH_NODES)
    return assigned, verify_constrained_allocation(grouping, assigned), search


def verify_constrained_allocation(grouping: ConservativeGrouping,
                                  partitions: Mapping[str, Iterable[AnalysisRecord]]) -> dict:
    grouping, groups, physical, members, vectors, quotas, baseline = _inputs(grouping)
    if not isinstance(partitions, Mapping) or set(partitions) != set(SPLITS):
        raise ContractError("invalid_partitions", "Exactly the three prescribed partitions are required.")
    try:
        assigned = {name: tuple(partitions[name]) for name in SPLITS}
    except TypeError as exc:
        raise ContractError("invalid_partitions", "Partition members must be iterable.") from exc
    if any(not isinstance(r, AnalysisRecord) for rows in assigned.values() for r in rows):
        raise ContractError("membership_mismatch", "Every original record must be present once, unmodified.")
    combined = tuple(sorted((r for rows in assigned.values() for r in rows), key=lambda r: r.original.image_id))
    if combined != grouping.records:
        raise ContractError("membership_mismatch", "Every original record must be present once, unmodified.")
    owner, counts = {}, {}
    for name, rows in assigned.items():
        for r in rows:
            for field, value in (("image", r.original.image_id), ("lesion", r.original.lesion_id),
                                 ("hash", r.original.sha256), ("physical", r.analysis_lesion_id),
                                 ("conservative", r.split_group_id)):
                if (field, value) in owner and owner[field, value] != name:
                    raise ContractError("cross_split_overlap", "An image, original lesion, exact view, physical identity or uncertainty component crosses partitions.",
                                        field=field, value=value)
                owner[field, value] = name
        counts[name] = {dx: {"physical_units": len({r.analysis_lesion_id for r in rows if r.original.dx == dx}),
                            "split_groups_with_class": len({r.split_group_id for r in rows if r.original.dx == dx}),
                            "original_records": sum(r.original.dx == dx for r in rows),
                            "active_views": sum(r.active and r.original.dx == dx for r in rows)} for dx in CLASS_ORDER}
    for dx in CLASS_ORDER:
        observed = tuple(counts[name][dx]["physical_units"] for name in SPLITS)
        if observed != quotas[dx]:
            raise ContractError("quota_mismatch", "Class quotas count unchanged physical units, not uncertainty clusters.", dx=dx)
    moved = sum(owner["physical", p] != baseline[p] for p in physical)
    mixed = {g: tuple(dx for dx, count in zip(CLASS_ORDER, vectors[g]) if count)
             for g in groups if sum(bool(v) for v in vectors[g]) > 1}
    return {"algorithm": ALGORITHM, "seed": SEED, "percentages": PERCENTAGES,
            "quota_unit": "physical_analysis_lesion", "class_physical_quotas": quotas,
            "original_records": len(grouping.records), "active_views": sum(r.active for r in grouping.records),
            "physical_units": len(physical), "split_groups": len(groups),
            "nontrivial_split_groups": sum(len(ids) > 1 for ids in members.values()),
            "mixed_label_groups": mixed, "physical_units_moved_from_fixed_baseline": moved,
            "matches_fixed_physical_allocation": moved == 0, "zero_overlap": True,
            "overlap_dimensions": ("image", "original_lesion", "exact_hash", "physical_identity", "conservative_component"),
            "counts": counts, "verification_scope": "immutable_membership_overlap_exact_physical_quotas",
            "movement_optimality_claim": False, "rare_class_adequacy_claim": False,
            "status": "proposed_assignment_verified_not_frozen", "production_freeze": False}
