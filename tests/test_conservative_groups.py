"""Invented metadata and review provenance only; no real assignments or I/O."""
from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.analysis import CandidateDecision, resolve_analysis, validate_analysis
from dermdetect.data.conservative_groups import (
    ConservativeGrouping, ReviewedUncertainty, build_conservative_groups,
    validate_conservative_groups, conservative_group_report,
    allocate_conservative_groups, verify_conservative_allocation,
)
from dermdetect.data.group_allocation import allocate_groups, SPLITS
from dermdetect.data.schemas import CLASS_ORDER, ContractError, ManifestRecord


def record(image, lesion=None, dx="nv", payload=None):
    return ManifestRecord(image, lesion or image, dx, "invented", "v1", "invented/" + image,
                          hashlib.sha256((payload or image).encode()).hexdigest())


def link(a, b):
    return ReviewedUncertainty(a, b, "invented full-view evidence", "invented reviewer",
                               "invented ambiguity retained", "2026-10-02", full_view_inspected=True)


class GroupTests(unittest.TestCase):
    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_transitive_whole_physical_components_preserve_exact_participation(self):
        source = [record("a", "la", payload="same"), record("copy", "la", payload="same"),
                  record("other", "la"), record("b", "lb"), record("c", "lc")]
        physical = resolve_analysis(source, [])
        grouped = build_conservative_groups(physical, [link("copy", "b"), link("b", "c")])
        self.assertEqual(len({r.split_group_id for r in grouped.records}), 1)
        self.assertEqual(len({r.analysis_lesion_id for r in grouped.records}), 3)
        self.assertEqual(grouped.physical_records, physical)
        for original, changed in zip(physical, grouped.records):
            self.assertEqual(replace(changed, split_group_id=changed.analysis_lesion_id), original)
        self.assertEqual(sum(r.active for r in grouped.records), 4)
        self.assertEqual(next(r for r in grouped.records if r.original.image_id == "copy").representative_image_id, "a")
        self.assertEqual(grouped, build_conservative_groups(reversed(physical), [link("c", "b"), link("b", "copy")]))
        self.assertEqual(validate_conservative_groups(grouped), grouped)
        self.assert_code("invalid_analysis_group", lambda: validate_analysis(grouped.records))
        with self.assertRaises(FrozenInstanceError):
            grouped.policy = "forged"

    def test_confirmed_physical_identity_is_not_replaced_by_uncertain_identity(self):
        source = [record("a"), record("b"), record("c")]
        physical = resolve_analysis(source, [CandidateDecision("a", "b", "invented", "same_lesion", "r", "why")])
        grouped = build_conservative_groups(physical, [link("a", "c")])
        self.assertEqual(len({r.analysis_lesion_id for r in grouped.records}), 2)
        self.assertEqual(len({r.split_group_id for r in grouped.records}), 1)

    def test_unknown_self_duplicate_component_and_uninspected_links_rejected(self):
        physical = resolve_analysis([record("a", "la"), record("a2", "la"), record("b")], [])
        for invalid in [link("a", "missing"), link("a", "a"), link("a", "a2")]:
            self.assert_code("invalid_uncertainty", lambda: build_conservative_groups(physical, [invalid]))
        for flag in (False, 1, "true", None):
            self.assert_code("uninspected_uncertainty", lambda: build_conservative_groups(physical, [replace(link("a", "b"), full_view_inspected=flag)]))
        self.assert_code("repeated_uncertainty", lambda: build_conservative_groups(physical, [link("a", "b"), link("b", "a")]))
        omitted = ReviewedUncertainty("a", "b", "evidence", "reviewer", "rationale", "2026-10-02")
        self.assert_code("uninspected_uncertainty", lambda: build_conservative_groups(physical, [omitted]))

    def test_distinct_view_evidence_between_same_components_is_retained(self):
        physical = resolve_analysis([record("a1", "la"), record("a2", "la"),
                                     record("b1", "lb"), record("b2", "lb")], [])
        links = [link("a1", "b1"), link("a2", "b2")]
        g = build_conservative_groups(physical, links)
        self.assertEqual(len(g.uncertainties), 2)
        self.assertEqual(len({r.split_group_id for r in g.records}), 1)
        self.assertEqual(g, build_conservative_groups(reversed(physical), reversed(links)))
        self.assertEqual(g.records, build_conservative_groups(physical, links[:1]).records)

    def test_provenance_and_malformed_inputs_fail_closed(self):
        physical = resolve_analysis([record("a"), record("b")], [])
        for field in ("evidence", "reviewer", "rationale", "review_date"):
            self.assert_code("missing_uncertainty_evidence", lambda: build_conservative_groups(physical, [replace(link("a", "b"), **{field: ""})]))
        for value in ("20261002", "2026-13-01", "today"):
            self.assert_code("invalid_review_date", lambda: build_conservative_groups(physical, [replace(link("a", "b"), review_date=value)]))
        self.assert_code("invalid_uncertainty", lambda: build_conservative_groups(physical, [object()]))
        self.assert_code("invalid_uncertainty", lambda: build_conservative_groups(physical, None))
        self.assert_code("invalid_physical_records", lambda: build_conservative_groups([object()], []))
        self.assert_code("invalid_conservative_grouping", lambda: validate_conservative_groups(physical))

    def test_tampered_or_incomplete_grouping_rejected_against_anchor(self):
        physical = resolve_analysis([record("a", "la"), record("a2", "la"), record("b"), record("c")], [])
        g = build_conservative_groups(physical, [link("a", "b")])
        variants = [replace(g, policy="forged"), replace(g, records=g.records[:-1]),
                    replace(g, uncertainties=()), replace(g, records=list(g.records)),
                    replace(g, records=(replace(g.records[0], active=False),) + g.records[1:]),
                    replace(g, records=(replace(g.records[0], split_group_id=g.records[-1].split_group_id),) + g.records[1:]),
                    replace(g, records=(replace(g.records[0], analysis_lesion_id=g.records[-1].analysis_lesion_id),) + g.records[1:])]
        for forged in variants:
            self.assert_code("invalid_conservative_grouping", lambda: validate_conservative_groups(forged))

    def test_mixed_labels_retained_and_allocation_refused(self):
        physical = resolve_analysis([record("a"), record("b", dx="bkl")], [])
        g = build_conservative_groups(physical, [link("a", "b")])
        report = conservative_group_report(g)
        self.assertEqual((report["physical_units"], report["split_groups"]), (2, 1))
        self.assertFalse(report["same_label_groups"])
        self.assertFalse(report["allocation_feasible"])
        self.assertEqual({r.original.dx for r in g.records}, {"nv", "bkl"})
        self.assert_code("mixed_label_split_group", lambda: allocate_conservative_groups(g))
        self.assert_code("mixed_label_split_group", lambda: verify_conservative_allocation(g, {name: () for name in SPLITS}))


class AllocationTests(unittest.TestCase):
    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def setUp(self):
        source = [record(f"{dx}_{i}_{view}", f"{dx}_{i}", dx)
                  for dx in CLASS_ORDER for i in range(10) for view in range(1 + i % 2)]
        source.append(record("nv_copy", "nv_0", "nv", "nv_0_0"))
        self.physical = resolve_analysis(source, [])
        self.fixed = allocate_groups(self.physical)

    def links_between(self, names):
        return [next(r.original.image_id for r in self.fixed[name] if r.original.dx == "nv") for name in names]

    def remap(self, g, partitions):
        by_image = {r.original.image_id: r for r in g.records}
        return {name: tuple(by_image[r.original.image_id] for r in rows) for name, rows in partitions.items()}

    def test_singletons_keep_original_deterministic_physical_order_and_quotas(self):
        g = build_conservative_groups(self.physical, [])
        assigned = allocate_conservative_groups(g)
        report = verify_conservative_allocation(g, assigned)
        self.assertTrue(report["matches_fixed_physical_allocation"])
        self.assertTrue(report["zero_overlap"])
        self.assertEqual(report["quota_unit"], "physical_analysis_lesion")
        for dx in CLASS_ORDER:
            self.assertEqual(tuple(report["counts"][s][dx]["physical_units"] for s in SPLITS), (7, 2, 1))
        self.assertEqual(assigned, self.remap(g, self.fixed))
        self.assertFalse(report["production_freeze"])

    def test_colocation_already_together_preserves_fixed_assignment_not_cluster_quotas(self):
        ids = list(dict.fromkeys(r.analysis_lesion_id for r in self.fixed["train"] if r.original.dx == "nv"))[:3]
        image = {r.analysis_lesion_id: r.original.image_id for r in self.physical}
        g = build_conservative_groups(self.physical, [link(image[ids[0]], image[ids[1]]), link(image[ids[1]], image[ids[2]])])
        report = verify_conservative_allocation(g, allocate_conservative_groups(g))
        self.assertTrue(report["matches_fixed_physical_allocation"])
        self.assertEqual(report["counts"]["train"]["nv"]["physical_units"], 7)
        self.assertEqual(report["counts"]["train"]["nv"]["split_groups"], 5)
        self.assertEqual((report["physical_units"], report["split_groups"]), (70, 68))

    def test_crossing_fixed_assignment_stops_without_swaps_but_manual_feasible_verifies(self):
        a, b = self.links_between(["train", "validation"])
        g = build_conservative_groups(self.physical, [link(a, b)])
        report = conservative_group_report(g)
        self.assertTrue(report["allocation_policy_required"])
        self.assert_code("allocation_policy_required", lambda: allocate_conservative_groups(g))
        self.assert_code("cross_split_overlap", lambda: verify_conservative_allocation(g, self.remap(g, self.fixed)))
        # Invent a feasible quota-preserving assignment: swap whole physical units.
        bunit = next(r.analysis_lesion_id for r in self.physical if r.original.image_id == b)
        aunit = next(r.analysis_lesion_id for r in self.physical if r.original.image_id == a)
        swapunit = next(r.analysis_lesion_id for r in self.fixed["train"] if r.original.dx == "nv" and r.analysis_lesion_id != aunit)
        changed = dict(self.fixed)
        changed["train"] = tuple(r for r in self.fixed["train"] if r.analysis_lesion_id != swapunit) + tuple(r for r in self.fixed["validation"] if r.analysis_lesion_id == bunit)
        changed["validation"] = tuple(r for r in self.fixed["validation"] if r.analysis_lesion_id != bunit) + tuple(r for r in self.fixed["train"] if r.analysis_lesion_id == swapunit)
        report = verify_conservative_allocation(g, self.remap(g, changed))
        self.assertFalse(report["matches_fixed_physical_allocation"])
        self.assertTrue(report["zero_overlap"])

    def test_missing_class_infeasible_counts_omission_tamper_and_wrong_quota(self):
        incomplete = build_conservative_groups([r for r in self.physical if r.original.dx != "df"], [])
        self.assert_code("infeasible_conservative_quotas", lambda: allocate_conservative_groups(incomplete))
        tiny = build_conservative_groups(resolve_analysis([record(dx, dx, dx) for dx in CLASS_ORDER], []), [])
        self.assert_code("infeasible_conservative_quotas", lambda: allocate_conservative_groups(tiny))
        g = build_conservative_groups(self.physical, [])
        assigned = allocate_conservative_groups(g)
        changed = dict(assigned); changed["train"] = changed["train"][1:]
        self.assert_code("membership_mismatch", lambda: verify_conservative_allocation(g, changed))
        changed = dict(assigned); changed["train"] = (replace(changed["train"][0], active=False),) + changed["train"][1:]
        self.assert_code("membership_mismatch", lambda: verify_conservative_allocation(g, changed))
        unit = assigned["train"][0].analysis_lesion_id
        changed = dict(assigned)
        moved = tuple(r for r in assigned["train"] if r.analysis_lesion_id == unit)
        changed["train"] = tuple(r for r in assigned["train"] if r.analysis_lesion_id != unit)
        changed["validation"] += moved
        self.assert_code("quota_mismatch", lambda: verify_conservative_allocation(g, changed))


if __name__ == "__main__":
    unittest.main()
