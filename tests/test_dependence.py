"""Invented metadata only; no predictions or dataset payloads."""
from dataclasses import replace
import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermdetect.data.analysis import resolve_analysis
from dermdetect.data.conservative_groups import build_conservative_groups, ReviewedUncertainty
from dermdetect.data.dependence import (
    CONSERVATIVE_CLUSTERS, PHYSICAL_UNITS, build_dependence_plan,
    dependence_report, paired_class_stratified_cluster_blueprint,
    validate_aggregation_membership, validate_dependence_plan,
)
from dermdetect.data.schemas import ContractError, ManifestRecord


def row(image, lesion, dx="nv", payload=None):
    return ManifestRecord(image, lesion, dx, "invented", "fixture-v1", image + ".txt",
                          hashlib.sha256((payload or image).encode()).hexdigest())


def uncertainty(a, b):
    return ReviewedUncertainty(a, b, "invented-review-evidence", "Synthetic reviewer",
                               "Identity remains unclear", "2026-10-02", True)


def grouping(mixed=False, full=False):
    # Three physical units in one nv cluster, two singleton nv clusters; one bkl.
    rows = [row("a1", "a", payload="view-a"), row("a2", "a", payload="view-a"),
            row("a3", "a"), row("b1", "b", "bkl" if mixed else "nv"),
            row("c1", "c"), row("d1", "d"), row("e1", "e"), row("f1", "f", "bkl")]
    if full:
        rows.extend(row(dx, dx, dx) for dx in ("akiec", "bcc", "df", "mel", "vasc"))
    physical = resolve_analysis(rows, ())
    return build_conservative_groups(physical, (uncertainty("a1", "b1"), uncertainty("b1", "c1")))


class DependenceTests(unittest.TestCase):
    def assert_code(self, code, function, *args, **kwargs):
        with self.assertRaises(ContractError) as caught:
            function(*args, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def plan(self, full=False, **kwargs):
        return build_dependence_plan(grouping(full=full), resampling_unit_policy=CONSERVATIVE_CLUSTERS, **kwargs)

    def test_physical_membership_distinct_views_and_accurate_counts(self):
        plan = self.plan()
        report = dependence_report(plan)
        self.assertEqual((report["physical_units"], report["clusters"], report["original_records"],
                          report["active_distinct_views"]), (6, 4, 8, 7))
        self.assertEqual(report["class_counts"]["nv"], {
            "physical_units": 5, "active_distinct_views": 6, "homogeneous_clusters": 3})
        unit = next(u for u in plan.physical_units if "a1" in u.original_image_ids)
        self.assertEqual(unit.active_image_ids, ("a1", "a3"))
        self.assertEqual(report["aggregation_unit"], "physical-analysis-identity")
        self.assertFalse(report["patient_independence_claim"])
        self.assertFalse(report["weighting_policy_selected"])
        self.assertFalse(report["production_freeze"])

    def test_policy_required_invalid_and_unresolved_independence(self):
        with self.assertRaises(TypeError):
            build_dependence_plan(grouping())
        for policy in (None, "", "patient-independent", []):
            self.assert_code("resampling_policy_required", build_dependence_plan, grouping(),
                             resampling_unit_policy=policy)
        self.assert_code("unresolved_dependence", build_dependence_plan, grouping(),
                         resampling_unit_policy=PHYSICAL_UNITS)
        singleton = build_conservative_groups(resolve_analysis((row("one", "lesion"),), ()), ())
        report = dependence_report(build_dependence_plan(singleton, resampling_unit_policy=PHYSICAL_UNITS))
        self.assertFalse(report["physical_independence_claim"])
        self.assertFalse(report["approved_production_policy"])

    def test_anchor_omission_inactive_and_plan_tampering_rejected(self):
        source = grouping()
        self.assert_code("invalid_conservative_grouping", build_dependence_plan,
                         replace(source, records=source.records[:-1]), resampling_unit_policy=CONSERVATIVE_CLUSTERS)
        # An entire absent-active unit cannot be repaired by silently dropping it.
        bad = tuple(replace(r, active=False) if r.original.lesion_id == "f" else r for r in source.records)
        self.assert_code("invalid_conservative_grouping", build_dependence_plan,
                         replace(source, records=bad), resampling_unit_policy=CONSERVATIVE_CLUSTERS)
        plan = self.plan()
        for forged in (replace(plan, physical_units=plan.physical_units[:-1]),
                       replace(plan, clusters=plan.clusters[:-1]),
                       replace(plan, membership_sha256="0" * 64),
                       replace(plan, physical_units=(replace(plan.physical_units[0], label="mel"),) + plan.physical_units[1:])):
            self.assert_code("invalid_dependence_plan", validate_dependence_plan, forged)

    def test_system_aggregation_requires_exact_active_membership(self):
        plan = self.plan()
        good = {u.physical_id: u.active_image_ids for u in plan.physical_units}
        actual = validate_aggregation_membership(plan, {"a": good, "b": {k: reversed(v) for k, v in good.items()}})
        self.assertEqual(actual["physical_units"], 6)
        unit = next(u for u in plan.physical_units if "a1" in u.original_image_ids)
        for wrong in (unit.active_image_ids + ("a1",), unit.active_image_ids + ("a2",), ("a1",), ("b1",)):
            bad = {**good, unit.physical_id: wrong}
            self.assert_code("aggregation_membership_mismatch", validate_aggregation_membership, plan, {"a": good, "b": bad})
        missing = dict(good)
        missing.pop(unit.physical_id)
        self.assert_code("aggregation_membership_mismatch", validate_aggregation_membership, plan, {"a": missing})
        by_cluster = {c.cluster_id: c.physical_ids for c in plan.clusters}
        self.assert_code("aggregation_membership_mismatch", validate_aggregation_membership, plan, {"a": by_cluster})

    def test_unequal_cluster_sizes_pairing_and_determinism(self):
        plan = self.plan(full=True)
        a = tuple(u.physical_id for u in plan.physical_units)
        b = tuple(reversed(a))
        result = paired_class_stratified_cluster_blueprint(plan, a, b, seed=42, replicate_count=100)
        again = paired_class_stratified_cluster_blueprint(plan, a, b, seed=42, replicate_count=100)
        self.assertEqual(result, again)
        self.assertNotEqual(result.draws, paired_class_stratified_cluster_blueprint(plan, a, b, seed=43, replicate_count=100).draws)
        self.assertEqual(result.scope, "unpartitioned-synthetic-contract-population")
        self.assertEqual(result.observed_classes, ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))
        self.assertEqual(result.missing_classes, ())
        self.assertTrue(result.seven_class_membership_complete)
        self.assertFalse(result.physical_class_counts_fixed)
        self.assertFalse(result.approved_production_policy)
        self.assertFalse(result.production_freeze)
        lengths = set()
        for draw in result.draws:
            multiplicities = dict(draw.physical_multiplicity)
            self.assertEqual(set(multiplicities), set(a))
            self.assertEqual(tuple(a[i] for i in draw.indices_a), tuple(b[i] for i in draw.indices_b))
            self.assertEqual(len(draw.selected_cluster_ids), 9)
            lengths.add(len(draw.indices_a))
            for cluster in plan.clusters:
                count = draw.selected_cluster_ids.count(cluster.cluster_id)
                self.assertEqual({multiplicities[key] for key in cluster.physical_ids}, {count})
            for dx in ("nv", "bkl"):
                self.assertTrue(any(multiplicities[u.physical_id] > 0 for u in plan.physical_units if u.label == dx))
        self.assertGreater(len(lengths), 1)  # Fixed cluster draws do not imply fixed lesion denominators.

    def test_paired_coverage_and_caller_configuration(self):
        plan = self.plan(full=True)
        ids = tuple(u.physical_id for u in plan.physical_units)
        for bad in (ids[:-1], ids + (ids[0],), ids[:-1] + ("unknown",)):
            self.assert_code("paired_coverage_mismatch", paired_class_stratified_cluster_blueprint,
                             plan, ids, bad, seed=1, replicate_count=3)
        with self.assertRaises(TypeError):
            paired_class_stratified_cluster_blueprint(plan, ids, ids)
        for seed, count in ((True, 1), (-1, 1), (0, False), (0, 0), (1.5, 3), (2, 1.5)):
            self.assert_code("invalid_resampling_configuration", paired_class_stratified_cluster_blueprint,
                             plan, ids, ids, seed=seed, replicate_count=count)

    def test_mixed_label_stratification_is_not_majority_labeled(self):
        plan = build_dependence_plan(grouping(mixed=True), resampling_unit_policy=CONSERVATIVE_CLUSTERS)
        self.assertEqual(len(dependence_report(plan)["mixed_label_clusters"]), 1)
        ids = tuple(u.physical_id for u in plan.physical_units)
        self.assert_code("mixed_label_cluster_stratification", paired_class_stratified_cluster_blueprint,
                         plan, ids, ids, seed=1, replicate_count=3)

    def test_partition_membership_leakage_and_explicit_scope(self):
        source = grouping(full=True)
        first = next(r.split_group_id for r in source.records if r.original.image_id == "a1")
        train = tuple(r for r in source.records if r.split_group_id == first)
        test = tuple(r for r in source.records if r.split_group_id != first)
        partitions = {"train": train, "internal_test": test}
        plan = build_dependence_plan(source, resampling_unit_policy=CONSERVATIVE_CLUSTERS, partitions=partitions)
        ids = tuple(u.physical_id for u in plan.physical_units)
        self.assert_code("partition_scope_required", paired_class_stratified_cluster_blueprint,
                         plan, ids, ids, seed=1, replicate_count=3)
        self.assert_code("cross_split_overlap", build_dependence_plan, source,
                         resampling_unit_policy=CONSERVATIVE_CLUSTERS,
                         partitions={"train": train[1:], "internal_test": test + train[:1]})
        self.assert_code("membership_mismatch", build_dependence_plan, source,
                         resampling_unit_policy=CONSERVATIVE_CLUSTERS, partitions={"train": train, "internal_test": test[:-1]})
        self.assert_code("membership_mismatch", build_dependence_plan, source,
                         resampling_unit_policy=CONSERVATIVE_CLUSTERS, partitions={"train": train + train[:1], "internal_test": test})
        selected_ids = tuple(sorted({r.analysis_lesion_id for r in test}))
        result = paired_class_stratified_cluster_blueprint(plan, selected_ids, reversed(selected_ids),
                                                         partition="internal_test", seed=7, replicate_count=5)
        self.assertEqual(result.physical_ids, selected_ids)
        self.assertEqual(result.scope, "partition:internal_test")
        self.assert_code("paired_coverage_mismatch", paired_class_stratified_cluster_blueprint,
                         plan, ids, ids, partition="internal_test", seed=7, replicate_count=5)

    def test_all_seven_class_membership_reported_without_production_approval(self):
        from dermdetect.data.schemas import CLASS_ORDER
        source = build_conservative_groups(resolve_analysis(tuple(row(dx, dx, dx) for dx in CLASS_ORDER), ()), ())
        plan = build_dependence_plan(source, resampling_unit_policy=PHYSICAL_UNITS)
        ids = tuple(u.physical_id for u in plan.physical_units)
        result = paired_class_stratified_cluster_blueprint(plan, ids, ids, seed=0, replicate_count=2)
        self.assertEqual(result.observed_classes, CLASS_ORDER)
        self.assertEqual(result.missing_classes, ())
        self.assertTrue(result.seven_class_membership_complete)
        self.assertFalse(result.approved_production_policy)

    def test_missing_class_strata_and_malformed_iterables_fail_closed(self):
        plan = self.plan()
        ids = tuple(u.physical_id for u in plan.physical_units)
        self.assert_code("missing_class_strata", paired_class_stratified_cluster_blueprint,
                         plan, ids, ids, seed=0, replicate_count=2)
        self.assert_code("paired_coverage_mismatch", paired_class_stratified_cluster_blueprint,
                         plan, None, ids, seed=0, replicate_count=2)
        self.assert_code("paired_coverage_mismatch", paired_class_stratified_cluster_blueprint,
                         plan, ids[0], ids, seed=0, replicate_count=2)
        good = {u.physical_id: u.active_image_ids for u in plan.physical_units}
        bad = dict(good)
        bad[ids[0]] = None
        self.assert_code("aggregation_membership_mismatch", validate_aggregation_membership, plan, {"a": bad})
        self.assert_code("invalid_conservative_grouping", validate_dependence_plan, replace(plan, grouping=None))
        self.assert_code("invalid_partitions", build_dependence_plan, grouping(),
                         resampling_unit_policy=CONSERVATIVE_CLUSTERS, partitions={"train": None})


if __name__ == "__main__":
    unittest.main()
