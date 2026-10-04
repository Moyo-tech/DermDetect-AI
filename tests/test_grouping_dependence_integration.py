"""Cross-module contracts using invented metadata/reviews only; no file I/O."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.analysis import resolve_analysis
from dermdetect.data.conservative_groups import (
    ReviewedUncertainty, allocate_conservative_groups, build_conservative_groups,
    verify_conservative_allocation,
)
from dermdetect.data.dependence import (
    CONSERVATIVE_CLUSTERS, PHYSICAL_UNITS, build_dependence_plan,
    paired_class_stratified_cluster_blueprint, validate_aggregation_membership,
)
from dermdetect.data.group_allocation import allocate_groups
from dermdetect.data.schemas import CLASS_ORDER, ContractError, ManifestRecord


def invented_record(image, lesion, dx, payload=None):
    return ManifestRecord(image, lesion, dx, "invented", "v1", "invented/" + image,
                          hashlib.sha256((payload or image).encode()).hexdigest())


def inspected_uncertainty(left, right):
    return ReviewedUncertainty(left, right, "invented evidence", "invented reviewer",
                               "invented unresolved identity", "2026-10-02",
                               full_view_inspected=True)


class GroupingDependenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        source = [invented_record(f"{dx}_{i:02}_{view}", f"{dx}_{i:02}", dx)
                  for dx in CLASS_ORDER for i in range(10) for view in range(2)]
        source.append(invented_record("nv_00_copy", "nv_00", "nv", "nv_00_0"))
        self.physical = resolve_analysis(source, [])
        self.fixed = allocate_groups(self.physical)

    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_active_physical_membership_and_pairing_survive_split_only_colocation(self):
        # Select a branch condition from a fixed invented assignment, not a
        # research seed or an outcome. This exercises already-colocated links.
        train_nv = sorted({r.analysis_lesion_id for r in self.fixed["train"]
                           if r.original.dx == "nv"})
        image = {identity: min(r.original.image_id for r in self.physical
                               if r.analysis_lesion_id == identity)
                 for identity in train_nv[:2]}
        grouped = build_conservative_groups(self.physical, [inspected_uncertainty(
            image[train_nv[0]], image[train_nv[1]])])
        assigned = allocate_conservative_groups(grouped)
        report = verify_conservative_allocation(grouped, assigned)
        self.assertTrue(report["matches_fixed_physical_allocation"])
        self.assertEqual(report["counts"]["train"]["nv"]["physical_units"], 7)
        self.assertEqual(report["counts"]["train"]["nv"]["split_groups"], 6)
        self.assertFalse(report["production_freeze"])
        self.assertEqual(sum(r.active for r in grouped.records), 140)
        for original, changed in zip(self.physical, grouped.records):
            self.assertEqual(replace(changed, split_group_id=changed.analysis_lesion_id), original)

        plan = build_dependence_plan(grouped, resampling_unit_policy=CONSERVATIVE_CLUSTERS,
                                     partitions=assigned)
        units = tuple(u for u in plan.physical_units
                      if u.physical_id in {r.analysis_lesion_id for r in assigned["train"]})
        membership = {u.physical_id: u.active_image_ids for u in units}
        validation = validate_aggregation_membership(plan, {"A": membership, "B": membership},
                                                     partition="train")
        self.assertEqual(validation["physical_units"], 49)
        self.assertFalse(validation["weighting_policy_selected"])
        a = tuple(membership)
        b = tuple(reversed(a))
        blueprint = paired_class_stratified_cluster_blueprint(
            plan, a, b, seed=123, replicate_count=12, partition="train")
        for draw in blueprint.draws:
            count = dict(draw.physical_multiplicity)
            self.assertEqual(count[train_nv[0]], count[train_nv[1]])
            self.assertEqual(tuple(a[i] for i in draw.indices_a),
                             tuple(b[i] for i in draw.indices_b))
            self.assertEqual(sum(count.values()), len(draw.indices_a))
        self.assertFalse(blueprint.approved_production_policy)
        self.assertFalse(blueprint.production_freeze)

        physical_scope = {u.physical_id: u.active_image_ids for u in plan.physical_units}
        copied = {**physical_scope}
        unit = next(u for u in plan.physical_units if "nv_00_copy" in u.original_image_ids)
        copied[unit.physical_id] += ("nv_00_copy",)
        self.assert_code("aggregation_membership_mismatch", lambda: validate_aggregation_membership(
            build_dependence_plan(grouped, resampling_unit_policy=CONSERVATIVE_CLUSTERS),
            {"A": physical_scope, "B": copied}))
        self.assert_code("unresolved_dependence", lambda: build_dependence_plan(
            grouped, resampling_unit_policy=PHYSICAL_UNITS))

    def test_mixed_label_uncertainty_chain_preserves_membership_and_blocks_unapproved_policies(self):
        bcc=sorted((r for r in self.physical if r.original.dx=="bcc"),
                   key=lambda r:r.original.image_id)
        a=bcc[0]
        b=next(r for r in bcc if r.analysis_lesion_id!=a.analysis_lesion_id)
        c=next(r for r in self.physical if r.original.dx=="akiec")
        # The same-label edge belongs to the full mixed-label component.
        grouped=build_conservative_groups(self.physical,[
            inspected_uncertainty(a.original.image_id,b.original.image_id),
            inspected_uncertainty(b.original.image_id,c.original.image_id)])
        by_image={r.original.image_id:r for r in grouped.records}
        linked=[by_image[r.original.image_id] for r in (a,b,c)]
        self.assertEqual(len({r.split_group_id for r in linked}),1)
        self.assertEqual(len({r.analysis_lesion_id for r in linked}),3)
        self.assertEqual({r.original.dx for r in linked},{"bcc","akiec"})
        for original,changed in zip(self.physical,grouped.records):
            self.assertEqual(replace(changed,split_group_id=changed.analysis_lesion_id),original)
        self.assert_code("mixed_label_split_group",lambda:allocate_conservative_groups(grouped))
        plan=build_dependence_plan(grouped,resampling_unit_policy=CONSERVATIVE_CLUSTERS)
        membership={u.physical_id:u.active_image_ids for u in plan.physical_units}
        result=validate_aggregation_membership(plan,{"A":membership,"B":membership})
        self.assertFalse(result["weighting_policy_selected"])
        ids=tuple(membership)
        self.assert_code("mixed_label_cluster_stratification",lambda:paired_class_stratified_cluster_blueprint(
            plan,ids,tuple(reversed(ids)),seed=123,replicate_count=2))
        self.assert_code("unresolved_dependence",lambda:build_dependence_plan(
            grouped,resampling_unit_policy=PHYSICAL_UNITS))

    def test_cross_partition_uncertainty_stops_assignment_and_dependence_scope(self):
        a = next(r for r in self.fixed["train"] if r.original.dx == "nv")
        b = next(r for r in self.fixed["validation"] if r.original.dx == "nv")
        grouped = build_conservative_groups(self.physical, [inspected_uncertainty(
            a.original.image_id, b.original.image_id)])
        self.assert_code("allocation_policy_required", lambda: allocate_conservative_groups(grouped))
        by_image = {r.original.image_id: r for r in grouped.records}
        leaked = {name: tuple(by_image[r.original.image_id] for r in members)
                  for name, members in self.fixed.items()}
        self.assert_code("cross_split_overlap", lambda: verify_conservative_allocation(grouped, leaked))
        self.assert_code("cross_split_overlap", lambda: build_dependence_plan(
            grouped, resampling_unit_policy=CONSERVATIVE_CLUSTERS, partitions=leaked))


if __name__ == "__main__":
    unittest.main()
