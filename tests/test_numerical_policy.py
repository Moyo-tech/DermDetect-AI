"""Invented numerical observations only; no dataset predictions or outcomes."""
from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.numerical_policy import (CLASS_ORDER, View, training_weights,
    mean_physical_logits, equal_physical_nll, paired_cluster_draws, paired_indices,
    percentile_interval)


def fixture():
    rows = [View(c, c, c, c, c) for c in CLASS_ORDER]
    rows += [View("nv2", "nv", "nv", "nv", "view2"),
             View("nv-copy", "nv", "nv", "nv", "nv", False),
             View("nv-other", "nv-other", "nv", "nv", "other"),
             View("nv-single", "nv-single", "nv-single", "nv", "single")]
    return tuple(rows)


class NumericalPolicyTests(unittest.TestCase):
    def test_class_mass_physical_mass_and_exact_inactive_zero(self):
        rows = fixture(); result = training_weights(rows); w = result["image_weights"]
        self.assertEqual(sum(w.values()), result["active_views"])
        self.assertEqual(w["nv-copy"], 0)
        for c in CLASS_ORDER:
            self.assertEqual(sum(w[v.image_id] for v in rows if v.label == c), Fraction(result["active_views"], 7))
        self.assertEqual(w["nv"] + w["nv2"], w["nv-other"])
        self.assertEqual(w["nv-other"], w["nv-single"])

    def test_mixed_training_colocation_does_not_change_weights_or_labels(self):
        rows = fixture(); changed = tuple(replace(v, cluster_id="nv") if v.label == "bcc" else v for v in rows)
        self.assertEqual(training_weights(rows), training_weights(changed))

    def test_invalid_membership_and_missing_class_stop(self):
        rows = fixture()
        for bad in (rows + (rows[0],), tuple(v for v in rows if v.label != "df"),
                    tuple(replace(v, active=True) if v.image_id == "nv-copy" else v for v in rows),
                    tuple(replace(v, label="mel") if v.image_id == "nv2" else v for v in rows),
                    tuple(replace(v, active=False) if v.label == "df" else v for v in rows)):
            with self.assertRaises(ValueError): training_weights(bad)

    def test_logit_mean_uses_active_views_and_keeps_linked_units_separate(self):
        rows = fixture(); logits = {v.image_id: [0.] * 7 for v in rows if v.active}
        logits["nv"] = [2.] * 7; logits["nv2"] = [4.] * 7
        reduced = mean_physical_logits(rows, logits)
        self.assertEqual(reduced["nv"], (3.,)*7)
        self.assertEqual(reduced["nv-other"], (0.,)*7)
        with self.assertRaises(ValueError): mean_physical_logits(rows, {**logits,"nv-copy":[99.]*7})
        with self.assertRaises(ValueError): mean_physical_logits(rows, {**logits,"nv":[float("nan")]*7})

    def test_calibration_nll_equal_units_stable_and_finite(self):
        logits = {"one": (1000.,)*7, "two": (0.,)*7}
        self.assertAlmostEqual(equal_physical_nll(logits, {"one":"nv","two":"bcc"}, 1.), math.log(7))
        for temp in (0., -1., float("nan"), True):
            with self.assertRaises(ValueError): equal_physical_nll(logits, {"one":"nv","two":"bcc"}, temp)
        with self.assertRaises(ValueError): equal_physical_nll(logits, {"one":"nv"}, 1.)

    def test_paired_groups_unequal_denominators_strata_and_row_order(self):
        rows = fixture(); draws = list(paired_cluster_draws(rows, partition="validation", replicate_count=50))
        self.assertEqual(draws, list(paired_cluster_draws(tuple(reversed(rows)),partition="validation",replicate_count=50)))
        lengths = set()
        for selected, multiplicity in draws:
            self.assertEqual(multiplicity["nv"], multiplicity["nv-other"])
            self.assertEqual(len(selected), 8)
            a = tuple(multiplicity); b = tuple(reversed(a)); ia, ib = paired_indices(multiplicity,a,b)
            self.assertEqual(tuple(a[i] for i in ia),tuple(b[i] for i in ib)); lengths.add(len(ia))
            self.assertTrue(all(sum(multiplicity[v.physical_id] for v in rows if v.label == c) > 0 for c in CLASS_ORDER))
        self.assertGreater(len(lengths),1)

    def test_mixed_evaluation_and_train_bootstrap_refused(self):
        mixed = tuple(replace(v, cluster_id="nv") if v.label == "bcc" else v for v in fixture())
        with self.assertRaises(ValueError): list(paired_cluster_draws(mixed, partition="validation"))
        with self.assertRaises(ValueError): list(paired_cluster_draws(fixture(), partition="train"))

    def test_quantiles_and_no_nonfinite_replica_dropping(self):
        self.assertEqual(percentile_interval([0.,100.]),(2.5,97.5))
        for values in ([],[1.],[1.,float("nan")]):
            with self.assertRaises(ValueError): percentile_interval(values)
        with self.assertRaises(ValueError): paired_indices({"a":1},["a"],["missing"])


if __name__ == "__main__": unittest.main()
