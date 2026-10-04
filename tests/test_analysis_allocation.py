"""Invented lesion identities and byte hashes; no real images or assignments."""
from dataclasses import replace
import hashlib
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.analysis import CandidateDecision, resolve_analysis, validate_analysis
from dermdetect.data.group_allocation import allocate_groups, group_quotas, verify_allocation
from dermdetect.data.image_integrity import dhash_from_luminance, near_duplicate_pairs
from dermdetect.data.schemas import CLASS_ORDER, ContractError, ManifestRecord


def record(image, lesion, dx="nv", payload=None):
    digest = hashlib.sha256((payload or image).encode()).hexdigest()
    return ManifestRecord(image, lesion, dx, "invented", "v1", f"invented/{image}.jpg", digest)


def decision(a, b, value="same_lesion"):
    return CandidateDecision(a, b, "invented visual review", value, "invented reviewer", "invented rationale")


class AnalysisTests(unittest.TestCase):
    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_AAB_uses_each_distinct_view_once_preserving_inventory(self):
        source = [record("a", "lesion", payload="A"), record("copy", "lesion", payload="A"),
                  record("b", "lesion", payload="B")]
        rows = resolve_analysis(source, [])
        self.assertEqual({r.original for r in rows}, set(source))
        active = [r for r in rows if r.active]
        self.assertEqual([r.original.image_id for r in active], ["a", "b"])
        copy = next(r for r in rows if not r.active)
        self.assertEqual((copy.representative_image_id, copy.inactive_reason), ("a", "repeated_exact_view"))
        logits = {"a": 0, "copy": 0, "b": 3}
        self.assertEqual(sum(logits[r.original.image_id] for r in active) / len(active), 1.5)
        self.assertEqual(sum(logits[r.original.image_id] for r in rows) / len(rows), 1)
        self.assertEqual(rows, resolve_analysis(reversed(source), []))

    def test_transitive_connections_join_whole_original_lesions(self):
        source = [record("a1", "a"), record("a2", "a"), record("b", "b"), record("c", "c")]
        reviews = [decision("a1", "b"), decision("b", "c")]
        rows = resolve_analysis(source, reviews)
        self.assertEqual(len({r.analysis_lesion_id for r in rows}), 1)
        self.assertEqual(rows, resolve_analysis(source, reversed(reviews)))
        self.assertEqual(len({r.original.lesion_id for r in rows}), 3)

    def test_different_lesions_remain_separate(self):
        rows = resolve_analysis([record("a", "a"), record("b", "b")],
                                [decision("a", "b", "different_lesions")])
        self.assertEqual(len({r.analysis_lesion_id for r in rows}), 2)

    def test_distinct_decision_conflicting_with_transitive_identity_stops_in_any_order(self):
        rows = [record("a", "a"), record("b", "b"), record("c", "c")]
        reviews = [decision("a", "c", "different_lesions"), decision("a", "b"), decision("b", "c")]
        self.assert_code("contradictory_review", lambda: resolve_analysis(rows, reviews))
        self.assert_code("contradictory_review", lambda: resolve_analysis(rows, reversed(reviews)))

    def test_pending_missing_evidence_unknown_repeated_candidates_stop(self):
        rows = [record("a", "a"), record("b", "b")]
        self.assert_code("unresolved_candidate", lambda: resolve_analysis(rows, [CandidateDecision("a", "b", "paper")]))
        self.assert_code("missing_review_evidence", lambda: resolve_analysis(rows, [replace(decision("a", "b"), reviewer="")]))
        self.assert_code("invalid_candidate", lambda: resolve_analysis(rows, [decision("a", "absent")]))
        self.assert_code("repeated_candidate", lambda: resolve_analysis(rows, [decision("a", "b"), decision("b", "a")]))

    def test_cross_label_identity_is_never_automatically_merged(self):
        self.assert_code("component_label_conflict", lambda: resolve_analysis(
            [record("a", "a"), record("b", "b", "bkl")], [decision("a", "b")]))

    def test_cross_id_exact_hash_needs_explicit_identity_review(self):
        rows = [record("a", "a", payload="same"), record("b", "b", payload="same")]
        self.assert_code("unresolved_exact_identity", lambda: resolve_analysis(rows, []))
        self.assert_code("contradictory_review", lambda: resolve_analysis(rows, [decision("a", "b", "different_lesions")]))
        resolved = resolve_analysis(rows, [decision("a", "b")])
        self.assertEqual(sum(r.active for r in resolved), 1)

    def test_cannot_reactivate_duplicate_or_drop_distinct_view(self):
        rows = resolve_analysis([record("a", "lesion", payload="same"), record("b", "lesion", payload="same")], [])
        self.assert_code("invalid_participation", lambda: validate_analysis([rows[0], replace(rows[1], active=True)]))
        self.assert_code("invalid_participation", lambda: validate_analysis([replace(rows[0], active=False), rows[1]]))

    def test_component_ids_and_split_units_are_checked(self):
        rows = resolve_analysis([record("a", "a")], [])
        self.assert_code("invalid_analysis_group", lambda: validate_analysis([replace(rows[0], split_group_id="other")]))
        self.assert_code("invalid_analysis_group", lambda: validate_analysis([
            replace(rows[0], split_group_id="forged", analysis_lesion_id="forged")]))


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.source = [record(f"{dx}_{i}_{view}", f"{dx}_{i}", dx)
                       for dx in CLASS_ORDER for i in range(10) for view in range(1 + i % 4)]
        self.source.append(record("nv_0_copy", "nv_0", "nv", "nv_0_0"))
        self.rows = resolve_analysis(self.source, [])

    def assert_code(self, code, call):
        with self.assertRaises(ContractError) as caught:
            call()
        self.assertEqual(caught.exception.code, code)

    def test_exact_largest_remainder_and_fixed_ties(self):
        self.assertEqual(group_quotas(10), (7, 2, 1))
        self.assertEqual(group_quotas(73), (51, 11, 11))
        self.assertEqual(group_quotas(98), (68, 15, 15))
        self.assertEqual(group_quotas(5403), (3782, 811, 810))
        for count in range(7, 501):
            quotas = group_quotas(count)
            self.assertEqual(sum(quotas), count)
            self.assertTrue(all(abs(q - count * p / 100) < 1 for q, p in zip(quotas, (70, 15, 15))))

    def test_allocator_preserves_rows_duplicates_components_and_quotas(self):
        assigned = allocate_groups(self.rows)
        report = verify_allocation(self.rows, assigned)
        self.assertTrue(report["zero_overlap"])
        self.assertFalse(report["production_freeze"])
        self.assertFalse(report["rare_class_adequacy_claim"])
        for dx in CLASS_ORDER:
            self.assertEqual(tuple(report["counts"][name][dx]["groups"] for name in assigned), (7, 2, 1))
        self.assertEqual(sum(r.active for members in assigned.values() for r in members), len(self.source) - 1)

    def test_reordering_and_extra_views_do_not_change_group_assignment(self):
        original = allocate_groups(self.rows)
        def owners(assigned):
            return {r.original.lesion_id: name for name, members in assigned.items() for r in members}
        self.assertEqual(original, allocate_groups(reversed(self.rows)))
        changed = resolve_analysis(self.source + [record("nv_0_extra", "nv_0")], [])
        self.assertEqual(owners(original), owners(allocate_groups(changed)))

    def test_confirmed_cross_id_component_cannot_leak(self):
        reviewed = resolve_analysis(self.source, [decision("nv_0_0", "nv_1_0")])
        assigned = allocate_groups(reviewed)
        owners = {r.original.lesion_id: name for name, members in assigned.items() for r in members}
        self.assertEqual(owners["nv_0"], owners["nv_1"])
        self.assertEqual(len({r.analysis_lesion_id for r in reviewed if r.original.dx == "nv"}), 9)

    def test_omission_alteration_and_cross_partition_view_move_fail(self):
        assigned = allocate_groups(self.rows)
        changed = dict(assigned)
        changed["train"] = changed["train"][1:]
        self.assert_code("membership_mismatch", lambda: verify_allocation(self.rows, changed))
        changed = dict(assigned)
        changed["train"] = (replace(changed["train"][0], active=False),) + changed["train"][1:]
        self.assert_code("membership_mismatch", lambda: verify_allocation(self.rows, changed))
        # Move one row of an intact multi-view group to another partition.
        multi = next(r for r in assigned["train"] if sum(x.split_group_id == r.split_group_id for x in assigned["train"]) > 1)
        changed = dict(assigned)
        changed["train"] = tuple(r for r in changed["train"] if r != multi)
        changed["validation"] += (multi,)
        self.assert_code("cross_split_overlap", lambda: verify_allocation(self.rows, changed))

    def test_missing_class_and_infeasible_quota_stop(self):
        self.assert_code("missing_class", lambda: allocate_groups([r for r in self.rows if r.original.dx != "df"]))
        self.assert_code("missing_partition_class", lambda: group_quotas(3))


class CandidateHashTests(unittest.TestCase):
    def test_luminance_order_and_ties_have_known_bits(self):
        self.assertEqual(dhash_from_luminance(bytes(range(9)) * 8), 0)
        self.assertEqual(dhash_from_luminance(bytes(reversed(range(9))) * 8), 2**64 - 1)
        self.assertEqual(dhash_from_luminance(bytes([5] * 72)), 0)
        with self.assertRaises(ContractError):
            dhash_from_luminance(bytes(71))

    def test_band_index_equals_exhaustive_distance_screen(self):
        rng = random.Random(42)
        hashes = {f"invented_{i:03}": rng.getrandbits(64) for i in range(60)}
        for i in range(20):
            origin = hashes[f"invented_{i:03}"]
            flips = sum(1 << bit for bit in rng.sample(range(64), i % 6))
            hashes[f"mutated_{i:03}"] = origin ^ flips
        expected = tuple((a, b) for a in sorted(hashes) for b in sorted(hashes)
                         if a < b and (hashes[a] ^ hashes[b]).bit_count() <= 4)
        self.assertEqual(near_duplicate_pairs(hashes), expected)
        self.assertEqual(near_duplicate_pairs(dict(reversed(list(hashes.items())))), expected)

    def test_four_altered_bands_still_find_unchanged_fifth_band(self):
        value = sum(1 << bit for bit in (0, 13, 26, 39))
        self.assertEqual(near_duplicate_pairs({"a": 0, "b": value}), (("a", "b"),))
        self.assertEqual(near_duplicate_pairs({"a": 0, "b": value | (1 << 52)}), ())

    def test_invalid_fingerprint_fails_and_queue_is_never_silently_truncated(self):
        from unittest.mock import patch
        for value in (-1, 2**64, True):
            with self.assertRaises(ContractError):
                near_duplicate_pairs({"a": value})
        with patch("dermdetect.data.image_integrity.MAX_CANDIDATES", 1):
            with self.assertRaises(ContractError) as caught:
                near_duplicate_pairs({"a": 0, "b": 0, "c": 0})
            self.assertEqual(caught.exception.code, "candidate_limit_exceeded")


if __name__ == "__main__":
    unittest.main()
