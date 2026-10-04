"""Exact quota/whole-component solver tests on invented metadata only."""
from dataclasses import replace
import hashlib
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dermdetect.data.analysis import resolve_analysis
from dermdetect.data.conservative_groups import ReviewedUncertainty, build_conservative_groups
from dermdetect.data.constrained_allocation import (
    allocate_constrained_groups, propose_constrained_groups, verify_constrained_allocation)
from dermdetect.data.group_allocation import allocate_groups, group_quotas, SPLITS
from dermdetect.data.schemas import CLASS_ORDER, ContractError, ManifestRecord


def record(image,dx,lesion=None,payload=None):
    return ManifestRecord(image,lesion or image,dx,"invented","v1",image+".txt",
                          hashlib.sha256((payload or image).encode()).hexdigest())

def link(a,b):
    return ReviewedUncertainty(a,b,"invented full-view evidence","invented reviewer",
        "invented unresolved identity","2026-10-02",full_view_inspected=True)


class ConstrainedAllocationTests(unittest.TestCase):
    def setUp(self):
        rows=[record(f"{dx}_{i:02}",dx) for dx in CLASS_ORDER for i in range(10)]
        rows.extend([record("nv_00_view","nv","nv_00"),record("nv_00_copy","nv","nv_00","nv_00")])
        self.physical=resolve_analysis(rows,[])
        self.baseline=allocate_groups(self.physical)

    def assert_code(self,code,call):
        with self.assertRaises(ContractError) as caught:call()
        self.assertEqual(caught.exception.code,code)

    def test_no_uncertainty_preserves_baseline_and_exact_representatives(self):
        grouping=build_conservative_groups(self.physical,[])
        result,report,search=propose_constrained_groups(grouping)
        self.assertEqual({s:[r.original.image_id for r in rows] for s,rows in result.items()},
                         {s:[r.original.image_id for r in rows] for s,rows in self.baseline.items()})
        self.assertTrue(report["zero_overlap"])
        self.assertEqual(report["physical_units_moved_from_fixed_baseline"],0)
        self.assertEqual(report["active_views"],71)
        self.assertEqual(search["nontrivial_components"],0)
        self.assertFalse(report["production_freeze"])

    def test_cross_partition_mixed_chain_is_colocated_without_relabel_or_identity_merge(self):
        a=next(r for r in self.baseline["train"] if r.original.dx=="nv")
        b=next(r for r in self.baseline["validation"] if r.original.dx=="nv")
        c=next(r for r in self.baseline["internal_test"] if r.original.dx=="bcc")
        grouping=build_conservative_groups(self.physical,[link(a.original.image_id,b.original.image_id),
                                                              link(b.original.image_id,c.original.image_id)])
        assigned,report,search=propose_constrained_groups(grouping)
        owners={r.original.image_id:name for name,rows in assigned.items() for r in rows}
        self.assertEqual(len({owners[r.original.image_id] for r in (a,b,c)}),1)
        for before,after in zip(self.physical,grouping.records):
            self.assertEqual(replace(after,split_group_id=after.analysis_lesion_id),before)
        self.assertEqual(len(report["mixed_label_groups"]),1)
        self.assertGreater(report["physical_units_moved_from_fixed_baseline"],0)
        for dx in CLASS_ORDER:
            self.assertEqual(tuple(report["counts"][s][dx]["physical_units"] for s in SPLITS),group_quotas(10))
        self.assertEqual(search["nontrivial_components"],1)

    def test_input_and_link_order_do_not_change_assignment(self):
        edges=[link("nv_00","bcc_00"),link("bcc_00","bcc_01"),link("mel_00","nv_01")]
        a=allocate_constrained_groups(build_conservative_groups(self.physical,edges))
        b=allocate_constrained_groups(build_conservative_groups(reversed(self.physical),[
            replace(e,left=e.right,right=e.left) for e in reversed(edges)]))
        self.assertEqual(a,b)

    def test_exhaustive_small_oracle_distinguishes_feasible_and_infeasible_cluster_sizes(self):
        # Enumerate all possible nv cluster placements independently of solver.
        for sizes in ((2,2),(3,2),(5,5),(2,2,2,2,2),(7,2,1)):
            with self.subTest(sizes=sizes):
                ids=[f"nv_{i:02}" for i in range(10)];offset=0;edges=[]
                for size in sizes:
                    group=ids[offset:offset+size];offset+=size
                    edges.extend(link(group[0],i) for i in group[1:])
                clusters=list(sizes)+[1]*(10-sum(sizes))
                quotas=group_quotas(10)
                possible=any(tuple(sum(size for size,s in zip(clusters,owners) if s==p) for p in range(3))==quotas
                             for owners in itertools.product(range(3),repeat=len(clusters)))
                grouping=build_conservative_groups(self.physical,edges)
                if possible:
                    self.assertTrue(verify_constrained_allocation(grouping,allocate_constrained_groups(grouping))["zero_overlap"])
                else:
                    self.assert_code("infeasible_constrained_allocation",lambda:allocate_constrained_groups(grouping))

    def test_search_budget_is_not_reported_as_proved_infeasibility(self):
        grouping=build_conservative_groups(self.physical,[link("nv_00","nv_01")])
        self.assert_code("allocation_search_limit",lambda:allocate_constrained_groups(grouping,search_node_limit=1))
        for value in (0,-1,True,"100"):
            self.assert_code("invalid_search_limit",lambda:allocate_constrained_groups(grouping,search_node_limit=value))

    def test_feasible_mixed_problem_backtracks_from_baseline_preference(self):
        def representatives(rows,dx):
            result={r.analysis_lesion_id:r.original.image_id for r in rows if r.original.dx==dx}
            return sorted(result.values())
        nv=representatives(self.physical,'nv');bcc=representatives(self.physical,'bcc')
        flex_nv=representatives(self.baseline['train'],'nv')[:2]
        flex_bcc=representatives(self.baseline['train'],'bcc')[:2]
        flexible=flex_nv+flex_bcc
        remaining_nv=[i for i in nv if i not in flex_nv]
        remaining_bcc=[i for i in bcc if i not in flex_bcc]
        components=[flexible,remaining_nv[:3],remaining_nv[3:6],remaining_bcc[:3],remaining_bcc[3:6]]
        edges=[link(group[0],i) for group in components for i in group[1:]]
        grouping=build_conservative_groups(self.physical,edges)
        assigned,report,search=propose_constrained_groups(grouping)
        # NV/BCC each have 7/2/1 quotas. Their two three-unit components
        # require train, so the flexible 2NV+2BCC component must use validation.
        owners={r.original.image_id:s for s,rows in assigned.items() for r in rows}
        self.assertEqual({owners[i] for i in flexible},{'validation'})
        self.assertGreater(search['failed_states'],0)
        self.assertTrue(report['zero_overlap'])

    def test_verifier_rejects_missing_changed_duplicate_and_crossing_rows(self):
        grouping=build_conservative_groups(self.physical,[link("nv_00","nv_01")])
        assigned=allocate_constrained_groups(grouping)
        changed={s:list(rows) for s,rows in assigned.items()};changed["train"].pop()
        self.assert_code("membership_mismatch",lambda:verify_constrained_allocation(grouping,changed))
        changed={s:list(rows) for s,rows in assigned.items()};changed["train"].append(changed["train"][0])
        self.assert_code("membership_mismatch",lambda:verify_constrained_allocation(grouping,changed))
        changed={s:list(rows) for s,rows in assigned.items()};changed["train"][0]=replace(changed["train"][0],active=False)
        self.assert_code("membership_mismatch",lambda:verify_constrained_allocation(grouping,changed))
        origin=next(s for s,rows in assigned.items() if any(r.original.image_id=="nv_00" for r in rows))
        target=next(s for s in SPLITS if s!=origin)
        changed={s:list(rows) for s,rows in assigned.items()}
        move=next(r for r in changed[origin] if r.original.image_id=="nv_00")
        changed[origin].remove(move);changed[target].append(move)
        self.assert_code("cross_split_overlap",lambda:verify_constrained_allocation(grouping,changed))


if __name__=="__main__":unittest.main()
