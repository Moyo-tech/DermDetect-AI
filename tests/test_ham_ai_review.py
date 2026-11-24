"""AI-audit contracts on invented identifiers, text views and observations."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import prepare_ham_ai_review as review
from dermdetect.data.manifests import canonical_json_bytes,file_sha256,manifest_csv_bytes,read_manifest,sha256_bytes
from dermdetect.data.analysis import CandidateDecision, resolve_analysis
from dermdetect.data.conservative_groups import ReviewedUncertainty, build_conservative_groups
from dermdetect.data.dependence import CONSERVATIVE_CLUSTERS, build_dependence_plan, validate_aggregation_membership
from dermdetect.data.schemas import ContractError,ManifestRecord
from dermdetect.data.review_policy import RANKED_PROCEDURE,RANKED_STATUS,APPROVAL_REFERENCE
from resolve_ham_analysis import VIEW_TRANSFORM,QUEUE_FIELDS


class AIAuditTests(unittest.TestCase):
    def setUp(self):
        self.make_fixture()

    def make_fixture(self,source_count=2,source_anchor=True,unclear_count=1):
        temporary=tempfile.TemporaryDirectory(prefix="dermdetect-ai-audit-test-")
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        manifest=self.root/"manifest.csv";screen=self.root/"screen";screen.mkdir()
        index=self.root/"visual_index.json";plan_path=self.root/"plan.json"
        source_assignment=self.root/"published18.json";legacy_path=self.root/"source_agent.json"
        records=[];queue=[];views=[];source=[];assigned=[];observations=[];legacy=[]
        def add(name,kind,recommendation,dx_right="nv",anchor=None,source_pair=False):
            left,right="invented_"+name+"_a","invented_"+name+"_b"
            ids=[left,right];payloads={i:sha256_bytes(i.encode()) for i in ids}
            for i,dx in ((left,"nv"),(right,dx_right)):
                lesion=anchor if i==left and anchor else "lesion_"+i
                records.append(ManifestRecord(i,lesion,dx,"invented","v1",i+".jpg",payloads[i]))
            view_file=name+".png";content=("INVENTED TEXT VIEW "+name).encode()
            (self.root/view_file).write_bytes(content);digest=sha256_bytes(content)
            queue.append(dict(zip(QUEUE_FIELDS,(left,right,"0" if "equal_dhash" in kind else "12",kind,"pending","","",""))))
            views.append({"image_ids":ids,"source_payload_sha256":[payloads[i] for i in ids],
                          "view_file":view_file,"view_sha256":digest,"view_transform":VIEW_TRANSFORM})
            assignment={"left":left,"right":right,"view_file":view_file,"view_sha256":digest,"source_payload_sha256":payloads}
            if source_pair:
                source.append(assignment)
                legacy.append({"image_ids":ids,"source_payload_sha256":[payloads[i] for i in ids],
                    "view_sha256":digest,"view_transform":VIEW_TRANSFORM,"agent_recommendation":"same_lesion",
                    "reviewer":"Codex invented source-review fixture","agent_review_date":"2026-10-02",
                    "rationale":"invented matching distinctive structures; not actual images","human_review_status":"not_performed"})
            else:
                assigned.append(assignment)
                observations.append({**assignment,"agent_recommendation":recommendation,
                    "agent_reviewer":"Codex invented fixture","agent_rationale":"invented observable landmark comparison; not actual images",
                    "review_date":"2026-10-02","full_resolution_viewed":True,"human_review_status":"not_performed"})
        for i in range(source_count):
            add(f"source{i}","independent_published_top1000_cross_id|reported_missed_duplicate","same_lesion",
                anchor="anchor_nv" if i==0 and source_anchor else None,source_pair=True)
        for s in range(4):
            for i in range(25):
                kind="independent_published_top1000_cross_id" if s<2 else "equal_dhash64_not_identity"
                add(f"negative{s}_{i:02}",kind,"different_lesions",dx_right="bcc" if s%2 else "nv",
                    anchor="anchor_nv" if s<2 else f"anchor{s}")
        add("new_same","independent_published_top1000_cross_id","same_lesion",anchor="anchor_nv")
        for i in range(unclear_count):
            add("unclear" if unclear_count==1 else f"unclear_{i:03}",
                "independent_published_top1000_cross_id|reported_similar_pair_different_lesions","uncertain")
        manifest.write_bytes(manifest_csv_bytes(records))
        with (screen/"candidates.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=QUEUE_FIELDS,lineterminator="\n");writer.writeheader();writer.writerows(queue)
        bindings={"manifest_sha256":file_sha256(manifest),"candidate_queue_sha256":file_sha256(screen/"candidates.csv")}
        index.write_bytes(canonical_json_bytes({"manifest_sha256":bindings["manifest_sha256"],
            "queue_sha256":"v"*64,"status":"visual_evidence_prepared_not_adjudicated","pairs":views}))
        bindings["visual_index_sha256"]=file_sha256(index)
        report={"manifest_sha256":bindings["manifest_sha256"],"candidates_sha256":bindings["candidate_queue_sha256"],
            "procedure":RANKED_PROCEDURE,"status":RANKED_STATUS,"procedure_approval":"approved_candidate_selection_only",
            "approval_reference":APPROVAL_REFERENCE,"human_decisions_recorded":0,"broad_outside_status":"unadjudicated",
            "production_freeze":False,"split_ready":False,"review_candidates":len(queue),"view_queue_sha256":"v"*64,
            "broad_pairs_outside_approved_review":0,"completeness_limit":"Synthetic fixture only.","source_applicability_limit":"Invented source."}
        (screen/"screen_report.json").write_bytes(canonical_json_bytes(report))
        plan={**bindings,"policy":review.POLICY,"approval_reference":review.APPROVAL_REFERENCE,"audit_seed":20260729,
              "human_audit":{"agent_different_sample":60,"published_same_sample":6,"negative_strata":list(review.STRATA)},
              "no_real_split_until_human_audit_and_acceptance":True,
              "audit_limit":"Spot-check monitor; not an accuracy/error-rate validation guarantee."}
        plan_path.write_bytes(canonical_json_bytes(plan))
        self.enterContext(patch.object(review,"AUDIT_PLAN_SHA256",file_sha256(plan_path)))
        self.enterContext(patch.object(review,"TOTAL_COUNT",len(queue)))
        self.enterContext(patch.object(review,"SOURCE_COUNT",len(source)))
        source_assignment.write_bytes(canonical_json_bytes({**bindings,"pairs":source}))
        legacy_path.write_bytes(canonical_json_bytes({"manifest_sha256":bindings["manifest_sha256"],"pairs":legacy}))
        assignment=self.root/"assigned.json";assignment.write_bytes(canonical_json_bytes({**bindings,"pairs":assigned}))
        observation=self.root/"observed.json"
        observation.write_bytes(canonical_json_bytes({**bindings,"assigned_file_sha256":file_sha256(assignment),
            "runtime":{"model":"invented/test","reasoning":"invented/test","start_utc":"2026-10-02T00:00:00Z",
                       "end_utc":"2026-10-02T00:00:01Z","elapsed_seconds":1},"pairs":observations}))
        self.args=(manifest,screen,index,plan_path,[assignment],[observation],source_assignment,legacy_path,self.root/"review")
        self.observation=observation;self.bindings=bindings

    def invoke(self,**kwargs):return review.prepare(*self.args,**kwargs)

    def blocked(self):
        with self.assertRaises(ContractError):self.invoke()
        self.assertFalse(self.root.joinpath("review.agent.json").exists())

    def test_complete_ai_coverage_produces_fixed_pending_audit_not_identity_results(self):
        result=self.invoke()
        self.assertEqual(result["new_observations"],102)
        self.assertEqual(result["negative_spot_check_count"],60)
        self.assertEqual(set(result["negative_spot_check_quotas"].values()),{15})
        self.assertEqual(result["source_same_spot_check_count"],2)
        self.assertFalse(result["production_freeze"]);self.assertFalse(result["human_audit_complete"])
        self.assertFalse(result["human_identity_decisions_complete"])
        self.assertEqual(result["human_inspected_unclear"],0)
        self.assertFalse(result["human_acceptance_recorded"])
        aggregate=json.loads((self.root/"review.agent.json").read_text())
        self.assertEqual(len(aggregate["pairs"]),104)
        self.assertTrue(all(p["human_review_status"]=="not_performed" for p in aggregate["pairs"]))
        draft=json.loads((self.root/"review.human.draft.json").read_text())
        self.assertTrue(draft["draft"])
        self.assertNotEqual(draft["candidate_queue_sha256"],draft["full_candidate_queue_sha256"])
        self.assertEqual(draft["candidate_queue_sha256"],file_sha256(self.root/"review.human_candidates.csv"))
        self.assertTrue(all(p["decision"]=="pending" and not p["full_view_inspected"] for p in draft["decisions"]))
        self.assertFalse(draft["human_acceptance"]["accept_identity_and_participation"])

    def test_stratified_small_bucket_census_redistribution_and_order_invariance(self):
        buckets={s:{(f"left_{s}_{i}",f"right_{i}") for i in range(n)} for s,n in zip(review.STRATA,(100,5,0,5))}
        selected,quotas=review.negative_sample(buckets)
        self.assertEqual(len(selected),60);self.assertEqual(list(quotas.values()),[50,5,0,5])
        reordered={s:set(reversed(sorted(v))) for s,v in reversed(list(buckets.items()))}
        self.assertEqual((selected,quotas),review.negative_sample(reordered))

    def test_missing_repeated_falsely_human_or_unviewed_observations_stop(self):
        original=self.observation.read_bytes()
        for change in ("missing","repeat","human","unviewed","view_hash","source_hash","runtime","date"):
            with self.subTest(change=change):
                value=json.loads(original)
                if change=="missing":value["pairs"].pop()
                if change=="repeat":value["pairs"].append(value["pairs"][0])
                if change=="human":value["pairs"][0]["human_review_status"]="performed"
                if change=="unviewed":value["pairs"][0]["full_resolution_viewed"]=False
                if change=="view_hash":value["pairs"][0]["view_sha256"]="0"*64
                if change=="source_hash":value["pairs"][0]["source_payload_sha256"][value["pairs"][0]["left"]]="0"*64
                if change=="runtime":value["runtime"]=[]
                if change=="date":value["pairs"][0]["review_date"]="2026-02-30"
                self.observation.write_bytes(canonical_json_bytes(value));self.blocked()
        self.observation.write_bytes(original)

    def test_stale_plan_assignment_or_visual_view_bytes_block_before_output(self):
        self.args[3].write_bytes(self.args[3].read_bytes()+b" ");self.blocked()

    def human_rows(self,draft,rows):
        human=json.loads(json.dumps(draft));human["decisions"]=rows
        path=self.root/"human.json";path.write_bytes(canonical_json_bytes(human));return path

    def reviewed(self,row,decision):
        return {**row,"decision":decision,"reviewer":"invented human","review_date":"2026-10-02",
                "rationale":"invented personally inspected landmarks","full_view_inspected":True}

    def test_negative_miss_expands_stratum_and_component_two_strata_all_negatives(self):
        initial=self.invoke();draft=json.loads((self.root/"review.human.draft.json").read_text())
        strata={}
        for row in draft["decisions"]:
            if row["agent_recommendation"]=="different_lesions":
                name="published_ranking" if "independent_published" in row["screen_reasons"] else "equal_dhash"
                mixed=row["source_applicability"] # labels derive fixture pair naming here
                label="mixed_label" if "negative1_" in row["left"] or "negative3_" in row["left"] else "same_label"
                strata.setdefault(name+"/"+label,row)
        first=strata[review.STRATA[0]];human=self.human_rows(draft,[self.reviewed(first,"pending")])
        self.args=(*self.args[:-1],self.root/"expanded")
        expanded=self.invoke(human_audit=human)
        self.assertEqual(expanded["failed_negative_strata"],[review.STRATA[0]])
        self.assertGreater(expanded["human_required_pairs"],initial["human_required_pairs"])
        required={p["left"] for p in expanded["required_pairs"]}
        self.assertTrue(all(f"invented_negative0_{i:02}_a" in required for i in range(25)))
        second=strata[review.STRATA[2]]
        human=self.human_rows(draft,[self.reviewed(first,"same_lesion"),self.reviewed(second,"pending")])
        self.args=(*self.args[:-1],self.root/"all_negative")
        expanded=self.invoke(human_audit=human)
        self.assertEqual(expanded["human_required_reason_counts"]["expanded_all_negatives_two_failed_strata"],100)

    def test_source_disagreement_expands_all_source_and_associated_components(self):
        self.invoke();draft=json.loads((self.root/"review.human.draft.json").read_text())
        source=next(r for r in draft["decisions"] if "source0" in r["left"])
        human=self.human_rows(draft,[self.reviewed(source,"different_lesions")])
        self.args=(*self.args[:-1],self.root/"source_expanded")
        result=self.invoke(human_audit=human)
        self.assertTrue(result["source_same_disagreement_or_unclear"])
        self.assertEqual(result["human_required_reason_counts"]["expanded_all_source18_disagreement"],2)
        self.assertGreater(result["human_required_reason_counts"]["expanded_affected_candidate_component"],2)

    def test_expanded_source18_draft_preserves_exact_annotations_and_new_pairs_blank(self):
        self.make_fixture(source_count=18,source_anchor=False,unclear_count=287)
        initial=self.invoke()
        initial_draft=json.loads((self.root/"review.human.draft.json").read_text())
        annotations=[]
        unclear=0
        for row in initial_draft["decisions"]:
            decision="pending" if row["agent_recommendation"]=="uncertain" else row["agent_recommendation"]
            if decision=="pending":
                unclear+=1
                if unclear>12:decision="different_lesions"
            # A sampled published same disagreement expands review to all 18.
            if "reported_missed_duplicate" in row["screen_reasons"]:
                decision="different_lesions" if not any("reported_missed_duplicate" in r["screen_reasons"] for r in annotations) else "same_lesion"
            annotated=self.reviewed(row,decision)
            annotated["reviewer"]=" Invented reviewer Ω "
            annotated["rationale"]=" Invented inspection: landmarks differ.\nUncertainty retained where applicable. Ω "
            annotations.append(annotated)
        input_annotations=canonical_json_bytes(annotations)
        human=self.human_rows(initial_draft,annotations)
        # Imported acceptance and stale presentation fields must not become
        # regenerated acceptance or override the current expanded screen flags.
        imported=json.loads(human.read_text())
        imported["human_acceptance"]={"reviewer":"Invented acceptance", "accept_procedure":True,
                                      "accept_identity_and_participation":True}
        for row in imported["decisions"]:
            row["screen_reasons"]="stale presentation flags"
            row["agent_rationale"]="stale agent presentation"
        human.write_bytes(canonical_json_bytes(imported))
        self.args=(*self.args[:-1],self.root/"source18_expanded")
        result=self.invoke(human_audit=human)
        expanded=json.loads((self.root/"source18_expanded.human.draft.json").read_text())
        fields=("decision","reviewer","review_date","rationale","full_view_inspected")
        by_key={(r["left"],r["right"]):r for r in expanded["decisions"]}
        old_keys={(r["left"],r["right"]) for r in annotations}
        self.assertEqual(len(old_keys),354)
        self.assertEqual(len(by_key),366)
        for annotated in annotations:
            actual=by_key[(annotated["left"],annotated["right"])]
            self.assertEqual(canonical_json_bytes({f:actual[f] for f in fields}),
                             canonical_json_bytes({f:annotated[f] for f in fields}))
            for field in ("evidence","view_sha256","source_payload_sha256","source_applicability"):
                self.assertEqual(actual[field],annotated[field])
            self.assertNotEqual(actual["screen_reasons"],"stale presentation flags")
            self.assertNotEqual(actual["agent_rationale"],"stale agent presentation")
        new=[r for k,r in by_key.items() if k not in old_keys]
        self.assertEqual(len(new),12)
        self.assertEqual(sum("reported_missed_duplicate" in r["screen_reasons"] for r in new),12)
        self.assertTrue(all((r["left"],r["right"]) not in old_keys for r in expanded["decisions"][:len(new)]))
        self.assertTrue(all((r["left"],r["right"]) in old_keys for r in expanded["decisions"][len(new):]))
        for row in new:
            self.assertEqual({f:row[f] for f in fields}, {"decision":"pending","reviewer":"",
                "review_date":"","rationale":"","full_view_inspected":False})
        self.assertEqual(canonical_json_bytes(annotations),input_annotations)
        self.assertTrue(result["source_same_disagreement_or_unclear"])
        self.assertEqual(result["human_required_reason_counts"]["expanded_all_source18_disagreement"],18)
        self.assertGreater(result["human_required_pairs"],initial["human_required_pairs"])
        self.assertEqual(result["human_audit_rows_recorded_as_attestations"],len(annotations))
        self.assertEqual(result["human_inspected_unclear"],12)
        self.assertFalse(result["human_audit_complete"])
        self.assertFalse(result["human_identity_decisions_complete"])
        self.assertFalse(result["split_ready"])
        self.assertFalse(result["production_freeze"])
        self.assertFalse(result["human_acceptance_recorded"])
        self.assertTrue(expanded["draft"])
        self.assertEqual(expanded["human_acceptance"],{"reviewer":"","date":"","approval_reference":"",
            "accept_procedure":False,"accept_identity_and_participation":False,"decision_sha256":""})

    def test_secondary_uncertainty_forces_human_flag_without_primary_override(self):
        value=json.loads(self.observation.read_text());item=dict(value["pairs"][0])
        item["agent_recommendation"]="uncertain";item["agent_reviewer"]="Codex secondary synthetic fixture"
        secondary=self.root/"secondary.json"
        secondary.write_bytes(canonical_json_bytes({**self.bindings,"pairs":[item],"timestamp_utc":"invented timestamp"}))
        result=self.invoke(secondary_observations=secondary)
        self.assertEqual(result["human_required_reason_counts"]["secondary_agent_disagreement_or_uncertainty"],1)
        aggregate=json.loads((self.root/"review.agent.json").read_text())
        primary=next(p for p in aggregate["pairs"] if p["left"]==item["left"])
        self.assertEqual(primary["agent_recommendation"],"different_lesions")
        self.assertEqual(aggregate["secondary_observations"][0]["agent_recommendation"],"uncertain")

    def test_completed_source_expansion_and_explicit_revisions_preserve_identity_gates(self):
        # Invented lifecycle matching the 354 -> 366 workload and three explicit
        # annotation revisions. It grants no real identity or acceptance policy.
        self.make_fixture(source_count=18,source_anchor=False,unclear_count=287)
        self.invoke()
        draft=json.loads((self.root/"review.human.draft.json").read_text())
        annotations=[];unclear_count=0;source_count=0
        for row in draft["decisions"]:
            decision=row["agent_recommendation"]
            if decision=="uncertain":
                unclear_count+=1
                decision="same_lesion" if unclear_count<=5 else "pending" if unclear_count<=17 else "different_lesions"
            if "reported_missed_duplicate" in row["screen_reasons"]:
                source_count+=1
                decision="different_lesions" if source_count==1 else "same_lesion"
            annotations.append(self.reviewed(row,decision))
        self.assertEqual(len(annotations),354)
        human=self.human_rows(draft,annotations)
        self.args=(*self.args[:-1],self.root/"expanded_lifecycle")
        self.invoke(human_audit=human)
        expanded=json.loads((self.root/"expanded_lifecycle.human.draft.json").read_text())
        old_keys={(r["left"],r["right"]) for r in annotations}
        completed=[];revised_same=False;revised_different=0
        for row in expanded["decisions"]:
            revised=dict(row)
            if (row["left"],row["right"]) not in old_keys:
                revised=self.reviewed(row,"same_lesion")
            elif row["agent_recommendation"]=="uncertain":
                if row["decision"]=="same_lesion" and not revised_same:
                    revised["decision"]="pending";revised_same=True
                    revised["rationale"]="Invented explicit revised uncertainty."
                elif row["decision"]=="different_lesions" and revised_different<2:
                    revised["decision"]="pending";revised_different+=1
                    revised["rationale"]="Invented explicit revised uncertainty."
            completed.append(revised)
        human=self.human_rows(expanded,completed)
        self.args=(*self.args[:-1],self.root/"completed_lifecycle")
        result=self.invoke(human_audit=human)
        self.assertEqual(result["human_required_pairs"],366)
        self.assertTrue(result["human_audit_complete"])
        self.assertTrue(result["human_identity_consistent"])
        self.assertEqual(result["human_inspected_unclear"],15)
        self.assertFalse(result["human_identity_decisions_complete"])
        self.assertEqual(result["failed_negative_strata"],[])
        self.assertTrue(result["source_same_disagreement_or_unclear"])
        self.assertFalse(result["human_acceptance_recorded"])
        self.assertFalse(result["production_freeze"])
        self.assertFalse(result["split_ready"])
        regenerated=json.loads((self.root/"completed_lifecycle.human.draft.json").read_text())
        by_key={(r["left"],r["right"]):r for r in regenerated["decisions"]}
        for row in completed:
            actual=by_key[(row["left"],row["right"])]
            for field in ("decision","reviewer","review_date","rationale","full_view_inspected"):
                self.assertEqual(actual[field],row[field])
        self.assertEqual(sum(r["decision"]=="same_lesion" for r in completed),22)
        self.assertEqual(sum(r["decision"]=="different_lesions" for r in completed),329)
        records=read_manifest(self.args[0])
        decisions=[CandidateDecision(r["left"],r["right"],r["evidence"],r["decision"],r["reviewer"],r["rationale"])
                   for r in completed]
        with self.assertRaises(ContractError) as caught:
            resolve_analysis(records,decisions)
        self.assertEqual(caught.exception.code,"unresolved_candidate")
        # Synthetic staging explicitly separates unresolved split-only links.
        # This is not a real resolver adapter or permission to omit candidates.
        physical=resolve_analysis(records,[d for d in decisions if d.decision!="pending"])
        uncertain=[ReviewedUncertainty(r["left"],r["right"],r["evidence"],r["reviewer"],r["rationale"],
                   r["review_date"],full_view_inspected=r["full_view_inspected"])
                   for r in completed if r["decision"]=="pending"]
        grouping=build_conservative_groups(physical,uncertain)
        physical_by_image={r.original.image_id:r for r in physical}
        for row in grouping.records:
            self.assertEqual(row.analysis_lesion_id,physical_by_image[row.original.image_id].analysis_lesion_id)
            self.assertEqual(row.original,physical_by_image[row.original.image_id].original)
            self.assertEqual(row.active,physical_by_image[row.original.image_id].active)
        plan=build_dependence_plan(grouping,resampling_unit_policy=CONSERVATIVE_CLUSTERS)
        membership={u.physical_id:u.active_image_ids for u in plan.physical_units}
        verified=validate_aggregation_membership(plan,{"A":membership,"B":membership})
        self.assertFalse(verified["weighting_policy_selected"])

    def test_cross_label_human_same_is_preserved_but_flagged_inconsistent(self):
        self.invoke();draft=json.loads((self.root/"review.human.draft.json").read_text())
        cross=next(r for r in draft["decisions"] if "invented_negative1_" in r["left"])
        annotated=self.reviewed(cross,"same_lesion")
        human=self.human_rows(draft,[annotated])
        self.args=(*self.args[:-1],self.root/"label_conflict")
        result=self.invoke(human_audit=human)
        self.assertEqual(result["human_same_label_conflicts"],[{"left":cross["left"],"right":cross["right"],
            "labels":{cross["left"]:"nv",cross["right"]:"bcc"}}])
        self.assertFalse(result["human_identity_consistent"])
        self.assertEqual(result["human_required_reason_counts"]["human_same_label_conflict"],1)
        expanded=json.loads((self.root/"label_conflict.human.draft.json").read_text())
        preserved=next(r for r in expanded["decisions"] if r["left"]==cross["left"])
        for field in ("decision","reviewer","review_date","rationale","full_view_inspected"):
            self.assertEqual(preserved[field],annotated[field])
        self.assertIn("human_same_label_conflict",preserved["screen_reasons"])
        self.assertFalse(result["split_ready"])
        self.assertFalse(result["production_freeze"])
        self.assertFalse(result["human_acceptance_recorded"])

    def test_human_transitive_contradictions_use_only_human_same_edges(self):
        by_id={i:ManifestRecord(i,"lesion_"+i,"nv","invented","v1",i+".jpg",sha256_bytes(i.encode()))
               for i in ("a","b","c")}
        human={("a","b"):{"decision":"same_lesion"},("b","c"):{"decision":"same_lesion"},
               ("a","c"):{"decision":"different_lesions"}}
        labels,conflicts=review.human_identity_diagnostics(human,by_id)
        self.assertEqual(labels,[])
        self.assertEqual(conflicts,[{"left":"a","right":"c","original_lesion_ids":["lesion_a","lesion_c"],
            "human_same_pairs":[{"left":"a","right":"b"},{"left":"b","right":"c"}]}])
        # Removing an inspected same edge must not infer it from source/agent evidence.
        del human[("b","c")]
        self.assertEqual(review.human_identity_diagnostics(human,by_id),([],[]))

    def test_invalid_secondary_hash_and_false_full_view_claim_are_rejected(self):
        value=json.loads(self.observation.read_text());item=dict(value["pairs"][0])
        secondary=self.root/"secondary.json"
        for change in ("view_hash","not_viewed"):
            with self.subTest(change=change):
                bad=dict(item)
                if change=="view_hash":bad["view_sha256"]="0"*64
                if change=="not_viewed":bad["full_resolution_viewed"]=False
                secondary.write_bytes(canonical_json_bytes({**self.bindings,"pairs":[bad]}))
                with self.assertRaises(ContractError):self.invoke(secondary_observations=secondary)
                self.assertFalse(self.root.joinpath("review.agent.json").exists())

    def test_completed_human_inspection_can_preserve_unclear_identity_without_freeze(self):
        self.invoke();draft=json.loads((self.root/"review.human.draft.json").read_text())
        reviewed=[self.reviewed(row,"pending" if row["agent_recommendation"]=="uncertain"
                                else row["agent_recommendation"]) for row in draft["decisions"]]
        human=self.human_rows(draft,reviewed)
        self.args=(*self.args[:-1],self.root/"inspected")
        result=self.invoke(human_audit=human)
        self.assertTrue(result["human_audit_complete"])
        self.assertFalse(result["human_identity_decisions_complete"])
        self.assertEqual(result["human_inspected_unclear"],1)
        self.assertFalse(result["human_acceptance_recorded"])
        self.assertFalse(result["split_ready"])
        self.assertFalse(result["production_freeze"])
        aggregate=json.loads((self.root/"inspected.agent.json").read_text())
        unclear=next(p for p in aggregate["pairs"] if "unclear" in p["left"])
        self.assertEqual(unclear["agent_recommendation"],"uncertain")


if __name__=="__main__":unittest.main()
