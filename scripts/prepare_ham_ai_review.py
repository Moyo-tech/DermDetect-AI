"""Validate AI observations and prepare the prespecified pending human audit.

No identity resolution, split, exclusion, model execution or outcome access.
Views are hash-checked as saved evidence; this script does not inspect pixels.
"""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import io
import csv
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, rooted_file, sha256_bytes
from dermdetect.data.review_policy import RANKED_PROCEDURE, validate_review_screen
from dermdetect.data.schemas import ContractError
from resolve_ham_analysis import VIEW_TRANSFORM, pair_key, read_candidate_queue, read_json_object
import prepare_ham_review_package as viewer

POLICY = "m2-ai-assisted-audit-v1"
APPROVAL_REFERENCE = "user_yes_to_ai_assisted_review_proposal_2026-10-02"
AUDIT_PLAN_SHA256 = "11d8fe45e959af87d0d48cd08a32027a2dadb23d8b12731b9e0b1b70ef092fad"
TOTAL_COUNT, SOURCE_COUNT = 665, 18
STRATA = ("published_ranking/same_label", "published_ranking/mixed_label",
          "equal_dhash/same_label", "equal_dhash/mixed_label")
HASH_FIELDS = ("manifest_sha256", "candidate_queue_sha256", "visual_index_sha256")


def sample_key(stratum, key):
    value = json.dumps([POLICY, 20260729, stratum, *key], separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest(), key


def negative_sample(buckets):
    """Fixed quotas, census small buckets, largest-remaining redistribution."""
    ordered = {s: sorted(buckets.get(s, ()), key=lambda k: sample_key(s, k)) for s in STRATA}
    quotas = {s: min(15, len(ordered[s])) for s in STRATA}
    target = min(60, sum(map(len, ordered.values())))
    while sum(quotas.values()) < target:
        available = [s for s in STRATA if quotas[s] < len(ordered[s])]
        chosen = min(available, key=lambda s: (-(len(ordered[s]) - quotas[s]), s))
        quotas[chosen] += 1
    return {k for s in STRATA for k in ordered[s][:quotas[s]]}, quotas


def require_bindings(value, expected):
    if any(value.get(k) != v for k, v in expected.items()):
        raise ContractError("ai_review_binding_mismatch", "AI review artifacts must bind the same prespecified evidence.")


def validate_date(value):
    try:
        if not isinstance(value, str) or datetime.date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except (ValueError, TypeError):
        raise ContractError("invalid_agent_review_date", "A canonical ISO review date is required.")


def pairs(value):
    result = value.get("pairs")
    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
        raise ContractError("invalid_ai_observation", "Pair records must be a list of objects.")
    return result


def validate_agent(item):
    if (item.get("agent_recommendation") not in {"same_lesion", "different_lesions", "uncertain"}
            or not isinstance(item.get("agent_rationale"), str) or not item["agent_rationale"].strip()
            or not isinstance(item.get("agent_reviewer"), str) or not item["agent_reviewer"].startswith("Codex")
            or item.get("full_resolution_viewed") is not True
            or item.get("human_review_status") != "not_performed"
            or "decision" in item or "human_acceptance" in item):
        raise ContractError("invalid_ai_observation", "Actual attributed full-view observations are recommendations, never human decisions.")
    validate_date(item.get("review_date"))


def graph_components(queue, by_id):
    parent = {r.lesion_id: r.lesion_id for r in by_id.values()}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for key in queue:
        a, b = root(by_id[key[0]].lesion_id), root(by_id[key[1]].lesion_id)
        parent[max(a, b)] = min(a, b)
    return {key: root(by_id[key[0]].lesion_id) for key in queue}


def human_identity_diagnostics(human_rows, by_id):
    """Check validated inspected human decisions, without assuming source merges."""
    same = {k:r for k,r in human_rows.items() if r["decision"] == "same_lesion"}
    label_conflicts = [{"left": k[0], "right": k[1], "labels": {i:by_id[i].dx for i in k}}
                       for k in sorted(same) if by_id[k[0]].dx != by_id[k[1]].dx]
    components = graph_components(same, by_id)
    members, edges = defaultdict(set), defaultdict(list)
    for key, component in sorted(components.items()):
        members[component].update(by_id[i].lesion_id for i in key)
        edges[component].append({"left":key[0], "right":key[1]})
    contradictions = []
    for key, row in sorted(human_rows.items()):
        if row["decision"] != "different_lesions":
            continue
        a, b = (by_id[i].lesion_id for i in key)
        connected = next((component for component, ids in members.items() if a in ids and b in ids), None)
        if a == b or connected is not None:
            contradictions.append({"left":key[0], "right":key[1],
                "original_lesion_ids":[a,b], "human_same_pairs":edges[connected] if connected is not None else []})
    return label_conflicts, contradictions


def prepare(manifest: Path, screen_dir: Path, visual_index: Path, audit_plan: Path,
            assignments: list[Path], observation_paths: list[Path], published18: Path,
            source_agent_review: Path, output_prefix: Path, *, human_audit: Path | None = None,
            pilot_only: bool = False, secondary_observations: Path | None = None) -> dict:
    suffixes = {"agent": ".agent.json", "audit": ".audit.json", "draft": ".human.draft.json",
                "page": ".html", "queue": ".human_candidates.csv"}
    outputs = {name: output_prefix.with_suffix(suffix) for name, suffix in suffixes.items()}
    snapshot = output_prefix.with_suffix(".source_snapshot")
    if any(p.exists() or p.is_symlink() for p in (*outputs.values(), snapshot)):
        raise ContractError("output_exists", "Refusing to overwrite AI/human audit evidence.")
    plan_digest = file_sha256(audit_plan)
    plan = read_json_object(audit_plan)
    if (plan_digest != AUDIT_PLAN_SHA256 or plan.get("policy") != POLICY
            or plan.get("approval_reference") != APPROVAL_REFERENCE
            or plan.get("audit_seed") != 20260729
            or plan.get("human_audit", {}).get("agent_different_sample") != 60
            or plan.get("human_audit", {}).get("published_same_sample") != 6
            or plan.get("human_audit", {}).get("negative_strata") != list(STRATA)
            or plan.get("no_real_split_until_human_audit_and_acceptance") is not True):
        raise ContractError("audit_plan_mismatch", "Use the unchanged audit plan recorded before AI observations.")
    root = output_prefix.parent.resolve(strict=True)
    index_path = visual_index.resolve(strict=True)
    if not index_path.is_relative_to(root):
        raise ContractError("unsafe_path", "Human package root must contain the saved visual index.")
    by_id = {r.image_id: r for r in read_manifest(manifest)}
    report_path, queue_path = screen_dir / "screen_report.json", screen_dir / "candidates.csv"
    report = read_json_object(report_path)
    if validate_review_screen(report, file_sha256(manifest), file_sha256(queue_path)) != RANKED_PROCEDURE:
        raise ContractError("ai_review_binding_mismatch", "AI workflow requires the approved ranked queue.")
    bindings = {"manifest_sha256": file_sha256(manifest), "candidate_queue_sha256": file_sha256(queue_path),
                "visual_index_sha256": file_sha256(index_path)}
    require_bindings(plan, bindings)
    rows = read_candidate_queue(queue_path, by_id)
    queue = {pair_key(r["left"], r["right"], by_id, code="invalid_candidate_queue"): r for r in rows}
    if len(rows) != len(queue) or len(queue) != TOTAL_COUNT or report.get("review_candidates") != len(queue):
        raise ContractError("candidate_membership_mismatch", "Approved candidate coverage must be complete and unique.")
    index = read_json_object(index_path)
    if (index.get("manifest_sha256") != bindings["manifest_sha256"]
            or index.get("status") != "visual_evidence_prepared_not_adjudicated"
            or index.get("queue_sha256") != report.get("view_queue_sha256")):
        raise ContractError("visual_evidence_mismatch", "Visual index must bind the approved view queue.")
    views = {}
    for item in pairs(index):
        ids = item.get("image_ids", ())
        if len(ids) != 2:
            raise ContractError("visual_evidence_mismatch", "Every view needs two inventory images.")
        key = pair_key(*ids, by_id, code="visual_evidence_mismatch")
        if (key in views or key not in queue or item.get("view_transform") != VIEW_TRANSFORM
                or item.get("source_payload_sha256") != [by_id[i].sha256 for i in ids]
                or file_sha256(rooted_file(index_path.parent, item.get("view_file", ""))) != item.get("view_sha256")):
            raise ContractError("visual_evidence_mismatch", "Preserved full-resolution view hashes must match exactly.")
        views[key] = item
    if set(views) != set(queue):
        raise ContractError("visual_evidence_mismatch", "Every approved candidate requires preserved full-resolution evidence.")

    def verify_pair(item):
        key = pair_key(item.get("left"), item.get("right"), by_id, code="invalid_ai_observation")
        view = views.get(key)
        if (view is None or item.get("view_sha256") != view["view_sha256"]
                or item.get("source_payload_sha256") != {i: by_id[i].sha256 for i in key}):
            raise ContractError("visual_evidence_mismatch", "Observation/source/view correspondence must match.")
        if "view_file" in item and item["view_file"] != view["view_file"]:
            raise ContractError("visual_evidence_mismatch", "Observation route must name the preserved view.")
        return key

    source_assignment = read_json_object(published18)
    require_bindings(source_assignment, bindings)
    source_keys = set()
    for item in pairs(source_assignment):
        key = verify_pair(item)
        if key in source_keys:
            raise ContractError("candidate_membership_mismatch", "Source-backed pair membership cannot repeat.")
        source_keys.add(key)
    expected_source = {k for k, r in queue.items() if "reported_missed_duplicate" in r["reasons"].split("|")}
    if source_keys != expected_source or len(source_keys) != SOURCE_COUNT:
        raise ContractError("candidate_membership_mismatch", "Reuse precisely the source-backed 18 pair set.")
    legacy = read_json_object(source_agent_review)
    require_bindings(legacy, {"manifest_sha256": bindings["manifest_sha256"]})
    source_observations = {}
    for item in pairs(legacy):
        ids = item.get("image_ids", ())
        if len(ids) != 2:
            raise ContractError("invalid_ai_observation", "Legacy observations require image pairs.")
        key = pair_key(*ids, by_id, code="invalid_ai_observation")
        if key not in source_keys:
            continue
        if key in source_observations or item.get("agent_recommendation") != "same_lesion":
            raise ContractError("invalid_ai_observation", "Source reuse requires one existing same-lesion recommendation.")
        normalized = {"left": key[0], "right": key[1], "agent_recommendation": "same_lesion",
            "agent_rationale": item.get("rationale"), "agent_reviewer": item.get("reviewer"),
            "review_date": item.get("agent_review_date"), "view_file": views[key]["view_file"],
            "view_sha256": item.get("view_sha256"),
            "source_payload_sha256": dict(zip(ids, item.get("source_payload_sha256", ()))),
            "full_resolution_viewed": True, "human_review_status": "not_performed",
            "observation_origin": "reused_original_source18_agent_observation"}
        verify_pair(normalized)
        if item.get("human_review_status") != "not_performed" or item.get("view_transform") != VIEW_TRANSFORM:
            raise ContractError("invalid_ai_observation", "Reused source recommendations are not human decisions.")
        validate_date(normalized["review_date"])
        if (not isinstance(normalized["agent_reviewer"], str) or not normalized["agent_reviewer"].startswith("Codex")
                or not isinstance(normalized["agent_rationale"], str) or not normalized["agent_rationale"].strip()):
            raise ContractError("invalid_ai_observation", "Legacy agent attribution and observations are required.")
        source_observations[key] = normalized
    if set(source_observations) != source_keys:
        raise ContractError("candidate_membership_mismatch", "Every source-backed pair needs its preserved local observation.")
    assignment_sets, assignment_members = {}, set()
    for path in assignments:
        value = read_json_object(path)
        require_bindings(value, bindings)
        digest = file_sha256(path)
        if digest in assignment_sets:
            raise ContractError("candidate_membership_mismatch", "Assignments must be disjoint, unique files.")
        keys = set()
        for item in pairs(value):
            key = verify_pair(item)
            if key in keys or key in assignment_members or key in source_keys:
                raise ContractError("candidate_membership_mismatch", "Assigned new observations must occur exactly once.")
            keys.add(key)
        assignment_sets[digest] = keys
        assignment_members.update(keys)
    if not pilot_only and assignment_members | source_keys != set(queue):
        raise ContractError("candidate_membership_mismatch", "Assignments must cover all 647 new candidates plus source18.")
    observations, runtimes, observation_files = {}, [], []
    for path in observation_paths:
        value = read_json_object(path)
        require_bindings(value, bindings)
        if "audit_plan_sha256" in value and value["audit_plan_sha256"] != plan_digest:
            raise ContractError("audit_plan_mismatch", "Observation batch refers to another audit plan.")
        assigned_digest = value.get("assigned_file_sha256")
        if assigned_digest not in assignment_sets:
            raise ContractError("ai_review_binding_mismatch", "Every observation batch must bind its assigned file.")
        runtime = value.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ContractError("missing_agent_runtime", "Runtime records must be objects.")
        start = runtime.get("start_utc", runtime.get("started_at"))
        end = runtime.get("end_utc", runtime.get("completed_at"))
        elapsed = runtime.get("elapsed_seconds")
        if (not isinstance(runtime, dict) or any(not isinstance(v, str) or not v.strip()
                for v in (runtime.get("model"), runtime.get("reasoning"), start, end))
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise ContractError("missing_agent_runtime", "Each batch requires attributed model/reasoning/start/end/elapsed evidence.")
        batch_keys = set()
        for item in pairs(value):
            key = verify_pair(item)
            if key in observations or key in source_keys or key in batch_keys:
                raise ContractError("candidate_membership_mismatch", "New AI observations must be exactly once, without source reuse.")
            validate_agent(item)
            observations[key] = {**item, "view_file": views[key]["view_file"], "observation_origin": "new_ai_observation"}
            batch_keys.add(key)
        if batch_keys != assignment_sets[assigned_digest]:
            raise ContractError("candidate_membership_mismatch", "Each completed batch must exactly cover its assignment.")
        runtimes.append({"assigned_file_sha256": assigned_digest, "runtime": runtime})
        observation_files.append({"path": str(path), "sha256": file_sha256(path)})
    if set(observations) != assignment_members:
        raise ContractError("candidate_membership_mismatch", "Every supplied assignment needs complete observations.")
    all_observations = {**source_observations, **observations}
    if not pilot_only and set(all_observations) != set(queue):
        raise ContractError("candidate_membership_mismatch", "Exactly 665 source/AI pair observations are required.")
    secondary = {}
    secondary_record = None
    if secondary_observations is not None:
        value = read_json_object(secondary_observations)
        require_bindings(value, bindings)
        for item in pairs(value):
            key = verify_pair(item)
            validate_agent(item)
            if key in secondary or key not in all_observations:
                raise ContractError("candidate_membership_mismatch", "Secondary observations require unique primary-observed pairs.")
            secondary[key] = item
        secondary_record = {"path": str(secondary_observations), "sha256": file_sha256(secondary_observations),
                            "timestamp_utc": value.get("timestamp_utc")}
    aggregate = {**bindings, "audit_plan_sha256": plan_digest, "policy": POLICY,
        "status": "pilot_observations_pending_human_audit" if pilot_only else "ai_observations_pending_human_audit",
        "production_freeze": False, "approval_reference": APPROVAL_REFERENCE,
        "source_agent_review_sha256": file_sha256(source_agent_review), "published18_sha256": file_sha256(published18),
        "assignments_sha256": sorted(assignment_sets), "observation_files": observation_files, "runtime_records": runtimes,
        "secondary_observation_file": secondary_record,
        "secondary_observations": [secondary[k] for k in sorted(secondary)],
        "pairs": [all_observations[k] for k in sorted(all_observations)]}
    aggregate_digest = sha256_bytes(canonical_json_bytes(aggregate))
    strata = {}
    for key, row in queue.items():
        source = "published_ranking" if "independent_published_top1000_cross_id" in row["reasons"].split("|") else "equal_dhash"
        labels = "same_label" if by_id[key[0]].dx == by_id[key[1]].dx else "mixed_label"
        strata[key] = source + "/" + labels
    negatives = {k for k, o in observations.items() if o["agent_recommendation"] == "different_lesions"}
    buckets = {s: {k for k in negatives if strata[k] == s} for s in STRATA}
    sampled, quotas = negative_sample(buckets)
    source_sample = set(sorted(source_keys, key=lambda k: sample_key("source_verified_same", k))[:6])
    required = defaultdict(set)
    for key, item in observations.items():
        if item["agent_recommendation"] == "same_lesion": required[key].add("new_same_lesion")
        if item["agent_recommendation"] == "uncertain": required[key].add("agent_uncertain")
    for key, row in queue.items():
        if key in all_observations and "reported_similar_pair_different_lesions" in row["reasons"].split("|"):
            required[key].add("published_ambiguous_source_wording")
    for key in sampled: required[key].add("fixed_negative_spot_check")
    for key in source_sample: required[key].add("fixed_source_same_spot_check")
    for key, item in secondary.items():
        if item["agent_recommendation"] == "uncertain" or item["agent_recommendation"] != all_observations[key]["agent_recommendation"]:
            required[key].add("secondary_agent_disagreement_or_uncertainty")
    same_edges = {k for k, o in all_observations.items() if o["agent_recommendation"] == "same_lesion"}
    positive_components = graph_components({k: queue[k] for k in same_edges}, by_id)
    positives_by_component = defaultdict(set)
    positive_lesions = defaultdict(set)
    for key, component in positive_components.items():
        positives_by_component[component].add(key)
        positive_lesions[component].update(by_id[i].lesion_id for i in key)
        if by_id[key[0]].dx != by_id[key[1]].dx: required[key].add("label_conflicting_same_proposal")
    for component, members in positive_lesions.items():
        if len(members) > 2:
            for key in positives_by_component[component]: required[key].add("high_impact_transitive_same_component")
    for key in negatives:
        a, b = (by_id[i].lesion_id for i in key)
        for component, members in positive_lesions.items():
            if a in members and b in members:
                required[key].add("transitive_identity_conflict")
                for edge in positives_by_component[component]: required[edge].add("transitive_identity_conflict")
    components = graph_components(queue, by_id)
    human_rows, failed_strata, failed_components = {}, set(), set()
    source_failure = False
    if human_audit is not None:
        human = read_json_object(human_audit)
        require_bindings(human, {"manifest_sha256": bindings["manifest_sha256"],
            "full_candidate_queue_sha256": bindings["candidate_queue_sha256"],
            "visual_index_sha256": bindings["visual_index_sha256"], "audit_plan_sha256": plan_digest,
            "agent_aggregate_sha256": aggregate_digest})
        human_seen = set()
        if not isinstance(human.get("decisions"), list) or any(not isinstance(v,dict) for v in human["decisions"]):
            raise ContractError("invalid_human_audit", "Human audit decisions must be objects.")
        for row in human["decisions"]:
            key = verify_pair(row)
            if (key in human_seen or key not in all_observations
                    or row.get("evidence") != views[key]["view_file"]
                    or row.get("source_applicability") != {i:{"source":by_id[i].source,"source_version":by_id[i].source_version} for i in key}):
                raise ContractError("invalid_human_audit", "Human audit membership must be unique and observed.")
            human_seen.add(key)
            if row.get("decision") not in {"same_lesion", "different_lesions", "pending"}:
                raise ContractError("invalid_human_audit", "Human uncertainty must remain pending.")
            if row.get("full_view_inspected") is not True:
                if row.get("decision") != "pending" or any(row.get(k) for k in ("reviewer", "review_date", "rationale")):
                    raise ContractError("invalid_human_audit", "Non-pending annotations require full-view human inspection evidence.")
                continue
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("reviewer", "rationale")):
                raise ContractError("invalid_human_audit", "Inspected audit decisions need reviewer and rationale.")
            validate_date(row.get("review_date"))
            human_rows[key] = row
            if key in negatives and row["decision"] != "different_lesions":
                failed_strata.add(strata[key]);failed_components.add(components[key])
            if key in source_keys and row["decision"] != "same_lesion":
                source_failure = True;failed_components.add(components[key])
        for key in negatives:
            if strata[key] in failed_strata: required[key].add("expanded_failed_negative_stratum")
            if len(failed_strata) >= 2: required[key].add("expanded_all_negatives_two_failed_strata")
        for key, component in components.items():
            if component in failed_components and key in all_observations: required[key].add("expanded_affected_candidate_component")
        if source_failure:
            for key in source_keys: required[key].add("expanded_all_source18_disagreement")
    human_label_conflicts, human_transitive_conflicts = human_identity_diagnostics(human_rows, by_id)
    for conflict in human_label_conflicts:
        required[conflict["left"], conflict["right"]].add("human_same_label_conflict")
    for conflict in human_transitive_conflicts:
        required[conflict["left"], conflict["right"]].add("human_transitive_identity_conflict")
        for edge in conflict["human_same_pairs"]:
            required[edge["left"], edge["right"]].add("human_transitive_identity_conflict")
    required_keys = set(required)
    # Inspection can be complete while physical identity remains unclear.
    # Such uncertainty is preserved for the separately gated split-only policy;
    # it never licenses forced identity decisions or a freeze here.
    human_complete = all(k in human_rows for k in required_keys)
    human_identity_complete = all(k in human_rows and human_rows[k]["decision"] != "pending" for k in required_keys)
    human_inspected_unclear = sum(row["decision"] == "pending" for row in human_rows.values())
    audit = {**bindings, "audit_plan_sha256": plan_digest, "agent_aggregate_sha256": aggregate_digest,
        "policy": POLICY, "procedure": RANKED_PROCEDURE, "production_freeze": False, "split_ready": False,
        "status": "pilot_audit_preparation_only" if pilot_only else "human_audit_and_acceptance_pending",
        "agent_recommendations": dict(Counter(o["agent_recommendation"] for o in all_observations.values())),
        "reused_source_pairs": len(source_keys), "new_observations": len(observations),
        "negative_spot_check_count": len(sampled), "negative_spot_check_quotas": quotas,
        "source_same_spot_check_count": len(source_sample),
        "secondary_quality_observations": len(secondary),
        "human_required_pairs": len(required_keys), "human_required_reason_counts": dict(Counter(r for rs in required.values() for r in rs)),
        "failed_negative_strata": sorted(failed_strata), "source_same_disagreement_or_unclear": source_failure,
        "human_audit_rows_recorded_as_attestations": len(human_rows), "human_audit_complete": human_complete,
        "human_identity_decisions_complete": human_identity_complete,
        "human_identity_consistent": not bool(human_label_conflicts or human_transitive_conflicts),
        "human_identity_consistency_scope": "inspected_human_decisions_and_original_lesion_membership",
        "human_same_label_conflicts": human_label_conflicts,
        "human_transitive_identity_conflicts": human_transitive_conflicts,
        "human_inspected_unclear": human_inspected_unclear,
        "human_acceptance_recorded": False, "audit_limit": plan.get("audit_limit"),
        "script_sha256": file_sha256(Path(__file__)), "viewer_source_sha256": file_sha256(Path(viewer.__file__)),
        "source_applicability_limit": report.get("source_applicability_limit"), "completeness_limit": report.get("completeness_limit"),
        "broad_pairs_outside_approved_review": report.get("broad_pairs_outside_approved_review"),
        "required_pairs": [{"left": k[0], "right": k[1], "reasons": sorted(required[k])} for k in sorted(required_keys)],
        "limitations": ["AI observations and hash-bound human annotations are attestations, not authenticated biological identity.",
            "The 60 negative spot checks do not validate an error rate or rule out a single missed duplicate.",
            "Unclear physical identity remains unresolved. Separate conservative split-only grouping is not implemented here.",
            "No real split/freeze may proceed before explicit acceptance of audit, flags and participation."]}
    # Focused audit uses its own exact queue digest, retaining the full queue
    # digest separately. It cannot be passed off as final complete adjudication.
    ordered_keys = sorted(required_keys, key=lambda k: (k in human_rows,
                                                       "published_ambiguous_source_wording" not in required[k],
                                                       "agent_uncertain" not in required[k], k))
    # Explicit audit context in the derived CSV guarantees a different browser
    # storage key even when expansion eventually requests every original pair.
    audit_mark = "|ai_human_audit:" + POLICY + "@" + plan_digest + "@" + aggregate_digest
    focused_rows = [{**queue[k], "reasons": queue[k]["reasons"] + audit_mark,
                     "decision": "pending", "reviewer": "", "rationale": "", "evidence": ""} for k in ordered_keys]
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader();writer.writerows(focused_rows)
    focused_csv = csv_stream.getvalue().encode()
    decisions, urls = [], []
    for key in ordered_keys:
        view, observation = views[key], all_observations[key]
        secondary_note = secondary.get(key)
        rationale = observation["agent_rationale"]
        if secondary_note:
            rationale += " Secondary agent quality observation (not ground truth): " + secondary_note["agent_recommendation"] + ". " + secondary_note["agent_rationale"]
        decision = {"left": key[0], "right": key[1], "decision": "pending", "reviewer": "", "review_date": "",
            "rationale": "", "full_view_inspected": False, "evidence": view["view_file"], "view_sha256": view["view_sha256"],
            "source_payload_sha256": {i: by_id[i].sha256 for i in key},
            "source_applicability": {i:{"source":by_id[i].source,"source_version":by_id[i].source_version} for i in key},
            "view_repetition": "exact_content_repeat" if by_id[key[0]].sha256 == by_id[key[1]].sha256 else "byte_distinct_views",
            "screen_reasons": queue[key]["reasons"] + "|human_audit:" + "|".join(sorted(required[key])),
            "dhash_distance": int(queue[key]["dhash_distance"]), "agent_recommendation": observation["agent_recommendation"],
            "agent_rationale": rationale, "agent_recommendation_is_human_decision": False}
        # Carry only validated human annotations. Rebuild all source/view and
        # current audit evidence above; acceptance remains a separate blank gate.
        if key in human_rows:
            for field in ("decision", "reviewer", "review_date", "rationale", "full_view_inspected"):
                decision[field] = human_rows[key][field]
        decisions.append(decision)
        urls.append((index_path.parent / view["view_file"]).relative_to(root).as_posix())
    draft = {"manifest_sha256": bindings["manifest_sha256"], "procedure": RANKED_PROCEDURE, "audit_policy": POLICY,
        "candidate_queue_sha256": sha256_bytes(focused_csv), "full_candidate_queue_sha256": bindings["candidate_queue_sha256"],
        "screen_report_sha256": file_sha256(report_path), "agent_aggregate_sha256": aggregate_digest,
        "audit_plan_sha256": plan_digest, "visual_index": index_path.relative_to(root).as_posix(),
        "visual_index_sha256": bindings["visual_index_sha256"], "draft": True, "decisions": decisions,
        "human_acceptance": {"reviewer":"","date":"","approval_reference":"","accept_procedure":False,
                             "accept_identity_and_participation":False,"decision_sha256":""}}
    summary = {**{k:v for k,v in audit.items() if v is not None}, "review_candidates": len(decisions), "candidates_with_existing_full_resolution_views": len(decisions)}
    page = viewer.render_review_page(draft, urls, summary, [("Pending audit draft",outputs["draft"].name),
        ("Prespecified audit and flags",outputs["audit"].name), ("All AI/source observations",outputs["agent"].name)])
    payloads = {outputs["agent"]:canonical_json_bytes(aggregate),outputs["audit"]:canonical_json_bytes(audit),
        outputs["draft"]:canonical_json_bytes(draft),outputs["queue"]:focused_csv,outputs["page"]:page.encode()}
    for path, content in payloads.items():
        with path.open("xb") as handle: handle.write(content)
    snapshot.mkdir()
    for path in (Path(__file__), Path(viewer.__file__), audit_plan):
        shutil.copyfile(path,snapshot/path.name)
    return audit


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","screen-dir","visual-index","audit-plan","published18","source-agent-review","output-prefix"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--assignment",type=Path,action="append",required=True)
    parser.add_argument("--observations",type=Path,action="append",required=True)
    parser.add_argument("--human-audit",type=Path)
    parser.add_argument("--secondary-observations",type=Path)
    parser.add_argument("--pilot-only",action="store_true")
    args=parser.parse_args()
    try:
        result=prepare(args.manifest,args.screen_dir,args.visual_index,args.audit_plan,args.assignment,args.observations,
                       args.published18,args.source_agent_review,args.output_prefix,human_audit=args.human_audit,
                       pilot_only=args.pilot_only,secondary_observations=args.secondary_observations)
    except ContractError as exc:
        print(json.dumps({"status":"failed","error":exc.to_dict()}),file=sys.stderr);raise SystemExit(2)
    print(json.dumps({k:result[k] for k in ("status","new_observations","human_required_pairs","production_freeze")}))
