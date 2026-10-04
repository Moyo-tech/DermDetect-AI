"""Prepare accepted-audit analysis and a proposed constrained split for review.

Reads saved HAM integrity/review evidence only. No archive read, pixel decode,
development image export, freeze, sealing, model execution or outcome access.
The accepted raw export remains a draft; a separate hash-bound user approval
record authorizes this proposal without fabricating its human acceptance fields.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import io
import csv
import datetime
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermdetect.data.audited_analysis import POLICY, resolve_audited_analysis
from dermdetect.data.constrained_allocation import configuration, propose_constrained_groups, allocate_constrained_groups
from dermdetect.data.development_bundle import bundle_input_hashes
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, rooted_file, sha256_bytes
from dermdetect.data.review_policy import RANKED_PROCEDURE, validate_review_screen
from dermdetect.data.schemas import ContractError
from prepare_ham_ai_review import AUDIT_PLAN_SHA256
from resolve_ham_analysis import ANALYSIS_FIELDS, read_json_object, read_candidate_queue, verify_visual_review

ACCEPTANCE_POLICY = "accepted-audit-proposed-constrained-assignment-v1"
INPUTS = ("human_audit", "human_baseline", "agent_aggregate", "audit_report", "audit_plan",
          "manifest", "candidate_queue", "screen_report", "visual_index")
EDITABLE = {"decision", "reviewer", "review_date", "rationale", "full_view_inspected"}


def _csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANALYSIS_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        value = row.to_dict(); value["active"] = str(value["active"]).lower()
        writer.writerow(value)
    return stream.getvalue().encode()


def prepare(acceptance_path: Path, output: Path, *, evidence_root: Path = ROOT,
            protected_fingerprints=()) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite proposed analysis or assignment evidence.")
    evidence_root = evidence_root.resolve(strict=True)
    if not output.absolute().is_relative_to(evidence_root) or ".." in output.absolute().parts:
        raise ContractError("unsafe_path", "Proposal outputs must remain beneath the explicit evidence root.")
    if any(p.is_symlink() for p in (output.absolute(), *output.absolute().parents)):
        raise ContractError("unsafe_path", "Proposal routes cannot contain symlinks.")
    approval = read_json_object(acceptance_path)
    if (approval.get("policy") != ACCEPTANCE_POLICY
            or approval.get("accept_audit_decisions_and_revisions") is not True
            or approval.get("retain_all15_unclear") is not True
            or approval.get("authorize_proposed_analysis_and_assignment") is not True
            or approval.get("authorize_assignment_freeze") is not False
            or approval.get("authorize_development_payload_export") is not False
            or any(not isinstance(approval.get(k), str) or not approval[k].strip()
                   for k in ("approval_reference", "approval_date", "approver", "user_quote"))):
        raise ContractError("audit_acceptance_pending", "Explicit source-qualified audit/proposal approval with freeze/export withheld is required.")
    try:
        if datetime.date.fromisoformat(approval["approval_date"]).isoformat() != approval["approval_date"]:
            raise ValueError()
    except (TypeError, ValueError):
        raise ContractError("audit_acceptance_pending", "Approval needs an ISO calendar date.")
    paths, hashes = {}, {}
    for key in INPUTS:
        paths[key] = rooted_file(evidence_root, approval.get("input_paths", {}).get(key, ""))
        hashes[key] = file_sha256(paths[key])
        if approval.get("input_sha256", {}).get(key) != hashes[key]:
            raise ContractError("accepted_input_changed", "Accepted source/review bytes changed.", input=key)
    if hashes["audit_plan"] != AUDIT_PLAN_SHA256:
        raise ContractError("audit_plan_mismatch", "Use the unchanged prespecified audit plan.")
    records = read_manifest(paths["manifest"]); by_id = {r.image_id: r for r in records}
    screen, audit, aggregate, human, baseline, plan = (read_json_object(paths[k]) for k in
        ("screen_report", "audit_report", "agent_aggregate", "human_audit", "human_baseline", "audit_plan"))
    if validate_review_screen(screen, hashes["manifest"], hashes["candidate_queue"]) != RANKED_PROCEDURE:
        raise ContractError("review_binding_mismatch", "Use the approved ranked candidate procedure.")
    bindings = {"manifest_sha256": hashes["manifest"], "candidate_queue_sha256": hashes["candidate_queue"],
                "visual_index_sha256": hashes["visual_index"], "audit_plan_sha256": hashes["audit_plan"]}
    for value in (aggregate, audit):
        if any(value.get(k) != v for k, v in bindings.items()):
            raise ContractError("review_binding_mismatch", "Audit/observation inputs must bind the full accepted evidence.")
    if (any(plan.get(k) != v for k, v in bindings.items() if k != "audit_plan_sha256")
            or audit.get("agent_aggregate_sha256") != hashes["agent_aggregate"]
            or aggregate.get("production_freeze") is not False
            or audit.get("human_audit_complete") is not True
            or audit.get("human_identity_consistent") is not True
            or audit.get("production_freeze") is not False
            or audit.get("split_ready") is not False
            or human.get("full_candidate_queue_sha256") != hashes["candidate_queue"]
            or human.get("agent_aggregate_sha256") != hashes["agent_aggregate"]
            or any(human.get(k) != bindings[k] for k in ("manifest_sha256", "visual_index_sha256", "audit_plan_sha256"))):
        raise ContractError("human_audit_incomplete", "The accepted audit must be inspected, consistent and evidence-bound.")
    if ({k:v for k,v in human.items() if k != "decisions"}
            != {k:v for k,v in baseline.items() if k != "decisions"}):
        raise ContractError("review_binding_mismatch", "Raw submitted draft bindings must equal the preserved human baseline.")
    def index(rows):
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ContractError("invalid_review_artifact", "Pair records must be object arrays.")
        result = {(r.get("left"), r.get("right")): r for r in rows}
        if len(result) != len(rows):
            raise ContractError("candidate_membership_mismatch", "Duplicate pair entries are forbidden.")
        return result
    human_rows, base_rows = index(human.get("decisions")), index(baseline.get("decisions"))
    if human_rows.keys() != base_rows.keys():
        raise ContractError("candidate_membership_mismatch", "Human submission must cover the exact baseline queue.")
    for key, row in human_rows.items():
        if ({f:v for f,v in row.items() if f not in EDITABLE}
                != {f:v for f,v in base_rows[key].items() if f not in EDITABLE}):
            raise ContractError("visual_evidence_mismatch", "Immutable human source/view evidence changed.")
    required = audit.get("required_pairs")
    required_rows = index(required)
    if (set(human_rows) != set(required_rows)
            or audit.get("human_required_pairs") != len(human_rows)
            or audit.get("human_audit_rows_recorded_as_attestations") != len(human_rows)):
        raise ContractError("human_audit_incomplete", "All current required audit pairs must be inspected.")
    queue = read_candidate_queue(paths["candidate_queue"], by_id)
    observations = aggregate.get("pairs")
    observed = index(observations)
    if len(queue) != screen.get("review_candidates") or len(observed) != len(queue):
        raise ContractError("candidate_membership_mismatch", "The full selected queue must be preserved.")
    visual_rows = []
    for key, row in observed.items():
        if any(i not in by_id for i in key):
            raise ContractError("candidate_membership_mismatch", "Observed image IDs must be inventory members.")
        visual_rows.append({"left":key[0], "right":key[1], "evidence":row.get("view_file"),
            "view_sha256":row.get("view_sha256"), "source_payload_sha256":row.get("source_payload_sha256"),
            "source_applicability":{i:{"source":by_id[i].source,"source_version":by_id[i].source_version} for i in key},
            "view_repetition":"exact_content_repeat" if by_id[key[0]].sha256==by_id[key[1]].sha256 else "byte_distinct_views"})
    verify_visual_review(records, visual_rows,
        {"visual_index":paths["visual_index"].relative_to(evidence_root).as_posix(), "visual_index_sha256":hashes["visual_index"]},
        evidence_root/"context-only-not-written.json",hashes["manifest"])
    resolved = resolve_audited_analysis(records, candidate_pairs=queue, agent_observations=observations,
        human_decisions=list(human_rows.values()), required_human_pairs=required)
    if (audit.get("human_inspected_unclear") != sum(r["decision"] == "pending" for r in human_rows.values())
            or plan.get("policy") != "m2-ai-assisted-audit-v1"
            or plan.get("audit_seed") != 20260729):
        raise ContractError("audit_plan_mismatch", "Prespecified plan and retained-uncertainty counts must agree.")
    grouping = resolved.grouping
    partitions, verification, search = propose_constrained_groups(grouping)
    # Independent input-order replay must produce exactly the same membership.
    replay = allocate_constrained_groups(grouping)
    if replay != partitions:
        raise ContractError("nondeterministic_allocation", "Repeated allocation must reproduce every record.")
    code_names = ("audited_analysis.py", "constrained_allocation.py", "conservative_groups.py", "development_bundle.py")
    code = {"data/"+name:(ROOT/"src/dermdetect/data"/name).read_bytes() for name in code_names}
    integrity_path = paths["manifest"].parent/"integrity_report.json"
    decode_path = rooted_file(evidence_root, "outputs/m2-image-integrity/decode_checkpoint.json")
    integrity, decode = read_json_object(integrity_path), read_json_object(decode_path)
    archive_hashes = {r.get("alias"):r.get("sha256") for r in integrity.get("archives", []) if isinstance(r, dict)}
    if (integrity.get("manifest_sha256") != hashes["manifest"]
            or integrity.get("archive_crc") != "passed_for_all_referenced_members"
            or set(archive_hashes) != {"part1", "part2"}
            or any(not re.fullmatch(r"[0-9a-f]{64}", str(v)) for v in archive_hashes.values())
            or not re.fullmatch(r"[0-9a-f]{64}", str(integrity.get("metadata_sha256", "")))
            or decode.get("manifest_sha256") != hashes["manifest"]
            or decode.get("archive_sha256") != archive_hashes
            or decode.get("records_decoded") != len(records)
            or decode.get("status") != "all_listed_jpegs_decoded_candidate_screen_pending"):
        raise ContractError("integrity_evidence_mismatch", "Inventory and preserved all-image decode evidence must bind these sources.")
    source_fingerprints = (*archive_hashes.values(), integrity["metadata_sha256"],
                           file_sha256(integrity_path), file_sha256(decode_path))
    protected = tuple(dict.fromkeys((*protected_fingerprints, *source_fingerprints,
                                    *hashes.values(), file_sha256(acceptance_path))))
    preview = bundle_input_hashes(records=grouping.records, partitions=partitions, grouping=grouping,
                                 code_files=code, protected_content_sha256=protected)
    payloads = {"physical_analysis_manifest.csv":_csv(grouping.physical_records),
                "analysis_manifest.csv":_csv(grouping.records),
                "effective_decisions.json":canonical_json_bytes(resolved.effective_decisions),
                "uncertainty_links.json":canonical_json_bytes([asdict(link) for link in grouping.uncertainties]),
                "allocation_config.json":canonical_json_bytes(configuration()),
                **{name+".csv":_csv(rows) for name,rows in partitions.items()}}
    result = {"status":"accepted_analysis_and_proposed_assignment_verified_not_frozen", "policy":POLICY,
        "accepted_input_sha256":hashes, "acceptance_sha256":file_sha256(acceptance_path),
        "approval_reference":approval["approval_reference"], "approval_kind":"source-qualified user attestation, not software authentication",
        "effective_decision_counts":dict(Counter(r["decision"] for r in resolved.effective_decisions)),
        "effective_attribution_counts":dict(Counter(r["decision_attribution"] for r in resolved.effective_decisions)),
        "unclear_pairs_retained":len(grouping.uncertainties), "original_records":len(grouping.records),
        "original_lesion_ids":len({r.original.lesion_id for r in grouping.records}),
        "analysis_lesions":len({r.analysis_lesion_id for r in grouping.records}),
        "active_views":sum(r.active for r in grouping.records), "inactive_exact_copies":sum(not r.active for r in grouping.records),
        "allocation_verification":verification, "allocation_search":search,
        "deterministic_membership_replay":True, "source_labels_and_original_records_unchanged":True,
        "bundle_adapter_preview":{"input_hashes":preview,"code_scope":"data-contract source preview; not final development worker/runtime",
            "image_payloads_exported":False,"export_approval_present":False,"production_seal_validated":False,
            "protected_source_and_evidence_fingerprints":protected},
        "inventory_report_sha256":file_sha256(integrity_path), "decode_checkpoint_sha256":file_sha256(decode_path),
        "artifact_sha256":{name:sha256_bytes(content) for name,content in payloads.items()},
        "script_sha256":file_sha256(Path(__file__)),
        "source_sha256":{name:sha256_bytes(content) for name,content in code.items()},
        "broad_outside_pairs_unadjudicated":screen.get("broad_pairs_outside_approved_review"),
        "broad_outside_pairs_status":"unadjudicated; never known negatives",
        "numerical_weighting_resampling_approved":False,"final_assignment_accepted":False,
        "production_freeze":False,"production_seal_validated":False,"training_or_outcome_access":False,
        "remaining_gates":["review proposed counts/membership and final assignment","settle mixed-label dependence/weighting numerical policy",
                           "approve/verify real development composition and runtime isolation before production freeze"]}
    payloads["proposal_report.json"] = canonical_json_bytes(result)
    output.mkdir(parents=True, exist_ok=False)
    for name,content in payloads.items():
        with (output/name).open("xb") as handle:handle.write(content)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=prepare(args.acceptance,args.output_dir)
    except ContractError as exc:
        import json
        print(json.dumps({"status":"failed","error":exc.to_dict()}),file=sys.stderr);raise SystemExit(2)
    import json
    print(json.dumps({k:result[k] for k in ("status","original_records","analysis_lesions","active_views","unclear_pairs_retained","production_freeze")}))
