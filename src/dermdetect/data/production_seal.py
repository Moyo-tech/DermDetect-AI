"""Fail-closed M2 CPU seal assembly from independently hash-bound evidence.

This is an evaluator record, not a development input or owner-proof barrier.
File/runtime/boundary verification is performed by the closeout driver first.
"""
import re

POLICY = "approved-ham-physical-split-cpu-production-seal-v1"
REQUIRED_BINDINGS = ("approval", "approved_package", "proposal_report", "numerical_policy",
                     "code_manifest", "runtime_inventory", "bundle_manifest",
                     "evaluator_provenance", "boundary_profile", "boundary_config",
                     "worker_result", "boundary_run", "composition", "preservation", "tests")
AUTHORIZATIONS = ("accept_exact_assignment", "approve_numerical_policy",
                  "authorize_train_validation_export", "authorize_cpu_boundary_verification",
                  "authorize_freeze_only_after_all_checks_pass")


def assemble_seal(*, approval, proposal, tests, composition, worker, boundary,
                  preservation, bindings, paths, recorded_utc):
    if any(approval.get(k) is not True for k in AUTHORIZATIONS):
        raise ValueError("Explicit final approval and conditional freeze authority required")
    if any(approval.get(k) is not False for k in
           ("training_authorized", "outcome_access_authorized", "commit_push_authorized")):
        raise ValueError("M2 closeout cannot imply later experiment/commit authority")
    if (set(bindings) != set(REQUIRED_BINDINGS)
            or any(not isinstance(h,str) or not re.fullmatch(r"[0-9a-f]{64}",h) for h in bindings.values())
            or approval.get("proposal_report_sha256") != bindings["proposal_report"]
            or approval.get("approved_package_sha256") != bindings["approved_package"]):
        raise ValueError("Exact complete approval/evidence hash bindings required")
    if (proposal.get("production_freeze") is not False
            or proposal.get("source_labels_and_original_records_unchanged") is not True
            or proposal.get("deterministic_membership_replay") is not True
            or proposal.get("unclear_pairs_retained") != 15
            or proposal.get("analysis_lesions") != 7449
            or proposal.get("active_views") != 10013
            or proposal.get("original_records") != 10015):
        raise ValueError("The exact approved preserved proposal is required")
    if (tests.get("returncode") != 0 or type(tests.get("passed")) is not int
            or tests["passed"] < 215 or tests.get("input_scope") != "invented fixtures only"):
        raise ValueError("Current complete passing synthetic test evidence required")
    if (composition.get("status") != "production_composition_verified"
            or composition.get("active_development_payloads") != 8505
            or composition.get("no_protected_membership_or_payloads") is not True
            or composition.get("all_payload_sha256_passed") is not True):
        raise ValueError("Exact development composition verification required")
    expected = {"train": (7000,5214,5203), "validation": (1505,1118,1117)}
    if worker.get("status") != "development_cpu_contracts_verified":
        raise ValueError("Actual final development worker must pass")
    for split,(images,physical,clusters) in expected.items():
        observed=worker.get("partitions",{}).get(split,{})
        if (observed.get("active_views"),observed.get("physical_units"),observed.get("conservative_groups")) != (images,physical,clusters):
            raise ValueError("Worker membership differs from approved split")
        if observed.get("class_physical_counts") != {
            c:v["physical_units"] for c,v in proposal["allocation_verification"]["counts"][split].items()}:
            raise ValueError("Worker class quotas differ from approved split")
    if (worker.get("weight_sum") != "7000" or set(worker.get("class_loss_mass",{}).values()) != {"1/7"}
            or len(worker.get("class_loss_mass",{})) != 7 or worker.get("validation_paired_draws") != 1000
            or worker.get("predictions_or_outcomes_read") is not False):
        raise ValueError("Approved real-membership numerical contract checks required")
    if (boundary.get("status") != "production_cpu_boundary_passed" or boundary.get("returncode") != 0
            or boundary.get("bundle_runtime_unchanged") is not True
            or boundary.get("fresh_worker_launch") is not True
            or boundary.get("inherited_source_descriptors") != 0
            or worker.get("output_write_allowed") is not True
            or not worker.get("boundary_checks")
            or not all(v is True for v in worker["boundary_checks"].values())):
        raise ValueError("Actual final enforced access boundary evidence required")
    required_checks = {"network_denied","helper_exec_denied","write_code_denied","write_runtime_denied",
                       "write_evaluator_denied","source_alias_denied","test_membership_alias_denied",
                       "future_outcome_sentinel_denied","future_image_sentinel_denied"}
    if not required_checks <= worker["boundary_checks"].keys():
        raise ValueError("Production denial coverage incomplete")
    if (preservation.get("status") != "accepted_sources_review_and_proposal_preserved"
            or preservation.get("exact_approved_membership_replayed") is not True
            or preservation.get("zero_overlap_dimensions") != 5
            or preservation.get("all15_unclear_retained_and_colocated") is not True
            or preservation.get("original_source_hashes_unchanged") is not True
            or preservation.get("raw_exports_unchanged") is not True):
        raise ValueError("Final source/review/proposal preservation checks required")
    if not isinstance(recorded_utc,str) or not recorded_utc or not paths:
        raise ValueError("Seal provenance required")
    return {"policy":POLICY,"status":"m2_complete_cpu_split_and_access_boundary_frozen",
        "recorded_utc":recorded_utc,"production_freeze":True,"production_seal_validated":True,
        "final_assignment_accepted":True,"numerical_policy_approved":True,"m2_complete":True,
        "training_authorized":False,"outcome_access_authorized":False,"commit_push_authorized":False,
        "evidence_sha256":dict(bindings),"paths":dict(paths),"proposal_artifact_sha256":proposal["artifact_sha256"],
        "partition_physical_units":{"train":5214,"validation":1118,"internal_test":1117},
        "original_records":10015,"active_views":10013,"retained_unclear_pairs":15,
        "outside_broad_pairs_unadjudicated":30072,"tested_runtime_scope":"inspected standard-library CPU data worker only",
        "limitations":["Machine owner and processes outside the profile retain access; no owner-proof or hostile-code isolation claim.",
                       "A changed code/runtime/input composition requires renewed verification before use.",
                       "Recorded uncertainty groups do not establish physical/patient independence or duplicate completeness.",
                       "No model runtime, training, predictions, internal-test/DDI outcomes or commit are authorized."]}
