"""Recognized review procedures; approval references are recorded attestations."""
from .candidate_screen import PROCEDURE as BROAD_PROCEDURE
from .schemas import ContractError

RANKED_PROCEDURE = "published-top1000-plus-equal-dhash-review-v1"
RANKED_STATUS = "approved_review_queue_adjudication_pending"
APPROVAL_REFERENCE = "Sentinel_0d21a1d9e7788191b024bff15c5920e5"


def validate_review_screen(report: dict, manifest_sha256: str, candidates_sha256: str) -> str:
    """Recognize an unchanged pending queue, never authenticate human identity."""
    if (report.get("manifest_sha256") != manifest_sha256
            or report.get("candidates_sha256") != candidates_sha256
            or report.get("production_freeze") is not False
            or report.get("split_ready") is not False):
        raise ContractError("review_binding_mismatch", "Review queue must bind this inventory and remain unfrozen.")
    procedure = report.get("procedure")
    if procedure == BROAD_PROCEDURE and report.get("status") == "enumeration_complete_adjudication_pending":
        return procedure
    if (procedure == RANKED_PROCEDURE and report.get("status") == RANKED_STATUS
            and report.get("procedure_approval") == "approved_candidate_selection_only"
            and report.get("approval_reference") == APPROVAL_REFERENCE
            and report.get("human_decisions_recorded") == 0
            and report.get("broad_outside_status") == "unadjudicated"):
        return procedure
    raise ContractError("review_binding_mismatch", "Review procedure/status or recorded selection approval is invalid.")
