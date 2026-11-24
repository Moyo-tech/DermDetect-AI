"""Resolve preserved inventory only after complete, hash-bound human review.

Never assigns partitions or reads images. Approval references are attestations,
not authentication; no claim of personal human inspection is inferred by code.
"""
import argparse
import csv
import datetime
import io
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data import analysis
from dermdetect.data.analysis import CandidateDecision, POLICY, resolve_analysis
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, rooted_file, sha256_bytes
from dermdetect.data.review_policy import RANKED_PROCEDURE, validate_review_screen
from dermdetect.data.schemas import ContractError, MANIFEST_FIELDS

ANALYSIS_FIELDS = (*MANIFEST_FIELDS, "analysis_lesion_id", "split_group_id", "active",
                   "representative_image_id", "inactive_reason")
VIEW_TRANSFORM = "EXIF transpose/RGB; full resolution; no crop/resize"
QUEUE_FIELDS = ("left", "right", "dhash_distance", "reasons", "decision", "reviewer", "rationale", "evidence")


def read_json_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("invalid_review_artifact", "Review artifacts must be readable JSON objects.", path=str(path)) from exc
    if not isinstance(value, dict):
        raise ContractError("invalid_review_artifact", "Review artifacts must be JSON objects.", path=str(path))
    return value


def pair_key(left, right, by_id, *, code):
    if (any(not isinstance(value, str) or not value or value != value.strip()
            for value in (left, right)) or left == right
            or left not in by_id or right not in by_id):
        raise ContractError(code, "Each pair must name two distinct canonical inventory IDs.")
    return tuple(sorted((left, right)))


def read_candidate_queue(path, by_id):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(QUEUE_FIELDS):
                raise ContractError("invalid_candidate_queue", "Candidate CSV must use the exact versioned eight-column header.")
            result = []
            for row in reader:
                if None in row or any(not isinstance(value, str) for value in row.values()):
                    raise ContractError("invalid_candidate_queue", "Candidate CSV rows must contain exactly eight values.")
                pair_key(row["left"], row["right"], by_id, code="invalid_candidate_queue")
                if (not re.fullmatch(r"(?:0|[1-9][0-9]*)", row["dhash_distance"])
                        or int(row["dhash_distance"]) > 64 or not row["reasons"].strip()):
                    raise ContractError("invalid_candidate_queue", "Candidate distance/reason fields are invalid.")
                result.append(row)
            return result
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("invalid_candidate_queue", "Candidate CSV is unreadable or malformed.") from exc


def verify_visual_review(records, decisions, review, review_path, manifest_sha256):
    """Bind each human decision to preserved, full-resolution view evidence.

    Hashes establish artifact correspondence, not authenticity of a reviewer's
    identity or proof that the reviewer inspected a view. Those remain recorded
    human attestations. Never accept an arbitrary nonblank evidence string.
    """
    reference = review.get("visual_index")
    if not isinstance(reference, str) or not reference:
        raise ContractError("missing_visual_evidence", "A rooted visual index and recorded hash are required.")
    index_path = rooted_file(review_path.parent, reference)
    digest = file_sha256(index_path)
    if digest != review.get("visual_index_sha256"):
        raise ContractError("visual_evidence_mismatch", "Visual index differs from its recorded hash.")
    index = read_json_object(index_path)
    if (index.get("manifest_sha256") != manifest_sha256
            or index.get("status") != "visual_evidence_prepared_not_adjudicated"
            or not isinstance(index.get("pairs"), list)):
        raise ContractError("visual_evidence_mismatch", "Visual index must match the original inventory.")
    by_id = {r.image_id: r for r in records}
    indexed = {}
    view_routes = {}
    for view in index["pairs"]:
        if not isinstance(view, dict):
            raise ContractError("visual_evidence_mismatch", "Visual-index pair entries must be objects.")
        ids = view.get("image_ids")
        if not isinstance(ids, list) or len(ids) != 2:
            raise ContractError("visual_evidence_mismatch", "View must name two inventory images.")
        key = pair_key(ids[0], ids[1], by_id, code="visual_evidence_mismatch")
        if key in indexed:
            raise ContractError("visual_evidence_mismatch", "Repeated visual-index pairs are ambiguous.")
        if (view.get("source_payload_sha256") != [by_id[i].sha256 for i in ids]
                or view.get("view_transform") != VIEW_TRANSFORM):
            raise ContractError("visual_evidence_mismatch", "View hashes and full-resolution transform must match.")
        route = view.get("view_file")
        if (not isinstance(route, str) or not route
                or not isinstance(view.get("view_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", view["view_sha256"])):
            raise ContractError("missing_visual_evidence", "Each view needs a relative route and canonical SHA-256.")
        if route in view_routes:
            raise ContractError("visual_evidence_mismatch", "Different pairs cannot reuse one visual file.")
        view_routes[route] = key
        indexed[key] = view
    verified_files = {}
    for item in decisions:
        key = tuple(sorted((item["left"], item["right"])))
        view = indexed.get(key)
        if view is None:
            raise ContractError("missing_visual_evidence", "Every candidate decision needs its full-resolution view.", image_ids=key)
        expected_sources = {i: {"source": by_id[i].source, "source_version": by_id[i].source_version} for i in key}
        expected_payloads = {i: by_id[i].sha256 for i in key}
        repetition = "exact_content_repeat" if by_id[key[0]].sha256 == by_id[key[1]].sha256 else "byte_distinct_views"
        if (item.get("source_applicability") != expected_sources
                or item.get("source_payload_sha256") != expected_payloads
                or item.get("view_repetition") != repetition
                or item.get("evidence") != view.get("view_file")
                or item.get("view_sha256") != view.get("view_sha256")):
            raise ContractError("visual_evidence_mismatch", "Decision must bind source/version, payloads, view and repetition status.", image_ids=key)
        reference = view.get("view_file")
        if not isinstance(reference, str):
            raise ContractError("missing_visual_evidence", "A rooted view file is required.")
        if reference not in verified_files:
            verified_files[reference] = file_sha256(rooted_file(index_path.parent, reference))
        if verified_files[reference] != view.get("view_sha256"):
            raise ContractError("visual_evidence_mismatch", "Full-resolution view bytes changed.", image_ids=key)
    return digest


def resolve(manifest: Path, screen_dir: Path, review_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite analysis evidence.")
    report_path = screen_dir / "screen_report.json"
    queue_path = screen_dir / "candidates.csv"
    report = read_json_object(report_path)
    review = read_json_object(review_path)
    try:
        records = read_manifest(manifest)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("invalid_review_artifact", "Inventory manifest is unreadable or malformed.") from exc
    by_id = {row.image_id: row for row in records}
    bindings = {"manifest_sha256": file_sha256(manifest),
                "candidate_queue_sha256": file_sha256(queue_path),
                "screen_report_sha256": file_sha256(report_path)}
    bindings["procedure"] = validate_review_screen(report, bindings["manifest_sha256"], bindings["candidate_queue_sha256"])
    if any(review.get(k) != v for k, v in bindings.items()):
        raise ContractError("review_binding_mismatch", "Review must bind the complete queue and inventory.")
    queue = read_candidate_queue(queue_path, by_id)
    keys = [pair_key(c["left"], c["right"], by_id, code="invalid_candidate_queue") for c in queue]
    if (type(report.get("review_candidates")) is not int or report["review_candidates"] < 0
            or len(keys) != len(set(keys)) or len(keys) != report["review_candidates"]):
        raise ContractError("candidate_membership_mismatch", "Complete candidate membership is inconsistent.")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise ContractError("invalid_review", "A complete decision list is required.")
    if any(not isinstance(item, dict) for item in decisions):
        raise ContractError("invalid_review", "Each decision must be a JSON object.")
    decision_keys = [pair_key(c.get("left"), c.get("right"), by_id, code="invalid_review") for c in decisions]
    if len(decision_keys) != len(set(decision_keys)) or set(decision_keys) != set(keys):
        raise ContractError("candidate_membership_mismatch", "Review every candidate exactly once; no filtering.")
    for item in decisions:
        try:
            date = item["review_date"]
            if datetime.date.fromisoformat(date).isoformat() != date:
                raise ValueError()
        except (KeyError, ValueError, TypeError):
            raise ContractError("missing_review_date", "Each decision needs an ISO review date.")
    if bindings["procedure"] == RANKED_PROCEDURE and any(
            item.get("full_view_inspected") is not True for item in decisions):
        raise ContractError("missing_full_view_attestation", "Each selected pair requires recorded full-resolution inspection.")
    # This underlying contract stops on pending identity, label conflicts,
    # transitive contradictions, missing evidence and cross-ID exact ambiguity.
    visual_digest = verify_visual_review(records, decisions, review, review_path, bindings["manifest_sha256"])
    resolved = resolve_analysis(records, [CandidateDecision(**{k: c.get(k, "") for k in
        ("left", "right", "evidence", "decision", "reviewer", "rationale")}) for c in decisions])
    acceptance = review.get("human_acceptance", {})
    if not isinstance(acceptance, dict):
        raise ContractError("human_acceptance_pending", "Human acceptance must be a recorded JSON object.")
    digest = sha256_bytes(canonical_json_bytes(decisions))
    if (review.get("draft") is not False
            or acceptance.get("decision_sha256") != digest
            or acceptance.get("accept_procedure") is not True
            or acceptance.get("accept_identity_and_participation") is not True
            or any(not isinstance(acceptance.get(k), str) or not acceptance[k].strip()
                   for k in ("reviewer", "date", "approval_reference"))):
        raise ContractError("human_acceptance_pending", "Record explicit human acceptance of this procedure and these decisions.")
    try:
        if datetime.date.fromisoformat(acceptance["date"]).isoformat() != acceptance["date"]:
            raise ValueError()
    except (TypeError, ValueError):
        raise ContractError("missing_review_date", "Human acceptance needs an ISO date.")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ANALYSIS_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in resolved:
        value = row.to_dict()
        value["active"] = str(value["active"]).lower()
        writer.writerow(value)
    content = stream.getvalue().encode()
    result = {**bindings, "status": "analysis_resolved_no_assignment_or_freeze", "policy": POLICY,
              "original_records": len(resolved), "active_views": sum(r.active for r in resolved),
              "inactive_exact_copies": sum(not r.active for r in resolved),
              "original_lesion_ids": len({r.original.lesion_id for r in resolved}),
              "analysis_lesions": len({r.analysis_lesion_id for r in resolved}),
              "decision_count": len(decisions), "decision_sha256": digest,
              "review_sha256": file_sha256(review_path), "human_acceptance": acceptance,
              "human_acceptance_kind": "recorded_attestation_not_authenticated_by_software",
              "analysis_manifest_sha256": sha256_bytes(content),
              "script_sha256": file_sha256(Path(__file__)),
              "analysis_source_sha256": file_sha256(Path(analysis.__file__)),
              "visual_index_sha256": visual_digest, "production_freeze": False}
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis_manifest.csv").write_bytes(content)
    (output / "analysis_report.json").write_bytes(canonical_json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "screen-dir", "review", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        report = resolve(args.manifest, args.screen_dir, args.review, args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(report))
