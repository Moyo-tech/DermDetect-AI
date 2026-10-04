"""Reuse verified pending views and merge exact active-review coverage.

Reads saved manifests, queue/index JSON and derived view bytes only. Never opens
archives, decodes images, edits preserved views or records identity decisions.
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, rooted_file
from dermdetect.data.review_policy import APPROVAL_REFERENCE, RANKED_PROCEDURE, RANKED_STATUS
from dermdetect.data.schemas import ContractError
from resolve_ham_analysis import VIEW_TRANSFORM, pair_key, read_json_object

VIEW_STATUS = "visual_evidence_prepared_not_adjudicated"


def safe_path(path, *, new=False):
    value = path.absolute()
    if ".." in value.parts or any(p.is_symlink() for p in (value, *value.parents)):
        raise ContractError("unsafe_path", "Review artifact routes cannot traverse or contain symlinks.")
    if new:
        if value.exists() or not value.parent.is_dir():
            raise ContractError("output_exists", "Use a new artifact path with an existing parent.")
    elif not value.is_file():
        raise ContractError("unreadable_file", "A saved ordinary artifact file is required.")
    return value.resolve()


def active_inputs(manifest, active_view_queue):
    manifest, active_view_queue = safe_path(manifest), safe_path(active_view_queue)
    rows = {row.image_id: row for row in read_manifest(manifest)}
    queue = read_json_object(active_view_queue)
    digest = file_sha256(manifest)
    if (queue.get("manifest_sha256") != digest or queue.get("procedure") != RANKED_PROCEDURE
            or queue.get("status") != RANKED_STATUS
            or queue.get("approval_reference") != APPROVAL_REFERENCE
            or queue.get("production_freeze") is not False
            or not isinstance(queue.get("candidates"), list)):
        raise ContractError("review_binding_mismatch", "Use the approved pending active-view queue for this manifest.")
    keys = []
    for candidate in queue["candidates"]:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("kind"), str) or not candidate["kind"].strip():
            raise ContractError("invalid_candidate", "Active queue entries need IDs and source kinds.")
        keys.append(pair_key(candidate.get("left"), candidate.get("right"), rows, code="invalid_candidate"))
        if candidate.get("decision", "pending") != "pending":
            raise ContractError("review_binding_mismatch", "View preparation does not inherit identity decisions.")
    if not keys or len(keys) != len(set(keys)):
        raise ContractError("candidate_membership_mismatch", "Each active candidate must occur exactly once.")
    return rows, queue, keys, digest


def verified_index(index_path, rows, manifest_sha256):
    index_path = safe_path(index_path)
    index = read_json_object(index_path)
    if (index.get("manifest_sha256") != manifest_sha256 or index.get("status") != VIEW_STATUS
            or index.get("production_freeze") is not False or not isinstance(index.get("pairs"), list)):
        raise ContractError("visual_evidence_mismatch", "Only pending full-resolution evidence for this inventory may be reused.")
    views, routes = {}, set()
    for view in index["pairs"]:
        if not isinstance(view, dict) or not isinstance(view.get("image_ids"), list) or len(view["image_ids"]) != 2:
            raise ContractError("visual_evidence_mismatch", "Index pair entries must name two images.")
        ids = view["image_ids"]
        key = pair_key(ids[0], ids[1], rows, code="visual_evidence_mismatch")
        if (key in views or view.get("source_payload_sha256") != [rows[i].sha256 for i in ids]
                or view.get("original_lesion_ids") != [rows[i].lesion_id for i in ids]
                or view.get("view_transform") != VIEW_TRANSFORM
                or view.get("decision") != "pending"
                or view.get("reviewer") != "" or view.get("rationale") != ""):
            raise ContractError("visual_evidence_mismatch", "Preserved view source/transform/pending bindings must remain unchanged.")
        reference, expected = view.get("view_file"), view.get("view_sha256")
        if (not isinstance(reference, str) or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise ContractError("visual_evidence_mismatch", "Each saved view needs a rooted file and canonical SHA-256.")
        # rooted_file validates canonical relative references; reject even
        # within-root symlinks before returning its resolved path.
        path = rooted_file(index_path.parent, reference)
        safe_path(index_path.parent / reference)
        if path in routes or file_sha256(path) != expected:
            raise ContractError("visual_evidence_mismatch", "View bytes changed or multiple pairs reuse one file.")
        routes.add(path)
        views[key] = (view, path)
    return index_path, index, views


def remaining_content(manifest, active_view_queue, existing_index):
    rows, queue, keys, manifest_digest = active_inputs(manifest, active_view_queue)
    old_path, old, views = verified_index(existing_index, rows, manifest_digest)
    if not set(views).issubset(keys):
        raise ContractError("candidate_membership_mismatch", "Preserved views must belong to the active review queue.")
    remaining = [candidate for candidate, key in zip(queue["candidates"], keys) if key not in views]
    result = {**queue, "candidates": remaining,
        "derivation_status": "remaining_views_pending_no_adjudication",
        "active_view_queue_sha256": file_sha256(active_view_queue),
        "preserved_visual_index_sha256": file_sha256(old_path),
        "script_sha256": file_sha256(Path(__file__)), "active_candidates": len(keys),
        "preserved_views": len(views), "remaining_views": len(remaining)}
    return result


def remaining(manifest, active_view_queue, existing_index, output_queue):
    output = safe_path(output_queue, new=True)
    result = remaining_content(manifest, active_view_queue, existing_index)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(result))
    return result


def merge(manifest, active_view_queue, existing_index, remaining_queue, new_index, output_index):
    output = safe_path(output_index, new=True)
    root = output.parent
    expected_remaining = remaining_content(manifest, active_view_queue, existing_index)
    remaining_path = safe_path(remaining_queue)
    if read_json_object(remaining_path) != expected_remaining:
        raise ContractError("review_binding_mismatch", "Remaining queue must exactly match the verified derivation.")
    rows, queue, keys, manifest_digest = active_inputs(manifest, active_view_queue)
    old_path, old, old_views = verified_index(existing_index, rows, manifest_digest)
    new_path, new, new_views = verified_index(new_index, rows, manifest_digest)
    remaining_keys = {pair_key(candidate["left"], candidate["right"], rows, code="invalid_candidate")
                      for candidate in expected_remaining["candidates"]}
    if (new.get("queue_sha256") != file_sha256(remaining_path)
            or set(new_views) != remaining_keys or set(old_views) & set(new_views)
            or set(old_views) | set(new_views) != set(keys)):
        raise ContractError("candidate_membership_mismatch", "Old and new views must cover the active queue exactly once.")
    all_views = {**old_views, **new_views}
    views, paths = [], set()
    for number, key in enumerate(keys, 1):
        view, path = all_views[key]
        if not path.is_relative_to(root) or path in paths:
            raise ContractError("unsafe_path", "Merged index must reside above all distinct source views.")
        paths.add(path)
        views.append({**view, "view_file": path.relative_to(root).as_posix(), "active_pair_number": number})
    sources = []
    for name, path, index, members in (("preserved", old_path, old, old_views), ("new", new_path, new, new_views)):
        if not path.is_relative_to(root):
            raise ContractError("unsafe_path", "Source indexes must remain beneath the merged index root.")
        sources.append({"role": name, "path": path.relative_to(root).as_posix(),
                        "sha256": file_sha256(path), "queue_sha256": index.get("queue_sha256"),
                        "pairs": len(members)})
    result = {"manifest_sha256": manifest_digest, "queue_sha256": file_sha256(active_view_queue),
        "procedure": queue["procedure"], "status": VIEW_STATUS, "production_freeze": False,
        "human_decisions_recorded": 0, "pairs": views, "source_indices": sources,
        "remaining_queue_sha256": file_sha256(remaining_path), "script_sha256": file_sha256(Path(__file__)),
        "preserved_views": len(old_views), "new_views": len(new_views),
        "view_preservation": "Source view files referenced unchanged; no PNGs copied, resized or regenerated."}
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("remaining", "merge"):
        sub = commands.add_parser(command)
        for name in ("manifest", "active-view-queue", "existing-index"):
            sub.add_argument("--" + name, type=Path, required=True)
        if command == "remaining":
            sub.add_argument("--output-queue", type=Path, required=True)
        else:
            for name in ("remaining-queue", "new-index", "output-index"):
                sub.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "remaining":
            result = remaining(args.manifest, args.active_view_queue, args.existing_index, args.output_queue)
            summary = {key: result[key] for key in ("derivation_status", "active_candidates", "preserved_views", "remaining_views")}
        else:
            result = merge(args.manifest, args.active_view_queue, args.existing_index, args.remaining_queue, args.new_index, args.output_index)
            summary = {key: result[key] for key in ("status", "preserved_views", "new_views", "production_freeze")}
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(summary))
