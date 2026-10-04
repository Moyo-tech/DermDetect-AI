"""Apply approved candidate selection using saved text/fingerprints only.

Approval is limited to the candidate procedure; no image identity, split,
freeze or outcomes are approved here. Preserve the broad unadjudicated audit.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data import candidate_screen, image_integrity, manifests, review_policy, schemas
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, sha256_bytes
from dermdetect.data.review_policy import APPROVAL_REFERENCE, RANKED_PROCEDURE, RANKED_STATUS, validate_review_screen
from dermdetect.data.schemas import ContractError
import complete_ham_candidate_screen as complete_screen
import prepare_ham_candidate_review as published_review
import prepare_ham_ranked_proposal as ranked_proposal


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=complete_screen.FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def prepare(manifest: Path, screen_dir: Path, checkpoint: Path, checkpoint_sha256: str,
            ranking: Path, published_queue: Path, ranking_source: Path,
            published_source_dir: Path, output: Path, *, approval_reference: str) -> dict:
    if approval_reference != APPROVAL_REFERENCE:
        raise ContractError("procedure_approval_required", "Require the exact recorded candidate-selection approval reference.")
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite review evidence.")
    broad_report_path = screen_dir / "screen_report.json"
    broad_queue_path = screen_dir / "candidates.csv"
    broad_matches_path = screen_dir / "hash_matches.csv"
    report = json.loads(broad_report_path.read_text())
    manifest_digest = file_sha256(manifest)
    validate_review_screen(report, manifest_digest, file_sha256(broad_queue_path))
    if report.get("procedure") != review_policy.BROAD_PROCEDURE:
        raise ContractError("review_binding_mismatch", "Select only from the preserved complete broad-screen evidence.")
    source = json.loads(ranking_source.read_text())
    if (file_sha256(ranking) != ranked_proposal.RANK_SHA256
            or file_sha256(published_queue) != ranked_proposal.PUBLISHED_SHA256
            or source.get("sha256") != ranked_proposal.RANK_SHA256
            or source.get("commit") != ranked_proposal.COMMIT
            or type(source.get("bytes")) is not int or source["bytes"] != ranking.stat().st_size):
        raise ContractError("candidate_source_mismatch", "Ranking and public source record must match their pinned revision/bytes.")
    if (file_sha256(checkpoint) != checkpoint_sha256
            or checkpoint_sha256 != report.get("checkpoint_sha256")
            or report.get("published_queue_sha256") != ranked_proposal.PUBLISHED_SHA256
            or report.get("decoder_source_sha256") != file_sha256(Path(image_integrity.__file__))
            or report.get("screen_source_sha256") != file_sha256(Path(candidate_screen.__file__))
            or report.get("script_sha256") != file_sha256(Path(complete_screen.__file__))
            or report.get("published_preparation_source_sha256") != file_sha256(Path(published_review.__file__))
            or report.get("hash_matches_sha256") != file_sha256(broad_matches_path)):
        raise ContractError("review_binding_mismatch", "Broad decoding/source/procedure bindings must remain unchanged.")
    inventory_path = screen_dir / "input_snapshot" / "inventory_report.json"
    # Broad screen records only its inventory digest; its original location is
    # explicit from the adjacent manifest inventory directory, never inferred
    # from a raw source path or silently replaced with new inventory values.
    if not inventory_path.exists():
        inventory_path = manifest.parent / "integrity_report.json"
    if file_sha256(inventory_path) != report.get("inventory_report_sha256"):
        raise ContractError("review_binding_mismatch", "Inventory report differs from the completed broad run.")
    with tempfile.TemporaryDirectory(prefix="dermdetect-ranked-review-") as temporary:
        replay_dir = Path(temporary) / "replayed_broad"
        replay = complete_screen.prepare(manifest, inventory_path, checkpoint, checkpoint_sha256,
            published_queue, replay_dir, published_sha256=ranked_proposal.PUBLISHED_SHA256,
            published_source_dir=published_source_dir)
        for key in ("candidates_sha256", "hash_matches_sha256", "review_candidates", "all_hash_matches",
                    "manifest_sha256", "archive_sha256", "local_config_sha256"):
            if report.get(key) != replay.get(key):
                raise ContractError("candidate_membership_mismatch", "Broad evidence must contain the entire unchanged replayed screen.")
        proposed = ranked_proposal.prepare(manifest, screen_dir, ranking, published_queue,
                                          Path(temporary) / "selection.json")
    rows = {r.image_id: r for r in read_manifest(manifest)}
    fingerprints = json.loads(checkpoint.read_text())["fingerprints"]
    selected = []
    selected_keys = set()
    for candidate in proposed["candidates"]:
        left, right = candidate["left"], candidate["right"]
        if left not in rows or right not in rows or left not in fingerprints or right not in fingerprints:
            raise ContractError("candidate_correspondence_failure", "Every selected pair requires inventory and checkpoint lookup.")
        reasons = set(candidate["kind"].split("|"))
        if fingerprints[left]["rgb_sha256"] == fingerprints[right]["rgb_sha256"]:
            reasons.add("identical_oriented_decoded_rgb")
        if rows[left].sha256 == rows[right].sha256:
            reasons.add("identical_payload_sha256")
        selected.append(dict(zip(complete_screen.FIELDS, (left, right,
            (fingerprints[left]["dhash64"] ^ fingerprints[right]["dhash64"]).bit_count(),
            "|".join(sorted(reasons)), "pending", "", "", ""))))
        selected_keys.add((left, right))
    with broad_queue_path.open(newline="") as handle:
        outside = [dict(row, decision="unadjudicated") for row in csv.DictReader(handle)
                   if (row["left"], row["right"]) not in selected_keys]
    view_queue = {"manifest_sha256": manifest_digest, "procedure": RANKED_PROCEDURE,
        "status": RANKED_STATUS, "approval_reference": approval_reference, "production_freeze": False,
        "candidates": [{"left": r["left"], "right": r["right"], "kind": r["reasons"]} for r in selected]}
    source_paths = {"manifest.csv": manifest, "decode_checkpoint.json": checkpoint,
        "broad_candidates.csv": broad_queue_path, "broad_hash_matches.csv": broad_matches_path,
        "broad_screen_report.json": broad_report_path, "inventory_report.json": inventory_path,
        "duplicates_1000.csv": ranking, "ranking_source.json": ranking_source,
        "published_queue.json": published_queue, "published_source.json": published_source_dir / "source.json"}
    source_paths.update({name: published_source_dir / name for name in published_review.SOURCES})
    code_paths = {Path(module.__file__).name: Path(module.__file__) for module in
        (complete_screen, ranked_proposal, published_review, candidate_screen, image_integrity,
         manifests, review_policy, schemas)}
    code_paths[Path(__file__).name] = Path(__file__)
    result = {**{k: report[k] for k in ("manifest_sha256", "archive_sha256", "local_config_sha256", "decoder_source_sha256")},
        "procedure": RANKED_PROCEDURE, "status": RANKED_STATUS,
        "procedure_approval": "approved_candidate_selection_only", "approval_reference": approval_reference,
        "approval_kind": "recorded_human_instruction_not_authenticated_by_software",
        "production_freeze": False, "split_ready": False, "human_decisions_recorded": 0,
        "review_candidates": len(selected), "pending_decisions": len(selected),
        "candidates_sha256": sha256_bytes(csv_bytes(selected)),
        "view_queue_sha256": sha256_bytes(canonical_json_bytes(view_queue)),
        "broad_outside_sha256": sha256_bytes(csv_bytes(outside)), "broad_outside_status": "unadjudicated",
        "broad_pairs_outside_approved_review": len(outside),
        "broad_review_candidates": report["review_candidates"],
        "independent_ranked_pairs_absent_from_broad_queue": proposed["independent_ranked_pairs_absent_from_broad_queue"],
        "ranked_cross_id_pairs": proposed["ranked_cross_id_pairs"],
        "equal_dhash_cross_id_pairs": proposed["equal_dhash_cross_id_pairs"],
        "mandatory_pairs": proposed["mandatory_pairs"], "checkpoint_sha256": checkpoint_sha256,
        "broad_queue_sha256": file_sha256(broad_queue_path), "broad_report_sha256": file_sha256(broad_report_path),
        "ranking_sha256": file_sha256(ranking), "public_candidate_commit": ranked_proposal.COMMIT,
        "published_queue_sha256": file_sha256(published_queue),
        "source_applicability_limit": proposed["source_applicability_limit"],
        "completeness_limit": "Approved sensitivity-limited selection; no proof of unknown-duplicate completeness. Broad outside pairs remain unadjudicated.",
        "read_scope": "saved metadata/text/fingerprints only; no image/archive/model/outcome reads",
        "runtime": {"python_version": sys.version, "python_executable": sys.executable},
        "input_snapshot_sha256": {name: file_sha256(path) for name, path in source_paths.items()},
        "source_snapshot_sha256": {name: file_sha256(path) for name, path in code_paths.items()}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "candidates.csv").write_bytes(csv_bytes(selected))
    (output / "broad_outside.csv").write_bytes(csv_bytes(outside))
    (output / "view_queue.json").write_bytes(canonical_json_bytes(view_queue))
    for directory, paths in (("input_snapshot", source_paths), ("source_snapshot", code_paths)):
        destination = output / directory
        destination.mkdir()
        for name, path in paths.items():
            shutil.copyfile(path, destination / name)
            expected = result["input_snapshot_sha256" if directory == "input_snapshot" else "source_snapshot_sha256"][name]
            if file_sha256(destination / name) != expected:
                raise ContractError("snapshot_mismatch", "Source/input changed during preserved snapshot creation.")
    (output / "screen_report.json").write_bytes(canonical_json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "screen-dir", "checkpoint", "ranking", "published-queue",
                 "ranking-source", "published-source-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--approval-reference", required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.screen_dir, args.checkpoint, args.checkpoint_sha256,
            args.ranking, args.published_queue, args.ranking_source, args.published_source_dir,
            args.output_dir, approval_reference=args.approval_reference)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: result[k] for k in ("status", "review_candidates", "broad_pairs_outside_approved_review", "production_freeze")}))
