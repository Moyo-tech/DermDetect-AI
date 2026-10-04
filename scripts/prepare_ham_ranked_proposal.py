"""Prepare a prospective smaller review set; never replace active M2 rules.

Union the pinned independent ranking's cross-ID pairs with all equal-dHash
pairs and mandatory source pairs. All decisions stay pending. Broad-screen
rows outside this proposal remain unadjudicated, never declared independent.
"""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest
from dermdetect.data.schemas import ContractError

COMMIT = "ac655ac5a5b93264c57d69d6d520778d915bfc89"
RANK_SHA256 = "02f669afc330e998aaeb971bcc93d9c000506dbf2aa33381bfbccf0049819e47"
PUBLISHED_SHA256 = "06b63e78bda8b12ec8610708c1ffc8292944aeac0ed9e2a2444ca8ef641cfd61"
RANK_COUNT = 1000
PROPOSAL = "published-top1000-plus-equal-dhash-review-v1"


def prepare(manifest: Path, screen_dir: Path, ranking: Path,
            published_queue: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite proposal evidence.")
    if file_sha256(ranking) != RANK_SHA256 or file_sha256(published_queue) != PUBLISHED_SHA256:
        raise ContractError("candidate_source_mismatch", "Both public ranking and mandatory queue must match their recorded pins.")
    rows = {r.image_id: r for r in read_manifest(manifest)}
    report_path, queue_path = screen_dir / "screen_report.json", screen_dir / "candidates.csv"
    report = json.loads(report_path.read_text())
    if (report.get("status") != "enumeration_complete_adjudication_pending"
            or report.get("manifest_sha256") != file_sha256(manifest)
            or report.get("candidates_sha256") != file_sha256(queue_path)):
        raise ContractError("review_binding_mismatch", "Proposal requires the complete unchanged broad screen.")
    published = json.loads(published_queue.read_text())
    if published.get("manifest_sha256") != file_sha256(manifest):
        raise ContractError("review_binding_mismatch", "Mandatory candidates must match this inventory.")
    proposed, ranked, rank_scores = defaultdict(set), {}, []
    with ranking.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["similarity", "from_img", "to_img"]:
            raise ContractError("invalid_candidate_source", "Unexpected ranked source header.")
        for position, item in enumerate(reader, 1):
            key = tuple(sorted(Path(item[k]).stem for k in ("from_img", "to_img")))
            try:
                score = Decimal(item["similarity"])
                if not score.is_finite() or not 0 <= score <= 1:
                    raise InvalidOperation()
            except (InvalidOperation, TypeError):
                raise ContractError("invalid_candidate_source", "Ranked similarities must be finite values in [0,1].")
            if key[0] == key[1] or key in ranked or any(i not in rows for i in key):
                raise ContractError("candidate_correspondence_failure", "Every ranked pair must match two unique inventoried IDs.")
            ranked[key] = {"rank": position, "similarity": str(score)}
            rank_scores.append(score)
            if rows[key[0]].lesion_id != rows[key[1]].lesion_id:
                proposed[key].add("independent_published_top1000_cross_id")
    if len(ranked) != RANK_COUNT or rank_scores != sorted(rank_scores, reverse=True):
        raise ContractError("invalid_candidate_source", "Rank count/order must match the prespecified published list.")
    with queue_path.open(newline="") as handle:
        broad = list(csv.DictReader(handle))
    broad_keys = {tuple(sorted((r["left"], r["right"]))) for r in broad}
    if len(broad_keys) != len(broad) or len(broad) != report.get("review_candidates"):
        raise ContractError("candidate_membership_mismatch", "Broad-screen membership is inconsistent.")
    zero_keys = set()
    for item in broad:
        key = tuple(sorted((item["left"], item["right"])))
        if any(i not in rows for i in key):
            raise ContractError("candidate_correspondence_failure", "Broad pairs must match inventory.")
        if int(item["dhash_distance"]) == 0:
            zero_keys.add(key)
            proposed[key].add("equal_dhash64_not_identity")
    for item in published["candidates"]:
        key = tuple(sorted((item["left"], item["right"])))
        if key not in ranked or any(i not in rows for i in key):
            raise ContractError("candidate_correspondence_failure", "All mandatory pairs must remain in the independent ranking.")
        proposed[key].add(item["kind"])
    # Exact cross-ID byte matches remain unconditional. Decoded-RGB matches
    # have equal dHash by the validated checkpoint consistency contract, so
    # the complete zero-distance union already includes those cross-ID pairs.
    hashes = defaultdict(list)
    for row in rows.values():
        hashes[row.sha256].append(row.image_id)
    for members in hashes.values():
        if len(members) > 100:
            raise ContractError("proposal_resource_bound", "Large exact groups require explicit review; no partial proposal.")
        for key in itertools.combinations(sorted(members), 2):
            if rows[key[0]].lesion_id != rows[key[1]].lesion_id:
                proposed[key].add("identical_payload_sha256")
    cross_ranked = {k for k in ranked if rows[k[0]].lesion_id != rows[k[1]].lesion_id}
    candidates = []
    for key in sorted(proposed, key=lambda k: (ranked.get(k, {}).get("rank", RANK_COUNT + 1), k)):
        candidates.append({"left": key[0], "right": key[1], "kind": "|".join(sorted(proposed[key])),
            "original_lesion_ids": [rows[i].lesion_id for i in key], "payload_sha256": [rows[i].sha256 for i in key],
            "decision": "pending", "reviewer": "", "rationale": "", **ranked.get(key, {})})
    result = {"status": "proposed_review_set_not_approved", "procedure": PROPOSAL,
        "production_freeze": False, "split_ready": False, "human_decisions_recorded": 0,
        "manifest_sha256": file_sha256(manifest), "ranking_sha256": file_sha256(ranking),
        "public_candidate_commit": COMMIT, "published_queue_sha256": file_sha256(published_queue),
        "broad_queue_sha256": file_sha256(queue_path), "broad_report_sha256": file_sha256(report_path),
        "script_sha256": file_sha256(Path(__file__)), "ranked_pairs": len(ranked),
        "ranked_cross_id_pairs": len(cross_ranked), "equal_dhash_cross_id_pairs": len(zero_keys),
        "ranked_pairs_already_in_broad_queue": len(cross_ranked & broad_keys),
        "independent_ranked_pairs_absent_from_broad_queue": len(cross_ranked - broad_keys),
        "proposed_review_pairs": len(candidates), "mandatory_pairs": len(published["candidates"]),
        "broad_pairs_outside_proposal": len(broad_keys - set(proposed)),
        "unreviewed_pair_status": "Not locally adjudicated; never inferred different lesions.",
        "source_applicability_limit": "IDs map to local inventory; author per-image byte hashes/encoder checksum are unavailable. Ranking was not reproduced locally.",
        "completeness_limit": "Sensitivity-limited candidate procedure proposal; no proof of unknown-duplicate completeness.",
        "candidates": candidates}
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "screen-dir", "ranking", "published-queue", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.screen_dir, args.ranking, args.published_queue, args.output)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: result[k] for k in ("status", "proposed_review_pairs", "ranked_cross_id_pairs", "broad_pairs_outside_proposal", "production_freeze")}))
