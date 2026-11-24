"""Prepare a complete review queue from hash-bound HAM decoding evidence.

Reads saved integrity artifacts only, never original ZIPs/images. This is a
reviewable procedure amendment, not approval of adjudication, split or freeze.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data import candidate_screen, image_integrity
from dermdetect.data.candidate_screen import iter_hash_pairs, PROCEDURE, MAX_OUTPUT_BYTES
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError
import prepare_ham_candidate_review as published_review

FIELDS = ("left", "right", "dhash_distance", "reasons", "decision", "reviewer", "rationale", "evidence")


def prepare(manifest: Path, inventory_report: Path, checkpoint_path: Path,
            checkpoint_sha256: str, published_path: Path, output: Path, *,
            published_sha256: str | None = None, published_source_dir: Path | None = None) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite integrity evidence.")
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise ContractError("decode_checkpoint_mismatch", "Require the explicit recorded checkpoint hash.")
    rows = read_manifest(manifest)
    by_id = {r.image_id: r for r in rows}
    checkpoint = json.loads(checkpoint_path.read_text())
    inventory = json.loads(inventory_report.read_text())
    published = json.loads(published_path.read_text())
    published_digest = file_sha256(published_path)
    real_inventory = inventory.get("data_kind") == "real" or any(r.source == "HAM10000" for r in rows)
    if published_sha256 is not None and published_digest != published_sha256:
        raise ContractError("candidate_source_mismatch", "Published queue differs from its explicit recorded hash.")
    if real_inventory and (published_sha256 is None or published_source_dir is None):
        raise ContractError("candidate_source_binding_required",
                            "Real HAM screening requires the recorded published queue hash and pinned textual source directory.")
    published_bindings = {"published_queue_sha256": published_digest,
                          "published_queue_binding_kind": "synthetic_fixture"}
    if published_source_dir is not None:
        # Reconstruct expected membership and source evidence from pinned CSV
        # texts. Never trust counts, hashes or pair IDs asserted by the queue.
        expected = published_review.prepare(manifest, published_source_dir)
        if published != expected:
            raise ContractError("candidate_source_mismatch",
                                "Published queue must exactly match the complete source-pinned preparation.")
        published_bindings.update({
            "published_queue_binding_kind": "reconstructed_from_pinned_textual_sources",
            "expected_published_queue_sha256": sha256_bytes(canonical_json_bytes(expected)),
            "public_candidate_commit": published_review.COMMIT,
            "published_source_files_sha256": {name: file_sha256(published_source_dir / name)
                                               for name in published_review.SOURCES},
            "published_source_record_sha256": file_sha256(published_source_dir / "source.json"),
            "published_preparation_source_sha256": file_sha256(Path(published_review.__file__)),
        })
    manifest_hash = file_sha256(manifest)
    bindings = {"manifest_sha256": manifest_hash,
                "local_config_sha256": inventory["local_config_sha256"],
                "archive_sha256": {a["alias"]: a["sha256"] for a in inventory["archives"]},
                "decoder_source_sha256": file_sha256(Path(image_integrity.__file__))}
    if (any(checkpoint.get(k) != v for k, v in bindings.items())
            or checkpoint.get("status") != "all_listed_jpegs_decoded_candidate_screen_pending"
            or checkpoint.get("records_decoded") != len(rows)
            or published.get("manifest_sha256") != manifest_hash):
        raise ContractError("decode_checkpoint_mismatch", "Saved evidence bindings do not match.")
    fingerprints = checkpoint.get("fingerprints", {})
    if set(fingerprints) != set(by_id):
        raise ContractError("decode_checkpoint_mismatch", "Every inventory image needs decoding evidence.")
    rgb_signatures = {}
    for key, value in fingerprints.items():
        if (not isinstance(value, dict) or type(value.get("width")) is not int
                or type(value.get("height")) is not int
                or min(value["width"], value["height"]) < 1
                or value["width"] * value["height"] > image_integrity.MAX_PIXELS
                or not isinstance(value.get("rgb_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", value["rgb_sha256"])):
            raise ContractError("invalid_fingerprint", "Malformed cached decoding evidence.")
        signature = (value["width"], value["height"], value.get("dhash64"))
        previous = rgb_signatures.setdefault(value["rgb_sha256"], signature)
        if previous != signature:
            raise ContractError("inconsistent_exact_fingerprint",
                                "Identical decoded RGB requires identical dimensions and dHash.")
    hashes = {key: value.get("dhash64") for key, value in fingerprints.items()}
    # Iteration validates all hashes before output creation, even if no matches.
    iterator = iter_hash_pairs(hashes)
    first = next(iterator, None)
    mandatory = defaultdict(set)
    for item in published["candidates"]:
        pair = tuple(sorted((item["left"], item["right"])))
        if pair[0] == pair[1] or any(i not in by_id for i in pair) or not item.get("kind"):
            raise ContractError("invalid_candidate", "Published candidate must match two inventory IDs.")
        mandatory[pair].add(item["kind"])
    # Validate equality consistency in O(records) space. Once validated, every
    # RGB/byte equality has equal dHash and necessarily appears in the stream;
    # no quadratic equality-pair dictionary is needed.
    byte_fingerprints = {}
    for row in rows:
        fingerprint = fingerprints[row.image_id]
        if byte_fingerprints.setdefault(row.sha256, fingerprint) != fingerprint:
            raise ContractError("inconsistent_exact_fingerprint",
                                "Identical encoded bytes require identical complete cached fingerprints.")
    output.mkdir(parents=True, exist_ok=False)
    total_bytes = 0
    all_counts, cross_counts = Counter(), Counter()
    review_count = 0
    started = time.perf_counter()
    all_path = output / "hash_matches.csv.partial"
    queue_path = output / "candidates.csv.partial"

    def write(handle, fields):
        nonlocal total_bytes
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerow(fields)
        content = buffer.getvalue().encode()
        total_bytes += len(content)
        if total_bytes > MAX_OUTPUT_BYTES:
            raise ContractError("screen_storage_bound", "Complete output exceeds 128 MiB; no queue certified.")
        handle.write(content)

    def write_candidate(handle, left, right, distance, reasons):
        nonlocal review_count
        write(handle, (left, right, distance, "|".join(sorted(reasons)), "pending", "", "", ""))
        review_count += 1

    try:
        with all_path.open("xb") as all_handle, queue_path.open("xb") as queue_handle:
            write(all_handle, ("left", "right", "dhash_distance", "same_original_lesion"))
            write(queue_handle, FIELDS)
            import itertools
            for left, right, distance in itertools.chain(() if first is None else (first,), iterator):
                same_id = by_id[left].lesion_id == by_id[right].lesion_id
                write(all_handle, (left, right, distance, str(same_id).lower()))
                all_counts[distance] += 1
                if not same_id:
                    cross_counts[distance] += 1
                    reasons = mandatory.pop((left, right), set()) | {"dhash64_hamming_at_most_4"}
                    if fingerprints[left]["rgb_sha256"] == fingerprints[right]["rgb_sha256"]:
                        reasons.add("identical_oriented_decoded_rgb")
                    if by_id[left].sha256 == by_id[right].sha256:
                        reasons.add("identical_payload_sha256")
                    write_candidate(queue_handle, left, right, distance, reasons)
            # Known candidates missed by the screen are unconditional, not tuned
            # against scores. Order is deterministic: streamed matches then extras.
            for (left, right), reasons in sorted(mandatory.items()):
                write_candidate(queue_handle, left, right, (hashes[left] ^ hashes[right]).bit_count(), reasons)
    except Exception as exc:
        failure = {"status": "streamed_screen_failed", "production_freeze": False,
                   "checkpoint_sha256": checkpoint_sha256, "procedure": PROCEDURE,
                   "error": exc.to_dict() if isinstance(exc, ContractError) else {"type": type(exc).__name__}}
        (output / "failure.json").write_bytes(canonical_json_bytes(failure))
        raise
    all_path = all_path.rename(output / "hash_matches.csv")
    queue_path = queue_path.rename(output / "candidates.csv")
    report = {**bindings, **published_bindings, "procedure": PROCEDURE, "status": "enumeration_complete_adjudication_pending",
              "procedure_approval": "pending_human_review", "production_freeze": False,
              "split_ready": False, "records_decoded_in_reused_checkpoint": len(rows),
              "checkpoint_sha256": checkpoint_sha256, "inventory_report_sha256": file_sha256(inventory_report),
              "script_sha256": file_sha256(Path(__file__)),
              "screen_source_sha256": file_sha256(Path(candidate_screen.__file__)),
              "hash_matches_sha256": file_sha256(all_path), "candidates_sha256": file_sha256(queue_path),
              "all_hash_matches": sum(all_counts.values()), "cross_id_hash_matches": sum(cross_counts.values()),
              "within_original_id_hash_matches": sum(all_counts.values()) - sum(cross_counts.values()),
              "hash_matches_by_distance": dict(all_counts), "cross_id_matches_by_distance": dict(cross_counts),
              "mandatory_pairs": len({tuple(sorted((c["left"], c["right"]))) for c in published["candidates"]}),
              "review_candidates": review_count, "pending_decisions": review_count,
              "elapsed_screen_seconds": round(time.perf_counter() - started, 6), "output_csv_bytes": total_bytes,
              "read_scope": "saved evidence only; no original archive reads or new image decoding",
              "completeness_limit": "Exhaustive for dHash <=4 plus mandatory candidates, not for physical lesion identity.",
              "resource_bounds": {"records": candidate_screen.MAX_RECORDS, "csv_bytes": MAX_OUTPUT_BYTES}}
    (output / "screen_report.json").write_bytes(canonical_json_bytes(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "inventory-report", "checkpoint", "published-queue", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--published-queue-sha256")
    parser.add_argument("--published-source-dir", type=Path)
    args = parser.parse_args()
    try:
        report = prepare(args.manifest, args.inventory_report, args.checkpoint,
                         args.checkpoint_sha256, args.published_queue, args.output_dir,
                         published_sha256=args.published_queue_sha256,
                         published_source_dir=args.published_source_dir)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: report[k] for k in ("status", "all_hash_matches", "review_candidates", "elapsed_screen_seconds", "production_freeze")}))
