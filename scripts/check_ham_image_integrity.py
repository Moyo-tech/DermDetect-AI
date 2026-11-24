"""Proposed unsplit HAM-only JPEG scan; requires separate real-decode approval.

Stream bounded ZIP members, never extract images. All review candidates stay
unresolved; this command neither creates an analysis manifest nor assigns splits.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.image_integrity import decode_fingerprint, near_duplicate_pairs, MAX_BYTES
from dermdetect.data import image_integrity as decoder_module
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError


def scan(config_path: Path, inventory_dir: Path, review_queue: Path, output: Path,
         decode_checkpoint: Path | None = None, checkpoint_sha256: str | None = None) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite integrity evidence.")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    inventory = json.loads((inventory_dir / "integrity_report.json").read_text())
    if inventory["local_config_sha256"] != sha256_bytes(config_bytes):
        raise ContractError("configuration_mismatch", "Reuse exactly the inventoried archive configuration.")
    manifest = inventory_dir / "manifest.csv"
    rows = read_manifest(manifest)
    queue = json.loads(review_queue.read_text())
    if queue["manifest_sha256"] != sha256_bytes(manifest.read_bytes()):
        raise ContractError("candidate_manifest_mismatch", "Candidate queue must refer to this inventory.")
    source = Path(config["source_review_path"])
    if file_sha256(source) != config["source_review_sha256"]:
        raise ContractError("source_review_mismatch", "Pinned source review changed.")
    archive_hashes = {item["alias"]: item["sha256"] for item in inventory["archives"]}
    if set(config["image_archives"]) != set(archive_hashes):
        raise ContractError("configuration_mismatch", "Only the inventoried archives may be read.")
    paths = {alias: Path(path) for alias, path in config["image_archives"].items()}
    for alias, path in paths.items():
        if not path.is_absolute() or file_sha256(path) != archive_hashes[alias]:
            raise ContractError("archive_checksum_mismatch", "Source archive changed since inventory.")
    by_archive = defaultdict(list)
    for row in rows:
        alias, name = row.path.split("/")
        if alias not in paths or name != row.image_id + ".jpg":
            raise ContractError("invalid_inventory_path", "Inventory path must name one approved flat JPG.")
        by_archive[alias].append(row)
    bindings = {"manifest_sha256": sha256_bytes(manifest.read_bytes()),
                "local_config_sha256": sha256_bytes(config_bytes),
                "archive_sha256": archive_hashes,
                "decoder_source_sha256": file_sha256(Path(decoder_module.__file__))}
    if decode_checkpoint is not None:
        if checkpoint_sha256 is None or file_sha256(decode_checkpoint) != checkpoint_sha256:
            raise ContractError("decode_checkpoint_mismatch", "Reused decoding evidence needs its recorded hash.")
        checkpoint = json.loads(decode_checkpoint.read_text())
        if any(checkpoint.get(key) != value for key, value in bindings.items()):
            raise ContractError("decode_checkpoint_mismatch", "Decoding evidence differs in source/configuration/procedure.")
        fingerprints = checkpoint["fingerprints"]
        if set(fingerprints) != {r.image_id for r in rows} or checkpoint["records_decoded"] != len(rows):
            raise ContractError("decode_checkpoint_mismatch", "Decoding checkpoint has incomplete membership.")
    else:
        fingerprints = {}
        for alias, members in sorted(by_archive.items()):
            with zipfile.ZipFile(paths[alias]) as archive:
                for row in members:
                    info = archive.getinfo(row.image_id + ".jpg")
                    if not 0 < info.file_size <= MAX_BYTES:
                        raise ContractError("invalid_image_payload", "Archive member exceeds JPEG scan bound.")
                    with archive.open(info) as handle:
                        payload = handle.read(MAX_BYTES + 1)
                    if len(payload) != info.file_size or sha256_bytes(payload) != row.sha256:
                        raise ContractError("image_checksum_mismatch", "Image bytes changed since inventory.")
                    try:
                        fingerprints[row.image_id] = decode_fingerprint(payload)
                    except ContractError as exc:
                        raise ContractError(exc.code, str(exc), image_id=row.image_id, **exc.details) from exc
        checkpoint = {**bindings, "status": "all_listed_jpegs_decoded_candidate_screen_pending",
                      "records_decoded": len(fingerprints), "fingerprints": fingerprints}
    # Preserve complete decoding evidence separately. A failed candidate stage
    # must not erase validated readability or force unnecessary repeat decoding.
    output.mkdir(parents=True, exist_ok=False)
    (output / "decode_checkpoint.json").write_bytes(canonical_json_bytes(checkpoint))
    by_id = {r.image_id: r for r in rows}
    candidates = {tuple(sorted((c["left"], c["right"]))): set([c["kind"]]) for c in queue["candidates"]}
    def add(pair, reason):
        candidates.setdefault(tuple(sorted(pair)), set()).add(reason)
    try:
        screened = near_duplicate_pairs({key: value["dhash64"] for key, value in fingerprints.items()})
    except ContractError as exc:
        failure = {**bindings, "status": "candidate_screen_failed", "error": exc.to_dict(),
                   "records_decoded": len(fingerprints), "production_freeze": False,
                   "decode_checkpoint_sha256": file_sha256(output / "decode_checkpoint.json")}
        (output / "candidate_failure.json").write_bytes(canonical_json_bytes(failure))
        raise
    for pair in screened:
        if by_id[pair[0]].lesion_id != by_id[pair[1]].lesion_id:
            add(pair, "dhash64_hamming_at_most_4")
    # Decoded-pixel matches can expose alternate encodings missed by byte SHA.
    rgb_groups = defaultdict(list)
    for image_id, value in fingerprints.items():
        rgb_groups[value["rgb_sha256"]].append(image_id)
    for members in rgb_groups.values():
        for i, left in enumerate(sorted(members)):
            for right in sorted(members)[i + 1:]:
                if by_id[left].lesion_id != by_id[right].lesion_id:
                    add((left, right), "identical_oriented_decoded_rgb")
    report = {"status": "decoded_pending_candidate_adjudication", "production_freeze": False,
              "manifest_sha256": sha256_bytes(manifest.read_bytes()),
              "review_queue_sha256": file_sha256(review_queue),
              "local_config_sha256": sha256_bytes(config_bytes),
              "records_decoded": len(fingerprints), "fingerprints": fingerprints,
              "candidates": [{"left": pair[0], "right": pair[1], "reasons": sorted(reasons),
                              "decision": "pending", "reviewer": "", "rationale": "", "visual_evidence": ""}
                             for pair, reasons in sorted(candidates.items())],
              "procedure": "JPEG verify/full-load; EXIF-oriented RGB SHA; bilinear-L dHash64 <=4; all 20 published pairs mandatory",
              "unreadable_policy": "hard_fail_no_exclusion",
              "completeness_limitation": "This screen cannot establish absence of perceptually related lesions."}
    (output / "image_integrity.json").write_bytes(canonical_json_bytes(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    args = parser.parse_args()
    try:
        report = scan(args.config, args.inventory_dir, args.review_queue, args.output_dir,
                      args.decode_checkpoint, args.checkpoint_sha256)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({"status": report["status"], "records_decoded": report["records_decoded"],
                      "candidate_pairs": len(report["candidates"]), "production_freeze": False}))
