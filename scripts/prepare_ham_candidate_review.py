"""Prepare an unresolved review queue from saved inventory and public ID lists.

No source archives, images, assignments, or model outcomes are opened.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.manifests import canonical_json_bytes, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError

SOURCES = {
    "newly_discovered_duplicates.csv": ("reported_missed_duplicate", 18,
        "fafe916c6a97c4d6c6de3c7c93eae1f7eb601680692c25ea82c8ef595fca73c7"),
    "near_duplicates_different_times.csv": ("reported_similar_pair_different_lesions", 2,
        "6c4aff9e4b138c25765045783de8a47273051448da13670bad9087f87530492a"),
}
COMMIT = "ac655ac5a5b93264c57d69d6d520778d915bfc89"


def prepare(manifest: Path, source_dir: Path) -> dict:
    rows = read_manifest(manifest)
    by_id = {row.image_id: row for row in rows}
    provenance = json.loads((source_dir / "source.json").read_text())
    if provenance["commit"] != COMMIT:
        raise ContractError("candidate_source_mismatch", "Public candidate revision must be pinned.")
    candidates = []
    seen = set()
    for name, (kind, count, digest) in SOURCES.items():
        content = (source_dir / name).read_bytes()
        if sha256_bytes(content) != digest:
            raise ContractError("candidate_source_mismatch", "Public candidate bytes changed.")
        reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
        if reader.fieldnames != ["similarity", "from_img", "to_img"]:
            raise ContractError("invalid_candidate_source", "Unexpected candidate header.")
        start = len(candidates)
        for item in reader:
            pair = tuple(sorted(Path(item[key]).stem for key in ("from_img", "to_img")))
            if pair in seen or pair[0] == pair[1] or any(i not in by_id for i in pair):
                raise ContractError("candidate_correspondence_failure", "Pair must match two unique inventory IDs.")
            seen.add(pair)
            left, right = (by_id[i] for i in pair)
            candidates.append({"left": left.image_id, "right": right.image_id,
                "original_lesion_ids": [left.lesion_id, right.lesion_id],
                "labels": [left.dx, right.dx], "payload_sha256": [left.sha256, right.sha256],
                "inventory_paths": [left.path, right.path], "kind": kind,
                "published_similarity": item["similarity"],
                "source_file": name, "source_file_sha256": digest,
                "decision": "pending", "reviewer": "", "rationale": "", "visual_evidence": ""})
        if len(candidates) - start != count:
            raise ContractError("invalid_candidate_source", "Published pair count changed.")
    by_hash = defaultdict(list)
    by_lesion = defaultdict(list)
    for row in rows:
        by_hash[row.sha256].append(row)
        by_lesion[row.lesion_id].append(row)
    exact = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        lesions = {row.lesion_id for row in members}
        exact.append({"sha256": digest, "image_ids": sorted(row.image_id for row in members),
            "original_lesion_ids": sorted(lesions),
            "same_original_lesion": len(lesions) == 1,
            "representative_image_id": min(row.image_id for row in members) if len(lesions) == 1 else None,
            "lesion_record_count": len(by_lesion[members[0].lesion_id]) if len(lesions) == 1 else None,
            "lesion_distinct_view_count": len({r.sha256 for r in by_lesion[members[0].lesion_id]})
                                          if len(lesions) == 1 else None,
            "policy": "retain_inventory_use_one_canonical_view",
            "cross_id_identity_review_required": len(lesions) != 1})
    return {"status": "pending_visual_adjudication_and_full_candidate_scan",
            "production_freeze": False, "manifest_sha256": sha256_bytes(manifest.read_bytes()),
            "source_versions": sorted({r.source_version for r in rows}),
            "public_candidate_commit": COMMIT,
            "identifier_correspondence_checked": True, "local_visual_identity_confirmed": False,
            "candidates": candidates, "exact_content_groups": exact,
            "reported_18_pair_labels": dict(Counter(c["labels"][0] for c in candidates
                                        if c["kind"] == "reported_missed_duplicate")),
            "reported_18_different_lesion_ids": all(c["original_lesion_ids"][0] != c["original_lesion_ids"][1]
                                        for c in candidates if c["kind"] == "reported_missed_duplicate"),
            "reported_18_different_payloads": all(c["payload_sha256"][0] != c["payload_sha256"][1]
                                        for c in candidates if c["kind"] == "reported_missed_duplicate")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.manifest, args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json_bytes(report))
    print(json.dumps({"status": report["status"], "published_pairs": len(report["candidates"]),
                      "exact_content_groups": len(report["exact_content_groups"])}))
