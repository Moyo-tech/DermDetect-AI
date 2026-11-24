"""Authorized pre-split integrity views, preserving original archives.

Full-resolution EXIF-oriented pairs are local review artifacts, never training
inputs or diagnostic outputs. Only IDs in the explicit queue are opened.
"""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
import io
import json
from pathlib import Path
import sys
import zipfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.image_integrity import decode_fingerprint, MAX_BYTES
from dermdetect.data import image_integrity
from dermdetect.data.manifests import canonical_json_bytes, file_sha256, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError


def prepare(config_path: Path, manifest: Path, queue_path: Path, output: Path, *, progress: bool = False) -> dict:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite visual evidence.")
    config = json.loads(config_path.read_text())
    rows = {r.image_id: r for r in read_manifest(manifest)}
    queue = json.loads(queue_path.read_text())
    if queue["manifest_sha256"] != sha256_bytes(manifest.read_bytes()):
        raise ContractError("candidate_manifest_mismatch", "Visual queue must match this inventory.")
    if len(queue["candidates"]) > 1000:
        raise ContractError("visual_resource_bound", "At most 1,000 full-resolution pairs per review batch.")
    pairs = [tuple(sorted((p["left"], p["right"]))) for p in queue["candidates"]]
    if len(pairs) != len(set(pairs)) or any(a == b or a not in rows or b not in rows for a, b in pairs):
        raise ContractError("invalid_candidate", "Each view must name two unique inventory images.")
    output.mkdir(parents=True, exist_ok=False)
    views, output_bytes = [], 0
    started = time.perf_counter()
    archives, cache = {}, OrderedDict()
    stack = ExitStack()

    def load(image_id):
        if image_id in cache:
            cache.move_to_end(image_id)
            return cache[image_id].copy()
        row = rows[image_id]
        alias, name = row.path.split("/")
        if alias not in archives:
            archives[alias] = stack.enter_context(zipfile.ZipFile(config["image_archives"][alias]))
        archive = archives[alias]
        if archive.getinfo(name).file_size > MAX_BYTES:
            raise ContractError("invalid_image_payload", "Pair payload exceeds the approved bound.")
        with archive.open(name) as member:
            payload = member.read(MAX_BYTES + 1)
        if sha256_bytes(payload) != row.sha256:
            raise ContractError("image_checksum_mismatch", "Visual payload differs from inventory.")
        decode_fingerprint(payload)
        with Image.open(io.BytesIO(payload)) as image:
            with ImageOps.exif_transpose(image) as oriented:
                cache[image_id] = oriented.convert("RGB")
        if len(cache) > 32:
            _, evicted = cache.popitem(last=False)
            evicted.close()
        return cache[image_id].copy()

    try:
        for index, pair in enumerate(queue["candidates"], 1):
            images = []
            ids = [pair["left"], pair["right"]]
            for image_id in ids:
                images.append(load(image_id))
            width = sum(image.width for image in images)
            height = max(image.height for image in images) + 76
            with Image.new("RGB", (width, height), "white") as sheet:
                draw = ImageDraw.Draw(sheet)
                font = ImageFont.load_default(size=14)
                offset = 0
                for image_id, image in zip(ids, images):
                    row = rows[image_id]
                    draw.text((offset + 8, 6), f"Pair {index:02}: {image_id}", fill="black", font=font)
                    draw.text((offset + 8, 26), f"Original lesion: {row.lesion_id}", fill="black", font=font)
                    draw.text((offset + 8, 46), f"SHA256: {row.sha256[:24]}... (full hash in index)", fill="black", font=font)
                    sheet.paste(image, (offset, 76))
                    offset += image.width
                    image.close()
                path = output / f"pair_{index:02}.png"
                sheet.save(path)
            output_bytes += path.stat().st_size
            if output_bytes > 1536 * 1024 * 1024:
                raise ContractError("visual_resource_bound", "Full-resolution review output exceeds 1.5 GiB; no complete index.")
            views.append({"pair_number": index, "image_ids": ids,
                          "original_lesion_ids": [rows[i].lesion_id for i in ids],
                          "source_payload_sha256": [rows[i].sha256 for i in ids],
                          "view_file": path.name, "view_sha256": sha256_bytes(path.read_bytes()),
                          "view_transform": "EXIF transpose/RGB; full resolution; no crop/resize",
                          "source_kind": pair.get("kind", pair.get("reasons")),
                          "decision": "pending", "reviewer": "", "rationale": ""})
            if progress and (index % 50 == 0 or index == len(queue["candidates"])):
                print(json.dumps({"pairs_prepared": index, "total_pairs": len(queue["candidates"])}), flush=True)
    finally:
        for image in cache.values():
            image.close()
        stack.close()
    report = {"manifest_sha256": sha256_bytes(manifest.read_bytes()),
              "queue_sha256": sha256_bytes(queue_path.read_bytes()),
              "status": "visual_evidence_prepared_not_adjudicated", "pairs": views,
              "script_sha256": file_sha256(Path(__file__)), "output_png_bytes": output_bytes,
              "decoder_source_sha256": file_sha256(Path(image_integrity.__file__)),
              "elapsed_view_seconds": round(time.perf_counter() - started, 6),
              "bounded_image_cache": 32, "local_config_sha256": file_sha256(config_path),
              "production_freeze": False}
    (output / "index.json").write_bytes(canonical_json_bytes(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    report = prepare(args.config, args.manifest, args.queue, args.output_dir, progress=args.progress)
    print(json.dumps({"pairs_prepared": len(report["pairs"]), "status": report["status"]}))
