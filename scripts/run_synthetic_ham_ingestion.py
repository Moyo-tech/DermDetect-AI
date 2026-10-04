"""Offline M2 preparation demonstration; no real-input switches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dermdetect.data.ham import ingest_ham, verify_ham_files
from dermdetect.data.manifests import canonical_json_bytes, manifest_csv_bytes, read_manifest
from dermdetect.data.schemas import CLASS_ORDER, ContractError


def run_demo(output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite an existing destination.")
    with tempfile.TemporaryDirectory(prefix="dermdetect-ham-demo-") as directory:
        scratch = Path(directory)
        roots = {alias: scratch / alias for alias in ("part1", "part2")}
        for root in roots.values():
            root.mkdir()
        metadata = scratch / "invented_metadata.csv"
        with metadata.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("image_id", "lesion_id", "dx", "dx_type"))
            for dx in CLASS_ORDER:
                for index, alias in enumerate(roots, 1):
                    image_id = f"invented_{dx}_{index:02d}"
                    writer.writerow((image_id, f"invented_lesion_{dx}", dx, "invented"))
                    (roots[alias] / f"{image_id}.jpg").write_text(
                        f"SYNTHETIC TEXT; NOT AN IMAGE: {image_id}\n", encoding="utf-8")
        result = ingest_ham(metadata, roots, source_version="invented-fixture-v1",
                            provenance_reference="scripts/run_synthetic_ham_ingestion.py",
                            permitted_use_reference="Invented metadata and text only; no medical data",
                            data_kind="synthetic")
        report = result.integrity_report()
        (scratch / "manifest.csv").write_bytes(manifest_csv_bytes(result.records))
        reloaded = read_manifest(scratch / "manifest.csv")
        if reloaded != result.records:
            raise ContractError("reload_mismatch", "Manifest reload changed membership.")
        verify_ham_files(reloaded, roots)
        (scratch / "integrity_report.json").write_bytes(canonical_json_bytes(report))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scratch, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 HAM-format preparation on invented text only.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "m2-synthetic-ingestion")
    args = parser.parse_args()
    try:
        report = run_demo(args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "data_kind": "synthetic", "error": exc.to_dict()}),
              file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"status": "failed", "data_kind": "synthetic",
                          "error": {"code": "io_error", "message": str(exc)}}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "data_kind": "synthetic",
                      "records": report["records"], "lesions": report["lesions"],
                      "split_ready": False, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
