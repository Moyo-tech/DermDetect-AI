"""Offline M1 fixture runner. No user-supplied manifest or dataset input is accepted."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dermdetect.data.manifests import (build_manifest, canonical_json_bytes, manifest_csv_bytes,
                                      read_manifest, sha256_bytes, verify_files)
from dermdetect.data.overlap import require_unique_content
from dermdetect.data.schemas import CLASS_ORDER, ContractError
from dermdetect.data.splits import (SPLIT_ORDER, SplitConfig, partition_counts,
                                   split_manifest, verify_partition_membership)

FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_contract.json"
CASES = ("valid", "duplicate-id", "missing-field", "unknown-class", "label-conflict",
         "duplicate-content", "checksum-mismatch", "overlap")


def load_fixture() -> dict:
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if spec.get("data_kind") != "synthetic" or spec.get("source") != "invented-fixture":
        raise ContractError("invalid_fixture", "Only the bundled invented fixture is accepted.")
    return spec


def materialize_fixture(root: Path, spec: dict) -> list[dict]:
    """Create artificial text, never encoded medical or demo images."""
    if spec.get("data_kind") != "synthetic" or spec.get("source") != "invented-fixture":
        raise ContractError("invalid_fixture", "Fixture must be explicitly invented and synthetic.")
    payloads = root / "payloads"
    payloads.mkdir()
    rows = []
    for lesion in spec["lesions"]:
        lesion_id = lesion["lesion_id"]
        count = lesion["image_count"]
        if (not isinstance(lesion_id, str) or not re.fullmatch(r"toy_[A-Za-z0-9_]+", lesion_id)
                or isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ContractError("invalid_fixture", "Invented lesion ID/count is invalid.")
        for index in range(1, count + 1):
            image_id = f"{lesion_id}_image{index:02d}"
            reference = f"payloads/{image_id}.txt"
            content = f"DermDetect-R SYNTHETIC TEXT; NOT AN IMAGE.\n{image_id}\n"
            path = root / reference
            if path.exists():
                raise ContractError("invalid_fixture", "Fixture repeats a payload identifier.")
            path.write_text(content, encoding="utf-8")
            rows.append({"image_id": image_id, "lesion_id": lesion_id, "dx": lesion["dx"],
                         "source": spec["source"], "source_version": spec["source_version"],
                         "path": reference})
    return rows


def inject_failure(case: str, metadata: list[dict], root: Path) -> None:
    if case == "duplicate-id":
        metadata[1]["image_id"] = metadata[0]["image_id"]
    elif case == "missing-field":
        metadata[0].pop("lesion_id")
    elif case == "unknown-class":
        metadata[0]["dx"] = "unknown"
    elif case == "label-conflict":
        # L02 has two images: rows 1 and 2 must agree.
        metadata[2]["dx"] = "bcc"
    elif case == "duplicate-content":
        (root / metadata[1]["path"]).write_bytes((root / metadata[0]["path"]).read_bytes())


def run_demo(output: Path, case: str = "valid") -> dict:
    if case not in CASES:
        raise ContractError("invalid_case", "Unknown synthetic failure case.")
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite an existing output location.",
                            path=str(output))
    spec = load_fixture()
    config = SplitConfig.from_mapping(spec["split_config"])
    with tempfile.TemporaryDirectory(prefix="dermdetect-synthetic-") as directory:
        scratch = Path(directory)
        metadata = materialize_fixture(scratch, spec)
        inject_failure(case, metadata, scratch)
        rows = build_manifest(metadata, scratch)
        if case == "checksum-mismatch":
            (scratch / rows[0].path).write_text("SYNTHETIC altered payload\n", encoding="utf-8")
        verify_files(rows, scratch)
        require_unique_content(rows)
        partitions = split_manifest(rows, config)
        if case == "overlap":
            partitions["validation"] += (partitions["train"][0],)
        integrity = verify_partition_membership(rows, partitions)
        counts = partition_counts(partitions, config)
        expected = spec["expected"]
        if (len(rows) != expected["images"]
                or len({row.lesion_id for row in rows}) != expected["lesions"]
                or any(sum(row.dx == dx for row in rows) != expected["images_per_class"]
                       for dx in CLASS_ORDER)
                or any(len({row.lesion_id for row in rows if row.dx == dx}) != expected["lesions_per_class"]
                       for dx in CLASS_ORDER)
                or any(counts[name]["lesions"] != expected["lesions_per_split"][name]
                       for name in SPLIT_ORDER)):
            raise ContractError("fixture_expectation_mismatch", "Fixture totals disagree with its known specification.")
        artifacts = {"manifest.csv": manifest_csv_bytes(rows),
                     "split_config.json": canonical_json_bytes(config.to_dict())}
        artifacts.update({f"{name}.csv": manifest_csv_bytes(partitions[name]) for name in SPLIT_ORDER})
        report = {"data_kind": "synthetic", "status": "passed", "schema_version": 1,
                  "fixture_sha256": sha256_bytes(canonical_json_bytes(spec)),
                  "class_order": list(CLASS_ORDER), "images": len(rows),
                  "lesions": len({row.lesion_id for row in rows}),
                  "config": config.to_dict(), "counts": counts, "integrity": integrity,
                  "artifact_sha256": {name: sha256_bytes(content) for name, content in sorted(artifacts.items())},
                  "limitations": ["No real data or models used.",
                                  "Text payloads do not test image decoding or perceptual duplicates.",
                                  "Toy allocator targets lesion quotas, not joint image/lesion optimization.",
                                  "Production split settings and sealed-outcome access controls are not validated."]}
        artifacts["integrity_report.json"] = canonical_json_bytes(report)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copytree(scratch / "payloads", output / "payloads")
        for name, content in artifacts.items():
            (output / name).write_bytes(content)
        # Re-read the generated contract and verify persisted payload identity.
        reloaded = read_manifest(output / "manifest.csv")
        if reloaded != rows:
            raise ContractError("reload_mismatch", "Manifest reload changed membership.")
        verify_files(reloaded, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic-only M1 data contracts offline.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "m1-synthetic")
    parser.add_argument("--case", choices=CASES, default="valid", help="Inject one known synthetic failure.")
    parser.add_argument("--verify", action="store_true", help="Run the standard-library test suite first.")
    args = parser.parse_args()
    if args.verify:
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)
        if result.returncode:
            return result.returncode
    try:
        report = run_demo(args.output_dir, args.case)
    except ContractError as exc:
        print(json.dumps({"data_kind": "synthetic", "status": "failed", "error": exc.to_dict()},
                         sort_keys=True), file=sys.stderr)
        return 2
    except OSError as exc:
        print(json.dumps({"data_kind": "synthetic", "status": "failed",
                          "error": {"code": "io_error", "message": str(exc)}}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"data_kind": "synthetic", "status": "passed", "images": report["images"],
                      "lesions": report["lesions"], "zero_overlap": report["integrity"]["zero_overlap"],
                      "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
