"""Build an unsplit local inventory from an explicitly authorized path config."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dermdetect.data.ham_archives import ingest_ham_archives
from dermdetect.data.manifests import canonical_json_bytes, manifest_csv_bytes, read_manifest, sha256_bytes
from dermdetect.data.schemas import ContractError


def run(config_path: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ContractError("output_exists", "Refusing to overwrite an existing inventory.")
    config_content = config_path.read_bytes()
    config = json.loads(config_content)
    required = {"schema_version", "metadata_path", "image_archives", "source_version",
                "provenance_reference", "permitted_use_reference", "data_kind",
                "expected_archive_md5", "source_review_path", "source_review_sha256"}
    if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
        raise ContractError("invalid_config", "Unsupported archive inventory configuration.")
    if not isinstance(config["image_archives"], dict):
        raise ContractError("invalid_config", "Named archives must be explicitly supplied.")
    for reference in (config["metadata_path"], config["source_review_path"], *config["image_archives"].values()):
        if not isinstance(reference, str) or not Path(reference).is_absolute():
            raise ContractError("invalid_config", "Input paths must be explicit absolute paths.")
    review = Path(config["source_review_path"]).read_bytes()
    if sha256_bytes(review) != config["source_review_sha256"]:
        raise ContractError("source_review_mismatch", "Saved source review differs from the configuration hash.")
    result = ingest_ham_archives(Path(config["metadata_path"]),
                                 {alias: Path(path) for alias, path in config["image_archives"].items()},
                                 **{key: config[key] for key in ("source_version", "provenance_reference",
                                                               "permitted_use_reference", "data_kind",
                                                               "expected_archive_md5")})
    report = {**result.integrity_report(), "local_config_sha256": sha256_bytes(config_content),
              "source_review_sha256": sha256_bytes(review)}
    # Write only small derived artifacts after all source/member integrity checks.
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.csv").write_bytes(manifest_csv_bytes(result.records))
    if read_manifest(output / "manifest.csv") != result.records:
        raise ContractError("reload_mismatch", "Persisted manifest differs from in-memory membership.")
    (output / "integrity_report.json").write_bytes(canonical_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        report = run(args.config, args.output_dir)
    except ContractError as exc:
        print(json.dumps({"status": "failed", "error": exc.to_dict()}), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "failed", "error": {"code": "invalid_input", "message": str(exc)}}), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "records": report["records"], "lesions": report["lesions"],
                      "duplicate_candidate_groups": len(report["exact_duplicate_candidates"]),
                      "publisher_archive_checksums_match": all(item["publisher_md5_match"] for item in report["archives"]),
                      "split_ready": False, "engineering_elapsed_seconds": round(time.monotonic() - started, 3)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
