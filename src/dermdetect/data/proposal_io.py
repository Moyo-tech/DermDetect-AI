"""Hash-bound replay of unfrozen proposed splits for evaluator/bundle adapters.

The authoritative report hash must come from an independently retained review
record. This loader grants no assignment acceptance, export or freeze approval.
It reads metadata only, reconstructs the physical anchor and reviewed uncertainty
wrapper, and validates canonical deterministic allocation and exact membership.
"""
import csv
from pathlib import Path

from .analysis import AnalysisRecord, validate_analysis
from .conservative_groups import ReviewedUncertainty, build_conservative_groups
from .constrained_allocation import allocate_constrained_groups, configuration, verify_constrained_allocation
from .manifests import canonical_json_bytes, file_sha256, rooted_file
from .schemas import ContractError, ManifestRecord

FIELDS = ("image_id", "lesion_id", "dx", "source", "source_version", "path", "sha256",
          "analysis_lesion_id", "split_group_id", "active", "representative_image_id", "inactive_reason")


def _json(path):
    import json
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated JSON key")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError("invalid_proposal_artifact", "Saved proposal JSON must be readable without duplicate keys.") from exc


def _csv(path):
    result = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ContractError("invalid_analysis_csv", "Exact versioned analysis columns are required.")
            for row in reader:
                if (None in row or any(not isinstance(v, str) for v in row.values())
                        or row["active"] not in ("true", "false")):
                    raise ContractError("invalid_analysis_csv", "Canonical rows and boolean participation are required.")
                original = ManifestRecord(*(row[name] for name in FIELDS[:7]))
                result.append(AnalysisRecord(original, row["analysis_lesion_id"], row["split_group_id"],
                    row["active"] == "true", row["representative_image_id"], row["inactive_reason"]))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ContractError("invalid_analysis_csv", "Proposed analysis CSV must be readable.") from exc
    return tuple(result)


def load_proposed_split(directory: Path, *, expected_report_sha256: str):
    if any(p.is_symlink() for p in (directory.absolute(), *directory.absolute().parents)):
        raise ContractError("unsafe_path", "Proposal paths cannot contain symlinks.")
    directory = directory.resolve(strict=True)
    report_path = rooted_file(directory, "proposal_report.json")
    if file_sha256(report_path) != expected_report_sha256:
        raise ContractError("proposal_report_mismatch", "Proposal differs from the independently retained report hash.")
    report = _json(report_path)
    if (not isinstance(report, dict) or report.get("production_freeze") is not False
            or report.get("production_seal_validated") is not False
            or report.get("final_assignment_accepted") is not False
            or report.get("status") != "accepted_analysis_and_proposed_assignment_verified_not_frozen"):
        raise ContractError("invalid_proposal_artifact", "This loader consumes reviewed proposals, never production seals.")
    hashes = report.get("artifact_sha256")
    required = {"physical_analysis_manifest.csv", "analysis_manifest.csv", "effective_decisions.json",
                "uncertainty_links.json", "allocation_config.json", "train.csv", "validation.csv", "internal_test.csv"}
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ContractError("invalid_proposal_artifact", "Exact complete proposal artifact membership is required.")
    files = tuple(directory.rglob("*"))
    if any(p.is_symlink() or (not p.is_dir() and not p.is_file()) for p in files):
        raise ContractError("unsafe_path", "Proposal evidence contains ordinary files only.")
    if {p.relative_to(directory).as_posix() for p in files if p.is_file()} != required | {"proposal_report.json"}:
        raise ContractError("unexpected_proposal_files", "Missing or unlisted proposal files are forbidden.")
    for reference, expected in hashes.items():
        if file_sha256(rooted_file(directory, reference)) != expected:
            raise ContractError("proposal_payload_mismatch", "A proposed metadata artifact changed.", artifact=reference)
    if canonical_json_bytes(_json(directory / "allocation_config.json")) != canonical_json_bytes(configuration()):
        raise ContractError("nonprescribed_assignment", "Saved configuration must equal the versioned fixed-seed allocator.")
    physical = validate_analysis(_csv(directory / "physical_analysis_manifest.csv"))
    links = _json(directory / "uncertainty_links.json")
    if not isinstance(links, list) or any(not isinstance(link, dict) for link in links):
        raise ContractError("invalid_proposal_artifact", "Structured uncertainty links are required.")
    try:
        uncertainties = tuple(ReviewedUncertainty(**link) for link in links)
    except TypeError as exc:
        raise ContractError("invalid_proposal_artifact", "Uncertainty link fields must match the versioned contract.") from exc
    grouping = build_conservative_groups(physical, uncertainties)
    if _csv(directory / "analysis_manifest.csv") != grouping.records:
        raise ContractError("invalid_conservative_grouping", "Serialized groups must equal the complete physical/uncertainty anchor.")
    partitions = {name: _csv(directory / (name + ".csv")) for name in ("train", "validation", "internal_test")}
    verification = verify_constrained_allocation(grouping, partitions)
    if canonical_json_bytes(verification) != canonical_json_bytes(report.get("allocation_verification")):
        raise ContractError("unvalidated_allocation", "Saved allocation report differs from reconstructed membership.")
    if partitions != allocate_constrained_groups(grouping):
        raise ContractError("nonprescribed_assignment", "Saved partitions must equal the fixed-seed constrained algorithm.")
    return grouping, partitions, report
