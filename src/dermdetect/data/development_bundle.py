"""Staged train/validation bundle composition, never production seal evidence.

This byte-only API does not ingest archives or assign real data. Callers must
provide reviewed inputs and hash-bound approval records. Full source context is
written separately for an evaluator, never copied into development inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Iterable, Mapping

from .analysis import AnalysisRecord, validate_analysis
from .conservative_groups import ConservativeGrouping, validate_conservative_groups
from .constrained_allocation import (ALGORITHM as CONSTRAINED_ALGORITHM,
                                     allocate_constrained_groups, verify_constrained_allocation)
from .group_allocation import SPLITS, allocate_groups, verify_allocation
from .manifests import canonical_json_bytes, sha256_bytes
from .schemas import ContractError

POLICY = "staged-development-bundle-v1"
APPROVAL_SCOPES = ("resolved_analysis", "fixed_assignment", "inspected_code", "development_export")


@dataclass(frozen=True)
class BundleApproval:
    scope: str
    input_sha256: str
    reviewer: str
    evidence: str
    approved: bool = False


def _relative(value: str) -> str:
    if not isinstance(value, str):
        raise ContractError("unsafe_bundle_path", "Bundle file names must be strings.")
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or path.as_posix() != value or value == "."):
        raise ContractError("unsafe_bundle_path", "Bundle file names must be canonical relative paths.")
    return value


def _code_manifest(code_files: Mapping[str, bytes]) -> dict[str, str]:
    if not code_files:
        raise ContractError("missing_bundle_code", "An inspected code payload is required.")
    result = {}
    for name, payload in sorted(code_files.items()):
        _relative(name)
        if type(payload) is not bytes:
            raise ContractError("invalid_bundle_payload", "Code payloads must be bytes.")
        result[name] = sha256_bytes(payload)
    names = tuple(result)
    if any(PurePosixPath(a) in PurePosixPath(b).parents for a in names for b in names if a != b):
        raise ContractError("unsafe_bundle_path", "File/directory name collisions are forbidden.")
    return result


def _inputs(records: Iterable[AnalysisRecord], partitions: Mapping[str, Iterable[AnalysisRecord]],
            code_files: Mapping[str, bytes], protected_content_sha256: Iterable[str],
            grouping: ConservativeGrouping | None = None):
    if grouping is None:
        rows = validate_analysis(records)
    else:
        grouping = validate_conservative_groups(grouping)
        rows = tuple(records)
        if rows != grouping.records:
            raise ContractError("invalid_conservative_grouping", "Bundle records must equal the complete validated grouping anchor.")
    assigned = {name: tuple(members) for name, members in partitions.items()}
    verification = (verify_allocation(rows, assigned) if grouping is None
                    else verify_constrained_allocation(grouping, assigned))
    expected = allocate_groups(rows) if grouping is None else allocate_constrained_groups(grouping)
    if any(tuple(sorted(assigned[name], key=lambda row: row.original.image_id)) != expected[name]
           for name in SPLITS):
        raise ContractError("nonprescribed_assignment", "Inputs must match the approved deterministic allocation.")
    protected = frozenset(protected_content_sha256)
    if not protected or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in protected):
        raise ContractError("missing_protected_fingerprints", "Evaluator-side source/evidence hashes are required.")
    code = _code_manifest(code_files)
    development = tuple(row for name in ("train", "validation") for row in expected[name] if row.active)
    # Original paths are deliberately ignored; payload names derive from IDs.
    for row in development:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row.original.image_id):
            raise ContractError("unsafe_bundle_path", "Image identifiers cannot become filesystem routes.")
    protected_payloads = protected | {row.original.sha256 for row in expected["internal_test"]}
    if set(code.values()) & protected_payloads:
        raise ContractError("protected_content_copy", "Inspected code cannot contain a byte-identical protected artifact.")
    analysis_content = [row.to_dict() for row in rows]
    grouping_digest = None
    if grouping is not None:
        grouping_digest = sha256_bytes(canonical_json_bytes({"policy": grouping.policy,
            "physical_records": [row.to_dict() for row in grouping.physical_records],
            "uncertainties": [asdict(link) for link in grouping.uncertainties]}))
        analysis_content = {"records": analysis_content, "grouping_sha256": grouping_digest}
    analysis_hash = sha256_bytes(canonical_json_bytes(analysis_content))
    partition_hash = sha256_bytes(canonical_json_bytes(
        {name: [row.original.image_id for row in expected[name]] for name in SPLITS}))
    code_hash = sha256_bytes(canonical_json_bytes(code))
    export_context = {"policy": POLICY,
        "analysis_sha256": analysis_hash, "partition_sha256": partition_hash,
        "code_manifest_sha256": code_hash, "protected_content_sha256": sorted(protected)}
    if grouping is not None:
        export_context.update(allocation_policy=CONSTRAINED_ALGORITHM, grouping_sha256=grouping_digest)
    export_hash = sha256_bytes(canonical_json_bytes(export_context))
    hashes = dict(zip(APPROVAL_SCOPES, (analysis_hash, partition_hash, code_hash, export_hash)))
    return rows, expected, development, verification, code, protected_payloads, hashes


def bundle_input_hashes(*, records: Iterable[AnalysisRecord],
                        partitions: Mapping[str, Iterable[AnalysisRecord]],
                        code_files: Mapping[str, bytes], protected_content_sha256: Iterable[str],
                        grouping: ConservativeGrouping | None = None) -> dict[str, str]:
    """Return proposed input hashes for review; grants no export approval."""
    return _inputs(records, partitions, code_files, protected_content_sha256, grouping)[-1]


def _safe_location(path: Path, *, existing: bool = False) -> Path:
    absolute = path.absolute()
    if ".." in absolute.parts:
        raise ContractError("unsafe_bundle_path", "Filesystem traversal is forbidden.")
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ContractError("unsafe_bundle_path", "Bundle/provenance routes cannot contain symlinks.")
    if existing:
        if not absolute.is_dir():
            raise ContractError("invalid_bundle_location", "Existing directory required.")
    elif absolute.exists() or not absolute.parent.is_dir():
        raise ContractError("bundle_overwrite", "Destination must be new with an existing parent directory.")
    return absolute.resolve()


def verify_staged_bundle(*, destination: Path, expected_manifest_sha256: str) -> dict:
    """Verify exact staged bytes against an evaluator-retained manifest hash.

    This is composition verification, not runtime or OS isolation evidence.
    """
    target = _safe_location(destination, existing=True)
    paths = tuple(target.rglob("*"))
    if any(path.is_symlink() or (not path.is_dir() and not path.is_file()) for path in paths):
        raise ContractError("unsafe_bundle_path", "Staged bundles contain only ordinary files/directories.")
    manifest_path = target / "bundle_manifest.json"
    if not manifest_path.is_file():
        raise ContractError("bundle_manifest_mismatch", "Staged manifest is absent.")
    payload = manifest_path.read_bytes()
    if sha256_bytes(payload) != expected_manifest_sha256:
        raise ContractError("bundle_manifest_mismatch", "Staged manifest differs from the evaluator-retained hash.")
    manifest = json.loads(payload)
    if (manifest.get("policy") != POLICY or manifest.get("status") != "staged_only"
            or manifest.get("production_seal_validated") is not False):
        raise ContractError("invalid_bundle_manifest", "This API verifies staging only.")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected:
        raise ContractError("invalid_bundle_manifest", "The complete file/hash manifest is required.")
    for reference, digest in expected.items():
        _relative(reference)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ContractError("invalid_bundle_manifest", "File digests must be canonical SHA-256 values.")
    observed = {path.relative_to(target).as_posix() for path in paths if path.is_file()}
    if observed != set(expected) | {"bundle_manifest.json"}:
        raise ContractError("unexpected_bundle_files", "Unlisted or missing staged files are forbidden.")
    for reference, digest in expected.items():
        if sha256_bytes((target / reference).read_bytes()) != digest:
            raise ContractError("bundle_payload_mismatch", "A staged file differs from its reviewed bytes.")
    return {"status": "staged_composition_verified", "production_seal_validated": False,
            "bundle_manifest_sha256": expected_manifest_sha256, "verified_files": len(expected)}


def stage_development_bundle(*, destination: Path, provenance_path: Path,
                             records: Iterable[AnalysisRecord],
                             partitions: Mapping[str, Iterable[AnalysisRecord]],
                             allocation_report: Mapping, code_files: Mapping[str, bytes],
                             payloads: Mapping[str, bytes], approvals: Iterable[BundleApproval],
                             protected_roots: Iterable[Path], protected_content_sha256: Iterable[str],
                             evidence_context: str, grouping: ConservativeGrouping | None = None) -> dict:
    """Write an approved staging bundle; no source reads, OS probe or seal.

    Protected fingerprints must include complete sources and evaluator evidence.
    Copy detection covers byte-identical artifacts; semantic/redacted copies
    require the recorded code inspection. Approval records are attestations, not
    an authentication system. Final runtime inspection and OS denial checks are
    separate requirements.
    """
    rows, assigned, development, verification, code, protected, hashes = _inputs(
        records, partitions, code_files, protected_content_sha256, grouping)
    if dict(allocation_report) != verification:
        raise ContractError("unvalidated_allocation", "The complete current verification report is required.")
    approval_rows = tuple(approvals)
    if (len(approval_rows) != len(APPROVAL_SCOPES)
            or any(not isinstance(item, BundleApproval) for item in approval_rows)
            or {item.scope for item in approval_rows} != set(APPROVAL_SCOPES)):
        raise ContractError("missing_bundle_approval", "Each recorded input approval is required exactly once.")
    for item in approval_rows:
        if (item.approved is not True or item.input_sha256 != hashes[item.scope]
                or not isinstance(item.reviewer, str) or not item.reviewer.strip()
                or not isinstance(item.evidence, str) or not item.evidence.strip()):
            raise ContractError("unapproved_bundle_inputs", "Approval must explicitly bind these inputs and record evidence.")
    if not isinstance(evidence_context, str) or not evidence_context.strip():
        raise ContractError("missing_bundle_context", "Evaluator provenance needs source/review context.")
    target, provenance = _safe_location(destination), _safe_location(provenance_path)
    roots = tuple(_safe_location(root, existing=True) for root in protected_roots)
    if not roots or any(target.is_relative_to(root) or root.is_relative_to(target) for root in roots):
        raise ContractError("unsafe_bundle_location", "Development bundle must be separate from all protected roots.")
    if provenance.is_relative_to(target) or target.is_relative_to(provenance):
        raise ContractError("unsafe_bundle_location", "Evaluator provenance must remain outside the development bundle.")
    if not any(provenance.is_relative_to(root) for root in roots):
        raise ContractError("unsafe_bundle_location", "Evaluator provenance must reside beneath a protected root.")
    expected_ids = {row.original.image_id for row in development}
    if set(payloads) != expected_ids:
        raise ContractError("unexpected_bundle_membership", "Supply only active train/validation byte payloads.")
    for row in development:
        payload = payloads[row.original.image_id]
        if type(payload) is not bytes or sha256_bytes(payload) != row.original.sha256:
            raise ContractError("bundle_payload_mismatch", "Development payload differs from reviewed analysis.")
        if row.original.sha256 in protected:
            raise ContractError("protected_content_copy", "Protected source/evidence cannot be a development payload.")
    files = {"code/" + name: payload for name, payload in code_files.items()}
    for name in ("train", "validation"):
        safe_rows = []
        for row in assigned[name]:
            if not row.active:
                continue
            image_id = row.original.image_id
            reference = f"data/{name}/{image_id}.payload"
            files[reference] = payloads[image_id]
            safe_rows.append({"image_id": image_id, "dx": row.original.dx,
                              "analysis_lesion_id": row.analysis_lesion_id,
                              "split_group_id": row.split_group_id,
                              "path": reference, "sha256": row.original.sha256})
        files[f"manifests/{name}.json"] = canonical_json_bytes(safe_rows)
    data_manifest = {name: sha256_bytes(payload) for name, payload in sorted(files.items())}
    bundle_manifest = {"policy": POLICY, "files": data_manifest,
                       "production_seal_validated": False, "status": "staged_only"}
    manifest_bytes = canonical_json_bytes(bundle_manifest)
    evaluator_record = {"policy": POLICY, "status": "staged_only", "production_seal_validated": False,
        "bundle_manifest_sha256": sha256_bytes(manifest_bytes), "input_hashes": hashes,
        "source_review_context": evidence_context, "approvals": [asdict(item) for item in approval_rows],
        "analysis": [row.to_dict() for row in rows],
        "partitions": {name: [row.original.image_id for row in assigned[name]] for name in SPLITS},
        "allocation_report": verification, "protected_roots": [str(root) for root in roots],
        "code_manifest": code,
        "limitations": ["No runtime inspection or enforced route-denial validation.",
                        "Byte-identical copy checks require supplied protected fingerprints.",
                        "Recorded approvals require human verification; no authentication is claimed."]}
    if grouping is not None:
        evaluator_record["allocation_policy"] = CONSTRAINED_ALGORITHM
        evaluator_record["physical_analysis_anchor"] = [row.to_dict() for row in grouping.physical_records]
        evaluator_record["reviewed_uncertainties"] = [asdict(link) for link in grouping.uncertainties]
    provenance_bytes = canonical_json_bytes(evaluator_record)
    # Exclusive creation; validation finishes before any output is created.
    target.mkdir()
    provenance_created = False
    try:
        for reference, payload in sorted(files.items()):
            path = target / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
        with (target / "bundle_manifest.json").open("xb") as handle:
            handle.write(manifest_bytes)
        with provenance.open("xb") as handle:
            provenance_created = True
            handle.write(provenance_bytes)
    except BaseException:
        shutil.rmtree(target)
        if provenance_created:
            provenance.unlink()
        raise
    return {"status": "staged_only", "production_seal_validated": False,
            "bundle_manifest_sha256": sha256_bytes(manifest_bytes),
            "evaluator_provenance_sha256": sha256_bytes(provenance_bytes),
            "exported_active_images": len(development), "code_files": len(code)}
