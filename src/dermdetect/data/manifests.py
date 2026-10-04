"""Canonical manifests and explicitly rooted file verification."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from .schemas import ContractError, MANIFEST_FIELDS, ManifestRecord, validate_manifest


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ContractError("unreadable_file", "Cannot read the referenced payload.",
                            path=str(path)) from exc
    return digest.hexdigest()


def rooted_file(root: Path, reference: str) -> Path:
    """Reject absolute/traversing references and symlink escapes before opening."""
    relative = PurePosixPath(reference)
    if (not reference or relative.is_absolute() or ".." in relative.parts
            or "\\" in reference or relative.as_posix() != reference):
        raise ContractError("unsafe_path", "Payload reference must be canonical and relative.",
                            path=reference)
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / reference).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ContractError("unsafe_path", "Payload reference escapes its explicitly authorized root.",
                            path=reference)
    if not candidate.is_file():
        raise ContractError("unreadable_file", "Referenced payload is not a readable file.",
                            path=reference)
    return candidate


def build_manifest(metadata: Iterable[Mapping[str, object]], root: Path) -> tuple[ManifestRecord, ...]:
    """Hash explicitly authorized payloads. M1 calls this only on invented text."""
    records = []
    for raw in metadata:
        row = dict(raw)
        # Verify metadata before touching its file; checksum is computed below.
        ManifestRecord.from_mapping({**row, "sha256": "0" * 64})
        row["sha256"] = file_sha256(rooted_file(root, row["path"]))
        records.append(ManifestRecord.from_mapping(row))
    return validate_manifest(records)


def verify_files(records: Iterable[ManifestRecord], root: Path) -> None:
    for row in validate_manifest(records):
        actual = file_sha256(rooted_file(root, row.path))
        if actual != row.sha256:
            raise ContractError("checksum_mismatch", "Payload content differs from its manifest.",
                                image_id=row.image_id, expected=row.sha256, actual=actual)


def manifest_csv_bytes(records: Iterable[ManifestRecord]) -> bytes:
    rows = validate_manifest(records)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row.to_dict() for row in rows)
    return stream.getvalue().encode("utf-8")


def read_manifest(path: Path) -> tuple[ManifestRecord, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ContractError("invalid_header", "CSV header does not match the versioned schema.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ContractError("invalid_csv_row", "CSV row has the wrong number of fields.")
            rows.append(ManifestRecord.from_mapping(row))
    return validate_manifest(rows)
