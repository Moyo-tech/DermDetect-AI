"""HAM-format ingestion preparation; callers must separately authorize real reads.

This adapter does not decode images, adjudicate duplicates, generate a split, or
enforce an outcome seal. Tests use invented CSV rows and text bytes only.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
from typing import Mapping

from .manifests import file_sha256, manifest_csv_bytes, rooted_file, sha256_bytes
from .overlap import exact_duplicates
from .schemas import CLASS_ORDER, ContractError, ManifestRecord, validate_manifest

REQUIRED_COLUMNS = ("image_id", "lesion_id", "dx")
ADAPTER_VERSION = "ham-csv-jpg-v1"


@dataclass(frozen=True)
class HamIngestion:
    records: tuple[ManifestRecord, ...]
    metadata_sha256: str
    source_version: str
    provenance_reference: str
    permitted_use_reference: str
    data_kind: str

    def integrity_report(self) -> dict:
        duplicates = exact_duplicates(self.records)
        return {
            "adapter_version": ADAPTER_VERSION,
            "data_kind": self.data_kind,
            "status": "pending_integrity_review",
            "split_ready": False,
            "metadata_sha256": self.metadata_sha256,
            "manifest_sha256": sha256_bytes(manifest_csv_bytes(self.records)),
            "source": "HAM10000",
            "source_version": self.source_version,
            "provenance_reference": self.provenance_reference,
            "permitted_use_reference": self.permitted_use_reference,
            "evidence_status": "caller_supplied_not_independently_verified",
            "records": len(self.records),
            "lesions": len({row.lesion_id for row in self.records}),
            "missing_classes": [dx for dx in CLASS_ORDER
                                if not any(row.dx == dx for row in self.records)],
            "exact_duplicate_candidates": duplicates,
            "image_decoding": "not_checked",
            "near_duplicates": "not_checked",
            "patient_grouping": "not_verified",
            "split": "not_created",
        }


def parse_ham_metadata(content: bytes, source_version: str) -> tuple[ManifestRecord, ...]:
    """Validate the entire CSV before any payload is opened; retain all rows.

    Additional columns are permitted, but source bytes are hashed at ingestion.
    Patient IDs are never inferred from those columns. Returned paths/checksums
    are placeholders until explicitly rooted file resolution below.
    """
    if not isinstance(source_version, str) or not source_version.strip() or source_version != source_version.strip():
        raise ContractError("invalid_source_record", "An explicit canonical source version is required.")
    try:
        decoded = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
        header = reader.fieldnames
        if (not header or len(header) != len(set(header))
                or any(not name or name != name.strip() for name in header)
                or not set(REQUIRED_COLUMNS).issubset(header)):
            raise ContractError("invalid_ham_header", "Unique columns including image_id, lesion_id and dx are required.")
        records = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ContractError("invalid_csv_row", "CSV row has the wrong number of fields.")
            record = ManifestRecord.from_mapping({
                **{name: row[name] for name in REQUIRED_COLUMNS},
                "source": "HAM10000", "source_version": source_version,
                "path": "pending.jpg", "sha256": "0" * 64,
            })
            if not re.fullmatch(r"[A-Za-z0-9_-]+", record.image_id):
                raise ContractError("unsafe_image_id", "Image ID must be a filename-safe stem.",
                                    image_id=record.image_id)
            records.append(record)
        return validate_manifest(records)
    except (UnicodeError, csv.Error) as exc:
        raise ContractError("invalid_ham_csv", "Metadata must be valid UTF-8 CSV.") from exc


def _roots(image_roots: Mapping[str, Path]) -> dict[str, Path]:
    if not image_roots:
        raise ContractError("invalid_image_roots", "At least one explicit image root is required.")
    resolved = {}
    for alias, root in sorted(image_roots.items()):
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
            raise ContractError("invalid_image_roots", "Root aliases must be filename-safe.")
        try:
            path = Path(root).resolve(strict=True)
        except OSError as exc:
            raise ContractError("invalid_image_roots", "An image root is unavailable.", alias=alias) from exc
        if not path.is_dir() or any(path.is_relative_to(other) or other.is_relative_to(path)
                                    for other in resolved.values()):
            raise ContractError("invalid_image_roots", "Image roots must be directories without overlap.", alias=alias)
        resolved[alias] = path
    return resolved


def ingest_ham(metadata_path: Path, image_roots: Mapping[str, Path], *,
               source_version: str, provenance_reference: str,
               permitted_use_reference: str, data_kind: str) -> HamIngestion:
    """Read only supplied CSV and '<image_id>.jpg' files in supplied roots.

    Manifest paths begin with a logical root alias, avoiding machine paths.
    Unrelated files are not scanned; every metadata ID must resolve exactly
    once. References record the caller's evidence, not an access approval.
    No real invocation is authorized solely by the existence of this function.
    """
    if data_kind not in ("synthetic", "real"):
        raise ContractError("invalid_source_record", "Explicit synthetic or real data kind is required.")
    for value in (provenance_reference, permitted_use_reference):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ContractError("invalid_source_record", "Canonical provenance and permitted-use references are required.")
    try:
        content = Path(metadata_path).read_bytes()
    except OSError as exc:
        raise ContractError("unreadable_metadata", "Cannot read the supplied metadata CSV.") from exc
    metadata = parse_ham_metadata(content, source_version)
    roots = _roots(image_roots)
    references = []
    for row in metadata:
        matches = []
        filename = f"{row.image_id}.jpg"
        for alias, root in roots.items():
            candidate = root / filename
            if candidate.exists() or candidate.is_symlink():
                matches.append((alias, rooted_file(root, filename)))
        if not matches:
            raise ContractError("missing_image", "Metadata ID has no referenced JPG file.", image_id=row.image_id)
        if len(matches) != 1:
            raise ContractError("ambiguous_image", "Metadata ID occurs in multiple image roots.", image_id=row.image_id)
        alias, path = matches[0]
        references.append((row, alias, path))
    # Resolve all membership before hashing any payload; failures never drop rows.
    records = validate_manifest(ManifestRecord(
        row.image_id, row.lesion_id, row.dx, row.source, row.source_version,
        f"{alias}/{row.image_id}.jpg", file_sha256(path),
    ) for row, alias, path in references)
    return HamIngestion(records, sha256_bytes(content), source_version,
                        provenance_reference, permitted_use_reference, data_kind)


def verify_ham_files(records: tuple[ManifestRecord, ...], image_roots: Mapping[str, Path]) -> None:
    """Recheck aliased manifest paths against the same explicitly supplied roots."""
    roots = _roots(image_roots)
    for row in validate_manifest(records):
        alias, separator, filename = row.path.partition("/")
        if (not separator or alias not in roots or filename != f"{row.image_id}.jpg"
                or not re.fullmatch(r"[A-Za-z0-9_-]+", row.image_id)):
            raise ContractError("unsafe_path", "Manifest path must match its ID and a supplied root alias.")
        actual = file_sha256(rooted_file(roots[alias], filename))
        if actual != row.sha256:
            raise ContractError("checksum_mismatch", "Payload differs from the HAM manifest.", image_id=row.image_id)
