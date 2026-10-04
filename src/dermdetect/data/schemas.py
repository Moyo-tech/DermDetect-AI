"""Manifest schema and structured failures. No images or models are loaded."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping

CLASS_ORDER = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MANIFEST_FIELDS = (
    "image_id", "lesion_id", "dx", "source", "source_version", "path", "sha256",
)


class ContractError(ValueError):
    """A machine-readable integrity failure, never a silently dropped record."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "details": self.details}


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    lesion_id: str
    dx: str
    source: str
    source_version: str
    path: str
    sha256: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> ManifestRecord:
        missing = [field for field in MANIFEST_FIELDS
                   if not isinstance(row.get(field), str) or not row[field].strip()]
        if missing:
            raise ContractError("missing_field", "Required fields must be nonempty strings.",
                                fields=missing)
        if any(row[field] != row[field].strip() for field in MANIFEST_FIELDS):
            raise ContractError("noncanonical_field", "Surrounding whitespace is rejected.")
        if row["dx"] not in CLASS_ORDER:
            raise ContractError("unknown_class", "Diagnosis code is outside the seven-class taxonomy.",
                                image_id=row["image_id"], dx=row["dx"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ContractError("invalid_checksum", "SHA-256 must be 64 lowercase hex characters.",
                                image_id=row["image_id"])
        return cls(**{field: row[field] for field in MANIFEST_FIELDS})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_manifest(records: Iterable[ManifestRecord]) -> tuple[ManifestRecord, ...]:
    """Validate membership; duplicate content is audited separately."""
    rows = tuple(records)
    if not rows:
        raise ContractError("empty_manifest", "A manifest cannot be empty.")
    seen_ids: set[str] = set()
    lesion_labels: dict[str, str] = {}
    for row in rows:
        ManifestRecord.from_mapping(row.to_dict())
        if row.image_id in seen_ids:
            raise ContractError("duplicate_image_id", "Image ID appears more than once.",
                                image_id=row.image_id)
        seen_ids.add(row.image_id)
        if row.lesion_id in lesion_labels and lesion_labels[row.lesion_id] != row.dx:
            raise ContractError("lesion_label_conflict", "Images of one lesion disagree on class.",
                                lesion_id=row.lesion_id)
        lesion_labels[row.lesion_id] = row.dx
    return tuple(sorted(rows, key=lambda row: row.image_id))
