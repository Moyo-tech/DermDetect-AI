"""Read explicitly supplied flat HAM JPG ZIPs without extracting or decoding.

Only named archives are opened. This is an unsplit integrity inventory, not a
production split, duplicate adjudication or an outcome-access boundary.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import stat
from typing import Mapping
import zipfile
import zlib

from .ham import HamIngestion, parse_ham_metadata
from .manifests import sha256_bytes
from .schemas import ContractError, ManifestRecord, validate_manifest

ARCHIVE_ADAPTER_VERSION = "ham-csv-zip-jpg-v1"
BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class HamArchiveIngestion(HamIngestion):
    archive_inventory: tuple[dict, ...]

    def integrity_report(self) -> dict:
        return {**super().integrity_report(),
                "adapter_version": ARCHIVE_ADAPTER_VERSION,
                "storage_mode": "zip_member_streams",
                "archives": list(self.archive_inventory),
                "archive_crc": "passed_for_all_referenced_members",
                "archive_membership": "all_jpg_members_referenced_exactly_once",
                "metadata_source_match": "local_bytes_hashed; original_csv_checksum_not_independently_verified"}


def _digests(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)  # Match the publisher's identity checksum only.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def ingest_ham_archives(metadata_path: Path, image_archives: Mapping[str, Path], *,
                        source_version: str, provenance_reference: str,
                        permitted_use_reference: str, data_kind: str,
                        expected_archive_md5: Mapping[str, str] | None = None,
                        max_member_bytes: int = 32 * 1024 * 1024,
                        max_total_bytes: int = 8_000_000_000) -> HamArchiveIngestion:
    """Hash each JPG through a bounded buffer, check CRC, retain every CSV row.

    All metadata and member membership are checked before any member payload
    is opened. Flat JPG layout is deliberate; no extraction or recursive scan
    occurs. Maxima are resource guards: exceeding them stops the entire run.
    Caller references record evidence and do not themselves authorize access.
    """
    if (data_kind not in ("synthetic", "real")
            or any(not isinstance(value, str) or not value.strip() or value != value.strip()
                   for value in (provenance_reference, permitted_use_reference))):
        raise ContractError("invalid_source_record", "Explicit source/use references and data kind are required.")
    if (not image_archives or any(not isinstance(alias, str)
                                 or not re.fullmatch(r"[A-Za-z0-9_-]+", alias)
                                 for alias in image_archives)):
        raise ContractError("invalid_image_archives", "Explicit filename-safe archive aliases are required.")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (max_member_bytes, max_total_bytes)):
        raise ContractError("invalid_archive_limits", "Archive resource limits must be positive integers.")
    if expected_archive_md5 is not None and (
            set(expected_archive_md5) != set(image_archives)
            or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{32}", digest)
                   for digest in expected_archive_md5.values())):
        raise ContractError("invalid_source_record", "Publisher MD5 evidence must cover exactly the supplied archives.")
    try:
        content = Path(metadata_path).read_bytes()
        metadata = parse_ham_metadata(content, source_version)
        paths = {alias: Path(path).resolve(strict=True) for alias, path in sorted(image_archives.items())}
        if len(set(paths.values())) != len(paths) or any(not path.is_file() for path in paths.values()):
            raise ContractError("invalid_image_archives", "Archives must be distinct existing files.")
        if sum(path.stat().st_size for path in paths.values()) > max_total_bytes:
            raise ContractError("archive_limit", "Combined archive bytes exceed the resource limit.")
        members = {}
        inventory = []
        total_bytes = 0
        with ExitStack() as stack:
            archives = {alias: stack.enter_context(zipfile.ZipFile(path)) for alias, path in paths.items()}
            for alias, archive in archives.items():
                seen = set()
                sizes = 0
                for info in archive.infolist():
                    if info.filename in seen:
                        raise ContractError("duplicate_archive_member", "ZIP contains a repeated member name.", alias=alias)
                    seen.add(info.filename)
                    if (info.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]+\.jpg", info.filename)
                            or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
                            or info.flag_bits & 1):
                        raise ContractError("unsafe_archive_member", "Only flat, unencrypted regular JPG members are supported.", alias=alias)
                    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                        raise ContractError("unsupported_archive_compression", "Only stored or deflated JPG members are supported.")
                    if info.file_size > max_member_bytes:
                        raise ContractError("archive_limit", "A member exceeds the resource limit.", alias=alias)
                    if info.filename in members:
                        raise ContractError("ambiguous_image", "Image occurs in multiple archives.")
                    members[info.filename] = (alias, info)
                    sizes += info.file_size
                total_bytes += sizes
                inventory.append({"alias": alias, "filename": paths[alias].name,
                                  "bytes": paths[alias].stat().st_size,
                                  "members": len(seen), "uncompressed_bytes": sizes})
            if total_bytes > max_total_bytes:
                raise ContractError("archive_limit", "Combined uncompressed bytes exceed the resource limit.")
            wanted = {f"{row.image_id}.jpg" for row in metadata}
            missing, unlisted = wanted - set(members), set(members) - wanted
            if missing or unlisted:
                raise ContractError("archive_membership_mismatch", "Metadata and archive JPG membership disagree.",
                                    missing_count=len(missing), unlisted_count=len(unlisted))
            # Source archive identity checks precede member payload reads.
            for item in inventory:
                alias = item["alias"]
                item["sha256"], item["md5"] = _digests(paths[alias])
                item["publisher_md5_match"] = None
                if expected_archive_md5 is not None:
                    if item["md5"] != expected_archive_md5[alias]:
                        raise ContractError("archive_source_mismatch", "Archive differs from the publisher checksum.", alias=alias)
                    item["publisher_md5_match"] = True
            records = []
            for row in metadata:
                alias, info = members[f"{row.image_id}.jpg"]
                digest = hashlib.sha256()
                count = 0
                with archives[alias].open(info) as handle:
                    for block in iter(lambda: handle.read(BLOCK_BYTES), b""):
                        count += len(block)
                        if count > max_member_bytes:
                            raise ContractError("archive_limit", "Stream exceeds the member resource limit.")
                        digest.update(block)
                if count != info.file_size:
                    raise ContractError("archive_size_mismatch", "Member byte count disagrees with its ZIP record.")
                records.append(ManifestRecord(row.image_id, row.lesion_id, row.dx, row.source,
                                              row.source_version, f"{alias}/{info.filename}", digest.hexdigest()))
        return HamArchiveIngestion(validate_manifest(records), sha256_bytes(content), source_version,
                                   provenance_reference, permitted_use_reference, data_kind, tuple(inventory))
    except ContractError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, zlib.error) as exc:
        raise ContractError("unreadable_archive_input", "Cannot read metadata or a valid supplied ZIP/member.",
                            error_type=type(exc).__name__) from exc
