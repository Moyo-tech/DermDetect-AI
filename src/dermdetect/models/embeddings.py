"""Offline frozen image adapter and immutable, hash-bound float32 cache."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(*, manifest_sha256, image_sha256, revision, processor, software, dimension,
            normalization="l2", model_id="google/medsiglip-448"):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Immutable checkpoint commit required")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) or not image_sha256:
        raise ValueError("Manifest/image hashes required")
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in image_sha256.values()):
        raise ValueError("Invalid image checksum")
    if type(dimension) is not int or dimension < 1 or normalization != "l2" or not processor or not software:
        raise ValueError("Explicit processor, versions, output dimension and normalization required")
    return dict(manifest_sha256=manifest_sha256, image_sha256=dict(sorted(image_sha256.items())),
                model_id=model_id, revision=revision, processor=processor, software=software,
                dimension=dimension, normalization=normalization, output="image_embeds", dtype="float32")


def freeze_image_features(model, pixel_values):
    """SigLIP image_embeds convention: projected image features, L2 normalized.

    The caller supplies an offline, previously approved model and pinned processor.
    No tokenizer/text encoder computation or automatic model loading occurs here.
    """
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
        if features.ndim != 2 or not torch.isfinite(features).all() or (features.norm(dim=-1) == 0).any():
            raise ValueError("Invalid image features")
        return torch.nn.functional.normalize(features, dim=-1).float().cpu()


def save_cache(directory, ids, features, key):
    ids = tuple(ids)
    array = np.asarray(features, dtype=np.float32)
    validate(ids, array, key)
    directory = Path(directory)
    directory.mkdir()  # Never overwrite any existing cache.
    np.save(directory / "features.npy", array, allow_pickle=False)
    metadata = {"binding": key, "ids": ids, "features_sha256": sha(directory / "features.npy")}
    (directory / "cache.json").write_text(json.dumps(metadata, sort_keys=True, allow_nan=False) + "\n")
    return sha(directory / "cache.json")


def validate(ids, array, key):
    if len(ids) != len(set(ids)) or set(ids) != set(key["image_sha256"]):
        raise ValueError("Exact one-to-one manifest coverage required")
    if array.dtype != np.float32 or array.shape != (len(ids), key["dimension"]) or not np.isfinite(array).all():
        raise ValueError("Finite float32 matrix with bound dimension required")
    if len(ids) < 2 or not np.any(np.var(array, axis=0) > 0):
        raise ValueError("Nonzero feature variance required")
    if not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Expected normalized image_embeds")


def load_cache(directory, expected_key, expected_metadata_sha256):
    directory = Path(directory)
    if sha(directory / "cache.json") != expected_metadata_sha256:
        raise ValueError("Cache metadata changed")
    metadata = json.loads((directory / "cache.json").read_text())
    if metadata["binding"] != expected_key or sha(directory / "features.npy") != metadata["features_sha256"]:
        raise ValueError("Cache binding/payload changed")
    array = np.load(directory / "features.npy", allow_pickle=False)
    ids = tuple(metadata["ids"])
    validate(ids, array, expected_key)
    return ids, array


def evaluation_resize(image):
    """Proposed TF-compatible half-pixel bilinear resize, antialias=False.

    Analytic synthetic checks only; reference TF/pinned official processor validation
    is required before real extraction. uint8 truncation follows the model card.
    """
    from PIL import Image
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    resized = torch.nn.functional.interpolate(tensor, size=(448, 448), mode="bilinear",
                                              align_corners=False, antialias=False)
    return Image.fromarray(resized[0].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8))


def load_local_encoder(snapshot, *, expected_files, revision):
    """Load an externally approved, hash-bound local snapshot; never fetch remote code.

    Terms/download authorization remains a human gate outside this offline adapter.
    expected_files must cover every snapshot file, including config and processor.
    """
    from transformers import AutoModel, AutoImageProcessor
    snapshot = Path(snapshot)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not expected_files:
        raise ValueError("Pinned revision and exact local file hashes required")
    paths = tuple(snapshot.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Local snapshot symlinks must be materialized and inspected")
    actual = {p.relative_to(snapshot).as_posix():sha(p) for p in paths if p.is_file()}
    if actual != expected_files:
        raise ValueError("Local checkpoint composition/hash mismatch")
    model = AutoModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                                     use_safetensors=True).cpu().eval().requires_grad_(False)
    processor = AutoImageProcessor.from_pretrained(snapshot,local_files_only=True,
                                                  trust_remote_code=False,use_fast=False)
    return model, processor
