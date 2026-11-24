"""Inspected standard-library CPU loader. No decoder, model or outcome API.

Run only through the evaluator's hash-checking sandbox launcher. Inputs are the
staged train/validation bundle; outputs are data-contract verification evidence.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from numerical_policy import (CLASS_ORDER, POLICY, View, training_weights,
                              paired_cluster_draws)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError("Repeated JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=unique)


def verify_bundle(bundle, expected_hash):
    bundle = bundle.resolve(strict=True)
    manifest_path = bundle / "bundle_manifest.json"
    if digest(manifest_path) != expected_hash: raise ValueError("Bundle manifest changed")
    manifest = read_json(manifest_path)
    if (manifest.get("policy") != "staged-development-bundle-v1"
            or manifest.get("status") != "staged_only"
            or manifest.get("production_seal_validated") is not False):
        raise ValueError("Unrecognized immutable staging manifest")
    paths = tuple(bundle.rglob("*"))
    if any(p.is_symlink() or (not p.is_file() and not p.is_dir()) for p in paths):
        raise ValueError("Bundle contains a symlink or special file")
    expected = manifest.get("files")
    if not isinstance(expected, dict) or not expected: raise ValueError("Missing exact file manifest")
    if {p.relative_to(bundle).as_posix() for p in paths if p.is_file()} != set(expected) | {"bundle_manifest.json"}:
        raise ValueError("Unexpected bundle membership")
    for name, checksum in expected.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or not re.fullmatch(r"[0-9a-f]{64}",str(checksum))):
            raise ValueError("Unsafe manifest route or digest")
        if digest(bundle / name) != checksum: raise ValueError("Bundle payload changed")
    views = {}; image_ids = set(); physical_owner = {}; group_owner = {}
    for partition in ("train", "validation"):
        rows = read_json(bundle / "manifests" / (partition + ".json"))
        if not isinstance(rows,list) or not rows: raise ValueError("Empty partition")
        selected = []
        for row in rows:
            if set(row) != {"image_id","dx","analysis_lesion_id","split_group_id","path","sha256"}:
                raise ValueError("Only redacted development fields allowed")
            image = row["image_id"]
            if not isinstance(image,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",image) or image in image_ids:
                raise ValueError("Invalid or repeated development image")
            image_ids.add(image)
            if row["path"] != f"data/{partition}/{image}.payload" or expected.get(row["path"]) != row["sha256"]:
                raise ValueError("Partition payload route/hash mismatch")
            for key,owners in (("analysis_lesion_id",physical_owner),("split_group_id",group_owner)):
                if row[key] in owners and owners[row[key]] != partition:
                    raise ValueError("Physical or conservative group crosses partitions")
                owners[row[key]] = partition
            selected.append(View(image,row["analysis_lesion_id"],row["split_group_id"],row["dx"],row["sha256"]))
        views[partition] = tuple(selected)
    listed_payloads = {name for name in expected if name.startswith("data/")}
    if listed_payloads != {f"data/{p}/{v.image_id}.payload" for p,rows in views.items() for v in rows}:
        raise ValueError("Unmanifested development payload")
    allowed_metadata = {"manifests/train.json","manifests/validation.json"}
    if any(not (name.startswith("code/") or name in listed_payloads or name in allowed_metadata) for name in expected):
        raise ValueError("Unexpected source/evidence artifact in development")
    return manifest, views


def probe_boundary(config):
    """Protected-file direct probes never read contents, even on a failed denial."""
    checks = {}
    for name, path in config["denied_open"].items():
        try:
            fd = os.open(path, os.O_RDONLY)
        except PermissionError: checks[name] = True
        except Exception: checks[name] = False
        else:
            os.close(fd); checks[name] = False
    for name, path in config["denied_zip"].items():
        try:
            with zipfile.ZipFile(path) as archive: archive.infolist()
        except PermissionError: checks[name] = True
        except Exception: checks[name] = False
        else: checks[name] = False
    for name, path in config["denied_write"].items():
        try:
            fd = os.open(path,os.O_WRONLY|os.O_APPEND)
        except PermissionError: checks[name] = True
        except Exception: checks[name] = False
        else:
            os.close(fd); checks[name] = False
    for name,path in config["denied_metadata"].items():
        try: os.stat(path)
        except PermissionError: checks[name] = True
        except Exception: checks[name] = False
        else: checks[name] = False
    try:
        with socket.socket() as connection: connection.connect(("127.0.0.1",9))
    except PermissionError: checks["network_denied"] = True
    except Exception: checks["network_denied"] = False
    else: checks["network_denied"] = False
    try: subprocess.run(["/bin/echo","invented-helper"],check=True,capture_output=True)
    except PermissionError: checks["helper_exec_denied"] = True
    except Exception: checks["helper_exec_denied"] = False
    else: checks["helper_exec_denied"] = False
    if not all(checks.values()): raise ValueError("Boundary refusal failed: " + repr([k for k,v in checks.items() if not v]))
    return checks


def verify_and_report(bundle, expected_hash, output, probe_config=None, probe_hash=None):
    manifest, views = verify_bundle(bundle,expected_hash)
    weights = training_weights(views["train"])
    summaries = {}
    for partition,rows in views.items():
        units = {v.physical_id:v.label for v in rows}
        summaries[partition] = {"active_views":len(rows),"physical_units":len(units),
            "class_physical_counts":dict(Counter(units.values())),"conservative_groups":len({v.cluster_id for v in rows})}
    h = hashlib.sha256(); sizes = Counter(); draw_count = 0
    for selected,multiplicity in paired_cluster_draws(views["validation"],partition="validation"):
        h.update(canonical([selected,multiplicity])); sizes[sum(multiplicity.values())] += 1; draw_count += 1
    evidence = {"status":"development_cpu_contracts_verified","bundle_manifest_sha256":expected_hash,
        "policy":POLICY,"partitions":summaries,"weight_sum":str(sum(weights["image_weights"].values())),
        "class_loss_mass":{c:str(sum(weights["image_weights"][v.image_id] for v in views["train"] if v.label==c)/len(views["train"])) for c in CLASS_ORDER},
        "validation_paired_draws":draw_count,"validation_draws_sha256":h.hexdigest(),
        "validation_replicate_physical_sizes":dict(sizes),"predictions_or_outcomes_read":False,
        "network_or_model_resources_used":False,"production_seal_validated":False}
    if probe_config is not None:
        if digest(probe_config) != probe_hash: raise ValueError("Boundary probe configuration changed")
        evidence["boundary_checks"] = probe_boundary(read_json(probe_config))
    output = output.resolve(strict=True)
    evidence["output_write_allowed"] = True
    (output / "worker_verification.json").write_bytes(canonical(evidence)+b"\n")
    serialized = {k: {image: [w.numerator,w.denominator] for image,w in weights[k].items()}
                  for k in ("class_weights","image_weights")}
    (output / "training_weights.json").write_bytes(canonical({"policy":POLICY,**serialized})+b"\n")
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--expected-manifest-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--probe-config",type=Path)
    parser.add_argument("--probe-sha256")
    args=parser.parse_args()
    try:
        evidence=verify_and_report(args.bundle,args.expected_manifest_sha256,args.output,args.probe_config,args.probe_sha256)
        print(json.dumps({"status":evidence["status"],"boundary_checks":len(evidence.get("boundary_checks",{})),"partitions":evidence["partitions"]},sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status":"failed","error":type(exc).__name__,"message":str(exc)}),file=sys.stderr)
        raise SystemExit(2)
