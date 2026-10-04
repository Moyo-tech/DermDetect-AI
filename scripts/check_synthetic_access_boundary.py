"""Run a real OS boundary check on invented sentinels, never dataset contents."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.access_boundary import development_profile

PROBE = '''import json, pathlib, sys, zipfile, socket, subprocess
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
results = {}
for name, path in config["allowed"].items():
    results[name] = pathlib.Path(path).read_text() == "INVENTED DEVELOPMENT INPUT\\n"
for name, path in config["denied"].items():
    try:
        if name == "original_zip_loader":
            with zipfile.ZipFile(path) as archive:
                archive.read("invented_test.jpg")
        elif name.startswith("real_original_zip_loader"):
            with zipfile.ZipFile(path) as archive:
                archive.infolist()
        else:
            with pathlib.Path(path).open("rb") as handle:
                handle.read(1)
    except PermissionError:
        results[name] = True
    except Exception:
        results[name] = False
    else:
        results[name] = False
try:
    pathlib.Path(config["forbidden_write"]).write_text("INVENTED")
except PermissionError:
    results["write_outside_outputs_denied"] = True
else:
    results["write_outside_outputs_denied"] = False
pathlib.Path(config["allowed_output"]).write_text("INVENTED OUTPUT\\n")
results["output_write_allowed"] = True
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.connect(("127.0.0.1", 9))
except PermissionError:
    results["network_access_denied"] = True
except Exception:
    results["network_access_denied"] = False
else:
    results["network_access_denied"] = False
try:
    subprocess.run(["/bin/echo", "INVENTED HELPER"], check=True, capture_output=True)
except PermissionError:
    results["unapproved_helper_exec_denied"] = True
except Exception:
    results["unapproved_helper_exec_denied"] = False
else:
    results["unapproved_helper_exec_denied"] = False
print(json.dumps(results, sort_keys=True))
sys.exit(0 if all(results.values()) else 1)
'''


def check(protected_config: Path | None = None) -> dict:
    import zipfile
    with tempfile.TemporaryDirectory(prefix="dermdetect-boundary-") as temporary:
        root = Path(temporary).resolve()
        bundle, outputs, sources = (root / name for name in ("development", "outputs", "evaluator"))
        for folder in (bundle, outputs, sources):
            folder.mkdir()
        allowed = {}
        for name in ("train", "validation"):
            path = bundle / f"{name}.txt"
            path.write_text("INVENTED DEVELOPMENT INPUT\n")
            allowed[name] = str(path)
        denied = {}
        for name in ("full_metadata", "unsplit_manifest", "test_membership", "test_image", "test_label", "future_test_logits"):
            path = sources / f"{name}.txt"
            path.write_text("INVENTED EVALUATOR INPUT\n")
            denied[name] = str(path)
        archive_path = sources / "original.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("invented_test.jpg", "INVENTED TEXT; NOT AN IMAGE")
        denied["original_zip_loader"] = str(archive_path)
        denied["original_zip_direct"] = str(archive_path)
        alias = bundle / "escaped_alias"
        alias.symlink_to(sources / "test_label.txt")
        denied["symlink_to_test"] = str(alias)
        protected = [sources]
        if protected_config is not None:
            # Actual-path probes concern only already-authorized HAM/evidence.
            # Never probe contents of ISIC/DDI or any held-out outcome file.
            paths = json.loads(protected_config.read_text())
            original_paths = [paths["metadata_path"], *paths["image_archives"].values()]
            protected.extend({Path(path).resolve(strict=True).parent for path in original_paths})
            repository = Path(__file__).resolve().parents[1]
            protected.extend([repository / "outputs", protected_config.resolve().parent])
            for i, path in enumerate(original_paths):
                denied[f"real_original_direct_{i}"] = str(Path(path).resolve(strict=True))
            for alias_name, path in paths["image_archives"].items():
                denied[f"real_original_zip_loader_{alias_name}"] = str(Path(path).resolve(strict=True))
            for name, path in {
                "real_unsplit_manifest": repository / "outputs/m2-ham-archive-inventory/manifest.csv",
                "real_full_decode_checkpoint": repository / "outputs/m2-image-integrity/decode_checkpoint.json",
                "real_candidate_queue": repository / "outputs/m2-candidate-review/adjudication_queue.json",
                "real_original_path_config": protected_config.resolve(),
            }.items():
                if not path.is_file():
                    raise RuntimeError(f"Required protected-route evidence is absent: {name}")
                denied[name] = str(path)
            source_alias = bundle / "real_source_alias"
            source_alias.symlink_to(original_paths[0])
            denied["real_source_symlink"] = str(source_alias)
        config = {"allowed": allowed, "denied": denied,
                  "forbidden_write": str(sources / "forbidden_write.txt"),
                  "allowed_output": str(outputs / "result.txt")}
        (bundle / "probe_config.json").write_text(json.dumps(config))
        (bundle / "probe.py").write_text(PROBE)
        profile = development_profile(bundle=bundle, output=outputs,
                                      runtime_roots=[Path(sys.prefix), Path(sys.base_prefix).parents[3],
                                                     Path(sys.base_prefix), Path(sys.base_prefix).resolve().parents[3]],
                                      interpreter=Path(sys.executable), protected_roots=protected)
        (root / "profile.sb").write_text(profile)
        result = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(root / "profile.sb"),
                                 str(Path(sys.executable).absolute()), "-B", "-I",
                                 str(bundle / "probe.py"), str(bundle / "probe_config.json")],
                                cwd=bundle, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Boundary failed ({result.returncode}): {result.stderr.strip()} {result.stdout.strip()}")
        checks = json.loads(result.stdout)
        if not checks or not all(checks.values()):
            raise RuntimeError("Boundary did not deny every invented protected route.")
        return {"status": "synthetic_os_boundary_passed", "checks": checks,
                "profile": profile, "production_seal_validated": False,
                "actual_ham_source_routes_checked": protected_config is not None,
                "limitation": "Machine owner and processes outside this profile retain access."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-config", type=Path)
    args = parser.parse_args()
    try:
        report = check(args.protected_config)
    except RuntimeError as exc:
        report = {"status": "synthetic_os_boundary_failed", "error": str(exc),
                  "production_seal_validated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(report.get("checks", {})),
                      "production_seal_validated": False}))
    raise SystemExit(0 if report["status"] == "synthetic_os_boundary_passed" else 1)
