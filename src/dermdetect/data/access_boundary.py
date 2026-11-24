"""macOS process-boundary prototype; production freeze still requires evidence.

Only an allowlisted development bundle/runtime/output tree is readable. This
does not restrict the machine owner or processes launched outside this profile.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import ContractError


def development_profile(*, bundle: Path, output: Path, runtime_roots: Iterable[Path],
                        interpreter: Path, protected_roots: Iterable[Path]) -> str:
    bundle, output, interpreter = (p.resolve(strict=True) for p in (bundle, output, interpreter))
    supplied_runtimes = tuple(p.absolute() for p in runtime_roots)
    runtimes = tuple(dict.fromkeys((*supplied_runtimes,
                                   *(p.resolve(strict=True) for p in supplied_runtimes))))
    protected = tuple(p.resolve(strict=True) for p in protected_roots)
    if not bundle.is_dir() or not output.is_dir() or not interpreter.is_file() or not runtimes or not protected:
        raise ContractError("invalid_boundary", "Existing bundle, output, runtime and protected roots are required.")
    allowed = (bundle, output, *runtimes, Path("/System/Library"), Path("/usr/lib"),
               Path("/usr/share"), Path("/Library/Apple"),
               Path("/System/Volumes/Preboot"), Path("/private/var/db/timezone"))
    immutable_roots = (allowed[0], *allowed[2:])
    if (any(output.is_relative_to(root) or root.is_relative_to(output) for root in immutable_roots)
            or interpreter.is_relative_to(output)):
        raise ContractError("unsafe_boundary", "Writable outputs must be disjoint from inspected code and runtime.")
    for root in allowed:
        if root == Path("/") or any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
            raise ContractError("unsafe_boundary", "Protected data cannot overlap any readable root.")
    def subpath(path: Path) -> str:
        return f"(subpath {json.dumps(str(path))})"
    read_rules = " ".join(subpath(p) for p in allowed)
    # OS-runtime paths follow Apple's installed system.sb/bsd.sb. A literal
    # root permits cwd lookup/listing, never recursive read access to the disk.
    read_rules += (' (literal "/") (literal "/dev/null") (literal "/dev/urandom")'
                   ' (literal "/dev/random") (subpath "/dev/fd")'
                   ' (literal "/private/etc/passwd")'
                   ' (literal "/private/etc/localtime")')
    executables = [interpreter]
    companion = interpreter.parent.parent / "Resources/Python.app/Contents/MacOS/Python"
    if companion.is_file():
        executables.append(companion.resolve(strict=True))
    exec_rules = ' '.join(f"(literal {json.dumps(str(p))})" for p in executables)
    return ("(version 1)\n(allow default)\n(deny network*)\n"
            f"(deny file-read-data (require-not (require-any {read_rules})))\n"
            f"(deny file-read-metadata (require-any {' '.join(subpath(p) for p in protected)}))\n"
            f"(deny file-write* (require-not {subpath(output)}))\n"
            f"(deny process-exec (require-not (require-any {exec_rules})))\n")
