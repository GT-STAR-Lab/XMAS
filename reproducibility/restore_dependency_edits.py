"""Check or restore captured library edits in the active Python environment.

Run with the target environment's Python before importing the affected packages.
Unknown versions or source hashes abort before any files are changed.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path(__file__).resolve().parent / "dependency_audit_20261117")
    parser.add_argument("--packages", nargs="+", default=["torchrl", "torch-geometric"],
                        choices=("torchrl", "torch-geometric", "statannot"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Restore edits after all files pass preflight checks")
    mode.add_argument("--check", action="store_true", help="Verify that all selected edits are present (default)")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "patch_manifest.json").read_text())
    selected = set(args.packages)
    operations, errors, existing = [], [], 0
    for package in manifest["packages"]:
        if package["name"] not in selected:
            continue
        try:
            installed = metadata.distribution(package["name"])
        except metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package['name']}=={package['version']}")
            continue
        if installed.version != package["version"]:
            errors.append(f"Wrong version: {package['name']} {installed.version}; expected {package['version']}")
            continue
        patch = bundle / package["patch"]
        if sha(patch) != package["patch_sha256"]:
            errors.append(f"Changed patch artifact: {patch}")
        for entry in package["files"]:
            relative = Path(entry["relative_path"])
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"Invalid package-relative path: {relative}")
                continue
            target = Path(installed.locate_file(relative))
            snapshot = bundle / entry["snapshot"]
            if not snapshot.is_file() or sha(snapshot) != entry["after_sha256"]:
                errors.append(f"Changed/missing source snapshot: {snapshot}")
                continue
            if not target.is_file():
                errors.append(f"Missing installed source: {target}")
                continue
            current = sha(target)
            if current == entry["after_sha256"]:
                existing += 1
            elif current == entry["before_sha256"]:
                operations.append((target, snapshot, entry["after_sha256"]))
            else:
                errors.append(f"Unrecognized installed source; refusing to replace: {target}")
    if errors:
        print(json.dumps(dict(status="preflight_failed", errors=errors, files_changed=0), indent=2))
        return 2
    if operations and not args.apply:
        print(json.dumps(dict(status="edits_required", already_patched_files=existing,
                              files_requiring_edits=len(operations), files_changed=0), indent=2))
        return 1
    for target, snapshot, expected in operations:
        permissions = target.stat().st_mode & 0o777
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name+".restore.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(snapshot.read_bytes())
        try:
            os.chmod(temporary, permissions)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        assert sha(target) == expected
    print(json.dumps(dict(status="verified", packages=sorted(selected),
        already_patched_files=existing, restored_files=len(operations),
        environment_prefix=sys.prefix), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
