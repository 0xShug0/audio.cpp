#!/usr/bin/env python3
"""Fail when installable catalog packages reference unregistered loaders.

The machine-readable contract from PR #74 makes package ``family`` and
``audiocpp_cli --list-loaders`` authoritative for integrators. Catalog entries
must not advertise installable packages for families that are commented out or
missing from ``src/framework/runtime/registry.cpp``.

See docs/maintainers/loader_and_catalog.md.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "src" / "framework" / "runtime" / "registry.cpp"
MODEL_MANAGER_PATH = REPO_ROOT / "tools" / "model_manager.py"

_LOADER_CALL_RE = re.compile(r"\bmake_([a-z0-9_]+)_loader\s*\(\s*\)")


def parse_registry_loaders(registry_text: str) -> tuple[set[str], set[str]]:
    """Return (active_families, commented_families) from registry.cpp."""
    active: set[str] = set()
    commented: set[str] = set()
    for raw_line in registry_text.splitlines():
        line = raw_line.strip()
        match = _LOADER_CALL_RE.search(line)
        if not match:
            continue
        family = match.group(1)
        if line.startswith("//"):
            commented.add(family)
        else:
            active.add(family)
    return active, commented


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Path to registry.cpp",
    )
    args = parser.parse_args()

    if not args.registry.is_file():
        print(f"error: registry not found: {args.registry}", file=sys.stderr)
        return 2
    if not MODEL_MANAGER_PATH.is_file():
        print(f"error: model manager not found: {MODEL_MANAGER_PATH}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(MODEL_MANAGER_PATH.parent))
    import model_manager as mm  # noqa: E402

    active, commented = parse_registry_loaders(args.registry.read_text(encoding="utf-8"))
    if not active:
        print("error: no active loaders parsed from registry.cpp", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []

    for package in mm.CATALOG:
        payload = mm.package_payload(package)
        package_id = str(payload.get("id") or "")
        family = str(payload.get("family") or "").strip()
        installable = bool(payload.get("installable"))
        standalone = bool(payload.get("standalone", True))
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_kind = str(source.get("kind") or "")

        if not family:
            errors.append(f"{package_id}: missing family (set ModelPackage.family or fix id)")
            continue

        if not installable or source_kind == "unsupported":
            # Parked / unavailable packages may keep a family for documentation.
            if family in active:
                warnings.append(
                    f"{package_id}: UnsupportedSource but family '{family}' is already "
                    "registered — restore a real SnapshotSource/Composite/Converter"
                )
            continue

        if not standalone:
            # Dependency / subcomponent packages do not need their own loader.
            continue

        if family not in active:
            hint = ""
            if family in commented:
                hint = " (commented out in registry.cpp)"
            elif family == "higgs_audio_tts" and "higgs_tts" in commented:
                hint = " (registry stub uses higgs_tts; keep family ids consistent)"
            errors.append(
                f"{package_id}: installable standalone package family '{family}' "
                f"is not registered in registry.cpp{hint}"
            )

    print(f"active_loaders={len(active)} catalog_packages={len(mm.CATALOG)}")
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("loader/catalog sync failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: either register the loader in registry.cpp + model_specs/, "
            "or mark the package UnsupportedSource. See "
            "docs/maintainers/loader_and_catalog.md",
            file=sys.stderr,
        )
        return 1

    print("ok: installable catalog families match registered loaders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
