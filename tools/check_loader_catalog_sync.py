#!/usr/bin/env python3
"""Fail when installable catalog packages reference unregistered loaders.

``model_manager list --json`` family fields and ``audiocpp_cli --list-loaders``
must stay aligned. Installable standalone packages cannot advertise a family
that is missing or commented out in ``src/framework/runtime/registry.cpp``.

See docs/maintainers/loader_and_catalog.md.
"""

from __future__ import annotations

import argparse
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "src" / "framework" / "runtime" / "registry.cpp"
MODEL_MANAGER_PATH = REPO_ROOT / "tools" / "model_manager.py"
README_PATH = REPO_ROOT / "README.md"

_LOADER_CALL_RE = re.compile(r"\bmake_([a-z0-9_]+)_loader\s*\(\s*\)")

# Optional: catalog family strings that map to a differently named parked stub.
PARKED_FAMILY_ALIASES: dict[str, set[str]] = {
    "higgs_tts": {"higgs_audio_tts"},
}

# Registered loaders that intentionally have no installable ModelPackage.
BUNDLED_LOADERS_WITHOUT_PACKAGE: set[str] = {
    "silero_vad",
}


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


def family_is_registered(family: str, active: set[str]) -> bool:
    return family in active


def family_is_parked(family: str, commented: set[str]) -> bool:
    if family in commented:
        return True
    for stub, aliases in PARKED_FAMILY_ALIASES.items():
        if stub in commented and family in aliases:
            return True
    return False


def parked_hint(family: str, commented: set[str]) -> str:
    if family in commented:
        return " (commented out in registry.cpp)"
    for stub, aliases in PARKED_FAMILY_ALIASES.items():
        if stub in commented and family in aliases:
            return (
                f" (parked registry stub is '{stub}'; catalog family '{family}' "
                "must stay UnsupportedSource until one id is registered)"
            )
    return ""


def parse_readme_package_table(readme_text: str) -> dict[str, str]:
    """Map package id -> status cell from the recommended install table."""
    match = re.search(
        r"\| Package id \| Model \| HF ready-to-use repo \|\n\|[^\n]+\n((?:\|.*\n)+)",
        readme_text,
    )
    if not match:
        return {}
    rows: dict[str, str] = {}
    for line in match.group(1).splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 3 or not cols[0].startswith("`"):
            continue
        package_id = cols[0].strip("`")
        rows[package_id] = cols[2]
    return rows


def check_catalog(
    *,
    active: set[str],
    commented: set[str],
    packages: list,
    package_payload,
    default_family_from_package_id,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    installable_families: set[str] = set()

    for package in packages:
        payload = package_payload(package)
        package_id = str(payload.get("id") or "")
        family = str(payload.get("family") or "").strip()
        installable = bool(payload.get("installable"))
        standalone = bool(payload.get("standalone", True))
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        source_kind = str(source.get("kind") or "")
        inferred = default_family_from_package_id(package_id)
        explicit = getattr(package, "family", None)

        if not family:
            errors.append(
                f"{package_id}: missing family (set ModelPackage.family or fix id)"
            )
            continue

        if not installable or source_kind == "unsupported":
            if family_is_registered(family, active):
                warnings.append(
                    f"{package_id}: UnsupportedSource but family '{family}' is already "
                    "registered — restore a real SnapshotSource/Composite/Converter"
                )
            elif not family_is_parked(family, commented):
                warnings.append(
                    f"{package_id}: UnsupportedSource family '{family}' is neither "
                    "registered nor a parked registry stub/alias"
                )
            continue

        if not standalone:
            continue

        installable_families.add(family)

        if explicit is None and inferred != family:
            errors.append(
                f"{package_id}: resolved family '{family}' != id inference "
                f"'{inferred}' without ModelPackage.family set"
            )

        if explicit is None and inferred not in active and family in active:
            errors.append(
                f"{package_id}: set ModelPackage.family explicitly "
                f"(id inference '{inferred}' is not a registered loader)"
            )

        if not family_is_registered(family, active):
            errors.append(
                f"{package_id}: installable standalone package family '{family}' "
                f"is not registered in registry.cpp{parked_hint(family, commented)}"
            )
        elif family_is_parked(family, commented):
            errors.append(
                f"{package_id}: family '{family}' is both active and commented in registry.cpp"
            )

    for stub in sorted(commented):
        aliases = {stub} | PARKED_FAMILY_ALIASES.get(stub, set())
        leaked = sorted(fam for fam in aliases if fam in installable_families)
        if leaked:
            errors.append(
                f"parked loader '{stub}' still has installable catalog families: "
                + ", ".join(leaked)
            )

    for family in sorted(active):
        if family in BUNDLED_LOADERS_WITHOUT_PACKAGE:
            continue
        if family not in installable_families:
            warnings.append(
                f"registered loader '{family}' has no installable standalone "
                "ModelPackage (add a package or list it in "
                "BUNDLED_LOADERS_WITHOUT_PACKAGE if intentional)"
            )

    return errors, warnings


def check_readme(
    *,
    readme_text: str,
    packages: list,
    package_payload,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    table = parse_readme_package_table(readme_text)
    if not table:
        errors.append("README.md: could not parse recommended package table")
        return errors, warnings

    catalog_by_id = {}
    for package in packages:
        payload = package_payload(package)
        catalog_by_id[str(payload["id"])] = payload

    for package_id, status in sorted(table.items()):
        payload = catalog_by_id.get(package_id)
        if payload is None:
            errors.append(f"README.md: package `{package_id}` not in model_manager CATALOG")
            continue
        installable = bool(payload.get("installable"))
        unavailable = "unavailable" in status.lower()
        if unavailable and installable:
            errors.append(
                f"README.md: `{package_id}` marked Unavailable but catalog is installable"
            )
        if not unavailable and not installable:
            errors.append(
                f"README.md: `{package_id}` looks installable in the table but catalog "
                "is UnsupportedSource — mark Unavailable or restore a source"
            )

    for package_id, payload in sorted(catalog_by_id.items()):
        if not payload.get("installable") or not payload.get("standalone", True):
            continue
        if package_id not in table:
            errors.append(
                f"README.md: installable standalone package `{package_id}` missing "
                "from recommended package table"
            )

    return errors, warnings


class _SyncCheckSelfTests(unittest.TestCase):
    def test_parse_active_and_commented(self) -> None:
        text = """
        // make_family_a_loader(),
        make_family_b_loader(),
        make_family_c_loader(), // trailing comment still active
        """
        active, commented = parse_registry_loaders(text)
        self.assertEqual(active, {"family_b", "family_c"})
        self.assertEqual(commented, {"family_a"})

    def test_parked_alias_blocks_installable(self) -> None:
        class Pkg:
            def __init__(self, family=None):
                self.family = family

        stub = next(iter(PARKED_FAMILY_ALIASES))
        alias = next(iter(PARKED_FAMILY_ALIASES[stub]))

        def payload(package):
            return {
                "id": "pkg_alias",
                "family": alias,
                "installable": True,
                "standalone": True,
                "source": {"kind": "huggingface_snapshot"},
            }

        errors, _ = check_catalog(
            active={"family_b"},
            commented={stub},
            packages=[Pkg(family=alias)],
            package_payload=payload,
            default_family_from_package_id=lambda _pid: "pkg_alias",
        )
        self.assertTrue(any(alias in e for e in errors))
        self.assertTrue(any(f"parked loader '{stub}'" in e for e in errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_PATH,
        help="Path to registry.cpp",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Skip README package-table cross-check",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in unit tests and exit",
    )
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SyncCheckSelfTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

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

    errors, warnings = check_catalog(
        active=active,
        commented=commented,
        packages=list(mm.CATALOG),
        package_payload=mm.package_payload,
        default_family_from_package_id=mm._default_family_from_package_id,
    )

    if not args.skip_readme:
        if not README_PATH.is_file():
            errors.append(f"README.md not found: {README_PATH}")
        else:
            readme_errors, readme_warnings = check_readme(
                readme_text=README_PATH.read_text(encoding="utf-8"),
                packages=list(mm.CATALOG),
                package_payload=mm.package_payload,
            )
            errors.extend(readme_errors)
            warnings.extend(readme_warnings)

    print(
        f"active_loaders={len(active)} commented_loaders={len(commented)} "
        f"catalog_packages={len(mm.CATALOG)}"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("loader/catalog sync failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: register the loader in registry.cpp (with sources in-tree), "
            "or mark the package UnsupportedSource / update README. See "
            "docs/maintainers/loader_and_catalog.md",
            file=sys.stderr,
        )
        return 1

    print("ok: installable catalog families match registered loaders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
