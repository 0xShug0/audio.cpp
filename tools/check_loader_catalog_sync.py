#!/usr/bin/env python3
"""Check sync between runtime loaders, model_specs packages, and model_manager_v2.

Schema correctness is owned by the typed model-spec validator. This script only
checks cross-system drift:

- registered loader families vs model_specs families
- model_specs packages vs model_manager_v2 package output
See docs/maintainers/loader_and_catalog.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE_PATH = REPO_ROOT / "CMakeLists.txt"
REGISTRY_PATH = REPO_ROOT / "src" / "framework" / "runtime" / "registry.cpp"
SPECS_DIR = REPO_ROOT / "model_specs"

_LOADER_CALL_RE = re.compile(r"\bmake_([a-z0-9_]+)_loader(?:\s*\(\s*\))?")

BUNDLED_LOADERS_WITHOUT_SPEC = {
    "marblenet_vad",
    "silero_vad",
}


@dataclass(frozen=True)
class SpecPackage:
    family: str
    id: str
    format: str
    target_directory: str
    default: bool


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_loader_declarations(text: str, comment_prefix: str) -> tuple[set[str], set[str]]:
    active: set[str] = set()
    commented: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _LOADER_CALL_RE.search(line)
        if not match:
            continue
        family = match.group(1)
        if line.startswith(comment_prefix):
            commented.add(family)
        else:
            active.add(family)
    return active, commented


def parse_declared_loaders(cmake_text: str, registry_text: str) -> tuple[set[str], set[str]]:
    cmake_active, cmake_commented = parse_loader_declarations(cmake_text, "#")
    registry_active, registry_commented = parse_loader_declarations(registry_text, "//")
    return cmake_active | registry_active, cmake_commented | registry_commented


def loader_families_from_json(payload: Any) -> set[str]:
    if not isinstance(payload, list):
        raise ValueError("loader JSON must be a list")
    families: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not isinstance(row.get("family"), str):
            raise ValueError(f"loader JSON row {index} must contain string family")
        families.add(row["family"])
    return families


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_spec_packages(specs_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, SpecPackage], list[str]]:
    specs_by_family: dict[str, dict[str, Any]] = {}
    packages_by_id: dict[str, SpecPackage] = {}
    errors: list[str] = []
    for path in sorted(specs_dir.glob("*.json")):
        try:
            spec = load_json(path)
        except json.JSONDecodeError as exc:
            errors.append(f"{rel(path)}: invalid JSON: {exc}")
            continue
        if not isinstance(spec, dict):
            errors.append(f"{rel(path)}: top-level JSON must be an object")
            continue
        family = spec.get("family")
        if not isinstance(family, str) or not family:
            errors.append(f"{rel(path)}: missing family")
            continue
        if family != path.stem:
            errors.append(f"{rel(path)}: family '{family}' does not match filename stem '{path.stem}'")
        if family in specs_by_family:
            errors.append(f"{rel(path)}: duplicate spec family '{family}'")
        specs_by_family[family] = spec

        packages = spec.get("packages", [])
        if packages is None:
            packages = []
        if not isinstance(packages, list):
            errors.append(f"{rel(path)}: packages must be a list for model_manager_v2")
            continue
        for index, package in enumerate(packages):
            if not isinstance(package, dict):
                errors.append(f"{rel(path)}: packages[{index}] must be an object for model_manager_v2")
                continue
            package_id = package.get("id")
            if not isinstance(package_id, str) or not package_id:
                errors.append(f"{rel(path)}: packages[{index}] missing id")
                continue
            if package_id in packages_by_id:
                errors.append(
                    f"{rel(path)}: duplicate package id '{package_id}' already declared by "
                    f"model_specs/{packages_by_id[package_id].family}.json"
                )
                continue
            packages_by_id[package_id] = SpecPackage(
                family=family,
                id=package_id,
                format=str(package.get("format") or ""),
                target_directory=str(package.get("target_directory") or ""),
                default=package.get("default") is True,
            )
    return specs_by_family, packages_by_id, errors


def load_manager_packages(specs_dir: Path) -> tuple[dict[str, Any], list[str]]:
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import model_manager_v2  # noqa: E402

    errors: list[str] = []
    try:
        records = model_manager_v2.flatten_packages(model_manager_v2.load_specs(specs_dir))
    except Exception as exc:
        return {}, [f"model_manager_v2 failed to load {rel(specs_dir)}: {exc}"]
    packages: dict[str, Any] = {}
    for record in records:
        if record.id in packages:
            errors.append(f"model_manager_v2 produced duplicate package id '{record.id}'")
        packages[record.id] = record
    return packages, errors


def check_loader_spec_sync(active_loaders: set[str], specs_by_family: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    spec_families = set(specs_by_family)
    for family in sorted(spec_families - active_loaders):
        errors.append(f"model_specs/{family}.json has no registered loader family")
    for family in sorted(active_loaders - spec_families - BUNDLED_LOADERS_WITHOUT_SPEC):
        errors.append(f"registered loader '{family}' has no model_specs/{family}.json")
    return errors


def check_manager_sync(spec_packages: dict[str, SpecPackage], manager_packages: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    spec_ids = set(spec_packages)
    manager_ids = set(manager_packages)
    for package_id in sorted(spec_ids - manager_ids):
        errors.append(f"model_specs package '{package_id}' is missing from model_manager_v2 output")
    for package_id in sorted(manager_ids - spec_ids):
        errors.append(f"model_manager_v2 package '{package_id}' is not declared in model_specs")

    defaults_by_family: dict[str, list[str]] = {}
    for package in spec_packages.values():
        if package.default:
            defaults_by_family.setdefault(package.family, []).append(package.id)
        if package.format != "gguf":
            warnings.append(f"model_specs/{package.family}.json package '{package.id}' is format={package.format}")

    for family in sorted({package.family for package in spec_packages.values()}):
        defaults = defaults_by_family.get(family, [])
        if len(defaults) != 1:
            errors.append(f"model_specs/{family}.json must expose exactly one default package for v2 family installs")
            continue
        default_id = defaults[0]
        if spec_packages[default_id].format != "gguf":
            errors.append(f"model_specs/{family}.json default package '{default_id}' must be GGUF")

    for package_id, spec_package in spec_packages.items():
        manager_package = manager_packages.get(package_id)
        if manager_package is None:
            continue
        if manager_package.family != spec_package.family:
            errors.append(
                f"model_manager_v2 package '{package_id}' family '{manager_package.family}' "
                f"does not match model_specs family '{spec_package.family}'"
            )
        if manager_package.target_directory != spec_package.target_directory:
            errors.append(
                f"model_manager_v2 package '{package_id}' target_directory '{manager_package.target_directory}' "
                f"does not match model_specs target_directory '{spec_package.target_directory}'"
            )
    return errors, warnings


class _SyncCheckSelfTests(unittest.TestCase):
    def test_parse_loader_declarations(self) -> None:
        text = """
        # engine::models::old::make_old_loader
        engine::models::new_family::make_new_family_loader
        engine::models::other::make_other_loader # active trailing comment
        """
        active, commented = parse_loader_declarations(text, "#")
        self.assertEqual(active, {"new_family", "other"})
        self.assertEqual(commented, {"old"})

    def test_loader_json_family_parse(self) -> None:
        families = loader_families_from_json([{"family": "a"}, {"family": "b"}])
        self.assertEqual(families, {"a", "b"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", type=Path, default=CMAKE_PATH, help="Path to top-level CMakeLists.txt")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="Path to registry.cpp")
    parser.add_argument("--specs-dir", type=Path, default=SPECS_DIR, help="Directory containing model spec JSON files")
    parser.add_argument(
        "--loader-json",
        type=Path,
        default=None,
        help="Optional audiocpp_cli --list-loaders --json output. Use '-' to read stdin.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit tests and exit")
    args = parser.parse_args()

    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SyncCheckSelfTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    errors: list[str] = []
    warnings: list[str] = []
    if args.loader_json is not None:
        try:
            loader_text = sys.stdin.read() if str(args.loader_json) == "-" else args.loader_json.read_text(encoding="utf-8")
            active_loaders = loader_families_from_json(json.loads(loader_text))
            commented_loaders: set[str] = set()
        except Exception as exc:
            print(f"error: failed to read loader JSON: {exc}", file=sys.stderr)
            return 2
    else:
        if not args.cmake.is_file():
            print(f"error: CMakeLists.txt not found: {args.cmake}", file=sys.stderr)
            return 2
        if not args.registry.is_file():
            print(f"error: registry not found: {args.registry}", file=sys.stderr)
            return 2
        active_loaders, commented_loaders = parse_declared_loaders(
            args.cmake.read_text(encoding="utf-8"),
            args.registry.read_text(encoding="utf-8"),
        )
    if not active_loaders:
        print("error: no active loaders found", file=sys.stderr)
        return 2

    specs_by_family, spec_packages, spec_errors = load_spec_packages(args.specs_dir)
    manager_packages, manager_errors = load_manager_packages(args.specs_dir)
    errors.extend(spec_errors)
    errors.extend(manager_errors)
    errors.extend(check_loader_spec_sync(active_loaders, specs_by_family))
    manager_sync_errors, manager_sync_warnings = check_manager_sync(spec_packages, manager_packages)
    errors.extend(manager_sync_errors)
    warnings.extend(manager_sync_warnings)

    print(
        f"active_loaders={len(active_loaders)} commented_loaders={len(commented_loaders)} "
        f"specs={len(specs_by_family)} packages={len(spec_packages)} "
        f"manager_packages={len(manager_packages)}"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("loader/spec sync failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: keep model_specs/*.json, model_manager_v2.py, registered loaders, "
            "and published default GGUF packages aligned. Schema-level validation "
            "belongs to the typed model-spec validator, and WebUI placement is checked separately.",
            file=sys.stderr,
        )
        return 1

    print("ok: runtime loaders, model_specs, and model_manager_v2 are in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
