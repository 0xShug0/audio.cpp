#!/usr/bin/env python3
"""Check sync between runtime loaders, model_specs packages, model_manager_v2,
and the WebUI catalog.

Schema correctness is owned by the typed model-spec validator. This script only
checks cross-system drift:

- registered loader families vs model_specs families
- model_specs packages vs model_manager_v2 package output
- model_specs packages vs the packages the native package manager publishes
- webui/configs/models_catalog.json vs model_specs (family, download id,
  install path, task/mode vocabulary) and vs the packages a catalog entry can
  actually reach
- webui/configs/model_params.json parameter groups vs the request options the
  owning spec documents
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
CATALOG_PATH = REPO_ROOT / "webui" / "configs" / "models_catalog.json"
MODEL_PARAMS_PATH = REPO_ROOT / "webui" / "configs" / "model_params.json"
SESSION_PATH = REPO_ROOT / "src" / "framework" / "runtime" / "session.cpp"

_LOADER_CALL_RE = re.compile(r"\bmake_([a-z0-9_]+)_loader(?:\s*\(\s*\))?")
_VALUE_COMPARE_RE = re.compile(r'value\s*==\s*"([a-z0-9_]+)"')

BUNDLED_LOADERS_WITHOUT_SPEC = {
    "marblenet_vad",
    "silero_vad",
}

# Bundled loaders ship inside the repository instead of a downloadable package,
# so their catalog entries carry no download id and point outside models/.
BUNDLED_CATALOG_PATH_PREFIX = "assets/framework/models/"

# src/framework/package_manager/manager.cpp drops every package whose download
# kind is not a Hugging Face snapshot, so those packages are invisible to the
# server and to the WebUI install flow no matter what model_manager_v2 lists.
INSTALLABLE_DOWNLOAD_KIND = "huggingface_snapshot"


@dataclass(frozen=True)
class SpecPackage:
    family: str
    id: str
    format: str
    target_directory: str
    default: bool
    download_kind: str = ""
    files: tuple[str, ...] = ()
    strip_prefix: str = ""

    @property
    def installable(self) -> bool:
        return self.download_kind == INSTALLABLE_DOWNLOAD_KIND

    @property
    def install_paths(self) -> set[str]:
        """Catalog paths that resolve to this package once it is installed.

        Either the install directory itself or one of the files the package
        drops into it. Both spellings appear in models_catalog.json and both
        are accepted by the loaders.
        """
        root = normalize_catalog_path(f"models/{self.target_directory}")
        paths = {root}
        prefix = normalize_catalog_path(self.strip_prefix)
        for remote in self.files:
            relative = normalize_catalog_path(remote)
            if prefix and relative.startswith(f"{prefix}/"):
                relative = relative[len(prefix) + 1:]
            if relative:
                paths.add(normalize_catalog_path(f"{root}/{relative}"))
        return paths


@dataclass(frozen=True)
class CatalogEntry:
    index: int
    id: str
    family: str
    path: str
    task: str
    mode: str
    download_id: str


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_catalog_path(value: str) -> str:
    text = value.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = re.sub(r"/+", "/", text)
    return text.rstrip("/")


def parse_string_vocabulary(text: str, signature: str) -> set[str]:
    """Collect the `value == "..."` literals a session.cpp parser accepts."""
    start = text.find(signature)
    if start < 0:
        raise ValueError(f"'{signature}' not found")
    end = text.find("\n}\n", start)
    body = text[start:] if end < 0 else text[start:end]
    values = set(_VALUE_COMPARE_RE.findall(body))
    if not values:
        raise ValueError(f"'{signature}' declares no accepted values")
    return values


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

        spec_defaults = spec.get("package_defaults")
        default_download = {}
        if isinstance(spec_defaults, dict) and isinstance(spec_defaults.get("download"), dict):
            default_download = spec_defaults["download"]

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
            download = package.get("download")
            if not isinstance(download, dict):
                download = {}
            files = package.get("files")
            if not isinstance(files, list):
                files = []
            packages_by_id[package_id] = SpecPackage(
                family=family,
                id=package_id,
                format=str(package.get("format") or ""),
                target_directory=str(package.get("target_directory") or ""),
                default=package.get("default") is True,
                download_kind=str(download.get("kind") or default_download.get("kind") or ""),
                files=tuple(str(item) for item in files if isinstance(item, str)),
                strip_prefix=str(package.get("strip_prefix") or ""),
            )
    return specs_by_family, packages_by_id, errors


def recommended_package_id(spec: dict[str, Any]) -> str:
    ui = spec.get("ui")
    if not isinstance(ui, dict):
        return ""
    value = ui.get("recommended_package")
    return value if isinstance(value, str) else ""


def load_catalog(path: Path) -> tuple[list[CatalogEntry], list[str]]:
    errors: list[str] = []
    try:
        payload = load_json(path)
    except json.JSONDecodeError as exc:
        return [], [f"{rel(path)}: invalid JSON: {exc}"]
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        return [], [f"{rel(path)}: top-level JSON must be an object with a 'models' list"]

    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(payload["models"]):
        if not isinstance(row, dict):
            errors.append(f"{rel(path)}: models[{index}] must be an object")
            continue
        entry_id = row.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"{rel(path)}: models[{index}] missing id")
            continue
        if entry_id in seen_ids:
            errors.append(f"{rel(path)}: duplicate catalog entry id '{entry_id}'")
            continue
        seen_ids.add(entry_id)
        family = row.get("family")
        if not isinstance(family, str) or not family:
            errors.append(f"{rel(path)}: catalog entry '{entry_id}' missing family")
            continue
        entries.append(CatalogEntry(
            index=index,
            id=entry_id,
            family=family,
            path=str(row.get("path") or ""),
            task=str(row.get("task") or ""),
            mode=str(row.get("mode") or ""),
            download_id=str(row.get("download_id") or ""),
        ))
    return entries, errors


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


def check_native_manager_sync(spec_packages: dict[str, SpecPackage]) -> list[str]:
    """Report packages model_manager_v2 lists but the native manager drops.

    tools/model_manager_v2.py flattens every declared package; the server keeps
    only Hugging Face snapshots (src/framework/package_manager/manager.cpp), so
    the two package counts are not interchangeable.
    """
    warnings: list[str] = []
    for package in sorted(spec_packages.values(), key=lambda item: item.id):
        if package.installable:
            continue
        warnings.append(
            f"model_specs/{package.family}.json package '{package.id}' has download.kind="
            f"'{package.download_kind or 'missing'}' and is not published by the native "
            f"package manager (server-side package list excludes it)"
        )
    return warnings


def check_catalog_sync(
    catalog_entries: list[CatalogEntry],
    specs_by_family: dict[str, dict[str, Any]],
    spec_packages: dict[str, SpecPackage],
    task_kinds: set[str],
    run_modes: set[str],
    catalog_path: Path,
) -> tuple[list[str], list[str]]:
    """Check webui/configs/models_catalog.json against model_specs.

    A catalog entry resolves to exactly one package through its download id.
    The WebUI then offers every GGUF package that installs into the resolved
    package's target directory (webui/native/src/lib/catalog.ts), so a target
    directory no entry resolves into is unreachable from the UI.
    """
    errors: list[str] = []
    warnings: list[str] = []
    name = rel(catalog_path)

    packages_by_family: dict[str, list[SpecPackage]] = {}
    for package in spec_packages.values():
        packages_by_family.setdefault(package.family, []).append(package)

    reachable_ids: set[str] = set()
    families_in_catalog: set[str] = set()

    for entry in catalog_entries:
        if entry.task and entry.task not in task_kinds:
            errors.append(
                f"{name}: catalog entry '{entry.id}' task '{entry.task}' is not accepted by "
                f"parse_voice_task_kind (expected one of {', '.join(sorted(task_kinds))})"
            )
        if entry.mode and entry.mode not in run_modes:
            errors.append(
                f"{name}: catalog entry '{entry.id}' mode '{entry.mode}' is not accepted by "
                f"parse_run_mode (expected one of {', '.join(sorted(run_modes))})"
            )

        if entry.family in BUNDLED_LOADERS_WITHOUT_SPEC:
            if entry.download_id:
                errors.append(
                    f"{name}: catalog entry '{entry.id}' is a bundled loader but names "
                    f"download_id '{entry.download_id}'"
                )
            if not normalize_catalog_path(entry.path).startswith(BUNDLED_CATALOG_PATH_PREFIX):
                errors.append(
                    f"{name}: catalog entry '{entry.id}' is bundled and must point inside "
                    f"{BUNDLED_CATALOG_PATH_PREFIX} (path '{entry.path}')"
                )
            continue

        if entry.family not in specs_by_family:
            errors.append(
                f"{name}: catalog entry '{entry.id}' names family '{entry.family}' "
                f"which has no model_specs/{entry.family}.json"
            )
            continue
        families_in_catalog.add(entry.family)

        family_packages = packages_by_family.get(entry.family, [])
        recommended_id = recommended_package_id(specs_by_family[entry.family])
        resolved = spec_packages.get(entry.download_id) if entry.download_id else None
        degraded = False
        if not entry.download_id:
            warnings.append(
                f"{name}: catalog entry '{entry.id}' has no download_id; its install location "
                f"cannot be checked against model_specs/{entry.family}.json"
            )
        elif resolved is None:
            errors.append(
                f"{name}: catalog entry '{entry.id}' download_id '{entry.download_id}' is not a "
                f"packages[].id in model_specs/{entry.family}.json"
            )
        elif resolved.family != entry.family:
            errors.append(
                f"{name}: catalog entry '{entry.id}' download_id '{entry.download_id}' belongs to "
                f"model_specs/{resolved.family}.json, not to family '{entry.family}'"
            )
            resolved = None
        if resolved is None and entry.download_id:
            # Fall back to the family recommendation so a broken download id
            # does not also suppress the path and reachability checks. The
            # download id error above is the one to fix first.
            fallback = spec_packages.get(recommended_id)
            if fallback is not None and fallback.family == entry.family:
                resolved = fallback
                degraded = True

        if resolved is None:
            continue

        if entry.path:
            candidate = normalize_catalog_path(entry.path)
            accepted = resolved.install_paths
            source = (
                f"model_specs/{entry.family}.json recommended package '{resolved.id}'"
                if degraded else f"package '{resolved.id}'"
            )
            if candidate not in accepted:
                lowered = {value.lower() for value in accepted}
                if candidate.lower() in lowered:
                    warnings.append(
                        f"{name}: catalog entry '{entry.id}' path '{entry.path}' differs in case "
                        f"from the install location of {source}"
                    )
                else:
                    errors.append(
                        f"{name}: catalog entry '{entry.id}' path '{entry.path}' is neither the "
                        f"target_directory nor an installed file of {source} "
                        f"(expected models/{resolved.target_directory}[/<installed file>])"
                    )

        exposed = [
            package for package in family_packages
            if package.target_directory == resolved.target_directory
            and package.format == "gguf" and package.installable
        ]
        if not exposed:
            warnings.append(
                f"{name}: catalog entry '{entry.id}' resolves to package '{resolved.id}' "
                f"(format={resolved.format or 'unknown'}), which the native model manager "
                f"cannot install; the entry has no install choice"
            )
        reachable_ids.update(package.id for package in exposed)

        if (not degraded and recommended_id and resolved.id != recommended_id
                and not resolved.default):
            recommended = spec_packages.get(recommended_id)
            if recommended is not None and recommended.target_directory == resolved.target_directory:
                warnings.append(
                    f"{name}: catalog entry '{entry.id}' names package '{resolved.id}' while "
                    f"model_specs/{entry.family}.json recommends '{recommended_id}' from the "
                    f"same install location"
                )

    for family in sorted(specs_by_family):
        family_packages = packages_by_family.get(family, [])
        installable = [
            package for package in family_packages
            if package.format == "gguf" and package.installable
        ]
        if not installable:
            # Families whose packages are all non-distributable (download.kind
            # is not a Hugging Face snapshot) are deliberately absent from the
            # UI, so their absence is not reported.
            continue
        if family not in families_in_catalog:
            warnings.append(
                f"model_specs/{family}.json publishes {len(installable)} installable GGUF "
                f"package(s) but no {name} entry exposes the family"
            )
            continue

        directories: dict[str, list[str]] = {}
        for package in installable:
            directories.setdefault(package.target_directory, []).append(package.id)
        for directory in sorted(directories):
            if any(package_id in reachable_ids for package_id in sorted(directories[directory])):
                continue
            warnings.append(
                f"model_specs/{family}.json installs {', '.join(sorted(directories[directory]))} "
                f"into '{directory}', which no {name} entry reaches"
            )

        recommended_id = recommended_package_id(specs_by_family[family])
        recommended = spec_packages.get(recommended_id) if recommended_id else None
        if recommended_id and (recommended is None or recommended.family != family):
            errors.append(
                f"model_specs/{family}.json ui.recommended_package '{recommended_id}' is not a "
                f"packages[].id in that spec"
            )
        elif recommended is not None and not (recommended.format == "gguf" and recommended.installable):
            errors.append(
                f"model_specs/{family}.json ui.recommended_package '{recommended_id}' is "
                f"format={recommended.format or 'unknown'} download.kind="
                f"'{recommended.download_kind or 'missing'}' and can never be offered by the UI"
            )
        elif recommended_id and recommended_id not in reachable_ids:
            errors.append(
                f"model_specs/{family}.json ui.recommended_package '{recommended_id}' is not "
                f"reachable from any {name} entry"
            )
        for package in sorted(installable, key=lambda item: item.id):
            if package.default and package.id not in reachable_ids:
                errors.append(
                    f"model_specs/{family}.json default package '{package.id}' is not reachable "
                    f"from any {name} entry"
                )
    return errors, warnings


def check_model_params_sync(
    params: Any,
    specs_by_family: dict[str, dict[str, Any]],
    catalog_entries: list[CatalogEntry],
    params_path: Path,
) -> tuple[list[str], list[str]]:
    """Check webui/configs/model_params.json controls against the specs.

    Parameter groups are keyed by spec family or by catalog entry id. A control
    whose name is absent from the owning spec's options.request is dead: strict
    loaders reject the unknown key and the request fails, lenient ones drop it
    silently. Only families that actually document their request options are
    checked, because an empty options.request means "undocumented", not
    "unsupported".
    """
    errors: list[str] = []
    warnings: list[str] = []
    name = rel(params_path)
    if not isinstance(params, dict):
        return [f"{name}: top-level JSON must be an object"], warnings

    family_by_entry_id = {entry.id: entry.family for entry in catalog_entries}
    for group, controls in params.items():
        if not isinstance(controls, list):
            continue
        family = group if group in specs_by_family else family_by_entry_id.get(group, "")
        if not family and group in BUNDLED_LOADERS_WITHOUT_SPEC:
            # Bundled families own no spec, so they are absent from
            # specs_by_family, and their catalog ids are hyphenated while the
            # group is keyed by the family name. Resolve them by name.
            family = group
        if not family:
            errors.append(
                f"{name}: parameter group '{group}' matches no model_specs family and no "
                f"catalog entry id"
            )
            continue
        if family in BUNDLED_LOADERS_WITHOUT_SPEC:
            # Bundled loaders have no spec by design, so there is no declared
            # request surface to check a control against. They also validate
            # nothing at runtime, so an unknown key is dropped rather than
            # rejected: the controls are safe, just unverifiable from here.
            continue
        spec = specs_by_family.get(family)
        if spec is None:
            errors.append(
                f"{name}: parameter group '{group}' resolves to family '{family}' "
                f"which has no model_specs/{family}.json"
            )
            continue
        options = spec.get("options")
        request = options.get("request") if isinstance(options, dict) else None
        if not isinstance(request, list) or not request:
            continue
        declared = {
            option.get("name") for option in request
            if isinstance(option, dict) and isinstance(option.get("name"), str)
        }
        for control in controls:
            if not isinstance(control, dict):
                continue
            control_name = control.get("name")
            if not isinstance(control_name, str) or not control_name:
                errors.append(f"{name}: parameter group '{group}' has a control without a name")
                continue
            if control_name not in declared:
                warnings.append(
                    f"{name}: parameter group '{group}' control '{control_name}' is not in "
                    f"model_specs/{family}.json options.request"
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

    def test_parse_string_vocabulary(self) -> None:
        text = (
            'VoiceTaskKind parse_voice_task_kind(const std::string & value) {\n'
            '    if (value == "vad") {\n        return VoiceTaskKind::Vad;\n    }\n'
            '    if (value == "tts") {\n        return VoiceTaskKind::Tts;\n    }\n'
            '    throw std::runtime_error("unsupported task: " + value);\n'
            '}\n'
            'RunMode parse_run_mode(const std::string & value) {\n'
            '    if (value == "offline") {\n        return RunMode::Offline;\n    }\n'
            '}\n'
        )
        self.assertEqual(
            parse_string_vocabulary(text, "parse_voice_task_kind(const std::string & value)"),
            {"vad", "tts"},
        )
        self.assertEqual(
            parse_string_vocabulary(text, "parse_run_mode(const std::string & value)"),
            {"offline"},
        )
        with self.assertRaises(ValueError):
            parse_string_vocabulary(text, "parse_missing(const std::string & value)")

    def test_package_install_paths(self) -> None:
        package = _self_test_package(
            "demo_q8_0",
            files=("Demo-GGUF/demo-q8_0.gguf", "Demo-GGUF/config.json"),
            strip_prefix="Demo-GGUF",
        )
        self.assertEqual(package.install_paths, {
            "models/Demo-GGUF",
            "models/Demo-GGUF/demo-q8_0.gguf",
            "models/Demo-GGUF/config.json",
        })

    def test_catalog_download_id_must_be_a_package_id(self) -> None:
        errors, _ = _self_test_catalog_check(
            [_self_test_entry(download_id="demo")],
            {"demo_q8_0": _self_test_package("demo_q8_0")},
        )
        self.assertTrue(any("download_id 'demo' is not a packages[].id" in error for error in errors))

    def test_catalog_path_must_match_the_package(self) -> None:
        packages = {"demo_q8_0": _self_test_package("demo_q8_0", files=("demo-q8_0.gguf",))}
        errors, _ = _self_test_catalog_check(
            [_self_test_entry(path="models/Demo")], packages)
        self.assertTrue(any("path 'models/Demo' is neither" in error for error in errors))
        errors, _ = _self_test_catalog_check(
            [_self_test_entry(path="models/Demo-GGUF/demo-q8_0.gguf")], packages)
        self.assertEqual(errors, [])

    def test_catalog_rejects_unknown_task_and_family(self) -> None:
        errors, _ = _self_test_catalog_check(
            [_self_test_entry(task="nope"), _self_test_entry(entry_id="x", family="ghost")],
            {"demo_q8_0": _self_test_package("demo_q8_0")},
        )
        self.assertTrue(any("task 'nope' is not accepted" in error for error in errors))
        self.assertTrue(any("no model_specs/ghost.json" in error for error in errors))

    def test_catalog_reports_unreachable_directory_and_default(self) -> None:
        packages = {
            "demo_q8_0": _self_test_package("demo_q8_0", default=True),
            "demo_extra_q8_0": _self_test_package(
                "demo_extra_q8_0", target_directory="Demo-GGUF/extra"),
        }
        _, warnings = _self_test_catalog_check([_self_test_entry()], packages)
        self.assertTrue(any("which no" in warning and "extra" in warning for warning in warnings))
        errors, _ = _self_test_catalog_check(
            [_self_test_entry(download_id="demo_extra_q8_0", path="models/Demo-GGUF/extra")],
            packages,
        )
        self.assertTrue(any("default package 'demo_q8_0' is not reachable" in e for e in errors))

    def test_catalog_reports_unsupported_downloads_as_not_installable(self) -> None:
        packages = {"demo_q8_0": _self_test_package("demo_q8_0", download_kind="unsupported")}
        errors, warnings = _self_test_catalog_check([], packages)
        self.assertEqual(errors, [])
        self.assertEqual([w for w in warnings if "no model_specs" in w], [])
        self.assertEqual(check_native_manager_sync(packages) != [], True)

    def test_model_params_group_must_resolve(self) -> None:
        specs = {"demo": {"family": "demo", "options": {"request": [{"name": "speed"}]}}}
        errors, warnings = check_model_params_sync(
            {"demo": [{"name": "speed"}, {"name": "gone"}], "ghost": [{"name": "speed"}]},
            specs,
            [],
            MODEL_PARAMS_PATH,
        )
        self.assertTrue(any("parameter group 'ghost' matches no" in error for error in errors))
        self.assertTrue(any("control 'gone' is not in" in warning for warning in warnings))


def _self_test_package(
    package_id: str,
    *,
    family: str = "demo",
    target_directory: str = "Demo-GGUF",
    files: tuple[str, ...] = (),
    strip_prefix: str = "",
    default: bool = False,
    download_kind: str = INSTALLABLE_DOWNLOAD_KIND,
) -> SpecPackage:
    return SpecPackage(
        family=family,
        id=package_id,
        format="gguf",
        target_directory=target_directory,
        default=default,
        download_kind=download_kind,
        files=files,
        strip_prefix=strip_prefix,
    )


def _self_test_entry(
    *,
    entry_id: str = "demo",
    family: str = "demo",
    path: str = "models/Demo-GGUF",
    task: str = "tts",
    mode: str = "offline",
    download_id: str = "demo_q8_0",
) -> CatalogEntry:
    return CatalogEntry(
        index=0,
        id=entry_id,
        family=family,
        path=path,
        task=task,
        mode=mode,
        download_id=download_id,
    )


def _self_test_catalog_check(
    entries: list[CatalogEntry],
    packages: dict[str, SpecPackage],
) -> tuple[list[str], list[str]]:
    specs = {"demo": {"family": "demo", "ui": {"recommended_package": "demo_q8_0"}}}
    return check_catalog_sync(
        entries, specs, packages, {"tts", "asr"}, {"offline", "streaming"}, CATALOG_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmake", type=Path, default=CMAKE_PATH, help="Path to top-level CMakeLists.txt")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH, help="Path to registry.cpp")
    parser.add_argument("--specs-dir", type=Path, default=SPECS_DIR, help="Directory containing model spec JSON files")
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH, help="Path to models_catalog.json")
    parser.add_argument("--model-params", type=Path, default=MODEL_PARAMS_PATH, help="Path to model_params.json")
    parser.add_argument("--session", type=Path, default=SESSION_PATH, help="Path to session.cpp")
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

    for path, label in ((args.catalog, "catalog"), (args.model_params, "model params"), (args.session, "session")):
        if not path.is_file():
            print(f"error: {label} not found: {path}", file=sys.stderr)
            return 2
    try:
        session_text = args.session.read_text(encoding="utf-8")
        task_kinds = parse_string_vocabulary(session_text, "parse_voice_task_kind(const std::string & value)")
        run_modes = parse_string_vocabulary(session_text, "parse_run_mode(const std::string & value)")
    except Exception as exc:
        print(f"error: failed to read task kinds from {rel(args.session)}: {exc}", file=sys.stderr)
        return 2

    specs_by_family, spec_packages, spec_errors = load_spec_packages(args.specs_dir)
    manager_packages, manager_errors = load_manager_packages(args.specs_dir)
    catalog_entries, catalog_errors = load_catalog(args.catalog)
    errors.extend(spec_errors)
    errors.extend(manager_errors)
    errors.extend(catalog_errors)
    errors.extend(check_loader_spec_sync(active_loaders, specs_by_family))
    manager_sync_errors, manager_sync_warnings = check_manager_sync(spec_packages, manager_packages)
    errors.extend(manager_sync_errors)
    warnings.extend(manager_sync_warnings)
    warnings.extend(check_native_manager_sync(spec_packages))
    catalog_sync_errors, catalog_sync_warnings = check_catalog_sync(
        catalog_entries, specs_by_family, spec_packages, task_kinds, run_modes, args.catalog)
    errors.extend(catalog_sync_errors)
    warnings.extend(catalog_sync_warnings)
    try:
        params_payload = load_json(args.model_params)
    except json.JSONDecodeError as exc:
        errors.append(f"{rel(args.model_params)}: invalid JSON: {exc}")
        params_payload = {}
    params_errors, params_warnings = check_model_params_sync(
        params_payload, specs_by_family, catalog_entries, args.model_params)
    errors.extend(params_errors)
    warnings.extend(params_warnings)

    server_packages = sum(1 for package in spec_packages.values() if package.installable)
    print(
        f"active_loaders={len(active_loaders)} commented_loaders={len(commented_loaders)} "
        f"specs={len(specs_by_family)} packages={len(spec_packages)} "
        f"manager_packages={len(manager_packages)} server_packages={server_packages} "
        f"catalog_entries={len(catalog_entries)} task_kinds={len(task_kinds)}"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    if errors:
        print("loader/spec/catalog sync failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nFix: keep model_specs/*.json, model_manager_v2.py, registered loaders, "
            "published default GGUF packages, and webui/configs/models_catalog.json aligned. "
            "A catalog entry must name a real packages[].id and install into that package's "
            "location. Schema-level validation belongs to the typed model-spec validator.",
            file=sys.stderr,
        )
        return 1

    print("ok: runtime loaders, model_specs, model_manager_v2, and the WebUI catalog are in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
