#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECS_DIR = REPO_ROOT / "model_specs"


class ManagerError(RuntimeError):
    pass


class InstallCancelled(ManagerError):
    pass


@dataclass(frozen=True)
class PackageRecord:
    family: str
    id: str
    display_name: str
    target_directory: str
    format: str
    precision: str
    files: tuple[str, ...]
    strip_prefix: str
    download: dict[str, Any]
    default: bool


@dataclass(frozen=True)
class RemoteFileInfo:
    size: int | None
    revision: str
    etag: str


def huggingface_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        cached = token_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    return None


def http_headers() -> dict[str, str]:
    headers = {"User-Agent": "audio.cpp model_manager_v2.py"}
    token = huggingface_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def quote_repo_path(value: str) -> str:
    return "/".join(quote(part, safe="") for part in value.split("/"))


def validate_relative_path(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ManagerError(f"{label} must be a safe relative path: {value}")
    if not value or value == ".":
        raise ManagerError(f"{label} must not be empty")
    return path


def stripped_path(remote_path: str, strip_prefix: str) -> Path:
    result = remote_path
    if strip_prefix:
        prefix = strip_prefix.rstrip("/")
        if result == prefix:
            raise ManagerError(f"strip_prefix removes the whole file path: {remote_path}")
        if not result.startswith(prefix + "/"):
            raise ManagerError(f"file path does not start with strip_prefix '{strip_prefix}': {remote_path}")
        result = result[len(prefix) + 1 :]
    return validate_relative_path(result, "local file path")


def load_specs(specs_dir: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in sorted(specs_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            spec = json.load(handle)
        spec["_spec_path"] = str(path)
        specs.append(spec)
    return specs


def merged_download(spec: dict[str, Any], package: dict[str, Any]) -> dict[str, Any]:
    defaults = spec.get("package_defaults", {}).get("download", {})
    download = dict(defaults)
    download.update(package.get("download", {}))
    return download


def flatten_packages(specs: list[dict[str, Any]]) -> list[PackageRecord]:
    records: list[PackageRecord] = []
    for spec in specs:
        family = spec["family"]
        for package in spec.get("packages", []):
            records.append(
                PackageRecord(
                    family=family,
                    id=package["id"],
                    display_name=package["display_name"],
                    target_directory=package["target_directory"],
                    format=package["format"],
                    precision=package["precision"],
                    files=tuple(package["files"]),
                    strip_prefix=package.get("strip_prefix", ""),
                    download=merged_download(spec, package),
                    default=bool(package.get("default", False)),
                )
            )
    return records


def select_package(records: list[PackageRecord], args: argparse.Namespace) -> PackageRecord:
    format_filter = getattr(args, "format", None)
    precision_filter = getattr(args, "precision", None)
    exact = [record for record in records if record.id == args.package]
    if exact:
        if format_filter or precision_filter:
            raise ManagerError("format/precision filters are only used when selecting by family")
        return exact[0]

    family = [record for record in records if record.family == args.package]
    if not family:
        raise ManagerError(f"unknown package or family: {args.package}")
    candidates = family
    if format_filter:
        candidates = [record for record in candidates if record.format == format_filter]
    if precision_filter:
        candidates = [record for record in candidates if record.precision == precision_filter]
    if not candidates:
        raise ManagerError(f"no package for family '{args.package}' matches the requested filters")
    default_candidates = [record for record in candidates if record.default]
    if len(default_candidates) == 1:
        return default_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    ids = ", ".join(record.id for record in candidates)
    raise ManagerError(f"ambiguous family '{args.package}', choose one package id: {ids}")


def hf_url(repo: str, revision: str, remote_path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote_repo_path(remote_path)}"


def check_remote_file(package: PackageRecord, remote_path: str) -> RemoteFileInfo:
    repo = package.download["repo"]
    revision = package.download.get("revision", "main")
    request = Request(hf_url(repo, revision, remote_path), headers=http_headers(), method="HEAD")
    try:
        with urlopen(request, timeout=60) as response:
            size = response.headers.get("Content-Length")
            return RemoteFileInfo(
                size=int(size) if size else None,
                revision=response.headers.get("X-Repo-Commit", ""),
                etag=response.headers.get("ETag", "").strip('"'),
            )
    except HTTPError as error:
        if package.download.get("gated") is True and error.code in (401, 403):
            return RemoteFileInfo(size=None, revision="", etag="")
        raise ManagerError(f"remote file is not accessible: {repo}/{remote_path} ({error.code})") from error


def cancellation_requested(cancel_file: Path | None) -> bool:
    return cancel_file is not None and cancel_file.is_file()


def download_file(
    package: PackageRecord,
    remote_path: str,
    output_path: Path,
    progress=None,
    cancel_file: Path | None = None,
) -> None:
    repo = package.download["repo"]
    revision = package.download.get("revision", "main")
    request = Request(hf_url(repo, revision, remote_path), headers=http_headers())
    try:
        with urlopen(request, timeout=300) as response:
            expected_header = response.headers.get("Content-Length")
            expected = int(expected_header) if expected_header is not None else None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            last_report = 0
            with output_path.open("wb") as handle:
                while True:
                    if cancellation_requested(cancel_file):
                        raise InstallCancelled("download cancelled by user")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    total += len(chunk)
                    if progress is not None and total - last_report >= 8 * 1024 * 1024:
                        progress(total, expected)
                        last_report = total
            if progress is not None:
                progress(total, expected)
            if cancellation_requested(cancel_file):
                raise InstallCancelled("download cancelled by user")
            if expected is not None and total != expected:
                raise ManagerError(f"downloaded size mismatch for {output_path}: {total} != {expected}")
    except HTTPError as error:
        if package.download.get("gated") is True and error.code in (401, 403):
            raise ManagerError(
                f"{repo}/{remote_path} requires accepted Hugging Face access and a valid HF token"
            ) from error
        raise ManagerError(f"failed to download {repo}/{remote_path}: HTTP {error.code}") from error


def ensure_hf_package(package: PackageRecord) -> None:
    kind = package.download.get("kind")
    if kind != "huggingface_snapshot":
        raise ManagerError(
            f"{package.id} uses download kind '{kind}'. model_manager_v2 only installs huggingface_snapshot packages; "
            "use tools/model_manager.py for legacy composite or converter installs."
        )
    if not package.download.get("repo"):
        raise ManagerError(f"{package.id} has no Hugging Face repo")


def package_manifest_path(package: PackageRecord, models_root: Path) -> Path:
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    return models_root / target_dir / f".audiocpp-package-{package.id}.json"


def read_package_manifest(package: PackageRecord, models_root: Path | None) -> dict[str, Any] | None:
    if models_root is None:
        return None
    path = package_manifest_path(package, models_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def write_package_manifest(
    package: PackageRecord,
    models_root: Path,
    resolved_revision: str,
    remote_files: dict[str, RemoteFileInfo],
) -> None:
    path = package_manifest_path(package, models_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "package_id": package.id,
        "repo": package.download.get("repo", ""),
        "requested_revision": package.download.get("revision", "main"),
        "resolved_revision": resolved_revision,
        "installed_at_unix": int(time.time()),
        "files": {
            name: {"size": info.size, "etag": info.etag}
            for name, info in remote_files.items()
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def reusable_package_outputs(
    package: PackageRecord,
    records: list[PackageRecord],
    models_root: Path,
) -> set[Path]:
    """Return files safely shared with another completely installed package."""
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    final_dir = models_root / target_dir
    requested = {
        (remote, final_dir / stripped_path(remote, package.strip_prefix))
        for remote in package.files
    }
    reusable: set[Path] = set()
    for other in records:
        if other.id == package.id or other.download != package.download:
            continue
        if not package_is_installed(other, models_root):
            continue
        other_dir = models_root / validate_relative_path(other.target_directory, "target_directory")
        for remote in other.files:
            output = other_dir / stripped_path(remote, other.strip_prefix)
            if (remote, output) in requested:
                reusable.add(output)
    return reusable


def install_package(package: PackageRecord, records: list[PackageRecord], args: argparse.Namespace) -> None:
    ensure_hf_package(package)
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    models_root = Path(args.models_root)
    final_dir = models_root / target_dir
    plan = [(remote, final_dir / stripped_path(remote, package.strip_prefix)) for remote in package.files]
    full_plan = list(plan)

    print(f"selected {package.id} ({package.family})")
    print(f"repo {package.download['repo']}@{package.download.get('revision', 'main')}")
    print(f"target {final_dir}")
    for remote, output in plan:
        if args.check:
            info = check_remote_file(package, remote)
            if info.size is None and package.download.get("gated") is True:
                suffix = " gated_access_required"
            else:
                suffix = f" size={info.size}" if info.size is not None else ""
            print(f"check {remote}{suffix}")
        else:
            print(f"file remote={remote} local={output.relative_to(models_root)}")

    if args.dry_run or args.check:
        return
    existing_outputs = [output for _remote, output in plan if output.exists()]
    if existing_outputs and not args.overwrite:
        if len(existing_outputs) == len(plan) and all(output.is_file() for output in existing_outputs):
            print(f"already installed {package.id} -> {final_dir}")
            return
        reusable = reusable_package_outputs(package, records, models_root)
        conflicts = [output for output in existing_outputs if output not in reusable]
        if conflicts:
            raise ManagerError(f"some package files already exist in: {final_dir} (use --overwrite)")
        plan = [(remote, output) for remote, output in plan if output not in reusable]
    models_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{package.target_directory.replace('/', '_')}.", dir=models_root))
    cancel_file = Path(args.cancel_file) if getattr(args, "cancel_file", "") else None
    try:
        if cancellation_requested(cancel_file):
            raise InstallCancelled("download cancelled by user")
        progress_enabled = bool(getattr(args, "progress", False))
        remote_files: dict[str, RemoteFileInfo] = {}
        if progress_enabled:
            remote_files = {remote: check_remote_file(package, remote) for remote, _output in plan}
        elif plan:
            remote_files[plan[0][0]] = check_remote_file(package, plan[0][0])
        expected_sizes = [remote_files[remote].size for remote, _output in plan] if progress_enabled else []
        total_expected = sum(size for size in expected_sizes if size is not None)
        if any(size is None for size in expected_sizes):
            total_expected = 0
        completed = 0

        def emit_progress(downloaded: int) -> None:
            print(
                f"AUDIOCPP_PROGRESS downloaded={downloaded} total={total_expected}",
                flush=True,
            )

        if progress_enabled:
            emit_progress(0)
        for remote, output in plan:
            destination = staging / output.relative_to(final_dir)
            if progress_enabled:
                download_file(
                    package,
                    remote,
                    destination,
                    lambda file_bytes, _expected, base=completed: emit_progress(base + file_bytes),
                    cancel_file,
                )
                completed += destination.stat().st_size
                emit_progress(completed)
            else:
                download_file(package, remote, destination, cancel_file=cancel_file)
        if cancellation_requested(cancel_file):
            raise InstallCancelled("download cancelled by user")
        # Record every package-owned file, including sidecars reused from a
        # sibling precision. Version checks compare these per-file ETags and
        # must not depend on a repository-wide revision.
        for remote, _output in full_plan:
            if remote not in remote_files:
                remote_files[remote] = check_remote_file(package, remote)
        if not final_dir.exists():
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(final_dir)
        else:
            # Precision variants commonly share one target directory. Merge the
            # fully downloaded staging tree so Q8 and F16/BF16 can coexist. An
            # overwrite replaces only this package's files, preserving siblings.
            for staged_file in sorted(path for path in staging.rglob("*") if path.is_file()):
                relative = staged_file.relative_to(staging)
                destination = final_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if not args.overwrite:
                        raise ManagerError(f"package file already exists: {destination} (use --overwrite)")
                    destination.unlink()
                staged_file.replace(destination)
            shutil.rmtree(staging)
        resolved_revision = next(
            (info.revision for info in remote_files.values() if info.revision),
            package.download.get("revision", "main"),
        )
        write_package_manifest(package, models_root, resolved_revision, remote_files)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed {package.id} -> {final_dir}")


def uninstall_package(package: PackageRecord, records: list[PackageRecord], args: argparse.Namespace) -> None:
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    models_root = Path(args.models_root)
    final_dir = models_root / target_dir
    package_files = [final_dir / stripped_path(remote, package.strip_prefix) for remote in package.files]
    manifest_path = package_manifest_path(package, models_root)
    existing = [path for path in package_files if path.is_file() or path.is_symlink()]
    if not existing:
        raise ManagerError(f"package is not installed: {package.id}")

    reusable = reusable_package_outputs(package, records, models_root)
    for path in existing:
        if path not in reusable:
            path.unlink()
    manifest_path.unlink(missing_ok=True)

    # Remove only directories made empty by this package. Other precision
    # variants and unrelated files in the shared target remain untouched.
    cleanup: set[Path] = set()
    for path in package_files:
        parent = path.parent
        while parent != final_dir.parent:
            cleanup.add(parent)
            parent = parent.parent
    for directory in sorted(cleanup, key=lambda path: len(path.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    print(f"removed {package.id} -> {final_dir}")


def clean_partial_package(package: PackageRecord, args: argparse.Namespace) -> None:
    models_root = Path(args.models_root).resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    prefix = f".{package.target_directory.replace('/', '_')}."
    removed = 0
    for candidate in models_root.glob(prefix + "*"):
        resolved = candidate.resolve()
        try:
            resolved.relative_to(models_root)
        except ValueError as error:
            raise ManagerError(f"refusing to clean staging path outside models root: {resolved}") from error
        if candidate.is_dir():
            shutil.rmtree(candidate)
            removed += 1
    print(f"cleaned {removed} partial download director{'y' if removed == 1 else 'ies'} for {package.id}")


def command_list(records: list[PackageRecord], args: argparse.Namespace) -> None:
    rows = [
        {
            "family": record.family,
            "id": record.id,
            "display_name": record.display_name,
            "format": record.format,
            "precision": record.precision,
            "default": record.default,
            "target_directory": record.target_directory,
            "repo": record.download.get("repo", ""),
        }
        for record in records
    ]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return
    for row in rows:
        default = " default" if row["default"] else ""
        print(f"{row['id']:<44} {row['family']:<24} {row['format']:<11} {row['precision']:<8}{default}")


def command_info(records: list[PackageRecord], args: argparse.Namespace) -> None:
    package = select_package(records, args)
    row = {
        "family": package.family,
        "id": package.id,
        "display_name": package.display_name,
        "format": package.format,
        "precision": package.precision,
        "default": package.default,
        "target_directory": package.target_directory,
        "files": list(package.files),
        "strip_prefix": package.strip_prefix,
        "download": package.download,
    }
    if args.json:
        print(json.dumps(row, indent=2, ensure_ascii=False))
        return
    print(f"id: {package.id}")
    print(f"family: {package.family}")
    print(f"name: {package.display_name}")
    print(f"format: {package.format}")
    print(f"precision: {package.precision}")
    print(f"target: {package.target_directory}")
    print(f"download: {package.download.get('kind')} {package.download.get('repo', '')}")
    for remote in package.files:
        print(f"file: {remote}")


def package_is_installed(package: PackageRecord, models_root: Path | None) -> bool:
    if models_root is None:
        return False
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    final_dir = models_root / target_dir
    if not final_dir.is_dir():
        return False
    return bool(package.files) and all(
        (final_dir / stripped_path(remote_path, package.strip_prefix)).is_file()
        for remote_path in package.files
    )


def package_version_state(
    installed: bool,
    manifest: dict[str, Any] | None,
    remote_files: dict[str, RemoteFileInfo],
) -> str:
    """Compare only files owned by this package, not the repository commit.

    Hugging Face's X-Repo-Commit changes whenever *any* file in a repository is
    uploaded. Several audio.cpp packages share one repository, so comparing
    that repository-wide revision incorrectly marked every installed package
    as outdated after an unrelated model upload. Per-file ETags identify the
    actual package payload and avoid that false positive.
    """
    if not installed:
        return "not_installed"
    manifest_files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(manifest_files, dict):
        return "unknown"

    compared = 0
    for remote_path, stored in manifest_files.items():
        if not isinstance(stored, dict):
            continue
        local_etag = str(stored.get("etag", "")).strip('"')
        remote_etag = remote_files.get(remote_path, RemoteFileInfo(None, "", "")).etag.strip('"')
        if not local_etag or not remote_etag:
            continue
        compared += 1
        if local_etag != remote_etag:
            return "update_available"
    return "up_to_date" if compared else "unknown"


def package_size_record(package: PackageRecord, models_root: Path | None = None) -> dict[str, Any]:
    installed = package_is_installed(package, models_root)
    try:
        ensure_hf_package(package)
        total = 0
        unknown = False
        remote_revision = ""
        remote_files: dict[str, RemoteFileInfo] = {}
        for remote_path in package.files:
            info = check_remote_file(package, remote_path)
            remote_files[remote_path] = info
            if info.size is None:
                unknown = True
            else:
                total += info.size
            if not remote_revision and info.revision:
                remote_revision = info.revision
        manifest = read_package_manifest(package, models_root)
        local_revision = str(manifest.get("resolved_revision", "")) if manifest else ""
        version_state = package_version_state(installed, manifest, remote_files)
        state = "gated" if unknown and package.download.get("gated") is True else "unknown" if unknown else "ok"
        return {
            "id": package.id,
            "size_bytes": None if unknown else total,
            "state": state,
            "message": "Hugging Face access and a valid token are required" if state == "gated" else "",
            "installed": installed,
            "version_state": version_state,
            "local_revision": local_revision,
            "remote_revision": remote_revision,
        }
    except ManagerError as error:
        return {
            "id": package.id,
            "size_bytes": None,
            "state": "error",
            "message": str(error),
            "installed": installed,
            "version_state": "unknown" if installed else "not_installed",
            "local_revision": "",
            "remote_revision": "",
        }


def command_sizes(records: list[PackageRecord], args: argparse.Namespace) -> None:
    selected = records
    if args.package:
        requested = set(args.package)
        selected = [record for record in records if record.id in requested]
        missing = sorted(requested - {record.id for record in selected})
        if missing:
            raise ManagerError(f"unknown package ids: {', '.join(missing)}")
    models_root = Path(args.models_root) if args.models_root else None
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda package: package_size_record(package, models_root), selected))
    print(json.dumps(rows, ensure_ascii=False))


def command_installed(records: list[PackageRecord], args: argparse.Namespace) -> None:
    models_root = Path(args.models_root)
    rows = [
        {
            "id": package.id,
            "size_bytes": None,
            "state": "pending",
            "message": "",
            "installed": package_is_installed(package, models_root),
            "version_state": "unknown" if package_is_installed(package, models_root) else "not_installed",
            "local_revision": "",
            "remote_revision": "",
        }
        for package in records
    ]
    print(json.dumps(rows, ensure_ascii=False))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install audio.cpp model packages from model_specs/*.json.")
    parser.add_argument("--specs-dir", default=str(DEFAULT_SPECS_DIR), help="directory containing model spec JSON files")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list installable packages")
    list_parser.add_argument("--json", action="store_true")

    info_parser = sub.add_parser("info", help="show one package or family default")
    info_parser.add_argument("package")
    info_parser.add_argument("--format")
    info_parser.add_argument("--precision")
    info_parser.add_argument("--json", action="store_true")

    sizes_parser = sub.add_parser("sizes", help="check package download sizes without downloading files")
    sizes_parser.add_argument("package", nargs="*", help="optional package ids; default checks every package")
    sizes_parser.add_argument("--jobs", type=int, default=12, help="parallel metadata checks")
    sizes_parser.add_argument("--models-root", help="also report packages whose required files are installed")
    sizes_parser.add_argument("--json", action="store_true", help="retained for command symmetry; output is JSON")

    installed_parser = sub.add_parser("installed", help="report locally installed packages without network access")
    installed_parser.add_argument("--models-root", default="models")
    installed_parser.add_argument("--json", action="store_true", help="retained for command symmetry; output is JSON")

    uninstall_parser = sub.add_parser("uninstall", help="remove only the files belonging to one installed package")
    uninstall_parser.add_argument("package", help="package id or family")
    uninstall_parser.add_argument("--format")
    uninstall_parser.add_argument("--precision")
    uninstall_parser.add_argument("--models-root", default="models")

    clean_parser = sub.add_parser("clean-partial", help="remove abandoned staging downloads for one package")
    clean_parser.add_argument("package", help="package id or family")
    clean_parser.add_argument("--models-root", default="models")

    install_parser = sub.add_parser("install", help="install one Hugging Face snapshot package")
    install_parser.add_argument("package", help="package id or family")
    install_parser.add_argument("--format")
    install_parser.add_argument("--precision")
    install_parser.add_argument("--models-root", default="models")
    install_parser.add_argument("--overwrite", action="store_true")
    install_parser.add_argument("--cancel-file", default="", help=argparse.SUPPRESS)
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--check", action="store_true", help="check remote files without downloading")
    install_parser.add_argument(
        "--progress",
        action="store_true",
        help="emit machine-readable AUDIOCPP_PROGRESS lines while downloading",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        records = flatten_packages(load_specs(Path(args.specs_dir)))
        if args.command == "list":
            command_list(records, args)
        elif args.command == "info":
            command_info(records, args)
        elif args.command == "sizes":
            command_sizes(records, args)
        elif args.command == "installed":
            command_installed(records, args)
        elif args.command == "uninstall":
            uninstall_package(select_package(records, args), records, args)
        elif args.command == "clean-partial":
            clean_partial_package(select_package(records, args), args)
        elif args.command == "install":
            install_package(select_package(records, args), records, args)
        else:
            raise ManagerError(f"unknown command: {args.command}")
        return 0
    except ManagerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
