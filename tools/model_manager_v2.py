#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
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
    exact = [record for record in records if record.id == args.package]
    if exact:
        if args.format or args.precision:
            raise ManagerError("format/precision filters are only used when selecting by family")
        return exact[0]

    family = [record for record in records if record.family == args.package]
    if not family:
        raise ManagerError(f"unknown package or family: {args.package}")
    candidates = family
    if args.format:
        candidates = [record for record in candidates if record.format == args.format]
    if args.precision:
        candidates = [record for record in candidates if record.precision == args.precision]
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


def check_remote_file(package: PackageRecord, remote_path: str) -> int | None:
    repo = package.download["repo"]
    revision = package.download.get("revision", "main")
    request = Request(hf_url(repo, revision, remote_path), headers=http_headers(), method="HEAD")
    try:
        with urlopen(request, timeout=60) as response:
            size = response.headers.get("Content-Length")
            return int(size) if size else None
    except HTTPError as error:
        if package.download.get("gated") is True and error.code in (401, 403):
            return None
        raise ManagerError(f"remote file is not accessible: {repo}/{remote_path} ({error.code})") from error


def download_file(package: PackageRecord, remote_path: str, output_path: Path) -> None:
    repo = package.download["repo"]
    revision = package.download.get("revision", "main")
    request = Request(hf_url(repo, revision, remote_path), headers=http_headers())
    try:
        with urlopen(request, timeout=300) as response:
            expected = response.headers.get("Content-Length")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with output_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    total += len(chunk)
            if expected is not None and total != int(expected):
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


def install_package(package: PackageRecord, args: argparse.Namespace) -> None:
    ensure_hf_package(package)
    target_dir = validate_relative_path(package.target_directory, "target_directory")
    models_root = Path(args.models_root)
    final_dir = models_root / target_dir
    plan = [(remote, final_dir / stripped_path(remote, package.strip_prefix)) for remote in package.files]

    print(f"selected {package.id} ({package.family})")
    print(f"repo {package.download['repo']}@{package.download.get('revision', 'main')}")
    print(f"target {final_dir}")
    for remote, output in plan:
        if args.check:
            size = check_remote_file(package, remote)
            if size is None and package.download.get("gated") is True:
                suffix = " gated_access_required"
            else:
                suffix = f" size={size}" if size is not None else ""
            print(f"check {remote}{suffix}")
        else:
            print(f"file remote={remote} local={output.relative_to(models_root)}")

    if args.dry_run or args.check:
        return
    if final_dir.exists() and not args.overwrite:
        raise ManagerError(f"target already exists: {final_dir} (use --overwrite)")
    models_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{package.target_directory.replace('/', '_')}.", dir=models_root))
    try:
        for remote, output in plan:
            download_file(package, remote, staging / output.relative_to(final_dir))
        if final_dir.exists():
            shutil.rmtree(final_dir)
        staging.rename(final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"installed {package.id} -> {final_dir}")


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

    install_parser = sub.add_parser("install", help="install one Hugging Face snapshot package")
    install_parser.add_argument("package", help="package id or family")
    install_parser.add_argument("--format")
    install_parser.add_argument("--precision")
    install_parser.add_argument("--models-root", default="models")
    install_parser.add_argument("--overwrite", action="store_true")
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--check", action="store_true", help="check remote files without downloading")
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
        elif args.command == "install":
            install_package(select_package(records, args), args)
        else:
            raise ManagerError(f"unknown command: {args.command}")
        return 0
    except ManagerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
