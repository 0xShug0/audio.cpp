#!/usr/bin/env python3
"""WebUI front-end for ``tools/model_manager.py`` — same CLI, sturdier downloads.

The Gradio WebUI kicks off model installs as a background subprocess
(``model_manager_webui.py install <package_id>``), where two things matter that
the upstream tool intentionally does not do:

* **Resumable, no-Torch downloads.** Weights land in a ``<file>.part`` sidecar and
  resume across runs via HTTP ``Range`` (or the hub client's own resume for
  Xet-backed repos), so a dropped multi-GB transfer picks up instead of
  restarting. A plain snapshot download must not import Torch either — its native
  DLLs can fail to load in some environments (e.g. conda Python on Windows) and
  would needlessly block downloads for models whose inference runs in the C++
  server.
* **Windows-tolerant finalization.** Defender/indexers and the WebUI's own
  progress scan can briefly hold a directory handle right after the last shard is
  written, so promoting a staged directory retries past transient sharing
  violations.

Rather than fork the 2700-line upstream tool, this module imports it and
overrides only the download path, then delegates to its unchanged CLI. Everything
not redefined here — the catalog, argument parsing, ``list``/``info``, converter
installs, all the ``convert_*`` post-processing — comes straight from
``model_manager`` (referenced as ``mm.*`` below, so what is overridden vs.
inherited stays obvious).
"""
from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import model_manager as mm

# huggingface_hub is imported lazily (see _ensure_download_deps) so that
# ``list``/``info`` keep working without it, exactly as upstream keeps Torch lazy.
hf_hub_download = None


# --- dependency loading ------------------------------------------------------

def _ensure_download_deps() -> None:
    """Import only what a plain snapshot/file download needs: huggingface_hub + yaml.

    Downloading weights must not drag in Torch (see the module docstring). yaml is
    published into the upstream module's namespace because its config/convert
    helpers read ``mm.yaml``."""
    global hf_hub_download
    if hf_hub_download is not None:
        return
    from huggingface_hub import hf_hub_download as _hf_hub_download
    import yaml as _yaml

    hf_hub_download = _hf_hub_download
    mm.yaml = _yaml


def _ensure_install_deps() -> None:
    """Everything _ensure_download_deps loads, plus Torch/safetensors for conversions.

    Only the composite/converter paths need this; plain snapshot downloads call
    _ensure_download_deps() so they stay Torch-free. Torch et al. are populated on
    the upstream module (mm.torch, mm.safe_open, ...) because its convert_* helpers
    read them from there."""
    _ensure_download_deps()
    mm._ensure_install_deps()


# --- plain-URL downloads (resumable) -----------------------------------------

def download_file(
    url: str,
    target: Path,
    expected_size: int | None,
    label: str | None = None,
) -> int:
    """Download ``url`` into ``target``, resuming across runs when possible.

    Bytes land in a sidecar ``<target>.part`` file first. If a previous run was
    interrupted, an HTTP ``Range`` request fetches only the missing tail instead
    of restarting the transfer (critical for multi-GB weights on flaky links);
    the ``.part`` file is promoted to ``target`` only once the full length has
    arrived. Files already present at the expected size are skipped outright.
    """
    name = label or target.name
    if expected_size is not None and target.is_file() and target.stat().st_size == expected_size:
        print(f"skip {name} (already complete)")
        return expected_size

    part = target.with_name(target.name + ".part")
    existing = part.stat().st_size if part.is_file() else 0
    if expected_size is not None and existing >= expected_size:
        # A finished-but-unpromoted leftover promotes as-is; anything larger than
        # expected is corrupt, so discard it and start over.
        if existing == expected_size:
            part.replace(target)
            print(f"skip {name} (already complete)")
            return expected_size
        part.unlink()
        existing = 0

    headers = mm.http_headers()
    mode = "wb"
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"resume {name} (from {existing} bytes)")
    else:
        print(f"download {name}")

    request = Request(url, headers=headers)
    try:
        response = urlopen(request)
    except HTTPError as ex:
        if ex.code == 416 and existing > 0:
            # Requested range past EOF: the server has nothing more to send.
            if expected_size is None or existing == expected_size:
                part.replace(target)
                return existing
            part.unlink(missing_ok=True)
        raise

    with response:
        status = getattr(response, "status", None) or response.getcode()
        if existing > 0 and status != 206:
            # Server ignored the Range header (answered 200) — start over.
            print(f"restart {name} (server ignored resume)")
            existing = 0
            mode = "wb"
        written = existing
        with part.open(mode) as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)

    if expected_size is not None and written != expected_size:
        raise RuntimeError(f"downloaded size mismatch for {target}: {written} != {expected_size}")
    part.replace(target)
    return written


# --- huggingface_hub downloads (Xet-backed repos) ----------------------------

def download_hf_file(
    source,
    relative_path: str,
    destination_root: Path,
    expected_size: int | None,
) -> None:
    """Fetch one repo file into ``destination_root/relative_path`` via huggingface_hub.

    Plain HTTP against the resolve URL cannot be used here: Xet-backed repos redirect
    to a CDN that rejects ordinary GETs, so only the hub client (with ``hf_xet``) can
    pull their weights. ``local_dir`` writes the file straight to its final location
    instead of duplicating it in the shared blob cache, and the hub client does its
    own resume, so an interrupted run picks up where it left off just like
    ``download_file`` does for the plain-URL callers.
    """
    destination = destination_root / relative_path
    if expected_size is not None and destination.is_file() and destination.stat().st_size == expected_size:
        print(f"skip {relative_path} (already complete)")
        return
    print(f"download {relative_path}")
    hf_hub_download(
        repo_id=source.repo_id,
        filename=relative_path,
        revision=source.revision,
        local_dir=destination_root,
        token=mm.huggingface_token(),
    )


def prune_hf_local_dir_cache(destination_root: Path) -> None:
    """Drop the ``.cache/huggingface`` bookkeeping tree hf_hub_download leaves behind.

    With ``local_dir=`` the hub client stores per-file resume metadata under
    ``<local_dir>/.cache/huggingface``. It is invisible to validation (which only
    checks that required files exist), but without this it would be promoted into
    the installed model directory as stray junk. Only safe to call once every file
    for this destination has arrived, since removing it discards resume state.
    """
    cache_dir = destination_root / ".cache" / "huggingface"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)
    parent = destination_root / ".cache"
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def install_snapshot_into_dir(
    source,
    destination_root: Path,
    required_files: Iterable[str],
    *,
    validate: bool = True,
) -> None:
    files = mm.list_hf_files(source)
    for relative, expected_size in files:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_hf_file(source, relative, destination_root, expected_size)
    prune_hf_local_dir_cache(destination_root)
    if validate:
        mm.validate_required_files_list(required_files, destination_root, source.repo_id)


# --- resumable staging + Windows-tolerant finalization -----------------------

def staging_dir_name(package) -> str:
    """Deterministic staging directory name so an interrupted install resumes
    into the same place on the next run (a random ``mkdtemp`` name would strand
    the partial downloads)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", package.target_directory)
    return f"{safe}.partial"


def prune_staging_root(staging_root: Path) -> None:
    try:
        if staging_root.exists() and not any(staging_root.iterdir()):
            staging_root.rmdir()
    except OSError:
        pass


def promote_staging_directory(source: Path, destination: Path) -> None:
    """Rename a completed staging directory, tolerating transient Windows locks.

    Defender/indexers and the WebUI progress scan can briefly retain a directory
    enumeration handle just after the final shard is written. Windows then reports
    access denied/sharing violation even though the ACL and destination are valid.
    """
    retry_delays = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            source.rename(destination)
            return
        except OSError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in {5, 32}:
                raise
            print(
                f"retry model directory finalization ({attempt}/{len(retry_delays)}): "
                f"{source} -> {destination} ({error})"
            )
            time.sleep(delay)
    source.rename(destination)


def install_snapshot(package, source, models_root: Path, overwrite: bool) -> Path:
    target_dir = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = staging_root / staging_dir_name(package)
    staging_dir.mkdir(parents=True, exist_ok=True)
    try:
        pre_validate_files = tuple(relative for relative in package.required_files if relative != "audiovae.safetensors")
        install_snapshot_into_dir(source, staging_dir, pre_validate_files, validate=package.id != "voxcpm2")
        if package.id == "voxcpm2":
            mm.convert_voxcpm2_audiovae(staging_dir)
            mm.validate_required_files(package, staging_dir)
        elif package.id == "moss_tts_nano_100m_model":
            mm.convert_moss_tts_weights(staging_dir)
        if target_dir.exists():
            if not overwrite:
                raise RuntimeError(f"model directory already exists: {target_dir}")
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        promote_staging_directory(staging_dir, target_dir)
        return target_dir
    finally:
        # On success the staging dir was renamed away; on failure it is kept so
        # the next run resumes partial downloads. Drop the parent when empty.
        prune_staging_root(staging_root)


def install_composite_snapshot(package, source, models_root: Path, overwrite: bool) -> Path:
    package_root = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_bundle = staging_root / staging_dir_name(package)
    staging_bundle.mkdir(parents=True, exist_ok=True)
    staged_roots: dict[Path, Path] = {}
    try:
        staged_package_root = staging_bundle / package.target_directory
        for placement in source.placements:
            destination_root = mm.normalized_join(staged_package_root, placement.target_subdir)
            final_root = mm.normalized_join(package_root, placement.target_subdir)
            if final_root.exists() and not overwrite:
                mm.validate_required_files_list(placement.required_files, final_root, str(final_root))
                continue
            destination_root.mkdir(parents=True, exist_ok=True)
            install_snapshot_into_dir(placement.source, destination_root, placement.required_files)
            staged_roots[final_root] = destination_root

        if package.id == "vevo2":
            whisper_root = staging_bundle / "whisper-medium"
            mm.install_whisper_medium_dependency(whisper_root)
            staged_roots[package_root.parent / "whisper-medium"] = whisper_root
            mm.prepare_vevo2_snapshot_layout(staged_package_root)
        elif package.id == "ace_step":
            mm.convert_ace_step_silence_latent(staged_package_root / "acestep-v15-turbo")
            mm.convert_ace_step_silence_latent(staged_package_root / "acestep-v15-base")
        elif package.id == "moss_tts_nano_100m":
            mm.convert_moss_tts_weights(staged_package_root)
        elif package.id in {"irodori_tts_500m_v3", "irodori_tts_600m_v3_voice_design"}:
            mm.write_irodori_model_config(staged_package_root)
            dacvae_root = staged_package_root.parent / "Semantic-DACVAE-Japanese-32dim"
            if dacvae_root.exists():
                mm.convert_irodori_dacvae_weights(dacvae_root)
        elif package.id == "outetts_1_0_1b":
            dac_root = staged_package_root.parent / "DAC.speech.v1.0"
            if dac_root.exists():
                mm.convert_outetts_dac_weights(dac_root)
        elif package.id == "vibevoice_asr":
            mm.copy_bundled_model_manager_assets(
                "vibevoice_1_5b",
                staged_package_root,
                ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
            )
        elif package.id in {"vibevoice_1_5b", "vibevoice_7b"}:
            # VibeVoice 1.5B and 7B share the same Qwen2.5 tokenizer, and neither
            # upstream repo ships the tokenizer files, so both reuse one bundle.
            mm.copy_bundled_model_manager_assets(
                "vibevoice_1_5b",
                staged_package_root,
                ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
            )
        mm.validate_composite_required_files(package, staged_package_root, package_root)

        top_level_roots: list[Path] = []
        for final_root in sorted(staged_roots.keys(), key=lambda path: len(path.parts)):
            if any(final_root.is_relative_to(existing) for existing in top_level_roots):
                continue
            top_level_roots.append(final_root)

        for final_root in sorted(top_level_roots, key=lambda path: len(path.parts), reverse=True):
            if final_root.exists():
                if not overwrite:
                    raise RuntimeError(f"model directory already exists: {final_root}")
                shutil.rmtree(final_root)

        for final_root in sorted(top_level_roots, key=lambda path: len(path.parts)):
            destination_root = staged_roots[final_root]
            final_root.parent.mkdir(parents=True, exist_ok=True)
            promote_staging_directory(destination_root, final_root)
        shutil.rmtree(staging_bundle, ignore_errors=True)
        return package_root
    finally:
        # Failed runs keep their staging bundle so partial downloads resume;
        # successful runs already removed it above. Drop the parent when empty.
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass


# --- install command (chooses download-only vs. full deps) -------------------

def command_install(args) -> int:
    package = mm.PACKAGE_BY_ID.get(args.package_id)
    if package is None:
        raise RuntimeError(f"unknown package id: {args.package_id}")
    models_root = mm.resolve_path(args.models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    source = package.source
    if isinstance(source, mm.UnsupportedSource):
        raise RuntimeError(f"{package.id} is not installable: {source.reason}")
    if isinstance(source, mm.SnapshotSource):
        # Plain download: huggingface_hub only, no Torch DLLs.
        _ensure_download_deps()
        install_path = install_snapshot(package, source, models_root, args.overwrite)
    elif isinstance(source, mm.CompositeSnapshotSource):
        # Composite snapshots may run a Torch post-process step, so bring in full deps.
        _ensure_install_deps()
        install_path = install_composite_snapshot(package, source, models_root, args.overwrite)
    else:
        _ensure_install_deps()
        install_path = mm.install_converter(
            package,
            source,
            models_root,
            args.overwrite,
            args.source_file,
            args.output_file,
            args.source_dir,
            args.variant,
        )
    print(f"installed {package.id} -> {install_path}")
    return 0


# Patch the two entry points the upstream call-graph resolves by name at run time:
#   * main() dispatches ``install`` to ``command_install`` looked up in mm's globals;
#   * mm's own dependency installers (whisper-medium, demucs, converters) call
#     ``download_file`` in mm's globals — routing them through the resumable version
#     matches what a single combined module would do.
# Everything else this module overrides is reached only from command_install below,
# so it resolves within this module's own namespace and needs no patching.
mm.download_file = download_file
mm.command_install = command_install


if __name__ == "__main__":
    raise SystemExit(mm.main())
