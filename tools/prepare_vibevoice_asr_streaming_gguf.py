#!/usr/bin/env python3
"""Build (and optionally publish) the VibeVoice ASR Streaming GGUF packages.

Produces the BF16/Q8_0/Q4_K GGUFs named by `model_specs/vibevoice_asr_streaming.json`
for either size, and can create the Hugging Face repo and upload them.

  # convert only, into ./out
  tools/prepare_vibevoice_asr_streaming_gguf.py --size 1.5b --out out

  # convert, check each package transcribes, then publish
  tools/prepare_vibevoice_asr_streaming_gguf.py --size 1.5b --out out \
      --verify-with build/bin/audiocpp_cli \
      --upload audio-cpp/VibeVoice-ASR-Streaming-1.5B-GGUF

Nothing is uploaded unless --upload is passed. Uploading needs `huggingface_hub`
and a token with write access (`hf auth login`, or HF_TOKEN in the environment).
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

SIZES = {
    "1.5b": {"source": "microsoft/VibeVoice-ASR-Streaming-1.5B", "slug": "1.5b",
             "repo": "audio-cpp/VibeVoice-ASR-Streaming-1.5B-GGUF"},
    "7b": {"source": "microsoft/VibeVoice-ASR-Streaming-7B", "slug": "7b",
           "repo": "audio-cpp/VibeVoice-ASR-Streaming-7B-GGUF"},
}

# Embedded verbatim so the GGUF is standalone. Everything else in the checkout --
# the README, the safetensors index -- would otherwise be swept in by `--root .`,
# since that embeds any file under 64 MB that is not itself weights.
SIDECARS = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "special_tokens_map.json", "added_tokens.json",
            "preprocessor_config.json"]

FAMILY = "vibevoice_asr_streaming"


def find_converter(explicit):
    if explicit:
        return pathlib.Path(explicit)
    root = pathlib.Path(__file__).resolve().parent.parent
    found = sorted(root.glob("build*/bin/audiocpp_gguf")) + sorted(root.glob("build*/*/bin/audiocpp_gguf"))
    if not found:
        sys.exit("no audiocpp_gguf found under build*/; pass --gguf, or build the "
                 "target with: cmake --build build --target audiocpp_gguf")
    return found[0]


def fetch_source(source, cache):
    """A local directory is used as-is; anything else is a repo id to snapshot."""
    local = pathlib.Path(source)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is needed to download the source weights: "
                 "pip install huggingface_hub (or pass a local --source directory)")
    print(f"downloading {source} ...")
    return pathlib.Path(snapshot_download(repo_id=source, cache_dir=cache))


def stage_sidecars(source, staging):
    """Copy the files that belong inside the GGUF, and nothing else."""
    staging.mkdir(parents=True, exist_ok=True)
    staged = []
    for name in SIDECARS:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, staging / name)
            staged.append(name)
    missing = {"config.json", "tokenizer.json"} - set(staged)
    if missing:
        sys.exit(f"source is missing required files: {', '.join(sorted(missing))}")
    return staged


def convert(converter, source, staging, output, precision):
    index = source / "model.safetensors.index.json"
    single = source / "model.safetensors"
    weights = index if index.is_file() else single
    if not weights.is_file():
        sys.exit(f"no safetensors weights found in {source}")
    command = [str(converter), "--input", str(weights), "--output", str(output),
               "--type", precision, "--family", FAMILY, "--root", str(staging), "--overwrite"]
    print("  " + " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # Print the converter's own words. A conversion that fails quietly and
        # leaves a truncated file behind is worse than one that stops here.
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(f"conversion to {precision} failed (exit {result.returncode})")
    for line in result.stdout.splitlines():
        if line.startswith(("embedded_sidecar_count=", "embedded_model_spec=", "weight_type=")):
            print("    " + line)


def verify(cli, output, assets, backend):
    """Score the package rather than just checking the file exists.

    ⚠ The backend is explicit because it changes the answer. CPU and CUDA quantize
    activations differently upstream (Q8_K/Q8_0 against Q8_1), so a quantized package
    scores differently on each, and an unlabelled WER number is not interpretable.
    """
    harness = pathlib.Path(__file__).resolve().parent / "asr_wer.py"
    command = [sys.executable, str(harness), "--cli", str(cli), "--family", FAMILY,
               "--model", str(output), "--backend", backend,
               "--label", f"{output.stem} [{backend}]"]
    if assets:
        command += ["--assets", assets]
    result = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write("".join("    " + line + "\n" for line in result.stdout.strip().splitlines()))
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(f"verification of {output.name} failed (exit {result.returncode})")


def upload(repo, files):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is needed to upload: pip install huggingface_hub")
    api = HfApi()
    print(f"creating {repo} if it does not exist ...")
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    for path in files:
        print(f"uploading {path.name} ({path.stat().st_size / 1e9:.2f} GB) ...")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo, repo_type="model")
    print(f"done: https://huggingface.co/{repo}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument("--size", choices=sorted(SIZES), default="1.5b")
    parser.add_argument("--source", help="Local weights directory, or a HF repo id. "
                                         "Defaults to the upstream repo for --size.")
    parser.add_argument("--out", default="out", help="Directory for the built GGUFs.")
    parser.add_argument("--precision", action="append", choices=["bf16", "q8_0", "q4_k"],
                        help="Repeatable. Defaults to all three.")
    parser.add_argument("--gguf", help="Path to audiocpp_gguf; searched under build*/ if omitted.")
    parser.add_argument("--cache", help="Hugging Face download cache directory.")
    parser.add_argument("--verify-with", metavar="AUDIOCPP_CLI",
                        help="Score each built package with tools/asr_wer.py using this CLI.")
    parser.add_argument("--verify-backend", default="cuda", choices=["cpu", "cuda", "metal", "vulkan"],
                        help="Backend for --verify-with. Quantized packages score differently per "
                             "backend, so this is recorded in the label.")
    parser.add_argument("--assets", help="Clip directory for --verify-with; defaults to the "
                                         "LibriSpeech clips in assets/asr_validation/librispeech.")
    parser.add_argument("--upload", metavar="REPO", nargs="?", const="",
                        help="Publish the built GGUFs. Bare --upload uses the repo named in "
                             "the model spec for this size.")
    args = parser.parse_args()

    size = SIZES[args.size]
    precisions = args.precision or ["bf16", "q8_0", "q4_k"]
    converter = find_converter(args.gguf)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = fetch_source(args.source or size["source"], args.cache)
    staging = out / "sidecars"
    staged = stage_sidecars(source, staging)
    print(f"source: {source}\nconverter: {converter}\nsidecars: {len(staged)} files")

    built = []
    for precision in precisions:
        output = out / f"vibevoice-asr-streaming-{size['slug']}-{precision}.gguf"
        print(f"\n{precision} -> {output.name}")
        convert(converter, source, staging, output, precision)
        print(f"    {output.stat().st_size / 1e9:.2f} GB")
        if args.verify_with:
            verify(args.verify_with, output, args.assets, args.verify_backend)
        built.append(output)

    shutil.rmtree(staging, ignore_errors=True)

    if args.upload is None:
        print("\nBuilt:")
        for path in built:
            print(f"  {path}")
        print("\nNothing was uploaded. Re-run with --upload to publish.")
        return 0

    upload(args.upload or size["repo"], built)
    return 0


if __name__ == "__main__":
    sys.exit(main())
