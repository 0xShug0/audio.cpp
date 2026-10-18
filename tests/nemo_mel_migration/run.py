#!/usr/bin/env python3
"""Capture and compare migration tensors from the production NeMo-style frontends."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


CASES = {
    "parakeet_tdt": ("parakeet", ("center", "no_center")),
    "sortformer_diar": ("sortformer_v1", ("",)),
    "sortformer_diar_v2": ("sortformer_v2", ("offline", "stream")),
    "hviske_asr": ("hviske", ("",)),
    "citrinet_asr": ("citrinet", ("",)),
    "canary_asr": ("canary", ("",)),
    "cohere_asr": ("cohere", ("",)),
    "nemotron_asr": ("nemotron", ("center", "no_center")),
    "granite5asr": ("granite", ("",)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_models(path: Path) -> dict[str, Path]:
    paths = json.loads(path.read_text())
    if not isinstance(paths, dict) or set(paths) != set(CASES) or not all(isinstance(value, str) for value in paths.values()):
        raise SystemExit(f"model map must contain exactly these families: {', '.join(CASES)}")
    models = {family: (path.parent / value).resolve() for family, value in paths.items()}
    for family, model in models.items():
        if not model.is_file():
            raise SystemExit(f"missing model for {family}: {model}")
    return models


def model_sha256(case: dict) -> str:
    if "model_sha256" in case:
        return case["model_sha256"]
    # Older local captures recorded the model path instead of its digest.
    if "model" in case:
        return sha256(Path(case["model"]))
    raise SystemExit("capture manifest has no model identity")


def capture(args: argparse.Namespace) -> None:
    models = load_models(args.models_json)
    args.out.mkdir(parents=True, exist_ok=True)
    for family, (prefix, suffixes) in CASES.items():
        model = models[family]
        command = [str(args.probe.resolve()), "--family", family, "--model", str(model),
                   "--audio", str(args.audio), "--out", str(args.out / prefix),
                   "--log", str(args.out / f"{prefix}.log")]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (args.out / f"{prefix}.stdout").write_text(result.stdout)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit(f"{family} probe failed with exit code {result.returncode}")
    index(args)


def index(args: argparse.Namespace) -> None:
    models = load_models(args.models_json)
    manifest = {"audio_sha256": sha256(args.audio), "cases": {}}
    for family, (prefix, suffixes) in CASES.items():
        model = models[family]
        outputs = [prefix + (f".{suffix}" if suffix else "") for suffix in suffixes]
        for name in outputs:
            if not (args.out / f"{name}.meta").is_file():
                raise SystemExit(f"missing feature metadata: {name}")
        manifest["cases"][family] = {
            "model_sha256": sha256(model),
            "outputs": {name: sha256(args.out / f"{name}.f32") for name in outputs},
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def compare(args: argparse.Namespace) -> None:
    old = json.loads((args.before / "manifest.json").read_text())
    new = json.loads((args.after / "manifest.json").read_text())
    if old["audio_sha256"] != new["audio_sha256"]:
        raise SystemExit("audio fixture hash mismatch")
    report = []
    for family, (prefix, suffixes) in CASES.items():
        if model_sha256(old["cases"][family]) != model_sha256(new["cases"][family]):
            raise SystemExit(f"{family} model hash mismatch")
        for suffix in suffixes:
            name = prefix + (f".{suffix}" if suffix else "")
            command = [sys.executable, str(Path(__file__).with_name("compare.py")),
                       str(args.before / name), str(args.after / name),
                       "--max-abs", str(args.max_abs)]
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, check=False)
            report.append(result.stdout.rstrip())
            print(result.stdout, end="")
            if result.returncode:
                raise SystemExit(f"{name} comparison failed")
    (args.after / "comparison.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("capture", "index"):
        cap = sub.add_parser(action)
        if action == "capture":
            cap.add_argument("--probe", type=Path, required=True)
        cap.add_argument("--models-json", type=Path, required=True)
        cap.add_argument("--audio", type=Path, required=True)
        cap.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--before", type=Path, required=True)
    cmp.add_argument("--after", type=Path, required=True)
    cmp.add_argument("--max-abs", type=float, default=0.0)
    args = parser.parse_args()
    if args.action == "capture":
        capture(args)
    elif args.action == "index":
        index(args)
    else:
        compare(args)
