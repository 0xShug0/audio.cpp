#!/usr/bin/env python3
"""Convert YuE2 weights into the audio.cpp native GGUF package layout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


SIDECARS = [
    ("YuE2-3B/config.json", "sidecars/yue2-model-config.json"),
    ("YuE2-3B/yue2_generation_config.json", "sidecars/yue2-generation-config.json"),
    ("YuE2-3B/qwen.tiktoken", "sidecars/yue2-qwen.tiktoken"),
    ("YuE2-Vae/config.json", "sidecars/yue2-vae-config.json"),
]


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_sidecars(source: Path, output: Path) -> None:
    for src_rel, dst_rel in SIDECARS:
        src = require_file(source / src_rel)
        dst = output / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Desktop" / "YuE2",
        help="Directory containing YuE2-3B and YuE2-Vae snapshots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "Yue2-3B-GGUF",
        help="Output GGUF package directory.",
    )
    parser.add_argument(
        "--audiocpp-gguf",
        type=Path,
        default=REPO_ROOT / "build" / "debug" / "bin" / "audiocpp_gguf",
        help="Path to the audio.cpp GGUF converter.",
    )
    parser.add_argument(
        "--model-type",
        default="q8_0",
        choices=["orig", "f16", "bf16", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"],
        help="Storage type for the YuE2 AR/NAR model GGUF.",
    )
    parser.add_argument(
        "--vae-type",
        default="f16",
        choices=["orig", "f16", "bf16", "q8_0"],
        help="Storage type for the Oobleck VAE GGUF.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def typed_name(stem: str, storage_type: str) -> str:
    if storage_type == "orig":
        if stem == "yue2-3b":
            return f"{stem}-bf16.gguf"
        if stem == "yue2-vae":
            return f"{stem}-f32.gguf"
    return f"{stem}-{storage_type}.gguf"


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    converter = require_file(args.audiocpp_gguf.resolve())
    model_spec = require_file(REPO_ROOT / "model_specs" / "yue2.json")

    model_weights = require_file(source / "YuE2-3B" / "model.safetensors")
    vae_weights = require_file(source / "YuE2-Vae" / "model.safetensors")
    output.mkdir(parents=True, exist_ok=True)
    copy_sidecars(source, output)
    model_output = output / typed_name("yue2-3b", args.model_type)
    vae_output = output / typed_name("yue2-vae", args.vae_type)

    model_cmd = [
        str(converter),
        "--input",
        f"model_weights={model_weights}",
        "--input",
        f"vae_weights={vae_weights}",
        "--root",
        str(output),
        "--model-spec",
        str(model_spec),
        "--family",
        "yue2",
        "--output",
        str(model_output),
        "--type",
        args.model_type,
        "--no-sidecars",
        "--allow-missing-model-spec",
        "--exclude-prefix",
        "vae_weights/",
    ]
    vae_cmd = [
        str(converter),
        "--input",
        f"model_weights={model_weights}",
        "--input",
        f"vae_weights={vae_weights}",
        "--root",
        str(output),
        "--model-spec",
        str(model_spec),
        "--family",
        "yue2",
        "--output",
        str(vae_output),
        "--type",
        args.vae_type,
        "--no-sidecars",
        "--allow-missing-model-spec",
        "--exclude-prefix",
        "model_weights/",
        "--fold-weight-norm",
        "vae_weights/*",
    ]
    if args.overwrite:
        model_cmd.append("--overwrite")
        vae_cmd.append("--overwrite")

    run(model_cmd)
    run(vae_cmd)
    print("[done]", output)
    print("[entry]", model_output)


if __name__ == "__main__":
    main()
