#!/usr/bin/env python3
"""Convert MiniMax Music 3 weights into the audio.cpp GGUF package layout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


SIDECARS = [
    ("config.json", "config.json"),
    ("language_model/config.json", "language_model/config.json"),
    ("rvq_depth_decoder/config.json", "rvq_depth_decoder/config.json"),
    ("condition_encoder/config.json", "condition_encoder/config.json"),
    ("transformer/config.json", "transformer/config.json"),
    ("vocoder/config.json", "vocoder/config.json"),
    ("tokenizer/tokenizer.json", "tokenizer/tokenizer.json"),
    ("tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json"),
]


COMPONENTS = [
    ("language_model", "language_model/model.safetensors.index.json", "language_model.gguf"),
    ("rvq_depth_decoder", "rvq_depth_decoder/diffusion_pytorch_model.safetensors", "rvq_depth_decoder.gguf"),
    ("condition_encoder", "condition_encoder/diffusion_pytorch_model.safetensors", "condition_encoder.gguf"),
    ("transformer", "transformer/diffusion_pytorch_model.safetensors.index.json", "transformer.gguf"),
    ("vocoder", "vocoder/diffusion_pytorch_model.safetensors", "vocoder.gguf"),
]

# Conv kernels must not be stored BF16: the CUDA conv lowering corrupts on BF16 kernel
# types (and the runtime guards this at load). Keep them F32 on disk for every target
# type.
KEEP_TYPES = {
    "transformer": ["preprocess_conv*=f32", "postprocess_conv*=f32"],
    "condition_encoder": ["proj.weight=f32"],
    "vocoder": ["dec_in_proj*=f32"],
}

# Config sidecars the runtime requires next to the component GGUFs.
CONFIG_SIDECARS = {
    "language_model": "language_model/config.json",
    "rvq_depth_decoder": "rvq_depth_decoder/config.json",
    "condition_encoder": "condition_encoder/config.json",
    "transformer": "transformer/config.json",
    "vocoder": "vocoder/config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Desktop" / "MiniMax-Music",
        help="MiniMax Music 3 source snapshot directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Desktop" / "MiniMax-Music3-GGUF",
        help="Output GGUF package directory.",
    )
    parser.add_argument(
        "--audiocpp-gguf",
        type=Path,
        default=REPO_ROOT / "build" / "debug" / "bin" / "audiocpp_gguf",
        help="Path to the audio.cpp GGUF conversion binary.",
    )
    parser.add_argument(
        "--type",
        default="orig",
        choices=["orig", "f16", "bf16", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"],
        help="GGUF storage type passed to audiocpp_gguf.",
    )
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated components to convert, or 'all'.",
    )
    parser.add_argument(
        "--suffix-components",
        default="language_model,transformer",
        help="Comma-separated component names that receive _<type> in non-orig outputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GGUF outputs.")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def copy_sidecars(source: Path, output: Path) -> None:
    for src_rel, dst_rel in SIDECARS:
        src = source / src_rel
        dst = output / dst_rel
        require_file(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def selected_components(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.components == "all":
        return COMPONENTS
    wanted = {name.strip() for name in args.components.split(",") if name.strip()}
    known = {name for name, _input, _output in COMPONENTS}
    unknown = wanted - known
    if unknown:
        raise ValueError(f"unknown MiniMax Music3 component(s): {', '.join(sorted(unknown))}")
    return [component for component in COMPONENTS if component[0] in wanted]


def output_name(args: argparse.Namespace, component: tuple[str, str, str]) -> str:
    name, _input_rel, output_rel = component
    suffix_components = {item.strip() for item in args.suffix_components.split(",") if item.strip()}
    if args.type == "orig" or name not in suffix_components:
        return output_rel
    path = Path(output_rel)
    return f"{path.stem}_{args.type}{path.suffix}"


def write_variant_spec(args: argparse.Namespace) -> tempfile.TemporaryDirectory[str]:
    tempdir = tempfile.TemporaryDirectory(prefix="minimax_music3_spec_")
    spec_path = Path(tempdir.name) / "minimax_music3.json"
    with (REPO_ROOT / "model_specs" / "minimax_music3.json").open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    output_by_name = {name: output_name(args, component) for name, _input, _base in COMPONENTS for component in [next(c for c in COMPONENTS if c[0] == name)]}
    gguf_source = next(source for source in spec["sources"] if source["format"] == "gguf")
    gguf_source.setdefault("tensors", {})
    gguf_source["tensors"]["language_model_weights"] = f"model:{output_by_name['language_model']}"
    gguf_source["tensors"]["depth_decoder_weights"] = f"model:{output_by_name['rvq_depth_decoder']}"
    gguf_source["tensors"]["condition_encoder_weights"] = f"model:{output_by_name['condition_encoder']}"
    gguf_source["tensors"]["transformer_weights"] = f"model:{output_by_name['transformer']}"
    gguf_source["tensors"]["vocoder_weights"] = f"model:{output_by_name['vocoder']}"

    with spec_path.open("w", encoding="utf-8") as handle:
        json.dump(spec, handle, indent=2)
        handle.write("\n")
    args.model_spec = spec_path
    return tempdir


def run_conversion(args: argparse.Namespace, component: tuple[str, str, str]) -> None:
    name, input_rel, output_rel = component
    input_path = args.source / input_rel
    output_path = args.output / output_name(args, component)
    require_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.audiocpp_gguf),
        "--input",
        str(input_path),
        "--root",
        str(args.source),
        "--output",
        str(output_path),
        "--type",
        args.type,
        "--family",
        "minimax_music3",
        "--model-spec",
        str(args.model_spec),
        "--no-sidecars",
        # The runtime opens component GGUFs by filename rather than through spec tensor
        # ids, and the spec's gguf source does not yet match audiocpp_gguf's
        # source-matching rules, so conversion proceeds without a matching package spec.
        "--allow-missing-model-spec",
    ]
    for keep in KEEP_TYPES.get(name, []):
        cmd.extend(["--keep-type", keep])
    if args.overwrite:
        cmd.append("--overwrite")
    print("[convert]", name, "->", output_path, flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    sidecar = CONFIG_SIDECARS.get(name)
    if sidecar is not None:
        config_dir = Path(args.output) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(args.source) / sidecar, config_dir / f"{name}.json")


def main() -> None:
    args = parse_args()
    require_file(args.audiocpp_gguf)
    require_file(REPO_ROOT / "model_specs" / "minimax_music3.json")
    args.output.mkdir(parents=True, exist_ok=True)
    copy_sidecars(args.source, args.output)
    with write_variant_spec(args):
        for component in selected_components(args):
            run_conversion(args, component)
    print("[done]", args.output)
    language_model = next(component for component in COMPONENTS if component[0] == "language_model")
    print("[entry]", args.output / output_name(args, language_model))


if __name__ == "__main__":
    main()
