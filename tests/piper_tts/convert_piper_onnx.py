#!/usr/bin/env python3
"""Convert an official Piper ONNX voice to audio.cpp safetensors assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def semantic_name(name: str, consumers: dict[str, str]) -> str:
    if name == "sid":
        return "enc_p.emb.weight"
    if not name.startswith("onnx::Conv_"):
        return name
    node = consumers.get(name, "")
    if not node.startswith("/flow/flows.") or not node.endswith("/Conv"):
        raise RuntimeError(f"cannot name anonymous initializer {name}: {node}")
    return node.removeprefix("/").removesuffix("/Conv").replace("/", ".") + ".weight"


def main() -> None:
    args = parse_args()
    model = onnx.load(args.onnx, load_external_data=True)
    consumers = {value: node.name for node in model.graph.node for value in node.input}
    tensors: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        value = numpy_helper.to_array(initializer)
        if not np.issubdtype(value.dtype, np.floating):
            continue
        if (
            initializer.name.startswith("/")
            and initializer.name != "/dp/flows.0/Exp_output_0"
        ):
            continue
        name = semantic_name(initializer.name, consumers)
        tensors[name] = np.ascontiguousarray(value.astype(np.float32, copy=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output_dir / "model.safetensors")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["format"] = "piper_tts_inference_config_v1"
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(tensors)} tensors to {args.output_dir}")


if __name__ == "__main__":
    main()
