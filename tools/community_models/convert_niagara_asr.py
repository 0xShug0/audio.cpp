#!/usr/bin/env python3
"""Convert ABR Niagara batch ASR PT2 CPU packages to audio.cpp GGUF.

The upstream Niagara repositories publish Torch 2.11 PT2 archives with raw
numbered tensor payloads and a serialized graph description. This converter
stages those tensors under the graph input names, then delegates GGUF writing,
quantization, sidecar embedding, and model-spec embedding to audiocpp_gguf.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "model_specs" / "niagara_asr.json"
REPOS = {
    "19m": {
        "repo": "abr-ai/niagara-19m-batch.en",
        "base": "niagara-19m-batch.en",
        "tokenizer": "sentencepiece_256.model",
        "output": "Niagara-19M-Batch-EN-GGUF/niagara-19m-batch.en.gguf",
    },
    "38m": {
        "repo": "abr-ai/niagara-38m-batch.en",
        "base": "niagara-38m-batch.en",
        "tokenizer": "sentencepiece_1024.model",
        "output": "Niagara-38M-Batch-EN-GGUF/niagara-38m-batch.en.gguf",
    },
}
TORCH_DTYPE = {
    4: torch.int32,
    5: torch.int64,
    7: torch.float32,
}


def tensor_shape(meta: dict[str, Any]) -> list[int]:
    return [int(item["as_int"]) for item in meta.get("sizes", [])]


def load_raw_tensor(archive: zipfile.ZipFile, path: str, dtype_id: int, shape: list[int]) -> torch.Tensor:
    try:
        dtype = TORCH_DTYPE[dtype_id]
    except KeyError as error:
        raise RuntimeError(f"unsupported Niagara PT2 tensor dtype id: {dtype_id}") from error
    data = archive.read(path)
    tensor = torch.frombuffer(bytearray(data), dtype=dtype).clone()
    return tensor.reshape(shape)


def download_if_missing(model_dir: Path, variant: str) -> None:
    spec = REPOS[variant]
    required = [
        f"{spec['base']}-cpu.pt2",
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        spec["tokenizer"],
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if not missing:
        return
    for name in missing:
        print(f"downloading {spec['repo']}:{name}", flush=True)
        hf_hub_download(spec["repo"], name, local_dir=model_dir)


def stage_from_pt2(model_dir: Path, variant: str, staging: Path) -> Path:
    spec = REPOS[variant]
    pt2 = model_dir / f"{spec['base']}-cpu.pt2"
    if not pt2.exists():
        raise FileNotFoundError(f"missing Niagara CPU PT2 package: {pt2}")

    with zipfile.ZipFile(pt2) as archive:
        base = archive.namelist()[0].split("/")[0]
        graph = json.loads(archive.read(f"{base}/models/model.json"))
        constants = json.loads(archive.read(f"{base}/data/constants/model_constants_config.json"))["config"]
        input_specs = graph["graph_module"]["signature"]["input_specs"]
        tensor_values = graph["graph_module"]["graph"]["tensor_values"]

        tensors: dict[str, torch.Tensor] = {}
        parameter_specs = [item["parameter"] for item in input_specs if "parameter" in item]
        for index, parameter in enumerate(parameter_specs):
            name = parameter["arg"]["name"]
            meta = tensor_values[name]
            tensors[name] = load_raw_tensor(
                archive,
                f"{base}/data/weights/weight_{index}",
                int(meta["dtype"]),
                tensor_shape(meta),
            )

        for item in input_specs:
            if "tensor_constant" not in item:
                continue
            constant = item["tensor_constant"]
            name = constant["arg"]["name"]
            info = constants[constant["tensor_constant_name"]]
            meta = info["tensor_meta"]
            tensors[name] = load_raw_tensor(
                archive,
                f"{base}/data/constants/{info['path_name']}",
                int(meta["dtype"]),
                tensor_shape(meta),
            )

    staging.mkdir(parents=True, exist_ok=True)
    weights = staging / "niagara_asr.safetensors"
    save_file(tensors, weights)
    shutil.copyfile(model_dir / "config.json", staging / "config.json")
    shutil.copyfile(model_dir / "preprocessor_config.json", staging / "preprocessor_config.json")
    shutil.copyfile(model_dir / "tokenizer_config.json", staging / "tokenizer_config.json")
    shutil.copyfile(model_dir / spec["tokenizer"], staging / "sentencepiece.model")
    return weights


def run_converter(converter: Path, staged_weights: Path, staging: Path, output: Path, quant_type: str, overwrite: bool) -> None:
    command = [
        str(converter),
        "--input",
        str(staged_weights),
        "--root",
        str(staging),
        "--family",
        "niagara_asr",
        "--model-spec",
        str(SPEC),
        "--type",
        quant_type,
        "--output",
        str(output),
    ]
    if overwrite:
        command.append("--overwrite")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(REPOS), required=True)
    parser.add_argument("--model-dir", type=Path, help="directory containing the Niagara HF files")
    parser.add_argument("--converter", type=Path, required=True, help="path to audiocpp_gguf")
    parser.add_argument("--output", type=Path, help="output GGUF path")
    parser.add_argument("--type", default="orig", choices=["orig", "f16", "bf16", "q8_0", "q4_k", "q5_k", "q6_k"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-staging", type=Path, help="keep staged safetensors and sidecars in this directory")
    parser.add_argument("--no-download", action="store_true", help="require all HF files to already exist locally")
    args = parser.parse_args()

    spec = REPOS[args.variant]
    model_dir = args.model_dir or (Path("models") / spec["base"])
    if not args.no_download:
        download_if_missing(model_dir, args.variant)
    output = args.output or Path(spec["output"])

    if args.keep_staging is not None:
        staging = args.keep_staging
        staged_weights = stage_from_pt2(model_dir, args.variant, staging)
        run_converter(args.converter, staged_weights, staging, output, args.type, args.overwrite)
        return

    with tempfile.TemporaryDirectory(prefix=f"niagara-{args.variant}-") as tmp:
        staging = Path(tmp)
        staged_weights = stage_from_pt2(model_dir, args.variant, staging)
        run_converter(args.converter, staged_weights, staging, output, args.type, args.overwrite)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
