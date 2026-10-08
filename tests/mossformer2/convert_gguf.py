#!/usr/bin/env python3
"""Package the official ClearVoice MossFormer2 separation checkpoint as GGUF."""

import argparse
import json
import logging
from pathlib import Path
import subprocess
import tempfile

import gguf
import numpy as np
import torch
import yaml
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--type", choices=("f32", "f16"), default="f32")
    parser.add_argument("--audiocpp-gguf", type=Path, default=ROOT / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    source_config = yaml.safe_load(args.config.read_text())
    if source_config["network"] != "MossFormer2_SS_16K":
        raise ValueError("Expected the MossFormer2 speech separation configuration")
    config = {"model_type": "mossformer2"}
    for key in ("sampling_rate", "num_spks", "encoder_kernel_size", "encoder_embedding_dim",
                "mossformer_sequence_dim", "num_mossformer_layer"):
        config[key] = source_config[key]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key: value.clone().contiguous() for key, value in checkpoint["model"].items()}
    del checkpoint
    if any(value.dtype != torch.float32 for value in state.values()):
        raise ValueError("Expected original F32 checkpoint tensors")
    if state["enc.conv1d.weight"].shape != (
            config["encoder_embedding_dim"], 1, config["encoder_kernel_size"]):
        raise ValueError("Encoder configuration does not match checkpoint")
    if state["mask_net.conv1d_out.weight"].shape[0] != config["num_spks"] * config["mossformer_sequence_dim"]:
        raise ValueError("Speaker count does not match checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mossformer2-convert-", dir=args.output.parent) as temp:
        staging = Path(temp)
        save_file(state, staging / "model.safetensors")
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        command = [str(args.audiocpp_gguf.resolve()), "--input", f"weights={staging / 'model.safetensors'}",
                   "--output", str(args.output.resolve()), "--type", "orig" if args.type == "f32" else args.type, "--family", "mossformer2",
                   "--model-spec", str(ROOT / "model_specs/mossformer2.json"), "--root", str(staging)]
        # Affine and positional tensors feed F32 elementwise operations.
        keep_f32 = {key for key, value in state.items()
                    if value.ndim < 2 or ".qk_offset_scale." in key}
        if args.type == "f16":
            for key in sorted(keep_f32):
                command.extend(["--keep-type", f"weights/{key}=f32"])
        if args.overwrite:
            command.append("--overwrite")
        logging.info("command=%s", command)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("%s", completed.stdout)
        completed.check_returncode()
    reader = gguf.GGUFReader(str(args.output))
    expected = {f"weights/{key}": value.numpy() for key, value in state.items()}
    names = reader.fields["audiocpp.tensor_names"].contents()
    if len(names) != len(reader.tensors) or set(names) != set(expected):
        raise ValueError("GGUF tensor names differ from checkpoint")
    for name, tensor in zip(names, reader.tensors):
        original = expected[name]
        if args.type == "f16" and name.removeprefix("weights/") not in keep_f32:
            original = original.astype(np.float16)
        if not np.array_equal(tensor.data.reshape(-1).view(np.uint8), original.reshape(-1).view(np.uint8)):
            raise ValueError(f"GGUF tensor bytes differ: {name}")
    logging.info("Verified %d tensors against source with %s conversion", len(expected), args.type)


if __name__ == "__main__":
    main()
