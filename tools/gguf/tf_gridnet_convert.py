#!/usr/bin/env python3
"""Package an ESPnet TF-GridNet checkpoint and its training YAML as GGUF."""

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
    parser.add_argument("--audiocpp-gguf", type=Path, default=ROOT / "build/debug/bin/audiocpp_gguf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    training = yaml.safe_load(args.config.read_text())
    if training["separator"] != "tfgridnet":
        raise ValueError("Expected an ESPnet tfgridnet checkpoint")
    config = dict(training["separator_conf"])
    config.update(model_type="tf_gridnet", sample_rate=training["sample_rate"])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if any(not key.startswith("separator.") for key in state):
        raise ValueError("Checkpoint contains weights outside the TF-GridNet separator")
    state = {key.removeprefix("separator."): value.contiguous() for key, value in state.items()}
    if state["deconv.weight"].shape[1] != config["n_srcs"] * 2:
        raise ValueError("Output weights disagree with separator_conf.n_srcs")
    windows = {"hann": torch.hann_window, "hanning": torch.hann_window, "hamming": torch.hamming_window}
    window = (torch.ones(config["n_fft"]) if config["window"] is None
              else windows[config["window"]](config["n_fft"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tf-gridnet-convert-") as temp:
        staging = Path(temp)
        save_file(state, staging / "model.safetensors")
        save_file({"window": window}, staging / "frontend.safetensors")
        (staging / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        command = [str(args.audiocpp_gguf.resolve()),
                   "--input", f"weights={staging / 'model.safetensors'}",
                   "--input", f"frontend={staging / 'frontend.safetensors'}",
                   "--output", str(args.output.resolve()), "--type", "orig",
                   "--family", "tf_gridnet", "--model-spec", str(ROOT / "model_specs/tf_gridnet.json"),
                   "--root", str(staging)]
        if args.overwrite:
            command.append("--overwrite")
        logging.info("command=%s", command)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logging.info("%s", completed.stdout)
        completed.check_returncode()
    expected = {f"weights/{key}": value.numpy() for key, value in state.items()}
    expected["frontend/window"] = window.numpy()
    reader = gguf.GGUFReader(str(args.output))
    if {tensor.name for tensor in reader.tensors} != set(expected):
        raise ValueError("GGUF tensor names differ from checkpoint")
    for tensor in reader.tensors:
        original = expected[tensor.name]
        if not np.array_equal(tensor.data.reshape(-1).view(np.uint8), original.reshape(-1).view(np.uint8)):
            raise ValueError(f"GGUF tensor bytes differ: {tensor.name}")
    logging.info("Verified %d tensors byte-exact; speakers=%d", len(expected), config["n_srcs"])


if __name__ == "__main__":
    main()
