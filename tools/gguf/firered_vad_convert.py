#!/usr/bin/env python3
"""Convert official FireRed VAD or Stream-VAD checkpoints to self-contained GGUF."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import kaldiio
import numpy as np
import torch
from safetensors.numpy import save_file

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--type", choices=("f32", "f16", "q8_0"), default="f32")
    args = parser.parse_args()
    checkpoint = torch.load(args.source / "model.pth.tar", map_location="cpu", weights_only=False)
    config = vars(checkpoint["args"])
    expected = dict(idim=80, odim=1, R=8, M=1, H=256, P=128, N1=20, S1=1, S2=1)
    if any(config.get(key) != value for key, value in expected.items()) or config.get("N2") not in (0, 20):
        raise ValueError(f"Unsupported FireRed VAD architecture: {config}")
    tensors = {name: np.ascontiguousarray(value.detach().numpy(), dtype=np.float32)
               for name, value in checkpoint["model_state_dict"].items()}
    stats = kaldiio.load_mat(str(args.source / "cmvn.ark"))
    if stats.shape != (2, 81) or stats[0, -1] < 1:
        raise ValueError("Expected 80-dimensional Kaldi CMVN statistics")
    mean = stats[0, :80] / stats[0, -1]
    variance = np.maximum(stats[1, :80] / stats[0, -1] - mean * mean, 1e-20)
    tensors["frontend.cmvn_shift"] = np.asarray(-mean, dtype=np.float32)
    tensors["frontend.cmvn_scale"] = np.asarray(1 / np.sqrt(variance), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="firered-vad-convert-") as temporary:
        root = Path(temporary)
        save_file(tensors, root / "weights.safetensors")
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        spec = json.loads((REPO / "model_specs/firered_vad.json").read_text())
        spec["modes"] = ["offline", "streaming"] if config["N2"] == 0 else ["offline"]
        excluded = {"merge_silence_duration_ms", "speech_pad_ms"} if config["N2"] == 0 else {"speech_start_pad_ms"}
        spec["options"]["request"] = [option for option in spec["options"]["request"] if option["name"] not in excluded]
        (root / "model_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        subprocess.run([str(args.converter.resolve()), "--input", f"weights={root / 'weights.safetensors'}",
                        "--root", str(root),
                        "--output", str(args.output.resolve()), "--type", args.type,
                        "--family", "firered_vad", "--model-spec", str(root / "model_spec.json"),
                        "--keep-type", "weights/frontend.*=f32",
                        "--overwrite"], check=True)
    print(json.dumps({"variant": "Stream-VAD" if config["N2"] == 0 else "VAD",
                      "tensors": len(tensors), "type": args.type, "output": str(args.output)}))


if __name__ == "__main__":
    main()
