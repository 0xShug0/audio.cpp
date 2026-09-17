#!/usr/bin/env python3
"""Prepare PulseVAD FP32 checkpoints for the native audio.cpp tensor loader."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    from pulsevad.frontend_np import _HANN_WINDOW, get_mel_filterbank
    from pulsevad.model import PulseVAD
    from pulsevad.quantize import FoldedPulseVAD, fold_batchnorm

    args.output.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    features = torch.randn(32, 64, 21)
    data = args.upstream / "pulsevad" / "data"
    with args.log.open("w") as log, torch.inference_mode():
        revision = subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip()
        print(f"upstream_revision={revision}", file=log, flush=True)
        for variant, filename in (("2.1k", "pulsevad_2.1k.pth"),
                                  ("81k", "pulsevad_teacher_81k.pth")):
            checkpoint = torch.load(data / filename, map_location="cpu", weights_only=True)
            if variant == "2.1k":
                model = FoldedPulseVAD(checkpoint["dims"]).eval()
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                reference = torch.jit.load(str(data / "pulsevad_2.1k.jit")).eval()
                for name, value in model.state_dict().items():
                    if not torch.equal(value, reference.state_dict()[name]):
                        raise ValueError(f"Student checkpoint/JIT mismatch: {name}")
            else:
                reference = PulseVAD().eval()
                reference.load_state_dict(checkpoint["state_dict"], strict=True)
                model = fold_batchnorm(reference)
            expected = reference(features)
            actual = model(features)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            print(f"{variant} folding_max_abs={(actual - expected).abs().max().item():.9g}",
                  file=log, flush=True)
            tensors = {name: value.detach().contiguous() for name, value in model.state_dict().items()}
            tensors["frontend.mel_filterbank"] = torch.from_numpy(
                np.ascontiguousarray(get_mel_filterbank().T))
            tensors["frontend.window"] = torch.from_numpy(_HANN_WINDOW.copy())
            path = args.output / variant / "pulsevad-f32.safetensors"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(path), metadata={"upstream_revision": revision, "family": "pulsevad"})
            restored = load_file(str(path))
            for name, value in tensors.items():
                if not torch.equal(value, restored[name]):
                    raise ValueError(f"Safetensors round-trip mismatch: {name}")
            print(f"{variant} safetensors_exact=true tensors={len(tensors)} path={path}",
                  file=log, flush=True)


if __name__ == "__main__":
    main()
