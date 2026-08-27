#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
convert_soprano.py -- Convert Soprano-1.1-80M for audio.cpp's ``soprano_tts`` family.

The HF checkpoint ships two artifacts:
  * ``model.safetensors``   - the Qwen3-style causal LM (no weight norm).
  * ``decoder.pth``         - a PyTorch ``SopranoDecoder`` that applies
                              ``torch.nn.utils.weight_norm`` on its conv/linear
                              weights. audio.cpp cannot evaluate weight-norm at
                              load time, so this script *folds* it and emits a
                              plain ``decoder.safetensors``.

Output layout (matches ``model_specs/soprano_tts.json`` ``sources``):

  <out>/
    config.json               (passthrough)
    generation_config.json    (generated)
    tokenizer.json            (passthrough)
    model.safetensors         (passthrough)
    decoder.safetensors       (weight-norm folded)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# Weight-norm folding
# --------------------------------------------------------------------------- #
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def load_pt_checkpoint(path: Path):
    """Load ``decoder.pth`` state dict (requires torch to read the pickle)."""
    if _torch_available():
        import torch
        return torch.load(path, map_location="cpu", weights_only=True)
    raise RuntimeError(
        "Soprano conversion requires torch to read decoder.pth (weight-norm "
        "state). Install torch or convert on a host that has it."
    )


def fold_weight_norm(state: dict) -> dict:
    """Fold ``torch.nn.utils.weight_norm`` ``weight_g`` / ``weight_v`` pairs.

    Emits the folded ``weight`` and removes the ``weight_g``/``weight_v`` keys,
    matching AudioCpp's assumption of plain weight tensors.
    """
    folded = dict(state)
    to_fold = []
    for key in list(state.keys()):
        if not key.endswith(".weight_g"):
            continue
        base = key[: -len(".weight_g")]
        v_key = base + ".weight_v"
        g_key = base + ".weight_g"
        if v_key not in state:
            continue
        g = state[g_key]
        v = state[v_key]
        if hasattr(v, "is_cuda") and v.is_cuda:
            g = g.to("cpu")
            v = v.to("cpu")
        norm = (v * v).sum(dim=list(range(1, v.dim())), keepdim=True).sqrt()
        norm = norm.clamp_min(1e-12)
        folded_w = (v / norm) * g
        to_fold.append((base + ".weight", g_key, v_key, folded_w))
    for weight_key, g_key, v_key, w in to_fold:
        folded[weight_key] = w.detach().float()
        del folded[g_key]
        del folded[v_key]
    return folded


# ---------------------------------------------------------------------------- #
# Decoder tensor renaming (Soprano nn.Module -> audio.cpp binding keys)
# ---------------------------------------------------------------------------- #
def rename_decoder_keys(state: dict) -> dict:
    """Map SopranoDecoder state names onto the keys the audio.cpp decoder
    loader expects (the same prefixes used by the `vevo2` Vocoder):

        decoder.embed                Conv1d(512 -> 768, k=1, pad 0, bias=true)
        decoder.norm                 LayerNorm(768)
        decoder.convnext.<i>.dwconv  DepthwiseConv1d(k=3, groups=768, bias=false)
        decoder.convnext.<i>.norm    LayerNorm(768)
        decoder.convnext.<i>.pwconv1 Linear(768 -> 2304, bias=true)
        decoder.convnext.<i>.pwconv2 Linear(2304 -> 768, bias=true)
        decoder.convnext.<i>.gamma   layer-scale, [768]
        decoder.final_layer_norm     LayerNorm(768)
        decoder.head.out             Linear(768 -> n_fft+2, bias=true)
    """
    out = {}
    for key, value in state.items():
        new = _map_key(key)
        if new:
            out[new] = value
    return out


def _map_key(key: str):
    """Map a Soprano state-dict key to the audio.cpp 'decoder.' namespace.

    Soprano names: ``decoder.embed.weight``, ``decoder.norm.weight``,
    ``decoder.convnext.N.dwconv.weight``, ``decoder.convnext.N.gamma``,
    ``decoder.final_layer_norm.weight``, ``decoder.head.out.weight``.
    """
    parts = key.split(".")
    # Drop a redundant `decoder.` / `model.` front prefix if present.
    while len(parts) >= 2 and parts[0] in ("decoder", "model") and parts[1] in ("decoder",):
        parts = parts[2:]
    key = ".".join(parts)
    if key.startswith("decoder."):
        return key
    return "decoder." + key


# ---------------------------------------------------------------------------- #
# Safetensors writer (torch-free)
# ---------------------------------------------------------------------------- #
def _write_safetensors(tensors: dict, path: Path) -> None:
    import numpy as np
    header = {}
    data = bytearray()
    for name, tensor in tensors.items():
        if tensor.dtype in (torch_dtype_f16(),):
            arr = np.asarray(tensor.detach().cpu()).view("int16") if False else None
            raw = tensor.detach().float().numpy().astype("<f4").flatten().tobytes()
            dtype = "F32"
        else:
            arr = tensor.detach().float().cpu().numpy()
            dtype = "F32"
            raw = arr.astype("<f4").flatten().tobytes()
        shape = list(tensor.shape)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    pad = (8 - (len(header_bytes) + 8) % 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(8, "little"))
        fh.write(header_bytes)
        fh.write(bytes(data))


def torch_dtype_f16():
    import torch
    return torch.float16


# ---------------------------------------------------------------------------- #
# main
# ---------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, help="HF Soprano-1.1-80M directory")
    ap.add_argument("--output-dir", "--out", required=True)
    args = ap.parse_args()

    src = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for p in ("config.json", "tokenizer.json", "model.safetensors", "decoder.pth"):
        if not (src / p).exists():
            raise SystemExit(f"missing input file: {src / p}")

    # Pass-through sidecars.
    for name in ("config.json", "tokenizer.json"):
        (out / name).write_bytes((src / name).read_bytes())
    (out / "generation_config.json").write_text(
        json.dumps({"max_new_tokens": 512, "bos_token_id": 3, "eos_token_id": 3},
                   indent=2) + "\n", encoding="utf-8")

    # LM safetensors is passed through byte-for-byte.
    (out / "model.safetensors").write_bytes((src / "model.safetensors").read_bytes())

    # Fold + split decoder.
    state = load_pt_checkpoint(src / "decoder.pth")
    state = fold_weight_norm(state)
    renamed = rename_decoder_keys(state)
    if not renamed:
        raise SystemExit("decoder.pth contained no recognised tensors")
    _write_safetensors(renamed, out / "decoder.safetensors")

    print(f"[convert_soprano] wrote {out}")
    print(f"[convert_soprano] decoder tensors: {len(renamed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())