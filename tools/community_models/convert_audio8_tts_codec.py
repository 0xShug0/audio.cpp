#!/usr/bin/env python3
"""Convert the Audio8 TTS Preview codec.pth checkpoint to safetensors.

The official checkpoint stores raw PyTorch module state-dict entries where the
encoder/decoder causal convolutions carry new-style weight-norm
parametrizations (`*.parametrizations.weight.original{0,1}`) and the vector
quantizer projections carry legacy weight-norm parameters (`*.weight_g/v`).
audio.cpp binds fused plain weights (`*.conv.weight`, `*.in_proj.weight`,
...), mirroring the forward math of `modeling_arktts_codec.py`
(`weight = g * v / ||v||`, norm over all but the output-channel axis).

The converter is torch-free: it reads the zipfile checkpoint with a stub
unpickler, materializes tensors with numpy, fuses both weight-norm variants,
keeps every other entry verbatim, validates structural anchors from
`modeling_arktts_codec.py`, and writes a plain safetensors file that
`assets.cpp` loads as the `codec_weights` tensor source.

Usage:
    python3 tools/community_models/convert_audio8_tts_codec.py \
        /path/to/codec.pth /path/to/codec.safetensors [--overwrite]
"""

from __future__ import annotations

import argparse
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

# torch storage class name -> numpy dtype
STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}

DTYPE_NAMES = {
    "FloatStorage": "f32",
    "DoubleStorage": "f64",
    "HalfStorage": "f16",
    "BFloat16Storage": "bf16->f32",
    "LongStorage": "i64",
}


class TensorRef:
    """Lazy reference to one tensor inside the checkpoint zip."""

    __slots__ = ("storage_key", "dtype_name", "offset", "size", "stride")

    def __init__(self, storage_key, dtype_name, offset, size, stride):
        self.storage_key = storage_key
        self.dtype_name = dtype_name
        self.offset = offset
        self.size = list(size)
        self.stride = list(stride)


class CheckpointUnpickler(pickle.Unpickler):
    """Stub unpickler capturing state-dict structure without torch."""

    def find_class(self, module, name):
        if module == "torch" and (name in STORAGE_DTYPES or name == "BFloat16Storage"):
            return type(name, (), {"__name__": name})
        if module == "torch._utils":
            if name.startswith("_rebuild_tensor"):
                def rebuild(storage, offset=0, size=None, stride=None, *_rest):
                    storage_type, key = storage[1], storage[2]
                    dtype_name = getattr(storage_type, "__name__", "UnknownStorage")
                    if stride is None:
                        stride = [1] * len(size or [])
                    return TensorRef(key, dtype_name, offset, size, stride)
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *_rest: data
        if module == "collections" and name == "OrderedDict":
            return dict
        return type(name, (), {"__name__": name})

    def persistent_load(self, pid):
        # pid = ("storage", storage_type, key, location, numel); keep whole id
        return ("storage",) + tuple(pid[1:])


def load_state_dict(path: Path) -> dict:
    zf = zipfile.ZipFile(path)
    pkl_names = [n for n in zf.namelist() if n.endswith(".pkl")]
    if len(pkl_names) != 1:
        raise SystemExit(f"expected exactly one data.pkl, found {pkl_names}")
    root = pkl_names[0].rsplit("/", 1)[0]

    with zf.open(pkl_names[0]) as fh:
        obj = CheckpointUnpickler(io.BytesIO(fh.read())).load()
    if not isinstance(obj, dict):
        raise SystemExit("checkpoint pickle did not contain a dict")

    def materialize(ref: TensorRef) -> np.ndarray:
        member = f"{root}/data/{ref.storage_key}"
        if ref.dtype_name == "BFloat16Storage":
            raw = np.frombuffer(zf.read(member), dtype=np.uint16)
            flat = raw[ref.offset : ref.offset + int(np.prod(ref.size))].astype(np.uint32)
            values = (flat << 16).view(np.float32)
        else:
            dtype = STORAGE_DTYPES.get(ref.dtype_name)
            if dtype is None:
                raise SystemExit(f"unsupported storage type: {ref.dtype_name}")
            flat = np.frombuffer(zf.read(member), dtype=dtype)
            span = int(np.prod(ref.size))
            last = ref.offset + sum(
                (s - 1) * st for s, st in zip(ref.size, ref.stride)
            ) + 1
            values = flat[ref.offset : max(last, ref.offset + span)]
        strided = np.lib.stride_tricks.as_strided(
            values,
            shape=tuple(ref.size),
            strides=[s * values.itemsize for s in ref.stride],
        )
        return np.ascontiguousarray(strided)

    return {name: materialize(ref) if isinstance(ref, TensorRef) else ref for name, ref in obj.items()}


def fuse_weight_norm(tensors: dict) -> dict:
    """Fuse g*v/||v|| pairs into plain weights; return a new dict."""
    out: dict = {}
    for name, value in tensors.items():
        if name.endswith(".parametrizations.weight.original0"):
            base = name[: -len(".parametrizations.weight.original0")]
            magnitude, direction = value, tensors[f"{base}.parametrizations.weight.original1"]
            norm = np.sqrt(
                np.sum(direction.astype(np.float32) ** 2, axis=tuple(range(1, direction.ndim)), keepdims=True)
            )
            fused = magnitude.astype(np.float32) * direction.astype(np.float32) / norm
            out[f"{base}.weight"] = fused
        elif name.endswith(".parametrizations.weight.original1"):
            continue
        elif name.endswith(".weight_g"):
            base = name[: -len("_g")]
            magnitude, direction = value, tensors[f"{base}_v"]
            norm = np.sqrt(
                np.sum(direction.astype(np.float32) ** 2, axis=tuple(range(1, direction.ndim)), keepdims=True)
            )
            out[base] = magnitude.astype(np.float32) * direction.astype(np.float32) / norm
        elif name.endswith(".weight_v"):
            continue
        else:
            out[name] = value
    return out


def validate(tensors: dict) -> None:
    def expect(name, shape):
        found = tensors.get(name)
        if found is None:
            raise SystemExit(f"missing anchor tensor: {name}")
        if list(found.shape) != shape:
            raise SystemExit(f"anchor {name} has shape {list(found.shape)}, expected {shape}")

    # modeling_arktts_codec.py structural anchors
    expect("quantizer.semantic_quantizer.quantizers.0.codebook.weight", [4096, 8])
    expect("quantizer.semantic_quantizer.quantizers.0.in_proj.weight", [8, 1024, 1])
    expect("quantizer.semantic_quantizer.quantizers.0.out_proj.weight", [1024, 8, 1])
    for book in range(9):
        expect(f"quantizer.quantizer.quantizers.{book}.codebook.weight", [1024, 8])
    for layer in range(8):
        expect(f"quantizer.pre_module.layers.{layer}.attention.wqkv.weight", [3072, 1024])
        expect(f"quantizer.post_module.layers.{layer}.attention.wqkv.weight", [2048, 1024])
    expect("quantizer.pre_module.norm.weight", [1024])
    expect("decoder.model.0.conv.weight", [1536, 1024, 7])
    expect("encoder.block.0.conv.weight", [64, 1, 7])

    leftovers = [
        n for n in tensors
        if "parametrizations" in n or n.endswith(".weight_g") or n.endswith(".weight_v")
    ]
    if leftovers:
        raise SystemExit(f"unfused weight-norm entries remain: {leftovers[:5]}")

    for name, tensor in tensors.items():
        if np.issubdtype(tensor.dtype, np.floating) and not np.isfinite(tensor.astype(np.float64)).all():
            raise SystemExit(f"non-finite values in {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Audio8 TTS codec.pth")
    parser.add_argument("output", type=Path, help="output codec.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {args.output}")

    print(f"loading {args.input} ...")
    raw = load_state_dict(args.input)
    print(f"  {len(raw)} state-dict entries")

    tensors = fuse_weight_norm(raw)
    fused_pairs = len(tensors) - len(raw)
    validate(tensors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(args.output),
        metadata={
            "format": "pt",
            "source": "Audio8/Audio8-TTS-Preview-0.6b",
            "checkpoint": args.input.name,
        },
    )

    dtypes: dict[str, int] = {}
    labels = {
        "<f4": "f32",
        "<f8": "f64",
        "<f2": "f16",
        "<i8": "i64",
        "<i4": "i32",
        "|u1": "u8",
    }
    for tensor in tensors.values():
        label = labels.get(tensor.dtype.str, str(tensor.dtype))
        dtypes[label] = dtypes.get(label, 0) + 1
    summary = ", ".join(f"{count}x {label}" for label, count in sorted(dtypes.items()))
    print(f"wrote {args.output} ({len(tensors)} tensors, {args.output.stat().st_size >> 20} MiB)")
    print(f"dtypes: {summary}; fused weight-norm pairs net change: {fused_pairs:+d}")


if __name__ == "__main__":
    main()
