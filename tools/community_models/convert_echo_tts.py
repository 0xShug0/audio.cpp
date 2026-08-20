#!/usr/bin/env python3
"""Convert the published Echo-TTS checkpoint into the GGUF layout audio.cpp loads.

Inputs (download them yourself; this tool does not fetch anything):

    jordand/echo-tts-base : pytorch_model.safetensors, pca_state.safetensors

Usage:

    python3 convert_echo_tts.py \
        --model-dir /path/to/echo-tts-base \
        --outfile Echo-TTS-GGUF/model.gguf \
        --precision orig

Tensors are emitted under three prefixes matching model_specs/echo_tts.json:

    dit_weights/*   the EchoDiT, its text encoder and its speaker encoder
    pca/*           the PCA basis mapping 80-D latents to Fish z_q space
    ae/*            the Fish S1-DAC autoencoder, weight norm folded

The Fish S1-DAC autoencoder IS packaged here, under the codec_weights prefix,
when --fish-dir is supplied. audio.cpp already implements this codec for the
fish_audio family and Echo reuses that implementation -- but not its weights:
fish_audio ships Fish Audio S2 Pro, while Echo is trained against the S1 DAC
(jordand/fish-s1-dac-min). Packaging the S1 weights alongside the DiT keeps the
two independent and guarantees Echo gets the exact autoencoder its PCA basis was
fitted to.

Weight normalisation is folded during conversion. The checkpoint stores it in
two forms -- modern `conv.parametrizations.weight.original0/original1` and legacy
`weight_g`/`weight_v` -- and both reduce to `w = g * v / ||v||`, with the norm
taken over every axis except 0. Note that for ConvTranspose1d axis 0 is the
INPUT channel count, not the output, so g is sized differently there.

The six registered buffers (three causal masks and three RoPE tables, 305 MB of
the 1.87 GB checkpoint) are regenerable and are dropped.

Blockwise-continuation weights (latent_encoder, latent_norm, w{k,v}_latent) are
dropped by default, mirroring inference.py's delete_blockwise_modules=True. That
is 420M of the checkpoint's 2.80B parameters. Pass --keep-blockwise to retain
them; nothing in the port consumes them yet.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import gguf
except ImportError:  # pragma: no cover - guidance path
    gguf = None  # reported in main(), so --help still works without it


# --- architecture -------------------------------------------------------
# Not stored in the checkpoint; these are the EchoDiT constructor arguments in
# inference.py::load_model_from_hf. They are re-emitted as GGUF metadata so the
# C++ loader can cross-check rather than hardcode.
ARCH = "echo_tts"

# GGML_MAX_NAME. gguf.cpp rejects any tensor name of this length or longer, and
# the failure surfaces only at load time as "tensor name N is too long".
GGML_MAX_NAME = 64

# The codec prefix is terse because it has to be. The longest name codec.cpp
# loads is 60 characters
# ("encoder.block.4.block.5.layers.3.attention_layer_scale.gamma"), leaving room
# for a three-character prefix and nothing more. "codec_weights." would push
# 157 of the 455 codec tensors past the limit.
# The namespace separator is "/", not ".". PrefixedTensorSourceView in
# src/framework/assets/tensor_source.cpp matches on `prefix + "/"`, so a
# dot-separated name is simply never routed and the namespace looks empty.
NAMESPACE_SEPARATOR = "/"
DIT_TENSOR_PREFIX = "dit_weights" + NAMESPACE_SEPARATOR
PCA_TENSOR_PREFIX = "pca" + NAMESPACE_SEPARATOR
CODEC_TENSOR_PREFIX = "ae" + NAMESPACE_SEPARATOR

CONFIG: Dict[str, int] = {
    "latent_size": 80,
    "model_size": 2048,
    "num_layers": 24,
    "num_heads": 16,
    "intermediate_size": 5888,
    "text_vocab_size": 256,
    "text_model_size": 1280,
    "text_num_layers": 14,
    "text_num_heads": 10,
    "text_intermediate_size": 3328,
    "speaker_patch_size": 4,
    "speaker_model_size": 1280,
    "speaker_num_layers": 14,
    "speaker_num_heads": 10,
    "speaker_intermediate_size": 3328,
    "timestep_embed_size": 512,
    "adaln_rank": 256,
    "max_sequence_length": 640,
    "max_text_length": 768,
    "max_speaker_latent_length": 6400,
    "ae_downsample_factor": 2048,
    "ae_latent_dim": 1024,
    "sample_rate": 44100,
}
NORM_EPS = 1.0e-5

# Upstream's own filter for the blockwise path, kept verbatim so the two stay
# in sync: inference.py::load_model_from_hf.
# Registered buffers, recomputed at graph build time rather than stored.
CODEC_BUFFER_SUFFIXES = ("causal_mask", "freqs_cis")

BLOCKWISE_PREFIXES = ("latent_encoder.", "latent_norm")
BLOCKWISE_SUBSTRINGS = (".wk_latent", ".wv_latent")

# Tensors that must stay F32 regardless of --precision: norm scales are tiny and
# quantising them costs accuracy for no meaningful saving, and biases likewise.
KEEP_F32_SUFFIXES = (".bias", ".alpha")
KEEP_F32_SUBSTRINGS = (
    "_norm.",
    "norm.weight",
    "q_norm",
    "k_norm",
    # LayerScale and ConvNeXt scales, and the Snake activation's alpha. These
    # are small in count (0.08 MB total) and small in magnitude: LayerScale
    # initialises around 1e-6, below F16's smallest normal of 6.1e-05, and the
    # Snake activation uses alpha's reciprocal, which amplifies any error.
    "gamma",
    ".alpha",
    # Codebook entries are summed to form z_q, which is exactly the latent
    # Echo's PCA basis maps from, so quantising them perturbs the speaker
    # conditioning directly. 0.43 MB to keep exact.
    "codebook",
)


def is_blockwise(name: str) -> bool:
    return name.startswith(BLOCKWISE_PREFIXES) or any(
        token in name for token in BLOCKWISE_SUBSTRINGS
    )


def keep_f32(name: str) -> bool:
    return name.endswith(KEEP_F32_SUFFIXES) or any(
        token in name for token in KEEP_F32_SUBSTRINGS
    )


# --- safetensors reading ------------------------------------------------

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def read_safetensors(path: Path) -> Dict[str, np.ndarray]:
    """Minimal safetensors reader with explicit bfloat16 handling.

    numpy has no bfloat16, and the Echo checkpoint is stored in it, so BF16 is
    widened to float32 by placing the 16 stored bits in the high half of the
    f32 mantissa/exponent. That is exact -- bf16 and f32 share an exponent
    layout -- so nothing is lost on the way in.
    """
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length).decode("utf-8"))
        payload_start = 8 + header_length
        handle.seek(0, 2)
        payload_bytes = handle.tell() - payload_start

        tensors: Dict[str, np.ndarray] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            if end > payload_bytes:
                raise RuntimeError(f"{path.name}: tensor {name} runs past end of file")
            shape = tuple(int(dim) for dim in entry["shape"])
            dtype_name = entry["dtype"]
            handle.seek(payload_start + start)
            raw = handle.read(end - start)

            if dtype_name == "BF16":
                bits = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                array = bits.view(np.float32).reshape(shape)
            elif dtype_name in _DTYPES:
                array = np.frombuffer(raw, dtype=_DTYPES[dtype_name]).reshape(shape)
            else:
                raise RuntimeError(f"{path.name}: unsupported dtype {dtype_name}")
            tensors[name] = np.ascontiguousarray(array)
    return tensors


# --- Fish S1-DAC codec ---------------------------------------------------


def fold_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reconstruct a weight-normalised tensor: w = g * v / ||v||.

    torch's weight_norm(dim=0) normalises over every axis except the first, so
    the reduction axes are the same regardless of layer type. What differs is
    what axis 0 *means*: for Conv1d it is out_channels, for ConvTranspose1d it
    is in_channels. Because the reduction is expressed relative to axis 0 rather
    than to a named channel count, one implementation covers both -- and the
    shapes of g and v carry the distinction for free.
    """
    axes = tuple(range(1, v.ndim))
    norm = np.sqrt(np.sum(v.astype(np.float64) ** 2, axis=axes, keepdims=True))
    if not np.all(norm > 0):
        raise RuntimeError("weight_norm folding hit a zero-norm direction vector")
    return (g.astype(np.float64) * v.astype(np.float64) / norm).astype(np.float32)


def split_fused_qkv(name: str, array: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a fused wqkv projection into the q/k/v codec.cpp loads separately.

    autoencoder.py::Attention keeps one nn.Linear and splits its output into
    three equal kv_size blocks (`wqkv(x).split([kv_size]*3, dim=-1)`), so the
    weight rows partition in the same order. codec.cpp instead loads
    attention.q_proj / k_proj / v_proj, one square matrix each.
    """
    base = name[: -len(".wqkv.weight")]
    rows = array.shape[0]
    if rows % 3 != 0:
        raise RuntimeError(f"{name}: fused qkv has {rows} rows, not divisible by 3")
    size = rows // 3
    if array.shape[1] != size:
        raise RuntimeError(
            f"{name}: expected square projections, got {array.shape} -> {size}")
    return {
        f"{base}.q_proj.weight": np.ascontiguousarray(array[:size]),
        f"{base}.k_proj.weight": np.ascontiguousarray(array[size:2 * size]),
        f"{base}.v_proj.weight": np.ascontiguousarray(array[2 * size:]),
    }


def resolve_codec_tensors(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Fold weight norm, split fused qkv, and drop buffers, leaving the names
    codec.cpp expects."""
    out: Dict[str, np.ndarray] = {}
    dropped_buffers = 0
    folded = 0
    split = 0
    for name, array in raw.items():
        if name.endswith(CODEC_BUFFER_SUFFIXES):
            dropped_buffers += 1
            continue
        if name.endswith(".parametrizations.weight.original1"):
            base = name[: -len(".parametrizations.weight.original1")]
            g = raw[base + ".parametrizations.weight.original0"]
            out[base + ".weight"] = fold_weight_norm(g, array)
            folded += 1
            continue
        if name.endswith(".parametrizations.weight.original0"):
            continue
        if name.endswith(".weight_v"):
            base = name[: -len(".weight_v")]
            out[base + ".weight"] = fold_weight_norm(raw[base + ".weight_g"], array)
            folded += 1
            continue
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".wqkv.weight"):
            out.update(split_fused_qkv(name, array))
            split += 1
            continue
        out[name] = array
    print(f"  folded {folded} weight-normalised tensors, split {split} fused qkv "
          f"projections, dropped {dropped_buffers} buffers")
    return out


# --- expected tensor manifest ------------------------------------------
# PyTorch state_dict keys follow the nn.Module attribute path, so this manifest
# is derived directly from model.py. It is checked against the checkpoint at
# convert time: a mismatch aborts rather than silently dropping weights.


def encoder_block_tensors(prefix: str, dim: int, ff: int, heads: int) -> List[Tuple[str, Tuple[int, ...]]]:
    head_dim = dim // heads
    out: List[Tuple[str, Tuple[int, ...]]] = []
    for name in ("wq", "wk", "wv", "wo", "gate"):
        out.append((f"{prefix}.attention.{name}.weight", (dim, dim)))
    out.append((f"{prefix}.attention.q_norm.weight", (heads, head_dim)))
    out.append((f"{prefix}.attention.k_norm.weight", (heads, head_dim)))
    out.append((f"{prefix}.mlp.w1.weight", (ff, dim)))
    out.append((f"{prefix}.mlp.w3.weight", (ff, dim)))
    out.append((f"{prefix}.mlp.w2.weight", (dim, ff)))
    out.append((f"{prefix}.attention_norm.weight", (dim,)))
    out.append((f"{prefix}.mlp_norm.weight", (dim,)))
    return out


def expected_tensors(keep_blockwise: bool) -> Dict[str, Tuple[int, ...]]:
    c = CONFIG
    D, TD, SD = c["model_size"], c["text_model_size"], c["speaker_model_size"]
    L, RANK = c["latent_size"], c["adaln_rank"]
    expected: Dict[str, Tuple[int, ...]] = {}

    expected["text_encoder.text_embedding.weight"] = (c["text_vocab_size"], TD)
    for i in range(c["text_num_layers"]):
        for name, shape in encoder_block_tensors(
            f"text_encoder.blocks.{i}", TD, c["text_intermediate_size"], c["text_num_heads"]
        ):
            expected[name] = shape

    speaker_stacks = ["speaker_encoder"] + (["latent_encoder"] if keep_blockwise else [])
    for stack in speaker_stacks:
        expected[f"{stack}.in_proj.weight"] = (SD, L * c["speaker_patch_size"])
        expected[f"{stack}.in_proj.bias"] = (SD,)
        for i in range(c["speaker_num_layers"]):
            for name, shape in encoder_block_tensors(
                f"{stack}.blocks.{i}", SD, c["speaker_intermediate_size"], c["speaker_num_heads"]
            ):
                expected[name] = shape

    expected["text_norm.weight"] = (TD,)
    expected["speaker_norm.weight"] = (SD,)
    if keep_blockwise:
        expected["latent_norm.weight"] = (SD,)

    # nn.Sequential(Linear, SiLU, Linear, SiLU, Linear) -> indices 0, 2, 4.
    expected["cond_module.0.weight"] = (D, c["timestep_embed_size"])
    expected["cond_module.2.weight"] = (D, D)
    expected["cond_module.4.weight"] = (D * 3, D)

    expected["in_proj.weight"] = (D, L)
    expected["in_proj.bias"] = (D,)

    head_dim = D // c["num_heads"]
    for i in range(c["num_layers"]):
        p = f"blocks.{i}.attention"
        for name in ("wq", "wk", "wv", "gate", "wo"):
            expected[f"{p}.{name}.weight"] = (D, D)
        expected[f"{p}.wk_text.weight"] = (D, TD)
        expected[f"{p}.wv_text.weight"] = (D, TD)
        expected[f"{p}.wk_speaker.weight"] = (D, SD)
        expected[f"{p}.wv_speaker.weight"] = (D, SD)
        if keep_blockwise:
            expected[f"{p}.wk_latent.weight"] = (D, SD)
            expected[f"{p}.wv_latent.weight"] = (D, SD)
        expected[f"{p}.q_norm.weight"] = (c["num_heads"], head_dim)
        expected[f"{p}.k_norm.weight"] = (c["num_heads"], head_dim)

        expected[f"blocks.{i}.mlp.w1.weight"] = (c["intermediate_size"], D)
        expected[f"blocks.{i}.mlp.w3.weight"] = (c["intermediate_size"], D)
        expected[f"blocks.{i}.mlp.w2.weight"] = (D, c["intermediate_size"])

        for adaln in ("attention_adaln", "mlp_adaln"):
            a = f"blocks.{i}.{adaln}"
            for field in ("shift", "scale", "gate"):
                expected[f"{a}.{field}_down.weight"] = (RANK, D)
                expected[f"{a}.{field}_up.weight"] = (D, RANK)
                expected[f"{a}.{field}_up.bias"] = (D,)

    expected["out_norm.weight"] = (D,)
    expected["out_proj.weight"] = (L, D)
    expected["out_proj.bias"] = (L,)
    return expected


def verify_manifest(
    found: Dict[str, np.ndarray], keep_blockwise: bool, strict: bool
) -> None:
    expected = expected_tensors(keep_blockwise)
    kept = {k: v for k, v in found.items() if keep_blockwise or not is_blockwise(k)}

    missing = sorted(set(expected) - set(kept))
    unexpected = sorted(set(kept) - set(expected))
    mismatched = [
        (name, expected[name], kept[name].shape)
        for name in sorted(set(expected) & set(kept))
        if tuple(kept[name].shape) != expected[name]
    ]

    for name in missing:
        print(f"  MISSING   {name} {expected[name]}", file=sys.stderr)
    for name in unexpected:
        print(f"  UNEXPECTED {name} {tuple(kept[name].shape)}", file=sys.stderr)
    for name, want, got in mismatched:
        print(f"  SHAPE     {name}: expected {want}, got {got}", file=sys.stderr)

    if missing or mismatched or (unexpected and strict):
        raise RuntimeError(
            "checkpoint does not match the expected Echo-TTS manifest; "
            "the architecture may have changed upstream"
        )


# --- conversion ---------------------------------------------------------


# Q8_0 stores 32 weights per block with one shared F16 scale, so a tensor is
# only quantisable when its fastest-varying axis is a multiple of 32. In GGUF
# that axis is the LAST logical dimension.
Q8_0_BLOCK = 32


def q8_0_eligible(name: str, array: np.ndarray) -> bool:
    if keep_f32(name) or array.ndim < 2:
        return False
    # 3-D tensors here are convolution kernels. ggml_conv_1d has no quantised
    # path, which is why codec.cpp takes matmul and conv storage types
    # separately; quantising them would fail at graph build, not at load.
    if array.ndim > 2:
        return False
    return array.shape[-1] % Q8_0_BLOCK == 0


def resolve_dtype(name: str, array: np.ndarray, precision: str):
    if precision == "f32" or keep_f32(name) or array.ndim < 2:
        return gguf.GGMLQuantizationType.F32, array.astype(np.float32)
    if precision in ("orig", "f16"):
        return gguf.GGMLQuantizationType.F16, array.astype(np.float16)
    if precision == "q8_0":
        if not q8_0_eligible(name, array):
            return gguf.GGMLQuantizationType.F16, array.astype(np.float16)
        quantised = gguf.quants.quantize(
            np.ascontiguousarray(array, dtype=np.float32),
            gguf.GGMLQuantizationType.Q8_0,
        )
        return gguf.GGMLQuantizationType.Q8_0, quantised
    raise RuntimeError(f"unknown precision {precision}")


def load_model_spec_json(explicit: str | None) -> str:
    """Read the spec that will be embedded, and sanity-check it."""
    if explicit:
        path = Path(explicit)
    else:
        # tools/community_models/convert_echo_tts.py -> model_specs/echo_tts.json
        path = Path(__file__).resolve().parents[2] / "model_specs" / f"{ARCH}.json"
    if not path.is_file():
        raise RuntimeError(
            f"model spec not found at {path}; pass --model-spec explicitly")
    text = path.read_text(encoding="utf-8")
    spec = json.loads(text)
    if spec.get("family") != ARCH:
        raise RuntimeError(
            f"{path} declares family {spec.get('family')!r}, expected {ARCH!r}")
    if spec.get("schema_version") != 1:
        raise RuntimeError(f"{path} is not a schema_version 1 spec")
    return text


def convert(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model_path = model_dir / "pytorch_model.safetensors"
    pca_path = model_dir / "pca_state.safetensors"
    for path in (model_path, pca_path):
        if not path.is_file():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2

    print(f"reading {model_path}")
    weights = read_safetensors(model_path)
    print(f"  {len(weights)} tensors")

    print(f"reading {pca_path}")
    pca = read_safetensors(pca_path)

    for required in ("pca_components", "pca_mean", "latent_scale"):
        if required not in pca:
            print(f"pca_state is missing {required}", file=sys.stderr)
            return 2

    components = pca["pca_components"].astype(np.float32)
    mean = pca["pca_mean"].astype(np.float32)
    scale = float(np.asarray(pca["latent_scale"]).reshape(-1)[0])

    want = (CONFIG["latent_size"], CONFIG["ae_latent_dim"])
    if components.shape != want:
        # ae_encode does `... @ pca_components.T` into 80-D, so the basis must be
        # (80, 1024). Accept the transpose but say so loudly.
        if components.shape == want[::-1]:
            print(
                f"  note: pca_components stored as {components.shape}, transposing to {want}",
                file=sys.stderr,
            )
            components = np.ascontiguousarray(components.T)
        else:
            print(
                f"pca_components has shape {components.shape}, expected {want}",
                file=sys.stderr,
            )
            return 2
    if mean.shape != (CONFIG["ae_latent_dim"],):
        print(f"pca_mean has shape {mean.shape}, expected {(CONFIG['ae_latent_dim'],)}",
              file=sys.stderr)
        return 2

    codec_tensors: Dict[str, np.ndarray] = {}
    if not args.no_codec:
        if not args.fish_dir:
            print(
                "--fish-dir is required (or pass --no-codec).\n"
                "Echo decodes through the Fish S1 DAC and its PCA basis is fitted to "
                "that codec's latent space; audio.cpp's fish_audio package ships S2 Pro, "
                "which is a different model.",
                file=sys.stderr,
            )
            return 2
        fish_path = Path(args.fish_dir) / "pytorch_model.safetensors"
        if not fish_path.is_file():
            print(f"missing required input: {fish_path}", file=sys.stderr)
            return 2
        print(f"reading {fish_path}")
        codec_tensors = resolve_codec_tensors(read_safetensors(fish_path))
        print(f"  {len(codec_tensors)} codec tensors")

    try:
        spec_json = load_model_spec_json(args.model_spec)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"embedding model spec ({len(spec_json)} bytes)")

    print("verifying tensor manifest against model.py")
    verify_manifest(weights, args.keep_blockwise, strict=not args.allow_extra)
    print("  manifest OK")

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(outfile), ARCH)

    over_limit: List[str] = []
    # ggml_n_dims() ignores trailing dimensions of size 1, so a (1, C, 1) snake
    # alpha reads back as (C, 1) and fails codec.cpp's {1, C, 1} shape check.
    # audio.cpp preserves exact logical shapes through two parallel arrays, in
    # tensor order: audiocpp.tensor_ranks (INT32) and the concatenated
    # audiocpp.tensor_shapes (INT64). Both must be present or neither.
    tensor_ranks: List[int] = []
    tensor_shapes: List[int] = []

    def emit(name: str, data, dtype, logical_shape) -> None:
        # Checked here rather than trusted, because ggml only reports this at
        # load time and the message does not say which tensor is at fault.
        if len(name) >= GGML_MAX_NAME:
            over_limit.append(name)
        # audiocpp.tensor_shapes records the LOGICAL shape; a quantised payload
        # arrives packed, so it has to come from the source array. gguf's own
        # raw_shape, by contrast, wants the packed byte shape and derives the
        # logical one itself, so it is left to default.
        tensor_ranks.append(len(logical_shape))
        tensor_shapes.extend(int(dim) for dim in logical_shape)
        writer.add_tensor(name, data, raw_dtype=dtype)

    for key, value in CONFIG.items():
        writer.add_uint32(f"{ARCH}.{key}", int(value))
    writer.add_float32(f"{ARCH}.norm_eps", NORM_EPS)
    writer.add_float32(f"{ARCH}.pca_latent_scale", scale)
    writer.add_bool(f"{ARCH}.has_blockwise_modules", bool(args.keep_blockwise))
    writer.add_bool(f"{ARCH}.has_codec_weights", not args.no_codec)
    writer.add_string(f"{ARCH}.source_precision", args.precision)
    writer.add_string("general.license", "cc-by-nc-sa-4.0")
    writer.add_string("general.name", "Echo-TTS")

    # A published GGUF must carry its own model spec: package.cpp refuses to load
    # one that does not, so that a distributed file is self-describing and does
    # not depend on a matching model_specs/ checkout.
    writer.add_uint32("audiocpp.model_spec.version", 1)
    writer.add_string("audiocpp.model_spec.family", ARCH)
    writer.add_string("audiocpp.model_spec.json", spec_json)

    dropped = 0
    written = 0
    total_bytes = 0
    for name in sorted(weights):
        if not args.keep_blockwise and is_blockwise(name):
            dropped += 1
            continue
        array = weights[name]
        dtype, data = resolve_dtype(name, array, args.precision)
        emit(f"{DIT_TENSOR_PREFIX}{name}", data, dtype, array.shape)
        written += 1
        total_bytes += data.nbytes

    emit(f"{PCA_TENSOR_PREFIX}components", components, gguf.GGMLQuantizationType.F32, components.shape)
    emit(f"{PCA_TENSOR_PREFIX}mean", mean, gguf.GGMLQuantizationType.F32, mean.shape)

    codec_bytes = 0
    for name in sorted(codec_tensors):
        array = codec_tensors[name]
        dtype, data = resolve_dtype(name, array, args.precision)
        emit(f"{CODEC_TENSOR_PREFIX}{name}", data, dtype, array.shape)
        codec_bytes += data.nbytes

    writer.add_key_value(
        "audiocpp.tensor_ranks",
        tensor_ranks,
        gguf.GGUFValueType.ARRAY,
        sub_type=gguf.GGUFValueType.INT32,
    )
    writer.add_key_value(
        "audiocpp.tensor_shapes",
        tensor_shapes,
        gguf.GGUFValueType.ARRAY,
        sub_type=gguf.GGUFValueType.INT64,
    )

    if over_limit:
        longest = max(over_limit, key=len)
        print(
            f"{len(over_limit)} tensor names reach or exceed GGML_MAX_NAME "
            f"({GGML_MAX_NAME}); longest is {len(longest)} chars:\n  {longest}\n"
            "Shorten a tensor prefix; the GGUF would fail to load.",
            file=sys.stderr,
        )
        writer.close()
        return 2

    print(f"writing {outfile}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"  {written} DiT tensors written, {dropped} blockwise tensors dropped")
    if codec_tensors:
        print(f"  {len(codec_tensors)} codec tensors written ({codec_bytes / 1e9:.2f} GB)")
        total_bytes += codec_bytes
    print(f"  approx tensor payload: {total_bytes / 1e9:.2f} GB")
    print(f"  pca latent_scale: {scale}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True,
                        help="directory holding pytorch_model.safetensors and pca_state.safetensors")
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--precision", default="orig", choices=("orig", "f16", "f32", "q8_0"),
                        help="orig and f16 both emit F16 matmul weights; the checkpoint "
                             "ships bf16, which has no GGUF matmul equivalent")
    parser.add_argument("--fish-dir",
                        help="directory holding the Fish S1-DAC checkpoint "
                             "(jordand/fish-s1-dac-min/pytorch_model.safetensors). "
                             "Required unless --no-codec is passed.")
    parser.add_argument("--no-codec", action="store_true",
                        help="omit the autoencoder; the resulting GGUF cannot synthesise")
    parser.add_argument("--keep-blockwise", action="store_true",
                        help="retain latent_encoder / w{k,v}_latent (+840 MB, unused today)")
    parser.add_argument("--model-spec",
                        help="path to model_specs/echo_tts.json to embed. Defaults to the "
                             "copy alongside this script's checkout.")
    parser.add_argument("--allow-extra", action="store_true",
                        help="tolerate checkpoint tensors absent from the expected manifest")
    args = parser.parse_args()
    if gguf is None:
        print("the 'gguf' package is required (pip install gguf)", file=sys.stderr)
        return 2
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
