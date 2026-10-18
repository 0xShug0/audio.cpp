#!/usr/bin/env python3
"""Verify a converted Echo-TTS GGUF before trying to load it in audio.cpp.

Checks the artifact rather than the conversion process: tensor coverage against
what the C++ loaders ask for, shapes against the architecture, and dtypes
against the precision policy. Catches a bad convert in a second instead of
after a model load and a wasted GPU run.

    python3 verify_echo_gguf.py /path/to/echo-tts.gguf

Exits non-zero if anything is wrong. Requires the `gguf` package.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Set, Tuple

try:
    import gguf
except ImportError:  # pragma: no cover - guidance path
    gguf = None

DIT_PREFIX = "dit_weights/"
PCA_PREFIX = "pca/"
CODEC_PREFIX = "ae/"

# gguf.cpp rejects names at or beyond GGML_MAX_NAME, and only at load time.
GGML_MAX_NAME = 64

# Groups that must stay F32 regardless of --precision. Snake's alpha is used via
# its reciprocal, LayerScale/ConvNeXt gammas initialise below F16's smallest
# normal (6.1e-05), and codebook entries sum to form the z_q that the PCA basis
# maps from.
F32_REQUIRED_MARKERS = ("gamma", ".alpha", "codebook", "norm.weight", "_norm.")


def codec_probe_names() -> List[str]:
    """Names src/framework/codecs/fish_dac_codec_runtime.cpp loads, transcribed from its paths."""
    want: List[str] = ["encoder.block.0.conv.weight", "encoder.block.0.conv.bias"]
    for b in range(1, 5):
        for r in range(3):
            want += [
                f"encoder.block.{b}.block.{r}.block.0.alpha",
                f"encoder.block.{b}.block.{r}.block.1.conv.weight",
                f"encoder.block.{b}.block.{r}.block.2.alpha",
                f"encoder.block.{b}.block.{r}.block.3.conv.weight",
            ]
        want += [
            f"encoder.block.{b}.block.3.alpha",
            f"encoder.block.{b}.block.4.conv.weight",
        ]
    for layer in range(4):
        p = f"encoder.block.4.block.5.layers.{layer}"
        want += [
            f"{p}.attention.q_proj.weight",
            f"{p}.attention.k_proj.weight",
            f"{p}.attention.v_proj.weight",
            f"{p}.attention.wo.weight",
            f"{p}.attention_norm.weight",
            f"{p}.ffn_norm.weight",
            f"{p}.feed_forward.w1.weight",
            f"{p}.feed_forward.w2.weight",
            f"{p}.feed_forward.w3.weight",
            f"{p}.attention_layer_scale.gamma",
            f"{p}.ffn_layer_scale.gamma",
        ]
    want += [
        "encoder.block.4.block.5.norm.weight",
        "encoder.block.5.alpha",
        "encoder.block.6.conv.weight",
    ]
    for stage in ("pre_module", "post_module"):
        for layer in range(8):
            want += [
                f"quantizer.{stage}.layers.{layer}.attention.q_proj.weight",
                f"quantizer.{stage}.layers.{layer}.attention.k_proj.weight",
                f"quantizer.{stage}.layers.{layer}.attention.v_proj.weight",
                f"quantizer.{stage}.layers.{layer}.attention.wo.weight",
            ]
        want += [f"quantizer.{stage}.norm.weight"]
    want += [
        "quantizer.semantic_quantizer.quantizers.0.codebook.weight",
        "quantizer.semantic_quantizer.quantizers.0.in_proj.weight",
        "quantizer.semantic_quantizer.quantizers.0.out_proj.weight",
    ]
    for q in range(9):
        want += [
            f"quantizer.quantizer.quantizers.{q}.codebook.weight",
            f"quantizer.quantizer.quantizers.{q}.in_proj.weight",
            f"quantizer.quantizer.quantizers.{q}.out_proj.weight",
        ]
    for stage in ("downsample", "upsample"):
        for i in range(2):
            want += [
                f"quantizer.{stage}.{i}.0.conv.weight",
                f"quantizer.{stage}.{i}.1.dwconv.conv.weight",
                f"quantizer.{stage}.{i}.1.pwconv1.weight",
                f"quantizer.{stage}.{i}.1.pwconv2.weight",
                f"quantizer.{stage}.{i}.1.norm.weight",
                f"quantizer.{stage}.{i}.1.gamma",
            ]
    want += ["decoder.model.0.conv.weight"]
    for b in range(1, 5):
        want += [
            f"decoder.model.{b}.block.0.alpha",
            f"decoder.model.{b}.block.1.conv.weight",
        ]
        for r in range(3):
            want += [
                f"decoder.model.{b}.block.{r + 2}.block.0.alpha",
                f"decoder.model.{b}.block.{r + 2}.block.1.conv.weight",
            ]
    want += ["decoder.model.5.alpha", "decoder.model.6.conv.weight"]
    return want


def dit_probe_names(config: Dict[str, int]) -> List[str]:
    want = ["text_encoder.text_embedding.weight", "in_proj.weight", "in_proj.bias"]
    want += ["cond_module.0.weight", "cond_module.2.weight", "cond_module.4.weight"]
    want += ["text_norm.weight", "speaker_norm.weight", "out_norm.weight"]
    want += ["out_proj.weight", "out_proj.bias"]
    want += ["speaker_encoder.in_proj.weight", "speaker_encoder.in_proj.bias"]
    for i in range(config["text_num_layers"]):
        want += [
            f"text_encoder.blocks.{i}.attention.wq.weight",
            f"text_encoder.blocks.{i}.attention.q_norm.weight",
            f"text_encoder.blocks.{i}.mlp.w1.weight",
            f"text_encoder.blocks.{i}.attention_norm.weight",
        ]
    for i in range(config["speaker_num_layers"]):
        want += [f"speaker_encoder.blocks.{i}.attention.wq.weight"]
    for i in range(config["num_layers"]):
        want += [
            f"blocks.{i}.attention.wq.weight",
            f"blocks.{i}.attention.wk_text.weight",
            f"blocks.{i}.attention.wv_speaker.weight",
            f"blocks.{i}.attention.q_norm.weight",
            f"blocks.{i}.mlp.w2.weight",
            f"blocks.{i}.attention_adaln.shift_down.weight",
            f"blocks.{i}.attention_adaln.shift_up.bias",
            f"blocks.{i}.mlp_adaln.gate_up.weight",
        ]
    return want


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", help="path to the converted Echo-TTS GGUF")
    args = parser.parse_args()

    if gguf is None:
        print("the 'gguf' package is required (pip install gguf)", file=sys.stderr)
        return 2

    try:
        reader = gguf.GGUFReader(args.gguf)
        tensors = {t.name: t for t in reader.tensors}
        fields = reader.fields
    except Exception as error:  # noqa: BLE001 - any parse failure is a bad file
        print(f"could not read {args.gguf} as a GGUF: {error}", file=sys.stderr)
        print("the file may be truncated or still being written", file=sys.stderr)
        return 1

    def kv(name):
        field = fields.get(name)
        if field is None:
            return None
        try:
            return field.parts[field.data[0]][0]
        except Exception:  # noqa: BLE001 - metadata shape varies by writer
            return None

    problems: List[str] = []
    too_long = [n for n in tensors if len(n) >= GGML_MAX_NAME]
    if too_long:
        longest = max(too_long, key=len)
        problems.append(
            f"{len(too_long)} tensor names reach GGML_MAX_NAME ({GGML_MAX_NAME}); "
            f"longest is {len(longest)} chars: {longest}")

    notes: List[str] = []

    config = {}
    for key in ("num_layers", "text_num_layers", "speaker_num_layers",
                "model_size", "latent_size", "ae_latent_dim"):
        value = kv(f"echo_tts.{key}")
        if value is None:
            problems.append(f"missing metadata echo_tts.{key}")
        else:
            config[key] = int(value)

    by_prefix: Dict[str, int] = {}
    for name in tensors:
        head = name.split("/")[0] + "/" if "/" in name else "(no namespace)"
        by_prefix[head] = by_prefix.get(head, 0) + 1

    print(f"file      : {args.gguf}")
    print(f"tensors   : {len(tensors)}")
    for head in sorted(by_prefix):
        print(f"  {head:16} {by_prefix[head]}")
    scale = kv("echo_tts.pca_latent_scale")
    has_codec = kv("echo_tts.has_codec_weights")
    print(f"latent_scale       : {scale}")
    print(f"has_codec_weights  : {has_codec}")

    # PrefixedTensorSourceView routes on `prefix + "/"`. A dot-separated name
    # is never routed and the namespace reports as non-existent at load time,
    # so check the separator explicitly rather than inferring it from coverage.
    # package.cpp refuses to load a published GGUF that does not embed its spec.
    spec_json = None
    for key, kind in (("audiocpp.model_spec.version", "uint32"),
                      ("audiocpp.model_spec.family", "string"),
                      ("audiocpp.model_spec.json", "string")):
        field = fields.get(key)
        if field is None:
            problems.append(f"missing embedded model spec key '{key}'")
            continue
        raw = field.parts[field.data[0]]
        if kind == "string":
            value = bytes(raw).decode("utf-8", "replace")
            if key.endswith(".family") and value != "echo_tts":
                problems.append(f"{key} is {value!r}, expected 'echo_tts'")
            if key.endswith(".json"):
                spec_json = value
        elif int(raw[0]) != 1:
            problems.append(f"{key} is {int(raw[0])}, expected 1")
    if spec_json is not None:
        try:
            import json as _json
            embedded = _json.loads(spec_json)
            if embedded.get("schema_version") != 1:
                problems.append("embedded model spec is not schema_version 1")
        except Exception as error:  # noqa: BLE001
            problems.append(f"embedded model spec is not valid JSON: {error}")

    unrouted = [n for n in tensors if "/" not in n]
    if unrouted:
        problems.append(
            f"{len(unrouted)} tensors are outside any namespace (no '/' separator), "
            f"e.g. {unrouted[:3]}; the loader will report the namespace as missing")
    for namespace in ("dit_weights/", "pca/", "ae/" if has_codec else None):
        if namespace and not any(n.startswith(namespace) for n in tensors):
            problems.append(f"namespace '{namespace}' is empty; the loader will refuse to open it")

    if not config:
        print("\nno echo_tts metadata found; is this an Echo-TTS GGUF?", file=sys.stderr)
        return 1

    # --- coverage ---
    dit_names = {n[len(DIT_PREFIX):] for n in tensors if n.startswith(DIT_PREFIX)}
    missing_dit = [n for n in dit_probe_names(config) if n not in dit_names]
    if missing_dit:
        problems.append(f"{len(missing_dit)} DiT tensors missing, e.g. {missing_dit[:3]}")

    for required in ("components", "mean"):
        if PCA_PREFIX + required not in tensors:
            problems.append(f"missing {PCA_PREFIX}{required}")

    codec_names = {n[len(CODEC_PREFIX):] for n in tensors if n.startswith(CODEC_PREFIX)}
    if has_codec:
        missing_codec = [n for n in codec_probe_names() if n not in codec_names]
        if missing_codec:
            problems.append(
                f"{len(missing_codec)} codec tensors missing, e.g. {missing_codec[:3]}")
        leftovers = [n for n in codec_names
                     if "parametrizations" in n or n.endswith(("weight_g", "weight_v"))]
        if leftovers:
            problems.append(
                f"{len(leftovers)} codec tensors still weight-normalised, e.g. {leftovers[:2]}")
        buffers = [n for n in codec_names if n.endswith(("causal_mask", "freqs_cis"))]
        if buffers:
            notes.append(f"{len(buffers)} regenerable buffers were packaged (harmless, wastes space)")
    else:
        problems.append("has_codec_weights is false; re-run the converter with --fish-dir")

    # --- shapes ---
    # ggml_n_dims() ignores trailing 1s, so a (1, C, 1) tensor would read back as
    # (C, 1). audio.cpp restores exact logical shapes from two parallel arrays in
    # tensor order; without them the loader falls back to the lossy inference.
    exact_shapes = {}
    rank_field = fields.get("audiocpp.tensor_ranks")
    shape_field = fields.get("audiocpp.tensor_shapes")
    if rank_field is None or shape_field is None:
        problems.append(
            "missing audiocpp.tensor_ranks/tensor_shapes; tensors with trailing "
            "size-1 dimensions (snake alphas) will fail their shape checks")
    else:
        ranks = [int(rank_field.parts[i][0]) for i in rank_field.data]
        flat = [int(shape_field.parts[i][0]) for i in shape_field.data]
        order = list(reader.tensors)
        if len(ranks) != len(order):
            problems.append(
                f"audiocpp.tensor_ranks has {len(ranks)} entries for {len(order)} tensors")
        elif sum(ranks) != len(flat):
            problems.append(
                f"audiocpp.tensor_shapes has {len(flat)} values, expected {sum(ranks)}")
        else:
            cursor = 0
            for tensor, rank in zip(order, ranks):
                exact_shapes[tensor.name] = tuple(flat[cursor:cursor + rank])
                cursor += rank

    def shape(name):
        if name in exact_shapes:
            return exact_shapes[name]
        t = tensors.get(name)
        # GGUF stores dimensions reversed relative to the logical order.
        return tuple(int(d) for d in reversed(t.shape)) if t is not None else None

    expect_shapes = {
        PCA_PREFIX + "components": (config["latent_size"], config["ae_latent_dim"]),
        PCA_PREFIX + "mean": (config["ae_latent_dim"],),
        DIT_PREFIX + "out_proj.weight": (config["latent_size"], config["model_size"]),
    }
    for name, want in expect_shapes.items():
        got = shape(name)
        if got is not None and got != want:
            problems.append(f"{name}: shape {got}, expected {want}")

    # --- dtype policy ---
    wrong_dtype = []
    for name, tensor in tensors.items():
        if any(marker in name for marker in F32_REQUIRED_MARKERS) or name.endswith(".bias"):
            if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
                wrong_dtype.append((name, tensor.tensor_type.name))
    if wrong_dtype:
        problems.append(
            f"{len(wrong_dtype)} precision-sensitive tensors are not F32, "
            f"e.g. {wrong_dtype[:3]} -- re-run with the current converter")

    counts: Dict[str, int] = {}
    for tensor in tensors.values():
        counts[tensor.tensor_type.name] = counts.get(tensor.tensor_type.name, 0) + 1
    print(f"dtypes    : {counts}")

    print()
    for note in notes:
        print(f"  note: {note}")
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}")
        return 1
    print("  GGUF looks good: tensor coverage, shapes, and precision policy all check out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
