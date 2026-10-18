#!/usr/bin/env python3
"""Repack the third-party `cstr/chatterbox-turbo-GGUF` pair into one audio.cpp-native,
self-contained GGUF.

The upstream package (published by `cstr` for the CrispASR project, MIT-relicensed; not
published by ResembleAI or audio.cpp) ships as two loose GGUF files with a flat dot-separated
tensor namespace and abbreviated S3Gen tensor names. This script repacks both into audio.cpp's
own "/"-delimited packed-GGUF namespace convention, renames S3Gen tensors back to the exact names
base Chatterbox's own S3Gen/HiFT-vocoder loader code already expects (so that loader runs
unmodified, no runtime name-translation bridge needed), and produces the tokenizer as plain
vocab/merges/special-token sidecar files instead of runtime GGUF-metadata reads. The actual GGUF
writing (quantization, embedded package-spec metadata, embedded sidecars) is done by this
project's own `audiocpp_gguf` converter -- this script only stages inputs for it, exactly like
tools/community_models/convert_voxcpm1.py does for VoxCPM1.

Only the T3 GPT2 backbone, the built-in default-voice conditionals, and the S3Gen flow/HiFT
vocoder are repacked. The upstream GGUF's `ve.*` (LSTM speaker-verification voice encoder) and
`s3.se.*`/`s3.tok.*` (ResNet speaker encoder / S3 speech tokenizer) sections are not staged: they
are not read by any loader in this codebase yet (custom voice cloning -- a caller-supplied
reference clip -- is not implemented; only the built-in baked-in voice is), and their tensor
layout has not been validated.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.numpy import save_file


# ---------------------------------------------------------------------------------------------
# S3Gen tensor-name mapping. This is the exact inverse of the rule tables that used to live in
# src/community_models/chatterbox_turbo/components/s3gen_name_bridge.cpp (deleted -- the repacked
# GGUF no longer needs a runtime bridge). Keep this in sync by hand if that history is ever
# revisited; the rules are transcribed here, not derived programmatically, to keep this script
# self-contained.
# ---------------------------------------------------------------------------------------------

FLOW_PREFIX_RULES = [
    ("flow.encoder.encoders.", "fe.enc."),
    ("flow.encoder.up_encoders.", "fe.ue."),
    ("flow.encoder.up_layer.", "fe.ul."),
    ("flow.encoder.up_embed.", "fe.uemb."),
    ("flow.encoder.pre_lookahead_layer.", "fe.pla."),
    ("flow.encoder.embed.", "fe.embed."),
    ("flow.encoder.after_norm", "fe.an"),
    ("flow.decoder.estimator.down_blocks.", "fd.db."),
    ("flow.decoder.estimator.mid_blocks.", "fd.mb."),
    ("flow.decoder.estimator.up_blocks.", "fd.ub."),
    ("flow.decoder.estimator.final_block", "fd.fb"),
    ("flow.decoder.estimator.final_proj", "fd.fp"),
    ("flow.decoder.estimator.time_mlp.", "fd.tm."),
    ("flow.decoder.estimator.time_embed_mixer", "fd.tmx"),
    # flow.encoder_proj / flow.spk_embed_affine_layer / flow.input_embedding: unchanged.
]

SEGMENT_RULES = [
    (".self_attn.linear_q", ".sa.lq"),
    (".self_attn.linear_k", ".sa.lk"),
    (".self_attn.linear_v", ".sa.lv"),
    (".self_attn.linear_out", ".sa.lo"),
    (".self_attn.linear_pos", ".sa.lp"),
    (".self_attn.pos_bias_u", ".sa.pbu"),
    (".self_attn.pos_bias_v", ".sa.pbv"),
    (".norm_mha", ".nmha"),
    (".norm_ff", ".nff"),
    (".feed_forward.w_1", ".ff.w_1"),
    (".feed_forward.w_2", ".ff.w_2"),
    (".block1.block.0", ".b1.0"),
    (".block1.block.2", ".b1.2"),
    (".block2.block.0", ".b2.0"),
    (".block2.block.2", ".b2.2"),
]

TRANSFORMER_BLOCK_RULES = [
    (".res_conv", ".rc"),
    (".attn1.to_q", ".attn1.q"),
    (".attn1.to_k", ".attn1.k"),
    (".attn1.to_v", ".attn1.v"),
    (".attn1.to_out.0", ".attn1.o"),
    (".ff.net.0.proj", ".ff.up"),
]

VOCODER_PREFIX_RULES = [
    ("conv_pre", "cpre"),
    ("conv_post", "cpost"),
    ("resblocks.", "rb."),
    ("source_downs.", "sd."),
    ("source_resblocks.", "srb."),
    ("f0_predictor.condnet.", "f0.cn."),
]


def reverse_translate_flow_name(name):
    """abbrev (post 's3.'-strip, no leading 'v.') -> canonical base-chatterbox flow name."""
    translated = name
    # Forward order was: prefix, then segment rules, then ".ff.net.2"->".ff.down", then
    # transformer-block rules. Invert in the opposite order.
    for canonical, abbrev in TRANSFORMER_BLOCK_RULES:
        translated = translated.replace(abbrev, canonical)
    translated = translated.replace(".ff.down", ".ff.net.2")
    for canonical, abbrev in SEGMENT_RULES:
        translated = translated.replace(abbrev, canonical)
    for canonical, abbrev in FLOW_PREFIX_RULES:
        if translated.startswith(abbrev):
            translated = canonical + translated[len(abbrev):]
            break
    return translated


def reverse_translate_vocoder_name(name):
    """abbrev (post 's3.v.'-strip) -> canonical HiFT vocoder tensor name (without the 'v.' the
    C++ loader's HiftVocoderConfig.tensor_prefix adds back on)."""
    translated = name
    if translated == "f0.cls" or translated.startswith("f0.cls."):
        translated = "f0_predictor.classifier" + translated[len("f0.cls"):]
    elif translated == "ms.ll" or translated.startswith("ms.ll."):
        translated = "m_source.l_linear" + translated[len("ms.ll"):]
    else:
        translated = translated.replace(".c1.", ".convs1.")
        translated = translated.replace(".c2.", ".convs2.")
        translated = translated.replace(".a1.", ".activations1.")
        translated = translated.replace(".a2.", ".activations2.")
    for canonical, abbrev in VOCODER_PREFIX_RULES:
        if translated.startswith(abbrev):
            translated = canonical + translated[len(abbrev):]
            break
    return translated


def reverse_translate_s3gen_name(name_without_s3_prefix):
    if name_without_s3_prefix.startswith("v."):
        return "v." + reverse_translate_vocoder_name(name_without_s3_prefix[len("v."):])
    return reverse_translate_flow_name(name_without_s3_prefix)


# ---------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repack the cstr/chatterbox-turbo-GGUF pair into an audio.cpp-native GGUF."
    )
    parser.add_argument(
        "--t3-source",
        type=Path,
        required=True,
        help="Path to the upstream chatterbox-turbo-t3-*.gguf file.",
    )
    parser.add_argument(
        "--s3gen-source",
        type=Path,
        required=True,
        help="Path to the upstream chatterbox-turbo-s3gen-*.gguf file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output native GGUF path.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("build/debug/bin/audiocpp_gguf"),
        help="Path to the audiocpp_gguf converter.",
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=Path("model_specs/chatterbox_turbo.json"),
        help="Chatterbox Turbo model spec JSON.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("build/chatterbox_turbo_native_gguf"),
        help="Directory for staged safetensors and sidecars.",
    )
    parser.add_argument("--type", default="q8_0", help="Main model conversion type.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output GGUF.")
    return parser.parse_args()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_tensor_float(reader_tensor):
    return dequantize(reader_tensor.data, reader_tensor.tensor_type).astype(np.float32)


def stage_t3(t3_reader, staging):
    t3_tensors = {}
    conds_tensors = {}
    for tensor in t3_reader.tensors:
        name = tensor.name
        if name.startswith("t3."):
            key = name[len("t3."):]
            bucket = t3_tensors
        elif name.startswith("conds."):
            key = name[len("conds."):]
            bucket = conds_tensors
        elif name.startswith("ve."):
            continue  # LSTM speaker-verification encoder: unused, out of scope.
        else:
            raise KeyError(f"unrecognized top-level T3 GGUF tensor namespace: {name}")

        if key in ("gen.prompt_token", "t3.speech_prompt_tokens"):
            # i32 speech-token id arrays: keep integer dtype, do not dequantize. tensor.data is
            # already a properly shaped (not reversed) numpy view for GGUFReader tensors.
            bucket[key] = np.array(tensor.data, dtype=np.int32)
        else:
            bucket[key] = read_tensor_float(tensor)

    save_file(t3_tensors, staging / "t3.safetensors")
    save_file(conds_tensors, staging / "conds.safetensors")
    return t3_tensors, conds_tensors


def stage_s3gen(s3gen_reader, staging):
    s3gen_tensors = {}
    for tensor in s3gen_reader.tensors:
        name = tensor.name
        if not name.startswith("s3."):
            raise KeyError(f"unrecognized top-level S3Gen GGUF tensor namespace: {name}")
        suffix = name[len("s3."):]
        if suffix.startswith("se.") or suffix.startswith("tok."):
            continue  # ResNet speaker encoder / S3 speech tokenizer: unused, out of scope.

        mapped = reverse_translate_s3gen_name(suffix)
        if mapped in s3gen_tensors:
            raise RuntimeError(f"duplicate mapped S3Gen tensor: {mapped} (from {name})")
        value = read_tensor_float(tensor)

        # Known shape fix: every nn.Linear(in, 1).weight in this checkpoint (out_features=1) is
        # squeezed to 1-D [in] in the upstream GGUF; base Chatterbox's HiFT vocoder loader
        # (load_linear in src/framework/modules/vocoders/hift_vocoder.cpp) does a strict
        # {out_features, in_features} shape check, so restore the canonical 2-D [1, in] shape.
        if mapped in ("v.f0_predictor.classifier.weight", "v.m_source.l_linear.weight") and value.ndim == 1:
            value = value.reshape(1, -1)

        s3gen_tensors[mapped] = value

    save_file(s3gen_tensors, staging / "s3gen.safetensors")
    return s3gen_tensors


def looks_like_bracket_tag(token):
    return len(token) >= 3 and token.startswith("[") and token.endswith("]")


def stage_tokenizer(t3_reader, staging):
    tokens = t3_reader.get_field("tokenizer.ggml.tokens").contents()
    merges = t3_reader.get_field("tokenizer.ggml.merges").contents()
    if not tokens:
        raise RuntimeError("T3 GGUF has an empty tokenizer.ggml.tokens array")

    # Exact port of the trailing-bracket-tag scan that used to run at C++ load time in
    # text_tokenizer_turbo.cpp: the 19 emotion/style control tags ([laugh], [sigh], ...) sit at
    # the tail of the vocab and were never trained into the BPE merge table, so they must be
    # registered as atomic special tokens rather than plain vocab entries.
    special_tokens_start = len(tokens)
    for token_id in range(len(tokens) - 1, -1, -1):
        if not looks_like_bracket_tag(tokens[token_id]):
            break
        special_tokens_start = token_id

    vocab = {token: token_id for token_id, token in enumerate(tokens) if token_id < special_tokens_start}
    special_tokens = [
        {"token": tokens[token_id], "id": token_id}
        for token_id in range(special_tokens_start, len(tokens))
    ]

    with open(staging / "chatterbox_turbo_vocab.json", "w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False)
    with open(staging / "chatterbox_turbo_merges.txt", "w", encoding="utf-8") as handle:
        for merge in merges:
            handle.write(merge + "\n")
    with open(staging / "chatterbox_turbo_special_tokens.json", "w", encoding="utf-8") as handle:
        json.dump(special_tokens, handle, ensure_ascii=False, indent=2)


def run_converter(args, staging):
    command = [
        str(args.converter),
        "--input", f"t3={staging / 't3.safetensors'}",
        "--input", f"conds={staging / 'conds.safetensors'}",
        "--input", f"s3gen={staging / 's3gen.safetensors'}",
        "--root", str(staging),
        "--output", str(args.output),
        "--type", args.type,
        "--keep-type", "conds/gen.prompt_token=orig",
        "--keep-type", "conds/t3.speech_prompt_tokens=orig",
        "--family", "chatterbox_turbo",
        "--model-spec", str(args.model_spec),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    require_file(args.t3_source)
    require_file(args.s3gen_source)
    require_file(args.converter)
    require_file(args.model_spec)

    staging = args.work_dir / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    t3_reader = GGUFReader(str(args.t3_source))
    s3gen_reader = GGUFReader(str(args.s3gen_source))

    stage_t3(t3_reader, staging)
    stage_s3gen(s3gen_reader, staging)
    stage_tokenizer(t3_reader, staging)

    run_converter(args, staging)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
