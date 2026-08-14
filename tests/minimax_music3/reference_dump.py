#!/usr/bin/env python3
"""Dump MiniMax-Music3 reference component outputs for parity testing.

Runs one diffusers reference component on a deterministic input, saves the raw
float32 input and output pair that tests/minimax_music3/minimax_music3_component_probe.cpp
consumes, and prints the tensor shapes.

Example:

  .venv-music3/bin/python tests/minimax_music3/reference_dump.py \
    --snapshot models/MiniMax-Music3-hf --component vocoder --out-dir /tmp/parity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def dump_vocoder(snapshot: Path, out_dir: Path, device: str, length: int) -> None:
    from diffusers import MiniMaxMusic3Vocoder

    vocoder = MiniMaxMusic3Vocoder.from_pretrained(snapshot / "vocoder", torch_dtype=torch.float32)
    vocoder = vocoder.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    latents = torch.randn((1, vocoder.config.latent_channels, length), generator=generator)
    with torch.no_grad():
        waveform = vocoder(latents.to(device)).cpu().float()
    latents.numpy().astype(np.float32).tofile(out_dir / "vocoder_input.f32")
    # Save interleaved stereo to match the C++ decode() output.
    interleaved = waveform[0].transpose(0, 1).contiguous().numpy().astype(np.float32)
    interleaved.tofile(out_dir / "vocoder_ref.f32")
    print(f"vocoder: latents {tuple(latents.shape)} -> waveform {tuple(waveform.shape)}")


def dump_condition_encoder(snapshot: Path, out_dir: Path, device: str, frames: int) -> None:
    from diffusers import MiniMaxMusic3ConditionEncoder

    encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
        snapshot / "condition_encoder", torch_dtype=torch.float32)
    encoder = encoder.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    hidden = torch.randn(
        (1, frames, encoder.config.num_condition_layers * encoder.config.condition_hidden_dim),
        generator=generator)
    with torch.no_grad():
        condition = encoder(hidden.to(device)).cpu().float()
    hidden.numpy().astype(np.float32).tofile(out_dir / "condition_encoder_input.f32")
    condition.numpy().astype(np.float32).tofile(out_dir / "condition_encoder_ref.f32")
    print(f"condition_encoder: hidden {tuple(hidden.shape)} -> condition {tuple(condition.shape)}")


AUDIO_CODE_OFFSET = 151675
AR_CFG_SCALE = 1.5


def dump_depth_decoder(snapshot: Path, out_dir: Path, device: str, semantic_code: int) -> None:
    import json

    from diffusers import MiniMaxMusic3RVQDepthDecoder
    from safetensors import safe_open

    decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        snapshot / "rvq_depth_decoder", torch_dtype=torch.float32)
    decoder = decoder.to(device).eval()
    index = json.loads((snapshot / "language_model" / "model.safetensors.index.json").read_text())
    embed_shard = index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(snapshot / "language_model" / embed_shard, framework="pt") as handle:
        embed_tokens = handle.get_tensor("model.embed_tokens.weight").to(torch.float32).to(device)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    last_hidden = torch.randn((2, decoder.config.hidden_size), generator=generator).to(device)

    num_codebooks = decoder.config.num_codebooks
    vocab = decoder.config.audio_vocab_size
    with torch.no_grad():
        sequence = [decoder.projection(last_hidden).unsqueeze(1)]
        code_embed = embed_tokens[semantic_code + AUDIO_CODE_OFFSET].unsqueeze(0).expand(2, -1)
        sequence.append(decoder.projection(code_embed).unsqueeze(1))
        codes = [semantic_code]
        hidden_parts = []
        for index_cb in range(1, num_codebooks):
            hidden = decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(hidden[:1])
            logits = decoder.audio_heads[index_cb - 1](hidden)
            conditional, unconditional = logits[:1].float(), logits[1:2].float()
            guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
            code = int(guided.argmax(dim=-1).item())
            codes.append(code)
            if index_cb < num_codebooks - 1:
                embed = decoder.audio_embeddings(
                    torch.tensor([code + (index_cb - 1) * vocab], device=device)).expand(2, -1)
                sequence.append(decoder.projection(embed).unsqueeze(1))
        depth_hidden = torch.cat(hidden_parts, dim=-1)
        feedback = embed_tokens[semantic_code + AUDIO_CODE_OFFSET].clone()
        for index_cb in range(1, num_codebooks):
            feedback += decoder.audio_embeddings.weight[codes[index_cb] + (index_cb - 1) * vocab]
        feedback = feedback * num_codebooks ** -0.5

    last_hidden.cpu().numpy().astype(np.float32).tofile(out_dir / "depth_decoder_input.f32")
    packed = np.concatenate([
        np.asarray(codes, dtype=np.float32),
        depth_hidden.cpu().numpy().astype(np.float32).reshape(-1),
        feedback.cpu().numpy().astype(np.float32).reshape(-1),
    ])
    packed.tofile(out_dir / "depth_decoder_ref.f32")
    print(f"depth_decoder: codes {codes}")


_CAPTION = """Global Metadata
Basic Attributes: bpm is 92. key is E, and scale is minor. Electric Blues / Blues Rock.
Vocal Details
Vocal Gender & Timbre: Singer A (Male). A deep, gravelly baritone with a raspy quality.
Arrangement
Primary: A clean-to-slightly-overdriven electric guitar drives the track from Intro to Outro."""

_LYRICS = """[verse]
I'm learning how to fill up
every space I used to leave,
teaching my own heart the patience
that my family needs from me.
[pre-chorus]
Breathe a little deeper,
love is here to heal.
[chorus]
You gotta let love lift what used to fall…
[outro]
We're gonna let love stay."""

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_CFG_TOKEN_ID = 151654


def dump_tokenizer(snapshot: Path, out_dir: Path) -> None:
    import importlib.util
    import sys

    from transformers import Qwen2Tokenizer

    spec = importlib.util.find_spec("diffusers.modular_pipelines.minimax_music3.encoders")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    tokenizer = Qwen2Tokenizer.from_pretrained(snapshot / "tokenizer")
    text = (
        f"{_IM_START}{_CAPTION_START}{module._clean_caption(_CAPTION)}{_CAPTION_END}"
        f"{_LYRICS_START}{module._normalize_lyrics(_LYRICS)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
    )
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    uncond = list(ids)
    for index in range(1, len(uncond) - 2):
        uncond[index] = _AUDIO_CFG_TOKEN_ID
    (out_dir / "tokenizer_caption.txt").write_text(_CAPTION)
    (out_dir / "tokenizer_lyrics.txt").write_text(_LYRICS)
    packed = np.asarray([len(ids)] + ids + uncond, dtype=np.float32)
    packed.tofile(out_dir / "tokenizer_ref.f32")
    print(f"tokenizer: {len(ids)} prompt tokens")


def dump_dit(snapshot: Path, out_dir: Path, device: str, length: int) -> None:
    from diffusers import MiniMaxMusic3Transformer1DModel

    transformer = MiniMaxMusic3Transformer1DModel.from_pretrained(
        snapshot / "transformer", torch_dtype=torch.float32)
    transformer = transformer.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(1234)
    latents = torch.randn((1, transformer.config.in_channels, length), generator=generator)
    condition = torch.randn((1, length, transformer.config.condition_dim), generator=generator)
    timestep = torch.tensor([0.5])
    with torch.no_grad():
        velocity = transformer(
            hidden_states=latents.to(device),
            timestep=timestep.to(device),
            encoder_hidden_states=condition.to(device),
            return_dict=False,
        )[0].cpu().float()
    packed_in = np.concatenate([
        latents.numpy().astype(np.float32).reshape(-1),
        condition[0].transpose(0, 1).contiguous().numpy().astype(np.float32).reshape(-1),
    ])
    packed_in.tofile(out_dir / "dit_input.f32")
    velocity.numpy().astype(np.float32).tofile(out_dir / "dit_ref.f32")
    print(f"dit: latents {tuple(latents.shape)} -> velocity {tuple(velocity.shape)}")


def dump_lm_prefill(snapshot: Path, out_dir: Path, device: str) -> None:
    from transformers import Qwen3ForCausalLM

    ref_ids = np.fromfile(out_dir / "tokenizer_ref.f32", dtype=np.float32).astype(np.int64)
    count = int(ref_ids[0])
    cond = ref_ids[1 : 1 + count]
    uncond = ref_ids[1 + count : 1 + 2 * count]
    model = Qwen3ForCausalLM.from_pretrained(
        snapshot / "language_model", dtype=torch.bfloat16)
    model = model.to(device).eval()
    ids = torch.tensor(np.stack([cond, uncond]), device=device)
    with torch.no_grad():
        output = model.model(input_ids=ids)
        hidden = output.last_hidden_state[:, -1]
        logits = model.lm_head(hidden).float()
    rows = [151670] + list(range(151675, 151675 + 16384))
    sliced = logits[:, rows]
    packed = np.concatenate([
        sliced[0].cpu().numpy().astype(np.float32),
        sliced[1].cpu().numpy().astype(np.float32),
        hidden[0].float().cpu().numpy().astype(np.float32),
        hidden[1].float().cpu().numpy().astype(np.float32),
    ])
    packed.tofile(out_dir / "lm_prefill_ref.f32")
    cond_uncond = np.concatenate([cond, uncond]).astype(np.float32)
    cond_uncond.tofile(out_dir / "lm_prefill_input.f32")
    print(f"lm_prefill: {count} tokens, argmax cond {int(sliced[0].argmax())}")


def dump_lm_decode(snapshot: Path, out_dir: Path, device: str) -> None:
    from transformers import Qwen3ForCausalLM

    ref_ids = np.fromfile(out_dir / "tokenizer_ref.f32", dtype=np.float32).astype(np.int64)
    count = int(ref_ids[0])
    cond = ref_ids[1 : 1 + count]
    uncond = ref_ids[1 + count : 1 + 2 * count]
    model = Qwen3ForCausalLM.from_pretrained(snapshot / "language_model", dtype=torch.bfloat16)
    model = model.to(device).eval()
    ids = torch.tensor(np.stack([cond, uncond]), device=device)
    generator = torch.Generator(device="cpu").manual_seed(77)
    embedding = (torch.randn((4096,), generator=generator) * 0.02).to(torch.bfloat16)
    with torch.no_grad():
        output = model.model(input_ids=ids, use_cache=True)
        feed = embedding.to(device).unsqueeze(0).unsqueeze(0).expand(2, 1, -1)
        step = model.model(
            inputs_embeds=feed, past_key_values=output.past_key_values, use_cache=True)
        hidden = step.last_hidden_state[:, -1]
        logits = model.lm_head(hidden).float()
    rows = [151670] + list(range(151675, 151675 + 16384))
    sliced = logits[:, rows]
    packed = np.concatenate([
        sliced[0].cpu().numpy().astype(np.float32),
        sliced[1].cpu().numpy().astype(np.float32),
        hidden[0].float().cpu().numpy().astype(np.float32),
        hidden[1].float().cpu().numpy().astype(np.float32),
    ])
    packed.tofile(out_dir / "lm_decode_ref.f32")
    embed_and_ids = np.concatenate([
        embedding.float().numpy().astype(np.float32),
        np.concatenate([cond, uncond]).astype(np.float32),
    ])
    embed_and_ids.tofile(out_dir / "lm_decode_input.f32")
    print(f"lm_decode: argmax cond {int(sliced[0].argmax())} uncond {int(sliced[1].argmax())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--component",
        required=True,
        choices=("vocoder", "condition_encoder", "depth_decoder", "tokenizer", "dit", "lm_prefill", "lm_decode"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--length", type=int, default=173, help="latent length for the vocoder dump")
    parser.add_argument("--frames", type=int, default=50, help="AR frames for the condition encoder dump")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.component == "vocoder":
        dump_vocoder(args.snapshot, args.out_dir, args.device, args.length)
    elif args.component == "condition_encoder":
        dump_condition_encoder(args.snapshot, args.out_dir, args.device, args.frames)
    elif args.component == "depth_decoder":
        dump_depth_decoder(args.snapshot, args.out_dir, args.device, semantic_code=1234)
    elif args.component == "tokenizer":
        dump_tokenizer(args.snapshot, args.out_dir)
    elif args.component == "dit":
        dump_dit(args.snapshot, args.out_dir, args.device, length=344)
    elif args.component == "lm_prefill":
        dump_lm_prefill(args.snapshot, args.out_dir, args.device)
    elif args.component == "lm_decode":
        dump_lm_decode(args.snapshot, args.out_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
