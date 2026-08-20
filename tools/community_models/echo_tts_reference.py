#!/usr/bin/env python3
"""Dump reference activations from upstream Echo-TTS for port parity testing.

Run this from inside a checkout of https://github.com/jordandare/echo-tts with
its requirements installed and a GPU available:

    cd /path/to/echo-tts
    python3 echo_tts_reference.py --speaker audio_prompts/<some>.wav -o echo_ref.npz

Everything is pinned to a fixed seed and a fixed text so the C++ port can be
compared stage by stage. Small tensors are dumped in full; the 24 DiT block
activations are dumped as statistics plus a value prefix unless --full-blocks is
passed, which keeps the archive to a few MB rather than a few hundred.

The DiT forward pass is captured at a single fixed timestep with a fixed input
latent, deliberately *not* the sampler's own trajectory: that isolates a wrong
block from a wrong integration step. The sampler is then run separately.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict

import numpy as np
import torch

try:
    from inference import (
        get_speaker_latent_and_mask,
        get_text_input_ids_and_mask,
        ae_decode,
        ae_encode,
        load_audio,
        load_fish_ae_from_hf,
        load_model_from_hf,
        load_pca_state_from_hf,
        sample_euler_cfg_independent_guidances,
        tokenizer_encode,
    )
except ImportError as error:  # pragma: no cover - guidance path
    print(
        f"could not import the upstream inference module ({error}).\n"
        "Run this script from inside a checkout of jordandare/echo-tts.",
        file=sys.stderr,
    )
    raise SystemExit(2)

DEFAULT_TEXT = (
    "[S1] Alright, I'm going to demo this new model called Echo TTS. "
    "Hopefully this works, I'm super excited to try this and see what it can do."
)

SEED = 0
FIXED_T = 0.7
SEQUENCE_LENGTH = 640


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def summarize(name: str, tensor: torch.Tensor, out: Dict[str, Any], prefix: int = 64) -> None:
    """Store shape, moments, and a value prefix -- enough to localise drift."""
    array = to_numpy(tensor)
    flat = array.reshape(-1)
    out[f"{name}.shape"] = np.array(array.shape, dtype=np.int64)
    out[f"{name}.stats"] = np.array(
        [flat.mean(), flat.std(), flat.min(), flat.max()], dtype=np.float64
    )
    out[f"{name}.prefix"] = flat[:prefix].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", required=True, help="reference wav for cloning")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("-o", "--output", default="echo_ref.npz")
    parser.add_argument(
        "--full-blocks",
        action="store_true",
        help="dump every DiT block activation in full (large)",
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="sampler steps (default: 40)"
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    out: Dict[str, Any] = {}

    model = load_model_from_hf(delete_blockwise_modules=True)
    fish_ae = load_fish_ae_from_hf()
    pca_state = load_pca_state_from_hf()
    device, dtype = model.device, model.dtype

    out["pca.components"] = to_numpy(pca_state.pca_components)
    out["pca.mean"] = to_numpy(pca_state.pca_mean)
    out["pca.latent_scale"] = np.array([pca_state.latent_scale], dtype=np.float64)

    # ---- tokenizer -------------------------------------------------------
    ids, normalized = tokenizer_encode(args.text, return_normalized_text=True)
    out["tokenizer.input_ids"] = to_numpy(ids).astype(np.int32)
    out["tokenizer.normalized_text"] = np.array([normalized.encode("utf-8")])

    text_input_ids, text_mask = get_text_input_ids_and_mask(
        [args.text], max_length=None, device=device
    )
    out["text.input_ids"] = to_numpy(text_input_ids).astype(np.int32)
    out["text.mask"] = to_numpy(text_mask).astype(np.int32)

    # ---- autoencoder round trip -----------------------------------------
    speaker_audio = load_audio(args.speaker).to(device)
    out["audio.speaker_input"] = to_numpy(speaker_audio)

    z_q = fish_ae.encode_zq(speaker_audio.unsqueeze(0).to(fish_ae.dtype))
    summarize("ae.encode_zq", z_q, out)

    latent_from_audio = ae_encode(fish_ae, pca_state, speaker_audio.unsqueeze(0).to(fish_ae.dtype))
    summarize("ae.encode_pca", latent_from_audio, out)

    reconstructed = ae_decode(fish_ae, pca_state, latent_from_audio)
    summarize("ae.decode_roundtrip", reconstructed, out)

    speaker_latent, speaker_mask = get_speaker_latent_and_mask(
        fish_ae, pca_state, speaker_audio.to(fish_ae.dtype)
    )
    out["speaker.latent"] = to_numpy(speaker_latent)
    out["speaker.mask"] = to_numpy(speaker_mask).astype(np.int32)

    # ---- conditioning encoders ------------------------------------------
    with torch.inference_mode():
        text_state = model.text_encoder(text_input_ids, text_mask)
        text_state = model.text_norm(text_state)
        summarize("text_encoder.output", text_state, out)

        speaker_state = model.speaker_encoder(speaker_latent.to(dtype))
        speaker_state = model.speaker_norm(speaker_state)
        summarize("speaker_encoder.output", speaker_state, out)

        kv_text = model.get_kv_cache_text(text_input_ids, text_mask)
        kv_speaker = model.get_kv_cache_speaker(speaker_latent.to(dtype))
        for layer in (0, len(kv_text) // 2, len(kv_text) - 1):
            summarize(f"kv_text.{layer}.k", kv_text[layer][0], out)
            summarize(f"kv_text.{layer}.v", kv_text[layer][1], out)
            summarize(f"kv_speaker.{layer}.k", kv_speaker[layer][0], out)
            summarize(f"kv_speaker.{layer}.v", kv_speaker[layer][1], out)

        # ---- single fixed-timestep DiT forward, with per-block hooks -----
        block_outputs: Dict[int, torch.Tensor] = {}

        def make_hook(index: int):
            def hook(_module, _inputs, output):
                block_outputs[index] = output.detach()

            return hook

        handles = [
            block.register_forward_hook(make_hook(index))
            for index, block in enumerate(model.blocks)
        ]

        generator = torch.Generator(device=device).manual_seed(SEED)
        x_fixed = torch.randn(
            (1, SEQUENCE_LENGTH, 80),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        out["dit.x_input"] = to_numpy(x_fixed)
        out["dit.t"] = np.array([FIXED_T], dtype=np.float64)

        t_tensor = (torch.ones((1,), device=device) * FIXED_T).to(dtype)
        v_pred = model(
            x=x_fixed.to(dtype),
            t=t_tensor,
            text_mask=text_mask,
            speaker_mask=speaker_mask,
            kv_cache_text=kv_text,
            kv_cache_speaker=kv_speaker,
        )
        for handle in handles:
            handle.remove()

        out["dit.v_pred"] = to_numpy(v_pred)
        for index, value in sorted(block_outputs.items()):
            if args.full_blocks:
                out[f"dit.block.{index}"] = to_numpy(value).astype(np.float16)
            summarize(f"dit.block.{index}", value, out)

        # ---- full sampler + decode --------------------------------------
        latent_out = sample_euler_cfg_independent_guidances(
            model=model,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            rng_seed=SEED,
            num_steps=args.steps,
            cfg_scale_text=3.0,
            cfg_scale_speaker=8.0,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
            truncation_factor=0.8,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=SEQUENCE_LENGTH,
        )
        out["sampler.latent"] = to_numpy(latent_out)

        # The initial noise, reproduced exactly as the sampler draws it, so the
        # C++ Philox path can be checked independently of the model.
        noise_generator = torch.Generator(device=device).manual_seed(SEED)
        noise = torch.randn(
            (1, SEQUENCE_LENGTH, 80),
            device=device,
            dtype=torch.float32,
            generator=noise_generator,
        )
        out["sampler.initial_noise"] = to_numpy(noise)

        audio_out = ae_decode(fish_ae, pca_state, latent_out)
        summarize("decode.audio_full", audio_out, out)
        out["decode.audio_prefix"] = to_numpy(audio_out).reshape(-1)[:44100]

    out["config.seed"] = np.array([SEED], dtype=np.int64)
    out["config.steps"] = np.array([args.steps], dtype=np.int64)
    out["config.sequence_length"] = np.array([SEQUENCE_LENGTH], dtype=np.int64)
    out["config.model_dtype"] = np.array([str(dtype).encode("utf-8")])

    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output} ({len(out)} arrays)")
    for key in sorted(out):
        if key.endswith(".stats"):
            mean, std, low, high = out[key]
            print(f"  {key[:-6]:<40} mean={mean:+.6f} std={std:.6f} "
                  f"min={low:+.4f} max={high:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
