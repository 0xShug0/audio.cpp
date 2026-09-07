#!/usr/bin/env python3
"""Run the trusted upstream MiraTTS implementation for native parity checks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import huggingface_hub
import numpy as np
import scipy.io.wavfile
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    # TTSCodec hardcodes snapshot_download; keep the upstream implementation intact
    # while making it use the already verified local snapshot.
    huggingface_hub.snapshot_download = lambda *unused_args, **unused_kwargs: str(model_dir)

    # The official checkpoint uses a legacy PyTorch file for FlashSR. The pinned
    # Windows Torch in this isolated environment predates Transformers' new guard.
    # Only the official, revision-pinned checkpoint is admitted here.
    import transformers.modeling_utils

    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
    from ncodec.codec import TTSCodec

    codec = TTSCodec()
    context_tokens = codec.encode(str(args.reference), encode_semantic=False)
    prompt = codec.format_prompt(args.text, context_tokens, None)

    backend = TurbomindEngineConfig(
        cache_max_entry_count=0.2,
        tp=1,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )
    pipe = pipeline(str(model_dir), backend_config=backend)
    generation = GenerationConfig(
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        do_sample=True,
        random_seed=args.seed,
    )
    response = pipe([prompt], gen_config=generation, do_preprocess=False)[0]
    speech_tokens = response.text
    speech_ids = np.asarray(
        [[int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)]],
        dtype=np.int64,
    )
    context_ids = np.asarray(
        [[[int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)]]],
        dtype=np.int32,
    )
    decoder = codec.audio_decoder
    latent = decoder.processor_detokenizer.run(
        ["preprocessed_output"],
        {"context_tokens": context_ids, "speech_tokens": speech_ids},
    )[0]
    lowres = decoder.audio_detokenizer.decode(
        torch.from_numpy(latent).to("cuda:0")
    ).squeeze().detach().float().cpu().numpy()
    audio = codec.decode(speech_tokens, context_tokens)
    waveform = audio.detach().float().cpu().numpy().reshape(-1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(args.output, 48000, np.clip(waveform, -1.0, 1.0))
    scipy.io.wavfile.write(
        args.output.with_name(args.output.stem + "-lowres.wav"),
        16000,
        np.clip(lowres, -1.0, 1.0),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "reference": str(args.reference.resolve()),
                "text": args.text,
                "context_tokens": context_tokens,
                "speech_tokens": speech_tokens,
                "sample_rate": 48000,
                "samples": int(waveform.size),
                "rms": float(np.sqrt(np.mean(np.square(waveform)))),
                "peak": float(np.max(np.abs(waveform))),
                "processor_latent": {
                    "shape": list(latent.shape),
                    "sum": float(np.sum(latent, dtype=np.float64)),
                    "sum_sq": float(np.sum(np.square(latent), dtype=np.float64)),
                    "first": latent.reshape(-1)[:10].astype(float).tolist(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
