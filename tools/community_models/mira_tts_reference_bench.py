#!/usr/bin/env python3
"""Run a long-lived upstream MiraTTS request sequence for parity testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import huggingface_hub
import numpy as np
import scipy.io.wavfile
import torch


def audio_sha256(waveform: np.ndarray) -> str:
    pcm = np.clip(waveform, -1.0, 1.0).astype(np.float32, copy=False)
    return hashlib.sha256(pcm.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    huggingface_hub.snapshot_download = (
        lambda *unused_args, **unused_kwargs: str(model_dir)
    )

    # The official checkpoint contains a revision-pinned legacy PyTorch file.
    import transformers.modeling_utils

    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

    from lmdeploy import GenerationConfig, TurbomindEngineConfig, pipeline
    from ncodec.codec import TTSCodec

    requests = json.loads(args.request_file.read_text(encoding="utf-8"))["requests"]
    if not requests:
        raise RuntimeError("MiraTTS reference request file is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)

    codec = TTSCodec()
    context_tokens = codec.encode(str(args.reference.resolve()), encode_semantic=False)
    context_ids = np.asarray(
        [[[int(token) for token in re.findall(r"context_token_(\d+)", context_tokens)]]],
        dtype=np.int32,
    )
    backend = TurbomindEngineConfig(
        cache_max_entry_count=0.2,
        tp=1,
        dtype="bfloat16",
        enable_prefix_caching=False,
    )
    pipe = pipeline(str(model_dir), backend_config=backend)
    decoder = codec.audio_decoder

    results: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        name = request.get("name", f"request_{index}")
        prompt = codec.format_prompt(request["text"], context_tokens, None)
        generation = GenerationConfig(
            top_p=float(request.get("top_p", 0.95)),
            top_k=int(request.get("top_k", 50)),
            temperature=float(request.get("temperature", 0.8)),
            max_new_tokens=int(request.get("max_tokens", 1024)),
            repetition_penalty=float(request.get("repetition_penalty", 1.2)),
            min_p=float(request.get("min_p", 0.05)),
            do_sample=True,
            random_seed=int(request.get("seed", 1234)),
        )
        started = time.perf_counter()
        response = pipe([prompt], gen_config=generation, do_preprocess=False)[0]
        speech_tokens = response.text
        speech_ids = np.asarray(
            [[int(token) for token in re.findall(r"speech_token_(\d+)", speech_tokens)]],
            dtype=np.int64,
        )
        latent = decoder.processor_detokenizer.run(
            ["preprocessed_output"],
            {"context_tokens": context_ids, "speech_tokens": speech_ids},
        )[0]
        audio = codec.decode(speech_tokens, context_tokens)
        waveform = audio.detach().float().cpu().numpy().reshape(-1)
        wall_ms = (time.perf_counter() - started) * 1000.0
        output = args.output_dir / f"{name}.wav"
        scipy.io.wavfile.write(output, 48000, np.clip(waveform, -1.0, 1.0))
        results.append(
            {
                "name": name,
                "wall_ms": wall_ms,
                "audio_seconds": waveform.size / 48000.0,
                "rtf": wall_ms / 1000.0 / (waveform.size / 48000.0),
                "sample_rate": 48000,
                "samples": int(waveform.size),
                "speech_token_count": int(speech_ids.size),
                "audio_sha256_f32": audio_sha256(waveform),
                "audio_out": str(output),
                "processor_latent": {
                    "shape": list(latent.shape),
                    "sum": float(np.sum(latent, dtype=np.float64)),
                    "sum_sq": float(np.sum(np.square(latent), dtype=np.float64)),
                },
            }
        )
        print(json.dumps(results[-1]))

    summary = {
        "family": "mira_tts",
        "implementation": "upstream_python",
        "model_dir": str(model_dir),
        "reference": str(args.reference.resolve()),
        "results": results,
    }
    args.summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(args.summary_file)


if __name__ == "__main__":
    main()
