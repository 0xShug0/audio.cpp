#!/usr/bin/env python3
"""Per-request benchmark for the official GLM-TTS Python implementation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
import torchaudio


class IdentityNormalizer:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def normalize(self, text: str) -> str:
        return text


def install_wetext_identity_stub() -> None:
    modules = {
        "tn": types.ModuleType("tn"),
        "tn.chinese": types.ModuleType("tn.chinese"),
        "tn.chinese.normalizer": types.ModuleType(
            "tn.chinese.normalizer"
        ),
        "tn.english": types.ModuleType("tn.english"),
        "tn.english.normalizer": types.ModuleType(
            "tn.english.normalizer"
        ),
    }
    modules["tn.chinese.normalizer"].Normalizer = IdentityNormalizer
    modules["tn.english.normalizer"].Normalizer = IdentityNormalizer
    sys.modules.update(modules)


def load_reference_module(root: Path):
    entry = root / "glmtts_inference.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    install_wetext_identity_stub()
    if not hasattr(torch, "npu"):
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "glmtts_reference_inference", entry
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {entry}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def flatten_token_chunks(chunks) -> list[int]:
    output: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, (int, float)):
            output.append(int(chunk))
        elif torch.is_tensor(chunk) and chunk.ndim == 0:
            output.append(int(chunk.item()))
        else:
            output.extend(int(value) for value in chunk)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--voice-ref", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Replace upstream RAS sampling with argmax for token parity.",
    )
    args = parser.parse_args()

    root = args.reference_root.resolve()
    request_file = args.request_file.resolve()
    voice_ref = args.voice_ref.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    module = load_reference_module(root)
    frontend, text_frontend, _, llama, flow = module.load_models(
        use_phoneme=False, sample_rate=24000
    )
    if args.greedy:
        import llm.glmtts as glmtts_module

        glmtts_module.common.ras_sampling = (
            lambda weighted_scores, decoded_tokens, sampling, **kwargs:
            weighted_scores.argmax()
        )
    requests = json.loads(
        request_file.read_text(encoding="utf-8")
    )["requests"]
    results = []
    for request in requests:
        name = request["name"]
        seed = int(request.get("seed", 0))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        started = time.perf_counter()

        prompt_text = text_frontend.text_normalize(
            args.reference_text
        )
        target_text = text_frontend.text_normalize(request["text"])
        prompt_text_tokens = frontend._extract_text_token(
            prompt_text + " "
        )
        prompt_speech_tokens = frontend._extract_speech_token(
            [str(voice_ref)]
        )
        speech_features = frontend._extract_speech_feat(
            str(voice_ref), sample_rate=24000
        )
        speaker_embedding = frontend._extract_spk_embedding(
            str(voice_ref)
        )
        prompt_speech_list = (
            prompt_speech_tokens.squeeze().tolist()
        )
        cache = {
            "cache_text": [prompt_text],
            "cache_text_token": [prompt_text_tokens],
            "cache_speech_token": [prompt_speech_list],
            "use_cache": True,
        }
        flow_prompt_tokens = torch.tensor(
            [prompt_speech_list], dtype=torch.int32
        ).to(module.DEVICE)
        waveform, mel, token_chunks, normalized = (
            module.generate_long(
                frontend=frontend,
                text_frontend=text_frontend,
                llm=llama,
                flow=flow,
                text_info=[name, target_text],
                cache=cache,
                embedding=speaker_embedding,
                seed=seed,
                flow_prompt_token=flow_prompt_tokens,
                speech_feat=speech_features,
                device=module.DEVICE,
                use_phoneme=False,
            )
        )
        synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        output_path = out_dir / f"{name}.wav"
        torchaudio.save(str(output_path), waveform.cpu(), 24000)
        frames = int(waveform.shape[-1])
        audio_seconds = frames / 24000.0
        peak_vram_mib = (
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if torch.cuda.is_available()
            else 0.0
        )
        result = {
            "name": name,
            "seed": seed,
            "wall_ms": wall_ms,
            "audio_seconds": audio_seconds,
            "rtf": wall_ms / 1000.0 / audio_seconds,
            "frames": frames,
            "mel_frames": int(mel.shape[-1]) if mel is not None else 0,
            "prompt_text_ids": [
                int(value)
                for value in prompt_text_tokens.squeeze().tolist()
            ],
            "prompt_speech_ids": [
                int(value) for value in prompt_speech_list
            ],
            "generated_speech_ids": flatten_token_chunks(
                token_chunks
            ),
            "normalized_text": (
                normalized
                if isinstance(
                    normalized,
                    (str, int, float, bool, list, dict, type(None)),
                )
                else str(normalized)
            ),
            "peak_vram_mib": peak_vram_mib,
            "audio_out": str(output_path),
        }
        results.append(result)
        print(f"request={name}")
        print(f"wall_ms={wall_ms}")
        print(f"audio_seconds={audio_seconds}")
        print(f"rtf={result['rtf']}")
        print(f"frames={frames}")
        print(f"peak_vram_mib={peak_vram_mib}")
        print(f"audio_out={output_path}")

    result_path = out_dir / "results.json"
    result_path.write_text(
        json.dumps({"requests": results}, indent=2),
        encoding="utf-8",
    )
    print(f"results_out={result_path}")


if __name__ == "__main__":
    main()
