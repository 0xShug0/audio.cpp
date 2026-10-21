#!/usr/bin/env python3
"""Run the official Ming3 Audio Thinker and preserve its AR input/output boundary."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import soundfile as sf
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/audiocpp_cli"))
from run_audiocpp_cli_path_tests import ResourceMonitor
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference/Ming-3.0-audio-flash/examples"))
from transformers_audio import build_messages, process_messages
from types import SimpleNamespace


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true",
                        help="Save processor tensors, projected audio, AR prompt embeddings and generated IDs")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--log", action="store_true", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if isinstance(requests, dict):
        requests = requests["requests"]
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation=args.attention).eval()
    manifest = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(model.dtype),
        "attention": args.attention,
        "config_sha256": hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
        "requests": [],
        "parameter_dtypes": {name: str(model.get_parameter(name).dtype) for name in (
            "model.word_embeddings.weight", "lm_head.weight",
            "model.layers.0.input_layernorm.weight", "model.layers.0.attention.A_log",
            "model.layers.0.attention.dt_bias", "model.layers.1.mlp.gate.weight")},
        "buffer_dtypes": {"model.layers.1.mlp.gate.expert_bias":
                          str(model.get_buffer("model.layers.1.mlp.gate.expert_bias").dtype)},
    }
    monitor = ResourceMonitor(os.getpid(), 0, 50)
    stop = threading.Event()

    def sample_resources():
        while not stop.is_set():
            monitor.sample()
            stop.wait(0.05)

    executor = ThreadPoolExecutor(max_workers=1)
    sampling_job = executor.submit(sample_resources)
    try:
        for request in requests:
            request = {**request, **request.get("options", {})}
            name = request.get("name", request.get("id"))
            if not name:
                raise ValueError("Each request requires a name or id")
            torch.manual_seed(request.get("seed", 42))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            audio = Path(request["audio"]) if "audio" in request else None
            if audio:
                chat = "system_prompt" in request
                messages = build_messages(SimpleNamespace(
                    task="chat" if chat else "asr", prompt_as_system=False,
                    prompt=request.get("prompt", None if chat else processor.ASR_PROMPT),
                    system_prompt=request.get("system_prompt", "")), str(audio.resolve()))
                inputs = process_messages(processor, messages, request.get("enable_thinking", False))
                info = sf.info(audio)
                duration = info.frames / info.samplerate
            else:
                prompt = processor.tokenizer.apply_chat_template(
                    [{"role": "user", "content": request["text"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = processor.tokenizer(prompt, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                duration = None
            inputs = inputs.to(model.device)
            if "audio_feats" in inputs:
                inputs["audio_feats"] = inputs["audio_feats"].to(model.dtype)
            tensors = {key: value.detach().cpu().contiguous() for key, value in inputs.items()}
            capture_handle = None
            if args.capture:
                def capture_prefill(module, positional, keywords):
                    if "prompt_embeddings" in tensors:
                        return
                    embeddings = keywords["inputs_embeds"]
                    tensors["prompt_embeddings"] = embeddings.detach().cpu().contiguous()
                    if audio:
                        mask = inputs["input_ids"] == model.config.audio_token_id
                        tensors["audio_embeddings"] = embeddings[mask].reshape(
                            1, -1, embeddings.shape[-1]).detach().cpu().contiguous()

                capture_handle = model.model.register_forward_pre_hook(capture_prefill, with_kwargs=True)
            try:
                output = model.generate(**inputs, max_new_tokens=request.get("max_tokens", 256), do_sample=False)
            finally:
                if capture_handle is not None:
                    capture_handle.remove()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - start) * 1000
            monitor.sample()
            generated = output[:, inputs["input_ids"].shape[1]:]
            text = processor.decode(generated[0], skip_special_tokens=True).strip()
            tensors["generated_ids"] = generated.cpu().contiguous()
            (args.output_dir / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            if args.capture:
                save_file(tensors, str(args.output_dir / f"{name}.safetensors"))
            metrics = {
                "name": name, "text": text, "session.wall_ms": wall_ms,
                "input_tokens": inputs["input_ids"].shape[1], "generated_tokens": generated.shape[1],
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                "audio_duration_sec": duration,
                "rtf": wall_ms / (duration * 1000) if duration else None,
                "instrumented": args.capture,
                "capture_boundary": "actual_thinker_prefill_inputs" if args.capture else None,
                **monitor.metrics.to_json(),
            }
            manifest["requests"].append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            (args.output_dir / "metrics.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        stop.set()
        sampling_job.result()
        executor.shutdown()
        monitor.close()


if __name__ == "__main__":
    main()
