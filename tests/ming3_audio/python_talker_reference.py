#!/usr/bin/env python3
"""Run the official Talker; capture only its AR input and final latent/waveform boundaries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/audiocpp_cli"))
from run_audiocpp_cli_path_tests import ResourceMonitor


class TalkerCapture:
    def __init__(self, model):
        self.model = model
        self.tensors = {}
        self.prompts = []
        self.encoder_layer_calls = []
        self.original = model.generate_latents
        self.decode = model.audio_vae.decode
        self.inputs = model.input_embeddings

    def __enter__(self):
        self.encoder_hooks = [layer.register_forward_hook(
            lambda module, inputs, output, index=index: self.encoder_layer_calls.append(index))
            for index, layer in enumerate(self.model.audio_vae.encoder.encoder.layers)]
        self.encoder_boundary_hook = self.model.audio_vae.encoder.register_forward_hook(
            lambda module, inputs, output: self.tensors.update({
                "reference.encoder_parameters": output[0].detach().cpu().contiguous(),
                "reference.waveform": output[1].detach().cpu().contiguous()}))
        self.speaker_hook = self.model.spk_head.register_forward_pre_hook(
            lambda module, inputs: self.tensors.update({
                "reference.speaker_features": inputs[0].detach().cpu().contiguous()}))

        def prompt(*args, **kwargs):
            name = f"chunk.{len(self.prompts)}"
            handle = self.model.model.get_input_embeddings().register_forward_pre_hook(
                lambda module, inputs: self.tensors.update({
                    name + ".prompt_ids": inputs[0].detach().to(torch.int32).cpu().contiguous()}))
            try:
                result = self.inputs(*args, **kwargs)
            finally:
                handle.remove()
            self.prompts.append({"text": args[0], "user_prompt": args[1], "instruction": args[2],
                                 "reference_text": args[4]})
            return result

        def generate(embeddings, reference, max_steps, cfg, sigma, temperature):
            name = f"chunk.{len(self.prompts) - 1}"
            offset_words = torch.cuda.default_generators[0].get_offset()
            self.tensors[name + ".initial_rng_offset_words"] = torch.tensor(offset_words, dtype=torch.int64)
            self.tensors[name + ".initial_rng_offset_blocks"] = torch.tensor(offset_words // 4, dtype=torch.int64)
            self.tensors[name + ".prompt_embeddings"] = embeddings.detach().cpu().contiguous()
            if reference is not None:
                self.tensors[name + ".reference_latents"] = reference.detach().cpu().contiguous()
            result = self.original(embeddings, reference, max_steps, cfg, sigma, temperature)
            self.tensors[name + ".generated_latents"] = result.detach().cpu().contiguous()
            return result

        def decode(latents, *args, **kwargs):
            result = self.decode(latents, *args, **kwargs)
            self.tensors[f"chunk.{len(self.prompts) - 1}.waveform"] = result[0].detach().cpu().contiguous()
            return result

        self.model.input_embeddings = prompt
        self.model.generate_latents = generate
        self.model.audio_vae.decode = decode
        return self

    def __exit__(self, *exc):
        for handle in self.encoder_hooks:
            handle.remove()
        self.encoder_boundary_hook.remove()
        self.speaker_hook.remove()
        self.model.generate_latents = self.original
        self.model.audio_vae.decode = self.decode
        self.model.input_embeddings = self.inputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--decode-fixture", type=Path,
                        help="Teacher-force captured AR output through the official VAE only")
    parser.add_argument("--vae-attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--log", action="store_true", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model_dir.resolve()))
    from talker import MingTalker

    model = MingTalker.from_pretrained(args.model_dir / "talker", dtype=torch.bfloat16, device="cuda:0")
    model.audio_vae.decoder.decoder.set_attn_implementation(args.vae_attention)
    if args.capture and args.decode_fixture:
        model.audio_vae.decoder.upsampling.upsampler.register_forward_hook(
            lambda module, inputs, output: print(json.dumps({"vae.upsampler.input_dtype": str(inputs[0].dtype),
                                                           "vae.upsampler.output_dtype": str(output.dtype)}), flush=True))
        model.audio_vae.decoder.decoder.register_forward_pre_hook(
            lambda module, inputs, kwargs: print(json.dumps({"vae.decoder.input_dtype": str(kwargs["inputs_embeds"].dtype),
                                                            "vae.rope.inv_freq_dtype": str(module.rotary_emb.inv_freq.dtype)}), flush=True),
            with_kwargs=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "requests": []}
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if isinstance(requests, dict):
        requests = requests["requests"]
    for request in requests:
        request = {**request, **request.get("options", {})}
        name = request.get("name", request.get("id"))
        if not name:
            raise ValueError("Each request requires a name or id")
        reference = None
        reference_audio = request.get("reference_audio", request.get("voice_ref"))
        if reference_audio:
            samples, sample_rate = sf.read(reference_audio, dtype="float32", always_2d=True)
            reference = {"samples": np.mean(samples, axis=1), "sample_rate": sample_rate}
        monitor = ResourceMonitor(os.getpid(), 0, 20)
        stop = threading.Event()

        def sample_resources():
            while not stop.is_set():
                monitor.sample()
                stop.wait(0.02)

        capture = TalkerCapture(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with ThreadPoolExecutor(max_workers=1) as executor:
            job = executor.submit(sample_resources)
            try:
                start = time.perf_counter()
                if args.capture and not args.decode_fixture:
                    capture.__enter__()
                if args.decode_fixture:
                    tensors = load_file(args.decode_fixture, device="cuda:0")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        waveform = model.audio_vae.decode(tensors["chunk.0.generated_latents"], use_cache=False, last_chunk=True)[0]
                    result = {"audio": waveform.detach().float().cpu(), "sr": model.audio_vae.config.sample_rate}
                else:
                    result = model.synthesize(
                        request["text"], ip_name=request.get("voice_id", "US_James"), reference=reference,
                        prompt_text=request.get("reference_text"), language=request.get("language"),
                        max_steps=request.get("max_audio_tokens", 1000), seed=request.get("seed", 42),
                        cfg=request.get("guidance_scale", 2), sigma=request.get("sigma", 0.25),
                        temperature=request.get("temperature", 0))
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - start) * 1000
            finally:
                if args.capture and not args.decode_fixture:
                    capture.__exit__()
                stop.set()
                job.result()
                monitor.close()
        waveform = result["audio"].reshape(-1).numpy()
        sample_rate = int(result["sr"])
        sf.write(args.output_dir / f"{name}.wav", waveform, sample_rate, subtype="FLOAT")
        if args.capture and not args.decode_fixture:
            save_file(capture.tensors, str(args.output_dir / f"{name}.safetensors"))
        duration = len(waveform) / sample_rate
        record = {**request, "session.wall_ms": wall_ms, "audio_duration_sec": duration,
                  "rtf": wall_ms / (duration * 1000), "instrumented": args.capture,
                  "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
                  **monitor.metrics.to_json(), "decode_only": bool(args.decode_fixture),
                  "vae_attention": args.vae_attention, "prompts": capture.prompts,
                  "reference_encoder_modules": len(model.audio_vae.encoder.encoder.layers),
                  "reference_encoder_config_layers": model.audio_vae.encoder.encoder.config.num_hidden_layers,
                  "reference_encoder_shared_config": model.audio_vae.encoder.encoder.config is model.audio_vae.encoder.aggregator.config,
                  "reference_encoder_executed_layers": capture.encoder_layer_calls}
        manifest["requests"].append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (args.output_dir / "metrics.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
