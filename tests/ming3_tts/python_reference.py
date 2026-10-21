#!/usr/bin/env python3
"""Run the official Ming pipeline; optionally save AR/flow/VAE boundaries."""

import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import soundfile as sf
import torch
from safetensors.torch import load_file, save_file

from infer import seed_everything
from ming_tts import MingTTS
from fm.CFM import get_epss_timesteps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/audiocpp_cli"))
from run_audiocpp_cli_path_tests import ResourceMonitor


class ComponentCapture:
    def __init__(self, model):
        self.model = model
        self.tensors = {}
        self.flow_count = 0
        self.prompt_count = 0
        self.decode_count = 0
        self.originals = {}

    def __enter__(self):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            t = get_epss_timesteps(10, next(self.model.parameters()).device, torch.bfloat16)
            t = t - (torch.cos(torch.pi / 2 * t) - 1 + t)
        self.tensors["sampler.timesteps"] = t.detach().cpu().contiguous()
        def prepare(*args, **kwargs):
            result = self.originals["prepare"](*args, **kwargs)
            prefix = f"prompt.{self.prompt_count}"
            self.tensors[prefix + ".ids"] = result[0].detach().cpu().contiguous()
            self.tensors[prefix + ".embeddings"] = result[1].detach().cpu().contiguous()
            for index, speaker in enumerate(kwargs.get("spk_emb") or []):
                self.tensors[prefix + f".speaker.{index}"] = speaker.detach().cpu().contiguous()
            if kwargs.get("prompt_latent") is not None:
                self.tensors[prefix + ".latents"] = kwargs["prompt_latent"].detach().cpu().contiguous()
            self.prompt_count += 1
            return result

        def flow(*args, **kwargs):
            prefix = f"flow.{self.flow_count}"
            for key in ("noise", "c", "latent_history", "sde_noise"):
                self.tensors[prefix + "." + key] = kwargs[key].detach().cpu().contiguous()
            result = self.originals["flow"](*args, **kwargs)
            self.tensors[prefix + ".latent"] = result.detach().cpu().contiguous()
            self.flow_count += 1
            return result

        def decode(latent, *args, **kwargs):
            prefix = f"decode.{self.decode_count}"
            self.tensors[prefix + ".latent"] = latent.detach().cpu().contiguous()
            result = self.originals["decode"](latent, *args, **kwargs)
            self.tensors[prefix + ".waveform"] = result[0].detach().cpu().contiguous()
            self.decode_count += 1
            return result

        def head_output(module, inputs, result):
            prefix = f"decode.{self.decode_count}"
            self.tensors[prefix + ".hidden"] = inputs[0].detach().cpu().contiguous()
            self.tensors[prefix + ".hidden_strides"] = torch.tensor(inputs[0].stride())
            self.tensors[prefix + ".spectrum"] = result[1].transpose(1, 2).detach().cpu().contiguous()
            log_magnitude, phase = result[1].chunk(2, dim=1)
            self.tensors[prefix + ".magnitude"] = log_magnitude.exp().detach().cpu().contiguous()
            self.tensors[prefix + ".cos_phase"] = phase.cos().detach().cpu().contiguous()
            window = module.istft.window
            frames = result[1].shape[-1]
            envelope = torch.nn.functional.fold(
                window.square().expand(1, frames, -1).transpose(1, 2),
                output_size=(1, (frames - 1) * module.istft.hop_length + window.numel()),
                kernel_size=(1, window.numel()), stride=(1, module.istft.hop_length))
            self.tensors[prefix + ".envelope"] = envelope.detach().cpu().contiguous()

        def decoder_input(module, inputs, kwargs):
            self.tensors[f"decode.{self.decode_count}.input_embeddings"] = kwargs["inputs_embeds"].detach().cpu().contiguous()

        def audio_qkv(module, inputs, result):
            prefix = f"decode.{self.decode_count}"
            self.tensors[prefix + ".qkv_input"] = inputs[0].detach().cpu().contiguous()
            self.tensors[prefix + ".qkv_input_strides"] = torch.tensor(inputs[0].stride())
            self.tensors[prefix + ".q_projection"] = result.detach().cpu().contiguous()
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            attention = self.model.audio.decoder.decoder.layers[0].self_attn
            q = result.view(result.shape[0], result.shape[1], attention.num_heads, attention.head_dim).transpose(1, 2)
            rotated = apply_rotary_pos_emb(q, q, *self.rotary_output)[0]
            self.tensors[prefix + ".q_rotary"] = rotated.detach().cpu().contiguous()

        def audio_rotary(module, inputs, result):
            self.rotary_output = result
            for name, value in zip(("cos", "sin"), result):
                self.tensors[f"decode.{self.decode_count}.rotary.{name}"] = value.detach().cpu().contiguous()

        def audio_projection(module, inputs, result, name):
            prefix = f"decode.{self.decode_count}.{name}"
            self.tensors[prefix + ".input"] = inputs[0].detach().cpu().contiguous()
            self.tensors[prefix + ".input_strides"] = torch.tensor(inputs[0].stride())
            self.tensors[prefix + ".output"] = result.detach().cpu().contiguous()
            self.tensors[prefix + ".output_strides"] = torch.tensor(result.stride())

        self.originals = {"prepare": self.model.prepare_input_embed,
                          "flow": self.model.flowloss.cfm.sample,
                          "decode": self.model.audio.decode}
        self.model.prepare_input_embed = prepare
        self.model.flowloss.cfm.sample = flow
        self.model.audio.decode = decode
        self.head_hook = self.model.audio.decoder.head.register_forward_hook(head_output)
        self.decoder_hook = self.model.audio.decoder.decoder.register_forward_pre_hook(decoder_input, with_kwargs=True)
        self.audio_qkv_hook = self.model.audio.decoder.decoder.layers[0].self_attn.q_proj.register_forward_hook(audio_qkv)
        self.audio_rotary_hook = self.model.audio.decoder.decoder.rotary_emb.register_forward_hook(audio_rotary)
        block = self.model.audio.decoder.decoder.layers[0]
        self.audio_projection_hooks = [module.register_forward_hook(
            lambda module, inputs, result, name=name: audio_projection(module, inputs, result, name))
            for name, module in (("o_projection", block.self_attn.o_proj),
                                 ("mlp_gate", block.mlp.gate_proj),
                                 ("mlp_up", block.mlp.up_proj),
                                 ("mlp_down", block.mlp.down_proj))]
        return self

    def __exit__(self, *exc):
        self.head_hook.remove()
        self.decoder_hook.remove()
        self.audio_qkv_hook.remove()
        self.audio_rotary_hook.remove()
        for hook in self.audio_projection_hooks:
            hook.remove()
        self.model.prepare_input_embed = self.originals["prepare"]
        self.model.flowloss.cfm.sample = self.originals["flow"]
        self.model.audio.decode = self.originals["decode"]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture", action="store_true",
                        help="Capture component boundaries; timings are instrumented, not benchmarks")
    parser.add_argument("--only", help="Run one named request from the request list")
    parser.add_argument("--speaker-reference", type=Path,
                        help="Capture the official CAMPPlus input and embedding without AR generation")
    parser.add_argument("--encoder-reference", type=Path,
                        help="Capture the official reference VAE input, posterior noise and reconstruction")
    parser.add_argument("--decode-fixture", type=Path,
                        help="Replay captured AR latents through the official VAE only")
    parser.add_argument("--flow-fixture", type=Path,
                        help="Validate one isolated flow component call using captured inputs")
    parser.add_argument("--attention-backend", choices=("default", "math", "eager"), default="default")
    parser.add_argument("--log", action="store_true", required=True)
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if args.only:
        requests = [request for request in requests if request["name"] == args.only]
        if not requests:
            raise ValueError(f"request not found: {args.only}")
    if args.decode_fixture and len(requests) != 1:
        raise ValueError("--decode-fixture requires a single selected request")
    if args.attention_backend == "eager" and not args.decode_fixture:
        raise ValueError("Eager attention comparison requires --decode-fixture")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.speaker_reference:
        import torchaudio
        from spkemb_extractor import SpkembExtractor

        start = time.perf_counter()
        waveform, sample_rate = torchaudio.load(args.speaker_reference, backend="soundfile")
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        features = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, dither=0, sample_frequency=16000)
        features = features - features.mean(dim=0, keepdim=True)
        embedding = SpkembExtractor(str(args.model_dir / "campplus.onnx"))(waveform)
        save_file({"speaker.features": features.contiguous(), "speaker.embedding": embedding.contiguous()},
                  args.output_dir / "speaker.safetensors")
        torch.save(embedding, args.output_dir / "speaker.pt")
        provenance = {"reference_sha256": hashlib.sha256(args.speaker_reference.read_bytes()).hexdigest(),
                      "onnx_sha256": hashlib.sha256((args.model_dir / "campplus.onnx").read_bytes()).hexdigest(),
                      "frames": features.shape[0], "session.wall_ms": (time.perf_counter() - start) * 1000}
        (args.output_dir / "speaker.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance), flush=True)
        return
    model = MingTTS(str(args.model_dir), device="cuda:0", text_normalization=False)
    if args.encoder_reference:
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution

        waveform, _ = model._load_prompt([str(args.encoder_reference)])
        waveform = waveform.cuda().to(torch.bfloat16)
        seed_everything(1895)
        start = time.perf_counter()
        encoder_head = {}
        def capture_encoder_head(module, inputs, result):
            encoder_head["encoder.head_input"] = inputs[0].detach().cpu().contiguous()
            encoder_head["encoder.head_input_strides"] = torch.tensor(inputs[0].stride())
        hook = model.model.audio.encoder.fc3.register_forward_hook(capture_encoder_head)
        try:
            parameters = model.model.audio.encoder(waveform)[0]
        finally:
            hook.remove()
        posterior = OobleckDiagonalGaussianDistribution(parameters.transpose(1, 2))
        state = torch.cuda.get_rng_state()
        noise = torch.randn(posterior.mean.shape, device="cuda", dtype=posterior.mean.dtype)
        torch.cuda.set_rng_state(state)
        latent = posterior.sample().transpose(1, 2)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reconstruction = model.model.audio.decode(latent, use_cache=False, last_chunk=True)[0]
        torch.cuda.synchronize()
        save_file({**encoder_head, "encoder.waveform": waveform.cpu().contiguous(),
                   "encoder.noise": noise.cpu().contiguous(),
                   "encoder.parameters": parameters.cpu().contiguous(),
                   "encoder.latent": latent.cpu().contiguous(),
                   "decode.0.latent": latent.cpu().contiguous(),
                   "decode.0.waveform": reconstruction.cpu().contiguous()}, args.output_dir / "encoder.safetensors")
        provenance = {"reference_sha256": hashlib.sha256(args.encoder_reference.read_bytes()).hexdigest(),
                      "seed": 1895, "samples": waveform.numel(), "latent_frames": latent.shape[1],
                      "session.wall_ms": (time.perf_counter() - start) * 1000}
        (args.output_dir / "encoder.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(provenance), flush=True)
        return
    provenance = {"reference_commit": subprocess.check_output(
        ["git", "-C", str(args.reference_dir), "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(), "capture": args.capture,
        "attention_backend": args.attention_backend,
        "vae_rotary_inv_freq": model.model.audio.decoder.decoder.rotary_emb.inv_freq.float().cpu().tolist(),
        "vae_rotary_dtype": str(model.model.audio.decoder.decoder.rotary_emb.inv_freq.dtype),
        "requests": requests}
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    measurements = []
    for index, item in enumerate(requests):
        request = dict(item)
        name = request.pop("name")
        seed_everything(request.pop("seed", 1895))
        fragments = request.pop("fragments", None)
        if fragments is not None and (not fragments or not all(isinstance(text, str) and text.strip() for text in fragments)
                                      or request.get("raw_text") is not True):
            raise ValueError("Explicit comparison fragments require nonempty strings and raw_text=true")
        monitor = ResourceMonitor(os.getpid(), 0, 20)
        stop_monitor = threading.Event()

        def sample_resources():
            while not stop_monitor.is_set():
                monitor.sample()
                stop_monitor.wait(0.02)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with ThreadPoolExecutor(max_workers=1) as executor:
            resource_job = executor.submit(sample_resources)
            try:
                start = time.perf_counter()
                attention_context = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                                     if args.attention_backend == "math" else nullcontext())
                if args.attention_backend == "eager":
                    def eager_attention(module, inputs, kwargs):
                        return inputs, {**kwargs, "output_attentions": True}
                    attention_context = model.model.audio.decoder.decoder.register_forward_pre_hook(
                        eager_attention, with_kwargs=True)
                capture = ComponentCapture(model.model) if args.capture else nullcontext()
                with attention_context, capture:
                    if args.flow_fixture:
                        fixture = load_file(args.flow_fixture, device="cuda")
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            velocity = model.model.flowloss.cfm.model.forward_with_cfg(
                                fixture["flow.0.noise"], torch.tensor(0.0, device="cuda", dtype=torch.bfloat16),
                                fixture["flow.0.c"], fixture["flow.0.latent_history"], 1)
                            block = model.model.flowloss.cfm.model.blocks[0]
                            block_inputs = []
                            block_outputs = []
                            norm_outputs = []
                            attention_outputs = []
                            projections = {}
                            input_hook = block.register_forward_pre_hook(lambda module, inputs: block_inputs.append(inputs[0].detach().cpu().contiguous()))
                            output_hook = block.register_forward_hook(lambda module, inputs, output: block_outputs.append(output.detach().cpu().contiguous()))
                            norm_hook = block.norm1.register_forward_hook(lambda module, inputs, output: norm_outputs.append(output.detach().cpu().contiguous()))
                            attention_hook = block.attn.register_forward_hook(lambda module, inputs, output: attention_outputs.append(output.detach().cpu().contiguous()))
                            projection_hooks = [getattr(block.attn, "to_" + name).register_forward_hook(
                                lambda module, inputs, output, key=name: projections.update({key: output.detach().cpu().contiguous()}))
                                for name in ("q", "k", "v")]
                            try:
                                velocity_half = model.model.flowloss.cfm.model.forward_with_cfg(
                                    fixture["flow.0.noise"], torch.tensor(0.5, device="cuda", dtype=torch.bfloat16),
                                    fixture["flow.0.c"], fixture["flow.0.latent_history"], 1)
                            finally:
                                input_hook.remove()
                                output_hook.remove()
                                norm_hook.remove()
                                attention_hook.remove()
                                for hook in projection_hooks:
                                    hook.remove()
                            t_half = torch.full((2,), 0.5, device="cuda", dtype=torch.bfloat16)
                            time_module = model.model.flowloss.cfm.model.t_embedder
                            time_embedding = time_module.time_embed(t_half).to(t_half.dtype)
                            time_projected = time_module(t_half)
                        if not args.capture:
                            raise ValueError("--flow-fixture requires --capture")
                        capture.tensors["flow.unit.velocity"] = velocity.detach().cpu().contiguous()
                        capture.tensors["flow.unit.velocity_half"] = velocity_half.detach().cpu().contiguous()
                        capture.tensors["flow.unit.time_embedding"] = time_embedding.detach().cpu().contiguous()
                        capture.tensors["flow.unit.time_projected"] = time_projected.detach().cpu().contiguous()
                        capture.tensors["flow.unit.block0.input"] = block_inputs[0]
                        capture.tensors["flow.unit.block0.output"] = block_outputs[0]
                        capture.tensors["flow.unit.block0.normalized"] = norm_outputs[0]
                        capture.tensors["flow.unit.block0.attention"] = attention_outputs[0]
                        for key, value in projections.items():
                            capture.tensors["flow.unit.block0." + key] = value
                        from fm.modules import apply_rotary_pos_emb
                        freqs, scale = model.model.flowloss.cfm.model.rotary_embed.forward_from_seq_len(3)
                        for key in ("q", "k"):
                            projected = projections[key].cuda().view(2, 3, 16, 64).transpose(1, 2)
                            capture.tensors["flow.unit.block0." + key + "_rope"] = apply_rotary_pos_emb(
                                projected, freqs, scale).detach().cpu().contiguous()
                        for key in ("c", "latent_history", "noise", "sde_noise", "latent"):
                            capture.tensors["flow.0." + key] = fixture["flow.0." + key].cpu().contiguous()
                    if args.decode_fixture:
                        latent = load_file(args.decode_fixture)["decode.0.latent"].cuda()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            waveform = model.model.audio.decode(latent, use_cache=False, last_chunk=True)[0].view(1, -1)
                    elif fragments is not None:
                        waveform = torch.cat([model.synthesize(**{**request, "text": fragment})
                                              for fragment in fragments], dim=-1)
                    else:
                        waveform = model.synthesize(**request)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            finally:
                stop_monitor.set()
                try:
                    resource_job.result()
                finally:
                    monitor.close()
        duration = waveform.shape[-1] / model.sample_rate
        row = {"name": name, "request_index": index, "instrumented": args.capture,
               "session.wall_ms": elapsed, "audio.duration_ms": duration * 1000,
               "rtf": elapsed / (duration * 1000),
               "cuda.peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
               "cuda.peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
               **monitor.metrics.to_json()}
        sf.write(args.output_dir / (name + ".wav"), waveform.float().cpu().view(-1).numpy(),
                 model.sample_rate, subtype="FLOAT")
        if args.capture:
            save_file(capture.tensors, args.output_dir / (name + ".safetensors"))
        measurements.append(row)
        print(json.dumps(row), flush=True)
    (args.output_dir / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")


if __name__ == "__main__":
    main()
