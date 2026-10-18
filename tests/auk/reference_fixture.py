#!/usr/bin/env python3
"""Capture AuK reference boundaries for ordered native parity testing."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import statistics

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("reference/AuK"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--sway", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", action="store_true", required=True)
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--qkv-fixture", type=Path)
    parser.add_argument("--audio-replay-fixture", type=Path)
    parser.add_argument("--audio-features", type=Path)
    parser.add_argument("--full-precision-reduction", action="store_true")
    parser.add_argument("--decode-fixture", type=Path)
    parser.add_argument("--flow-fixture", type=Path)
    parser.add_argument("--flow-cfg", action="store_true")
    parser.add_argument("--flow-sample", action="store_true")
    parser.add_argument("--benchmark-runs", type=int, default=0)
    args = parser.parse_args()
    if bool(args.audio_replay_fixture) != bool(args.audio_features):
        parser.error("--audio-replay-fixture and --audio-features must be supplied together")
    if args.decode_fixture and args.flow_fixture:
        parser.error("choose only one component fixture")
    if args.flow_cfg and not args.flow_fixture:
        parser.error("--flow-cfg requires --flow-fixture")
    if args.flow_sample and not args.flow_fixture:
        parser.error("--flow-sample requires --flow-fixture")
    if args.benchmark_runs < 0 or (args.benchmark_runs and args.flow_fixture):
        parser.error("--benchmark-runs requires a nonnegative count and VAE-only or full inference")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "reference.log"), logging.StreamHandler()],
        force=True,
    )
    sys.path.insert(0, str(args.reference.resolve() / "src"))
    from auk.infer.infer_auk import AukInfer, save_audio

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.full_precision_reduction:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.flow_fixture:
        from auk.model.flux2_edit import Flux2Edit
        from omegaconf import OmegaConf
        manifest = json.loads((args.flow_fixture / "manifest.json").read_text())
        if manifest["dtype"] != "fp32":
            raise ValueError("flow embedding fixture currently targets FP32 only")
        config = OmegaConf.load(args.model_dir / "config.yaml")
        arch = OmegaConf.to_container(config.model.arch, resolve=True)
        arch["attn_backend"] = "torch"
        model = Flux2Edit(**arch, latent_dim=64)
        checkpoint = load_file(args.model_dir / "auk_base.safetensors")
        model.load_state_dict({key.removeprefix("transformer."): value
                               for key, value in checkpoint.items() if key.startswith("transformer.")}, strict=True)
        del checkpoint
        model = model.cuda().eval()
        fixture = load_file(args.flow_fixture / "reference.safetensors")
        reference_latents = fixture["fm.first.ref"].cuda()
        reference_mask = fixture["fm.first.ref_mask"].cuda() if reference_latents.shape[1] else None
        if not fixture["fm.first.c_mask"].all():
            raise ValueError("flow fixture currently requires all-valid text tokens")
        if args.flow_sample:
            from torchdiffeq import odeint
            steps, strength = manifest["steps"], manifest["cfg"]
            times = torch.linspace(0, 1, steps + 1, device="cuda", dtype=torch.float32)
            times = times + manifest.get("sway", -1.0) * (torch.cos(torch.pi / 2 * times) - 1 + times)
            text = fixture["fm.first.text"].cuda()
            mask = fixture["fm.first.c_mask"].cuda()
            states = {"times": times.cpu()}
            with torch.inference_mode():
                def velocity(time, latent):
                    prediction = model(x=latent, text=text, time=time, c_mask=mask, cfg_infer=True, cache=True,
                                       ref=reference_latents, ref_mask=reference_mask)
                    conditional, unconditional = prediction.chunk(2, dim=0)
                    return conditional + (conditional - unconditional) * strength
                trajectory = odeint(velocity, fixture["fm.first.x"].cuda(), times, method="euler")
                states["trajectory"] = trajectory.cpu().contiguous()
                states["fm.output"] = trajectory[-1].cpu().contiguous().clone()
                expected = fixture["vae.normalized_input"] if reference_latents.shape[1] else fixture["fm.output"]
                error = (states["fm.output"] - expected).abs().max().item()
                logging.info("Flow-only replay versus full Python output max_error=%g", error)
                torch.testing.assert_close(states["fm.output"], expected, rtol=0, atol=0)
            save_file(states, args.output_dir / "flow-trajectory.safetensors")
            return
        stages = {}
        with torch.inference_mode():
            stages["text"] = model.project_text(fixture["fm.first.text"].cuda()).cpu().contiguous()
            stages["audio"] = model.audio_embed(fixture["fm.first.x"].cuda()).cpu().contiguous()
            if reference_latents.shape[1]:
                stages["reference"] = model.audio_embed._embed(reference_latents, reference_mask).cpu().contiguous()
                stages["reference.unconditional"] = model.audio_embed._embed(
                    torch.zeros_like(reference_latents), reference_mask).cpu().contiguous()
            for index, step in enumerate((0.0, 0.25, 0.75)):
                time_embedding = model.time_embed(torch.tensor([step], device="cuda"))
                stages[f"time.{index}"] = time_embedding.cpu().contiguous()
                audio_out, audio_mask, prompt_len = model._embed_audio(
                    fixture["fm.first.x"].cuda(), reference_latents, False, None, reference_mask)
                text_out = stages["text"].cuda()
                text_mask = fixture["fm.first.c_mask"].cuda()
                if args.flow_cfg:
                    time_embedding = torch.cat((time_embedding, time_embedding), dim=0)
                    unconditional_audio, _, _ = model._embed_audio(
                        fixture["fm.first.x"].cuda(), reference_latents, True, None, reference_mask)
                    audio_out = torch.cat((audio_out, unconditional_audio), dim=0)
                    if audio_mask is not None:
                        audio_mask = torch.cat((audio_mask, audio_mask), dim=0)
                    text_mask = torch.cat((text_mask, text_mask), dim=0)
                    text_out = torch.cat((text_out, torch.zeros_like(text_out)), dim=0)
                handles = []
                first = model.transformer_blocks[0]
                for branch in ("x", "c"):
                    suffix = "" if branch == "x" else "_c"
                    for label, module in (
                        ("mod", getattr(first, "attn_norm_" + branch)),
                        ("qkv", getattr(first.attn, "to_qkv" + suffix)),
                        ("attn_output", first.attn.to_out[0] if branch == "x" else first.attn.to_out_c),
                        ("q_norm", first.attn.q_norm if branch == "x" else first.attn.c_q_norm),
                        ("k_norm", first.attn.k_norm if branch == "x" else first.attn.c_k_norm),
                        ("ff_output", getattr(first, "ff_" + branch)),
                    ):
                        name = f"first.{branch}.{label}.{index}"
                        def capture(module, inputs, output, name=name, label=label):
                            value = output[0] if isinstance(output, tuple) else output
                            if label in ("q_norm", "k_norm"):
                                value = value.transpose(1, 2)
                            stages[name] = value.detach().cpu().contiguous()
                        handles.append(module.register_forward_hook(capture))
                    name = f"first.{branch}.ff_input.{index}"
                    def capture_input(module, inputs, name=name):
                        stages[name] = inputs[0].detach().cpu().contiguous()
                    handles.append(getattr(first, "ff_" + branch).register_forward_pre_hook(capture_input))
                    name = f"first.{branch}.attention.{index}"
                    def capture_attention(module, inputs, name=name):
                        stages[name] = inputs[0].detach().cpu().contiguous()
                    projection = first.attn.to_out[0] if branch == "x" else first.attn.to_out_c
                    handles.append(projection.register_forward_pre_hook(capture_attention))
                for layer, block in enumerate(model.transformer_blocks):
                    text_out, audio_out = block(
                        audio_out, text_out, time_embedding,
                        mask=audio_mask, c_mask=text_mask,
                        rope=model.rotary_embed.forward_from_seq_len(audio_out.shape[1]),
                        c_rope=model.rotary_embed.forward_from_seq_len(stages["text"].shape[1]))
                    if layer == 0:
                        stages[f"block.audio.{index}"] = audio_out.cpu().contiguous()
                        stages[f"block.text.{index}"] = text_out.cpu().contiguous()
                        for handle in handles:
                            handle.remove()
                stages[f"double.audio.{index}"] = audio_out.cpu().contiguous()
                stages[f"double.text.{index}"] = text_out.cpu().contiguous()
                hidden = torch.cat((text_out, audio_out), dim=1)
                rope = model.rotary_embed.forward_from_seq_len(hidden.shape[1])
                single_mask = torch.cat((text_mask, audio_mask), dim=1) if audio_mask is not None else None
                for block in model.single_transformer_blocks:
                    hidden = block(hidden, time_embedding, mask=single_mask, rope=rope)
                stages[f"single.{index}"] = hidden.cpu().contiguous()
                projected = model.proj_out(model.norm_out(hidden[:, text_out.shape[1] + prompt_len:], time_embedding))
                direct = model(x=fixture["fm.first.x"].cuda(), text=fixture["fm.first.text"].cuda(),
                               time=torch.tensor([step], device="cuda"),
                               c_mask=fixture["fm.first.c_mask"].cuda(), cfg_infer=args.flow_cfg,
                               ref=reference_latents, ref_mask=reference_mask)
                torch.testing.assert_close(projected, direct, rtol=0, atol=0)
                stages[f"velocity.{index}"] = direct.cpu().contiguous()
                if args.flow_cfg:
                    conditional, unconditional = direct.chunk(2, dim=0)
                    stages[f"conditional.{index}"] = conditional.cpu().contiguous().clone()
                    stages[f"unconditional.{index}"] = unconditional.cpu().contiguous().clone()
                    stages[f"guided.{index}"] = (conditional + (conditional - unconditional) * 2.0).cpu().contiguous()
        save_file(stages, args.output_dir / ("flow-cfg.safetensors" if args.flow_cfg else "flow-embeddings.safetensors"))
        logging.info("Saved FP32 flow boundaries and velocity at three timesteps (CFG=%s); composition matches original forward exactly", args.flow_cfg)
        return

    if args.decode_fixture:
        from auk.model.vae import BigVGANFlowVAEConfig, load_vae_model
        from omegaconf import OmegaConf
        config = OmegaConf.load(args.model_dir / "config.yaml")
        vae_config = BigVGANFlowVAEConfig.from_dict(
            OmegaConf.to_container(config.model.vae.model_init_kwargs, resolve=True))
        vae = load_vae_model("BigVGANFlowVAE", vae_config,
                             str(args.model_dir / "vae.safetensors"), map_location="cpu").cuda().eval()
        fixture = load_file(args.decode_fixture / "reference.safetensors")
        stages = {}
        handles = []
        def save_stage(name):
            def hook(_module, _inputs, output):
                stages[name] = output.detach().cpu().contiguous().clone()
            return hook
        names = ("conv_pre", "ups.0.0", "resblocks.0.activations.0", "resblocks.0.convs1.0")
        for name, module in vae.named_modules():
            if name in names:
                handles.append(module.register_forward_hook(save_stage(name)))
        with torch.inference_mode():
            latents = vae.denormalize(fixture["fm.output"].cuda()).transpose(1, 2).contiguous()
            stages["input"] = latents.cpu()
            stages["waveform"] = vae.inference_from_latents(latents).cpu()
        for handle in handles:
            handle.remove()
        logging.info("Decoder rerun max waveform error: %.9g",
                     (stages["waveform"].flatten() - fixture["waveform"].flatten()).abs().max())
        save_file(stages, args.output_dir / "vae-stages.safetensors")
        if args.benchmark_runs:
            milliseconds = []
            with torch.inference_mode():
                # Match native timing: CPU normalized latents through CPU waveform;
                # exclude weight loading, graph construction and WAV serialization.
                for run in range(args.benchmark_runs + 2):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    decoder_input = vae.denormalize(fixture["fm.output"].cuda()).transpose(1, 2)
                    waveform = vae.inference_from_latents(decoder_input).cpu()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    if not torch.equal(waveform, stages["waveform"]):
                        raise RuntimeError("benchmark VAE output changed")
                    if run >= 2:
                        milliseconds.append(elapsed)
                        logging.info("VAE decode_ms=%.6f", elapsed)
            result = {"runs": args.benchmark_runs, "milliseconds": milliseconds,
                      "median_ms": statistics.median(milliseconds), "weights": "original F32",
                      "scope": "CPU normalized latents to CPU waveform; no capture hooks"}
            (args.output_dir / "vae-performance.json").write_text(json.dumps(result, indent=2) + "\n")
            logging.info("VAE median_ms=%.6f runs=%d", result["median_ms"], args.benchmark_runs)
        return

    engine = AukInfer(
        str(args.model_dir / "config.yaml"),
        str(args.model_dir / "auk_base.safetensors"),
        device="cuda",
        dtype=args.dtype,
        qwen_path=str(args.qwen_dir),
        cpu_offload=False,
    )
    if args.audio_replay_fixture:
        if args.dtype != "fp32":
            raise ValueError("audio feature replay requires FP32")
        fixture = load_file(args.audio_replay_fixture / "reference.safetensors")
        lengths = fixture["qwen.audio.feature_lens"].cuda()
        if lengths.numel() != 1:
            raise ValueError("audio feature replay requires one reference")
        frames = int(lengths.item())
        original = fixture["condition.input_features"][0, :, :frames].cuda()
        native = load_file(args.audio_features)["features"].cuda()
        if native.shape != original.shape:
            raise ValueError("replayed audio feature shape differs")
        expected = fixture["qwen.audio_output"].cuda()
        with torch.inference_mode():
            outputs = {}
            for name, features in (("original", original), ("native_features", native)):
                output = engine.model.text_encoder.audio_tower(
                    features, feature_lens=lengths).last_hidden_state
                x, y = output.double().flatten(), expected.double().flatten()
                cosine = torch.dot(x, y) / (x.norm() * y.norm())
                rmse = (x - y).square().mean().sqrt()
                logging.info("Python audio replay %s: cosine=%.12g RMSE=%.12g",
                             name, cosine.item(), rmse.item())
                outputs[name] = output.cpu().contiguous()
        save_file(outputs, args.output_dir / "audio-replay.safetensors")
        return
    captured: dict[str, torch.Tensor] = {}

    def capture_inputs(prefix):
        def hook(_module, _args, kwargs):
            for name, value in kwargs.items():
                key = f"{prefix}.{name}"
                if isinstance(value, torch.Tensor) and key not in captured:
                    captured[key] = value.detach().cpu().contiguous().clone()
        return hook

    handles = [
        engine.model.text_encoder.register_forward_pre_hook(capture_inputs("condition"), with_kwargs=True),
        engine.model.text_encoder.model.register_forward_pre_hook(capture_inputs("qwen"), with_kwargs=True),
        engine.model.transformer.register_forward_pre_hook(capture_inputs("fm.first"), with_kwargs=True),
    ]
    if args.qkv_fixture:
        if not args.capture_layers or args.benchmark_runs:
            raise ValueError("QKV replay requires layer capture and cannot be benchmarked")
        qkv_fixture = load_file(args.qkv_fixture)
        def replay_projection(name):
            def hook(_module, _inputs, output):
                replacement = qkv_fixture[name].to(device=output.device, dtype=output.dtype)
                if replacement.shape != output.shape:
                    raise ValueError(f"QKV replay shape mismatch: {name}")
                return replacement
            return hook
        for name in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(engine.model.text_encoder.model.layers[0].self_attn, name)
                           .register_forward_hook(replay_projection(name)))
    if args.capture_layers:
        def capture_layer(index):
            def hook(_module, _inputs, output):
                captured[f"qwen.layer.{index}"] = output.detach().cpu().contiguous().clone()
            return hook
        for index, layer in enumerate(engine.model.text_encoder.model.layers):
            handles.append(layer.register_forward_hook(capture_layer(index)))
        for name in ("q_proj", "k_proj", "v_proj"):
            handles.append(getattr(engine.model.text_encoder.model.layers[0].self_attn, name)
                           .register_forward_hook(capture_layer(name)))
        def capture_attention_input(_module, inputs):
            captured["qwen.attention_output"] = inputs[0].detach().cpu().contiguous().clone()
        handles.append(engine.model.text_encoder.model.layers[0].self_attn.o_proj
                       .register_forward_pre_hook(capture_attention_input))
    original_sample = engine.model.sample

    def capture_sample(*sample_args, **sample_kwargs):
        for name, value in sample_kwargs.items():
            if isinstance(value, torch.Tensor):
                captured[f"fm.sample.{name}"] = value.detach().cpu().contiguous().clone()
        output, trajectory = original_sample(*sample_args, **sample_kwargs)
        captured["fm.output"] = output.detach().cpu().contiguous()
        return output, trajectory

    engine.model.sample = capture_sample
    original_denormalize = engine.vae_model.denormalize
    def capture_denormalize(latents):
        captured["vae.normalized_input"] = latents.detach().cpu().contiguous().clone()
        return original_denormalize(latents)
    engine.vae_model.denormalize = capture_denormalize
    original_encoding = engine.vae_model.encoding_and_normalization
    if args.audio:
        def capture_encoder(_module, inputs, output):
            captured["vae.encoder_input"] = inputs[0].detach().cpu().contiguous().clone()
            captured["vae.encoder_stats"] = output.detach().cpu().contiguous().clone()
        handles.append(engine.vae_model.audio_encoder.register_forward_hook(capture_encoder))
        def capture_encoding(*encoding_args, **encoding_kwargs):
            original_randn_like = torch.randn_like
            def capture_noise(*noise_args, **noise_kwargs):
                noise = original_randn_like(*noise_args, **noise_kwargs)
                captured["vae.encoder_noise"] = noise.detach().cpu().contiguous().clone()
                return noise
            torch.randn_like = capture_noise
            try:
                output, lengths = original_encoding(*encoding_args, **encoding_kwargs)
            finally:
                torch.randn_like = original_randn_like
            captured["vae.reference_latents"] = output.detach().cpu().contiguous().clone()
            captured["vae.reference_lengths"] = lengths.detach().cpu().contiguous().clone()
            return output, lengths
        engine.vae_model.encoding_and_normalization = capture_encoding
        def capture_audio_encoder(_module, _inputs, output):
            captured["qwen.audio_output"] = output.last_hidden_state.detach().cpu().contiguous().clone()
        handles.append(engine.model.text_encoder.audio_tower.register_forward_pre_hook(
            capture_inputs("qwen.audio"), with_kwargs=True))
        handles.append(engine.model.text_encoder.audio_tower.register_forward_hook(capture_audio_encoder))
        def capture_audio_boundary(name):
            def hook(_module, inputs, output):
                captured[f"qwen.audio.{name}.input"] = inputs[0].detach().cpu().contiguous().clone()
                value = output[0] if isinstance(output, tuple) else output
                captured[f"qwen.audio.{name}.output"] = value.detach().cpu().contiguous().clone()
            return hook
        audio_tower = engine.model.text_encoder.audio_tower
        for name in ("conv1", "conv2", "ln_post", "proj"):
            handles.append(getattr(audio_tower, name).register_forward_hook(capture_audio_boundary(name)))
        for index in (0, len(audio_tower.layers) - 1):
            handles.append(audio_tower.layers[index].register_forward_hook(capture_audio_boundary(f"layer.{index}")))
        captured["qwen.audio.positional_embedding"] = (
            audio_tower.positional_embedding.positional_embedding.detach().cpu().contiguous().clone())

    def capture_decoder_input(_module, inputs):
        captured["vae.input"] = inputs[0].detach().cpu().contiguous().clone()

    handles.append(engine.vae_model.conv_pre.register_forward_pre_hook(capture_decoder_input))
    content = [{"type": "text", "text": args.instruction}]
    if args.audio:
        content.append({"type": "audio", "audio": str(args.audio.resolve())})
    messages = [{"role": "user", "content": content}]
    # Initialization consumes RNG. Reset before VAE reference sampling as well
    # as the diffusion seed set inside the reference sample implementation.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        waveform, sample_rate = engine.generate(
            messages,
            gen_seconds=args.duration,
            nfe=args.steps,
            cfg_strength=args.cfg,
            sway_sampling_coef=args.sway,
            seed=args.seed,
        )
    finally:
        engine.model.sample = original_sample
        engine.vae_model.denormalize = original_denormalize
        engine.vae_model.encoding_and_normalization = original_encoding
        for handle in handles:
            handle.remove()

    captured["waveform"] = waveform.contiguous()
    save_file(captured, args.output_dir / "reference.safetensors")
    if args.qkv_fixture:
        for name in ("q_proj", "k_proj", "v_proj"):
            if not torch.equal(captured[f"qwen.layer.{name}"].float(), qkv_fixture[name]):
                raise RuntimeError(f"Replayed projection differs: {name}")
        actual = qkv_fixture["attention_output"].double().flatten()
        expected = captured["qwen.attention_output"].double().flatten()
        cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
        rmse = (actual - expected).square().mean().sqrt().item()
        logging.info("Matched QKV attention cosine=%.12f rmse=%.12g", cosine, rmse)
    save_audio(waveform, sample_rate, str(args.output_dir / "reference.wav"))
    manifest = {
        "instruction": args.instruction,
        "messages": messages,
        "seed": args.seed,
        "duration": args.duration,
        "steps": args.steps,
        "cfg": args.cfg,
        "sway": args.sway,
        "dtype": args.dtype,
        "qkv_replay_fixture": str(args.qkv_fixture) if args.qkv_fixture else None,
        "torch": torch.__version__,
        "tf32": False,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "sample_rate": sample_rate,
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in captured.items()},
        "performance_measurement": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logging.info("Saved reference fixture: %s", args.output_dir)
    if args.benchmark_runs:
        milliseconds = []
        expected_waveform = waveform.cpu()
        for run in range(args.benchmark_runs + 2):
            if args.audio:
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            repeated, repeated_rate = engine.generate(messages, gen_seconds=args.duration,
                nfe=args.steps, cfg_strength=args.cfg, sway_sampling_coef=args.sway, seed=args.seed)
            repeated = repeated.cpu()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if repeated_rate != sample_rate or not torch.equal(repeated, expected_waveform):
                raise RuntimeError("benchmark waveform changed")
            if run >= 2:
                milliseconds.append(elapsed)
                logging.info("AuK inference_ms=%.6f", elapsed)
        result = {"runs": args.benchmark_runs, "milliseconds": milliseconds,
                  "median_ms": statistics.median(milliseconds), "dtype": args.dtype,
                  "rtf": statistics.median(milliseconds) / (1000 * args.duration),
                  "scope": "loaded original Python generate to CPU waveform; capture hooks removed; excludes WAV writing"}
        (args.output_dir / "inference-performance.json").write_text(json.dumps(result, indent=2) + "\n")
        logging.info("AuK inference_median_ms=%.6f RTF=%.6f", result["median_ms"], result["rtf"])


if __name__ == "__main__":
    main()
