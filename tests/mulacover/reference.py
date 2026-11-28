#!/usr/bin/env python3
"""Run official MuLaCover and save AR inputs, output codes and final waveform."""

import argparse
import logging
from pathlib import Path
import sys
import time

import soundfile as sf
import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lyrics", type=Path, required=True)
    parser.add_argument("--tags", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-audio", type=Path)
    source.add_argument("--melody-file", type=Path)
    parser.add_argument("--chord-file", type=Path)
    parser.add_argument("--drum-file", type=Path)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--duration-sec", type=float, default=120)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Save prompt and invariant conditioning boundaries without running AR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.melody_file and not args.chord_file:
        parser.error("--melody-file requires --chord-file")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(args.log), logging.StreamHandler()], force=True)
    sys.path.insert(0, str(args.upstream.resolve() / "src"))
    from mulacover import MuLaCoverGenPipeline

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    pipeline = MuLaCoverGenPipeline.from_pretrained(
        str(args.model), device=torch.device(args.device),
        dtype={"mulacover": getattr(torch, args.dtype), "qwen": torch.float32,
               "codec": torch.float32, "transcriptor": torch.float32}, lazy_load=True)
    inputs = {"lyrics": args.lyrics.read_text(encoding="utf-8"), "tags": args.tags}
    if args.reference_audio:
        inputs.update(ref_audio=str(args.reference_audio), bpm=args.bpm)
    else:
        inputs.update(melody_midi=str(args.melody_file), chord_midi=str(args.chord_file),
                      drum_midi=str(args.drum_file) if args.drum_file else None)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        start = time.perf_counter()
        prepared = pipeline.preprocess(inputs, cfg_scale=args.guidance_scale,
                                       max_audio_length_ms=int(args.duration_sec * 1000))
        fixtures = {name: tensor.detach().cpu().contiguous() for name, tensor in prepared.items()}
        # Capture the AR boundary outside generation: the documented prefix
        # embedding composition, with every conditional/unconditional input.
        model = pipeline.mulacover
        device = pipeline.devices["mulacover"]
        tokens = prepared["tokens"].to(device)
        batch_size = tokens.shape[0]
        uncond = (torch.arange(batch_size, device=device) >= batch_size // 2
                  if args.guidance_scale > 1 and batch_size > 1 else None)
        embeddings = (model._embed_tokens(tokens, uncond) *
                      prepared["tokens_mask"].to(device).unsqueeze(-1)).sum(dim=2)
        qwen = model.qwen_linear(prepared["qwen_embedding"].to(device, model.qwen_linear.weight.dtype))
        muq = model.muq_linear(prepared["muq_embedding"].to(device, model.muq_linear.weight.dtype))
        if uncond is not None:
            unconditional = model.unconditional_text_embedding(torch.zeros(1, dtype=torch.long, device=device))
            qwen = torch.where(uncond[:, None], unconditional, qwen)
            muq = torch.where(uncond[:, None], unconditional, muq)
        rows = torch.arange(batch_size, device=device)
        embeddings[rows, prepared["qwen_indices"].to(device)] = qwen.to(embeddings.dtype)
        embeddings[rows, prepared["muq_indices"].to(device)] = muq.to(embeddings.dtype)
        fixtures["prompt_embeddings"] = embeddings.float().cpu().contiguous()
        fixtures["seed"] = torch.tensor(args.seed, dtype=torch.int64)
        if args.prepare_only:
            piano = prepared["pianoroll"].to(device)
            drums = prepared["drum_pianoroll"].to(device)
            chords = prepared["chord"].to(device)
            style = prepared["qwen_embedding"].to(device).unsqueeze(1)
            if uncond is not None:
                piano = torch.where(uncond[:, None, None], -torch.ones_like(piano), piano)
                drums = torch.where(uncond[:, None, None], -torch.ones_like(drums), drums)
                chords = torch.where(uncond[:, None, None], -torch.ones_like(chords), chords)
                style = torch.where(uncond[:, None, None], -torch.ones_like(style), style)
            for name, value in (("pianoroll", piano), ("drum_pianoroll", drums), ("chord", chords)):
                fixtures[name] = value[0].float().cpu().contiguous().clone()
            fixtures["style"] = style[0, 0].float().cpu().contiguous().clone()
            context = None
            first_layer = len(model.backbone.layers) - model.config.peft_layer_num
            for index, layer_index in enumerate(range(first_layer, len(model.backbone.layers))):
                layer = model.peft_layers[layer_index]
                current = layer.adaptor(piano, drums, chords, prepared["context_mask"].to(device))
                context = current if context is None else context + current
                scale, shift = model.tag_mlp_layers[layer_index](style)
                gate_context = layer.cross_attn.context_gate_proj(context)
                values = {"context": context, "style_scale": scale, "style_shift": shift,
                          "gatecontext": gate_context}
                for prefix, attention, source in (("", layer.cross_attn.attn, context),
                                                  ("gate_", layer.cross_attn.gate, gate_context)):
                    key = attention.k_proj(source).view(batch_size, source.shape[1], attention.num_kv_heads, attention.head_dim)
                    key = attention.key_rope(key).transpose(1, 2)
                    value = attention.v_proj(source).view(batch_size, source.shape[1], attention.num_kv_heads, attention.head_dim).transpose(1, 2)
                    values[prefix + "key"] = key
                    values[prefix + "value"] = value
                for name, value in values.items():
                    fixtures[f"layer{index}.{name}"] = value.detach().float().cpu().contiguous().clone()
            save_file(fixtures, args.output / "reference.safetensors")
            logging.info("session.wall_ms=%.3f validation.boundary=prompt_and_conditioning",
                         (time.perf_counter() - start) * 1000)
            return
        del model, embeddings, tokens, qwen, muq
        generated = pipeline._forward(
            prepared, max_audio_length_ms=int(args.duration_sec * 1000),
            temperature=args.temperature, topk=args.top_k, cfg_scale=args.guidance_scale,
            disable_progress=True)
        fixtures["codes"] = generated["frames"].detach().cpu().contiguous()
        fixtures["codec_rng_state"] = (torch.cuda.get_rng_state(args.device).cpu()
                                       if args.device.startswith("cuda") else torch.get_rng_state())
        if args.device.startswith("cuda"):
            # CUDA's generator records offsets in 32-bit draws; the framework
            # sampling helper accepts four-draw Philox blocks.
            offset = int.from_bytes(bytes(fixtures["codec_rng_state"])[8:16], "little")
            if offset % 4:
                raise ValueError("Unaligned CUDA Philox offset")
            fixtures["codec_randn_offset"] = torch.tensor(offset // 4, dtype=torch.int64)
        codec = pipeline.codec
        waveform = codec.detokenize(generated["frames"].to(args.device), duration=29.76,
                                    num_steps=10, guidance_scale=1.25, disable_progress=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        fixtures["waveform"] = waveform.detach().float().cpu().contiguous()
        elapsed = time.perf_counter() - start
        save_file(fixtures, args.output / "reference.safetensors")
        sf.write(args.output / "python.wav", fixtures["waveform"].numpy().T, codec.sample_rate,
                 subtype="FLOAT")
        seconds = waveform.shape[-1] / codec.sample_rate
        logging.info("session.wall_ms=%.3f audio.duration_sec=%.3f session.rtf=%.6f",
                     elapsed * 1000, seconds, elapsed / seconds)


if __name__ == "__main__":
    main()
