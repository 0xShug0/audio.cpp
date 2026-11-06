#!/usr/bin/env python3
"""Official MuLaCover conditioning boundaries for native component validation."""

import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", choices=("style", "frontend", "transcriber", "chord", "codec"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--codes-file", type=Path)
    parser.add_argument("--tags", default="topic:[hope]; genre:[folk]; instrument:[piano]; mood:[warm]")
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.FileHandler(args.log), logging.StreamHandler()], force=True)
    sys.path.insert(0, str(args.upstream.resolve() / "src"))
    torch.set_num_threads(8)
    torch.manual_seed(1234)
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    logging.info("reference.cudnn.allow_tf32=%s reference.matmul.allow_tf32=%s",
                 torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tensors = {}
    with torch.inference_mode():
        if args.component == "codec":
            if not args.codes_file or not args.device.startswith("cuda"):
                parser.error("codec mode requires --codes-file with a CUDA reference")
            from mulacover._codec.modeling import HeartCodec
            reference = load_file(args.codes_file)
            tensors["seed"] = reference["seed"].clone()
            tensors["codec_randn_offset"] = reference["codec_randn_offset"].clone()
            tensors["flow_steps"] = torch.tensor(10, dtype=torch.int64)
            tensors["flow_guidance"] = torch.tensor(1.25, dtype=torch.float32)
            codec = HeartCodec.from_pretrained(args.model / "HeartCodec-oss", device_map=args.device,
                                               dtype=torch.float32, local_files_only=True).eval()
            torch.cuda.set_rng_state(reference["codec_rng_state"], args.device)
            original_flow = codec.flow_matching.inference_codes
            original_decode = codec.scalar_model.decode

            def capture_flow(codes, *arguments, **keywords):
                result = original_flow(codes, *arguments, **keywords)
                if "codec_codes" not in tensors:
                    tensors["codec_codes"] = codes[0].detach().cpu().to(torch.int32)
                return result

            def capture_condition(module, arguments, result):
                if "conditioning" not in tensors:
                    tensors["conditioning"] = result.repeat_interleave(2, dim=1).float().cpu()

            def capture_estimator(module, arguments, keywords, result):
                if "flow_input" not in tensors:
                    tensors["flow_input"] = arguments[0].detach().float().cpu()
                    tensors["flow_timestep"] = keywords["timestep"].detach().float().cpu()
                    tensors["flow_output"] = result.detach().float().cpu()

            def capture_decode(latents, *arguments, **keywords):
                result = original_decode(latents, *arguments, **keywords)
                if "scalar_latents" not in tensors:
                    tensors["scalar_latents"] = latents.detach().float().cpu()
                    tensors["scalar_waveform"] = result.detach().float().cpu()
                return result

            codec.flow_matching.inference_codes = capture_flow
            codec.scalar_model.decode = capture_decode
            cond_hook = codec.flow_matching.cond_feature_emb.register_forward_hook(capture_condition)
            estimator_hook = codec.flow_matching.estimator.register_forward_hook(capture_estimator, with_kwargs=True)
            try:
                tensors["waveform"] = codec.detokenize(
                    reference["codes"].to(args.device), duration=29.76, num_steps=10,
                    guidance_scale=1.25, disable_progress=True).detach().float().cpu()
            finally:
                cond_hook.remove()
                estimator_hook.remove()
                codec.flow_matching.inference_codes = original_flow
                codec.scalar_model.decode = original_decode
        elif args.component == "style":
            from transformers import AutoModel, AutoTokenizer
            root = args.model / "Qwen3-Embedding-0.6B"
            tokenizer = AutoTokenizer.from_pretrained(root, padding_side="left", local_files_only=True)
            model = AutoModel.from_pretrained(root, dtype=torch.float32, local_files_only=True).to(args.device).eval()
            inputs = tokenizer([args.tags], padding=True, truncation=False, return_tensors="pt")
            hidden = model(**{k: v.to(args.device) for k, v in inputs.items()}).last_hidden_state
            tensors["style"] = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)[0].cpu()
            tensors["input_ids"] = inputs["input_ids"].to(torch.int32).cpu()
        else:
            if not args.input:
                parser.error("--input is required for audio components")
            audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
            tensors["waveform"] = torch.from_numpy(audio.copy())
            tensors["sample_rate"] = torch.tensor(sample_rate, dtype=torch.int64)
            if args.component in ("frontend", "chord"):
                import librosa
                bpm = args.bpm
                if bpm == 0:
                    original, original_rate = librosa.load(args.input, sr=None)
                    tempo, _ = librosa.beat.beat_track(y=original, sr=original_rate)
                    bpm = float(tempo.item())
                tensors["bpm"] = torch.tensor(bpm, dtype=torch.float64)
                waveform, _ = librosa.load(args.input, sr=22050, mono=True)
                tuning = librosa.estimate_tuning(y=waveform, sr=22050, bins_per_octave=36)
                magnitude = np.abs(librosa.hybrid_cqt(waveform, sr=22050, bins_per_octave=36,
                    fmin=librosa.note_to_hz("F#0"), n_bins=288, tuning=None, hop_length=512)).T
                cqt = magnitude[:, 18:270]
                tensors["cqt"] = torch.from_numpy(cqt.copy())
                tensors["tuning"] = torch.tensor(tuning, dtype=torch.float64)
                if args.component == "chord":
                    from mulacover._symbolic_transcription.harmony import ChordTranscriber
                    chord = ChordTranscriber(args.model / "SymbolicTranscriptor/chord",
                                              torch.device(args.device), torch.float32)
                    # Compare the full network's probability boundaries separately
                    # from the model-specific chord vocabulary/Viterbi decisions.
                    features = torch.from_numpy(magnitude.copy()).to(args.device)
                    predictions = [m.inference(features) for m in chord.models]
                    for head in range(6):
                        tensors[f"probabilities.{head}"] = torch.from_numpy(
                            np.mean([p[head] for p in predictions], axis=0))
                    labels = chord.decoder.decode_segments(
                        [tensors[f"probabilities.{h}"].numpy() for h in range(6)], 512 / 22050)
                    import json
                    tensors["chord_times"] = torch.tensor([[onset, offset] for _, onset, offset in labels], dtype=torch.float64).reshape(-1, 2)
                    args.output.with_suffix(".json").write_text(json.dumps({"chord_labels": [label for label, _, _ in labels]}) + "\n")
            else:
                from mulacover._symbolic_transcription.melody import MelodyTranscriber
                model = MelodyTranscriber(args.model / "SymbolicTranscriptor/yourmt3/last.ckpt",
                                           args.device, torch.float32)
                notes = model.transcribe(args.input)
                tensors["notes"] = torch.tensor([
                    [n.onset, n.pitch, n.offset - n.onset, n.program, int(n.is_drum)] for n in notes
                ], dtype=torch.float64).reshape(-1, 5)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: (v.float() if v.is_floating_point() else v).contiguous()
               for k, v in tensors.items()}, args.output)
    logging.info("session.wall_ms=%.3f", (time.perf_counter() - start) * 1000)


if __name__ == "__main__":
    main()
