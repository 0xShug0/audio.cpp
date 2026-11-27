#!/usr/bin/env python3
"""Run the official Hugging Face implementation on a list of audio inputs.

Works with GigaAM v3/Multilingual snapshots (``pytorch_model.bin``) and with
GigaAM-v4 dual-head snapshots (``model.safetensors``; pick the head with
``--head``). The snapshot directory must contain the model's own
``modeling_gigaam.py`` next to its weights and tokenizers.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--log", action="store_true", required=True)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument("--reference-dir", type=Path,
                        help="Official GigaAM source checkout for the timestamp-enabled decoder")
    parser.add_argument("--chunk-duration-sec", type=float, default=0,
                        help="Compare fixed-chunk longform; zero runs the complete input directly")
    parser.add_argument("--head", choices=["e2e", "raw"], default="e2e",
                        help="GigaAM-v4 head (matches gigaam_asr.head); ignored by single-head snapshots")
    args = parser.parse_args()
    if args.chunk_duration_sec and not 0.02 <= args.chunk_duration_sec <= 25:
        parser.error("--chunk-duration-sec must be zero or between 0.02 and 25")
    if args.chunk_duration_sec and args.save_features:
        parser.error("--save-features requires an unchunked input")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.out_dir / "python.log", level=logging.INFO)
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    sys.path.insert(0, str(args.model.resolve()))
    import modeling_gigaam
    from modeling_gigaam import GigaAMConfig, GigaAMModel

    # Avoid Transformers' meta-device construction: this upstream frontend builds
    # torchaudio constants on CPU. Keep the official model and default dtypes.
    config = GigaAMConfig.from_pretrained(str(args.model.resolve()))
    config.name_or_path = str(args.model.resolve())
    model = GigaAMModel(config)
    if (args.model / "pytorch_model.bin").exists():
        state = torch.load(args.model / "pytorch_model.bin", map_location="cpu", weights_only=True)
    else:
        from safetensors.torch import load_file
        state = load_file(str(args.model / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(args.backend).eval()
    dual_heads = getattr(model.model, "heads", None)
    if dual_heads is not None:
        # GigaAM-v4: one encoder, separate RNN-T heads and tokenizers; its own
        # modeling file provides the decoder and the timestamp helpers.
        head, decoding = dual_heads[args.head], model.model.decodings[args.head]
        compute_frame_shift, frames_to_words = modeling_gigaam.compute_frame_shift, modeling_gigaam.frames_to_words
    elif args.word_timestamps and args.reference_dir is None:
        parser.error("--word-timestamps requires --reference-dir for single-head snapshots")
    elif args.word_timestamps:
        # Some HF v3 snapshots predate the official timestamp API. Use the
        # upstream decoder and timestamp utility with the unchanged loaded head.
        sys.path.insert(0, str(args.reference_dir.resolve()))
        from gigaam.decoding import CTCGreedyDecoding, RNNTGreedyDecoding
        from gigaam.timestamps_utils import compute_frame_shift, frames_to_words
        decoding = config.cfg["model"]["cfg"]["decoding"]
        decoder_class = RNNTGreedyDecoding if "RNNT" in decoding["_target_"] else CTCGreedyDecoding
        kwargs = {"vocabulary": decoding["vocabulary"],
                  "model_path": str(args.model / "tokenizer.model") if decoding.get("model_path") else None}
        if decoder_class is RNNTGreedyDecoding:
            kwargs["max_symbols_per_step"] = decoding.get("max_symbols_per_step", 10)
        model.model.decoding = decoder_class(**kwargs)
    if dual_heads is None:
        head, decoding = model.model.head, model.model.decoding
    logging.info("model=%s backend=%s torch=%s threads=%d head=%s", args.model, args.backend, torch.__version__,
                 args.threads, args.head if dual_heads is not None else "single")
    records = []
    for path in args.audio:
        waveform, lengths = model.model.prepare_wav(str(path))
        if args.save_features:
            with torch.inference_mode():
                features, _ = model.model.preprocessor(waveform, lengths)
            import numpy as np
            np.save(args.out_dir / (path.stem + "-features.npy"), features.cpu().numpy())
            def save_stage(name):
                def hook(module, inputs, output):
                    value = output[0] if isinstance(output, tuple) else output
                    np.save(args.out_dir / (path.stem + "-" + name + ".npy"), value.detach().cpu().float().numpy())
                return hook
            encoder = model.model.encoder
            if hasattr(encoder, "pre_encode"):
                stages = [(encoder.pre_encode, "subsampling"), (encoder.layers[0], "layer0")]
            else:  # Zapformer: Conv2d/ConvNeXt frontend, then the first layer of the first stack
                stages = [(encoder.encoder_embed, "subsampling"), (encoder.encoder.encoders[0].layers[0], "layer0")]
            handles = [module.register_forward_hook(save_stage(name)) for module, name in stages]
        duration = waveform.shape[-1] / 16000
        chunk_size = int(args.chunk_duration_sec * 16000) if args.chunk_duration_sec else waveform.shape[-1]
        boundaries = list(range(0, waveform.shape[-1], chunk_size))
        if len(boundaries) > 1 and waveform.shape[-1] - boundaries[-1] < 320:
            boundaries.pop()
        boundaries.append(waveform.shape[-1])
        runs = []
        with torch.inference_mode():
            for iteration in range(args.repeat + 1):
                if args.backend == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                parts = []
                words = []
                for begin, end in zip(boundaries, boundaries[1:]):
                    chunk = waveform[:, begin:end]
                    chunk_lengths = lengths.new_full((1,), end - begin)
                    encoded, encoded_lengths = model(chunk, chunk_lengths)
                    if dual_heads is not None:  # as GigaAMDualASR._decode_head: F16 encoder, F32 heads
                        encoded = encoded.to(next(head.parameters()).dtype)
                    texts = decoding.decode(head, encoded, encoded_lengths)
                    part = texts[0]
                    parts.append(part if isinstance(part, str) else part[0])
                    if args.word_timestamps:
                        shift = compute_frame_shift(end - begin, int(encoded_lengths[0]))
                        for word in frames_to_words(decoding.tokenizer, part[1], part[2], shift):
                            words.append({"word": word.text, "start": word.start + begin / 16000,
                                          "end": word.end + begin / 16000})
                if args.backend == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                text = " ".join(part for part in parts if part)
                row = {"iteration": iteration, "cold": iteration == 0,
                       "text": text, "wall_ms": elapsed, "rtf": elapsed / (duration * 1000)}
                if args.word_timestamps:
                    row["words"] = words
                if args.backend == "cuda":
                    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                runs.append(row)
                logging.info("%s", json.dumps(row, ensure_ascii=False))
        record = {"audio": str(path), "duration_sec": duration,
                  "head": args.head if dual_heads is not None else None,
                  "chunk_duration_sec": args.chunk_duration_sec, "runs": runs}
        records.append(record)
        if args.save_features:
            for handle in handles:
                handle.remove()
        print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.out_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
