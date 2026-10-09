#!/usr/bin/env python3
"""Export official features, frame probabilities and timestamps; measure warmed inference."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def median_ms(fn, cuda=False):
    for _ in range(5):
        fn()
    times = []
    for _ in range(21):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--gguf", type=Path, help="Use dequantized GGUF weights to isolate backend error from quantization drift.")
    parser.add_argument("--max-speech-frames", type=int, default=2000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference.resolve()))
    from fireredvad.vad import FireRedVad, FireRedVadConfig
    from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig
    torch.set_num_threads(args.threads)
    config = torch.load(args.source / "model.pth.tar", map_location="cpu", weights_only=False)["args"]
    causal = config.N2 == 0
    cuda = args.device == "cuda"
    options = (FireRedStreamVadConfig if causal else FireRedVadConfig)(max_speech_frame=args.max_speech_frames, use_gpu=cuda)
    model = (FireRedStreamVad if causal else FireRedVad).from_pretrained(str(args.source), options)
    if args.gguf:
        import gguf
        tensors = {}
        shapes = {name: tuple(value.shape) for name, value in model.vad_model.state_dict().items()}
        for tensor in gguf.GGUFReader(args.gguf).tensors:
            name = tensor.name.removeprefix("weights/")
            shape = shapes.get(name, tuple(reversed(tensor.shape)))
            value = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
            tensors[name] = torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
        model.vad_model.load_state_dict({name: tensors[name] for name in model.vad_model.state_dict()}, strict=True)
        model.audio_feat.cmvn.means = -tensors["frontend.cmvn_shift"].numpy().astype(np.float64)
        model.audio_feat.cmvn.inverse_std_variances = tensors["frontend.cmvn_scale"].numpy().astype(np.float64)
    pcm, rate = sf.read(args.audio, dtype="int16")
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError("Expected mono 16 kHz PCM")
    audio = (pcm, rate)
    detect = (lambda: model.detect_full(audio)) if causal else (lambda: model.detect(audio))
    cold_start = time.perf_counter()
    cold_result = detect()
    cold_ms = (time.perf_counter() - cold_start) * 1000
    features, duration = model.audio_feat.extract(audio)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(args.output_prefix)
    with torch.inference_mode():
        network_features = features.to(args.device)
        probs = model.vad_model(network_features.unsqueeze(0))[0].flatten()
        features.numpy().astype("<f4").tofile(prefix + ".features.f32")
        probs.cpu().numpy().astype("<f4").tofile(prefix + ".probs.f32")
        if causal:
            cached = None
            chunks = []
            for chunk in network_features.split(16):
                chunk_probs, cached = model.vad_model(chunk.unsqueeze(0), cached)
                chunks.append(chunk_probs.flatten())
            torch.cat(chunks).cpu().numpy().astype("<f4").tofile(prefix + ".chunked_probs.f32")
            _, result = cold_result
        else:
            result, _ = cold_result
        Path(prefix + ".segments.json").write_text(json.dumps(result["timestamps"]) + "\n")
        metrics = {"threads": args.threads, "duration": duration, "frames": features.shape[0],
                   "cold_ms": cold_ms,
                   "frontend_ms": median_ms(lambda: model.audio_feat.extract(audio)),
                   "network_ms": median_ms(lambda: model.vad_model(network_features.unsqueeze(0)), cuda),
                   "total_ms": median_ms(detect, cuda)}
        metrics["rtf"] = metrics["total_ms"] / (duration * 1000)
        if args.log:
            print(f"session.wall_ms={metrics['total_ms']:.6f}")
        Path(prefix + ".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
