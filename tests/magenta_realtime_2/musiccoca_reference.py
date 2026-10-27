#!/usr/bin/env python3
"""Save official MusicCoCa outputs for the independent C++ parity test."""

import argparse
import logging
from pathlib import Path
import time

import numpy as np
from safetensors.numpy import save_file
import soundfile as sf
from scipy.signal import resample_poly
from threadpoolctl import threadpool_limits

from magenta_rt import musiccoca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(args.log), logging.StreamHandler()], force=True)
    model = musiccoca.MusicCoCa(args.resources)
    outputs = {}
    prompts = ("disco funk", "A gentle acoustic guitar with warm piano", "Driving electronic drums and deep bass")
    noise = np.random.RandomState(0).randn(768).astype(np.float32)
    outputs["mapper_noise"] = noise
    with threadpool_limits(limits=8):
        for i, prompt in enumerate(prompts):
            start = time.perf_counter()
            text = model.embed(prompt, use_mapper=False)
            mapped = model.embed(prompt, use_mapper=True, seed=0)
            codes = model.tokenize(mapped).astype(np.int32)
            outputs[f"text_{i}"] = text.reshape(-1)
            outputs[f"mapped_{i}"] = mapped.reshape(-1)
            outputs[f"codes_{i}"] = codes.reshape(-1)
            logging.info("prompt=%r session.wall_ms=%.3f", prompt, (time.perf_counter()-start)*1000)
        audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        divisor = np.gcd(sample_rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype(np.float32)
        audio = np.pad(audio[:160000], (0, max(0, 160000-len(audio))))
        prep = model._audio_preprocessor
        prep.set_tensor(prep.get_input_details()[0]["index"], audio.reshape(1, -1))
        prep.invoke()
        features = prep.get_tensor(prep.get_output_details()[0]["index"])
        encoder = model._music_encoder
        encoder.set_tensor(encoder.get_input_details()[0]["index"], features)
        encoder.invoke()
        outputs["audio_features"] = features.reshape(-1)
        outputs["audio_input"] = audio
        outputs["audio_embedding"] = encoder.get_tensor(encoder.get_output_details()[0]["index"]).reshape(-1)
        save_file(outputs, args.output)


if __name__ == "__main__":
    main()
