#!/usr/bin/env python3
"""Measure the unmodified official Python system using the C++ validation cases."""

import argparse
import csv
import json
import logging
from pathlib import Path
import time

import jax
import soundfile as sf
from ai_edge_litert.interpreter import Interpreter

from magenta_rt import paths, musiccoca
from magenta_rt.config import MUSICCOCA
from magenta_rt.jax.system import MagentaRT2System


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", choices=("small", "base"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.log)], force=True)
    paths.set_magenta_home(args.source)
    def interpreter_with_threads(model_path):
        interpreter = Interpreter(model_path=str(model_path), num_threads=args.threads)
        interpreter.allocate_tensors()
        return interpreter
    musiccoca._make_interpreter = interpreter_with_threads
    model = MagentaRT2System(size="mrt2_" + args.variant)
    cases = json.loads(args.cases.read_text())
    with (args.output / "metrics.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=("case", "audio_sec", "wall_ms", "rtf"))
        writer.writeheader()
        for case in cases:
            if case["seed"] != 0:
                raise ValueError("The upstream Python system fixes its generation RNG seed to zero")
            jax.block_until_ready(model._params)
            start = time.perf_counter()
            embedding = model.embed_style(case["text"], use_mapper=True)
            # Keep generation on the official, combined AR+codec streaming function.
            waveform, _ = model.generate(conditioning={MUSICCOCA.key: embedding}, frames=round(case["duration_sec"] * 25))
            wall_ms = (time.perf_counter() - start) * 1000
            duration = len(waveform.samples) / waveform.sample_rate
            sf.write(args.output / (case["id"] + ".wav"), waveform.samples, waveform.sample_rate, subtype="PCM_16")
            row = {"case": case["id"], "audio_sec": duration, "wall_ms": wall_ms, "rtf": wall_ms / (1000 * duration)}
            writer.writerow(row)
            output.flush()
            logging.info("case=%s session.wall_ms=%.3f audio.duration_sec=%.3f RTF=%.6f", case["id"], wall_ms, duration, row["rtf"])


if __name__ == "__main__":
    main()
