#!/usr/bin/env python3
"""Run pinned, unmodified ClearVoice SE/SR inference on a session case list."""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--variant", choices=("se", "sr"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(args.log)], force=True)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.upstream / "clearvoice"))
    from clearvoice.utils.decode import decode_one_audio_mossformer2_se_48k, decode_one_audio_mossformer2_sr_48k
    config_root = args.upstream / "clearvoice/clearvoice/config/inference"
    if args.variant == "se":
        from clearvoice.models.mossformer2_se.mossformer2_se_wrapper import TestNet
        config = yaml.safe_load((config_root / "MossFormer2_SE_48K.yaml").read_text())
        model = TestNet()
        model.load_state_dict(torch.load(args.source / "last_best_checkpoint.pt", map_location="cpu", weights_only=True))
        model = model.eval().to(args.backend)
        decode = decode_one_audio_mossformer2_se_48k
    else:
        from clearvoice.models.mossformer2_sr.generator import Mossformer, Generator
        config = yaml.safe_load((config_root / "MossFormer2_SR_48K.yaml").read_text())
        config.update(json.loads((config_root / "MossFormer2_SR_48K.json").read_text()))
        backbone, vocoder = Mossformer(), Generator(SimpleNamespace(**config))
        backbone.load_state_dict(torch.load(args.source / "last_best_checkpoint_m.pt", map_location="cpu", weights_only=True)["mossformer"])
        vocoder.load_state_dict(torch.load(args.source / "last_best_checkpoint_g.pt", map_location="cpu", weights_only=True)["generator"])
        vocoder.remove_weight_norm()
        model = [backbone.eval().to(args.backend), vocoder.eval().to(args.backend)]
        decode = decode_one_audio_mossformer2_sr_48k
    config = SimpleNamespace(**config)
    with (args.output / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=("case", "audio_sec", "wall_ms", "rtf"))
        writer.writeheader()
        for case in json.loads(args.cases.read_text()):
            samples, rate = sf.read(args.cases.parent / case["audio"], dtype="float32")
            if rate != 48000 or samples.ndim != 1:
                raise ValueError("Reference cases require aligned 48 kHz mono input")
            if case.get("options"):
                raise ValueError("This reference runner uses the unmodified upstream defaults")
            if args.backend == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = decode(model, torch.device(args.backend), samples[None], config)
            # ClearVoice's public file wrapper trims decode output to the input length.
            output = output[:len(samples)]
            if args.backend == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if not np.isfinite(output).all():
                raise ValueError("Non-finite reference output")
            sf.write(args.output / (case["id"] + ".wav"), output, rate, subtype="FLOAT")
            duration = len(output) / rate
            row = dict(case=case["id"], audio_sec=duration, wall_ms=elapsed, rtf=elapsed / (duration * 1000))
            writer.writerow(row)
            f.flush()
            logging.info("case=%s session.wall_ms=%.3f audio.duration_sec=%.6f RTF=%.6f", case["id"], elapsed, duration, row["rtf"])


if __name__ == "__main__":
    main()
