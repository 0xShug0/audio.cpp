#!/usr/bin/env python3
"""The Python side of the audio comparison for the tiny-voca port.

    reference_audio.py write  --checkpoint tiny-voca.pt --texts TEXTS --out DIR
    reference_audio.py compare --out DIR --cpp CPP_DIR

``write`` speaks every text with every voice through the PyTorch model
(``TinyVoca.synthesize``), with noise it stores, and writes the cases for
``tiny_voca_synth_probe``: the same phoneme ids and the same noise. ``compare``
reads what the probe wrote and reports whether the durations are equal, the
largest difference between the waveforms as a fraction of the peak (the
measure of tiny-voca's ``scripts/check_onnx.py``), the difference as a level
below the signal, and how far the pitch of the frames is apart.

Needs tiny_voca and dizione on the Python path, and DIZIONE_DATA.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def write(args):
    import torch
    from tiny_voca.infer import load_model, match_frontend
    from tiny_voca.text import phonemize
    torch.set_num_threads(args.threads)
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    hop = model.config.hop
    args.out.mkdir(parents=True, exist_ok=True)
    texts = [line for line in args.texts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        raise ValueError("texts must not be empty")
    settings = [(1.0, 0.0, 1.0)]
    if args.vary:
        settings += [(0.8, 0.0, 1.0), (1.3, 0.1155, 1.0), (1.0, -0.1155, 1.4)]
    rows, cases = [], []
    caught = {}
    # the pitch the model hands its waveform head, frame by frame
    model.vocoder.register_forward_pre_hook(lambda module, inputs: caught.update(f0=inputs[1]))
    for number, text in enumerate(texts):
        phones = phonemize(text)
        for voice, name in enumerate(model.config.speakers):
            for variant, (speed, shift, spread) in enumerate(settings if number % 4 == 0 else settings[:1]):
                tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
                seed = number
                wave, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]),
                                                          speed, shift, seed, spread)
                frames = int(lengths[0]) // hop
                # the noise model.synthesize drew for this row
                noise = torch.randn(frames * hop, generator=torch.Generator().manual_seed(seed + 0)).numpy().astype(np.float32)
                key = f"u{number:04d}-{name}-{variant}"
                if (frames <= 0 or not torch.isfinite(wave).all() or
                        not torch.isfinite(caught["f0"]).all()):
                    raise ValueError(f"{key}: empty or nonfinite reference output")
                if int(lengths[0]) != frames * hop or len(phones.ids) != len(phones.stress):
                    raise ValueError(f"{key}: inconsistent reference dimensions")
                wave[0, :int(lengths[0])].numpy().astype(np.float32).tofile(args.out / f"{key}.f32")
                noise.tofile(args.out / f"{key}.noise")
                caught["f0"][0, :frames].numpy().astype(np.float32).tofile(args.out / f"{key}.f0")
                durations = details["durations"][0].tolist()
                rows.append({"name": key, "text": text, "voice": name, "speed": speed, "pitch_shift": shift, "pitch_range": spread,
                             "tokens": len(phones), "durations": durations, "seconds": int(lengths[0]) / model.config.sample_rate})
                cases.append("\t".join([key, str(voice), repr(speed), repr(shift), repr(spread), " ".join(map(str, phones.ids)),
                                        " ".join(map(str, phones.stress)), str((args.out / f"{key}.noise").resolve())]))
    if not rows:
        raise ValueError("reference generated no utterances")
    (args.out / "cases.tsv").write_text("\n".join(cases) + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} utterances, {sum(row['seconds'] for row in rows):.1f} s of audio")


def compare(args):
    rows = json.loads((args.out / "rows.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows.json must contain a nonempty list")
    names = set()
    for row in rows:
        name = row["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in names:
            raise ValueError("invalid or duplicate row name")
        names.add(name)
        expected = row["durations"]
        if (not isinstance(expected, list) or not expected or
                any(type(x) is not int or x < 0 for x in expected) or
                len(expected) != row["tokens"] or sum(expected) <= 0):
            raise ValueError(f"{name}: invalid reference durations")
        durations = [int(x) for x in (args.cpp / f"{name}.dur").read_text().split()]
        if durations != expected:
            raise ValueError(f"{name}: duration mismatch")
        arrays = {}
        for suffix in ("f32", "f0"):
            pair = []
            for directory in (args.out, args.cpp):
                path = directory / f"{name}.{suffix}"
                if path.stat().st_size == 0 or path.stat().st_size % 4:
                    raise ValueError(f"{path}: empty or malformed float32 data")
                values = np.fromfile(path, dtype=np.float32).astype(np.float64)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: nonfinite values")
                pair.append(values)
            if pair[0].shape != pair[1].shape:
                raise ValueError(f"{name}: {suffix} shape mismatch")
            arrays[suffix] = pair
        reference, ported = arrays["f32"]
        f0, actual_f0 = arrays["f0"]
        if len(f0) != sum(expected) or len(reference) % len(f0):
            raise ValueError(f"{name}: inconsistent frame/sample counts")
        peak = float(np.max(np.abs(reference)))
        if peak == 0:
            raise ValueError(f"{name}: zero reference waveform norm")
        error = float(np.max(np.abs(ported - reference)) / peak)
        pitch_error = float(np.max(np.abs(actual_f0 - f0)))
        print(f"{name}: peak_relative_error={error:.9g} pitch_max_abs={pitch_error:.9g}")
        if error > args.max_peak_relative_error or pitch_error > args.max_pitch_abs_error:
            raise ValueError(f"{name}: numerical acceptance failed")
    print(f"PASS: {len(rows)} utterances")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("write", "compare"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--texts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vary", action="store_true", help="also speak every fourth text at other speeds and pitches")
    parser.add_argument("--max-peak-relative-error", type=float,
                        help="required comparison acceptance; no unverified default")
    parser.add_argument("--max-pitch-abs-error", type=float,
                        help="required absolute pitch tolerance in Hz")
    args = parser.parse_args()
    if args.command == "compare":
        if args.cpp is None:
            parser.error("compare requires --cpp")
        for value in (args.max_peak_relative_error, args.max_pitch_abs_error):
            if value is None or not np.isfinite(value) or value < 0:
                parser.error("compare requires finite nonnegative numerical acceptance parameters")
    elif args.checkpoint is None or args.texts is None:
        parser.error("write requires --checkpoint and --texts")
    {"write": write, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
