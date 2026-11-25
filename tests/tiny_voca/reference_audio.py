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
                wave[0, :int(lengths[0])].numpy().astype(np.float32).tofile(args.out / f"{key}.f32")
                noise.tofile(args.out / f"{key}.noise")
                caught["f0"][0, :frames].numpy().astype(np.float32).tofile(args.out / f"{key}.f0")
                durations = details["durations"][0].tolist()
                rows.append({"name": key, "text": text, "voice": name, "speed": speed, "pitch_shift": shift, "pitch_range": spread,
                             "tokens": len(phones), "durations": durations, "seconds": int(lengths[0]) / model.config.sample_rate})
                cases.append("\t".join([key, str(voice), repr(speed), repr(shift), repr(spread), " ".join(map(str, phones.ids)),
                                        " ".join(map(str, phones.stress)), str((args.out / f"{key}.noise").resolve())]))
    (args.out / "cases.tsv").write_text("\n".join(cases) + "\n", encoding="utf-8")
    (args.out / "rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} utterances, {sum(row['seconds'] for row in rows):.1f} s of audio")


def compare(args):
    rows = json.loads((args.out / "rows.json").read_text(encoding="utf-8"))
    errors, same, lengths, worst, pitch, level = [], 0, 0, [], [], []
    for row in rows:
        reference = np.fromfile(args.out / f"{row['name']}.f32", dtype=np.float32)
        path = args.cpp / f"{row['name']}.f32"
        if not path.exists():
            print("MISSING", row["name"])
            continue
        ported = np.fromfile(path, dtype=np.float32)
        durations = [int(x) for x in (args.cpp / f"{row['name']}.dur").read_text().split()]
        equal = durations == row["durations"]
        same += equal
        if len(ported) == len(reference):
            lengths += 1
            error = float(np.abs(ported - reference).max() / np.abs(reference).max())
            errors.append(error)
            worst.append((error, row["name"]))
            level.append(float(np.sqrt(np.mean((ported - reference) ** 2)) / np.sqrt(np.mean(reference ** 2))))
            ours, theirs = args.cpp / f"{row['name']}.f0", args.out / f"{row['name']}.f0"
            if ours.exists() and theirs.exists():
                a, b = np.fromfile(ours, dtype=np.float32), np.fromfile(theirs, dtype=np.float32)
                if len(a) == len(b):
                    pitch.append(float((np.abs(a - b) / b).max()) if len(b) else 0.0)
        if not equal:
            where = [(i, a, b) for i, (a, b) in enumerate(zip(row["durations"], durations)) if a != b]
            print(f"DURATIONS {row['name']}: {len(where)} of {len(durations)} tokens differ, {where[:6]}; text: {row['text'][:70]}")
    errors = np.array(errors)
    print(f"{len(rows)} utterances; durations equal in {same}; same length in {lengths}")
    if len(errors):
        print(f"waveform error as a fraction of the peak, where the lengths agree: median {np.median(errors):.2e}, "
              f"mean {errors.mean():.2e}, largest {errors.max():.2e}")
        for error, name in sorted(worst, reverse=True)[:5]:
            print(f"  {name}: {error:.2e}")
        level = np.array(level)
        print(f"difference over signal (RMS): median {20 * np.log10(np.median(level)):.0f} dB, largest {20 * np.log10(level.max()):.0f} dB")
    if pitch:
        pitch = np.array(pitch)
        print(f"pitch of the frames, largest relative difference in an utterance: median {np.median(pitch):.1e}, largest {pitch.max():.1e} "
              f"({len(pitch)} utterances)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("write", "compare"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--texts", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpp", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--vary", action="store_true", help="also speak every fourth text at other speeds and pitches")
    args = parser.parse_args()
    {"write": write, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
