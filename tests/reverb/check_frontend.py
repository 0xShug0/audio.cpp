"""Parity of the C++ Reverb frontend against reverb_tts.text and reverb_tts.audio.

    python tests/reverb/check_frontend.py build/bin/reverb_frontend_probe

Needs reverb-tts (pip install reverb-tts, or the upstream checkout) and torchaudio. Randomized
multilingual texts check preparation, the length estimate and long-form splitting exactly; synthetic
audio checks kaiser resampling (5 rate pairs), raise_quiet, speech_length and trim_silence."""
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from reverb_tts.audio import RESAMPLE, raise_quiet, speech_length, trim_silence
from reverb_tts.text import BRACKETS, estimate_frames, split_text

probe = subprocess.Popen([sys.argv[1]], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, encoding="utf-8")


def ask(line):
    probe.stdin.write(line + "\n")
    probe.stdin.flush()
    return probe.stdout.readline().rstrip("\n")


rng = random.Random(0)
words = ("the voice control quick brown fox Zażółć gęślą jaźń café naïve über straße où l'été "
         "¿Qué tal? señor. Hello! Really? yes, no; 12 3,5 % $ — … (aside) [note] {x}").split()
fails = 0
counts = {}

# Text: preparation (brackets + whitespace), length estimate, long-form splitting.
reference = "Some call me nature. Others call me Mother Nature, I have been here for over four billion years."
for trial in range(300):
    text = " ".join(rng.choice(words) for _ in range(rng.randint(1, 120)))
    text = text.replace(" ", rng.choice([" ", "  ", "\u00a0", " \u2003 "]), rng.randint(0, 3))
    prepared = " ".join(BRACKETS.sub("", text).split())
    frames = rng.randint(20, 400)
    got = ask(f"prepare 0\t{text}\t{reference}")
    want = prepared
    if got != want:
        fails += 1
        counts["prepare"] = counts.get("prepare", 0) + 1
    if int(ask(f"estimate {frames}\t{prepared}\t{reference}")) != estimate_frames(prepared, reference, frames):
        fails += 1
        counts["estimate"] = counts.get("estimate", 0) + 1
    split = ask(f"split {frames}\t{prepared}\t{reference}").split("\x1f")[:-1]
    if split != split_text(prepared, reference, frames, 750, 375):
        fails += 1
        counts["split"] = counts.get("split", 0) + 1
print(f"text: 300 random texts checked, mismatches: {counts}")

# Audio: resampling, quiet-reference raise, pace measure, silence trim.
tmp = Path(tempfile.mkdtemp())
t = np.linspace(0, 1, 24000, dtype=np.float32)
for (src_rate, dst_rate) in ((48000, 24000), (44100, 24000), (16000, 24000), (22050, 24000), (24000, 16000)):
    n = int(src_rate * 0.7) + 13
    x = (np.random.RandomState(src_rate).randn(n) * 0.1).astype(np.float32)
    x[: n // 3] *= 0.0005  # quiet lead-in, silence-ish
    (tmp / "x.bin").write_bytes(x.tobytes())
    ask(f"resample {tmp / 'x.bin'} {src_rate} {dst_rate} {tmp / 'y.bin'}")
    got = np.frombuffer((tmp / "y.bin").read_bytes(), dtype=np.float32)
    want = torchaudio.functional.resample(torch.from_numpy(x), src_rate, dst_rate, **RESAMPLE).numpy()
    err = np.abs(got - want).max() if got.shape == want.shape else float("inf")
    ok = got.shape == want.shape and err < 1e-5
    fails += not ok
    print(f"resample {src_rate}->{dst_rate}: {len(got)} vs {len(want)} samples, max abs err {err:.2e}", "" if ok else "FAIL")

    q = x * 0.001
    (tmp / "q.bin").write_bytes(q.tobytes())
    ask(f"quiet {tmp / 'q.bin'} {tmp / 'r.bin'}")
    got = np.frombuffer((tmp / "r.bin").read_bytes(), dtype=np.float32)
    want = raise_quiet(torch.from_numpy(q)).numpy()
    err = np.abs(got - want).max()
    fails += err > 1e-6
    sp_got = float(ask(f"speech {tmp / 'x.bin'} {src_rate}"))
    sp_want = speech_length(x, src_rate)
    fails += abs(sp_got - sp_want) > 1e-9
    ask(f"trim {tmp / 'x.bin'} {src_rate} {tmp / 't.bin'}")
    got = np.frombuffer((tmp / "t.bin").read_bytes(), dtype=np.float32)
    want = trim_silence(x, src_rate)
    fails += not (got.shape == want.shape and np.array_equal(got, want))
    print(f"   raise_quiet err {err:.1e}; speech_length {sp_got:.4f} vs {sp_want:.4f}; trim {len(got)} vs {len(want)}")

print("FAILURES:", fails)
sys.exit(1 if fails else 0)
