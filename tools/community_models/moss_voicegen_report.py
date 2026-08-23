"""Report the acoustic properties that matter for MOSS-VoiceGenerator takes.

The model designs a speaker rather than reproducing one, so takes differ from run to run
in ways a diff cannot show. This reports what a caller needs to accept or reject a take:
duration, how much of it is speech, the longest interior pause, the peak level, and the
median F0 that stands in for speaker identity.

    python3 tools/community_models/moss_voicegen_report.py <dir-or-wav> [--reference NAME]

With --reference, every file is also reported in semitones relative to that one, which is
how two takes are checked for being the same speaker.
"""

import argparse
import pathlib
import wave

import numpy as np

QUIET_RMS = 0.005      # ~-46 dBFS, below any voiced speech
FRAME_SECONDS = 0.02


def load(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768, sr


def envelope(x, sr):
    n = int(FRAME_SECONDS * sr)
    return np.array([np.sqrt((x[i:i+n] ** 2).mean()) for i in range(0, len(x) - n, n)])


def longest_quiet_run(env):
    best = current = 0
    for quiet in env < QUIET_RMS:
        current = current + 1 if quiet else 0
        best = max(best, current)
    return best * FRAME_SECONDS


def median_f0(x, sr, fmin=60.0, fmax=320.0):
    n, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    picks = []
    for i in range(0, len(x) - n, hop):
        frame = x[i:i+n]
        if np.sqrt((frame ** 2).mean()) < 0.02:
            continue
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, mode="full")[n-1:]
        if ac[0] <= 0 or hi <= lo:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] < 0.3:      # unvoiced
            continue
        picks.append(sr / lag)
    return float(np.median(picks)) if picks else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--reference", help="file name to report semitone distances against")
    args = parser.parse_args()

    root = pathlib.Path(args.path)
    files = sorted(root.glob("*.wav")) if root.is_dir() else [root]

    print(f"{'file':<30} {'dur':>6} {'speech':>7} {'pause':>7} {'peak':>6} {'F0':>7}")
    f0s = {}
    for path in files:
        x, sr = load(path)
        env = envelope(x, sr)
        f0s[path.name] = median_f0(x, sr)
        print(f"{path.name:<30} {len(x)/sr:5.2f}s {100*(env >= QUIET_RMS).mean():6.1f}% "
              f"{longest_quiet_run(env):6.2f}s {np.abs(x).max():6.3f} {f0s[path.name]:6.1f}Hz")

    if args.reference and args.reference in f0s:
        base = f0s[args.reference]
        print(f"\nsemitones vs {args.reference} (a speaker change shows up here, not in the numbers above):")
        for name, hz in f0s.items():
            if not np.isnan(hz) and not np.isnan(base):
                print(f"  {name:<30} {12*np.log2(hz/base):+5.1f} st")


if __name__ == "__main__":
    main()
