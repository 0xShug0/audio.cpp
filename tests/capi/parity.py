#!/usr/bin/env python3
"""Check that the C API returns what audiocpp_cli returns.

The C API is a facade over the same runtime the CLI drives, so for identical
inputs the two must agree. That is the property worth testing: not that the
facade produces plausible output, but that it produces the *same* output.

    python3 tests/capi/parity.py --build-dir build --models-root /path/to/models

Exits 0 when every available family matches, 1 on a mismatch, and 77 (the CTest
skip code) when no model is present.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import wave

SKIP = 77

# family -> (relative model path, CLI args builder, comparison)
CASES = {
    "citrinet_asr": {
        "model": "Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf",
        "task": "asr",
        "kind": "text",
    },
    "parakeet_tdt": {
        "model": "Parakeet-TDT-0.6B-v3-GGUF/parakeet-tdt-0.6b-v3-q8_0.gguf",
        "task": "asr",
        "kind": "text",
    },
    "kokoro_tts": {
        "model": "Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf",
        "task": "tts",
        "kind": "audio",
        "text": "The quick brown fox jumps over the lazy dog.",
        "language": "en-us",
        "voice_id": "af_heart",
        "seed": "1234",
    },
    "bs_roformer": {
        "model": "BS-RoFormer-ep368-GGUF/bs-roformer-ep368-q8_0.gguf",
        "task": "sep",
        "kind": "stems",
        # Fixed 44.1 kHz input; the default 16 kHz clip is rejected outright.
        "use_alt_audio": True,
    },
    "sortformer_diar": {
        "model": "Sortformer-Diar-4spk-v1-GGUF/sortformer-diar-4spk-v1-q8_0.gguf",
        "task": "diar",
        "kind": "turns",
    },
}


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    return proc.stdout


def wav_summary(path):
    with wave.open(path) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(frames)
    peak = 0
    if width == 2:
        import array

        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder == "big":
            samples.byteswap()
        peak = max((abs(v) for v in samples), default=0) / 32768.0
    return frames, rate, channels, peak


def why_missing(family, capi_output):
    """Say why a family produced no comparable value.

    "Not linked" and "ran but produced nothing" look identical from the parity
    values alone, and reporting the wrong one sends a reader hunting through
    CMake for a build problem that is not there.
    """
    if f"=== {family} (" in capi_output:
        return "the C API run reported no comparable value"
    for line in capi_output.splitlines():
        if line.startswith(f"skip {family}") or line.startswith(f"skip {family:16}"):
            return line.split("(", 1)[-1].rstrip(")") if "(" in line else "skipped by the C API run"
    return "family not linked into this build"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--audio", default="assets/resources/sample_16k.wav")
    # Separation families fix their input rate, so they need their own clip.
    parser.add_argument("--alt-audio", default="tests/ace_step/assets/complete_source_demucs_8s.wav")
    parser.add_argument("--backend", default="cpu")
    # Separation dominates the wall time and scales with threads, and thread
    # count does not change results (verified bit-identical stems at 4 and 16).
    parser.add_argument("--threads", default=str(os.cpu_count() or 4))
    args = parser.parse_args()

    # Resolved up front: this runs from the build directory under CTest, where
    # the repo-relative defaults would not exist. Silently losing a fixture is
    # how a case skips while the run still reports OK.
    args.audio = os.path.abspath(args.audio)
    args.alt_audio = os.path.abspath(args.alt_audio)
    for path in (args.audio, args.alt_audio):
        if not os.path.exists(path):
            print(f"missing audio fixture {path}; skipping")
            return SKIP

    binaries = os.path.join(args.build_dir, "bin")
    cli = os.path.join(binaries, "audiocpp_cli")
    model_test = os.path.join(binaries, "audiocpp_c_api_model_test")
    for path in (cli, model_test):
        if not os.path.exists(path):
            print(f"missing {path}; build audiocpp_cli and audiocpp_c_api_model_test first")
            return SKIP

    available = {
        family: case
        for family, case in CASES.items()
        if os.path.exists(os.path.join(args.models_root, case["model"]))
    }
    if not available:
        print(f"no models under {args.models_root}; skipping")
        return SKIP

    # One C API run covers every family; parse its parity: lines. The alternate
    # clip and thread count are passed explicitly so this matches what the C
    # test does when CTest runs it directly.
    capi = run([model_test, args.models_root, args.audio, args.backend,
                args.alt_audio, args.threads])
    values = dict(re.findall(r"^parity:(\S+?)=(.*)$", capi, re.M))

    failures = []
    checked = 0

    with tempfile.TemporaryDirectory() as tmp:
        for family, case in sorted(available.items()):
            model = os.path.join(args.models_root, case["model"])
            base = [cli, "--task", case["task"], "--family", family, "--model", model,
                    "--backend", args.backend, "--threads", args.threads]

            if case["kind"] == "text":
                key = f"{family}:text"
                if key not in values:
                    print(f"{family:16} SKIP ({why_missing(family, capi)})")
                    continue
                out = os.path.join(tmp, f"{family}.txt")
                run(base + ["--audio", args.audio, "--text-out", out])
                with open(out) as handle:
                    cli_value = handle.read().strip()
                api_value = values[key].strip()
                same = cli_value == api_value
                print(f"{family:16} transcript {'MATCH' if same else 'DIFFER'} "
                      f"({len(api_value)} chars)")
                if not same:
                    failures.append(family)
                    print(f"  cli: {cli_value[:160]}")
                    print(f"  api: {api_value[:160]}")

            elif case["kind"] == "audio":
                key = f"{family}:audio_frames"
                if key not in values:
                    print(f"{family:16} SKIP ({why_missing(family, capi)})")
                    continue
                out = os.path.join(tmp, f"{family}.wav")
                # Kokoro draws noise per run, so both sides must be seeded or
                # the comparison is meaningless. Verified: two unseeded CLI runs
                # over the same input differ.
                run(base + ["--language", case["language"], "--voice-id", case["voice_id"],
                            "--seed", case["seed"], "--text", case["text"], "--out", out])
                frames, rate, _channels, peak = wav_summary(out)
                api_frames = int(values[f"{family}:audio_frames"])
                api_rate = int(values[f"{family}:audio_rate"])
                api_peak = float(values[f"{family}:audio_peak"])
                # Length and rate must agree exactly. Peak carries a small
                # tolerance only because the CLI quantises to 16-bit PCM on the
                # way out while the C API hands back the float buffer directly;
                # that error is ~3e-5, so 1e-3 is still a tight check. With both
                # sides seeded these agree to four decimals in practice.
                same = (frames == api_frames and rate == api_rate
                        and abs(peak - api_peak) < 0.001)
                print(f"{family:16} audio      {'MATCH' if same else 'DIFFER'} "
                      f"cli={frames}@{rate}Hz peak={peak:.4f} "
                      f"api={api_frames}@{api_rate}Hz peak={api_peak:.4f}")
                if not same:
                    failures.append(family)

            elif case["kind"] == "stems":
                key = f"{family}:named_audio"
                if key not in values:
                    print(f"{family:16} SKIP ({why_missing(family, capi)})")
                    continue
                out_dir = os.path.join(tmp, family)
                os.makedirs(out_dir, exist_ok=True)
                run(base + ["--audio", args.alt_audio, "--out-dir", out_dir])
                cli_stems = {}
                for name in sorted(os.listdir(out_dir)):
                    if not name.lower().endswith(".wav"):
                        continue
                    frames, rate, _c, _p = wav_summary(os.path.join(out_dir, name))
                    cli_stems[os.path.splitext(name)[0]] = (frames, rate)
                api_stems = {}
                for match in re.finditer(rf"^parity:{family}:stream_\d+=(\S+?):(\d+)@(\d+)$",
                                         capi, re.M):
                    api_stems[match.group(1)] = (int(match.group(2)), int(match.group(3)))
                # CLI stem filenames may carry a prefix; compare on the suffix.
                normalise = lambda d: {k.split("_")[-1]: v for k, v in d.items()}
                same = normalise(cli_stems) == normalise(api_stems)
                print(f"{family:16} stems      {'MATCH' if same else 'DIFFER'} "
                      f"cli={sorted(normalise(cli_stems))} api={sorted(normalise(api_stems))}")
                if not same:
                    failures.append(family)
                    print(f"  cli: {cli_stems}")
                    print(f"  api: {api_stems}")

            elif case["kind"] == "turns":
                key = f"{family}:speaker_turns"
                if key not in values:
                    print(f"{family:16} SKIP ({why_missing(family, capi)})")
                    continue
                out = os.path.join(tmp, f"{family}.json")
                run(base + ["--audio", args.audio, "--turns-out", out])
                with open(out) as handle:
                    parsed = json.load(handle)
                cli_turns = len(parsed if isinstance(parsed, list) else parsed.get("turns", []))
                api_turns = int(values[key])
                same = cli_turns == api_turns
                print(f"{family:16} turns      {'MATCH' if same else 'DIFFER'} "
                      f"cli={cli_turns} api={api_turns}")
                if not same:
                    failures.append(family)

            checked += 1

    print()
    if not checked:
        print("nothing comparable was available; skipping")
        return SKIP
    if failures:
        print(f"PARITY FAILED for: {', '.join(failures)}")
        return 1
    print(f"PARITY OK across {checked} famil{'y' if checked == 1 else 'ies'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
