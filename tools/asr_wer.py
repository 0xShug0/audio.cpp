#!/usr/bin/env python3
"""Word error rate for an ASR family against the LibriSpeech clips in this tree.

ASR is one of the few audio tasks with a ground truth shipped next to the audio, so it
can be scored rather than listened to. `assets/asr_validation/librispeech/` holds four
clips with their reference transcripts; this runs a model over them and reports WER.

⚠ FOUR CLIPS IS A SANITY CHECK, NOT A BENCHMARK. About 120 words total, so one bad
substitution moves the number by nearly a point. It answers "does this model work and
roughly how does it compare", not "which model is better".

  tools/asr_wer.py --cli build/bin/audiocpp_cli --family vibevoice_asr_streaming \
                   --model path/to.gguf [--backend cuda] [--label 7B]
"""
import argparse, json, pathlib, re, subprocess, sys


def normalize(text):
    """LibriSpeech references are uppercase and unpunctuated; match that."""
    text = text.upper().replace("-", " ")
    text = re.sub(r"[^A-Z' ]", " ", text)
    return text.split()


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", default="16")
    parser.add_argument("--label", default="")
    parser.add_argument("--assets", default="assets/asr_validation/librispeech")
    args = parser.parse_args()

    clips = sorted(pathlib.Path(args.assets).glob("*.wav"))
    if not clips:
        print(f"no clips under {args.assets}", file=sys.stderr)
        return 2

    total_errors = total_words = 0
    print(f"{'clip':46s} {'WER':>7s}  {'err/words':>10s}")
    for clip in clips:
        reference = normalize(clip.with_suffix(".txt").read_text())
        result = subprocess.run(
            [args.cli, "--task", "asr", "--family", args.family, "--model", args.model,
             "--backend", args.backend, "--threads", args.threads, "--audio", str(clip)],
            capture_output=True, text=True, timeout=1800)
        # ⚠ Two output shapes. A diarizing family (vibevoice_asr_streaming) emits
        # `speaker_turns=[{...}]` plus a MULTI-LINE text_output with "Speaker N:"
        # prefixes; a plain one (parakeet_tdt) puts everything after `text_output=` on
        # one line. Reading only the remainder of the text_output line scores a working
        # model at 100% WER, which is how this was found.
        hypothesis = ""
        for line in result.stdout.splitlines():
            if line.startswith("speaker_turns="):
                try:
                    turns = json.loads(line[len("speaker_turns="):])
                    hypothesis = " ".join(turn.get("text", "") for turn in turns)
                except json.JSONDecodeError:
                    pass
        if not hypothesis.strip():
            collecting = False
            parts = []
            for line in result.stdout.splitlines():
                if line.startswith("text_output="):
                    collecting = True
                    parts.append(line[len("text_output="):])
                elif collecting:
                    if "=" in line.split(" ")[0] and not line.startswith(" "):
                        break          # the next key=value field, so the value ended
                    parts.append(line)
            hypothesis = " ".join(parts)
        hypothesis = re.sub(r"Speaker\s+\d+\s*:", " ", hypothesis)
        if not hypothesis.strip() and result.returncode != 0:
            hypothesis = ""
            print(f"  (exit {result.returncode}) {result.stderr.strip().splitlines()[-1][:80] if result.stderr.strip() else ''}")
        words = normalize(hypothesis)
        errors = edit_distance(reference, words)
        total_errors += errors
        total_words += len(reference)
        flag = "" if words else "   <- NO OUTPUT"
        print(f"{clip.stem[:46]:46s} {100*errors/max(len(reference),1):6.1f}%  {errors:4d}/{len(reference):<5d}{flag}")

    label = f" [{args.label}]" if args.label else ""
    print(f"\nWER{label}: {100*total_errors/max(total_words,1):.2f}%  "
          f"({total_errors} errors over {total_words} words, {len(clips)} clips)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
