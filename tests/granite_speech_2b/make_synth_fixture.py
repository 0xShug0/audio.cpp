#!/usr/bin/env python3
"""Build the E2 synthetic >60 s chunk-requiring fixture (finding: no natural
long-audio in-tree fixture exists; the plan allows synthetic concatenations
for joins/limits).

Composition (all 16 kHz mono 16-bit, sample-aligned concat, deterministic):
  librispeech 6930-75918-0001 (14.225 s) x4 + 6930-75918-0000 (3.505 s) x2
  = 63.91 s, 188 reference words.
Reference = the six per-clip reference texts joined with a single space
(WER folding splits on whitespace, so reference joins are exact).

Provenance for tests/granite_speech_2b/fixtures/e2_synth_60s_plus.{wav,txt}.
Re-running this script must reproduce both files byte-identically
(sha256 recorded in tests/granite_speech_2b/eval_manifest.jsonl).
"""
import hashlib
import pathlib
import wave

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
FIX = REPO / "tests/granite_speech_2b/fixtures"
LIB = REPO / "assets/asr_validation/librispeech"

LONG = "librispeech_test_clean_6930-75918-0001"
SHORT = "librispeech_test_clean_6930-75918-0000"
PARTS = [LONG] * 4 + [SHORT] * 2


def main():
    frames = b""
    key = None
    params = None
    texts = []
    for part in PARTS:
        with wave.open(str(LIB / f"{part}.wav"), "rb") as w:
            p = w.getparams()
            k = (p.nchannels, p.sampwidth, p.framerate, p.comptype)
            if key is None:
                key, params = k, p
            assert k == key, f"format mismatch in {part}: {k} vs {key}"
            frames += w.readframes(w.getnframes())
        texts.append((LIB / f"{part}.txt").read_text().strip())
    out_wav = FIX / "e2_synth_60s_plus.wav"
    with wave.open(str(out_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    out_txt = FIX / "e2_synth_60s_plus.txt"
    out_txt.write_text(" ".join(texts) + "\n")
    digest = hashlib.sha256(out_wav.read_bytes()).hexdigest()
    dur = len(frames) / (params.framerate * params.sampwidth * params.nchannels)
    print(f"wav: {out_wav} {len(frames)} frames {dur:.2f} s sha256={digest}")
    print(f"txt: {out_txt} {sum(len(t.split()) for t in texts)} words")


if __name__ == "__main__":
    main()
