#!/usr/bin/env python3
"""Build the F ja->en long-form fixture (deterministic sample concat, E2 style).

Composition (16 kHz mono 16-bit, sample-aligned concat, stdlib wave only):
  common_voice_ja_20461197 (3.14 s) x4 = ~12.6 s, multi-chunk under fixed 8 s.
Reference (translation scoring) = the implementer-authored English rendering
  "I read a book written in Japanese." x4, joined with a single space
  (fixtures/f_tr_long_ja_en.txt). The Japanese transcript x4 ships alongside
  as provenance (fixtures/f_ja_long_ja.txt).

Re-running this script must reproduce both files byte-identically
(sha256 recorded in tests/granite_speech_2b/eval_manifest.jsonl).
"""
import hashlib
import pathlib
import wave

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
FIX = REPO / "tests/granite_speech_2b/fixtures"
JA = REPO / "assets/asr_validation/common_voice_ja/common_voice_ja_20461197.wav"

EN_RENDERING = "I read a book written in Japanese."


def main():
    frames = b""
    key = None
    params = None
    with wave.open(str(JA), "rb") as w:
        p = w.getparams()
        key, params = (p.nchannels, p.sampwidth, p.framerate, p.comptype), p
        one = w.readframes(w.getnframes())
    ja_text = (REPO / "assets/asr_validation/common_voice_ja" /
               "common_voice_ja_20461197.txt").read_text().strip()
    frames = one * 4
    out_wav = FIX / "f_ja_long.wav"
    with wave.open(str(out_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)
    (FIX / "f_ja_long_ja.txt").write_text((" ".join([ja_text] * 4)) + "\n")
    (FIX / "f_tr_long_ja_en.txt").write_text((" ".join([EN_RENDERING] * 4)) + "\n")
    digest = hashlib.sha256(out_wav.read_bytes()).hexdigest()
    dur = len(frames) / (params.framerate * params.sampwidth * params.nchannels)
    print(f"wav: {out_wav} {len(frames)} frames {dur:.2f} s sha256={digest}")


if __name__ == "__main__":
    main()
