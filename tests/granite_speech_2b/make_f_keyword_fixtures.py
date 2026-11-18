#!/usr/bin/env python3
"""Build the F keyword-biasing rare-name fixture (espeak synth, deterministic).

Rationale (brief F1): paired runs on a clip containing rare names (keywords IN
the intended transcript = recall probe) plus clips where the list must
produce NO insertions. No natural rare-name audio ships in-tree, so the
fixture is synthesized with the local espeak path (brief: espeak check).

Generation (stdlib + espeak CLI only; no sox dependency — the committed wav
keeps espeak-native 22050 Hz mono 16-bit and the app resamples):
  espeak -v en -s 150 -a 100 "Xochitl Quetzal arrived in Tlalnepantla with
  Zbigniew Brzezinski." -w f_kw_rare.wav

Pinned: eSpeak text-to-speech 1.48.15 (16.Apr.15) as shipped on this box.
Re-running with the same espeak must reproduce the wav byte-identically
(sha256 recorded in tests/granite_speech_2b/eval_manifest.jsonl); a different
espeak version changes the bytes and must be re-pinned, never silently kept.

Reference = the intended sentence (exact by construction: it is the synth
input). The model baseline mangles the names; the keyword run is scored by
recall of the three multiword terms. espeak prosody is robotic — the fixture
measures the PAIRED with/without-keywords delta on identical audio, never
absolute quality.
"""
import hashlib
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
FIX = REPO / "tests/granite_speech_2b/fixtures"

SENTENCE = "Xochitl Quetzal arrived in Tlalnepantla with Zbigniew Brzezinski."


def main():
    out_wav = FIX / "f_kw_rare.wav"
    subprocess.run(
        ["espeak", "-v", "en", "-s", "150", "-a", "100", SENTENCE,
         "-w", str(out_wav)],
        check=True)
    out_txt = FIX / "f_kw_rare.txt"
    out_txt.write_text(SENTENCE + "\n")
    digest = hashlib.sha256(out_wav.read_bytes()).hexdigest()
    print(f"wav: {out_wav} sha256={digest}")
    print(f"txt: {out_txt}")


if __name__ == "__main__":
    main()
