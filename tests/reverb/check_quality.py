"""Intelligibility check of Reverb packages with Whisper small.en: preset and cloned voices.

    python tests/reverb/check_quality.py --cli build/bin/audiocpp_cli --whisper whisper-small.en \
        --model Reverb-TTS-GGUF/reverb-tts-q8_0.gguf [--model Reverb-TTS-GGUF/reverb-tts-f16.gguf] \
        [--reverb Reverb-TTS --codec higgs-audio-v2-tokenizer]   # also score upstream reverb-tts

Six sentences, each spoken by the en_female_narrator preset and by a voice cloned from
assets/resources/b.wav, through one long-lived audiocpp_cli session per package (--request-sequence).
Reports word error rate against the input text. Needs transformers, torchaudio and soundfile.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Please remind me to call the dentist tomorrow morning before work.",
    "Voice control on Linux should feel fast, private and dependable.",
    "She opened the window and let the cool evening air into the room.",
    "Our team will review the proposal and send feedback by Friday.",
    "If the build fails again, check the logs before you restart the server.",
]
REFERENCE = ROOT / "assets/resources/b.wav"
# Spoken form, as Verbalis prepares it upstream (audio.cpp does not verbalize numbers).
REFERENCE_TEXT = ("Some call me nature. Others call me Mother Nature. I've been here for over four point five "
                  "billion years. twenty-two thousand five hundred times longer than you.")


def words(text):
    return re.sub(r"[^a-z' ]", " ", text.lower().replace("-", " ")).split()


def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def requests():
    items = []
    for i, text in enumerate(SENTENCES):
        items.append({"id": f"preset_{i}", "text": text, "seed": i, "voice_id": "en_female_narrator"})
        items.append({"id": f"clone_{i}", "text": text, "seed": i, "voice_ref": str(REFERENCE),
                      "reference_text": REFERENCE_TEXT, "language": "en"})
    return items


def run_upstream(args, out):
    from reverb_tts.tts import TTS
    tts = TTS.from_pretrained(args.reverb, variant="fp16", device="cpu", codec=args.codec)
    voices = {"preset": tts.preset_voice("en_female_narrator"),
              "clone": tts.voice(str(REFERENCE), REFERENCE_TEXT, "en", prepare=False)}
    for item in requests():
        kind = item["id"].split("_")[0]
        sf.write(out / f"{item['id']}.wav", tts.generate(item["text"], voices[kind], "en", seed=item["seed"]),
                 tts.sample_rate)


def run_cli(args, model, out):
    sequence = out / "requests.json"
    sequence.write_text(json.dumps({"requests": requests()}))
    subprocess.run([args.cli, "--task", "tts", "--family", "reverb", "--model", model, "--backend", args.backend,
                    "--threads", str(args.threads), "--request-sequence", str(sequence), "--out-dir", str(out)],
                   check=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", required=True)
    parser.add_argument("--whisper", required=True, help="openai/whisper-small.en snapshot")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--reverb")
    parser.add_argument("--codec")
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(args.whisper)
    asr = WhisperForConditionalGeneration.from_pretrained(args.whisper).eval()
    runs = ([("reverb-tts (upstream)", None)] if args.reverb and args.codec else []) + [(m, m) for m in args.model]
    worst = 0.0
    for label, model in runs:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_upstream(args, out) if model is None else run_cli(args, model, out)
            errors = total = 0
            for item in requests():
                audio, sr = sf.read(out / f"{item['id']}.wav", dtype="float32")
                audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
                features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
                with torch.inference_mode():
                    hyp = processor.batch_decode(asr.generate(features), skip_special_tokens=True)[0]
                e = distance(words(item["text"]), words(hyp))
                errors, total = errors + e, total + len(words(item["text"]))
                if e:
                    print(f"   {item['id']}: {hyp.strip()}")
            wer = 100.0 * errors / total
            worst = max(worst, wer)
            print(f"{label}: WER {wer:.2f}% ({errors}/{total} words)")
    return 0 if worst <= 5.0 else 1


if __name__ == "__main__":
    sys.exit(main())
