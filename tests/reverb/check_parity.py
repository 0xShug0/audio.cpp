"""End-to-end parity of the audio.cpp `reverb` family against upstream reverb-tts.

    python tests/reverb/check_parity.py --cli build/bin/audiocpp_cli --package Reverb-TTS-package \
        --reverb Reverb-TTS --codec higgs-audio-v2-tokenizer

Runs the same text, preset voice and seed through reverb_tts.TTS.generate (fp32 on the CPU, from the
fp16 checkpoint) and through audiocpp_cli with reverb.weight_type=f32 on the safetensors package, and
compares the waveforms. Identical generated codes give a correlation above 0.9999 (the remaining
difference is the codec's float order and 16-bit WAV rounding); a single differing code is far below.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from reverb_tts.tts import TTS

CASES = [
    ("Hello from Reverb, running inside audio dot cpp on the CPU.", "en_female_narrator", 0),
    ("Our team will review the proposal and send feedback by Friday.", "en_male_narrator_1", 7),
    ("Bonjour, je parle français maintenant.", "fr_female_standard", 3),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", required=True)
    parser.add_argument("--package", required=True, help="output of tools/gguf/reverb_convert.py")
    parser.add_argument("--reverb", required=True, help="software-mansion/Reverb-TTS snapshot")
    parser.add_argument("--codec", required=True, help="eustlb/higgs-audio-v2-tokenizer snapshot")
    parser.add_argument("--model-spec", default=str(Path(__file__).resolve().parents[2] / "model_specs"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    tts = TTS.from_pretrained(args.reverb, variant="fp16", device="cpu", codec=args.codec)
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for text, voice, seed in CASES:
            want = tts.generate(text, tts.preset_voice(voice), voice[:2], seed=seed)
            out = Path(tmp) / "cpp.wav"
            subprocess.run([args.cli, "--task", "tts", "--family", "reverb", "--model", args.package,
                            "--model-spec-override", args.model_spec, "--backend", "cpu",
                            "--threads", str(args.threads), "--text", text, "--request-option", f"voice_id={voice}",
                            "--request-option", f"seed={seed}", "--session-option", "reverb.weight_type=f32",
                            "--out", str(out)], check=True, capture_output=True)
            got, _ = sf.read(out, dtype="float32")
            same_length = len(got) == len(want)
            corr = float(np.corrcoef(got, want)[0, 1]) if same_length else float("nan")
            ok = same_length and corr > 0.9999
            failures += not ok
            print(f"{voice} seed {seed}: {len(got)} vs {len(want)} samples, correlation {corr:.8f}",
                  "ok" if ok else "MISMATCH")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
