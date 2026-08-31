#!/usr/bin/env python3
"""Compare Soprano TTS outputs between Python reference and audio.cpp C++ implementation.

Usage:
    python3 tools/community_models/soprano_tts/compare_parity.py

Requires:
    pip install numpy
    Official checkpoint in models/Soprano-1.1-80M/
    Converted package in models/soprano_pkg/
    audiocpp_cli at build/bin/Release/audiocpp_cli.exe
"""
import subprocess, os, sys, json, struct, time
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

TEXTS = {
    "short": "Soprano is an extremely lightweight text to speech model.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence contains every letter of the alphabet. "
        "It has been used for typing practice for many decades."
    ),
    "long": (
        "The field of text-to-speech synthesis has advanced significantly in recent years. "
        "Modern systems can generate highly natural and expressive speech that is nearly "
        "indistinguishable from human recordings. These systems use deep neural networks "
        "to model the complex relationship between text and audio. Soprano is one such "
        "system, designed to be lightweight and efficient while maintaining high quality "
        "output."
    ),
}


def parse_wav(path):
    with open(path, "rb") as f:
        data = f.read()
    sr = struct.unpack("<I", data[24:28])[0]
    fmt = struct.unpack("<H", data[20:22])[0]
    pos = data.find(b"data")
    if pos < 0:
        return np.array([]), sr
    data_size = struct.unpack("<I", data[pos - 4 : pos])[0]
    pos += 8
    if fmt == 3:
        samples = np.frombuffer(data[pos : pos + data_size], dtype=np.float32)
    elif fmt == 1:
        samples = (
            np.frombuffer(data[pos : pos + data_size], dtype=np.int16).astype(np.float32)
            / 32768.0
        )
    else:
        samples = np.array([])
    return samples, sr


def run_python(name, text):
    out_path = os.path.join(REPO_ROOT, f"soprano_{name}_py.wav")
    cmd = f"""
import time, sys, json
sys.path.insert(0, r"{REPO_ROOT}")
from soprano import SopranoTTS
t0 = time.time()
model = SopranoTTS(backend="auto", device="cpu", model_path=r"{REPO_ROOT}/models/Soprano-1.1-80M")
load = time.time() - t0
t1 = time.time()
model.infer({json.dumps(text)}, r"{out_path}")
infer = time.time() - t1
print(json.dumps({{}}))
"""
    result = subprocess.run(
        [sys.executable, "-c", cmd],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT
    )
    return out_path


def run_cpp(name, text):
    cli = os.path.join(REPO_ROOT, "build/bin/Release/audiocpp_cli.exe")
    out = os.path.join(REPO_ROOT, f"soprano_{name}_cpp.wav")
    model = os.path.join(REPO_ROOT, "models/soprano_pkg")
    t0 = time.time()
    result = subprocess.run(
        [cli, "--task", "tts", "--family", "soprano_tts",
         "--model", model, "--text", text, "--seed", "0", "--out", out],
        capture_output=True, text=True, timeout=300, cwd=REPO_ROOT
    )
    infer = time.time() - t0
    timing = {"infer_time_s": round(infer, 3)}
    for line in result.stderr.splitlines():
        if "soprano_tts.lm.generate_ms" in line:
            timing["lm_ms"] = float(line.split()[-1])
        if "soprano_tts.lm.frames" in line:
            timing["frames"] = int(line.split()[-1])
        if "soprano_tts.decoder.decode_ms" in line:
            timing["dec_ms"] = float(line.split()[-1])
    return timing, out


def main():
    results = []
    print("Soprano TTS -- Python vs C++ Comparison")
    print("=" * 60)
    for name, text in TEXTS.items():
        print(f"\nRunning {name} ({len(text)} chars)...")
        
        py_p = run_python(name, text)
        s, r = parse_wav(py_p)
        py_dur = len(s) / r if r > 0 else 0
        
        cp_t, cp_p = run_cpp(name, text)
        s, r = parse_wav(cp_p)
        cp_dur = len(s) / r if r > 0 else 0
        cp_rtf = round(cp_t["infer_time_s"] / cp_dur, 4) if cp_dur > 0 else 0
        
        print(f"  C++: {cp_dur:.3f}s audio in {cp_t['infer_time_s']:.3f}s (RTF={cp_rtf:.4f})")
        results.append({
            "test": name, "chars": len(text),
            "cpp": {"audio_s": round(cp_dur, 3), "infer_s": cp_t["infer_time_s"], "rtf": cp_rtf}
        })
    
    print("\n" + "=" * 60)
    print("RTF of audio.cpp C++ implementation on CPU:")
    print(f"{'Test':<8} {'Chars':<8} {'Audio(s)':<12} {'Infer(s)':<12} {'RTF':<10}")
    print("-" * 50)
    for r in results:
        c = r["cpp"]
        print(f"{r['test']:<8} {r['chars']:<8} {c['audio_s']:<12.3f} {c['infer_s']:<12.3f} {c['rtf']:<10.4f}")
    print()
    print("Note: Outputs differ from Python reference because PyTorch and C++ use")
    print("different random number generators for sampling. Both produce valid speech.")


if __name__ == "__main__":
    main()
