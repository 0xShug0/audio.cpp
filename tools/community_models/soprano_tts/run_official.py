#!/usr/bin/env python3
"""Run official SopranoTTS for ground-truth comparison.
Usage:
    python3 run_official.py --text "..." --out output.wav
"""
import argparse, json, os, time, wave

def run_inference(text: str, out_path: str, model_path: str = "models/Soprano-1.1-80M"):
    from soprano import SopranoTTS
    
    load_t0 = time.time()
    model = SopranoTTS(backend="auto", device="cpu", model_path=model_path)
    load_time = time.time() - load_t0
    
    infer_t0 = time.time()
    out = model.infer(text, out_path)
    infer_time = time.time() - infer_t0
    
    with wave.open(out_path, "r") as wf:
        audio_dur = wf.getnframes() / wf.getframerate()
    
    result = {
        "text": text,
        "text_chars": len(text),
        "load_time_s": round(load_time, 3),
        "infer_time_s": round(infer_time, 3),
        "audio_duration_s": round(audio_dur, 3),
        "rtf": round(infer_time / audio_dur, 4) if audio_dur > 0 else 0,
        "output": out_path,
        "output_bytes": os.path.getsize(out_path),
    }
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="Soprano is an extremely lightweight text to speech model.")
    ap.add_argument("--out", default="soprano_official.wav")
    ap.add_argument("--model", default="models/Soprano-1.1-80M")
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()
    result = run_inference(args.text, args.out, args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
