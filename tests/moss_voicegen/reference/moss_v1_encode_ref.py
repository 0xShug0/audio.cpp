# Capture matched prepared samples and independent Python encoder codes.
# v1 uses mono 24 kHz / 1920-sample frames; v2 uses stereo 48 kHz / 3840.
# Generated files stay in the explicitly supplied external output directory.
import argparse, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=("v1", "v2"), default="v1")
ap.add_argument("--codec", required=True)
ap.add_argument("--wav", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

model = AutoModel.from_pretrained(args.codec, trust_remote_code=True).eval().to(args.device)
sr_model = int(model.sampling_rate)
print(f"codec sampling_rate={sr_model}", flush=True)

wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)   # [samples, channels]
wav = wav.T                                                     # [channels, samples]
if args.variant == "v1":
    if wav.shape[0] != 1:
        wav = wav.mean(axis=0, keepdims=True)
else:
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    if wav.shape[0] != 2:
        raise ValueError("v2 capture requires mono or stereo input")
if sr != sr_model:
    raise SystemExit(f"fixture is {sr} Hz, codec wants {sr_model}; pick a native-rate fixture "
                     "rather than resampling here, so resampling cannot explain a mismatch")

# Trim to a whole number of codec frames. The C++ side reports valid_frames from
# the same arithmetic; leaving a partial frame makes the last column ambiguous.
spf = 1920 if args.variant == "v1" else 3840
frames = wav.shape[1] // spf
wav = wav[:, : frames * spf]
print(f"prepared waveform: {wav.shape} ({frames} frames of {spf})", flush=True)

x = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0).to(args.device)  # [1, 1, samples]
with torch.no_grad():
    enc = model.encode(x, return_dict=True)
codes = enc.audio_codes                      # [num_quantizers, batch, frames]
codes = codes.squeeze(1).cpu().numpy().astype(np.int32)
print(f"codes: {codes.shape} (num_quantizers, frames)", flush=True)

wav.astype(np.float32).tofile(args.out_prefix + "_prepared.f32")
with open(args.out_prefix + "_codes.csv", "w") as f:
    for row in codes:
        f.write(",".join(str(int(v)) for v in row) + "\n")
print(f"wrote {args.out_prefix}_prepared.f32  channels={wav.shape[0]} samples={wav.shape[1]}")
print(f"wrote {args.out_prefix}_codes.csv     rows={codes.shape[0]} cols={codes.shape[1]}")
