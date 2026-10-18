# Reference capture for the MOSS-Audio-Tokenizer v1 encode path.
#
# The v1 encoder is implemented in the framework but no model calls it and no
# test covers it (v2 encode and v1 decode are both covered). This dumps the
# prepared waveform exactly as fed to the reference, plus the codes it produces,
# so the C++ side can be compared on identical input.
import argparse, sys
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("--codec", default="/mnt/data/models/MOSS-Audio-Tokenizer")
ap.add_argument("--wav", default="/home/chris/Programming/audio.cpp-fork/tests/omnivoice/assets/ref_audio_01.wav")
ap.add_argument("--out-prefix", default="/tmp/claude-1000/-home-chris-Programming-audio-cpp-fork/c0faadcf-4d9b-4eed-8a8d-ff0b7d0a030c/scratchpad/v1enc")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()

model = AutoModel.from_pretrained(args.codec, trust_remote_code=True).eval().to(args.device)
sr_model = int(model.sampling_rate)
print(f"codec sampling_rate={sr_model}", flush=True)

wav, sr = sf.read(args.wav, dtype="float32", always_2d=True)   # [samples, channels]
wav = wav.T                                                     # [channels, samples]
if wav.shape[0] != 1:
    wav = wav.mean(axis=0, keepdims=True)
if sr != sr_model:
    raise SystemExit(f"fixture is {sr} Hz, codec wants {sr_model}; pick a native-rate fixture "
                     "rather than resampling here, so resampling cannot explain a mismatch")

# Trim to a whole number of codec frames. The C++ side reports valid_frames from
# the same arithmetic; leaving a partial frame makes the last column ambiguous.
spf = 1920
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
