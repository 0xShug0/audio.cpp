# Cloning-prompt fixture: the path that carries reference codes into the prompt.
import sys, json, os
import numpy as np, soundfile as sf, torch
from transformers import AutoModel, AutoProcessor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--codec", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--reference-audio", type=Path, required=True)
args = parser.parse_args()
MODEL, CODEC, OUT = str(args.model), str(args.codec), str(args.output)
args.output.mkdir(parents=True, exist_ok=True)
REF = str(args.reference_audio)

os.makedirs(OUT, exist_ok=True)
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
pm=sys.modules[type(proc).__module__]

tok=proc.audio_tokenizer
x,sr=sf.read(REF,dtype="float32",always_2d=True); x=x.T.mean(axis=0,keepdims=True)
x=x[:, :(x.shape[1]//1920)*1920]
with torch.no_grad():
    enc=tok.encode(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(tok.device), return_dict=True)
codes=enc.audio_codes.squeeze(1)                    # (n_vq, frames)
print("reference codes:", tuple(codes.shape), flush=True)

NAME="en_clone"
TEXT="This sentence should be spoken in the voice from the reference recording."
msg=pm.UserMessage(text=TEXT, reference=[codes.transpose(0,1).contiguous()])
ids=proc([msg], mode="generation", n_vq=32)["input_ids"]
print("prompt rows:", tuple(ids.shape), flush=True)

json.dump({
  "name": NAME,
  "text": msg.text, "instruction": msg.instruction, "language": msg.language, "tokens": msg.tokens,
  "content": msg._content,
  "reference_codes": codes.cpu().tolist(),          # [n_vq][frames], codec layout
  "input_ids": ids[0].tolist(),
}, open(f"{OUT}/ref_prompt_{NAME}.json","w"))
print(f"wrote {OUT}/ref_prompt_{NAME}.json")
