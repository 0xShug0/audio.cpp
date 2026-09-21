# Dump backbone parity fixtures for MOSS-TTS-v1.5, in the schema the existing
# moss_voicegen backbone_parity harness already reads.
import sys, json, torch
from transformers import AutoModel, AutoProcessor
MODEL="/mnt/data/models/MOSS-TTS-v1.5"; CODEC="/mnt/data/models/MOSS-Audio-Tokenizer"
OUT="/home/chris/Programming/audio.cpp-fork/tests/moss_tts_delay/reference"
import os; os.makedirs(OUT, exist_ok=True)
dev="cuda"
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
model=AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).eval().to(dev)
pm=sys.modules[type(proc).__module__]

NAME="en_instruction"
TEXT="The quick brown fox jumps over the lazy dog, and then says hello."
INSTR="A warm, friendly female voice speaking clearly at a natural pace."
msg=pm.UserMessage(text=TEXT, instruction=INSTR)
ids=proc([msg],mode="generation")["input_ids"]           # (1, rows, 1+n_vq)
rows=ids.shape[1]; nvq=ids.shape[2]-1
print(f"prompt rows={rows} channels={ids.shape[2]} (1+{nvq})", flush=True)

with torch.no_grad():
    out=model(input_ids=ids.to(dev), attention_mask=torch.ones(ids.shape[:2],dtype=torch.bool,device=dev))
# hidden_states[-1] is the post-final-norm state the heads consume, which is what
# the C++ prefill() returns.
hid=out.hidden_states[-1][0].float().cpu()               # (rows, hidden)
hidden_size=hid.shape[1]
print(f"hidden: {tuple(hid.shape)}", flush=True)

probe=[0, rows//2, rows-1]
# Record the fields as the message actually carried them: writing a value the
# message did not use makes the fixture describe a prompt nobody rendered.
json.dump({
  "name": NAME,
  "text": msg.text,
  "instruction": msg.instruction,
  "language": msg.language,
  "tokens": msg.tokens,
  "content": msg._content,
  "input_ids": ids[0].tolist(),
}, open(f"{OUT}/ref_prompt_{NAME}.json","w"))
json.dump({
  "name": NAME, "rows": rows, "hidden_size": hidden_size,
  "last_hidden": hid[-1].tolist(),
  "last_hidden_abs_max": float(hid[-1].abs().max()),
  "probe_positions": probe,
  "probe_hidden": [hid[p].tolist() for p in probe],
}, open(f"{OUT}/ref_hidden_{NAME}.json","w"))
print(f"wrote {OUT}/ref_prompt_{NAME}.json and ref_hidden_{NAME}.json")
print(f"last_hidden abs_max={float(hid[-1].abs().max()):.4f}")
