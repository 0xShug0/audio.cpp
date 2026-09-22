# Dump prompt parity fixtures for MOSS-TTSD, in the schema the moss_tts_v15
# prompt_parity harness reads.
#
# Prompt-only, so the model weights are never loaded: the processor is built for
# its tokenizer and its message rendering. The codec IS loaded, because
# AutoProcessor constructs it -- a reference with audio needs real codes and
# there is no way to ask the reference for a prompt without one.
import json, os, sys
import torch
from transformers import AutoProcessor

MODEL = "/mnt/data/models/MOSS-TTSD-v1.0"
CODEC = "/mnt/data/models/MOSS-Audio-Tokenizer"
OUT = os.path.dirname(os.path.abspath(__file__))

proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
pm = sys.modules[type(proc).__module__]
n_vq = int(proc.model_config.n_vq)
sr = int(proc.model_config.sampling_rate)

def dump(name, msg):
    ids = proc([msg], mode="generation")["input_ids"]   # (1, rows, 1+n_vq)
    rec = {
        "name": name,
        "text": msg.text,
        "instruction": msg.instruction,
        "language": msg.language,
        # Recorded as the message actually carried them. Writing a value the
        # message did not use makes the fixture describe a prompt nobody rendered.
        "quality": msg.quality,
        "sound_event": msg.sound_event,
        "ambient_sound": msg.ambient_sound,
        "content": msg._content,
        "input_ids": ids[0].tolist(),
    }
    # Positional, one entry per speaker, null where that speaker brought no
    # audio. The layout IS the test: a fixture that only listed the present
    # speakers could not tell a "[S2]: None" line from a missing one.
    #
    # Stored [n_vq][frames], the codec layout the C++ ReferenceAudio takes, while
    # the message wants [frames][channels] -- hence the transpose. Truncated to
    # n_vq first: the tokenizer emits 32 RVQ layers and this model reads the
    # first 16 (processing_moss_tts.py, "Always follow model RVQ channels"), so a
    # fixture carrying all 32 would describe codes the model never saw.
    if isinstance(msg.reference, list):
        rec["reference_codes"] = [
            None if r is None else r[:, :n_vq].transpose(0, 1).contiguous().tolist()
            for r in msg.reference
        ]
    json.dump(rec, open(f"{OUT}/ref_prompt_{name}.json", "w"))
    print(f"wrote ref_prompt_{name}.json  rows={ids.shape[1]} channels={ids.shape[2]} (1+{n_vq})")

DIALOGUE = "[S1] So what did you make of it? [S2] Honestly, I was not expecting that ending."

# 1. Two speakers, neither cloned: the "[S<n>]: None" branch, twice.
dump("dialogue_none", pm.UserMessage(text=DIALOGUE, reference=[None, None]))

# 2. No speakers named at all: the `reference is None` branch.
dump("dialogue_no_reference", pm.UserMessage(text=DIALOGUE))

# 3. One cloned, one not -- the mixed case, which is where a shape error in
#    either branch shows up as a token shift rather than a wrong count.
torch.manual_seed(0)
wav = torch.zeros(1, sr)            # one second; content is irrelevant to prompt shape
wav[0, ::137] = 0.2                 # some structure so the codec does not emit a constant
codes = proc.encode_audios_from_wav([wav], sampling_rate=sr)
dump("dialogue_mixed", pm.UserMessage(text=DIALOGUE, reference=[codes[0], None]))

# 4. Continuation: the mode TTSD actually clones in. The user turn names the
#    speakers, and an assistant turn carries the concatenated reference audio
#    that the model continues from. The assistant span uses a different pair of
#    slot tokens from a user span -- gen for the frames, delay for the trailing
#    n_vq - 1 -- and the turn is left unterminated so generation continues it.
cont_msgs = [
    pm.UserMessage(text=DIALOGUE, reference=[codes[0], None]),
    pm.AssistantMessage(audio_codes_list=[codes[0]]),
]
ids = proc([cont_msgs], mode="continuation")["input_ids"]
json.dump({
    "name": "dialogue_continuation",
    "text": DIALOGUE,
    "instruction": None, "language": None,
    "quality": None, "sound_event": None, "ambient_sound": None,
    "content": cont_msgs[0]._content,
    "reference_codes": [codes[0][:, :n_vq].transpose(0, 1).contiguous().tolist(), None],
    "assistant_codes": codes[0][:, :n_vq].transpose(0, 1).contiguous().tolist(),
    "input_ids": ids[0].tolist(),
}, open(f"{OUT}/ref_prompt_dialogue_continuation.json", "w"))
print(f"wrote ref_prompt_dialogue_continuation.json  rows={ids.shape[1]}")
