import json, sys, torch, soundfile as sf
from transformers import AutoProcessor
MODEL="/mnt/data/models/MOSS-TTSD-v1.0"; CODEC="/mnt/data/models/MOSS-Audio-Tokenizer"
S=sys.argv[1]; OUT=sys.argv[2]
proc=AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, codec_path=CODEC)
sr=int(proc.model_config.sampling_rate); n_vq=int(proc.model_config.n_vq)
def load(p):
    x,r=sf.read(p, dtype="float32")
    if x.ndim>1: x=x.mean(axis=1)
    t=torch.from_numpy(x)[None,:]
    if r!=sr:
        import torchaudio; t=torchaudio.functional.resample(t,r,sr)
    return t
w1,w2=load(f"{S}/ref_s1.wav"),load(f"{S}/ref_s2.wav")
ref=proc.encode_audios_from_wav([w1,w2], sampling_rate=sr)
prompt_audio=proc.encode_audios_from_wav([torch.cat([w1,w2],dim=-1)], sampling_rate=sr)[0]
REFTEXT=("[S1] This is the first voice, and it speaks quite clearly. "
         "[S2] And this is the second voice, which sounds rather different.")
TEXT=("[S1] The train leaves at four in the afternoon. [S2] I packed the blue suitcase already. "
      "[S1] We should bring something to read. [S2] There is a bookshop by the platform.")
um=proc.build_user_message(text=f"{REFTEXT} {TEXT}", reference=ref)
msgs=[um, proc.build_assistant_message(audio_codes_list=[prompt_audio])]
ids=proc([msgs], mode="continuation")["input_ids"]
t=lambda c: c[:, :n_vq].transpose(0,1).contiguous().tolist()
json.dump({
 "name":"real_clone","text":f"{REFTEXT} {TEXT}",
 "instruction":None,"language":None,"quality":None,"sound_event":None,"ambient_sound":None,
 "content":um["content"] if isinstance(um,dict) else um._content,
 "reference_codes":[t(ref[0]), t(ref[1])],
 "assistant_codes":t(prompt_audio),
 "input_ids":ids[0].tolist(),
}, open(f"{OUT}/ref_prompt_real_clone.json","w"))
print("prefix rows:", ids.shape[1], "| ref frames:", ref[0].shape[0], ref[1].shape[0],
      "| assistant frames:", prompt_audio.shape[0])
