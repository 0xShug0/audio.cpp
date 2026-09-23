"""Check the MOSS codec attention contexts against the checkpoints they describe.

audio.cpp hardcodes each codec's per-stage attention context as a step count.
The checkpoint states it as a DURATION, per stage, which the reference turns into
steps with `int(round(current_frame_rate * context_duration))` -- where the rate
is the one *before* that transformer, not after its patch.

Getting that wrong is silent: the decoder still runs, still produces speech, and
only diverges once a sequence outgrows the real window, so short clips look
perfect and long ones drift. That is exactly what #663 reported, and it was the
Nano decoder, whose contexts were each twice what they should have been.

Needs the checkpoints' config.json files, so it is a tool rather than a test:

    python3 tools/check_moss_codec_contexts.py
"""
import json, re, sys

def derive(cfg_path):
    c = json.load(open(cfg_path))
    default = float(c.get("causal_transformer_context_duration", 10.0))
    ci = 2 if c.get("enable_channel_interleave") else 1
    code_rate = c["sampling_rate"] / c["downsample_rate"]
    out = {}
    # decoder: starts at the code rate, rate *= patch after each module
    rate = code_rate; dec = []
    for m in c["decoder_kwargs"]:
        if m["module_type"] == "PatchedPretransform":
            rate *= m["patch_size"]
        else:
            dec.append(int(round(rate * float(m.get("context_duration", default)))))
    out["decoder"] = (dec, rate, c["sampling_rate"] * ci)
    # encoder: starts at the output rate, rate /= patch
    rate = c["sampling_rate"] * ci; enc = []
    for m in c["encoder_kwargs"]:
        if m["module_type"] == "PatchedPretransform":
            rate /= m["patch_size"]
        else:
            enc.append(int(round(rate * float(m.get("context_duration", default)))))
    out["encoder"] = (enc, rate, code_rate)
    return out

def hardcoded(src, fn):
    body = src.split(f"MossAudioTokenizerConfig {fn}()")[1].split("\n}")[0]
    out = {}
    for side in ("encoder_stages", "decoder_stages"):
        block = body.split(f"config.{side} = {{")[1].split("};")[0]
        out[side.split("_")[0]] = [int(r.split(",")[6]) for r in
                                   re.findall(r"\{([^}]*)\}", block)]
    return out

src = open("src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp").read()
for label, cfg, fn in [
    ("v1",   "/mnt/data/models/MOSS-Audio-Tokenizer/config.json",        "moss_audio_tokenizer_v1_config"),
    ("v2",   "/mnt/data/models/MOSS-Audio-Tokenizer-v2-cfg/config.json", "moss_audio_tokenizer_v2_config"),
    ("nano", "/mnt/data/models/MOSS-Audio-Tokenizer-Nano/config.json",   "moss_audio_tokenizer_nano_config"),
]:
    d = derive(cfg); h = hardcoded(src, fn)
    print(f"=== {label}")
    for side in ("encoder", "decoder"):
        want, final, expect = d[side]
        got = h[side]
        ok = want == got
        print(f"   {side:8s} want {want}")
        print(f"   {'':8s} got  {got}   {'OK' if ok else '<-- MISMATCH'}")
        if abs(final - expect) > 1:
            print(f"   {'':8s} frame-rate chain does not close: {final} vs {expect}")
