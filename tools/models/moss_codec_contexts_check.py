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

    python3 tools/models/moss_codec_contexts_check.py --nano-config /path/to/config.json
"""
import argparse
import json
from pathlib import Path
import re
import sys

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-config", type=Path)
    parser.add_argument("--v2-config", type=Path)
    parser.add_argument("--nano-config", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] /
                        "src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp")
    args = parser.parse_args()
    if not any((args.v1_config, args.v2_config, args.nano_config)):
        parser.error("supply at least one of --v1-config, --v2-config or --nano-config")
    src = args.source.read_text()
    failed = False
    for label, cfg in (("v1", args.v1_config), ("v2", args.v2_config), ("nano", args.nano_config)):
        if cfg is None:
            continue
        derived = derive(cfg)
        actual = hardcoded(src, f"moss_audio_tokenizer_{label}_config")
        print(f"=== {label}")
        for side in ("encoder", "decoder"):
            want, final, expect = derived[side]
            got = actual[side]
            ok = want == got
            print(f"   {side:8s} want {want}")
            print(f"   {'':8s} got  {got}   {'OK' if ok else '<-- MISMATCH'}")
            failed |= not ok
            if abs(final - expect) > 1:
                print(f"   {'':8s} frame-rate chain does not close: {final} vs {expect}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
