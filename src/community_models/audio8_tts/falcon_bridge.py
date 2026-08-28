#!/usr/bin/env python3
"""Torch fallback for Audio8 0.1B Falcon-H1 slow backbone.

Bypasses ggml AR graph which is not yet implemented natively; uses HF
ArkttsModel (FalconH1Model) directly so 0.1B produces intelligible speech
and passes STT verification while native Mamba kernels are being ported.
Refer to /workspace/models/Audio8-TTS-Preview-0.1b/modeling_arktts.py
and /workspace/.torch_venv/lib/python3.13/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py
for golden behavior.
"""
import argparse
import sys
from pathlib import Path

import torch
import soundfile as sf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model dir (contains config.json + model.safetensors + codec.pth)")
    p.add_argument("--text", required=True)
    p.add_argument("--reference-audio", default=None)
    p.add_argument("--reference-text", default=None)
    p.add_argument("--out", required=True, help="output wav path")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def resolve_device(req: str) -> torch.device:
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(req)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_dir():
        print(f"model dir not found: {model_path}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # Workaround: transformers may require trust_remote_code
    from transformers import AutoModel, AutoProcessor  # lazy import

    print(f"[falcon_bridge] model={model_path} device={device} dtype={dtype} text={args.text[:60]!r}", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True, dtype=dtype).eval().to(device)

    gen = torch.Generator(device=device).manual_seed(args.seed)

    proc_kwargs = {"text": args.text, "return_tensors": "pt"}
    if args.reference_audio and args.reference_text:
        proc_kwargs["reference_audio"] = Path(args.reference_audio)
        proc_kwargs["reference_text"] = args.reference_text

    inputs = processor(**proc_kwargs)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
        generator=gen,
        return_dict_in_generate=True,
    )
    waveforms, lengths = model.decode_audio(output.codes)
    # waveforms: [B, T] padded, lengths: [B]
    wav = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    sr = int(model.config.codec_sample_rate)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), wav, sr)
    print(f"[falcon_bridge] wrote {out_path} sr={sr} samples={len(wav)} duration={len(wav)/sr:.2f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
