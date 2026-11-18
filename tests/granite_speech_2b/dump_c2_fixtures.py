#!/usr/bin/env python3
"""Dump C2 reference fixtures: YaRN inv_freq + a text-only 4096-crossing case.

1. rope_yarn_inv_freq.json: the pinned model's rotary_emb.inv_freq buffer
   (64 float64 values) + attention_scaling + the yarn scalars, read live from
   the checkpoint config/model (never hardcoded).
2. text_only_4096_crossing/: input_ids.json (a >4100-token text-only prompt)
   + last_pos_logits.f32 from model.language_model (no audio path), proving
   the YaRN interpolation regime past position 4096.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

    torch.set_num_threads(8)
    model, _ = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True,
        output_loading_info=True, attn_implementation="eager")
    model.eval()

    txt = model.config.text_config
    rope = model.language_model.model.rotary_emb
    inv_freq = rope.inv_freq.detach().to(torch.float64).cpu().numpy()
    assert inv_freq.shape == (64,), inv_freq.shape
    yarn = {
        "head_dim": 128,
        "theta": float(txt.rope_theta),
        "factor": float(txt.rope_scaling["factor"]),
        "original_max_position_embeddings": int(
            txt.rope_scaling["original_max_position_embeddings"]),
        "max_position_embeddings": int(txt.max_position_embeddings),
        "beta_fast": float(txt.rope_scaling.get("beta_fast", 32)),
        "beta_slow": float(txt.rope_scaling.get("beta_slow", 1)),
        "attention_scaling": float(rope.attention_scaling),
        "inv_freq": [float(v) for v in inv_freq],
    }
    out = args.fixtures_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "rope_yarn_inv_freq.json").write_text(json.dumps(yarn, indent=1))

    # Text-only prompt crossing position 4096: repeat a mid-length sentence.
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sentence = "Concord returned to its place amidst the tents, and the evening settled over the quiet camp. "
    ids = tok(sentence, add_special_tokens=False)["input_ids"]
    reps = 4100 // len(ids) + 2
    input_ids = (ids * reps)[:4200]
    assert len(input_ids) > 4096, len(input_ids)
    (out / "text_only_4096_crossing_ids.json").write_text(json.dumps(input_ids))

    with torch.inference_mode():
        out_tensors = model.language_model(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            use_cache=False)
        last = out_tensors.logits[0, -1].detach().to(torch.float32).cpu().numpy()
    assert last.shape == (100353,), last.shape
    (out / "text_only_4096_crossing_last_logits.f32").write_bytes(
        np.ascontiguousarray(last, dtype="<f4").tobytes())
    top = int(np.argmax(last))
    print(json.dumps({"ids_len": len(input_ids), "top1": top,
                      "top1_logit": float(last[top])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
