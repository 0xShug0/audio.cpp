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


def dump_prompt_references(tokenizer, out: Path) -> None:
    # Independent Python tokenizer/chat-template oracle; no C++ output is read.
    from reference import build_prompt, LANGUAGE_NAMES

    def emit(name, content, **metadata):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        marker = tokenizer.convert_tokens_to_ids("<|audio|>")
        assert ids.count(marker) == 1
        count = 33
        index = ids.index(marker)
        data = dict(name=name, user_content=content, prompt=prompt,
                    one_marker_ids=ids, audio_token_id=marker, audio_tokens=count,
                    expanded_ids=ids[:index] + [marker] * count + ids[index + 1:],
                    audio_run=1, **metadata)
        (out / (name + ".json")).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    for language in (None, "en", "ja"):
        content, _ = build_prompt(language)
        emit("c1_prompt_" + (language or "none"), content, language=language)

    rare = ["Xochitl Quetzal", "Tlalnepantla", "Zbigniew Brzezinski"]
    keywords = [
        ("rare_hint", "en", rare), ("rare_nohint", "", rare),
        ("absent", "en", ["xylophone", "quantum"]),
        ("accented", "en", ["café", "têntes", "日本語"]),
        ("multiword", "en", ["returned to its place", "amidst the tents"]),
    ]
    for name, source, terms in keywords:
        prefix = (f"The speech is in {LANGUAGE_NAMES[source]}. Transcribe it exactly; "
                  "do not translate. ") if source else ""
        content = prefix + "<|audio|>transcribe the speech to text. Keywords: " + ", ".join(terms)
        emit("f_prompt_kw_" + name, content, source=source, target="", terms=terms)

    names = dict(LANGUAGE_NAMES, it="Italian", zh="Mandarin")
    translations = [("en_" + target, "", target, [])
                    for target in ("fr", "de", "es", "pt", "ja", "it", "zh")]
    translations += [(source + "_en", "", "en", []) for source in ("fr", "de", "es", "pt", "ja")]
    translations += [("en_fr_src", "en", "fr", []),
                     ("en_fr_kw", "en", "fr", ["Concord", "tents"]),
                     ("ja_en_src", "ja", "en", [])]
    for name, source, target, terms in translations:
        prefix = f"The speech is in {names[source]}. " if source else ""
        content = (prefix + f"<|audio|>translate the speech to {names[target]} "
                   "with proper punctuation and capitalization.")
        if terms:
            content += " Keywords: " + ", ".join(terms)
        emit("f_prompt_tr_" + name, content, source=source, target=target, terms=terms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures-dir", type=Path, required=True)
    parser.add_argument("--prompts-only", action="store_true",
                        help="Generate tokenizer references without loading model weights")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    args.fixtures_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dump_prompt_references(tokenizer, args.fixtures_dir)
    if args.prompts_only:
        return 0

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
