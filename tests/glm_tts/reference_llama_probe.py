#!/usr/bin/env python3
"""Emit upstream GLM-TTS prompt and sampled speech token ids."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import types
from pathlib import Path

import torch

from reference_full_pipeline import install_wetext_identity_stub


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("reference_wav")
    parser.add_argument("reference_text")
    parser.add_argument("text")
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    root = args.reference_root.resolve()
    install_wetext_identity_stub()
    if not hasattr(torch, "npu"):
        torch.npu = types.SimpleNamespace(is_available=lambda: False)
    os.chdir(root)
    sys.path.insert(0, str(root))
    namespace = runpy.run_path(
        str(root / "glmtts_inference.py"), run_name="_glmtts_reference"
    )

    yaml_util = namespace["yaml_util"]
    speech_model, feature_extractor = yaml_util.load_speech_tokenizer(
        os.path.join("ckpt", "speech_tokenizer")
    )
    speech_tokenizer = namespace["SpeechTokenizer"](
        speech_model, feature_extractor
    )
    frontend, text_frontend = namespace["load_frontends"](
        speech_tokenizer, sample_rate=24000, use_phoneme=False
    )
    prompt_text = text_frontend.text_normalize(args.reference_text)
    target_text = text_frontend.text_normalize(args.text)
    prompt_text_ids = frontend._extract_text_token(prompt_text + " ")
    target_text_ids = frontend._extract_text_token(target_text)
    prompt_speech_ids = frontend._extract_speech_token([args.reference_wav])

    llama_path = os.path.join("ckpt", "llm")
    llama = namespace["GLMTTS"](
        llama_cfg_path=os.path.join(llama_path, "config.json"),
        mode="PRETRAIN",
    )
    llama.llama = namespace["LlamaForCausalLM"].from_pretrained(
        llama_path, dtype=torch.float32
    ).to(namespace["DEVICE"])
    llama.llama_embedding = llama.llama.model.embed_tokens
    special = namespace["get_special_token_ids"](frontend.tokenize_fn)
    llama.set_runtime_vars(special_token_ids=special)
    if args.greedy:
        import llm.glmtts as glmtts_module

        glmtts_module.common.ras_sampling = (
            lambda weighted_scores, decoded_tokens, sampling, **kwargs:
            weighted_scores.argmax()
        )
    namespace["seed_util"].set_seed(0)
    generated = namespace["local_llm_forward"](
        llama,
        prompt_text_ids,
        target_text_ids,
        prompt_speech_ids,
    )

    begin = special["boa"]
    full_prompt = (
        prompt_text_ids.squeeze(0).tolist()
        + target_text_ids.squeeze(0).tolist()
        + [begin]
        + [special["ats"] + token for token in prompt_speech_ids.squeeze(0).tolist()]
    )
    print(
        json.dumps(
            {
                "normalized_reference_text": prompt_text,
                "normalized_text": target_text,
                "prompt_ids": full_prompt,
                "speech_tokens": generated,
            }
        )
    )


if __name__ == "__main__":
    main()
