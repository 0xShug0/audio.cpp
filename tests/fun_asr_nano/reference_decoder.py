#!/usr/bin/env python3
"""Generate pinned Fun-ASR-Nano prompt and Qwen decoder references."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from array import array
from pathlib import Path


TRANSFORMERS_COMMIT = "f9966442ac24fff57060774ce22e1884760f4a3b"
TRANSFORMERS_SOURCE_SHA256 = {
    "src/transformers/models/fun_asr_nano/configuration_fun_asr_nano.py": (
        "174abb3aad5b38f7208c7eae140531be900eafb3a5d7f9f5562c300c93240e2d"
    ),
    "src/transformers/models/fun_asr_nano/modeling_fun_asr_nano.py": (
        "70f8230fd5a75e8a119cea1c8dc919c3f51d8aa7c5d8f1c23c5402b856ebebfb"
    ),
    "src/transformers/models/fun_asr_nano/processing_fun_asr_nano.py": (
        "bc1601a59f6acef6e5be743e511891fad688afa192d10e722fdea2dc42b5f72a"
    ),
    "src/transformers/models/qwen3/modeling_qwen3.py": (
        "966626c5f2c0f729dd6b61b99c000038602805f7d7ad73957507a63f8c2de563"
    ),
}
MODEL_REVISION = "854d88f94205cd17d2afdb24332130d86fbe654a"
MODEL_CONFIG_SHA256 = "c7c4a30316929631ac5fabc5fb3c0dd3278dcc9809670720c5920186285d004a"
MODEL_SAFETENSORS_SHA256 = (
    "335ca3e74917f1156690400e2c344350112950165789cf78ce3d0a367affd821"
)
ITN_PROMPT = "语音转写："
NO_ITN_PROMPT = "语音转写，不进行文本规整："
AUDIO_TOKENS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformers-src", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_revision(transformers_src: Path) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=transformers_src, text=True
    ).strip()
    if actual != TRANSFORMERS_COMMIT:
        raise RuntimeError(
            f"expected Transformers {TRANSFORMERS_COMMIT}, found {actual}"
        )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=transformers_src,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Transformers reference checkout has tracked modifications")
    for relative_path, expected_hash in TRANSFORMERS_SOURCE_SHA256.items():
        actual_hash = sha256(transformers_src / relative_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"unexpected Transformers source hash for {relative_path}: "
                f"{actual_hash}"
            )


def chat_text(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|object_ref_start|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def expanded_prompt(
    tokenizer, prompt: str, audio_token_id: int
) -> tuple[list[int], list[int]]:
    ids = tokenizer.encode(chat_text(prompt), add_special_tokens=False)
    if ids.count(audio_token_id) != 1:
        raise RuntimeError("expected one audio placeholder in Fun-ASR-Nano prompt")
    expanded: list[int] = []
    positions: list[int] = []
    for token_id in ids:
        if token_id == audio_token_id:
            for _ in range(AUDIO_TOKENS):
                positions.append(len(expanded))
                expanded.append(token_id)
        else:
            expanded.append(token_id)
    return expanded, positions


def deterministic_audio(torch):
    values = [
        ((index % 97) - 48) / 97.0 + math.sin(index * 0.013) * 0.1
        for index in range(AUDIO_TOKENS * 1024)
    ]
    return torch.tensor(values, dtype=torch.float32).reshape(AUDIO_TOKENS, 1024)


def append_f32(blob: array, values) -> dict[str, int]:
    flat = values.detach().to(device="cpu", dtype=values.new_zeros(()).float().dtype)
    flat = flat.contiguous().view(-1).tolist()
    offset = len(blob)
    blob.extend(flat)
    return {"offset_f32": offset, "count": len(flat)}


def main() -> None:
    args = parse_args()
    require_revision(args.transformers_src)
    sys.path.insert(0, str(args.transformers_src / "src"))

    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor, FunAsrNanoForConditionalGeneration

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    config_path = args.model_dir / "config.json"
    model_path = args.model_dir / "model.safetensors"
    if sha256(config_path) != MODEL_CONFIG_SHA256:
        raise RuntimeError("unexpected Fun-ASR-Nano config hash")
    if sha256(model_path) != MODEL_SAFETENSORS_SHA256:
        raise RuntimeError("unexpected Fun-ASR-Nano model hash")

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = FunAsrNanoForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    audio_token_id = model.config.audio_token_id
    itn_ids, itn_positions = expanded_prompt(
        processor.tokenizer, ITN_PROMPT, audio_token_id
    )
    no_itn_ids, no_itn_positions = expanded_prompt(
        processor.tokenizer, NO_ITN_PROMPT, audio_token_id
    )
    if len(itn_positions) != AUDIO_TOKENS or len(no_itn_positions) != AUDIO_TOKENS:
        raise RuntimeError("expanded audio token count mismatch")

    input_ids = torch.tensor([itn_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)
    audio = deterministic_audio(torch).to(args.device)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds[0, itn_positions] = audio

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    logits = [outputs.logits[0, -1].float()]
    greedy_tokens = [int(torch.argmax(logits[0]).item())]
    past_key_values = outputs.past_key_values

    for _ in range(2):
        step_id = torch.tensor(
            [[greedy_tokens[-1]]], dtype=torch.long, device=args.device
        )
        attention_mask = torch.ones(
            (1, attention_mask.shape[1] + 1), dtype=torch.long, device=args.device
        )
        outputs = model(
            input_ids=step_id,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits.append(outputs.logits[0, -1].float())
        greedy_tokens.append(int(torch.argmax(logits[-1]).item()))

    with safe_open(model_path, framework="pt", device="cpu") as handle:
        text_names = sorted(
            name for name in handle.keys() if name.startswith("model.language_model.")
        )
    catalog_hash = hashlib.sha256("\n".join(text_names).encode()).hexdigest()

    blob = array("f")
    audio_descriptor = append_f32(blob, audio)
    audio_descriptor["shape"] = [AUDIO_TOKENS, 1024]
    logit_descriptors = []
    for values in logits:
        descriptor = append_f32(blob, values)
        descriptor["shape"] = [151936]
        logit_descriptors.append(descriptor)
    if sys.byteorder != "little":
        blob.byteswap()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.output.with_suffix(".bin")
    with data_path.open("wb") as handle:
        blob.tofile(handle)

    metadata = {
        "schema_version": 1,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "transformers_source_sha256": TRANSFORMERS_SOURCE_SHA256,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_safetensors_sha256": MODEL_SAFETENSORS_SHA256,
        "tokenizer_json_sha256": sha256(args.model_dir / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(args.model_dir / "tokenizer_config.json"),
        "text_weight_catalog_count": len(text_names),
        "text_weight_catalog_sha256": catalog_hash,
        "torch_version": torch.__version__,
        "device": str(args.device),
        "data_file": data_path.name,
        "data_format": "little-endian-float32",
        "audio_token_id": audio_token_id,
        "im_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_token_id": processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "eos_token_id": model.config.text_config.eos_token_id,
        "itn": {
            "prompt": ITN_PROMPT,
            "input_ids": itn_ids,
            "audio_token_positions": itn_positions,
        },
        "no_itn": {
            "prompt": NO_ITN_PROMPT,
            "input_ids": no_itn_ids,
            "audio_token_positions": no_itn_positions,
        },
        "audio_embeddings": audio_descriptor,
        "greedy_token_ids": greedy_tokens,
        "generated_text": processor.tokenizer.decode(
            greedy_tokens, skip_special_tokens=True
        ),
        "logits": logit_descriptors,
        "data_sha256": sha256(data_path),
    }
    args.output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
