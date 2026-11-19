#!/usr/bin/env python3
"""Pinned FP32 reference trace generator for ``granite_speech5``.

Runs IBM ``granite-speech-5.0-2b`` at a pinned checkpoint revision on CPU in
FP32 (eager attention) and dumps every intermediate tensor that the C++ parity
work in milestones B/C must reproduce, plus a ``trace.json`` manifest.

Pinned reference (do not change without a new milestone)::

    REV = ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7
    torch==2.8.0 torchaudio==2.8.0 transformers==4.57.0, CPU, FP32,
    attn_implementation="eager", torch.set_num_threads(8),
    generate(max_new_tokens=MAX, do_sample=False, num_beams=1).

Pipeline stages are dumped in order (see ``PIPELINE``); all ``*.f32`` files
are little-endian float32, contiguous, time-major.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

REVISION = "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7"
MODEL_SHA256 = "4151269881d2dd825d45ca75d6552b2a20e6e39d0934ed84c4db826af8bf6d88"
DEFAULT_USER_PROMPT = (
    "<|audio|>transcribe the speech with proper punctuation and capitalization."
)
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}
NUM_THREADS = 8
N_FORCED_STEPS = 8

# (stage, dump file, max_abs threshold or None, rel_rms threshold or None).
# Initial limits per the plan: frontend max_abs 2e-4; encoder/projector/prefill
# normalized RMS 1e-3; logits max_abs 2e-2 plus normalized RMS 1e-3.
PIPELINE: list[tuple[str, str, float | None, float | None]] = [
    ("mel", "mel.f32", 2e-4, 1e-3),
    ("stacked", "stacked.f32", 2e-4, 1e-3),
    ("input_proj", "input_proj.f32", None, 1e-3),
    ("encoder_block1", "encoder_block1.f32", None, 1e-3),
    ("encoder_layer8_pre_feedback", "encoder_layer8_pre_feedback.f32", None, 1e-3),
    ("encoder_layer8_post_feedback", "encoder_layer8_post_feedback.f32", None, 1e-3),
    ("encoder_last", "encoder_last.f32", None, 1e-3),
    ("projected", "projected.f32", None, 1e-3),
    ("merged_prefill", "merged_prefill.f32", None, 1e-3),
    ("scaled_prefill", "scaled_prefill.f32", None, 1e-3),
    ("last_pos_logits", "last_pos_logits.f32", 2e-2, 1e-3),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_f32(path: Path, tensor) -> list[int]:
    """Write a CPU float32 tensor little-endian; return its shape."""
    import numpy as np
    import torch

    arr = np.ascontiguousarray(
        tensor.detach().to(dtype=torch.float32).cpu().numpy(), dtype="<f4")
    path.write_bytes(arr.tobytes())
    return list(arr.shape)


def build_prompt(language: str | None) -> tuple[str, str | None]:
    if language is None:
        return DEFAULT_USER_PROMPT, None
    name = LANGUAGE_NAMES[language]
    hinted = (
        f"The speech is in {name}. Transcribe it exactly; do not translate. "
        + DEFAULT_USER_PROMPT
    )
    return hinted, language


def dump_model_config(model) -> dict:
    """Derive the trace's model_config from the loaded checkpoint config.

    Every value is read from ``model.config`` / ``model.generation_config``.
    Anything missing is an explicit RuntimeError (loud FAIL in main), never a
    silent constant, so the trace cannot lie about a drifted checkpoint.
    """
    def need(obj, field: str, where: str):
        try:
            value = getattr(obj, field)
        except AttributeError:
            raise RuntimeError(f"FAIL: model config missing {where}.{field}; "
                               f"refusing to emit trace with guessed constants")
        if value is None:
            raise RuntimeError(f"FAIL: model config {where}.{field} is None; "
                               f"refusing to emit trace with guessed constants")
        return value

    cfg = need(model, "config", "model")
    enc = need(cfg, "encoder_config", "config")
    proj = need(cfg, "projector_config", "config")
    txt = need(cfg, "text_config", "config")
    gen = need(model, "generation_config", "model")
    rope = need(txt, "rope_scaling", "text_config")
    if isinstance(rope, dict):
        rope_keys = ("factor", "original_max_position_embeddings", "rope_type")
        missing = [k for k in rope_keys if k not in rope or rope[k] is None]
        if missing:
            raise RuntimeError(f"FAIL: model config text_config.rope_scaling missing {missing}; "
                               f"refusing to emit trace with guessed constants")
        rope_map = {k: rope[k] for k in rope_keys}
    else:
        rope_map = {k: need(rope, k, "text_config.rope_scaling") for k in
                    ("factor", "original_max_position_embeddings", "rope_type")}
    return {
        "audio_token_id": need(cfg, "audio_token_id", "config"),
        "encoder": {
            "input_dim": need(enc, "input_dim", "encoder_config"),
            "hidden_dim": need(enc, "hidden_dim", "encoder_config"),
            "num_layers": need(enc, "num_layers", "encoder_config"),
            "num_heads": need(enc, "num_heads", "encoder_config"),
            "dim_head": need(enc, "dim_head", "encoder_config"),
            "context_size": need(enc, "context_size", "encoder_config"),
            "output_dim": need(enc, "output_dim", "encoder_config"),
        },
        "projector": {
            "window_size": need(cfg, "window_size", "config"),
            "downsample_rate": need(cfg, "downsample_rate", "config"),
            "hidden_size": need(proj, "hidden_size", "projector_config"),
            "qformer_layers": need(proj, "num_hidden_layers", "projector_config"),
            "qformer_heads": need(proj, "num_attention_heads", "projector_config"),
        },
        "text": {
            "num_layers": need(txt, "num_hidden_layers", "text_config"),
            "hidden_size": need(txt, "hidden_size", "text_config"),
            "num_attention_heads": need(txt, "num_attention_heads", "text_config"),
            "num_key_value_heads": need(txt, "num_key_value_heads", "text_config"),
            "intermediate_size": need(txt, "intermediate_size", "text_config"),
            "vocab_size": need(txt, "vocab_size", "text_config"),
            "bos_token_id": need(gen, "bos_token_id", "generation_config"),
            "eos_token_id": need(gen, "eos_token_id", "generation_config"),
            "pad_token_id": need(gen, "pad_token_id", "generation_config"),
            "embedding_multiplier": need(txt, "embedding_multiplier", "text_config"),
            "attention_multiplier": need(txt, "attention_multiplier", "text_config"),
            "logits_scaling": need(txt, "logits_scaling", "text_config"),
            "residual_multiplier": need(txt, "residual_multiplier", "text_config"),
            "rope_theta": need(txt, "rope_theta", "text_config"),
            "rope_scaling": rope_map,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", choices=sorted(LANGUAGE_NAMES), default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--force-ids", type=Path, default=None,
                        help="JSON list of token IDs used as the forced continuation "
                             "(default: the greedy output of this run)")
    parser.add_argument("--fixture-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--license", default=None)
    parser.add_argument("--human-ref-txt", type=Path, default=None)
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="proceed even if model.safetensors does not match the pinned "
                             "SHA-256 (trace is then recorded as unpinned)")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    torch.set_num_threads(NUM_THREADS)
    out_dir: Path = args.out_dir

    samples, rate = sf.read(args.audio, dtype="float32")
    assert rate == 16000 and samples.ndim == 1, f"need mono 16 kHz, got sr={rate} ndim={samples.ndim}"
    num_samples = int(samples.shape[0])
    audio_sha = sha256_file(args.audio)

    model_sha = sha256_file(args.model / "model.safetensors")
    pinned = (model_sha == MODEL_SHA256)
    if not pinned and not args.allow_unpinned:
        print(f"FAIL: model.safetensors sha256 {model_sha} != pinned {MODEL_SHA256}; "
              f"refusing to emit a pinned trace (pass --allow-unpinned to override)",
              file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model, load_info = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch.float32, local_files_only=True,
        output_loading_info=True, attn_implementation="eager",
    )
    model.eval()

    user_content, language_hint = build_prompt(args.language)
    prompt = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True,
    )
    waveform = torch.from_numpy(samples)[None]
    inputs = processor(prompt, waveform, device="cpu", return_tensors="pt")
    input_ids: torch.Tensor = inputs["input_ids"]
    prompt_len = int(input_ids.shape[-1])

    # Length contract: M raw mel frames, T encoder steps, W windows, A audio tokens.
    mel_frames = num_samples // 160 + 1
    enc_steps = mel_frames // 2  # odd trailing frame is dropped before stacking
    windows = (enc_steps + 15 - 1) // 15
    audio_tokens = 3 * windows
    audio_token_id = int(model.config.audio_token_id)
    assert (input_ids[0] == audio_token_id).sum().item() == audio_tokens, (
        "audio marker count mismatch: "
        f"{(input_ids[0] == audio_token_id).sum().item()} != {audio_tokens}"
    )

    # Raw 80-dim mel capture: same submodule + same pointwise ops as
    # GraniteSpeechFeatureExtractor._extract_mel_spectrograms (pinned
    # transformers), stashed before the odd-frame drop + pairwise stacking.
    # Self-checked below against the processor's own stacked output.
    fe = processor.audio_processor
    with torch.no_grad():
        mel = fe.mel_filters(waveform.float())
        logmel = mel.transpose(-1, -2).clip_(min=1e-10).log10_()
        mx = logmel.amax(dim=(-2, -1), keepdim=True)
        mel_raw = torch.maximum(logmel, mx - 8.0).div_(4).add_(1)
    assert int(mel_raw.shape[1]) == mel_frames, (mel_raw.shape, mel_frames)
    stacked_ref = mel_raw[:, : mel_frames - (mel_frames % 2)].reshape(1, -1, 160)
    assert torch.equal(stacked_ref, inputs["input_features"].to(stacked_ref.dtype)), \
        "mel restack does not reproduce processor input_features"

    # Stage hooks (active only for the manual prefill forward below).
    caps: dict[str, torch.Tensor] = {}

    def save(name: str):
        def hook(_mod, _inp, out):
            caps[name] = out.detach().to(torch.float32).cpu()
        return hook

    def save_clone(name: str):
        # Clone at hook time: the encoder applies the block-8 CTC feedback
        # with an in-place += after layers[7] returns, which would otherwise
        # mutate this hook's aliased (detach/cpu no-op on CPU FP32) storage
        # before the dump and void the pre-feedback stage.
        def hook(_mod, _inp, out):
            caps[name] = out.detach().to(torch.float32).cpu().clone()
        return hook

    handles = [
        model.encoder.input_linear.register_forward_hook(save("input_proj")),
        model.encoder.layers[0].register_forward_hook(save("encoder_block1")),
        model.encoder.layers[7].register_forward_hook(save_clone("encoder_layer8_pre_feedback")),
        model.encoder.layers[8].register_forward_pre_hook(
            lambda _mod, inp: caps.__setitem__(
                "encoder_layer8_post_feedback", inp[0].detach().to(torch.float32).cpu())),
        model.encoder.layers[15].register_forward_hook(save("encoder_last")),
        model.projector.register_forward_hook(save("projected")),
        model.language_model.model.layers[0].register_forward_pre_hook(
            lambda _mod, inp: caps.__setitem__(
                "scaled_prefill", inp[0].detach().to(torch.float32).cpu())),
        model.language_model.lm_head.register_forward_hook(
            lambda _mod, _inp, out: caps.__setitem__(
                "last_pos_logits", out[0, -1].detach().to(torch.float32).cpu())),
    ]
    orig_merge = model.get_merged_audio_embeddings

    def merge_wrap(**kwargs):
        out = orig_merge(**kwargs)
        caps["merged_prefill"] = out.detach().to(torch.float32).cpu()
        return out

    model.get_merged_audio_embeddings = merge_wrap  # type: ignore[method-assign]

    with torch.inference_mode():
        attention_mask = torch.ones(1, prompt_len, dtype=torch.long)
        prefill = model(
            input_ids=input_ids,
            input_features=inputs["input_features"],
            input_features_mask=inputs.get("input_features_mask"),
            attention_mask=attention_mask,
            use_cache=True,
        )
        past = prefill.past_key_values
        prefill_last_logits = prefill.logits[0, -1].detach().to(torch.float32).cpu()
    for h in handles:
        h.remove()
    model.get_merged_audio_embeddings = orig_merge  # type: ignore[method-assign]

    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_tokens, do_sample=False, num_beams=1)
    gen_ids = output[0, prompt_len:].tolist()
    eos_id = int(model.generation_config.eos_token_id)
    observed_stop = "eos" if gen_ids and gen_ids[-1] == eos_id else "budget"
    text = processor.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Forced cached decode: teacher-force the known continuation through the
    # single-token cached path and dump each step's full logit vector.
    # forced_step{k} predicts continuation[k]: step 0 is the prefill
    # last-position distribution; step k>=1 feeds continuation[k-1] at
    # position prompt_len+k-1 (exactly what generate() does per step).
    if args.force_ids is not None:
        continuation = json.loads(args.force_ids.read_text())
    else:
        continuation = gen_ids
    n_steps = min(N_FORCED_STEPS, len(continuation))
    forced_logits: list[torch.Tensor] = [prefill_last_logits][:n_steps]
    with torch.inference_mode():
        for k in range(1, n_steps):
            tok = torch.tensor([[continuation[k - 1]]], dtype=torch.long)
            pos = torch.tensor([prompt_len + k - 1], dtype=torch.long)
            attention_mask = torch.ones(1, prompt_len + k, dtype=torch.long)
            step = model(
                input_ids=tok, past_key_values=past,
                attention_mask=attention_mask, cache_position=pos, use_cache=True)
            forced_logits.append(step.logits[0, -1].detach().to(torch.float32).cpu())
            past = step.past_key_values
    if args.force_ids is None:
        # Teacher forcing the greedy path must reproduce it exactly.
        replay = [int(t.argmax().item()) for t in forced_logits[:n_steps]]
        assert replay == continuation[:n_steps], \
            f"forced replay diverges: {replay} != {continuation[:n_steps]}"

    # Dumps.
    shapes: dict[str, list[int]] = {}
    shapes["mel"] = dump_f32(out_dir / "mel.f32", mel_raw[0])
    shapes["stacked"] = dump_f32(out_dir / "stacked.f32", inputs["input_features"][0])
    for stage, fname, _max_abs, _rel_rms in PIPELINE[2:]:
        shapes[stage] = dump_f32(out_dir / fname, caps[stage][0]
                                 if caps[stage].ndim == 3 else caps[stage])
    for k, logits_k in enumerate(forced_logits):
        shapes[f"forced_step{k}"] = dump_f32(out_dir / f"forced_step{k}.f32", logits_k)

    import transformers as _tf
    import numpy as _np

    try:
        derived_config = dump_model_config(model)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    trace = {
        "family": "granite_speech5",
        "revision": REVISION,
        "model_sha256": model_sha,
        "audio_sha256": audio_sha,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _tf.__version__,
            "numpy": _np.__version__,
        },
        "backend": "cpu",
        "attn_implementation": "eager",
        "dtype": "float32",
        "threads": NUM_THREADS,
        "through": "generate",
        "sample_rate": rate,
        "audio": {"path": str(args.audio), "num_samples": num_samples,
                  "span": [0, num_samples]},
        "lengths": {"mel_frames": mel_frames, "encoder_steps": enc_steps,
                    "windows": windows, "audio_tokens": audio_tokens,
                    "prompt_len": prompt_len, "generated_len": len(gen_ids)},
        "prompt": prompt,
        "language_hint": language_hint,
        "input_ids": input_ids[0].tolist(),
        "max_tokens": args.max_tokens,
        "generated_ids": gen_ids,
        "observed_stop": observed_stop,
        "eos_token_id": eos_id,
        "text": text,
        "pinned": pinned,
        "allow_unpinned": args.allow_unpinned,
        "model_config": derived_config,
        "load_info": {k: list(v) if isinstance(v, (list, tuple)) else v
                      for k, v in (load_info.items() if isinstance(load_info, dict) else [])},
        "stages": [
            {"name": name, "file": fname, "shape": shapes[name],
             "dtype": "float32", "max_abs": max_abs, "rel_rms": rel_rms}
            for name, fname, max_abs, rel_rms in PIPELINE
        ] + [
            {"name": f"forced_step{k}", "file": f"forced_step{k}.f32",
             "shape": shapes[f"forced_step{k}"], "dtype": "float32",
             "max_abs": 2e-2, "rel_rms": 1e-3}
            for k in range(n_steps)
        ],
        "cutoffs": {
            "frontend": "stacked",
            "encoder": "encoder_last",
            "projector": "projected",
            "prefill": "last_pos_logits",
            "generate": f"forced_step{n_steps - 1}" if n_steps else "last_pos_logits",
        },
    }
    (out_dir / "trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    human_ref = args.human_ref_txt.read_text(encoding="utf-8").strip() \
        if args.human_ref_txt is not None else None
    manifest_row = {
        "id": args.fixture_id or args.audio.stem,
        "audio_path": str(args.audio),
        "audio_sha256": audio_sha,
        "dataset": args.dataset,
        "license": args.license,
        "language": language_hint,
        "options": {"max_tokens": args.max_tokens, "language_hint": language_hint},
        "human_ref": human_ref,
    }
    (out_dir / "reference_manifest.jsonl").write_text(
        json.dumps(manifest_row, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "text": text,
                      "generated_len": len(gen_ids), "observed_stop": observed_stop,
                      "prompt_len": prompt_len, "audio_tokens": audio_tokens,
                      "load_info": trace["load_info"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
