#!/usr/bin/env python3
"""Compare archived native prompt/logit traces with local upstream assets.

The current runtime has no custom tensor-dump hooks. This script accepts traces
saved during the original port, rather than capturing a new session. It loads
only local HF safetensors; no download or pickle checkpoint loading is used.
Current component parity is checked separately by check_native_components.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python-package", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="Bruno")
    parser.add_argument("--voices", type=Path, help="Complete prepared voice index, if separate from the upstream model directory")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-lm", action="store_true")
    parser.add_argument("--emotion", action="store_true", help="Include the upstream emotion-control prefix for expression text")
    parser.add_argument("--max-logit-error", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    # These two upstream modules have no runtime dependencies. Load them directly
    # so parity does not import the full TTS package and its optional normalizer.
    def upstream_module(name):
        path = args.python_package / "kittenml" / "kittentts2" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"kitten_parity_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    prompt_module = upstream_module("prompt")
    TokenMap = upstream_module("tokens").TokenMap
    from transformers import AutoTokenizer

    config = json.loads((args.model / "config.json").read_text(encoding="utf-8"))
    voice = json.loads((args.voices or args.model / "cpp" / "default" / "voices.json").read_text(encoding="utf-8"))[args.voice]
    tm = TokenMap.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model / "lm", local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    prefix = prompt_module.build_reference_prefix(tm, encode(voice["transcript"]), voice["reference_tokens"])
    emotion = encode(config["generation"]["emotion_control"]) if args.emotion else None
    expected_prompt = prompt_module.build_generation_prompt(tm, encode(args.text), prefix, emotion)
    native_prompt = json.loads((args.trace / "prompt.json").read_text(encoding="utf-8"))
    if native_prompt != expected_prompt:
        raise AssertionError("native prompt differs from upstream tokenization / prompt assembly")
    report = {"prompt_tokens": len(native_prompt), "prompt_exact": True}
    if not args.skip_lm:
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForCausalLM
        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights
        lm_config = AutoConfig.from_pretrained(args.model / "lm", local_files_only=True)
        with no_init_weights():
            model = AutoModelForCausalLM.from_config(lm_config, dtype=torch.float32, attn_implementation="sdpa")
        state = load_file(str(args.model / "lm" / "model.safetensors"))
        state = {key: value.float() for key, value in state.items() if not key.startswith("spk_proj.")}
        missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
        if set(missing) - {"lm_head.weight"} or unexpected:
            raise AssertionError((missing, unexpected))
        model.tie_weights()
        model.eval()
        with torch.inference_mode():
            embeddings = model.get_input_embeddings()(torch.tensor([native_prompt]))
            embeddings[:, 0, :] = torch.tensor(voice["speaker"], dtype=torch.float32)
            full = model(inputs_embeds=embeddings, use_cache=False).logits[0, -1].float().numpy()
        reference = np.concatenate((full[tm.audio_id_base:tm.audio_id_end],
                                    full[[tm.speech_end_id, tm.stop_id]]))
        native = np.fromfile(args.trace / "prefill_logits.f32", dtype=np.float32)
        if reference.shape != native.shape or not np.isfinite(native).all():
            raise AssertionError("invalid native logit trace")
        error = np.abs(reference - native)
        report.update(max_logit_error=float(error.max()), mean_logit_error=float(error.mean()),
                      top_token_native=int(native.argmax()), top_token_reference=int(reference.argmax()),
                      cosine=float(np.dot(reference, native) / (np.linalg.norm(reference) * np.linalg.norm(native))))
        if error.max() > args.max_logit_error or native.argmax() != reference.argmax():
            raise AssertionError(report)
        del model, state
    codes = json.loads((args.trace / "codes.json").read_text(encoding="utf-8"))
    native_audio, rate = sf.read(args.wav, dtype="float32")
    expected_frames = 2*(len(voice['prompt_token'][0])+len(codes)+3)-len(voice['prompt_feat'][0])
    if rate != 24000 or native_audio.shape != (expected_frames*480,):
        raise AssertionError("waveform rate/shape mismatch")
    if not np.isfinite(native_audio).all() or np.mean(native_audio**2) < 1e-8:
        raise AssertionError("non-finite or silent audio")
    report.update(audio_tokens=len(codes), audio_samples=len(native_audio))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
