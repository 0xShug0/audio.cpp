#!/usr/bin/env python3
"""Run MOSS-TTS-Local test cases through the Python reference (transformers).

Produces, for every case, a result directory that mirrors the ``audiocpp_cli``
path-test layout so ``compare_audiocpp_cli_path_results.py`` and
``moss_tts_local_report.py`` can diff C++ vs Python symmetrically:

    <out_root>/<case_id>/command.json      (marks the dir as a case)
    <out_root>/<case_id>/outputs/<request_id>.wav
    <out_root>/<case_id>/stdout.log        (request_id= + [TIMING] lines)
    <out_root>/<case_id>/memory.json       (per-request RSS / CUDA VRAM)

Each case loads the model + codec once and runs its requests in sequence, so a
multi-request case is a genuine long-lived session (mirroring the C++ session).

Run inside the ``moss_tts_local`` conda env (transformers>=5.0, torch, torchaudio).

Examples
--------
    # fp32 parity reference (matches the C++ f32 path; no KV cache for exactness)
    python moss_tts_local_reference.py --only moss_tts_local_text_only_greedy \
        --dtype fp32 --device cuda --out-root build/logs/moss_ref_fp32

    # bf16 CUDA reference (for realistic performance comparison)
    python moss_tts_local_reference.py --family moss_tts_local \
        --dtype bf16 --device cuda --out-root build/logs/moss_ref_bf16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = REPO_ROOT / "tools" / "audiocpp_cli" / "audiocpp_cli_path_cases.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "MOSS-TTS-Local-Transformer-v1.5"
FAMILY = "moss_tts_local"


def load_cases(path: Path) -> list[dict[str, Any]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return catalog.get("cases", [])


def select(cases: list[dict[str, Any]], only: set[str]) -> list[dict[str, Any]]:
    out = []
    for case in cases:
        if case.get("family") != FAMILY:
            continue
        if only and case["id"] not in only:
            continue
        out.append(case)
    if only:
        missing = sorted(only - {c["id"] for c in out})
        if missing:
            raise SystemExit(f"unknown/non-moss case id(s): {', '.join(missing)}")
    return out


def resolve_resource(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def opt_float(req: dict[str, Any], key: str) -> Optional[float]:
    return float(req[key]) if key in req and req[key] is not None else None


def opt_int(req: dict[str, Any], key: str) -> Optional[int]:
    return int(req[key]) if key in req and req[key] is not None else None


def opt_bool(req: dict[str, Any], key: str, default: bool) -> bool:
    if key not in req or req[key] is None:
        return default
    value = req[key]
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class MemoryProbe:
    """Per-request process RSS and CUDA VRAM snapshots."""

    def __init__(self, torch_mod: Any, device: str) -> None:
        self.torch = torch_mod
        self.device = device
        try:
            import psutil  # noqa: WPS433 - optional dependency

            self.proc = psutil.Process()
        except Exception:  # noqa: BLE001 - psutil is optional
            self.proc = None

    def reset_peak(self) -> None:
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.proc is not None:
            out["rss_mb"] = self.proc.memory_info().rss / (1024.0 * 1024.0)
        if self.device == "cuda" and self.torch.cuda.is_available():
            out["cuda_alloc_mb"] = self.torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            out["cuda_peak_mb"] = self.torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            out["cuda_reserved_mb"] = self.torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        return out


def run_case(
    case: dict[str, Any],
    processor: Any,
    model: Any,
    torch_mod: Any,
    torchaudio_mod: Any,
    device: str,
    use_kv_cache: bool,
    probe: MemoryProbe,
    out_root: Path,
) -> None:
    case_dir = out_root / case["id"]
    outputs = case_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (case_dir / "command.json").write_text(
        json.dumps({"reference": "moss_tts_local_reference.py", "case": case["id"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    sampling_rate = int(processor.model_config.sampling_rate)
    stdout_lines: list[str] = [f"family={FAMILY}", f"task={case.get('task', 'tts')}", "mode=offline"]
    memory: dict[str, Any] = {"case": case["id"], "dtype": str(model.dtype), "requests": []}

    session_start = time.perf_counter()
    for index, req in enumerate(case["requests"]):
        request_id = req["id"]
        language = req.get("language")
        if language in (None, "", "Auto"):
            language = None
        reference = None
        if "voice_ref" in req and req["voice_ref"]:
            reference = [resolve_resource(req["voice_ref"])]

        message_kwargs: dict[str, Any] = {"text": req["text"], "language": language}
        if reference is not None:
            message_kwargs["reference"] = reference
        if opt_int(req, "tokens") is not None:
            message_kwargs["tokens"] = opt_int(req, "tokens")
        conversation = [processor.build_user_message(**message_kwargs)]

        seed = opt_int(req, "seed")
        if seed is not None:
            torch_mod.manual_seed(seed)
            if device == "cuda":
                torch_mod.cuda.manual_seed_all(seed)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": opt_int(req, "max_tokens") or 4096,
            "do_sample": opt_bool(req, "do_sample", True),
            "use_kv_cache": use_kv_cache,
        }
        if opt_float(req, "temperature") is not None:
            gen_kwargs["audio_temperature"] = opt_float(req, "temperature")
        if opt_int(req, "top_k") is not None:
            gen_kwargs["audio_top_k"] = opt_int(req, "top_k")
        if opt_float(req, "top_p") is not None:
            gen_kwargs["audio_top_p"] = opt_float(req, "top_p")
        if opt_float(req, "repetition_penalty") is not None:
            gen_kwargs["audio_repetition_penalty"] = opt_float(req, "repetition_penalty")

        probe.reset_peak()
        run_start = time.perf_counter()
        with torch_mod.no_grad():
            batch = processor([conversation], mode="generation")
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            gen_out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            audio = None
            for message in processor.decode(gen_out):
                if message is None:
                    continue
                audio = message.audio_codes_list[0]
                break
        if device == "cuda":
            torch_mod.cuda.synchronize()
        wall_ms = (time.perf_counter() - run_start) * 1000.0

        if audio is None:
            raise RuntimeError(f"{case['id']}/{request_id}: reference produced no audio")
        wav_path = outputs / f"{request_id}.wav"
        torchaudio_mod.save(str(wav_path), audio.to(torch_mod.float32).cpu(), sampling_rate)

        snap = probe.snapshot()
        snap.update({"request_id": request_id, "wall_ms": wall_ms})
        audio_seconds = float(audio.shape[-1]) / float(sampling_rate)
        snap["audio_seconds"] = audio_seconds
        snap["rtf"] = (wall_ms / 1000.0) / audio_seconds if audio_seconds > 0 else 0.0
        memory["requests"].append(snap)

        stdout_lines.append(f"request_index={index}")
        stdout_lines.append(f"request_id={request_id}")
        stdout_lines.append(f"[TIMING] request.{request_id}.wall_ms {wall_ms}")
        stdout_lines.append(f"audio_out={wav_path}")
        print(
            f"[REF] {case['id']}/{request_id}: {wall_ms:.1f} ms, "
            f"{audio_seconds:.2f} s audio, rtf={snap['rtf']:.3f}",
            flush=True,
        )

    session_wall_ms = (time.perf_counter() - session_start) * 1000.0
    stdout_lines.append(f"[TIMING] session.wall_ms {session_wall_ms}")
    (case_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    memory["session_wall_ms"] = session_wall_ms
    (case_dir / "memory.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--only", action="append", default=[], help="case id(s), comma-separated allowed")
    parser.add_argument("--family", action="store_true", help="run all moss_tts_local cases (default if --only omitted)")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--use-kv-cache",
        choices=["auto", "true", "false"],
        default="auto",
        help="auto: off for fp32 (exact parity with C++), on otherwise",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:  # noqa: BLE001 - guide the user to the right env
        print(f"[FAIL] import error ({exc}). Activate the moss_tts_local conda env.", file=sys.stderr)
        return 1

    only = {item.strip() for raw in args.only for item in raw.split(",") if item.strip()}
    cases = select(load_cases(args.cases), only)
    if not cases:
        print("[FAIL] no matching moss_tts_local cases", file=sys.stderr)
        return 1

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if args.use_kv_cache == "auto":
        use_kv_cache = args.dtype != "fp32"
    else:
        use_kv_cache = args.use_kv_cache == "true"
    attn = "sdpa" if device == "cuda" else "eager"
    if device == "cuda" and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    print(
        f"[INFO] model={args.model} dtype={args.dtype} device={device} "
        f"attn={attn} use_kv_cache={use_kv_cache}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    if args.dtype == "fp32":
        # The codec loads in bf16 by default; cast its weights to fp32 (and set
        # its compute dtype to fp32) so decode stays fp32 for parity with the
        # C++ f32 path. Casting params avoids the fp32-input/bf16-weight mismatch.
        processor.audio_tokenizer = processor.audio_tokenizer.to(device=device, dtype=torch.float32)
        if hasattr(processor.audio_tokenizer, "set_compute_dtype"):
            processor.audio_tokenizer.set_compute_dtype("fp32")
    else:
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    model = AutoModel.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        attn_implementation=attn,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    probe = MemoryProbe(torch, device)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for case in cases:
        print(f"[RUN] {case['id']}: {case.get('coverage', '')}", flush=True)
        run_case(case, processor, model, torch, torchaudio, device, use_kv_cache, probe, args.out_root)
    print(f"[DONE] {len(cases)} case(s), out_root={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
