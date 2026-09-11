#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reference" / "Confucius4-TTS"
DEFAULT_CONFIG = REFERENCE_ROOT / "config" / "inference_config.yaml"
DEFAULT_CASE_CATALOG = REPO_ROOT / "tests" / "confucius4_tts" / "confucius4_tts_warm_bench_cases.json"
DEFAULT_W2V_BERT = REPO_ROOT / "models" / "facebook-w2v-bert-2.0"
DEFAULT_BIGVGAN = REPO_ROOT / "models" / "bigvgan_v2_22khz_80band_256x"
DEFAULT_CAMPPLUS = REPO_ROOT / "models" / "funasr-campplus" / "campplus_cn_common.bin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Confucius4-TTS warmbench.")
    parser.add_argument("--family", default="confucius4_tts")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "Confucius4-TTS")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--w2v-bert", type=Path, default=DEFAULT_W2V_BERT)
    parser.add_argument("--bigvgan", type=Path, default=DEFAULT_BIGVGAN)
    parser.add_argument("--campplus", type=Path, default=DEFAULT_CAMPPLUS)
    parser.add_argument("--backend", choices=("cuda",), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--request-json", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--warmup-request-json", default="")
    parser.add_argument("--case-catalog", type=Path, default=DEFAULT_CASE_CATALOG)
    parser.add_argument("--case-name", default="multi_request_logic_paths")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=10.0)
    parser.add_argument("--n-timesteps", type=int, default=25)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--text-chunk-size", type=int, default=80)
    parser.add_argument("--cross-fade-duration", type=float, default=0.3)
    parser.add_argument("--edge-fade-duration", type=float, default=0.1)
    parser.add_argument("--edge-pad-duration", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--timing-file",
        type=Path,
        default=REPO_ROOT
        / "build"
        / "debug"
        / "logs"
        / "confucius4_tts"
        / "python_warm_bench"
        / "python.timing.log",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-out", type=Path, default=Path("confucius4_tts_python_audio.wav"))
    parser.add_argument("--summary-file", type=Path, default=None)
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


@contextlib.contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def add_reference_path(reference_root: Path) -> Path:
    root = resolve_repo_path(reference_root).resolve()
    package = root / "confuciustts" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"missing Confucius4-TTS reference package: {package}")
    use_transformers452()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_reference_symbols(reference_root: Path) -> tuple[Any, Path]:
    root = add_reference_path(reference_root)
    from confuciustts.cli.inference import ConfuciusTTS

    module_path = Path(sys.modules[ConfuciusTTS.__module__].__file__).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"Confucius4-TTS imported from {module_path}, expected under {root}") from exc
    return ConfuciusTTS, module_path


def use_transformers452() -> None:
    if "transformers" in sys.modules:
        module = sys.modules["transformers"]
        if getattr(module, "__name__", "") != "transformers452":
            raise RuntimeError("Confucius4-TTS warmbench must select transformers452 before transformers is imported")
        return
    import transformers452

    sys.modules["transformers"] = transformers452


def load_bigvgan_local(model_root: Path, device: torch.device) -> Any:
    from external.bigvgan.bigvgan import BigVGAN, load_hparams_from_json

    config_path = model_root / "config.json"
    weights_path = model_root / "bigvgan_generator.pt"
    if not config_path.is_file():
        raise RuntimeError(f"missing local BigVGAN config: {config_path}")
    if not weights_path.is_file():
        raise RuntimeError(f"missing local BigVGAN weights: {weights_path}")
    model = BigVGAN(load_hparams_from_json(config_path), use_cuda_kernel=False)
    checkpoint = torch.load(weights_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise RuntimeError(f"local BigVGAN checkpoint missing generator state: {weights_path}")
    removed_weight_norm = False
    try:
        model.load_state_dict(checkpoint["generator"])
    except RuntimeError:
        model.remove_weight_norm()
        removed_weight_norm = True
        model.load_state_dict(checkpoint["generator"])
    if not removed_weight_norm:
        model.remove_weight_norm()
    model.eval().to(device)
    return model


def make_local_weight_confucius_tts(ConfuciusTTS: Any, args: argparse.Namespace, reference_root: Path) -> Any:
    import safetensors.torch
    from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

    from external.campplus import CAMPPlus
    from confuciustts.flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
    from confuciustts.frontend.text_normalizer import TextNormalizer
    from confuciustts.llm.llm import Text2Semantic, Text2SemanticConfig

    class LocalWeightConfuciusTTS(ConfuciusTTS):
        def __init__(self, config_path: str, device_name: str) -> None:
            self.device = torch.device(device_name)
            with open(config_path, "r", encoding="utf-8") as handle:
                self.cfg = yaml.safe_load(handle)
            paths = self.cfg["paths"]
            model_root = resolve_repo_path(args.model).resolve()

            self.sample_rate = self.cfg["audio"]["target_sample_rate"]
            self.n_mels = self.cfg["audio"]["n_mels"]
            self.n_fft = self.cfg["audio"]["n_fft"]
            self.hop_length = self.cfg["audio"]["hop_length"]
            self.win_length = self.cfg["audio"]["win_length"]
            self.fmin = self.cfg["audio"]["fmin"]
            self.fmax = self.cfg["audio"]["fmax"]

            self.normalizer = TextNormalizer()

            self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(paths["w2v_bert_path"])
            self.w2v_model = Wav2Vec2BertModel.from_pretrained(paths["w2v_bert_path"]).eval().to(self.device)
            stats = torch.load(paths["w2v_stat"], map_location="cpu")
            self.semantic_mean = stats["mean"].to(self.device)
            self.semantic_std = torch.sqrt(stats["var"]).to(self.device)

            spk_cfg = paths["style_encoder"]
            self.style_encoder = CAMPPlus(**spk_cfg.get("init_args", {}))
            style_path = resolve_repo_path(Path("models/funasr-campplus") / spk_cfg["checkpoint"])
            if not style_path.is_file():
                raise RuntimeError(f"missing local CAMPPlus checkpoint: {style_path}")
            spk_state = torch.load(style_path, map_location="cpu")
            if isinstance(spk_state, dict) and "state_dict" in spk_state:
                spk_state = spk_state["state_dict"]
            self.style_encoder.load_state_dict(spk_state, strict=False)
            self.style_encoder.eval().to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"])
            t2s_config = Text2SemanticConfig(**self.cfg["t2s_model"])
            self.t2s_model = Text2Semantic(t2s_config)
            self.t2s_model.config.vocab_size = t2s_config.semantic_vocab_size
            t2s_path = model_root / paths["t2s_checkpoint"]
            if not t2s_path.is_file():
                raise RuntimeError(f"missing local T2S checkpoint: {t2s_path}")
            self.t2s_model.load_state_dict(safetensors.torch.load_file(str(t2s_path), device="cpu"))
            self.t2s_model.eval().to(self.device)

            s2a_config = MaskedDiffWithXvecConfig(**self.cfg["s2a_model"])
            self.s2a_model = MaskedDiffWithXvec(s2a_config)
            s2a_path = model_root / paths["s2a_checkpoint"]
            if not s2a_path.is_file():
                raise RuntimeError(f"missing local S2A checkpoint: {s2a_path}")
            self.s2a_model.load_state_dict(torch.load(s2a_path, map_location="cpu", weights_only=False))
            self.s2a_model.eval().to(self.device)

            self.bigvgan = load_bigvgan_local(resolve_repo_path(args.bigvgan).resolve(), self.device)

    with pushd(reference_root):
        return LocalWeightConfuciusTTS(str(write_runtime_config(args)), f"cuda:{args.device}")


def configure_runtime(args: argparse.Namespace) -> torch.device:
    if args.threads != 8:
        raise RuntimeError("Confucius4-TTS Python warmbench requires --threads 8")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Confucius4-TTS warmbench requested CUDA, but torch.cuda.is_available() is false")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(f"cuda:{args.device}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Confucius4-TTS warmbench request JSON entries must be objects")
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Confucius4-TTS warmbench request requires non-empty text")
    lang = payload.get("language", "")
    if not isinstance(lang, str) or not lang.strip():
        raise RuntimeError("Confucius4-TTS warmbench request requires non-empty language")
    prompt_wav = payload.get("prompt_wav", payload.get("reference_audio", ""))
    if not isinstance(prompt_wav, str) or not prompt_wav.strip():
        raise RuntimeError("Confucius4-TTS warmbench request requires prompt_wav/reference_audio")
    path = resolve_repo_path(Path(prompt_wav))
    if not path.is_file():
        raise RuntimeError(f"Confucius4-TTS reference audio not found: {path}")
    return dict(payload)


def load_case(catalog_path: Path, case_name: str) -> dict[str, Any] | None:
    path = resolve_repo_path(catalog_path)
    if not path.is_file():
        return None
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise RuntimeError(f"Confucius4-TTS case catalog must contain an object: {path}")
    case = catalog.get(case_name)
    if case is None:
        available = ", ".join(sorted(str(key) for key in catalog))
        raise RuntimeError(f"unknown Confucius4-TTS case name {case_name!r}; available: {available}")
    if not isinstance(case, dict):
        raise RuntimeError(f"Confucius4-TTS case {case_name!r} must contain an object")
    requests = case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"Confucius4-TTS case {case_name!r} must contain non-empty requests")
    return case


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.request_sequence_json:
        payload = json.loads(args.request_sequence_json)
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("--request-sequence-json must decode to a non-empty list")
        return [require_request(item) for item in payload]
    if args.request_json:
        return [require_request(json.loads(args.request_json))]
    case = load_case(args.case_catalog, args.case_name)
    if case is None:
        raise RuntimeError("Confucius4-TTS warmbench requires --request-json, --request-sequence-json, or --case-catalog")
    return [require_request(item) for item in case["requests"]]


def load_warmup_request(args: argparse.Namespace, requests: list[dict[str, Any]]) -> dict[str, Any]:
    if args.warmup_request_json:
        return require_request(json.loads(args.warmup_request_json))
    case = load_case(args.case_catalog, args.case_name)
    if case is not None and isinstance(case.get("warmup"), dict):
        return require_request(case["warmup"])
    return requests[0]


def request_float(request: dict[str, Any], key: str, fallback: float) -> float:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise RuntimeError(f"Confucius4-TTS request field {key} must be numeric")
    return float(value)


def request_int(request: dict[str, Any], key: str, fallback: int) -> int:
    value = request.get(key, fallback)
    if isinstance(value, bool):
        raise RuntimeError(f"Confucius4-TTS request field {key} must be an integer")
    return int(value)


def request_language(request: dict[str, Any]) -> str:
    return str(request.get("language", "zh"))


def request_reference_audio(request: dict[str, Any]) -> Path:
    return resolve_repo_path(Path(str(request.get("prompt_wav", request.get("reference_audio", ""))))).resolve()


def audio_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    audio = np.asarray(audio, dtype=np.float32)
    flat = audio.reshape(-1)
    if flat.size == 0:
        raise RuntimeError("Confucius4-TTS Python warmbench received empty audio")
    if audio.ndim == 1:
        frames = int(audio.shape[0])
        channels = 1
    elif audio.ndim == 2:
        frames = int(audio.shape[-1])
        channels = int(audio.shape[0])
    else:
        raise RuntimeError(f"Confucius4-TTS warmbench expected 1D or 2D audio, got shape {audio.shape}")
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "samples": int(flat.size),
        "frames": frames,
        "duration_sec": float(frames / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def tensor_to_audio(audio: torch.Tensor) -> np.ndarray:
    value = audio.detach().float().cpu().numpy()
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    return np.asarray(value, dtype=np.float32)


def write_runtime_config(args: argparse.Namespace) -> Path:
    source_path = resolve_repo_path(args.config).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"missing Confucius4-TTS config: {source_path}")
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("paths"), dict):
        raise RuntimeError(f"invalid Confucius4-TTS config: {source_path}")
    paths = payload["paths"]
    w2v_path = resolve_repo_path(args.w2v_bert).resolve()
    bigvgan_path = resolve_repo_path(args.bigvgan).resolve()
    campplus_path = resolve_repo_path(args.campplus).resolve()
    if not w2v_path.is_dir():
        raise RuntimeError(f"missing local W2V-BERT model directory: {w2v_path}")
    if not (bigvgan_path / "bigvgan_generator.pt").is_file():
        raise RuntimeError(f"missing local BigVGAN weights: {bigvgan_path}")
    if not campplus_path.is_file():
        raise RuntimeError(f"missing local CAMPPlus checkpoint: {campplus_path}")
    paths["w2v_bert_path"] = str(w2v_path)
    paths["vocoder_path"] = str(bigvgan_path)
    paths["style_encoder"]["checkpoint"] = str(campplus_path)
    output_path = resolve_repo_path(args.timing_file).with_name("inference_config.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


def load_model(args: argparse.Namespace, ConfuciusTTS: Any, reference_root: Path) -> Any:
    return make_local_weight_confucius_tts(ConfuciusTTS, args, reference_root)


def run_request(model: Any, request: dict[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, int]:
    seed_all(request_int(request, "seed", args.seed))
    sync_device(device)
    with torch.no_grad():
        audio = model.generate(
            text=str(request["text"]),
            lang=request_language(request),
            prompt_wav=str(request_reference_audio(request)),
            temperature=request_float(request, "temperature", args.temperature),
            top_p=request_float(request, "top_p", args.top_p),
            top_k=request_int(request, "top_k", args.top_k),
            num_beams=request_int(request, "num_beams", args.num_beams),
            repetition_penalty=request_float(request, "repetition_penalty", args.repetition_penalty),
            n_timesteps=request_int(request, "num_inference_steps", args.n_timesteps),
            inference_cfg_rate=request_float(request, "guidance_scale", args.inference_cfg_rate),
            max_text_tokens_per_segment=request_int(request, "text_chunk_size", args.text_chunk_size),
            cross_fade_duration=request_float(request, "cross_fade_duration", args.cross_fade_duration),
            edge_fade_duration=request_float(request, "edge_fade_duration", args.edge_fade_duration),
            edge_pad_duration=request_float(request, "edge_pad_duration", args.edge_pad_duration),
            verbose=False,
        )
    sync_device(device)
    return tensor_to_audio(audio), int(model.sample_rate)


def main() -> None:
    args = parse_args()
    if args.iterations != 1:
        raise RuntimeError("Confucius4-TTS warmbench records raw per-request timing; --iterations must be 1")
    device = configure_runtime(args)
    reference_root = add_reference_path(args.reference_root)
    requests = load_requests(args)
    warmup_request = load_warmup_request(args, requests)
    ConfuciusTTS, module_path = load_reference_symbols(reference_root)
    model = load_model(args, ConfuciusTTS, reference_root)

    for _ in range(args.warmup):
        run_request(model, warmup_request, args, device)

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    args.timing_file.parent.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        started = time.perf_counter()
        audio, sample_rate = run_request(model, request, args, device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        audio_path = ""
        if output_dir is not None:
            path = output_dir / f"audio_{request_index}.wav"
            sf.write(path, audio, sample_rate)
            audio_path = str(path)
        elif request_index == 0:
            sf.write(args.audio_out, audio, sample_rate)
            audio_path = str(args.audio_out)
        text_length = len(str(request["text"]))
        lang = request_language(request)
        print(f"confucius4_tts.request[{request_index}].id={request.get('id', '')}")
        print(f"confucius4_tts.request[{request_index}].language={lang}")
        print(f"confucius4_tts.request[{request_index}].length={text_length}")
        print(f"confucius4_tts.request[{request_index}].wall_ms={wall_ms:.6f}")
        stem = {"name": "audio", "summary": audio_summary(audio, sample_rate)}
        if audio_path:
            stem["audio"] = audio_path
        steps.append({
            "request_index": request_index,
            "id": str(request.get("id", "")),
            "language": lang,
            "text_length": text_length,
            "reference_audio": str(request_reference_audio(request)),
            "stems": [stem],
            "metrics": {"wall_ms": wall_ms},
        })

    summary = {
        "family": args.family,
        "backend": args.backend,
        "device": args.device,
        "threads": args.threads,
        "case_name": args.case_name,
        "timing_boundary": "per-request inference only; model load is excluded",
        "reference_module": str(module_path),
        "transformers_module": sys.modules["transformers"].__name__,
        "transformers_version": str(getattr(sys.modules["transformers"], "__version__", "")),
        "sequence_steps": steps,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    args.timing_file.write_text(summary_text + "\n", encoding="utf-8")
    if args.summary_file is not None:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(summary_text + "\n", encoding="utf-8")
    print("summary_json=" + summary_text)


if __name__ == "__main__":
    main()
