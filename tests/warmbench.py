from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
ACE_STEP_BASELINE_CPP_AUDIO = REPO_ROOT / "build/logs/warmbench/ace_step_20s_baseline/ace_step_offline_cuda/cpp_audio/audio_0.wav"
VOXCPM2_MAX_LEN_CAP = 521
PYTHON_EXE = sys.executable
WHISPER_CONDA_ENV = "qwen3-tts"
WHISPER_MODEL = "tiny.en"
WHISPER_DEVICE = "cpu"
WHISPER_WORD_TIME_TOLERANCE_SEC = 0.05
OMNIVOICE_DEFAULT_WARMUP_TEXT = "This is a fixed warmup request for the OmniVoice session benchmark."
CJK_ASR_TRANSLATION = str.maketrans({
    "軟": "软",
    "聲": "声",
    "說": "说",
    "話": "话",
    "讓": "让",
    "聽": "听",
    "來": "来",
    "這": "这",
    "個": "个",
    "們": "们",
    "會": "会",
    "還": "还",
    "裡": "里",
    "裏": "里",
    "嗎": "吗",
    "對": "对",
    "為": "为",
    "風": "风",
    "體": "体",
    "現": "现",
    "溫": "温",
    "過": "过",
    "實": "实",
    "問": "问",
    "題": "题",
    "開": "开",
    "閉": "闭",
    "發": "发",
    "聰": "聪",
    "學": "学",
    "習": "习",
    "語": "语",
    "氣": "气",
    "號": "号",
    "節": "节",
})

FAMILY_CONFIG: dict[str, dict[str, Any]] = {
    "kokoro": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/kokoro_tts_warm_bench",
        "python_script": "tests/kokoro_tts/kokoro_tts_python_warm_bench.py",
        "model": "models/kokoro-82m-v1_0-ggml",
        "case_catalog": "tests/kokoro_tts/kokoro_tts_warm_bench_cases.txt",
        "voice_id": "af_heart",
        "speaking_rate": 1.0,
        "max_input_tokens": 512,
        "min_text_length": 65,
    },
    "pocket_tts": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/pocket_tts_warm_bench",
        "python_script": "tests/pocket_tts/pocket_tts_python_warm_bench.py",
        "model": "models/pocket-tts/languages/english",
        "case_catalog": "tests/pocket_tts/pocket_tts_warm_bench_cases.txt",
        "voice_id": "alba",
        "language": "english",
        "min_text_length": 65,
    },
    "qwen3_tts": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/qwen3_tts_warm_bench",
        "python_script": "tests/qwen3_tts/qwen3_tts_python_warm_bench.py",
        "model": "models/Qwen3-TTS-12Hz-0.6B-Base",
        "clone_audio": "resources/sample.wav",
        "reference_text": "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you.",
        "case_catalog": "tests/qwen3_tts/qwen3_tts_warm_bench_cases.txt",
        "max_new_tokens": 512,
        "min_text_length": 70,
    },
    "qwen3_tts_voice_design": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/qwen3_tts_warm_bench",
        "python_script": "tests/qwen3_tts/qwen3_tts_python_warm_bench.py",
        "model": "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "task": "voice_design",
        "language": "Chinese",
        "voice_design_instruct": "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        "case_catalog": "tests/qwen3_tts/qwen3_tts_voice_design_warm_bench_cases.txt",
        "max_new_tokens": 2048,
        "min_text_length": 18,
        "whisper_model": "small",
        "whisper_language": "Chinese",
        "asr_compact_lcs_min": 0.65,
    },
    "qwen3_tts_custom_voice": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/qwen3_tts_warm_bench",
        "python_script": "tests/qwen3_tts/qwen3_tts_python_warm_bench.py",
        "model": "models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "task": "custom_voice",
        "language": "English",
        "speaker": "Ryan",
        "custom_voice_instruct": "Very happy.",
        "case_catalog": "tests/qwen3_tts/qwen3_tts_custom_voice_warm_bench_cases.txt",
        "max_new_tokens": 2048,
        "min_text_length": 65,
        "whisper_model": "tiny.en",
    },
    "chatterbox": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/chatterbox_warm_bench",
        "python_script": "tests/chatterbox/chatterbox_python_warm_bench.py",
        "model": "models/chatterbox",
        "clone_audio": "resources/sample.wav",
        "case_catalog": "tests/chatterbox/chatterbox_warm_bench_cases.txt",
        "default_warmup": 0,
        "max_new_tokens": 1000,
        "min_text_length": 70,
        "whisper_model": "tiny.en",
        "asr_compact_lcs_min": 0.65,
        "min_rms": 1.0e-3,
    },
    "omnivoice_voice_clone": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/omnivoice_warm_bench",
        "python_script": "tests/omnivoice/omnivoice_python_warm_bench.py",
        "model": "models/OmniVoice",
        "task": "voice_clone",
        "clone_audio": "resources/sample.wav",
        "reference_text": "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you.",
        "language": "en",
        "case_catalog": "tests/omnivoice/omnivoice_warm_bench_cases.json",
        "default_case_name": "default",
    },
    "omnivoice_voice_design": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/omnivoice_warm_bench",
        "python_script": "tests/omnivoice/omnivoice_python_warm_bench.py",
        "model": "models/OmniVoice",
        "task": "voice_design",
        "language": "en",
        "voice_design_instruct": "female, high pitch",
        "case_catalog": "tests/omnivoice/omnivoice_warm_bench_cases.json",
        "default_case_name": "default",
    },
    "omnivoice_auto_voice": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/omnivoice_warm_bench",
        "python_script": "tests/omnivoice/omnivoice_python_warm_bench.py",
        "model": "models/OmniVoice",
        "task": "auto_voice",
        "language": "en",
        "case_catalog": "tests/omnivoice/omnivoice_warm_bench_cases.json",
        "default_case_name": "default",
    },
    "mel_band_roformer": {
        "kind": "separation",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/roformer_warm_bench",
        "python_script": "tests/roformer/roformer_python_warm_bench.py",
        "model": "models/mel-roformer-mlx",
        "python_model": "models/melbandroformer/MelBandRoformer.ckpt",
        "python_config": "reference/Mel-Band-Roformer-Vocal-Model/configs/config_vocals_mel_band_roformer.yaml",
        "case_catalog": "tests/roformer/roformer_warm_bench_cases.json",
        "default_case_name": "default",
    },
    "htdemucs": {
        "kind": "separation",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/demucs_warm_bench",
        "python_script": "tests/demucs/demucs_python_warm_bench.py",
        "model": "models/htdemucs",
        "python_model": "models/htdemucs",
        "python_name": "htdemucs",
        "case_catalog": "tests/demucs/demucs_warm_bench_cases.json",
        "default_case_name": "default",
    },
    "moss_tts_nano": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/moss_tts_nano_warm_bench",
        "python_script": "tests/moss_tts_nano/moss_tts_nano_python_warm_bench.py",
        "model": "models/MOSS-TTS-Nano-100M",
        "audio_tokenizer_model": "models/MOSS-Audio-Tokenizer-Nano",
        "clone_audio": "resources/sample.wav",
        "case_catalog": "tests/moss_tts_nano/moss_tts_nano_warm_bench_cases.json",
        "default_case_name": "default",
        "max_new_frames": 300,
        "active_codebooks": 16,
        "do_sample": "true",
        "text_temperature": 1.5,
        "text_top_p": 1.0,
        "text_top_k": 50,
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 25,
        "repetition_penalty": 1.0,
        "use_kv_cache": "true",
        "voice_clone_max_text_tokens": 50,
        "voice_clone_max_memory_per_sample_gb": 1.0,
        "tts_max_batch_size": 0,
        "codec_max_batch_size": 0,
        "run_asr_check": False,
        "log_mel_cosine_min": 0.80,
        "length_ratio_min": 0.90,
        "whisper_word_timestamps": False,
    },
    "moss_tts_local": {
        "kind": "tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/moss_tts_local_warm_bench",
        "python_script": "tests/moss_tts_local/moss_tts_local_python_warm_bench.py",
        "model": "models/MOSS-TTS-Local-Transformer-v1.5",
        "clone_audio": "resources/a.wav",
        "case_catalog": "tests/moss_tts_local/moss_tts_local_warm_bench_cases.json",
        "default_case_name": "long_lived_session",
        "max_new_frames": 4096,
        "do_sample": "true",
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
        "dtype": "fp32",
        "cpp_session_options": ["moss_tts_local.weight_type=f32"],
        "use_kv_cache": "true",
        "default_requests_per_session": 4,
        "asr_compact_lcs_min": 0.95,
        "run_asr_check": False,
        "log_mel_cosine_min": 0.80,
        "length_ratio_min": 0.80,
        "whisper_word_timestamps": False,
    },
    "qwen3_asr": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/qwen3_asr_warm_bench",
        "python_script": "tests/qwen3_asr/qwen3_asr_python_warm_bench.py",
        "model": "models/Qwen3-ASR-0.6B",
        "case_catalog": "tests/qwen3_asr/qwen3_asr_warm_bench_cases.json",
        "max_new_tokens": 512,
        "strict_text": True,
        "check_language": True,
    },
    "higgs_audio_stt": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/higgs_audio_stt_warm_bench",
        "python_script": "tests/higgs_audio_stt/higgs_audio_stt_python_warm_bench.py",
        "model": "models/higgs-audio-v3-stt",
        "case_catalog": "tests/higgs_audio_stt/higgs_audio_stt_warm_bench_cases.json",
        "default_requests_per_session": 1,
    },
    "hviske_asr": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/hviske_asr_warm_bench",
        "python_script": "tests/hviske_asr/hviske_asr_python_warm_bench.py",
        "model": "models/hviske-v5.3",
        "case_catalog": "tests/hviske_asr/hviske_asr_warm_bench_cases.json",
        "default_requests_per_session": 1,
    },
    "nemotron_asr": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/nemotron_asr_warm_bench",
        "python_script": "tests/nemotron_asr/nemotron_asr_python_warm_bench.py",
        "model": "models/nemotron-3.5-asr-streaming-0.6b",
        "case_catalog": "tests/nemotron_asr/nemotron_asr_warm_bench_cases.json",
        "default_requests_per_session": 1,
    },
    "vibevoice_asr": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/vibevoice_asr_warm_bench",
        "python_script": "tests/vibevoice_asr/vibevoice_asr_python_warm_bench.py",
        "model": "models/VibeVoice-ASR",
        "case_catalog": "tests/vibevoice_asr/vibevoice_asr_warm_bench_cases.json",
        "default_case_name": "default",
        "default_requests_per_session": 1,
    },
    "qwen3_forced_aligner": {
        "kind": "alignment",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/qwen3_forced_aligner_warm_bench",
        "python_script": "tests/qwen3_forced_aligner/qwen3_forced_aligner_python_warm_bench.py",
        "model": "models/Qwen3-ForcedAligner-0.6B",
        "case_catalog": "tests/qwen3_forced_aligner/qwen3_forced_aligner_warm_bench_cases.json",
        "strict_alignment": True,
    },
    "ace_step": {
        "kind": "ace_step",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/ace_step_warm_bench",
        "python_script": "tests/ace_step/ace_step_python_warm_bench.py",
        "python_conda_env": "index-tts",
        "model": "models/Ace-Step1.5",
        "case_catalog": "tests/ace_step/ace_step_warm_bench_cases.json",
        "default_case_name": "parity_find",
    },
    "vevo2": {
        "kind": "vevo2",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/vevo2_warm_bench",
        "python_script": "tests/vevo2/vevo2_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/Vevo2",
        "case_catalog": "tests/vevo2/vevo2_warm_bench_cases.json",
        "default_case_name": "route_zero_shot_tts",
        "wav_cosine_min": 0.90,
        "log_mel_cosine_min": 0.90,
        "length_ratio_min": 0.80,
        "cpp_session_options": ["vevo2.weight_type=f32"],
    },
    "seed_vc": {
        "kind": "seed_vc",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/seed_vc_warm_bench",
        "python_script": "tests/seed_vc/seed_vc_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/SeedVC-MLX",
        "python_model": "models/SeedVC",
        "case_catalog": "tests/seed_vc/seed_vc_warm_bench_cases.json",
        "default_case_name": "v2_vc_example",
        "wav_cosine_min": 0.80,
    },
    "miocodec": {
        "kind": "miocodec",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/miocodec_warm_bench",
        "python_script": "tests/miocodec/miocodec_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/MioCodec-25Hz-44.1kHz-v2",
        "case_catalog": "tests/miocodec/miocodec_warm_bench_cases.json",
        "default_case_name": "vc_metis",
        "wav_cosine_min": 0.80,
        "log_mel_cosine_min": 0.80,
        "mrstft_spectral_convergence_max": 0.20,
        "mrstft_logmag_mae_max": 0.30,
        "similarity_vote_required": 2,
    },
    "voxcpm2": {
        "kind": "voxcpm2",
        "modes": ["offline", "streaming"],
        "cpp_bin": "build/debug/bin/voxcpm2_warm_bench",
        "python_script": "tests/voxcpm2/voxcpm2_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/VoxCPM2",
        "case_catalog": "tests/voxcpm2/voxcpm2_warm_bench_cases.json",
        "default_case_name": "cuda_parity_512char",
        "wav_cosine_min": 0.95,
        "log_mel_cosine_min": 0.95,
    },
    "supertonic": {
        "kind": "supertonic",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/supertonic_warm_bench",
        "python_script": "tests/supertonic/supertonic_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/supertonic-3",
        "case_catalog": "tests/supertonic/supertonic_warm_bench_cases.json",
        "default_case_name": "multi_path_voice_lang_steps_speed_chunking",
        "default_requests_per_session": 6,
        "default_warmup": 0,
        "wav_cosine_min": 0.90,
        "log_mel_cosine_min": 0.90,
    },
    "vibevoice": {
        "kind": "vibevoice",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/vibevoice_warm_bench",
        "python_script": "tests/vibevoice/vibevoice_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/VibeVoice-1.5B",
        "case_catalog": "tests/vibevoice/vibevoice_warm_bench_cases.json",
        "default_case_name": "short",
        "default_requests_per_session": 1,
        "default_warmup": 0,
        "wav_cosine_min": 0.90,
        "log_mel_cosine_min": 0.90,
        "length_ratio_is_diagnostic": True,
    },
    "irodori_tts": {
        "kind": "irodori_tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/irodori_tts_warm_bench",
        "python_script": "tests/irodori_tts/irodori_tts_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/Irodori-TTS-500M-v3",
        "case_catalog": "tests/irodori_tts/irodori_tts_warm_bench_cases.json",
        "default_case_name": "short_emoji",
        "default_requests_per_session": 1,
        "default_warmup": 0,
        "wav_cosine_min": 0.98,
        "log_mel_cosine_min": 0.98,
        "cpp_session_options": ["irodori_tts.weight_type=f32", "irodori_tts.codec_weight_type=f32"],
    },
    "heartmula": {
        "kind": "heartmula",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/heartmula_warm_bench",
        "python_script": "tests/heartmula/heartmula_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/HeartMuLa",
        "case_catalog": "tests/heartmula/heartmula_warm_bench_cases.json",
        "default_case_name": "default",
        "default_requests_per_session": 1,
        "default_warmup": 0,
        "wav_cosine_min": 0.90,
        "log_mel_cosine_min": 0.90,
        "cpp_session_options": ["heartmula.weight_type=f32"],
    },
    "higgs_tts": {
        "kind": "higgs_tts",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/higgs_tts_warm_bench",
        "python_script": "tests/higgs_tts/higgs_tts_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/higgs-audio-v3-tts-4b",
        "case_catalog": "tests/higgs_tts/higgs_tts_warm_bench_cases.json",
        "default_case_name": "default",
        "default_requests_per_session": 1,
        "default_warmup": 0,
        "wav_cosine_min": 0.90,
        "log_mel_cosine_min": 0.90,
        "length_ratio_min": 0.98,
        "cpp_session_options": ["higgs_tts.codec_weight_type=f32"],
    },
    "index_tts2": {
        "kind": "index_tts2",
        "display_name": "IndexTTS2",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/index_tts2_warm_bench",
        "python_script": "tests/index_tts2/index_tts2_python_warm_bench.py",
        "python_conda_env": "qwen3-tts",
        "model": "models/IndexTTS-2",
        "case_catalog": "tests/index_tts2/index_tts2_warm_bench_cases.json",
        "default_case_name": "voice_clone",
        "default_requests_per_session": 1,
        "default_warmup": 0,
        "cpp_session_options": ["index_tts2.weight_type=f32", "index_tts2.conv_weight_type=f32"],
        "wav_cosine_min": 0.98,
        "log_mel_cosine_min": 0.98,
        "similarity_vote_required": 1,
    },
    "parakeet": {
        "kind": "asr",
        "modes": ["offline", "longform", "streaming"],
        "cpp_bin": "build/debug/bin/parakeet_warm_bench",
        "python_script": "tests/parakeet_tdt/parakeet_python_warm_bench.py",
        "model": "models/parakeet-tdt-0.6b-v3",
    },
    "sortformer": {
        "kind": "diar",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/sortformer_diar_warm_bench",
        "python_script": "tests/sortformer_diar/sortformer_diar_python_warm_bench.py",
        "model": "models/diar_sortformer_4spk-v1",
        "python_model": "nvidia/diar_sortformer_4spk-v1",
    },
    "citrinet_asr": {
        "kind": "asr",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/citrinet_asr_warm_bench",
        "python_script": "tests/citrinet_asr/citrinet_asr_python_warm_bench.py",
        "model": "models/citrinet",
    },
    "marblenet_vad": {
        "kind": "vad",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/marblenet_vad_warm_bench",
        "python_script": "tests/marblenet_vad/marblenet_vad_python_warm_bench.py",
        "model": "models/marblenet_vad",
    },
    "silero_vad": {
        "kind": "vad",
        "modes": ["offline"],
        "cpp_bin": "build/debug/bin/silero_vad_warm_bench",
        "python_script": "tests/silero_vad/silero_vad_python_warm_bench.py",
        "model": "assets/framework/models/silero_vad",
    },
}


def timestamp_seconds_local() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def next_artifact_dir(base: Path) -> Path:
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = base.with_name(f"{base.name}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-lived multi-request audio.cpp warmbench scenarios sequentially.")
    parser.add_argument("--family", action="append", choices=tuple(FAMILY_CONFIG.keys()) + ("all",), dest="families", default=[])
    parser.add_argument("--mode", choices=("offline", "longform", "streaming", "all"), default="all")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--requests-per-session", type=int, default=2)
    parser.add_argument(
        "--audio-durations",
        default="",
        help="Comma-separated audio request durations in seconds for ASR/diarization benchmarks.",
    )
    parser.add_argument(
        "--audio-warmup-duration",
        type=float,
        default=1.6,
        help="Warmup audio duration in seconds for ASR/diarization benchmarks.",
    )
    parser.add_argument("--text", action="append", dest="texts", default=[], help="Repeat to provide exact TTS request texts.")
    parser.add_argument(
        "--case-name",
        action="append",
        dest="case_names",
        default=[],
        help="Repeat to select named TTS case-catalog sections in order.",
    )
    parser.add_argument("--warmup-text", default="", help="TTS warmup text passed to both Python and C++ warm benches.")
    parser.add_argument("--clone-audio", default="", help="Reference audio used for voice clone parity.")
    parser.add_argument("--model", default="", help="Override the model path for the selected family.")
    parser.add_argument("--python-model", default="", help="Override the Python reference model path for the selected family.")
    parser.add_argument(
        "--voice-design-instruct",
        action="append",
        dest="voice_design_instructs",
        default=[],
        help="Repeat to provide per-request Qwen3 voice-design/custom-voice instructions.",
    )
    parser.add_argument(
        "--request-instruct",
        action="append",
        dest="voice_design_instructs",
        default=[],
        help="Alias for --voice-design-instruct.",
    )
    parser.add_argument(
        "--warmup-voice-design-instruct",
        default="",
        help="Qwen3 voice-design/custom-voice instruction used for warmup; defaults to the family instruction.",
    )
    parser.add_argument(
        "--warmup-instruct",
        dest="warmup_voice_design_instruct",
        default="",
        help="Alias for --warmup-voice-design-instruct.",
    )
    parser.add_argument("--speaker", default="", help="Qwen3 custom-voice speaker override.")
    parser.add_argument("--warmup-speaker", default="", help="Qwen3 custom-voice warmup speaker override.")
    parser.add_argument(
        "--request-speaker",
        action="append",
        dest="request_speakers",
        default=[],
        help="Repeat to provide per-request Qwen3 custom-voice speakers.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Override Qwen3 TTS max generated code frames.")
    parser.add_argument("--do-sample", choices=("true", "false"), default="true", help="Qwen3 TTS main talker sampling mode.")
    parser.add_argument(
        "--subtalker-do-sample",
        choices=("true", "false"),
        default="true",
        help="Qwen3 TTS code-predictor sampling mode.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Qwen3 TTS main talker top-k.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Qwen3 TTS main talker top-p.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Qwen3 TTS main talker temperature.")
    parser.add_argument("--subtalker-top-k", type=int, default=50, help="Qwen3 TTS code-predictor top-k.")
    parser.add_argument("--subtalker-top-p", type=float, default=1.0, help="Qwen3 TTS code-predictor top-p.")
    parser.add_argument("--subtalker-temperature", type=float, default=0.9, help="Qwen3 TTS code-predictor temperature.")
    parser.add_argument("--chatterbox-language", default="en", help="Chatterbox language code.")
    parser.add_argument("--chatterbox-s3gen-cfg-rate", type=float, default=0.7, help="Chatterbox S3Gen CFM CFG rate.")
    parser.add_argument(
        "--cpp-session-option",
        action="append",
        default=[],
        help="Repeat key=value C++ session options for model-specific experiments.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--test-noise-file",
        default="",
        help="Optional shared ACE-Step DiT initial-noise file (.f32). When set, both Python and C++ use this file instead of sampling noise.",
    )
    parser.add_argument("--artifact-stamp", default="")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--keep-output-dir", action="store_true")
    args = parser.parse_args()
    args.warmup_was_explicit = any(arg == "--warmup" or arg.startswith("--warmup=") for arg in sys.argv[1:])
    args.requests_per_session_was_explicit = any(
        arg == "--requests-per-session" or arg.startswith("--requests-per-session=")
        for arg in sys.argv[1:]
    )
    args.seed_was_explicit = any(arg == "--seed" or arg.startswith("--seed=") for arg in sys.argv[1:])
    return args


def selected_families(args: argparse.Namespace) -> list[str]:
    if not args.families or "all" in args.families:
        return list(FAMILY_CONFIG.keys())
    return args.families


def selected_backends(args: argparse.Namespace) -> list[str]:
    return ["cpu", "cuda"] if args.backend == "all" else [args.backend]


def python_reference_backend(backend: str) -> str:
    return "cuda" if backend == "vulkan" else backend


def set_command_backend(command: list[str], backend: str) -> None:
    for index, value in enumerate(command[:-1]):
        if value == "--backend":
            command[index + 1] = backend
            return
    raise RuntimeError("warmbench command is missing --backend")


def selected_modes(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    modes = list(config["modes"])
    if args.mode == "all":
        return modes
    return [args.mode] if args.mode in modes else []


def effective_warmup(config: dict[str, Any], args: argparse.Namespace) -> int:
    if args.warmup_was_explicit:
        return args.warmup
    return int(config.get("default_warmup", args.warmup))


def aggregate_warmup_value(args: argparse.Namespace) -> int:
    if args.warmup_was_explicit:
        return args.warmup
    values = {
        effective_warmup(FAMILY_CONFIG[family], args)
        for family in selected_families(args)
    }
    if len(values) == 1:
        return next(iter(values))
    return args.warmup


def aggregate_requests_per_session_value(args: argparse.Namespace) -> int:
    if args.requests_per_session_was_explicit:
        return args.requests_per_session
    values = {
        int(FAMILY_CONFIG[family].get("default_requests_per_session", args.requests_per_session))
        for family in selected_families(args)
    }
    if len(values) == 1:
        return next(iter(values))
    return args.requests_per_session


def load_omnivoice_case_catalog(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid OmniVoice case catalog: {path}")
    return payload


def resolve_omnivoice_case(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    if args.case_names and args.texts:
        raise RuntimeError("--case-name and --text are mutually exclusive for OmniVoice warmbench")

    scenario_config = dict(config)
    warmup_text = args.warmup_text or OMNIVOICE_DEFAULT_WARMUP_TEXT
    request_manifest: dict[str, Any] = {"warmup_text": warmup_text}
    if args.texts:
        if len(args.texts) < args.requests_per_session:
            raise RuntimeError(
                f"need {args.requests_per_session} --text values, only received {len(args.texts)}"
            )
        texts = args.texts[: args.requests_per_session]
        request_manifest["texts"] = texts
        return texts, request_manifest, scenario_config

    task_name = str(config["task"])
    catalog = load_omnivoice_case_catalog(REPO_ROOT / str(config["case_catalog"]))
    task_cases = catalog.get(task_name)
    if not isinstance(task_cases, dict):
        raise RuntimeError(f"OmniVoice case catalog is missing task {task_name!r}")

    if len(args.case_names) > 1:
        raise RuntimeError("OmniVoice warmbench accepts at most one --case-name because a case carries scenario metadata")
    case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
    case = task_cases.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(task_cases))
        raise RuntimeError(f"unknown OmniVoice case name {case_name!r} for task {task_name}; available: {available}")

    texts = case.get("texts")
    if not isinstance(texts, list) or not texts:
        raise RuntimeError(f"OmniVoice case {task_name}.{case_name} has no texts")
    if len(texts) < args.requests_per_session:
        raise RuntimeError(
            f"OmniVoice case {task_name}.{case_name} needs at least {args.requests_per_session} texts, only has {len(texts)}"
        )
    texts = [str(text) for text in texts[: args.requests_per_session]]
    for key in ("language", "voice_design_instruct", "reference_text", "whisper_model", "whisper_language", "asr_compact_lcs_min"):
        if key in case:
            scenario_config[key] = case[key]
    if "warmup_text" in case and not args.warmup_text:
        request_manifest["warmup_text"] = str(case["warmup_text"])
    request_manifest["texts"] = texts
    request_manifest["case_names"] = [case_name]
    return texts, request_manifest, scenario_config


def resolve_tts_texts(
    family: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    mode: str,
) -> tuple[list[str], dict[str, Any]]:
    if args.case_names and args.texts:
        raise RuntimeError("--case-name and --text are mutually exclusive for TTS benchmarks")

    if family == "moss_tts_local" and not args.texts:
        if len(args.case_names) > 1:
            raise RuntimeError("MOSS-TTS-Local warmbench accepts at most one --case-name")
        case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "long_lived_session"))
        catalog_path = REPO_ROOT / str(config["case_catalog"])
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        case = catalog.get(case_name)
        if not isinstance(case, dict):
            available = ", ".join(sorted(catalog))
            raise RuntimeError(f"unknown MOSS-TTS-Local case name {case_name!r}; available: {available}")
        requests = case.get("requests")
        if not isinstance(requests, list) or not requests:
            raise RuntimeError(f"MOSS-TTS-Local case {case_name!r} has no requests")
        if len(requests) < args.requests_per_session:
            raise RuntimeError(
                f"MOSS-TTS-Local case {case_name!r} needs at least {args.requests_per_session} requests, only has {len(requests)}"
            )
        selected = [dict(request) for request in requests[: args.requests_per_session]]
        if getattr(args, "seed_was_explicit", False):
            for request in selected:
                request["seed"] = args.seed
        texts = [str(request.get("text", "")) for request in selected]
        if any(not text for text in texts):
            raise RuntimeError(f"MOSS-TTS-Local case {case_name!r} contains an empty text")
        warmup_text = args.warmup_text or str(case.get("warmup_text", ""))
        return texts, {
            "texts": texts,
            "warmup_text": warmup_text,
            "case_names": [case_name],
            "requests": selected,
        }

    if family == "moss_tts_nano" and not args.texts:
        if len(args.case_names) > 1:
            raise RuntimeError("MOSS-TTS-Nano warmbench accepts at most one --case-name")
        case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
        catalog_path = REPO_ROOT / str(config["case_catalog"])
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        case = catalog.get(case_name)
        if not isinstance(case, dict):
            available = ", ".join(sorted(catalog))
            raise RuntimeError(f"unknown MOSS-TTS-Nano case name {case_name!r}; available: {available}")
        texts = case.get("texts")
        if not isinstance(texts, list) or not texts:
            raise RuntimeError(f"MOSS-TTS-Nano case {case_name!r} has no texts")
        if len(texts) < args.requests_per_session:
            raise RuntimeError(
                f"MOSS-TTS-Nano case {case_name!r} needs at least {args.requests_per_session} texts, only has {len(texts)}"
            )
        selected = [str(text) for text in texts[: args.requests_per_session]]
        if any(not text for text in selected):
            raise RuntimeError(f"MOSS-TTS-Nano case {case_name!r} contains an empty text")
        request_manifest: dict[str, Any] = {
            "texts": selected,
            "case_names": [case_name],
        }
        if "warmup_text" in case and not args.warmup_text:
            request_manifest["warmup_text"] = str(case["warmup_text"])
        options = case.get("options")
        if isinstance(options, dict):
            request_manifest["options"] = dict(options)
        return selected, request_manifest

    case_catalog_path = REPO_ROOT / config["case_catalog"]
    used_case_names: list[str] = []
    if args.case_names:
        used_case_names = list(args.case_names)
        texts = load_named_case_texts(case_catalog_path, used_case_names, args.requests_per_session)
    elif args.texts:
        if len(args.texts) < args.requests_per_session:
            raise RuntimeError(
                f"need {args.requests_per_session} --text values, only received {len(args.texts)}"
            )
        texts = args.texts[: args.requests_per_session]
    else:
        texts = choose_diverse_texts(
            load_case_texts(case_catalog_path),
            args.requests_per_session,
            scenario_seed(args.seed, family, mode),
            int(config.get("min_text_length", 12)),
        )

    warmup_text = args.warmup_text
    if not warmup_text and family == "chatterbox" and used_case_names:
        warmup_text = load_named_case_warmup_text(case_catalog_path, used_case_names)
    request_manifest: dict[str, Any] = {"texts": texts, "warmup_text": warmup_text}
    if used_case_names:
        request_manifest["case_names"] = used_case_names
    return texts, request_manifest


def resolve_ace_step_case(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(args.case_names) > 1:
        raise RuntimeError("ACE-Step warmbench accepts at most one --case-name")
    if args.texts:
        raise RuntimeError("ACE-Step warmbench uses JSON case requests; --text is not supported")
    case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
    catalog_path = REPO_ROOT / str(config["case_catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    case = catalog.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(catalog))
        raise RuntimeError(f"unknown ACE-Step case name {case_name!r}; available: {available}")
    requests = case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"ACE-Step case {case_name!r} has no requests")
    if len(requests) < args.requests_per_session:
        raise RuntimeError(
            f"ACE-Step case {case_name!r} needs at least {args.requests_per_session} requests, only has {len(requests)}"
        )
    selected = [dict(request) for request in requests[: args.requests_per_session]]
    if getattr(args, "seed_was_explicit", False):
        for request in selected:
            request["seed"] = args.seed
    assert_fields = case.get("assert_optional_request_fields", [])
    if assert_fields:
        if not isinstance(assert_fields, list) or not all(isinstance(field, str) for field in assert_fields):
            raise RuntimeError(f"ACE-Step case {case_name!r} has invalid assert_optional_request_fields")
        for request_index, request in enumerate(selected):
            for field in assert_fields:
                if field not in request:
                    raise RuntimeError(
                        f"ACE-Step case {case_name!r} request {request_index} missing asserted optional field {field!r}"
                    )
                value = request[field]
                if value is None or value == "" or value == []:
                    raise RuntimeError(
                        f"ACE-Step case {case_name!r} request {request_index} has empty asserted optional field {field!r}"
                    )
    return selected, {"case_name": case_name, "requests": selected, "assert_optional_request_fields": assert_fields}


def resolve_vevo2_case(config: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(args.case_names) > 1:
        raise RuntimeError("Vevo2 warmbench accepts at most one --case-name")
    if args.texts:
        raise RuntimeError("Vevo2 warmbench uses JSON case requests; --text is not supported")
    case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
    catalog_path = REPO_ROOT / str(config["case_catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    case = catalog.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(catalog))
        raise RuntimeError(f"unknown Vevo2 case name {case_name!r}; available: {available}")
    requests = case.get("requests")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"Vevo2 case {case_name!r} has no requests")
    if len(requests) < args.requests_per_session:
        raise RuntimeError(
            f"Vevo2 case {case_name!r} needs at least {args.requests_per_session} requests, only has {len(requests)}"
        )
    selected = [dict(request) for request in requests[: args.requests_per_session]]
    if getattr(args, "seed_was_explicit", False):
        for request in selected:
            request["seed"] = args.seed
    return selected, {"case_name": case_name, "requests": selected}


def resolve_miocodec_case(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if len(args.case_names) > 1:
        raise RuntimeError("MioCodec warmbench accepts at most one --case-name")
    if args.texts:
        raise RuntimeError("MioCodec warmbench uses JSON case requests; --text is not supported")
    case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
    catalog_path = REPO_ROOT / str(config["case_catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    case = catalog.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(catalog))
        raise RuntimeError(f"unknown MioCodec case name {case_name!r}; available: {available}")
    warmup = case.get("warmup")
    requests = case.get("requests")
    if not isinstance(warmup, dict):
        raise RuntimeError(f"MioCodec case {case_name!r} is missing warmup request")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"MioCodec case {case_name!r} has no requests")
    if len(requests) < args.requests_per_session:
        raise RuntimeError(
            f"MioCodec case {case_name!r} needs at least {args.requests_per_session} requests, only has {len(requests)}"
        )
    selected = [dict(request) for request in requests[: args.requests_per_session]]
    return dict(warmup), selected, {"case_name": case_name, "warmup": warmup, "requests": selected}


def resolve_voxcpm2_case(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if len(args.case_names) > 1:
        raise RuntimeError("VoxCPM2 warmbench accepts at most one --case-name")
    if args.texts:
        raise RuntimeError("VoxCPM2 warmbench uses JSON case requests; --text is not supported")
    case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
    catalog_path = REPO_ROOT / str(config["case_catalog"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    case = catalog.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(catalog))
        raise RuntimeError(f"unknown VoxCPM2 case name {case_name!r}; available: {available}")
    warmup = case.get("warmup")
    requests = case.get("requests")
    if not isinstance(warmup, dict):
        raise RuntimeError(f"VoxCPM2 case {case_name!r} is missing warmup request")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"VoxCPM2 case {case_name!r} has no requests")
    if len(requests) < args.requests_per_session:
        raise RuntimeError(
            f"VoxCPM2 case {case_name!r} needs at least {args.requests_per_session} requests, only has {len(requests)}"
        )
    capped_warmup = dict(warmup)
    if "max_len" in capped_warmup:
        capped_warmup["max_len"] = min(int(capped_warmup["max_len"]), VOXCPM2_MAX_LEN_CAP)
    selected = []
    for request in requests[: args.requests_per_session]:
        capped_request = dict(request)
        if "max_len" in capped_request:
            capped_request["max_len"] = min(int(capped_request["max_len"]), VOXCPM2_MAX_LEN_CAP)
        selected.append(capped_request)
    if getattr(args, "seed_was_explicit", False):
        for request in selected:
            request["seed"] = args.seed
    return capped_warmup, selected, {"case_name": case_name, "warmup": capped_warmup, "requests": selected}


def omnivoice_instruct_values(
    config: dict[str, Any],
    args: argparse.Namespace,
    request_count: int,
) -> tuple[str, list[str]]:
    default_instruct = str(config.get("voice_design_instruct", ""))
    warmup_instruct = args.warmup_voice_design_instruct or default_instruct
    if args.voice_design_instructs:
        if len(args.voice_design_instructs) < request_count:
            raise RuntimeError(
                f"need {request_count} --voice-design-instruct values for measured requests, "
                f"only received {len(args.voice_design_instructs)}"
            )
        return warmup_instruct, args.voice_design_instructs[:request_count]
    return warmup_instruct, [default_instruct] * request_count


def scenario_seed(base_seed: int, family: str, mode: str) -> int:
    return base_seed + len(family) + len(mode)


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{log_timestamp()}] {message}\n")


def write_timestamped_lines(path: Path, prefix: str, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in text.splitlines():
            handle.write(f"[{log_timestamp()}] {prefix} {line}\n")


def parse_summary_lines(text: str) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    warmup_summaries: list[dict[str, Any]] = []
    texts: list[str] = []
    warmup_texts: list[str] = []
    audio_out: str | None = None
    audio_outs: dict[int, str] = {}
    final_summary: dict[str, Any] | None = None
    current_average: dict[str, float] | None = None
    average_metrics: list[dict[str, float]] = []
    for line in text.splitlines():
        if line.startswith("average["):
            current_average = {}
            average_metrics.append(current_average)
            continue
        if line.startswith("warmup_text["):
            warmup_texts.append(line.split("=", 1)[1])
            continue
        if line.startswith("warmup_summary_json["):
            warmup_summaries.append(json.loads(line.split("=", 1)[1]))
        elif line.startswith("text["):
            texts.append(line.split("=", 1)[1])
        elif line.startswith("summary_json["):
            summaries.append(json.loads(line.split("=", 1)[1]))
        elif line.startswith("summary_json="):
            final_summary = json.loads(line.split("=", 1)[1])
        elif line.startswith("{"):
            payload = json.loads(line)
            if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
                final_summary = {
                    "family": payload.get("family"),
                    "backend": payload.get("backend"),
                    "sequence_steps": payload["steps"],
                }
        elif line.startswith("audio_out["):
            index_text = line.split("]", 1)[0][len("audio_out["):]
            audio_outs[int(index_text)] = line.split("=", 1)[1]
        elif line.startswith("audio_out="):
            audio_out = line.split("=", 1)[1]
        elif current_average is not None and "=" in line:
            key, value = line.split("=", 1)
            try:
                current_average[key] = float(value)
            except ValueError:
                continue
    metrics: dict[str, float] = {}
    if average_metrics:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for average in average_metrics:
            for key, value in average.items():
                totals[key] = totals.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        metrics = {key: totals[key] / counts[key] for key in totals}
    return {
        "summaries": summaries,
        "warmup_summaries": warmup_summaries,
        "texts": texts,
        "warmup_texts": warmup_texts,
        "summary": final_summary,
        "audio_out": audio_out,
        "audio_outs": audio_outs,
        "metrics": metrics,
        "request_metrics": average_metrics,
    }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip().lower()).translate(CJK_ASR_TRANSLATION)
    chars: list[str] = []
    for char in text:
        if unicodedata.category(char).startswith("P"):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def normalize_whisper_text(text: str) -> str:
    return normalize_text(text)


def normalize_whisper_word(word: str) -> str:
    return normalize_text(word)


def edit_distance(a: str, b: str, stop_after: int = 2) -> int:
    if abs(len(a) - len(b)) > stop_after:
        return stop_after + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = current[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > stop_after:
            return stop_after + 1
        previous = current
    return previous[-1]


def lcs_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for ca in a:
        current = [0]
        for j, cb in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if ca == cb else max(previous[j], current[-1]))
        previous = current
    return previous[-1] / float(max(len(a), len(b)))


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def run_whisper(
    audio_path: Path,
    artifact_dir: Path,
    threads: int,
    model: str = WHISPER_MODEL,
    language: str | None = None,
    word_timestamps: bool = True,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "-n",
        WHISPER_CONDA_ENV,
        "whisper",
        str(audio_path),
        "--model",
        model,
        "--device",
        WHISPER_DEVICE,
        "--output_dir",
        str(artifact_dir),
        "--output_format",
        "json",
        "--word_timestamps",
        "True" if word_timestamps else "False",
        "--fp16",
        "False",
        "--threads",
        str(threads),
        "--verbose",
        "False",
    ]
    if language:
        command.extend(["--language", language])
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"whisper failed ({completed.returncode}) for {audio_path}\n{completed.stdout}\n{completed.stderr}"
        )
    output_json = artifact_dir / f"{audio_path.stem}.json"
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    text = " ".join(segment.get("text", "").strip() for segment in payload.get("segments", [])).strip()
    words: list[dict[str, Any]] = []
    for segment in payload.get("segments", []):
        for word in segment.get("words", []):
            words.append(
                {
                    "word": word.get("word", "").strip(),
                    "word_normalized": normalize_whisper_word(word.get("word", "")),
                    "start": word.get("start"),
                    "end": word.get("end"),
                }
            )
    return {
        "audio_path": str(audio_path),
        "json_path": str(output_json),
        "text": text,
        "text_normalized": normalize_whisper_text(text),
        "words": words,
    }


def compare_whisper_outputs(
    cpp_whisper: dict[str, Any],
    py_whisper: dict[str, Any],
    compact_lcs_min: float = 0.0,
) -> dict[str, Any]:
    mismatches: list[str] = []
    timing_mismatches: list[str] = []
    cpp_text = str(cpp_whisper.get("text_normalized", ""))
    py_text = str(py_whisper.get("text_normalized", ""))
    if not cpp_text or not py_text:
        mismatches.append("empty_text")
    text_distance = edit_distance(cpp_text, py_text, 1)
    cpp_compact = cpp_text.replace(" ", "")
    py_compact = py_text.replace(" ", "")
    compact_text_distance = edit_distance(cpp_compact, py_compact, 1)
    compact_lcs = lcs_ratio(cpp_compact, py_compact)
    if text_distance > 1 and compact_text_distance > 1 and compact_lcs < compact_lcs_min:
        mismatches.append("text")
    cpp_words = cpp_whisper.get("words", [])
    py_words = py_whisper.get("words", [])
    if len(cpp_words) != len(py_words):
        timing_mismatches.append("word_count")
    else:
        for index, (cpp_word, py_word) in enumerate(zip(cpp_words, py_words)):
            if cpp_word.get("word_normalized") != py_word.get("word_normalized"):
                timing_mismatches.append(f"word[{index}]")
                break
            cpp_start = cpp_word.get("start")
            py_start = py_word.get("start")
            if cpp_start is None or py_start is None or abs(float(cpp_start) - float(py_start)) > WHISPER_WORD_TIME_TOLERANCE_SEC:
                timing_mismatches.append(f"start[{index}]")
                break
            cpp_end = cpp_word.get("end")
            py_end = py_word.get("end")
            if cpp_end is None or py_end is None or abs(float(cpp_end) - float(py_end)) > WHISPER_WORD_TIME_TOLERANCE_SEC:
                timing_mismatches.append(f"end[{index}]")
                break
    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "text_distance": text_distance,
        "compact_text_distance": compact_text_distance,
        "compact_lcs": compact_lcs,
        "compact_lcs_min": compact_lcs_min,
        "timing_mismatches": timing_mismatches,
    }


def load_case_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        texts.append(line)
    deduped = list(dict.fromkeys(texts))
    if not deduped:
        raise RuntimeError(f"no texts found in case catalog: {path}")
    return deduped


def load_text_case_catalog(path: Path) -> dict[str, list[str]]:
    cases: dict[str, list[str]] = {}
    current_case = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_case = line[1:-1]
            cases.setdefault(current_case, [])
            continue
        if not current_case:
            raise RuntimeError(f"case catalog entry is missing a [case] header: {path}")
        cases[current_case].append(line)
    return cases


def load_named_case_texts(path: Path, case_names: list[str], count: int) -> list[str]:
    catalog = load_text_case_catalog(path)
    texts: list[str] = []
    for case_name in case_names:
        if case_name not in catalog:
            available = ", ".join(sorted(catalog))
            raise RuntimeError(f"unknown case name {case_name!r} in {path}; available: {available}")
        texts.extend(catalog[case_name])
    if len(texts) < count:
        raise RuntimeError(f"need {count} texts from selected cases, only found {len(texts)}")
    return texts[:count]


def load_named_case_warmup_text(path: Path, case_names: list[str]) -> str:
    if not case_names:
        return ""
    catalog = load_text_case_catalog(path)
    warmup_case = f"{case_names[0]}_warmup"
    warmup_texts = catalog.get(warmup_case, [])
    return warmup_texts[0] if warmup_texts else ""


def load_qwen3_asr_cases(path: Path, count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"no Qwen3 ASR requests found in case catalog: {path}")
    if len(requests) < count:
        raise RuntimeError(f"need {count} Qwen3 ASR requests, only found {len(requests)}")
    warmup = payload.get("warmup") or requests[0]
    return warmup, requests[:count]


def load_qwen3_forced_aligner_cases(path: Path, count: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"no Qwen3 forced aligner requests found in case catalog: {path}")
    if len(requests) < count:
        raise RuntimeError(f"need {count} Qwen3 forced aligner requests, only found {len(requests)}")
    warmup = payload.get("warmup") or requests[0]
    return warmup, requests[:count]


def load_catalog_asr_cases(
    path: Path,
    count: int,
    family_label: str,
    case_name: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {family_label} case catalog: {path}")
    selected_case_name = case_name
    case_payload: dict[str, Any]
    if selected_case_name:
        cases = payload.get("cases")
        if isinstance(cases, dict) and selected_case_name in cases:
            selected = cases[selected_case_name]
        else:
            selected = payload.get(selected_case_name)
        if not isinstance(selected, dict):
            available = []
            if isinstance(cases, dict):
                available.extend(str(key) for key in cases.keys())
            available.extend(str(key) for key, value in payload.items() if isinstance(value, dict) and key != "cases")
            raise RuntimeError(
                f"unknown {family_label} case name {selected_case_name!r} in {path}; available: {', '.join(sorted(set(available)))}"
            )
        case_payload = selected
    elif isinstance(payload.get("requests"), list):
        case_payload = payload
    else:
        available = [str(key) for key, value in payload.items() if isinstance(value, dict)]
        raise RuntimeError(f"{family_label} case catalog requires --case-name; available: {', '.join(sorted(available))}")
    requests = case_payload.get("requests", [])
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"{family_label} case {selected_case_name or '<default>'!r} has no requests")
    if len(requests) < count:
        raise RuntimeError(f"need {count} {family_label} requests, only found {len(requests)}")
    warmup = (
        case_payload.get("warmup")
        or case_payload.get("offline_warmup")
        or case_payload.get("streaming_warmup")
        or requests[0]
    )
    if not isinstance(warmup, dict):
        raise RuntimeError(f"{family_label} case {selected_case_name or '<default>'!r} has invalid warmup")
    return dict(warmup), [dict(item) for item in requests[:count]], {
        "case_name": selected_case_name,
        "warmup": warmup,
        "requests": requests[:count],
    }


def load_separation_cases(path: Path, case_name: str, count: int, family_label: str) -> tuple[Path, list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid {family_label} case catalog: {path}")
    case = payload.get(case_name)
    if not isinstance(case, dict):
        available = ", ".join(sorted(payload))
        raise RuntimeError(f"unknown {family_label} case name {case_name!r} in {path}; available: {available}")
    warmup = case.get("warmup_audio")
    requests = case.get("requests", [])
    if not isinstance(warmup, str) or not warmup:
        raise RuntimeError(f"{family_label} case {case_name!r} is missing warmup_audio")
    if not isinstance(requests, list) or not requests:
        raise RuntimeError(f"{family_label} case {case_name!r} has no requests")
    if len(requests) < count:
        raise RuntimeError(f"need {count} {family_label} requests, only found {len(requests)} in case {case_name!r}")
    return Path(warmup), [Path(str(item)) for item in requests[:count]]


def choose_diverse_texts(pool: list[str], count: int, seed: int, min_length: int = 12) -> list[str]:
    filtered_pool = [text for text in pool if len(text) >= min_length]
    if len(filtered_pool) >= count:
        pool = filtered_pool
    if len(pool) < count:
        raise RuntimeError(f"need at least {count} texts, only found {len(pool)}")
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda item: (len(item), item))
    chosen: list[str] = []
    used: set[str] = set()
    for bucket_index in range(count):
        bucket_start = (len(ordered) * bucket_index) // count
        bucket_end = (len(ordered) * (bucket_index + 1)) // count
        bucket = [text for text in ordered[bucket_start:bucket_end] if text not in used]
        if not bucket:
            bucket = [text for text in ordered if text not in used]
        pick = bucket[rng.randrange(len(bucket))]
        chosen.append(pick)
        used.add(pick)
    return chosen


def read_wave_pcm16_mono(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM WAV, got sample width {sample_width} for {path}")
    if channels == 2:
        stereo = array.array("h")
        stereo.frombytes(frames)
        mono = array.array("h")
        for index in range(0, len(stereo), 2):
            mono.append(int((int(stereo[index]) + int(stereo[index + 1])) / 2))
        frames = mono.tobytes()
    elif channels != 1:
        raise RuntimeError(f"expected mono or stereo WAV, got {channels} channels for {path}")
    return sample_rate, frames


def write_wave_pcm16_mono(path: Path, sample_rate: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def ensure_16k_mono_audio(source: Path, dest: Path) -> Path:
    sample_rate, frames = read_wave_pcm16_mono(source)
    if sample_rate != 16000:
        samples = array.array("h")
        samples.frombytes(frames)
        target_length = max(1, round(len(samples) * 16000 / sample_rate))
        resampled = array.array("h")
        if len(samples) == 1:
            resampled.extend([samples[0]] * target_length)
        else:
            scale = (len(samples) - 1) / max(1, target_length - 1)
            for output_index in range(target_length):
                position = output_index * scale
                left_index = int(position)
                right_index = min(left_index + 1, len(samples) - 1)
                fraction = position - left_index
                left_value = int(samples[left_index])
                right_value = int(samples[right_index])
                interpolated = int(round(left_value * (1.0 - fraction) + right_value * fraction))
                resampled.append(interpolated)
        frames = resampled.tobytes()
        sample_rate = 16000
    write_wave_pcm16_mono(dest, sample_rate, frames)
    return dest


def materialize_qwen3_asr_16k_audio(source: Path, dest: Path, start_sec: float | None, end_sec: float | None) -> Path:
    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = librosa.load(source, sr=None, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio.mean(axis=-1).astype(np.float32)
    if start_sec is not None or end_sec is not None:
        start = max(0, int(round((start_sec or 0.0) * sample_rate)))
        end = len(audio) if end_sec is None else min(len(audio), int(round(end_sec * sample_rate)))
        if end <= start:
            raise RuntimeError(f"invalid Qwen3 ASR crop window for {source}: {start_sec}..{end_sec}")
        audio = audio[start:end]
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000).astype(np.float32)
    if audio.size == 0:
        raise RuntimeError(f"Qwen3 ASR materialized empty audio: {source}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, audio, 16000, subtype="PCM_16")
    return dest


def materialize_qwen3_asr_audio(case: dict[str, Any], output_dir: Path, index: int, role: str) -> Path:
    source = REPO_ROOT / str(case["audio"])
    if "start_sec" not in case and "end_sec" not in case:
        sample_rate, _ = read_wave_pcm16_mono(source)
        if sample_rate == 16000:
            return source
        output_dir.mkdir(parents=True, exist_ok=True)
        return materialize_qwen3_asr_16k_audio(source, output_dir / f"{role}_{index:02d}.wav", None, None)
    if "end_sec" not in case:
        raise RuntimeError(f"Qwen3 ASR crop requires end_sec for {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_sec = float(case.get("start_sec", 0.0))
    end_sec = float(case["end_sec"])
    if end_sec <= start_sec:
        raise RuntimeError(f"invalid Qwen3 ASR crop window for {source}: {start_sec}..{end_sec}")
    return materialize_qwen3_asr_16k_audio(source, output_dir / f"{role}_{index:02d}.wav", start_sec, end_sec)


def parse_audio_durations(value: str) -> list[float]:
    if not value:
        return []
    durations: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        duration = float(item)
        if duration <= 0.0:
            raise RuntimeError(f"audio duration must be positive: {item}")
        durations.append(duration)
    return durations


def build_audio_requests(
    request_dir: Path,
    count: int,
    seed: int,
    durations_sec: list[float],
    warmup_duration_sec: float,
) -> tuple[Path, list[Path]]:
    request_dir.mkdir(parents=True, exist_ok=True)
    base_source = REPO_ROOT / "resources" / "sample.wav"
    if not base_source.exists():
        raise RuntimeError(f"missing base audio for request generation: {base_source}")
    base_16k = ensure_16k_mono_audio(base_source, request_dir / "base_16k.wav")
    sample_rate, frames = read_wave_pcm16_mono(base_16k)
    total_frames = len(frames) // 2
    if not durations_sec:
        durations_sec = [2.2, 3.7, 5.4, 6.8, 8.1]
    if len(durations_sec) < count:
        raise RuntimeError(f"need {count} audio durations, only received {len(durations_sec)}")
    chosen_durations = durations_sec[:count]
    rng = random.Random(seed)
    requests: list[Path] = []
    for index, duration_sec in enumerate(chosen_durations):
        request_frames = min(total_frames, int(duration_sec * sample_rate))
        max_start = max(0, total_frames - request_frames)
        start_frame = 0 if max_start == 0 else rng.randrange(0, max_start + 1)
        byte_start = start_frame * 2
        byte_end = byte_start + request_frames * 2
        request_path = request_dir / f"request_{index:02d}_{request_frames}.wav"
        write_wave_pcm16_mono(request_path, sample_rate, frames[byte_start:byte_end])
        requests.append(request_path)
    if warmup_duration_sec <= 0.0:
        raise RuntimeError(f"audio warmup duration must be positive: {warmup_duration_sec}")
    warmup_frames = min(total_frames, int(warmup_duration_sec * sample_rate))
    warmup_path = request_dir / "warmup.wav"
    write_wave_pcm16_mono(warmup_path, sample_rate, frames[: warmup_frames * 2])
    return warmup_path, requests


def compare_float(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def compare_kokoro(cpp_summary: dict[str, Any], py_summary: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for key in ("sample_rate", "channels", "samples"):
        if cpp_summary.get(key) != py_summary.get(key):
            mismatches.append(key)
    for key in ("sum", "mean_abs", "rms", "min", "max"):
        if not compare_float(float(cpp_summary.get(key, 0.0)), float(py_summary.get(key, 0.0)), 1e-4):
            mismatches.append(key)
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_pocket(cpp_summary: dict[str, Any], py_summary: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for key in (
        "sample_rate",
        "channels",
        "samples",
        "request_char_count",
        "generated_steps",
        "test_noise_mode",
        "test_noise_seed",
        "test_noise_steps",
        "test_noise_latent_dim",
        "test_noise_hash",
        "test_noise_file",
    ):
        if key in cpp_summary and key in py_summary and cpp_summary.get(key) != py_summary.get(key):
            mismatches.append(key)
    tolerances = {"sum": 1e-3, "mean_abs": 1e-4, "rms": 1e-4, "min": 1e-4, "max": 1e-4}
    for key, tolerance in tolerances.items():
        if key in cpp_summary and key in py_summary and not compare_float(float(cpp_summary.get(key, 0.0)), float(py_summary.get(key, 0.0)), tolerance):
            mismatches.append(key)
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_moss_tts(
    cpp_summary: dict[str, Any],
    py_summary: dict[str, Any],
    cpp_audio_path: Path,
    py_audio_path: Path,
    log_mel_cosine_min: float,
    length_ratio_min: float,
) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    mismatches: list[str] = []
    for key in ("sample_rate", "channels", "request_char_count"):
        if key in cpp_summary and key in py_summary and cpp_summary.get(key) != py_summary.get(key):
            mismatches.append(key)

    cpp_audio, cpp_sr = sf.read(str(cpp_audio_path), always_2d=True)
    py_audio, py_sr = sf.read(str(py_audio_path), always_2d=True)
    if int(cpp_sr) != int(py_sr):
        mismatches.append("sample_rate")
    cpp_audio = np.asarray(cpp_audio, dtype=np.float32)
    py_audio = np.asarray(py_audio, dtype=np.float32)
    common_frames = min(cpp_audio.shape[0], py_audio.shape[0])
    if common_frames <= 0:
        return {
            "ok": False,
            "reason": "mismatch:empty_audio",
            "mismatches": ["empty_audio"],
            "metrics": {"log_mel_cosine_min": log_mel_cosine_min},
        }
    length_ratio = common_frames / float(max(cpp_audio.shape[0], py_audio.shape[0]))
    if length_ratio < length_ratio_min:
        mismatches.append("length_ratio")
    cpp_mono = cpp_audio[:common_frames].mean(axis=1)
    py_mono = py_audio[:common_frames].mean(axis=1)
    mel_kwargs = {"sr": int(py_sr), "n_fft": 1024, "hop_length": 256, "n_mels": 80, "power": 2.0}
    cpp_log_mel = np.log(np.maximum(librosa.feature.melspectrogram(y=cpp_mono, **mel_kwargs), 1.0e-10))
    py_log_mel = np.log(np.maximum(librosa.feature.melspectrogram(y=py_mono, **mel_kwargs), 1.0e-10))
    common_cols = min(cpp_log_mel.shape[1], py_log_mel.shape[1])
    cpp_flat = cpp_log_mel[:, :common_cols].reshape(-1).astype(np.float64, copy=False)
    py_flat = py_log_mel[:, :common_cols].reshape(-1).astype(np.float64, copy=False)
    denom = float(np.linalg.norm(cpp_flat) * np.linalg.norm(py_flat))
    log_mel_cosine = 1.0 if denom == 0.0 else float(np.dot(cpp_flat, py_flat) / denom)
    if log_mel_cosine < log_mel_cosine_min:
        mismatches.append("log_mel_cosine")
    metrics = {
        "log_mel_cosine": log_mel_cosine,
        "log_mel_cosine_min": log_mel_cosine_min,
        "length_ratio": length_ratio,
        "length_ratio_min": length_ratio_min,
        "cpp_frames": int(cpp_audio.shape[0]),
        "python_frames": int(py_audio.shape[0]),
        "common_frames": int(common_frames),
        "sample_rate": int(py_sr),
    }
    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "metrics": metrics,
    }


def summarize_tts_asr(whisper_output: dict[str, Any]) -> str:
    return f"text={json.dumps(whisper_output.get('text', ''), ensure_ascii=False)} words={len(whisper_output.get('words', []))} json={whisper_output.get('json_path', '')}"


def compare_parakeet_step(cpp_step: dict[str, Any], py_step: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    sample_tolerance = 1
    if normalize_text(str(cpp_step.get("text_output", ""))) != normalize_text(str(py_step.get("text_output", ""))):
        mismatches.append("text_output")
    cpp_words = cpp_step.get("word_timestamps", [])
    py_words = py_step.get("word_timestamps", [])
    if len(cpp_words) != len(py_words):
        mismatches.append("word_count")
    else:
        for index, (cpp_word, py_word) in enumerate(zip(cpp_words, py_words)):
            if normalize_text(str(cpp_word.get("word", ""))) != normalize_text(str(py_word.get("word", ""))):
                mismatches.append(f"word[{index}]")
                break
            if abs(int(cpp_word.get("start_sample", 0)) - int(py_word.get("start_sample", 0))) > sample_tolerance:
                mismatches.append(f"start_sample[{index}]")
                break
            if abs(int(cpp_word.get("end_sample", 0)) - int(py_word.get("end_sample", 0))) > sample_tolerance:
                mismatches.append(f"end_sample[{index}]")
                break
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_qwen3_asr_step(
    cpp_step: dict[str, Any],
    py_step: dict[str, Any],
    expected_fragments: list[str],
    check_language: bool,
) -> dict[str, Any]:
    mismatches: list[str] = []
    cpp_text = normalize_text(str(cpp_step.get("text_output", "")))
    py_text = normalize_text(str(py_step.get("text_output", "")))
    if cpp_text != py_text:
        mismatches.append("text_output")
    if check_language and normalize_text(str(cpp_step.get("language", ""))) != normalize_text(str(py_step.get("language", ""))):
        mismatches.append("language")
    missing_fragments: list[str] = []
    for fragment in expected_fragments:
        expected = normalize_text(fragment)
        if expected and (expected not in cpp_text or expected not in py_text):
            missing_fragments.append(fragment)
    if missing_fragments:
        mismatches.append("expected_fragments")
    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "cpp_text_normalized": cpp_text,
        "python_text_normalized": py_text,
        "expected_fragments": expected_fragments,
        "missing_fragments": missing_fragments,
    }


def normalize_alignment_word(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def compare_qwen3_forced_aligner_step(
    cpp_step: dict[str, Any],
    py_step: dict[str, Any],
    expected_words: list[str],
) -> dict[str, Any]:
    mismatches: list[str] = []
    cpp_words = cpp_step.get("word_timestamps", [])
    py_words = py_step.get("word_timestamps", [])
    if len(cpp_words) != len(py_words):
        mismatches.append("word_count")
    else:
        for index, (cpp_word, py_word) in enumerate(zip(cpp_words, py_words)):
            if normalize_alignment_word(str(cpp_word.get("word", ""))) != normalize_alignment_word(str(py_word.get("word", ""))):
                mismatches.append(f"word[{index}]")
                break
            if abs(int(cpp_word.get("start_sample", 0)) - int(py_word.get("start_sample", 0))) > 1:
                mismatches.append(f"start_sample[{index}]")
                break
            if abs(int(cpp_word.get("end_sample", 0)) - int(py_word.get("end_sample", 0))) > 1:
                mismatches.append(f"end_sample[{index}]")
                break
    normalized_cpp_words = {normalize_alignment_word(str(item.get("word", ""))) for item in cpp_words}
    missing_words = [
        word for word in expected_words
        if normalize_alignment_word(word) not in normalized_cpp_words
    ]
    if missing_words:
        mismatches.append("expected_words")
    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "expected_words": expected_words,
        "missing_words": missing_words,
    }


def compare_sortformer_step(cpp_step: dict[str, Any], py_step: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    sample_tolerance = 1
    cpp_turns = cpp_step.get("speaker_turns", [])
    py_turns = py_step.get("speaker_turns", [])
    if len(cpp_turns) != len(py_turns):
        mismatches.append("turn_count")
    else:
        for index, (cpp_turn, py_turn) in enumerate(zip(cpp_turns, py_turns)):
            if cpp_turn.get("speaker_id") != py_turn.get("speaker_id"):
                mismatches.append(f"speaker_id[{index}]")
                break
            if abs(int(cpp_turn.get("start_sample", 0)) - int(py_turn.get("start_sample", 0))) > sample_tolerance:
                mismatches.append(f"start_sample[{index}]")
                break
            if abs(int(cpp_turn.get("end_sample", 0)) - int(py_turn.get("end_sample", 0))) > sample_tolerance:
                mismatches.append(f"end_sample[{index}]")
                break
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_vad_step(cpp_step: dict[str, Any], py_step: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    cpp_segments = cpp_step.get("speech_segments", [])
    py_segments = py_step.get("speech_segments", [])
    if len(cpp_segments) != len(py_segments):
        mismatches.append("segment_count")
    else:
        for index, (cpp_segment, py_segment) in enumerate(zip(cpp_segments, py_segments)):
            if int(cpp_segment.get("start_sample", 0)) != int(py_segment.get("start_sample", 0)):
                mismatches.append(f"start_sample[{index}]")
                break
            if int(cpp_segment.get("end_sample", 0)) != int(py_segment.get("end_sample", 0)):
                mismatches.append(f"end_sample[{index}]")
                break
            if abs(float(cpp_segment.get("confidence", 0.0)) - float(py_segment.get("confidence", 0.0))) > 1e-4:
                mismatches.append(f"confidence[{index}]")
                break
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_speaker_step(cpp_step: dict[str, Any], py_step: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    if str(cpp_step.get("label", "")) != str(py_step.get("label", "")):
        mismatches.append("label")
    if int(cpp_step.get("index", -1)) != int(py_step.get("index", -1)):
        mismatches.append("index")
    if abs(float(cpp_step.get("score", 0.0)) - float(py_step.get("score", 0.0))) > 1e-4:
        mismatches.append("score")
    return {"ok": not mismatches, "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}", "mismatches": mismatches}


def compare_separation_step(cpp_step: dict[str, Any], py_step: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    cpp_stems = {
        str(stem.get("name", "")): stem
        for stem in cpp_step.get("stems", [])
        if isinstance(stem, dict) and stem.get("name")
    }
    py_stems = {
        str(stem.get("name", "")): stem
        for stem in py_step.get("stems", [])
        if isinstance(stem, dict) and stem.get("name")
    }
    mismatches: list[str] = []
    metrics: dict[str, Any] = {}
    if set(cpp_stems) != set(py_stems):
        mismatches.append("stem_names")
        return {
            "ok": False,
            "reason": f"mismatch:{mismatches[0]}",
            "mismatches": mismatches,
            "metrics": metrics,
        }

    for stem_name in sorted(py_stems):
        cpp_stem = cpp_stems[stem_name]
        py_stem = py_stems[stem_name]
        cpp_summary = cpp_stem.get("summary", {})
        py_summary = py_stem.get("summary", {})
        stem_mismatches: list[str] = []
        for key in ("sample_rate", "channels", "frames", "samples"):
            if cpp_summary.get(key) != py_summary.get(key):
                stem_mismatches.append(key)
        cpp_audio_path = resolve_repo_path(str(cpp_stem.get("audio", "")))
        py_audio_path = resolve_repo_path(str(py_stem.get("audio", "")))
        if cpp_audio_path is None or py_audio_path is None or not cpp_audio_path.exists() or not py_audio_path.exists():
            stem_mismatches.append("audio_path")
            metrics[stem_name] = {"ok": False, "mismatches": stem_mismatches}
            mismatches.append(f"{stem_name}:{stem_mismatches[0]}")
            continue

        cpp_audio, cpp_sr = sf.read(str(cpp_audio_path), always_2d=True)
        py_audio, py_sr = sf.read(str(py_audio_path), always_2d=True)
        cpp_audio = np.asarray(cpp_audio, dtype=np.float32)
        py_audio = np.asarray(py_audio, dtype=np.float32)
        if cpp_sr != py_sr:
            stem_mismatches.append("sample_rate")
        if cpp_audio.shape != py_audio.shape:
            stem_mismatches.append("shape")
            metrics[stem_name] = {
                "ok": False,
                "mismatches": stem_mismatches,
                "cpp_shape": list(cpp_audio.shape),
                "python_shape": list(py_audio.shape),
            }
            mismatches.append(f"{stem_name}:{stem_mismatches[0]}")
            continue

        diff = cpp_audio - py_audio
        ref = py_audio
        mae = float(np.mean(np.abs(diff), dtype=np.float64))
        rmse = float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))
        max_abs = float(np.max(np.abs(diff)))
        cpp_flat = cpp_audio.reshape(-1).astype(np.float64, copy=False)
        py_flat = py_audio.reshape(-1).astype(np.float64, copy=False)
        denom = float(np.linalg.norm(cpp_flat) * np.linalg.norm(py_flat))
        cosine = 1.0 if denom == 0.0 else float(np.dot(cpp_flat, py_flat) / denom)
        ref_power = float(np.sum(np.square(ref, dtype=np.float64)))
        err_power = float(np.sum(np.square(diff, dtype=np.float64)))
        ref_rms = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64))))
        snr_db = float("inf") if err_power == 0.0 else float(10.0 * np.log10(max(ref_power, 1e-30) / err_power))
        low_energy_ref = ref_rms < 1e-3

        if mae > 1e-3:
            stem_mismatches.append("mae")
        if rmse > 2e-3:
            stem_mismatches.append("rmse")
        if max_abs > 2e-2:
            stem_mismatches.append("max_abs")
        if not low_energy_ref and cosine < 0.999:
            stem_mismatches.append("cosine")
        if not low_energy_ref and snr_db < 40.0:
            stem_mismatches.append("snr_db")

        metrics[stem_name] = {
            "ok": not stem_mismatches,
            "mismatches": stem_mismatches,
            "mae": mae,
            "rmse": rmse,
            "max_abs": max_abs,
            "cosine": cosine,
            "snr_db": snr_db,
            "ref_rms": ref_rms,
            "low_energy_ref": low_energy_ref,
        }
        if stem_mismatches:
            mismatches.append(f"{stem_name}:{stem_mismatches[0]}")

    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "metrics": metrics,
    }


def compare_single_audio_step(
    cpp_step: dict[str, Any],
    py_step: dict[str, Any],
    cosine_min: float,
    log_mel_cosine_min: float | None = None,
    mrstft_spectral_convergence_max: float | None = None,
    mrstft_logmag_mae_max: float | None = None,
    similarity_vote_required: int | None = None,
    length_ratio_min: float = 0.98,
    length_ratio_is_diagnostic: bool = False,
) -> dict[str, Any]:
    import librosa
    import numpy as np
    import soundfile as sf

    cpp_stems = cpp_step.get("stems", [])
    py_stems = py_step.get("stems", [])
    mismatches: list[str] = []
    if not cpp_stems or not py_stems:
        return {"ok": False, "reason": "missing_stem", "mismatches": ["missing_stem"], "metrics": {}}
    cpp_audio_path = resolve_repo_path(str(cpp_stems[0].get("audio", "")))
    py_audio_path = resolve_repo_path(str(py_stems[0].get("audio", "")))
    if cpp_audio_path is None or py_audio_path is None or not cpp_audio_path.exists() or not py_audio_path.exists():
        return {"ok": False, "reason": "missing_audio_path", "mismatches": ["audio_path"], "metrics": {}}

    cpp_audio, cpp_sr = sf.read(str(cpp_audio_path), always_2d=True)
    py_audio, py_sr = sf.read(str(py_audio_path), always_2d=True)
    cpp_audio = np.asarray(cpp_audio, dtype=np.float32)
    py_audio = np.asarray(py_audio, dtype=np.float32)
    if cpp_sr != py_sr:
        mismatches.append("sample_rate")
    cpp_flat = cpp_audio.reshape(-1).astype(np.float64, copy=False)
    py_flat = py_audio.reshape(-1).astype(np.float64, copy=False)
    common = min(cpp_flat.size, py_flat.size)
    if common <= 0:
        mismatches.append("empty_audio")
        common = 0
    length_ratio = 0.0 if max(cpp_flat.size, py_flat.size) == 0 else common / float(max(cpp_flat.size, py_flat.size))
    if length_ratio < length_ratio_min:
        mismatches.append("length_ratio")
    if common > 0:
        cpp_common = cpp_flat[:common]
        py_common = py_flat[:common]
        diff = cpp_common - py_common
        denom = float(np.linalg.norm(cpp_common) * np.linalg.norm(py_common))
        cosine = 1.0 if denom == 0.0 else float(np.dot(cpp_common, py_common) / denom)
        mae = float(np.mean(np.abs(diff), dtype=np.float64))
        rmse = float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))
        max_abs = float(np.max(np.abs(diff)))
    else:
        cosine = 0.0
        mae = float("inf")
        rmse = float("inf")
        max_abs = float("inf")
    waveform_ok = cosine >= cosine_min
    if not waveform_ok:
        mismatches.append("cosine")
    if log_mel_cosine_min is not None and cpp_sr == py_sr and common > 0:
        cpp_mono = cpp_audio.mean(axis=1)[:common].astype(np.float32, copy=False)
        py_mono = py_audio.mean(axis=1)[:common].astype(np.float32, copy=False)
        mel_kwargs = {"sr": int(cpp_sr), "n_fft": 1024, "hop_length": 256, "n_mels": 80, "power": 2.0}
        cpp_log_mel = np.log(np.maximum(librosa.feature.melspectrogram(y=cpp_mono, **mel_kwargs), 1.0e-10))
        py_log_mel = np.log(np.maximum(librosa.feature.melspectrogram(y=py_mono, **mel_kwargs), 1.0e-10))
        common_cols = min(cpp_log_mel.shape[1], py_log_mel.shape[1])
        if common_cols > 0:
            cpp_mel_flat = cpp_log_mel[:, :common_cols].reshape(-1).astype(np.float64, copy=False)
            py_mel_flat = py_log_mel[:, :common_cols].reshape(-1).astype(np.float64, copy=False)
            mel_denom = float(np.linalg.norm(cpp_mel_flat) * np.linalg.norm(py_mel_flat))
            log_mel_cosine = 1.0 if mel_denom == 0.0 else float(np.dot(cpp_mel_flat, py_mel_flat) / mel_denom)
        else:
            log_mel_cosine = 0.0
        log_mel_ok = log_mel_cosine >= log_mel_cosine_min
        if not log_mel_ok:
            mismatches.append("log_mel_cosine")
    else:
        log_mel_cosine = None
        log_mel_ok = None
    if cpp_sr == py_sr and common > 0:
        if "cpp_mono" not in locals():
            cpp_mono = cpp_audio.mean(axis=1)[:common].astype(np.float32, copy=False)
            py_mono = py_audio.mean(axis=1)[:common].astype(np.float32, copy=False)
        spectral_convergences: list[float] = []
        logmag_maes: list[float] = []
        for n_fft, hop_length in ((512, 128), (1024, 256), (2048, 512)):
            cpp_stft = librosa.stft(cpp_mono, n_fft=n_fft, hop_length=hop_length, center=True)
            py_stft = librosa.stft(py_mono, n_fft=n_fft, hop_length=hop_length, center=True)
            common_cols = min(cpp_stft.shape[1], py_stft.shape[1])
            if common_cols <= 0:
                continue
            cpp_mag = np.abs(cpp_stft[:, :common_cols]).astype(np.float64, copy=False)
            py_mag = np.abs(py_stft[:, :common_cols]).astype(np.float64, copy=False)
            denom = float(np.linalg.norm(py_mag))
            spectral_convergences.append(float(np.linalg.norm(cpp_mag - py_mag) / max(denom, 1.0e-12)))
            cpp_logmag = np.log(np.maximum(cpp_mag, 1.0e-7))
            py_logmag = np.log(np.maximum(py_mag, 1.0e-7))
            logmag_maes.append(float(np.mean(np.abs(cpp_logmag - py_logmag), dtype=np.float64)))
        mrstft_spectral_convergence = float(np.mean(spectral_convergences, dtype=np.float64)) if spectral_convergences else None
        mrstft_logmag_mae = float(np.mean(logmag_maes, dtype=np.float64)) if logmag_maes else None
    else:
        mrstft_spectral_convergence = None
        mrstft_logmag_mae = None
    mrstft_ok = None
    if mrstft_spectral_convergence_max is not None and mrstft_logmag_mae_max is not None:
        mrstft_ok = (
            mrstft_spectral_convergence is not None and
            mrstft_logmag_mae is not None and
            mrstft_spectral_convergence <= mrstft_spectral_convergence_max and
            mrstft_logmag_mae <= mrstft_logmag_mae_max
        )
        if not mrstft_ok:
            mismatches.append("mrstft")
    metrics = {
        "cosine": cosine,
        "cosine_min": cosine_min,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
        "cpp_samples": int(cpp_flat.size),
        "python_samples": int(py_flat.size),
        "common_samples": int(common),
        "length_ratio": length_ratio,
        "length_ratio_min": length_ratio_min,
        "cpp_sample_rate": int(cpp_sr),
        "python_sample_rate": int(py_sr),
    }
    if log_mel_cosine_min is not None:
        metrics["log_mel_cosine"] = log_mel_cosine
        metrics["log_mel_cosine_min"] = log_mel_cosine_min
        metrics["log_mel_ok"] = log_mel_ok
    metrics["mrstft_spectral_convergence"] = mrstft_spectral_convergence
    if mrstft_spectral_convergence_max is not None:
        metrics["mrstft_spectral_convergence_max"] = mrstft_spectral_convergence_max
    metrics["mrstft_logmag_mae"] = mrstft_logmag_mae
    if mrstft_logmag_mae_max is not None:
        metrics["mrstft_logmag_mae_max"] = mrstft_logmag_mae_max
    if mrstft_ok is not None:
        metrics["mrstft_ok"] = mrstft_ok
    metrics["waveform_ok"] = waveform_ok
    if similarity_vote_required is not None:
        vote_results = [waveform_ok]
        if log_mel_ok is not None:
            vote_results.append(log_mel_ok)
        if mrstft_ok is not None:
            vote_results.append(mrstft_ok)
        similarity_votes = sum(1 for item in vote_results if item)
        metrics["similarity_votes"] = similarity_votes
        metrics["similarity_vote_required"] = similarity_vote_required
        if similarity_votes >= similarity_vote_required:
            mismatches = [item for item in mismatches if item not in {"cosine", "log_mel_cosine", "mrstft"}]
    elif log_mel_ok is not None:
        metrics["similarity_gate"] = "waveform_or_log_mel"
        if waveform_ok or log_mel_ok:
            mismatches = [item for item in mismatches if item not in {"cosine", "log_mel_cosine"}]
            if length_ratio_is_diagnostic:
                mismatches = [item for item in mismatches if item != "length_ratio"]
    return {
        "ok": not mismatches,
        "reason": "ok" if not mismatches else f"mismatch:{mismatches[0]}",
        "mismatches": mismatches,
        "metrics": metrics,
    }


def read_status_memory_kb(pid: int) -> dict[str, int]:
    status_path = Path("/proc") / str(pid) / "status"
    memory: dict[str, int] = {}
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:") or line.startswith("VmSize:"):
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts:
                    memory[key] = int(parts[0])
    except OSError:
        pass
    return memory


def sample_process_memory_kb(pid: int) -> dict[str, int]:
    memory = read_status_memory_kb(pid)
    return {
        "rss_kb": memory.get("VmRSS", 0),
        "vms_kb": memory.get("VmSize", 0),
    }


def run_command(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> tuple[str, dict[str, int]]:
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    peak_memory = {"peak_rss_kb": 0, "peak_vms_kb": 0}
    while True:
        sample = sample_process_memory_kb(process.pid)
        peak_memory["peak_rss_kb"] = max(peak_memory["peak_rss_kb"], sample["rss_kb"])
        peak_memory["peak_vms_kb"] = max(peak_memory["peak_vms_kb"], sample["vms_kb"])
        try:
            stdout, stderr = process.communicate(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            time.sleep(0.02)
    write_timestamped_lines(log_path, "STDOUT", stdout)
    with log_path.open("a", encoding="utf-8") as handle:
        for line in stderr.splitlines():
            handle.write(f"[{log_timestamp()}] STDERR {line}\n")
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{stdout}\n{stderr}"
        )
    return stdout, peak_memory


def build_tts_commands(
    family: str,
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    texts: list[str],
) -> tuple[list[str], list[str]]:
    timing_py = scenario_dir / "python.timing.log"
    timing_cpp = scenario_dir / "cpp.timing.log"
    audio_py = scenario_dir / "python_audio.wav"
    audio_cpp = scenario_dir / "cpp_audio.wav"
    audio_py_dir = scenario_dir / "python_audio"
    audio_cpp_dir = scenario_dir / "cpp_audio"
    common_text_args: list[str] = []
    for text in texts:
        common_text_args.extend(["--text", text])
    common_warmup_args: list[str] = []
    warmup_text = args.warmup_text or str(config.get("warmup_text", ""))
    if warmup_text:
        common_warmup_args = ["--warmup-text", warmup_text]
    model_path = args.model or config["model"]
    if family == "kokoro":
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            config["model"],
            "--voice-id",
            config["voice_id"],
            "--speaking-rate",
            str(config.get("speaking_rate", 1.0)),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + common_warmup_args + common_text_args
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            config["model"],
            "--voice-id",
            config["voice_id"],
            "--speaking-rate",
            str(config.get("speaking_rate", 1.0)),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-input-tokens",
            str(config.get("max_input_tokens", 512)),
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
        ] + common_warmup_args + common_text_args
        for option in config.get("cpp_session_options", []):
            cpp_command.extend(["--session-option", option])
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family in {"qwen3_tts", "qwen3_tts_voice_design", "qwen3_tts_custom_voice"}:
        max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else config["max_new_tokens"]
        task_name = config.get("task", "tts")
        clone_audio = args.clone_audio or config.get("clone_audio", "")
        prompt_control_args: list[str] = []
        if task_name in {"voice_design", "custom_voice"}:
            default_instruct = config.get("voice_design_instruct", config.get("custom_voice_instruct", ""))
            warmup_instruct = args.warmup_voice_design_instruct or default_instruct
            prompt_control_args.extend(["--instruct", default_instruct])
            prompt_control_args.extend(["--warmup-instruct", warmup_instruct])
            if args.voice_design_instructs:
                if len(args.voice_design_instructs) < len(texts):
                    raise RuntimeError(
                        f"need {len(texts)} --voice-design-instruct values for measured requests, "
                        f"only received {len(args.voice_design_instructs)}"
                    )
                for instruct in args.voice_design_instructs[: len(texts)]:
                    prompt_control_args.extend(["--request-instruct", instruct])
        if task_name == "custom_voice":
            speaker = args.speaker or config["speaker"]
            prompt_control_args.extend(["--speaker", speaker])
            prompt_control_args.extend(["--warmup-speaker", args.warmup_speaker or speaker])
            if args.request_speakers:
                if len(args.request_speakers) < len(texts):
                    raise RuntimeError(
                        f"need {len(texts)} --request-speaker values for measured requests, "
                        f"only received {len(args.request_speakers)}"
                    )
                for speaker_name in args.request_speakers[: len(texts)]:
                    prompt_control_args.extend(["--request-speaker", speaker_name])
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            model_path,
            "--task",
            task_name,
            "--language",
            config.get("language", "Auto"),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-tokens",
            str(max_new_tokens),
            "--seed",
            str(args.seed),
            "--do-sample",
            args.do_sample,
            "--subtalker-do-sample",
            args.subtalker_do_sample,
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--temperature",
            str(args.temperature),
            "--subtalker-top-k",
            str(args.subtalker_top_k),
            "--subtalker-top-p",
            str(args.subtalker_top_p),
            "--subtalker-temperature",
            str(args.subtalker_temperature),
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + common_warmup_args + common_text_args
        if task_name in {"voice_design", "custom_voice"}:
            python_command.extend(prompt_control_args)
        else:
            python_command.extend(["--clone-audio", clone_audio, "--reference-text", config["reference_text"]])
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            model_path,
            "--task",
            task_name,
            "--language",
            config.get("language", "Auto"),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-tokens",
            str(max_new_tokens),
            "--seed",
            str(args.seed),
            "--do-sample",
            args.do_sample,
            "--subtalker-do-sample",
            args.subtalker_do_sample,
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--temperature",
            str(args.temperature),
            "--subtalker-top-k",
            str(args.subtalker_top_k),
            "--subtalker-top-p",
            str(args.subtalker_top_p),
            "--subtalker-temperature",
            str(args.subtalker_temperature),
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
        ] + common_warmup_args + common_text_args
        if task_name in {"voice_design", "custom_voice"}:
            cpp_command.extend(prompt_control_args)
        else:
            cpp_command.extend(["--clone-audio", clone_audio, "--reference-text", config["reference_text"]])
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family == "chatterbox":
        max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else config["max_new_tokens"]
        clone_audio = args.clone_audio or config["clone_audio"]
        warmup = effective_warmup(config, args)
        chatterbox_args = [
            "--seed",
            str(args.seed),
            "--language",
            args.chatterbox_language,
            "--exaggeration",
            "0.5",
            "--cfg-weight",
            "0.5",
            "--s3gen-cfg-rate",
            str(args.chatterbox_s3gen_cfg_rate),
            "--temperature",
            "0.8",
            "--repetition-penalty",
            "1.2",
            "--min-p",
            "0.05",
            "--top-p",
            "1.0",
        ]
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            model_path,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(warmup),
            "--iterations",
            str(args.iterations),
            "--clone-audio",
            clone_audio,
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + chatterbox_args + common_warmup_args + common_text_args
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            model_path,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-tokens",
            str(max_new_tokens),
            "--clone-audio",
            clone_audio,
            "--do-sample",
            "true",
            "--stop-on-eos",
            "true",
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
        ] + chatterbox_args + common_warmup_args + common_text_args
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family.startswith("omnivoice_"):
        task_name = str(config.get("task", "voice_clone"))
        clone_audio = args.clone_audio or config.get("clone_audio", "")
        omnivoice_warmup_args = common_warmup_args if common_warmup_args else ["--warmup-text", OMNIVOICE_DEFAULT_WARMUP_TEXT]
        prompt_control_args: list[str] = ["--task", task_name]
        if task_name == "voice_clone":
            prompt_control_args.extend(["--clone-audio", clone_audio, "--reference-text", config["reference_text"]])
        elif task_name == "voice_design":
            warmup_instruct, request_instructs = omnivoice_instruct_values(config, args, len(texts))
            default_instruct = str(config.get("voice_design_instruct", ""))
            prompt_control_args.extend(["--instruct", default_instruct, "--warmup-instruct", warmup_instruct])
            for instruct in request_instructs:
                prompt_control_args.extend(["--request-instruct", instruct])
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            model_path,
            "--language",
            config.get("language", "en"),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--num-step",
            "32",
            "--guidance-scale",
            "2.0",
            "--speed",
            "1.0",
            "--t-shift",
            "0.1",
            "--denoise",
            "true",
            "--postprocess-output",
            "true",
            "--layer-penalty-factor",
            "5.0",
            "--position-temperature",
            "0.0",
            "--class-temperature",
            "0.0",
            "--audio-chunk-duration",
            "15.0",
            "--audio-chunk-threshold",
            "30.0",
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + omnivoice_warmup_args + common_text_args + prompt_control_args
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            model_path,
            "--language",
            config.get("language", "en"),
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--num-step",
            "32",
            "--guidance-scale",
            "2.0",
            "--speed",
            "1.0",
            "--t-shift",
            "0.1",
            "--denoise",
            "true",
            "--postprocess-output",
            "true",
            "--layer-penalty-factor",
            "5.0",
            "--position-temperature",
            "0.0",
            "--class-temperature",
            "0.0",
            "--audio-chunk-duration",
            "15.0",
            "--audio-chunk-threshold",
            "30.0",
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
        ] + omnivoice_warmup_args + common_text_args + prompt_control_args
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family == "moss_tts_local":
        clone_audio = args.clone_audio or config["clone_audio"]
        max_new_frames = int(config.get("max_new_frames", 256))
        do_sample = str(config.get("do_sample", "true"))
        audio_temperature = float(config.get("audio_temperature", 1.0))
        audio_top_p = float(config.get("audio_top_p", 0.95))
        audio_top_k = int(config.get("audio_top_k", 50))
        audio_repetition_penalty = float(config.get("audio_repetition_penalty", 1.0))
        dtype = str(config.get("dtype", "bf16"))
        use_kv_cache = str(config.get("use_kv_cache", "true"))
        request_file_args: list[str] = []
        request_file = config.get("moss_tts_local_request_file")
        if request_file:
            request_file_args = ["--request-file", str(request_file)]
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            model_path,
            "--clone-audio",
            clone_audio,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-frames",
            str(max_new_frames),
            "--seed",
            str(args.seed),
            "--do-sample",
            do_sample,
            "--audio-temperature",
            str(audio_temperature),
            "--audio-top-p",
            str(audio_top_p),
            "--audio-top-k",
            str(audio_top_k),
            "--audio-repetition-penalty",
            str(audio_repetition_penalty),
            "--dtype",
            dtype,
            "--use-kv-cache",
            use_kv_cache,
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + common_warmup_args + request_file_args + ([] if request_file_args else common_text_args)
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            model_path,
            "--clone-audio",
            clone_audio,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-frames",
            str(max_new_frames),
            "--seed",
            str(args.seed),
            "--do-sample",
            do_sample,
            "--audio-temperature",
            str(audio_temperature),
            "--audio-top-p",
            str(audio_top_p),
            "--audio-top-k",
            str(audio_top_k),
            "--audio-repetition-penalty",
            str(audio_repetition_penalty),
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
            "--log-file",
            str(scenario_dir / "cpp.framework.log"),
        ] + common_warmup_args + request_file_args + ([] if request_file_args else common_text_args)
        for option in config.get("cpp_session_options", []):
            cpp_command.extend(["--session-option", option])
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family == "moss_tts_nano":
        audio_tokenizer_model = Path(model_path) / "audio_tokenizer"
        if not audio_tokenizer_model.exists():
            audio_tokenizer_model = resolve_repo_path(str(config["audio_tokenizer_model"])) or Path(config["audio_tokenizer_model"])
        clone_audio = args.clone_audio or config["clone_audio"]
        max_new_frames = int(config.get("max_new_frames", 300))
        active_codebooks = int(config.get("active_codebooks", 16))
        do_sample = str(config.get("do_sample", "true"))
        text_temperature = float(config.get("text_temperature", 1.5))
        text_top_p = float(config.get("text_top_p", 1.0))
        text_top_k = int(config.get("text_top_k", 50))
        audio_temperature = float(config.get("temperature", 1.7))
        audio_top_p = float(config.get("top_p", 0.8))
        audio_top_k = int(config.get("top_k", 25))
        audio_repetition_penalty = float(config.get("repetition_penalty", 1.0))
        use_kv_cache = str(config.get("use_kv_cache", "true"))
        voice_clone_max_text_tokens = int(config.get("voice_clone_max_text_tokens", 50))
        voice_clone_max_memory_per_sample_gb = float(config.get("voice_clone_max_memory_per_sample_gb", 1.0))
        tts_max_batch_size = int(config.get("tts_max_batch_size", 0))
        codec_max_batch_size = int(config.get("codec_max_batch_size", 0))
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            model_path,
            "--audio-tokenizer-model",
            str(audio_tokenizer_model),
            "--clone-audio",
            clone_audio,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-frames",
            str(max_new_frames),
            "--active-codebooks",
            str(active_codebooks),
            "--do-sample",
            do_sample,
            "--text-temperature",
            str(text_temperature),
            "--text-top-p",
            str(text_top_p),
            "--text-top-k",
            str(text_top_k),
            "--audio-temperature",
            str(audio_temperature),
            "--audio-top-p",
            str(audio_top_p),
            "--audio-top-k",
            str(audio_top_k),
            "--audio-repetition-penalty",
            str(audio_repetition_penalty),
            "--seed",
            str(args.seed),
            "--use-kv-cache",
            use_kv_cache,
            "--voice-clone-max-text-tokens",
            str(voice_clone_max_text_tokens),
            "--voice-clone-max-memory-per-sample-gb",
            str(voice_clone_max_memory_per_sample_gb),
            "--tts-max-batch-size",
            str(tts_max_batch_size),
            "--codec-max-batch-size",
            str(codec_max_batch_size),
            "--audio-out",
            str(audio_py),
            "--audio-out-dir",
            str(audio_py_dir),
            "--timing-file",
            str(timing_py),
        ] + common_warmup_args + common_text_args
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            model_path,
            "--clone-audio",
            clone_audio,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--audio-out",
            str(audio_cpp),
            "--audio-out-dir",
            str(audio_cpp_dir),
            "--timing-file",
            str(timing_cpp),
        ] + common_warmup_args + common_text_args
        cpp_request_options = {
            "max_tokens": str(max_new_frames),
            "active_codebooks": str(active_codebooks),
            "do_sample": do_sample,
            "use_kv_cache": use_kv_cache,
            "text_temperature": str(text_temperature),
            "text_top_p": str(text_top_p),
            "text_top_k": str(text_top_k),
            "temperature": str(audio_temperature),
            "top_p": str(audio_top_p),
            "top_k": str(audio_top_k),
            "repetition_penalty": str(audio_repetition_penalty),
            "seed": str(args.seed),
        }
        for key, value in cpp_request_options.items():
            cpp_command.extend(["--request-option", f"{key}={value}"])
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    noise_file = scenario_dir / "shared_noise.bin"
    voice_args = ["--clone-audio", args.clone_audio] if args.clone_audio else ["--voice-id", config["voice_id"]]
    python_command = [
        PYTHON_EXE,
        str(REPO_ROOT / config["python_script"]),
        "--mode",
        "parity",
        "--model",
        config["model"],
        "--language",
        config["language"],
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--noise-file",
        str(noise_file),
        "--audio-out",
        str(audio_py),
        "--audio-out-dir",
        str(audio_py_dir),
        "--timing-file",
        str(timing_py),
    ] + voice_args + common_warmup_args + common_text_args
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
        "--mode",
        "parity",
        "--model",
        config["model"],
        "--language",
        config["language"],
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--noise-file",
        str(noise_file),
        "--audio-out",
        str(audio_cpp),
        "--audio-out-dir",
        str(audio_cpp_dir),
        "--timing-file",
        str(timing_cpp),
    ] + voice_args + common_warmup_args + common_text_args
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_ace_step_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    noise_file = args.test_noise_file or ""
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--checkpoint-dir",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        # ACE-Step planner parity is sensitive to LM precision on CUDA, so the
        # warmbench default pins the Python LM to fp32.
        "--lm-dtype",
        "float32",
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in args.cpp_session_option:
        if option.startswith("ace_step.dit_model_path="):
            python_command.extend(["--config-path", option.split("=", 1)[1]])
        elif option.startswith("ace_step.lm_model_path="):
            python_command.extend(["--lm-model-path", option.split("=", 1)[1]])
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        # Keep the full ACE-Step GGML stack on fp32 by default for parity
        # runs, including the VAE. The C++ VAE decoder now chunks long
        # latent sequences so warmbench does not need a native-weight escape
        # hatch that would hide real fp32 issues.
        "--session-option",
        "ace_step.weight_type=f32",
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    has_tensor_debug = any(option.startswith("ace_step.dit_tensor_") for option in args.cpp_session_option)
    has_tensor_compare_result = any(
        option.startswith("ace_step.dit_tensor_compare_result=")
        for option in args.cpp_session_option
    )
    if has_tensor_debug and not has_tensor_compare_result:
        cpp_command.extend([
            "--session-option",
            f"ace_step.dit_tensor_compare_result={scenario_dir / 'ace_step_dit_compare.json'}",
        ])
    return python_command, cpp_command


def build_vevo2_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    cpp_model_path = args.model or config["model"]
    python_model_path = args.model or config.get("python_model", config["model"])
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    noise_file = args.test_noise_file or ""
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        python_model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        cpp_model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_audio_commands(
    family: str,
    config: dict[str, Any],
    backend: str,
    mode: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    warmup_audio: Path,
    requests: list[Path],
) -> tuple[list[str], list[str]]:
    audio_sequence = ",".join(str(path) for path in requests)
    first_audio = str(requests[0])
    if family == "qwen3_asr":
        request_languages = getattr(args, "qwen3_asr_request_languages", [])
        request_contexts = getattr(args, "qwen3_asr_request_contexts", [])
        warmup_language = getattr(args, "qwen3_asr_warmup_language", "English")
        warmup_context = getattr(args, "qwen3_asr_warmup_context", "")
        max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else config.get("max_new_tokens", 512)
        common = [
            "--model",
            args.model or config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--max-new-tokens",
            str(max_new_tokens),
            "--language",
            "English",
            "--warmup-language",
            warmup_language,
            "--warmup-context",
            warmup_context,
            "--language-sequence",
            ",".join(request_languages),
            "--context-sequence",
            ",".join(request_contexts),
        ]
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--timing-file",
            str(scenario_dir / "python.timing.log"),
        ] + common
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--timing-file",
            str(scenario_dir / "cpp.timing.log"),
        ] + common
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family == "qwen3_forced_aligner":
        request_languages = getattr(args, "qwen3_forced_aligner_request_languages", [])
        request_transcripts = getattr(args, "qwen3_forced_aligner_request_transcripts", [])
        warmup_language = getattr(args, "qwen3_forced_aligner_warmup_language", "English")
        warmup_transcript = getattr(args, "qwen3_forced_aligner_warmup_transcript", "")
        common = [
            "--model",
            args.model or config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--language",
            "English",
            "--warmup-language",
            warmup_language,
            "--warmup-transcript",
            warmup_transcript,
        ]
        request_args: list[str] = []
        for language in request_languages:
            request_args.extend(["--request-language", language])
        for transcript in request_transcripts:
            request_args.extend(["--request-transcript", transcript])
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--timing-file",
            str(scenario_dir / "python.timing.log"),
        ] + common + request_args
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--timing-file",
            str(scenario_dir / "cpp.timing.log"),
        ] + common + request_args
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if family == "parakeet":
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--run-mode",
            mode,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--run-mode",
            mode,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--timing-file",
            str(scenario_dir / "cpp.timing.log"),
        ]
        if mode == "longform":
            cpp_command.extend(["--encoder-variant", "long_context"])
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    if config["kind"] in {"asr", "vad", "spk"}:
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
        ]
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            config["model"],
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--timing-file",
            str(scenario_dir / "cpp.timing.log"),
        ]
        return python_command, cpp_command

    if config["kind"] == "separation":
        python_command = [
            PYTHON_EXE,
            str(REPO_ROOT / config["python_script"]),
            "--model",
            config.get("python_model", args.model or config["model"]),
            "--family",
            family,
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--timing-file",
            str(scenario_dir / "python.timing.log"),
            "--output-dir",
            str(scenario_dir / "python_audio"),
        ]
        python_config = str(config.get("python_config", ""))
        if python_config:
            python_command.extend(["--config-path", python_config])
        python_repo = str(config.get("python_repo", ""))
        if python_repo:
            python_command.extend(["--repo", python_repo])
        python_name = str(config.get("python_name", ""))
        if python_name:
            python_command.extend(["--model-name", python_name])
        cpp_command = [
            str(REPO_ROOT / config["cpp_bin"]),
            "--model",
            args.model or config["model"],
            "--family",
            family,
            "--audio",
            first_audio,
            "--warmup-audio",
            str(warmup_audio),
            "--audio-sequence",
            audio_sequence,
            "--backend",
            backend,
            "--device",
            str(args.device),
            "--threads",
            str(args.threads),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--timing-file",
            str(scenario_dir / "cpp.timing.log"),
            "--output-dir",
            str(scenario_dir / "cpp_audio"),
        ]
        for option in args.cpp_session_option:
            cpp_command.extend(["--session-option", option])
        return python_command, cpp_command

    python_command = [
        PYTHON_EXE,
        str(REPO_ROOT / config["python_script"]),
        "--model",
        config["python_model"],
        "--audio",
        first_audio,
        "--warmup-audio",
        str(warmup_audio),
        "--audio-sequence",
        audio_sequence,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        config["model"],
        "--audio",
        first_audio,
        "--warmup-audio",
        str(warmup_audio),
        "--audio-sequence",
        audio_sequence,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
    ]
    return python_command, cpp_command


def catalog_case_audio(case: dict[str, Any], scenario_dir: Path, role: str, index: int) -> Path:
    source = REPO_ROOT / str(case["audio"])
    if "start_sec" not in case and "end_sec" not in case:
        return source
    return materialize_qwen3_asr_audio(case, scenario_dir / "catalog_audio", index, role)


def csv_values(cases: list[dict[str, Any]], key: str, default: Any = "") -> str:
    return ",".join(str(case.get(key, default)) for case in cases)


def csv_bools(cases: list[dict[str, Any]], key: str, default: bool) -> str:
    return ",".join("true" if bool(case.get(key, default)) else "false" for case in cases)


def catalog_asr_model_path(config: dict[str, Any], args: argparse.Namespace) -> str:
    return args.model or str(config["model"])


def build_catalog_asr_commands(
    family: str,
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    warmup_case: dict[str, Any],
    request_cases: list[dict[str, Any]],
) -> tuple[list[str], list[str], dict[str, Any]]:
    warmup_audio = catalog_case_audio(warmup_case, scenario_dir, "warmup", 0)
    audio_requests = [
        catalog_case_audio(case, scenario_dir, "request", index)
        for index, case in enumerate(request_cases)
    ]
    model_path = catalog_asr_model_path(config, args)
    common = [
        "--model",
        model_path,
        "--audio",
        str(audio_requests[0]),
        "--warmup-audio",
        str(warmup_audio),
        "--audio-sequence",
        ",".join(str(path) for path in audio_requests),
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    python_extra: list[str] = []
    model_parent = Path(model_path).parent
    if family == "higgs_audio_stt":
        common.extend([
            "--prompt",
            str(warmup_case.get("prompt", "")),
            "--prompt-sequence",
            csv_values(request_cases, "prompt", warmup_case.get("prompt", "")),
            "--max-tokens",
            str(warmup_case.get("max_tokens", 1024)),
            "--max-tokens-sequence",
            csv_values(request_cases, "max_tokens", warmup_case.get("max_tokens", 1024)),
            "--enable-thinking",
            "true" if bool(warmup_case.get("enable_thinking", True)) else "false",
            "--enable-thinking-sequence",
            csv_bools(request_cases, "enable_thinking", bool(warmup_case.get("enable_thinking", True))),
        ])
        python_extra.extend(["--whisper-processor", str(model_parent / "whisper-large-v3")])
    elif family == "hviske_asr":
        common.extend([
            "--language",
            str(request_cases[0].get("language", warmup_case.get("language", "da"))),
            "--warmup-language",
            str(warmup_case.get("language", request_cases[0].get("language", "da"))),
            "--language-sequence",
            csv_values(request_cases, "language", "da"),
            "--punctuation",
            "true" if bool(warmup_case.get("punctuation", True)) else "false",
            "--max-tokens",
            str(warmup_case.get("max_tokens", 256)),
            "--max-tokens-sequence",
            csv_values(request_cases, "max_tokens", warmup_case.get("max_tokens", 256)),
            "--num-beams",
            str(warmup_case.get("num_beams", 1)),
            "--num-beams-sequence",
            csv_values(request_cases, "num_beams", warmup_case.get("num_beams", 1)),
            "--length-penalty",
            str(warmup_case.get("length_penalty", 1.0)),
            "--length-penalty-sequence",
            csv_values(request_cases, "length_penalty", warmup_case.get("length_penalty", 1.0)),
            "--do-sample",
            "true" if bool(warmup_case.get("do_sample", False)) else "false",
            "--do-sample-sequence",
            csv_bools(request_cases, "do_sample", bool(warmup_case.get("do_sample", False))),
            "--temperature",
            str(warmup_case.get("temperature", 1.0)),
            "--temperature-sequence",
            csv_values(request_cases, "temperature", warmup_case.get("temperature", 1.0)),
            "--top-k",
            str(warmup_case.get("top_k", 50)),
            "--top-k-sequence",
            csv_values(request_cases, "top_k", warmup_case.get("top_k", 50)),
            "--top-p",
            str(warmup_case.get("top_p", 1.0)),
            "--top-p-sequence",
            csv_values(request_cases, "top_p", warmup_case.get("top_p", 1.0)),
            "--seed",
            str(warmup_case.get("seed", args.seed)),
            "--seed-sequence",
            csv_values(request_cases, "seed", args.seed),
        ])
    elif family == "nemotron_asr":
        common.extend([
            "--language",
            str(request_cases[0].get("language", warmup_case.get("language", "en-US"))),
            "--warmup-language",
            str(warmup_case.get("language", request_cases[0].get("language", "en-US"))),
            "--language-sequence",
            csv_values(request_cases, "language", "en-US"),
            "--lookahead-tokens",
            str(warmup_case.get("lookahead_tokens", 3)),
            "--lookahead-tokens-sequence",
            csv_values(request_cases, "lookahead_tokens", warmup_case.get("lookahead_tokens", 3)),
            "--max-tokens",
            str(warmup_case.get("max_tokens", 256)),
            "--max-tokens-sequence",
            csv_values(request_cases, "max_tokens", warmup_case.get("max_tokens", 256)),
            "--streaming",
            "true" if bool(warmup_case.get("streaming", False)) else "false",
            "--streaming-sequence",
            csv_bools(request_cases, "streaming", bool(warmup_case.get("streaming", False))),
            "--keep-language-tags",
            "true" if bool(warmup_case.get("keep_language_tags", False)) else "false",
            "--keep-language-tags-sequence",
            csv_bools(request_cases, "keep_language_tags", bool(warmup_case.get("keep_language_tags", False))),
        ])
    elif family == "vibevoice_asr":
        common.extend([
            "--context",
            str(warmup_case.get("context", "")),
            "--context-sequence",
            csv_values(request_cases, "context", warmup_case.get("context", "")),
            "--max-tokens",
            str(warmup_case.get("max_tokens", 32768)),
            "--max-tokens-sequence",
            csv_values(request_cases, "max_tokens", warmup_case.get("max_tokens", 32768)),
            "--temperature",
            str(warmup_case.get("temperature", 0.0)),
            "--temperature-sequence",
            csv_values(request_cases, "temperature", warmup_case.get("temperature", 0.0)),
            "--top-p",
            str(warmup_case.get("top_p", 1.0)),
            "--top-p-sequence",
            csv_values(request_cases, "top_p", warmup_case.get("top_p", 1.0)),
            "--top-k",
            str(warmup_case.get("top_k", 50)),
            "--top-k-sequence",
            csv_values(request_cases, "top_k", warmup_case.get("top_k", 50)),
            "--num-beams",
            str(warmup_case.get("num_beams", 1)),
            "--num-beams-sequence",
            csv_values(request_cases, "num_beams", warmup_case.get("num_beams", 1)),
            "--seed",
            str(warmup_case.get("seed", args.seed)),
            "--seed-sequence",
            csv_values(request_cases, "seed", args.seed),
        ])
        python_extra.extend(["--tokenizer", model_path])
    else:
        raise RuntimeError(f"catalog ASR warmbench is not wired for family {family}")

    python_command = [
        PYTHON_EXE,
        str(REPO_ROOT / config["python_script"]),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
    ] + python_extra + common
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
    ] + common
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    request_manifest = {
        "warmup_audio": str(warmup_audio),
        "audio_sequence": [str(path) for path in audio_requests],
        "warmup": warmup_case,
        "requests": request_cases,
    }
    return python_command, cpp_command, request_manifest


def build_miocodec_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    warmup_request: dict[str, Any],
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False)
    warmup_request_json = json.dumps(warmup_request, ensure_ascii=False)
    common = [
        "--model",
        args.model or config["model"],
        "--request-sequence-json",
        request_sequence_json,
        "--warmup-request-json",
        warmup_request_json,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
    ]
    python_command = [
        PYTHON_EXE,
        str(REPO_ROOT / config["python_script"]),
    ] + common + [
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
    ]
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
    ] + common + [
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
    ]
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_voxcpm2_commands(
    config: dict[str, Any],
    mode: str,
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    warmup_request: dict[str, Any],
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False)
    warmup_request_json = json.dumps(warmup_request, ensure_ascii=False)
    common = [
        "--model",
        args.model or config["model"],
        "--request-sequence-json",
        request_sequence_json,
        "--warmup-request-json",
        warmup_request_json,
        "--case-name",
        args.case_names[0] if args.case_names else str(config.get("default_case_name", "default")),
        "--run-mode",
        mode,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
    ]
    python_command = [
        PYTHON_EXE,
        str(REPO_ROOT / config["python_script"]),
    ] + common + [
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
    ]
    if args.test_noise_file:
        python_command.extend(["--noise-file", args.test_noise_file])
    cpp_command = [
        str(REPO_ROOT / config["cpp_bin"]),
    ] + common + [
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
    ]
    if args.test_noise_file:
        cpp_command.extend(["--noise-file", args.test_noise_file])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_vibevoice_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    model_root = Path(model_path)
    if not model_root.is_absolute():
        model_root = REPO_ROOT / model_root
    model_config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
    latent_size = int(model_config["acoustic_vae_dim"])
    decoder_capacity = int(model_config["decoder_config"]["max_position_embeddings"])
    total_speakers = sum(max(1, len(request.get("voice_samples", []))) for request in requests)
    prompt_noise_count = total_speakers + total_speakers * decoder_capacity * latent_size
    prompt_noise_path = scenario_dir / "shared_prompt_noise.f32"
    prompt_rng = random.Random(args.seed + 17)
    prompt_values = array.array("f", (prompt_rng.gauss(0.0, 1.0) for _ in range(prompt_noise_count)))
    prompt_noise_path.parent.mkdir(parents=True, exist_ok=True)
    with prompt_noise_path.open("wb") as handle:
        prompt_values.tofile(handle)
    prompt_noise_file = str(prompt_noise_path)
    noise_file = args.test_noise_file
    if not noise_file:
        max_new_tokens = max([int(request.get("max_new_tokens", 0)) for request in requests] + [0])
        if max_new_tokens <= 0:
            max_new_tokens = decoder_capacity
        noise_count = max_new_tokens * 2 * len(requests) * latent_size
        noise_path = scenario_dir / "shared_diffusion_noise.f32"
        rng = random.Random(args.seed)
        values = array.array("f", (rng.gauss(0.0, 1.0) for _ in range(noise_count)))
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        with noise_path.open("wb") as handle:
            values.tofile(handle)
        noise_file = str(noise_path)
    warmup_count = 0
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(warmup_count),
        "--iterations",
        str(args.iterations),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--prompt-noise-file",
        prompt_noise_file,
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    if any(bool(request.get("batch")) for request in requests):
        python_command.append("--batch")
    cpp_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(warmup_count),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--log-file",
        str(scenario_dir / "cpp.trace.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--prompt-noise-file",
        prompt_noise_file,
        "--noise-file",
        noise_file,
        "--request-sequence-json",
        request_sequence_json,
    ]
    if any(bool(request.get("batch")) for request in requests):
        cpp_command.append("--batch")
    return python_command, cpp_command


def build_irodori_tts_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if backend != "cuda":
        raise RuntimeError("Irodori-TTS warmbench is CUDA-only")
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_model_path = args.python_model or config.get("python_model", model_path)
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        python_model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--request-sequence-json",
        request_sequence_json,
        "--model-precision",
        "fp32",
        "--codec-precision",
        "fp32",
    ]
    cpp_command = [
        "env",
        "NVIDIA_TF32_OVERRIDE=0",
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_supertonic_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_model_path = args.python_model or config.get("python_model", model_path)
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        python_model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    cpp_command = [
        "env",
        "NVIDIA_TF32_OVERRIDE=0",
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_index_tts2_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_model_path = args.python_model or config.get("python_model", model_path)
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        python_model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--audio-out-dir",
        str(scenario_dir / "python_audio"),
        "--summary-file",
        str(scenario_dir / "python.summary.json"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    cpp_command = [
        "env",
        "NVIDIA_TF32_OVERRIDE=0",
        "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_heartmula_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if backend != "cuda":
        raise RuntimeError("HeartMuLa warmbench is CUDA-only")
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--request-sequence-json",
        request_sequence_json,
        "--mula-dtype",
        "float32",
        "--codec-dtype",
        "float32",
        "--lazy-load",
    ]
    cpp_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def build_higgs_tts_commands(
    config: dict[str, Any],
    backend: str,
    args: argparse.Namespace,
    scenario_dir: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    if backend != "cuda":
        raise RuntimeError("Higgs TTS warmbench is CUDA-only")
    request_sequence_json = json.dumps(requests, ensure_ascii=False, separators=(",", ":"))
    model_path = args.model or config["model"]
    python_env = str(config.get("python_conda_env", "qwen3-tts"))
    python_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        "python",
        str(REPO_ROOT / config["python_script"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "python.timing.log"),
        "--output-dir",
        str(scenario_dir / "python_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    cpp_command = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        python_env,
        str(REPO_ROOT / config["cpp_bin"]),
        "--model",
        model_path,
        "--backend",
        backend,
        "--device",
        str(args.device),
        "--threads",
        str(args.threads),
        "--warmup",
        str(effective_warmup(config, args)),
        "--iterations",
        str(args.iterations),
        "--timing-file",
        str(scenario_dir / "cpp.timing.log"),
        "--output-dir",
        str(scenario_dir / "cpp_audio"),
        "--request-sequence-json",
        request_sequence_json,
    ]
    if os.environ.get("HIGGS_TTS_WARMBENCH_TRACE") == "1":
        python_command.append("--enable-trace")
        cpp_command.extend(["--enable-trace", "--log-file", str(scenario_dir / "cpp.trace.log")])
    for option in config.get("cpp_session_options", []):
        cpp_command.extend(["--session-option", option])
    for option in args.cpp_session_option:
        cpp_command.extend(["--session-option", option])
    return python_command, cpp_command


def validate_tts_result(
    parsed: dict[str, Any],
    request_count: int,
    min_rms: float = 0.0,
) -> dict[str, Any]:
    summaries = parsed.get("summaries", [])
    audio_path = resolve_repo_path(parsed.get("audio_out"))
    indexed_audio_outs = parsed.get("audio_outs", {})
    summary_valid = len(summaries) == request_count and all(
        int(item.get("samples", 0)) > 0 and float(item.get("rms", 0.0)) >= min_rms
        for item in summaries
    )
    per_request_outputs: list[dict[str, Any]] = []
    for request_index in range(request_count):
        request_audio_path = resolve_repo_path(indexed_audio_outs.get(request_index))
        summary_rms = float(summaries[request_index].get("rms", 0.0)) if request_index < len(summaries) else 0.0
        request_valid = (
            request_audio_path is not None and
            request_audio_path.exists() and
            request_audio_path.stat().st_size > 44 and
            summary_rms >= min_rms
        )
        per_request_outputs.append({
            "request_index": request_index,
            "path": str(request_audio_path) if request_audio_path is not None else "",
            "ok": request_valid,
            "rms": summary_rms,
        })
    audio_valid = bool(per_request_outputs) and all(item["ok"] for item in per_request_outputs)
    final_audio_valid = audio_path is not None and audio_path.exists() and audio_path.stat().st_size > 44
    return {
        "ok": summary_valid and audio_valid and final_audio_valid,
        "summary_valid": summary_valid,
        "audio_valid": audio_valid,
        "final_audio_valid": final_audio_valid,
        "audio_path": str(audio_path) if audio_path is not None else "",
        "per_request_outputs": per_request_outputs,
    }


def validate_sequence_result(summary: dict[str, Any], request_count: int, kind: str) -> dict[str, Any]:
    steps = summary.get("sequence_steps", []) if isinstance(summary, dict) else []
    count_valid = len(steps) == request_count
    if kind == "asr":
        payload_valid = all(
            isinstance(step.get("text_output", ""), str)
            and isinstance(step.get("word_timestamps", []), list)
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind == "alignment":
        payload_valid = all(
            isinstance(step.get("text_output", ""), str)
            and isinstance(step.get("word_timestamps", []), list)
            and len(step.get("word_timestamps", [])) > 0
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind == "vad":
        payload_valid = all(
            isinstance(step.get("speech_segments", []), list)
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind == "spk":
        payload_valid = all(
            isinstance(step.get("label", ""), str)
            and isinstance(step.get("index", None), (int, float))
            and isinstance(step.get("score", None), (int, float))
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind == "separation":
        payload_valid = all(
            isinstance(step.get("stems", []), list)
            and len(step.get("stems", [])) > 0
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind == "ace_step":
        payload_valid = all(
            isinstance(step.get("stems", []), list)
            and len(step.get("stems", [])) > 0
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    elif kind in {"vevo2", "seed_vc", "miocodec", "voxcpm2", "supertonic", "vibevoice", "irodori_tts", "heartmula", "higgs_tts", "index_tts2"}:
        payload_valid = all(
            isinstance(step.get("stems", []), list)
            and len(step.get("stems", [])) > 0
            and isinstance(step.get("metrics", {}), dict)
            for step in steps)
    else:
        payload_valid = all(isinstance(step.get("speaker_turns", []), list) and len(step.get("speaker_turns", [])) > 0 for step in steps)
    return {"ok": count_valid and payload_valid, "count_valid": count_valid, "payload_valid": payload_valid}


def write_sequence_step_artifacts(steps: list[dict[str, Any]], output_dir: Path, prefix: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for request_index, step in enumerate(steps):
        path = output_dir / f"{prefix}_request_{request_index:02d}.json"
        path.write_text(json.dumps(step, indent=2, ensure_ascii=False), encoding="utf-8")
        paths.append(str(path))
    return paths


def file_is_nonempty(path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    return path.exists() and path.stat().st_size > 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_frame_count(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return int(handle.getnframes())


def first_sequence_audio(summary: dict[str, Any]) -> Path:
    steps = summary.get("sequence_steps", [])
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("ACE-Step hash compare requires at least one C++ sequence step")
    stems = steps[0].get("stems", [])
    if not isinstance(stems, list) or not stems:
        raise RuntimeError("ACE-Step hash compare requires at least one C++ audio stem")
    audio_path = resolve_repo_path(str(stems[0].get("audio", "")))
    if audio_path is None:
        raise RuntimeError("ACE-Step hash compare C++ audio path is missing")
    return audio_path


def ace_step_baseline_hash_compare(cpp_summary: dict[str, Any]) -> dict[str, Any]:
    current_audio = first_sequence_audio(cpp_summary)
    if not ACE_STEP_BASELINE_CPP_AUDIO.exists():
        raise RuntimeError(f"ACE-Step baseline audio is missing: {ACE_STEP_BASELINE_CPP_AUDIO}")
    if not current_audio.exists():
        raise RuntimeError(f"ACE-Step C++ audio is missing: {current_audio}")
    baseline_sha256 = sha256_file(ACE_STEP_BASELINE_CPP_AUDIO)
    current_sha256 = sha256_file(current_audio)
    baseline_frames = wav_frame_count(ACE_STEP_BASELINE_CPP_AUDIO)
    current_frames = wav_frame_count(current_audio)
    comparable = baseline_frames == current_frames
    return {
        "baseline_path": str(ACE_STEP_BASELINE_CPP_AUDIO),
        "current_path": str(current_audio),
        "baseline_frames": baseline_frames,
        "current_frames": current_frames,
        "comparable": comparable,
        "baseline_sha256": baseline_sha256,
        "current_sha256": current_sha256,
        "match": baseline_sha256 == current_sha256,
    }


def sanitized_result_payload(kind: str, parsed: dict[str, Any]) -> dict[str, Any]:
    if kind == "tts":
        return {
            "summaries": parsed.get("summaries", []),
            "warmup_summaries": parsed.get("warmup_summaries", []),
            "texts": parsed.get("texts", []),
            "warmup_texts": parsed.get("warmup_texts", []),
            "audio_out": parsed.get("audio_out"),
            "audio_outs": parsed.get("audio_outs", {}),
            "metrics": parsed.get("metrics", {}),
            "request_metrics": parsed.get("request_metrics", []),
        }
    return parsed.get("summary") or {}


def missing_parity(request_index: int, request: str, reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "reason": reason,
        "mismatches": [reason],
        "request_index": request_index,
        "request": request,
    }


def run_scenario(
    family: str,
    config: dict[str, Any],
    mode: str,
    backend: str,
    args: argparse.Namespace,
    root_output_dir: Path,
    master_log: Path,
) -> dict[str, Any]:
    scenario_config = dict(config)
    original_requests_per_session = args.requests_per_session
    if (
        not getattr(args, "requests_per_session_was_explicit", False)
        and "default_requests_per_session" in scenario_config
    ):
        args.requests_per_session = int(scenario_config["default_requests_per_session"])
    scenario_name = f"{family}_{mode}_{backend}"
    scenario_dir = root_output_dir / scenario_name
    if scenario_dir.exists() and not args.keep_output_dir:
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    append_log(master_log, f"SCENARIO START family={family} mode={mode} backend={backend} dir={scenario_dir}")

    if scenario_config["kind"] == "ace_step":
        ace_requests, request_manifest = resolve_ace_step_case(config, args)
        python_command, cpp_command = build_ace_step_commands(scenario_config, backend, args, scenario_dir, ace_requests)
    elif scenario_config["kind"] == "vibevoice":
        vibevoice_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(vibevoice_requests)
        python_command, cpp_command = build_vibevoice_commands(scenario_config, backend, args, scenario_dir, vibevoice_requests)
    elif scenario_config["kind"] == "heartmula":
        heartmula_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(heartmula_requests)
        python_command, cpp_command = build_heartmula_commands(scenario_config, backend, args, scenario_dir, heartmula_requests)
    elif scenario_config["kind"] == "supertonic":
        supertonic_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(supertonic_requests)
        python_command, cpp_command = build_supertonic_commands(scenario_config, backend, args, scenario_dir, supertonic_requests)
    elif scenario_config["kind"] == "irodori_tts":
        irodori_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(irodori_requests)
        python_command, cpp_command = build_irodori_tts_commands(scenario_config, backend, args, scenario_dir, irodori_requests)
    elif scenario_config["kind"] == "higgs_tts":
        higgs_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(higgs_requests)
        python_command, cpp_command = build_higgs_tts_commands(scenario_config, backend, args, scenario_dir, higgs_requests)
    elif scenario_config["kind"] == "index_tts2":
        index_tts2_requests, request_manifest = resolve_vevo2_case(config, args)
        args.requests_per_session = len(index_tts2_requests)
        python_command, cpp_command = build_index_tts2_commands(scenario_config, backend, args, scenario_dir, index_tts2_requests)
    elif scenario_config["kind"] in {"vevo2", "seed_vc"}:
        vevo2_requests, request_manifest = resolve_vevo2_case(config, args)
        python_command, cpp_command = build_vevo2_commands(scenario_config, backend, args, scenario_dir, vevo2_requests)
    elif scenario_config["kind"] == "tts":
        if family.startswith("omnivoice_"):
            texts, request_manifest, scenario_config = resolve_omnivoice_case(config, args)
        else:
            texts, request_manifest = resolve_tts_texts(family, config, args, mode)
        if family == "chatterbox" and not args.warmup_text and request_manifest.get("warmup_text"):
            scenario_config["warmup_text"] = request_manifest["warmup_text"]
        if family == "moss_tts_nano" and isinstance(request_manifest.get("options"), dict):
            scenario_config.update(request_manifest["options"])
        if family in {"qwen3_tts_voice_design", "qwen3_tts_custom_voice"}:
            default_instruct = scenario_config.get("voice_design_instruct", scenario_config.get("custom_voice_instruct", ""))
            request_manifest["warmup_instruct"] = args.warmup_voice_design_instruct or default_instruct
            request_manifest["instructs"] = (
                args.voice_design_instructs[: len(texts)]
                if args.voice_design_instructs
                else [default_instruct] * len(texts)
            )
        if family.startswith("omnivoice_") and scenario_config.get("task") == "voice_design":
            warmup_instruct, request_instructs = omnivoice_instruct_values(scenario_config, args, len(texts))
            request_manifest["warmup_instruct"] = warmup_instruct
            request_manifest["instructs"] = request_instructs
        if family == "qwen3_tts_custom_voice":
            speaker = args.speaker or scenario_config["speaker"]
            request_manifest["warmup_speaker"] = args.warmup_speaker or speaker
            request_manifest["speakers"] = (
                args.request_speakers[: len(texts)]
                if args.request_speakers
                else [speaker] * len(texts)
            )
        if family == "moss_tts_local" and "requests" in request_manifest:
            request_file = scenario_dir / "moss_tts_local_requests.json"
            request_file.write_text(
                json.dumps({"requests": request_manifest["requests"]}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            scenario_config["moss_tts_local_request_file"] = str(request_file)
        python_command, cpp_command = build_tts_commands(family, scenario_config, backend, args, scenario_dir, texts)
    else:
        if family in {"mel_band_roformer", "htdemucs"}:
            if len(args.case_names) > 1:
                raise RuntimeError(f"{family} warmbench accepts at most one --case-name")
            case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", "default"))
            family_label = "RoFormer" if family == "mel_band_roformer" else "HTDemucs"
            warmup_case, request_cases = load_separation_cases(REPO_ROOT / config["case_catalog"], case_name, args.requests_per_session, family_label)
            warmup_audio = REPO_ROOT / warmup_case if not warmup_case.is_absolute() else warmup_case
            audio_requests = [
                REPO_ROOT / item if not item.is_absolute() else item
                for item in request_cases
            ]
            request_manifest = {
                "case_name": case_name,
                "warmup_audio": str(warmup_audio),
                "audio_sequence": [str(path) for path in audio_requests],
            }
        elif family == "qwen3_asr":
            warmup_case, request_cases = load_qwen3_asr_cases(REPO_ROOT / config["case_catalog"], args.requests_per_session)
            qwen_audio_dir = scenario_dir / "qwen3_asr_audio"
            warmup_audio = materialize_qwen3_asr_audio(warmup_case, qwen_audio_dir, 0, "warmup")
            audio_requests = [
                materialize_qwen3_asr_audio(item, qwen_audio_dir, index, "request")
                for index, item in enumerate(request_cases)
            ]
            args.qwen3_asr_warmup_language = str(warmup_case.get("language", "English"))
            args.qwen3_asr_warmup_context = str(warmup_case.get("context", ""))
            args.qwen3_asr_request_languages = [str(item.get("language", "English")) for item in request_cases]
            args.qwen3_asr_request_contexts = [str(item.get("context", "")) for item in request_cases]
            request_manifest = {
                "warmup_audio": str(warmup_audio),
                "warmup_language": args.qwen3_asr_warmup_language,
                "warmup_context": args.qwen3_asr_warmup_context,
                "audio_sequence": [str(path) for path in audio_requests],
                "languages": args.qwen3_asr_request_languages,
                "contexts": args.qwen3_asr_request_contexts,
                "expected_fragments": [item.get("expected_fragments", []) for item in request_cases],
            }
        elif family == "qwen3_forced_aligner":
            warmup_case, request_cases = load_qwen3_forced_aligner_cases(REPO_ROOT / config["case_catalog"], args.requests_per_session)
            qwen_audio_dir = scenario_dir / "qwen3_forced_aligner_audio"
            warmup_audio = materialize_qwen3_asr_audio(warmup_case, qwen_audio_dir, 0, "warmup")
            audio_requests = [
                materialize_qwen3_asr_audio(item, qwen_audio_dir, index, "request")
                for index, item in enumerate(request_cases)
            ]
            args.qwen3_forced_aligner_warmup_language = str(warmup_case.get("language", "English"))
            args.qwen3_forced_aligner_warmup_transcript = str(warmup_case.get("transcript", ""))
            args.qwen3_forced_aligner_request_languages = [str(item.get("language", "English")) for item in request_cases]
            args.qwen3_forced_aligner_request_transcripts = [str(item.get("transcript", "")) for item in request_cases]
            request_manifest = {
                "warmup_audio": str(warmup_audio),
                "warmup_language": args.qwen3_forced_aligner_warmup_language,
                "warmup_transcript": args.qwen3_forced_aligner_warmup_transcript,
                "audio_sequence": [str(path) for path in audio_requests],
                "languages": args.qwen3_forced_aligner_request_languages,
                "transcripts": args.qwen3_forced_aligner_request_transcripts,
                "expected_words": [item.get("expected_words", []) for item in request_cases],
            }
        elif family in {"higgs_audio_stt", "hviske_asr", "nemotron_asr", "vibevoice_asr"}:
            if len(args.case_names) > 1:
                raise RuntimeError(f"{family} warmbench accepts at most one --case-name")
            case_name = args.case_names[0] if args.case_names else str(config.get("default_case_name", ""))
            warmup_case, request_cases, catalog_manifest = load_catalog_asr_cases(
                REPO_ROOT / str(config["case_catalog"]),
                args.requests_per_session,
                family,
                case_name,
            )
            python_command, cpp_command, request_manifest = build_catalog_asr_commands(
                family,
                scenario_config,
                backend,
                args,
                scenario_dir,
                warmup_case,
                request_cases,
            )
            request_manifest.update(catalog_manifest)
        elif family == "miocodec":
            warmup_request, request_cases, request_manifest = resolve_miocodec_case(config, args)
            python_command, cpp_command = build_miocodec_commands(
                scenario_config,
                backend,
                args,
                scenario_dir,
                warmup_request,
                request_cases,
            )
        elif family == "voxcpm2":
            warmup_request, request_cases, request_manifest = resolve_voxcpm2_case(config, args)
            python_command, cpp_command = build_voxcpm2_commands(
                scenario_config,
                mode,
                backend,
                args,
                scenario_dir,
                warmup_request,
                request_cases,
            )
        else:
            request_dir = scenario_dir / "requests"
            warmup_audio, audio_requests = build_audio_requests(
                request_dir,
                args.requests_per_session,
                scenario_seed(args.seed, family, mode),
                parse_audio_durations(args.audio_durations),
                args.audio_warmup_duration,
            )
            request_manifest = {
                "warmup_audio": str(warmup_audio),
                "audio_sequence": [str(path) for path in audio_requests],
            }
        if family not in {"miocodec", "voxcpm2", "higgs_audio_stt", "hviske_asr", "nemotron_asr", "vibevoice_asr"}:
            python_command, cpp_command = build_audio_commands(family, scenario_config, backend, mode, args, scenario_dir, warmup_audio, audio_requests)
    (scenario_dir / "request_manifest.json").write_text(json.dumps(request_manifest, indent=2), encoding="utf-8")

    set_command_backend(python_command, python_reference_backend(backend))

    append_log(master_log, f"PYTHON START family={family} mode={mode} backend={backend} command={' '.join(python_command)}")
    python_env = os.environ.copy()
    python_env.pop("LD_LIBRARY_PATH", None)
    if scenario_config["kind"] == "ace_step":
        python_env["ACESTEP_PROJECT_ROOT"] = str(scenario_dir)
    python_stdout, python_memory = run_command(python_command, scenario_dir / "python.log", env=python_env)
    append_log(master_log, f"PYTHON END family={family} mode={mode} backend={backend}")
    append_log(
        master_log,
        f"PYTHON MEMORY family={family} mode={mode} backend={backend} "
        f"peak_rss_kb={python_memory['peak_rss_kb']} peak_vms_kb={python_memory['peak_vms_kb']}",
    )
    python_parsed = parse_summary_lines(python_stdout)
    python_result_payload = sanitized_result_payload(scenario_config["kind"], python_parsed)
    python_summary = python_parsed.get("summary") or {}
    python_summary_path = scenario_dir / "python.result.json"
    python_summary_path.write_text(json.dumps(python_result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(master_log, f"PYTHON RESULT family={family} mode={mode} backend={backend} path={python_summary_path}")

    append_log(master_log, f"CPP START family={family} mode={mode} backend={backend} command={' '.join(cpp_command)}")
    cpp_stdout, cpp_memory = run_command(cpp_command, scenario_dir / "cpp.log")
    append_log(master_log, f"CPP END family={family} mode={mode} backend={backend}")
    append_log(
        master_log,
        f"CPP MEMORY family={family} mode={mode} backend={backend} "
        f"peak_rss_kb={cpp_memory['peak_rss_kb']} peak_vms_kb={cpp_memory['peak_vms_kb']}",
    )
    cpp_parsed = parse_summary_lines(cpp_stdout)
    cpp_result_payload = sanitized_result_payload(scenario_config["kind"], cpp_parsed)
    cpp_summary = cpp_parsed.get("summary") or {}
    cpp_summary_path = scenario_dir / "cpp.result.json"
    cpp_summary_path.write_text(json.dumps(cpp_result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(master_log, f"CPP RESULT family={family} mode={mode} backend={backend} path={cpp_summary_path}")

    parity_results: list[dict[str, Any]] = []
    if scenario_config["kind"] == "tts":
        min_rms = float(scenario_config.get("min_rms", 0.0))
        python_valid = validate_tts_result(python_parsed, args.requests_per_session, min_rms)
        cpp_valid = validate_tts_result(cpp_parsed, args.requests_per_session, min_rms)
        waveform_compare_fn = compare_kokoro if family == "kokoro" else compare_pocket
        python_whisper_results: list[dict[str, Any]] = []
        cpp_whisper_results: list[dict[str, Any]] = []
        run_asr_check = bool(scenario_config.get("run_asr_check", True))
        for request_index in range(args.requests_per_session):
            python_output = python_valid["per_request_outputs"][request_index]
            cpp_output = cpp_valid["per_request_outputs"][request_index]
            append_log(master_log, f"PYTHON OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={python_output['path']} valid={int(python_output['ok'])}")
            append_log(master_log, f"CPP OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={cpp_output['path']} valid={int(cpp_output['ok'])}")
            if request_index >= len(python_parsed["summaries"]):
                parity = missing_parity(request_index, request_manifest["texts"][request_index], "missing_python_summary")
            elif request_index >= len(cpp_parsed["summaries"]):
                parity = missing_parity(request_index, request_manifest["texts"][request_index], "missing_cpp_summary")
            else:
                python_audio_path = resolve_repo_path(python_output["path"])
                cpp_audio_path = resolve_repo_path(cpp_output["path"])
                if python_audio_path is None or cpp_audio_path is None:
                    parity = missing_parity(request_index, request_manifest["texts"][request_index], "missing_audio_path")
                else:
                    if run_asr_check:
                        python_whisper = run_whisper(
                            python_audio_path,
                            scenario_dir / "whisper" / f"python_request_{request_index:02d}",
                            args.threads,
                            scenario_config.get("whisper_model", WHISPER_MODEL),
                            scenario_config.get("whisper_language"),
                            bool(scenario_config.get("whisper_word_timestamps", True)),
                        )
                        cpp_whisper = run_whisper(
                            cpp_audio_path,
                            scenario_dir / "whisper" / f"cpp_request_{request_index:02d}",
                            args.threads,
                            scenario_config.get("whisper_model", WHISPER_MODEL),
                            scenario_config.get("whisper_language"),
                            bool(scenario_config.get("whisper_word_timestamps", True)),
                        )
                        python_whisper_results.append(python_whisper)
                        cpp_whisper_results.append(cpp_whisper)
                        append_log(master_log, f"PYTHON ASR family={family} mode={mode} backend={backend} request={request_index} {summarize_tts_asr(python_whisper)}")
                        append_log(master_log, f"CPP ASR family={family} mode={mode} backend={backend} request={request_index} {summarize_tts_asr(cpp_whisper)}")
                        asr_parity = compare_whisper_outputs(
                            cpp_whisper,
                            python_whisper,
                            float(scenario_config.get("asr_compact_lcs_min", 0.0)),
                        )
                    else:
                        asr_parity = {
                            "ok": True,
                            "reason": "skipped",
                            "mismatches": [],
                            "text_distance": 0,
                            "compact_text_distance": 0,
                            "compact_lcs": 1.0,
                            "compact_lcs_min": float(scenario_config.get("asr_compact_lcs_min", 0.0)),
                            "timing_mismatches": [],
                        }
                        append_log(master_log, f"ASR SKIPPED family={family} mode={mode} backend={backend} request={request_index}")
                    if family in {"moss_tts_nano", "moss_tts_local"}:
                        waveform_parity = compare_moss_tts(
                            cpp_parsed["summaries"][request_index],
                            python_parsed["summaries"][request_index],
                            cpp_audio_path,
                            python_audio_path,
                            float(scenario_config.get("log_mel_cosine_min", 0.80)),
                            float(scenario_config.get("length_ratio_min", 0.98)),
                        )
                        if (
                            family == "moss_tts_local"
                            and asr_parity["mismatches"] == ["empty_text"]
                            and waveform_parity["ok"]
                        ):
                            asr_parity = {**asr_parity, "ok": True, "reason": "ok", "mismatches": []}
                        parity_ok = asr_parity["ok"] and waveform_parity["ok"]
                        parity_reason = "ok" if parity_ok else (
                            asr_parity["reason"] if not asr_parity["ok"] else waveform_parity["reason"]
                        )
                    else:
                        waveform_parity = waveform_compare_fn(cpp_parsed["summaries"][request_index], python_parsed["summaries"][request_index])
                        parity_ok = asr_parity["ok"]
                        parity_reason = asr_parity["reason"]
                    parity = {
                        "ok": parity_ok,
                        "reason": parity_reason,
                        "mismatches": asr_parity["mismatches"] + waveform_parity["mismatches"],
                        "asr": asr_parity,
                        "waveform": waveform_parity,
                    }
                parity["request_index"] = request_index
                parity["request"] = request_manifest["texts"][request_index]
            parity_results.append(parity)
            if "asr" in parity and "waveform" in parity:
                append_log(
                    master_log,
                    f"PARITY family={family} mode={mode} backend={backend} request={request_index} ok={int(parity['ok'])} reason={parity['reason']} asr_reason={parity['asr']['reason']} waveform_reason={parity['waveform']['reason']}",
                )
            elif "waveform" in parity:
                append_log(
                    master_log,
                    f"PARITY family={family} mode={mode} backend={backend} request={request_index} ok={int(parity['ok'])} reason={parity['reason']} metrics={json.dumps(parity['waveform'].get('metrics', {}), sort_keys=True)}",
                )
            else:
                append_log(master_log, f"PARITY family={family} mode={mode} backend={backend} request={request_index} ok={int(parity['ok'])} reason={parity['reason']}")
        python_result_payload["whisper"] = python_whisper_results
        cpp_result_payload["whisper"] = cpp_whisper_results
    elif scenario_config["kind"] == "ace_step":
        python_valid = validate_sequence_result(python_summary, args.requests_per_session, "ace_step")
        cpp_valid = validate_sequence_result(cpp_summary, args.requests_per_session, "ace_step")
        hash_compare = ace_step_baseline_hash_compare(cpp_summary)
        append_log(
            master_log,
            f"ACE_STEP_HASH_COMPARE family={family} mode={mode} backend={backend} "
            f"match={int(hash_compare['match'])} baseline_sha256={hash_compare['baseline_sha256']} "
            f"current_sha256={hash_compare['current_sha256']}",
        )
        python_step_paths = write_sequence_step_artifacts(python_summary.get("sequence_steps", []), scenario_dir / "python_json", "python")
        cpp_step_paths = write_sequence_step_artifacts(cpp_summary.get("sequence_steps", []), scenario_dir / "cpp_json", "cpp")
        for request_index in range(args.requests_per_session):
            python_step_path = python_step_paths[request_index] if request_index < len(python_step_paths) else ""
            cpp_step_path = cpp_step_paths[request_index] if request_index < len(cpp_step_paths) else ""
            append_log(master_log, f"PYTHON OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={python_step_path} valid={int(file_is_nonempty(python_step_path))}")
            append_log(master_log, f"CPP OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={cpp_step_path} valid={int(file_is_nonempty(cpp_step_path))}")
    elif scenario_config["kind"] in {"vevo2", "seed_vc", "miocodec", "voxcpm2", "supertonic", "vibevoice", "irodori_tts", "heartmula", "higgs_tts", "index_tts2"}:
        python_valid = validate_sequence_result(python_summary, args.requests_per_session, scenario_config["kind"])
        cpp_valid = validate_sequence_result(cpp_summary, args.requests_per_session, scenario_config["kind"])
        python_step_paths = write_sequence_step_artifacts(python_summary.get("sequence_steps", []), scenario_dir / "python_json", "python")
        cpp_step_paths = write_sequence_step_artifacts(cpp_summary.get("sequence_steps", []), scenario_dir / "cpp_json", "cpp")
        for request_index in range(args.requests_per_session):
            python_step_path = python_step_paths[request_index] if request_index < len(python_step_paths) else ""
            cpp_step_path = cpp_step_paths[request_index] if request_index < len(cpp_step_paths) else ""
            append_log(master_log, f"PYTHON OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={python_step_path} valid={int(file_is_nonempty(python_step_path))}")
            append_log(master_log, f"CPP OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={cpp_step_path} valid={int(file_is_nonempty(cpp_step_path))}")
            if request_index >= len(python_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["requests"][request_index], "missing_python_step")
            elif request_index >= len(cpp_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["requests"][request_index], "missing_cpp_step")
            else:
                parity = compare_single_audio_step(
                    cpp_summary["sequence_steps"][request_index],
                    python_summary["sequence_steps"][request_index],
                    float(scenario_config.get("wav_cosine_min", 0.80)),
                    float(scenario_config["log_mel_cosine_min"]) if "log_mel_cosine_min" in scenario_config else None,
                    float(scenario_config["mrstft_spectral_convergence_max"]) if "mrstft_spectral_convergence_max" in scenario_config else None,
                    float(scenario_config["mrstft_logmag_mae_max"]) if "mrstft_logmag_mae_max" in scenario_config else None,
                    int(scenario_config["similarity_vote_required"]) if "similarity_vote_required" in scenario_config else None,
                    float(scenario_config.get("length_ratio_min", 0.98)),
                    bool(scenario_config.get("length_ratio_is_diagnostic", False)),
                )
                parity["request_index"] = request_index
                parity["request"] = request_manifest["requests"][request_index]
            parity_results.append(parity)
            append_log(
                master_log,
                f"PARITY family={family} mode={mode} backend={backend} request={request_index} "
                f"ok={int(parity['ok'])} reason={parity['reason']} metrics={json.dumps(parity.get('metrics', {}), sort_keys=True)}",
            )
    elif scenario_config["kind"] == "asr":
        python_valid = validate_sequence_result(python_summary, args.requests_per_session, "asr")
        cpp_valid = validate_sequence_result(cpp_summary, args.requests_per_session, "asr")
        python_step_paths = write_sequence_step_artifacts(python_summary.get("sequence_steps", []), scenario_dir / "python_json", "python")
        cpp_step_paths = write_sequence_step_artifacts(cpp_summary.get("sequence_steps", []), scenario_dir / "cpp_json", "cpp")
        for request_index in range(args.requests_per_session):
            python_step_path = python_step_paths[request_index] if request_index < len(python_step_paths) else ""
            cpp_step_path = cpp_step_paths[request_index] if request_index < len(cpp_step_paths) else ""
            append_log(master_log, f"PYTHON OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={python_step_path} valid={int(file_is_nonempty(python_step_path))}")
            append_log(master_log, f"CPP OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={cpp_step_path} valid={int(file_is_nonempty(cpp_step_path))}")
            if request_index >= len(python_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["audio_sequence"][request_index], "missing_python_step")
            elif request_index >= len(cpp_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["audio_sequence"][request_index], "missing_cpp_step")
            else:
                if family == "qwen3_asr":
                    expected_fragments = request_manifest.get("expected_fragments", [])
                    parity = compare_qwen3_asr_step(
                        cpp_summary["sequence_steps"][request_index],
                        python_summary["sequence_steps"][request_index],
                        expected_fragments[request_index] if request_index < len(expected_fragments) else [],
                        bool(config.get("check_language", False)),
                    )
                else:
                    parity = compare_parakeet_step(cpp_summary["sequence_steps"][request_index], python_summary["sequence_steps"][request_index])
                parity["request_index"] = request_index
                parity["request"] = request_manifest["audio_sequence"][request_index]
            parity_results.append(parity)
            append_log(master_log, f"PARITY family={family} mode={mode} backend={backend} request={request_index} ok={int(parity['ok'])} reason={parity['reason']}")
    else:
        sequence_kind = scenario_config["kind"]
        python_valid = validate_sequence_result(python_summary, args.requests_per_session, sequence_kind)
        cpp_valid = validate_sequence_result(cpp_summary, args.requests_per_session, sequence_kind)
        python_step_paths = write_sequence_step_artifacts(python_summary.get("sequence_steps", []), scenario_dir / "python_json", "python")
        cpp_step_paths = write_sequence_step_artifacts(cpp_summary.get("sequence_steps", []), scenario_dir / "cpp_json", "cpp")
        for request_index in range(args.requests_per_session):
            python_step_path = python_step_paths[request_index] if request_index < len(python_step_paths) else ""
            cpp_step_path = cpp_step_paths[request_index] if request_index < len(cpp_step_paths) else ""
            append_log(master_log, f"PYTHON OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={python_step_path} valid={int(file_is_nonempty(python_step_path))}")
            append_log(master_log, f"CPP OUTPUT family={family} mode={mode} backend={backend} request={request_index} path={cpp_step_path} valid={int(file_is_nonempty(cpp_step_path))}")
            if request_index >= len(python_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["audio_sequence"][request_index], "missing_python_step")
            elif request_index >= len(cpp_summary.get("sequence_steps", [])):
                parity = missing_parity(request_index, request_manifest["audio_sequence"][request_index], "missing_cpp_step")
            else:
                if sequence_kind == "vad":
                    parity = compare_vad_step(cpp_summary["sequence_steps"][request_index], python_summary["sequence_steps"][request_index])
                elif sequence_kind == "spk":
                    parity = compare_speaker_step(cpp_summary["sequence_steps"][request_index], python_summary["sequence_steps"][request_index])
                elif sequence_kind == "separation":
                    parity = compare_separation_step(cpp_summary["sequence_steps"][request_index], python_summary["sequence_steps"][request_index])
                elif sequence_kind == "alignment":
                    expected_words = request_manifest.get("expected_words", [])
                    parity = compare_qwen3_forced_aligner_step(
                        cpp_summary["sequence_steps"][request_index],
                        python_summary["sequence_steps"][request_index],
                        expected_words[request_index] if request_index < len(expected_words) else [],
                    )
                else:
                    parity = compare_sortformer_step(cpp_summary["sequence_steps"][request_index], python_summary["sequence_steps"][request_index])
                parity["request_index"] = request_index
                parity["request"] = request_manifest["audio_sequence"][request_index]
            parity_results.append(parity)
            append_log(master_log, f"PARITY family={family} mode={mode} backend={backend} request={request_index} ok={int(parity['ok'])} reason={parity['reason']}")

    python_summary_path.write_text(json.dumps(python_result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    cpp_summary_path.write_text(json.dumps(cpp_result_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    scenario_summary = {
        "family": family,
        "mode": mode,
        "backend": backend,
        "warmup": effective_warmup(scenario_config, args),
        "request_manifest": request_manifest,
        "python": {
            "summary_path": str(python_summary_path),
            "parsed": python_result_payload,
            "valid": python_valid,
            "memory": python_memory,
        },
        "cpp": {
            "summary_path": str(cpp_summary_path),
            "parsed": cpp_result_payload,
            "valid": cpp_valid,
            "memory": cpp_memory,
        },
        "parity": parity_results,
        "ok": python_valid["ok"] and cpp_valid["ok"] and all(item["ok"] for item in parity_results),
    }
    if scenario_config["kind"] == "ace_step":
        scenario_summary["hash_compare"] = hash_compare
        if hash_compare["comparable"]:
            scenario_summary["ok"] = scenario_summary["ok"] and hash_compare["match"]
        dit_compare_path = scenario_dir / "ace_step_dit_compare.json"
        if dit_compare_path.exists():
            dit_compare = json.loads(dit_compare_path.read_text(encoding="utf-8"))
            scenario_summary["dit_tensor_compare"] = {
                "path": str(dit_compare_path),
                "result": dit_compare,
            }
            scenario_summary["ok"] = scenario_summary["ok"] and bool(dit_compare.get("ok", False))
    (scenario_dir / "scenario_summary.json").write_text(json.dumps(scenario_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(master_log, f"SCENARIO END family={family} mode={mode} backend={backend} ok={int(scenario_summary['ok'])}")
    args.requests_per_session = original_requests_per_session
    return scenario_summary


def main() -> int:
    args = parse_args()
    requested_stamp = args.artifact_stamp or timestamp_seconds_local()
    requested_output_dir = REPO_ROOT / (
        args.output_dir if args.output_dir else f"build/logs/warmbench/{requested_stamp}")
    root_output_dir = requested_output_dir if args.keep_output_dir else next_artifact_dir(requested_output_dir)
    stamp = root_output_dir.name
    aggregate_json = root_output_dir / (Path(args.output_json).name if args.output_json else "summary.json")
    root_output_dir.mkdir(parents=True, exist_ok=True)
    master_log = root_output_dir / "run.log"

    aggregate_requests_per_session = aggregate_requests_per_session_value(args)
    append_log(master_log, f"RUN START stamp={stamp} modes={args.mode} backends={selected_backends(args)} requests_per_session={aggregate_requests_per_session}")
    summary: dict[str, Any] = {
        "artifact_stamp": stamp,
        "warmup": aggregate_warmup_value(args),
        "warmup_arg": args.warmup,
        "warmup_was_explicit": args.warmup_was_explicit,
        "iterations": args.iterations,
        "threads": args.threads,
        "device": args.device,
        "requests_per_session": aggregate_requests_per_session,
        "scenarios": [],
    }
    for family in selected_families(args):
        config = FAMILY_CONFIG[family]
        for backend in selected_backends(args):
            for mode in selected_modes(args, config):
                scenario_summary = run_scenario(family, config, mode, backend, args, root_output_dir, master_log)
                summary["scenarios"].append(scenario_summary)

    summary["ok"] = all(item["ok"] for item in summary["scenarios"])
    aggregate_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    append_log(master_log, f"RUN END ok={int(summary['ok'])} summary={aggregate_json}")
    print(f"summary_json={aggregate_json}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
