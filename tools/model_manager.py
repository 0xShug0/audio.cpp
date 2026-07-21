#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import fractions
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import warnings
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

# Heavy install/convert deps are imported lazily so ``list --json`` / ``info``
# work without Torch in the environment.
torch: Any = None
safe_open: Any = None
load_file: Any = None
save_file: Any = None
yaml: Any = None


def _ensure_install_deps() -> None:
    """Import Torch / safetensors / yaml on first install/convert use."""
    global torch, safe_open, load_file, save_file, yaml
    if torch is not None:
        return
    import torch as _torch
    from safetensors import safe_open as _safe_open
    from safetensors.torch import load_file as _load_file
    from safetensors.torch import save_file as _save_file
    import yaml as _yaml

    torch = _torch
    safe_open = _safe_open
    load_file = _load_file
    save_file = _save_file
    yaml = _yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANAGER_ASSETS = REPO_ROOT / "assets" / "model_manager"
DEMUCS_ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
WHISPER_MEDIUM_PT_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
)
DEMUCS_REMOTE_MODELS = {
    "955717e8": DEMUCS_ROOT_URL + "hybrid_transformer/955717e8-8726e21a.th",
    "f7e0c4bc": DEMUCS_ROOT_URL + "hybrid_transformer/f7e0c4bc-ba3fe64a.th",
    "d12395a8": DEMUCS_ROOT_URL + "hybrid_transformer/d12395a8-e57c48e6.th",
    "92cfc3b6": DEMUCS_ROOT_URL + "hybrid_transformer/92cfc3b6-ef3bcb9c.th",
    "04573f0d": DEMUCS_ROOT_URL + "hybrid_transformer/04573f0d-f3cf25b2.th",
    "75fc33f5": DEMUCS_ROOT_URL + "hybrid_transformer/75fc33f5-1941ce65.th",
    "5c90dfd2": DEMUCS_ROOT_URL + "5c90dfd2-34c22ccb.th",
}
DEMUCS_BAGS: dict[str, dict[str, Any]] = {
    "htdemucs": {"models": ["955717e8"]},
    "htdemucs_ft": {
        "models": ["f7e0c4bc", "d12395a8", "92cfc3b6", "04573f0d"],
        "weights": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    },
    "htdemucs_6s": {"models": ["5c90dfd2"]},
    "hdemucs_mmi": {"models": ["75fc33f5"], "segment": 44},
}


def huggingface_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        cached = token_path.read_text(encoding="utf-8").strip()
        if cached:
            return cached
    return None


def http_headers() -> dict[str, str]:
    headers = {"User-Agent": "audio.cpp model_manager.py"}
    token = huggingface_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@dataclasses.dataclass(frozen=True)
class SnapshotSource:
    repo_id: str
    revision: str = "main"
    include_prefixes: tuple[str, ...] = ()
    include_suffixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SnapshotPlacement:
    source: SnapshotSource
    target_subdir: str = ""
    required_files: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CompositeSnapshotSource:
    placements: tuple[SnapshotPlacement, ...]


@dataclasses.dataclass(frozen=True)
class ConverterSource:
    kind: str
    description: str
    url: str = ""
    output_weights_file: str = ""
    config_kind: str = ""


@dataclasses.dataclass(frozen=True)
class UnsupportedSource:
    reason: str


@dataclasses.dataclass(frozen=True)
class ModelPackage:
    id: str
    display_name: str
    target_directory: str
    required_files: tuple[str, ...]
    source: SnapshotSource | CompositeSnapshotSource | ConverterSource | UnsupportedSource
    description: str = ""
    # Optional identity fields for ``list --json`` consumers. When unset,
    # family/tasks/standalone/gated are derived from local package metadata only.
    family: str | None = None
    standalone: bool | None = None
    parent_package_id: str | None = None
    tasks: tuple[str, ...] = ()
    gated: bool | None = None


UTILITY_CONVERTER_KINDS = {"pytorch_to_safetensors"}
POSTPROCESS_SNAPSHOT_PACKAGE_IDS = {"voxcpm2"}
# HF repos that require accepting a license / access grant before download.
_GATED_REPO_MARKERS = (
    "kyutai/pocket-tts",
    "stabilityai/",
)


def package_install_kind(package: ModelPackage) -> str:
    source = package.source
    if isinstance(source, SnapshotSource):
        return "composite" if package.id in POSTPROCESS_SNAPSHOT_PACKAGE_IDS else "snapshot"
    if isinstance(source, CompositeSnapshotSource):
        return "composite"
    if isinstance(source, ConverterSource):
        return "utility" if source.kind in UTILITY_CONVERTER_KINDS else "composite"
    return "unsupported"


def package_usage_examples(package: ModelPackage) -> list[str]:
    if package.id == "voxcpm2_audiovae":
        return [
            "python tools/model_manager.py install voxcpm2_audiovae --source-file models/VoxCPM2/audiovae.pth --models-root models --overwrite",
            "python tools/model_manager.py install voxcpm2_audiovae --source-file /path/to/audiovae.pth --output-file /path/to/audiovae.safetensors --overwrite",
        ]
    return []


CATALOG: tuple[ModelPackage, ...] = (
    ModelPackage(
        id="ace_step",
        display_name="ACE-Step 1.5",
        target_directory="Ace-Step1.5",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="ACE-Step/Ace-Step1.5"),
                    required_files=(
                        "config.json",
                        "Qwen3-Embedding-0.6B/model.safetensors",
                        "acestep-5Hz-lm-1.7B/model.safetensors",
                        "acestep-v15-turbo/model.safetensors",
                        "acestep-v15-turbo/silence_latent.pt",
                        "vae/diffusion_pytorch_model.safetensors",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="ACE-Step/acestep-v15-base"),
                    target_subdir="acestep-v15-base",
                    required_files=("config.json", "model.safetensors", "silence_latent.pt"),
                ),
            ),
        ),
        required_files=(
            "Qwen3-Embedding-0.6B/model.safetensors",
            "acestep-5Hz-lm-1.7B/model.safetensors",
            "acestep-v15-base/model.safetensors",
            "acestep-v15-base/silence_latent.safetensors",
            "acestep-v15-turbo/model.safetensors",
            "acestep-v15-turbo/silence_latent.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ),
    ),
    ModelPackage(
        id="kokoro_82m_bf16",
        display_name="Kokoro 82M bf16",
        target_directory="Kokoro-82M-bf16",
        source=SnapshotSource(repo_id="mlx-community/Kokoro-82M-bf16"),
        required_files=("config.json", "kokoro-v1_0.safetensors", "voices/af_heart.safetensors"),
        family="kokoro_tts",
        tasks=("tts",),
    ),
    ModelPackage(
        id="moss_tts_nano_100m",
        display_name="MOSS-TTS-Nano 100M",
        target_directory="MOSS-TTS-Nano-100M",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-TTS-Nano-100M"),
                    required_files=("config.json", "pytorch_model.bin", "tokenizer.model", "tokenizer_config.json"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"),
                    target_subdir="audio_tokenizer",
                    required_files=("config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "model.safetensors",
            "tokenizer.model",
            "tokenizer_config.json",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model-00001-of-00001.safetensors",
            "audio_tokenizer/model.safetensors.index.json",
        ),
        description="Recommended MOSS-TTS-Nano package; assembles the TTS model and MOSS-Audio-Tokenizer-Nano dependency.",
    ),
    ModelPackage(
        id="moss_tts_nano_100m_model",
        display_name="MOSS-TTS-Nano 100M model subcomponent",
        target_directory="MOSS-TTS-Nano-100M",
        source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-TTS-Nano-100M"),
        required_files=("config.json", "pytorch_model.bin", "tokenizer.model", "tokenizer_config.json"),
        description="Subcomponent only. Use moss_tts_nano_100m for the full framework-ready MOSS runtime layout.",
    ),
    ModelPackage(
        id="moss_audio_tokenizer_nano",
        display_name="MOSS Audio Tokenizer Nano subcomponent",
        target_directory="MOSS-Audio-Tokenizer-Nano",
        source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"),
        required_files=("config.json", "model-00001-of-00001.safetensors", "model.safetensors.index.json"),
        description="Subcomponent only. Use moss_tts_nano_100m for the full framework-ready MOSS Nano runtime layout.",
    ),
    ModelPackage(
        id="moss_audio_tokenizer_v2",
        display_name="MOSS Audio Tokenizer v2 subcomponent",
        target_directory="MOSS-Audio-Tokenizer-v2",
        source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"),
        required_files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
        description="Subcomponent only. Use moss_tts_local_v1_5 for the full framework-ready MOSS-TTS-Local runtime layout.",
    ),
    ModelPackage(
        id="moss_tts_local_v1_5",
        display_name="MOSS-TTS-Local Transformer v1.5",
        target_directory="MOSS-TTS-Local-Transformer-v1.5",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"),
                    required_files=(
                        "config.json",
                        "model.safetensors",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "chat_template.jinja",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"),
                    target_subdir="audio_tokenizer",
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00003.safetensors",
                        "model-00002-of-00003.safetensors",
                        "model-00003-of-00003.safetensors",
                    ),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors.index.json",
            "audio_tokenizer/model-00001-of-00003.safetensors",
            "audio_tokenizer/model-00002-of-00003.safetensors",
            "audio_tokenizer/model-00003-of-00003.safetensors",
        ),
        description="MOSS-TTS-Local Transformer v1.5 with the MOSS-Audio-Tokenizer-v2 codec dependency.",
    ),
    ModelPackage(
        id="omnivoice",
        display_name="OmniVoice",
        target_directory="OmniVoice",
        source=SnapshotSource(repo_id="k2-fsa/OmniVoice"),
        required_files=(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "audio_tokenizer/config.json",
            "audio_tokenizer/model.safetensors",
        ),
    ),
    ModelPackage(
        id="qwen3_asr_0_6b",
        display_name="Qwen3 ASR 0.6B",
        target_directory="Qwen3-ASR-0.6B",
        source=SnapshotSource(repo_id="Qwen/Qwen3-ASR-0.6B"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="qwen3_asr_1_7b_hf",
        display_name="Qwen3 ASR 1.7B HF",
        target_directory="Qwen3-ASR-1.7B-hf",
        source=SnapshotSource(
            repo_id="Qwen/Qwen3-ASR-1.7B-hf",
            include_prefixes=(
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ),
        ),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
        ),
        description="Native Hugging Face Transformers checkpoint; no conversion is required.",
    ),
    ModelPackage(
        id="voxtral_realtime",
        display_name="Voxtral Mini 4B Realtime",
        target_directory="Voxtral-Mini-4B-Realtime-2602",
        source=SnapshotSource(
            repo_id="mistralai/Voxtral-Mini-4B-Realtime-2602",
            include_prefixes=(
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "params.json",
                "processor_config.json",
                "tekken.json",
            ),
        ),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "params.json",
            "processor_config.json",
            "tekken.json",
        ),
        family="voxtral_realtime",
        tasks=("asr",),
        description="Native Hugging Face checkpoint for Voxtral realtime ASR; no conversion is required.",
    ),
    ModelPackage(
        id="higgs_audio_stt",
        display_name="Higgs Audio STT",
        target_directory="higgs-audio-v3-stt",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="bosonai/higgs-audio-v3-stt"),
                    required_files=(
                        "config.json",
                        "generation_config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="openai/whisper-large-v3",
                        include_prefixes=("preprocessor_config.json",),
                    ),
                    target_subdir="../whisper-large-v3",
                    required_files=("preprocessor_config.json",),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "../whisper-large-v3/preprocessor_config.json",
        ),
        description="Installs Higgs Audio STT plus the sibling Whisper Large v3 preprocessor config required by the framework runtime.",
    ),
    ModelPackage(
        id="hviske_asr",
        display_name="Hviske ASR",
        target_directory="hviske-v5.3",
        source=SnapshotSource(repo_id="syvai/hviske-v5.3"),
        required_files=("config.json", "generation_config.json", "model.safetensors", "tokenizer.model"),
    ),
    ModelPackage(
        id="nemotron_asr",
        display_name="Nemotron ASR",
        target_directory="nemotron-3.5-asr-streaming-0.6b",
        source=SnapshotSource(repo_id="nvidia/nemotron-3.5-asr-streaming-0.6b"),
        required_files=("config.json", "model.safetensors", "processor_config.json", "tokenizer.json"),
    ),
    ModelPackage(
        id="qwen3_forced_aligner_0_6b",
        display_name="Qwen3 Forced Aligner 0.6B",
        target_directory="Qwen3-ForcedAligner-0.6B",
        source=SnapshotSource(repo_id="Qwen/Qwen3-ForcedAligner-0.6B"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="qwen3_tts_0_6b_base",
        display_name="Qwen3 TTS 12Hz 0.6B Base",
        target_directory="Qwen3-TTS-12Hz-0.6B-Base",
        source=SnapshotSource(repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="vietneu_tts_v3_turbo",
        display_name="VieNeu-TTS v3 Turbo Base",
        target_directory="VieNeu-TTS-v3-Turbo",
        source=SnapshotSource(repo_id="phuocnguyen90/VieNeu-TTS-v3-Turbo-GGUF"),
        required_files=(
            "config.json",
            "model.gguf",
            "speech_tokenizer/config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ),
        description="Installs VieNeu-TTS v3 Turbo GGUF model and configuration sidecars for C++ inference.",
    ),
    ModelPackage(
        id="qwen3_tts_1_7b_base",
        display_name="Qwen3 TTS 12Hz 1.7B Base",
        target_directory="Qwen3-TTS-12Hz-1.7B-Base",
        source=SnapshotSource(repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="qwen3_tts_1_7b_custom_voice",
        display_name="Qwen3 TTS 12Hz 1.7B Custom Voice",
        target_directory="Qwen3-TTS-12Hz-1.7B-CustomVoice",
        source=SnapshotSource(repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="qwen3_tts_1_7b_voice_design",
        display_name="Qwen3 TTS 12Hz 1.7B Voice Design",
        target_directory="Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        source=SnapshotSource(repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="qwen3_tts_tokenizer_12hz",
        display_name="Qwen3 TTS Tokenizer 12Hz subcomponent",
        target_directory="Qwen3-TTS-Tokenizer-12Hz",
        source=SnapshotSource(repo_id="Qwen/Qwen3-TTS-Tokenizer-12Hz"),
        required_files=("config.json", "model.safetensors"),
        description="Subcomponent only. Normal Qwen3 TTS packages already include the speech tokenizer needed by the framework.",
    ),
    ModelPackage(
        id="chatterbox",
        display_name="Chatterbox",
        target_directory="chatterbox",
        source=SnapshotSource(repo_id="ResembleAI/chatterbox"),
        required_files=(
            "ve.safetensors",
            "t3_cfg.safetensors",
            "t3_mtl23ls_v2.safetensors",
            "t3_mtl23ls_v3.safetensors",
            "s3gen.safetensors",
            "tokenizer.json",
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
            "conds.pt",
        ),
    ),
    ModelPackage(
        id="sortformer_diar_4spk_v1",
        display_name="Sortformer diarization 4 speaker v1",
        target_directory="diar_sortformer_4spk-v1",
        source=SnapshotSource(repo_id="nvidia/diar_sortformer_4spk-v1"),
        required_files=("config.json", "model.safetensors", "processor_config.json"),
    ),
    ModelPackage(
        id="parakeet_tdt_0_6b_v3",
        display_name="Parakeet TDT 0.6B v3",
        target_directory="parakeet-tdt-0.6b-v3",
        source=SnapshotSource(repo_id="nvidia/parakeet-tdt-0.6b-v3"),
        required_files=("config.json", "model.safetensors", "processor_config.json", "tokenizer.json"),
    ),
    ModelPackage(
        id="pocket_tts",
        display_name="PocketTTS",
        target_directory="pocket-tts",
        source=SnapshotSource(
            repo_id="kyutai/pocket-tts",
            include_prefixes=("languages/english/",),
        ),
        required_files=(
            "languages/english/model.safetensors",
            "languages/english/tokenizer.model",
            "languages/english/embeddings/alba.safetensors",
        ),
    ),
    ModelPackage(
        id="miocodec_25hz_44k_v2",
        display_name="MioCodec 25Hz 44.1kHz v2",
        target_directory="MioCodec-25Hz-44.1kHz-v2",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="Aratako/MioCodec-25Hz-44.1kHz-v2",
                        include_prefixes=("config.yaml", "model.safetensors"),
                    ),
                    required_files=("config.yaml", "model.safetensors"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="mlx-community/wavlm-base-plus-mlx",
                        include_prefixes=("config.json", "weights.safetensors"),
                    ),
                    target_subdir="wavlm-base-plus-mlx",
                    required_files=("config.json", "weights.safetensors"),
                ),
            ),
        ),
        required_files=(
            "config.yaml",
            "model.safetensors",
            "wavlm-base-plus-mlx/config.json",
            "wavlm-base-plus-mlx/weights.safetensors",
        ),
    ),
    ModelPackage(
        id="miotts_1_7b",
        display_name="MioTTS 1.7B",
        target_directory="MioTTS-1.7B",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="Aratako/MioTTS-1.7B",
                        include_prefixes=(
                            "config.json",
                            "generation_config.json",
                            "tokenizer_config.json",
                            "tokenizer.json",
                            "vocab.json",
                            "merges.txt",
                            "model.safetensors",
                        ),
                    ),
                    required_files=(
                        "config.json",
                        "generation_config.json",
                        "tokenizer_config.json",
                        "tokenizer.json",
                        "vocab.json",
                        "merges.txt",
                        "model.safetensors",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="Aratako/MioCodec-25Hz-44.1kHz-v2",
                        include_prefixes=("config.yaml", "model.safetensors"),
                    ),
                    target_subdir="../MioCodec-25Hz-44.1kHz-v2",
                    required_files=("config.yaml", "model.safetensors"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="mlx-community/wavlm-base-plus-mlx",
                        include_prefixes=("config.json", "weights.safetensors"),
                    ),
                    target_subdir="../MioCodec-25Hz-44.1kHz-v2/wavlm-base-plus-mlx",
                    required_files=("config.json", "weights.safetensors"),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ),
        description="Installs MioTTS plus the sibling MioCodec dependency required by the framework runtime.",
    ),
    ModelPackage(
        id="vibevoice_asr",
        display_name="VibeVoice ASR",
        target_directory="VibeVoice-ASR",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="microsoft/VibeVoice-ASR"),
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00008.safetensors",
                        "model-00002-of-00008.safetensors",
                        "model-00003-of-00008.safetensors",
                        "model-00004-of-00008.safetensors",
                        "model-00005-of-00008.safetensors",
                        "model-00006-of-00008.safetensors",
                        "model-00007-of-00008.safetensors",
                        "model-00008-of-00008.safetensors",
                    ),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00008.safetensors",
            "model-00002-of-00008.safetensors",
            "model-00003-of-00008.safetensors",
            "model-00004-of-00008.safetensors",
            "model-00005-of-00008.safetensors",
            "model-00006-of-00008.safetensors",
            "model-00007-of-00008.safetensors",
            "model-00008-of-00008.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
        description="Installs VibeVoice ASR plus the Qwen2.5 tokenizer files required by the framework runtime.",
    ),
    ModelPackage(
        id="vibevoice_1_5b",
        display_name="VibeVoice 1.5B",
        target_directory="VibeVoice-1.5B",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="microsoft/VibeVoice-1.5B"),
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00003.safetensors",
                        "model-00002-of-00003.safetensors",
                        "model-00003-of-00003.safetensors",
                        "preprocessor_config.json",
                    ),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="vibevoice_7b",
        display_name="VibeVoice 7B",
        target_directory="VibeVoice-7B",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="vibevoice/VibeVoice-7B"),
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00010.safetensors",
                        "model-00002-of-00010.safetensors",
                        "model-00003-of-00010.safetensors",
                        "model-00004-of-00010.safetensors",
                        "model-00005-of-00010.safetensors",
                        "model-00006-of-00010.safetensors",
                        "model-00007-of-00010.safetensors",
                        "model-00008-of-00010.safetensors",
                        "model-00009-of-00010.safetensors",
                        "model-00010-of-00010.safetensors",
                        "preprocessor_config.json",
                    ),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00010.safetensors",
            "model-00002-of-00010.safetensors",
            "model-00003-of-00010.safetensors",
            "model-00004-of-00010.safetensors",
            "model-00005-of-00010.safetensors",
            "model-00006-of-00010.safetensors",
            "model-00007-of-00010.safetensors",
            "model-00008-of-00010.safetensors",
            "model-00009-of-00010.safetensors",
            "model-00010-of-00010.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ),
    ),
    ModelPackage(
        id="higgs_audio_v3_tts_4b",
        display_name="Higgs Audio v3 TTS 4B",
        target_directory="higgs-audio-v3-tts-4b",
        source=SnapshotSource(repo_id="bosonai/higgs-audio-v3-tts-4b"),
        required_files=(
            "chat_template.jinja",
            "config.json",
            "model.safetensors.index.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ),
        family="higgs_audio_tts",
        tasks=("tts",),
    ),
    ModelPackage(
        id="heartmula",
        display_name="HeartMuLa",
        target_directory="HeartMuLa",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="HeartMuLa/HeartMuLaGen"),
                    required_files=("gen_config.json", "tokenizer.json"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="HeartMuLa/HeartMuLa-oss-3B"),
                    target_subdir="HeartMuLa-oss-3B",
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00004.safetensors",
                        "model-00002-of-00004.safetensors",
                        "model-00003-of-00004.safetensors",
                        "model-00004-of-00004.safetensors",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="HeartMuLa/HeartCodec-oss-20260123"),
                    target_subdir="HeartCodec-oss",
                    required_files=(
                        "config.json",
                        "model.safetensors.index.json",
                        "model-00001-of-00002.safetensors",
                        "model-00002-of-00002.safetensors",
                    ),
                ),
            ),
        ),
        required_files=(
            "tokenizer.json",
            "gen_config.json",
            "HeartMuLa-oss-3B/config.json",
            "HeartMuLa-oss-3B/model.safetensors.index.json",
            "HeartMuLa-oss-3B/model-00001-of-00004.safetensors",
            "HeartMuLa-oss-3B/model-00002-of-00004.safetensors",
            "HeartMuLa-oss-3B/model-00003-of-00004.safetensors",
            "HeartMuLa-oss-3B/model-00004-of-00004.safetensors",
            "HeartCodec-oss/config.json",
            "HeartCodec-oss/model.safetensors.index.json",
            "HeartCodec-oss/model-00001-of-00002.safetensors",
            "HeartCodec-oss/model-00002-of-00002.safetensors",
        ),
    ),
    ModelPackage(
        id="irodori_tts_500m_v3",
        display_name="Irodori-TTS 500M v3",
        target_directory="Irodori-TTS-500M-v3",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="Aratako/Irodori-TTS-500M-v3"),
                    required_files=("model.safetensors",),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="llm-jp/llm-jp-3-150m"),
                    target_subdir="../llm-jp-3-150m",
                    required_files=("tokenizer.json", "model.safetensors", "config.json"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="Aratako/Semantic-DACVAE-Japanese-32dim"),
                    target_subdir="../Semantic-DACVAE-Japanese-32dim",
                    required_files=("weights.pth",),
                ),
            ),
        ),
        required_files=(
            "model.safetensors",
            "model_config.json",
            "../llm-jp-3-150m/tokenizer.json",
            "../Semantic-DACVAE-Japanese-32dim/weights.safetensors",
        ),
        description="Installs Irodori-TTS plus the sibling llm-jp tokenizer and DACVAE codec dependencies required by the framework runtime.",
    ),
    ModelPackage(
        id="irodori_tts_600m_v3_voice_design",
        display_name="Irodori-TTS 600M v3 VoiceDesign",
        target_directory="Irodori-TTS-600M-v3-VoiceDesign",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="Aratako/Irodori-TTS-600M-v3-VoiceDesign"),
                    required_files=("model.safetensors",),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="llm-jp/llm-jp-3-150m"),
                    target_subdir="../llm-jp-3-150m",
                    required_files=("tokenizer.json", "model.safetensors", "config.json"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="Aratako/Semantic-DACVAE-Japanese-32dim"),
                    target_subdir="../Semantic-DACVAE-Japanese-32dim",
                    required_files=("weights.pth",),
                ),
            ),
        ),
        required_files=(
            "model.safetensors",
            "model_config.json",
            "../llm-jp-3-150m/tokenizer.json",
            "../Semantic-DACVAE-Japanese-32dim/weights.safetensors",
        ),
        description="Installs Irodori-TTS VoiceDesign plus the sibling llm-jp tokenizer and DACVAE codec dependencies required by the framework runtime.",
    ),
    ModelPackage(
        id="outetts_1_0_1b",
        display_name="OuteTTS 1.0 1B",
        target_directory="Llama-OuteTTS-1.0-1B",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="OuteAI/Llama-OuteTTS-1.0-1B"),
                    required_files=(
                        "config.json",
                        "generation_config.json",
                        "model.safetensors",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                    ),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="ibm-research/DAC.speech.v1.0"),
                    target_subdir="../DAC.speech.v1.0",
                    required_files=("config.json", "weights_24khz_1.5kbps_v1.0.pth"),
                ),
                SnapshotPlacement(
                    source=SnapshotSource(repo_id="Qwen/Qwen3-ForcedAligner-0.6B"),
                    target_subdir="../Qwen3-ForcedAligner-0.6B",
                    required_files=(
                        "config.json",
                        "generation_config.json",
                        "model.safetensors",
                        "preprocessor_config.json",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                    ),
                ),
            ),
        ),
        required_files=(
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "../DAC.speech.v1.0/config.json",
            "../DAC.speech.v1.0/model.safetensors",
            "../Qwen3-ForcedAligner-0.6B/config.json",
            "../Qwen3-ForcedAligner-0.6B/generation_config.json",
            "../Qwen3-ForcedAligner-0.6B/model.safetensors",
            "../Qwen3-ForcedAligner-0.6B/preprocessor_config.json",
            "../Qwen3-ForcedAligner-0.6B/tokenizer_config.json",
            "../Qwen3-ForcedAligner-0.6B/vocab.json",
            "../Qwen3-ForcedAligner-0.6B/merges.txt",
        ),
        description="Installs OuteTTS, its IBM DAC 1.5 kbps codec, and Qwen3 Forced Aligner for reliable voice cloning.",
    ),
    ModelPackage(
        id="stable_audio_3_small_music",
        display_name="Stable Audio 3 Small Music",
        target_directory="stable-audio-3-small-music",
        source=SnapshotSource(repo_id="stabilityai/stable-audio-3-small-music"),
        required_files=(
            "model_config.json",
            "model.safetensors",
            "t5gemma-b-b-ul2/config.json",
            "t5gemma-b-b-ul2/model.safetensors",
            "t5gemma-b-b-ul2/tokenizer.json",
            "t5gemma-b-b-ul2/tokenizer.model",
        ),
    ),
    ModelPackage(
        id="stable_audio_3_small_sfx",
        display_name="Stable Audio 3 Small SFX",
        target_directory="stable-audio-3-small-sfx",
        source=SnapshotSource(repo_id="stabilityai/stable-audio-3-small-sfx"),
        required_files=(
            "model_config.json",
            "model.safetensors",
            "t5gemma-b-b-ul2/config.json",
            "t5gemma-b-b-ul2/model.safetensors",
            "t5gemma-b-b-ul2/tokenizer.json",
            "t5gemma-b-b-ul2/tokenizer.model",
        ),
    ),
    ModelPackage(
        id="stable_audio_3_medium",
        display_name="Stable Audio 3 Medium",
        target_directory="stable-audio-3-medium",
        source=SnapshotSource(repo_id="stabilityai/stable-audio-3-medium"),
        required_files=(
            "model_config.json",
            "model.safetensors",
            "t5gemma-b-b-ul2/config.json",
            "t5gemma-b-b-ul2/model.safetensors",
            "t5gemma-b-b-ul2/tokenizer.json",
            "t5gemma-b-b-ul2/tokenizer.model",
        ),
    ),
    ModelPackage(
        id="supertonic_3",
        display_name="Supertonic 3",
        target_directory="supertonic-3",
        source=SnapshotSource(repo_id="mlx-community/supertonic-3-mlx"),
        required_files=(
            "config/tts.json",
            "config/unicode_indexer.json",
            "ggml/supertonic.safetensors",
            "voice_styles/M1.json",
        ),
    ),
    ModelPackage(
        id="index_tts2",
        display_name="IndexTTS2",
        target_directory="IndexTTS-2",
        source=SnapshotSource(repo_id="mlx-community/index-tts2-mlx"),
        required_files=(
            "config.yaml",
            "bpe.model",
            "gpt.safetensors",
            "s2mel.safetensors",
            "feat1.safetensors",
            "feat2.safetensors",
            "wav2vec2bert_stats.safetensors",
            "semantic_codec_model.safetensors",
            "campplus.safetensors",
            "w2v-bert-2.0/config.json",
            "w2v-bert-2.0/preprocessor_config.json",
            "w2v-bert-2.0/model.safetensors",
            "bigvgan/config.json",
            "bigvgan/model.safetensors",
            "qwen0.6bemo4-merge/config.json",
            "qwen0.6bemo4-merge/generation_config.json",
            "qwen0.6bemo4-merge/tokenizer.json",
            "qwen0.6bemo4-merge/tokenizer_config.json",
            "qwen0.6bemo4-merge/vocab.json",
            "qwen0.6bemo4-merge/merges.txt",
            "qwen0.6bemo4-merge/model.safetensors",
        ),
        description="Framework-ready IndexTTS2 safetensors layout with shared components at the model root.",
    ),
    ModelPackage(
        id="mel_band_roformer",
        display_name="Mel RoFormer MLX",
        target_directory="mel-roformer-mlx",
        source=SnapshotSource(repo_id="mlx-community/mel-roformer-mlx"),
        required_files=("config.json", "model.safetensors"),
    ),
    ModelPackage(
        id="vevo2",
        display_name="Vevo2",
        target_directory="Vevo2",
        source=CompositeSnapshotSource(
            placements=(
                SnapshotPlacement(
                    source=SnapshotSource(
                        repo_id="RMSnow/Vevo2",
                        include_prefixes=(
                            "acoustic_modeling/fm_emilia101k_singnet7k_repa/",
                            "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/",
                            "contentstyle_modeling/posttrained/",
                            "contentstyle_modeling/pretrained/",
                            "tokenizer/contentstyle_fvq16384_12.5hz/",
                            "tokenizer/prosody_fvq512_6.25hz/",
                            "vocoder/",
                        ),
                        exclude_prefixes=(
                            "contentstyle_modeling/posttrained/optimizer.pt",
                            "contentstyle_modeling/posttrained/rng_state_",
                            "contentstyle_modeling/posttrained/scheduler.pt",
                            "contentstyle_modeling/posttrained/trainer_state.json",
                            "contentstyle_modeling/posttrained/training_args.bin",
                            "contentstyle_modeling/pretrained/optimizer.bin",
                            "contentstyle_modeling/pretrained/random_states_",
                            "contentstyle_modeling/pretrained/scheduler.bin",
                        ),
                    ),
                    required_files=(
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json",
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa/model.safetensors",
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.pt",
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/config.json",
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/model.safetensors",
                        "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/whisper_stats.pt",
                        "contentstyle_modeling/posttrained/amphion_config.json",
                        "contentstyle_modeling/posttrained/config.json",
                        "contentstyle_modeling/posttrained/generation_config.json",
                        "contentstyle_modeling/posttrained/merges.txt",
                        "contentstyle_modeling/posttrained/model.safetensors",
                        "contentstyle_modeling/posttrained/tokenizer.json",
                        "contentstyle_modeling/posttrained/tokenizer_config.json",
                        "contentstyle_modeling/posttrained/vocab.json",
                        "contentstyle_modeling/pretrained/config.json",
                        "contentstyle_modeling/pretrained/generation_config.json",
                        "contentstyle_modeling/pretrained/merges.txt",
                        "contentstyle_modeling/pretrained/model.safetensors",
                        "contentstyle_modeling/pretrained/tokenizer.json",
                        "contentstyle_modeling/pretrained/tokenizer_config.json",
                        "contentstyle_modeling/pretrained/vocab.json",
                        "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors",
                        "tokenizer/prosody_fvq512_6.25hz/model.safetensors",
                        "vocoder/config.json",
                        "vocoder/model.safetensors",
                        "vocoder/model_1.safetensors",
                        "vocoder/model_2.safetensors",
                    ),
                ),
            ),
        ),
        required_files=(
            "acoustic_modeling/fm_emilia101k_singnet7k_repa/config.json",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa/model.safetensors",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa/whisper_stats.safetensors",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/config.json",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/model.safetensors",
            "acoustic_modeling/fm_emilia101k_singnet7k_repa_text/whisper_stats.safetensors",
            "contentstyle_modeling/posttrained/amphion_config.json",
            "contentstyle_modeling/posttrained/config.json",
            "contentstyle_modeling/posttrained/generation_config.json",
            "contentstyle_modeling/posttrained/merges.txt",
            "contentstyle_modeling/posttrained/model.safetensors",
            "contentstyle_modeling/posttrained/tokenizer.json",
            "contentstyle_modeling/posttrained/tokenizer_config.json",
            "contentstyle_modeling/posttrained/vocab.json",
            "contentstyle_modeling/pretrained/config.json",
            "contentstyle_modeling/pretrained/generation_config.json",
            "contentstyle_modeling/pretrained/merges.txt",
            "contentstyle_modeling/pretrained/model.safetensors",
            "contentstyle_modeling/pretrained/tokenizer.json",
            "contentstyle_modeling/pretrained/tokenizer_config.json",
            "contentstyle_modeling/pretrained/vocab.json",
            "tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors",
            "tokenizer/prosody_fvq512_6.25hz/model.safetensors",
            "vocoder/config.json",
            "vocoder/model.safetensors",
            "vocoder/model_1.safetensors",
            "vocoder/model_2.safetensors",
        ),
        description="Installs Vevo2 plus the sibling whisper-medium dependency required by the framework runtime.",
    ),
    ModelPackage(
        id="seed_vc",
        display_name="SeedVC-MLX",
        target_directory="SeedVC-MLX",
        source=SnapshotSource(repo_id="mlx-community/SeedVC-MLX"),
        required_files=(
            "seed_vc_manifest.json",
            "v2/vc_wrapper.json",
            "v2/ar.safetensors",
            "v2/cfm.safetensors",
            "v1/svc.json",
            "v1/svc.safetensors",
            "v1/whisper_bigvgan.json",
            "v1/whisper_bigvgan.safetensors",
            "v1/xlsr_hift.json",
            "v1/xlsr_hift.safetensors",
            "astral/bsq32.json",
            "astral/bsq32.safetensors",
            "astral/bsq2048.json",
            "astral/bsq2048.safetensors",
            "campplus/model.safetensors",
            "rmvpe/model.safetensors",
            "hift/config.json",
            "hift/model.safetensors",
            "bigvgan/v2_22khz_80band_256x/config.json",
            "bigvgan/v2_22khz_80band_256x/model.safetensors",
            "bigvgan/v2_44khz_128band_512x/config.json",
            "bigvgan/v2_44khz_128band_512x/model.safetensors",
            "whisper-small/config.json",
            "whisper-small/model.safetensors",
            "hubert-large-ll60k/config.json",
            "hubert-large-ll60k/model.safetensors",
            "wav2vec2-xls-r-300m/config.json",
            "wav2vec2-xls-r-300m/model.safetensors",
        ),
        description="Framework-ready SeedVC-MLX bundle. The Hugging Face snapshot already contains the converted safetensors layout.",
    ),
    ModelPackage(
        id="citrinet_asr",
        display_name="Citrinet ASR converted layout",
        target_directory="citrinet",
        source=ConverterSource(
            kind="nemo_archive",
            description="Download and convert the official NeMo archive into framework-friendly safetensors and sidecars.",
            url="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo",
            output_weights_file="citrinet_256.safetensors",
            config_kind="citrinet",
        ),
        required_files=(
            "citrinet_256.safetensors",
            "citrinet_256_config.json",
            "citrinet_256_tokenizer.model",
            "citrinet_256_vocab.txt",
        ),
    ),
    ModelPackage(
        id="marblenet_vad",
        display_name="MarbleNet VAD converted layout",
        target_directory="marblenet_vad",
        source=ConverterSource(
            kind="nemo_archive",
            description="Download and convert the official NeMo archive into framework-friendly safetensors and sidecars.",
            url="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/vad_multilingual_frame_marblenet/versions/1.20.0/files/vad_multilingual_frame_marblenet.nemo",
            output_weights_file="marblenet_vad.safetensors",
            config_kind="marblenet",
        ),
        required_files=(
            "marblenet_vad.safetensors",
            "marblenet_vad_config.json",
            "marblenet_vad_labels.txt",
        ),
    ),
    ModelPackage(
        id="voxcpm2",
        display_name="VoxCPM2",
        target_directory="VoxCPM2",
        source=SnapshotSource(repo_id="OpenBMB/VoxCPM2"),
        required_files=(
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "audiovae.pth",
            "audiovae.safetensors",
        ),
    ),
    ModelPackage(
        id="voxcpm2_audiovae",
        display_name="VoxCPM2 AudioVAE local conversion utility",
        target_directory="VoxCPM2",
        source=ConverterSource(
            kind="pytorch_to_safetensors",
            description="Convert a local PyTorch checkpoint into safetensors.",
        ),
        required_files=("audiovae.safetensors",),
        description="Utility only. Use voxcpm2 for the full framework-ready VoxCPM2 package.",
    ),
    ModelPackage(
        id="htdemucs",
        display_name="HTDemucs",
        target_directory="htdemucs",
        source=ConverterSource(
            kind="demucs_reference",
            description="Download official Demucs .th checkpoints, or use --source-dir, and assemble framework safetensors plus a manifest.",
        ),
        required_files=(
            "manifest.json",
            "955717e8/config.json",
            "955717e8/model.safetensors",
        ),
    ),
)

PACKAGE_BY_ID = {package.id: package for package in CATALOG}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or convert official model packages into a local models layout."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available model packages.")
    list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    info_parser = subparsers.add_parser("info", help="Show one package.")
    info_parser.add_argument("package_id")
    info_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    install_parser = subparsers.add_parser("install", help="Install one package into a models root.")
    install_parser.add_argument("package_id")
    install_parser.add_argument("--models-root", default="models", help="Target models root.")
    install_parser.add_argument("--overwrite", action="store_true", help="Replace an existing installed directory.")
    install_parser.add_argument("--source-file", help="Local source checkpoint for utility packages.")
    install_parser.add_argument("--output-file", help="Optional output file for single-file utility packages.")
    install_parser.add_argument("--source-dir", help="Optional local source directory for packages that can use checkpoint folders.")
    install_parser.add_argument("--variant", help="Variant or model name for packages that support variants.")

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


_FAMILY_ID_STRIP_PATTERNS = (
    re.compile(r"_\d+_\d+b(?:_base|_custom_voice|_voice_design)?$"),
    re.compile(r"_\d+b(?:_base|_custom_voice|_voice_design)?$"),
    re.compile(r"_\d+m(?:_bf16)?$"),
    re.compile(r"_bf16$"),
    re.compile(r"_\d+spk_v\d+$"),
    re.compile(r"_\d+hz_\d+k_v\d+$"),
    re.compile(r"_3_small_(?:music|sfx)$"),
    re.compile(r"_3_medium$"),
    re.compile(r"_v\d+(?:_\d+)?$"),
    re.compile(r"_\d+$"),
    re.compile(r"_model$"),
    re.compile(r"_hf$"),
    re.compile(r"_12hz$"),
    re.compile(r"_(?:base|custom_voice|voice_design)$"),
)


def _default_family_from_package_id(package_id: str) -> str:
    """Strip size/version suffixes from a package id. Prefer ``ModelPackage.family`` for exceptions."""
    key = str(package_id or "").strip().lower()
    if not key:
        return ""
    changed = True
    while changed and key:
        changed = False
        for pattern in _FAMILY_ID_STRIP_PATTERNS:
            updated = pattern.sub("", key)
            if updated and updated != key:
                key = updated
                changed = True
                break
    return key


def _default_tasks_from_family(family: str) -> list[str]:
    key = str(family or "").strip().lower()
    if not key:
        return []
    if "forced_aligner" in key or key.endswith("_aligner") or key.endswith("_align"):
        return ["align"]
    if key.endswith("_asr") or key.endswith("_stt") or key in {"parakeet_tdt", "whisper", "voxtral_realtime"}:
        return ["asr"]
    if "vad" in key:
        return ["vad"]
    if "diar" in key:
        return ["diar"]
    if any(token in key for token in ("demucs", "roformer", "separator")):
        return ["sep"]
    if key in {"stable_audio", "ace_step", "heartmula"} or key.endswith("_gen"):
        return ["gen"]
    if key.endswith("_vc") or key in {"seed_vc", "vevo2"}:
        return ["vc"]
    if key.endswith("_codec") or key == "miocodec":
        return ["codec"]
    if key.endswith("_asr") or key.endswith("_stt"):
        return ["asr"]
    if "tts" in key or key in {
        "kokoro",
        "kokoro_tts",
        "chatterbox",
        "voxcpm2",
        "omnivoice",
        "supertonic",
        "miotts",
        "pocket_tts",
        "irodori_tts",
        "index_tts2",
        "vibevoice",
        "moss_tts_nano",
        "moss_tts_local",
    }:
        return ["tts"]
    return []


def _collect_package_repo_ids(package: ModelPackage) -> list[str]:
    source = package.source
    repos: list[str] = []
    if isinstance(source, SnapshotSource):
        repos.append(source.repo_id)
    elif isinstance(source, CompositeSnapshotSource):
        for placement in source.placements:
            repos.append(placement.source.repo_id)
    return [repo for repo in repos if isinstance(repo, str) and repo]


def _package_is_gated(package: ModelPackage) -> bool:
    if isinstance(package.gated, bool):
        return package.gated
    repos = [repo.lower() for repo in _collect_package_repo_ids(package)]
    return any(any(marker in repo for marker in _GATED_REPO_MARKERS) for repo in repos)


def _package_standalone_fields(package: ModelPackage) -> tuple[bool, str | None]:
    if isinstance(package.standalone, bool):
        return package.standalone, package.parent_package_id
    desc = str(package.description or "").strip().lower()
    kind = package_install_kind(package)
    if desc.startswith("subcomponent only.") or desc.startswith("utility only.") or kind == "utility":
        parent = package.parent_package_id
        if not parent:
            for token in desc.replace(".", " ").replace(",", " ").split():
                if token in PACKAGE_BY_ID and token != package.id:
                    parent = token
                    break
        return False, parent
    if "tokenizer" in package.id.lower() or "audiovae" in package.id.lower():
        return False, package.parent_package_id
    return True, None


def package_payload(package: ModelPackage) -> dict[str, object]:
    source = package.source
    if isinstance(source, SnapshotSource):
        source_payload: dict[str, object] = {
            "kind": "huggingface_snapshot",
            "repo_id": source.repo_id,
            "revision": source.revision,
            "include_prefixes": list(source.include_prefixes),
            "include_suffixes": list(source.include_suffixes),
            "exclude_prefixes": list(source.exclude_prefixes),
        }
        installable = True
    elif isinstance(source, CompositeSnapshotSource):
        source_payload = {
            "kind": "composite_snapshot",
            "placements": [
                {
                    "repo_id": placement.source.repo_id,
                    "revision": placement.source.revision,
                    "target_subdir": placement.target_subdir,
                    "required_files": list(placement.required_files),
                    "include_prefixes": list(placement.source.include_prefixes),
                    "include_suffixes": list(placement.source.include_suffixes),
                    "exclude_prefixes": list(placement.source.exclude_prefixes),
                }
                for placement in source.placements
            ],
            "repo_ids": _collect_package_repo_ids(package),
        }
        installable = True
    elif isinstance(source, ConverterSource):
        utility = source.kind in UTILITY_CONVERTER_KINDS
        source_payload = {
            "kind": "utility" if utility else "composite",
            "operation_kind": source.kind,
            "description": source.description,
        }
        if source.url:
            source_payload["url"] = source.url
        if source.output_weights_file:
            source_payload["output_weights_file"] = source.output_weights_file
        if source.config_kind:
            source_payload["config_kind"] = source.config_kind
        installable = True
    else:
        source_payload = {
            "kind": "unsupported",
            "reason": source.reason,
        }
        installable = False
    family = package.family or _default_family_from_package_id(package.id)
    tasks = list(package.tasks) if package.tasks else _default_tasks_from_family(family)
    standalone, parent_package_id = _package_standalone_fields(package)
    return {
        "id": package.id,
        "display_name": package.display_name,
        "target_directory": package.target_directory,
        "description": package.description,
        "installable": installable,
        "install_kind": package_install_kind(package),
        "usage_examples": package_usage_examples(package),
        "required_files": list(package.required_files),
        "source": source_payload,
        "family": family,
        "tasks": tasks,
        "modes": ["offline"] if tasks else [],
        "standalone": standalone,
        "parent_package_id": parent_package_id,
        "gated": _package_is_gated(package),
    }


def print_package_table(packages: Iterable[ModelPackage]) -> None:
    rows = []
    for package in packages:
        status = package_install_kind(package)
        rows.append((package.id, status, package.target_directory, package.display_name))
    width_id = max(len("id"), *(len(row[0]) for row in rows))
    width_status = max(len("type"), *(len(row[1]) for row in rows))
    width_target = max(len("target"), *(len(row[2]) for row in rows))
    print(f"{'id':<{width_id}}  {'type':<{width_status}}  {'target':<{width_target}}  name")
    for row in rows:
        print(f"{row[0]:<{width_id}}  {row[1]:<{width_status}}  {row[2]:<{width_target}}  {row[3]}")


def hf_tree_url(source: SnapshotSource) -> str:
    revision = quote(source.revision, safe="")
    repo_id = source.repo_id
    return f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"


def hf_resolve_url(source: SnapshotSource, relative_path: str) -> str:
    revision = quote(source.revision, safe="")
    path = quote(relative_path, safe="/")
    return f"https://huggingface.co/{source.repo_id}/resolve/{revision}/{path}"


def http_json(url: str) -> object:
    request = Request(url, headers=http_headers())
    with urlopen(request) as response:
        return json.load(response)


def list_hf_files(source: SnapshotSource) -> list[tuple[str, int | None]]:
    payload = http_json(hf_tree_url(source))
    if not isinstance(payload, list):
        raise RuntimeError(f"unexpected HuggingFace tree payload for {source.repo_id}")
    files: list[tuple[str, int | None]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        entry_type = entry.get("type")
        if not isinstance(path, str) or entry_type != "file":
            continue
        if source.include_prefixes and not any(path.startswith(prefix) for prefix in source.include_prefixes):
            continue
        if source.include_suffixes and not any(path.endswith(suffix) for suffix in source.include_suffixes):
            continue
        if any(path.startswith(prefix) for prefix in source.exclude_prefixes):
            continue
        size = entry.get("size")
        files.append((path, size if isinstance(size, int) else None))
    if not files:
        raise RuntimeError(f"no installable files found for {source.repo_id}")
    return files


def download_file(url: str, target: Path, expected_size: int | None) -> int:
    request = Request(url, headers=http_headers())
    with urlopen(request) as response, target.open("wb") as handle:
        written = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
    if expected_size is not None and written != expected_size:
        raise RuntimeError(f"downloaded size mismatch for {target}: {written} != {expected_size}")
    return written


def validate_required_files(package: ModelPackage, root: Path) -> None:
    missing = [relative for relative in package.required_files if not (root / relative).exists()]
    if missing:
        raise RuntimeError(f"installed package is missing required files: {missing}")


def validate_composite_required_files(package: ModelPackage, staged_root: Path, final_root: Path) -> None:
    missing = [
        relative
        for relative in package.required_files
        if not normalized_join(staged_root, relative).exists() and not normalized_join(final_root, relative).exists()
    ]
    if missing:
        raise RuntimeError(f"installed package is missing required files: {missing}")


def validate_required_files_list(required_files: Iterable[str], root: Path, label: str) -> None:
    missing = [relative for relative in required_files if not (root / relative).exists()]
    if missing:
        raise RuntimeError(f"{label} is missing required files: {missing}")


def install_snapshot_into_dir(
    source: SnapshotSource,
    destination_root: Path,
    required_files: Iterable[str],
    *,
    validate: bool = True,
) -> None:
    files = list_hf_files(source)
    for relative, expected_size in files:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {relative}")
        download_file(hf_resolve_url(source, relative), destination, expected_size)
    if validate:
        validate_required_files_list(required_files, destination_root, source.repo_id)


def write_vevo2_whisper_stats(target_path: Path, source_file: Path) -> None:
    payload = torch.load(source_file, map_location="cpu", weights_only=False)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    if set(tensors.keys()) != {"mean", "std"}:
        raise RuntimeError(f"Vevo2 whisper stats must contain mean/std only: {sorted(tensors.keys())}")
    save_file(tensors, str(target_path))


def whisper_dims(payload: dict[str, Any]) -> dict[str, int]:
    raw_dims = payload.get("dims")
    if not isinstance(raw_dims, dict):
        raise RuntimeError("Whisper checkpoint is missing dims")
    dims: dict[str, int] = {}
    for key, value in raw_dims.items():
        if not isinstance(key, str) or not isinstance(value, int):
            raise RuntimeError(f"invalid Whisper dim entry: {key!r}={value!r}")
        dims[key] = value
    return dims


def whisper_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    raw_state = payload.get("model_state_dict")
    if not isinstance(raw_state, dict):
        raise RuntimeError("Whisper checkpoint is missing model_state_dict")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str):
            raise RuntimeError(f"invalid non-string Whisper tensor key: {key!r}")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"invalid non-tensor Whisper state entry: {key}")
        state[key] = value.detach().cpu().contiguous()
    return state


def whisper_medium_config(dims: dict[str, int], tensor_count: int) -> dict[str, Any]:
    return {
        "model_type": "whisper",
        "variant": "medium",
        "source_format": "openai_whisper_pt",
        "weight_file": "model.safetensors",
        "tensor_count": tensor_count,
        "dims": dims,
        "audio_encoder": {
            "n_mels": dims["n_mels"],
            "n_audio_ctx": dims["n_audio_ctx"],
            "n_audio_state": dims["n_audio_state"],
            "n_audio_head": dims["n_audio_head"],
            "n_audio_layer": dims["n_audio_layer"],
        },
        "text_decoder": {
            "n_vocab": dims["n_vocab"],
            "n_text_ctx": dims["n_text_ctx"],
            "n_text_state": dims["n_text_state"],
            "n_text_head": dims["n_text_head"],
            "n_text_layer": dims["n_text_layer"],
        },
    }


def install_whisper_medium_dependency(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ckpt_path = root / "medium.pt"
    model_path = root / "model.safetensors"
    config_path = root / "config.json"
    download_file(WHISPER_MEDIUM_PT_URL, ckpt_path, None)
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected Whisper checkpoint payload type: {type(payload)}")
        dims = whisper_dims(payload)
        state = whisper_state(payload)
        save_file(state, str(model_path))
        reloaded = load_file(str(model_path))
        if set(reloaded.keys()) != set(state.keys()):
            raise RuntimeError("saved Whisper safetensors key set does not match source checkpoint")
        for key, value in state.items():
            if not torch.equal(value, reloaded[key]):
                raise RuntimeError(f"saved Whisper safetensors changed tensor: {key}")
        config_path.write_text(
            json.dumps(whisper_medium_config(dims, len(state)), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    finally:
        ckpt_path.unlink(missing_ok=True)


def prepare_vevo2_snapshot_layout(root: Path) -> None:
    stats_dirs = (
        root / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa",
        root / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa_text",
    )
    if all((stats_dir / "whisper_stats.safetensors").is_file() for stats_dir in stats_dirs):
        return
    for stats_dir in stats_dirs:
        stats_pt = stats_dir / "whisper_stats.pt"
        if not stats_pt.is_file():
            raise RuntimeError(f"Vevo2 snapshot is missing {stats_pt}")
        write_vevo2_whisper_stats(stats_dir / "whisper_stats.safetensors", stats_pt)


def convert_voxcpm2_audiovae(root: Path) -> None:
    input_path = root / "audiovae.pth"
    output_path = root / "audiovae.safetensors"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    write_checked_safetensors(tensors, output_path, input_path, overwrite=True)


def convert_moss_tts_weights(root: Path) -> None:
    input_path = root / "pytorch_model.bin"
    output_path = root / "model.safetensors"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    write_checked_safetensors(tensors, output_path, input_path, overwrite=True)


def convert_ace_step_silence_latent(root: Path) -> None:
    input_path = root / "silence_latent.pt"
    output_path = root / "silence_latent.safetensors"
    tensor = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(f"ACE-Step silence latent must be a tensor: {input_path}")
    write_checked_safetensors({"silence_latent": tensor.detach().cpu().contiguous()}, output_path, input_path, overwrite=True)


def convert_irodori_dacvae_weights(root: Path) -> None:
    input_path = root / "weights.pth"
    output_path = root / "weights.safetensors"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    write_checked_safetensors(tensors, output_path, input_path, overwrite=True)


def convert_outetts_dac_weights(root: Path) -> None:
    input_path = root / "weights_24khz_1.5kbps_v1.0.pth"
    output_path = root / "model.safetensors"
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    expected = {
        "quantizer.quantizers.0.codebook.weight": (1024, 8),
        "quantizer.quantizers.1.codebook.weight": (1024, 8),
        "decoder.model.0.weight_v": (1536, 1024, 7),
        "decoder.model.6.weight_v": (1, 96, 7),
    }
    for name, shape in expected.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise RuntimeError(f"unexpected OuteTTS DAC tensor {name}: {getattr(tensors.get(name), 'shape', None)}")
    write_checked_safetensors(tensors, output_path, input_path, overwrite=True)


def write_irodori_model_config(root: Path) -> None:
    input_path = root / "model.safetensors"
    output_path = root / "model_config.json"
    with safe_open(input_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    config_json = metadata.get("config_json")
    if config_json is None:
        raise RuntimeError(f"Irodori-TTS model.safetensors is missing config_json metadata: {input_path}")
    output_path.write_text(config_json, encoding="utf-8")


def copy_bundled_model_manager_assets(asset_subdir: str, destination_root: Path, required_files: Iterable[str]) -> None:
    source_root = MODEL_MANAGER_ASSETS / asset_subdir
    validate_required_files_list(required_files, source_root, f"bundled model-manager assets {asset_subdir}")
    for relative in required_files:
        source_path = source_root / relative
        destination_path = destination_root / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def install_snapshot(package: ModelPackage, source: SnapshotSource, models_root: Path, overwrite: bool) -> Path:
    target_dir = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{package.target_directory}.", dir=staging_root))
    try:
        pre_validate_files = tuple(relative for relative in package.required_files if relative != "audiovae.safetensors")
        install_snapshot_into_dir(source, staging_dir, pre_validate_files, validate=package.id != "voxcpm2")
        if package.id == "voxcpm2":
            convert_voxcpm2_audiovae(staging_dir)
            validate_required_files(package, staging_dir)
        elif package.id == "moss_tts_nano_100m_model":
            convert_moss_tts_weights(staging_dir)
        if target_dir.exists():
            if not overwrite:
                raise RuntimeError(f"model directory already exists: {target_dir}")
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(target_dir)
        return target_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass


def normalized_join(base: Path, relative: str) -> Path:
    if not relative:
        return base
    return Path(os.path.normpath(str(base / relative)))


def install_composite_snapshot(
    package: ModelPackage,
    source: CompositeSnapshotSource,
    models_root: Path,
    overwrite: bool,
) -> Path:
    package_root = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_bundle = Path(tempfile.mkdtemp(prefix=f"{package.target_directory}.", dir=staging_root))
    staged_roots: dict[Path, Path] = {}
    try:
        staged_package_root = staging_bundle / package.target_directory
        for placement in source.placements:
            destination_root = normalized_join(staged_package_root, placement.target_subdir)
            final_root = normalized_join(package_root, placement.target_subdir)
            if final_root.exists() and not overwrite:
                validate_required_files_list(placement.required_files, final_root, str(final_root))
                continue
            destination_root.mkdir(parents=True, exist_ok=True)
            install_snapshot_into_dir(placement.source, destination_root, placement.required_files)
            staged_roots[final_root] = destination_root

        if package.id == "vevo2":
            whisper_root = staging_bundle / "whisper-medium"
            install_whisper_medium_dependency(whisper_root)
            staged_roots[package_root.parent / "whisper-medium"] = whisper_root
            prepare_vevo2_snapshot_layout(staged_package_root)
        elif package.id == "ace_step":
            convert_ace_step_silence_latent(staged_package_root / "acestep-v15-turbo")
            convert_ace_step_silence_latent(staged_package_root / "acestep-v15-base")
        elif package.id == "moss_tts_nano_100m":
            convert_moss_tts_weights(staged_package_root)
        elif package.id in {"irodori_tts_500m_v3", "irodori_tts_600m_v3_voice_design"}:
            write_irodori_model_config(staged_package_root)
            dacvae_root = staged_package_root.parent / "Semantic-DACVAE-Japanese-32dim"
            if dacvae_root.exists():
                convert_irodori_dacvae_weights(dacvae_root)
        elif package.id == "outetts_1_0_1b":
            dac_root = staged_package_root.parent / "DAC.speech.v1.0"
            if dac_root.exists():
                convert_outetts_dac_weights(dac_root)
        elif package.id == "vibevoice_asr":
            copy_bundled_model_manager_assets(
                "vibevoice_1_5b",
                staged_package_root,
                ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
            )
        elif package.id in {"vibevoice_1_5b", "vibevoice_7b"}:
            # VibeVoice 1.5B and 7B share the same Qwen2.5 tokenizer, and neither
            # upstream repo ships the tokenizer files, so both reuse one bundle.
            copy_bundled_model_manager_assets(
                "vibevoice_1_5b",
                staged_package_root,
                ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
            )
        validate_composite_required_files(package, staged_package_root, package_root)

        top_level_roots: list[Path] = []
        for final_root in sorted(staged_roots.keys(), key=lambda path: len(path.parts)):
            if any(final_root.is_relative_to(existing) for existing in top_level_roots):
                continue
            top_level_roots.append(final_root)

        for final_root in sorted(top_level_roots, key=lambda path: len(path.parts), reverse=True):
            if final_root.exists():
                if not overwrite:
                    raise RuntimeError(f"model directory already exists: {final_root}")
                shutil.rmtree(final_root)

        for final_root in sorted(top_level_roots, key=lambda path: len(path.parts)):
            destination_root = staged_roots[final_root]
            final_root.parent.mkdir(parents=True, exist_ok=True)
            destination_root.rename(final_root)
        return package_root
    except Exception:
        shutil.rmtree(staging_bundle, ignore_errors=True)
        raise
    finally:
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass


def checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a dict, got {type(payload).__name__}")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"checkpoint state_dict must be a dict, got {type(state).__name__}")
    return state


def install_demucs_import_stubs() -> None:
    if "demucs.htdemucs" not in sys.modules:
        demucs_pkg = sys.modules.get("demucs")
        if demucs_pkg is None:
            demucs_pkg = types.ModuleType("demucs")
            demucs_pkg.__path__ = []  # type: ignore[attr-defined]
            sys.modules["demucs"] = demucs_pkg
        htdemucs = types.ModuleType("demucs.htdemucs")

        class HTDemucs:
            pass

        htdemucs.HTDemucs = HTDemucs
        sys.modules["demucs.htdemucs"] = htdemucs
        setattr(demucs_pkg, "htdemucs", htdemucs)

    if "openunmix.filtering" not in sys.modules:
        filtering = types.ModuleType("openunmix.filtering")

        def _wiener_stub(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("openunmix.filtering.wiener stub should not be called during Demucs conversion")

        filtering.wiener = _wiener_stub
        openunmix = types.ModuleType("openunmix")
        openunmix.filtering = filtering
        sys.modules["openunmix"] = openunmix
        sys.modules["openunmix.filtering"] = filtering

    if "julius" not in sys.modules:
        julius = types.ModuleType("julius")

        def _resample_frac_stub(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("julius.resample_frac stub should not be called during Demucs conversion")

        julius.resample_frac = _resample_frac_stub
        sys.modules["julius"] = julius

    if "dora.log" not in sys.modules:
        log = types.ModuleType("dora.log")

        def _fatal_stub(message: str) -> None:
            raise RuntimeError(message)

        log.fatal = _fatal_stub
        dora = types.ModuleType("dora")
        dora.log = log
        sys.modules["dora"] = dora
        sys.modules["dora.log"] = log


def load_demucs_package(path: Path) -> dict[str, Any]:
    install_demucs_import_stubs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        package = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(package, dict):
        raise RuntimeError(f"unexpected Demucs package type in {path}: {type(package)}")
    required = {"klass", "args", "kwargs", "state"}
    missing = required.difference(package)
    if missing:
        raise RuntimeError(f"Demucs package {path} is missing keys: {sorted(missing)}")
    return package


def find_demucs_checkpoint(repo_dir: Path, signature: str) -> Path:
    matches = sorted(repo_dir.glob(f"{signature}*.th"))
    if not matches:
        raise RuntimeError(f"could not find checkpoint for signature '{signature}' in {repo_dir}")
    if len(matches) > 1:
        exact = [path for path in matches if path.stem == signature or path.stem.startswith(signature + "-")]
        if len(exact) == 1:
            return exact[0]
        raise RuntimeError(f"ambiguous checkpoints for signature '{signature}' in {repo_dir}: {matches}")
    return matches[0]


def normalize_manifest_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, fractions.Fraction):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [normalize_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_manifest_value(item) for key, item in value.items()}
    return repr(value)


def export_demucs_model_config(package: dict[str, Any], checkpoint_path: Path, signature: str) -> dict[str, Any]:
    klass = package["klass"]
    kwargs = {str(key): normalize_manifest_value(value) for key, value in package["kwargs"].items()}
    return {
        "model_type": "demucs",
        "class_name": getattr(klass, "__name__", str(klass)),
        "signature": signature,
        "checkpoint_file": checkpoint_path.name,
        "args": [normalize_manifest_value(value) for value in package["args"]],
        "kwargs": kwargs,
        "sources": normalize_manifest_value(kwargs.get("sources")),
        "samplerate": normalize_manifest_value(kwargs.get("samplerate")),
        "audio_channels": normalize_manifest_value(kwargs.get("audio_channels")),
        "segment": normalize_manifest_value(kwargs.get("segment")),
    }


def export_demucs_state_tensors(package: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = package["state"]
    if not isinstance(state, dict):
        raise RuntimeError(f"unexpected Demucs state payload type: {type(state)}")
    converted: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "__quantized":
            raise RuntimeError("Demucs quantized checkpoints are not supported by this converter yet")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"unexpected non-tensor entry in Demucs state: {key} -> {type(value)}")
        converted[str(key)] = value.detach().cpu().contiguous()
    return converted


def load_demucs_bag_manifest(name: str) -> dict[str, Any] | None:
    payload = DEMUCS_BAGS.get(name)
    if payload is None:
        return None
    return dict(payload)


def demucs_variant_signatures(name: str) -> list[str]:
    bag = load_demucs_bag_manifest(name)
    if bag is None:
        return [name]
    signatures = bag.get("models")
    if not isinstance(signatures, list) or not signatures:
        raise RuntimeError(f"Demucs bag manifest has invalid model list: {name}")
    return [str(signature) for signature in signatures]


def ensure_demucs_checkpoints(repo_dir: Path, name: str) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    for signature in demucs_variant_signatures(name):
        if any(repo_dir.glob(f"{signature}*.th")):
            continue
        url = DEMUCS_REMOTE_MODELS.get(signature)
        if url is None:
            raise RuntimeError(f"no remote checkpoint URL found for Demucs signature: {signature}")
        filename = Path(url).name
        print(f"download {filename}")
        download_file(url, repo_dir / filename, None)


def convert_demucs_single(signature: str, repo_dir: Path, output_root: Path) -> dict[str, Any]:
    checkpoint_path = find_demucs_checkpoint(repo_dir, signature)
    package = load_demucs_package(checkpoint_path)
    state = export_demucs_state_tensors(package)
    model_dir = output_root / signature
    model_dir.mkdir(parents=True, exist_ok=True)
    save_file(state, str(model_dir / "model.safetensors"))
    write_json(model_dir / "config.json", export_demucs_model_config(package, checkpoint_path, signature))
    return {
        "signature": signature,
        "checkpoint_file": checkpoint_path.name,
        "output_dir": str(model_dir.relative_to(output_root)),
        "tensor_count": len(state),
    }


def install_demucs_reference(
    package: ModelPackage,
    models_root: Path,
    overwrite: bool,
    source_dir_arg: str | None,
    variant_arg: str | None,
) -> Path:
    name = variant_arg or "htdemucs"
    target_dir = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="htdemucs.", dir=staging_root))
    try:
        if source_dir_arg:
            repo_dir = resolve_path(source_dir_arg)
            if not repo_dir.is_dir():
                raise RuntimeError(f"checkpoint repo directory does not exist: {repo_dir}")
        else:
            repo_dir = staging_dir / "downloaded_checkpoints"
            ensure_demucs_checkpoints(repo_dir, name)
        bag = load_demucs_bag_manifest(name)
        if bag is None:
            converted = [convert_demucs_single(name, repo_dir, staging_dir)]
            manifest = {
                "model_type": "demucs_single",
                "name": name,
                "model": converted[0],
            }
        else:
            signatures = bag["models"]
            if not isinstance(signatures, list) or not signatures:
                raise RuntimeError(f"Demucs bag manifest has invalid model list: {name}")
            converted = [convert_demucs_single(str(signature), repo_dir, staging_dir) for signature in signatures]
            weights = normalize_manifest_value(bag.get("weights"))
            segment = normalize_manifest_value(bag.get("segment"))
            if len(converted) == 1 and weights is None and segment is None:
                manifest = {
                    "model_type": "demucs_single_alias",
                    "name": name,
                    "alias_of": converted[0]["signature"],
                    "model": converted[0],
                }
            else:
                manifest = {
                    "model_type": "demucs_bag",
                    "name": name,
                    "models": converted,
                    "weights": weights,
                    "segment": segment,
                }
        write_json(staging_dir / "manifest.json", manifest)
        validate_required_files(package, staging_dir)
        if target_dir.exists():
            if not overwrite:
                raise RuntimeError(f"model directory already exists: {target_dir}")
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(target_dir)
        return target_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass


def archive_member(archive: tarfile.TarFile, name: str) -> bytes:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        normalized = member.name.removeprefix("./")
        if normalized != name:
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            break
        return extracted.read()
    raise RuntimeError(f"missing archive member: {name}")


def tensor_state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    seen_storages: set[int] = set()
    for key, value in state.items():
        if not isinstance(key, str):
            raise RuntimeError(f"checkpoint contains a non-string tensor key: {key!r}")
        if not torch.is_tensor(value):
            raise RuntimeError(f"checkpoint state contains a non-tensor value at key: {key}")
        tensor = value.detach().cpu().contiguous()
        storage_ptr = tensor.untyped_storage().data_ptr()
        if storage_ptr in seen_storages:
            tensor = tensor.clone()
            storage_ptr = tensor.untyped_storage().data_ptr()
        seen_storages.add(storage_ptr)
        tensors[key] = tensor
    if not tensors:
        raise RuntimeError("refusing to write an empty safetensors file")
    return tensors


def write_checked_safetensors(tensors: dict[str, torch.Tensor], output_path: Path, source_file: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"output file already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_format": "pytorch",
        "source_file": str(source_file),
        "tensor_count": str(len(tensors)),
    }
    save_file(tensors, str(output_path), metadata=metadata)
    reloaded = load_file(str(output_path), device="cpu")
    if set(reloaded.keys()) != set(tensors.keys()):
        raise RuntimeError(f"saved safetensors key set does not match source: {output_path}")
    for key, value in tensors.items():
        actual = reloaded[key]
        if actual.shape != value.shape:
            raise RuntimeError(f"saved tensor shape changed for {key}: {tuple(value.shape)} -> {tuple(actual.shape)}")
        if actual.dtype != value.dtype:
            raise RuntimeError(f"saved tensor dtype changed for {key}: {value.dtype} -> {actual.dtype}")
        if not torch.equal(actual, value):
            raise RuntimeError(f"saved tensor value changed for {key}")


def yaml_to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): yaml_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [yaml_to_json(item) for item in value]
    return value


def expect_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a map")
    return {str(key): item for key, item in value.items()}


def map_at(root: dict[str, Any], key: str) -> dict[str, Any]:
    return expect_map(root.get(key), key)


def list_at(root: dict[str, Any], key: str) -> list[Any]:
    value = root.get(key)
    if not isinstance(value, list):
        raise RuntimeError(f"{key} must be a list")
    return value


def int_at(root: dict[str, Any], key: str) -> int:
    value = root.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise RuntimeError(f"{key} must be an integer")


def number_at(root: dict[str, Any], key: str, default: float | None = None) -> float:
    value = root.get(key)
    if value is None and default is not None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"{key} must be a number")


def string_at(root: dict[str, Any], key: str) -> str:
    value = root.get(key)
    if isinstance(value, str):
        return value
    raise RuntimeError(f"{key} must be a string")


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list")
    return [str(item) for item in value]


def jasper_block(
    block: dict[str, Any],
    *,
    flatten_triples: bool,
    include_dropout: bool,
    include_residual_mode: bool,
    se_metadata: str,
) -> dict[str, Any]:
    def maybe_flatten(key: str) -> Any:
        raw = block.get(key)
        if not flatten_triples:
            return raw
        if isinstance(raw, list) and len(raw) == 1:
            return raw[0]
        return raw

    out: dict[str, Any] = {
        "dilation": maybe_flatten("dilation") if block.get("dilation") is not None else 1,
        "filters": block.get("filters"),
        "kernel": maybe_flatten("kernel"),
        "repeat": block.get("repeat"),
        "residual": block.get("residual", False),
        "separable": block.get("separable", False),
        "stride": maybe_flatten("stride") if block.get("stride") is not None else 1,
    }
    if include_dropout and "dropout" in block:
        out["dropout"] = block["dropout"]
    if include_residual_mode:
        out["residual_mode"] = block.get("residual_mode")
    if "se" in block:
        out["se"] = block["se"]
    if se_metadata == "reduction_ratio" and "se" in block:
        out["se_reduction_ratio"] = block.get("se_reduction_ratio", 8)
    if se_metadata == "context_size" and "se_context_size" in block:
        out["se_context_size"] = block["se_context_size"]
    return out


def jasper_blocks(
    blocks: list[Any],
    *,
    flatten_triples: bool,
    include_dropout: bool,
    include_residual_mode: bool,
    se_metadata: str,
) -> list[dict[str, Any]]:
    return [
        jasper_block(
            expect_map(block, "jasper block"),
            flatten_triples=flatten_triples,
            include_dropout=include_dropout,
            include_residual_mode=include_residual_mode,
            se_metadata=se_metadata,
        )
        for block in blocks
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_citrinet_sidecars(root: dict[str, Any], archive: tarfile.TarFile, output_dir: Path) -> None:
    tokenizer_name = "citrinet_256_tokenizer.model"
    vocab_name = "citrinet_256_vocab.txt"
    (output_dir / tokenizer_name).write_bytes(archive_member(archive, "tokenizer.model"))
    (output_dir / vocab_name).write_bytes(archive_member(archive, "vocab.txt"))
    preprocessor = map_at(root, "preprocessor")
    encoder = map_at(root, "encoder")
    decoder = map_at(root, "decoder")
    vocab_size = int_at(decoder, "num_classes")
    config = {
        "blank_id": vocab_size,
        "jasper": jasper_blocks(
            list_at(encoder, "jasper"),
            flatten_triples=True,
            include_dropout=True,
            include_residual_mode=True,
            se_metadata="reduction_ratio",
        ),
        "n_fft": int_at(preprocessor, "n_fft"),
        "n_mels": int_at(preprocessor, "features"),
        "normalize": string_at(preprocessor, "normalize"),
        "num_classes": vocab_size + 1,
        "pad_to": int_at(preprocessor, "pad_to"),
        "sample_rate": int_at(preprocessor, "sample_rate"),
        "target": str(root.get("target", "")),
        "tokenizer_model_file": tokenizer_name,
        "vocab_count": vocab_size,
        "vocab_file": vocab_name,
        "vocab_size": vocab_size,
        "window": string_at(preprocessor, "window"),
        "window_size": number_at(preprocessor, "window_size"),
        "window_stride": number_at(preprocessor, "window_stride"),
    }
    write_json(output_dir / "citrinet_256_config.json", config)


def write_marblenet_sidecars(root: dict[str, Any], output_dir: Path) -> None:
    labels_name = "marblenet_vad_labels.txt"
    labels = string_list(root.get("labels"), "labels")
    (output_dir / labels_name).write_text("\n".join(labels) + "\n", encoding="utf-8")
    preprocessor = map_at(root, "preprocessor")
    encoder = map_at(root, "encoder")
    decoder = map_at(root, "decoder")
    config = {
        "jasper": jasper_blocks(
            list_at(encoder, "jasper"),
            flatten_triples=True,
            include_dropout=False,
            include_residual_mode=False,
            se_metadata="none",
        ),
        "label_count": len(labels),
        "labels": labels,
        "labels_file": labels_name,
        "n_fft": int_at(preprocessor, "n_fft"),
        "n_mels": int_at(preprocessor, "features"),
        "normalize": string_at(preprocessor, "normalize"),
        "num_classes": int_at(decoder, "num_classes"),
        "pad_to": int_at(preprocessor, "pad_to"),
        "sample_rate": int_at(preprocessor, "sample_rate"),
        "target": str(root.get("target", "")),
        "window": string_at(preprocessor, "window"),
        "window_size": number_at(preprocessor, "window_size"),
        "window_stride": number_at(preprocessor, "window_stride"),
    }
    write_json(output_dir / "marblenet_vad_config.json", config)


def write_nemo_sidecars(config_kind: str, yaml_text: str, archive: tarfile.TarFile, output_dir: Path) -> None:
    root = yaml_to_json(yaml.safe_load(yaml_text))
    if not isinstance(root, dict):
        raise RuntimeError("NeMo model_config.yaml root must be a map")
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_kind == "citrinet":
        write_citrinet_sidecars(root, archive, output_dir)
        return
    if config_kind == "marblenet":
        write_marblenet_sidecars(root, output_dir)
        return
    raise RuntimeError(f"unsupported NeMo config kind: {config_kind}")


def install_nemo_archive(package: ModelPackage, source: ConverterSource, models_root: Path, overwrite: bool) -> Path:
    target_dir = models_root / package.target_directory
    staging_root = models_root / ".engine_model_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{package.target_directory}.", dir=staging_root))
    try:
        archive_path = staging_dir / Path(source.url).name
        print(f"download {archive_path.name}")
        download_file(source.url, archive_path, None)
        raw_bytes = archive_path.read_bytes()
        with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:*") as archive:
            weight_bytes = archive_member(archive, "model_weights.ckpt")
            config_bytes = archive_member(archive, "model_config.yaml")
            payload = torch.load(io.BytesIO(weight_bytes), map_location="cpu", weights_only=False)
            state = checkpoint_state_dict(payload)
            tensors = tensor_state_dict(state)
            write_checked_safetensors(
                tensors,
                staging_dir / source.output_weights_file,
                archive_path,
                overwrite=True,
            )
            write_nemo_sidecars(source.config_kind, config_bytes.decode("utf-8"), archive, staging_dir)
        archive_path.unlink()
        validate_required_files(package, staging_dir)
        if target_dir.exists():
            if not overwrite:
                raise RuntimeError(f"model directory already exists: {target_dir}")
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(target_dir)
        return target_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError:
            pass


def install_converter(
    package: ModelPackage,
    source: ConverterSource,
    models_root: Path,
    overwrite: bool,
    source_file_arg: str | None,
    output_file_arg: str | None,
    source_dir_arg: str | None,
    variant_arg: str | None,
) -> Path:
    target_dir = models_root / package.target_directory
    if source.kind == "nemo_archive":
        return install_nemo_archive(package, source, models_root, overwrite)
    if source.kind == "demucs_reference":
        return install_demucs_reference(package, models_root, overwrite, source_dir_arg, variant_arg)
    if source.kind != "pytorch_to_safetensors":
        raise RuntimeError(f"unsupported converter kind: {source.kind}")
    if not source_file_arg:
        raise RuntimeError(f"{package.id} requires --source-file")
    source_file = resolve_path(source_file_arg)
    if not source_file.is_file():
        raise RuntimeError(f"source checkpoint does not exist: {source_file}")
    if output_file_arg:
        output_path = resolve_path(output_file_arg)
    else:
        output_path = target_dir / package.required_files[0]
    payload = torch.load(source_file, map_location="cpu", weights_only=True)
    state = checkpoint_state_dict(payload)
    tensors = tensor_state_dict(state)
    write_checked_safetensors(tensors, output_path, source_file, overwrite)
    if output_path.parent == target_dir:
        validate_required_files(package, target_dir)
    return target_dir


def command_list(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps([package_payload(package) for package in CATALOG], indent=2))
    else:
        print_package_table(CATALOG)
    return 0


def command_info(args: argparse.Namespace) -> int:
    package = PACKAGE_BY_ID.get(args.package_id)
    if package is None:
        raise RuntimeError(f"unknown package id: {args.package_id}")
    payload = package_payload(package)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


def command_install(args: argparse.Namespace) -> int:
    _ensure_install_deps()
    package = PACKAGE_BY_ID.get(args.package_id)
    if package is None:
        raise RuntimeError(f"unknown package id: {args.package_id}")
    models_root = resolve_path(args.models_root)
    models_root.mkdir(parents=True, exist_ok=True)
    source = package.source
    if isinstance(source, UnsupportedSource):
        raise RuntimeError(f"{package.id} is not installable: {source.reason}")
    if isinstance(source, SnapshotSource):
        install_path = install_snapshot(package, source, models_root, args.overwrite)
    elif isinstance(source, CompositeSnapshotSource):
        install_path = install_composite_snapshot(package, source, models_root, args.overwrite)
    else:
        install_path = install_converter(
            package,
            source,
            models_root,
            args.overwrite,
            args.source_file,
            args.output_file,
            args.source_dir,
            args.variant,
        )
    print(f"installed {package.id} -> {install_path}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "list":
            return command_list(args)
        if args.command == "info":
            return command_info(args)
        if args.command == "install":
            return command_install(args)
        raise RuntimeError(f"unsupported command: {args.command}")
    except HTTPError as ex:
        print(f"http error: {ex.code} {ex.reason}", file=sys.stderr)
        return 1
    except Exception as ex:
        print(str(ex), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
