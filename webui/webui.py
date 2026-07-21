"""
audio.cpp WebUI (Route B) — a thin Gradio frontend that proxies to the local
audiocpp_server HTTP API.

Model loading is on demand: instead of preloading a model at server startup, this
WebUI reads models_catalog.json, lets you pick a model, and (re)starts
audiocpp_server with a single-model config only when you actually load/run it.
One model lives in VRAM at a time — picking a different model swaps it.

Uploaded files are saved by Gradio to local temp paths, which we pass to the
server as `voice_ref` / `audio` (frontend + server run on the same machine).

Just launch this (it starts the server for you):
    venv\\Scripts\\python audiocpp-portable\\webui.py

Env overrides:
    AUDIOCPP_BACKEND=gpu|cpu     which bin dir to launch the server from
                                 (default: auto — gpu when an NVIDIA driver and the
                                 gpu server build are both present, else cpu)
    AUDIOCPP_THREADS=N           ggml compute threads (default 1; cpu backend
                                 defaults to all cores minus one)
    AUDIOCPP_SERVER=http://...   talk to an already-running server instead of managing one
    AUDIOCPP_LOAD_TIMEOUT=300    seconds to wait for a model to finish loading
    AUDIOCPP_NO_BROWSER=1        don't open a browser tab
"""
import atexit
import base64
import glob
import io
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
from urllib.parse import urlparse

import numpy as np
import requests
import gradio as gr

try:
    from ui_i18n import (
        LANGUAGE_CHOICES,
        get_language,
        load_language,
        param_spec as localized_param_spec,
        save_language,
        set_language,
        text as ui_text,
    )
except ImportError as _exc:
    # Fall back to the package-qualified name only when ui_i18n itself is not on
    # the path (imported as ``webui.webui`` in tests). A missing *transitive*
    # dependency of ui_i18n — e.g. opencc — must surface as-is instead of being
    # masked by a misleading "no module named webui.ui_i18n".
    if _exc.name not in ("ui_i18n", "webui"):
        raise
    from webui.ui_i18n import (
        LANGUAGE_CHOICES,
        get_language,
        load_language,
        param_spec as localized_param_spec,
        save_language,
        set_language,
        text as ui_text,
    )

# 降噪：屏蔽 Gradio 内部触发、每次请求都会刷屏的 Starlette 弃用告警。
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE.*")


def _silence_proactor_connection_reset():
    """Windows: swallow the benign `ConnectionResetError [WinError 10054]` that
    asyncio's proactor prints when a browser/HTTP connection drops abruptly."""
    if sys.platform != "win32":
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport
    except Exception:
        return
    _orig = _ProactorBasePipeTransport._call_connection_lost

    def _patched(self, exc):
        if isinstance(exc, ConnectionResetError):
            exc = None  # peer reset == normal close; still run the cleanup below
        try:
            return _orig(self, exc)
        except ConnectionResetError:
            # _orig's own sock.shutdown() raced a peer reset (WinError 10054),
            # which skips the rest of its cleanup — finish it here.
            sock = getattr(self, "_sock", None)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                self._sock = None
            server = getattr(self, "_server", None)
            if server is not None:
                try:
                    server._detach()
                except Exception:
                    pass
                self._server = None
            self._called_connection_lost = True

    _ProactorBasePipeTransport._call_connection_lost = _patched


def _silence_h11_content_length_race():
    """Large uploads (e.g. a multi-minute reference wav) can have the browser
    abort/replace an in-flight preview fetch for the same file while uvicorn
    is still streaming its body; h11 then raises LocalProtocolError trying to
    close out that half-sent response. It's a benign race — verified the
    aborted request doesn't affect the server or any other request, the
    browser's follow-up fetch of the same file completes fine — but uvicorn
    logs it as a full "Exception in ASGI application" traceback per occurrence.
    Drop just that one exception type from uvicorn's logger instead of hiding
    all uvicorn.error output."""
    try:
        from h11 import LocalProtocolError
    except Exception:
        return

    class _DropContentLengthRace(logging.Filter):
        def filter(self, record):
            exc = record.exc_info[1] if record.exc_info else None
            msg = str(exc or "")
            if isinstance(exc, LocalProtocolError) and "declared Content-Length" in msg:
                return False
            # uvicorn with httptools raises these RuntimeErrors on the same
            # preview race: "shorter" when the browser aborts/replaces the
            # fetch, "longer" when gradio sized Content-Length off a large
            # upload still being written to disk and the file grew mid-stream.
            # The request is already dead / the browser refetches; do not spam
            # a full ASGI traceback for either direction.
            if isinstance(exc, RuntimeError) and "than Content-Length" in msg:
                return False
            return True

    logging.getLogger("uvicorn.error").addFilter(_DropContentLengthRace())


def _patch_gradio_render_config_race():
    """Gradio 6.19 上游竞态（gradio-app/gradio#9991 只冻结了 blocks、漏了 fns）：
    本页有 7 个 @gr.render 动态参数区，页面加载/模型切换时多个渲染事件并发，
    一个事件在 get_config 里遍历 session 的 blocks/fns 字典的同时，另一个渲染
    正往里注册组件，偶发 "RuntimeError: dictionary changed size during
    iteration"（前端表现为该次渲染丢失/报错）。get_config 是纯只读操作，
    撞上竞态时稍等重读即可收敛。"""
    try:
        BlocksConfig = gr.blocks.BlocksConfig
        orig = BlocksConfig.get_config
    except AttributeError:
        return

    def _get_config_retry(self, renderable=None):
        for _ in range(10):
            try:
                return orig(self, renderable)
            except RuntimeError:
                time.sleep(0.02)
        return orig(self, renderable)

    BlocksConfig.get_config = _get_config_retry


_silence_proactor_connection_reset()
_silence_h11_content_length_race()
_patch_gradio_render_config_race()


def _t(zh, en, language=None, **values):
    """Select UI copy in the active language."""
    return ui_text(zh, en, language=language, **values)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# WebUI-local working dirs, kept next to this file and created on startup.
CONFIG_DIR = os.path.join(HERE, "configs")
OUTPUT_DIR = os.path.join(HERE, "output")
VOICE_DIR = os.path.join(HERE, "voice")
LOG_DIR = os.path.join(HERE, "logs")
for _d in (CONFIG_DIR, OUTPUT_DIR, VOICE_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

INITIAL_LANGUAGE = load_language()

PROMPTS_DIR = VOICE_DIR                                    # built-in / reference voices
CATALOG_PATH = os.path.join(CONFIG_DIR, "models_catalog.json")
MODEL_PARAMS_PATH = os.path.join(CONFIG_DIR, "model_params.json")
REQUIRED_FILES_PATH = os.path.join(CONFIG_DIR, "required_files.json")


# Executable/name conventions differ per OS: Windows binaries carry a .exe suffix, POSIX ones don't.
EXE_SUFFIX = ".exe" if os.name == "nt" else ""
SERVER_EXE_NAME = "audiocpp_server" + EXE_SUFFIX
GGUF_EXE_NAME = "audiocpp_gguf" + EXE_SUFFIX
# The standalone server launcher, named only in messages telling the user which
# script may be holding the port.
SERVER_LAUNCHER = "run_server.bat" if os.name == "nt" else "run_server.sh"


def _cmake_cache_backend(bin_dir):
    """gpu/cpu for a build tree whose directory name doesn't say, read from its
    CMakeCache.txt (GGML_CUDA:BOOL=ON). Returns None when there is no cache to
    read — an installed tree, or a layout we don't recognize."""
    # Walk up from the bin dir to find the build tree's CMakeCache.txt. Single-config
    # generators put the exe in build/bin (cache one level up); multi-config ones
    # (Visual Studio) nest it in build/bin/Release, so the cache sits two levels up.
    cache = None
    d = os.path.dirname(bin_dir)
    for _ in range(4):
        cand = os.path.join(d, "CMakeCache.txt")
        if os.path.isfile(cand):
            cache = cand
            break
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if cache is None:
        return None
    try:
        with open(cache, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("GGML_CUDA:"):
                    return "gpu" if line.rstrip().upper().endswith("ON") else "cpu"
    except OSError:
        return None
    return "cpu"


def _discover_dev_bin_dirs():
    """Locate from-source build outputs, newest-wins per backend.

    Two layouts are in the wild. The one README documents is
    build/<os>-<backend>-<type>/bin (windows-cuda-release, linux-cpu-release, …),
    where the backend is in the directory name. A plain `cmake -B build` instead
    lands in build/bin and says nothing about the backend, so that one is
    classified from its CMakeCache."""
    out = {}
    for backend, keyword in (("gpu", "*cuda*"), ("cpu", "*cpu*")):
        hits = sorted(d for d in glob.glob(os.path.join(PROJECT_ROOT, "build", keyword, "bin"))
                      if os.path.isfile(os.path.join(d, SERVER_EXE_NAME)))
        if hits:
            out[backend] = hits[-1]
    # A plain `cmake -B build` lands in build/bin with single-config generators
    # (Ninja, Makefiles); multi-config generators (Visual Studio, Xcode) nest the
    # exe in a per-config subdir, so also look in build/bin/Release and .../Debug.
    for plain in (os.path.join(PROJECT_ROOT, "build", "bin"),
                  os.path.join(PROJECT_ROOT, "build", "bin", "Release"),
                  os.path.join(PROJECT_ROOT, "build", "bin", "Debug")):
        if not os.path.isfile(os.path.join(plain, SERVER_EXE_NAME)):
            continue
        backend = _cmake_cache_backend(plain)
        # Only fill a backend the named-directory scan didn't already find, so an
        # explicit build/linux-cuda-release still wins over a stale plain build/.
        if backend and backend not in out:
            out[backend] = plain
    return out


DEV_BIN_DIRS = _discover_dev_bin_dirs()


def _dev_server_exe(backend):
    d = DEV_BIN_DIRS.get(backend)
    return os.path.join(d, SERVER_EXE_NAME) if d else ""


def _find_bundle_root():
    """Locate the root that holds models/ (and, when packaged, cpu/ gpu/ tools/).

    Three layouts: a from-source dev tree (binaries under build/, models under
    PROJECT_ROOT/models), the packaged bundle next to webui/, and webui/ copied
    under the bundle for distribution. Override with AUDIOCPP_BUNDLE."""
    env = os.environ.get("AUDIOCPP_BUNDLE")
    if env:
        return env
    for c in (HERE,                                         # webui.py directly in the bundle
              PROJECT_ROOT,                                 # webui/ shipped under the bundle
              os.path.join(PROJECT_ROOT, "audiocpp-portable")):
        if os.path.isdir(os.path.join(c, "gpu")) or os.path.isdir(os.path.join(c, "cpu")):
            return c
    if any(os.path.isfile(_dev_server_exe(b)) for b in DEV_BIN_DIRS):
        return PROJECT_ROOT
    return os.path.join(PROJECT_ROOT, "audiocpp-portable")


BUNDLE_ROOT = _find_bundle_root()

def _detect_backend():
    """Which bundle build (gpu/ or cpu/) to launch. AUDIOCPP_BACKEND=gpu|cuda|cpu
    wins; otherwise auto-detect: gpu when an NVIDIA driver AND the gpu server
    build are both present, else cpu (the server runs fine on the cpu backend,
    just slower and with lower model coverage)."""
    env = os.environ.get("AUDIOCPP_BACKEND", "").strip().lower()
    if env in ("gpu", "cuda"):
        return "gpu"
    if env:
        return env
    if os.name == "nt":
        has_nvidia = os.path.isfile(os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll"))
    else:
        has_nvidia = shutil.which("nvidia-smi") is not None
    if has_nvidia and (os.path.isfile(os.path.join(BUNDLE_ROOT, "gpu", SERVER_EXE_NAME))
                       or os.path.isfile(_dev_server_exe("gpu"))):
        return "gpu"
    if (os.path.isfile(os.path.join(BUNDLE_ROOT, "cpu", SERVER_EXE_NAME))
            or os.path.isfile(_dev_server_exe("cpu"))):
        return "cpu"
    return "gpu"


BACKEND = _detect_backend()
# The server's own backend name: bin dirs are gpu/ vs cpu/, but the server takes
# cuda|cpu|vulkan|metal, and its config default is "cuda" — which a CPU-only
# build rejects at startup, so the temp config must always spell it out.
SERVER_BACKEND = "cuda" if BACKEND == "gpu" else BACKEND
SERVER_EXE = os.path.join(BUNDLE_ROOT, BACKEND, SERVER_EXE_NAME)
if not os.path.isfile(SERVER_EXE) and os.path.isfile(_dev_server_exe(BACKEND)):
    SERVER_EXE = _dev_server_exe(BACKEND)
LOG_PATH = os.path.join(LOG_DIR, "audiocpp_server_webui.log")
LOAD_TIMEOUT = int(os.environ.get("AUDIOCPP_LOAD_TIMEOUT", "300"))
GGUF_TYPES = ("orig", "f16", "bf16", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k", "q6_k")

# Only families with a model package spec can load the runtime GGUF produced by
# audiocpp_gguf.  Keep this aligned with model_specs/*.json; a safetensors file
# by itself is not evidence that the corresponding C++ loader supports GGUF.
GGUF_NATIVE_FAMILIES = frozenset({
    "citrinet_asr",
    "higgs_audio_stt",
    "hviske_asr",
    "index_tts2",
    "irodori_tts",
    "moss_tts_local",
    "moss_tts_nano",
    "nemotron_asr",
    "omnivoice",
    "qwen3_asr",
    "qwen3_forced_aligner",
    "qwen3_tts",
    "supertonic",
    "vibevoice_asr",
})

# These families have input layouts the WebUI can assemble without guessing.
# Other native-GGUF composite packages remain available through the converter
# CLI until their explicit multi-input layout is added here.
GGUF_SIMPLE_MODEL_FAMILIES = frozenset({
    "higgs_audio_stt",
    "hviske_asr",
    "nemotron_asr",
    "qwen3_asr",
    "qwen3_forced_aligner",
    "vibevoice_asr",
})
GGUF_WEBUI_CONVERTIBLE_FAMILIES = GGUF_SIMPLE_MODEL_FAMILIES | {"qwen3_tts"}


def _find_gguf_exe():
    """Find the converter in a development build or an integrated bundle.

    Keep this separate from SERVER_EXE: developers normally run the executable
    directly from their build tree's bin/, whereas portable users have it beside
    the server binary under gpu/ or cpu/. Dev locations come from the same
    DEV_BIN_DIRS scan the server uses, so non-Windows build-directory names
    (linux-cuda-release, a plain build/bin, …) are covered too; the active
    backend is tried first, then the other one.
    """
    other = "cpu" if BACKEND == "gpu" else "gpu"
    candidates = [
        os.environ.get("AUDIOCPP_GGUF"),
        *(os.path.join(DEV_BIN_DIRS[b], GGUF_EXE_NAME)
          for b in (BACKEND, other) if b in DEV_BIN_DIRS),
        os.path.join(BUNDLE_ROOT, BACKEND, GGUF_EXE_NAME),
        os.path.join(BUNDLE_ROOT, "gpu", GGUF_EXE_NAME),
        os.path.join(BUNDLE_ROOT, "cpu", GGUF_EXE_NAME),
        os.path.join(PROJECT_ROOT, "audiocpp-portable", "gpu", GGUF_EXE_NAME),
        os.path.join(PROJECT_ROOT, "audiocpp-portable", "cpu", GGUF_EXE_NAME),
    ]
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.normpath(candidate)
        key = os.path.normcase(candidate)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(candidate):
            return candidate
    return None

# model_manager_webui.py downloads not-yet-installed models in the background. It
# wraps the upstream tools/model_manager.py CLI with resumable, Torch-free
# downloads (falling back to the upstream tool if the wrapper isn't present).
MODELS_ROOT = os.path.join(BUNDLE_ROOT, "models")


def _find_model_manager():
    for c in (os.environ.get("AUDIOCPP_MODEL_MANAGER"),
              os.path.join(BUNDLE_ROOT, "tools", "model_manager_webui.py"),
              os.path.join(PROJECT_ROOT, "tools", "model_manager_webui.py"),
              os.path.join(BUNDLE_ROOT, "tools", "model_manager.py"),
              os.path.join(PROJECT_ROOT, "tools", "model_manager.py")):
        if c and os.path.isfile(c):
            return c
    return None


MODEL_MANAGER = _find_model_manager()


def _detect_vram_gb():
    """本机 NVIDIA 显卡显存总量（GB，多卡取最大）；无 nvidia-smi/无 N 卡返回 None。
    用于对照 catalog 条目的 min_vram_gb 估算值，提示“下载了也可能跑不动”。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        vals = [float(line) for line in out.stdout.split() if line.strip()]
        return round(max(vals) / 1024, 1) if vals else None
    except Exception:
        return None


LOCAL_VRAM_GB = _detect_vram_gb()


def _vram_shortfall(entry):
    """条目估算显存超过本机显存时返回 (需要GB, 本机GB)，否则 None。"""
    if BACKEND == "cpu":
        return None  # CPU 后端跑在系统内存里，显存对照不适用
    need = entry.get("min_vram_gb")
    if need and LOCAL_VRAM_GB and float(need) > LOCAL_VRAM_GB:
        return float(need), LOCAL_VRAM_GB
    return None

# TTS 语言下拉的语种集合。下拉首位会合成一个 ("Auto", "") 选项（空串=模型默认/
# 自动检测），所以这里不要再放字面量 "Auto"——带 lang_map 的家族会拒绝它，
# 且界面上会出现两个分不清的 Auto。
LANGS = ["", "english", "chinese", "french", "german", "italian",
         "japanese", "korean", "portuguese", "russian", "spanish"]

# Which catalog task tokens each tab can drive. do_tts sends text + optional
# reference voice, which fits both plain TTS ("tts") and voice cloning ("clon").
TTS_TASKS = ("tts", "clon")
ASR_TASKS = ("asr",)
GEN_TASKS = ("gen",)   # music/SFX generation, served via the generic /v1/tasks/run route
VC_TASKS = ("vc", "svc", "s2s")           # 声音转换：源音频 + 目标音色，走 /v1/tasks/run
SEP_TASKS = ("sep",)                      # 音源分离：多轨 named_audio_outputs
ANALYZE_TASKS = ("vad", "diar", "align")  # 音频分析：segments / speaker_turns / words
VDES_TASKS = ("vdes",)                    # 声音设计：文字 + 音色描述，走 /v1/audio/speech

# Per-family behavior for the TTS tab, keyed by catalog `family`. This is how we
# cope with each model wanting different input formats / options: the shared UI
# stays simple, and each family gets its own hint, optional text transform, and
# default request options. A catalog entry can also carry its own `input_hint` /
# `default_options` to override the family profile without editing this file, and
# the "高级参数 (JSON)" box lets you pass ANY model-specific option at run time.
MODEL_PROFILES = {
    "vibevoice": {
        "input_hint": (
            "**VibeVoice** 多说话人脚本：每行 `Speaker N: 内容`（纯文字自动包装）；"
            "多音色用高级参数 `voice_samples`，不要传参考音频。"),
        "wrap_speaker_script": True,
        # VibeVoice has no internal text chunking. VRAM is bounded since the
        # layerwise-prefill/gallocr decode-graph fix, so chunks no longer need to be
        # tiny; 600 chars keeps each chunk's generation inside the default
        # max_tokens=1200 budget (~1.5 frames/CJK char) and the KV capacity tier
        # <= 2048 (~7.1 GB peak on 8 GB GPUs).
        "chunk_chars": 600,
        # Advanced-parameter widgets are sent only when the user edits them, so an
        # untouched num_inference_steps control (displayed 10) silently falls back to
        # the model config's ddpm_num_inference_steps=20 server-side. Send 10 (the
        # official demo default) explicitly; widgets/JSON still override.
        "default_options": {"num_inference_steps": 10},
    },
    "voxcpm2": {
        "input_hint": (
            "**VoxCPM2** 声音克隆：上传干净的单人参考音色并填『参考文本』；长文本自动分段；"
            "支持⚡流式生成（生成模式选『流式』，边生成边播放）。"),
        # 服务端流式（server mode=streaming + /v1/audio/speech stream_format=sse）：
        # delta 事件是 base64 的裸 PCM16，不带采样率——只能客户端自带。取值来自模型
        # config.json 的 audio_vae.out_sample_rate（48000）。流式生成要求显式
        # retry_badcase=false（generator.cpp:1259，重试逻辑与已推流的音频冲突）。
        "supports_streaming": True,
        "stream_sample_rate": 48000,
        # 8G 4060 实测标定（2026-07-05，CLI --log + nvidia-smi 抓峰值）：
        # - 峰值 ≈ 固定基线 + audiovae 解码图。基线（权重+KV+生成图）与文本长度/
        #   max_tokens 无关：max_tokens 128/300/1200 峰值都是 7634MiB（生成会提前
        #   遇停止符，解码图按实际帧数而非 max_tokens 建）。所以 chunk_chars / max_tokens
        #   都压不动基线——真正的杠杆是**权重量化**（catalog session_options 里已设
        #   voxcpm2.weight_type=q8_0：bf16 权重 4.6G，峰值 7634→5549MiB）。
        # - audiovae 解码图仍随每段音频长度涨：容量取「≥latent_frames 的最小 2 的幂」
        #   (audiovae.cpp:724)，~1.28MB/帧。q8_0 下实测 ~53 字→cap512→6366MiB，
        #   ~106 字→cap1024→7722MiB（偏紧）。所以每段要短：60 字→cap≤512→~6.4G，
        #   留 ~1.7G 给其它占 GPU 的程序。想少接缝可上调，但注意 8G 边界。
        "chunk_chars": 60,
    },
    "qwen3_tts": {
        "input_hint": (
            "**Qwen3-TTS** 声音克隆：建议上传参考音色并填『参考文本』，否则可能提前截断。"),
    },
    "pocket_tts": {
        "input_hint": "**PocketTTS**：必须提供参考音色（上传/录制/内置）。",
    },
    "chatterbox": {
        "input_hint": (
            "**Chatterbox** 声音克隆：必须提供参考音色；语言仅支持英/西/法/德/意/葡/韩"
            "（**无中文**），留空=英语。"),
        # Chatterbox validates against 2-letter ISO codes (no zh/ja/ru, no auto),
        # so the shared dropdown's friendly names are translated here; names not
        # listed are genuinely unsupported by the model and rejected up front.
        "lang_map": {
            "english": "en", "spanish": "es", "french": "fr", "german": "de",
            "italian": "it", "portuguese": "pt", "korean": "ko",
        },
    },
    "qwen3_asr": {
        # Encoder cap: max_source_positions=1500 tokens at 13 tokens/second
        # (qwen3_asr_audio_encoder_token_count) -> ~115 s of audio per request.
        # 但 8G 卡上先撞显存：thinker prefill 图随时长超线性膨胀（4060 8G 实测
        # 65s 可过、70s 要 13.2GB、75s 要 14.6GB），所以客户端按 max_input_seconds
        # =60s 在静音处切段逐段转写再拼接，顺带解决 70~115s 音频原本的 OOM。
        "input_hint": ("**Qwen3-ASR**：长音频自动分段转写；"
                       "语种/上下文/对话模式见『转写选项』。"),
        "max_input_seconds": 60,
    },
    "voxtral_realtime": {
        "input_hint": (
            "**Voxtral Mini 4B Realtime**：自动语种转写；支持⚡流式转写，"
            "勾选后按模型原生音频分块边转边出字；不输出时间戳。"),
        "supports_streaming": True,
    },
    "ace_step": {
        "input_hint": (
            "**ACE-Step** 音乐生成/编辑：提示词写风格/乐器/情绪（英文最佳），可填歌词。"
            "编辑类 route 需上传源音频并建议先点『🔍 分析源音频』；参数详解见 webui/README.md。"),
        # 原版 turbo UI 默认 shift=3.0（C++ 端默认 1.0，仅 remix/extract 路由自带 3.0）。
        # 控件只发用户改过的项，所以这里显式发送，保证 UI 显示值=实际值。
        "default_options": {"shift": 3.0},
    },
    "stable_audio": {
        "input_hint": (
            "**Stable Audio**：提示词**仅英文**，不用歌词；"
            "上传源音频可做 init/inpaint（高级参数选 audio_input_kind）。"),
    },
    "heartmula": {
        "input_hint": (
            "**HeartMuLa**：高级参数 `tags` 必填（逗号分隔），『歌词』填唱词；"
            "峰值显存 ~25G，8G 卡跑不动。"),
    },
    "vevo2": {
        "input_hint": (
            "**Vevo2**：源语音 + 目标音色，默认只换音色（保留说话风格）；"
            "风格转换类 route 需 JSON 补 `style_ref` 等，详见 webui/README.md。"
            "长音频自动分段，参考音色自动截 ≤10s。"),
        # 8G 4060 实测标定（2026-07-04/07-05）：FM 图一次建图，序列长度 =
        # 目标音色(prompt) + 源(target) 帧数（均 50fps，见 fm.cpp:782 cond_frames =
        # prompt_frames + target_frames）。cond≈25s（源15s+参考10s）就把 8G 吃满、
        # 峰值溢出到共享显存（idle 已占 7.9G/8G）；cond≈20s 勉强、30s 必炸。所以按
        # (预算 − 参考时长) 反推每段源时长，并把参考截到 ≤ ref_max，令 cond 稳定
        # 落在预算内；各段源仍补零到等长以复用缓存图。带 target_text 的编辑类 route
        # 不适合分段（会把文本对不上），这类输入本身也放不进显存。
        # 权重加载后约占 5.5-6G，留给图的只有 ~2G；18s 源(cond≈21) 只剩 ~350MB，
        # 所以预算取 16（cond≈16，留 ~0.8-1.2G 给峰值/其它占用 GPU 的程序）。
        "vc_chunk_seconds": 15,          # 每段源时长上限（会被显存预算进一步压低）
        "vc_fm_budget_seconds": 16,      # 参考 + 每段源 的总时长预算（8G 安全线）
        "vc_ref_max_seconds": 10,        # 目标音色参考截断上限
        "vc_min_chunk_seconds": 6,       # 每段源时长下限，避免切得过碎
    },
    "seed_vc": {
        "input_hint": (
            "**Seed-VC**：源语音 + 目标音色参考（几秒干净人声）；"
            "默认 v2 路线，v1 旧路线在高级参数切换。"),
    },
    "miocodec": {
        "input_hint": "**MioCodec**：codec 重建式转换——源提供内容，参考提供音色。",
    },
    "htdemucs": {
        "input_hint": "**HTDemucs**：输出 drums / bass / other / vocals 四轨。",
    },
    "mel_band_roformer": {
        "input_hint": "**Mel-Band RoFormer**：输出人声轨 + 伴奏轨。",
    },
    "nemotron_asr": {
        "input_hint": ("**Nemotron ASR**：100+ 语种；支持⚡流式转写"
                       "（勾选后边转边出字，长音频不用干等）。"),
        "supports_streaming": True,
    },
    "higgs_audio_stt": {
        "input_hint": "**Higgs Audio STT**：支持⚡流式转写（勾选后边转边出字）。",
        "supports_streaming": True,
    },
    "vibevoice_asr": {
        "input_hint": "**VibeVoice-ASR**：离线转写，支持自动语种和说话人分段。",
    },
    "silero_vad": {
        "input_hint": "**Silero VAD**：检测音频中的语音段。",
    },
    "marblenet_vad": {
        "input_hint": "**MarbleNet VAD**：帧级语音活动检测，输出语音段列表。",
    },
    "sortformer_diar": {
        "input_hint": "**Sortformer**：说话人分离（谁在何时说话，≤4 人）。",
    },
    "qwen3_forced_aligner": {
        "input_hint": "**Qwen3 强制对齐**：『对齐文本』填音频原文，输出逐词时间戳（≤115 秒）。",
    },
    "index_tts2": {
        "input_hint": (
            "**IndexTTS2** 中/英声音克隆：**必须**提供参考音色（上传/录制/内置）；"
            "情感控制在『高级参数』：emotion_text 填情绪参考文本（如“你吓死我了！”）+ "
            "emotion_alpha 调强度，或 use_emotion_text 从朗读文本自动推断。"),
        # 模型只认 zh/en 语种标签（docs/tts.md），共享下拉的其它语言直接拒绝。
        "lang_map": {"chinese": "zh", "english": "en"},
        "require_voice": True,
    },
    "irodori_tts": {
        "input_hint": (
            "**Irodori-TTS**（日语）：默认无参考直接生成；上传参考音色即自动切换克隆模式。"
            "VoiceDesign 版走『声音设计』标签页，用日语 caption 描述音色。"),
        "lang_map": {"japanese": "ja"},
        # 会话默认 no_ref=true（忽略参考音频），带参考时必须显式关掉才走克隆路径。
        "no_ref_toggle": True,
        # 声音设计标签页的『音色描述』对本家族要发 options.caption（qwen3_tts 走
        # 服务器的 instructions→instruct 映射，irodori session 只读 caption）。
        "vdes_option_key": "caption",
    },
    "moss_tts_local": {
        "input_hint": (
            "**MOSS-TTS-Local**：纯文本直接生成；克隆时上传参考音色并尽量填『参考文本』。"
            "输出 48kHz 立体声；语言下拉可留空（自动）或选择语言作为提示。"),
    },
    "moss_tts_nano": {
        "input_hint": (
            "**MOSS-TTS-Nano** 100M 轻量：无参考=文本续写式生成（音色随机）；"
            "上传参考音色即声音克隆。"),
    },
    "supertonic": {
        "input_hint": (
            "**Supertonic 3** 预置音色多语种 TTS：在『高级参数』选 voice（M1-M5 男声 / "
            "F1-F5 女声）和语速 speaking_rate；支持⚡流式生成；"
            "**不支持**参考音频克隆（**无中文**）。"),
        # Supertonic 的 C++ 会话支持 mode=streaming，并通过 pull events 按文本段
        # 输出音频。SSE delta 是不带采样率的裸 PCM16，客户端需使用模型的 44.1kHz。
        "supports_streaming": True,
        "stream_sample_rate": 44100,
        # 模型收 ISO 语种码（en/ko/ja/...），共享下拉的友好名在此转换；chinese 不在
        # 支持列表所以不映射（选中会被 resolve_language 拒绝并提示）。
        "lang_map": {
            "english": "en", "french": "fr", "german": "de", "italian": "it",
            "japanese": "ja", "korean": "ko", "portuguese": "pt",
            "russian": "ru", "spanish": "es",
        },
    },
}

# Concise English hints.  The Chinese catalog/profile copy remains the detailed
# reference; English intentionally keeps these notes short so model cards stay
# aligned on narrower screens.
MODEL_HINTS_EN = {
    "vibevoice": "**VibeVoice**: use one `Speaker N:` line per speaker. Use `voice_samples` for multiple voices.",
    "voxcpm2": "**VoxCPM2**: upload a clean voice reference and its transcript. Streaming is supported.",
    "qwen3_tts": "**Qwen3-TTS**: a voice reference and matching transcript are recommended.",
    "pocket_tts": "**PocketTTS** requires a voice reference.",
    "chatterbox": "**Chatterbox** requires a voice reference and supports en/es/fr/de/it/pt/ko.",
    "qwen3_asr": "**Qwen3-ASR** automatically splits long audio. Language and context are optional.",
    "voxtral_realtime": "**Voxtral Mini 4B Realtime** auto-detects language and supports streaming transcription. Timestamps are not exposed.",
    "ace_step": "**ACE-Step**: describe style, instruments and mood. Editing routes require source audio.",
    "stable_audio": "**Stable Audio** accepts English prompts only. Source audio enables init/inpaint.",
    "heartmula": "**HeartMuLa** requires `tags` and lyrics. Estimated peak VRAM is about 25 GB.",
    "vevo2": "**Vevo2**: provide source audio and a target voice. Long audio is split automatically.",
    "seed_vc": "**Seed-VC**: provide source audio and a clean target voice reference.",
    "miocodec": "**MioCodec** reconstructs source content with the reference voice.",
    "htdemucs": "**HTDemucs** outputs drums, bass, other and vocals.",
    "mel_band_roformer": "**Mel-Band RoFormer** outputs vocals and accompaniment.",
    "nemotron_asr": "**Nemotron ASR** supports 100+ languages and streaming transcription.",
    "higgs_audio_stt": "**Higgs Audio STT** supports streaming transcription.",
    "vibevoice_asr": "**VibeVoice-ASR** uses offline transcription with automatic language and speaker segmentation.",
    "silero_vad": "**Silero VAD** detects speech segments.",
    "marblenet_vad": "**MarbleNet VAD** detects frame-level speech activity.",
    "sortformer_diar": "**Sortformer** identifies who spoke when (up to four speakers).",
    "qwen3_forced_aligner": "**Qwen3 Forced Aligner** requires the source transcript and returns word timestamps.",
    "index_tts2": "**IndexTTS2** (zh/en) requires a voice reference; emotion controls live in advanced parameters.",
    "irodori_tts": "**Irodori-TTS** (Japanese) works without a reference; uploading one enables voice cloning.",
    "moss_tts_local": "**MOSS-TTS-Local**: plain text works; add a voice reference and its transcript to clone. 48 kHz stereo output.",
    "moss_tts_nano": "**MOSS-TTS-Nano** 100M: continuation mode without a reference, voice clone with one.",
    "supertonic": "**Supertonic 3**: preset voices and streaming are supported; no voice cloning, no Chinese.",
}
# Qwen3-ASR 可强制的语种（模型 config.json 的 support_languages，prompt 里用英文名；
# 留空/Auto = 自动检测）。citrinet 等其它 ASR 族忽略该字段。
QWEN3_ASR_LANGUAGES = [
    ("中文", "Chinese"), ("英语", "English"), ("粤语", "Cantonese"),
    ("日语", "Japanese"), ("韩语", "Korean"), ("俄语", "Russian"),
    ("法语", "French"), ("德语", "German"), ("西班牙语", "Spanish"),
    ("葡萄牙语", "Portuguese"), ("意大利语", "Italian"), ("阿拉伯语", "Arabic"),
    ("印尼语", "Indonesian"), ("泰语", "Thai"), ("越南语", "Vietnamese"),
    ("土耳其语", "Turkish"), ("印地语", "Hindi"), ("马来语", "Malay"),
    ("荷兰语", "Dutch"), ("瑞典语", "Swedish"), ("丹麦语", "Danish"),
    ("芬兰语", "Finnish"), ("波兰语", "Polish"), ("捷克语", "Czech"),
    ("菲律宾语", "Filipino"), ("波斯语", "Persian"), ("希腊语", "Greek"),
    ("罗马尼亚语", "Romanian"), ("匈牙利语", "Hungarian"), ("马其顿语", "Macedonian"),
]

DEFAULT_PROFILE = {"input_hint": "", "input_hint_en": "", "wrap_speaker_script": False,
                   "default_options": {},
                   # C++ 会话实现了 IStreamingVoiceTaskSession 的家族（server 需以
                   # mode=streaming 加载才走流式路由）；见 registry 各家族 loader。
                   "supports_streaming": False,
                   # Families with internal chunking handle long text fine; the client
                   # split only exists to bound each HTTP request (no 900 s timeout)
                   # and surface progress, so the budget can stay coarse.
                   "chunk_chars": 1000}

# One "Speaker N:" line (any speaker index) is enough to treat text as a script.
_SPEAKER_RE = re.compile(r"^\s*Speaker\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)


def profile_for(entry):
    prof = {**DEFAULT_PROFILE, **MODEL_PROFILES.get(entry.get("family", ""), {})}
    prof["input_hint_en"] = MODEL_HINTS_EN.get(entry.get("family", ""), "")
    if entry.get("input_hint"):
        prof["input_hint"] = entry["input_hint"]
    if entry.get("input_hint_en"):
        prof["input_hint_en"] = entry["input_hint_en"]
    if entry.get("default_options"):
        prof["default_options"] = {**prof.get("default_options", {}), **entry["default_options"]}
    return prof


def supports_streaming(model_id):
    entry = catalog_by_id(model_id) if model_id else None
    return bool(entry) and bool(profile_for(entry).get("supports_streaming"))


def asr_stream_update(model_id):
    """Reset and show the ASR streaming toggle for the selected model."""
    return gr.update(visible=supports_streaming(model_id), value=False)


def model_hint_for(model_id, language=None):
    entry = catalog_by_id(model_id) if model_id else None
    if not entry:
        return ""
    language = language or get_language()
    prof = profile_for(entry)
    hint = _t(prof["input_hint"], prof["input_hint_en"], language)
    short = _vram_shortfall(entry)
    if short:
        warn = _t(
            "⚠️ **显存提示**：该模型估算需 **≥{need:g}G** 显存，本机为 **{local:g}G**，运行可能很慢{tail}",
            "⚠️ **VRAM**: estimated **≥{need:g} GB**, detected **{local:g} GB**. Performance may be poor{tail}",
            language, need=short[0], local=short[1],
            tail=(_t("，请谨慎下载。", ".", language) if not entry["installed"] else
                  _t("。", ".", language)))
        hint = warn + ("\n\n" + hint if hint else "")
    return hint


def resolve_language(prof, language):
    """Translate the shared language dropdown into what the selected family
    expects. Most families (Qwen3-TTS, VibeVoice, …) take the UI's friendly
    names as-is, so they have no `lang_map` and the value passes through. A
    family with a restricted language set (Chatterbox: 2-letter ISO codes, no
    Chinese/Japanese/Russian, no auto-detect) supplies a `lang_map`; its names
    are converted and anything it can't do — including "Auto" — is rejected here
    with an actionable message instead of a raw server 500."""
    lang = (language or "").strip()
    lang_map = prof.get("lang_map")
    if not lang_map:
        return lang
    if not lang:
        return ""                       # 留空 = 用模型默认
    code = lang_map.get(lang.lower())
    if code is not None:
        return code
    raise gr.Error(_t(
        "所选模型不支持语言「{selected}」。请改选：{supported}，或选 Auto 使用模型默认。",
        "This model does not support {selected}. Choose {supported}, or choose Auto for the model default.",
        selected=language, supported=" / ".join(lang_map)))


def _as_speaker_script(text):
    """Wrap plain text into `Speaker 0:` lines when it isn't already a script."""
    if _SPEAKER_RE.search(text or ""):
        return text
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(f"Speaker 0: {ln}" for ln in lines) if lines else text


def _vibevoice_punctuate_script(text):
    """VibeVoice is much less stable on tiny lines without sentence punctuation."""
    out = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_LINE_RE.match(line)
        if not m:
            out.append(line)
            continue
        prefix, body = m.group(1), m.group(2).strip()
        if body and body[-1] not in "。！？!?.,;；:":
            body += "。"
        out.append(f"{prefix} {body}" if body else prefix)
    return "\n".join(out) if out else text


def _vibevoice_text_max_tokens(chunk, ui_default=1200):
    """Estimate VibeVoice speech tokens from text, not reference-prompt length."""
    body = re.sub(r"(?im)^\s*Speaker\s+\d+\s*:\s*", "", chunk or "")
    cjk = sum(1 for ch in body if "\u4e00" <= ch <= "\u9fff")
    non_space = sum(1 for ch in body if not ch.isspace())
    non_cjk = max(0, non_space - cjk)
    pauses = sum(1 for ch in body if ch in "，。！？；：,.!?;:")
    estimate = int(cjk * 2.2 + non_cjk * 0.45 + pauses * 3.0 + 8)
    return max(18, min(int(ui_default), estimate))


# VibeVoice 1.5B 对超短脚本从第 1 帧起整段胡言乱语——模型级缺陷（长播客数据训练，
# 短文本 OOD），与参考音色/CFG/扩散步数/中英文/seed 全部无关（2026-07-08 A/B+ASR
# 转写实测：27 字必炸、40 字逐字正确；按上面估算公式 69 token 炸 / 87 token 过）。
# 低于阈值直接拦截；生成出来只会是垃圾，调参数救不了。
_VIBEVOICE_MIN_EST_TOKENS = 85


def _merge_short_vibevoice_tail(chunks):
    """分段时留下的过短尾段同样会胡言乱语：并回前一段。前段最多超预算几十字，
    仍在 max_tokens=1200 封顶的显存包络内。"""
    while len(chunks) > 1 and (
            _vibevoice_text_max_tokens(chunks[-1]) < _VIBEVOICE_MIN_EST_TOKENS):
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + "\n" + tail
    return chunks


# --- client-side long-text chunking ------------------------------------------
# Long text is synthesized as one HTTP request per chunk and concatenated here.
# That keeps every request bounded (no 900 s timeout, works for families without
# internal chunking) and lets the UI show real per-chunk progress.
_SPEAKER_LINE_RE = re.compile(r"^\s*(Speaker\s+\d+\s*:)\s*(.*)$", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^。！？!?；;…]*[。！？!?；;…]+|[^。！？!?；;…]+$")


def _split_long_line(line, budget):
    """Split one overlong line at sentence ends into pieces of <= budget chars,
    re-attaching its `Speaker N:` prefix (if any) to every piece."""
    m = _SPEAKER_LINE_RE.match(line)
    prefix, body = (m.group(1) + " ", m.group(2)) if m else ("", line.strip())
    pieces, cur = [], ""
    for sent in _SENTENCE_RE.findall(body):
        if cur and len(cur) + len(sent) > budget:
            pieces.append(prefix + cur)
            cur = ""
        cur += sent
    if cur:
        pieces.append(prefix + cur)
    return pieces or [line]


def _split_tts_chunks(text, budget):
    """Group non-empty lines into chunks of <= budget chars. A line is never
    split across chunks unless it alone exceeds the budget (then it is split at
    sentence boundaries). Returns a list of chunk strings."""
    units = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue
        units.extend(_split_long_line(ln, budget) if len(ln) > budget else [ln])
    chunks, cur, cur_len = [], [], 0
    for unit in units:
        sep = 1 if cur else 0  # the "\n" join separator counts toward the budget
        if cur and cur_len + sep + len(unit) > budget:
            chunks.append("\n".join(cur))
            cur, cur_len, sep = [], 0, 0
        cur.append(unit)
        cur_len += sep + len(unit)
    if cur:
        chunks.append("\n".join(cur))
    return chunks or ([text] if (text or "").strip() else [])


def _concat_wavs(blobs, out_path, keep_ratios=None):
    """Concatenate same-format WAV byte blobs into one file at out_path.
    keep_ratios[i]：每段只保留前一部分（分段转换把源补零到等长后，
    按有效占比截掉对应输出的尾部静音）。"""
    params, frames = None, []
    for i, blob in enumerate(blobs):
        with wave.open(io.BytesIO(blob)) as w:
            fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            if params is None:
                params = fmt
            elif fmt != params:
                raise gr.Error(_t("分段音频格式不一致：{left} != {right}",
                                  "Audio chunk formats differ: {left} != {right}",
                                  left=fmt, right=params))
            n = w.getnframes()
            if keep_ratios is not None:
                n = max(1, min(n, int(round(n * keep_ratios[i]))))
            frames.append(w.readframes(n))
    with wave.open(out_path, "wb") as w:
        w.setnchannels(params[0])
        w.setsampwidth(params[1])
        w.setframerate(params[2])
        for data in frames:
            w.writeframes(data)


def _audio_duration_seconds(path):
    """Duration of a local audio file, or None when not measurable. wave covers
    WAV, i.e. Gradio mic recordings and the typical uploads here; other formats
    just skip the duration note instead of failing the request."""
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            return (w.getnframes() / float(rate)) if rate else None
    except Exception:
        return None


# 已转码文件缓存：gradio 的临时路径按内容哈希命名，同一上传重复运行不重复转码。
_WAV_CACHE = {}


def _find_ffmpeg():
    """转码用的 ffmpeg：随 webui 分发的 ffmpeg（Windows 下为 ffmpeg.exe）优先，其次 PATH。"""
    bundled = os.path.join(HERE, "ffmpeg" + EXE_SUFFIX)
    if os.path.exists(bundled):
        return bundled
    found = shutil.which("ffmpeg")
    if not found:
        raise gr.Error(_t("找不到 ffmpeg，无法转码非 WAV 音频。",
                          "ffmpeg was not found; non-WAV audio cannot be converted."))
    return found


def _ensure_ascii_path(path):
    """若路径含非 ASCII 字符，复制到纯 ASCII 临时文件。
    C++ server 在 Windows 上无法打开含中文等字符的路径。"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    fd, out = tempfile.mkstemp(prefix="audiocpp_asc_", suffix=".wav")
    os.close(fd)
    shutil.copy2(path, out)
    return out


def _wait_file_stable(path, timeout=8.0, interval=0.08, stable_ticks=3):
    """Wait until a just-uploaded/recorded file stops changing on disk.
    On Windows, the browser can start/restart audio preview fetches while Gradio
    is still replacing a large temp file. Waiting for a few equal size samples
    before copying avoids half-written preview files."""
    if not path:
        return False
    deadline = time.time() + timeout
    last_size = -1
    ticks = 0
    while time.time() < deadline:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size > 0 and size == last_size:
            ticks += 1
            if ticks >= stable_ticks:
                return True
        else:
            last_size = size
            ticks = 0
        time.sleep(interval)
    return os.path.exists(path)


def _safe_audio_ext(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        ext.encode("ascii")
        if 0 < len(ext) <= 8:
            return ext
    except Exception:
        pass
    return ".wav"


def _stage_upload(path, force_copy=False):
    """上传/录制完成后，把输入音频换成短 ASCII 临时名再交回控件。

    关键点：长音频即使路径本身已经是短 ASCII，也强制复制到一个新的稳定文件。
    否则在同一个 gr.Audio 控件里点 X 清空再上传长音频时，旧预览 fetch 与新预览
    fetch 容易竞态，前端波形会卡住；短音频通常因为读取很快不明显。"""
    if not path or not os.path.exists(path):
        return path

    _wait_file_stable(path)

    # 已经是我们 staging 过的文件，不要再次复制，避免 .change / .upload 回写后循环。
    base = os.path.basename(path)
    if base.startswith(("audiocpp_up_", "audiocpp_rec_", "audiocpp_asc_")):
        return path

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0

    must_copy = force_copy or size >= 4 * 1024 * 1024
    try:
        path.encode("ascii")
        ascii_short = len(path) < 180
    except UnicodeEncodeError:
        ascii_short = False
        must_copy = True

    if ascii_short and not must_copy:
        return path

    ext = _safe_audio_ext(path)
    fd, out = tempfile.mkstemp(prefix="audiocpp_up_", suffix=ext)
    os.close(fd)
    shutil.copy2(path, out)
    return out


def _stage_recording(path):
    """麦克风录音结束后总是换成一个新的稳定临时文件。"""
    return _stage_upload(path, force_copy=True)


def _ensure_wav(path, target_sr=None):
    """server 端只有 WAV 读取器（其它格式报 invalid WAV RIFF header），Gradio 上传
    的 flac/mp3/ogg/m4a 等在这里先用 ffmpeg 转成 16-bit PCM WAV 临时文件再发；
    已是 RIFF/WAVE 的原样透传。target_sr：模型 prepare() 硬校验采样率的族（音源
    分离要 44.1k）传入目标值，采样率不符的输入（含 WAV）顺带重采样。"""
    if not path:
        return path
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError as e:
        raise gr.Error(_t("读不到音频文件 {path}：{error}",
                          "Cannot read audio file {path}: {error}", path=path, error=e))
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        if target_sr is None:
            return _ensure_ascii_path(path)
        try:
            with wave.open(path, "rb") as w:
                if w.getframerate() == target_sr:
                    return _ensure_ascii_path(path)
        except Exception:
            pass                    # 非 PCM WAV：wave 读不了，交给 ffmpeg 重写
    key = (path, target_sr)
    cached = _WAV_CACHE.get(key)
    if cached and os.path.exists(cached):
        return cached
    ffmpeg = _find_ffmpeg()
    ext = os.path.splitext(path)[1].lower() or "（无扩展名）"
    fd, out = tempfile.mkstemp(prefix="audiocpp_in_", suffix=".wav")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", path, "-map", "0:a:0"]
    if target_sr:
        cmd += ["-ar", str(target_sr)]
    cmd += ["-c:a", "pcm_s16le", out]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.getsize(out):
        try:
            os.remove(out)
        except OSError:
            pass
        err = (proc.stderr or "").strip()[-300:]
        raise gr.Error(_t("ffmpeg 转码 {ext} → wav 失败：{error}",
                          "ffmpeg conversion {ext} → wav failed: {error}",
                          ext=ext, error=err or _t("未知错误", "unknown error")))
    note = (_t("，重采样到 {sr}Hz", ", resampled to {sr}Hz", sr=target_sr)
            if target_sr else "")
    _ui_log(_t("输入转码：{name}（{ext}）→ 16-bit PCM WAV{note}",
               "input transcoded: {name} ({ext}) → 16-bit PCM WAV{note}",
               name=os.path.basename(path), ext=ext, note=note))
    _WAV_CACHE[key] = out
    return out


def _to_16k_mono_wav(path, target_sr=16000):
    """VAD / 说话人分离 / 强制对齐族要求 16 kHz 单声道输入（Silero、Sortformer 对
    非 16k 直接报错），这里用 wave+numpy 把 PCM WAV 转换成 16k 单声道临时文件。
    已是 16k 单声道、或非 PCM WAV（wave 读不了）时原样透传，由 server 决定成败。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except Exception:
        return path
    if sr == target_sr and ch == 1:
        return path
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i24 = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) |
               (b[:, 2].astype(np.int32) << 16))
        i24 -= (i24 & 0x800000) << 1  # sign-extend 24-bit
        data = i24.astype(np.float32) / 8388608.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        return path
    if ch > 1:
        data = data[: len(data) // ch * ch].reshape(-1, ch).mean(axis=1)
    if sr != target_sr and len(data) > 0:
        n_out = max(1, int(round(len(data) * target_sr / sr)))
        x_old = np.arange(len(data), dtype=np.float64) / sr
        x_new = np.arange(n_out, dtype=np.float64) / target_sr
        data = np.interp(x_new, x_old, data).astype(np.float32)
    fd, out = tempfile.mkstemp(prefix="audiocpp_16k_", suffix=".wav")
    os.close(fd)
    pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_sr)
        w.writeframes(pcm.tobytes())
    return out


def _split_wav_chunks(path, max_seconds, min_search_frac=0.6, win_ms=50,
                      pad_to_max=True):
    """把 PCM WAV 切成若干段临时 wav，返回 [(路径, 有效占比)]。用于 vevo2 歌声
    转换：FM 图按整段长度一次建图，8G 卡放不下长音频；且同一 server 上不同长度
    的请求会重建图并叠加占用显存（实测 20s 成功后 17s 反要 9.9GB），而形状相同
    的请求可复用缓存图（实测显存平稳）。所以**每段都补零到恰好 max_seconds**，
    占比供拼接时截掉补零对应的尾部输出。切点选在
    [起点+max*min_search_frac, 起点+max] 区间内能量最低的 win_ms 窗口中心
    （典型是呼吸/间奏处）；读不了的（非 PCM WAV）原样返回不分段。
    pad_to_max=False：不补零（ASR 分段转写用——补出的尾部静音只会浪费编码器
    token 甚至诱发幻听，转写也不需要各段图形状一致）。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
    except Exception:
        return [(path, 1.0)]            # 非 PCM WAV：不分段，交给 server
    if sw != 2 or n <= 0:
        return [(path, 1.0)]
    data = np.frombuffer(raw, dtype=np.int16)
    frames = data.reshape(-1, ch) if ch > 1 else data.reshape(-1, 1)
    mono = np.abs(frames.astype(np.float32)).mean(axis=1)
    win = max(1, int(sr * win_ms / 1000.0))
    env_len = max(1, len(mono) // win)
    env = mono[: env_len * win].reshape(env_len, win).mean(axis=1)

    max_f, lo_f = int(max_seconds * sr), int(max_seconds * min_search_frac * sr)
    spans, pos = [], 0
    while n - pos > max_f:
        w0 = min((pos + lo_f) // win, env_len - 1)
        w1 = min((pos + max_f) // win, env_len)
        wi = w0 + int(np.argmin(env[w0:w1])) if w1 > w0 else w1 - 1
        cut = min(n, wi * win + win // 2)
        spans.append((pos, cut))
        pos = cut
    if pos < n:
        spans.append((pos, n))
    outs = []
    for i, (a, b) in enumerate(spans):
        fd, out = tempfile.mkstemp(prefix=f"audiocpp_vcseg{i}_", suffix=".wav")
        os.close(fd)
        seg = frames[a:b]
        if pad_to_max and len(seg) < max_f:  # 全部段补零到等长，保证图形状一致
            pad = np.zeros((max_f - len(seg), ch), dtype=np.int16)
            seg = np.concatenate([seg, pad], axis=0)
        with wave.open(out, "wb") as ww:
            ww.setnchannels(ch)
            ww.setsampwidth(sw)
            ww.setframerate(sr)
            ww.writeframes(seg.tobytes())
        outs.append((out, (b - a) / float(max_f)))
    return outs


def _trim_wav_seconds(path, max_seconds):
    """把 PCM WAV 截到前 max_seconds 秒。vevo2 的音色参考超过约 10s 对音色几乎
    没有额外贡献，却会让 FM 图的 prompt 段变长、和源音频一起把显存吃满，所以
    转换前先截短。短于上限、或非 PCM WAV（wave 读不了）的原样返回。"""
    try:
        with wave.open(path, "rb") as w:
            sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
            keep = int(max_seconds * sr)
            if keep <= 0 or w.getnframes() <= keep:
                return path
            raw = w.readframes(keep)
    except Exception:
        return path
    fd, out = tempfile.mkstemp(prefix="audiocpp_ref_", suffix=".wav")
    os.close(fd)
    with wave.open(out, "wb") as ww:
        ww.setnchannels(ch)
        ww.setsampwidth(sw)
        ww.setframerate(sr)
        ww.writeframes(raw)
    return out


def _parse_adv_options(raw):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except Exception as e:
        raise gr.Error(_t("高级参数不是合法 JSON：{error}",
                          "Advanced options are not valid JSON: {error}", error=e))
    if not isinstance(obj, dict):
        raise gr.Error(_t('高级参数必须是 JSON 对象，例如 {"num_inference_steps": 10}',
                          'Advanced options must be a JSON object, e.g. {"num_inference_steps": 10}'))
    return obj


# Map known server-error fragments to an actionable Chinese hint, so a raw 500
# like "requires a session voice via --voice-ref" becomes "请上传参考音色".
# Ordered specific -> generic; server_error() takes the FIRST match.
ERROR_HINTS = [
    (re.compile(r"failed to allocate .{0,40}graph|out of memory|cudaMalloc", re.I),
     "🧠 显存不足：这次请求的计算图放不进剩余显存，通常是音频/文本太长。"
     "请剪短或分段后重试。"),
    (re.compile(r"invalid WAV RIFF header", re.I),
     "🎵 server 只支持 WAV 音频：上传的文件 webui 会自动转码，"
     "手动填路径的参数（如 voice_samples）请先转成 .wav。"),
    (re.compile(r"unsupported Chatterbox language", re.I),
     "🌐 Chatterbox 只支持 en/es/fr/de/it/pt/ko（无中文/日文/俄文，也没有自动检测）。"
     "请在“语言”里改选受支持的语言，或选 Auto 用默认（英语）。"),
    (re.compile(r"Stable Audio.{0,80}(English|prompt)|prompt.{0,80}(English|Stable Audio)", re.I),
     "🎵 Stable Audio 的提示词只支持英文。请把“提示词”改成英文后重试。"),
    (re.compile(r"max_source_positions", re.I),
     "⏱ 音频过长：Qwen3-ASR 编码器上限 1500 token（约 13 token/秒），"
     "单次最多约 115 秒。请把音频剪短或分段后再转写。"),
    (re.compile(r"exceeds fixed graph capacity|session_len_sec exceeds", re.I),
     "📏 音频超过模型的固定图容量（Sortformer 默认 20 秒、上限约 120 秒）。"
     "webui 的说话人分离/对话模式会按时长自动重载；仍报错说明超过 120 秒上限，"
     "请先剪短音频。"),
    (re.compile(r"combine voice_samples|voice_samples.{0,20}voice_ref", re.I),
     "🔀 voice_samples 与单个参考音色不能同时用：多说话人时请不要上传参考音色。"),
    (re.compile(r"cached voice id", re.I),
     "🎤 需要参考音频文件（不是 voice id）：请上传/录制一段参考音色。"),
    (re.compile(r"no valid Speaker|Speaker\s+N", re.I),
     "📝 需要多说话人脚本：每行写成 `Speaker 0: 内容`（多角色用 Speaker 0/1/…）。"),
    (re.compile(r"reference[-_ ]?text", re.I),
     "🗒 需要参考文本：在“参考文本”里填参考音频里说的原话。"),
    (re.compile(r"voice[-_ ]?ref|voice[-_ ]?id|session voice|speaker reference|"
                r"requires .{0,40}voice|requires audio", re.I),
     "🎤 该模型需要参考音色：上传/录制一段参考音频，或选一个内置参考音色后重试。"),
]

ERROR_HINTS_EN = [
    (ERROR_HINTS[0][0], "Not enough VRAM. Shorten or split the input."),
    (ERROR_HINTS[1][0], "The server requires WAV audio. Uploaded files are converted automatically."),
    (ERROR_HINTS[2][0], "Chatterbox supports en/es/fr/de/it/pt/ko only."),
    (ERROR_HINTS[3][0], "Stable Audio prompts must be in English."),
    (ERROR_HINTS[4][0], "The audio exceeds the Qwen3-ASR input limit. Split it and retry."),
    (ERROR_HINTS[5][0], "The audio exceeds this model's graph capacity."),
    (ERROR_HINTS[6][0], "Do not combine voice_samples with a single voice reference."),
    (ERROR_HINTS[7][0], "Upload a voice reference file."),
    (ERROR_HINTS[8][0], "Use one `Speaker N:` line per speaker."),
    (ERROR_HINTS[9][0], "Enter the transcript spoken in the reference audio."),
    (ERROR_HINTS[10][0], "This model requires a voice reference."),
]


def _extract_server_message(text):
    try:
        return json.loads(text)["error"]["message"] or text
    except Exception:
        return (text or "").strip()


def server_error(entry, status, text, extra=None):
    """Build a friendly gr.Error from a non-200 server response."""
    msg = _extract_server_message(text)
    language = get_language()
    hints = ERROR_HINTS_EN if language == "en" else ERROR_HINTS
    hint = next((h for pat, h in hints if pat.search(msg)), None)
    if hint:
        hint = _t(hint, hint, language)
    parts = [f"❌ server {status}"]
    if hint:
        parts.append("💡 " + hint)
    else:
        parts[0] = f"❌ server {status}：{msg[:400]}"
    if extra:
        parts.append(extra)
    if entry and not hint:
        prof = profile_for(entry)
        ih = prof.get("input_hint_en" if language == "en" else "input_hint")
        if ih:
            parts.append("ℹ️ " + _t(ih, ih, language))
    return gr.Error("\n\n".join(parts))


def connection_error(error):
    return gr.Error(_t(
        "无法连接 server @ {server}：{error}\n💡 server 可能已退出，请重新加载模型。",
        "Cannot connect to server @ {server}: {error}\nReload the model and retry.",
        server=SERVER, error=error))


def _msg_from_error(e):
    """Human-readable text from a gr.Error/exception, for inline (non-popup) display."""
    return getattr(e, "message", None) or str(e) or _t("未知错误", "Unknown error")


# Fallback catalog if models_catalog.json is missing/unreadable.
DEFAULT_CATALOG = {
    "host": "127.0.0.1", "port": 8080, "device": 0, "threads": 1,
    "models": [
        {"id": "qwen3-tts", "display_name": "Qwen3-TTS 0.6B (tts)",
         "family": "qwen3_tts", "path": "models/Qwen3-TTS-12Hz-0.6B-Base",
         "task": "tts", "mode": "offline"},
        {"id": "vibevoice", "display_name": "VibeVoice 1.5B (tts)",
         "family": "vibevoice", "path": "models/VibeVoice-1.5B",
         "task": "tts", "mode": "offline"},
        {"id": "qwen3-asr", "display_name": "Qwen3-ASR 0.6B (asr)",
         "family": "qwen3_asr", "path": "models/Qwen3-ASR-0.6B",
         "task": "asr", "mode": "offline"},
    ],
}


def _load_catalog():
    if os.path.isfile(CATALOG_PATH):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[webui] failed to read {CATALOG_PATH}: {e}; using defaults")
    return DEFAULT_CATALOG


def _load_model_params():
    """Per-model/family advanced-parameter specs (configs/model_params.json).
    Catalog id entries override family entries; a missing/broken file -> {}."""
    if os.path.isfile(MODEL_PARAMS_PATH):
        try:
            with open(MODEL_PARAMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[webui] failed to read {MODEL_PARAMS_PATH}: {e}; no param controls")
    return {}


def _load_required_files():
    """download_id -> 安装完成后模型目录里必须存在的文件清单（configs/required_files.json，
    由 model_manager.py CATALOG 的 required_files 预生成，含 .pt->.safetensors 等转换后
    的最终布局）。用于把“手动拷贝/下载中断的不完整目录”和“已安装”区分开——不完整目录
    server 端只会报 no registered model loader，用户看不出缺了什么。
    文件缺失/损坏 -> {}（完整性检查停用，退回“目录存在即已安装”的旧行为）。"""
    if os.path.isfile(REQUIRED_FILES_PATH):
        try:
            with open(REQUIRED_FILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception as e:
            print(f"[webui] failed to read {REQUIRED_FILES_PATH}: {e}; "
                  + _t("跳过模型完整性检查", "skipping model integrity check"))
    return {}


CATALOG = _load_catalog()
MODEL_PARAMS = _load_model_params()
REQUIRED_FILES = _load_required_files()
HOST = CATALOG.get("host", "127.0.0.1")
PORT = int(CATALOG.get("port", 8080))
DEVICE = int(CATALOG.get("device", 0))
THREADS = int(os.environ.get("AUDIOCPP_THREADS") or CATALOG.get("threads", 1))
if BACKEND == "cpu" and THREADS <= 1:
    # catalog 里的 threads=1 是按 CUDA 调的（GPU 路径不吃这个值）；CPU 后端的
    # ggml 计算线程数就是它，单线程没法用 —— 默认全核减一，留一个核给 UI/系统。
    THREADS = max(1, (os.cpu_count() or 4) - 1)

# If the user points us at an existing server, keep everything consistent with it.
_ENV_SERVER = os.environ.get("AUDIOCPP_SERVER")
if _ENV_SERVER:
    _u = urlparse(_ENV_SERVER)
    HOST = _u.hostname or HOST
    PORT = _u.port or PORT
    SERVER = _ENV_SERVER.rstrip("/")
else:
    SERVER = f"http://{HOST}:{PORT}"

# --- managed server process state ------------------------------------------
_proc_lock = threading.Lock()
_server_proc = None      # subprocess.Popen we launched, or None
_loaded_id = None        # model id our managed server is serving
_loaded_session_options = None   # 随本次加载写进 server config 的 session_options
_loaded_mode = None              # 本次加载的运行模式（offline/streaming），None=未加载


def _missing_required_files(entry):
    """目录已存在但缺失的必须文件（相对模型目录）；目录不存在或无清单时返回 []。"""
    req = REQUIRED_FILES.get(entry.get("download_id") or "")
    if not req or not os.path.isdir(entry["abs_path"]):
        return []
    return [f for f in req if not os.path.isfile(os.path.join(entry["abs_path"], f))]


def catalog_models():
    """Catalog entries annotated with abs_path / installed / incomplete / label.
    installed 要求目录存在且 required_files 清单齐全；目录在但缺文件记为
    incomplete（missing_files 列出缺什么），加载入口据此给出明确报错。"""
    out = []
    for m in CATALOG.get("models", []):
        rel = m.get("path", "")
        ap = rel if os.path.isabs(rel) else os.path.join(BUNDLE_ROOT, rel)
        entry = dict(m)
        entry["abs_path"] = os.path.normpath(ap).replace("\\", "/")
        entry["missing_files"] = _missing_required_files(entry)
        entry["incomplete"] = bool(entry["missing_files"])
        entry["installed"] = os.path.exists(entry["abs_path"]) and not entry["incomplete"]
        entry["label"] = m.get("display_name") or m.get("id", "?")
        out.append(entry)
    return out


def catalog_by_id(model_id):
    for m in catalog_models():
        if m.get("id") == model_id:
            return m
    return None


def choices_for_tasks(tasks, language=None):
    """[(label, id)] for catalog models whose task is in `tasks`; missing ones flagged.
    未安装且估算显存超过本机的条目额外标注最低显存，防止白下载。"""
    language = language or get_language()
    out = []
    for m in catalog_models():
        if m.get("task") not in tasks:
            continue
        label = m["label"]
        if language == "en":
            label = m.get("display_name_en") or (label if label.isascii() else m["id"])
        else:
            label = _t(label, label, language)
        if m["incomplete"]:
            label += _t(" · 目录不完整", " · incomplete", language)
        elif not m["installed"]:
            label += _t(" · 未安装", " · not installed", language)
            short = _vram_shortfall(m)
            if short:
                label += _t(" ⚠️估算需≥{need:g}G显存", " ⚠️≥{need:g} GB VRAM", language,
                            need=short[0])
        out.append((label, m["id"]))
    return out


def builtin_voices():
    if not os.path.isdir(PROMPTS_DIR):
        return []
    return sorted(f for f in os.listdir(PROMPTS_DIR) if f.lower().endswith(".wav"))


def _voice_name_from_path(path):
    """Original upload basename without its extension, for the save-name box."""
    if not path:
        return ""
    return os.path.splitext(os.path.basename(path))[0]


def _stage_tts_voice_upload(path):
    """Preserve the uploaded filename before replacing its preview path."""
    return _stage_upload(path, force_copy=True), _voice_name_from_path(path)


def _stage_tts_voice_recording(path):
    return _stage_recording(path), _voice_name_from_path(path)


def _load_voice_texts():
    """Built-in voice basename -> reference transcript, parsed from
    voice/prompt_text (each line is '<basename>|<transcript>')."""
    texts = {}
    try:
        with open(os.path.join(PROMPTS_DIR, "prompt_text"), "r", encoding="utf-8") as f:
            for line in f:
                name, sep, text = line.rstrip("\n").partition("|")
                if sep:
                    texts[name.strip()] = text
    except Exception:
        pass
    return texts


def refresh_builtin_voices(current):
    """刷新按钮：重新扫描 voice/ 目录的 wav 列表，并按当前选中项重读
    prompt_text 里的参考文本；选中项已被删除时回落到 '(none)'。
    必须在同一个 handler 里连带输出 wav/参考文本——拆成 .then 链会和
    下拉更新触发的 .change 并发执行，撞 Gradio get_config 的竞态
    （RuntimeError: dictionary changed size during iteration）。"""
    choices = ["(none)"] + builtin_voices()
    if current not in choices:
        current = "(none)"
    wav, ref, voice_name = on_tts_builtin_voice_change(current)
    return gr.update(choices=choices, value=current), wav, ref, voice_name


def on_builtin_voice_change(name):
    """Selecting a built-in voice mirrors its wav into the upload widget and
    fills the matching reference text; '(none)' clears both."""
    if not name or name == "(none)":
        return None, ""
    path = os.path.join(PROMPTS_DIR, name)
    ref = _load_voice_texts().get(os.path.splitext(name)[0], "")
    return (_ensure_ascii_path(path) if os.path.isfile(path) else None), ref


def on_tts_builtin_voice_change(name):
    wav, ref = on_builtin_voice_change(name)
    voice_name = (os.path.splitext(name)[0]
                  if name and name != "(none)" else "")
    return wav, ref, voice_name


def _builtin_voice_filename(voice_name):
    """Validate a user-facing voice name and normalize it to a WAV filename."""
    name = (voice_name or "").strip()
    if name.lower().endswith(".wav"):
        name = name[:-4]
    invalid = '<>:"/\\|?*'
    if (not name or name in (".", "..") or name.endswith((" ", "."))
            or any(ch in invalid or ord(ch) < 32 for ch in name)):
        raise ValueError(_t(
            "名称不能为空、不能是路径，也不能包含这些字符：{chars}",
            "The name cannot be empty or a path, and cannot contain: {chars}",
            chars=invalid))
    return name + ".wav"


def _write_voice_prompt(voice_stem, reference_text):
    """Insert or replace one '<voice stem>|<transcript>' prompt_text record."""
    prompt_path = os.path.join(PROMPTS_DIR, "prompt_text")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        lines = []

    transcript = " ".join((reference_text or "").splitlines()).strip()
    record = f"{voice_stem}|{transcript}"
    updated = False
    output = []
    for line in lines:
        key, sep, _text = line.partition("|")
        if sep and key.strip() == voice_stem:
            if not updated:
                output.append(record)
                updated = True
            continue
        output.append(line)
    if not updated:
        output.append(record)

    _save_voice_prompt_lines(output)


def _save_voice_prompt_lines(lines):
    """Atomically replace prompt_text with the supplied records."""
    prompt_path = os.path.join(PROMPTS_DIR, "prompt_text")

    fd, temp_path = tempfile.mkstemp(
        prefix="prompt_text_", suffix=".tmp", dir=PROMPTS_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        os.replace(temp_path, prompt_path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise


def _delete_voice_prompt(voice_stem):
    """Remove every prompt_text record for one built-in voice."""
    prompt_path = os.path.join(PROMPTS_DIR, "prompt_text")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return
    output = [
        line for line in lines
        if not (line.partition("|")[1]
                and line.partition("|")[0].strip() == voice_stem)
    ]
    if output != lines:
        _save_voice_prompt_lines(output)


def save_builtin_voice(uploaded_voice, voice_name, reference_text):
    """Copy a reference into voice/, update prompt_text, and refresh its list."""
    if not (voice_name or "").strip():
        return gr.skip(), _t(
            "❌ 请填写内置参考音色名称。",
            "❌ Enter a name for the built-in voice.")
    if not uploaded_voice or not os.path.isfile(uploaded_voice):
        return gr.skip(), _t(
            "❌ 请先上传或录制参考音频。",
            "❌ Upload or record a reference audio file first.")

    try:
        filename = _builtin_voice_filename(voice_name)
        source_wav = _ensure_wav(uploaded_voice)
        destination = os.path.join(PROMPTS_DIR, filename)
        if os.path.abspath(source_wav) != os.path.abspath(destination):
            fd, temp_path = tempfile.mkstemp(
                prefix="voice_", suffix=".wav", dir=PROMPTS_DIR)
            os.close(fd)
            try:
                shutil.copy2(source_wav, temp_path)
                os.replace(temp_path, destination)
            except Exception:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
                raise
        _write_voice_prompt(os.path.splitext(filename)[0], reference_text)
        choices = ["(none)"] + builtin_voices()
        return gr.update(choices=choices, value=filename), _t(
            "✅ 参考音色已保存：{filename}",
            "✅ Voice reference saved: {filename}", filename=filename)
    except Exception as e:
        return gr.skip(), _t(
            "❌ 保存参考音色失败：{error}",
            "❌ Failed to save voice reference: {error}", error=e)


def delete_builtin_voice(current):
    """Delete the selected built-in WAV and its prompt_text record."""
    if not current or current == "(none)":
        return tuple(gr.skip() for _ in range(5))
    try:
        if os.path.basename(current) != current or not current.lower().endswith(".wav"):
            raise ValueError(_t("无效的内置音色文件名。",
                                 "Invalid built-in voice filename."))
        path = os.path.join(PROMPTS_DIR, current)
        if not os.path.isfile(path):
            raise FileNotFoundError(_t(
                "找不到内置音色文件：{filename}",
                "Built-in voice file not found: {filename}", filename=current))
        os.remove(path)
        _delete_voice_prompt(os.path.splitext(current)[0])
        choices = ["(none)"] + builtin_voices()
        return (gr.update(choices=choices, value="(none)"), None, "", "",
                _t("✅ 已删除内置参考音色：{filename}",
                   "✅ Built-in voice deleted: {filename}", filename=current))
    except Exception as e:
        return (*tuple(gr.skip() for _ in range(4)), _t(
            "❌ 删除内置参考音色失败：{error}",
            "❌ Failed to delete built-in voice: {error}", error=e))


# --- config-driven advanced-parameter controls (TTS tab) -------------------
def params_for(model_id):
    """Advanced-parameter specs: catalog id override, then family fallback."""
    entry = catalog_by_id(model_id) if model_id else None
    if not entry:
        return []
    specs = MODEL_PARAMS.get(entry.get("id", ""))
    if specs is None:
        specs = MODEL_PARAMS.get(entry.get("family", ""), [])
    return specs if isinstance(specs, list) else []


def _make_param_component(p, language=None):
    """Build one Gradio control from a spec (type: slider|number|bool|text|choice).

    interactive=True is forced: inside @gr.render a control that is only wired to
    its own .change handler is otherwise inferred as output-only (read-only)."""
    p = localized_param_spec(p, language)
    t = p.get("type", "number")
    label = p.get("label", p.get("name", ""))
    info = p.get("info")
    if t == "bool":
        return gr.Checkbox(label=label, info=info, value=bool(p.get("default", False)),
                           interactive=True)
    if t == "text":
        return gr.Textbox(label=label, info=info, value=p.get("default", ""),
                          placeholder=p.get("placeholder", ""),
                          lines=int(p.get("lines", 1)), interactive=True)
    if t == "choice":
        return gr.Dropdown(label=label, info=info, choices=p.get("choices", []),
                           value=p.get("default"), interactive=True)
    if t == "slider":
        return gr.Slider(label=label, info=info,
                         minimum=p.get("minimum", 0), maximum=p.get("maximum", 1),
                         step=p.get("step", 0.01), value=p.get("default", 0),
                         interactive=True)
    return gr.Number(label=label, info=info, value=p.get("default"),
                     minimum=p.get("minimum"), maximum=p.get("maximum"),
                     step=p.get("step"), precision=p.get("precision"),
                     interactive=True)


def _adv_updater(name):
    """change-handler that writes one control's value into the shared advanced
    options state dict (keyed by the option name)."""
    def _fn(state, value):
        state = dict(state or {})
        state[name] = value
        return state
    return _fn


# --- server lifecycle ------------------------------------------------------
def _port_open(host, port, timeout=0.5):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def server_alive():
    try:
        requests.get(f"{SERVER}/health", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


def loaded_ids():
    try:
        r = requests.get(f"{SERVER}/v1/models", timeout=5)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]
    except Exception:
        return []


def _write_temp_config(entry):
    model = {
        "id": entry["id"],
        "family": entry["family"],
        "path": entry["abs_path"],
        "task": entry.get("task", "tts"),
        "mode": entry.get("mode", "offline"),
    }
    for key in ("config", "weight", "load_options", "session_options"):
        if entry.get(key) is not None:
            model[key] = entry[key]
    cfg = {"host": HOST, "port": PORT, "backend": SERVER_BACKEND, "device": DEVICE,
           "threads": THREADS, "models": [model]}
    fd, path = tempfile.mkstemp(prefix="audiocpp_webui_cfg_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return path


def _stop_server():
    global _server_proc, _loaded_id, _loaded_session_options, _loaded_mode
    _loaded_session_options = None
    _loaded_mode = None
    proc, _server_proc, _loaded_id = _server_proc, None, None
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:
            pass
    for _ in range(24):  # let the OS release the port
        if not _port_open(HOST, PORT):
            break
        time.sleep(0.25)


def _read_tail(path, n=30, max_bytes=65536):
    """Last n non-empty lines of a file. Reads only the file's tail, and treats
    \\r as a line break so tqdm-style progress (one giant \\r-line) shows its
    latest state instead of nothing."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        lines = [ln for ln in data.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                 if ln.strip()]
        return "\n".join(lines[-n:]).strip()
    except Exception:
        return ""


def _log_tail(n=30):
    return _read_tail(LOG_PATH, n)


# One shared append handle for the WebUI log: the pump thread (server output)
# and _ui_log (webui-side request events) both write through it, so lines
# interleave correctly instead of two handles overwriting each other.
_log_lock = threading.Lock()
_log_fh = None


def _open_log_file(truncate=False):
    global _log_fh
    with _log_lock:
        if _log_fh is not None:
            try:
                _log_fh.close()
            except Exception:
                pass
        _log_fh = open(LOG_PATH, "w" if truncate else "a",
                       encoding="utf-8", errors="replace")


def _log_write(text):
    global _log_fh
    with _log_lock:
        if _log_fh is None:
            try:
                _log_fh = open(LOG_PATH, "a", encoding="utf-8", errors="replace")
            except Exception:
                return
        try:
            _log_fh.write(text)
            _log_fh.flush()
        except Exception:
            pass


def _ts():
    return time.strftime("%H:%M:%S")


def _emit_log_line(text):
    """One already-formatted line to BOTH the console and the WebUI log file."""
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except Exception:
        pass
    _log_write(text)


def _ui_log(msg):
    """Timestamped webui-side event (request start/finish, model load, ...) so
    the console/log show when things began and ended, not just server spam."""
    _emit_log_line(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [webui] {msg}\n")


def _pump_server_output(proc):
    """Tee the server's combined stdout/stderr to console + log file, prefixing
    every line with a timestamp and collapsing consecutive duplicate lines
    (e.g. the repeated `CUDA graph warmup complete`) into a periodic counter."""
    last, repeats = None, 0
    try:
        for line in proc.stdout:
            if line == last:
                repeats += 1
                if repeats % 50 == 0:
                    _emit_log_line(f"[{_ts()}]   ... " + _t(
                        "上一行已重复 {count} 次", "previous line repeated {count} times",
                        count=repeats) + "\n")
                continue
            if repeats:
                _emit_log_line(f"[{_ts()}]   ... " + _t(
                    "（上一行共重复 {count} 次）", "(previous line repeated {count} times total)",
                    count=repeats) + "\n")
            last, repeats = line, 0
            _emit_log_line(f"[{_ts()}] {line}")
    except Exception:
        pass
    finally:
        if repeats:
            _emit_log_line(f"[{_ts()}]   ... " + _t(
                "（上一行共重复 {count} 次）", "(previous line repeated {count} times total)",
                count=repeats) + "\n")


def _start_server(entry):
    global _server_proc, _loaded_id
    if not os.path.isfile(SERVER_EXE):
        raise gr.Error(_t("找不到 server：{path}（可用 AUDIOCPP_BACKEND=gpu|cpu 指定）",
                          "Server not found: {path}. Set AUDIOCPP_BACKEND=gpu|cpu.",
                          path=SERVER_EXE))
    cfg = _write_temp_config(entry)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    _open_log_file(truncate=True)
    # --log 会打开 engine 的 [TRACE]/[TIMING] 调试输出，日常太吵，默认关闭；
    # 排查推理问题时设 AUDIOCPP_SERVER_DEBUG=1 再启动 webui。
    cmd = [SERVER_EXE, "--config", cfg, "--host", HOST, "--port", str(PORT)]
    if os.environ.get("AUDIOCPP_SERVER_DEBUG") == "1":
        cmd.append("--log")
    _server_proc = subprocess.Popen(
        cmd,
        cwd=BUNDLE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=flags, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    _loaded_id = entry["id"]
    extra = (_t("，threads={threads}", ", threads={threads}", threads=THREADS)
             if SERVER_BACKEND == "cpu" else "")
    _ui_log(_t("启动 audiocpp_server（backend={backend}{extra}），加载模型 {label} …",
               "starting audiocpp_server (backend={backend}{extra}), loading model {label} …",
               backend=SERVER_BACKEND, extra=extra, label=entry['label']))
    threading.Thread(target=_pump_server_output, args=(_server_proc,),
                     daemon=True).start()


def _wait_health(timeout):
    start = time.time()
    while time.time() - start < timeout:
        if _server_proc is not None and _server_proc.poll() is not None:
            return False  # process exited before becoming healthy
        if server_alive():
            return True
        time.sleep(0.5)
    return False


def ensure_model_loaded(model_id, expect_tasks=None, session_options=None, mode=None):
    """(Re)start the server so `model_id` is loaded. Returns a status string.
    session_options：额外写进本次 server config 的 session_options（string→string，
    如 sortformer 的 session_len_sec）；与上次加载不一致时会重启重载。
    mode：覆盖 catalog 条目的运行模式（"streaming"/"offline"），流式转写/生成用；
    与上次加载不一致时同样重启重载。"""
    global _loaded_session_options, _loaded_mode
    if not model_id:
        raise gr.Error(_t("请先选择一个模型", "Select a model first."))
    entry = catalog_by_id(model_id)
    if entry is None:
        raise gr.Error(_t("catalog 里没有模型 id：{model}",
                          "Model id not found in catalog: {model}", model=model_id))
    if not entry["installed"]:
        if entry["incomplete"]:
            missing = entry["missing_files"]
            shown = "、".join(missing[:8]) + (f" 等 {len(missing)} 个" if len(missing) > 8 else "")
            raise gr.Error(_t(
                "模型目录不完整：{path}\n缺少文件：{missing}\n请点『⬇️ 下载模型』重新安装。",
                "Model directory is incomplete: {path}\nMissing: {missing}\nClick Download to reinstall.",
                path=entry["abs_path"], missing=shown))
        raise gr.Error(_t("模型未安装：{path}", "Model is not installed: {path}",
                          path=entry["abs_path"]))
    if expect_tasks and entry.get("task") not in expect_tasks:
        raise gr.Error(_t(
            "模型 {model} 的 task 是 {actual}，此处需要 {expected}",
            "Model {model} has task {actual}; expected {expected}.",
            model=model_id, actual=entry.get("task"), expected="/".join(expect_tasks)))
    want_mode = mode or entry.get("mode", "offline")
    mode_note = (_t("（流式模式）", " (streaming)")
                 if want_mode == "streaming" else "")

    with _proc_lock:
        managed_alive = _server_proc is not None and _server_proc.poll() is None
        if (managed_alive and _loaded_id == model_id and server_alive()
                and (session_options or {}) == (_loaded_session_options or {})
                and want_mode == (_loaded_mode or entry.get("mode", "offline"))):
            return _t("✅ 已加载：{label}{mode}", "✅ Loaded: {label}{mode}",
                      label=entry["label"], mode=mode_note)

        if not managed_alive and server_alive():
            # A server we didn't launch is holding the port.
            if session_options:
                raise gr.Error(_t(
                    "{host}:{port} 上的外部 server 无法调整 session 配置，请先关闭。",
                    "The external server at {host}:{port} cannot change session settings. Stop it first.",
                    host=HOST, port=PORT))
            if mode and mode != entry.get("mode", "offline"):
                raise gr.Error(_t(
                    "{host}:{port} 上的外部 server 无法切换到 {mode} 模式，请先关闭。",
                    "The external server at {host}:{port} cannot switch to {mode}. Stop it first.",
                    host=HOST, port=PORT, mode=mode))
            if model_id in loaded_ids():
                return _t("✅ 复用外部 server：{label}",
                          "✅ Using external server: {label}", label=entry["label"])
            raise gr.Error(_t(
                "检测到外部 server 占用 {host}:{port}，请先关闭或设置 AUDIOCPP_SERVER。",
                "An external server is using {host}:{port}. Stop it or set AUDIOCPP_SERVER.",
                host=HOST, port=PORT))

        _stop_server()
        if session_options or want_mode != entry.get("mode", "offline"):
            entry = dict(entry)
            entry["mode"] = want_mode
            if session_options:
                entry["session_options"] = {
                    **(entry.get("session_options") or {}), **session_options}
        t0 = time.time()
        _start_server(entry)
        _loaded_session_options = dict(session_options) if session_options else None
        _loaded_mode = want_mode
        if not _wait_health(LOAD_TIMEOUT):
            tail = _log_tail()
            _stop_server()
            _ui_log(_t("模型 {label} 加载失败/超时（{timeout}s）",
                       "model {label} load failed/timed out ({timeout}s)",
                       label=entry['label'], timeout=LOAD_TIMEOUT))
            raise gr.Error(_t("加载 {label} 失败/超时（{timeout}s）。\n日志尾部：\n{tail}",
                              "Loading {label} failed or timed out ({timeout}s).\nLog tail:\n{tail}",
                              label=entry["label"], timeout=LOAD_TIMEOUT, tail=tail))
        _ui_log(_t("模型 {label} 加载完成{mode_note}，用时 {seconds:.1f}s",
                   "model {label} loaded{mode_note}, elapsed {seconds:.1f}s",
                   label=entry['label'], mode_note=mode_note, seconds=time.time() - t0))
        return _t("✅ 已加载：{label}{mode}", "✅ Loaded: {label}{mode}",
                  label=entry["label"], mode=mode_note)


def unload_model():
    """停止本 WebUI 启动的 server，释放全部显存（权重+常驻计算图缓冲）。
    下次生成/转写时 ensure_model_loaded 会自动重启重载（实测 ~5s），转写速度
    本身不受影响（计算图本来就按每次请求的音频长度分配/复用）。"""
    with _proc_lock:
        managed_alive = _server_proc is not None and _server_proc.poll() is None
        if managed_alive:
            label = _loaded_id or "(unknown)"
            _stop_server()
            _ui_log(_t("已卸载模型 {label} 并停止 server，显存已释放",
                       "unloaded model {label} and stopped server, VRAM released",
                       label=label))
            return (_t("🧹 已卸载模型并释放显存，下次运行时会自动重新加载。",
                       "🧹 Model unloaded and VRAM released. It will reload on the next run."),
                    server_status())
        if server_alive():
            return (_t("⚠️ 当前 server 不是本 WebUI 启动的，请在其启动窗口中关闭。",
                       "⚠️ This server was started externally. Stop it from its own window."),
                    server_status())
        return _t("⚪ server 未运行，无需释放。", "⚪ Server is not running."), server_status()


def server_status():
    if server_alive():
        ids = ", ".join(loaded_ids()) or "(none)"
        return f"✅ server @ {SERVER} · backend={BACKEND} · model id={ids}"
    return _t("⚪ server 未运行 @ {server} — 选择模型并点『📥 加载模型』",
              "⚪ Server is not running @ {server} — select a model and click Load.",
              server=SERVER)


def _api_usage_md(language=None):
    """状态行下方折叠区的第三方调用说明。端点/字段以 app/server/README.md 为准；
    URL 取运行时的 SERVER，避免和 AUDIOCPP_SERVER / catalog 配置不一致。"""
    language = language or get_language()
    if language == "en":
        return f"""
Other applications can call the local `audiocpp_server` started by this WebUI.

- Base URL: `{SERVER}/v1`
- TTS: `POST {SERVER}/v1/audio/speech`
- ASR: `POST {SERVER}/v1/audio/transcriptions`
- Other tasks: `POST {SERVER}/v1/tasks/run`
- `model` must be the currently loaded model id. Check it with `GET {SERVER}/v1/models`.
- Audio paths in API JSON are paths on the server machine, not browser uploads.

```bash
curl {SERVER}/v1/audio/speech -H "Content-Type: application/json" -o out.wav \\
  -d '{{"model":"qwen3-tts","input":"Hello from audio.cpp."}}'
```

The server stays running while the WebUI command window is open.
"""
    content = f"""
第三方应用可以直接调用本 WebUI 启动的 `audiocpp_server`（OpenAI 风格 HTTP API），不经过本页面。

- **URL 怎么填**：API 地址是 `{SERVER}`；OpenAI 兼容客户端的 Base URL 填 `{SERVER}/v1`。
  生成语音：`POST {SERVER}/v1/audio/speech` · 音频转写：`POST {SERVER}/v1/audio/transcriptions`
- **模型名称怎么填**：`model` 填模型 id（与本页模型列表一致，如 `qwen3-tts`、`vibevoice`），
  且必须是**当前已加载**的那个 —— server 同一时刻只驻留一个模型，先在本页点『📥 加载模型』；
  可用 `GET {SERVER}/v1/models` 查看当前可用的 id。
- **TTS 请求示例**（响应默认是 WAV 音频；加 `"response_format": "json"` 改为返回 base64 的 JSON）：

```bash
curl {SERVER}/v1/audio/speech -H "Content-Type: application/json" -o out.wav \\
  -d '{{"model": "qwen3-tts", "input": "你好，audio.cpp。", "voice_ref": "D:/voices/ref.wav", "reference_text": "参考音频里的原话", "seed": 1234}}'
```

- **ASR 请求示例**：`-d '{{"model": "qwen3-asr", "audio": "D:/audio/in.wav"}}'`；
  可选 `"language"`（强制语种，如 `"Chinese"`）和 `"context"`（人名/术语偏置提示），仅 qwen3-asr 生效。
  注意 `voice_ref` / `audio` 填的都是 **server 所在机器上的文件路径**（不是浏览器上传）。
- **音乐生成（gen 模型）**：走通用路由 `POST {SERVER}/v1/tasks/run`，body 形如
  `{{"model": "ace-step", "request": {{"text": "提示词", "lyrics": "歌词", "duration_seconds": 30,
  "options": {{"tags": "pop,bright"}}}}}}`，响应 JSON 的 `audio` 字段是 base64 WAV。
- **其它任务（vc/svc/s2s/sep/vad/diar/align）**：同样走 `POST {SERVER}/v1/tasks/run`，`request` 里用
  `audio`（源音频路径）/ `voice_ref`（目标音色）/ `text`（对齐文本）等字段；分离多轨在响应的
  `named_audio_outputs`，VAD/说话人/对齐结果在 `segments` / `speaker_turns` / `words`。
- server 的生命周期跟随本 WebUI 的**命令行窗口**：只关浏览器页面不影响，server 仍可被第三方调用；
  关掉命令窗口（webui.py 退出）才会连带关闭它。也可单独启动 server（如 {SERVER_LAUNCHER}）
  供第三方应用调用。
"""
    return _t(content, content, language)


# --- background model downloads (via tools/model_manager.py) ----------------
_dl_lock = threading.Lock()
_downloads = {}  # model_id -> {"proc": Popen, "log": path}


def _dl_log_path(model_id):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_id)
    return os.path.join(LOG_DIR, f"download_{safe}.log")


def _dir_size_bytes(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _fmt_bytes(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _download_progress_note(entry):
    """Bytes already on disk for a running download. model_manager stages into
    models/.engine_model_staging/<target>.partial/ and renames on completion;
    fall back to the whole staging root for packages with composite targets."""
    staging_root = os.path.join(MODELS_ROOT, ".engine_model_staging")
    base = os.path.basename(entry.get("path", "").rstrip("/\\"))
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    staging = os.path.join(staging_root, safe + ".partial")
    probe = staging if os.path.isdir(staging) else staging_root
    if not os.path.isdir(probe):
        return _t("尚未写入数据（正在连接/解析）", "Waiting for data…")
    return _t("已下载 {size}", "Downloaded {size}",
              size=_fmt_bytes(_dir_size_bytes(probe)))


def hf_token_present():
    """True if model_manager will find an HF token (env or cached login)."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return os.path.isfile(os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token"))


def download_model(model_id, hf_token="", proxy=""):
    """Kick off `model_manager.py install <download_id>` in the background."""
    if not model_id:
        return _t("❌ 请先选择一个模型", "❌ Select a model first.")
    entry = catalog_by_id(model_id)
    if entry is None:
        return _t("❌ catalog 里没有模型 id：{model}",
                  "❌ Model id not found: {model}", model=model_id)
    if entry["installed"]:
        return _t("✅ {label} 已安装，无需下载", "✅ {label} is already installed.",
                  label=entry["label"])
    dl_id = entry.get("download_id")
    if not dl_id:
        return _t("⚠️ {label} 没有 download_id，请手动安装。",
                  "⚠️ {label} has no download_id; install it manually.", label=entry["label"])
    if MODEL_MANAGER is None:
        return _t("❌ 找不到 tools/model_manager.py",
                  "❌ tools/model_manager.py was not found.")

    # Pass a token to the child so gated/private HF repos don't 401.
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"    # progress lines land in the log immediately
    env["PYTHONIOENCODING"] = "utf-8"
    tok = (hf_token or "").strip()
    if tok:
        env["HF_TOKEN"] = tok
        env["HUGGING_FACE_HUB_TOKEN"] = tok
    # Route the child's downloads through a proxy (urllib reads these env vars).
    px = (proxy or "").strip()
    proxy_note = ""
    if px:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[var] = px
        proxy_note = _t("🌐 通过代理 {proxy}\n\n", "🌐 Proxy: {proxy}\n\n", proxy=px)
    warn = "" if (tok or hf_token_present()) else _t(
        "⚠️ 未检测到 HF token，受限模型可能返回 401。\n\n",
        "⚠️ No HF token detected; gated models may return 401.\n\n")
    short = _vram_shortfall(entry)
    if short:
        warn = _t("⚠️ **显存不足**：估算需 **≥{need:g}G**，本机为 **{local:g}G**。\n\n",
                  "⚠️ **Low VRAM**: estimated **≥{need:g} GB**, detected **{local:g} GB**.\n\n",
                  need=short[0], local=short[1]) + warn
    if entry["incomplete"]:
        warn = _t("⚠️ {path} 不完整（缺 {count} 个文件），将覆盖重装。\n\n",
                  "⚠️ {path} is incomplete ({count} files missing); it will be reinstalled.\n\n",
                  path=entry["abs_path"], count=len(entry["missing_files"])) + warn

    with _dl_lock:
        rec = _downloads.get(model_id)
        if rec and rec["proc"].poll() is None:
            return _t("⏳ {label} 已在后台下载中…\n```\n{tail}\n```",
                      "⏳ {label} is already downloading…\n```\n{tail}\n```",
                      label=entry["label"], tail=_read_tail(rec["log"]))
        log = _dl_log_path(model_id)
        logf = open(log, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [sys.executable, "-u", MODEL_MANAGER, "install", dl_id,
             "--models-root", MODELS_ROOT, "--overwrite"],
            cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
        _downloads[model_id] = {"proc": proc, "log": log}
    _ui_log(_t("开始后台下载 {label}（{dl_id}），日志：{log}",
               "started background download {label} ({dl_id}), log: {log}",
               label=entry['label'], dl_id=dl_id, log=log))
    return warn + proxy_note + _t(
        "⏳ 已开始下载 **{label}**（{download_id}）。完成后刷新列表。\n日志：{log}",
        "⏳ Download started: **{label}** ({download_id}). Refresh the list when complete.\nLog: {log}",
        label=entry["label"], download_id=dl_id, log=log)


def download_status(model_id):
    entry = catalog_by_id(model_id) if model_id else None
    if entry is None:
        return ""
    if entry["installed"]:
        return _t("✅ {label} 已安装", "✅ {label} is installed.", label=entry["label"])
    rec = _downloads.get(model_id)
    if rec is None:
        if entry["incomplete"]:
            return _t("⚠️ {label} 目录不完整（缺 {count} 个文件），请重新下载。",
                      "⚠️ {label} is incomplete ({count} files missing). Download it again.",
                      label=entry["label"], count=len(entry["missing_files"]))
        return _t("⚪ {label} 未安装，未开始下载", "⚪ {label} is not installed.",
                  label=entry["label"])
    code = rec["proc"].poll()
    tail = _read_tail(rec["log"], n=12)
    if code is None:
        return _t("⏳ 正在下载 {label}… {progress} · 更新于 {time}\n```\n{tail}\n```",
                  "⏳ Downloading {label}… {progress} · {time}\n```\n{tail}\n```",
                  label=entry["label"], progress=_download_progress_note(entry),
                  time=_ts(), tail=tail)
    if not rec.get("reported"):
        rec["reported"] = True
        _ui_log(_t("{label} 下载进程结束 (exit {code})",
                   "{label} download process ended (exit {code})",
                   label=entry['label'], code=code))
    if code == 0:
        return _t("✅ {label} 下载完成，请刷新列表。\n```\n{tail}\n```",
                  "✅ {label} downloaded. Refresh the model list.\n```\n{tail}\n```",
                  label=entry["label"], tail=tail)
    return _t("❌ {label} 下载失败（exit {code}）。\n```\n{tail}\n```",
              "❌ {label} download failed (exit {code}).\n```\n{tail}\n```",
              label=entry["label"], code=code, tail=tail)


def _download_running(model_id):
    rec = _downloads.get(model_id) if model_id else None
    return rec is not None and rec["proc"].poll() is None


def download_start(model_id, hf_token="", proxy=""):
    """Click handler: kick off the download and arm the auto-refresh timer."""
    msg = download_model(model_id, hf_token, proxy)
    return msg, gr.Timer(active=_download_running(model_id))


def download_status_tick(model_id):
    """Timer tick: refresh status; stop the timer once the download is idle."""
    return download_status(model_id), gr.Timer(active=_download_running(model_id))


def _gguf_entry(model_id, require_installed=True):
    if not model_id:
        return None, _t("请先选择一个模型", "Select a model first.")
    entry = catalog_by_id(model_id)
    if entry is None:
        return None, _t("catalog 里没有模型 id：{model}",
                         "Model id not found in catalog: {model}", model=model_id)
    if require_installed and not entry["installed"]:
        return None, _t("模型未完整安装。", "The model is not fully installed.")
    return entry, ""


def _gguf_output_path(entry):
    model_path = entry["abs_path"]
    if os.path.isfile(model_path) and model_path.lower().endswith(".gguf"):
        return model_path
    root = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
    return os.path.join(root, "model.gguf")


def _gguf_tensor_entrypoint(model_dir):
    """Return a single-file or sharded safetensors entry point in model_dir."""
    for name in ("model.safetensors.index.json", "model.safetensors"):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _gguf_conversion_inputs(entry):
    """Build the converter's ordered (namespace, weights) input list."""
    if entry["family"] not in GGUF_WEBUI_CONVERTIBLE_FAMILIES:
        return []

    model_path = entry["abs_path"]
    if os.path.isfile(model_path):
        lower = model_path.lower()
        if lower.endswith(".safetensors") or lower.endswith(".safetensors.index.json"):
            return [("", model_path)]
        return []

    # Qwen3-TTS is a composite package. Its GGUF package spec requires both
    # tensor sources under the exact namespaces below.
    if entry["family"] == "qwen3_tts":
        model_weights = _gguf_tensor_entrypoint(model_path)
        speech_weights = _gguf_tensor_entrypoint(os.path.join(model_path, "speech_tokenizer"))
        if model_weights and speech_weights:
            return [
                ("model_weights", model_weights),
                ("speech_tokenizer_weights", speech_weights),
            ]
        return []

    source = _gguf_tensor_entrypoint(model_path)
    return [("", source)] if source else []


def _gguf_conversion_unavailable(entry):
    family = entry["family"]
    if family not in GGUF_NATIVE_FAMILIES:
        return _t("当前模型后端暂不支持原生 GGUF。",
                  "This model backend does not currently support native GGUF.")
    if family not in GGUF_WEBUI_CONVERTIBLE_FAMILIES:
        return _t("当前复合模型暂不能在 WebUI 自动转换。",
                  "This composite model cannot yet be converted automatically in the WebUI.")
    return ""


def gguf_status(model_id):
    entry, error = _gguf_entry(model_id, require_installed=False)
    if entry is None:
        return f"⚪ {error}"
    unavailable = _gguf_conversion_unavailable(entry)
    if unavailable:
        return f"⚠️ {unavailable}"
    if not entry["installed"]:
        return _t("🧊 可转换，但模型未完整安装。",
                  "🧊 Convertible, but the model is not fully installed.")
    output = _gguf_output_path(entry)
    converter = _find_gguf_exe()
    if os.path.isfile(output):
        return _t("🧊 已有GGUF，将优先加载该模型。", "🧊 GGUF is available and will be loaded first.")
    if converter is None:
        return _t("⚠️ 找不到转换器。", "⚠️ Converter not found.")
    inputs = _gguf_conversion_inputs(entry)
    if not inputs:
        return _t("⚠️ 未找到可转换的模型权重。", "⚠️ No convertible model weights found.")
    return _t("🧊 可转换。", "🧊 Ready to convert.")


def _gguf_inspection_summary(output, text):
    info, namespaces = {}, []
    for line in (text or "").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "namespace":
            namespaces.append(value)
        else:
            info[key] = value

    yes_no = lambda value: _t("是" if value == "true" else "否",
                              "Yes" if value == "true" else "No")
    rows = [
        _t("✅ 检查完成", "✅ Inspection complete"),
        _t("文件：`{name}`", "File: `{name}`", name=os.path.basename(output)),
        _t("路径：`{path}`", "Path: `{path}`", path=output),
        _t("张量：{count}", "Tensors: {count}", count=info.get("tensors", "-")),
        _t("内嵌资源：{value}", "Embedded sidecars: {value}",
           value=yes_no(info.get("embedded_sidecars"))),
        _t("内嵌模型配置：{value}", "Embedded model spec: {value}",
           value=yes_no(info.get("embedded_model_spec"))),
    ]
    if info.get("model_spec_family"):
        rows.append(_t("模型家族：{family}", "Model family: {family}",
                       family=info["model_spec_family"]))
    if namespaces:
        rows.append(_t("权重命名空间：{items}", "Weight namespaces: {items}",
                       items=", ".join(namespaces)))
    return "  \n".join(rows)


def inspect_gguf(model_id):
    entry, error = _gguf_entry(model_id)
    if entry is None:
        return f"❌ {error}"
    output = _gguf_output_path(entry)
    if not os.path.isfile(output):
        return _t("⚠️ 暂无 GGUF。", "⚠️ No GGUF yet.")
    converter = _find_gguf_exe()
    if converter is None:
        return _t("❌ 找不到 `{name}`；已检查开发构建和 portable 的 gpu/cpu 目录。",
                  "❌ {name} was not found in development or portable gpu/cpu paths.", name=GGUF_EXE_NAME)
    try:
        result = subprocess.run([converter, "--inspect", output], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=120)
    except Exception as exc:
        return _t("❌ GGUF 检查无法启动：{error}", "❌ Could not start GGUF inspection: {error}", error=exc)
    if result.returncode != 0:
        return _t("❌ 检查失败（exit {code}）。", "❌ Inspection failed (exit {code}).", code=result.returncode)
    _ui_log(_t("检查 GGUF：{output}", "checking GGUF: {output}", output=output))
    return _gguf_inspection_summary(output, result.stdout)


def convert_model_to_gguf(model_id, weight_type, progress=gr.Progress()):
    entry, error = _gguf_entry(model_id)
    if entry is None:
        return f"❌ {error}"
    unavailable = _gguf_conversion_unavailable(entry)
    if unavailable:
        return f"❌ {unavailable}"
    output = _gguf_output_path(entry)
    if os.path.isfile(output):
        return _t("⚠️ GGUF 已存在；请先检查或删除。", "⚠️ GGUF already exists; inspect or delete it first.")
    converter = _find_gguf_exe()
    if converter is None:
        return _t("❌ 找不到 `{name}`；已检查开发构建和 portable 的 gpu/cpu 目录。",
                  "❌ {name} was not found in development or portable gpu/cpu paths.", name=GGUF_EXE_NAME)
    inputs = _gguf_conversion_inputs(entry)
    if not inputs:
        return _t("❌ 未找到可自动转换的模型权重。", "❌ No convertible model weights found.")
    if weight_type not in GGUF_TYPES:
        return _t("❌ 不支持的 GGUF 类型：{type}", "❌ Unsupported GGUF type: {type}", type=weight_type)

    root = entry["abs_path"] if os.path.isdir(entry["abs_path"]) else os.path.dirname(inputs[0][1])
    cmd = [converter]
    for namespace, source in inputs:
        cmd.extend(["--input", f"{namespace}={source}" if namespace else source])
    cmd.extend(["--root", root, "--output", output,
                "--type", weight_type, "--family", entry["family"]])
    progress(0, desc=_t("正在转换 GGUF…", "Converting GGUF…"))
    _ui_log(_t("开始转换 GGUF：{label} ({weight_type})",
               "converting GGUF: {label} ({weight_type})",
               label=entry['label'], weight_type=weight_type))
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=7200)
    except subprocess.TimeoutExpired:
        return _t("❌ GGUF 转换超过 2 小时，已停止。", "❌ GGUF conversion exceeded two hours and was stopped.")
    except Exception as exc:
        return _t("❌ 无法启动 GGUF 转换：{error}", "❌ Could not start GGUF conversion: {error}", error=exc)

    if result.returncode != 0 or not os.path.isfile(output):
        _ui_log(_t("GGUF 转换失败：{label} (exit {code})",
                   "GGUF conversion failed: {label} (exit {code})",
                   label=entry['label'], code=result.returncode))
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        details = []
        if stdout:
            details.append(f"stdout:\n{stdout}")
        if stderr:
            details.append(f"stderr:\n{stderr}")
        process_output = "\n\n".join(details) or _t("（转换器没有输出）", "(The converter produced no output.)")
        return _t(
            "❌ 转换失败（exit {code}）。\n\n命令：\n```text\n{command}\n```\n\n详细信息：\n```text\n{output}\n```",
            "❌ Conversion failed (exit {code}).\n\nCommand:\n```text\n{command}\n```\n\nDetails:\n```text\n{output}\n```",
            code=result.returncode, command=subprocess.list2cmdline(cmd), output=process_output)

    stopped = False
    with _proc_lock:
        if _loaded_id == model_id and _server_proc is not None and _server_proc.poll() is None:
            _stop_server()
            stopped = True
    _ui_log(_t("GGUF 转换完成：{label} → {output}",
               "GGUF conversion done: {label} → {output}",
               label=entry['label'], output=output))
    stop_note = (_t("请点『加载模型』。", "Click Load.")
                 if stopped else
                 _t("点『加载模型』即可。", "Click Load to use it."))
    return _t("✅ 转换成功。{note}", "✅ Conversion complete. {note}", note=stop_note)


def delete_gguf(model_id):
    entry, error = _gguf_entry(model_id)
    if entry is None:
        return f"❌ {error}", server_status()
    output = _gguf_output_path(entry)
    if not os.path.isfile(output):
        return _t("⚠️ 暂无 GGUF。", "⚠️ No GGUF to delete."), server_status()
    with _proc_lock:
        if _loaded_id == model_id and _server_proc is not None and _server_proc.poll() is None:
            _stop_server()
        elif server_alive() and model_id in loaded_ids():
            return _t("⚠️ 外部 server 正在使用该 GGUF；请先关闭它再删除。",
                      "⚠️ An external server is using this GGUF. Stop it before deleting."), server_status()
    temporary = output + ".tmp"
    try:
        os.remove(output)
        if os.path.isfile(temporary):
            os.remove(temporary)
    except OSError as exc:
        return _t("❌ 删除 GGUF 失败：{error}", "❌ Could not delete GGUF: {error}", error=exc), server_status()
    _ui_log(_t("删除 GGUF：{output}", "deleting GGUF: {output}", output=output))
    return _t("✅ 已删除：`{path}`", "✅ Deleted: `{path}`", path=output), server_status()


# --- task handlers ---------------------------------------------------------
# Task handlers return (output, message): the reminder/status message is shown
# inline under the output widget instead of as a Gradio popup card.
def _merged_options(prof, adv_values, adv_options):
    """请求 options 合并：family 默认值 -> 生成控件（仅用户改过的项）-> JSON 兜底框。"""
    options = dict(prof.get("default_options", {}))
    if isinstance(adv_values, dict):
        options.update({k: v for k, v in adv_values.items()
                        if v is not None and v != ""})
    options.update(_parse_adv_options(adv_options))
    return options


def _run_task(entry, model, req, timeout, log_label):
    """POST /v1/tasks/run（通用任务路由）并返回响应 JSON；
    连接失败 / 非 200 统一转成带提示的 gr.Error。"""
    try:
        r = requests.post(f"{SERVER}/v1/tasks/run",
                          json={"model": model, "request": req}, timeout=timeout)
    except requests.RequestException as e:
        _ui_log(_t("{log_label}失败：无法连接 server",
                   "{log_label} failed: cannot connect to server",
                   log_label=log_label))
        raise connection_error(e)
    if r.status_code != 200:
        _ui_log(_t("{log_label}失败：server {code}",
                   "{log_label} failed: server {code}",
                   log_label=log_label, code=r.status_code))
        raise server_error(entry, r.status_code, r.text)
    return r.json()


def _resolve_seed(seed):
    """seed=-1 → 每次请求随机抽一个。server/C++ 侧 seed 一律按无符号整数解析
    （多数族 u32，seed_vc/stable_audio u64），负数会直接报错，所以 -1 只能在
    客户端消化；随机范围取 u32 全集，对所有族安全。返回 (seed, 消息后缀)——
    后缀把实际用的 seed 回显在结果里，方便复现。"""
    s = int(seed)
    if s != -1:
        return s, ""
    s = random.randrange(0, 2 ** 32)
    return s, f"🎲 seed={s}"


def do_tts(model, text, language, uploaded_voice, builtin_voice,
           reference_text, seed, max_tokens, adv_values, adv_options,
           progress=gr.Progress()):
    try:
        if not (text or "").strip():
            raise gr.Error(_t("请输入要合成的文字", "Enter text to synthesize."))

        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE

        if prof.get("wrap_speaker_script"):     # e.g. VibeVoice needs Speaker N: lines
            text = _as_speaker_script(text)
            text = _vibevoice_punctuate_script(text)

        # 超短文本拦截（见 _VIBEVOICE_MIN_EST_TOKENS）——放在模型加载之前，
        # 免得为一个注定被拒的请求重启 server。
        is_vibevoice = bool(entry) and entry.get("family") == "vibevoice"
        if is_vibevoice and _vibevoice_text_max_tokens(text) < _VIBEVOICE_MIN_EST_TOKENS:
            raise gr.Error(_t(
                "VibeVoice 是长文模型。请使用 ≥40 个汉字（英文约 ≥35 词），或改用短句模型。",
                "VibeVoice is for long-form text. Use at least ~35 English words, or choose a short-text model."))

        # 必须参考音色的家族（如 IndexTTS2）在加载模型前就拦下来，
        # 免得等几十秒加载后才收到 server 报错。
        if (prof.get("require_voice") and not uploaded_voice
                and (not builtin_voice or builtin_voice == "(none)")):
            raise gr.Error(_t(
                "该模型必须提供参考音色：请上传/录制参考音频，或选择内置音色。",
                "This model requires a voice reference: upload/record one or pick a built-in voice."))

        # 加载/模式切换（可能几十秒）单独计时：状态栏的"用时"只含合成本身，
        # 不报加载会让它看起来远小于实际等待时间。
        t_load = time.time()
        ensure_model_loaded(model, TTS_TASKS)
        load_s = time.time() - t_load
        load_note = (_t("，含模型加载 {seconds:.1f}s", ", model load {seconds:.1f}s",
                        seconds=load_s) if load_s >= 1.0 else "")

        # Model-specific knobs travel in a nested "options" object; the server merges
        # every key into the request options and each model reads what it understands.
        options = _merged_options(prof, adv_values, adv_options)

        voice_path = None
        if uploaded_voice:                      # gradio gives an absolute temp path
            voice_path = uploaded_voice
        elif builtin_voice and builtin_voice != "(none)":
            voice_path = os.path.join(PROMPTS_DIR, builtin_voice)
        has_voice_samples = "voice_samples" in options or "vibevoice.voice_samples" in options
        if is_vibevoice and voice_path and not has_voice_samples:
            options["voice_samples"] = _ensure_wav(voice_path)
            voice_path = None
        # voice_samples (multi-speaker) can't be combined with a single voice_ref.
        if has_voice_samples and voice_path:
            voice_path = None

        # 预置音色家族（Supertonic 的 M1-M5/F1-F5）：控件里的 voice 是请求顶层的
        # cached-voice id，不是 options 项，从 options 里挪出去；有参考音频时以参考为准。
        voice_preset = options.pop("voice", None)
        # Irodori 会话默认 no_ref=true（无参考直接生成），带参考时须显式关掉才走克隆。
        if prof.get("no_ref_toggle") and voice_path:
            options.setdefault("no_ref", False)
        # IndexTTS2：emotion_text 只在 use_emotion_text=true 时生效（request.cpp），
        # 填了情绪参考文本却没勾选是最常见的坑，替用户补上。
        if options.get("emotion_text") and "use_emotion_text" not in options:
            options["use_emotion_text"] = True

        seed, seed_note = _resolve_seed(seed)
        payload = {
            "model": model,
            "language": resolve_language(prof, language),
            "seed": seed,
            "max_tokens": int(max_tokens),
        }
        auto_vibevoice_max_tokens = (
            is_vibevoice and int(max_tokens) == 1200
        )
        if voice_path:
            payload["voice_ref"] = _ensure_wav(voice_path)
        elif voice_preset:
            payload["voice"] = voice_preset
        if (reference_text or "").strip():
            payload["reference_text"] = reference_text
        if options:
            payload["options"] = options

        # Long text goes out as several bounded requests (concatenated below), so a
        # whole chapter neither hits the per-request timeout nor runs blind.
        chunks = _split_tts_chunks(text, prof.get("chunk_chars", 1000))
        if is_vibevoice:
            chunks = _merge_short_vibevoice_tail(chunks)
        _ui_log(_t("TTS 开始：model={model}，{count} 段 / 共 {chars} 字",
                   "TTS started: model={model}, {count} chunks / {chars} chars total",
                   model=model, count=len(chunks), chars=sum(len(c) for c in chunks)))
        t_start = time.time()
        blobs = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                progress((i, len(chunks)), desc=_t("合成 {index}/{total} 段…",
                                                   "Synthesizing {index}/{total}…",
                                                   index=i + 1, total=len(chunks)))
            payload["input"] = chunk
            if auto_vibevoice_max_tokens:
                payload["max_tokens"] = _vibevoice_text_max_tokens(chunk, int(max_tokens))
            t_chunk = time.time()
            try:
                r = requests.post(f"{SERVER}/v1/audio/speech", json=payload, timeout=900)
            except requests.RequestException as e:
                _ui_log(_t("TTS 失败：段 {index}/{count} 无法连接 server",
                           "TTS failed: chunk {index}/{count} cannot connect to server",
                           index=i + 1, count=len(chunks)))
                raise connection_error(e)
            if r.status_code != 200:
                _ui_log(_t("TTS 失败：段 {index}/{count}，server {code}",
                           "TTS failed: chunk {index}/{count}, server {code}",
                           index=i + 1, count=len(chunks), code=r.status_code))
                raise server_error(entry, r.status_code, r.text)
            blobs.append(r.content)
            _ui_log(_t("TTS 段 {index}/{count} 完成（{chars} 字，{seconds:.1f}s）",
                       "TTS chunk {index}/{count} done ({chars} chars, {seconds:.1f}s)",
                       index=i + 1, count=len(chunks), chars=len(chunk),
                       seconds=time.time() - t_chunk))

        out = os.path.join(OUTPUT_DIR, f"audiocpp_tts_{int(time.time()*1000)}.wav")
        if len(blobs) == 1:
            with open(out, "wb") as f:
                f.write(blobs[0])
        else:
            _concat_wavs(blobs, out)
        elapsed = time.time() - t_start
        _ui_log(_t("TTS 完成：{out}，总用时 {seconds:.1f}s",
                   "TTS done: {out}, total {seconds:.1f}s",
                   out=out, seconds=elapsed))
        parts_note = (_t("（{count} 段）", " ({count} parts)", count=len(blobs))
                      if len(blobs) > 1 else "")
        return out, _t("✅ 生成完成{parts}，用时 {seconds:.1f}s{load}。{seed}",
                       "✅ Complete{parts} in {seconds:.1f}s{load}. {seed}",
                       parts=parts_note, seconds=elapsed, load=load_note, seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, _t("❌ 生成失败：{error}", "❌ Generation failed: {error}", error=e)


def do_tts_stream(model, text, language, uploaded_voice, builtin_voice,
                  reference_text, seed, max_tokens, adv_values, adv_options):
    """流式 TTS 生成器：产出 (音频增量, 最终文件, 状态)。
    音频增量是 (sr, np.int16 数组)，喂给 streaming=True 的 gr.Audio 逐段追加
    播放；结束时把完整音频写成 wav 一并给普通输出组件（可下载/回放）。
    server 端 /v1/audio/speech stream_format=sse 的 delta 是 base64 裸 PCM16。"""
    if not (text or "").strip():
        raise gr.Error(_t("请输入要合成的文字", "Enter text to synthesize."))
    entry = catalog_by_id(model)
    prof = profile_for(entry) if entry else DEFAULT_PROFILE
    if not prof.get("supports_streaming"):
        raise gr.Error(_t("模型 {model} 不支持流式生成。", "Model {model} does not support streaming.",
                          model=model))
    sr = int(prof.get("stream_sample_rate") or 0)
    if sr <= 0:
        raise gr.Error(_t("模型 {model} 缺少流式采样率配置。",
                          "Model {model} has no streaming sample-rate setting.", model=model))

    t_load = time.time()
    ensure_model_loaded(model, TTS_TASKS, mode="streaming")
    load_s = time.time() - t_load
    load_note = (_t("，含模型加载 {seconds:.1f}s", ", model load {seconds:.1f}s",
                    seconds=load_s) if load_s >= 1.0 else "")
    options = _merged_options(prof, adv_values, adv_options)
    family = entry.get("family") if entry else ""
    if family == "voxcpm2":
        # VoxCPM2 流式生成的硬性要求（generator.cpp:1259）：badcase 重试要
        # 重新生成整段，与已经推给播放器的音频冲突，所以流式下强制关闭。
        options["retry_badcase"] = False
        options.pop("voxcpm2.retry_badcase", None)

    voice_path = None
    if uploaded_voice:
        voice_path = uploaded_voice
    elif builtin_voice and builtin_voice != "(none)":
        voice_path = os.path.join(PROMPTS_DIR, builtin_voice)
    # Supertonic 等预置音色家族要求 voice 位于请求顶层，而不是 options。
    voice_preset = options.pop("voice", None)

    seed, seed_note = _resolve_seed(seed)
    payload = {
        "model": model,
        "language": resolve_language(prof, language),
        "seed": seed,
        "max_tokens": int(max_tokens),
        "stream": True,
        "stream_format": "sse",
        "response_format": "pcm",
        "options": options,
    }
    if voice_path:
        payload["voice_ref"] = _ensure_wav(voice_path)
    elif voice_preset:
        payload["voice"] = voice_preset
    if (reference_text or "").strip():
        payload["reference_text"] = reference_text

    chunks = _split_tts_chunks(text, prof.get("chunk_chars", 1000))
    _ui_log(_t("TTS 开始（流式）：model={model}，{count} 段 / 共 {chars} 字",
               "TTS started (streaming): model={model}, {count} chunks / {chars} chars total",
               model=model, count=len(chunks), chars=sum(len(c) for c in chunks)))
    t_start = time.time()
    all_parts = []          # 全部 PCM（拼最终 wav）
    pending = []            # 未推给播放器的 PCM 增量（凑批再推，免得刷屏）
    pending_samples = 0
    min_push = int(sr * 0.4)   # 每 ~0.4s 音频推一次播放器
    ttft_note = ""

    def _flush():
        nonlocal pending, pending_samples
        if not pending:
            return None
        arr = pending[0] if len(pending) == 1 else np.concatenate(pending)
        pending, pending_samples = [], 0
        return arr

    for i, chunk in enumerate(chunks):
        seg_note = (_t("（{index}/{total} 段）", " ({index}/{total})",
                       index=i + 1, total=len(chunks)) if len(chunks) > 1 else "")
        payload["input"] = chunk
        t_chunk = time.time()
        try:
            r = requests.post(f"{SERVER}/v1/audio/speech", json=payload,
                              stream=True, timeout=900)
        except requests.RequestException as e:
            _ui_log(_t("TTS 失败（流式）：段 {index}/{count} 无法连接 server",
                       "TTS failed (streaming): chunk {index}/{count} cannot connect to server",
                       index=i + 1, count=len(chunks)))
            raise connection_error(e)
        with r:
            if r.status_code != 200:
                _ui_log(_t("TTS 失败（流式）：段 {index}/{count}，server {code}",
                           "TTS failed (streaming): chunk {index}/{count}, server {code}",
                           index=i + 1, count=len(chunks), code=r.status_code))
                raise server_error(entry, r.status_code, r.text)
            for event in _iter_sse_events(r):
                etype = event.get("type")
                if etype == "speech.audio.delta":
                    pcm = base64.b64decode(event.get("audio") or "")
                    if not pcm:
                        continue
                    arr = np.frombuffer(pcm, dtype=np.int16)
                    all_parts.append(arr)
                    pending.append(arr)
                    pending_samples += arr.size
                    if pending_samples >= min_push:
                        out_arr = _flush()
                        done_s = sum(a.size for a in all_parts) / sr
                        yield ((sr, out_arr), None,
                               _t("⏳ 流式生成中{segment}…已生成 {seconds:.1f}s",
                                  "⏳ Streaming{segment}… {seconds:.1f}s generated",
                                  segment=seg_note, seconds=done_s))
                elif etype == "speech.audio.done":
                    if i == 0 and not ttft_note:
                        ttft = (event.get("timing") or {}).get("ttft_ms")
                        if ttft:
                            ttft_note = _t("，首包 {seconds:.1f}s", ", first audio {seconds:.1f}s",
                                           seconds=ttft / 1000)
        _ui_log(_t("TTS 段 {index}/{count} 完成（流式，{chars} 字，{seconds:.1f}s）",
                   "TTS chunk {index}/{count} done (streaming, {chars} chars, {seconds:.1f}s)",
                   index=i + 1, count=len(chunks), chars=len(chunk),
                   seconds=time.time() - t_chunk))

    tail = _flush()
    if tail is not None:
        yield (sr, tail), None, _t("⏳ 流式生成收尾…", "⏳ Finishing stream…")
    if not all_parts:
        raise gr.Error(_t("流式生成没有产出音频。", "Streaming produced no audio."))
    out = os.path.join(OUTPUT_DIR, f"audiocpp_tts_stream_{int(time.time() * 1000)}.wav")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.concatenate(all_parts).tobytes())
    elapsed = time.time() - t_start
    audio_s = sum(a.size for a in all_parts) / sr
    parts_note = (_t("（{count} 段）", " ({count} parts)", count=len(chunks))
                  if len(chunks) > 1 else "")
    _ui_log(_t("TTS 完成（流式）：{out}，音频 {audio:.1f}s，总用时 {seconds:.1f}s",
               "TTS done (streaming): {out}, audio {audio:.1f}s, total {seconds:.1f}s",
               out=out, audio=audio_s, seconds=elapsed))
    yield (None, out, _t(
        "✅ 流式生成完成{parts}，音频 {audio:.1f}s，用时 {elapsed:.1f}s{ttft}{load}。{seed}",
        "✅ Stream complete{parts}: {audio:.1f}s audio in {elapsed:.1f}s{ttft}{load}. {seed}",
        parts=parts_note, audio=audio_s, elapsed=elapsed, ttft=ttft_note,
        load=load_note, seed=seed_note))


def do_tts_or_stream(model, gen_mode, text, language, uploaded_voice, builtin_voice,
                     reference_text, seed, max_tokens, adv_values, adv_options,
                     progress=gr.Progress()):
    """TTS 按钮统一入口：离线模式原样走 do_tts（行为不变），流式模式走
    do_tts_stream。输出：(流式播放增量, 输出文件, 状态)。
    先 yield 一条即时提示——模型未加载/需切换模式时后续要静默等几十秒。"""
    yield None, None, _t("⏳ 生成中…模型切换时可能需要重新加载。",
                         "⏳ Generating… model switches may require a reload.")
    if gen_mode == "流式":
        try:
            yield from do_tts_stream(model, text, language, uploaded_voice,
                                     builtin_voice, reference_text, seed,
                                     max_tokens, adv_values, adv_options)
        except gr.Error as e:
            yield None, None, _msg_from_error(e)
        except Exception as e:
            yield None, None, _t("❌ 流式生成失败：{error}",
                                 "❌ Streaming failed: {error}", error=e)
        return
    # 离线合成本身是阻塞的 do_tts。旧版把它直接绑到按钮（普通函数），Gradio 会在
    # 输出组件上显示原生的“处理中 X.Xs”计时；现在统一入口是 generator，上面那次
    # 即时 yield 会顶掉原生计时器，于是这里自己计时：后台线程跑 do_tts，主生成器
    # 每 ~0.5s 吐一次“已用时 Xs”，把秒数显示找回来（也顺带覆盖模型重载的等待）。
    result = {}

    def _run_offline():
        result["value"] = do_tts(model, text, language, uploaded_voice, builtin_voice,
                                 reference_text, seed, max_tokens, adv_values, adv_options,
                                 progress=progress)

    worker = threading.Thread(target=_run_offline, daemon=True)
    t_offline = time.time()
    worker.start()
    while worker.is_alive():
        worker.join(0.5)
        if worker.is_alive():
            yield None, None, _t("⏳ 生成中…已用时 {seconds:.1f}s",
                                 "⏳ Generating… {seconds:.1f}s elapsed",
                                 seconds=time.time() - t_offline)
    out, msg = result.get("value",
                          (None, _t("❌ 生成失败：无返回结果。", "❌ Generation failed: no result.")))
    yield None, out, msg


def _iter_sse_events(response):
    """逐个产出 SSE 事件（dict）。server 每个事件一行 `data: {json}`，事件间空行
    分隔（write_sse）；`data: [DONE]` 表示流结束。type=error 的事件直接抛错。"""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if event.get("type") == "error":
            err = event.get("error") or {}
            raise gr.Error(_t("server 流式错误：{error}", "Server stream error: {error}",
                              error=err.get("message") or event))
        yield event


def _asr_transcribe_stream(model, entry, wav_path, extras, tag="ASR"):
    """流式转写一个 WAV（server 需已按 mode=streaming 加载）：产出
    (累计文本, 是否最终, ttft_ms)。整个文件一个请求——增量出字本身就解决了
    离线路径靠分段缓解的"长音频干等"问题。"""
    payload = {"model": model, "audio": wav_path, "stream": True, **extras}
    try:
        r = requests.post(f"{SERVER}/v1/audio/transcriptions", json=payload,
                          stream=True, timeout=900)
    except requests.RequestException as e:
        _ui_log(_t("{tag} 失败：无法连接 server",
                   "{tag} failed: cannot connect to server", tag=tag))
        raise connection_error(e)
    with r:
        if r.status_code != 200:
            _ui_log(_t("{tag} 失败：server {code}",
                       "{tag} failed: server {code}", tag=tag, code=r.status_code))
            raise server_error(entry, r.status_code, r.text)
        text = ""
        for event in _iter_sse_events(r):
            etype = event.get("type")
            if etype == "transcript.text.delta":
                delta = event.get("delta") or ""
                if entry.get("family") == "voxtral_realtime":
                    # Voxtral currently emits the cumulative transcript (and may
                    # dispatch the same event twice), despite the server exposing
                    # it as a delta. Replace instead of appending for this family.
                    if delta == text:
                        continue
                    text = delta
                else:
                    text += delta
                yield text, False, None
            elif etype == "transcript.text.done":
                final = (event.get("text") or text).strip()
                ttft = (event.get("timing") or {}).get("ttft_ms")
                yield final, True, ttft
                return
    raise gr.Error(_t("流式转写未收到最终结果。", "Streaming transcription returned no final result."))


def _asr_transcribe_wav(model, entry, prof, wav_path, extras, tag="ASR"):
    """转写一个 WAV 文件（调用方已加载好 ASR 模型）：超过单次上限（见 qwen3_asr
    profile：8G 卡显存实测取 60s）时在静音处切段逐段转写再拼接；短音频/非 PCM
    WAV（测不出时长）走单请求。extras 是可选的 language/context 请求字段。
    返回 (text, dur, 段数)。"""
    dur = _audio_duration_seconds(wav_path)
    dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
    max_s = prof.get("max_input_seconds")
    chunks = [wav_path]
    if max_s and dur is not None and dur > max_s:
        chunks = [p for p, _ in
                  _split_wav_chunks(wav_path, max_s, pad_to_max=False)]
        if len(chunks) > 1:
            _ui_log(_t("{tag} 长音频分段：{dur_note} → {count} 段（每段 ≤{max_s:.0f}s，静音处切分）",
                       "{tag} long-audio split: {dur_note} → {count} chunks (each ≤{max_s:.0f}s, split at silence)",
                       tag=tag, dur_note=dur_note, count=len(chunks), max_s=max_s))
    texts = []
    for i, chunk in enumerate(chunks):
        t_chunk = time.time()
        payload = {"model": model, "audio": chunk, **extras}
        try:
            r = requests.post(f"{SERVER}/v1/audio/transcriptions", json=payload, timeout=900)
        except requests.RequestException as e:
            _ui_log(_t("{tag} 失败：无法连接 server",
                       "{tag} failed: cannot connect to server", tag=tag))
            raise connection_error(e)
        if r.status_code != 200:
            seg_note = (_t("，第 {index}/{total} 段", ", part {index}/{total}",
                           index=i + 1, total=len(chunks)) if len(chunks) > 1 else "")
            _ui_log(_t("{tag} 失败：server {code}（音频 {dur_note}{seg_note}）",
                       "{tag} failed: server {code} (audio {dur_note}{seg_note})",
                       tag=tag, code=r.status_code, dur_note=dur_note, seg_note=seg_note))
            extra = (_t("⏱ 本次音频时长约 {seconds:.1f} 秒", "⏱ audio is about {seconds:.1f}s long",
                        seconds=dur) if dur is not None else None)
            raise server_error(entry, r.status_code, r.text, extra=extra)
        try:
            data = r.json()
            texts.append((data.get("text") or str(data)).strip())
        except Exception:
            texts.append(r.text)
        if len(chunks) > 1:
            _ui_log(_t("{tag} 段 {index}/{count} 完成（{seconds:.1f}s）",
                       "{tag} chunk {index}/{count} done ({seconds:.1f}s)",
                       tag=tag, index=i + 1, count=len(chunks),
                       seconds=time.time() - t_chunk))
    text = texts[0] if len(chunks) == 1 else "\n".join(t for t in texts if t)
    return text, dur, len(chunks)


def do_asr(model, audio_path, language="", context="", dialogue=False, stream=False):
    """生成器：非流式路径只 yield 一次最终结果（行为与旧版 return 完全一致——
    Gradio 对生成器处理器逐次刷新输出）；流式路径边收 SSE 增量边 yield。"""
    try:
        if not audio_path:
            raise gr.Error(_t("请上传或录制音频", "Upload or record audio."))
        audio_path = _ensure_wav(audio_path)
        # 可选转写参数：留空不发，请求体和原来完全一致（qwen3_asr 从 text_input
        # 读 context/language，其它族忽略；server 端 build_openai_transcription_request
        # 只在字段存在时才设置 text_input）。
        extras = {}
        if (language or "").strip():
            extras["language"] = language.strip()
        if (context or "").strip():
            extras["context"] = context.strip()
        if dialogue:
            # 对话模式走 Sortformer 切段 + 逐段离线转写，与流式互斥（勾了也忽略）。
            yield _asr_dialogue(model, audio_path, extras)
            return
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE

        if stream and prof.get("supports_streaming"):
            yield "", _t("⏳ 转写中…模型切换时可能需要重新加载。",
                          "⏳ Transcribing… model switches may require a reload.")
            t_load = time.time()
            ensure_model_loaded(model, ASR_TASKS, mode="streaming")
            load_s = time.time() - t_load
            load_note = (_t("，含模型加载 {seconds:.1f}s", ", model load {seconds:.1f}s",
                            seconds=load_s) if load_s >= 1.0 else "")
            extras_note = "".join(f"，{k}={v[:20]}" for k, v in extras.items())
            _ui_log(_t("ASR 开始（流式）：model={model}{extras_note}",
                       "ASR started (streaming): model={model}{extras_note}",
                       model=model, extras_note=extras_note))
            t_start = time.time()
            dur = _audio_duration_seconds(audio_path)
            dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
            last_yield = 0.0
            for text, is_final, ttft in _asr_transcribe_stream(
                    model, entry, audio_path, extras):
                if is_final:
                    elapsed = time.time() - t_start
                    ttft_note = (_t("，首字 {seconds:.1f}s", ", first text {seconds:.1f}s",
                                    seconds=ttft / 1000) if ttft else "")
                    _ui_log(_t("ASR 完成（流式）：音频 {dur_note}，用时 {seconds:.1f}s",
                               "ASR done (streaming): audio {dur_note}, elapsed {seconds:.1f}s",
                               dur_note=dur_note, seconds=elapsed))
                    yield text, _t(
                        "✅ 流式转写完成（音频 {duration}），用时 {elapsed:.1f}s{ttft}{load}。",
                        "✅ Streaming transcript complete ({duration}) in {elapsed:.1f}s{ttft}{load}.",
                        duration=dur_note, elapsed=elapsed, ttft=ttft_note, load=load_note)
                    return
                # 增量刷新节流：delta 可能非常密，0.15s 一次足够"边转边出字"的观感
                now = time.time()
                if now - last_yield >= 0.15:
                    last_yield = now
                    yield text, _t("⏳ 流式转写中…（音频 {duration}）",
                                   "⏳ Streaming transcript… ({duration})", duration=dur_note)
            return
        if stream and entry is not None:
            raise gr.Error(_t("模型 {model} 不支持流式转写。",
                              "Model {model} does not support streaming transcription.",
                              model=model))

        yield "", _t("⏳ 转写中…模型切换时可能需要重新加载。",
                      "⏳ Transcribing… model switches may require a reload.")
        t_load = time.time()
        ensure_model_loaded(model, ASR_TASKS)
        load_s = time.time() - t_load
        load_note = (_t("，含模型加载 {seconds:.1f}s", ", model load {seconds:.1f}s",
                        seconds=load_s) if load_s >= 1.0 else "")
        extras_note = "".join(f"，{k}={v[:20]}" for k, v in extras.items())
        _ui_log(_t("ASR 开始：model={model}{extras_note}",
                   "ASR started: model={model}{extras_note}",
                   model=model, extras_note=extras_note))
        t_start = time.time()
        text, dur, n = _asr_transcribe_wav(model, entry, prof, audio_path, extras)
        elapsed = time.time() - t_start
        dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
        parts_note = (_t("，{count} 段", ", {count} parts", count=n) if n > 1 else "")
        _ui_log(_t("ASR 完成：音频 {dur_note}{parts_note}，用时 {seconds:.1f}s",
                   "ASR done: audio {dur_note}{parts_note}, elapsed {seconds:.1f}s",
                   dur_note=dur_note, parts_note=parts_note, seconds=elapsed))
        yield text, _t("✅ 转写完成（音频 {duration}{parts}），用时 {elapsed:.1f}s{load}。",
                       "✅ Transcript complete ({duration}{parts}) in {elapsed:.1f}s{load}.",
                       duration=dur_note, parts=parts_note, elapsed=elapsed, load=load_note)
    except gr.Error as e:
        yield "", _msg_from_error(e)
    except Exception as e:
        yield "", _t("❌ 转写失败：{error}", "❌ Transcription failed: {error}", error=e)


# 对话模式：同一说话人相邻发言段合并的最大间隔 / 每段前后补的余量（防止
# Sortformer 边界切掉字头字尾）/ 短于该值的发言段丢弃（多为口头禅、气口）。
DIALOGUE_MERGE_GAP_S = 1.0
DIALOGUE_PAD_S = 0.25
DIALOGUE_MIN_SEG_S = 0.3

# Sortformer CUDA 下是固定图容量：session_len_sec 定容量（默认 20s），上限来自
# tf_encoder.max_source_positions=1500 @ 12.5 编码帧/秒 = 120s。
SORTFORMER_MAX_SEC = 120


def _diar_session_options(dur):
    """说话人分离的动态 session_len_sec：≤20s 用默认容量（不重载）；更长的按
    30s 步进向上取整重载（避免每个文件都重启 server）；超过 120s 上限报错。"""
    if dur is None or dur <= 20:
        return None
    if dur > SORTFORMER_MAX_SEC:
        raise gr.Error(_t("说话人分离最长约 {seconds} 秒，请先剪短音频。",
                          "Speaker diarization is limited to about {seconds}s. Shorten the audio.",
                          seconds=SORTFORMER_MAX_SEC))
    return {"session_len_sec": str(min(SORTFORMER_MAX_SEC,
                                       ((int(dur) // 30) + 1) * 30))}


def _fmt_mmss(sec):
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _first_installed_model(task):
    for m in catalog_models():
        if m.get("task") == task and m.get("installed"):
            return m
    return None


def _asr_dialogue(model, wav_path, extras):
    """对话模式：Sortformer 说话人分离 → 按说话人合并/切段 → 逐段 ASR →
    带说话人标签和时间戳的对话稿。server 一次只驻留一个模型，所以先换载 diar
    再换回 ASR（每次自动换载 ~5s）。"""
    diar = _first_installed_model("diar")
    if diar is None:
        raise gr.Error(_t("对话模式需要 Sortformer，请先在音频分析页下载安装。",
                          "Dialogue mode requires Sortformer. Install it from Audio analysis."))
    entry = catalog_by_id(model)
    prof = profile_for(entry) if entry else DEFAULT_PROFILE

    wav16 = _to_16k_mono_wav(wav_path)
    try:
        with wave.open(wav16, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
    except Exception as e:
        raise gr.Error(_t("无法读取对话音频：{error}",
                          "Cannot read dialogue audio: {error}", error=e))
    samples = np.frombuffer(raw, dtype=np.int16)
    total_dur = len(samples) / float(sr)

    t_start = time.time()
    _ui_log(_t("对话模式：先用 {id} 做说话人分离（音频 {seconds:.1f}s）",
               "dialogue mode: first run speaker diarization with {id} (audio {seconds:.1f}s)",
               id=diar['id'], seconds=total_dur))
    ensure_model_loaded(diar["id"], ("diar",),
                        session_options=_diar_session_options(total_dur))
    data = _run_task(diar, diar["id"], {"audio": wav16}, timeout=900,
                     log_label="说话人分离")
    turns = data.get("speaker_turns") or []
    if not turns:
        raise gr.Error(_t("没有检测到说话人发言段。", "No speaker turns were detected."))

    # 合并同一说话人的相邻发言段（间隔 ≤1s 且合并后不超过 ASR 单次上限），
    # 减少请求数并给 ASR 更完整的上下文。
    max_len = (prof.get("max_input_seconds") or 60) * sr
    merged = []                                     # [说话人, start, end] (样本数)
    for t in sorted(turns, key=lambda t: t["start_sample"]):
        spk, s, e = str(t.get("speaker_id", "?")), t["start_sample"], t["end_sample"]
        if (merged and merged[-1][0] == spk
                and s - merged[-1][2] <= DIALOGUE_MERGE_GAP_S * sr
                and e - merged[-1][1] <= max_len):
            merged[-1][2] = max(merged[-1][2], e)
        else:
            merged.append([spk, s, e])
    merged = [m for m in merged if m[2] - m[1] >= DIALOGUE_MIN_SEG_S * sr]
    if not merged:
        raise gr.Error(_t("说话人发言段都太短，无法转写。",
                          "All detected speaker turns are too short to transcribe."))
    _ui_log(_t("说话人分离完成：{turns} 个发言段 → 合并为 {merged} 段；换回 {model} 逐段转写",
               "speaker diarization done: {turns} turns → merged into {merged} segments; switching back to {model} for per-segment transcription",
               turns=len(turns), merged=len(merged), model=model))

    ensure_model_loaded(model, ASR_TASKS)
    pad = int(DIALOGUE_PAD_S * sr)
    speakers, lines = [], []
    for idx, (spk, s, e) in enumerate(merged, 1):
        a, b = max(0, s - pad), min(len(samples), e + pad)
        fd, seg_path = tempfile.mkstemp(prefix=f"audiocpp_dlg{idx}_", suffix=".wav")
        os.close(fd)
        with wave.open(seg_path, "wb") as ww:
            ww.setnchannels(1)
            ww.setsampwidth(2)
            ww.setframerate(sr)
            ww.writeframes(samples[a:b].tobytes())
        text, _, _ = _asr_transcribe_wav(model, entry, prof, seg_path, extras,
                                         tag=f"对话段 {idx}/{len(merged)}")
        if spk not in speakers:
            speakers.append(spk)
        label = _t("说话人{index}", "Speaker {index}", index=speakers.index(spk) + 1)
        if text:
            lines.append(f"[{_fmt_mmss(s / sr)}-{_fmt_mmss(e / sr)}] {label}: {text}")
        _ui_log(_t("对话段 {index}/{count} 完成（{label}，{seconds:.1f}s）",
                   "dialogue turn {index}/{count} done ({label}, {seconds:.1f}s)",
                   index=idx, count=len(merged), label=label, seconds=(e - s) / sr))

    elapsed = time.time() - t_start
    _ui_log(_t("对话转写完成：{merged} 段发言、{speakers} 个说话人，用时 {seconds:.1f}s",
               "dialogue transcription done: {merged} turns, {speakers} speakers, elapsed {seconds:.1f}s",
               merged=len(merged), speakers=len(speakers), seconds=elapsed))
    if not lines:
        return "", _t("⚠️ 检测到发言段，但没有转写出文字。",
                      "⚠️ Speaker turns were detected, but no text was transcribed.")
    return ("\n".join(lines), _t(
        "✅ 对话转写完成（音频 {duration:.1f}s，{turns} 段，{speakers} 人），用时 {elapsed:.1f}s。",
        "✅ Dialogue transcript complete ({duration:.1f}s, {turns} turns, {speakers} speakers) in {elapsed:.1f}s.",
        duration=total_dur, turns=len(merged), speakers=len(speakers), elapsed=elapsed))


def do_music_gen(model, text, lyrics, source_audio, duration, seed,
                 adv_values, adv_options):
    """Music/SFX generation via the generic /v1/tasks/run route. The request
    object uses the CLI request-JSON fields (text/lyrics/duration_seconds/
    task_route/audio + an options map); the response carries base64 WAV."""
    try:
        if not (text or "").strip():
            raise gr.Error(_t("请输入音乐/音效提示词", "Enter a music or sound prompt."))
        ensure_model_loaded(model, GEN_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        seed, seed_note = _resolve_seed(seed)
        req = {"text": text, "seed": seed}
        # task_route is a top-level request field (not a model option); the
        # generated controls funnel everything through `options`, so lift it out.
        route = options.pop("task_route", None)
        if route:
            req["task_route"] = route
        if (lyrics or "").strip():
            req["lyrics"] = lyrics
        if duration is not None and float(duration) != 0:
            req["duration_seconds"] = float(duration)
        if source_audio:
            req["audio"] = _ensure_wav(source_audio)
        if options:
            req["options"] = options

        dur_note = req.get("duration_seconds", _t("自动", "auto"))
        _ui_log(_t("音乐生成开始：model={model}，目标时长 {dur_note}s",
                   "music generation started: model={model}, target duration {dur_note}s",
                   model=model, dur_note=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=1800, log_label="音乐生成")
        b64 = data.get("audio")
        if not b64 and data.get("named_audio_outputs"):
            b64 = data["named_audio_outputs"][0].get("audio")
        if not b64:
            raise gr.Error(_t("server 没有返回音频数据。", "The server returned no audio."))
        out = os.path.join(OUTPUT_DIR, f"audiocpp_gen_{int(time.time()*1000)}.wav")
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        elapsed = time.time() - t_start
        _ui_log(_t("音乐生成完成：{out}，用时 {seconds:.1f}s",
                   "music generation done: {out}, elapsed {seconds:.1f}s",
                   out=out, seconds=elapsed))
        return out, _t("✅ 生成完成，用时 {seconds:.1f}s。{seed}",
                       "✅ Complete in {seconds:.1f}s. {seed}", seconds=elapsed, seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, _t("❌ 生成失败：{error}", "❌ Generation failed: {error}", error=e)


# analyze 返回的字段 -> ACE-Step 高级参数（model_params.json 里的 name）
_ANALYZE_FILL_MAP = (("caption", "source_caption"), ("lyrics", "source_lyrics"),
                     ("bpm", "bpm"), ("keyscale", "keyscale"),
                     ("timesignature", "timesignature"))


def do_music_analyze(model, source_audio, seed, adv_values):
    """『🔍 分析源音频』（仅 ACE-Step）：task_route=analyze 把源音频编码成语义 code，
    再用 5Hz LM 反推 caption/歌词/BPM/调性/拍号，回填到高级参数
    （写进 state 并通过 prefill state 触发控件重渲染，让填进去的值可见可改）。
    返回 (adv_state, prefill, message)。"""
    adv_values = dict(adv_values or {})
    try:
        if not source_audio:
            raise gr.Error(_t("请先上传源音频", "Upload source audio first."))
        entry = catalog_by_id(model)
        if not entry or entry.get("family") != "ace_step":
            raise gr.Error(_t("只有 ACE-Step 支持源音频分析。",
                              "Source analysis is available for ACE-Step only."))
        ensure_model_loaded(model, GEN_TASKS)

        # 分析要可复现：seed=-1（随机）时固定为 1234，不跟生成共享随机性；
        # 用户显式填的固定 seed 仍然生效（可换 seed 重抽歌词转写）。
        analyze_seed = 1234 if int(seed if seed is not None else -1) == -1 else int(seed)
        req = {"text": "analyze", "task_route": "analyze",
               "audio": _ensure_wav(source_audio), "seed": analyze_seed}
        dur = _audio_duration_seconds(req["audio"])
        dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
        _ui_log(_t("源音频分析开始：model={model}，音频 {dur_note}",
                   "source audio analysis started: model={model}, audio {dur_note}",
                   model=model, dur_note=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=1800, log_label="源音频分析")
        raw = data.get("text") or ""
        try:
            info = json.loads(raw)
        except Exception:
            raise gr.Error(_t("server 返回的分析结果不是 JSON：{result}",
                              "Server analysis result is not JSON: {result}", result=raw[:200]))
        elapsed = time.time() - t_start

        for src, dst in _ANALYZE_FILL_MAP:
            v = info.get(src)
            if v is None or v == "" or v == 0:
                continue
            adv_values[dst] = v

        lines = [_t("✅ 分析完成（音频 {duration}），用时 {elapsed:.1f}s；已回填高级参数。",
                    "✅ Analysis complete ({duration}) in {elapsed:.1f}s; advanced options updated.",
                    duration=dur_note, elapsed=elapsed)]
        labels = (("caption", _t("描述", "Caption")), ("bpm", "BPM"),
                  ("keyscale", _t("调性", "Key")),
                  ("timesignature", _t("拍号", "Time signature")),
                  ("language", _t("语言", "Language")),
                  ("genres", _t("流派", "Genres")),
                  ("duration", _t("时长(s)", "Duration (s)")))
        for key, label in labels:
            v = info.get(key)
            if v not in (None, "", 0):
                lines.append(f"- **{label}**：{v}")
        if info.get("lyrics"):
            lines.append(_t("- **歌词**：\n```\n{lyrics}\n```",
                            "- **Lyrics**:\n```\n{lyrics}\n```", lyrics=info["lyrics"]))
        _ui_log(_t("源音频分析完成：用时 {seconds:.1f}s",
                   "source audio analysis done: elapsed {seconds:.1f}s",
                   seconds=elapsed))
        return adv_values, dict(adv_values), "\n".join(lines)
    except gr.Error as e:
        return adv_values, gr.skip(), _msg_from_error(e)
    except Exception as e:
        return adv_values, gr.skip(), _t("❌ 分析失败：{error}",
                                         "❌ Analysis failed: {error}", error=e)


def do_vc(model, source_audio, target_upload, builtin_voice, seed,
          adv_values, adv_options, progress=gr.Progress()):
    """声音/歌声转换（vc/svc/s2s），走通用 /v1/tasks/run 路由：`audio` 是源音频，
    `voice_ref` 是目标音色。seed_vc/miocodec 直接用这两个字段；vevo2 也接受它们
    （audio_input/voice speaker 是 source_audio/target_voice 选项的回退），
    风格转换类 route 的额外字段（style_ref 等）由“其它参数(JSON)”兜底。
    profile 带 vc_chunk_seconds 的族（vevo2 FM 图按整段建，8G 卡长音频必炸）
    超限时按低能量点分段逐段转换，同一 voice_ref 保证各段音色一致，最后拼接。"""
    try:
        if not source_audio:
            raise gr.Error(_t("请上传要转换的源音频", "Upload source audio to convert."))
        ensure_model_loaded(model, VC_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        source_audio = _ensure_wav(source_audio)
        # 同一 seed 用于所有分段，保证各段结果一致可复现。
        seed, seed_note = _resolve_seed(seed)
        req = {"seed": seed}
        voice_path = target_upload or (
            os.path.join(PROMPTS_DIR, builtin_voice)
            if builtin_voice and builtin_voice != "(none)" else None)
        if voice_path:
            ref_wav = _ensure_wav(voice_path)
            ref_cap = prof.get("vc_ref_max_seconds")
            if ref_cap:                     # 参考音色截短，别让 prompt 段吃满显存
                ref_wav = _trim_wav_seconds(ref_wav, ref_cap)
            req["voice_ref"] = ref_wav
        if options:
            req["options"] = options

        # vevo2 的 FM 图一次建图，序列长度 = 参考音色(prompt) + 每段源(target)。按
        # 显存预算反推每段源时长（预算 − 参考时长），令 cond 稳定落在 8G 内；没有
        # 显存预算/参考时的族仍按 vc_chunk_seconds 上限或整段发送。
        chunk_cap = prof.get("vc_chunk_seconds")
        ref_sec = _audio_duration_seconds(req.get("voice_ref")) or 0.0
        if chunk_cap and prof.get("vc_fm_budget_seconds"):
            min_chunk = prof.get("vc_min_chunk_seconds", 6)
            budget = prof["vc_fm_budget_seconds"]
            chunk_cap = int(max(min_chunk, min(chunk_cap, round(budget - ref_sec))))
        pieces = (_split_wav_chunks(source_audio, chunk_cap)
                  if chunk_cap else [(source_audio, 1.0)])
        dur = _audio_duration_seconds(source_audio)
        dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
        seg_note = (_t("，参考 {reference:.0f}s，自适应分 {count} 段（每段 ≤{cap}s）",
                       ", {reference:.0f}s reference, {count} adaptive parts (≤{cap}s)",
                       reference=ref_sec, count=len(pieces), cap=chunk_cap)
                    if len(pieces) > 1 else "")
        _ui_log(_t("声音转换开始：model={model}，源音频 {dur_note}{seg_note}",
                   "voice conversion started: model={model}, source audio {dur_note}{seg_note}",
                   model=model, dur_note=dur_note, seg_note=seg_note))
        t_start = time.time()
        blobs, ratios = [], []
        for i, (piece, ratio) in enumerate(pieces):
            if len(pieces) > 1:
                progress((i, len(pieces)), desc=_t("转换 {index}/{total} 段…",
                                                   "Converting {index}/{total}…",
                                                   index=i + 1, total=len(pieces)))
            req["audio"] = piece
            t_seg = time.time()
            data = _run_task(entry, model, req, timeout=1800, log_label="声音转换")
            b64 = data.get("audio")
            if not b64 and data.get("named_audio_outputs"):
                b64 = data["named_audio_outputs"][0].get("audio")
            if not b64:
                raise gr.Error(_t("server 没有返回音频数据。", "The server returned no audio."))
            blobs.append(base64.b64decode(b64))
            ratios.append(ratio)
            if len(pieces) > 1:
                _ui_log(_t("声音转换段 {index}/{count} 完成（{seconds:.1f}s）",
                           "voice conversion segment {index}/{count} done ({seconds:.1f}s)",
                           index=i + 1, count=len(pieces), seconds=time.time() - t_seg))
        out = os.path.join(OUTPUT_DIR, f"audiocpp_vc_{int(time.time()*1000)}.wav")
        if len(blobs) == 1 and ratios[0] >= 1.0:
            with open(out, "wb") as f:
                f.write(blobs[0])
        else:
            _concat_wavs(blobs, out, keep_ratios=ratios)
        elapsed = time.time() - t_start
        _ui_log(_t("声音转换完成：{out}，用时 {seconds:.1f}s",
                   "voice conversion done: {out}, elapsed {seconds:.1f}s",
                   out=out, seconds=elapsed))
        parts_note = (_t("（{count} 段拼接）", " ({count} joined parts)", count=len(blobs))
                      if len(blobs) > 1 else "")
        return out, _t("✅ 转换完成{parts}，用时 {seconds:.1f}s。{seed}",
                       "✅ Conversion complete{parts} in {seconds:.1f}s. {seed}",
                       parts=parts_note, seconds=elapsed, seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, _t("❌ 转换失败：{error}", "❌ Conversion failed: {error}", error=e)


# 分轨 id -> 中文标签；未收录的 id 原样显示。
STEM_LABELS = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "其它",
               "instrumental": "伴奏", "accompaniment": "伴奏", "audio": "输出"}
STEM_LABELS_EN = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "other": "Other",
                  "instrumental": "Instrumental", "accompaniment": "Accompaniment",
                  "audio": "Output"}
MAX_SEP_STEMS = 4
# 分离族（htdemucs / mel-band-roformer）的 prepare() 硬校验 44.1kHz（模型配置
# sample_rate），不自己重采样；其它采样率的输入在 webui 侧先转到 44.1k。
SEP_SR = 44100


def do_sep(model, audio_path):
    """音源分离：响应里的 named_audio_outputs 每轨落盘成一个 wav，
    前 MAX_SEP_STEMS 轨直接放进播放器，全部轨放进文件下载列表。"""
    empty = [gr.update(value=None, visible=False) for _ in range(MAX_SEP_STEMS)]
    try:
        if not audio_path:
            raise gr.Error(_t("请上传要分离的音频", "Upload audio to separate."))
        audio_path = _ensure_wav(audio_path, target_sr=SEP_SR)
        ensure_model_loaded(model, SEP_TASKS)
        entry = catalog_by_id(model)
        dur = _audio_duration_seconds(audio_path)
        dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
        _ui_log(_t("音源分离开始：model={model}，音频时长 {dur_note}",
                   "source separation started: model={model}, audio duration {dur_note}",
                   model=model, dur_note=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, {"audio": audio_path},
                         timeout=1800, log_label="音源分离")
        stems = data.get("named_audio_outputs") or []
        if not stems and data.get("audio"):
            stems = [{"id": "audio", "audio": data["audio"]}]
        if not stems:
            raise gr.Error(_t("server 没有返回音轨数据。", "The server returned no tracks."))
        ts = int(time.time() * 1000)
        paths = []
        for stem in stems:
            sid = stem.get("id") or f"stem{len(paths)}"
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", sid)
            p = os.path.join(OUTPUT_DIR, f"audiocpp_sep_{ts}_{safe}.wav")
            with open(p, "wb") as f:
                f.write(base64.b64decode(stem["audio"]))
            paths.append((sid, p))
        updates = []
        for i in range(MAX_SEP_STEMS):
            if i < len(paths):
                sid, p = paths[i]
                language = get_language()
                labels = STEM_LABELS_EN if language == "en" else STEM_LABELS
                label = labels.get(sid)
                if label:
                    label = _t(label, label, language)
                updates.append(gr.update(
                    value=p, visible=True, label=f"{label}（{sid}）" if label else sid))
            else:
                updates.append(gr.update(value=None, visible=False))
        elapsed = time.time() - t_start
        _ui_log(_t("音源分离完成：{tracks} 轨，用时 {seconds:.1f}s",
                   "source separation done: {tracks} tracks, elapsed {seconds:.1f}s",
                   tracks=len(paths), seconds=elapsed))
        note = ("" if len(paths) <= MAX_SEP_STEMS else
                _t("，其余 {count} 轨见下载列表", "; {count} more in the download list",
                   count=len(paths) - MAX_SEP_STEMS))
        return (*updates, [p for _, p in paths], _t(
            "✅ 分离完成（{count} 轨）{note}，用时 {elapsed:.1f}s。",
            "✅ Separation complete ({count} tracks){note} in {elapsed:.1f}s.",
            count=len(paths), note=note, elapsed=elapsed))
    except gr.Error as e:
        return (*empty, None, _msg_from_error(e))
    except Exception as e:
        return (*empty, None, _t("❌ 分离失败：{error}",
                                 "❌ Separation failed: {error}", error=e))


# VAD/diar/align 输入统一转成 16 kHz 单声道后再发（见 _to_16k_mono_wav），
# 所以响应里的 start_sample/end_sample 一律按 16000 换算成秒。
SR_ANALYZE = 16000


def _fmt_ts(samples):
    return f"{samples / SR_ANALYZE:.2f}s"


def do_analyze(model, audio_path, transcript, language):
    """音频分析（vad/diar/align）：格式化 segments / speaker_turns / words 为
    可读文本，原始 JSON 落盘供下载。"""
    try:
        if not audio_path:
            raise gr.Error(_t("请上传或录制音频", "Upload or record audio."))
        entry = catalog_by_id(model)
        task = entry.get("task") if entry else ""
        req = {"audio": _to_16k_mono_wav(_ensure_wav(audio_path))}
        if task == "align":
            if not (transcript or "").strip():
                raise gr.Error(_t("强制对齐需要填写音频原文。",
                                  "Forced alignment requires the source transcript."))
            req["text"] = transcript.strip()
            if (language or "").strip():
                req["language"] = language.strip()
        dur = _audio_duration_seconds(req["audio"])
        # diar（Sortformer）是固定图容量，>20s 的音频按时长动态重载。
        ensure_model_loaded(model, ANALYZE_TASKS,
                            session_options=(_diar_session_options(dur)
                                             if task == "diar" else None))
        dur_note = f"{dur:.1f}s" if dur is not None else _t("未知", "unknown")
        _ui_log(_t("音频分析开始：model={model}（task={task}），音频 {dur_note}",
                   "audio analysis started: model={model} (task={task}), audio {dur_note}",
                   model=model, task=task, dur_note=dur_note))
        t_start = time.time()
        data = _run_task(entry, model, req, timeout=900, log_label="音频分析")

        lines = []
        if data.get("segments"):
            segs = data["segments"]
            lines.append(_t("共 {count} 个语音段：", "{count} speech segments:", count=len(segs)))
            speech = 0
            for i, s in enumerate(segs, 1):
                lines.append(_t("{index:3d}. {start} → {end}　置信度 {confidence:.2f}",
                                "{index:3d}. {start} → {end}  confidence {confidence:.2f}",
                                index=i, start=_fmt_ts(s["start_sample"]),
                                end=_fmt_ts(s["end_sample"]), confidence=s.get("confidence", 0)))
                speech += s["end_sample"] - s["start_sample"]
            lines.append(_t("语音总时长约 {seconds:.1f}s", "Total speech: {seconds:.1f}s",
                            seconds=speech / SR_ANALYZE))
        if data.get("speaker_turns"):
            turns = data["speaker_turns"]
            spk = sorted({t.get("speaker_id", "?") for t in turns})
            lines.append(_t("共 {turns} 个发言段、{speakers} 个说话人（{ids}）：",
                            "{turns} turns, {speakers} speakers ({ids}):",
                            turns=len(turns), speakers=len(spk), ids=", ".join(spk)))
            for i, t in enumerate(turns, 1):
                lines.append(_t("{index:3d}. {start} → {end}　{speaker}　置信度 {confidence:.2f}",
                                "{index:3d}. {start} → {end}  {speaker}  confidence {confidence:.2f}",
                                index=i, start=_fmt_ts(t["start_sample"]),
                                end=_fmt_ts(t["end_sample"]), speaker=t.get("speaker_id", "?"),
                                confidence=t.get("confidence", 0)))
        if data.get("words"):
            lines.append(_t("共 {count} 个词的时间戳：", "Timestamps for {count} words:",
                            count=len(data["words"])))
            for w in data["words"]:
                lines.append(f"{_fmt_ts(w['start_sample'])} → {_fmt_ts(w['end_sample'])}"
                             f"　{w.get('word', '')}")
        if data.get("text"):
            lines.append(_t("文本输出：{text}", "Text: {text}", text=data["text"]))
        if not lines:
            lines.append(_t("（模型没有返回可显示的分析结果）",
                            "(The model returned no displayable result.)"))

        json_path = os.path.join(OUTPUT_DIR, f"audiocpp_analyze_{int(time.time()*1000)}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        elapsed = time.time() - t_start
        _ui_log(_t("音频分析完成：用时 {seconds:.1f}s",
                   "audio analysis done: elapsed {seconds:.1f}s",
                   seconds=elapsed))
        return ("\n".join(lines), json_path,
                _t("✅ 分析完成（音频 {duration}），用时 {elapsed:.1f}s。",
                   "✅ Analysis complete ({duration}) in {elapsed:.1f}s.",
                   duration=dur_note, elapsed=elapsed))
    except gr.Error as e:
        return "", None, _msg_from_error(e)
    except Exception as e:
        return "", None, _t("❌ 分析失败：{error}", "❌ Analysis failed: {error}", error=e)


def _align_fields_visibility(model_id):
    """音频分析页：只有 align 任务的模型才显示『对齐文本/语言』输入。"""
    entry = catalog_by_id(model_id) if model_id else None
    show = bool(entry) and entry.get("task") == "align"
    return gr.update(visible=show), gr.update(visible=show)


def do_vdes(model, text, instruct, seed, max_tokens, adv_values, adv_options):
    """声音设计（vdes）：文字 + 音色描述走 /v1/audio/speech（instructions 字段
    映射到模型的 instruct 选项），响应是 WAV 音频。"""
    try:
        if not (text or "").strip():
            raise gr.Error(_t("请输入要合成的文字", "Enter text to synthesize."))
        if not (instruct or "").strip():
            raise gr.Error(_t("请填写音色描述。", "Enter a voice description."))
        ensure_model_loaded(model, VDES_TASKS)
        entry = catalog_by_id(model)
        prof = profile_for(entry) if entry else DEFAULT_PROFILE
        options = _merged_options(prof, adv_values, adv_options)

        seed, seed_note = _resolve_seed(seed)
        payload = {"model": model, "input": text,
                   "seed": seed, "max_tokens": int(max_tokens)}
        # 音色描述的落点按家族区分：qwen3_tts 走服务器的 instructions→instruct 映射；
        # irodori 等只读自家 options 键（profile 里的 vdes_option_key，如 caption）。
        vdes_key = prof.get("vdes_option_key")
        if vdes_key:
            options.setdefault(vdes_key, instruct)
        else:
            payload["instructions"] = instruct
        if options:
            payload["options"] = options
        _ui_log(_t("声音设计开始：model={model}，{chars} 字",
                   "voice design started: model={model}, {chars} chars",
                   model=model, chars=len(text)))
        t_start = time.time()
        try:
            r = requests.post(f"{SERVER}/v1/audio/speech", json=payload, timeout=900)
        except requests.RequestException as e:
            _ui_log(_t("声音设计失败：无法连接 server",
                       "voice design failed: cannot connect to server"))
            raise connection_error(e)
        if r.status_code != 200:
            _ui_log(_t("声音设计失败：server {code}",
                       "voice design failed: server {code}", code=r.status_code))
            raise server_error(entry, r.status_code, r.text)
        out = os.path.join(OUTPUT_DIR, f"audiocpp_vdes_{int(time.time()*1000)}.wav")
        with open(out, "wb") as f:
            f.write(r.content)
        elapsed = time.time() - t_start
        _ui_log(_t("声音设计完成：{out}，用时 {seconds:.1f}s",
                   "voice design done: {out}, elapsed {seconds:.1f}s",
                   out=out, seconds=elapsed))
        return out, _t("✅ 生成完成，用时 {seconds:.1f}s。{seed}",
                       "✅ Complete in {seconds:.1f}s. {seed}", seconds=elapsed, seed=seed_note)
    except gr.Error as e:
        return None, _msg_from_error(e)
    except Exception as e:
        return None, _t("❌ 生成失败：{error}", "❌ Generation failed: {error}", error=e)


def _make_load_handler(tasks):
    """『📥 加载模型』按钮的处理器工厂：按标签页各自的 task 集合校验并加载。"""
    def _load(model):
        try:
            s = ensure_model_loaded(model, tasks)
        except gr.Error as e:
            s = _msg_from_error(e)
        return s, server_status()
    return _load


# 标签页顺序（与 refresh() 的输出、_refresh_outputs 列表一一对应）：
# TTS、ASR、音乐生成、声音转换、音源分离、音频分析、声音设计。
TAB_SPECS = [TTS_TASKS, ASR_TASKS, GEN_TASKS, VC_TASKS,
             SEP_TASKS, ANALYZE_TASKS, VDES_TASKS]


def refresh():
    """重读 catalog / 参数配置，刷新每个标签页的模型下拉和提示。
    返回顺序：各页下拉更新（按 TAB_SPECS 顺序）、状态行、各页提示、
    ASR 流式选项。"""
    global CATALOG, MODEL_PARAMS, REQUIRED_FILES
    CATALOG = _load_catalog()
    MODEL_PARAMS = _load_model_params()
    REQUIRED_FILES = _load_required_files()
    dropdowns, hints = [], []
    for tasks in TAB_SPECS:
        choices = choices_for_tasks(tasks)
        value = choices[0][1] if choices else None
        dropdowns.append(gr.update(choices=choices, value=value))
        hints.append(model_hint_for(value))
    asr_model_id = dropdowns[1]["value"] if len(dropdowns) > 1 else None
    return (*dropdowns, server_status(), *hints, asr_stream_update(asr_model_id))


atexit.register(_stop_server)


_I18N_COMPONENTS = []


def _localized(component, **props):
    """Register translatable Gradio properties as ``prop=(zh, en)``."""
    for prop, pair in props.items():
        setattr(component, prop, _localized_prop_value(prop, pair, INITIAL_LANGUAGE))
    _I18N_COMPONENTS.append((component, props))
    return component


def _localized_prop_value(prop, pair, language):
    """Localize one component property without changing choice protocol values."""
    if prop == "choices" and language == "zh-Hant":
        localized = []
        for choice in pair[0]:
            if isinstance(choice, (tuple, list)) and len(choice) == 2:
                label, value = choice
            else:
                label = value = choice
            localized.append((_t(label, label, language), value))
        return localized
    return _t(pair[0], pair[1], language)


def _localized_updates(language):
    return [
        gr.update(**{
            prop: _localized_prop_value(prop, pair, language)
            for prop, pair in props.items()
        })
        for _component, props in _I18N_COMPONENTS
    ]


CUSTOM_CSS = """

.app-header {
  align-items: center !important;
  justify-content: space-between !important;
  gap: 12px !important;
  flex-wrap: nowrap !important;
  width: 100% !important;
}
.app-title { flex: 1 1 auto !important; min-width: 0 !important; }
.app-title h1 { margin: 0 !important; }
.language-switch {
  flex: 0 0 118px !important;
  width: 118px !important;
  min-width: 118px !important;
  max-width: 118px !important;
  margin: 0 0 0 auto !important;
  padding: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.audio-default { border: none !important; box-shadow: none !important; }

.mm-btn-row { gap: 10px !important; }
.mm-btn-row button {
  border-radius: var(--button-large-radius, var(--radius-lg)) !important;
  white-space: nowrap !important;
}
.mm-btn-row button span { white-space: nowrap !important; }
.gguf-btn-row { align-items: center !important; }
.gguf-btn-row > * { align-self: center !important; }

/* 长音频的波形出现横向滚动条时，WaveSurfer 的滚动层（58px 内容 + 滚动条）会
   溢出 Gradio 固定 58px 的 .waveform-container / #waveform，盖住下方的
   0:00/总时长标签。放开这两层的高度让标签随内容下移；短音频（无滚动条）时
   min-height 保证布局与原来一致。 */
.waveform-container, #waveform { height: auto !important; min-height: 58px; }

.hint-small { opacity: 0.7; font-size: 0.85em; margin-top: 2px; }

/* 内置参考音色行：Row 本身透明，会在按钮一列露出 gr.Group 的灰色面板底，
   和 secondary 按钮的灰底连成一片。把整行铺成和下拉 block 相同的白色卡片，
   按钮（灰底）落在卡片内自然形成对比；flex-end + margin 让按钮和下拉输入框
   本体底部对齐（下拉 block 自带 --block-padding 内边距）。 */
.voice-refresh-row {
  align-items: flex-end !important;
  background: var(--block-background-fill, #fff);
  border-radius: var(--block-radius, 8px);
}
.voice-refresh-row button {
  height: var(--size-10, 40px);
  /* 字面量，不能用 var(--block-padding)：该主题变量是双值 "10px 12px"，
     代入 margin 简写会让整条声明非法被丢弃。11px = block 下内边距 10px + 边框，
     让按钮和下拉输入框本体上下沿精确对齐。 */
  margin: 0 12px 11px 0;
  border-radius: var(--radius-lg) !important;
}
"""

# Gradio 的 Audio 播放器（WaveSurfer）换音频时复用同一实例，旧的播放进度会
# 原样带到新音频上（生成完成后进度条停在上一次的位置）。value=None 清空也挡
# 不住。修法：每个播放器宿主挂一次 loadedmetadata（每次换源触发、用户 seek
# 不触发），换源后开一个 ~3s 守护窗，暂停状态下把 shadow DOM 里 wavesurfer 的
# <audio>.currentTime 清回 0；窗口内用户 pointerdown（捕获阶段，躲开内部
# stopPropagation）立即取消守护，不干扰手动 seek/播放。
_RESET_AUDIO_SEEK_JS = """
() => {
  if (window.__audiocppSeekReset) return;
  window.__audiocppSeekReset = true;
  const seen = new WeakSet();
  const arm = (host) => {
    if (seen.has(host)) return;
    seen.add(host);
    const attach = () => {
      const a = host.shadowRoot && host.shadowRoot.querySelector('audio');
      if (!a) { setTimeout(attach, 250); return; }
      const guard = () => {
        let tries = 0, cancelled = false;
        const cancel = () => { cancelled = true; };
        host.addEventListener('pointerdown', cancel, { once: true, capture: true });
        const tick = () => {
          if (cancelled) return;
          if (a.paused && a.currentTime > 0.05) { try { a.currentTime = 0; } catch (e) {} }
          if (++tries < 12) setTimeout(tick, 250);
          else host.removeEventListener('pointerdown', cancel, { capture: true });
        };
        setTimeout(tick, 100);
      };
      a.addEventListener('loadedmetadata', guard);
      if (a.readyState >= 1) guard();
    };
    attach();
  };
  const scan = (root) => root.querySelectorAll('#waveform > div').forEach(arm);
  new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType !== 1) continue;
        if (n.matches && n.matches('#waveform > div')) arm(n);
        else if (n.querySelectorAll) scan(n);
      }
    }
  }).observe(document.body, { childList: true, subtree: true });
  scan(document);
}
"""


VDES_TEXT_DEFAULT_ZH = "你好，这是 audio.cpp 用文字描述设计出来的声音。"
VDES_TEXT_DEFAULT_ZH_HANT = _t(VDES_TEXT_DEFAULT_ZH, VDES_TEXT_DEFAULT_ZH, "zh-Hant")
VDES_TEXT_DEFAULT_EN = "Hello, this audio.cpp voice was created from a text description."


def _vdes_text_language_update(language, current):
    label = _t("要合成的文字", "Text to synthesize", language)
    if current in (VDES_TEXT_DEFAULT_ZH, VDES_TEXT_DEFAULT_ZH_HANT, VDES_TEXT_DEFAULT_EN):
        value = _t(VDES_TEXT_DEFAULT_ZH, VDES_TEXT_DEFAULT_EN, language)
        return gr.update(label=label, value=value)
    return gr.update(label=label)


with gr.Blocks(title="audio.cpp WebUI") as demo:
    with gr.Row(elem_classes="app-header"):
        _localized(
            gr.Markdown("# 🎙️ audio.cpp WebUI", elem_classes="app-title",
                        container=False, scale=0, min_width=250),
            value=("# 🎙️ audio.cpp WebUI", "# 🎙️ audio.cpp WebUI"))
        ui_language = gr.Dropdown(
            choices=LANGUAGE_CHOICES, value=INITIAL_LANGUAGE,
            interactive=True, filterable=False, allow_custom_value=False,
            show_label=False, container=False,
            elem_classes="language-switch", scale=0, min_width=118)
    status = gr.Markdown(server_status())
    with _localized(
            gr.Accordion("🔌 API 调用说明", open=False),
            label=("🔌 API 调用说明", "🔌 API usage")):
        api_usage = gr.Markdown(_api_usage_md())
    with _localized(
            gr.Accordion("🔐 下载设置（可选，不保存）", open=False),
            label=("🔐 下载设置（可选，不保存）", "🔐 Download settings (optional)")):
        with gr.Row():
            hf_token = _localized(gr.Textbox(
                label="HF token (下载受限模型时用)", type="password",
                placeholder="hf_xxx —— 或先运行 huggingface-cli login"),
                label=("HF token（下载受限模型时使用）", "HF token (for gated models)"),
                placeholder=("hf_xxx —— 或先运行 huggingface-cli login",
                             "hf_xxx — or run huggingface-cli login"))
            proxy = _localized(gr.Textbox(
                label="代理 (仅下载模型时使用)",
                placeholder="http://127.0.0.1:7890"),
                label=("代理（仅下载时使用）", "Proxy (downloads only)"),
                placeholder=("http://127.0.0.1:7890", "http://127.0.0.1:7890"))

    # ---- 每个标签页共用的“模型管理”卡片、接线与高级参数渲染 ----
    def _model_manager_block(task_label, tasks):
        """标准“模型管理”卡片：模型下拉、加载/下载与 GGUF 操作 + 状态区。
        返回组件 dict；接线见 _wire_model_manager（刷新按钮统一接在文件末尾）。"""
        choices = choices_for_tasks(tasks)
        with gr.Group():
            _localized(gr.Markdown("#### 🧩 模型管理"),
                       value=("#### 🧩 模型管理", "#### 🧩 Model"))
            model = _localized(gr.Dropdown(
                label=f"模型列表 (task={task_label})", choices=choices,
                value=(choices[0][1] if choices else None)),
                label=(f"模型列表（task={task_label}）", f"Model (task={task_label})"),
                choices=(choices_for_tasks(tasks, "zh"), choices_for_tasks(tasks, "en")))
            with gr.Row(elem_classes="mm-btn-row"):
                load_btn = _localized(
                    gr.Button("📥 加载模型", variant="primary", size="lg", min_width=100),
                    value=("📥 加载模型", "📥 Load"))
                refresh_btn = _localized(
                    gr.Button("🔄 刷新列表", variant="primary", size="lg", min_width=100),
                    value=("🔄 刷新列表", "🔄 Refresh"))
                dl_btn = _localized(
                    gr.Button("⬇️ 下载模型", variant="primary", size="lg", min_width=100),
                    value=("⬇️ 下载模型", "⬇️\u00a0Download"))
                dl_stat_btn = _localized(
                    gr.Button("📊 下载进度", variant="primary", size="lg", min_width=100),
                    value=("📊 下载进度", "📊 Progress"))
                unload_btn = _localized(
                    gr.Button("🧹 释放显存", variant="secondary", size="lg", min_width=100),
                    value=("🧹 释放显存", "🧹 Unload"))
            load_status = gr.Markdown("")
            dl_status = gr.Markdown("")
            with _localized(gr.Accordion("🧊 GGUF 工具（转换 / 检查 / 删除）", open=False),
                            label=("🧊 GGUF 工具（转换 / 检查 / 删除）",
                                   "🧊 GGUF tools (convert / inspect / delete)")):
                with gr.Row(elem_classes="mm-btn-row gguf-btn-row"):
                    gguf_type = _localized(gr.Dropdown(
                        label="类型", show_label=False, choices=list(GGUF_TYPES), value="q8_0", scale=1, min_width=80),
                        label=("类型", "Type"))
                    gguf_convert_btn = _localized(
                        gr.Button("🧊 转换", variant="secondary", scale=1, min_width=90),
                        value=("🧊 转换", "🧊 Convert"))
                    gguf_inspect_btn = _localized(
                        gr.Button("🔎 检查", variant="secondary", scale=1, min_width=90),
                        value=("🔎 检查", "🔎 Inspect"))
                    gguf_delete_btn = _localized(
                        gr.Button("🗑️ 删除", variant="stop", scale=1, min_width=90),
                        value=("🗑️ 删除", "🗑️ Delete"))
                gguf_message = gr.Markdown(gguf_status(model.value))
            timer = gr.Timer(3, active=False)
        return {"model": model, "load_btn": load_btn, "refresh_btn": refresh_btn,
                "dl_btn": dl_btn, "dl_stat_btn": dl_stat_btn, "unload_btn": unload_btn,
                "load_status": load_status, "dl_status": dl_status, "timer": timer,
                "gguf_type": gguf_type, "gguf_convert_btn": gguf_convert_btn,
                "gguf_inspect_btn": gguf_inspect_btn, "gguf_delete_btn": gguf_delete_btn,
                "gguf_status": gguf_message}

    def _wire_model_manager(mm, tasks, hint):
        mm["load_btn"].click(_make_load_handler(tasks), mm["model"],
                             [mm["load_status"], status])
        mm["dl_btn"].click(download_start, [mm["model"], hf_token, proxy],
                           [mm["dl_status"], mm["timer"]])
        mm["timer"].tick(download_status_tick, mm["model"],
                         [mm["dl_status"], mm["timer"]])
        mm["dl_stat_btn"].click(download_status, mm["model"], mm["dl_status"])
        mm["unload_btn"].click(unload_model, None, [mm["load_status"], status])
        mm["gguf_convert_btn"].click(convert_model_to_gguf,
                                      [mm["model"], mm["gguf_type"]], mm["gguf_status"])
        mm["gguf_inspect_btn"].click(inspect_gguf, mm["model"], mm["gguf_status"])
        mm["gguf_delete_btn"].click(delete_gguf, mm["model"], [mm["gguf_status"], status])
        mm["model"].change(model_hint_for, mm["model"], hint)
        mm["model"].change(gguf_status, mm["model"], mm["gguf_status"])

    def _render_param_controls(model_comp, state_comp, skip=(), prefill_comp=None):
        """“高级参数”折叠区内容：按所选模型的 family 动态生成控件（gr.render），
        控件值写进共享 state（只有用户改过的项会随请求发送）。
        传入 prefill_comp（gr.State dict）时它也是渲染输入：程序化回填
        （如『🔍 分析源音频』）写 prefill 触发重渲染，把值显示在控件上。"""
        inputs = ([model_comp, ui_language] if prefill_comp is None else
                  [model_comp, ui_language, prefill_comp])

        @gr.render(inputs=inputs)
        def _render(model_id, language, prefill=None):
            specs = [p for p in params_for(model_id) if p.get("name") not in skip]
            if not specs:
                gr.Markdown(_t("*该模型无可调高级参数。*",
                               "*No advanced options for this model.*", language))
                return
            prefill = prefill or {}
            for p in specs:
                if p["name"] in prefill:
                    p = {**p, "default": prefill[p["name"]]}
                comp = _make_param_component(p, language)
                comp.change(_adv_updater(p["name"]), [state_comp, comp], state_comp)

    # ---------------- TTS / 声音克隆 ----------------
    with _localized(gr.Tab("🗣️ TTS / 声音克隆"),
                    label=("🗣️ TTS / 声音克隆", "🗣️ TTS / Voice cloning")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 参考音频（声音克隆）
            with gr.Column(scale=1):
                tts_mm = _model_manager_block("tts", TTS_TASKS)
                tts_model = tts_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎧 参考音频（声音克隆）"),
                               value=("#### 🎧 参考音频（声音克隆）",
                                      "#### 🎧 Voice reference"))
                    with gr.Row(elem_classes="voice-refresh-row"):
                        tts_builtin = _localized(gr.Dropdown(
                            label="内置参考音色",
                            choices=["(none)"] + builtin_voices(),
                            value="(none)", scale=8),
                            label=("内置参考音色", "Built-in voice"))
                        tts_voice_refresh = gr.Button(
                            "🔄", scale=1, min_width=48)
                    tts_upload = _localized(gr.Audio(
                        label="上传/录制参考音色（可选）", type="filepath",
                        elem_classes="audio-default"),
                        label=("上传/录制参考音色（可选）", "Upload/record voice (optional)"))
                    tts_ref_text = _localized(gr.Textbox(
                        label="参考文本", lines=2,
                        placeholder="参考音频里说的原话（建议填写）"),
                        label=("参考文本", "Reference transcript"),
                        placeholder=("参考音频里说的原话（建议填写）",
                                     "Words spoken in the reference audio"))
                    tts_voice_name = _localized(gr.Textbox(
                        label="内置参考音色命名",
                        placeholder="例如：my_voice（将保存为 my_voice.wav）"),
                        label=("内置参考音色命名", "Built-in voice name"),
                        placeholder=("例如：my_voice（将保存为 my_voice.wav）",
                                     "Example: my_voice (saved as my_voice.wav)"))
                    with gr.Row():
                        tts_voice_save = _localized(
                            gr.Button("💾 保存参考音频"),
                            value=("💾 保存参考音频", "💾 Save reference audio"))
                        tts_voice_delete = _localized(
                            gr.Button("🗑️ 删除内置音频", variant="stop"),
                            value=("🗑️ 删除内置音频",
                                   "🗑️ Delete built-in audio"))

                # 切换模型后的提示，显示在参考音频整块下面
                tts_hint = gr.Markdown(model_hint_for(tts_model.value))

            # 右列：合成设置 + 合成内容 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### ⚙️ 合成设置"),
                               value=("#### ⚙️ 合成设置", "#### ⚙️ Synthesis settings"))
                    with gr.Row():
                        tts_seed = _localized(
                            gr.Number(label="seed（-1=随机）", value=1234, precision=0),
                            label=("seed（-1=随机）", "seed (-1=random)"))
                        tts_maxtok = _localized(
                            gr.Number(label="max_tokens", value=1200, precision=0),
                            label=("max_tokens", "max_tokens"))
                    tts_adv_state = gr.State({})
                    with _localized(
                            gr.Accordion("高级参数（只发送改动过的项）", open=False),
                            label=("高级参数（只发送改动过的项）", "Advanced options")):
                        _render_param_controls(tts_model, tts_adv_state)

                    with _localized(
                            gr.Accordion("其它参数（JSON，覆盖控件）", open=False),
                            label=("其它参数（JSON，覆盖控件）", "Other options (JSON override)")):
                        tts_adv = gr.Textbox(
                            label="",
                            placeholder='{"num_inference_steps": 10, "voice_samples": "D:/a.wav,D:/b.wav"}',
                            lines=3)

                with gr.Group():
                    _localized(gr.Markdown("#### ✍️ 合成内容"),
                               value=("#### ✍️ 合成内容", "#### ✍️ Text"))
                    tts_text = _localized(gr.Textbox(
                        label="要合成的文字", lines=5,
                        value="Hello, this is audio dot cpp speaking from a web page."),
                        label=("要合成的文字", "Text to synthesize"))
                    tts_lang = _localized(gr.Dropdown(
                        label="语言（Auto=模型默认）",
                        choices=[("Auto", "")] + [(lang, lang) for lang in LANGS if lang],
                        value=""),
                        label=("语言（Auto=模型默认）", "Language (Auto=model default)"))
                    tts_gen_mode = _localized(gr.Radio(
                        label="生成模式（流式=边生成边播放）",
                        choices=["离线", "流式"], value="离线",
                        visible=supports_streaming(tts_model.value)),
                        label=("生成模式（流式=边生成边播放）", "Generation mode"),
                        choices=([("离线", "离线"), ("流式", "流式")],
                                 [("Offline", "离线"), ("Streaming", "流式")]))

                tts_btn = _localized(
                    gr.Button("🎵 生成语音", variant="primary", size="lg"),
                    value=("🎵 生成语音", "🎵 Generate speech"))
                with gr.Group():
                    _localized(gr.Markdown("#### 🔊 输出音频"),
                               value=("#### 🔊 输出音频", "#### 🔊 Output"))
                    tts_out_stream = _localized(gr.Audio(
                        label="⚡ 流式播放（边生成边播）", streaming=True,
                        autoplay=True, visible=False,
                        elem_classes="audio-default"),
                        label=("⚡ 流式播放（边生成边播）", "⚡ Streaming audio"))
                    tts_out = _localized(
                        gr.Audio(label="输出音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("输出音频", "Output audio"))
                    tts_msg = gr.Markdown("")

        _wire_model_manager(tts_mm, TTS_TASKS, tts_hint)
        tts_model.change(lambda: {}, None, tts_adv_state)  # reset knobs on model switch
        # 模型切换：只有支持流式的家族（voxcpm2）显示生成模式选择，并重置回离线。
        tts_model.change(
            lambda m: gr.update(visible=supports_streaming(m), value="离线"),
            tts_model, tts_gen_mode)
        # 流式播放组件只在选了流式模式时出现（离线路径的输出组件保持原样）。
        tts_gen_mode.change(lambda v: gr.update(visible=(v == "流式")),
                            tts_gen_mode, tts_out_stream)
        tts_builtin.change(on_tts_builtin_voice_change, tts_builtin,
                           [tts_upload, tts_ref_text, tts_voice_name])
        tts_voice_refresh.click(refresh_builtin_voices, tts_builtin,
                                [tts_builtin, tts_upload, tts_ref_text,
                                 tts_voice_name])
        tts_voice_save.click(
            save_builtin_voice,
            [tts_upload, tts_voice_name, tts_ref_text],
            [tts_builtin, tts_msg])
        tts_voice_delete.click(
            delete_builtin_voice,
            tts_builtin,
            [tts_builtin, tts_upload, tts_ref_text, tts_voice_name, tts_msg])
        # Clear the previous run's audio + status message the moment 生成 is
        # clicked, so a prior message doesn't linger next to the new run's
        # progress indicator (outputs otherwise update only when do_tts returns).
        # 流式播放器同样要清：streaming 组件在新一轮事件开始时不会自动复位，
        # 上一轮的音频会留在播放器里直到新首包到达（模型重载时能有几十秒）。
        tts_btn.click(lambda: (None, None, ""), None,
                      [tts_out_stream, tts_out, tts_msg]).then(
            do_tts_or_stream,
            [tts_model, tts_gen_mode, tts_text, tts_lang, tts_upload, tts_builtin,
             tts_ref_text, tts_seed, tts_maxtok, tts_adv_state, tts_adv],
            [tts_out_stream, tts_out, tts_msg])

    # ---------------- ASR / 音频转写 ----------------
    with _localized(gr.Tab("📝 ASR / 音频转写"),
                    label=("📝 ASR / 音频转写", "📝 ASR / Transcription")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 音频输入
            with gr.Column(scale=1):
                asr_mm = _model_manager_block("asr", ASR_TASKS)
                asr_model = asr_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎤 音频输入"),
                               value=("#### 🎤 音频输入", "#### 🎤 Audio input"))
                    asr_audio = _localized(
                        gr.Audio(label="上传/录制音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("上传/录制音频", "Upload/record audio"))
                with _localized(gr.Accordion("🎛 转写选项（可选）", open=False),
                                label=("🎛 转写选项（可选）", "🎛 Options (optional)")):
                    asr_language = _localized(gr.Dropdown(
                        label="强制语种（留空=自动检测）",
                        choices=[("自动检测", "")] +
                                [(f"{zh} {en}", en) for zh, en in QWEN3_ASR_LANGUAGES],
                        value=""),
                        label=("强制语种（留空=自动检测）", "Language (blank=auto)"),
                        choices=([("自动检测", "")] +
                                 [(f"{zh} {en}", en) for zh, en in QWEN3_ASR_LANGUAGES],
                                 [("Auto", "")] + [(en, en) for _zh, en in QWEN3_ASR_LANGUAGES]))
                    asr_context = _localized(gr.Textbox(
                        label="上下文提示", lines=2,
                        placeholder="人名/术语等，帮助识别专有名词"),
                        label=("上下文提示", "Context"),
                        placeholder=("人名/术语等，帮助识别专有名词",
                                     "Names or terms that may improve recognition"))
                    asr_dialogue = _localized(gr.Checkbox(
                        label="🗣 对话模式（≤120s，需 Sortformer）"),
                        label=("🗣 对话模式（≤120s，需 Sortformer）",
                               "🗣 Dialogue mode (≤120s; needs Sortformer)"))
                asr_stream = _localized(gr.Checkbox(
                    label="⚡ 流式转写（边转边出字，长音频不用干等；与对话模式互斥）",
                    value=False, visible=supports_streaming(asr_model.value)),
                    label=("⚡ 流式转写（与对话模式互斥）", "⚡ Streaming transcription"))
                asr_hint = gr.Markdown(model_hint_for(asr_model.value))
                asr_btn = _localized(
                    gr.Button("📝 开始转写", variant="primary", size="lg"),
                    value=("📝 开始转写", "📝 Transcribe"))

            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### 📄 识别结果"),
                               value=("#### 📄 识别结果", "#### 📄 Transcript"))
                    asr_out = _localized(gr.Textbox(
                        label="转写文本", lines=10,
                        placeholder="转写结果将显示在这里……"),
                        label=("转写文本", "Transcript"),
                        placeholder=("转写结果将显示在这里……", "Transcript appears here…"))
                    asr_msg = gr.Markdown("")

        _wire_model_manager(asr_mm, ASR_TASKS, asr_hint)
        # 用户选模型时立即同步流式能力，不走默认队列；用 change 会同时接收
        # 后端刷新下拉框的更新，和未安装模型的其它状态回调交错后可能留下旧的
        # visible=False。程序化刷新由 refresh() 显式更新该控件。
        asr_model.input(asr_stream_update, asr_model, asr_stream,
                        queue=False, show_progress="hidden")
        asr_btn.click(lambda: ("", ""), None, [asr_out, asr_msg]).then(
            do_asr, [asr_model, asr_audio, asr_language, asr_context, asr_dialogue,
                     asr_stream],
            [asr_out, asr_msg])

    # ---------------- 音乐 / 音效生成 ----------------
    with _localized(gr.Tab("🎵 音乐生成"),
                    label=("🎵 音乐生成", "🎵 Music generation")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 源音频（编辑类用法）
            with gr.Column(scale=1):
                gen_mm = _model_manager_block("gen", GEN_TASKS)
                gen_model = gen_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎧 源音频（可选，仅编辑类用法）"),
                               value=("#### 🎧 源音频（可选，仅编辑类用法）",
                                      "#### 🎧 Source audio (optional)"))
                    gen_audio = _localized(gr.Audio(
                        label="上传源音频（编辑类 route 用）",
                        type="filepath", elem_classes="audio-default"),
                        label=("上传源音频（编辑类 route 用）", "Upload source audio"))
                    gen_ana_btn = _localized(gr.Button("🔍 分析源音频"),
                                             value=("🔍 分析源音频", "🔍 Analyze source"))
                    gen_ana_msg = gr.Markdown("")

                gen_hint = gr.Markdown(model_hint_for(gen_model.value))

            # 右列：生成设置 + 提示词/歌词 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### ⚙️ 生成设置"),
                               value=("#### ⚙️ 生成设置", "#### ⚙️ Generation settings"))
                    with gr.Row():
                        gen_duration = _localized(
                            gr.Number(label="时长(秒)，-1=自动", value=30, precision=1),
                            label=("时长（秒，-1=自动）", "Duration (s; -1=auto)"))
                        gen_seed = _localized(
                            gr.Number(label="seed（-1=随机）", value=1234, precision=0),
                            label=("seed（-1=随机）", "seed (-1=random)"))
                    gen_adv_state = gr.State({})
                    gen_prefill = gr.State({})   # 『🔍 分析源音频』回填 -> 触发控件重渲染
                    with _localized(
                            gr.Accordion("高级参数（只发送改动过的项）", open=False),
                            label=("高级参数（只发送改动过的项）", "Advanced options")):
                        _render_param_controls(gen_model, gen_adv_state,
                                               prefill_comp=gen_prefill)

                    with _localized(
                            gr.Accordion("其它参数（JSON，覆盖控件）", open=False),
                            label=("其它参数（JSON，覆盖控件）", "Other options (JSON override)")):
                        gen_adv = gr.Textbox(
                            label="",
                            placeholder='{"tags": "pop,bright,drums", "num_inference_steps": 8}',
                            lines=3)

                with gr.Group():
                    _localized(gr.Markdown("#### ✍️ 提示词与歌词"),
                               value=("#### ✍️ 提示词与歌词", "#### ✍️ Prompt and lyrics"))
                    gen_text = _localized(gr.Textbox(
                        label="提示词（英文效果最好）", lines=3,
                        value="uplifting pop with bright synths and driving drums"),
                        label=("提示词（英文效果最好）", "Prompt (English works best)"))
                    gen_lyrics = _localized(gr.Textbox(label="歌词（可选）", lines=4),
                                            label=("歌词（可选）", "Lyrics (optional)"))

                gen_btn = _localized(
                    gr.Button("🎵 生成音乐", variant="primary", size="lg"),
                    value=("🎵 生成音乐", "🎵 Generate music"))
                with gr.Group():
                    _localized(gr.Markdown("#### 🔊 输出音频"),
                               value=("#### 🔊 输出音频", "#### 🔊 Output"))
                    gen_out = _localized(
                        gr.Audio(label="输出音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("输出音频", "Output audio"))
                    gen_msg = gr.Markdown("")

        _wire_model_manager(gen_mm, GEN_TASKS, gen_hint)
        gen_model.change(lambda: ({}, {}, ""), None,
                         [gen_adv_state, gen_prefill, gen_ana_msg])  # 切换模型清空旋钮+分析信息
        gen_ana_btn.click(lambda: _t("⏳ 分析中……（需先加载模型）",
                                     "⏳ Analyzing… load the model first."),
                          None, gen_ana_msg).then(
            do_music_analyze, [gen_model, gen_audio, gen_seed, gen_adv_state],
            [gen_adv_state, gen_prefill, gen_ana_msg])
        gen_btn.click(lambda: (None, ""), None, [gen_out, gen_msg]).then(
            do_music_gen,
            [gen_model, gen_text, gen_lyrics, gen_audio, gen_duration, gen_seed,
             gen_adv_state, gen_adv],
            [gen_out, gen_msg])

    # ---------------- 声音转换 (vc / svc / s2s) ----------------
    with _localized(gr.Tab("🎭 声音转换"),
                    label=("🎭 声音转换", "🎭 Voice conversion")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 源音频 + 目标音色
            with gr.Column(scale=1):
                vc_mm = _model_manager_block("vc/svc/s2s", VC_TASKS)
                vc_model = vc_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎙️ 源音频（要转换的内容）"),
                               value=("#### 🎙️ 源音频（要转换的内容）",
                                      "#### 🎙️ Source audio"))
                    vc_source = _localized(
                        gr.Audio(label="上传/录制源音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("上传/录制源音频", "Upload/record source"))
                with gr.Group():
                    _localized(gr.Markdown("#### 🎧 目标音色（转换成谁的声音）"),
                               value=("#### 🎧 目标音色（转换成谁的声音）",
                                      "#### 🎧 Target voice"))
                    with gr.Row(elem_classes="voice-refresh-row"):
                        vc_builtin = _localized(gr.Dropdown(
                            label="内置参考音色",
                            choices=["(none)"] + builtin_voices(),
                            value="(none)", scale=8),
                            label=("内置参考音色", "Built-in voice"))
                        vc_voice_refresh = gr.Button(
                            "🔄", scale=1, min_width=48)
                    vc_target = _localized(
                        gr.Audio(label="上传/录制目标音色", type="filepath",
                                 elem_classes="audio-default"),
                        label=("上传/录制目标音色", "Upload/record target voice"))

                vc_hint = gr.Markdown(model_hint_for(vc_model.value))

            # 右列：转换设置 + 转换 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### ⚙️ 转换设置"),
                               value=("#### ⚙️ 转换设置", "#### ⚙️ Conversion settings"))
                    vc_seed = _localized(
                        gr.Number(label="seed（-1=随机）", value=1234, precision=0),
                        label=("seed（-1=随机）", "seed (-1=random)"))
                    vc_adv_state = gr.State({})
                    with _localized(
                            gr.Accordion("高级参数（只发送改动过的项）", open=False),
                            label=("高级参数（只发送改动过的项）", "Advanced options")):
                        _render_param_controls(vc_model, vc_adv_state)
                    with _localized(
                            gr.Accordion("其它参数（JSON，覆盖控件）", open=False),
                            label=("其它参数（JSON，覆盖控件）", "Other options (JSON override)")):
                        vc_adv = gr.Textbox(
                            label="",
                            placeholder='{"route": "style_converted_vc", "style_ref": "D:/style.wav", "target_text": "……"}',
                            lines=3)

                vc_btn = _localized(
                    gr.Button("🎭 开始转换", variant="primary", size="lg"),
                    value=("🎭 开始转换", "🎭 Convert"))
                with gr.Group():
                    _localized(gr.Markdown("#### 🔊 输出音频"),
                               value=("#### 🔊 输出音频", "#### 🔊 Output"))
                    vc_out = _localized(
                        gr.Audio(label="输出音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("输出音频", "Output audio"))
                    vc_msg = gr.Markdown("")

        _wire_model_manager(vc_mm, VC_TASKS, vc_hint)
        vc_model.change(lambda: {}, None, vc_adv_state)  # reset knobs on model switch
        vc_builtin.change(lambda n: on_builtin_voice_change(n)[0], vc_builtin, vc_target)
        vc_voice_refresh.click(lambda n: refresh_builtin_voices(n)[:2],
                               vc_builtin, [vc_builtin, vc_target])
        vc_btn.click(lambda: (None, ""), None, [vc_out, vc_msg]).then(
            do_vc,
            [vc_model, vc_source, vc_target, vc_builtin, vc_seed,
             vc_adv_state, vc_adv],
            [vc_out, vc_msg])

    # ---------------- 音源分离 (sep) ----------------
    with _localized(gr.Tab("🎚️ 音源分离"),
                    label=("🎚️ 音源分离", "🎚️ Source separation")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 输入音频
            with gr.Column(scale=1):
                sep_mm = _model_manager_block("sep", SEP_TASKS)
                sep_model = sep_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎵 输入音频"),
                               value=("#### 🎵 输入音频", "#### 🎵 Audio input"))
                    sep_audio = _localized(
                        gr.Audio(label="上传要分离的歌曲/音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("上传要分离的歌曲/音频", "Upload audio to separate"))
                sep_hint = gr.Markdown(model_hint_for(sep_model.value))
                sep_btn = _localized(
                    gr.Button("🎚️ 开始分离", variant="primary", size="lg"),
                    value=("🎚️ 开始分离", "🎚️ Separate"))

            # 右列：分离结果（各分轨播放器 + 文件下载）
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### 🔊 分离结果"),
                               value=("#### 🔊 分离结果", "#### 🔊 Separated tracks"))
                    sep_stems = [
                        _localized(
                            gr.Audio(label=f"音轨 {i + 1}", type="filepath",
                                     visible=False, elem_classes="audio-default"),
                            label=(f"音轨 {i + 1}", f"Track {i + 1}"))
                        for i in range(MAX_SEP_STEMS)
                    ]
                    sep_files = _localized(
                        gr.File(label="全部音轨文件", file_count="multiple", interactive=False),
                        label=("全部音轨文件", "All track files"))
                    sep_msg = gr.Markdown("")

        _wire_model_manager(sep_mm, SEP_TASKS, sep_hint)
        sep_btn.click(
            lambda: (*(gr.update(value=None, visible=False)
                       for _ in range(MAX_SEP_STEMS)), None, ""),
            None, [*sep_stems, sep_files, sep_msg]).then(
            do_sep, [sep_model, sep_audio], [*sep_stems, sep_files, sep_msg])

    # ---------------- 音频分析 (vad / diar / align) ----------------
    with _localized(gr.Tab("🔎 音频分析"),
                    label=("🔎 音频分析", "🔎 Audio analysis")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 音频输入（align 模型多出对齐文本/语言）
            with gr.Column(scale=1):
                ana_mm = _model_manager_block("vad/diar/align", ANALYZE_TASKS)
                ana_model = ana_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### 🎤 音频输入"),
                               value=("#### 🎤 音频输入", "#### 🎤 Audio input"))
                    ana_audio = _localized(
                        gr.Audio(label="上传/录制音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("上传/录制音频", "Upload/record audio"))
                    _localized(
                        gr.Markdown("*输入自动转 16 kHz 单声道。*", elem_classes="hint-small"),
                        value=("*输入自动转 16 kHz 单声道。*",
                               "*Input is converted to 16 kHz mono.*"))
                    _ana_is_align = bool(
                        ana_model.value and
                        (catalog_by_id(ana_model.value) or {}).get("task") == "align")
                    ana_text = _localized(gr.Textbox(
                        label="对齐文本（align 必填：音频原文）", lines=3,
                        visible=_ana_is_align),
                        label=("对齐文本（align 必填：音频原文）",
                               "Transcript (required for align)"))
                    ana_lang = _localized(gr.Textbox(
                        label="语言（可选，如 English / Chinese）",
                        visible=_ana_is_align),
                        label=("语言（可选，如 English / Chinese）",
                               "Language (optional; e.g. English)"))
                ana_hint = gr.Markdown(model_hint_for(ana_model.value))
                ana_btn = _localized(
                    gr.Button("🔎 开始分析", variant="primary", size="lg"),
                    value=("🔎 开始分析", "🔎 Analyze"))

            # 右列：分析结果
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### 📄 分析结果"),
                               value=("#### 📄 分析结果", "#### 📄 Results"))
                    ana_out = _localized(gr.Textbox(
                        label="结果（时间单位：秒）", lines=14,
                        placeholder="语音段 / 说话人 / 逐词时间戳将显示在这里……"),
                        label=("结果（时间单位：秒）", "Results (seconds)"),
                        placeholder=("语音段 / 说话人 / 逐词时间戳将显示在这里……",
                                     "Segments, speakers or word timestamps appear here…"))
                    ana_json = _localized(gr.File(label="原始 JSON 结果", interactive=False),
                                          label=("原始 JSON 结果", "Raw JSON"))
                    ana_msg = gr.Markdown("")

        _wire_model_manager(ana_mm, ANALYZE_TASKS, ana_hint)
        ana_model.change(_align_fields_visibility, ana_model, [ana_text, ana_lang])
        ana_btn.click(lambda: ("", None, ""), None, [ana_out, ana_json, ana_msg]).then(
            do_analyze, [ana_model, ana_audio, ana_text, ana_lang],
            [ana_out, ana_json, ana_msg])

    # ---------------- 声音设计 (vdes) ----------------
    with _localized(gr.Tab("🎨 声音设计"),
                    label=("🎨 声音设计", "🎨 Voice design")):
        with gr.Row(equal_height=False):
            # 左列：模型管理 + 生成设置
            with gr.Column(scale=1):
                vdes_mm = _model_manager_block("vdes", VDES_TASKS)
                vdes_model = vdes_mm["model"]

                with gr.Group():
                    _localized(gr.Markdown("#### ⚙️ 生成设置"),
                               value=("#### ⚙️ 生成设置", "#### ⚙️ Generation settings"))
                    with gr.Row():
                        vdes_seed = _localized(
                            gr.Number(label="seed（-1=随机）", value=1234, precision=0),
                            label=("seed（-1=随机）", "seed (-1=random)"))
                        vdes_maxtok = _localized(
                            gr.Number(label="max_tokens", value=1200, precision=0),
                            label=("max_tokens", "max_tokens"))
                    vdes_adv_state = gr.State({})
                    with _localized(
                            gr.Accordion("高级参数（只发送改动过的项）", open=False),
                            label=("高级参数（只发送改动过的项）", "Advanced options")):
                        # instruct 有专用的『音色描述』输入框，这里不重复生成
                        _render_param_controls(vdes_model, vdes_adv_state, skip=("instruct",))
                    with _localized(
                            gr.Accordion("其它参数（JSON，覆盖控件）", open=False),
                            label=("其它参数（JSON，覆盖控件）", "Other options (JSON override)")):
                        vdes_adv = gr.Textbox(label="", placeholder='{"temperature": 0.9}',
                                              lines=3)
                vdes_hint = gr.Markdown(model_hint_for(vdes_model.value))

            # 右列：设计内容 + 生成 + 输出
            with gr.Column(scale=1):
                with gr.Group():
                    _localized(gr.Markdown("#### ✍️ 设计内容"),
                               value=("#### ✍️ 设计内容", "#### ✍️ Voice description"))
                    vdes_instruct = _localized(gr.Textbox(
                        label="音色描述（用文字描述想要的声音）", lines=3,
                        placeholder="例：低沉磁性的中年男声，语速偏慢"),
                        label=("音色描述（用文字描述想要的声音）", "Voice description"),
                        placeholder=("例：低沉磁性的中年男声，语速偏慢",
                                     "Example: a calm, deep middle-aged male voice"))
                    vdes_text = gr.Textbox(
                        label=_t("要合成的文字", "Text to synthesize", INITIAL_LANGUAGE),
                        lines=5,
                        value=_t(VDES_TEXT_DEFAULT_ZH, VDES_TEXT_DEFAULT_EN,
                                 INITIAL_LANGUAGE))

                vdes_btn = _localized(
                    gr.Button("🎨 生成语音", variant="primary", size="lg"),
                    value=("🎨 生成语音", "🎨 Generate speech"))
                with gr.Group():
                    _localized(gr.Markdown("#### 🔊 输出音频"),
                               value=("#### 🔊 输出音频", "#### 🔊 Output"))
                    vdes_out = _localized(
                        gr.Audio(label="输出音频", type="filepath",
                                 elem_classes="audio-default"),
                        label=("输出音频", "Output audio"))
                    vdes_msg = gr.Markdown("")

        _wire_model_manager(vdes_mm, VDES_TASKS, vdes_hint)
        vdes_model.change(lambda: {}, None, vdes_adv_state)  # reset knobs on model switch
        vdes_btn.click(lambda: (None, ""), None, [vdes_out, vdes_msg]).then(
            do_vdes,
            [vdes_model, vdes_text, vdes_instruct, vdes_seed, vdes_maxtok,
             vdes_adv_state, vdes_adv],
            [vdes_out, vdes_msg])

    # 上传/录制后把文件换成短 ASCII 临时名，绕过 Gradio 在 Windows 上无法渲染
    # 含中文/超长文件名的声波图（见 _stage_upload）。只接输入类音频控件。
    tts_upload.upload(_stage_tts_voice_upload, tts_upload,
                      [tts_upload, tts_voice_name])
    tts_upload.stop_recording(_stage_tts_voice_recording, tts_upload,
                              [tts_upload, tts_voice_name])
    tts_upload.clear(lambda: (None, ""), None,
                     [tts_upload, tts_voice_name])

    for _in_audio in (asr_audio, gen_audio, vc_source, vc_target, sep_audio,
                      ana_audio):
        # upload: 长文件强制换成新的稳定临时文件，避免 X 清空后再次上传长音频时
        # 前端复用旧 preview 状态导致波形不渲染。
        _in_audio.upload(lambda p: _stage_upload(p, force_copy=True), _in_audio, _in_audio)
        # microphone: 录音完成不是 upload 事件，单独接 stop_recording。
        _in_audio.stop_recording(_stage_recording, _in_audio, _in_audio)
        # clear: 明确把后端值置空，避免长音频 preview fetch 被中断后控件状态残留。
        _in_audio.clear(lambda: None, None, _in_audio)

    # 顺序与 refresh()/TAB_SPECS 一致：各页下拉、状态行、各页提示、
    # ASR 流式选项。
    _refresh_outputs = [
        tts_model, asr_model, gen_model, vc_model, sep_model, ana_model, vdes_model,
        status,
        tts_hint, asr_hint, gen_hint, vc_hint, sep_hint, ana_hint, vdes_hint,
        asr_stream,
    ]
    for _mm in (tts_mm, asr_mm, gen_mm, vc_mm, sep_mm, ana_mm, vdes_mm):
        _mm["refresh_btn"].click(refresh, None, _refresh_outputs)
    # 顶部状态行在建界面时只求值一次，浏览器刷新会看到那份静态初值（即使模型
    # 已加载也显示“server 未运行”）。server 本身就是状态的唯一真相源
    # （/health + /v1/models），每次页面加载重新探测即可，无需额外状态文件。
    # 音频播放器换源后把旧播放进度清回 0:00（见 _RESET_AUDIO_SEEK_JS 注释）。
    demo.load(None, None, None, js=_RESET_AUDIO_SEEK_JS)
    _localized(
        gr.Markdown(
            "---\n<center><small>audio.cpp WebUI · 按需加载，同一时刻只驻留一个模型 · "
            "详细说明见 webui/README.md</small></center>"),
        value=(
            "---\n<center><small>audio.cpp WebUI · 按需加载，同一时刻只驻留一个模型 · "
            "详细说明见 webui/README.md</small></center>",
            "---\n<center><small>audio.cpp WebUI · On-demand loading · One model at a time · "
            "See webui/README.md</small></center>"))

    def _language_updates(language, *values):
        set_language(language)
        model_ids, current_vdes_text = values[:7], values[7]
        hints = [model_hint_for(model_id, language) for model_id in model_ids]
        return (*_localized_updates(language),
                _vdes_text_language_update(language, current_vdes_text),
                server_status(), _api_usage_md(), *hints)

    def _change_ui_language(language, *values):
        language = save_language(language)
        return _language_updates(language, *values)

    def _page_load_updates(language, *model_ids):
        """Refresh live status without re-rendering the already localized UI."""
        set_language(language)
        hints = [model_hint_for(model_id, language) for model_id in model_ids]
        return server_status(), _api_usage_md(language), *hints

    language_inputs = [
        ui_language, tts_model, asr_model, gen_model, vc_model,
        sep_model, ana_model, vdes_model, vdes_text,
    ]
    language_outputs = [
        *[_component for _component, _props in _I18N_COMPONENTS],
        vdes_text, status, api_usage,
        tts_hint, asr_hint, gen_hint, vc_hint, sep_hint, ana_hint, vdes_hint,
    ]

    ui_language.change(
        _change_ui_language,
        language_inputs,
        language_outputs,
        show_progress="hidden")
    demo.load(
        _page_load_updates,
        language_inputs[:8],
        [status, api_usage,
         tts_hint, asr_hint, gen_hint, vc_hint, sep_hint, ana_hint, vdes_hint],
        show_progress="hidden")


if __name__ == "__main__":
    open_browser = os.environ.get("AUDIOCPP_NO_BROWSER") != "1"
    # Gradio 6 moved css/theme from the Blocks constructor to launch().
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=open_browser,
                css=CUSTOM_CSS)
