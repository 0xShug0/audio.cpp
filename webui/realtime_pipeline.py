"""
Realtime voice pipeline for audio.cpp: VAD -> STT -> LLM -> TTS.

Everything runs in-process inside the standalone realtime server (see
realtime_server.py).  Speech detection is Python silero-vad (tiny, CPU); STT and
TTS are HTTP calls to two audio.cpp C++ servers (one ASR model, one TTS model),
so the realtime loop exercises audio.cpp's own ASR + TTS at the same time. The
LLM is any OpenAI-compatible Chat Completions endpoint (DeepSeek by default).

Topology (URLs/models are per-session, pushed from the browser Settings panel):

    mic PCM16 16k --> [silero VAD] --> speech segment
                                   --> [ASR server /v1/audio/transcriptions] --> text
                                   --> [LLM /chat/completions, streamed] --> reply tokens
                                   --> [TTS server /v1/audio/speech, per sentence] --> WAV
                                   --> resample to 16k PCM16 --> browser playback

Two independent audio.cpp servers are expected, e.g.:

    run_server.bat qwen3-tts 8080     (TTS)
    run_server.bat qwen3-asr 8081     (ASR)

Both may also be the *same* multi-model server; the pipeline only needs a URL +
model id for each stage, so either topology works.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import re
import tempfile
import threading
import wave
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import requests
import torch
import httpx

logger = logging.getLogger("realtime.pipeline")

# ── constants ──────────────────────────────────────────────────────────
PIPELINE_SR = 16000
VAD_FRAME = 512            # silero window at 16 kHz (~32 ms)
CHUNK_SAMPLES = 512        # samples per outbound audio chunk at 16 kHz
_SENTENCE_END = re.compile(r"[.!?。！？…\n]")
_CLAUSE_END = re.compile(r"[,;:，、；：]")
_SOFT_BREAK = re.compile(r"\s+")
_MIN_TTS_CHARS = 6         # don't synthesize tiny fragments on their own
_TARGET_TTS_CHARS = 36     # small chunks keep first audio responsive on local GPUs
_MAX_TTS_CHARS = 72        # force a chunk even if the LLM avoids punctuation


# ── cancel scope (cooperative cancellation for barge-in) ───────────────


class CancelScope:
    """Generation-counter pattern: each barge-in bumps ``generation``; the
    in-flight turn checks ``is_stale(gen)`` between steps and aborts early."""

    def __init__(self) -> None:
        self.generation: int = 0

    def cancel(self) -> None:
        self.generation += 1

    def is_stale(self, gen: int) -> bool:
        return gen != self.generation


# ── pipeline events (emitted by stages, consumed by the WebSocket layer) ─


@dataclass
class PipelineEvent:
    pass


@dataclass
class SpeechStarted(PipelineEvent):
    pass


@dataclass
class SpeechStopped(PipelineEvent):
    # Only the VAD->worker copy carries audio; the outbound copy leaves it None.
    audio: Optional[np.ndarray] = None


@dataclass
class UserTranscript(PipelineEvent):
    text: str
    partial: bool = False


@dataclass
class AssistantText(PipelineEvent):
    text: str


@dataclass
class AudioChunk(PipelineEvent):
    """One chunk of 16-bit PCM audio at 16 kHz."""
    pcm: bytes  # int16 LE


@dataclass
class ResponseDone(PipelineEvent):
    pass


@dataclass
class TurnDiscarded(PipelineEvent):
    """A VAD turn ended without producing a conversational response."""

    reason: str


# ── audio helpers ───────────────────────────────────────────────────────


def _wav_bytes_to_pcm16_16k(data: bytes) -> bytes:
    """Decode a WAV response and return mono PCM16 at 16 kHz.  audio.cpp models
    emit whatever native rate they run at (e.g. 24 kHz), but the browser
    playback worklet is fixed to 16 kHz, so we downmix + resample here."""
    if len(data) < 44 or data[:4] != b"RIFF":
        # Assume the server already handed back raw PCM16 @ 16 kHz.
        return data
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError) as exc:
        logger.error("TTS WAV decode failed: %r", exc)
        return b""
    if sampwidth != 2:
        logger.error("TTS returned %d-byte samples; only PCM16 is supported", sampwidth)
        return b""
    arr = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)
    if sr != PIPELINE_SR and arr.size:
        n_out = int(round(arr.size * PIPELINE_SR / sr))
        if n_out > 0:
            x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            arr = np.interp(x_new, x_old, arr.astype(np.float32))
    arr = np.clip(np.round(arr), -32768, 32767).astype("<i2")
    return arr.tobytes()


def _float_to_wav16k(audio: np.ndarray) -> bytes:
    """float32 [-1,1] @ 16 kHz mono -> WAV bytes (for the ASR request)."""
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(PIPELINE_SR)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── VAD stage (streaming silero, per-frame with hysteresis) ─────────────


class _VADStage:
    """Streaming voice-activity detection.  Fed raw mic chunks; runs silero on
    fixed 512-sample frames and emits SpeechStarted / SpeechStopped with the
    accumulated speech segment.  On speech start it also fires the barge-in
    cancel so a reply already playing is cut immediately."""

    def __init__(
        self,
        out_events: "queue.Queue[PipelineEvent]",
        turn_queue: "queue.Queue[SpeechStopped]",
        cancel_scope: CancelScope,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        min_silence_ms: int = 450,
    ):
        self._out = out_events
        self._turns = turn_queue
        self._cancel = cancel_scope
        model, _utils = torch.hub.load(
            repo_or_dir=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "third_party", "silero-vad",
            ),
            model="silero_vad",
            source="local",
            trust_repo=True,
        )
        self._model = model
        try:
            self._model.reset_states()
        except Exception:
            pass
        self._threshold = threshold
        self._min_speech_frames = max(1, int(min_speech_ms * PIPELINE_SR / 1000 / VAD_FRAME))
        self._min_silence_frames = max(1, int(min_silence_ms * PIPELINE_SR / 1000 / VAD_FRAME))
        self._pending = np.zeros(0, dtype=np.float32)   # leftover < one frame
        self._speaking = False
        self._segment: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        # Pre-speech ring buffer: keeps the last N frames while we wait for
        # _speech_run to reach _min_speech_frames.  Without this, the first
        # ~200 ms of every utterance is silently dropped (the frames that
        # *triggered* speech detection never make it into the segment), which
        # causes the ASR to miss the first word or two — especially noticeable
        # in Chinese where each character is ~100-150 ms.
        self._pre_speech_buf: list[np.ndarray] = []
        self._pre_speech_max = self._min_speech_frames + 4  # keep a few extra

    def add_chunk(self, pcm: bytes) -> None:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, arr]) if self._pending.size else arr

    def _speech_prob(self, frame: np.ndarray) -> float:
        with torch.no_grad():
            out = self._model(torch.from_numpy(frame), PIPELINE_SR)
        return float(out.reshape(-1)[0])

    def process(self) -> None:
        """Consume whole 512-sample frames from the pending buffer.  Called from
        the VAD ticker thread every ~30 ms worth of audio."""
        n = self._pending.size // VAD_FRAME
        if n == 0:
            return
        frames = self._pending[: n * VAD_FRAME].reshape(n, VAD_FRAME)
        self._pending = self._pending[n * VAD_FRAME:].copy()
        for frame in frames:
            voiced = self._speech_prob(frame) >= self._threshold
            if not self._speaking:
                if voiced:
                    self._speech_run += 1
                    # Accumulate into pre-speech ring buffer so these frames
                    # aren't lost when speech is confirmed below.
                    self._pre_speech_buf.append(frame.copy())
                    if len(self._pre_speech_buf) > self._pre_speech_max:
                        self._pre_speech_buf.pop(0)
                    if self._speech_run >= self._min_speech_frames:
                        self._speaking = True
                        self._silence_run = 0
                        # Include ALL pre-speech frames + current frame so the
                        # ASR gets the complete utterance from the very start.
                        self._segment = list(self._pre_speech_buf)
                        self._pre_speech_buf = []
                        self._out.put(SpeechStarted())
                        self._cancel.cancel()  # barge-in
                else:
                    self._speech_run = 0
                    self._pre_speech_buf = []
            else:
                self._segment.append(frame.copy())
                if voiced:
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                    if self._silence_run >= self._min_silence_frames:
                        seg = np.concatenate(self._segment) if self._segment else None
                        self._segment = []
                        self._speaking = False
                        self._speech_run = 0
                        self._out.put(SpeechStopped())
                        self._turns.put(SpeechStopped(audio=seg))


# ── STT stage (audio.cpp ASR server) ────────────────────────────────────


class _STTStage:
    """Transcribe a speech segment via an audio.cpp ASR server."""

    def __init__(self, server_url: str, model_id: str, language: str = ""):
        self._server_url = server_url.rstrip("/")
        self._model_id = model_id
        self._language = language
        self._http = requests.Session()

    def configure(self, server_url: Optional[str] = None, model_id: Optional[str] = None,
                  language: Optional[str] = None) -> None:
        if server_url:
            self._server_url = server_url.rstrip("/")
        if model_id:
            self._model_id = model_id
        if language is not None:
            self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        wav = _float_to_wav16k(audio)
        tmp = tempfile.NamedTemporaryFile(prefix="rt_asr_", suffix=".wav", delete=False)
        try:
            tmp.write(wav)
            tmp.close()
            payload: dict[str, Any] = {"model": self._model_id, "audio": tmp.name}
            if self._language:
                payload["language"] = self._language
            r = self._http.post(
                f"{self._server_url}/v1/audio/transcriptions", json=payload, timeout=60)
            if r.status_code != 200:
                logger.error("ASR error %s: %s", r.status_code, r.text[:200])
                return ""
            return (r.json().get("text") or "").strip()
        except requests.RequestException as exc:
            logger.error("ASR server unreachable (%s): %r", self._server_url, exc)
            return ""
        except (ValueError, KeyError) as exc:
            logger.error("ASR bad response: %r", exc)
            return ""
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


# ── LLM stage ───────────────────────────────────────────────────────────


class _LLMStage:
    """OpenAI-compatible Chat Completions (DeepSeek by default)."""

    DEFAULT_SYSTEM = (
        "You are a helpful voice assistant. Keep responses concise — under 3 "
        "sentences when possible. Always reply in the same language the user "
        "speaks. If the user speaks Chinese, use Simplified Chinese (简体中文), "
        "not Traditional Chinese."
    )

    def __init__(self, base_url: str, api_key: str = "", model: str = "deepseek-chat",
                 instructions: str = ""):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._system = instructions.strip() or self.DEFAULT_SYSTEM
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": self._system}]
        self._http = httpx.Client(timeout=60.0)

    def configure(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                  model: Optional[str] = None, instructions: Optional[str] = None) -> None:
        if base_url:
            self._base_url = base_url.rstrip("/")
        if api_key is not None:
            self._api_key = api_key
        if model:
            self._model = model
        if instructions is not None:
            new_system = instructions.strip() or self.DEFAULT_SYSTEM
            if new_system != self._system:
                self._system = new_system
                self._messages[0] = {"role": "system", "content": self._system}

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        self._messages.append({"role": "assistant", "content": text})
        if len(self._messages) > 21:  # system + last 20 turns
            self._messages = [self._messages[0]] + self._messages[-20:]

    def stream(self, cancel_scope: CancelScope, gen: int) -> Iterator[str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model, "messages": self._messages, "stream": True,
        }
        try:
            with self._http.stream("POST", f"{self._base_url}/chat/completions",
                                   json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    resp.read()
                    logger.error("LLM API error %s: %s", resp.status_code, resp.text[:300])
                    return
                for line in resp.iter_lines():
                    if cancel_scope.is_stale(gen):
                        return
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.RequestError as exc:
            logger.error("LLM API unreachable (%s): %r", self._base_url, exc)


# ── TTS stage (audio.cpp TTS server) ────────────────────────────────────


class _TTSStage:
    """Synthesize speech via an audio.cpp TTS server, resampled to 16 kHz."""

    def __init__(self, server_url: str):
        self._server_url = server_url.rstrip("/")
        self._http = requests.Session()

    def configure(self, server_url: Optional[str] = None) -> None:
        if server_url:
            self._server_url = server_url.rstrip("/")

    def synthesize(self, text: str, model_id: str, voice: str = "",
                   voice_ref: str = "", reference_text: str = "") -> bytes:
        payload: dict[str, Any] = {"model": model_id, "input": text}
        if voice:
            payload["voice"] = voice
        if voice_ref:
            payload["voice_ref"] = voice_ref
        if reference_text:
            payload["reference_text"] = reference_text
        try:
            r = self._http.post(
                f"{self._server_url}/v1/audio/speech", json=payload, timeout=120)
        except requests.RequestException as exc:
            logger.error("TTS server unreachable (%s): %r", self._server_url, exc)
            return b""
        if r.status_code != 200:
            logger.error("TTS error %s: %s", r.status_code, r.text[:200])
            return b""
        return _wav_bytes_to_pcm16_16k(r.content)


# ── orchestrator ────────────────────────────────────────────────────────


class RealtimePipeline:
    """Owns the full VAD->STT->LLM->TTS loop.  The WebSocket layer feeds raw mic
    audio via ``feed_audio`` and pulls outbound events via ``drain_events``.
    Per-session config (server URLs, model ids, voice, LLM key, instructions) is
    applied with ``configure`` from the browser's session.update."""

    def __init__(
        self,
        tts_server: str = "http://127.0.0.1:8080",
        tts_model: str = "qwen3-tts",
        tts_voice: str = "",
        tts_voice_ref: str = "",
        tts_reference_text: str = "",
        asr_server: str = "http://127.0.0.1:8081",
        asr_model: str = "qwen3-asr",
        asr_language: str = "",
        llm_base_url: str = "https://api.deepseek.com/v1",
        llm_api_key: str = "",
        llm_model: str = "deepseek-chat",
        instructions: str = "",
    ):
        self._out_events: "queue.Queue[PipelineEvent]" = queue.Queue()
        self._turn_queue: "queue.Queue[SpeechStopped]" = queue.Queue()
        self._cancel_scope = CancelScope()
        self._vad = _VADStage(self._out_events, self._turn_queue, self._cancel_scope)
        self._stt = _STTStage(asr_server, asr_model, asr_language)
        self._llm = _LLMStage(llm_base_url, llm_api_key, llm_model, instructions)
        self._tts = _TTSStage(tts_server)
        self._tts_model = tts_model
        self._tts_voice = tts_voice
        self._tts_voice_ref = tts_voice_ref
        self._tts_reference_text = tts_reference_text
        self._cfg_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        # TTS worker thread — LLM and TTS run in parallel via queue
        self._tts_queue: "queue.Queue[tuple]" = queue.Queue()
        self._tts_pending = 0
        self._tts_lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        for target, name in ((self._vad_loop, "realtime-vad"),
                             (self._pipeline_loop, "realtime-pipeline"),
                             (self._tts_worker, "realtime-tts")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        self._cancel_scope.cancel()
        self._tts_queue.put(None)  # wake TTS worker so it can exit

    def feed_audio(self, pcm: bytes) -> None:
        self._vad.add_chunk(pcm)

    def cancel(self) -> None:
        self._cancel_scope.cancel()

    def drain_events(self) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        while True:
            try:
                events.append(self._out_events.get_nowait())
            except queue.Empty:
                break
        return events

    def configure(self, cfg: dict[str, Any]) -> None:
        """Apply per-session config pushed from the browser (session.update).

        Empty-string values are treated as "no override" so the env defaults
        (set in run_realtime.bat) survive — e.g. qwen3-tts-Base requires a
        voice_ref; an empty browser field must not clobber the env-provided
        reference audio path.
        """
        def val(key: str) -> Optional[str]:
            v = cfg.get(key)
            return v if isinstance(v, str) and v else None

        with self._cfg_lock:
            self._stt.configure(
                server_url=val("asr_server"),
                model_id=val("asr_model"),
                language=val("asr_language"),
            )
            self._tts.configure(server_url=val("tts_server"))
            if val("tts_model"):
                self._tts_model = val("tts_model")
            if val("tts_voice"):
                self._tts_voice = val("tts_voice")
            if val("tts_voice_ref"):
                self._tts_voice_ref = val("tts_voice_ref")
            if val("tts_reference_text"):
                self._tts_reference_text = val("tts_reference_text")
            self._llm.configure(
                base_url=val("llm_base_url"),
                api_key=val("llm_api_key"),
                model=val("llm_model"),
                instructions=val("instructions"),
            )

    # ── background loops ──────────────────────────────────────────────

    def _vad_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._vad.process()
            except Exception:  # never let a bad frame kill the ticker
                logger.exception("VAD process error")
            self._stop_event.wait(0.03)

    def _pipeline_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stopped = self._turn_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            audio = stopped.audio
            if audio is None or len(audio) < PIPELINE_SR * 0.3:  # < 0.3s: ignore
                self._out_events.put(TurnDiscarded(reason="too_short"))
                continue
            try:
                self._run_turn(audio)
            except Exception:
                logger.exception("turn failed")
                self._out_events.put(ResponseDone())

    # ── TTS worker thread ──────────────────────────────────────────────

    def _tts_worker(self) -> None:
        """Pull sentences from _tts_queue, synthesize them, push AudioChunk
        events.  Runs in its own thread so the LLM continues generating while
        the TTS server synthesises the previous sentence."""
        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:           # shutdown sentinel
                break
            text, gen, model, voice, voice_ref, reference_text = item
            if self._cancel_scope.is_stale(gen):
                with self._tts_lock:
                    self._tts_pending -= 1
                continue
            try:
                pcm = self._tts.synthesize(
                    text, model_id=model, voice=voice,
                    voice_ref=voice_ref, reference_text=reference_text)
            except Exception:
                logger.exception("TTS synthesis failed")
                with self._tts_lock:
                    self._tts_pending -= 1
                continue
            step = CHUNK_SAMPLES * 2
            for i in range(0, len(pcm), step):
                if self._cancel_scope.is_stale(gen):
                    break
                chunk = pcm[i:i + step]
                if len(chunk) < step:
                    chunk = chunk.ljust(step, b"\x00")
                self._out_events.put(AudioChunk(pcm=chunk))
            with self._tts_lock:
                self._tts_pending -= 1

    # ── one conversational turn ───────────────────────────────────────

    def _run_turn(self, audio: np.ndarray) -> None:
        gen = self._cancel_scope.generation

        # ── STT ──
        text = self._stt.transcribe(audio)
        if self._cancel_scope.is_stale(gen):
            return
        if not text:
            logger.info("ASR returned no transcript; discarding realtime turn")
            self._out_events.put(TurnDiscarded(reason="empty_transcript"))
            return
        logger.info("USER: %s", text)
        self._out_events.put(UserTranscript(text=text, partial=False))
        self._llm.add_user_message(text)

        # ── LLM stream -> sentence-chunked TTS (parallel via queue) ──
        with self._cfg_lock:
            tts_model, tts_voice = self._tts_model, self._tts_voice
            tts_ref, tts_ref_text = self._tts_voice_ref, self._tts_reference_text
        pending = ""
        full = ""
        for token in self._llm.stream(self._cancel_scope, gen):
            if self._cancel_scope.is_stale(gen):
                # Barge-in: drain queued TTS items that haven't started yet,
                # then return immediately — the worker will discard in-flight
                # audio when it checks is_stale.
                drained = 0
                while True:
                    try:
                        self._tts_queue.get_nowait()
                        drained += 1
                    except queue.Empty:
                        break
                with self._tts_lock:
                    self._tts_pending = max(0, self._tts_pending - drained)
                return
            full += token
            pending += token
            while True:
                chunk, pending = self._split_speakable(pending)
                if chunk is None:
                    break
                self._speak(chunk, gen, tts_model, tts_voice, tts_ref, tts_ref_text)

        if self._cancel_scope.is_stale(gen):
            return
        if pending.strip():
            self._speak(pending, gen, tts_model, tts_voice, tts_ref, tts_ref_text)

        # Wait for all queued TTS items to finish before marking the turn done.
        while not self._stop_event.is_set():
            with self._tts_lock:
                if self._tts_pending <= 0:
                    break
            self._stop_event.wait(0.05)

        if full.strip():
            logger.info("ASSISTANT: %s", full.strip())
            self._llm.add_assistant_message(full.strip())
        self._out_events.put(ResponseDone())

    @staticmethod
    def _split_speakable(buf: str) -> tuple[Optional[str], str]:
        """Pull one short, speakable chunk from ``buf``.

        audio.cpp's HTTP TTS endpoint returns a whole WAV per request, so we
        simulate streaming by feeding it compact clauses instead of waiting for
        an entire assistant paragraph. Sentence punctuation wins; clause
        punctuation is allowed once the chunk is useful; a long punctuation-free
        span is force-split near a word boundary.
        """
        text = buf.lstrip()
        if not text:
            return None, ""

        sentence = _SENTENCE_END.search(text)
        if sentence and sentence.end() <= _TARGET_TTS_CHARS:
            end = sentence.end()
            head, tail = text[:end], text[end:]
            if len(head.strip()) >= _MIN_TTS_CHARS or tail.strip():
                return head.strip(), tail

        if len(text) >= _TARGET_TTS_CHARS:
            clause = None
            for match in _CLAUSE_END.finditer(text):
                if match.end() >= _MIN_TTS_CHARS:
                    clause = match
                    if match.end() >= _TARGET_TTS_CHARS:
                        break
            if clause and clause.end() <= _MAX_TTS_CHARS:
                end = clause.end()
                return text[:end].strip(), text[end:]

        if sentence and sentence.end() <= _MAX_TTS_CHARS:
            end = sentence.end()
            head, tail = text[:end], text[end:]
            if len(head.strip()) >= _MIN_TTS_CHARS or tail.strip():
                return head.strip(), tail

        if len(text) < _MAX_TTS_CHARS:
            return None, buf

        window = text[:_MAX_TTS_CHARS]
        split_at = 0
        for match in _SOFT_BREAK.finditer(window):
            if match.end() >= _TARGET_TTS_CHARS:
                split_at = match.end()
                break
            split_at = match.end()
        if split_at < _MIN_TTS_CHARS:
            split_at = _MAX_TTS_CHARS
        return text[:split_at].strip(), text[split_at:]

    def _speak(self, text: str, gen: int, model: str, voice: str,
               voice_ref: str, reference_text: str) -> None:
        """Queue a sentence for TTS synthesis.  The worker thread picks it up
        so LLM generation continues without blocking on the HTTP call."""
        text = text.strip()
        if not text:
            return
        # Match the original speech-to-speech architecture: text reaches the
        # client as soon as the LLM has a speakable sentence, while TTS works in
        # the background. This keeps the side panel from sitting blank during
        # local TTS latency.
        self._out_events.put(AssistantText(text=text))
        with self._tts_lock:
            self._tts_pending += 1
        self._tts_queue.put((text, gen, model, voice, voice_ref, reference_text))
