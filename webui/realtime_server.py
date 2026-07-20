"""
Background FastAPI server for the realtime voice pipeline.

Runs in a daemon thread inside the Gradio webui process.  Exposes::

    ws://127.0.0.1:8765/v1/realtime   — OpenAI Realtime protocol WebSocket
    http://127.0.0.1:8765/realtime/   — static orb frontend
    GET /health                         — liveness check

These are only *defaults*; the browser Settings panel overrides them per session
via ``session.update`` (see ``_config_from_session``).

Environment variables::

    AUDIOCPP_LLM_API_KEY     DeepSeek / OpenAI-compatible API key
    AUDIOCPP_LLM_BASE_URL    Chat Completions endpoint (default: https://api.deepseek.com/v1)
    AUDIOCPP_LLM_MODEL       Model ID (default: deepseek-chat)
    AUDIOCPP_TTS_SERVER      C++ TTS server URL (default: http://127.0.0.1:8080)
    AUDIOCPP_TTS_MODEL       TTS model id loaded in that server (default: qwen3-tts)
    AUDIOCPP_TTS_VOICE       Named cached voice id (optional; else use voice_ref)
    AUDIOCPP_TTS_VOICE_REF   Reference audio path for voice cloning
    AUDIOCPP_TTS_REF_TEXT    Text of the reference audio
    AUDIOCPP_ASR_SERVER      C++ ASR server URL (default: http://127.0.0.1:8081)
    AUDIOCPP_ASR_MODEL       ASR model id loaded in that server (default: qwen3-asr)
    AUDIOCPP_ASR_LANGUAGE    Optional language hint for the ASR model
    AUDIOCPP_INSTRUCTIONS    System prompt for the assistant
    AUDIOCPP_REALTIME_PORT   WebSocket server port (default: 8765)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from realtime_pipeline import (
    AssistantText,
    AudioChunk,
    PipelineEvent,
    RealtimePipeline,
    ResponseDone,
    SpeechStarted,
    SpeechStopped,
    TurnDiscarded,
    UserTranscript,
)

logger = logging.getLogger("realtime.server")

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "realtime_static")

# ── helpers ─────────────────────────────────────────────────────────────


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _generate_id(prefix: str = "id") -> str:
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _join_transcript(parts: list[str]) -> str:
    out = ""
    for part in parts:
        text = (part or "").strip()
        if not text:
            continue
        if not out:
            out = text
        elif (out[-1].isspace() or text[0].isspace()
              or text[0] in ",.;:!?，。！？、；：）]}》”’"
              or out[-1] in "([{（《“‘"
              or "\u3400" <= out[-1] <= "\u9fff"
              or "\u3400" <= text[0] <= "\u9fff"):
            out += text
        else:
            out += " " + text
    return out.strip()


def _resolve_voice_ref(path: str) -> str:
    """If *path* is relative, resolve it against the webui/ directory so the
    C++ TTS server (whose CWD may be anywhere) can open it."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(HERE, path))


def _default_config() -> dict:
    """Pipeline config from environment (browser Settings override per session)."""
    return {
        "tts_server": _env("AUDIOCPP_TTS_SERVER", "http://127.0.0.1:8080"),
        "tts_model": _env("AUDIOCPP_TTS_MODEL", "qwen3-tts"),
        "tts_voice": _env("AUDIOCPP_TTS_VOICE", ""),
        "tts_voice_ref": _resolve_voice_ref(_env("AUDIOCPP_TTS_VOICE_REF", "")),
        "tts_reference_text": _env("AUDIOCPP_TTS_REF_TEXT", ""),
        "asr_server": _env("AUDIOCPP_ASR_SERVER", "http://127.0.0.1:8081"),
        "asr_model": _env("AUDIOCPP_ASR_MODEL", "qwen3-asr"),
        "asr_language": _env("AUDIOCPP_ASR_LANGUAGE", ""),
        "llm_base_url": _env("AUDIOCPP_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        "llm_api_key": _env("AUDIOCPP_LLM_API_KEY", ""),
        "llm_model": _env("AUDIOCPP_LLM_MODEL", "deepseek-chat"),
        "instructions": _env("AUDIOCPP_INSTRUCTIONS", ""),
    }


# Keys the browser is allowed to override via session.update -> session.audiocpp.
_OVERRIDABLE_KEYS = {
    "tts_server", "tts_model", "tts_voice", "tts_voice_ref", "tts_reference_text",
    "asr_server", "asr_model", "asr_language",
    "llm_base_url", "llm_api_key", "llm_model", "instructions",
}


def _config_from_session(session: dict) -> dict:
    """Extract pipeline overrides from a session.update payload.  Custom fields
    live under ``session.audiocpp``; the standard OpenAI Realtime fields
    ``instructions`` and ``audio.output.voice`` are also honored."""
    cfg: dict[str, Any] = {}
    audiocpp = session.get("audiocpp")
    if isinstance(audiocpp, dict):
        for key, value in audiocpp.items():
            if key in _OVERRIDABLE_KEYS:
                cfg[key] = value
    if isinstance(session.get("instructions"), str):
        cfg["instructions"] = session["instructions"]
    audio = session.get("audio")
    if isinstance(audio, dict):
        output = audio.get("output")
        if isinstance(output, dict) and isinstance(output.get("voice"), str) and output["voice"]:
            cfg["tts_voice"] = output["voice"]
    # Resolve relative voice_ref paths against webui/ so the C++ TTS server
    # finds them regardless of its own working directory.
    if "tts_voice_ref" in cfg:
        cfg["tts_voice_ref"] = _resolve_voice_ref(cfg["tts_voice_ref"])
    return cfg


# ── FastAPI app factory ─────────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(title="audio.cpp Realtime")

    defaults = _default_config()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/config")
    async def config():
        """Return the env-default pipeline config so the browser Settings panel
        can pre-fill fields (TTS/ASR/LLM URLs, model ids, voice_ref, etc.).
        Do not echo the LLM API key back to the browser; the pipeline can still
        use the env-provided key as its server-side default."""
        public_defaults = dict(defaults)
        public_defaults["llm_api_key"] = ""
        public_defaults["llm_api_key_set"] = bool(defaults.get("llm_api_key"))
        return public_defaults

    @app.websocket("/v1/realtime")
    async def ws_realtime(ws: WebSocket):
        await ws.accept()
        session_id = _generate_id("session")

        pipeline = RealtimePipeline(**defaults)
        pipeline.start()

        # ── Send session.created ──
        await ws.send_json({
            "type": "session.created",
            "session": {
                "id": session_id,
                "type": "realtime",
                "audio": {
                    "input": {"format": {"type": "pcm16", "sample_rate": 16000, "channels": 1}},
                    "output": {"format": {"type": "pcm16", "sample_rate": 16000, "channels": 1}},
                },
            },
        })

        async def send_events():
            """Poll pipeline events and send them over the WebSocket."""
            current_resp_id: Optional[str] = None  # stable across one assistant turn
            _full_parts: list[str] = []    # committed sentences for response.done

            def _ensure_resp_id() -> str:
                nonlocal current_resp_id
                if current_resp_id is None:
                    current_resp_id = _generate_id("resp")
                return current_resp_id

            while True:
                await asyncio.sleep(0.01)
                for event in pipeline.drain_events():
                    if isinstance(event, SpeechStarted):
                        current_resp_id = None  # barge-in: new turn, new response
                        _full_parts = []
                        item_id = _generate_id("item")
                        await ws.send_json({
                            "type": "input_audio_buffer.speech_started",
                            "item_id": item_id,
                        })
                    elif isinstance(event, SpeechStopped):
                        await ws.send_json({
                            "type": "input_audio_buffer.speech_stopped",
                        })
                    elif isinstance(event, TurnDiscarded):
                        await ws.send_json({
                            "type": "input_audio_buffer.turn_discarded",
                            "reason": event.reason,
                        })
                    elif isinstance(event, UserTranscript):
                        item_id = _generate_id("item")
                        await ws.send_json({
                            "type": "conversation.item.input_audio_transcription.completed",
                            "item_id": item_id,
                            "transcript": event.text,
                        })
                    elif isinstance(event, AssistantText):
                        # Each delta is now a complete sentence, emitted when
                        # its TTS audio starts being sent — so the frontend
                        # shows the text synchronised with audio playback.
                        rid = _ensure_resp_id()
                        await ws.send_json({
                            "type": "response.output_audio_transcript.delta",
                            "response_id": rid,
                            "delta": event.text,
                        })
                        # Immediately commit as a done segment (the pipeline
                        # already split on sentence boundaries).
                        _full_parts.append(event.text)
                        await ws.send_json({
                            "type": "response.output_audio_transcript.done",
                            "response_id": rid,
                            "transcript": event.text,
                        })
                    elif isinstance(event, AudioChunk):
                        b64 = base64.b64encode(event.pcm).decode("ascii")
                        await ws.send_json({
                            "type": "response.output_audio.delta",
                            "response_id": _ensure_resp_id(),
                            "delta": b64,
                        })
                    elif isinstance(event, ResponseDone):
                        rid = current_resp_id or _ensure_resp_id()
                        full_transcript = _join_transcript(_full_parts)
                        await ws.send_json({
                            "type": "response.done",
                            "response": {
                                "id": rid,
                                "status": "completed",
                                "output": [{
                                    "type": "message",
                                    "content": [{
                                        "type": "audio",
                                        "transcript": full_transcript,
                                    }],
                                }] if full_transcript else [],
                            },
                        })
                        current_resp_id = None
                        _full_parts = []

        # Start the event sender as a background task
        send_task = asyncio.create_task(send_events())

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "session.update":
                    session = msg.get("session")
                    if isinstance(session, dict):
                        cfg = _config_from_session(session)
                        if cfg:
                            pipeline.configure(cfg)
                    await ws.send_json({"type": "session.updated"})

                elif msg_type == "input_audio_buffer.append":
                    audio_b64 = msg.get("audio", "")
                    if audio_b64:
                        try:
                            pcm = base64.b64decode(audio_b64)
                            pipeline.feed_audio(pcm)
                        except Exception:
                            pass

                elif msg_type == "response.cancel":
                    pipeline.cancel()

                elif msg_type == "input_audio_buffer.commit":
                    pass  # VAD handles boundaries automatically

                elif msg_type == "response.create":
                    pass  # LLM is triggered automatically after STT

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected: %s", session_id)
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass
            pipeline.stop()

    # Serve static orb frontend
    if os.path.isdir(STATIC_DIR):
        app.mount("/realtime", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


# ── server launcher (called from webui.py) ──────────────────────────────


class RealtimeServerThread:
    """Manages a uvicorn server in a background daemon thread."""

    def __init__(self, port: int = 8765):
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[uvicorn.Server] = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> None:
        if self._server is not None:
            return
        app = create_app()
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.serve())  # type: ignore[union-attr]

        self._thread = threading.Thread(target=_run, daemon=True, name="realtime-server")
        self._thread.start()
        logger.info("Realtime server started on ws://127.0.0.1:%d/v1/realtime", self.port)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        logger.info("Realtime server stopped")


# Module-level singleton
_server: Optional[RealtimeServerThread] = None


def get_server(port: int = 8765) -> RealtimeServerThread:
    global _server
    if _server is None:
        _server = RealtimeServerThread(port=port)
    return _server


# ── direct execution ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("AUDIOCPP_REALTIME_PORT", "8765"))
    server = get_server(port)
    server.start()
    print(f"Realtime server running at http://127.0.0.1:{port}/realtime/")
    print(f"WebSocket at ws://127.0.0.1:{port}/v1/realtime")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("Server stopped.")
