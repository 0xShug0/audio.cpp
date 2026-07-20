import queue
import threading
import unittest

import numpy as np

from realtime_pipeline import (
    RealtimePipeline,
    ResponseDone,
    SpeechStopped,
    TurnDiscarded,
    UserTranscript,
)


class _CancelScope:
    generation = 1

    def __init__(self, stale: bool = False):
        self._stale = stale

    def is_stale(self, generation: int) -> bool:
        return self._stale


class _STT:
    def __init__(self, text: str):
        self._text = text

    def transcribe(self, audio: np.ndarray) -> str:
        return self._text


class _LLM:
    def __init__(self):
        self.user_messages: list[str] = []

    def add_user_message(self, text: str) -> None:
        self.user_messages.append(text)

    def add_assistant_message(self, text: str) -> None:
        pass

    def stream(self, cancel_scope: _CancelScope, generation: int):
        return iter(())


class _SingleTurnQueue:
    def __init__(self, turn: SpeechStopped, stop_event: threading.Event):
        self._turn = turn
        self._stop_event = stop_event

    def get(self, timeout: float) -> SpeechStopped:
        self._stop_event.set()
        return self._turn


def _pipeline_for_transcript(text: str) -> RealtimePipeline:
    pipeline = RealtimePipeline.__new__(RealtimePipeline)
    pipeline._out_events = queue.Queue()
    pipeline._cancel_scope = _CancelScope()
    pipeline._stt = _STT(text)
    pipeline._llm = _LLM()
    pipeline._cfg_lock = threading.Lock()
    pipeline._tts_model = "unused"
    pipeline._tts_voice = ""
    pipeline._tts_voice_ref = ""
    pipeline._tts_reference_text = ""
    pipeline._tts_lock = threading.Lock()
    pipeline._tts_pending = 0
    pipeline._stop_event = threading.Event()
    return pipeline


class RealtimeTurnLifecycleTests(unittest.TestCase):
    def test_too_short_segment_discards_turn(self):
        pipeline = RealtimePipeline.__new__(RealtimePipeline)
        pipeline._out_events = queue.Queue()
        pipeline._stop_event = threading.Event()
        pipeline._turn_queue = _SingleTurnQueue(
            SpeechStopped(audio=np.zeros(100, dtype=np.float32)),
            pipeline._stop_event,
        )

        pipeline._pipeline_loop()

        self.assertEqual(
            pipeline.drain_events(),
            [TurnDiscarded(reason="too_short")],
        )

    def test_empty_transcript_discards_turn_without_starting_llm(self):
        pipeline = _pipeline_for_transcript("")

        pipeline._run_turn(np.zeros(16000, dtype=np.float32))

        events = pipeline.drain_events()
        self.assertEqual(events, [TurnDiscarded(reason="empty_transcript")])
        self.assertEqual(pipeline._llm.user_messages, [])

    def test_successful_transcript_keeps_existing_response_lifecycle(self):
        pipeline = _pipeline_for_transcript("hello")

        pipeline._run_turn(np.zeros(16000, dtype=np.float32))

        events = pipeline.drain_events()
        self.assertEqual(events, [UserTranscript(text="hello"), ResponseDone()])
        self.assertEqual(pipeline._llm.user_messages, ["hello"])


if __name__ == "__main__":
    unittest.main()
