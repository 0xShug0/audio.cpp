"""Regression tests for the exact-output optimization gate (no models needed)."""

import json
from pathlib import Path
import tempfile
import unittest

from compare_optimization import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.before, self.after = root / "before.json", root / "after.json"
        self.audio = [root / "before.wav", root / "after.wav"]
        for path in self.audio:
            path.write_bytes(b"test audio payload")
        self.payload = {
            "family": "mira_tts", "backend": "cpu", "model": "model.gguf",
            "mode": "offline", "results": [{
                "name": "repeat", "iteration": 1, "mode": "offline",
                "audio_hash": "same", "samples": 48000, "sample_rate": 48000,
                "channels": 1, "audio_seconds": 1.0, "event_count": 0,
                "wall_ms": 1000, "audio_out": str(self.audio[0]),
            }],
        }
        self.before.write_text(json.dumps(self.payload), encoding="utf-8")
        self.payload["results"][0]["audio_out"] = str(self.audio[1])
        self.payload["results"][0]["wall_ms"] = 900
        self.save_after()

    def save_after(self):
        self.after.write_text(json.dumps(self.payload), encoding="utf-8")

    def test_identical_output(self):
        self.assertIn("10.0%", compare(self.before, self.after)[0])

    def test_changed_wav_even_with_same_reported_hash(self):
        self.audio[1].write_bytes(b"changed audio")
        with self.assertRaisesRegex(ValueError, "WAV bytes differ"):
            compare(self.before, self.after)

    def test_different_metadata(self):
        self.payload["results"][0]["event_count"] = 2
        self.save_after()
        with self.assertRaisesRegex(ValueError, "event_count changed"):
            compare(self.before, self.after)

    def test_wrong_backend(self):
        self.payload["backend"] = "vulkan"
        self.save_after()
        with self.assertRaisesRegex(ValueError, "different backend"):
            compare(self.before, self.after)

    def test_empty_results(self):
        self.payload["results"] = []
        self.save_after()
        with self.assertRaisesRegex(ValueError, "empty results"):
            compare(self.before, self.after)

    def test_duplicate_results(self):
        self.payload["results"].append(self.payload["results"][0])
        self.save_after()
        with self.assertRaisesRegex(ValueError, "duplicate request"):
            compare(self.before, self.after)


if __name__ == "__main__":
    unittest.main()
