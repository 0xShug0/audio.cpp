"""Reference comparison must not hide missing words or a duration-frame shift."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("validate_phonon2", Path(__file__).with_name("validate_phonon2.py"))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class Row(list):
    def tolist(self):
        return list(self)


class TimestampComparisonTests(unittest.TestCase):
    def test_reference_frames_include_blank_skips_and_preserve_zero_duration(self):
        generated = SimpleNamespace(sequences=[Row([3, 3, 0, 1, 3, 2])],
                                    durations=[Row([0, 2, 0, 1, 4, 0])])
        captured = []
        def formatter(pieces, limit):
            captured.extend(pieces)
            self.assertEqual(limit, 1.0)
            return []
        validator.reference_words(generated, ["\u2581He", "llo", "\u2581world"], formatter, 1.)
        self.assertEqual(captured, [(" He", .16, 0), ("llo", .16, .08), (" world", .56, 0)])

    def test_missing_speech_timestamps_fail(self):
        self.assertFalse(validator.compare_words([], [{"text": "hi", "start": 0, "end": .08}])["timestamps_match"])
        self.assertTrue(validator.compare_words([], [])["timestamps_match"])

    def test_one_frame_shift_is_not_output_rounding(self):
        actual = [{"word": "hi", "start_sample": 1280, "end_sample": 2560}]
        expected = [{"text": "hi", "start": 0, "end": .16}]
        result = validator.compare_words(actual, expected)
        self.assertFalse(result["timestamps_match"])
        self.assertEqual(result["max_shift_ms"], 80.)
        expected[0]["start"] = .08
        self.assertTrue(validator.compare_words(actual, expected)["timestamps_match"])


if __name__ == "__main__":
    unittest.main()
