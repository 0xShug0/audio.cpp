"""Phonon-2 packing edge cases; no weights or framework dependencies required."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[2] / "tools/community_models/convert_phonon2.py"
spec = importlib.util.spec_from_file_location("convert_phonon2", MODULE)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class PhononConversionTests(unittest.TestCase):
    def test_timestamp_metadata_unicode_punctuation_and_word_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps({"model": {"vocab": {
                "\u2581": 0, "\u2581Hello": 1, "\u2581!": 2, "\u2581\u00e9": 3,
                "\u2581\u4e2d": 4, "\u2014": 5, "123": 6, "\u2581\u0661": 7}}}), encoding="utf-8")
            config = {}
            converter.configure_timestamps(config, path)
            self.assertEqual(config["word_timestamp_mode"], "token_duration")
            self.assertEqual(config["audiocpp_punctuation_token_ids"], [0, 2, 5])

    def test_projection_compensation_cancels_default_loader_scale(self):
        values = np.array([[-.03125, 0, .125, 17.375]], dtype=np.float32)
        for name in ("encoder.subsampling.linear.weight", "encoder.subsampling.linear.bias"):
            adjusted = converter.compensate_projection(name, values)
            np.testing.assert_array_equal(adjusted * np.float32(32), values)
        np.testing.assert_array_equal(converter.compensate_projection("joint.head.weight", values), values)

    def test_batch_norm_compensation_preserves_reference_fold(self):
        gamma = np.array([.5, 2, -.25], dtype=np.float32)
        variance = np.array([0, .000001, .125], dtype=np.float32)
        adjusted = converter.compensate_batch_norm("encoder.layers.0.conv.norm.weight", gamma,
            {"encoder.layers.0.conv.norm.running_var": variance})
        loaded = adjusted / np.sqrt(np.maximum(variance, np.float32(1e-5)) + np.float32(1e-5))
        reference = gamma / np.sqrt(variance + np.float32(1e-5))
        np.testing.assert_allclose(loaded, reference, rtol=2e-7)
        self.assertEqual(adjusted[2], gamma[2])

    def test_five_values_row_padding_and_nonzero_bit_order(self):
        # Six columns force a second trit byte. High/low bits are packed over
        # nonzeros only, across row boundaries (not over zero entries).
        codes = np.array([[0, 1, 2, 0, 2, 1], [2, 0, 1, 2, 1, 0]], dtype=np.uint8)
        padded = np.ones((2, 10), dtype=np.uint8)
        padded[:, :6] = codes
        trits = (padded.reshape(2, 2, 5) * (3 ** np.arange(5))).sum(-1).astype(np.uint8)
        bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
        lo = np.array([0.25, 0.5], dtype="<f2")
        hi = np.array([1., 2.], dtype="<f2")
        blob = trits.tobytes() + np.packbits(bits, bitorder="little").tobytes() + lo.tobytes() + hi.tobytes()
        got = converter.decode_record("five_value", (2, 6), blob)
        np.testing.assert_array_equal(got, [[-.25, 0, 1, -1, .25, 0], [2, -.5, 0, .5, 0, -2]])

    def test_int6_signed_endpoints_padding_and_exact_scale_product(self):
        # Seven weights require a padded group of four. Scale * 31 is not
        # representable in F16, so a second half conversion must not occur.
        q = np.array([-32, -1, 0, 1, 15, 30, 31, 0], dtype=np.int32)
        packed = ((q + 32).reshape(-1, 4).astype(np.uint32) << np.array([0, 6, 12, 18])).sum(1)
        body = np.stack([(packed >> s) & 255 for s in (0, 8, 16)], axis=1).astype(np.uint8).tobytes()
        scale = np.array([0.10004], dtype="<f2")
        got = converter.decode_record("int6", (1, 7), body + scale.tobytes())
        expected = q[:7].astype(np.float32) * np.float32(scale[0])
        np.testing.assert_array_equal(got[0], expected)
        self.assertNotEqual(float(got[0, -1]), float(np.float16(got[0, -1])))

    def test_int8_scales_per_row(self):
        q = np.array([[-128, 127], [-1, 1]], dtype=np.int8)
        scale = np.array([.5, 2], dtype="<f2")
        np.testing.assert_array_equal(converter.decode_record("int8", (2, 2), q.tobytes() + scale.tobytes()),
                                      [[-64, 63.5], [-2, 2]])

    def test_scalar_training_counter(self):
        self.assertTrue(np.isinf(converter.decode_record("fp16", (), np.array(np.inf, dtype="<f2").tobytes())))

    def test_bad_record_sizes_and_shapes(self):
        for kind, shape, blob in [("five_value", (1, 1), b""), ("int6", (1, 4), b"\0"),
                                  ("fp16", (2,), b"\0\0"), ("int8", (), b""),
                                  ("fp16", (-1,), b""), ("int4", (1, 1), b"\0\0")]:
            with self.subTest(kind=kind, shape=shape), self.assertRaises(ValueError):
                converter.decode_record(kind, shape, blob)

    def test_invalid_trit_byte(self):
        with self.assertRaises(ValueError):
            converter.decode_record("five_value", (1, 1), bytes([255, 0, 0, 0, 0]))

    def test_container_truncation_duplicates_and_trailing_bytes(self):
        entry = {"n": "x", "k": "fp16", "shape": [1], "b": 2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.fermion"
            for entries, data in [([entry], b"\0"), ([entry], b"\0\0extra"),
                                  ([entry, entry], b"\0\0\0\0")]:
                header = json.dumps({"format": converter.FORMAT, "index": entries}).encode()
                path.write_bytes(len(header).to_bytes(8, "little") + header + data)
                with self.assertRaises(ValueError):
                    converter.read_container(path)

    def test_unsupported_format_and_oversize_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.fermion"
            for data in [(2**63).to_bytes(8, "little"), (2).to_bytes(8, "little") + b"{}"]:
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    converter.read_container(path)


if __name__ == "__main__":
    unittest.main()
