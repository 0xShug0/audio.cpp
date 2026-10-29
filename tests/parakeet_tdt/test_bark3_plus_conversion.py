"""Weight-free coverage of Plus's ANS format, precision and embedded contract."""
import json
import lzma
from pathlib import Path
import sys
import tempfile
import unittest

import constriction
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/community_models"))
import convert_underdog_bark3_plus as plus


class BarkPlusConversionTest(unittest.TestCase):
    def test_levels_preserve_values_and_column_class_order(self):
        codes = np.array([[0, 1, 3, 4], [2, 4, 1, 0]], np.int32)
        classes = np.array([0, 1, 0, 15], np.uint8)
        levels = np.array([[.125, .75], [.25, 1.5]], np.float16)
        counts = np.ones((16, 5), dtype="<u4")
        coder = constriction.stream.stack.AnsCoder()
        for group in reversed(range(16)):
            columns = np.nonzero(classes == group)[0]
            model = constriction.stream.model.Categorical(np.ones(5) / 5, perfect=False)
            coder.encode_reverse(codes[:, columns].T.ravel(), model)
        class_bits = np.unpackbits(classes[:, None], axis=1)[:, -4:].ravel()
        blobs = dict(lev=lzma.compress(levels.tobytes()),
                     meta=np.packbits(class_bits).tobytes() + counts.tobytes(),
                     b=coder.get_compressed().astype("<u4").tobytes())
        entry = dict(k="levels2", K=2, shape=[2, 4], zl=True)
        actual = plus.decode_record(entry, blobs)
        np.testing.assert_array_equal(actual, np.array([[0, .125, -.125, -.75], [1.5, -1.5, .25, 0]], np.float32))
        with self.assertRaises(ValueError):
            plus.decode_record(dict(entry, shape=[2, 5]), blobs)

    def test_integer_scale_products_are_not_rounded_to_half(self):
        for bits in (5, 6):
            alphabet = 2 ** bits - 1
            codes = np.array([0, 1, alphabet - 1, 2, 3, alphabet - 2], np.int32)
            model = constriction.stream.model.Categorical(np.ones(alphabet) / alphabet, perfect=False)
            coder = constriction.stream.stack.AnsCoder()
            coder.encode_reverse(codes, model)
            scales = np.array([.123, .333], np.float16)
            blobs = dict(s=lzma.compress(scales.tobytes()), meta=np.ones(alphabet, dtype="<u4").tobytes(),
                         b=coder.get_compressed().astype("<u4").tobytes())
            actual = plus.decode_record(dict(k="ansint" + str(bits), shape=[2, 3], zs=True), blobs)
            expected = (codes.astype(np.float32) - (2 ** (bits - 1) - 1)).reshape(2, 3) * scales.astype(np.float32)[:, None]
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.array_equal(actual, actual.astype(np.float16).astype(np.float32)))

    def test_raw_sizes_lzma_bounds_and_container_boundaries(self):
        entry = dict(n="weight", k="raw", dtype="torch.float32", shape=[2], zr=True)
        blob = lzma.compress(np.array([.5, -.25], np.float16).tobytes())
        np.testing.assert_array_equal(plus.decode_record(entry, {"b": blob}), [.5, -.25])
        with self.assertRaises(ValueError):
            plus.unpack_blob(blob, True, 2)
        with self.assertRaises(ValueError):
            plus.unpack_blob(blob + b"extra", True, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bark"
            header = json.dumps(dict(format="bark-levels-v3", config="{}", index=[dict(entry, b=len(blob))])).encode()
            data = len(header).to_bytes(8, "little") + header + blob
            path.write_bytes(data)
            tensors, _ = plus.read_container(path)
            np.testing.assert_array_equal(tensors["weight"], [.5, -.25])
            for bad in (data[:-1], data + b"extra", (9999999).to_bytes(8, "little")):
                path.write_bytes(bad)
                with self.assertRaises((ValueError, EOFError)):
                    plus.read_container(path)

    def test_lossless_storage_retains_nonrepresentable_products(self):
        from safetensors.numpy import save_file
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            values = {"encoder.layers.0.attention.q_proj.weight": np.ones((2, 2), np.float32),
                      "decoder.embedding.weight": np.array([[1.001, 2.002]], np.float32)}
            save_file(values, staging / "model.safetensors")
            (staging / "tensor_manifest.json").write_text(json.dumps([dict(name=name) for name in values]))
            preserved, manifest = plus.lossless_f16_tensors(staging)
            self.assertEqual(preserved, ["decoder.embedding.weight"])
            self.assertEqual({row["name"]: row["dtype"] for row in manifest},
                             {"encoder.layers.0.attention.q_proj.weight": "F16", "decoder.embedding.weight": "F32"})

    def test_plus_contract_is_english_and_has_no_unpublished_downloads(self):
        original = (REPO / "model_specs/parakeet_tdt.json").read_bytes()
        spec = plus.package_spec(REPO)
        self.assertEqual(spec["display_name"], "Underdog Bark 3.0 Plus")
        self.assertEqual(spec["family"], "parakeet_tdt")
        self.assertEqual(spec["languages"], ["en"])
        self.assertEqual(spec["packages"], [])
        self.assertNotIn("package_defaults", spec)
        self.assertEqual(spec["ui"]["docs"], ["docs/models/underdog_bark3_plus.md"])
        self.assertEqual((REPO / "model_specs/parakeet_tdt.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
