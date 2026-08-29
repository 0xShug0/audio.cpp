from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "minimax_music3" / "repack_lm_head_gguf.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repack_lm_head_gguf", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompactLmHeadTest(unittest.TestCase):
    def test_token_ids_match_native_compact_sampler_order(self) -> None:
        module = load_module()
        token_ids = module.compact_token_ids()
        self.assertEqual(len(token_ids), 16_385)
        self.assertEqual(token_ids[0], 151_675)
        self.assertEqual(token_ids[-2], 168_058)
        self.assertEqual(token_ids[-1], 151_670)
        self.assertEqual(len(set(token_ids)), len(token_ids))

    def test_repack_copies_quantized_rows_verbatim(self) -> None:
        module = load_module()
        rows = 200_000
        row_bytes = 3
        source = np.arange(rows * row_bytes, dtype=np.uint8).reshape(rows, row_bytes)
        compact = module.compact_lm_head_rows(source)
        expected_ids = module.compact_token_ids()
        self.assertEqual(compact.shape, (16_385, row_bytes))
        np.testing.assert_array_equal(compact, source[expected_ids])
        self.assertTrue(compact.flags.c_contiguous)

    def test_repack_rejects_wrong_source_vocab(self) -> None:
        module = load_module()
        source = np.zeros((199_999, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "200000"):
            module.compact_lm_head_rows(source)

    def test_replace_logical_shape_preserves_hidden_width(self) -> None:
        module = load_module()
        shapes = {
            "lm_head.weight": (200_000, 4_096),
            "model.embed_tokens.weight": (200_000, 4_096),
        }
        replaced = module.compact_logical_shapes(shapes)
        self.assertEqual(replaced["lm_head.weight"], (16_385, 4_096))
        self.assertEqual(replaced["model.embed_tokens.weight"], (200_000, 4_096))
        self.assertEqual(shapes["lm_head.weight"], (200_000, 4_096))


if __name__ == "__main__":
    unittest.main()
