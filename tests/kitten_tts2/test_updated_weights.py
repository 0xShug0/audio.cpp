"""Small offline packing tests; requires NumPy, PyTorch and safetensors."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/community_models'))
import kitten_tts2_weights as weights


def dequantize(payload, bits):
    blocks = np.frombuffer(payload, np.uint8).reshape(-1, 18 if bits == 4 else 34)
    scales = blocks[:, :2].copy().view('<f2').astype(np.float32)
    if bits == 4:
        codes = np.concatenate((blocks[:, 2:] & 15, blocks[:, 2:] >> 4), 1).astype(np.float32) - 8
    else:
        codes = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
    return codes * scales


class PackingTests(unittest.TestCase):
    def test_exact_blocks_include_zero_and_small_scales(self):
        values = torch.tensor([-1, 0, 1, 0] * 8).repeat(4, 1).to(torch.bfloat16)
        values *= torch.tensor([0, 0.123046875, 2**-22, 2**-24], dtype=torch.bfloat16)[:, None]
        for bits in (4, 8):
            self.assertTrue(np.array_equal(dequantize(weights.exact_ternary_blocks(values, bits), bits), values.float().numpy()))

    def test_nonternary_and_unrepresentable_scales_rejected(self):
        values = torch.zeros(1, 32, dtype=torch.bfloat16)
        values[0, :2] = torch.tensor([1, 0.5])
        for bits in (4, 8):
            with self.assertRaises(ValueError): weights.exact_ternary_blocks(values, bits)
            with self.assertRaises(ValueError): weights.exact_ternary_blocks(torch.full_like(values, 2**-30), bits)

    def test_base3_digit_order_and_truncation(self):
        # Base-3 byte 102 contains digits 0,1,2,0,1, representing -1,0,1,-1,0.
        codes = torch.full((26,), 102, dtype=torch.uint8)
        expected = torch.tensor(([-1,0,1,-1,0] * 26)[:128], dtype=torch.bfloat16).reshape(1,128)
        actual = weights.unpack_ternary(codes, torch.ones(1,1,dtype=torch.bfloat16), [1,128])
        self.assertTrue(torch.equal(actual, expected))
        with self.assertRaises(ValueError): weights.unpack_ternary(codes[:-1], torch.ones(1,1), [1,128])
        with self.assertRaises(ValueError): weights.unpack_ternary(torch.full_like(codes,255), torch.ones(1,1), [1,128])


if __name__ == '__main__':
    unittest.main()
