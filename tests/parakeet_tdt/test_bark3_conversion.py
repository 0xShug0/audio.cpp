"""Weight-free checks for Bark's model-local conversion contract.

Run: python -m unittest discover -s tests/parakeet_tdt -p test_bark3_conversion.py
"""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/community_models"))
import convert_underdog_bark3 as bark


class BarkConversionTest(unittest.TestCase):
    def test_projection_compensation_is_exact(self):
        values = np.array([0., -0.00390625, 1., 16., 128.], dtype=np.float32)
        for suffix in ("weight", "bias"):
            actual = bark.compensate_projection("encoder.subsampling.linear." + suffix, values)
            np.testing.assert_array_equal(actual * 32, values)
        self.assertIs(bark.compensate_projection("decoder.embedding.weight", values), values)

    def test_batch_norm_compensation_restores_reference_gain(self):
        var = np.array([0., 1e-9, 1e-5, 1.], dtype=np.float32)
        gamma = np.array([1., 2., -3., 4.], dtype=np.float32)
        name = "encoder.layers.0.conv.norm.weight"
        actual = bark.compensate_batch_norm(name, gamma, {name[:-6] + "running_var": var})
        native_gain = actual / np.sqrt(np.maximum(var, 1e-5) + 1e-5)
        reference_gain = gamma / np.sqrt(var + 1e-5)
        np.testing.assert_allclose(native_gain, reference_gain, rtol=2e-7)
        with self.assertRaises(ValueError):
            bark.compensate_batch_norm(name, gamma, {name[:-6] + "running_var": -np.ones(4)})

    def test_architecture_is_explicit(self):
        cfg = dict(model_type="parakeet_tdt", vocab_size=8193, num_decoder_layers=2,
                   durations=[0, 1, 2, 3, 4], encoder_config=dict(hidden_size=1024,
                   num_hidden_layers=24, num_attention_heads=8, subsampling_factor=8,
                   conv_kernel_size=9, scale_input=False))
        bark.validate_config(cfg)
        for section, key, value in [(None, "model_type", "bark"),
                                    ("encoder_config", "scale_input", True),
                                    ("encoder_config", "hidden_size", 512),
                                    (None, "durations", [0, 1])]:
            changed = copy.deepcopy(cfg)
            (changed[section] if section else changed)[key] = value
            with self.assertRaises(ValueError):
                bark.validate_config(changed)

    def test_embedded_contract_does_not_inherit_other_variants(self):
        original = (REPO / "model_specs/parakeet_tdt.json").read_bytes()
        spec = bark.package_spec(REPO)
        self.assertEqual(spec["family"], "parakeet_tdt")
        self.assertEqual(spec["languages"], ["en"])
        self.assertEqual(spec["packages"], [])
        self.assertNotIn("package_defaults", spec)
        self.assertEqual([s["format"] for s in spec["sources"]], ["gguf"])
        requests = {o["name"]: o for o in spec["options"]["request"]}
        self.assertNotIn("hotwords", requests)
        self.assertNotIn("vad", requests["audio_chunk_mode"]["values"])
        sessions = {o["name"]: o for o in spec["options"]["session"]}
        self.assertEqual(sessions["offline_mode"]["default"], "full_context")
        self.assertNotIn("vad_model_path", sessions)
        self.assertEqual((REPO / "model_specs/parakeet_tdt.json").read_bytes(), original)
        json.dumps(spec)  # Embedded metadata must be serializable.


if __name__ == "__main__":
    unittest.main()
