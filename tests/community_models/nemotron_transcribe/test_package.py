"""Self-generated GGUF package roundtrip. Run with uv and gguf/numpy installed."""
from __future__ import annotations

import base64
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODULE = importlib.util.spec_from_file_location("prepare_nemotron_transcribe",
    ROOT / "tools/community_models/prepare_nemotron_transcribe.py")
assert MODULE and MODULE.loader
package = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(package)


class PackageTest(unittest.TestCase):
    def test_preserves_mixed_payload_and_embeds_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            for relative in package.SIDECARS:
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            backbone = {"hidden_size": 2688, "vocab_size": 131072, "num_attention_heads": 32,
                "head_dim": 128, "moe_intermediate_size": 1856, "moe_shared_expert_intermediate_size": 3712,
                "n_shared_experts": 1, "n_routed_experts": 128, "num_experts_per_tok": 6,
                "n_group": 1, "topk_group": 1, "conv_kernel": 4, "ssm_state_size": 128,
                "n_groups": 8, "mamba_num_heads": 64, "mamba_head_dim": 64,
                "norm_eps": 1e-5, "layer_norm_epsilon": 1e-5, "routed_scaling_factor": 2.5,
                "norm_topk_prob": True, "num_key_value_heads": 2,
                "layers_block_type": ["mamba", "moe", "attention"]}
            (source / "llm_backbone/config.json").write_text(json.dumps(backbone))
            (source / "generation_config.json").write_text('{"eos_token_id":[11]}')
            header = json.dumps({"tiny": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                package.WINDOW_KEY: {"dtype": "BF16", "shape": [400], "data_offsets": [4, 804]}}).encode()
            (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header +
                struct.pack("<f", 1.0) + np.full(400, 0x3f00, dtype="<u2").tobytes())
            identity = package.source_checkpoint_identity(source)
            tokens = [f"self-generated-{i}" for i in range(131072)]
            (source / "tokenizer.json").write_text(json.dumps({"model": {
                "vocab": {text: index for index, text in enumerate(tokens)}, "merges": [],
                "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True},
                "normalizer": None, "pre_tokenizer": package.PRE_TOKENIZER,
                "added_tokens": [{"id": 131072, "content": "<|audio|>", "special": True}]}))
            (source / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>",
                "clean_up_tokenization_spaces": False}))
            encoder = root / "Encoder.gguf"
            writer = gguf.GGUFWriter(str(encoder), "pee")
            writer.add_string("nemo_speech.checkpoint_id", identity)
            tokenizer = b"binary\0tokenizer\xff"
            writer.add_string("ctc.tokenizer.spm_model", base64.b64encode(tokenizer).decode())
            writer.add_tensor("test", np.zeros((1,), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            decoder = root / "Decoder.gguf"
            writer = gguf.GGUFWriter(str(decoder), "nemotron_h_moe")
            writer.add_string("nemo_speech.checkpoint_id", identity)
            writer.add_string("nemo_speech.speech_model", encoder.name)
            writer.add_array("tokenizer.ggml.tokens", tokens)
            writer.add_array("tokenizer.ggml.token_type", [1] * len(tokens))
            for source_key, native_key in package.BACKBONE_FIELDS.items():
                writer.add_uint32("nemotron_h_moe." + native_key, backbone[source_key])
            for key, value in (("attention.layer_norm_rms_epsilon", 1e-5), ("attention.layer_norm_epsilon", 1e-5),
                               ("expert_weights_scale", 2.5)):
                writer.add_float32("nemotron_h_moe." + key, value)
            writer.add_uint32("nemotron_h_moe.attention.value_length", 128)
            writer.add_uint32("nemotron_h_moe.ssm.inner_size", 4096)
            writer.add_bool("nemotron_h_moe.expert_weights_norm", True)
            writer.add_uint32("nemotron_h_moe.block_count", 4)
            writer.add_array("nemotron_h_moe.attention.head_count_kv", [0, 0, 2, 2])
            writer.add_array("nemotron_h_moe.feed_forward_length", [0, 1856, 0, 1856])
            writer.add_array("test.strings", ["one", "日本語"])
            writer.add_key_value("test.integers", [1, 7], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT32)
            payloads = {}
            for name, shape, dtype, qtype in (
                ("f32", (3, 8), np.float32, gguf.GGMLQuantizationType.F32),
                ("f16", (3, 8), np.float16, gguf.GGMLQuantizationType.F16),
                ("iq4", (2, 256), np.uint8, gguf.GGMLQuantizationType.IQ4_NL),
                ("q8", (2, 256), np.uint8, gguf.GGMLQuantizationType.Q8_0),
                ("bf16", (3, 8), np.uint8, gguf.GGMLQuantizationType.BF16),
            ):
                raw_shape = gguf.quant_shape_to_byte_shape(shape, qtype) if dtype == np.uint8 else shape
                data = np.arange(np.prod(raw_shape), dtype=dtype).reshape(raw_shape)
                payloads[name] = data.tobytes()
                writer.add_tensor(name, data, raw_dtype=qtype)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            output = root / "package" / "Model.gguf"
            spec = ROOT / "model_specs/nemotron_transcribe.json"
            receipt = package.prepare(decoder, encoder, source, output, spec)
            self.assertEqual(receipt["tensor_count"], 5)
            reader = gguf.GGUFReader(str(output))
            self.assertEqual(package.field(reader, "audiocpp.model_spec.family"), "nemotron_transcribe")
            self.assertEqual(package.field(reader, "test.strings"), ["one", "日本語"])
            self.assertEqual(package.field(reader, "test.integers"), [1, 7])
            names = package.field(reader, "audiocpp.embedded_files.names")
            contents = package.field(reader, "audiocpp.embedded_files.contents")
            self.assertEqual(dict(zip(names, contents))["generation_config.json"], '{"eos_token_id":[11]}')
            self.assertEqual(json.loads(dict(zip(names, contents))["frontend_window.json"]), [0.5] * 400)
            self.assertNotIn("alignment.model", names)
            with package.mapped_reader(output.parent / encoder.name) as packed_encoder:
                self.assertEqual(base64.b64decode(package.field(packed_encoder, "ctc.tokenizer.spm_model")), tokenizer)
            for tensor in reader.tensors:
                self.assertEqual(tensor.data.tobytes(), payloads[tensor.name])
            reader.data._mmap.close()
            with self.assertRaisesRegex(ValueError, "already exists"):
                package.prepare(decoder, encoder, source, output, spec)
            # A mixed checkpoint pair must fail before creating an output.
            bad = root / "wrong-encoder.gguf"
            writer = gguf.GGUFWriter(str(bad), "pee")
            writer.add_string("nemo_speech.checkpoint_id", "another-checkpoint")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.close()
            with self.assertRaisesRegex(ValueError, "checkpoint IDs differ"):
                package.prepare(decoder, bad, source, root / "bad.gguf", spec)
            self.assertFalse((root / "bad.gguf").exists())


if __name__ == "__main__":
    unittest.main()
