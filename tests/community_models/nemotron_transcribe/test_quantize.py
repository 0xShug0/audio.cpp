"""Self-generated GGUF row-padding/type audits; optional stock-quantizer proof.

Run with uv and gguf/numpy. Set NEMOTRON_TEST_QUANTIZER to the pristine pinned
llama-quantize executable to include the three real K-quant integration cases.
No checkpoint, benchmark data, inference or network access is needed.
"""
from __future__ import annotations

import importlib.util
import base64
import json
import os
from pathlib import Path
import sys
import struct
import tempfile
import unittest

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("quantize_nemotron_transcribe",
    ROOT / "tools/community_models/quantize_nemotron_transcribe.py")
assert SPEC and SPEC.loader
quant = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quant
SPEC.loader.exec_module(quant)


def bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").view(np.uint8).reshape(*values.shape[:-1], values.shape[-1] * 2)


def metadata(writer):
    # Two-block Mamba/MoE hparams sufficient for the stock quantizer. There are
    # no real model tensors, sidecars or tokenizer assets in this fixture.
    writer.add_context_length(128)
    writer.add_embedding_length(128)
    writer.add_block_count(2)
    writer.add_feed_forward_length([0, 64])
    writer.add_head_count([0, 0])
    writer.add_head_count_kv([0, 0])
    writer.add_key_length(4)
    writer.add_value_length(4)
    writer.add_layer_norm_eps(1e-5)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_inner_size(8)
    writer.add_ssm_state_size(128)
    writer.add_ssm_time_step_rank(2)
    writer.add_ssm_group_count(2)
    writer.add_expert_count(4)
    writer.add_expert_used_count(2)
    writer.add_expert_shared_count(1)
    writer.add_expert_feed_forward_length(64)
    writer.add_expert_shared_feed_forward_length(128)
    writer.add_expert_weights_norm(True)
    writer.add_expert_weights_scale(2.5)
    writer.add_expert_group_count(1)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(["a", "b", "c", "d", "e", "f", "g", "h"])
    writer.add_string("nemo_speech.checkpoint_id", "self-generated-checkpoint")
    writer.add_string("nemo_speech.speech_model", "Encoder-F16.gguf")
    writer.add_array("test.strings", ["one", "日本語"])
    writer.add_key_value("test.integers", [1, 7], gguf.GGUFValueType.ARRAY,
        sub_type=gguf.GGUFValueType.INT32)


def fixture(path: Path, omit_down=False, wrong_shape=False, quantized=False):
    rng = np.random.default_rng(1938)
    writer = gguf.GGUFWriter(str(path), quant.ARCH)
    metadata(writer)
    values = {
        "blk.1.ffn_up_exps.weight": rng.normal(0, 0.05, (4, 64, 128)).astype(np.float32),
        "blk.1.ffn_up_shexp.weight": rng.normal(0, 0.05, (128, 128)).astype(np.float32),
        "token_embd.weight": rng.normal(0, 0.05, (8, 128)).astype(np.float32),
        "output.weight": rng.normal(0, 0.05, (8, 128)).astype(np.float32),
    }
    if not omit_down:
        values["blk.1.ffn_down_exps.weight"] = rng.normal(0, 0.05,
            (3 if wrong_shape else 4, 128, 64)).astype(np.float32)
    payloads = {}
    for name, value in values.items():
        data = bf16(value)
        payloads[name] = data.copy()
        writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("output_norm.weight", np.ones(128, np.float32))
    if quantized:
        data = np.zeros(gguf.quant_shape_to_byte_shape((2, 128), gguf.GGMLQuantizationType.Q8_0), np.uint8)
        writer.add_tensor("test.quantized.weight", data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return payloads


def package_fixture(root: Path):
    """Valid helper-level package metadata with self-generated zero embeddings.

    The real tokenizer validator requires 131072 entries. At hidden128 these
    two BF16 tables total67MB, still independent of any real checkpoint.
    """
    helpers = quant.package_helpers()
    source = root / "checkpoint"
    for relative in helpers.SIDECARS:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    backbone = dict(hidden_size=128, vocab_size=131072, num_attention_heads=2, head_dim=4,
        moe_intermediate_size=64, moe_shared_expert_intermediate_size=128, n_shared_experts=1,
        n_routed_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1, conv_kernel=4,
        ssm_state_size=128, n_groups=2, mamba_num_heads=2, mamba_head_dim=4, norm_eps=1e-5,
        layer_norm_epsilon=1e-5, routed_scaling_factor=2.5, norm_topk_prob=True,
        num_key_value_heads=1, layers_block_type=["mamba", "moe"])
    (source / "llm_backbone/config.json").write_text(json.dumps(backbone), encoding="utf-8")
    (source / "generation_config.json").write_text('{"eos_token_id":[11]}', encoding="utf-8")
    header = json.dumps({"tiny": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        helpers.WINDOW_KEY: {"dtype": "BF16", "shape": [400], "data_offsets": [4, 804]},
        helpers.DIAR_WINDOW_KEY: {"dtype": "BF16", "shape": [400], "data_offsets": [804, 1604]}}).encode()
    (source / "model.safetensors").write_bytes(struct.pack("<Q", len(header))+header+struct.pack("<f", 1.0)+
        np.full(800, 0x3f00, dtype="<u2").tobytes())
    identity = helpers.source_checkpoint_identity(source)
    tokens = [f"self-generated-{i}" for i in range(131072)]
    (source / "tokenizer.json").write_text(json.dumps({"model": {
        "vocab": {text: index for index, text in enumerate(tokens)}, "merges": [], "type": "BPE",
        "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None,
        "fuse_unk": False, "byte_fallback": False, "ignore_merges": True}, "normalizer": None,
        "pre_tokenizer": helpers.PRE_TOKENIZER,
        "added_tokens": [{"id": 131072, "content": "<|audio|>", "special": True}]}), encoding="utf-8")
    (source / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>",
        "clean_up_tokenization_spaces": False}), encoding="utf-8")
    encoder = root / "Encoder-F16.gguf"
    writer = gguf.GGUFWriter(str(encoder), "pee")
    writer.add_string("nemo_speech.checkpoint_id", identity)
    writer.add_string("ctc.tokenizer.spm_model", base64.b64encode(b"binary\0tokenizer\xff").decode())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    native = root / "decoder-bf16.gguf"
    writer = gguf.GGUFWriter(str(native), quant.ARCH)
    writer.add_string("nemo_speech.checkpoint_id", identity)
    writer.add_string("nemo_speech.speech_model", encoder.name)
    for key, suffix in helpers.BACKBONE_FIELDS.items():
        writer.add_uint32(quant.ARCH + "." + suffix, backbone[key])
    writer.add_uint32(quant.ARCH + ".block_count", 2)
    writer.add_uint32(quant.ARCH + ".context_length", 128)
    writer.add_array(quant.ARCH + ".attention.head_count_kv", [0, 0])
    writer.add_array(quant.ARCH + ".feed_forward_length", [0, 64])
    writer.add_float32(quant.ARCH + ".attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32(quant.ARCH + ".attention.layer_norm_epsilon", 1e-5)
    writer.add_float32(quant.ARCH + ".expert_weights_scale", 2.5)
    writer.add_uint32(quant.ARCH + ".attention.value_length", 4)
    writer.add_uint32(quant.ARCH + ".ssm.inner_size", 8)
    writer.add_bool(quant.ARCH + ".expert_weights_norm", True)
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([1]*len(tokens))
    # Stale framework overrides must not survive the new physical expert layout.
    writer.add_string("audiocpp.tensor_shapes", "stale-self-generated-shapes")
    writer.add_string("audiocpp.tensor_ranks", "stale-self-generated-ranks")
    writer.add_tensor("blk.1.ffn_up_exps.weight", np.zeros((4, 64, 256), np.uint8), raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("blk.1.ffn_down_exps.weight", np.zeros((4, 128, 128), np.uint8), raw_dtype=gguf.GGMLQuantizationType.BF16)
    for name in ("token_embd.weight", "output.weight"):
        writer.add_tensor(name, np.zeros((len(tokens), 256), np.uint8), raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor("output_norm.weight", np.ones(128, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return native, encoder, source


class PaddingTest(unittest.TestCase):
    def test_only_compatible_k_quant_ids_are_exposed(self):
        self.assertEqual({name: int(kind) for name, kind in quant.PROFILES.items()},
            {"q4_k": 12, "q3_k": 11, "q2_k": 10})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                quant.convert(root / "source.gguf", Path(sys.executable), root / "output.gguf", "q2_0")

    def test_embedded_package_cache_roundtrip_and_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, encoder, source = package_fixture(root)
            spec = ROOT / "model_specs/nemotron_transcribe.json"
            with quant.mapped_reader(native) as reader:
                package = quant.package_payload(source, encoder, spec, reader)
            wrong_encoder = root / "wrong-encoder.gguf"
            writer = gguf.GGUFWriter(str(wrong_encoder), "pee")
            writer.add_string("nemo_speech.checkpoint_id", "different-checkpoint")
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.close()
            with quant.mapped_reader(native) as reader, self.assertRaisesRegex(ValueError, "checkpoint IDs differ"):
                quant.package_payload(source, wrong_encoder, spec, reader)
            cache = root / "padded.gguf"
            plans, _ = quant.padded_cache(native, cache, package)
            with quant.mapped_reader(cache) as reader:
                self.assertEqual(quant.field(reader, "audiocpp.model_spec.family"), "nemotron_transcribe")
                self.assertEqual(quant.field(reader, "audiocpp.model_spec.version"), 1)
                self.assertEqual(json.loads(quant.field(reader, "audiocpp.model_spec.json"))["family"], "nemotron_transcribe")
                files = dict(zip(quant.field(reader, "audiocpp.embedded_files.names"), quant.field(reader, "audiocpp.embedded_files.contents")))
                self.assertEqual(json.loads(files["frontend_window.json"]), [0.5]*400)
                self.assertEqual(files["generation_config.json"], '{"eos_token_id":[11]}')
                self.assertNotIn("audiocpp.tensor_shapes", reader.fields)
                self.assertNotIn("audiocpp.tensor_ranks", reader.fields)
            output_folder = root / "package"
            output_folder.mkdir()
            sibling = quant.ensure_encoder_sibling(encoder, output_folder, package)
            before = quant.stat_identity(sibling)
            self.assertEqual(quant.ensure_encoder_sibling(encoder, output_folder, package), sibling)
            self.assertEqual(quant.stat_identity(sibling), before)
            (source / "generation_config.json").write_text('{"eos_token_id":[11],"max_new_tokens":8}', encoding="utf-8")
            with quant.mapped_reader(native) as reader:
                changed = quant.package_payload(source, encoder, spec, reader)
            with self.assertRaisesRegex(ValueError, "changed"):
                quant.padded_cache(native, cache, changed)
            with self.assertRaises(ValueError):
                quant.padded_cache(native, cache)
            if os.environ.get("NEMOTRON_TEST_QUANTIZER"):
                # Restore matching package inputs and prove that stock quantizer
                # metadata preservation removes the post-quant packaging copy.
                (source / "generation_config.json").write_text('{"eos_token_id":[11]}', encoding="utf-8")
                report = quant.convert(native, Path(os.environ["NEMOTRON_TEST_QUANTIZER"]),
                    output_folder / "q3.gguf", "q3_k", threads=2, cache=cache,
                    package_source=source, encoder=encoder, model_spec=spec)
                self.assertTrue(report["cache"]["reused"])
                with quant.mapped_reader(output_folder / "q3.gguf") as reader:
                    self.assertEqual(quant.field(reader, "audiocpp.embedded_files.contents"), package["contents"])

    def test_streamed_rows_metadata_and_single_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "source.gguf", root / "padded.gguf"
            payloads = fixture(source)
            plans, receipt = quant.padded_cache(source, cache)
            self.assertFalse(receipt["reused"])
            stamp = quant.stat_identity(cache)
            with quant.mapped_reader(source) as before, quant.mapped_reader(cache) as after:
                self.assertEqual(after.fields[quant.ALIGNMENT_KEY].types, [gguf.GGUFValueType.UINT32])
                self.assertEqual(quant.field(after, quant.ALIGNMENT_KEY), 256)
                for name in ("test.strings", "test.integers", "nemo_speech.checkpoint_id", "nemo_speech.speech_model",
                             "nemotron_h_moe.embedding_length", "nemotron_h_moe.expert_feed_forward_length"):
                    self.assertEqual(before.fields[name].types, after.fields[name].types)
                    self.assertEqual(quant.field(before, name), quant.field(after, name))
                by_name = {t.name: t for t in after.tensors}
                for name in ("blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight"):
                    original = payloads[name]
                    padded = by_name[name].data
                    self.assertEqual(tuple(by_name[name].shape)[0], 256)
                    np.testing.assert_array_equal(padded[..., :original.shape[-1]], original)
                    self.assertTrue(np.all(padded[..., original.shape[-1]:] == 0))
                self.assertEqual(tuple(by_name["blk.1.ffn_up_exps.weight"].shape), (256, 64, 4))
                self.assertEqual(tuple(by_name["blk.1.ffn_down_exps.weight"].shape), (256, 128, 4))
                for tensor in before.tensors:
                    if "_exps." not in tensor.name:
                        self.assertEqual(tensor.data.tobytes(), by_name[tensor.name].data.tobytes())
            _, second = quant.padded_cache(source, cache)
            self.assertTrue(second["reused"])
            self.assertEqual(stamp, quant.stat_identity(cache))
            self.assertTrue(all(p.category != "routed" or p.padded_shape[1:] == p.shape[1:] for p in plans))

    def test_rejects_quantized_input_missing_pair_and_wrong_shapes(self):
        for options in ({"quantized": True}, {"omit_down": True}, {"wrong_shape": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root / "source.gguf", **options)
                with self.assertRaises(ValueError):
                    quant.padded_cache(root / "source.gguf", root / "cache.gguf")
                self.assertFalse((root / "cache.gguf").exists())

    def test_preserves_unverified_partial_cache_and_modified_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "source.gguf", root / "cache.gguf"
            fixture(source)
            partial = cache.with_name(cache.name + ".padding")
            partial.write_bytes(b"uninspected partial")
            with self.assertRaises(FileExistsError):
                quant.padded_cache(source, cache)
            self.assertEqual(partial.read_bytes(), b"uninspected partial")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "source.gguf", root / "cache.gguf"
            fixture(source)
            quant.padded_cache(source, cache)
            with cache.open("ab") as stream:
                stream.write(b"changed")
            changed = cache.read_bytes()
            with self.assertRaises(ValueError):
                quant.padded_cache(source, cache)
            self.assertEqual(cache.read_bytes(), changed)

    def test_actual_type_audit_rejects_unquantized_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / "source.gguf")
            plans, _ = quant.padded_cache(root / "source.gguf", root / "cache.gguf")
            with self.assertRaisesRegex(RuntimeError, "fallback"):
                quant.audit_output(root / "cache.gguf", root / "cache.gguf", plans, "q2_k")

    @unittest.skipUnless(os.environ.get("NEMOTRON_TEST_QUANTIZER"), "explicit pristine stock quantizer not provided")
    def test_stock_q4_q3_q2_types_and_reused_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "source.gguf", root / "cache.gguf"
            fixture(source)
            for index, profile in enumerate(quant.PROFILES):
                with self.subTest(profile=profile):
                    output = root / f"{profile}.gguf"
                    report = quant.convert(source, Path(os.environ["NEMOTRON_TEST_QUANTIZER"]), output,
                        profile, threads=2, cache=cache)
                    self.assertEqual(report["cache"]["reused"], index > 0)
                    self.assertEqual(report["counts"][quant.PROFILES[profile].name], 2)
                    self.assertEqual(report["counts"]["Q8_0"], 1)
                    self.assertEqual(report["counts"]["BF16"], 2)
                    self.assertEqual(report["counts"]["F32"], 1)
                    receipt = output.with_name(output.name + ".quantization.json")
                    self.assertEqual(json.loads(receipt.read_text())["profile"], profile)
                    existing = output.read_bytes()
                    with self.assertRaises(FileExistsError):
                        quant.convert(source, Path(os.environ["NEMOTRON_TEST_QUANTIZER"]), output,
                            profile, threads=2, cache=cache)
                    self.assertEqual(output.read_bytes(), existing)


if __name__ == "__main__":
    unittest.main()
