import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import gguf
import numpy as np
from safetensors.torch import save_file
from safetensors import SafetensorError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/lora"))
from bindings import BaseGGUF, BaseSources, Bindings
from merge_lora import main, normalize, read_adapter, sidecars
from checkpoint import EXTENSIONS, find_checkpoint, load_checkpoint


class MergeLoraTests(unittest.TestCase):
    def test_cosy_mixed_package_selection_and_sidecars(self):
        name = "model.layers.0.q_proj"
        bindings = Bindings("cosyvoice3", {
            "llm/llm.model." + name + ".weight": (4, 4),
            "llm/speech_embedding.weight": (4, 4),
            "llm/llm_decoder.weight": (4, 4),
        })
        native = {name + ".lora_A": torch.ones(2, 4), name + ".lora_B": torch.ones(4, 2)}
        peft = {name + ".lora_A.weight": torch.full((1, 4), 3.),
                name + ".lora_B.weight": torch.full((4, 1), 7.)}
        for extension in EXTENSIONS:
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps({"config": {"lora_r": 2, "lora_alpha": 4}}))
                (root / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}))
                for stem, state in (("lora_weights", native), ("adapter_model", peft),
                                    ("speech_embedding", {"weight": torch.full((4, 4), 5.)}),
                                    ("llm_decoder", {"weight": torch.full((4, 4), 6.)})):
                    file = root / (stem + extension)
                    if extension == ".safetensors":
                        save_file(state, file)
                    else:
                        torch.save(state, file)
                for path, is_native in ((root, True), (root / ("lora_weights" + extension), True),
                                        (root / ("adapter_model" + extension), False)):
                    loaded, config, _, native_config = read_adapter(path, "cosyvoice3", root, None)
                    self.assertEqual(native_config is not None, is_native)
                    output = {}
                    normalize(loaded, config, bindings, output, 1, {}, native_scaling=is_native)
                    target = "llm/llm.model." + name + ".weight"
                    self.assertEqual(output[target + ".alpha"].item(), 4 if is_native else 1)
                    expected_a = native[name + ".lora_A"] if is_native else peft[name + ".lora_A.weight"]
                    self.assertTrue(torch.equal(output[target + ".lora_A"], expected_a))
                    sidecars("cosyvoice3", root, native_config, None, bindings, output, root)
                    self.assertEqual("llm/speech_embedding.weight" in output, is_native)
                    self.assertEqual("llm/llm_decoder.weight" in output, is_native)
                (root / ("speech_embedding" + extension)).unlink()
                for path in (root, root / ("lora_weights" + extension)):
                    _, _, _, native_config = read_adapter(path, "cosyvoice3", root, None)
                    with self.assertRaises(FileNotFoundError):
                        sidecars("cosyvoice3", root, native_config, None, bindings, {}, root)

    def test_cosy_native_rank_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"config": {"lora_r": 3, "lora_alpha": 6}}))
            name = "model.layers.0.q_proj"
            save_file({name + ".lora_A": torch.ones(2, 4), name + ".lora_B": torch.ones(4, 2)}, root / "lora_weights.safetensors")
            loaded, config, _, native_config = read_adapter(root, "cosyvoice3", root, None)
            bindings = Bindings("cosyvoice3", {"llm/llm.model." + name + ".weight": (4, 4)})
            with self.assertRaisesRegex(ValueError, "Configured rank differs"):
                normalize(loaded, config, bindings, {}, 1, {}, native_scaling=native_config is not None)

    def test_adapter_layout_is_independent_of_container(self):
        for native in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if native:
                    (root / "config.json").write_text(json.dumps({"config": {"lora_r": 2, "lora_alpha": 4}}))
                else:
                    (root / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
                stem = "lora_weights" if native else "adapter_model"
                suffix = "" if native else ".weight"
                name = "model.layers.0.q_proj"
                state = {name + ".lora_A" + suffix: torch.arange(8.).reshape(2, 4),
                         name + ".lora_B" + suffix: torch.arange(8.).reshape(4, 2)}
                bindings = Bindings("cosyvoice3", {"llm/llm.model." + name + ".weight": (4, 4)})
                expected = None
                for extension in EXTENSIONS:
                    with self.subTest(native=native, extension=extension):
                        path = root / (stem + extension)
                        if extension == ".safetensors":
                            save_file(state, path)
                        else:
                            torch.save(state, path)
                        self.assertEqual(find_checkpoint(root, stem), path)
                        for supplied_path in (root, path):
                            loaded, config, _, native_config = read_adapter(supplied_path, "cosyvoice3", root, None)
                            self.assertEqual(native_config is not None, native)
                            output = {}
                            normalize(loaded, config, bindings, output, 1, {}, native_scaling=native)
                            if expected is None:
                                expected = output
                            self.assertEqual(output.keys(), expected.keys())
                            for key in output:
                                self.assertTrue(torch.equal(output[key], expected[key]), key)
                        path.unlink()

    def test_checkpoint_discovery_does_not_hide_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"weight": torch.ones(2)}, root / "adapter_model.pt")
            invalid = root / "adapter_model.safetensors"
            invalid.write_bytes(b"invalid")
            self.assertEqual(find_checkpoint(root, "adapter_model"), invalid)
            with self.assertRaises(SafetensorError):
                load_checkpoint(invalid)
            self.assertIsNone(find_checkpoint(root, "missing", required=False))
            with self.assertRaises(FileNotFoundError):
                find_checkpoint(root, "missing")

    def test_fish_training_layout_is_independent_of_container(self):
        state = {"model.layers.0.attention.wqkv.lora_A": torch.ones(2, 4),
                 "model.layers.0.attention.wqkv.lora_B": torch.ones(8, 2)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extension in EXTENSIONS:
                with self.subTest(extension=extension):
                    path = root / ("training" + extension)
                    if extension == ".safetensors":
                        save_file(state, path)
                    else:
                        torch.save({"state_dict": state, "epoch": 1}, path)
                    loaded, _, _, _ = read_adapter(path, "fish_audio", root, 4)
                    self.assertTrue(torch.equal(loaded["layers.0.attention.wqkv.lora_A"], state["model.layers.0.attention.wqkv.lora_A"]))
                    self.assertEqual(loaded["layers.0.attention.wqkv.alpha"].item(), 4)

    def test_prepared_fish_scaling_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "prepared.safetensors"
            state = {"layers.0.attention.wqkv.lora_A": torch.ones(2, 4),
                     "layers.0.attention.wqkv.lora_B": torch.ones(8, 2),
                     "layers.0.attention.wqkv.alpha": torch.tensor(4.)}
            save_file(state, path)
            loaded, _, _, _ = read_adapter(path, "fish_audio", root, 16)
            self.assertEqual(loaded.keys(), state.keys())
            for name in state:
                self.assertTrue(torch.equal(loaded[name], state[name]))

    def test_embedded_byte_array_is_read_in_bulk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "base.gguf"
            writer = gguf.GGUFWriter(file, "test")
            writer.add_array("audiocpp.embedded_files.names", ["vocab.txt"])
            writer.add_key_value("audiocpp.embedded_files.offsets", [0, 3],
                                 gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
            writer.add_array("audiocpp.embedded_files.data", b"ABC")
            writer.add_tensor("projection.weight", np.ones((2, 32), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            base = BaseGGUF(file)
            self.assertEqual(base.reader.fields["audiocpp.embedded_files.data"].contents(), [65, 66, 67])
            self.assertEqual(base.shapes["projection.weight"], (2, 32))
            self.assertTrue(torch.equal(base.read_f32("projection.weight"), torch.ones(2, 32)))
            destination = root / "sidecars"
            base.extract_sidecars(destination)
            self.assertEqual((destination / "vocab.txt").read_bytes(), b"ABC")

    def test_all_family_bindings(self):
        cases = [
            ("qwen3_tts", "model_weights/talker.model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("qwen3_asr", "model.language_model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("voxcpm2", "weights/base_lm.layers.0.self_attn.q_proj.weight", "base_lm.layers.0.self_attn.q_proj"),
            ("voxcpm1", "blk.0.attn_q.weight", "base_lm.layers.0.self_attn.q_proj"),
            ("fish_audio", "model_weights/fast_output.weight", "model.fast_output"),
            ("cosyvoice3", "llm/llm.model.model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("f5_tts", "transformer/ema_model.transformer.transformer_blocks.0.attn.to_q.weight", "transformer.transformer_blocks.0.attn.to_q"),
            ("outetts", "model_weights/model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj"),
            ("yue2", "model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj"),
            ("ace_step", "dit_turbo_weights/decoder.layers.0.q_proj.weight", "layers.0.q_proj"),
            ("vibevoice", "model.language_model.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q_proj"),
        ]
        for family, name, external in cases:
            with self.subTest(family=family):
                shapes = {name: (4, 4)}
                if family == "yue2":
                    shapes.update({name.replace("q_proj", part): (4, 4) for part in ("k_proj", "v_proj")})
                bindings = Bindings(family, shapes)
                self.assertEqual(bindings.resolve("base_model.model." + external)[0].name, name)

    def test_fused_fish_slices_keep_all_rows(self):
        shapes = {f"model_weights/layers.0.attention.{part}_proj.weight": (rows, 4)
                  for part, rows in (("q", 4), ("k", 2), ("v", 2))}
        bindings = Bindings("fish_audio", shapes)
        a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        b = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        state = {"model.layers.0.attention.wqkv.lora_A": a,
                 "model.layers.0.attention.wqkv.lora_B": b,
                 "model.layers.0.attention.wqkv.alpha": torch.tensor(4.)}
        output = {}
        normalize(state, {}, bindings, output, 1, {})
        offset = 0
        for part, rows in (("q", 4), ("k", 2), ("v", 2)):
            name = f"model_weights/layers.0.attention.{part}_proj.weight"
            self.assertTrue(torch.equal(output[name + ".lora_A"], a))
            self.assertTrue(torch.equal(output[name + ".lora_B"], b[offset:offset + rows]))
            self.assertEqual(output[name + ".alpha"].item(), 4)
            offset += rows
        with tempfile.TemporaryDirectory() as directory:
            save_file(output, Path(directory) / "normalized.safetensors")

    def test_fused_slices_reject_missing_and_excess_rows(self):
        for family, prefix, fused in (
            ("fish_audio", "model_weights/layers.0.attention", "model.layers.0.attention.wqkv"),
            ("yue2", "model.layers.0.self_attn", "layers.0.self_attn.qkv_proj"),
        ):
            shapes = {f"{prefix}.{part}_proj.weight": (rows, 4)
                      for part, rows in (("q", 4), ("k", 2), ("v", 2))}
            bindings = Bindings(family, shapes)
            for rows in (7, 9):
                with self.subTest(family=family, rows=rows):
                    state = {fused + ".lora_A": torch.ones(2, 4),
                             fused + ".lora_B": torch.ones(rows, 2),
                             fused + ".alpha": torch.tensor(4.)}
                    output = {}
                    with self.assertRaisesRegex(ValueError, "output row"):
                        normalize(state, {}, bindings, output, 1, {})
                    self.assertEqual(output, {})

    def test_mixed_lora_and_lokr_factors_are_rejected(self):
        name = "model.layers.0.q_proj"
        bindings = Bindings("qwen3_asr", {name + ".weight": (4, 4)})
        for factor in ("lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"):
            with self.subTest(factor=factor):
                state = {name + ".lora_A": torch.ones(2, 4),
                         name + ".lora_B": torch.ones(4, 2),
                         name + ".alpha": torch.tensor(2.),
                         name + "." + factor: torch.ones(2, 2)}
                output = {}
                with self.assertRaisesRegex(ValueError, "Mixed adapter algorithms"):
                    normalize(state, {}, bindings, output, 1, {})
                self.assertEqual(output, {})

    def test_peft_patterns_and_rslora_are_preserved(self):
        name = "model.layers.0.q_proj.weight"
        bindings = Bindings("qwen3_asr", {name: (4, 4)})
        source = "base_model.model.model.layers.0.q_proj"
        state = {source + ".lora_A.weight": torch.ones(2, 4), source + ".lora_B.weight": torch.ones(4, 2)}
        config = {"r": 8, "lora_alpha": 16, "rank_pattern": {"q_proj": 2}, "alpha_pattern": {"q_proj": 6}, "use_rslora": True}
        rs = {"rank_pattern": {}, "alpha_pattern": {}}
        output = {}
        normalize(state, config, bindings, output, 1, rs)
        self.assertEqual(list(rs["rank_pattern"].values()), [2])
        self.assertEqual(list(rs["alpha_pattern"].values()), [6])
        self.assertIn(name + ".lora_A.weight", output)

    def test_duplicate_aliases_cannot_overwrite_rslora(self):
        name = "model.layers.0.q_proj.weight"
        bindings = Bindings("qwen3_asr", {name: (4, 4)})
        source = "base_model.model.model.layers.0.q_proj"
        state = {source + ".lora_A.weight": torch.ones(2, 4),
                 source + ".lora_B.weight": torch.ones(4, 2)}
        config = {"r": 2, "lora_alpha": 2, "use_rslora": True}
        output = {}
        normalize(state, config, bindings, output, 1, {"rank_pattern": {}, "alpha_pattern": {}})
        with self.assertRaisesRegex(ValueError, "Multiple updates"):
            normalize(state, config, bindings, output, 1, {"rank_pattern": {}, "alpha_pattern": {}})

    def test_native_does_not_inherit_neighbor_peft_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "native.safetensors"
            save_file({"layers.0.self_attn.q_proj.lora_A": torch.ones(2, 4),
                       "layers.0.self_attn.q_proj.lora_B": torch.ones(4, 2)}, file)
            (root / "adapter_config.json").write_text(json.dumps({"r": 128, "lora_alpha": 256, "use_dora": True}))
            state, config, _, _ = read_adapter(file, "yue2", root, None)
            self.assertEqual(config, {})
            output = {}
            shapes = {f"model.layers.0.self_attn.{part}_proj.weight": (4, 4) for part in ("q", "k", "v")}
            normalize(state, config, Bindings("yue2", shapes),
                      output, 1, {}, require_scaling=False)
            self.assertEqual(output["model.layers.0.self_attn.q_proj.weight.alpha"].item(), 2)

    def test_missing_peft_metadata_and_empty_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "adapter.safetensors"
            save_file({"base_model.model.proj.lora_A.weight": torch.ones(2, 4),
                       "base_model.model.proj.lora_B.weight": torch.ones(4, 2)}, file)
            with self.assertRaisesRegex(ValueError, "adapter_config.json"):
                read_adapter(file, "qwen3_asr", root, None)
            save_file({}, file)
            with self.assertRaisesRegex(ValueError, "empty"):
                read_adapter(file, "qwen3_asr", root, None)

    def test_component_ambiguity_is_explicit(self):
        shapes = {f"{component}/decoder.proj.weight": (4, 4) for component in ("dit_base_weights", "dit_turbo_weights")}
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            Bindings("ace_step", shapes)
        bindings = Bindings("ace_step", shapes, "dit_turbo_weights")
        self.assertEqual(bindings.resolve("proj")[0].name, "dit_turbo_weights/decoder.proj.weight")

    def test_original_sharded_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"model.proj.weight": torch.ones(4, 4)}, root / "part.safetensors")
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"model.proj.weight": "part.safetensors"}}))
            base = BaseSources([f"model_weights={index}"], root)
            self.assertEqual(base.shapes, {"model_weights/model.proj.weight": (4, 4)})
            self.assertTrue(torch.equal(base.read_f32("model_weights/model.proj.weight"), torch.ones(4, 4)))
            original = (root / "part.safetensors").read_bytes()
            arguments = ["merge_lora.py", "--family", "qwen3_asr", "--input", str(index),
                         "--output", str(root / "part.safetensors"), "--overwrite", "--adapter", "unused"]
            with patch.object(sys, "argv", arguments), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((root / "part.safetensors").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
