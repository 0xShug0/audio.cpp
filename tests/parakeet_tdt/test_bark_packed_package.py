import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/community_models"))
import prepare_bark_packed as packed


class PackedPackageTests(unittest.TestCase):
    def test_native_contract_is_english_and_preserves_source_catalog(self):
        original = (REPO / "model_specs/parakeet_tdt.json").read_bytes()
        for profile in packed.PROFILES.values():
            spec = packed.native_package_spec(REPO, profile)
            self.assertEqual(spec["display_name"], profile.display_name)
            self.assertEqual(spec["family"], "parakeet_tdt")
            self.assertEqual(spec["languages"], ["en"])
            self.assertEqual(spec["sources"][0]["format"], "bark")
            self.assertEqual(spec["sources"][0]["tensors"], {"weights": "model:model.bark"})
            self.assertEqual(spec["packages"], [])
            self.assertNotIn("package_defaults", spec)
            self.assertEqual(spec["ui"]["docs"], ["docs/models/bark_packed_loading.md"])
            offline = next(o for o in spec["options"]["session"] if o["name"] == "offline_mode")
            self.assertEqual(offline["default"], "full_context")
        self.assertEqual((REPO / "model_specs/parakeet_tdt.json").read_bytes(), original)

    def test_checkpoint_profiles_are_separate_and_immutable(self):
        standard, plus = packed.PROFILES["standard"], packed.PROFILES["plus"]
        self.assertNotEqual(standard.source_sha256, plus.source_sha256)
        self.assertNotEqual(standard.variant, plus.variant)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            standard.variant = plus.variant
        with self.assertRaises(TypeError):
            packed.PROFILES["standard"] = plus

    def test_wrong_checkpoint_does_not_publish_a_package(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "model.bark").write_bytes(b"not the pinned checkpoint")
            target = Path(directory) / "package"
            for profile in packed.PROFILES.values():
                with self.assertRaises(ValueError):
                    packed.prepare_native_package(source, target, profile)
                self.assertFalse(target.exists())

    def test_existing_package_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "package"
            target.mkdir()
            sentinel = target / "keep"
            sentinel.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                packed.prepare_native_package(Path(directory) / "missing", target, packed.PROFILES["standard"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
