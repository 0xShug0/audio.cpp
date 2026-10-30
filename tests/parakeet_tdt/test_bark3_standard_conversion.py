"""Standard Bark source identity and shared-converter isolation checks."""
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/community_models"))
import convert_underdog_bark3_plus as packed
import convert_underdog_bark3_standard as standard


class BarkStandardConversionTest(unittest.TestCase):
    def test_profiles_are_immutable_and_do_not_contaminate_plus(self):
        original = packed.package_spec(REPO)
        spec = packed.package_spec(REPO, standard.STANDARD_PROFILE)
        self.assertEqual(spec["display_name"], "Underdog Bark 3.0")
        self.assertEqual(spec["family"], "parakeet_tdt")
        self.assertEqual(spec["languages"], ["en"])
        self.assertEqual(spec["packages"], [])
        self.assertEqual(spec["ui"]["docs"], ["docs/models/underdog_bark3.md"])
        self.assertEqual(packed.package_spec(REPO), original)
        self.assertEqual(packed.VARIANT, "underdog_bark3_plus")
        self.assertEqual(packed.PLUS_PROFILE.bf16_f32_prefixes, ())
        self.assertTrue(standard.STANDARD_PROFILE.bf16_f32_prefixes)
        with self.assertRaises(FrozenInstanceError):
            standard.STANDARD_PROFILE.variant = "underdog_bark3_plus"

    def test_cross_model_staging_is_rejected_before_packing(self):
        for profile, other in [(standard.STANDARD_PROFILE, packed.PLUS_PROFILE),
                               (packed.PLUS_PROFILE, standard.STANDARD_PROFILE)]:
            with tempfile.TemporaryDirectory() as directory:
                staging = Path(directory)
                metadata = dict(source=other.source_repo, revision=other.source_revision,
                                model_sha256=other.source_sha256, variant=other.variant)
                (staging / "conversion.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "different source/variant"):
                    packed.pack(staging, staging / "missing-tool", staging / "out.gguf", "mixed-f16", profile=profile)

    def test_unpinned_standard_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.bark").write_bytes(b"wrong checkpoint")
            with self.assertRaisesRegex(ValueError, "pinned revision"):
                packed.prepare(root, root / "staging", standard.STANDARD_PROFILE)


if __name__ == "__main__":
    unittest.main()
