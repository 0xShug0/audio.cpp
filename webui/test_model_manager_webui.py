"""The WebUI downloader must install files where required_files says they are.

model_manager_webui.py overrides the upstream download path with a resumable,
Torch-free one. list_hf_files() yields (repo path, installed path, size) and those
two differ for every package with a strip_prefix -- which is all the GGUF packages
in audio-cpp/audio.cpp-gguf, since they live in a repo subdirectory
(Vevo2-GGUF/vevo2-q8_0.gguf) that must not appear in the installed model directory.
Using the repo path as the destination installed them one level too deep, and the
install failed validation with "missing required files" (issue #113).

hf_hub_download is stubbed here: the point is where bytes land, not the transfer.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for path in (HERE, os.path.join(REPO_ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import model_manager as mm  # noqa: E402
import model_manager_webui as mmw  # noqa: E402


class _Source:
    repo_id = "audio-cpp/audio.cpp-gguf"
    revision = "main"


class _RecordingHub:
    """Stands in for hf_hub_download: writes `filename` under local_dir, as the real
    client does, and records what it was asked for."""

    def __init__(self, payload=b"gguf"):
        self.payload = payload
        self.requested = []

    def __call__(self, repo_id, filename, revision, local_dir, token):
        self.requested.append(filename)
        written = Path(local_dir) / filename
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(self.payload)
        return str(written)


class StripPrefixPlacementTests(unittest.TestCase):
    def setUp(self):
        self.hub = _RecordingHub()
        original = mmw.hf_hub_download
        mmw.hf_hub_download = self.hub
        self.addCleanup(setattr, mmw, "hf_hub_download", original)

    def _install(self, tmp, files):
        original = mm.list_hf_files
        mm.list_hf_files = lambda source: files
        self.addCleanup(setattr, mm, "list_hf_files", original)
        required = [local for _remote, local, _size in files]
        mmw.install_snapshot_into_dir(_Source(), Path(tmp), required)

    def test_strip_prefix_file_lands_at_its_installed_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._install(tmp, [("Vevo2-GGUF/vevo2-q8_0.gguf", "vevo2-q8_0.gguf", 4)])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "vevo2-q8_0.gguf")))
            # The repo subdirectory must not survive into the installed package.
            self.assertFalse(os.path.exists(os.path.join(tmp, "Vevo2-GGUF")))
            self.assertEqual(self.hub.requested, ["Vevo2-GGUF/vevo2-q8_0.gguf"],
                             "the repo path is what must be requested from the hub")

    def test_nested_paths_below_the_stripped_prefix_are_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._install(tmp, [("Pkg-GGUF/speech_tokenizer/config.json",
                                 "speech_tokenizer/config.json", 4)])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "speech_tokenizer", "config.json")))
            self.assertFalse(os.path.exists(os.path.join(tmp, "Pkg-GGUF")))

    def test_packages_without_a_strip_prefix_are_unchanged(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._install(tmp, [("config.json", "config.json", 4),
                                ("speech_tokenizer/model.safetensors",
                                 "speech_tokenizer/model.safetensors", 4)])
            self.assertTrue(os.path.isfile(os.path.join(tmp, "config.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp, "speech_tokenizer", "model.safetensors")))

    def test_a_complete_file_is_not_downloaded_again(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "vevo2-q8_0.gguf").write_bytes(b"gguf")
            self._install(tmp, [("Vevo2-GGUF/vevo2-q8_0.gguf", "vevo2-q8_0.gguf", 4)])
            self.assertEqual(self.hub.requested, [],
                             "an already-complete file at its installed path was re-downloaded")


class EveryStripPrefixPackageIsCoveredTests(unittest.TestCase):
    def test_gguf_repo_packages_declare_stripped_required_files(self):
        # Guards the invariant the placement fix restores: required_files is written
        # against the stripped path, so a downloader that keeps the repo prefix fails.
        catalog = mm.CATALOG
        packages = catalog.values() if isinstance(catalog, dict) else catalog
        checked = 0
        for package in packages:
            source = package.source
            if not isinstance(source, mm.SnapshotSource) or not source.strip_prefix:
                continue
            checked += 1
            for required in package.required_files:
                self.assertFalse(required.startswith(source.strip_prefix),
                                 f"{package.id} required_files still carries the strip_prefix")
        self.assertGreater(checked, 0, "no strip_prefix packages found to check")


if __name__ == "__main__":
    unittest.main()
