"""The WebUI-local model manager installs packages directly from model_specs/.

The WebUI should be relatively self-contained for downloads: no legacy
tools/model_manager.py catalog, no conversion path, and no safetensors package
bridge. model_manager_webui.py is the small spec-v1 downloader copied next to the
WebUI, so these tests cover the package selection and final on-disk layout.
"""
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model_manager_webui as mmw  # noqa: E402


def _package(**overrides):
    values = {
        "family": "demo",
        "id": "demo_q8_0",
        "display_name": "Demo Q8_0",
        "target_directory": "Demo-GGUF",
        "format": "gguf",
        "precision": "q8_0",
        "files": ("Demo-GGUF/model-q8_0.gguf",),
        "strip_prefix": "Demo-GGUF/",
        "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/audio.cpp-gguf"},
        "default": True,
    }
    values.update(overrides)
    return mmw.PackageRecord(**values)


class SpecPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = mmw.flatten_packages(mmw.load_specs(mmw.DEFAULT_SPECS_DIR))
        cls.by_id = {record.id: record for record in cls.records}

    def test_webui_manager_reads_default_spec_packages(self):
        voxcpm2 = self.by_id["voxcpm2_q8_0"]
        self.assertEqual(voxcpm2.family, "voxcpm2")
        self.assertEqual(voxcpm2.format, "gguf")
        self.assertEqual(voxcpm2.target_directory, "VoxCPM2-GGUF")
        self.assertEqual(voxcpm2.download["repo"], "audio-cpp/audio.cpp-gguf")

        inflect = self.by_id["inflect_micro_v2_orig"]
        self.assertEqual(inflect.family, "inflect_v2")
        self.assertEqual(inflect.format, "gguf")
        self.assertEqual(inflect.target_directory, "Inflect-Micro-v2-GGUF")

    def test_selecting_a_family_uses_its_default_package(self):
        selected = mmw.select_package(self.records, SimpleNamespace(
            package="voxcpm2", format=None, precision=None))
        self.assertEqual(selected.id, "voxcpm2_q8_0")

    def test_non_huggingface_package_is_rejected(self):
        package = _package(download={"kind": "local"})
        with self.assertRaises(mmw.ManagerError):
            mmw.ensure_hf_package(package)

    def test_package_size_record_sums_metadata_without_downloading(self):
        original = mmw.check_remote_file
        self.addCleanup(setattr, mmw, "check_remote_file", original)
        mmw.check_remote_file = lambda _package, remote: 10 if remote == "a" else 15
        row = mmw.package_size_record(_package(files=("a", "b")))
        self.assertEqual(row["state"], "ok")
        self.assertEqual(row["size_bytes"], 25)

    def test_gated_package_size_is_reported_without_failing_the_scan(self):
        original = mmw.check_remote_file
        self.addCleanup(setattr, mmw, "check_remote_file", original)
        mmw.check_remote_file = lambda _package, _remote: None
        row = mmw.package_size_record(_package(download={
            "kind": "huggingface_snapshot", "repo": "gated/demo", "gated": True}))
        self.assertEqual(row["state"], "gated")
        self.assertIsNone(row["size_bytes"])

    def test_package_status_requires_every_file_before_reporting_installed(self):
        root = Path(tempfile.mkdtemp(prefix="audiocpp_package_status_test_"))
        self.addCleanup(shutil.rmtree, root, True)
        package = _package(files=("Demo-GGUF/model-q8_0.gguf", "Demo-GGUF/config.json"))
        target = root / "Demo-GGUF"
        target.mkdir()
        (target / "model-q8_0.gguf").write_bytes(b"gguf")
        original = mmw.check_remote_file
        self.addCleanup(setattr, mmw, "check_remote_file", original)
        mmw.check_remote_file = lambda _package, _remote: 4
        self.assertFalse(mmw.package_size_record(package, root)["installed"])
        (target / "config.json").write_text("{}", encoding="utf-8")
        self.assertTrue(mmw.package_size_record(package, root)["installed"])


class InstallPlacementTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="audiocpp_webui_manager_test_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.calls = []
        original = mmw.download_file
        self.addCleanup(setattr, mmw, "download_file", original)

        def fake_download(package, remote_path, output_path, progress=None):
            self.calls.append((remote_path, output_path.name))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"gguf")
            if progress is not None:
                progress(4, 4)

        mmw.download_file = fake_download

    def _install(self, package, overwrite=False, progress=False):
        args = SimpleNamespace(
            models_root=self.root,
            overwrite=overwrite,
            check=False,
            dry_run=False,
            progress=progress,
        )
        mmw.install_package(package, args)

    def _uninstall(self, package):
        mmw.uninstall_package(package, SimpleNamespace(models_root=self.root))

    def test_strip_prefix_file_lands_at_its_installed_path(self):
        self._install(_package())
        self.assertTrue(os.path.isfile(os.path.join(self.root, "Demo-GGUF", "model-q8_0.gguf")))
        self.assertEqual(self.calls, [
            ("Demo-GGUF/model-q8_0.gguf", "model-q8_0.gguf"),
        ])

    def test_nested_paths_below_the_stripped_prefix_are_preserved(self):
        package = _package(files=("Demo-GGUF/tokenizer/config.json",))
        self._install(package)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "Demo-GGUF", "tokenizer", "config.json")))

    def test_nested_target_directory_creates_its_parent_before_atomic_rename(self):
        package = _package(
            target_directory="Demo-GGUF/english",
            files=("Demo-GGUF/english/model-q8_0.gguf",),
            strip_prefix="Demo-GGUF/english")
        self._install(package)
        self.assertTrue(os.path.isfile(os.path.join(
            self.root, "Demo-GGUF", "english", "model-q8_0.gguf")))

    def test_packages_without_a_strip_prefix_are_unchanged(self):
        package = _package(files=("config.json",), strip_prefix="")
        self._install(package)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "Demo-GGUF", "config.json")))

    def test_existing_target_requires_overwrite(self):
        target = Path(self.root) / "Demo-GGUF"
        target.mkdir()
        (target / "model-q8_0.gguf").write_bytes(b"old")
        with self.assertRaises(mmw.ManagerError):
            self._install(_package(), overwrite=False)
        self._install(_package(), overwrite=True)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "Demo-GGUF", "model-q8_0.gguf")))

    def test_precision_variants_can_share_a_target_directory(self):
        self._install(_package())
        self._install(_package(
            id="demo_f16",
            precision="f16",
            files=("Demo-GGUF/model-f16.gguf",),
        ))
        target = Path(self.root) / "Demo-GGUF"
        self.assertTrue((target / "model-q8_0.gguf").is_file())
        self.assertTrue((target / "model-f16.gguf").is_file())

    def test_overwrite_preserves_other_precision_variants(self):
        target = Path(self.root) / "Demo-GGUF"
        target.mkdir()
        (target / "model-q8_0.gguf").write_bytes(b"q8")
        (target / "model-f16.gguf").write_bytes(b"old")
        self._install(_package(
            id="demo_f16",
            precision="f16",
            files=("Demo-GGUF/model-f16.gguf",),
        ), overwrite=True)
        self.assertEqual((target / "model-q8_0.gguf").read_bytes(), b"q8")
        self.assertEqual((target / "model-f16.gguf").read_bytes(), b"gguf")

    def test_uninstall_removes_only_selected_precision(self):
        q8 = _package()
        f16 = _package(id="demo_f16", precision="f16", files=("Demo-GGUF/model-f16.gguf",))
        self._install(q8)
        self._install(f16)
        self._uninstall(q8)
        target = Path(self.root) / "Demo-GGUF"
        self.assertFalse((target / "model-q8_0.gguf").exists())
        self.assertTrue((target / "model-f16.gguf").is_file())

    def test_uninstall_last_package_removes_empty_target(self):
        package = _package()
        self._install(package)
        self._uninstall(package)
        self.assertFalse((Path(self.root) / "Demo-GGUF").exists())

    def test_progress_mode_emits_downloaded_and_total_bytes(self):
        original = mmw.check_remote_file
        self.addCleanup(setattr, mmw, "check_remote_file", original)
        mmw.check_remote_file = lambda _package, _remote: 4
        output = StringIO()
        with redirect_stdout(output):
            self._install(_package(), progress=True)
        self.assertIn("AUDIOCPP_PROGRESS downloaded=0 total=4", output.getvalue())
        self.assertIn("AUDIOCPP_PROGRESS downloaded=4 total=4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
