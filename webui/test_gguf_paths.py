"""The WebUI must agree with the C++ loader about which GGUF a model directory means.

find_directory_gguf() in src/framework/assets/tensor_source.cpp resolves a model
directory to model.gguf, or to the sole *.gguf in it. The WebUI used to match only
model.gguf, so an installed GGUF package that keeps its release name
(vevo2-q8_0.gguf, voxtral-mini-4b-realtime-2602-q8_0.gguf) read as "no GGUF yet" and
the GGUF tab offered to convert weights that were never downloaded (issue #113).
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

try:
    from webui import webui as app
except ImportError:
    import webui as app


def _entry(path, download_id=None):
    return {"abs_path": path, "download_id": download_id, "family": "vevo2",
            "id": "vevo2", "label": "Vevo2", "installed": True, "incomplete": False}


class ExistingGgufPathTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="audiocpp_webui_gguf_test_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.model_dir = os.path.join(self.root, "Vevo2-GGUF")
        os.makedirs(self.model_dir)

    def _touch(self, name):
        path = os.path.join(self.model_dir, name)
        open(path, "w").close()
        return path

    def test_release_named_package_gguf_is_found(self):
        expected = self._touch("vevo2-q8_0.gguf")
        self.assertEqual(app._existing_gguf_path(_entry(self.model_dir)), expected)

    def test_model_gguf_wins_over_a_release_named_sibling(self):
        self._touch("vevo2-q8_0.gguf")
        expected = self._touch("model.gguf")
        self.assertEqual(app._existing_gguf_path(_entry(self.model_dir)), expected)

    def test_several_release_named_ggufs_are_ambiguous(self):
        # The loader refuses to guess between them, so neither may be advertised.
        self._touch("vevo2-q8_0.gguf")
        self._touch("vevo2-f16.gguf")
        self.assertIsNone(app._existing_gguf_path(_entry(self.model_dir)))

    def test_directory_without_a_gguf_has_none(self):
        self._touch("model.safetensors")
        self.assertIsNone(app._existing_gguf_path(_entry(self.model_dir)))

    def test_explicit_gguf_model_path_is_itself(self):
        path = os.path.join(self.root, "direct.gguf")
        open(path, "w").close()
        self.assertEqual(app._existing_gguf_path(_entry(path)), path)

    def test_conversions_still_target_model_gguf(self):
        # model.gguf is what the loader prefers when a directory holds several, so
        # conversion output keeps that name even though discovery is now broader.
        self.assertEqual(app._gguf_output_path(_entry(self.model_dir)),
                         os.path.join(self.model_dir, "model.gguf"))


class GgufStatusTests(unittest.TestCase):
    def test_downloaded_package_reports_the_gguf_it_will_load(self):
        # vevo2 is not WebUI-convertible; before the fix this returned the
        # "cannot be converted automatically" warning even with a GGUF installed.
        root = tempfile.mkdtemp(prefix="audiocpp_webui_gguf_status_test_")
        self.addCleanup(shutil.rmtree, root, True)
        name = "vevo2-q8_0.gguf"
        open(os.path.join(root, name), "w").close()
        entry = _entry(root, download_id="vevo2_gguf")
        original = app.catalog_by_id
        app.catalog_by_id = lambda model_id: entry
        self.addCleanup(setattr, app, "catalog_by_id", original)
        self.assertIn(name, app.gguf_status("vevo2"))


class DeleteGgufTests(unittest.TestCase):
    def test_a_downloaded_package_gguf_is_not_deleted_as_conversion_output(self):
        root = tempfile.mkdtemp(prefix="audiocpp_webui_gguf_delete_test_")
        self.addCleanup(shutil.rmtree, root, True)
        name = "vevo2-q8_0.gguf"
        target = os.path.join(root, name)
        open(target, "w").close()
        entry = _entry(root, download_id="vevo2_gguf")
        original = app.catalog_by_id
        app.catalog_by_id = lambda model_id: entry
        self.addCleanup(setattr, app, "catalog_by_id", original)
        message, _status = app.delete_gguf("vevo2")
        self.assertIn(name, message)
        self.assertTrue(os.path.isfile(target),
                        "delete_gguf removed a downloaded model package")


if __name__ == "__main__":
    unittest.main()
