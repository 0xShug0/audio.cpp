"""The WebUI catalog must stay in step with the repo it ships next to.

configs/models_catalog.json and configs/required_files.json are hand-maintained
copies of facts that live in tools/model_manager.py (package ids, install
directories, post-install file lists) and src/framework/runtime/registry.cpp
(which families the server can actually load). Nothing fails loudly when they
drift: a renamed install directory just makes an installed model read as "not
installed", a dropped package makes the Download button run a package id that no
longer exists, and a new family is simply invisible in the UI. Release 0.4 did
all three at once, which is why these are tests.
"""
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if os.path.join(REPO_ROOT, "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import model_manager  # noqa: E402

try:
    from webui import webui as app
except ImportError:
    import webui as app

CATALOG_PATH = os.path.join(HERE, "configs", "models_catalog.json")
REQUIRED_FILES_PATH = os.path.join(HERE, "configs", "required_files.json")
REGISTRY_PATH = os.path.join(REPO_ROOT, "src", "framework", "runtime", "registry.cpp")

_LOADER_CALL_RE = re.compile(r"\bmake_([a-z0-9_]+)_loader\s*\(\s*\)")

# Families the server registers but the WebUI deliberately does not surface,
# with the reason. Keep this empty unless there is one.
UNLISTED_FAMILIES: dict[str, str] = {
    "kroko_asr": (
        "Kroko Community packages require a user-supplied free .data file; "
        "the WebUI installer cannot provide that local converter input."
    ),
}


def _packages():
    catalog = model_manager.CATALOG
    values = catalog.values() if isinstance(catalog, dict) else catalog
    return {p.id: p for p in values}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class CatalogSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packages = _packages()
        cls.entries = _load(CATALOG_PATH)["models"]
        cls.required = _load(REQUIRED_FILES_PATH)

    def test_every_download_id_is_a_model_manager_package(self):
        for entry in self.entries:
            download_id = entry.get("download_id")
            if download_id is None:
                continue
            self.assertIn(download_id, self.packages,
                          f"catalog entry {entry['id']} downloads {download_id!r}, "
                          "which tools/model_manager.py no longer offers")

    def test_install_directory_matches_the_package_target(self):
        for entry in self.entries:
            download_id = entry.get("download_id")
            if download_id not in self.packages:
                continue
            target = self.packages[download_id].target_directory
            self.assertEqual(os.path.basename(entry["path"]), target,
                             f"catalog entry {entry['id']} looks in {entry['path']!r}, "
                             f"but {download_id} installs into {target!r}")

    def test_entries_without_a_download_id_ship_with_the_repo(self):
        # No download_id means the weights are committed (bundled assets); the
        # path has to exist in a checkout or the model is simply unreachable.
        for entry in self.entries:
            if entry.get("download_id"):
                continue
            self.assertTrue(os.path.isdir(os.path.join(REPO_ROOT, entry["path"])),
                            f"catalog entry {entry['id']} has no download_id, so "
                            f"{entry['path']!r} must be a bundled directory in the repo")

    def test_required_files_mirrors_the_package_specs(self):
        expected = {pkg_id: list(p.required_files)
                    for pkg_id, p in self.packages.items() if p.required_files}
        actual = {k: v for k, v in self.required.items() if k != "_comment"}
        self.assertEqual(actual, expected,
                         "configs/required_files.json is stale; regenerate it from "
                         "tools/model_manager.py CATALOG (see its _comment)")

    def test_gguf_families_come_from_the_package_specs(self):
        specs = {os.path.splitext(f)[0]
                 for f in os.listdir(os.path.join(REPO_ROOT, "model_specs"))
                 if f.endswith(".json")}
        self.assertEqual(app.GGUF_NATIVE_FAMILIES, specs,
                         "GGUF_NATIVE_FAMILIES should be read from model_specs/")
        # The no-model_specs fallback may lag behind, but it must never claim
        # GGUF support for a family the runtime has no package spec for.
        self.assertLessEqual(app.GGUF_NATIVE_FAMILIES_FALLBACK, specs)
        self.assertLessEqual(app.GGUF_WEBUI_CONVERTIBLE_FAMILIES, specs)

    def test_every_registered_family_is_reachable_from_the_ui(self):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = f.read()
        registered = {m.group(1) for line in registry.splitlines()
                      for m in [_LOADER_CALL_RE.search(line.strip())]
                      if m and not line.strip().startswith("//")}
        listed = {e.get("family") for e in self.entries}
        missing = sorted(registered - listed - set(UNLISTED_FAMILIES))
        self.assertFalse(missing,
                         "families the server can load but the WebUI never offers: "
                         f"{missing}. Add a catalog entry, or record why not in "
                         "UNLISTED_FAMILIES.")


if __name__ == "__main__":
    unittest.main()
