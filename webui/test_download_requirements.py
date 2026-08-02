"""What a download costs has to be visible, and alarming, before it starts.

Catalog weights run from 40 MB to 17 GB. Nothing used to state the size up front, so a
package that could not fit filled the volume and failed near the end, leaving the partial
behind in models/.engine_model_staging.

The Download button now only proposes: it reports the download size, the resulting disk
utilization, and the share of VRAM (or of system RAM on the CPU backend) the model would
need, raises an alarm below DISK_FREE_ALARM_BYTES (or past MEMORY_USAGE_ALARM), and waits
for an explicit confirmation. A download that cannot fit at all is refused outright.

Sizes come from the Hugging Face tree API; it is stubbed here so the tests stay offline.
"""
import collections
import email.utils
import os
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from webui import webui as app
except ImportError:
    import webui as app

GB = 1000 ** 3
_Usage = collections.namedtuple("_Usage", "total used free")


def _usage(free, total=100 * GB):
    """A volume of `total` with `free` available, i.e. (total - free) already used."""
    return _Usage(total, total - free, free)


class _AppPatch(unittest.TestCase):
    """Patches module globals and restores them, so tests never touch the real disk."""

    def patch(self, **values):
        for name, value in values.items():
            self.addCleanup(setattr, app, name, getattr(app, name))
            setattr(app, name, value)

    def stub_disk(self, free, total=100 * GB):
        self.patch(_disk_usage=lambda path, u=_usage(free, total): u)

    def stub_size(self, total):
        self.patch(package_download_bytes=lambda download_id: total)


class RequirementsNoteTests(_AppPatch):
    def setUp(self):
        self.entry = {"id": "vevo2", "label": "Vevo2", "family": "vevo2",
                      "download_id": "vevo2_q8_0", "abs_path": "/nonexistent",
                      "installed": False, "incomplete": False, "missing_files": [],
                      "min_vram_gb": 6}
        self.stub_size(3 * GB)
        self.stub_disk(50 * GB)          # 50% used -> 53% after: below the alarm
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0)

    def test_note_states_size_utilization_and_vram(self):
        note, blocker = app.download_requirements(self.entry)
        self.assertIn("3.00 GB", note)
        self.assertIn("50.00 GB", note)
        self.assertIn("50% → 53%", note)
        self.assertIn("6 GB VRAM", note)
        self.assertIn("75%", note, "the VRAM share of the local card should be stated")
        self.assertEqual(blocker, "")

    def test_no_alarm_below_both_thresholds(self):
        note, blocker = app.download_requirements(self.entry)
        self.assertNotIn("🚨", note)
        self.assertEqual(blocker, "")

    def test_a_download_that_cannot_fit_is_blocked(self):
        self.stub_disk(1 * GB)
        note, blocker = app.download_requirements(self.entry)
        self.assertIn("3.00 GB", note)
        self.assertTrue(blocker, "a download larger than the free space was not blocked")
        self.assertIn("Vevo2", blocker)

    def test_an_unknown_size_is_not_treated_as_zero(self):
        # Unreachable repos report None; that must never look
        # like "0 bytes needed, definitely fits" nor block the download.
        self.stub_size(None)
        self.stub_disk(0)
        note, blocker = app.download_requirements(self.entry)
        self.assertEqual(blocker, "")
        self.assertNotIn("Download", note)
        self.assertIn("VRAM", note, "the VRAM estimate should still be reported")

    def test_vram_is_still_reported_on_the_cpu_backend(self):
        # "How much GPU memory does this need" is worth answering before a multi-GB
        # download whichever backend happens to be built.
        self.patch(BACKEND="cpu", LOCAL_RAM_GB=32.0)
        note, _blocker = app.download_requirements(self.entry)
        self.assertIn("6 GB", note)
        self.assertIn("system RAM", note)
        self.assertIsNone(app._vram_shortfall(self.entry),
                          "the CPU backend must not raise a VRAM shortfall warning")

    def test_vram_is_reported_without_a_detected_device(self):
        self.patch(LOCAL_VRAM_GB=None)
        note, _blocker = app.download_requirements(self.entry)
        self.assertIn("6 GB VRAM", note)


class DiskAlarmTests(_AppPatch):
    """Low remaining space after a download is still offered, but loudly."""

    def setUp(self):
        self.entry = {"id": "m", "label": "M", "family": "vevo2", "download_id": "vevo2_q8_0",
                      "abs_path": "/nonexistent", "installed": False, "incomplete": False,
                      "missing_files": []}
        self.stub_size(10 * GB)
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0)

    def test_the_free_space_threshold_is_twenty_gb(self):
        self.assertEqual(app.DISK_FREE_ALARM_BYTES, 20 * GB)

    def test_leaving_less_than_twenty_gb_alarms_without_blocking(self):
        self.stub_disk(29 * GB)          # 19 GB left after a 10 GB download
        note, blocker = app.download_requirements(self.entry)
        self.assertIn("🚨", note)
        self.assertIn("19.00 GB", note)
        self.assertIn("20.00 GB", note)
        self.assertEqual(blocker, "", "an alarm must not block; it is the user's disk")

    def test_leaving_at_least_twenty_gb_does_not_alarm(self):
        self.stub_disk(30 * GB)          # exactly 20 GB left after the download
        note, _blocker = app.download_requirements(self.entry)
        self.assertNotIn("🚨", note)

    def test_a_large_disk_with_plenty_left_does_not_alarm(self):
        # The reported issue: 82% full on a 7.27 TB disk still leaves about 1.4 TB free.
        self.stub_disk(1425 * GB, total=7270 * GB)
        note, _blocker = app.download_requirements(self.entry)
        self.assertIn("80% → 81%", note)
        self.assertNotIn("Disk alarm", note)


class MemoryAlarmTests(_AppPatch):
    """Above MEMORY_USAGE_ALARM of the device it will run on, the model is flagged."""

    def setUp(self):
        self.entry = {"id": "m", "label": "M", "family": "vevo2", "download_id": "vevo2_q8_0",
                      "abs_path": "/nonexistent", "installed": False, "incomplete": False,
                      "missing_files": [], "min_vram_gb": 7}
        self.stub_size(1 * GB)
        self.stub_disk(90 * GB)

    def test_the_threshold_is_eighty_percent(self):
        self.assertEqual(app.MEMORY_USAGE_ALARM, 0.80)

    def test_vram_above_the_threshold_alarms(self):
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0)     # 7/8 = 88%
        alarm = app.memory_alarm(self.entry)
        self.assertIsNotNone(alarm)
        self.assertEqual(alarm[3], "vram")
        note, blocker = app.download_requirements(self.entry)
        self.assertIn("VRAM alarm", note)
        self.assertIn("88%", note)
        self.assertEqual(blocker, "", "memory pressure must not block a download")

    def test_vram_below_the_threshold_is_quiet(self):
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=16.0)    # 7/16 = 44%
        self.assertIsNone(app.memory_alarm(self.entry))

    def test_the_cpu_backend_is_judged_against_system_ram(self):
        self.patch(BACKEND="cpu", LOCAL_RAM_GB=7.4, LOCAL_VRAM_GB=64.0)
        alarm = app.memory_alarm(self.entry)
        self.assertIsNotNone(alarm, "a CPU run must be judged against RAM, not an idle GPU")
        self.assertEqual(alarm[3], "ram")
        note, _blocker = app.download_requirements(self.entry)
        self.assertIn("RAM alarm", note)

    def test_plenty_of_system_ram_is_quiet_on_cpu(self):
        self.patch(BACKEND="cpu", LOCAL_RAM_GB=64.0)
        self.assertIsNone(app.memory_alarm(self.entry))

    def test_no_estimate_and_no_device_means_no_alarm(self):
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=None)
        self.assertIsNone(app.memory_alarm(self.entry))
        self.patch(LOCAL_VRAM_GB=8.0)
        self.assertIsNone(app.memory_alarm(dict(self.entry, min_vram_gb=None)))

    def test_system_ram_is_detected_on_this_machine(self):
        self.assertIsNotNone(app._system_ram_gb(), "could not read total system RAM")
        self.assertGreater(app._system_ram_gb(), 0)


class BlockedDownloadDoesNotStartTests(_AppPatch):
    def test_download_model_refuses_and_spawns_nothing(self):
        entry = {"id": "vevo2", "label": "Vevo2", "family": "vevo2",
                 "download_id": "vevo2_q8_0", "abs_path": "/nonexistent",
                 "installed": False, "incomplete": False, "missing_files": [],
                 "min_vram_gb": 6}
        self.patch(catalog_by_id=lambda model_id: entry)
        self.stub_size(100 * GB)
        self.stub_disk(1 * GB)
        started = []
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0)
        self.addCleanup(setattr, app.subprocess, "Popen", app.subprocess.Popen)
        app.subprocess.Popen = lambda *a, **k: started.append(a) or (_ for _ in ()).throw(
            AssertionError("a blocked download still spawned model_manager"))
        message = app.download_model("vevo2")
        self.assertIn("100.00 GB", message)
        self.assertEqual(started, [])
        self.assertNotIn("vevo2", app._downloads)


class WarningsSurviveTheProgressRefreshTests(_AppPatch):
    """download_status() rewrites the whole message every 3 seconds. Anything shown only
    by the initial Download click was replaced within three seconds, which is
    indistinguishable from never having been shown."""

    def setUp(self):
        self.entry = {"id": "m", "label": "Big Model", "family": "vevo2", "path": "models/M",
                      "download_id": "vevo2_q8_0", "abs_path": "/nonexistent", "installed": False,
                      "download_installed": False, "incomplete": False, "missing_files": [],
                      "min_vram_gb": 20}
        self.stub_size(17 * GB)
        self.stub_disk(50 * GB)
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0,
                   _staged_bytes=lambda entry: 5 * GB,
                   catalog_by_id=lambda model_id: self.entry,
                   _read_tail=lambda *a, **k: "")
        self.addCleanup(app._downloads.pop, "m", None)

    def _status(self, poll_result):
        app._downloads["m"] = {"proc": type("P", (), {"poll": lambda s: poll_result})(),
                               "log": os.devnull}
        return app.download_status("m")

    def test_memory_alarm_is_repeated_on_every_running_tick(self):
        self.assertIn("VRAM alarm", self._status(None))

    def test_vram_warning_survives_completion(self):
        self.assertIn("Low VRAM", self._status(0))

    def test_progress_is_reported_against_the_package_total(self):
        status = self._status(None)
        self.assertIn("5.00 GB of 17.00 GB", status)
        self.assertIn("29%", status)

    def test_disk_is_judged_against_what_is_still_to_fetch(self):
        # 12 GB left of a 17 GB download onto a volume that ends at 62% is healthy;
        # judging the full 17 GB against shrinking free space would raise a false alarm.
        self.stub_disk(50 * GB)
        self.assertNotIn("Disk alarm", self._status(None))

    def test_running_out_of_disk_mid_download_is_reported(self):
        self.stub_disk(3 * GB)
        self.assertIn("Will run out of disk", self._status(None))

    def test_progress_falls_back_to_bytes_when_the_total_is_unknown(self):
        self.stub_size(None)
        status = self._status(None)
        self.assertIn("Downloaded 5.00 GB", status)

    def test_size_and_vram_stay_on_screen_for_the_whole_download(self):
        status = self._status(None)
        self.assertIn("17.00 GB", status)
        self.assertIn("VRAM", status)


class ConfirmBeforeDownloadTests(_AppPatch):
    """The Download button proposes; only Confirm commits. A multi-GB fetch cannot be
    undone once the bytes are written, so one stray click must not start one."""

    def setUp(self):
        self.entry = {"id": "m", "label": "Big Model", "family": "vevo2", "path": "models/M",
                      "download_id": "vevo2_q8_0", "abs_path": "/nonexistent", "installed": False,
                      "incomplete": False, "missing_files": [], "min_vram_gb": 20}
        self.stub_size(17 * GB)
        self.stub_disk(50 * GB)
        self.patch(BACKEND="gpu", LOCAL_VRAM_GB=8.0,
                   catalog_by_id=lambda model_id: self.entry,
                   SPEC_MODEL_MANAGER="/path/to/model_manager_webui.py")
        self.spawned = []
        self.addCleanup(setattr, app.subprocess, "Popen", app.subprocess.Popen)

        def fake_popen(*a, **k):
            self.spawned.append(a)
            return type("P", (), {"poll": lambda s: None, "terminate": lambda s: None})()

        app.subprocess.Popen = fake_popen
        self.addCleanup(app._downloads.pop, "m", None)

    def test_clicking_download_starts_nothing(self):
        message, can_confirm = app.download_proposal("m")
        self.assertEqual(self.spawned, [], "the Download button started a download by itself")
        self.assertNotIn("m", app._downloads)
        self.assertTrue(can_confirm)
        self.assertIn("17.00 GB", message)
        self.assertIn("VRAM", message)

    def test_an_alarm_changes_the_question(self):
        message, can_confirm = app.download_proposal("m")   # 20 GB on an 8 GB card
        self.assertIn("Alarms raised", message)
        self.assertIn("really want it", message)
        self.assertTrue(can_confirm, "an alarm must still leave the choice with the user")

    def test_a_quiet_download_asks_plainly(self):
        self.entry["min_vram_gb"] = 2
        self.stub_size(1 * GB)
        message, can_confirm = app.download_proposal("m")
        self.assertNotIn("Alarms raised", message)
        self.assertTrue(can_confirm)

    def test_an_incomplete_directory_is_called_out_as_a_reinstall(self):
        self.entry.update(incomplete=True, missing_files=["a", "b"])
        message, can_confirm = app.download_proposal("m")
        self.assertIn("reinstalled", message)
        self.assertTrue(can_confirm)

    def test_cancelling_starts_nothing(self):
        app.download_proposal("m")
        message, *_hide = app.download_cancel("m")
        self.assertEqual(self.spawned, [])
        self.assertNotIn("m", app._downloads)
        self.assertIn("Cancelled", message)

    def test_nothing_to_confirm_when_it_cannot_fit(self):
        self.stub_disk(1 * GB)
        message, can_confirm = app.download_proposal("m")
        self.assertIn("Not enough disk space", message)
        self.assertFalse(can_confirm, "a download that cannot fit offered a Confirm")

    def test_nothing_to_confirm_when_download_package_is_already_installed(self):
        self.entry.update(installed=True, download_installed=True)
        _message, can_confirm = app.download_proposal("m")
        self.assertFalse(can_confirm)

    def test_existing_legacy_install_can_still_download_the_gguf_package(self):
        self.entry.update(installed=True, download_installed=False)
        _message, can_confirm = app.download_proposal("m")
        self.assertTrue(can_confirm)

    def test_nothing_to_confirm_without_a_selection(self):
        _message, can_confirm = app.download_proposal("")
        self.assertFalse(can_confirm)

    def test_nothing_to_confirm_while_a_download_runs(self):
        app._downloads["m"] = {"proc": type("P", (), {"poll": lambda s: None})(), "log": os.devnull}
        message, can_confirm = app.download_proposal("m")
        self.assertIn("already downloading", message)
        self.assertFalse(can_confirm)

    def test_confirming_starts_the_download(self):
        self.patch(_read_tail=lambda *a, **k: "")
        message, *_rest = app.download_start("m")
        self.assertEqual(len(self.spawned), 1, "Confirm did not start the download")
        self.assertIn("Download started", message)

    def test_the_click_handler_mirrors_the_decision(self):
        # download_preview only wraps download_proposal in Gradio visibility updates.
        message, confirm, cancel = app.download_preview("m")
        self.assertEqual(message, app.download_proposal("m")[0])
        self.assertIsNotNone(confirm)
        self.assertIsNotNone(cancel)


class SizeProbeTests(unittest.TestCase):
    def _stub_spec_package(self, files):
        original = app.SPEC_PACKAGE_BY_ID
        app.SPEC_PACKAGE_BY_ID = {
            **app.SPEC_PACKAGE_BY_ID,
            "pkg": {
                "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/audio.cpp-gguf"},
                "files": files,
            },
        }
        self.addCleanup(setattr, app, "SPEC_PACKAGE_BY_ID", original)
        app._dl_size_cache.pop("pkg", None)
        self.addCleanup(app._dl_size_cache.pop, "pkg", None)

    def _stub_head_lengths(self, lengths):
        calls = []
        original = app.requests.head

        class _Response:
            def __init__(self, length):
                self.headers = {} if length is None else {"Content-Length": str(length)}

            def raise_for_status(self):
                pass

        def fake_head(*args, **kwargs):
            index = len(calls)
            calls.append(args)
            return _Response(lengths[index])

        app.requests.head = fake_head
        self.addCleanup(setattr, app.requests, "head", original)
        return calls

    def test_sizes_are_summed_over_every_snapshot_source(self):
        self._stub_spec_package(("a.gguf", "b.gguf"))
        self._stub_head_lengths((1000, 2000))
        self.assertEqual(app.package_download_bytes("pkg"), 3000)

    def test_a_listing_without_sizes_reports_unknown(self):
        self._stub_spec_package(("a.gguf", "b.gguf"))
        self._stub_head_lengths((1000, None))
        self.assertIsNone(app.package_download_bytes("pkg"))

    def test_an_unreachable_repo_reports_unknown(self):
        self._stub_spec_package(("a.gguf",))
        original = app.requests.head

        def boom(*args, **kwargs):
            raise OSError("network is unreachable")

        app.requests.head = boom
        self.addCleanup(setattr, app.requests, "head", original)
        self.assertIsNone(app.package_download_bytes("pkg"))

    def test_packages_without_a_download_id_report_unknown(self):
        self.assertIsNone(app.package_download_bytes(None))
        self.assertIsNone(app.package_download_bytes(""))


class ModelUpdateNoteTests(_AppPatch):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory(prefix="audiocpp_webui_update_test_")
        self.addCleanup(self.root.cleanup)
        self.old_language = app.get_language()
        app.set_language("en")
        self.addCleanup(app.set_language, self.old_language)
        local = os.path.join(self.root.name, "model.gguf")
        open(local, "wb").close()
        self.local = local
        self.entry = {
            "id": "demo",
            "label": "Demo",
            "download_id": "demo_q8_0",
            "download_installed": True,
            "abs_path": self.root.name,
        }
        self.package = {
            "format": "gguf",
            "files": ("Demo-GGUF/model.gguf",),
            "download": {"kind": "huggingface_snapshot", "repo": "audio-cpp/audio.cpp-gguf"},
        }
        self.patch(SPEC_PACKAGE_BY_ID={"demo_q8_0": self.package},
                   REQUIRED_FILES={"demo_q8_0": ["model.gguf"]})
        app._model_update_cache.clear()
        self.addCleanup(app._model_update_cache.clear)

    def _head_last_modified(self, timestamp):
        original = app.requests.head

        class _Response:
            headers = {
                "Last-Modified": email.utils.formatdate(timestamp, usegmt=True),
                "Content-Length": str(os.path.getsize(self.local)),
            }

            def raise_for_status(self):
                pass

        app.requests.head = lambda *a, **k: _Response()
        self.addCleanup(setattr, app.requests, "head", original)

    def _head_content_length(self, size):
        original = app.requests.head

        class _Response:
            headers = {"Content-Length": str(size)}

            def raise_for_status(self):
                pass

        app.requests.head = lambda *a, **k: _Response()
        self.addCleanup(setattr, app.requests, "head", original)

    def test_newer_remote_package_is_reported(self):
        now = time.time()
        os.utime(self.local, (now - 7200, now - 7200))
        self._head_last_modified(now)

        self.assertIn("Model package update available", app.model_update_note(self.entry))

    def test_current_local_package_is_reported_current(self):
        now = time.time()
        os.utime(self.local, (now, now))
        self._head_last_modified(now - 7200)

        self.assertIn("Local model package is up to date", app.model_update_note(self.entry))

    def test_content_length_fallback_reports_current_package(self):
        self._head_content_length(os.path.getsize(self.local))

        self.assertIn("Local model package is up to date", app.model_update_note(self.entry))

    def test_content_length_fallback_reports_updated_package(self):
        self._head_content_length(os.path.getsize(self.local) + 1)

        self.assertIn("Model package update available", app.model_update_note(self.entry))

    def test_unreachable_remote_is_quiet(self):
        original = app.requests.head
        app.requests.head = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        self.addCleanup(setattr, app.requests, "head", original)

        self.assertEqual("", app.model_update_note(self.entry))

    def test_legacy_install_points_user_to_default_gguf_package(self):
        self.entry["installed"] = True
        self.entry["download_installed"] = False

        self.assertIn("install the default GGUF package", app.model_update_note(self.entry))


class DiskUsageTests(unittest.TestCase):
    def test_usage_walks_up_to_an_existing_ancestor(self):
        # models/ need not exist yet; the volume is still the right thing to measure.
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "models", "not", "created", "yet")
            self.assertIsNotNone(app._disk_usage(missing))
            self.assertEqual(app._disk_usage(missing).total, app._disk_usage(tmp).total)

    def test_fill_after_accounts_for_what_is_already_used(self):
        self.assertAlmostEqual(app._disk_fill_after(10 * GB, _usage(50 * GB)), 0.60)

    def test_fill_after_is_unknown_without_a_size_or_a_volume(self):
        self.assertIsNone(app._disk_fill_after(None, _usage(50 * GB)))
        self.assertIsNone(app._disk_fill_after(10 * GB, None))


if __name__ == "__main__":
    unittest.main()
