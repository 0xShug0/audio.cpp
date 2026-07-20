import os
import tempfile
import unittest
import wave
from unittest import mock

try:
    from webui import webui as app
except ImportError:
    import webui as app


class VoiceLibraryTests(unittest.TestCase):
    def setUp(self):
        self._previous_language = app.get_language()
        app.set_language("zh")

    def tearDown(self):
        app.set_language(self._previous_language)

    @staticmethod
    def _write_wav(path):
        with wave.open(path, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\0\0" * 160)

    def test_original_upload_name_is_available_before_staging(self):
        with mock.patch.object(app, "_stage_upload", return_value="staged.wav"):
            staged, name = app._stage_tts_voice_upload(
                os.path.join("somewhere", "Original Voice.wav"))
        self.assertEqual(staged, "staged.wav")
        self.assertEqual(name, "Original Voice")

    def test_save_copies_wav_and_updates_one_prompt_record(self):
        with tempfile.TemporaryDirectory() as directory:
            voice_dir = os.path.join(directory, "voice")
            os.makedirs(voice_dir)
            source = os.path.join(directory, "source.wav")
            self._write_wav(source)

            with mock.patch.object(app, "PROMPTS_DIR", voice_dir):
                _dropdown, message = app.save_builtin_voice(
                    source, "测试音色", "第一行\n第二行")
                app.save_builtin_voice(source, "测试音色.wav", "更新文本")

            self.assertTrue(os.path.isfile(os.path.join(voice_dir, "测试音色.wav")))
            with open(os.path.join(voice_dir, "prompt_text"),
                      "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "测试音色|更新文本\n")
            self.assertIn("测试音色.wav", message)

    def test_save_requires_name_and_audio(self):
        _dropdown, message = app.save_builtin_voice(None, "", "")
        self.assertIn("填写", message)
        _dropdown, message = app.save_builtin_voice(None, "voice", "")
        self.assertIn("上传", message)

    def test_delete_removes_wav_and_prompt_record(self):
        with tempfile.TemporaryDirectory() as directory:
            voice_dir = os.path.join(directory, "voice")
            os.makedirs(voice_dir)
            source = os.path.join(directory, "source.wav")
            self._write_wav(source)

            with mock.patch.object(app, "PROMPTS_DIR", voice_dir):
                app.save_builtin_voice(source, "delete_me", "参考文本")
                result = app.delete_builtin_voice("delete_me.wav")

            self.assertFalse(os.path.exists(
                os.path.join(voice_dir, "delete_me.wav")))
            with open(os.path.join(voice_dir, "prompt_text"),
                      "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "")
            self.assertIn("已删除", result[-1])

    def test_delete_none_does_nothing(self):
        self.assertEqual(
            app.delete_builtin_voice("(none)"),
            tuple(app.gr.skip() for _ in range(5)))

    def test_voice_name_rejects_paths(self):
        with self.assertRaises(ValueError):
            app._builtin_voice_filename("../outside")


if __name__ == "__main__":
    unittest.main()
