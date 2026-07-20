import json
import os
import tempfile
import unittest
from unittest import mock

try:
    from ui_i18n import (
        DEFAULT_LANGUAGE,
        LANGUAGE_CHOICES,
        get_language,
        load_language,
        normalize_language,
        param_spec,
        save_language,
        set_language,
        text,
    )
except ImportError:
    from webui.ui_i18n import (
        DEFAULT_LANGUAGE,
        LANGUAGE_CHOICES,
        get_language,
        load_language,
        normalize_language,
        param_spec,
        save_language,
        set_language,
        text,
    )


HERE = os.path.dirname(os.path.abspath(__file__))


class UiI18nTests(unittest.TestCase):
    def tearDown(self):
        set_language(DEFAULT_LANGUAGE)

    def test_english_is_the_default(self):
        set_language(DEFAULT_LANGUAGE)
        self.assertEqual(get_language(), "en")
        self.assertEqual(text("生成语音", "Generate speech"), "Generate speech")

    def test_manual_chinese_switch(self):
        set_language("zh")
        self.assertEqual(get_language(), "zh")
        self.assertEqual(text("生成语音", "Generate speech"), "生成语音")

    def test_language_choices_are_in_requested_order(self):
        self.assertEqual(LANGUAGE_CHOICES, [
            ("English", "en"),
            ("中文", "zh"),
            ("中文繁體", "zh-Hant"),
        ])

    def test_manual_traditional_chinese_switch(self):
        set_language("zh-Hant")
        self.assertEqual(get_language(), "zh-Hant")
        self.assertEqual(text("生成语音", "Generate speech"), "生成語音")

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(set_language("fr"), "en")

    def test_missing_language_config_defaults_to_english(self):
        set_language("zh")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing.json")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("AUDIOCPP_LANG", None)
                self.assertEqual(load_language(path), "en")
                self.assertEqual(get_language(), "en")

    def test_audiocpp_lang_supplies_the_default_when_nothing_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing.json")
            with mock.patch.dict(os.environ, {"AUDIOCPP_LANG": "zh_TW"}):
                self.assertEqual(load_language(path), "zh-Hant")

    def test_saved_language_beats_audiocpp_lang(self):
        """An explicit pick in the UI must survive a stale env var."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ui_language.json")
            save_language("zh", path)
            with mock.patch.dict(os.environ, {"AUDIOCPP_LANG": "en"}):
                self.assertEqual(load_language(path), "zh")

    def test_audiocpp_lang_aliases_normalize(self):
        for value, expected in (
            ("EN", "en"), ("english", "en"),
            ("zh-CN", "zh"), ("zh_Hans", "zh"), ("Chinese", "zh"),
            ("zh-Hant", "zh-Hant"), ("zh_TW", "zh-Hant"), ("tw", "zh-Hant"),
            ("  zh-hk  ", "zh-Hant"),
            ("klingon", "en"),
        ):
            with self.subTest(value=value):
                self.assertEqual(normalize_language(value), expected)

    def test_saved_language_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ui_language.json")
            self.assertEqual(save_language("en", path), "en")
            set_language("zh")
            self.assertEqual(load_language(path), "en")
            self.assertEqual(get_language(), "en")
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"language": "en"})

    def test_saved_traditional_chinese_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ui_language.json")
            self.assertEqual(save_language("zh-Hant", path), "zh-Hant")
            set_language("zh")
            self.assertEqual(load_language(path), "zh-Hant")
            self.assertEqual(get_language(), "zh-Hant")

    def test_english_param_keeps_identifier_and_drops_long_copy(self):
        localized = param_spec({
            "name": "guidance_scale",
            "label": "guidance_scale（引导强度）",
            "info": "这是一段很长的说明",
            "placeholder": "中文占位说明",
        }, "en")
        self.assertEqual(localized["label"], "guidance_scale")
        self.assertIsNone(localized["info"])
        self.assertEqual(localized["placeholder"], "")

    def test_traditional_param_converts_visible_copy_only(self):
        localized = param_spec({
            "name": "guidance_scale",
            "label": "引导强度",
            "info": "加载模型后生效",
            "placeholder": "请输入文件路径",
            "value": "保持不变",
        }, "zh-Hant")
        self.assertEqual(localized["name"], "guidance_scale")
        self.assertEqual(localized["label"], "引導強度")
        self.assertEqual(localized["info"], "加載模型後生效")
        self.assertEqual(localized["placeholder"], "請輸入文件路徑")
        self.assertEqual(localized["value"], "保持不变")

    def test_every_config_control_has_an_english_identifier_label(self):
        path = os.path.join(HERE, "configs", "model_params.json")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for family, specs in config.items():
            if family.startswith("_"):
                continue
            for spec in specs:
                with self.subTest(family=family, name=spec.get("name")):
                    self.assertTrue(param_spec(spec, "en").get("label"))


if __name__ == "__main__":
    unittest.main()
