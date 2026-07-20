"""Small multilingual helpers for the local Gradio WebUI."""

import json
import os

from opencc import OpenCC

LANG_ZH = "zh"
LANG_ZH_HANT = "zh-Hant"
LANG_EN = "en"
DEFAULT_LANGUAGE = LANG_EN
LANGUAGE_CHOICES = [
    ("English", LANG_EN),
    ("中文", LANG_ZH),
    ("中文繁體", LANG_ZH_HANT),
]
LANGUAGE_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "ui_language.json")

# Spellings accepted from AUDIOCPP_LANG, which people write in several ways.
_LANGUAGE_ALIASES = {
    "en": LANG_EN, "en-us": LANG_EN, "en_us": LANG_EN, "english": LANG_EN,
    "zh": LANG_ZH, "zh-cn": LANG_ZH, "zh_cn": LANG_ZH, "zh-hans": LANG_ZH,
    "zh_hans": LANG_ZH, "cn": LANG_ZH, "chinese": LANG_ZH,
    "zh-hant": LANG_ZH_HANT, "zh_hant": LANG_ZH_HANT, "zh-tw": LANG_ZH_HANT,
    "zh_tw": LANG_ZH_HANT, "zh-hk": LANG_ZH_HANT, "zh_hk": LANG_ZH_HANT,
    "tw": LANG_ZH_HANT, "hant": LANG_ZH_HANT, "traditional": LANG_ZH_HANT,
}

_language = DEFAULT_LANGUAGE
_s2t_converter = OpenCC("s2t")


def normalize_language(language):
    if language in (LANG_ZH, LANG_ZH_HANT, LANG_EN):
        return language
    if isinstance(language, str):
        alias = _LANGUAGE_ALIASES.get(language.strip().lower())
        if alias:
            return alias
    return DEFAULT_LANGUAGE


def to_traditional(value):
    """Convert visible Simplified Chinese copy while preserving non-strings."""
    return _s2t_converter.convert(value) if isinstance(value, str) else value


def set_language(language):
    global _language
    _language = normalize_language(language)
    return _language


def get_language():
    return _language


def load_language(path=LANGUAGE_CONFIG_PATH):
    """Resolve the startup UI language.

    A language saved from the in-UI picker wins, so an explicit click keeps
    surviving restarts. AUDIOCPP_LANG only supplies the default for a profile
    that has never chosen one (first run, or a scripted/headless launch);
    otherwise it would silently defeat the picker. Falls back to English.
    """
    saved = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            saved = data.get("language")
    except (OSError, ValueError, TypeError):
        pass
    if saved is None:
        saved = os.environ.get("AUDIOCPP_LANG") or DEFAULT_LANGUAGE
    return set_language(saved)


def save_language(language, path=LANGUAGE_CONFIG_PATH):
    """Persist the normalized UI language and return it."""
    language = normalize_language(language)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({"language": language}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp_path, path)
    return language


def text(zh, en, language=None, **values):
    """Choose and format localized UI copy."""
    language = normalize_language(language or _language)
    if language == LANG_EN:
        template = en
    elif language == LANG_ZH_HANT:
        template = to_traditional(zh)
    else:
        template = zh
    return template.format(**values) if values else template


def param_spec(spec, language=None):
    """Return a localized advanced-parameter spec without changing its value.

    Protocol names, choices and defaults must stay stable.  In English mode a
    missing translated label falls back to the option name, while untranslated
    explanatory copy is omitted to keep dynamic control rows compact.
    """
    language = normalize_language(language or _language)
    if language == LANG_ZH:
        return dict(spec)

    if language == LANG_ZH_HANT:
        out = dict(spec)
        for key in ("label", "info", "placeholder"):
            if key in out:
                out[key] = to_traditional(out[key])
        return out

    out = dict(spec)
    out["label"] = spec.get("label_en") or spec.get("name", "")
    out["info"] = spec.get("info_en") or None
    placeholder = spec.get("placeholder_en")
    if placeholder is None:
        original = spec.get("placeholder", "")
        placeholder = original if original.isascii() else ""
    out["placeholder"] = placeholder
    return out
