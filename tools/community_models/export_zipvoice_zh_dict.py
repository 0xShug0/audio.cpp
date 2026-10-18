#!/usr/bin/env python3
"""Export Chinese G2P dictionaries for the ZipVoice audio.cpp frontend.

The upstream ZipVoice EmiliaTokenizer converts Chinese text with
jieba + pypinyin (TONE3 style, tone sandhi, neutral tone = 5) and splits
every syllable into an initial token (e.g. "w0") plus a final token
(e.g. "o3"). audio.cpp implements the lookup in C++ but must reproduce
pypinyin's syllable splitting exactly, so this script bakes the split
into the exported tables:

  zh_chars.tsv      <char>\\t<initial0> <final><tone>     (default reading)
  zh_phrases.tsv    <phrase>\\t<tokens for every syllable>   (raw tones, no sandhi)
  zh_syllables.tsv  <tone3 syllable>\\t<tokens>              (for <pinyin> tags)

Tone sandhi (3-3 runs, 一/不) is applied by the C++ frontend at run time,
mirroring lazy_pinyin(tone_sandhi=True); the tables therefore stay raw.

This script also stages the jieba segmentation dictionaries used by the
C++ frontend (the model-local JiebaSegmenter, equivalent to python
jieba.cut with HMM on):

  zh_jieba_dict.txt   <word> <freq> <tag> lines (jieba jieba.dict.utf8)
  zh_hmm_model.txt    HMM start/trans/emit tables (jieba hmm_model.utf8)

The dictionaries are not vendored in this repository; they are downloaded
from the cppjieba mirror commit pinned below (SHA-256 verified, cached
under ~/.cache/audiocpp) so the packaged model is self-contained. Pass
--cppjieba-dict to copy from a local cppjieba checkout instead
(offline escape hatch).

Splitting follows pypinyin's to_initials(strict=False) +
to_finals_tone3(strict=False, neutral_tone_with_five=True): y/w ARE
initials ("yi3" -> "y0 i3"), and ju/qu/xu keep "u" (only nv/lv give "v").

Run with the same Python environment used for the upstream checkout, e.g.
  /path/to/ZipVoice/.venv/bin/python \\
      tools/community_models/export_zipvoice_zh_dict.py \\
      --output-dir /models/ZipVoice/zipvoice_distill
"""
import argparse
import hashlib
import unicodedata
import urllib.request
from pathlib import Path

# These jieba dictionaries are pinned to a cppjieba mirror commit (cppjieba
# redistributes the jieba dictionaries); byte-for-byte identity is enforced by
# SHA-256 so exported packages stay reproducible.
CPPJIEBA_COMMIT = "8f171de"
CPPJIEBA_DICT_URL = ("https://raw.githubusercontent.com/yanyiwu/cppjieba/"
                     + CPPJIEBA_COMMIT + "/dict/{name}")
CPPJIEBA_DICT_SHA256 = {
    "jieba.dict.utf8":
        "6f7d4350e8861ef4139b2e3a6fad05430c19ae71f4b8378190edecac8aae2e6a",
    "hmm_model.utf8":
        "f17790586ac86dd048c8adffed052c4bd2b28ed0682972c1275e59040c0589a7",
}
DEFAULT_DICT_CACHE_DIR = Path.home() / ".cache" / "audiocpp" / "cppjieba-dict"

_TONE_MARKS = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("v", 1), "ǘ": ("v", 2), "ǚ": ("v", 3), "ǜ": ("v", 4),
    "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4),
    "ḿ": ("m", 2),
}

_INITIALS = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k",
    "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
]


def to_tone3(syllable: str) -> str:
    """Convert an accented pinyin syllable (e.g. "xiǎo") to TONE3 ("xiao3")."""
    syllable = unicodedata.normalize("NFKC", syllable)
    out = []
    tone = 0
    i = 0
    while i < len(syllable):
        ch = syllable[i]
        two = syllable[i:i + 2]
        if two in _TONE_MARKS:
            base, t = _TONE_MARKS[two]
            out.append(base)
            tone = t
            i += 2
            continue
        if ch in _TONE_MARKS:
            base, t = _TONE_MARKS[ch]
            out.append(base)
            tone = t
        else:
            out.append(ch)
        i += 1
    if tone == 0:
        tone = 5  # neutral-tone syllables carry no mark
    return "".join(out) + str(tone)


def split_syllable(tone3: str) -> list[str]:
    """Split a TONE3 syllable into initial ("X0") + final ("Ytone") tokens."""
    tone = tone3[-1]
    body = tone3[:-1]
    if not body.isalpha():
        return [tone3]
    initial = ""
    rest = body
    for cand in _INITIALS:
        if body.startswith(cand):
            initial = cand
            rest = body[len(cand):]
            break
    tokens = []
    if initial:
        tokens.append(initial + "0")
    if rest:
        tokens.append(rest + tone)
    return tokens


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_jieba_dicts(output_dir: Path, cache_dir: Path,
                      override_dir: Path | None) -> None:
    """Write zh_jieba_dict.txt / zh_hmm_model.txt next to tokens.txt.

    Downloads the pinned upstream dictionaries into cache_dir on first use.
    --cppjieba-dict overrides the download with a local cppjieba dict/
    directory (copied as-is; its hashes are printed for visibility).
    """
    for name in ("jieba.dict.utf8", "hmm_model.utf8"):
        target = output_dir / ("zh_hmm_model.txt" if name.endswith("hmm_model.utf8")
                               else "zh_jieba_dict.txt")
        if override_dir is not None:
            source = override_dir / name
            if not source.is_file():
                raise SystemExit(f"--cppjieba-dict override missing {name}: {source}")
            data = source.read_bytes()
            note = f"copied from {source} (sha256 {_sha256(data)})"
        else:
            cache = cache_dir / name
            if cache.is_file() and _sha256(cache.read_bytes()) == CPPJIEBA_DICT_SHA256[name]:
                data = cache.read_bytes()
                note = f"cached {cache}"
            else:
                url = CPPJIEBA_DICT_URL.format(name=name)
                print(f"downloading {url}")
                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read()
                digest = _sha256(data)
                if digest != CPPJIEBA_DICT_SHA256[name]:
                    raise SystemExit(
                        f"downloaded {name} SHA-256 {digest} does not match the pinned "
                        f"{CPPJIEBA_DICT_SHA256[name]}; pass --cppjieba-dict to copy the "
                        "dictionaries from a local cppjieba checkout instead")
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(data)
                note = f"downloaded (pinned {CPPJIEBA_COMMIT}, SHA-256 verified)"
        target.write_bytes(data)
        print(f"staged {name} -> {target} [{note}]")


def export_pinyin_tables(output_dir: Path) -> None:
    # Deferred import: the dictionary staging above only needs the standard
    # library, so the download path runs in any Python; only the table baking
    # requires the upstream ZipVoice environment.
    try:
        from pypinyin.phrases_dict import phrases_dict
        from pypinyin.pinyin_dict import pinyin_dict
    except ImportError as error:
        raise SystemExit(
            "pypinyin is required to bake the pronunciation tables; run this "
            "script with the upstream ZipVoice python environment "
            f"(import failed: {error})") from error

    # Characters: pinyin_dict maps codepoint -> space-separated accented
    # pinyins; the first entry is pypinyin's default reading.
    bare_syllables = set()
    chars_path = output_dir / "zh_chars.tsv"
    with chars_path.open("w", encoding="utf-8") as f:
        for codepoint, pinyins in sorted(pinyin_dict.items()):
            if not (0x3400 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF):
                continue
            # Entries are comma-separated readings (e.g. "hǔ,hù"); the
            # first is pypinyin's default reading.
            first = to_tone3(pinyins.split(",")[0].strip())
            bare_syllables.add(first)
            tokens = split_syllable(first)
            f.write(f"{chr(codepoint)}\t{' '.join(tokens)}\n")
    print(f"wrote {chars_path}")

    # Phrases: raw per-syllable tones (no sandhi); longest-match lookup and
    # tone sandhi happen in C++.
    phrases_path = output_dir / "zh_phrases.tsv"
    count = 0
    with phrases_path.open("w", encoding="utf-8") as f:
        for phrase, syllables in sorted(phrases_dict.items()):
            if not all(len(s) == 1 for s in syllables):
                continue
            tokens = []
            for entry in syllables:
                tokens.extend(split_syllable(to_tone3(entry[0])))
            f.write(f"{phrase}\t{' '.join(tokens)}\n")
            count += 1
    print(f"wrote {phrases_path} ({count} entries)")

    # Valid bare tone3 syllables for <pinyin> overrides.
    syllables_path = output_dir / "zh_syllables.tsv"
    with syllables_path.open("w", encoding="utf-8") as f:
        for syllable in sorted(bare_syllables):
            f.write(f"{syllable}\t{' '.join(split_syllable(syllable))}\n")
    print(f"wrote {syllables_path} ({len(bare_syllables)} entries)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="model directory that contains tokens.txt")
    parser.add_argument("--cppjieba-dict", type=Path, default=None,
                        help="offline override: cppjieba dict/ directory to copy "
                             "from (default: download the pinned upstream "
                             "dictionaries, SHA-256 verified)")
    parser.add_argument("--dict-cache-dir", type=Path, default=DEFAULT_DICT_CACHE_DIR,
                        help="download cache directory (default: %(default)s)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # jieba segmentation dictionaries for the model-local segmenter.
    stage_jieba_dicts(args.output_dir, args.dict_cache_dir, args.cppjieba_dict)

    export_pinyin_tables(args.output_dir)


if __name__ == "__main__":
    main()
