#!/usr/bin/env python3
"""Fetch a multilingual FLEURS sample set for parakeet_tdt accuracy testing.

parakeet-tdt-0.6b-v3 is trained on 25 European languages; this pulls N clips
per language from the FLEURS test split, decodes them to the 16 kHz mono WAV
the engine expects, and writes a manifest with the reference transcript.
"""
import io
import json
import os
import sys
import urllib.request

import pyarrow.parquet as pq
import soundfile as sf

# The 25 European languages parakeet-tdt-0.6b-v3 supports, mapped to FLEURS
# config names. (FLEURS has no Maltese, so 24 of the 25 are covered here.)
LANGS = {
    "bg": "bg_bg", "hr": "hr_hr", "cs": "cs_cz", "da": "da_dk", "nl": "nl_nl",
    "en": "en_us", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "de": "de_de",
    "el": "el_gr", "hu": "hu_hu", "it": "it_it", "lv": "lv_lv", "lt": "lt_lt",
    "pl": "pl_pl", "pt": "pt_br", "ro": "ro_ro", "sk": "sk_sk", "sl": "sl_si",
    "es": "es_419", "sv": "sv_se", "ru": "ru_ru", "uk": "uk_ua",
}

OUT = os.environ.get("FLEURS_DIR", "tmp/fleurs")
PER_LANG = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_SECS = 20.0

os.makedirs(OUT, exist_ok=True)
manifest = []

def already_done(lang):
    """Resume support: a language is done if its clips and manifest rows exist."""
    return os.path.exists(f"{OUT}/{lang}_{PER_LANG-1:02d}.wav")


for lang, cfg in LANGS.items():
    if already_done(lang) and os.path.exists(f"{OUT}/done_{lang}.json"):
        manifest.extend(json.load(open(f"{OUT}/done_{lang}.json")))
        print(f"  {lang}: cached", flush=True)
        continue
    dst_parquet = f"{OUT}/{cfg}.parquet"
    if not os.path.exists(dst_parquet):
        url = f"https://huggingface.co/api/datasets/google/fleurs/parquet/{cfg}/test/0.parquet"
        try:
            urllib.request.urlretrieve(url, dst_parquet)
        except Exception as exc:  # noqa: BLE001
            print(f"  {lang}: download failed ({exc})", flush=True)
            continue
    try:
        table = pq.read_table(dst_parquet, columns=["audio", "transcription"])
    except Exception as exc:  # noqa: BLE001
        print(f"  {lang}: parquet read failed ({exc})", flush=True)
        os.remove(dst_parquet)
        continue

    lang_rows = []
    kept = 0
    for i in range(table.num_rows):
        if kept >= PER_LANG:
            break
        row = table.slice(i, 1).to_pylist()[0]
        audio, text = row["audio"], (row["transcription"] or "").strip()
        if not text:
            continue
        try:
            data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        except Exception:  # noqa: BLE001
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != 16000 or len(data) / rate > MAX_SECS or len(data) / rate < 2.0:
            continue
        name = f"{lang}_{kept:02d}.wav"
        sf.write(f"{OUT}/{name}", data, rate, subtype="PCM_16")
        lang_rows.append({"lang": lang, "file": name,
                          "secs": round(len(data) / rate, 2), "reference": text})
        kept += 1
    manifest.extend(lang_rows)
    json.dump(lang_rows, open(f"{OUT}/done_{lang}.json", "w"), ensure_ascii=False)
    # Each FLEURS test parquet is ~150-400 MB and we keep only a handful of
    # clips from it, so drop it immediately rather than accumulating ~10 GB.
    os.remove(dst_parquet)
    print(f"  {lang}: {kept} clips", flush=True)

with open(f"{OUT}/manifest.json", "w") as fh:
    json.dump(manifest, fh, ensure_ascii=False, indent=1)
total = sum(m["secs"] for m in manifest)
print(f"\n{len(manifest)} clips across {len({m['lang'] for m in manifest})} languages, "
      f"{total/60:.1f} min audio -> {OUT}/manifest.json")
