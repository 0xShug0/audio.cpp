#!/usr/bin/env python3
"""Fetch small FLEURS speech fixtures without downloading entire archives."""

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--languages", nargs="+", default=["en_us", "kk_kz", "uz_uz"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for language in args.languages:
        url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{language}/audio/test.tar.gz"
        with urllib.request.urlopen(url) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".wav"):
                        continue
                    data = archive.extractfile(member).read()
                    samples, rate = sf.read(io.BytesIO(data), dtype="float32")
                    if rate != 16000 or samples.ndim != 1:
                        raise ValueError(f"Unexpected FLEURS audio layout: {language}")
                    path = args.output / (language + ".wav")
                    sf.write(path, samples, rate, subtype="PCM_16")
                    record = {"language": language, "url": url, "member": member.name,
                              "source_sha256": hashlib.sha256(data).hexdigest(),
                              "audio": str(path), "duration_sec": len(samples) / rate}
                    manifest.append(record)
                    print(json.dumps(record), flush=True)
                    break
                else:
                    raise ValueError(f"No WAV found in {url}")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    requests = [{"id": row["language"], "audio": row["audio"]} for row in manifest]
    case = {"id": "gigaam_languages", "family": "gigaam_asr", "task": "asr",
            "model": "GigaAM-ASR-GGUF/gigaam-multilingual-ctc-f32.gguf",
            "coverage": "Independent FLEURS multilingual recordings.",
            "outputs": ["text"], "requests": requests}
    (args.output / "path_cases.json").write_text(json.dumps({"cases": [case]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
