#!/usr/bin/env python3
"""Create a reproducible multilingual Kroko smoke-test set with Edge TTS."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

import edge_tts


SAMPLES = {
    "de": (
        "de-DE-KatjaNeural",
        "Heute testen wir die lokale Spracherkennung mit einem kurzen und klaren Satz.",
    ),
    "en": (
        "en-US-JennyNeural",
        "Today we are testing local speech recognition with a short and clear sentence.",
    ),
    "es": (
        "es-ES-ElviraNeural",
        "Hoy probamos el reconocimiento de voz local con una frase corta y clara.",
    ),
    "fr": (
        "fr-FR-DeniseNeural",
        "Aujourd'hui, nous testons la reconnaissance vocale locale avec une phrase courte et claire.",
    ),
    "it": (
        "it-IT-ElsaNeural",
        "Oggi proviamo il riconoscimento vocale locale con una frase breve e chiara.",
    ),
    "he": (
        "he-IL-HilaNeural",
        "היום אנחנו בודקים זיהוי דיבור מקומי באמצעות משפט קצר וברור.",
    ),
    "nl": (
        "nl-NL-ColetteNeural",
        "Vandaag testen we lokale spraakherkenning met een korte en duidelijke zin.",
    ),
    "pt": (
        "pt-BR-FranciscaNeural",
        "Hoje testamos o reconhecimento de fala local com uma frase curta e clara.",
    ),
    "sv": (
        "sv-SE-SofieNeural",
        "I dag testar vi lokal taligenkänning med en kort och tydlig mening.",
    ),
    "tr": (
        "tr-TR-EmelNeural",
        "Bugün yerel konuşma tanımayı kısa ve anlaşılır bir cümleyle test ediyoruz.",
    ),
}


async def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    for language, (voice, text) in SAMPLES.items():
        mp3 = output / f"{language}.mp3"
        wav = output / f"{language}.wav"
        await edge_tts.Communicate(text, voice).save(str(mp3))
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(mp3),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(wav),
            ],
            check=True,
        )
        mp3.unlink()
        manifest[language] = {
            "voice": voice,
            "text": text,
            "audio": wav.name,
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(generate(args.output))


if __name__ == "__main__":
    main()
