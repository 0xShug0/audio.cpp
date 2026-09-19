#!/usr/bin/env python3
"""Source audio for AuK's 16 task types, at both rates the model needs.

⚠ A task tested on audio it cannot act on tests nothing. Asking "remove the background
noise" of a clean recording, or "keep the first speaker" of a single-speaker one, gives
output that cannot be judged: there is no noise to remove and no second speaker to drop.
The first pass of this demo did exactly that, and its results were unreadable as a
consequence.

So each fixture is chosen or built for the task that uses it, and everything synthesized
here says how it was made -- a degradation we introduced is one we can check the model
undid.

Every fixture is written twice: 16 kHz for the Thinker, 24 kHz for the VAE.
"""
import argparse, json, pathlib, sys
import numpy as np
import soundfile as sf


def load_mono(path, rate):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != rate:
        index = np.arange(int(len(audio) * rate / sr)) * sr / rate
        audio = np.interp(index, np.arange(len(audio)), audio).astype("float32")
    return audio


def normalize(audio, peak=0.7):
    highest = float(np.abs(audio).max())
    return audio if highest < 1e-6 else (audio * (peak / highest)).astype("float32")


def add_noise(clean, snr_db, seed=0):
    """Babble-ish noise at a stated SNR, so 'denoise' has something to remove."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, len(clean)).astype("float32")
    # tilt it away from white: real room noise is low-frequency heavy
    spectrum = np.fft.rfft(noise)
    tilt = 1.0 / (1.0 + np.linspace(0, 8, len(spectrum)))
    noise = np.fft.irfft(spectrum * tilt, n=len(clean)).astype("float32")
    speech_power = float((clean ** 2).mean())
    noise_power = float((noise ** 2).mean())
    scale = np.sqrt(speech_power / max(noise_power, 1e-12) / (10 ** (snr_db / 10)))
    return (clean + noise * scale).astype("float32")


def add_reverb(clean, rate, rt60=0.6, seed=1):
    """Exponentially decaying noise as a crude room impulse response."""
    rng = np.random.default_rng(seed)
    length = int(rate * rt60)
    impulse = rng.normal(0, 1, length).astype("float32") * np.exp(-np.linspace(0, 6, length))
    impulse[0] = 1.0
    # ⚠ FFT, not np.convolve. Direct convolution of 144k samples with a 14k-tap impulse
    # is ~2e9 multiply-adds for one six-second fixture -- it ran for 13 minutes across
    # seven cores before being killed. The FFT form is the same result in milliseconds.
    size = 1
    while size < len(clean) + len(impulse):
        size *= 2
    wet = np.fft.irfft(np.fft.rfft(clean, size) * np.fft.rfft(impulse, size), size)[: len(clean)]
    return normalize(0.6 * clean + 0.4 * wet.astype("float32"), peak=float(np.abs(clean).max()))


def overlap_speakers(first, second, rate, offset_s=1.0):
    """Two voices talking over each other, the second starting late."""
    offset = int(rate * offset_s)
    length = max(len(first), offset + len(second))
    mix = np.zeros(length, dtype="float32")
    mix[: len(first)] += first
    mix[offset : offset + len(second)] += second * 0.9
    return normalize(mix)


def band_limit(audio, rate, cutoff_hz=3000):
    """Drop the top end, so bandwidth extension has something to restore."""
    spectrum = np.fft.rfft(audio)
    frequencies = np.fft.rfftfreq(len(audio), 1.0 / rate)
    spectrum[frequencies > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum, n=len(audio)).astype("float32")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", type=pathlib.Path, required=True, help="audio.cpp checkout")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = args.engine / "assets"
    tests = args.engine / "tests"
    limit24 = int(args.seconds * 24000)

    clean = normalize(load_mono(src / "asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.wav", 24000))[:limit24]
    other = normalize(load_mono(src / "asr_validation/librispeech/librispeech_test_other_7902-96591-0001.wav", 24000))
    voices = normalize(load_mono(src / "resources/four_speaker_short.wav", 24000))[:limit24]
    reference = normalize(load_mono(tests / "omnivoice/assets/ref_audio_01.wav", 24000))[:limit24]
    music = normalize(load_mono(tests / "ace_step/assets/complete_source_demucs_8s.wav", 24000))[:limit24]
    second_voice = normalize(load_mono(tests / "parakeet_tdt/assets/2086-149220-0033.wav", 24000))[:limit24]

    fixtures = {
        # name: (audio at 24 kHz, what it is and why the task can act on it)
        "clean_speech": (clean, "LibriSpeech test-clean: read speech, no degradation"),
        "reference_voice": (reference, "a distinct speaker, for cloning a voice onto new text"),
        "noisy_speech": (add_noise(clean, snr_db=5.0), "clean speech + low-tilted noise at 5 dB SNR"),
        "reverb_speech": (add_reverb(clean, 24000), "clean speech through a synthetic RT60 0.6 s room"),
        "bandlimited_speech": (normalize(band_limit(clean, 24000, 3000)), "clean speech with everything above 3 kHz removed"),
        "two_speakers": (overlap_speakers(clean, second_voice, 24000), "two readers overlapping, the second entering at 1 s"),
        "multi_speaker": (voices, "four speakers in sequence"),
        "music_mix": (music, "a music excerpt with vocals over instruments"),
        "accented_speech": (normalize(np.tile(other, 3)[:limit24]), "LibriSpeech test-other: harder, more accented reading"),
        # ⚠ SUBSTITUTE, and labelled as one. The music excerpt has no intelligible vocal
        # -- ASR returns nothing from it and its side/mid ratio says the vocal is not
        # centred -- so "extract the singing" had nothing to extract. This mixes speech
        # over the music instead: a human voice over instruments, which the separation
        # tasks CAN act on and which ASR can verify afterwards. It is not singing, and a
        # real lyric-editing test still needs a recording of someone singing words.
        "voice_over_music": (normalize(0.7 * clean[:limit24] + 0.6 * music[:limit24]),
                             "clean speech mixed over the music bed (a stand-in for sung vocals)"),
    }

    manifest = {}
    for name, (audio24, description) in fixtures.items():
        index = np.arange(int(len(audio24) * 16000 / 24000)) * 24000 / 16000
        audio16 = np.interp(index, np.arange(len(audio24)), audio24).astype("float32")
        sf.write(str(args.out / f"{name}_24k.wav"), audio24, 24000)
        sf.write(str(args.out / f"{name}_16k.wav"), audio16, 16000)
        manifest[name] = {
            "description": description,
            "seconds": round(len(audio24) / 24000, 2),
            "rms": round(float(np.sqrt((audio24 ** 2).mean())), 4),
        }
        print(f"  {name:20s} {len(audio24)/24000:5.1f}s  rms={manifest[name]['rms']:.4f}  {description}")

    (args.out / "fixtures.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(fixtures)} fixtures (16 kHz and 24 kHz each) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
