#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


def summarize(values: torch.Tensor) -> dict:
    flat = values.detach().cpu().float().contiguous().view(-1)
    return {
        "frames": int(values.shape[0]),
        "dims": int(values.shape[1]),
        "sum": float(flat.double().sum()),
        "first": flat[:32].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    args = parser.parse_args()

    waveform, sample_rate = torchaudio.load(args.reference)
    waveform = waveform.mean(dim=0, keepdim=True)

    prompt = torchaudio.functional.resample(
        waveform, sample_rate, 24000
    ).squeeze(0)
    window = torch.hann_window(1920, periodic=True)
    padded = torch.nn.functional.pad(
        prompt.unsqueeze(0), (720, 720), mode="reflect"
    ).squeeze(0)
    stft = torch.stft(
        padded,
        n_fft=1920,
        hop_length=480,
        win_length=1920,
        window=window,
        center=False,
        return_complex=True,
    )
    magnitude = torch.sqrt(stft.real.square() + stft.imag.square() + 1e-9)
    mel_basis = torch.from_numpy(
        librosa.filters.mel(
            sr=24000,
            n_fft=1920,
            n_mels=80,
            fmin=0,
            fmax=8000,
            htk=False,
            norm="slaney",
        )
    ).float()
    mel = torch.log(torch.clamp(mel_basis @ magnitude, min=1e-5)).transpose(0, 1)

    campplus = torchaudio.functional.resample(
        waveform, sample_rate, 16000
    )
    fbank = torchaudio.compliance.kaldi.fbank(
        campplus,
        num_mel_bins=80,
        dither=0.0,
        sample_frequency=16000,
        window_type="povey",
        frame_length=25.0,
        frame_shift=10.0,
        snip_edges=True,
    )
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    print(json.dumps({"mel": summarize(mel), "fbank": summarize(fbank)}))


if __name__ == "__main__":
    main()
