# XTTS v2

XTTS v2 is Coqui's cross-lingual text-to-speech model with voice cloning. The
audio.cpp execution path runs the full offline pipeline on GGUF weights: GPT-2
mel-code generator with perceiver-resampled voice conditioning, latent
extraction, and the HiFi-GAN decoder with speaker-encoder conditioning.

| Field | Value |
|---|---|
| Family | `xtts_v2` |
| Model directory | `models/XTTS-v2` |
| Task | `tts` |
| Modes | `offline` |
| Languages | 17 (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi) |
| Voice input | Required speaker reference WAV (6-30 s recommended) |
| Output | mono 24 kHz WAV |

`model_specs/xtts_v2.json` is the source of truth for metadata, packages, and
normalized request options.

## Weights

Two packages are available: a Q8_0-quantized GPT (recommended, ~518 MB
total) and the full-precision F16/F32 original (~1.75 GB total). Each bundles:

- `xtts-v2-gpt[-q8_0|-f16].gguf` - GPT-2 generator (text/mel embeddings, 30
  transformer layers, perceiver resampler, conditioning encoder, BPE vocab).
- `xtts-v2-vocoder-f16.gguf` - HiFi-GAN decoder and SE-ResNet speaker encoder
  (kept F16 in both packages; it is small and quality-sensitive).

The upstream Coqui release is distributed under CC-BY-NC-4.0 (see
[License](#license)). Convert official checkpoints with the Coqui `model.pth`
(v2.0.3) using the GGUF export path; the layout is identical to the files in
the default package.

## Usage

```bash
audiocpp_cli --task tts --family xtts_v2 --model models/XTTS-v2 \
    --backend cpu --threads <n> \
    --text "Hello, my name is Anna and this is a test." \
    --voice-ref reference.wav \
    --out out.wav
```

Request options (also settable via `--request-option`): `language`,
`temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_mel_tokens`.
Defaults follow the official Coqui inference settings (temperature 0.75,
top-k 50, top-p 0.85, repetition penalty 10.0, max 605 mel codes).

The reference WAV may be any sample rate; it is resampled internally. For
best cloning quality use a clean 6-30 second recording of a single speaker,
matching the official recommendation.

## Implementation notes

- Conditioning: reference audio is truncated to 30 s, split into 6 s chunks;
  each chunk goes through the ConditioningEncoder and PerceiverResampler, and
  the resulting latents are averaged (matches `get_conditioning_latents`).
- The speaker encoder consumes the first 30 s of the reference at 16 kHz with
  pre-emphasis, Hamming-windowed log-mel features, and attentive statistics
  pooling.
- Generation uses causal attention with a KV cache; the text/conditioning
  prefix is processed in a single batched pass.
- Latents are extracted teacher-forced over the generated codes and decoded
  by HiFi-GAN with speaker conditioning at all five injection points.

## License

The XTTS v2 model weights are released by Coqui under CC-BY-NC-4.0
(https://creativecommons.org/licenses/by-nc/4.0/) - free for non-commercial
use. This execution code contains no Coqui-derived source; it re-implements
the published architecture against GGUF tensors.
