# Audio8-ASR-0.1B in audio.cpp

[Audio8-ASR-0.1B](https://huggingface.co/Audio8/Audio8-ASR-0.1B) is a compact
multilingual autoregressive ASR model (en / zh / yue / ja / ko / fr / de): a
Qwen3-ASR audio encoder adapted by an MLP tower into an 8-layer Qwen2-style
decoder with only ~103M language-model parameters (324M end-to-end). The
checkpoint ships under **CC-BY-NC-4.0**, so audio.cpp loads it from locally
converted weights only; the converted GGUF must not be redistributed.

## Architecture

- **Audio frontend**: 16 kHz Whisper log-mel, 128 bins, hop 160, n_fft 400
  (shared with the `qwen3_asr` family). The reference processor emits
  bfloat16 mel values; the audio8_asr frontend rounds to bfloat16 before
  encoding to match.
- **Audio encoder**: Qwen3-ASR audio tower (d_model 896, 18 layers, 14 heads,
  FFN 3584, 128 mel bins, output dim 1024) — bit-for-bit the same
  architecture as `Qwen/Qwen3-ASR-0.6B`, loaded through the shared
  `qwen3_asr` encoder implementation.
- **Adapter**: 4 pre-norm residual MLP blocks (1024 → 4096, erf GELU), a
  final LayerNorm, then an adaptive average pool (merge factor 4 against the
  mel-frame count) followed by LayerNorm + Linear(1024 → 512).
- **Decoder**: Qwen2-style causal LM, 8 layers, hidden 512, 8 heads,
  head_dim 64, SwiGLU FFN 1408, tied embeddings over a 151,936 Qwen BPE
  vocabulary, RoPE theta 1e6, RMSNorm eps 1e-6.
- **Prompt**: `<|user|><|begin_of_audio|><|audio|>×N<|end_of_audio|>Please
  transcribe this audio.<|assistant|>` where
  `N = max(floor(floor((floor(samples / hop) + 1) / 2) / merge_factor), 1)`
  (all divisions integer).

## Usage

```bash
# Convert locally (requires the audiocpp_gguf tool and a self-downloaded
# checkpoint from the Audio8/Audio8-ASR-0.1B HF repository):
python tools/community_models/convert_audio8_asr.py \
    --checkpoint models/Audio8-ASR-0.1B-hf \
    --converter build/debug/bin/audiocpp_gguf \
    --type q8_0 \
    --output models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf

# Transcribe
audiocpp_cli --task asr --family audio8_asr \
    --model models/Audio8-ASR-0.1B-GGUF/audio8-asr-0.1b-q8_0.gguf \
    --audio sample.wav

# The safetensors package loads directly, no conversion required:
audiocpp_cli --task asr --family audio8_asr \
    --model models/Audio8-ASR-0.1B-hf --audio sample.wav
```

## Parity

`tools/community_models/audio8_asr_reference.py` runs the Hugging Face
`trust_remote_code` reference (torch CPU, fp32) and
`tools/community_models/audio8_asr_stages.py` captures staged tensors (mel,
encoder output, projected audio embeddings) for comparison.

Greedy transcription on the repo test clips matched the fp32 reference
exactly with the Q8_0 GGUF on both the Metal and CPU backends:

| Audio | Reference (fp32) | audio.cpp (Q8_0 GGUF, Metal + CPU) |
|---|---|---|
| `assets/resources/a.wav` (5.95 s) | "This little work was finished in the year eighteen o three, and intended for immediate publication." | identical |
| `assets/resources/sample_16k.wav` (14.07 s) | "Some call me nature. Others call me Mother Nature. I've been here for over four point five billion years, twenty-two thousand five hundred times longer than you." | identical |

A 61-second clip transcribed through rate-correct 30-second windows matched
the per-window reference transcripts (also verified by an independent review
pass). `test_audio8_asr_golden_transcription` asserts the first row
end-to-end — run it manually once weights exist (it is not part of ctest,
matching the granite5asr golden-test convention); `test_audio8_asr_units`
covers the token-count formula and bfloat16 rounding and runs under ctest.

## Known limitations

- **Offline only**: no streaming mode, no word timestamps, no language-id
  output.
- **30-second windows**: audio longer than 30 s is transcribed in fixed
  windows sized at the input sample rate (0.5 s minimum tail folds into the
  previous window) and space-joined, without VAD segmentation. Unlike the
  single-pass reference, each window is peak-normalized independently, so
  relative loudness across a window boundary can shift.
- **CC-BY-NC-4.0**: non-commercial use only; convert locally, do not
  redistribute the converted GGUF. There is no release GGUF package and no
  WebUI catalog entry (the package manager cannot download
  unsupported-license packages).
- Hotword logit boosting from the reference implementation is not ported.
