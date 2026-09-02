# MiraTTS

MiraTTS is an experimental community port of
[ysharma3501/MiraTTS](https://github.com/ysharma3501/MiraTTS), a zero-shot
voice-cloning text-to-speech model. The native path includes the Qwen2 speech
token generator, ECAPA-TDNN plus Perceiver reference encoder, finite-scalar
speaker tokenizer, conditional acoustic processor, and DAC waveform decoder.

## Model and license

The upstream checkpoint is
[YatharthS/MiraTTS](https://huggingface.co/YatharthS/MiraTTS). Its model card
declares `CC-BY-NC-SA-4.0`; this is a non-commercial, attribution, share-alike
license. Review that license before downloading or redistributing converted
weights. audio.cpp does not redistribute the checkpoint.

No ready-to-run GGUF package is published yet, so MiraTTS intentionally has no
entry in the built-in download catalog. Convert a locally obtained upstream
checkpoint with:

```bash
python tools/community_models/convert_mira_tts.py /path/to/MiraTTS /path/to/mira-native
audiocpp_gguf \
  --input language_model=/path/to/mira-native/language_model.safetensors \
  --input speaker_encoder=/path/to/mira-native/speaker_encoder.safetensors \
  --input processor=/path/to/mira-native/processor.safetensors \
  --input decoder=/path/to/mira-native/decoder.safetensors \
  --input upsampler=/path/to/mira-native/upsampler.safetensors \
  --output /path/to/mira-native/mira-tts.gguf --type bf16 \
  --family mira_tts --root /path/to/mira-native
```

Use BF16 for the first parity-oriented conversion. Converting the natively
BF16 Qwen backbone to F16 can overflow and produce non-finite logits.

The converter also imports the official FastBiCodec and FlashSR component
checkpoints referenced by the upstream repository. Use `--help` to see its
component path overrides.

## Run

MiraTTS requires reference audio. The CLI voice-cloning request accepts the
converted model directory, target text, and a short clean reference WAV through
the normal audio.cpp TTS/clone arguments. Sampling defaults reproduce upstream:
temperature `0.8`, top-k `50`, top-p `0.95`, min-p `0.05`, and repetition
penalty `1.2`.

The upstream pipeline decodes at 16 kHz and applies its learned FlashSR
upsampler. The native runtime executes both stages and returns 48 kHz audio.

MiraTTS also exposes a streaming session. It splits long input at natural text
boundaries, reuses one encoded speaker identity for the whole request, and emits
each completed 48 kHz segment immediately. `text_chunk_size` controls the
maximum segment size (160 codepoints by default), while `text_chunk_mode`
selects the framework chunker. This is segment-level progressive synthesis;
the acoustic processor, DAC, and FlashSR still decode each segment as a unit.

## Validation status

- Official checkpoint conversion: validated.
- Native CUDA build: validated.
- Native CUDA smoke synthesis through all converted model components: validated.
- Segment-level streaming synthesis: validated through the native streaming
  session and `/v1/audio/speech/live` route.
- Deterministic upstream comparison with identical speech/context tokens:
  validated (48 kHz waveform correlation 0.99996, SNR 41.1 dB).
- End-to-end generation comparison: validated through matching tokenization,
  identical 32-token speaker codes, and the first six greedy LM tokens. Later
  autoregressive tokens can diverge between LMDeploy, Transformers, and the
  native backend because of backend floating-point differences.

Until end-to-end measurements are published, the family remains experimental
and is not advertised as an installable WebUI package.
