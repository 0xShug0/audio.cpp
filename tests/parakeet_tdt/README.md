# Parakeet TDT test inputs

The golden and streaming tests use the shared 16 kHz mono LibriSpeech clip:
`assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav`.
Its dataset transcript, source identity and CC BY 4.0 attribution live beside it.
There is no Parakeet-local copy of the recording.

The exact punctuated transcript in the C++ assertions was independently captured
in FP32 with `transformers.ParakeetForTDT` 5.12.1 and the official NVIDIA
checkpoint's generation configuration, pinned by the generator. The same
reference reproduced the previous NeMo-verified fixture transcript before the
input was replaced. Expected text is never populated from the C++ implementation.

To recapture the shared input:

```bash
conda run --no-capture-output -n qwen3-tts python \
  tests/parakeet_tdt/parity/dump_transcript_reference.py \
  --model models/parakeet-tdt-0.6b-v3 \
  --output build/logs/parakeet-reference.json \
  > build/logs/parakeet-reference.log 2>&1
```

The generator records the input SHA-256, reference version and generation-config
revision. Changing the input requires a new independent capture, not replacement
of only the path or expected text. `parity/dump_nemo_reference.py` remains the
separate NeMo activation-reference generator.

The native tests retain full-context/windowed transcription, word timestamps,
streaming partials, finalize and reset assertions. Both accept `--model` for an
installed Safetensors directory or standalone GGUF.
