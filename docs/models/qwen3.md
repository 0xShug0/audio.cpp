# Qwen3 Models

Qwen3 TTS, Qwen3 ASR, and Qwen3 Forced Aligner are separate model families
with different inputs and runtime contracts:

- [Qwen3 TTS](qwen3_tts.md) covers Base voice cloning, VoiceDesign,
  CustomVoice, sampling, and streaming.
- [Qwen3 ASR](qwen3_asr.md) covers transcription, long audio, streaming, and
  optional word alignment.
- [Qwen3 Forced Aligner](qwen3_forced_aligner.md) maps an exact transcript onto
  speech audio.

## GGUF checkpoints

All three families accept audio.cpp-native GGUF checkpoints. By default the
converter embeds JSON, tokenizer, processor, and chat-template sidecars found
beside the input checkpoint. Qwen3 ASR can therefore be distributed and loaded
as a standalone `model.gguf`:

```bash
audiocpp_gguf \
  --input models/Qwen3-ASR-1.7B-hf/model.safetensors \
  --output models/Qwen3-ASR-1.7B-hf/model.gguf \
  --type q8_0
```

On Windows:

```powershell
audiocpp_gguf.exe --input models\Qwen3-ASR-1.7B-hf\model.safetensors --output models\Qwen3-ASR-1.7B-hf\model.gguf --type q8_0
```

The forced aligner uses the same standalone layout:

```powershell
audiocpp_gguf.exe --input models\Qwen3-ForcedAligner-0.6B\model.safetensors --output models\Qwen3-ForcedAligner-0.6B-Q8_0\model.gguf --type q8_0
```

Pass the resulting GGUF file or its containing directory to
`qwen3_asr.forced_aligner_path`; no external sidecars are required.

Sidecar embedding is recursive and binary-safe, so nested tokenizer assets are
portable too. Pass `--no-sidecars` to produce the older tensor-only layout. The
converter supports `f16`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, and `q6_k`.
It quantizes eligible projection matrices but keeps embedding/codebook lookup
tables in F16 and leaves shapes that cannot use the selected block format
unquantized. This mixed layout works across more ggml backends than blindly
quantizing every 2-D tensor. Qwen loaders prefer `model.gguf` when both formats
are present. Use `--overwrite` to replace an existing output.

For Qwen3 TTS, convert the main `model.safetensors` and the separate
`speech_tokenizer/model.safetensors` independently, placing each output beside
its source as `model.gguf`.

Quantization can reduce the reliability of Qwen3 ASR automatic language
detection even when transcription quality remains good with a language hint.
For known-language audio, pass `--language` explicitly; use `f16` when maximum
parity with the original checkpoint is more important than file size.

Older tensor-only GGUF files still require configuration, generation settings,
processor files, and tokenizer files beside them and remain backward
compatible. GGUF files produced for llama.cpp or whisper.cpp do not
automatically work because those projects use architecture-specific tensor
names and metadata.

With word timestamps:

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf --backend cuda --audio assets/resources/sample_16k.wav --language English --text "" --text-out transcript.txt --words-out words.json --session-option qwen3_asr.forced_aligner_path=models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf --session-option qwen3_asr.vad_model_path=assets/framework/models/silero_vad
```
