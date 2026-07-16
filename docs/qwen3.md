# Qwen3 Models

This page covers Qwen3 TTS, Qwen3 ASR, and Qwen3 forced alignment. These are separate model packages with different inputs: Base TTS clones from audio, VoiceDesign uses an instruction, CustomVoice uses packaged speakers, ASR transcribes audio, and the aligner requires an exact transcript.

## Qwen3 TTS Base

Qwen3 TTS Base is the voice-clone TTS path. It needs reference audio and can use the reference transcript when available.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-Base` |
| Task | `tts` |
| Modes | `offline` |
| Voice input | Reference WAV through `--voice-ref` |
| Transcript | Optional `--reference-text` |

```bash
audiocpp_cli --task tts --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-Base --backend cuda --text "Hello from Qwen3 TTS." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--voice-ref` | WAV path | required | Voice clone reference audio. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. |
| `--max-tokens` | integer | `8192` | Maximum generated speech tokens per chunk. |

## Qwen3 TTS VoiceDesign

Qwen3 VoiceDesign creates a voice from an instruction. It does not require a speaker reference WAV.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Task | `vdes` |
| Modes | `offline` |
| Voice input | Instruction text through `--instruct` |
| Reference audio | Not required |

```bash
audiocpp_cli --task vdes --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --backend cuda --text "Hello from a designed voice." --instruct "A warm adult narrator" --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--instruct` | text | required | Voice design instruction. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. |
| `--max-tokens` | integer | `8192` | Maximum generated speech tokens per chunk. |

## Qwen3 TTS CustomVoice

Qwen3 CustomVoice uses speaker ids packaged with the model. The CLI passes the speaker name with `--speaker`; examples use `Vivian` and `Ryan`.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Task | `tts` |
| Modes | `offline` |
| Voice input | Built-in speaker id through `--speaker` |
| Style control | Optional instruction through `--instruct` |
| External voice WAV | Not used by this path |

```bash
audiocpp_cli --task tts --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice --backend cuda --text "Hello from a custom voice." --speaker Vivian --instruct "Very happy." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--speaker` | packaged speaker name | required | Built-in speaker id, such as `Vivian` or `Ryan`. |
| `--instruct` | text | empty string | Style or emotion instruction. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. |
| `--max-tokens` | integer | `8192` | Maximum generated speech tokens per chunk. |

## Qwen3 TTS Sampling

These sampling controls are shared by the Qwen3 TTS Base, VoiceDesign, and CustomVoice paths.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--do-sample` | `true`, `false` | `true` | Enable sampling. |
| `--temperature` | float | `0.9` | Main talker temperature. |
| `--top-k` | integer | `50` | Main talker top-k. |
| `--top-p` | float | `1.0` | Main talker top-p. |
| `--repetition-penalty` | float | `1.05` | Main talker repetition penalty. |
| `--request-option subtalker_do_sample=true|false` | bool | `true` | Subtalker sampling. |
| `--request-option subtalker_temperature=<float>` | float | `0.9` | Subtalker temperature. |
| `--request-option subtalker_top_k=<n>` | integer | `50` | Subtalker top-k. |
| `--request-option subtalker_top_p=<float>` | float | `1.0` | Subtalker top-p. |
| `--seed` | integer | random if omitted | Sampling seed. |

## Qwen3 ASR

Qwen3 ASR transcribes speech audio. Word timestamps are produced by running the
recognized transcript through Qwen3 Forced Aligner, so `--words-out` requires
the forced aligner model path. Long audio is split inside the model session
before ASR inference; word timestamps are shifted back onto the original audio
timeline.

`audio_chunk_mode=auto` is the default. For transcript-only ASR, Qwen3 ASR uses
fixed chunks. When word timestamps are requested, it uses bundled Silero VAD
internally to choose speech-aware chunks before running ASR and alignment.

| Field | Value |
|---|---|
| Family | `qwen3_asr` |
| Model directory | `models/Qwen3-ASR-0.6B` or `models/Qwen3-ASR-1.7B-hf` |
| Task | `asr` |
| Modes | `offline` |
| Input | Speech WAV through `--audio` |
| Output | Text to stdout or `--text-out`; optional word JSON through `--words-out` with a forced aligner |

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-0.6B --backend cuda --audio speech_16k.wav --text "" --text-out transcript.txt
```

The native Hugging Face Transformers layout of `Qwen/Qwen3-ASR-1.7B-hf` is
also accepted directly. It uses `processor_config.json`, `tokenizer.json`, and
the `model.audio_tower` / `model.language_model` tensor namespaces; no model
conversion is required:

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-1.7B-hf --backend cuda --audio speech_16k.wav --text "" --text-out transcript.txt
```

### GGUF checkpoints

Qwen3 ASR, Qwen3 Forced Aligner, and Qwen3 TTS accept audio.cpp-native GGUF
checkpoints. By default the converter embeds JSON, tokenizer, processor, and chat
template sidecars found beside the input checkpoint. Qwen3 ASR can therefore be
distributed and loaded as a standalone `model.gguf`:

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
`qwen3_asr.forced_aligner_model_path`; no external sidecars are required.

Sidecar embedding is recursive and binary-safe, so nested tokenizer assets are
portable too. Pass `--no-sidecars` to produce the older tensor-only layout. The converter supports
`f16`, `q8_0`, `q2_k`, `q3_k`, `q4_k`, `q5_k`, and
`q6_k`. It quantizes eligible projection matrices but keeps embedding/codebook
lookup tables in F16 and leaves shapes that cannot use the selected block format
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
processor files, and tokenizer files beside them and remain backward compatible.
GGUF files produced for llama.cpp or whisper.cpp do not automatically work because
those projects use architecture-specific tensor names and metadata.

With word timestamps:

```bash
audiocpp_cli --task asr --family qwen3_asr --model models/Qwen3-ASR-0.6B --backend cuda --audio speech_16k.wav --text "" --text-out transcript.txt --words-out words.json --session-option qwen3_asr.forced_aligner_model_path=models/Qwen3-ForcedAligner-0.6B
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech audio; use 16 kHz WAV for the examples. |
| `--text` | text | empty string | Context prompt. |
| `--language` | language code | empty string | Recognition language hint. |
| `--max-tokens` | integer | `512` | Maximum decode tokens. |
| `--audio-chunk-seconds` | float seconds | `30`, or `15` with `--words-out` | Target/max chunk length used before ASR inference. |
| `--audio-chunk-mode` | `auto`, `fixed`, `vad`, `none` | `auto` | Let the model choose, force fixed chunks, force internal VAD chunks, or disable model-side chunking. |
| `--text-out` | TXT path | not set | Transcript output. The transcript is also printed to stdout. |
| `--words-out` | JSON path | not set | Word timestamp output. Requires `qwen3_asr.forced_aligner_model_path`. |
| `--session-option qwen3_asr.forced_aligner_model_path=<path>` | model directory | not set | Qwen3 Forced Aligner model used to generate word timestamps after ASR. |
| `--session-option qwen3_asr.vad_model_path=<path>` | model directory | `assets/framework/models/silero_vad` | Optional internal VAD model override for timestamp-safe chunking. |

## Qwen3 Forced Aligner

The forced aligner maps an exact transcript onto speech audio. It is not an ASR route: the transcript is required input.
Standalone forced alignment does not chunk audio because exact transcript/audio
chunk pairing cannot be inferred safely from a raw transcript. For long-audio
timestamping, use Qwen3 ASR with `--words-out` and
`qwen3_asr.forced_aligner_model_path`; ASR chunks the audio first, then aligns
each recognized transcript to its matching audio chunk.

| Field | Value |
|---|---|
| Family | `qwen3_forced_aligner` |
| Model directory | `models/Qwen3-ForcedAligner-0.6B` |
| Task | `align` |
| Modes | `offline` |
| Input | Speech WAV plus exact transcript |
| Output | Word timestamp JSON through `--words-out` |

```bash
audiocpp_cli --task align --family qwen3_forced_aligner --model models/Qwen3-ForcedAligner-0.6B --backend cuda --audio speech_16k.wav --text "The exact transcript." --language en --words-out words.json
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech audio; use 16 kHz WAV for the examples. |
| `--text` | exact transcript | required | Transcript to align. |
| `--language` | language code | required | Transcript language. |
| `--audio-chunk-mode` | `auto`, `none` | `auto` | Standalone forced alignment runs one audio/transcript pair. `fixed` and `vad` are rejected because transcript chunk boundaries would be ambiguous. |
| `--words-out` | JSON path | not set | Word timestamp output. |

## Weight Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--session-option qwen3_tts.mem_saver=true|false` | bool | `false` | Release the TTS talker cached-step graph after each request to reduce post-request resident VRAM. Later requests rebuild that graph; voice prompt, prefill, code predictor, and speech decoder caches stay reusable. |
| `--session-option qwen3_tts.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | TTS graph weight type. |
| `--session-option qwen3_asr.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | ASR thinker weight type. |
| `--session-option qwen3_forced_aligner.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Aligner thinker weight type. |
