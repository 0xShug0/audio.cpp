# AudioCPP Command Usage

Use `audiocpp_cli` for direct model inference.

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend <backend> [inputs] [outputs]
```

## Common Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task` | `gen`, `tts`, `clon`, `vc`, `svc`, `s2s`, `asr`, `align`, `vad`, `diar`, `sep`, `vdes` | required | User task. |
| `--family` | model family name | required | Selects the model implementation. Must match a registered loader (`audiocpp_cli --list-loaders`). |
| `--model` | local model directory | required | Path to local model assets. |
| `--backend` | `cpu`, `cuda`, `vulkan`, `metal`, `best` | `cpu` | Inference backend. |
| `--mode` | `offline`, `streaming` | `offline` | Run mode. Most models are offline. |
| `--device` | integer | `0` | Backend device index. |
| `--threads` | integer | `4` | Backend/OpenMP worker threads. |
| `--log` | flag | off | Print progress and timing logs to stdout. |
| `--log-file` | path | not set | Stream progress and timing logs to a file. |

## Common Inputs And Outputs

| Option | Used by | Meaning |
|---|---|---|
| `--text` | generation, TTS, ASR context, alignment transcript | Input text. |
| `--audio` | generation/editing, ASR, VAD, diarization, separation, conversion, alignment | Input WAV. |
| `--voice-ref` | voice clone / voice design / some VC paths | Reference voice WAV. |
| `--language` | language-aware models | Language code. |
| `--out` | audio-producing models | Output WAV path. |
| `--out-dir` | multi-output or batch models | Output directory. |
| `--segments-out` | VAD | Speech segments JSON. |
| `--vad-chunks-out` | offline VAD | VAD-based audio chunk windows JSON. |
| `--turns-out` | diarization | Speaker turns JSON. |
| `--words-out` | ASR/alignment | Word timestamps JSON. |
| `--audio-chunk-seconds` | ASR | Split long audio before model inference, where supported. |
| `--audio-chunk-mode` | ASR/alignment | `auto`, `fixed`, `vad`, or `none`, where supported. |

## Common Generation Options

Omit these unless you need explicit control. If `--seed` is omitted, models that sample use a random seed.

| Option | Values | Meaning |
|---|---|---|
| `--seed` | integer | Reproducible random seed. |
| `--max-tokens` | integer | Maximum generated tokens for AR/LLM-style models. |
| `--max-steps` | integer | Maximum diffusion or generation steps for models that expose it. |
| `--temperature` | float | Sampling temperature. |
| `--top-k` | integer | Top-k sampling limit. |
| `--top-p` | float in `(0, 1]` | Nucleus sampling limit. |
| `--repetition-penalty` | float | Penalize repeated tokens. |
| `--do-sample` | `true`, `false` | Enable sampling instead of greedy decode. |
| `--guidance-scale` | float | Classifier-free guidance scale. |
| `--num-inference-steps` | integer | Diffusion/flow denoising steps. |
| `--text-chunk-size` | integer chars | Split long text where supported, including TTS text. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | Select text chunking mode where supported. |

## Batch Inputs

| Option | Meaning |
|---|---|
| `--batch-text-file <txt>` | One request per non-empty text line. |
| `--batch-text-dir <dir>` | One request per `.txt`, `.md`, or `.json` file; each file is normalized into a single paragraph. |
| `--batch-audio-dir <dir>` | One request per `.wav` file. |
| `--batch-audio-role audio|voice_ref|source_audio|target_voice|prosody_ref|style_ref` | How to use each batch WAV. |
| `--batch-merge-audio none|concat` | Keep outputs separate or concatenate generated audio. |
| `--batch-manifest-out <json>` | Write a batch output manifest. |

`--batch-text-dir` reads `.txt` and `.md` files as plain text. For `.json`, use either a JSON string root or an object with a string `input` or `text` field.

## Model Docs

| Need | Doc |
|---|---|
| Speech, voice clone, long-form TTS | [tts.md](tts.md) |
| Music and sound generation | [music_generation.md](music_generation.md) |
| ASR models | [asr.md](asr.md) |
| VAD and diarization | [speech_analysis.md](speech_analysis.md) |
| Voice conversion codec and source separation | [audio_tools.md](audio_tools.md) |
