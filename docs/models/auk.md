# AuK

Tencent AuK base and AuK-Flash are available as experimental `--family auk`
models for offline speech generation and instruction-based audio editing.
Output is mono 24 kHz.

## Model

Place the component GGUFs and sidecars in one model directory:

```text
models/AuK-Base-and-Flash-GGUF/
  config/auk-base.yaml
  config/auk-flash.yaml
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  qwen2.5-omni-3b-bf16.gguf
  qwen2.5-omni-3b-q8_0.gguf
  auk-base-f32.gguf
  auk-base-f16.gguf
  auk-base-q8_0.gguf
  auk-flash-f32.gguf
  auk-flash-f16.gguf
  auk-flash-q8_0.gguf
  auk-vae-f32.gguf
```

Pass the directory to `--model`; choose components with the session options
below. The defaults are Qwen BF16, AuK base F32, and VAE F32. Set
`auk.variant=flash` to use the upstream fixed four-step AuK-Flash schedule with
guidance disabled. The Qwen component is Qwen2.5-Omni-3B, not Qwen3.
The native model manager exposes each generator, Qwen, and VAE component as a
separate download. Model details and instruction examples are available from [Tencent AuK](https://huggingface.co/tencent/AuK),
[AuK-Flash](https://huggingface.co/tencent/AuK-Flash), and the
[upstream cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md).

## Speech Generation

For instruction TTS, pass the words in `--text` and a voice description in
`instruct`:

```bash
audiocpp_cli --task tts --family auk \
  --model models/AuK-Base-and-Flash-GGUF --backend cuda \
  --text "The next train leaves in ten minutes." \
  --request-option "instruct=A warm male voice speaking clear English." \
  --request-option duration_sec=4 --seed 42 --out speech.wav
```

For zero-shot TTS, provide reference speech with `--voice-ref`. Without
`instruct`, `--text` must contain the complete upstream instruction, not
just the words to speak.

## Audio Editing

Editing uses the CLI generation route, `--task gen`, with the instruction
in `--text` and source audio in `--audio`:

```bash
audiocpp_cli --task gen --family auk \
  --model models/AuK-Base-and-Flash-GGUF --backend cuda \
  --audio input.wav \
  --text "Remove the background noise and make the voice cleaner" \
  --seed 42 --out enhanced.wav
```

The same route accepts speech-content, lyric, pitch, speed, volume, emotion,
timbre, de-accent, nonverbal and whisper edits, speech enhancement, speech
and music separation, and target-speaker extraction. Use the upstream
cookbook for task-specific instructions. Do not pass `instruct` for editing.

## Common Options (use directly)

| Option | Default | Meaning |
|---|---|---|
| `--seed` | `-1` | Nonnegative seed, or `-1` for a random seed. |

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `instruct` | Unset | TTS voice description; wraps `text` in the upstream instruction. |
| `duration_sec` | Reference duration | Positive output duration in seconds; required without reference audio. |
| `num_inference_steps` | `32` | Base Euler steps; AuK-Flash always uses four. |
| `guidance_scale` | `2` | Base guidance strength; AuK-Flash always uses zero. |
| `sway_sampling_coef` | `-1` | Base schedule sway; AuK-Flash uses its fixed time grid. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `auk.variant` | `base` | Select the Base or AuK-Flash schedule and default generator. |
| `auk.model_gguf` | Variant F32 | Select an AuK Base or AuK-Flash generator GGUF from the model directory. |
| `auk.qwen_gguf` | `qwen2.5-omni-3b-bf16.gguf` | Select the Qwen conditioning GGUF. |
| `auk.vae_gguf` | `auk-vae-f32.gguf` | Select the VAE GGUF. |
| `auk.attention` | `auto` | Flow-model attention mode. |
| `auk.mem_saver` | `false` | Reduce peak VRAM by releasing conditioning and generation weights between stages. Later requests reload weights and rebuild graphs, so warm requests are much slower. |

Durations are rounded up to 20 ms audio frames. Set the duration explicitly
when a speed edit should change output length. Input audio is resampled
internally. Streaming is not supported.

For example, to use AuK-Flash F16 with Qwen Q8_0, add
`--session-option auk.variant=flash --session-option auk.model_gguf=auk-flash-f16.gguf`
and `--session-option auk.qwen_gguf=qwen2.5-omni-3b-q8_0.gguf`.
