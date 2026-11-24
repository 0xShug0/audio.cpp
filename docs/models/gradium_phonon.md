# Gradium Phonon

`gradium_phonon` runs Gradium's Phonon text-to-speech models. Phonon is a small
flow-matching language model with a Mimi decoder, a descendant of Kyutai's PocketTTS,
so this family reuses the `pocket_tts` graphs and session. Output is mono 24 kHz.

The default model is Phonon 31m English
([gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en), Apache-2.0).
French and multilingual 40m packages are also available:

| Package | Source | Weights |
|---|---|---|
| `gradium_phonon_31m_en_q8_0` (default) | [gradium/phonon-31m-en-audiocpp](https://huggingface.co/gradium/phonon-31m-en-audiocpp) | GGUF: flow transformer linears Q8_0, the rest as published |
| `gradium_phonon_31m_en_safetensors` | [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en), as published | Safetensors: BF16 flow model, F32 Mimi |
| `gradium_phonon_40m_fr_q8_0` | [audio.cpp GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Gradium-Phonon-GGUF/40m-fr), from [Gradium](https://huggingface.co/gradium/phonon-40m-fr) | Q8_0 flow transformer; French |
| `gradium_phonon_40m_multilingual_q8_0` | [audio.cpp GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Gradium-Phonon-GGUF/40m-multilingual), from [Gradium](https://huggingface.co/gradium/phonon-40m-multilingual) | Q8_0 flow transformer; English, French, German, Spanish, Portuguese |

> [!WARNING]
> The 40m weights, tokenizer and bundled voices use **CC BY-NC 4.0: noncommercial use only**. See each package's README and LICENSE. Conversion, quantization and optional download do not remove this restriction. The 31m packages retain their existing Apache-2.0 license.

The 40m checkpoints use the same implementation with nine layers. Packages install
under `models/Gradium-Phonon-GGUF/40m-fr` or
`models/Gradium-Phonon-GGUF/40m-multilingual`, with a default voice beside the GGUF.

## Usage

```bash
python3 tools/model_manager_v2.py install gradium_phonon_31m_en_q8_0 --models-root models

audiocpp_cli --task tts --family gradium_phonon \
  --model models/phonon-31m-en --backend metal \
  --text "Hello from Phonon." --out out.wav
```

The package's `default-voice.safetensors` is the `default` voice. `--mode streaming`
emits one audio event per generated frame. The safetensors package installs to
`models/phonon-31m-en-safetensors` and runs the same way.

The 40m packages use the same `gradium_phonon` runtime; layer counts and other
architecture settings come from the embedded checkpoint config. No variant flag
or spec override is needed. For example, after installing the French package:

```bash
audiocpp_cli --task tts --family gradium_phonon \
  --model models/Gradium-Phonon-GGUF/40m-fr --backend cuda \
  --mode streaming --text "Bonjour, ceci est un test de synthèse vocale." \
  --out french.wav --log
```

Use `models/Gradium-Phonon-GGUF/40m-multilingual` for multilingual text.
Offline and streaming generation are checked on CUDA, Vulkan and CPU for both
40m packages. Language is determined by the input text and checkpoint; these
packages do not add a runtime language-style control.

## Model

| Field | Value |
|---|---|
| Family | `gradium_phonon` |
| Tasks | `tts` |
| Modes | `offline`, `streaming` |
| Languages | `en`, `fr`, `de`, `es`, `pt` (checkpoint-dependent) |
| Default package | `gradium_phonon_31m_en_q8_0` |
| Input | Text; voice conditioning as shown in the examples |
| Output | Generated speech audio |

“Model/session default” means no explicit override. The runtime uses the
checkpoint or session setting, or leaves the optional value unset as described.

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--text` | Input text; see the examples for the expected prompt format. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `frames_after_eos` | `int` | `-1` | Frames kept after EOS, counting the EOS frame; -1 keeps 4 for texts of up to four words and 2 otherwise, as the reference does. Declared range: minimum -1. |
| `max_frames` | `int` | `0` | Maximum generated audio frames; zero estimates the cap from text length. Captured during session preparation. Declared range: minimum 0. |
| `text_chunk_tokens` | `int` | `50` | Tokenizer tokens per text chunk; must be positive, default 50. Set during preparation and overridable per request. Declared range: minimum 1. |
| `temperature` | `float` | `0.3` | Positive flow noise temperature. Captured during session preparation. |
| `seed` | `int` | random | Noise seed; omitted chooses a random seed. Captured during session preparation. Declared range: minimum 0, maximum 4294967295. |
| `noise_clamp` | `float` | `-1` | Noise clamp; a negative value disables clamping. Captured during session preparation. |
| `eos_threshold` | `float` | `4` | EOS detection threshold on the EOS logit. Captured during session preparation. |
| `noise_file` | `path` | Model/session default | Precomputed noise schedule, raw float32 [frames, 32], used as given. Captured during session preparation. |
| `voice_embedding_file` | `path` | unset | Voice file: a speaker_wavs latent file such as those of gradium/phonon-voices, or an exported voice state. Captured during session preparation. |
| `text_chunk_size` | `int` | unset | Optional per-request character chunk size; otherwise use sentence/token budgeting. |
| `export_voice_state_file` | `path` | unset | Export the prepared voice state during session preparation. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `gradium_phonon.voice_state_cache_slots` | `int` | `4` | Cached prepared voice states; zero disables reuse. Declared range: minimum 0. |
| `gradium_phonon.offline_graph_capacity_mode` | `fixed`, `double`, `tiered`, `grow` | Model/session default | Offline graph capacity policy; CPU/Metal default tiered, other backends double. |
| `gradium_phonon.prompt_graph_capacity` | `int` | Model/session default | Optional reserved prompt capacity. Declared range: minimum 1. |
| `gradium_phonon.generation_graph_capacity` | `int` | Model/session default | Optional reserved generation capacity. Declared range: minimum 1. |
| `gradium_phonon.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Shared matmul weight storage; component matmul settings inherit it. Convolution storage remains independently native unless overridden. |
| `gradium_phonon.matmul_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Weight storage type; component-specific settings inherit weight_type. |
| `gradium_phonon.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage. |
| `gradium_phonon.weight_context_mb` | `int` | `64` | Context or graph arena size in MiB. Declared range: minimum 1. |
| `gradium_phonon.flow_weight_context_mb` | `int` | Model/session default | Flow weight context in MiB; inherits gradium_phonon.weight_context_mb when omitted. Declared range: minimum 1. |
| `gradium_phonon.mimi_decoder_weight_context_mb` | `int` | Model/session default | Mimi decoder weight context in MiB; inherits gradium_phonon.weight_context_mb when omitted. Declared range: minimum 1. |
| `gradium_phonon.flow_weights_view_context_mb` | `int` | `256` | Context or graph arena size in MiB. Declared range: minimum 1. |
| `gradium_phonon.flow_step_graph_arena_mb` | `int` | `256` | Flow-step host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `gradium_phonon.mimi_conv_graph_arena_mb` | `int` | `32` | Mimi convolution host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `gradium_phonon.mimi_transformer_graph_arena_mb` | `int` | `96` | Mimi transformer host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `gradium_phonon.mimi_tail_graph_arena_mb` | `int` | `512` | Mimi tail host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `gradium_phonon.mimi_full_chunk_frames` | `int` | `90` | Full-sequence decoder chunk frames. Declared range: minimum 1. |
| `gradium_phonon.mimi_stage2_chunk_frames` | `int` | `900` | Second-stage decoder chunk frames. Declared range: minimum 1. |
| `gradium_phonon.use_full_mimi` | `bool` | `true` | Use full-sequence Mimi decoding. |

## Voices

Phonon has no voice cloning from audio: the released checkpoints carry no Mimi
encoder. Voices are `speaker_wavs` latent files (one tensor `[1, 512, 125]`), which the
checkpoint projects into its voice prompt. Every Phonon checkpoint reads the same files.

More voices are in [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices),
listed with their names and languages in `voices.jsonl`. Pass one with
`--request-option voice_embedding_file=/path/to/voice.safetensors`, or copy it into the
model folder as `embeddings/<name>.safetensors`: `--voice-id <name>` and the server's
`"voice": "<name>"` then select it, and `GET /v1/audio/voices` lists it.

## Text

The text goes to the SentencePiece tokenizer after the PocketTTS clean-up: whitespace
collapsed, first letter capitalized, a final period added. There is no number or symbol
normalization, so write numbers, currencies and abbreviations out in words for the best
results. Letters the tokenizer has no piece for, such as `û` and `ï`, reach the model as
byte tokens.

## Conversion

The Q8_0 package is built from the safetensors checkpoint with the second command
below; the first keeps the published types. `audiocpp_gguf` embeds `config.json` and
`tokenizer.model` in the GGUF; put `default-voice.safetensors` (or
`embeddings/default.safetensors`) beside it:

```bash
audiocpp_gguf --input models/phonon-31m-en-safetensors/model.safetensors \
  --root models/phonon-31m-en-safetensors \
  --family gradium_phonon --output phonon-31m-en-bf16.gguf --type orig \
  --exclude-prefix flow_lm.flow.w_s_t --overwrite

audiocpp_gguf --input models/phonon-31m-en-safetensors/model.safetensors \
  --root models/phonon-31m-en-safetensors \
  --family gradium_phonon --output phonon-31m-en-q8_0.gguf --type q8_0 \
  --exclude-prefix flow_lm.flow.w_s_t \
  --keep-type 'mimi.*=f32' --keep-type 'flow_lm.flow_net.*=bf16' \
  --keep-type 'flow_lm.condition_provider.*=bf16' \
  --keep-type 'flow_lm.input_linear.*=bf16' --keep-type 'flow_lm.out_eos.*=bf16' \
  --overwrite
```

`orig` keeps the checkpoint's types: BF16 for the flow model, F32 for Mimi. The Q8_0
recipe quantizes only the flow transformer's linear layers, as Gradium's own runtime
does; Mimi stays F32. `flow_lm.flow.w_s_t` is used only in training.

The loader reads the model sizes from `config.json`, uses the plain RMS time embedding
when it says `"time_rms_norm": "rms"`, and folds the summed conditioners
(`num_speakers` = 1, `duration_delta` = 0) into a bias on `input_linear`, which is what
the reference adds to every generated frame.

## Validation

Checked against Gradium's Rust runtime with its reference dumps, feeding the same
noise through `noise_file`: with F32 weights on the CPU, the token ids, the first EOS
frame and the frame count match exactly, and the per-frame latents agree to about 1e-5
relative. BF16 and Q8_0 weights and the Metal backend drift as usual for a lower
precision, with the same Whisper transcripts.
