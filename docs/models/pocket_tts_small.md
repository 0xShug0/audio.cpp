# PocketTTS Small

`pocket_tts_small` is the dedicated English PocketTTS Small family. It supports
text-to-speech (`tts`) and voice cloning (`clon`) in offline and streaming modes,
using a dedicated implementation.

## Packages

F32 packages are hosted under
[PocketTTS-Small-GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/PocketTTS-Small-GGUF)
in the `audio-cpp/audio.cpp-gguf` repository. The default is 18m ReFL.

| Package | GGUF file |
|---|---|
| `pocket_tts_small_english_18m_refl_f32` (default) | `english-18m-refl-f32.gguf` |
| `pocket_tts_small_english_18m_f32` | `english-18m-f32.gguf` |
| `pocket_tts_small_english_7m_refl_f32` | `english-7m-refl-f32.gguf` |
| `pocket_tts_small_english_7m_f32` | `english-7m-f32.gguf` |
| `pocket_tts_small_english_2m_refl_f32` | `english-2m-refl-f32.gguf` |
| `pocket_tts_small_english_2m_f32` | `english-2m-f32.gguf` |

Each package includes `embeddings/alba.safetensors` as its predefined `alba` voice.
GGUF sidecars provide `config.yaml` and `tokenizer.model`; the configuration is
required to describe the selected small model. The embedding remains beside the
GGUF under `embeddings/`.

The source layout follows PocketTTS: GGUF weights at the model root with the
configuration and tokenizer resolved through sidecars, or a Safetensors source
containing `model.safetensors`, `config.yaml`, and `tokenizer.model` at the
model root, with predefined voices under `embeddings/`. The six managed packages use GGUF.

## Quick start

```sh
audiocpp_cli --family pocket_tts_small --task tts \
  --model models/PocketTTS-Small-GGUF/english-18m-refl/english-18m-refl-f32.gguf \
  --backend cpu --voice-id alba \
  --text "Hello! This is the small PocketTTS model speaking locally." \
  --out output.wav --log
```

Use `--mode streaming` for streaming synthesis. For voice cloning, select
`--task clon` and pass `--voice-ref reference.wav` instead of `--voice-id alba`.

The checkpoints come from [Kyutai PocketTTS](https://huggingface.co/kyutai/pocket-tts)
and use the CC-BY-4.0 license. The implementation follows upstream revision
[`4cc57533`](https://github.com/kyutai-labs/pocket-tts/tree/4cc5753319ceddc5cbf70598d9011f1f22858ba0).

Offline synthesis uses bounded causal blocks of four latent frames by default
on every backend. `--session-option pocket_tts_small.use_full_mimi=true`
selects full-sequence decoding explicitly.

## Model

| Field | Value |
|---|---|
| Family | `pocket_tts_small` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Languages | `en` |
| Default package | `pocket_tts_small_english_18m_refl_f32` |
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
| `--voice-ref` | Reference audio for voice cloning. |
| `--out` | Output WAV file. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `frames_after_eos` | `int` | `-1` | Extra frames after EOS; -1 uses the model-derived value. Declared range: minimum -1. |
| `max_frames` | `int` | `0` | Maximum generated audio frames; zero estimates the cap from text length. Captured during session preparation. Declared range: minimum 0. |
| `text_chunk_tokens` | `int` | `50` | Tokenizer tokens per text chunk; must be positive, default 50. Set during preparation and overridable per request. Declared range: minimum 1. |
| `temperature` | `float` | Model/session default | Positive flow noise temperature; defaults to the model configuration. Captured during session preparation. |
| `seed` | `int` | Model/session default | Noise seed; omitted chooses a random seed. Captured during session preparation. Declared range: minimum 0, maximum 4294967295. |
| `noise_clamp` | `float` | `-1` | Noise clamp; a negative value disables clamping. Captured during session preparation. |
| `eos_threshold` | `float` | `-1` | EOS detection threshold; defaults to the checkpoint recommendation. |
| `noise_file` | `path` | Model/session default | Precomputed noise schedule file. Captured during session preparation. |
| `voice_embedding_file` | `path` | Model/session default | Precomputed voice-state embedding path. Captured during session preparation. |
| `truncate_clone_audio` | `bool` | `false` | Truncate voice clone audio. Captured during session preparation. |
| `text_chunk_size` | `int` | Model/session default | Optional per-request character chunk size; otherwise use sentence/token budgeting. |
| `export_voice_state_file` | `path` | Model/session default | Export the prepared voice state during session preparation. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `pocket_tts_small.attention` | `auto`, `eager`, `flash` | `auto` | GGML acoustic attention implementation. |
| `pocket_tts_small.kv_cache_type` | `f32`, `f16` | `f32` | GGML acoustic KV cache storage. Native CPU flow requires f32. |
| `pocket_tts_small.voice_state_cache_slots` | `int` | `4` | Cached prepared voice states; zero disables reuse. Declared range: minimum 0. |
| `pocket_tts_small.offline_graph_capacity_mode` | `fixed`, `double`, `tiered`, `grow` | Model/session default | Offline graph capacity policy; CPU/Metal default tiered, other backends double. |
| `pocket_tts_small.prompt_graph_capacity` | `int` | Model/session default | Optional reserved prompt capacity. Declared range: minimum 1. |
| `pocket_tts_small.generation_graph_capacity` | `int` | Model/session default | Optional reserved generation capacity. Declared range: minimum 1. |
| `pocket_tts_small.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | Model/session default | Shared matmul weight storage; component matmul settings inherit it. Convolution storage remains independently native unless overridden. |
| `pocket_tts_small.conv_weight_type` | `native`, `f32`, `f16` | `native` | Convolution weight storage. |
| `pocket_tts_small.weight_context_mb` | `int` | `64` | Context or graph arena size in MiB. Declared range: minimum 1. |
| `pocket_tts_small.flow_weight_context_mb` | `int` | Model/session default | Flow weight context in MiB; inherits pocket_tts_small.weight_context_mb when omitted. Declared range: minimum 1. |
| `pocket_tts_small.mimi_encoder_weight_context_mb` | `int` | Model/session default | Mimi encoder weight context in MiB; inherits pocket_tts_small.weight_context_mb when omitted. Declared range: minimum 1. |
| `pocket_tts_small.mimi_decoder_weight_context_mb` | `int` | Model/session default | Mimi decoder weight context in MiB; inherits pocket_tts_small.weight_context_mb when omitted. Declared range: minimum 1. |
| `pocket_tts_small.flow_weights_view_context_mb` | `int` | `256` | Context or graph arena size in MiB. Declared range: minimum 1. |
| `pocket_tts_small.flow_step_graph_arena_mb` | `int` | `256` | Flow-step host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `pocket_tts_small.mimi_encoder_graph_arena_mb` | `int` | `512` | Mimi encoder host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `pocket_tts_small.mimi_conv_graph_arena_mb` | `int` | `32` | Mimi convolution host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `pocket_tts_small.mimi_transformer_graph_arena_mb` | `int` | `96` | Mimi transformer host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `pocket_tts_small.mimi_tail_graph_arena_mb` | `int` | `512` | Mimi tail host graph-metadata arena size in MiB, not device workspace. Declared range: minimum 1. |
| `pocket_tts_small.mimi_chunk_frames` | `int` | `4` | Latent frames per bounded causal waveform-decoder block. Declared range: minimum 1. |
| `pocket_tts_small.mimi_stage2_chunk_frames` | `int` | `900` | Second-stage decoder chunk frames. Declared range: minimum 1. |
| `pocket_tts_small.use_full_mimi` | `bool` | `false` | Use a full-utterance waveform graph instead of bounded causal blocks. This can use more device memory. |
| `pocket_tts_small.exec_plan` | `string` | Model/session default | Component execution plan. Flow defaults to ggml; native is an experimental CPU F32 implementation with an optional threads parameter. |

## Experimental native CPU flow

The default implementation uses GGML on every backend. A build with
`-DENGINE_ENABLE_NEON=ON` also provides an experimental F32 CPU flow implementation.
The audio decoder continues to use GGML. Select it explicitly with:

```sh
--session-option 'pocket_tts_small.exec_plan={"flow":{"impl":"native","params":{"threads":1}}}'
```

The plan's `threads` parameter controls native operators; the CLI `--threads`
option controls the GGML decoder. Performance depends on model size and CPU:
benchmark both implementations for your workload. Native flow requires F32
weights and a CPU backend. GPU backends use GGML.

## Experimental offline batching

C++ callers can use `runtime::IBatchedOfflineVoiceTaskSession::run_batch` with
one `TaskRequest` per utterance. Results retain input order. Each request can
select its own `VoiceCondition` and override generation options such as `seed`,
`temperature`, and `max_frames`; omitted values use the prepared session defaults.
Each row has independent random state, cache positions, and EOS stopping.
Long requests retain sentence chunking and concatenate their chunks into one result.

Single-request, streaming, and batched generation share the same GGML acoustic
decode graph implementation; ordinary generation uses batch size one. Text prefill
and Mimi waveform decoding remain per-request. Graphs belong to the session and are
reused while their batch/cache capacities fit. The existing graph-capacity policy
also applies to batch cache sizing. Finished rows retain their slots until the
other rows finish, so mixed lengths can limit throughput. Memory grows with batch
size and cache capacity; a larger batch is not always faster.

Batching requires GGML flow and offline mode. Streaming and native CPU flow
continue to use the single-request interface. Batched floating-point execution
can change sampled audio; identical seeds do not promise identical WAVs across
batch sizes or compositions. Native CPU flow remains a separate implementation.

The model integration test `pocket_tts_small_batch` exercises the batch interface.
The CLI's `--batch-text-file` schedules requests one at a time through that same
batch-size-one runtime; C++ callers use the batch interface to group requests.

## Performance controls

`pocket_tts_small.attention` (`auto`, `eager`, or `flash`) and
`pocket_tts_small.kv_cache_type` (`f32` or `f16`) independently control acoustic
attention and KV storage. Defaults are automatic attention selection and F32 KV storage.
These session options apply equally to ordinary, streaming, and batched GGML
generation. Native CPU flow requires eager attention and F32 state.

For example, add `--session-option pocket_tts_small.attention=flash` and
`--session-option pocket_tts_small.kv_cache_type=f16` to the CLI command.
These controls do not change the GGUF weights. Reduced precision and attention
changes can change sampled speech; compare quality on your workload.

Waveform decoding defaults to bounded causal blocks of four latent frames.
`pocket_tts_small.mimi_chunk_frames` controls that block size; larger values can
improve throughput but use more memory. Graphs are reused between blocks and
utterances, while convolution history and transformer state reset for each new
utterance. The final block is padded and only its valid audio is returned.
`pocket_tts_small.use_full_mimi=true` selects a full-utterance graph instead, with
a higher peak-memory cost. Both paths implement the same causal decoder.
