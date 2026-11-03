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

## Controls

The request controls match PocketTTS: `frames_after_eos`, `max_frames`,
`text_chunk_tokens`, `temperature`, `seed`, `noise_clamp`, `eos_threshold`,
`noise_file`, `voice_embedding_file`, `truncate_clone_audio`, `text_chunk_size`,
and `export_voice_state_file`. Defaults and ranges are recorded in
[`model_specs/pocket_tts_small.json`](../../model_specs/pocket_tts_small.json).

Session controls use the `pocket_tts_small.` namespace, including
`pocket_tts_small.weight_type`, `pocket_tts_small.voice_state_cache_slots`,
`pocket_tts_small.offline_graph_capacity_mode`, and the corresponding PocketTTS
flow/Mimi storage and graph capacity controls. Use `--session-option name=value`
for session controls and `--request-option name=value` for request controls.
There are no public family or option aliases and no language load selector;
these packages are English.

Select `--family pocket_tts_small`, `--task tts` for a predefined voice or
`--task clon` with speaker reference audio. Both tasks accept `--mode offline`
and `--mode streaming`. Enable `--log` and save the log when running either mode.

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

On CPU, offline synthesis uses cached causal decoder windows by default; GPU
backends use full-sequence decoding. `--session-option pocket_tts_small.use_full_mimi=true`
selects full-sequence decoding explicitly.

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
attention and KV storage. Defaults are eager attention and F32 KV storage.
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
