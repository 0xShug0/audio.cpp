# Qwen3 TTS

Qwen3 TTS includes Base voice cloning, VoiceDesign, and CustomVoice variants.
For shared conversion guidance, see [Qwen3 GGUF checkpoints](qwen3.md#gguf-checkpoints).

## Qwen3 TTS Base

Qwen3 TTS Base is the voice-clone TTS path. It requires reference audio and,
by default, its transcript. To use reference audio without a transcript, pass
`--request-option x_vector_only_mode=true`.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-Base` |
| Tasks | `tts`, `clon` |
| Modes | `offline`, `streaming` |
| Voice input | Reference WAV through `--voice-ref` |
| Transcript | `--reference-text` required unless `x_vector_only_mode=true` |

```bash
audiocpp_cli --task tts --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-Base --backend cuda --text "Hello from Qwen3 TTS." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

`tts` and `clon` enter the same Base synthesis implementation.

### Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--voice-ref` | WAV path | required | Voice clone reference audio. |
| `--reference-text` | text | required by default | Transcript for reference audio; optional with `x_vector_only_mode=true`. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. Each chunk is voiced separately; with the session option `qwen3_tts.stitch_decode=true` the talker still runs per chunk but the codes of all chunks are voiced in one decoder pass. |
| `--max-tokens` | positive integer | checkpoint generation config (`2048` if absent) | Maximum generated speech tokens per chunk. |

## Qwen3 TTS CustomVoice

Qwen3 CustomVoice uses speaker ids packaged with the model. Use `--request-option voice_id=Vivian` to select one; legacy `--speaker Vivian` remains accepted. Examples use `Vivian` and `Ryan`.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Task | `tts` |
| Modes | `offline`, `streaming` |
| Voice input | Built-in speaker id through `--speaker` |
| Style control | Optional instruction through `--instruct` |
| External voice WAV | Not used by this path |

```bash
audiocpp_cli --task tts --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-CustomVoice --backend cuda --text "Hello from a custom voice." --speaker Vivian --instruct "Very happy." --out out.wav
```

### Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--speaker` | packaged speaker name | required | Built-in speaker id, such as `Vivian` or `Ryan`. |
| `--instruct` | text | empty string | Style or emotion instruction. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. Each chunk is voiced separately; with the session option `qwen3_tts.stitch_decode=true` the talker still runs per chunk but the codes of all chunks are voiced in one decoder pass. |
| `--max-tokens` | positive integer | checkpoint generation config (`2048` if absent) | Maximum generated speech tokens per chunk. |

## Qwen3 TTS VoiceDesign

Qwen3 VoiceDesign creates a voice from an instruction. It does not require a speaker reference WAV.

| Field | Value |
|---|---|
| Family | `qwen3_tts` |
| Model directory | `models/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Tasks | `tts`, `vdes` |
| Modes | `offline`, `streaming` |
| Voice input | Instruction text through `--instruct` |
| Reference audio | Not required |

```bash
audiocpp_cli --task vdes --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --backend cuda --text "Hello from a designed voice." --instruct "A warm adult narrator" --out out.wav
```

`tts` and `vdes` enter the same VoiceDesign synthesis implementation.

### Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Text to synthesize. |
| `--instruct` | text | empty string | Optional voice design instruction. |
| `--language` | language code | empty string | Text language hint. |
| `--text-chunk-size` | integer chars | `8192` | Long-form text chunk size. Each chunk is voiced separately; with the session option `qwen3_tts.stitch_decode=true` the talker still runs per chunk but the codes of all chunks are voiced in one decoder pass. |
| `--max-tokens` | positive integer | checkpoint generation config (`2048` if absent) | Maximum generated speech tokens per chunk. |

## Qwen3 TTS Streaming

All three TTS variants accept `--mode streaming`. Audio is decoded and emitted
while speech tokens are still being generated, rather than after the complete
utterance. The first audio chunk contains one codec frame (80 milliseconds at
24 kHz). Subsequent chunks double up to the configured `stream_frames_per_event`
maximum: with the default 12, the sequence is 1, 2, 4, 8, 12, 12, and so on.
These are audio durations, not latency guarantees.
Set `stream_chunk_policy=fixed` to use `stream_frames_per_event` from the first
event instead; the last event may contain fewer frames. A fixed size of 1 emits
one frame at a time. The default policy is `grow`.

Streaming retains 25 codec frames of left context and reuses a fixed-capacity
decoder graph, with a separate cached graph for the first frame. The final
partial chunk is flushed at end of generation. The
final audio is exactly the concatenation of the emitted chunks. Streaming uses
smaller codec windows than offline decoding, so waveform differences are expected;
the offline decoding path is unchanged.

For the server, configure the model with `"mode": "streaming"` and request
`"stream_format": "sse"` from `/v1/audio/speech` (streaming defaults to PCM).
Audio arrives through the existing `speech.audio.delta`
events. Base models require reference audio; the transcript can be omitted only
with `x_vector_only_mode=true`.
VoiceDesign accepts an empty instruction; CustomVoice requires a packaged speaker.

## Qwen3 TTS Sampling

These sampling controls are shared by the Qwen3 TTS Base, VoiceDesign, and CustomVoice paths.

### Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--do-sample` | `true`, `false` | `true` | Enable sampling. |
| `--temperature` | float | `0.9` | Main talker temperature; must be > 0 when `--do-sample=true`, ignored when false. |
| `--top-k` | integer | `50` | Main talker top-k. |
| `--top-p` | float | `1.0` | Main talker top-p. |
| `--repetition-penalty` | float > 0 | `1.05` | Main talker repetition penalty. |
| `--seed` | integer | random if omitted | Sampling seed. |

### Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `subtalker_do_sample` | bool | `true` | Enable subtalker sampling for the remaining audio codebooks. |
| `voice_id` | packaged speaker name | unset | CustomVoice preset selection. Legacy `speaker` remains accepted. An explicit request option takes precedence over the structured voice's cached id. |
| `subtalker_temperature` | float | `0.9` | Must be > 0 when `subtalker_do_sample=true`; ignored when false. |
| `subtalker_top_k` | integer | `50` | Subtalker top-k. |
| `subtalker_top_p` | float | `1.0` | Subtalker top-p. |
| `x_vector_only_mode` | bool | `false` | Base voice cloning using only the reference speaker embedding, without ICL reference codec prompting. |
| `stream_frames_per_event` | positive integer | `12` | Codec frames per fixed event, or the maximum for growing events. Smaller values reduce chunk latency but increase decoding overhead. Ignored offline. |
| `stream_chunk_policy` | `grow` or `fixed` | `grow` | Grow from one frame to the configured maximum, or emit fixed-size chunks. Ignored offline. |

### Session Options (use with `--session-option`)

These options apply to Base, VoiceDesign, and CustomVoice sessions. Arena and
constant-context sizes are in MiB.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `qwen3_tts.mem_saver` | bool | `false` | Release the talker cached-step graph after each request. Voice-prompt, prefill, code-predictor, and speech-decoder caches remain reusable. |
| `qwen3_tts.lora` | adapter directory or checkpoint file | not set | Load a PEFT Talker/Subtalker adapter at session creation. Keep `adapter_config.json` beside its weights. A trained custom voice may also provide `speaker_embedding` and `tts_config.json` sidecars; select its name with `voice_id`. |
| `qwen3_tts.lora_scale` | finite float | `1` | Strength multiplier applied after the adapter's alpha/r scaling. Speaker embedding replacements are not scaled. |
| `qwen3_tts.perf_mode` | `off`, `flash_attention`, `optimized` | `off` | `flash_attention` enables flash attention. `optimized` also selects F16 convolution weights and stateful streaming decoding unless explicitly overridden. KV alignment defaults to 256 in all modes. May change audio and increase peak VRAM. |
| `qwen3_tts.talker_kv_cache_alignment` | int ≥ 1 | `256` | Round talker KV capacity to this many rows. Can enable faster attention kernels, may use more memory, and can change numerical results. Set to `1` to disable padding. |
| `qwen3_tts.stream_decoder` | `windowed`, `stateful` | `windowed` | GGML streaming decoder. `stateful` carries convolution and attention history and caches graphs for exact chunk lengths. Uses more graph memory and can change audio; `windowed` preserves existing behavior. |
| `qwen3_tts.stitch_decode` | bool | `false` | Voice the concatenated speech codes of all text chunks in one decoder pass instead of voicing each chunk separately. Per-chunk voicing voices the voice-clone reference once per chunk and zeroes the stateful vocoder cache at every chunk boundary (measured on RK3588, 300 characters, 22.5 s reference: 7 vocoder runs for 159.9 s against 1 run for 57.5 s); the stitched pass costs device memory proportional to the utterance, since one decode call builds a vocoder graph sized by its frame count. |
| `qwen3_tts.voice_prompt_cache_slots` | nonnegative integer | `1` | Cached voice-clone prompts; `0` disables prompt caching. |
| `qwen3_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Talker weight storage fallback. |
| `qwen3_tts.talker_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | inherits `weight_type` | Explicit talker weight storage override. |
| `qwen3_tts.speech_encoder_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Reference speech encoder matmul weight storage. |
| `qwen3_tts.speech_decoder_weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Speech decoder matmul weight storage. |
| `qwen3_tts.conv_weight_type` | `native`, `f32`, `f16` | `f32` | Speech tokenizer and speaker encoder convolution weight storage. |
| `qwen3_tts.talker_graph_arena_mb` | positive integer | `32` | Talker graph arena. |
| `qwen3_tts.speech_encoder_graph_arena_mb` | positive integer | `32` | Reference speech encoder graph arena. |
| `qwen3_tts.speech_decoder_graph_arena_mb` | positive integer | `32` | Speech decoder graph arena. |
| `qwen3_tts.speaker_encoder_graph_arena_mb` | positive integer | `32` | Speaker encoder graph arena. |
| `qwen3_tts.talker_constant_context_mb` | positive integer | `4` | Talker constant tensor context. |
| `qwen3_tts.code_predictor_constant_context_mb` | positive integer | `4` | Code predictor constant tensor context. |
| `qwen3_tts.speech_decoder_constant_context_mb` | positive integer | `4` | Speech decoder constant tensor context. |
| `qwen3_tts.exec_plan` | JSON object of component plans, encoded as a string | `{}` | `{"speech_decoder":{"impl":"rknn","artifacts":{"main":"/models/decoder.rknn","first":"/models/decoder-first.rknn","window":"/models/decoder-window.rknn"}}}` selects the NPU decoder and skips GGML decoder weights. `main` is required; `first` optionally uses a smaller engine for the first streaming group; `window` optionally streams with window semantics. No decoder-specific params. See [component execution plans](../build/rockchip-npu.md#component-execution-plans). |

## Rockchip NPU decoder

The speech decoder stage can run on a Rockchip NPU through `librknnrt` instead
of ggml. The path is opt-in at build time:

```bash
cmake -S . -B build -DENGINE_ENABLE_RKNN=ON \
  -DENGINE_RKNN_INCLUDE_DIR=/usr/local/include \
  -DENGINE_RKNN_LIBRARY=/usr/lib/librknnrt.so
```

The `.rknn` must be the stateful fp16 vocoder export for the same checkpoint
and target SoC: inputs `audio_codes`, `freqs_in`, `cache_valid`, `cache_in`;
outputs `audio_values` and `cache_out`; 1920 samples per token at 24 kHz, with
the convolution and KV cache carried across steps in the graph I/O. The
optional first-group model runs the first token group at a smaller compiled
batch; the cache layout is batch-independent, so its cache feeds the main
model directly. The optional window model (`artifacts.window`) streams with the
ggml path's window semantics instead: every event runs a full decode of its
whole `[left context + new]` window from a zeroed cache and drops the context
frames' audio, so each event reproduces the CPU streaming function exactly
(25 frames re-voiced per event, at one NPU step of `25 + frames_per_event`
tokens). Naming the same file as `main` aliases it: the resident engine covers
the default 25 + 12 window; a dedicated export compiled at `--codes-length
25 + stream_frames_per_event` is cheaper per event. Offline decoding keeps
using `main`. A `.rknn` is compiled per SoC and is not portable between
chips. Build the file with
[`tools/rknn/prepare_qwen3_tts_rknn_vocoder.py`](../../tools/rknn/prepare_qwen3_tts_rknn_vocoder.py)
and see [the Rockchip NPU build guide](../build/rockchip-npu.md) for the toolkit
venv, the measured numbers, and the limits of this path. This path is opt-in
and validated only on RK3588.

