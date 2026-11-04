# VibeASR in audio.cpp

[VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) is Microsoft's CPU-first
port of the VibeVoice ASR stack, quantized end to end for edge inference: the
audio VAE encoder runs on INT8 weights *and* INT8 activations, and the Qwen2
decoder runs on BitNet-style ternary weights. This entry ports both halves, so
`--family vibeasr` transcribes end to end on CPU.

## Usage

VibeASR.cpp already ships both halves quantized, so there is nothing to
re-quantize. The two forks only disagree on the numeric type *ids* — the VibeASR
fork put I2_S/I8_S at 36/37, which upstream ggml had already spent on the retired
`IQ4_NL_4_4` / `IQ4_NL_4_8` slots, so audio.cpp registers them at 43/42. The
converter rewrites the 4-byte type field in each tensor info and copies
everything else through byte for byte:

```bash
# inspect first
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input vibeasr-vae-encoder-i8_s.gguf --list

# fix both GGUFs in place (703 MB encoder, 993 MB decoder)
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-vae-encoder-i8_s.gguf --in-place
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf --in-place

# confirm an already-converted package needs no further remapping
python3 tools/community_models/convert_vibeasr_gguf.py \
    --input models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf --check
```

Use `--output <path>` instead of `--in-place` to keep the original.

The package is two GGUFs plus the tokenizer, so `--model` points at the LM GGUF
and the spec is resolved from the repo — the same invocation shape as
[`minimax_h3`](minimax_h3.md):

```
models/vibeasr/
├── vibeasr-vae-encoder-i8_s.gguf
├── vibeasr-lm-i2_s-embed-q6_k.gguf
├── tokenizer.json
└── tokenizer_config.json
```

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target audiocpp_cli

./build/bin/audiocpp_cli \
    --task asr \
    --family vibeasr \
    --model models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf \
    --model-spec-override model_specs \
    --backend cpu \
    --threads 8 \
    --audio assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav \
    --metrics
```

```
text_output=Concord returned to its place amidst the tents.
metrics.wall_ms=1284.65
metrics.rtf=0.36652
```

Request options: `output_format` (`text` | `json`), `context` (a string folded
into the prompt to bias recognition), `max_tokens` (default 1024; legacy
`max_new_tokens` remains accepted). Session
options: `vibeasr.encoder_graph_arena_mb` (64),
`vibeasr.prefill_graph_arena_mb` (256), `vibeasr.decode_graph_arena_mb` (256).

Note that `output_format=json` returns an empty transcript on short
single-speaker clips — the model emits an immediate end-of-turn. VibeASR.cpp
behaves identically on the same input; this port does not paper over it.

## Model

| Field | Value |
|---|---|
| Family | `vibeasr` |
| Tasks | `asr` |
| Modes | `offline` |
| Languages | `en`, `zh`, `fr`, `it`, `ko`, `pt`, `vi` |
| Default package | `vibeasr_bitnet_i2_s` |
| Input | Speech audio |
| Output | Transcript text |

## Common Options (use directly)

| Option | Meaning |
|---|---|
| `--model` | Model file or directory shown in the examples. |
| `--backend` | Select an available backend supported by this model. |
| `--threads` | CPU thread count. |
| `--log` | Enable runtime logging. |
| `--audio` | Input audio, where required by the selected task. |

## Request Options (use with `--request-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `output_format` | `text`, `json` | `text` | Prompt suffix asked of the decoder: plain transcription text, or JSON rows with Start/End/Speaker/Content. |
| `context` | `string` | `` | Extra context injected into the prompt (names, jargon) to bias the transcription. |
| `max_tokens` | `int` | `1024` | Cap on decoded tokens for one request. Declared range: minimum 1. |

## Session Options (use with `--session-option`)

| Option | Type / values | Default | Description |
|---|---|---|---|
| `vibeasr.encoder_graph_arena_mb` | `int` | `64` | VAE encoder graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |
| `vibeasr.prefill_graph_arena_mb` | `int` | `256` | Decoder prefill graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |
| `vibeasr.decode_graph_arena_mb` | `int` | `256` | Decoder single-step graph arena size in MB. Must be positive; sufficient capacity depends on the graph. Declared range: minimum 1. |

## Related Model

Use [VibeVoice ASR](../asr.md#vibevoice-asr) for the standard F32/Q8 runtime.
VibeASR uses the separate INT8/ternary package and is CPU-only.

## Measured performance

Release build, CPU backend, 24 vCPU AMD EPYC 7V13, 3.505 s clip resampled to
24 kHz (26 speech frames, 72-token prompt, 13 generated tokens):

| Threads | encoder (both branches) | prefill | decode | wall | RTF |
|---|---|---|---|---|---|
| 8 | 758 ms | 214 ms | 239 ms | 1285 ms | 0.367 |
| 1 | 4652 ms | 1415 ms | 941 ms | 7081 ms | 2.020 |

The encoder dominates: it is run twice, once per branch, and it processes raw
samples rather than tokens. Decode is about 18 ms/token at 8 threads.

The encoder-only probe reports 546 ms for both branches at 8 threads because it
feeds the clip at its native 16 kHz (17 frames); the session resamples to 24 kHz
first (26 frames).

Peak RSS is 2.20 GB against 1.70 GB of weights: `BackendWeightStore` stages each
tensor before upload, so weight loading briefly holds roughly two copies of the
tensor being uploaded. Graph arenas are 64 MB (encoder) + 256 MB (prefill) +
256 MB (decode) by default.

## Known limitations

- **CPU only.** The fused I8_S ops and the I2_S matmul have no CUDA or Metal
  kernels; the session pins the backend to CPU.
- **Offline only**, like `vibevoice_asr` itself. Upstream's encoder is causal, so
  streaming is implementable, but the state machine is not ported.
- **Greedy only.** Upstream's sampling path (temperature, top-p) is not ported,
  and neither is `vibevoice_asr`'s beam search.
- **Text only.** No `--segments-out` / `--turns-out` equivalent; `output_format=json`
  is a prompt variant, not structured decoding.
- The package is two GGUFs, so it needs `--model <lm gguf>` plus
  `--model-spec-override model_specs` rather than a directory path.

## Upstream

- Model port: <https://github.com/microsoft/VibeASR.cpp> (`src/vae.cpp`,
  `src/lm.cpp`, `src/asr_server.cpp`, `utils/prompt_builder.h`)
- Base model: VibeVoice ASR, also in tree as [`vibevoice_asr`](../asr.md#vibevoice-asr)
- Weights: <https://huggingface.co/microsoft/VibeVoice-ASR-BitNet>
