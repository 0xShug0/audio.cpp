# MiraTTS Vulkan GPU validation

Local validation: 2026-09-05, Windows, RTX 3090, Vulkan device 1.

The generator now uses the selected Vulkan execution context instead of a CPU
fallback. Its decoder uses F32 projection precision, grouped flash attention
(materialized grouped K/V heads), and F32 input to the output projection.
DirectSetRows KV updates and compact logits remain enabled.

The previous GPU configuration truncated the short test to 0.74 seconds.
Projection precision alone and attention-layout changes alone did not resolve
that failure in local probes. This is a model-scoped configuration fix, not a
claim that an individual upstream Vulkan kernel has been conclusively diagnosed.

## Long-lived offline session

| Request | Q8 wall ms | BF16 wall ms | Audio seconds |
|---|---:|---:|---:|
| Cold | 1632.78 | 1286.46 | 5.12 |
| Repeat | 884.67 | 923.20 | 5.12 |
| Different prompt | 902.05 | 926.29 | 5.12 |
| Long-form | 4957.44 | 3108.20 | 15.36 |
| Repeat after long-form | 933.39 | 1007.46 | 5.12 |

The three identical requests have identical audio hashes within each precision.
These fixtures are token-capped; their duration is not evidence of complete
prompt coverage. Timing excludes process startup/model loading and is not a
controlled cross-backend performance comparison.

## Streaming

| Precision/request | Wall ms | First event ms | Events | Audio seconds |
|---|---:|---:|---:|---:|
| Q8 cold | 3770.73 | 1437.21 | 4 | 20.02 |
| Q8 repeat | 3717.18 | 1375.11 | 4 | 20.02 |
| BF16 cold | 4202.34 | 1686.40 | 4 | 20.60 |
| BF16 repeat | 3986.38 | 1459.62 | 4 | 20.60 |

Repeated streaming audio hashes match. Both precisions pass
`tests/mira_tts/validate_bench.py` with their offline/streaming summaries.
Use `--model-spec-override model_specs/mira_tts.json` for the streaming fixture's
chunking options, as in the test README.

## Full short sentence

Text: "Hello, this is a native Mira TTS test."
Reference: `assets/resources/b.wav`, seed 1234, default CLI token budget.

| Precision | Generation ms | Audio seconds |
|---|---:|---:|
| Q8 | 1154.30 | 5.56 |
| BF16 | 1363.06 | 6.60 |

Qwen3-ASR recognizes the complete sentence from both outputs (including Mira
and TTS). This is a transcription smoke check, not perceptual quality parity.
Samples and benchmark JSON/logs are in the local sibling `outputs` directory,
with names `MiraTTS-q8-Vulkan-GPU.wav`, `MiraTTS-bf16-Vulkan-GPU.wav`, and
`vulkan-gpu-{q8,bf16}-{offline,streaming}`.

CUDA waveform parity and other Vulkan vendors/devices are not claimed.
