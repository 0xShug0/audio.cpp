# YuE2 longform generation: host commit exhaustion in the NAR graph build context

On v0.9.0, generating the official `tonight-awake` longform test case (5,180 semantic
tokens) aborts the process with `GGML_ASSERT(ctx->mem_buffer != NULL) failed`
(`external/ggml/src/ggml.c:1685`) whenever Windows commit available is below roughly
6 GiB when the NAR stage starts. The failing allocation is the host-side `ggml_init`
of the NAR graph build context, whose size comes from `yue2.nar_graph_arena_mb`
(default 6144). The abort is a fail-fast (`exit code 0xC0000409`); in server mode it
takes the whole server down: the in-flight HTTP request gets `ECONNRESET`, every loaded
model disappears, and VRAM usage drops to 0.

This is a host-RAM failure, not a VRAM failure: at the crash point the GPU held only
about 2.5-3 GiB of its 8 GiB. The same failure was first observed in normal server-mode
use and is deterministic under low commit availability; the controlled reproduction
below reduces commit availability with a plain ballast allocation to keep the run
conditions fixed. Upstream discussion: issue
[#764](https://github.com/0xShug0/audio.cpp/issues/764), earlier CPU-backend report
[#656](https://github.com/0xShug0/audio.cpp/issues/656).

## Reproduction

Windows 11 x64, 16 GB system RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8 GiB, compute
capability 8.9), driver 610.74 (CUDA 13.3). audio.cpp v0.9.0 prebuilt
`bin-windows-x64-cuda13.3` archive plus the matching `cudart` archive extracted next to
the binaries. Model: `audio-cpp/Yue2-3B-GGUF` snapshot (`yue2-3b-q4_0.gguf`,
`yue2-vae-f16.gguf`, sidecars).

Request: `examples/tonight-awake.json` from `m-a-p/YuE2-3B` — Chinese lyrics, City Pop
style, `cot=full`, `seed=12300`, `num_inference_steps=8` (~207 s of audio at 48 kHz
stereo).

```
audiocpp_cli.exe --task gen --family yue2 --model models/Yue2-3B-GGUF --backend cuda --threads 8 \
  --text "<tonight-awake lyrics>" \
  --request-option "style=City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night" \
  --request-option cot=full --request-option seed=12300 --request-option num_inference_steps=8 \
  --session-option yue2.model_gguf=yue2-3b-q4_0.gguf \
  --out tonight.wav --log
```

With default session options and low commit availability, the whole AR/semantic
pipeline completes and the abort lands at the start of the NAR graph build, before
`yue2.nar.graph.build_ms` is logged:

```
[TIMING] yue2.semantic.tokens 5180
[TIMING] yue2.nar.weights.buffer_mb 861.63720703125
[TIMING] yue2.nar.weights.buffer_name CUDA0
[TIMING] yue2.nar.init_ms 909.5426
[TIMING] yue2.nar.synthesize.codec_tokens 5180
[TIMING] yue2.nar.synthesize.chunks 1
[TIMING] yue2.nar.synthesize.context 24576
D:\a\audio.cpp\audio.cpp\external\ggml\src\ggml.c:1685: GGML_ASSERT(ctx->mem_buffer != NULL) failed

=== EXIT code=3221226505 (0xC0000409) ===
```

Full log: [longform_host_memory_crash.log](longform_host_memory_crash.log).

## Default host contexts are over-provisioned

The YuE2 session defaults reserve multi-GiB host contexts whose actual metadata usage is
orders of magnitude smaller. Logged `ctx_reserved_mb` vs `ctx_used_mb` pairs from the
same run:

| Context | Default reserved | Logged used |
|---|---:|---:|
| `yue2.model_weight_context_mb` | 6144 MB | 0.108 MB |
| `yue2.ar.generation.weights` | 64 MB | 0.0007 MB |
| `yue2.ar_prefill_graph_arena_mb` | 4096 MB | ~3.0 MB |
| `yue2.ar_decode_graph_arena_mb` | 1536 MB | ~3.1 MB |
| `yue2.nar_graph_arena_mb` | 6144 MB | ~10.5 MB |
| `yue2.vae_graph_arena_mb` | 1536 MB | ~0.1 MB |

The defaults sum to ~19.1 GiB of host context (not all alive at once, but the 6144 MB
NAR build context alone is a single contiguous ~6 GiB allocation). The NAR compute
buffer itself for 5,180 frames is only 339,615,744 bytes (~324 MiB) on the GPU.

## Workaround

Reducing the host arena/context session options lets the identical request complete on
the same machine:

```
--session-option yue2.model_weight_context_mb=1024
--session-option yue2.ar_prefill_graph_arena_mb=1024
--session-option yue2.ar_decode_graph_arena_mb=512
--session-option yue2.nar_graph_arena_mb=2048
--session-option yue2.vae_graph_arena_mb=512
--session-option yue2.vae_weight_context_mb=512
```

Result: full 207.2 s stereo 48 kHz song, `session.wall_ms` 113,158 (RTF 0.55), NAR
compute buffer 339,615,744 bytes, sampled VRAM ~6.2 GiB mid-run. Full log:
[longform_host_memory_success.log](longform_host_memory_success.log).

## Inconsistent failure handling

While narrowing down the crash at intermediate memory pressures, two different failure
modes appeared for the same underlying exhaustion, depending on which allocation fails
first:

- CUDA-side buffer allocation failures are handled gracefully: the CLI exits with
  `audiocpp_cli failed: failed to allocate Yue2 NAR graph` (exit code 1).
- Host-side `ggml_init` failures hit the `GGML_ASSERT` and abort the process (and in
  server mode kill the server, losing every loaded session).

A clean error for the host-side path (and a per-request HTTP 5xx in server mode) would
make the behavior consistent.

## Status

Fixed on main by "Lower YuE2 context defaults"
([#767](https://github.com/0xShug0/audio.cpp/pull/767)), which lowers these defaults;
this report documents the v0.9.0 behavior and the workaround for anyone staying on that
release.
