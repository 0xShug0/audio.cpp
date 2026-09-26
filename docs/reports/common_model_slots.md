# Concurrent slots for one loaded model

Follow-up: [Vulkan compatibility validation](common_model_slots_vulkan.md)
includes real same-backend inference and the complete local Vulkan test results.

The server previously serialized requests through one mutable session per model.
Removing that lock would let requests overwrite the same graphs, KV/reference
caches, callbacks and sampler state. Running multiple server instances avoids
that conflict by loading a separate copy of the model in every process.

The common slots framework keeps one loaded model and leases isolated execution
sessions to concurrent requests. The first enabled adapter is Higgs Audio v3 TTS
on CUDA in offline mode: immutable AR and codec weights are shared, while each
slot owns its execution context, graphs, caches and RNG. Other adapters retain
their default single-slot behavior until they implement and validate the factory.
Shared CUDA kernels, precision, sampling and graph operations are unchanged.

## Measured generation time

Windows, RTX 3090 24 GiB, CUDA 12.4, MSVC Release; source based on `ab862077`
after the CUDA and Vulkan prefill memory improvements were merged. Eight CPU
threads per server; model `higgs-audio-v3-tts.gguf`; bundled `b.wav` reference
with its transcript; seed 1234, max_tokens 1024, remaining settings at defaults.
The model uses `model_specs/higgs_audio_tts.json` as a spec override.

Text:

> Ptaszki ćwierkają, że Apple chce wrócić na rynek serwerów dostarczając chłonnemu rynkowi AI swoje własne rozwiązania bazujące na zmodyfikowanych układach z serii M.

Each row launches that many simultaneous identical requests. Times are medians
of three warm repeats, measured from barrier release until all HTTP requests
complete, including response decoding and saving the WAVs. The same PR executable
is used for both configurations: one process with N slots, or N independent
processes each configured with one slot, all on GPU 0. The latter represents the
existing separate-instance workaround. Startup/model loading is excluded.

| Simultaneous requests | One slot, sequential (estimated) | Shared-model slots | Separate single-slot servers | Less time vs separate servers |
|---:|---:|---:|---:|---:|
| 1 | 2.15 s | 2.15 s | 2.15 s | 0.0% |
| 2 | 4.30 s | 3.73 s | 6.24 s | 40.2% |
| 3 | 6.45 s | 5.36 s | 9.39 s | 42.9% |
| 4 | 8.60 s | 7.04 s | 12.55 s | 43.9% |

The sequential column is an estimate: N times the measured warm single-slot
median, rather than a separately timed sequential batch. Four shared slots save
18.1% total time and improve throughput by
22.1% against that estimate. Against
four actual separate server instances, they save
43.9% total time. Individual request latency
increases with slot count; this improves simultaneous-batch throughput.

## Measured memory

Peaks include loaded weights and inference allocations. NVML is sampled roughly
every 10 ms; the initial idle driver reservation is subtracted. Values are the
maximum sampled peak across the three warm repeats. The GPU is otherwise idle.

| Requests | Shared-model slots: warm peak VRAM | Separate servers: warm peak VRAM |
|---:|---:|---:|
| 1 | 5.40 GiB | 5.40 GiB |
| 2 | 5.83 GiB | 10.80 GiB |
| 3 | 6.27 GiB | 16.20 GiB |
| 4 | 6.70 GiB | 21.60 GiB |

One shared-model process loads about 4.931 GiB of weights/state for every tested
slot count. Four separate processes load about 19.724 GiB before generation.
Four shared slots reduce the observed warm total peak by
69.0%
relative to four separate servers. Working memory still grows with slot count.

These are measurements for one prompt, GPU and operating system. Process context
scheduling, GPU memory contention, temperature and allocator overlap affect
results; separate-instance timings are not a universal multi-GPU or MPS claim.
Configurations were measured once in order (shared 1–4, then instances 2–4),
with three warm repeats each. Peak memory is sampled, not an allocator high-water
mark. [Raw timing samples and calculated comparisons](common_model_slots_metrics.json)
include cold measurements as well.

## Correctness and lifecycle validation

- Full CPU build with all model families, CLI/server/GGUF tool and default tests:
  **49/49 CTests passed**.
- CUDA build with Higgs and BS-RoFormer: **5/5 focused lifecycle/config tests
  passed**. These five tests are also wired into Linux/macOS/Windows CI.
- Test doubles cover offline/native batches, streaming isolation/reset,
  capability validation, partial construction rollback, lifetime ordering,
  queue draining, timeouts, management priority and single-slot compatibility.
- Real HTTP tests compare **27 Higgs WAVs** with matching validated upstream-based
  cold/warm references; every output is byte-identical. Cases include four-way
  lazy loading, six-request queue, errors, timeouts, unload/reload, eviction and
  targeted/all-model unload during first load.
- All **76 benchmark WAVs** are byte-identical to those matching single-slot
  cold/warm references. Concurrent activity was observed for every configuration.
- Real BS-RoFormer single-slot vocals and instrumental match the separate
  upstream-based reference server exactly; requesting two slots is rejected
  cleanly, and subsequent Higgs inference recovers.
- Loader/catalog synchronization and its self-tests pass.

Original Higgs cold and warm reference-cache paths produce different WAVs;
comparisons preserve the same cache state. Real concurrent streaming/native-batch
adapters, dynamic native UI reconfiguration and non-CUDA parallel adapters have
not been exercised. Continuous token batching and in-flight GPU cancellation are
outside this change. Sixteen is the framework capability limit, not a tested
VRAM guarantee; real-GPU measurements here cover one through four slots.

## Reproduction

Create a task `request.json` with the text above, an absolute `voice_ref` path to
`assets/resources/b.wav`, seed 1234 and max_tokens 1024. Set `reference_text` to:

> Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you.

Run on an idle NVIDIA GPU with NVML available:

```sh
python tests/higgs_audio_tts/server_slots_bench.py --server build/windows-cuda-release/bin/audiocpp_server.exe --model models/higgs-audio-v3-tts.gguf --spec model_specs/higgs_audio_tts.json --request-json request.json --output bench-results --iterations 3
```

The standard-library harness launches and stops its own servers, saves config,
logs, WAVs, memory/activity samples and results, and asserts audio parity. Four
independent instances require enough VRAM for four model copies. If that does not
fit, benchmark fewer counts with `--slots 1 2`. Model SHA-256 is recorded in the
metrics JSON. See [the adapter guide](../maintainers/parallel_sessions.md) for
the pool contract and enabling another model.
