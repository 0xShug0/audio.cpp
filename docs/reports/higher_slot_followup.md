# Higher-slot follow-up: CUDA and Vulkan

Windows/MSVC, RTX 3090 (24 GiB), 2026-09-27. Based on PR #706 commit
`daccd36e`.

## Changes and evidence

- **FireRedTTS3 / CUDA:** the historical four-slot waveform mismatch did not
  reproduce with the current source. No additional inference patch was needed.
  The earlier shared tensor-source mapping protection is already present, but
  these retests do not prove which earlier change resolved that particular run.
- **RVC / CUDA:** the historical four-slot connection reset also did not
  reproduce. No additional inference patch was needed. Both models receive
  four-slot admission after the checks below, rather than a claimed new kernel fix.
- **Qwen3 TTS / CUDA:** four-slot waveform differences still reproduced: two
  outputs differed from their reference by up to 215 and 327 signed PCM units.
  A traced stress run produced 66 byte-identical speech-code buffers, including
  those corresponding to mismatched waveforms. Guard only speech waveform
  decoding with a mutex shared by the loaded package's sessions. Token/talker
  generation remains parallel. The underlying decoder/backend cause is not
  established; this is a stage-scoped correctness workaround. CPU and Vulkan
  do not acquire the decoder guard.
- **ZipVoice / Vulkan:** a cold four-slot run crashed through a null callback
  during backend tensor-buffer allocation. The debugger stack resolved through
  `BackendWeightStore::upload` and `ggml_backend_alloc_ctx_tensors*`. The Vulkan
  device loader publishes its device before finishing buffer-type initialization;
  ZipVoice initializes its backend lazily inside concurrent first requests.
  Serialize cold loading for Vulkan and automatic backend selection. Cached
  inference retains separate graphs/backends per slot. Explicit CUDA and CPU
  loading do not acquire the additional guard. This fixes the observed same-model
  cold-start race; cross-model simultaneous lazy initialization was not tested.
- **Yue2 / Vulkan:** the old survey reached a 23,895 MiB GPU peak and failed
  allocating a 3,001,177,088-byte VAE buffer on a warm four-slot request. A fresh
  reproduction also failed the host `ggml_init` allocation assertion before
  completing a wave. Bound Yue2's `no_alloc` host contexts by tensor-descriptor
  counts and existing graph capacities, with the requested sizes as upper limits.
  Tensor data still resides in backend buffers; precision, graph operations and
  generation settings are unchanged. Release the Vulkan VAE runtime/weights after
  successful audio decoding so they do not overlap the next request's AR/NAR
  stages. CUDA/CPU retain the existing VAE weight caching. Host reservation
  bounds apply to Yue2 on all backends.

No ggml CUDA/Vulkan kernels, generic weight-cache implementations, or model
arithmetic were changed. Only these five family/task capacities rise to four;
the admitted model/task counts remain 89 CUDA and 75 Vulkan. Other limits and
CPU/streaming fallback policies are unchanged.

## Checkpoint-backed results

Every family passed two single-slot reference cases, their serial repeats, and
comparison against the saved pre-follow-up executable. Parallel waves exercised
same requests, different text/seed cases (different audio lengths for RVC),
reversed request order, post-parallel serial reuse, unload/reload, and independent
cold server starts. Yue2 used its default generation settings, producing
69.919-second and 112.039-second stereo clips. Qwen3 used the existing audit's
256-token request limit. No quality or sampling setting was lowered by the fixes.

Each parallel wave returned HTTP 200, reported all requested slots active, and
matched the backend-specific single-slot audio and artifacts byte-for-byte.
Every pool drained; unload removed the loaded model and reload preserved output.
No VRAM guard triggered. CUDA families used three fresh starts per slot count,
ZipVoice five, and Yue2 two. Reload within a process is additional to those starts.

| Backend | Tested package / task | Three-slot exact outputs | Four-slot exact outputs | Four-slot GPU peak (MiB) |
| --- | --- | ---: | ---: | ---: |
| CUDA | FireRedTTS3 Instruct Q8_0 / cloning | 21/21 | 28/28 | 18,244 |
| CUDA | Qwen3 TTS 1.7B Base Q8_0 v2 / TTS | 21/21 | 28/28 | 16,076 |
| CUDA | RVC F16 / voice conversion | 21/21 | 28/28 | 8,146 |
| Vulkan | ZipVoice Distill Q8_0 / cloning | 27/27 | 36/36 | 923 |
| Vulkan | Yue2 3B Q8_0 + VAE F16 / generation | 18/18 | 24/24 | 22,817 |

The table totals **252 exact parallel outputs**, excluding serial controls.
Qwen3 additionally passed 64/64 outputs in a separate sixteen-wave four-slot
decoder-guard stress run. Preliminary FireRedTTS3 and RVC runs each passed 16/16
four-slot outputs; ZipVoice passed 12/12 at three and 32/32 at four; Yue2 passed
8/8 at four before the mixed-input validation. The final CUDA executable also
passed another 8/8 four-slot Qwen3 outputs after the last rebuild.

GPU figures are sampled total device usage, including desktop, model weights,
graphs and caches, rather than incremental per-slot memory. Active HTTP slots
do not imply every guarded GPU stage runs concurrently. The tested packages,
requests and device establish this validation scope; other checkpoints, much
larger requests and smaller GPUs can have different memory requirements.

## Yue2 host memory

| Vulkan reference/check pass | Peak private committed host memory |
| --- | ---: |
| Original, one slot / two reference cases | 31.20 GiB |
| Updated, one slot / same two cases and their repeats | 12.58 GiB |
| Updated, three slots / cold, warm, mixed, reuse and reload | 22.30 GiB |
| Updated, four slots / cold, warm, mixed, reuse and reload | 27.21 GiB |

These are sampled process-private committed bytes on Windows, including host
resources used by the driver; they are not resident RAM or VRAM. The updated
one-slot pass includes two additional repeat requests. No successful original
four-slot host-memory reference was available because the fresh run asserted.

## Compatibility and regression checks

- Saved original versus updated single-slot outputs were exact for both test
  cases on the other affected backends: ZipVoice CUDA, Yue2 CUDA and Qwen3 TTS
  Vulkan. Both updated serial repeats also matched. This checks that the guarded
  Vulkan loader, Yue2 host bounds and CUDA-only decoder guard preserve those
  paths. No cross-backend waveform equivalence is claimed.
- CUDA server/CLI and Vulkan server/CLI builds passed. The Vulkan diagnostic
  `/MAP` linker flag is no longer set.
- Eleven focused CTests passed in each backend build (22 total): shared weight
  cache, GGUF tensor source, session pool, server slots, single-slot compatibility,
  busy guard, configuration, F5 CPU backend and RoFormer accumulation; additionally
  CUDA iSTFT determinism/FSQ codec reuse, or Vulkan attention-row/shared-weight
  cache tests. Slot-policy assertions now require the five validated capacities
  of four. Untested families, CPU fallback and streaming policies remain covered.
- `git diff --check` passed.

## Artifacts

Workspace artifacts are under `outputs/model-two-slot-audit/higher-slot-followup/`:
`baseline/`, `qwen-trace-stress/`, `qwen-decoder-guard/`, `zip-debug-0/`,
`candidate-3/`, `candidate-4/`, `final/{cuda,vulkan}/`, `final-binary/`, and
`backend-controls/{cuda,vulkan}/` contain JSON responses,
saved WAV files, server logs and peaks. The local drivers are
`probe_higher_followup.py`, `validate_higher_final.py`, and
`control_higher_backends.py` in the parent audit
directory. Temporary Qwen speech-code instrumentation and the diagnostic Vulkan
linker-map flag were removed before the final validation/builds.
