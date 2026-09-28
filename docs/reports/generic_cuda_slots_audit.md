# CUDA two-slot model survey

Local experimental framework: independent sessions from one loaded checkpoint. Legacy models can duplicate GPU weights. These tests cover offline CUDA only.

## Enabled policy in PR #706

The generic CUDA offline factory is restricted to the 75 passing family/task
pairs in `app/server/audited_model_slots.h`. Subsequent
[three/four-slot testing](generic_cuda_slots_3_4_audit.md) sets each pair's
highest validated capacity: 58 families at four, nine at three and eight at two.
Families
requiring investigation, unverified parity, CPU-only variants and hardware-blocked
packages remain single-slot on CUDA. Untested tasks are not enabled by this
CUDA table; Vulkan uses its own [audited table](generic_vulkan_slots_audit.md). Higgs retains its specialized shared-weight factory and existing
capacity. The survey below exercised the broader experimental factory to find
the supported subset; its failures are not enabled by the final allowlist.

94 families recorded. Status counts: BLOCKED_HARDWARE: 1, FAIL_OUTPUT_PARITY: 3, FAIL_TWO_SLOT_REQUEST: 9, NOT_TESTED_CPU_ONLY: 2, PASS_TWO_SLOTS_EXACT: 75, RUNS_PARITY_UNVERIFIED: 4.

An exact pass requires overlapping requests, successful cold and warm runs, output equality with a single-slot reference, and no remaining leases. Two slots passing does not validate higher slot counts or every supported task.

| Family | Status | 1-slot warm (s) | 2-request warm batch (s) | 1-slot / 2-slot GPU MiB after warmup |
|---|---|---:|---:|---:|
| ace_step | FAIL_TWO_SLOT_REQUEST | 7.437 | — | 5684 / — |
| apollo | PASS_TWO_SLOTS_EXACT | 0.277 | 0.518 | 1182 / 2136 |
| audio8_asr | PASS_TWO_SLOTS_EXACT | 0.055 | 0.067 | 898 / 1536 |
| audio8_tts | PASS_TWO_SLOTS_EXACT | 0.665 | 0.899 | 1586 / 2918 |
| audiosr | FAIL_OUTPUT_PARITY | 5.125 | 8.847 | 2930 / 5606 |
| auk | PASS_TWO_SLOTS_EXACT | 1.192 | 2.217 | 6030 / 11804 |
| breeze_tts | PASS_TWO_SLOTS_EXACT | 1.216 | 1.689 | 4244 / 8230 |
| bs_roformer | PASS_TWO_SLOTS_EXACT | 0.730 | 1.375 | 1516 / 2776 |
| canary_asr | PASS_TWO_SLOTS_EXACT | 0.059 | 0.081 | 626 / 998 |
| chatterbox | FAIL_TWO_SLOT_REQUEST | 0.760 | — | 2408 / — |
| chatterbox_turbo | FAIL_TWO_SLOT_REQUEST | 0.636 | — | 1852 / — |
| citrinet_asr | PASS_TWO_SLOTS_EXACT | 0.019 | 0.025 | 318 / 408 |
| cohere_asr | PASS_TWO_SLOTS_EXACT | 0.107 | 0.124 | 2802 / 5348 |
| confucius4_r2t2 | PASS_TWO_SLOTS_EXACT | 0.173 | 0.230 | 3040 / 5818 |
| confucius4_tts | PASS_TWO_SLOTS_EXACT | 1.926 | 3.335 | 6698 / 13142 |
| controlfoley | PASS_TWO_SLOTS_EXACT | 5.009 | 8.309 | 11806 / 23356 |
| cosyvoice3 | PASS_TWO_SLOTS_EXACT | 0.583 | 0.871 | 2118 / 3978 |
| dots_tts | PASS_TWO_SLOTS_EXACT | 1.703 | 2.534 | 4132 / 8010 |
| dramabox | FAIL_TWO_SLOT_REQUEST | 7.023 | — | 22212 / — |
| echo_tts | PASS_TWO_SLOTS_EXACT | 3.239 | 6.142 | 6316 / 12342 |
| f5_tts | FAIL_TWO_SLOT_REQUEST | 1.816 | — | 3250 / — |
| firered_audio | PASS_TWO_SLOTS_EXACT | 1.421 | 18.069 | 12654 / 24318 |
| fireredtts3 | PASS_TWO_SLOTS_EXACT | 1.150 | 1.685 | 4726 / 9196 |
| fish_audio | PASS_TWO_SLOTS_EXACT | 1.489 | 2.445 | 5900 / 11544 |
| fun_asr_nano | PASS_TWO_SLOTS_EXACT | 0.164 | 0.225 | 1954 / 3650 |
| glm_tts | PASS_TWO_SLOTS_EXACT | 0.777 | 1.233 | 5350 / 10442 |
| granite5asr | PASS_TWO_SLOTS_EXACT | 0.028 | 0.038 | 880 / 1502 |
| heartmula | PASS_TWO_SLOTS_EXACT | 9.646 | 16.610 | 7368 / 14454 |
| higgs_audio_stt | PASS_TWO_SLOTS_EXACT | 0.200 | 0.300 | 3466 / 6670 |
| higgs_audio_tts | PASS_TWO_SLOTS_EXACT | 0.631 | 1.076 | 5204 / 5322 |
| htdemucs | PASS_TWO_SLOTS_EXACT | 0.481 | 0.835 | 858 / 1456 |
| hviske_asr | PASS_TWO_SLOTS_EXACT | 0.152 | 0.224 | 2812 / 5360 |
| index_tts2 | PASS_TWO_SLOTS_EXACT | 1.222 | 1.919 | 5980 / 11706 |
| inflect_v2 | PASS_TWO_SLOTS_EXACT | 0.059 | 0.065 | 392 / 556 |
| irodori_tts | PASS_TWO_SLOTS_EXACT | 0.624 | 1.086 | 1544 / 2842 |
| kitten_tts | PASS_TWO_SLOTS_EXACT | 1.235 | 1.776 | 1700 / 3174 |
| kokoro_tts | PASS_TWO_SLOTS_EXACT | 0.165 | 0.222 | 1236 / 2214 |
| kroko_asr | PASS_TWO_SLOTS_EXACT | 0.120 | 0.181 | 526 / 798 |
| liveavatar | FAIL_TWO_SLOT_REQUEST | 14.454 | — | 1474 / — |
| magpie_tts | PASS_TWO_SLOTS_EXACT | 0.574 | 0.797 | 1362 / 2466 |
| meanvc2 | PASS_TWO_SLOTS_EXACT | 0.300 | 0.361 | 4430 / 8604 |
| mel_band_roformer | PASS_TWO_SLOTS_EXACT | 0.317 | 0.578 | 1390 / 2524 |
| midashenglm_gen | PASS_TWO_SLOTS_EXACT | 3.685 | 6.049 | 3890 / 7528 |
| minimax_h3 | BLOCKED_HARDWARE | — | — | — / — |
| minimax_music3 | PASS_TWO_SLOTS_EXACT | 34.615 | 61.727 | 1882 / 3506 |
| miocodec | RUNS_PARITY_UNVERIFIED | 0.060 | 0.079 | 3758 / 7282 |
| miotts | RUNS_PARITY_UNVERIFIED | 0.429 | 0.680 | 4662 / 9058 |
| mira_tts | PASS_TWO_SLOTS_EXACT | 0.626 | 1.011 | 2488 / 4716 |
| mms_forced_aligner | FAIL_TWO_SLOT_REQUEST | 0.048 | — | 762 / — |
| moonshine_asr | PASS_TWO_SLOTS_EXACT | 0.033 | 0.028 | 334 / 412 |
| moss_transcribe_diarize | PASS_TWO_SLOTS_EXACT | 0.168 | 0.260 | 1648 / 3036 |
| moss_tts_local | FAIL_OUTPUT_PARITY | 0.800 | 7.528 | 17012 / 24300 |
| moss_tts_nano | PASS_TWO_SLOTS_EXACT | 0.538 | 0.734 | 806 / 1358 |
| moss_tts_v15 | PASS_TWO_SLOTS_EXACT | 1.075 | 4.475 | 12926 / 24318 |
| moss_ttsd | FAIL_OUTPUT_PARITY | 2.141 | 15.211 | 13082 / 24316 |
| moss_voicegen | PASS_TWO_SLOTS_EXACT | 0.557 | 0.889 | 7790 / 15320 |
| muscriptor | PASS_TWO_SLOTS_EXACT | 0.052 | 0.075 | 730 / 1200 |
| nemotron_3_diar | PASS_TWO_SLOTS_EXACT | 0.028 | 0.044 | 388 / 518 |
| nemotron_asr | PASS_TWO_SLOTS_EXACT | 0.048 | 0.067 | 1294 / 2298 |
| neutts | RUNS_PARITY_UNVERIFIED | 1.065 | 1.234 | 1814 / 3364 |
| niagara_asr | NOT_TESTED_CPU_ONLY | — | — | — / — |
| omnivoice | PASS_TWO_SLOTS_EXACT | 0.815 | 1.461 | 1986 / 3714 |
| outetts | PASS_TWO_SLOTS_EXACT | 1.499 | 2.664 | 6784 / 13308 |
| parakeet_tdt | PASS_TWO_SLOTS_EXACT | 0.044 | 0.050 | 2040 / 3824 |
| personaplex | PASS_TWO_SLOTS_EXACT | 5.415 | 62.922 | 13954 / 24312 |
| piper_tts | PASS_TWO_SLOTS_EXACT | 0.040 | 0.060 | 418 / 612 |
| pocket_tts | PASS_TWO_SLOTS_EXACT | 0.062 | 0.091 | 630 / 998 |
| pulsevad | PASS_TWO_SLOTS_EXACT | 0.016 | 0.024 | 228 / 228 |
| qwen3_asr | PASS_TWO_SLOTS_EXACT | 0.201 | 0.270 | 3150 / 6038 |
| qwen3_forced_aligner | PASS_TWO_SLOTS_EXACT | 0.041 | 0.054 | 1544 / 2830 |
| qwen3_tts | PASS_TWO_SLOTS_EXACT | 0.773 | 0.989 | 3980 / 7706 |
| rvc | PASS_TWO_SLOTS_EXACT | 0.316 | 0.492 | 2228 / 4200 |
| sanotts | PASS_TWO_SLOTS_EXACT | 0.011 | 0.015 | 270 / 316 |
| seed_vc | FAIL_TWO_SLOT_REQUEST | 1.181 | — | 3102 / — |
| sense_asr | PASS_TWO_SLOTS_EXACT | 0.070 | 0.093 | 588 / 922 |
| sheetsage2 | PASS_TWO_SLOTS_EXACT | 2.142 | 3.435 | 2830 / 5430 |
| soprano_tts | PASS_TWO_SLOTS_EXACT | 0.122 | 0.146 | 744 / 1232 |
| sopro_tts | PASS_TWO_SLOTS_EXACT | 0.133 | 0.178 | 1010 / 1762 |
| sortformer_diar | PASS_TWO_SLOTS_EXACT | 0.033 | 0.055 | 2480 / 4732 |
| sortformer_diar_v2 | PASS_TWO_SLOTS_EXACT | 0.039 | 0.050 | 1238 / 2248 |
| stable_audio | PASS_TWO_SLOTS_EXACT | 3.030 | 5.735 | 3634 / 7012 |
| supertonic | PASS_TWO_SLOTS_EXACT | 0.078 | 0.081 | 700 / 1170 |
| universr | PASS_TWO_SLOTS_EXACT | 2.127 | 4.124 | 2542 / 4858 |
| vevo2 | FAIL_TWO_SLOT_REQUEST | 1.176 | — | 3574 / — |
| vibeasr | NOT_TESTED_CPU_ONLY | — | — | — / — |
| vibevoice | PASS_TWO_SLOTS_EXACT | 0.603 | 0.946 | 3926 / 7590 |
| vibevoice_asr | PASS_TWO_SLOTS_EXACT | 0.634 | 1.085 | 9808 / 19360 |
| vibevoice_asr_streaming | PASS_TWO_SLOTS_EXACT | 0.661 | 1.125 | 9748 / 19238 |
| vieneu_v3_turbo | RUNS_PARITY_UNVERIFIED | 0.477 | 0.564 | 1012 / 1768 |
| voxcpm1 | PASS_TWO_SLOTS_EXACT | 0.517 | 0.728 | 1488 / 2722 |
| voxcpm2 | PASS_TWO_SLOTS_EXACT | 1.003 | 1.515 | 2982 / 5710 |
| voxtral_realtime | PASS_TWO_SLOTS_EXACT | 0.635 | 1.056 | 5500 / 10744 |
| yue2 | PASS_TWO_SLOTS_EXACT | 18.255 | 29.993 | 796 / 1322 |
| zipvoice | PASS_TWO_SLOTS_EXACT | 0.201 | 0.333 | 552 / 850 |

GPU measurements include loaded weights, runtime caches and retained buffers; they are not isolated KV-cache measurements or request peaks.

## Cases needing investigation

- **ace_step — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [False, False].
- **audiosr — FAIL_OUTPUT_PARITY**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: 51.16–52.11 dB. This measures numerical difference, not perceptual quality.
- **chatterbox — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [False, False].
- **chatterbox_turbo — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [False, False].
- **dramabox — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [500, 200]; equal to serial output: [False, True].
- **f5_tts — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Diagnostic error: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host
- **liveavatar — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [True, True].
- **minimax_h3 — BLOCKED_HARDWARE**: "Package is 30.2 GiB, exceeding 24 GiB GPU; no safe two-slot CUDA baseline."
- **miocodec — RUNS_PARITY_UNVERIFIED**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: 110.58–111.89 dB. This measures numerical difference, not perceptual quality.
- **miotts — RUNS_PARITY_UNVERIFIED**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: 110.56–113.57 dB. This measures numerical difference, not perceptual quality.
- **mms_forced_aligner — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [True, True].
- **moss_tts_local — FAIL_OUTPUT_PARITY**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - At least one concurrent waveform has a different sample count from its reference.
- **moss_ttsd — FAIL_OUTPUT_PARITY**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: 52.44–52.44 dB. This measures numerical difference, not perceptual quality.
- **neutts — RUNS_PARITY_UNVERIFIED**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: 108.38–109.47 dB. This measures numerical difference, not perceptual quality.
- **niagara_asr — NOT_TESTED_CPU_ONLY**: "Selected variant uses CPU-only kernels; CUDA two-slot inference is inapplicable."
- **seed_vc — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [False, True].
- **vevo2 — FAIL_TWO_SLOT_REQUEST**: "two-slot: ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote host"
  - PCM signal-to-error ratio across comparable outputs: -2.99–-2.87 dB. This measures numerical difference, not perceptual quality.
  - Serial request loading two sessions: HTTP 200.
  - Pair after serial warmup: HTTP [200, 200]; equal to serial output: [True, True].
- **vibeasr — NOT_TESTED_CPU_ONLY**: "Selected variant uses CPU-only kernels; CUDA two-slot inference is inapplicable."
- **vieneu_v3_turbo — RUNS_PARITY_UNVERIFIED**: "Concurrent output differs; requires investigation, not proof of a concurrency defect."
  - PCM signal-to-error ratio across comparable outputs: -2.92–-1.78 dB. This measures numerical difference, not perceptual quality.

## Scope and limitations

- Windows, RTX 3090 24 GiB, full CUDA build. Tested one selected package and one offline task per family, using saved requests and fixtures; not every model variant or task.
- Main cases compare identical concurrent requests against cold/warm single-slot references and observe two loaded active slots. This is a concurrency smoke/parity survey, not exhaustive cross-request isolation validation.
- 75 exact passes are for these fixtures only. No guarantee is made for three or more slots, long prompts or larger contexts.
- AuK passed text-only TTS. Reference-audio conditioning requires Soxr, absent from this build.
- LiveAvatar used a 256x256 frame extracted from the provided video, five output frames and four inference steps; video quality was not evaluated.
- Some TTS cases use explicit token limits. Full request settings are in the JSON evidence.
- The CUDA table does not enable other backends. Vulkan now has a separate
  [audited offline table](generic_vulkan_slots_audit.md); CPU and streaming
  still require explicit specialized support.
- No CUDA kernels or model inference implementations were changed. The audited subset is enabled in PR #706; the failing models remain disabled for generic multi-slot execution.
- Framework checks: session_pool_test, server_model_slots_test, server_busy_guard_test and server_config_test all passed.
