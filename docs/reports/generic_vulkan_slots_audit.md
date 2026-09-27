# Vulkan parallel-slot model audit

The [allocation follow-up](vulkan_slots_allocation_investigation.md) validates
two-slot weight sharing for FireRed Audio, HeartMuLa, MOSS TTS v1.5 and
PersonaPlex. The [connection-reset follow-up](vulkan_slots_connection_resets.md)
also validates GLM TTS, OuteTTS, SheetSage2 and Stable Audio at two slots.
The [Inflect/Mel-Band follow-up](vulkan_inflect_mel_slots.md) fixes the Inflect
single-slot assertion and Mel-Band host race, validating both at four slots.
The audited Vulkan table now includes 73 family/task pairs; the original survey
results below remain a record of the earlier implementation.

Date: 2026-09-27. Windows/MSVC Release, RTX 3090 24 GiB, eight CPU threads.

Tested all 75 model/task packages that had passed the earlier CUDA two-slot audit, on the RTX 3090 using Vulkan device 1 and an audit server with temporary Vulkan eligibility. These results now define the separate Vulkan offline allowlist in `app/server/audited_model_slots.h`. The 3/4-slot test ran only for exact two-slot passes and within each family's existing audited slot capacity.

## Results

- **Two slots:** 63/75 passed cold and warm concurrent requests with exact normalized output parity, observed full overlap, and drained leases. The table gives concurrent batch wall time as cold/warm seconds.
- **Three/four slots:** 109 of 126 count/model cases passed with exact output parity and all requested slots observed active. The table gives cold/warm batch wall time and sampled peak GPU memory.
- Higher-slot cases: 13 skipped above current audited capacity; 1 skipped by VRAM preflight; 1 Vulkan allocation failure; 2 unstable ZipVoice cases.
- Of the 63 exact two-slot passes, 60 were within the audited limit for three slots and 53 for four slots.

Highest sampled peak among successful 3/4-slot cases: 22,832 MiB (22.30 GiB), echo_tts at 3 slots.
Yue2 at four slots reached 23,895 MiB before a warm request failed to allocate a 3,001,177,088-byte Vulkan buffer; the device guard was 24,064 MiB. Minimax Music3 separately peaked at 24,021 MiB during the two-slot run and passed exactness.

## Per-package outcomes

Higher-slot cells show observed batch wall time, not single-request latency.

| Model/task family | Tested package | Task | Two slots (cold/warm) | Three slots (cold/warm; peak) | Four slots (cold/warm; peak) |
|---|---|---|---|---|---|
| `bs_roformer` | `bs_roformer_q8_0` | `sep` | PASS 3.5/3.0 s cold/warm | PASS 5.3/4.5 s cold/warm; 8684 MiB peak | PASS 6.7/5.9 s cold/warm; 11569 MiB peak |
| `higgs_audio_tts` | `higgs_audio_tts_4b_q8_0` | `tts` | PASS 6.1/1.4 s cold/warm | PASS 11.0/2.2 s cold/warm; 5479 MiB peak | PASS 8.4/2.9 s cold/warm; 6018 MiB peak |
| `apollo` | `apollo_orig` | `s2s` | PASS 0.9/0.5 s cold/warm | PASS 1.3/0.7 s cold/warm; 3163 MiB peak | PASS 1.6/1.0 s cold/warm; 4214 MiB peak |
| `audio8_asr` | `audio8_asr_0_1b_q8_0` | `asr` | PASS 2.8/0.1 s cold/warm | PASS 3.3/0.1 s cold/warm; 1690 MiB peak | PASS 3.3/0.1 s cold/warm; 2248 MiB peak |
| `audio8_tts` | `audio8_tts_preview_0_6b_q8_0` | `tts` | PASS 4.3/1.1 s cold/warm | PASS 7.2/1.5 s cold/warm; 6658 MiB peak | PASS 8.0/2.0 s cold/warm; 8850 MiB peak |
| `auk` | `auk_base_q8_0` | `tts` | BLOCKED | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `breeze_tts` | `breeze_tts_2_q8_0` | `vdes` | PASS 12.2/3.6 s cold/warm | PASS 21.9/5.8 s cold/warm; 13571 MiB peak | PASS 20.5/7.4 s cold/warm; 17840 MiB peak |
| `canary_asr` | `canary_180m_flash_q8_0` | `asr` | PASS 0.7/0.1 s cold/warm | PASS 1.2/0.1 s cold/warm; 955 MiB peak | PASS 1.0/0.2 s cold/warm; 1268 MiB peak |
| `citrinet_asr` | `citrinet_asr_q8_0` | `asr` | PASS 0.3/0.0 s cold/warm | PASS 0.5/0.0 s cold/warm; 147 MiB peak | PASS 0.4/0.0 s cold/warm; 190 MiB peak |
| `cohere_asr` | `cohere_transcribe_q8_0` | `asr` | PASS 4.0/0.2 s cold/warm | PASS 7.2/0.3 s cold/warm; 7297 MiB peak | PASS 5.9/0.3 s cold/warm; 9722 MiB peak |
| `confucius4_r2t2` | `confucius4_r2t2_q8_0` | `asr` | PASS 4.0/0.3 s cold/warm | PASS 7.5/0.4 s cold/warm; 8122 MiB peak | PASS 6.9/0.5 s cold/warm; 10814 MiB peak |
| `confucius4_tts` | `confucius4_tts_orig` | `clon` | PASS 15.6/3.3 s cold/warm | PASS 28.6/5.0 s cold/warm; 20647 MiB peak | SKIP (audited cap) |
| `controlfoley` | `controlfoley_large_44k_q8_0` | `gen` | VRAM STOP | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `cosyvoice3` | `cosyvoice3_q8_0` | `clon` | PASS 5.6/1.2 s cold/warm | PASS 9.1/1.7 s cold/warm; 5662 MiB peak | PASS 9.4/2.2 s cold/warm; 8033 MiB peak |
| `dots_tts` | `dots_tts_soar_q8_0` | `tts` | PASS 8.5/4.0 s cold/warm | PASS 14.9/6.0 s cold/warm; 14955 MiB peak | PASS 15.4/8.2 s cold/warm; 18549 MiB peak |
| `echo_tts` | `echo_tts_q8_0` | `clon` | PASS 12.7/10.5 s cold/warm | PASS 22.0/15.8 s cold/warm; 22832 MiB peak | SKIP (audited cap) |
| `firered_audio` | `firered_audio_q8_0` | `clon` | OOM | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `fireredtts3` | `fireredtts3_instruct_q8_0` | `clon` | PASS 8.5/4.6 s cold/warm | PASS 14.9/7.1 s cold/warm; 12955 MiB peak | SKIP (audited cap) |
| `fish_audio` | `fish_audio_s2_pro_q8_0` | `tts` | PASS 13.2/2.6 s cold/warm | PASS 25.2/3.8 s cold/warm; 19246 MiB peak | SKIP (audited cap) |
| `fun_asr_nano` | `fun_asr_nano_2512_q8_0` | `asr` | PASS 2.8/0.2 s cold/warm | PASS 4.9/0.3 s cold/warm; 3701 MiB peak | PASS 4.6/0.4 s cold/warm; 4921 MiB peak |
| `glm_tts` | `glm_tts_q8_0` | `tts` | RESET | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `granite5asr` | `granite5asr_q8_0` | `asr` | PASS 0.7/0.1 s cold/warm | PASS 1.3/0.1 s cold/warm; 1877 MiB peak | PASS 1.0/0.1 s cold/warm; 2556 MiB peak |
| `heartmula` | `heartmula_q8_0` | `gen` | OOM | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `higgs_audio_stt` | `higgs_audio_stt_q8_0` | `asr` | PASS 4.3/0.4 s cold/warm | PASS 7.7/0.6 s cold/warm; 9637 MiB peak | PASS 6.7/0.7 s cold/warm; 12832 MiB peak |
| `htdemucs` | `htdemucs_q8_0` | `sep` | PASS 1.1/0.8 s cold/warm | PASS 1.6/1.1 s cold/warm; 1702 MiB peak | PASS 1.9/1.4 s cold/warm; 2262 MiB peak |
| `hviske_asr` | `hviske_asr_q8_0` | `asr` | PASS 5.1/0.3 s cold/warm | PASS 9.2/0.3 s cold/warm; 8275 MiB peak | PASS 9.2/0.4 s cold/warm; 12659 MiB peak |
| `index_tts2` | `index_tts2_q8_0` | `tts` | PASS 16.8/3.4 s cold/warm | PASS 30.5/5.3 s cold/warm; 22395 MiB peak | SKIP (VRAM estimate) |
| `inflect_v2` | `inflect_micro_v2_orig` | `tts` | SINGLE SLOT ASSERT | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `irodori_tts` | `irodori_tts_v4_small_q8_0` | `tts` | PASS 4.5/1.5 s cold/warm | PASS 7.7/2.1 s cold/warm; 8795 MiB peak | PASS 8.0/2.8 s cold/warm; 10396 MiB peak |
| `kitten_tts` | `kitten_tts_mini_0_8_orig` | `tts` | PASS 1.4/0.5 s cold/warm | PASS 2.3/0.7 s cold/warm; 2060 MiB peak | PASS 2.5/1.0 s cold/warm; 2740 MiB peak |
| `kokoro_tts` | `kokoro_82m_q8_0` | `tts` | PASS 2.5/0.2 s cold/warm | PASS 3.1/0.2 s cold/warm; 2653 MiB peak | PASS 3.3/0.3 s cold/warm; 3532 MiB peak |
| `kroko_asr` | `kroko_asr_community_q8_0` | `asr` | PASS 0.8/0.4 s cold/warm | PASS 1.2/0.6 s cold/warm; 557 MiB peak | PASS 1.3/0.8 s cold/warm; 736 MiB peak |
| `magpie_tts` | `magpie_tts_q8_0` | `tts` | PASS 6.2/2.3 s cold/warm | PASS 7.5/2.4 s cold/warm; 3894 MiB peak | PASS 7.4/3.1 s cold/warm; 4810 MiB peak |
| `meanvc2` | `meanvc2_120ms_40ms_f32` | `vc` | PASS 2.3/0.5 s cold/warm | PASS 4.7/0.7 s cold/warm; 14251 MiB peak | PASS 4.4/0.9 s cold/warm; 18398 MiB peak |
| `mel_band_roformer` | `mel_band_roformer_q8_0` | `sep` | PARITY UNVERIFIED | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `midashenglm_gen` | `midashenglm_gen_q8_0` | `gen` | PASS 13.2/11.4 s cold/warm | PASS 21.6/17.0 s cold/warm; 11456 MiB peak | PASS 25.4/22.5 s cold/warm; 15316 MiB peak |
| `minimax_music3` | `minimax_music3_q8_0` | `gen` | PASS 89.1/87.9 s cold/warm | SKIP (audited cap) | SKIP (audited cap) |
| `mira_tts` | `mira_tts_local_bf16` | `tts` | PASS 7.4/1.2 s cold/warm | PASS 12.0/1.9 s cold/warm; 7996 MiB peak | PASS 12.7/2.5 s cold/warm; 12195 MiB peak |
| `moonshine_asr` | `moonshine_streaming_tiny_q8_0` | `asr` | PASS 0.4/0.1 s cold/warm | PASS 0.5/0.1 s cold/warm; 290 MiB peak | PASS 0.5/0.1 s cold/warm; 366 MiB peak |
| `moss_transcribe_diarize` | `moss_transcribe_diarize_q8_0` | `asr` | PASS 2.8/0.3 s cold/warm | PASS 4.9/0.4 s cold/warm; 4237 MiB peak | PASS 4.6/0.5 s cold/warm; 5639 MiB peak |
| `moss_tts_nano` | `moss_tts_nano_100m_q8_0` | `tts` | PASS 1.6/1.0 s cold/warm | PASS 2.4/1.3 s cold/warm; 1607 MiB peak | PASS 2.6/1.7 s cold/warm; 2134 MiB peak |
| `moss_tts_v15` | `moss_tts_v15_q8_0_codec_f16` | `tts` | OOM | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `moss_voicegen` | `moss_voicegen_bf16_codec_f16_decode` | `vdes` | PASS 8.3/1.3 s cold/warm | PASS 11.9/2.0 s cold/warm; 22607 MiB peak | SKIP (audited cap) |
| `muscriptor` | `muscriptor_small_f32` | `midi` | PASS 0.7/0.1 s cold/warm | PASS 1.0/0.1 s cold/warm; 1513 MiB peak | PASS 1.0/0.1 s cold/warm; 2011 MiB peak |
| `nemotron_3_diar` | `nemotron_3_diar_q8_0` | `diar` | PASS 0.3/0.0 s cold/warm | PASS 0.5/0.0 s cold/warm; 395 MiB peak | PASS 0.5/0.1 s cold/warm; 500 MiB peak |
| `nemotron_asr` | `nemotron_asr_q8_0` | `asr` | PASS 1.8/0.1 s cold/warm | PASS 2.9/0.1 s cold/warm; 2960 MiB peak | PASS 2.9/0.2 s cold/warm; 3939 MiB peak |
| `omnivoice` | `omnivoice_q8_0` | `tts` | PASS 5.0/1.5 s cold/warm | PASS 8.1/2.3 s cold/warm; 6010 MiB peak | PASS 8.6/3.2 s cold/warm; 7814 MiB peak |
| `outetts` | `outetts_1_0_1b_q8_0` | `tts` | RESET | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `parakeet_tdt` | `parakeet_tdt_q8_0` | `asr` | PASS 3.0/0.1 s cold/warm | PASS 5.5/0.1 s cold/warm; 5225 MiB peak | PASS 5.4/0.1 s cold/warm; 6962 MiB peak |
| `personaplex` | `personaplex_7b_v1_q8_0` | `s2s` | OOM | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `piper_tts` | `piper_lessac_medium_orig` | `tts` | PASS 0.4/0.0 s cold/warm | PASS 0.5/0.1 s cold/warm; 559 MiB peak | PASS 0.5/0.1 s cold/warm; 741 MiB peak |
| `pocket_tts` | `pocket_tts_english_q8_0` | `tts` | PASS 0.5/0.1 s cold/warm | PASS 0.8/0.2 s cold/warm; 1121 MiB peak | PASS 0.8/0.2 s cold/warm; 1490 MiB peak |
| `pulsevad` | `pulsevad_2_1k_f32` | `vad` | PASS 0.2/0.0 s cold/warm | PASS 0.2/0.0 s cold/warm; 11 MiB peak | PASS 0.2/0.1 s cold/warm; 11 MiB peak |
| `qwen3_asr` | `qwen3_asr_1_7b_q8_0` | `asr` | PASS 4.0/0.3 s cold/warm | PASS 6.9/0.4 s cold/warm; 8449 MiB peak | PASS 7.3/0.5 s cold/warm; 11251 MiB peak |
| `qwen3_forced_aligner` | `qwen3_forced_aligner_0_6b_q8_0` | `align` | PASS 1.6/0.1 s cold/warm | PASS 3.2/0.1 s cold/warm; 3638 MiB peak | PASS 3.1/0.1 s cold/warm; 4843 MiB peak |
| `qwen3_tts` | `qwen3_tts_1_7b_base_q8_0` | `tts` | PASS 5.3/1.2 s cold/warm | PASS 9.3/1.7 s cold/warm; 11733 MiB peak | SKIP (audited cap) |
| `rvc` | `rvc_f16` | `vc` | PASS 21.3/0.6 s cold/warm | PASS 20.1/0.8 s cold/warm; 7040 MiB peak | SKIP (audited cap) |
| `sanotts` | `sanotts_heart_nano_orig` | `tts` | PASS 0.2/0.0 s cold/warm | PASS 0.2/0.0 s cold/warm; 25 MiB peak | PASS 0.2/0.0 s cold/warm; 31 MiB peak |
| `sense_asr` | `sensevoice_small_q8` | `asr` | PASS 0.6/0.1 s cold/warm | PASS 1.0/0.2 s cold/warm; 789 MiB peak | PASS 1.0/0.3 s cold/warm; 1046 MiB peak |
| `sheetsage2` | `sheetsage2_orig` | `midi` | RESET | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `soprano_tts` | `soprano_1_1_80m_q8_0` | `tts` | PASS 0.9/0.2 s cold/warm | PASS 1.3/0.2 s cold/warm; 1465 MiB peak | PASS 1.5/0.3 s cold/warm; 1907 MiB peak |
| `sopro_tts` | `sopro_v2_turbo_f16` | `tts` | PASS 1.6/0.2 s cold/warm | PASS 2.3/0.3 s cold/warm; 2181 MiB peak | PASS 2.6/0.4 s cold/warm; 2881 MiB peak |
| `sortformer_diar` | `sortformer_diar_4spk_v1_q8_0` | `diar` | PASS 0.9/0.0 s cold/warm | PASS 1.4/0.1 s cold/warm; 6636 MiB peak | PASS 1.6/0.1 s cold/warm; 8843 MiB peak |
| `sortformer_diar_v2` | `sortformer_diar_v2_1_f32_gguf_local` | `diar` | PASS 1.0/0.1 s cold/warm | PASS 1.5/0.1 s cold/warm; 2895 MiB peak | PASS 1.7/0.1 s cold/warm; 3856 MiB peak |
| `stable_audio` | `stable_audio_3_medium_q8_0` | `gen` | RESET | not run (two-slot baseline not exact/pass) | not run (two-slot baseline not exact/pass) |
| `supertonic` | `supertonic_3_q8_0` | `tts` | PASS 2.9/0.2 s cold/warm | PASS 3.4/0.2 s cold/warm; 1222 MiB peak | PASS 3.5/0.3 s cold/warm; 1625 MiB peak |
| `universr` | `universr_audio_orig` | `s2s` | PASS 5.0/4.4 s cold/warm | PASS 7.7/6.6 s cold/warm; 9035 MiB peak | PASS 10.0/8.9 s cold/warm; 11632 MiB peak |
| `vibevoice` | `vibevoice_1_5b_q8_0` | `tts` | PASS 5.0/1.5 s cold/warm | PASS 9.0/2.0 s cold/warm; 12435 MiB peak | PASS 8.3/2.7 s cold/warm; 15398 MiB peak |
| `vibevoice_asr` | `vibevoice_asr_q8_0` | `asr` | PASS 11.5/1.3 s cold/warm | SKIP (audited cap) | SKIP (audited cap) |
| `vibevoice_asr_streaming` | `vibevoice_asr_streaming_7b_q8_0` | `asr` | PASS 11.7/1.4 s cold/warm | SKIP (audited cap) | SKIP (audited cap) |
| `voxcpm1` | `voxcpm1_0_5b_q8_0` | `tts` | PASS 3.0/1.3 s cold/warm | PASS 5.3/1.9 s cold/warm; 6163 MiB peak | PASS 5.5/2.5 s cold/warm; 7824 MiB peak |
| `voxcpm2` | `voxcpm2_q8_0` | `tts` | PASS 5.8/2.2 s cold/warm | PASS 10.5/3.1 s cold/warm; 14247 MiB peak | PASS 10.0/4.1 s cold/warm; 19802 MiB peak |
| `voxtral_realtime` | `voxtral_realtime_q8_0` | `asr` | PASS 6.7/1.2 s cold/warm | PASS 11.6/1.8 s cold/warm; 15681 MiB peak | PASS 12.1/2.3 s cold/warm; 20973 MiB peak |
| `yue2` | `yue2_main_q8_0` | `gen` | PASS 29.7/26.6 s cold/warm | PASS 49.8/45.3 s cold/warm; 18040 MiB peak | OOM |
| `zipvoice` | `zipvoice_distill_gguf` | `clon` | PASS 1.5/0.6 s cold/warm | UNSTABLE | UNSTABLE |

## Two-slot cases needing follow-up

- **Vulkan allocation failures:** FireRed Audio, HeartMuLa, Moss TTS v1.5, and PersonaPlex returned `ErrorOutOfDeviceMemory` while allocating Vulkan model/inference buffers.
- **Stopped by VRAM guard:** ControlFoley reached the configured total-VRAM safety limit; its run was stopped before overcommit.
- **Connection resets:** GLM TTS, OuteTTS, SheetSage2, and Stable Audio reset the request connection without a matching Vulkan allocation error in their logs.
- **Single-slot blockers:** AuK reports that its native session currently requires CUDA. Inflect V2 hit a Vulkan GGML alignment assertion before its single-slot reference was established.
- **Parity baseline issue:** Mel-Band RoFormer varied between cold/warm single-slot outputs, so concurrent parity could not be assessed.

## Higher-slot exceptions

- **Yue2, four slots:** three-slot runs passed; four-slot cold wave passed, but one warm request failed with Vulkan `ErrorOutOfDeviceMemory` for a 3.0 GB VAE decode buffer. This is an allocation failure, not a parity defect.
- **ZipVoice, three/four slots:** repeated automated runs were inconsistent. Three slots reset in three harness runs, although a separate instrumented three-request batch succeeded with identical outputs and all slots active. Four slots passed twice and reset twice across four automated attempts. Neither count is treated as a reliable pass.

## Recorded test outcomes

- Raw local runs are summarized in the tables above. The two-slot sweep covers 75 packages; higher counts cover 63 exact two-slot passes.
- The compatibility sweep used a local audit build with temporary Vulkan admission; production now admits only the 63 validated family/task pairs at their validated capacities.
- Higgs Vulkan support commit on PR #706: `50c43a0f386b3e9f1a08d2409ba8866d53a86657`.

## Production admission checks

After adding the separate Vulkan allowlist, the full CUDA and Vulkan Release
builds succeeded. The five focused tests for session pool/policy, scheduler,
single-slot compatibility, busy guard and server configuration passed on both
backends (10/10). Using the regular Vulkan server binary, BS-Roformer, Higgs TTS
and Kokoro each passed cold/warm three- and four-slot waves with exact reference
outputs, full observed overlap and no leaked leases (6/6 cases).

The regular Vulkan server also rejected ZipVoice at three slots (capacity two),
RVC at four (capacity three), and Kokoro at five (capacity four), with HTTP 500
capacity errors and no published model or active lease after the failed loads.

The audit used the same task fixtures and request options as the CUDA survey,
with backend `vulkan` and device 1. The base synthesis text was "This is a test
of two independent audio requests." Reference-conditioned tasks used
`assets/resources/a.wav` and its recorded transcript; audio tasks used the
resampled speech fixture. Task-specific options, fixed seeds and token limits
were reused between single-slot references and concurrent waves. These are
compatibility checks for selected requests, not exhaustive quality or load tests.
