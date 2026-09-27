# CUDA three/four-slot validation

All 75 family/task pairs that passed the two-slot survey were assessed at three and four slots on an RTX 3090 (24 GiB). Counts predicted not to fit were skipped; tests approaching physical capacity were stopped.

Highest consecutively validated counts: 58 families at four slots, 9 at three slots, 8 at two slots.

The generic fallback uses these per-family/task limits. A failed or untested lower count prevents promoting the capacity even if a higher count passes. Higgs retains its specialized shared-weight factory and existing ceiling; the fresh survey confirms three and four slots.

## Tested packages and limits

| Family | Package | Task | Validated maximum | 3 slots | 4 slots | Observed peak GPU MiB (3 / 4) |
|---|---|---|---:|---|---|---:|
| apollo | apollo_orig | s2s | 4 | Exact pass | Exact pass | 3090 / 4044 |
| audio8_asr | audio8_asr_0_1b_q8_0 | asr | 4 | Exact pass | Exact pass | 2174 / 2812 |
| audio8_tts | audio8_tts_preview_0_6b_q8_0 | tts | 4 | Exact pass | Exact pass | 6084 / 8050 |
| auk | auk_base_q8_0 | tts | 4 | Exact pass | Exact pass | 17578 / 23352 |
| breeze_tts | breeze_tts_2_q8_0 | vdes | 4 | Exact pass | Exact pass | 12958 / 17142 |
| bs_roformer | bs_roformer_q8_0 | sep | 4 | Exact pass | Exact pass | 4036 / 5296 |
| canary_asr | canary_180m_flash_q8_0 | asr | 4 | Exact pass | Exact pass | 1370 / 1746 |
| citrinet_asr | citrinet_asr_q8_0 | asr | 4 | Exact pass | Exact pass | 498 / 428 |
| cohere_asr | cohere_transcribe_q8_0 | asr | 4 | Exact pass | Exact pass | 7892 / 10438 |
| confucius4_r2t2 | confucius4_r2t2_q8_0 | asr | 4 | Exact pass | Exact pass | 8598 / 11376 |
| confucius4_tts | confucius4_tts_orig | clon | 3 | Exact pass | VRAM estimate: skipped | 19588 / — |
| controlfoley | controlfoley_large_44k_q8_0 | gen | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| cosyvoice3 | cosyvoice3_q8_0 | clon | 4 | Exact pass | Exact pass | 5842 / 7700 |
| dots_tts | dots_tts_soar_q8_0 | tts | 4 | Exact pass | Exact pass | 13134 / 17486 |
| echo_tts | echo_tts_q8_0 | clon | 3 | Exact pass | VRAM estimate: skipped | 18432 / — |
| firered_audio | firered_audio_q8_0 | clon | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| fireredtts3 | fireredtts3_instruct_q8_0 | clon | 3 | Exact pass | Output differs | 13668 / 18140 |
| fish_audio | fish_audio_s2_pro_q8_0 | tts | 3 | Exact pass | VRAM guard: stopped | 19000 / 24302 |
| fun_asr_nano | fun_asr_nano_2512_q8_0 | asr | 4 | Exact pass | Exact pass | 5316 / 7010 |
| glm_tts | glm_tts_q8_0 | tts | 4 | Exact pass | Exact pass | 15538 / 20630 |
| granite5asr | granite5asr_q8_0 | asr | 4 | Exact pass | Exact pass | 2172 / 2746 |
| heartmula | heartmula_q8_0 | gen | 2 | VRAM guard: stopped | VRAM estimate: skipped | 24172 / — |
| higgs_audio_stt | higgs_audio_stt_q8_0 | asr | 4 | Exact pass | Exact pass | 9916 / 13134 |
| higgs_audio_tts | higgs_audio_tts_4b_q8_0 | tts | 4 | Exact pass | Exact pass | 5722 / 5954 |
| htdemucs | htdemucs_q8_0 | sep | 4 | Exact pass | Exact pass | 2054 / 2652 |
| hviske_asr | hviske_asr_q8_0 | asr | 4 | Exact pass | Exact pass | 8342 / 12558 |
| index_tts2 | index_tts2_q8_0 | tts | 4 | Exact pass | Exact pass | 17428 / 23104 |
| inflect_v2 | inflect_micro_v2_orig | tts | 4 | Exact pass | Exact pass | 720 / 884 |
| irodori_tts | irodori_tts_v4_small_q8_0 | tts | 4 | Exact pass | Exact pass | 6786 / 8964 |
| kitten_tts | kitten_tts_mini_0_8_orig | tts | 4 | Exact pass | Exact pass | 4644 / 6122 |
| kokoro_tts | kokoro_82m_q8_0 | tts | 4 | Exact pass | Exact pass | 3194 / 4172 |
| kroko_asr | kroko_asr_community_q8_0 | asr | 4 | Exact pass | Exact pass | 1070 / 1342 |
| magpie_tts | magpie_tts_q8_0 | tts | 4 | Exact pass | Exact pass | 3542 / 4648 |
| meanvc2 | meanvc2_120ms_40ms_f32 | vc | 4 | Exact pass | Exact pass | 12782 / 16956 |
| mel_band_roformer | mel_band_roformer_q8_0 | sep | 4 | Exact pass | Exact pass | 3658 / 4792 |
| midashenglm_gen | midashenglm_gen_q8_0 | gen | 4 | Exact pass | Exact pass | 11100 / 14670 |
| minimax_music3 | minimax_music3_q8_0 | gen | 2 | VRAM guard: stopped | VRAM guard: stopped | 24272 / 24208 |
| mira_tts | mira_tts_local_bf16 | tts | 4 | Exact pass | Exact pass | 7358 / 9998 |
| moonshine_asr | moonshine_streaming_tiny_q8_0 | asr | 4 | Exact pass | Exact pass | 490 / 618 |
| moss_transcribe_diarize | moss_transcribe_diarize_q8_0 | asr | 4 | Exact pass | Exact pass | 4428 / 5820 |
| moss_tts_nano | moss_tts_nano_100m_q8_0 | tts | 4 | Exact pass | Exact pass | 1916 / 2468 |
| moss_tts_v15 | moss_tts_v15_q8_0_codec_f16 | tts | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| moss_voicegen | moss_voicegen_bf16_codec_f16_decode | vdes | 3 | Exact pass | VRAM estimate: skipped | 22854 / — |
| muscriptor | muscriptor_small_f32 | midi | 4 | Exact pass | Exact pass | 1670 / 2142 |
| nemotron_3_diar | nemotron_3_diar_q8_0 | diar | 4 | Exact pass | Exact pass | 648 / 780 |
| nemotron_asr | nemotron_asr_q8_0 | asr | 4 | Exact pass | Exact pass | 3286 / 4372 |
| omnivoice | omnivoice_q8_0 | tts | 4 | Exact pass | Exact pass | 5444 / 7174 |
| outetts | outetts_1_0_1b_q8_0 | tts | 3 | Exact pass | VRAM estimate: skipped | 20396 / — |
| parakeet_tdt | parakeet_tdt_q8_0 | asr | 4 | Exact pass | Exact pass | 5608 / 7244 |
| personaplex | personaplex_7b_v1_q8_0 | s2s | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| piper_tts | piper_lessac_medium_orig | tts | 4 | Exact pass | Exact pass | 806 / 850 |
| pocket_tts | pocket_tts_english_q8_0 | tts | 4 | Exact pass | Exact pass | 1366 / 1734 |
| pulsevad | pulsevad_2_1k_f32 | vad | 4 | Exact pass | Exact pass | 228 / 228 |
| qwen3_asr | qwen3_asr_1_7b_q8_0 | asr | 4 | Exact pass | Exact pass | 8928 / 11816 |
| qwen3_forced_aligner | qwen3_forced_aligner_0_6b_q8_0 | align | 4 | Exact pass | Exact pass | 4106 / 5404 |
| qwen3_tts | qwen3_tts_1_7b_base_q8_0 | tts | 3 | Exact pass | Output differs | 11436 / 15158 |
| rvc | rvc_f16 | vc | 3 | Exact pass | Request failure | 6142 / 5410 |
| sanotts | sanotts_heart_nano_orig | tts | 4 | Exact pass | Exact pass | 262 / 406 |
| sense_asr | sensevoice_small_q8 | asr | 4 | Exact pass | Exact pass | 1184 / 1592 |
| sheetsage2 | sheetsage2_orig | midi | 3 | Exact pass | VRAM guard: stopped | 20650 / 24304 |
| soprano_tts | soprano_1_1_80m_q8_0 | tts | 4 | Exact pass | Exact pass | 1656 / 2178 |
| sopro_tts | sopro_v2_turbo_f16 | tts | 4 | Exact pass | Exact pass | 2518 / 3272 |
| sortformer_diar | sortformer_diar_4spk_v1_q8_0 | diar | 4 | Exact pass | Exact pass | 6952 / 9236 |
| sortformer_diar_v2 | sortformer_diar_v2_1_f32_gguf_local | diar | 4 | Exact pass | Exact pass | 3226 / 4268 |
| stable_audio | stable_audio_3_medium_q8_0 | gen | 4 | Exact pass | Exact pass | 10400 / 13778 |
| supertonic | supertonic_3_q8_0 | tts | 4 | Exact pass | Exact pass | 1634 / 2114 |
| universr | universr_audio_orig | s2s | 4 | Exact pass | Exact pass | 7174 / 9490 |
| vibevoice | vibevoice_1_5b_q8_0 | tts | 4 | Exact pass | Exact pass | 12016 / 17256 |
| vibevoice_asr | vibevoice_asr_q8_0 | asr | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| vibevoice_asr_streaming | vibevoice_asr_streaming_7b_q8_0 | asr | 2 | VRAM estimate: skipped | VRAM estimate: skipped | — / — |
| voxcpm1 | voxcpm1_0_5b_q8_0 | tts | 4 | Exact pass | Exact pass | 5930 / 7770 |
| voxcpm2 | voxcpm2_q8_0 | tts | 4 | Exact pass | Exact pass | 14022 / 18516 |
| voxtral_realtime | voxtral_realtime_q8_0 | asr | 4 | Exact pass | Exact pass | 15986 / 21232 |
| yue2 | yue2_main_q8_0 | gen | 4 | Exact pass | Exact pass | 17680 / 23462 |
| zipvoice | zipvoice_distill_gguf | clon | 4 | Exact pass | Exact pass | 1142 / 1444 |

## Final build and policy checks

- Full CUDA CLI/server build passed. Five focused CTests passed: session_pool_test, server_model_slots_test, server_busy_guard_test, server_config_test and server_single_slot_compat_test.
- Final compiled server accepted four concurrent BS-RoFormer requests with exact reference outputs. HTTP checks rejected BS-RoFormer at five slots, FireRedTTS3 and Qwen3 TTS at four, F5-TTS at two and ControlFoley at three, reporting the expected capacities and draining every lease.

## Higher-count investigations

- FireRedTTS3 at four slots: one cold waveform differed (same sample count, maximum PCM difference 352, signal-to-error ratio 34.32 dB). Three slots passed exactly.
- Qwen3 TTS at four slots: one warm waveform differed (same sample count, maximum PCM difference 449, signal-to-error ratio 40.69 dB). Three slots passed exactly.
- RVC at four slots: request connections reset. Three slots passed exactly.
- These differences require investigation; signal-to-error ratios measure numerical differences, not perceptual quality. These families remain capped at three slots.

## Method

- Same selected packages and saved requests as the two-slot survey, with fresh three/four-slot server processes.
- Identical concurrent requests, cold and warm waves, compared with the saved single-slot reference outputs. Exact pass requires all successful responses, exact result equality, the requested active slot count observed after loading, and no remaining leases.
- Preflight estimate: two-slot retained GPU MiB plus (slots − 2) times the measured one-to-two-slot increment. Skip if this exceeds available VRAM minus 1 GiB. This estimate is not an exact peak measurement.
- GPU memory sampled during loading/inference. Stop only the test process when total use reaches device capacity minus 512 MiB. A guarded stop is a resource-budget result, not proof of an inference bug.
- Generic sessions share the CPU checkpoint assets but can replicate GPU weights. More slots do not guarantee higher throughput.
- Coverage is one selected package/task per family, these request settings and CUDA offline mode. Higher counts, other variants/tasks, CPU/Vulkan and streaming are not certified by this survey.
- Some TTS requests use explicit token limits, seeds and reference transcripts. AuK uses text-only TTS because this build lacks Soxr for reference-audio conditioning.
- No CUDA kernels or model inference implementations were changed.
