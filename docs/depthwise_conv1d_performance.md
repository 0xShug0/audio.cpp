# Depthwise Conv1d Performance

Measured with `audiocpp_cli` path-test compare results from:

- Baseline: `build/logs/audiocpp_cli_path_tests/depthwise_global_baseline_cuda_20260709`
- Current: `build/logs/audiocpp_cli_path_tests/depthwise_global_current_cuda_20260709`

| Model | Case | Current | Baseline | Change |
| --- | --- | ---: | ---: | ---: |
| MioCodec | `miocodec_voice_conversion` | `125.813 ms` | `152.905 ms` | `-17.72%` |
| MioTTS | `miotts_long_text_voice_clone` | `2650.393 ms` | `2714.406 ms` | `-2.36%` |
| SeedVC | `seed_vc_v2_vc_long_bigvgan` | `2455.385 ms` | `2466.754 ms` | `-0.46%` |
| SeedVC | `seed_vc_v1_svc_long_rmvpe_bigvgan` | `24285.678 ms` | `24327.151 ms` | `-0.17%` |
| SeedVC | `seed_vc_v1_whisper_bigvgan_vc` | `1114.114 ms` | `1146.564 ms` | `-2.83%` |
| SeedVC | `seed_vc_v1_xlsr_hift_vc` | `617.485 ms` | `654.602 ms` | `-5.67%` |
| Vevo2 | `vevo2_s2s_routes` | `752.936 ms` | `770.326 ms` | `-2.26%` |
| Vevo2 | `vevo2_svc_routes` | `6945.930 ms` | `7268.863 ms` | `-4.44%` |
| Vevo2 | `vevo2_tts_routes` | `4291.218 ms` | `7642.226 ms` | `-43.85%` |
| Vevo2 | `vevo2_vc_routes` | `1253.280 ms` | `1273.592 ms` | `-1.59%` |
| VoxCPM2 | `voxcpm2_streaming_voice_design` | `1479.823 ms` | `1591.221 ms` | `-7.00%` |
| VoxCPM2 | `voxcpm2_tts_voice_design` | `1033.134 ms` | `1123.794 ms` | `-8.07%` |
| VoxCPM2 | `voxcpm2_voice_clone` | `1192.802 ms` | `1358.717 ms` | `-12.21%` |
| Citrinet ASR | `citrinet_asr_offline` | `84.098 ms` | `125.806 ms` | `-33.15%` |
| MarbleNet VAD | `marblenet_vad_offline` | `95.773 ms` | `108.697 ms` | `-11.89%` |
| VibeVoice TTS | `vibevoice_two_speaker_long` | `6699.695 ms` | `7770.973 ms` | `-13.79%` |
| Higgs Audio STT | `higgs_audio_stt_paths` | `1095.766 ms` | `1147.252 ms` | `-4.49%` |
| Nemotron ASR | `nemotron_asr_offline_paths` | `119.746 ms` | `138.110 ms` | `-13.30%` |
| Nemotron ASR | `nemotron_asr_streaming_paths` | `176.851 ms` | `186.050 ms` | `-4.94%` |
| VibeVoice ASR | `vibevoice_asr_paths` | `4171.806 ms` | `4779.468 ms` | `-12.71%` |
| VibeVoice ASR | `vibevoice_asr_structured_segments` | `2339.892 ms` | `2524.570 ms` | `-7.32%` |
| Sortformer Diar | `sortformer_diar_offline` | `48.584 ms` | `68.528 ms` | `-29.10%` |
| Hviske ASR | `hviske_asr_paths` | `756.612 ms` | `833.027 ms` | `-9.17%` |

CPU unit-test module timings:

| Shape source | Shape | Old lowering | New lowering | Change |
| --- | --- | ---: | ---: | ---: |
| MioCodec block | `B=1, C=384, T=2048, K=7` | `8.22533 ms` | `3.13917 ms` | `-61.8%` |
| MioCodec batched | `B=4, C=384, T=1024, K=7` | `15.1191 ms` | `5.89447 ms` | `-61.0%` |
| Nemotron encoder | `B=1, C=1024, T=1024, K=9` | `10.6603 ms` | `1.60797 ms` | `-84.9%` |
