# Module Optimizations

This page records framework-level module optimizations and the validation evidence used before enabling them broadly.

## ConvTranspose1d CUDA Col2im Fast Path

`ConvTranspose1dModule` automatically uses the CUDA col2im fast path when the graph, backend, and convolution shape are eligible. Callers do not need a model-specific flag. Ineligible shapes continue through the regular `ggml_conv_transpose_1d` path.

The fast path is intended for ConvTranspose1d decoder blocks that match the supported CUDA layout and stride/kernel conditions. It keeps the public module interface cleaner while still making the optimization conditional inside the framework.

Validation included:

- Debug build of `audiocpp_cli`.
- `conv_transpose_fast_path_test`, comparing CPU reference output against the automatic CUDA fast path.
- Path-test A/B comparison against the release baseline for models that exercise `ConvTranspose1dModule`.

### Path-Test Results

`before` is the release baseline. `after` is the build with automatic ConvTranspose1d fast-path selection.

| model/test | match | wav_cos | log_mel_cos | before ms | after ms | time change |
|---|---|---:|---:|---:|---:|---:|
| ace_step_base_cfg_text2music | no | 0.999451380 | 0.999974505 | 6987.509 | 6978.462 | -0.13% |
| ace_step_complete_source_audio | no | 0.999995143 | 0.999998052 | 4700.087 | 4637.143 | -1.34% |
| ace_step_cover_nofsq | no | 0.998803500 | 0.999783263 | 2890.072 | 2799.138 | -3.15% |
| ace_step_extract_vocals | no | 0.999913699 | 0.999972163 | 3658.467 | 3351.721 | -8.38% |
| ace_step_lego_guitar_from_piano | no | 0.999999391 | 0.999999537 | 5597.852 | 5577.165 | -0.37% |
| ace_step_repaint_middle | no | 0.999986492 | 0.999995485 | 3492.646 | 3397.707 | -2.72% |
| ace_step_turbo_text2music_long_chunk | no | 0.999970677 | 0.999988639 | 17722.696 | 18359.121 | +3.59% |
| chatterbox_de_multilingual | yes | exact | exact | 1907.115 | 1900.749 | -0.33% |
| chatterbox_en_long_cfg | yes | exact | exact | 4119.310 | 2736.709 | -33.56% |
| heartmula_music_generation | no | 0.999999949 | 0.999999962 | 25616.633 | 21253.878 | -17.03% |
| htdemucs_long_source_separation | no | 0.999430867 | 0.999852050 | 5812.936 | 5331.022 | -8.29% |
| miocodec_voice_conversion | no | 0.999998986 | 0.999999816 | 220.285 | 210.889 | -4.27% |
| omnivoice_clone_long | no | 0.999999619 | 0.999999829 | 416.612 | 377.363 | -9.42% |
| omnivoice_voice_design | no | 0.999999783 | 0.999999846 | 234.969 | 225.608 | -3.98% |
| pocket_tts_long_short_long | no | 0.999999403 | 0.999998684 | 467.139 | 302.158 | -35.32% |
| qwen3_tts_base_voice_clone_reuse | no | 0.999532211 | 0.999944397 | 20613.466 | 14305.990 | -30.60% |
| qwen3_tts_custom_voice | no | 0.999999933 | 0.999999917 | 1859.467 | 1734.749 | -6.71% |
| qwen3_tts_voice_design | no | 0.999739860 | 0.999910764 | 2725.197 | 2441.985 | -10.39% |
| stable_audio_medium_text | yes | exact | exact | 1015.399 | 1006.282 | -0.90% |
| stable_audio_music_init_audio | yes | exact | exact | 375.939 | 391.880 | +4.24% |
| stable_audio_music_inpaint_audio | yes | exact | exact | 368.180 | 383.208 | +4.08% |
| stable_audio_music_text | yes | exact | exact | 873.906 | 899.193 | +2.89% |
| stable_audio_sfx_text | yes | exact | exact | 353.219 | 334.724 | -5.24% |
| vevo2_s2s_routes | no | 0.999900936 | 0.999999530 | 1472.639 | 1502.221 | +2.01% |
| vevo2_svc_routes | no | 0.974615316 | 0.999747499 | 12496.319 | 12727.856 | +1.85% |
| vevo2_tts_routes | no | 0.997645912 | 0.999986043 | 8725.804 | 8595.433 | -1.49% |
| vevo2_vc_routes | no | 0.996030695 | 0.999985312 | 2411.683 | 2512.053 | +4.16% |
| vibevoice_two_speaker_long | no | 0.999740581 | 0.999969755 | 6905.524 | 1852.745 | -73.17% |
| voxcpm2_tts_voice_design | no | 0.999999734 | 0.999999800 | 1559.450 | 1329.636 | -14.74% |
| voxcpm2_voice_clone | no | 0.999999862 | 0.999999827 | 1837.453 | 1567.855 | -14.67% |

`match=yes` means the path-test artifacts matched exactly. `match=no` means at least one artifact hash changed, so the similarity columns should be used to judge numerical/audio closeness.

### Notes

- The fast path is selected by framework eligibility checks, not by per-model opt-in flags.
- The table reports one local debug-build validation snapshot. Timing can vary by GPU, driver, backend build options, and model/request shape.
- VibeVoice long-form generation is sampling-sensitive. Separate long-form checks should be used when validating narrative-level parity.
