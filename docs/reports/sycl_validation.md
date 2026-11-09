# SYCL validation

The models checked on the SYCL backend so far, and what happened. A model is listed as validated only when its output was checked, not because it ran to completion.

## Test configuration

Measured on an Intel Arc Pro B70 with 30.3 GiB, Linux 7.0, Level Zero, the oneAPI 2026.1 DPC++ compiler, Release builds with `-DENGINE_ENABLE_SYCL=ON`, `--backend sycl`, eight threads and logging enabled, on an otherwise idle machine. The tree is `main` at `7d20b414` with #1243 and the CPY and `CONV_2D_DW` additions applied.

Rows report a single first request in a new process, unless noted. Timing uses canonical `session.wall_ms`, which excludes model loading and includes first-use SYCL kernel compilation.

RTF is request time divided by generated audio duration, or input duration for ASR. **RTF below 1 means faster than real time.**

Peak VRAM is the largest per-process VRAM in the DRM fdinfo, sampled every 20 ms across loading and inference. It may miss short peaks.

## Validated

| Model | Task | GGUF format | Input | Audio seconds | Request seconds | RTF | Peak VRAM (MiB) |
|---|---|---|---|---:|---:|---:|---:|
| Kokoro-82M | TTS | Q8_0 | 237 chars | 14.600 | 1.835 | 0.1257 | 2820.1 |
| Supertonic 3 | TTS | Q8_0 | 237 chars | 14.558 | 0.981 | 0.0674 | 550.6 |
| OmniVoice | TTS | Q8_0 | 237 chars | 14.340 | 3.184 | 0.2220 | 1837.2 |
| Magpie Multilingual | TTS | Q8_0 | 237 chars | 12.539 | 20.383 | 1.6256 | 2646.2 |
| VoxCPM-0.5B | TTS | Q8_0 | 237 chars | 13.040 | 10.799 | 0.8281 | 2449.9 |
| Moonshine Streaming Medium | ASR | Q8_0 | 40 s audio | 40.000 | 1.663 | 0.0416 | 887.9 |
| GigaAM Multilingual Large | ASR | F16 | 40 s audio | 40.000 | 1.224 | 0.0306 | 1319.6 |
| Cohere Transcribe | ASR | Q4_0 | 40 s audio | 40.000 | 2.121 | 0.0530 | 2163.8 |
| YuE2 | GEN | Q8_0 + F16 VAE | Full song; median of 3 after warmup | 68.120 | 22.830 | 0.3351 | 8816.6 |

## Not working yet

These were tested with the same inputs. The first seven complete without an error and produce wrong output, so a successful exit code is not enough to treat a SYCL run as correct.

| Model | Task | GGUF format | Result on SYCL | CPU backend |
|---|---|---|---|---|
| PocketTTS | TTS | Q8_0 | Constant full-scale output | Nonconstant output |
| MOSS-TTS Nano | TTS | Q8_0 | Constant full-scale output. Also on `main` at `28aaff1c`, before the ggml-sycl kernels in #1242, with `GGML_SYCL_F16` on or off | Nonconstant output |
| HTDemucs | SEP | Q8_0 | Constant full-scale stems. `SOFT_MAX` on a `[2688,2688,8,1]` tensor produces non-finite values | Nonconstant output |
| Parakeet-TDT-0.6B-v3 | ASR | Q8_0 | Empty transcript | 1% word error rate |
| Qwen3-ASR-0.6B | ASR | Q8_0 | Unrelated text | 0% word error rate |
| SenseVoice-Small | ASR | Q8_0 | First part of the speech is missing; 75% word error rate against CPU | 2% word error rate |
| Fun-ASR Nano | ASR | Q8_0 | Repeated tokens over the first part of the speech; 70% word error rate against CPU | 2% word error rate |
| Qwen3 forced aligner | ALIGN | Q8_0 | Exits with `produced negative timestamp id` | Not run |
| KittenTTS 2 | TTS | Q8_0 | Aborts in `ggml_sycl_flash_attn_ext`: no kernel for its attention shape | Not run |

Disabling graph replay, oneDNN attention, oneDNN entirely or the reorder optimization (`GGML_SYCL_DISABLE_GRAPH=1`, `GGML_SYCL_FA_ONEDNN=0`, `GGML_SYCL_DISABLE_DNN=1`, `GGML_SYCL_DISABLE_OPT=1`) did not correct any of the first seven models.

## Workload and configuration

- TTS uses one 237-character English paragraph. Kokoro uses voice `af_heart`, Supertonic `F1`, and the others their defaults. OmniVoice and Magpie use `--language en`.
- ASR uses the first 40 seconds of `assets/resources/qwen3_tts_longform_asr_input.wav`, resampled to 16 kHz. Cohere Transcribe uses `language=en`.
- YuE2 uses city_lights, seed 831001, `cot=full`, eight NAR steps, guidance 1.0 and the default `sycl_perf` preset. With `GGML_SYCL_WEIGHT_F16_CACHE=1` it measures 3.02x realtime at 16.5 GiB peak VRAM.
- Packages: `kokoro_82m_q8_0`, `supertonic_3_q8_0`, `omnivoice_q8_0`, `magpie_tts_q8_0`, `voxcpm1_0_5b_q8_0`, `moonshine_streaming_medium_q8_0`, `gigaam_multilingual_large_ctc_f16`, `cohere_transcribe_q4_0`, `pocket_tts_english_q8_0`, `moss_tts_nano_100m_q8_0`, `htdemucs_q8_0`, `parakeet_tdt_q8_0`, `qwen3_asr_0_6b_q8_0`, `sensevoice_small_q8`, `fun_asr_nano_2512_q8_0`, `qwen3_forced_aligner_0_6b_q8_0` and `kitten_tts2_20261021_q8_0`.
- Word error rates are against a reference transcript of the same 40 seconds from Qwen3-ASR 1.7B on the CPU backend, unless stated otherwise.

## Output checks

The five validated TTS outputs are finite and nonconstant. Cohere Transcribe on SYCL returned the input text for all five with no word errors after normalizing punctuation, case and "harbour".

Moonshine, GigaAM and Cohere produced transcripts identical to the CPU backend for the same model and input, with 0%, 5.9% and 2.0% word error rate against the reference transcript.

The three YuE2 takes completed with no clipped samples and peaks from 0.65 to 0.83.

## Limits

Only the Arc Pro B70 on Linux was tested. Timings are single runs of a first request, except YuE2. Streaming, voice cloning and the other tasks of these families were not tested. Models not listed here have not been checked.

See the [SYCL build guide](../build/sycl.md) to build and run the backend.
