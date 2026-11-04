# Browser WebGPU validation

## Test configuration

Measured in Chrome 150 on an NVIDIA RTX 5090 with driver 595.84, using Emscripten 6.0.13, Debug builds with `-O3 -g`, WASM SIMD, `--backend webgpu`, one model thread and logging enabled. Requests run sequentially.

**Warmed** rows report the mean of three requests in the same model session after one excluded warmup request. Other rows report a single first request. Timing excludes model loading and WASM initialization. First-request timings include graph/shader setup performed during inference.

RTF is request time divided by generated audio duration, or input duration for ASR, alignment and separation, using canonical `session.wall_ms`. **RTF below 1 means faster than real time.** Audio seconds and request seconds are per request, including for warmed rows.

Peak VRAM is the observed NVML graphics-memory peak of a dedicated Chrome GPU process, sampled every 20 ms across loading, warmup and inference. It includes the browser baseline and may miss short peaks.

## Results

| Model | Task | GGUF format | Input | Audio seconds | Request seconds | RTF | Peak VRAM (MiB) | WASM |
|---|---|---|---|---:|---:|---:|---:|---|
| Irodori-TTS Small MF | TTS | Q8_0 | 205 chars | 41.800 | 4.766 | 0.1140 | 3581.3 | 32 |
| Irodori-TTS Small RF | TTS | Q8_0 | 205 chars | 41.800 | 13.736 | 0.3286 | 3708.9 | 32 |
| Kokoro-82M | TTS | Q8_0 | 245 chars | 15.075 | 1.086 | 0.0720 | 1811.0 | 32 |
| LFM3-Audio-2B | TTS | Q8_0 + F16 codec | 245 chars; warmed | 15.360 | 5.150 | 0.3353 | 2146.4 | 64 |
| Magpie Multilingual | TTS | Q8_0 | 245 chars | 13.375 | 50.691 | 3.7901 | 1457.2 | 32 |
| Ming3-TTS-0.5B | TTS | Q8_0 | 245 chars | 12.500 | 20.262 | 1.6210 | 1385.1 | 32 |
| MOSS-TTS Nano | TTS | Q8_0 | 245 chars | 14.000 | 15.075 | 1.0768 | 561.6 | 32 |
| OmniVoice | TTS | Q8_0 | 245 chars | 15.000 | 5.027 | 0.3351 | 1776.1 | 32 |
| PocketTTS | TTS | Q8_0 | 245 chars | 11.600 | 2.062 | 0.1778 | 735.4 | 32 |
| PocketTTS Small 18M | TTS | F32 | 245 chars; warmed | 11.040 | 3.360 | 0.3043 | 260.9 | 32 |
| Qwen3-TTS 1.7B CustomVoice | TTS | Q8_0 | 245 chars | 18.080 | 21.590 | 1.1941 | 3827.7 | 64 |
| Supertonic 3 | TTS | F32/I64 (Q8-labelled) | 245 chars | 16.023 | 0.687 | 0.0429 | 502.7 | 32 |
| VibeVoice 1.5B | TTS | Q8_0 | 256 chars | 13.200 | 21.878 | 1.6574 | 4501.0 | 64 |
| VoxCPM-0.5B | TTS | Q8_0 | 245 chars | 15.360 | 15.783 | 1.0276 | 2432.9 | 32 |
| Cohere Transcribe | ASR | Q4_0 | 40 s audio | 40.000 | 1.548 | 0.0387 | 2138.8 | 32 |
| Fun-ASR Nano | ASR | Q8_0 | 40 s audio | 40.000 | 2.609 | 0.0652 | 1795.4 | 32 |
| GigaAM Multilingual Large | ASR | F16 | 40 s audio | 40.000 | 0.329 | 0.0082 | 1216.2 | 32 |
| Moonshine Streaming Medium | ASR | Q8_0 | 40 s audio | 40.000 | 1.628 | 0.0407 | 730.2 | 32 |
| LFM3-Audio-2B | ASR | Q8_0 + F16 codec | 40 s audio; warmed | 40.000 | 1.601 | 0.0400 | 2358.9 | 64 |
| MOSS Transcribe Diarize | ASR | Q4_K | 40 s audio | 40.000 | 4.748 | 0.1187 | 1642.2 | 32 |
| Nemotron-3.5-ASR-Streaming-0.6B | ASR | Q8_0 | 40 s audio | 40.000 | 32.524 | 0.8131 | 1111.0 | 32 |
| OWSM-CTC-v4-1B | ASR | Q8_0 | 40 s audio | 40.000 | 1.115 | 0.0279 | 2891.5 | 32 |
| Parakeet-TDT-0.6B-v3 | ASR | Q8_0 | 40 s audio | 40.000 | 1.742 | 0.0436 | 2079.5 | 32 |
| Qwen3-ASR-0.6B | ASR | Q8_0 | 40 s audio | 40.000 | 2.334 | 0.0584 | 2359.9 | 32 |
| SenseVoice-Small | ASR | Q8_0 | 40 s audio | 40.000 | 0.425 | 0.0106 | 369.3 | 32 |
| Qwen3 forced aligner | ALIGN | Q8_0 | 40 s audio | 40.000 | 0.365 | 0.0091 | 1978.7 | 32 |
| BS-RoFormer | SEP | Q8_0 | 40 s audio | 40.000 | 37.927 | 0.9482 | 2133.7 | 32 |
| HTDemucs | SEP | Q8_0 | 40 s audio | 40.000 | 1.689 | 0.0422 | 502.0 | 32 |
| YuE2 | GEN | Q4_0 + F16 VAE | Instrumental; 1,500-token cap | 59.999 | 37.021 | 0.6170 | 3014.4 | 64 |
| NVIDIA NemotronLabs RE-USE | S2S enhancement | F32 | 14.07 s audio; offline | 14.072 | 4.335 | 0.3080 | 3090.0 | 32 |
| Realtime RE-USE | S2S enhancement | F32 | 14.07 s audio; streaming | 14.072 | 1.569 | 0.1115 | 522.5 | 32 |

## Workload and configuration

- TTS uses 245 English characters; Irodori uses 205 Japanese characters. VibeVoice's 256-character script includes its `Speaker 0:` label. ASR/alignment uses 40 seconds of speech; separation uses 40 seconds of stereo music.
- Irodori RF/MF uses `codec_decode_chunk_steps=256`. WASM32 model metadata arenas fit within its 4 GiB address space; browser GPU buffer limits are separate.
- LFM3-Audio uses the published Q8_0 backbone and Q8_0 mmproj with its F16 codec, WASM64, default model controls and the WASM sinc input resampler. TTS uses seed 1234 and the American English female voice.
- PocketTTS Small 18M uses F32 weights, the matching Alba voice state, seed 1234, automatic attention (resolved flash), F32 KV and bounded four-frame offline decoding.
- VibeVoice uses WASM64 with a 16 GiB host-memory ceiling, separate from GPU memory.
- YuE2 uses Q4 main weights, an F16 VAE, `cot=off`, eight NAR steps, `semantic_max_tokens=1500` and `vae_tile_frames=256`. Generation stops at the token cap.
- RE-USE uses 10-second offline chunks with one-second overlap. Realtime RE-USE uses streaming mode, exit layer 8 and zero lookahead.
- Supertonic's Q8-labelled package contains F32/I64 tensors; the table shows the actual types.

## Output checks

PocketTTS Small 18M and LFM3-Audio TTS produced finite, nonconstant 24 kHz audio. Their warmed outputs were identical across repeats and matched the independently checked first-request outputs: Whisper Small transcribed all 43 input words correctly after English normalization (0% WER). LFM3-Audio ASR returned the same English transcript across all four requests.

RE-USE and Realtime RE-USE produced finite, nonconstant 24 kHz audio with the input sample count; waveform cosine similarity against matched native CUDA runs exceeded 0.9999997. Realtime RE-USE's streaming result validates incremental inference and final audio collection, not progressive browser playback or TTFA.

All generated TTS/music WAVs were finite and nonconstant. Independent SenseVoice transcription matched seven of twelve other TTS clips after normalization: Kokoro, VoxCPM, OmniVoice, Ming3, Qwen3-TTS, Irodori RF and VibeVoice. PocketTTS, Magpie and Irodori MF had small transcript differences; Supertonic and MOSS Nano had larger differences. These checks are not a model-quality benchmark.

Alignment produced 101 ordered words within the 40-second input. Separation produced both BS-RoFormer stems and all four HTDemucs stems.

See the [WebGPU build guide](../build/webgpu.md) to run models with your own weights and inputs.
