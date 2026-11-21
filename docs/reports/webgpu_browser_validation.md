# Browser WebGPU validation

## Test configuration

Measured in Chrome 150 on an NVIDIA RTX 5090 with driver 595.84, using Emscripten 6.0.13, Debug builds with `-O3 -g`, WASM SIMD, `--backend webgpu`, one model thread and logging enabled. Requests run sequentially.

**Warmed** rows report the mean of three requests in the same model session after one excluded warmup request. Other rows report a single first request. Timing excludes model loading and WASM initialization. First-request timings include graph/shader setup performed during inference.

RTF is request time divided by generated audio duration for synthesis, or input duration for ASR, alignment, separation, enhancement, restoration and note transcription, using canonical `session.wall_ms`. **RTF below 1 means faster than real time.** Audio seconds and request seconds are per request, including for warmed rows.

Peak VRAM is the observed NVML graphics-memory peak of a dedicated Chrome GPU process, sampled every 20 ms across loading, warmup and inference. It includes the browser baseline and may miss short peaks.

## Results

| Model | Task | GGUF format | Input | Audio seconds | Request seconds | RTF | Peak VRAM (MiB) | WASM |
|---|---|---|---|---:|---:|---:|---:|---|
| Irodori-TTS Small MF | TTS | Q8_0 | 205 chars | 41.800 | 4.766 | 0.1140 | 3581.3 | 32 |
| Irodori-TTS Small RF | TTS | Q8_0 | 205 chars | 41.800 | 13.736 | 0.3286 | 3708.9 | 32 |
| Kokoro-82M | TTS | Q8_0 | 245 chars | 15.075 | 1.086 | 0.0720 | 1811.0 | 32 |
| LFM3-Audio-2B | TTS | Q8_0 + F16 codec | 245 chars; warmed | 15.360 | 5.150 | 0.3353 | 2146.4 | 64 |
| M5-TTS CustomVoice | TTS | Q8_0 + F32 codec | 245 chars; warmed | 15.080 | 42.384 | 2.8106 | 2342.5 | 64 |
| M5-TTS VoiceClone | Clone | Q8_0 + F32 codec | 245 chars + reference; warmed | 25.480 | 73.328 | 2.8779 | 4611.9 | 64 |
| Magpie Multilingual | TTS | Q8_0 | 245 chars | 13.375 | 50.691 | 3.7901 | 1457.2 | 32 |
| Ming3-TTS-0.5B | TTS | Q8_0 | 245 chars | 12.500 | 20.262 | 1.6210 | 1385.1 | 32 |
| MOSS-TTS Nano | TTS | Q8_0 | 245 chars | 14.000 | 15.075 | 1.0768 | 561.6 | 32 |
| OmniVoice | TTS | Q8_0 | 245 chars | 15.000 | 5.027 | 0.3351 | 1776.1 | 32 |
| PocketTTS | TTS | Q8_0 | 245 chars | 11.600 | 2.062 | 0.1778 | 735.4 | 32 |
| PocketTTS Small 18M | TTS | F32 | 245 chars; warmed | 11.040 | 3.360 | 0.3043 | 260.9 | 32 |
| Qwen3-TTS 1.7B CustomVoice | TTS | Q8_0 | 245 chars | 18.080 | 21.590 | 1.1941 | 3827.7 | 64 |
| Supertonic 3 | TTS | F32/I64 (Q8-labelled) | 245 chars | 16.023 | 0.687 | 0.0429 | 502.7 | 32 |
| Tiny Voca | TTS | F32 | 245 chars; warmed | 12.299 | 0.219 | 0.0178 | 138.1 | 64 |
| VibeVoice 1.5B | TTS | Q8_0 | 256 chars | 13.200 | 21.878 | 1.6574 | 4501.0 | 64 |
| VoxCPM-0.5B | TTS | Q8_0 | 245 chars | 15.360 | 15.783 | 1.0276 | 2432.9 | 32 |
| VoxMel-TTS 0.6B | Clone | Q8_0 + F32 vocoder | 245 chars + reference; warmed | 15.179 | 25.317 | 1.6679 | 1592.1 | 64 |
| Cohere Transcribe | ASR | Q4_0 | 40 s audio | 40.000 | 1.548 | 0.0387 | 2138.8 | 32 |
| Fun-ASR Nano | ASR | Q8_0 | 40 s audio | 40.000 | 2.609 | 0.0652 | 1795.4 | 32 |
| GigaAM Multilingual Large | ASR | F16 | 40 s audio | 40.000 | 0.329 | 0.0082 | 1216.2 | 32 |
| Kroko XL English Offline | ASR | Q8_0 | 40 s audio; warmed | 40.000 | 0.131 | 0.0033 | 682.5 | 64 |
| LFM3-Audio-2B | ASR | Q8_0 + F16 codec | 40 s audio; warmed | 40.000 | 1.601 | 0.0400 | 2358.9 | 64 |
| Moonshine Streaming Medium | ASR | Q8_0 | 40 s audio | 40.000 | 1.628 | 0.0407 | 730.2 | 32 |
| MOSS Transcribe Diarize | ASR | Q4_K | 40 s audio | 40.000 | 4.748 | 0.1187 | 1642.2 | 32 |
| Nemotron-3.5-ASR-Streaming-0.6B | ASR | Q8_0 | 40 s audio | 40.000 | 32.524 | 0.8131 | 1111.0 | 32 |
| OWSM-CTC-v4-1B | ASR | Q8_0 | 40 s audio | 40.000 | 1.115 | 0.0279 | 2891.5 | 32 |
| Parakeet-TDT-0.6B-v3 | ASR | Q8_0 | 40 s audio | 40.000 | 1.742 | 0.0436 | 2079.5 | 32 |
| Qwen3-ASR-0.6B | ASR | Q8_0 | 40 s audio | 40.000 | 2.334 | 0.0584 | 2359.9 | 32 |
| SenseVoice-Small | ASR | Q8_0 | 40 s audio | 40.000 | 0.425 | 0.0106 | 369.3 | 32 |
| Qwen3 forced aligner | ALIGN | Q8_0 | 40 s audio | 40.000 | 0.365 | 0.0091 | 1978.7 | 32 |
| BS-RoFormer | SEP | Q8_0 | 40 s audio | 40.000 | 37.927 | 0.9482 | 2133.7 | 32 |
| HTDemucs | SEP | Q8_0 | 40 s audio | 40.000 | 1.689 | 0.0422 | 502.0 | 32 |
| MossFormer2 SE 48K | S2S enhancement | F32 | 14.07 s audio; warmed | 14.072 | 0.188 | 0.0134 | 424.0 | 64 |
| MossFormer2 SR 48K | S2S restoration | F16 | 14.07 s audio; warmed | 14.072 | 1.120 | 0.0796 | 2081.0 | 64 |
| NVIDIA NemotronLabs RE-USE | S2S enhancement | F32 | 14.07 s audio; offline | 14.072 | 4.335 | 0.3080 | 3090.0 | 32 |
| Realtime RE-USE | S2S enhancement | F32 | 14.07 s audio; streaming | 14.072 | 1.569 | 0.1115 | 522.5 | 32 |
| Inworld Voice Isolator | S2S enhancement | F32 | 10.77 s speech; offline, warmed | 10.770 | 1.738 | 0.1614 | 905.4 | 64 |
| Inworld Voice Isolator Realtime | S2S enhancement | F32 | 10.77 s speech; streaming | 10.770 | 4.820 | 0.4475 | 835.5 | 64 |
| MOSS-SoundEffect v2 | Sound effects | Q8_0 | Dog barks, 10 steps; warmed | 10.000 | 16.289 | 1.6289 | 5160.5 | 64 |
| YuE2 | GEN | Q4_0 + F16 VAE | Instrumental; 1,500-token cap | 59.999 | 37.021 | 0.6170 | 3014.4 | 64 |
| SoulX-Singer SVC | Singing voice conversion | Q8_0 | 8.29 s vocal + reference; warmed | 8.294 | 6.037 | 0.7279 | 2086.7 | 64 |
| SoulX-Singer SVS | Singing synthesis | Q8_0 | English score + reference; warmed | 6.900 | 5.231 | 0.7581 | 1396.9 | 64 |
| ROSVOT | Singing-note transcription | F32 | 8.29 s vocal; warmed | 8.294 | 0.392 | 0.0472 | 800.5 | 64 |

## Workload and configuration

- TTS uses 245 English characters; Irodori uses 205 Japanese characters. VibeVoice's 256-character script includes its `Speaker 0:` label. ASR/alignment uses 40 seconds of speech; separation uses 40 seconds of stereo music.
- Irodori RF/MF uses `codec_decode_chunk_steps=256`. WASM32 model metadata arenas fit within its 4 GiB address space; browser GPU buffer limits are separate.
- Kroko XL uses the 40-second speech input.
- LFM3-Audio uses the published Q8_0 backbone and Q8_0 mmproj with its F16 codec, WASM64, default model controls and the WASM sinc input resampler. TTS uses seed 1234 and the American English female voice.
- M5-TTS uses Q8 LM weights with the packaged F32 codec and seed 1234. CustomVoice selects speaker 1; VoiceClone uses a five-second Mandarin reference with its matching transcript.
- MOSS-SoundEffect v2 uses Q8, a dog-bark prompt, ten-second output, the full trained 30-second canvas, 10 Euler steps, guidance 4 and sigma shift 5. BF16 operands retain their values and accumulate in F32; this path does not substitute F16 arithmetic. WebGPU currently lacks the CUDA/Vulkan denoiser epilogue fusions.
- MossFormer2 SE/SR use the 14.072-second, 16 kHz speech input, exercising SR bandwidth restoration. Their RTF uses input duration; SR output is shortened by less than one vocoder hop.
- PocketTTS Small 18M uses F32 weights, the matching Alba voice state, seed 1234, automatic attention (resolved flash), F32 KV and bounded four-frame offline decoding.
- RE-USE uses 10-second offline chunks with one-second overlap. Realtime RE-USE uses streaming mode, exit layer 8 and zero lookahead.
- Voice Isolator Realtime uses the default F16 attention cache, two lookahead frames and four-frame blocks.
- ROSVOT uses the official 8.294-second SoulX English target vocal. SoulX SVS uses the official English reference/target score, producing 6.9 seconds with the default 32 steps and guidance 3. SVC uses that target vocal and the official SVC reference, also with default sampling. Both source and reference are supplied at 16 kHz for SVC: other rates require libsoxr, which this browser build cannot load dynamically. Offline conversion to 16 kHz is excluded from its timing; model inference and its remaining preprocessing are included.
- Supertonic's Q8-labelled package contains F32/I64 tensors; the table shows the actual types.
- Tiny Voca uses its F32 package, voice M2 and seed 1234.
- VibeVoice uses WASM64 with a 16 GiB host-memory ceiling, separate from GPU memory.
- VoxMel uses Q8 weights with its F32 vocoder, seed 1234 and a five-second Mandarin reference with its matching transcript.
- YuE2 uses Q4 main weights, an F16 VAE, `cot=off`, eight NAR steps, `semantic_max_tokens=1500` and `vae_tile_frames=256`. Generation stops at the token cap.

## Output checks

All generated audio is finite and nonconstant. Speech-content checks use independent ASR after English normalization. They are sanity checks, not a model-quality benchmark or a waveform-parity claim.

Whisper Small checks had 0% WER for LFM3-Audio TTS, both M5-TTS modes, MossFormer2 SE/SR, PocketTTS Small, both SoulX-Singer modes, Tiny Voca and VoxMel, using target text for TTS and reference ASR transcripts for enhancement and singing.

SenseVoice transcription matched the target text for Kokoro, VoxCPM, OmniVoice, Ming3, Qwen3-TTS, Irodori RF and VibeVoice. PocketTTS, Magpie and Irodori MF had small transcript differences; Supertonic and MOSS Nano had larger differences.

Repeated LFM3-Audio TTS, M5-TTS, MossFormer2 SR, MOSS-SoundEffect, PocketTTS Small, SoulX-Singer, Tiny Voca and VoxMel outputs are byte-identical within each tested session. MossFormer2 SE repeats differ numerically because of frontend dithering. LFM3-Audio ASR repeats the same transcript. Kroko repeats the same transcript and 101 bounded word timestamps across four requests. ROSVOT returns 24 ordered notes within the input duration, with identical JSON on repeats.

RE-USE and Realtime RE-USE produce 24 kHz audio with the input sample count; waveform cosine similarity against matched native CUDA runs exceeds 0.9999997. Realtime RE-USE validates streaming inference and final audio collection, not progressive browser playback or TTFA.

Voice Isolator produced finite, nonconstant 48 kHz audio with the input sample count. Speech waveform cosine against native CUDA was 0.99999999 offline and 0.99924 streaming; offline repeats were byte-identical.

Alignment produces 101 ordered words within the 40-second input. Separation produces both BS-RoFormer stems and all four HTDemucs stems.

## Backend operation checks

WebGPU operations passed the GGML harness against its independent CPU reference: 268 pooling cases, 14 transposed-convolution cases and ten BF16 matmul cases. Coverage includes F32/F16 pooling, average/max padding semantics, offset and padded-row views, transposed-convolution overlap and stride gaps, odd channel/tile tails, batched matmul and BF16 values beyond F16's range. Existing tolerances were not relaxed.

Pooling uses a single dispatch with no scratch buffer. Transposed convolution uses shared-memory tiles without inserting zeros or using floating-point atomics. BF16 matrix–matrix multiplication reuses the existing tiled F32-accumulation path, preserving BF16 range. Existing F32/F16 matmul dispatch is unchanged. The upstream 30-second queue-wait timeout is unchanged; the ten-step MOSS-SoundEffect run completed within it. No model code or inference precision was changed.

See the [WebGPU build guide](../build/webgpu.md) to run models with your own weights and inputs.
