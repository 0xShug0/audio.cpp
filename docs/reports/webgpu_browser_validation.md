# Browser WebGPU validation

## Test configuration

These are historical, sequential, single-request measurements on an NVIDIA RTX 5090 with driver 595.84, Chrome 150 and Emscripten 6.0.13. Builds used Debug with `-O3 -g`, WASM SIMD, `--backend webgpu`, `--threads 1` and `--log`. Some cases used pthreads for asynchronous prefetch. The results span development builds and are not a fresh benchmark of the final browser integration.

RTF is request time divided by generated audio duration, or input audio duration for ASR, alignment and separation. Request time was collected from `session.wall_ms`; model loading and WebAssembly initialization are excluded, while first-request graph/shader work is included. **RTF below 1 means faster than real time.**

Peak VRAM is the observed NVML graphics-memory peak of a dedicated Chrome GPU process, sampled nominally every 20 ms across loading and inference. It includes approximately 32–34 MiB of browser baseline memory and can miss short peaks. It is not an exact WebGPU allocation counter.

## Results

| Model | Task | GGUF format | Input | Audio seconds | Request seconds | RTF | Peak VRAM (MiB) | WASM |
|---|---|---|---|---:|---:|---:|---:|---|
| Irodori-TTS Small MF | TTS | Q8_0 | 205 chars | 41.800 | 4.766 | 0.1140 | 3581.3 | 32 |
| Irodori-TTS Small RF | TTS | Q8_0 | 205 chars | 41.800 | 13.736 | 0.3286 | 3708.9 | 32 |
| Kokoro-82M | TTS | Q8_0 | 245 chars | 15.075 | 1.086 | 0.0720 | 1811.0 | 32 |
| Magpie Multilingual | TTS | Q8_0 | 245 chars | 13.375 | 50.691 | 3.7901 | 1457.2 | 32 |
| Ming3-TTS-0.5B | TTS | Q8_0 | 245 chars | 12.500 | 20.262 | 1.6210 | 1385.1 | 32 |
| MOSS-TTS Nano | TTS | Q8_0 | 245 chars | 14.000 | 15.075 | 1.0768 | 561.6 | 32 |
| OmniVoice | TTS | Q8_0 | 245 chars | 15.000 | 5.027 | 0.3351 | 1776.1 | 32 |
| PocketTTS | TTS | Q8_0 | 245 chars | 11.600 | 2.062 | 0.1778 | 735.4 | 32 |
| Qwen3-TTS 1.7B CustomVoice | TTS | Q8_0 | 245 chars | 18.080 | 21.590 | 1.1941 | 3827.7 | 64 |
| Supertonic 3 | TTS | F32/I64 (Q8-labelled) | 245 chars | 16.023 | 0.687 | 0.0429 | 502.7 | 32 |
| VibeVoice 1.5B | TTS | Q8_0 | 256 chars | 13.200 | 21.878 | 1.6574 | 4501.0 | 64 |
| VoxCPM-0.5B | TTS | Q8_0 | 245 chars | 15.360 | 15.783 | 1.0276 | 2432.9 | 32 |
| Cohere Transcribe | ASR | Q4_0 | 40 s audio | 40.000 | 1.548 | 0.0387 | 2138.8 | 32 |
| Fun-ASR Nano | ASR | Q8_0 | 40 s audio | 40.000 | 2.609 | 0.0652 | 1795.4 | 32 |
| GigaAM Multilingual Large | ASR | F16 | 40 s audio | 40.000 | 0.329 | 0.0082 | 1216.2 | 32 |
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
| YuE2 | GEN | Q4_0 + F16 VAE | Instrumental; 1,500-token cap | 59.999 | 37.021 | 0.6170 | 3014.4 | 64 |


## Workload and configuration notes

- TTS used 245 English characters; Irodori used 205 Japanese characters. VibeVoice's 256-character script includes its `Speaker 0:` control label. ASR/alignment used 40 seconds of speech, and separation used 40 seconds of stereo music.
- Irodori RF/MF used `codec_decode_chunk_steps=256` to fit the browser storage-buffer binding limit. Several WASM32 cases used reduced model metadata arenas. These were request/session settings, not model-default changes.
- VibeVoice used Q8 weights and WASM64 with a 16 GiB host-memory ceiling. That ceiling is separate from GPU memory.
- YuE2 used Q4 main weights, an F16 VAE, `cot=off`, eight NAR steps, `semantic_max_tokens=1500` and `vae_tile_frames=256`. It reached the token cap and generated approximately 60 seconds.
- Supertonic's Q8-labelled package contains F32/I64 tensors; the table reports the actual tensor types rather than inferring them from the package name.
- Moonshine/Cohere request timings came from development-only instrumentation that is excluded from the browser integration. Their historical timing values are included here; a current build does not emit those added timing keys.
- These observations have no cold-cache guarantee and do not establish warmed throughput, hardware-independent performance or a complete regression comparison. This report does not include the input assets or per-case configurations required for exact reproduction. See [the build guide](../build/webgpu.md) to run models with your own GGUFs and inputs.

## Output checks

All generated TTS/music WAVs were finite and nonconstant. Independent SenseVoice ASR found seven of twelve TTS transcripts exact after normalizing punctuation, whitespace and case: Kokoro, VoxCPM, OmniVoice, Ming3, Qwen3-TTS, Irodori RF and VibeVoice. Pocket, Magpie and Irodori MF had small differences; Supertonic and MOSS Nano had larger discrepancies requiring further verification. ASR differences alone cannot be attributed to synthesis, and these checks do not establish waveform parity or a model-quality score.

Alignment produced 101 ordered words within the 40-second input. Separation produced both BS-RoFormer stems and all four HTDemucs stems. IndexTTS2.5, AudioSR and ACE-Step Turbo were excluded from the sweep.

## Integration checks

The browser integration also passed Debug WASM32/WASM64 `--version` checks and sequential short generation smokes for PocketTTS, Kokoro with eSpeak data, and Qwen3-TTS with a GGUF over 2 GiB. All three produced finite, nonconstant WAVs. These were functional checks, not a new performance sweep.

The GGML operator checks passed 9/9 focused matrix cases and 169/169 copy, concat, masking, mean, row-sum and reflection-padding cases against CPU reference. An invalid F16 wide-range fixture was removed because its inputs exceed FP16 range; regular F16 coverage and valid F32/BF16/Q8 overflow checks remain. A full-suite pass has not been established.

WASM64 with static eSpeak also passed a Kokoro browser generation check. The child library’s C/C++ and Debug flags were verified to match the parent, including the WASM64 architecture and pthread flags.
