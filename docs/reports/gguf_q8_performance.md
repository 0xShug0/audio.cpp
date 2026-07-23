# GGUF Q8 Performance

This report compares tested 16-bit GGUF packages against Q8_0 GGUF packages on CUDA.
In the measured routes, Q8_0 improves wall time by up to **1.53x** and lowers
peak VRAM by up to about **37%** compared with the matching 16-bit GGUF package.

Notes:

- Offline session rows exclude the warmup request.
- "Speed vs real time" means generated or processed audio duration divided by wall time.
- "Q8 vs 16-bit" means 16-bit wall time divided by Q8 wall time; values above `1.0x` mean Q8 was faster.
- `qwen3_tts` uses the regenerated `q8_v2` GGUF, which keeps speaker-sensitive tensors in 16-bit storage.
- `higgs_audio_tts` 16-bit long-form uses `text_chunk_size=256`.

## Summary

Q8 gives the clearest end-to-end wins on the larger AR-style models:

- `higgs_audio_tts`: Q8 is **1.38x-1.53x faster** on warmed requests and lowers peak VRAM from **9.1 GiB to 5.9 GiB**.
- `fish_audio`: Q8 is **1.26x-1.34x faster** on warmed requests and lowers peak VRAM from **11.8 GiB to 7.5 GiB**.
- `voxtral_realtime`: Q8 is **1.31x-1.38x faster** in offline ASR and lowers peak VRAM from **10.7 GiB to 7.6 GiB**.

Smaller or already memory-light models still load and run with Q8, but the speed gain can be modest. Treat Q8 as a measured route choice, not a guaranteed win for every model.

## TTS Offline Long-Lived Session

| Model | 16-bit speed vs real time | Q8 speed vs real time | Q8 vs 16-bit | 16-bit peak VRAM | Q8 peak VRAM |
|---|---:|---:|---:|---:|---:|
| `pocket_tts` | 68.5x-99.0x | 78.7x-102.0x | 1.01x-1.21x | 2060 MiB | 1932 MiB |
| `chatterbox` | 6.2x-8.2x | 6.2x-8.7x | 1.01x-1.11x | 4333 MiB | 4309 MiB |
| `omnivoice` | 7.2x-24.4x | 8.2x-29.0x | 1.08x-1.22x | 3250 MiB | 3157 MiB |
| `qwen3_tts` | 5.3x-7.0x | 5.8x-7.9x | 1.01x-1.12x | 7873 MiB | 6397 MiB |
| `fish_audio` | 2.5x-2.6x | 3.1x-3.4x | 1.26x-1.34x | 12093 MiB | 7669 MiB |
| `higgs_audio_tts` | 6.1x-6.7x | 8.8x-10.1x | 1.38x-1.53x | 9326 MiB | 6024 MiB |

## TTS Offline Long-Form

| Model | 16-bit speed vs real time | Q8 speed vs real time | Q8 vs 16-bit | 16-bit peak VRAM | Q8 peak VRAM |
|---|---:|---:|---:|---:|---:|
| `pocket_tts` | 82.6x | 85.5x | 1.04x | 2098 MiB | 2298 MiB |
| `chatterbox` | 7.9x | 8.5x | 1.06x | 5033 MiB | 4554 MiB |
| `omnivoice` | 38.6x | 43.7x | 1.13x | 3175 MiB | 3069 MiB |
| `qwen3_tts` | 5.9x | 6.6x | 1.12x | 9412 MiB | 8138 MiB |
| `fish_audio` | 2.5x | 3.3x | 1.27x | 12261 MiB | 9228 MiB |
| `higgs_audio_tts` | 6.1x | 8.5x | 1.41x | 11878 MiB | 9129 MiB |

## ASR Offline Long-Lived Session

| Model | 16-bit speed vs real time | Q8 speed vs real time | Q8 vs 16-bit | 16-bit peak VRAM | Q8 peak VRAM |
|---|---:|---:|---:|---:|---:|
| `voxtral_realtime` | 11.1x-12.5x | 14.7x-16.7x | 1.31x-1.38x | 10909 MiB | 7754 MiB |
| `nemotron_asr` | 277.8x-384.6x | 285.7x-400.0x | 1.03x-1.12x | 5125 MiB | 4028 MiB |

## ASR Streaming Long Audio

| Model | 16-bit server TTFT | Q8 server TTFT | 16-bit client TTFT | Q8 client TTFT | 16-bit speed vs real time | Q8 speed vs real time | 16-bit peak VRAM | Q8 peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `voxtral_realtime` | 207.308 ms | 179.896 ms | 550.526 ms | 530.558 ms | 4.7x | 5.4x | 12616 MiB | 8972 MiB |
| `nemotron_asr` | 205.007 ms | 214.822 ms | 488.629 ms | 499.453 ms | 31.6x | 33.4x | 2816 MiB | 2497 MiB |
