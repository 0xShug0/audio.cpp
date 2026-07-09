# Memory Saver

`mem_saver` session options reduce graph workspace VRAM or post-request resident VRAM by using tighter graph allocators or releasing staged graph and cache state. They are intended for deployments that prefer lower memory over maximum graph/cache reuse between later requests.

Use this page to collect memory-saver validation results across models. Add new rows with the same columns so results stay comparable.

## Validation Protocol

Use server validation with one measurement running at a time.

For each model:

1. Run the default/native configuration in a fresh `audiocpp_server`.
2. Run the matching `<family>.mem_saver=true` configuration in a fresh `audiocpp_server`.
3. Wait for `/health`, sample VRAM during exactly one `/v1/tasks/run`, then stop the server before the next case.
4. Record peak VRAM, final resident VRAM, server wall time, generated audio length, and RTF.
5. Use the same request, native/default weights, backend, seed, duration, and sampling settings for the default and `mem_saver` pair.
6. Use a request that actually stresses the model memory path. For longform music models, the existing checks used 120 second targets and long lyrics/text.
7. On a stress request, `mem_saver` should usually lower peak or final resident VRAM. If neither drops, check the release logs and confirm the request exercised the graph/cache state that `mem_saver` is supposed to affect.

## Results

Native/default weights were used for all rows.

- ACE-Step base used a corrected long-lyrics request: 4146 lyric characters, 120 second target audio.
- HeartMuLa used a corrected long-lyrics request: 12278 text characters, 120 second target audio.
- Stable Audio rows use existing 120 second server measurements.
- OmniVoice used default generation parameters, native/default weights, no explicit text chunk size, and no seed for the memory-stat pair.
- Chatterbox used one fixed-seed voice-clone request with native/default weights and no explicit max token cap.
- Qwen3 TTS Base used a five-request voice-clone server sequence: two small requests, one 6026-character long request with `max_tokens=4096`, then two small requests. Peak VRAM was sampled during each request; resident VRAM is the post-response value after the long request.
- VoxCPM2 used the OpenAI-compatible offline speech endpoint with a 2048-character voice-design request, `seed=1234`, `max_tokens=512`, `num_inference_steps=10`, and `guidance_scale=2.0`. The default and `mem_saver` WAV outputs were byte-identical.

| Model | Mode | Peak VRAM | Resident VRAM | Server wall | Audio | RTF |
|---|---|---:|---:|---:|---:|---:|
| ACE-Step base | default | 20098 MiB | 19540 MiB | 30402.5 ms | 120s | 0.253354 |
| ACE-Step base | mem_saver | 20098 MiB | 10332 MiB | 27470.2 ms | 120s | 0.228918 |
| HeartMuLa | default | 25600 MiB | 25600 MiB | 47248.6 ms | 120.08s | 0.393476 |
| HeartMuLa | mem_saver | 25600 MiB | 21762 MiB | 47349.2 ms | 120.08s | 0.394313 |
| Stable Audio 3 small music | default | 3658 MiB | 3652 MiB | 1485.97 ms | 120s | 0.0123831 |
| Stable Audio 3 small music | mem_saver | 3070 MiB | 2868 MiB | 1445.73 ms | 120s | 0.0120478 |
| Stable Audio 3 small SFX | default | 3652 MiB | 3652 MiB | 1407.58 ms | 120s | 0.0117298 |
| Stable Audio 3 small SFX | mem_saver | 2980 MiB | 2868 MiB | 1415.61 ms | 120s | 0.0117968 |
| Stable Audio 3 medium | default | 10440 MiB | 10440 MiB | 3847.98 ms | 120s | 0.0320665 |
| Stable Audio 3 medium | mem_saver | 10324 MiB | 9468 MiB | 3830.24 ms | 120s | 0.0319186 |
| OmniVoice | default | 11396 MiB | 11396 MiB | 947.392 ms | 4.32s | 0.219304 |
| OmniVoice | mem_saver | 10526 MiB | 3662 MiB | 968.823 ms | 4.32s | 0.224264 |
| Chatterbox | default | 13408 MiB | 13408 MiB | 4264.11 ms | 6.76s | 0.630786 |
| Chatterbox | mem_saver | 12272 MiB | 5074 MiB | 4832.43 ms | 6.76s | 0.714856 |
| Qwen3 TTS Base voice clone | default | 7518 MiB | 7518 MiB | 122013.55 ms | 327.6s | 0.372447 |
| Qwen3 TTS Base voice clone | mem_saver | 7520 MiB | 5684 MiB | 121919.91 ms | 327.6s | 0.372161 |
| VoxCPM2 offline voice design | default | 13080 MiB | 13039 MiB | 15325.7 ms | 93.12s | 0.16458 |
| VoxCPM2 offline voice design | mem_saver | 12456 MiB | 5966 MiB | 14924 ms | 93.12s | 0.160266 |
