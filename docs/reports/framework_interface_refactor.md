# PReLU, Qwen3 Audio Encoder, and S3Gen Interface Refactor

| CUDA longform case | Before (`4b30d92d`) warm time (s) | After (`4a8eb721`) warm time (s) | Before RTF | After RTF | Before peak VRAM (MiB) | After peak VRAM (MiB) | A/B output parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Chatterbox | 50.041 | 49.791 | 0.12085 | 0.12024 | 5160 | 5126 | Exact |
| Supertonic | 1.615 | 1.624 | 0.00426 | 0.00428 | 1152 | 1152 | Exact |
| Chatterbox Turbo | 32.242 | 31.914 | 0.09904 | 0.09803 | 3504 | 3352 | Exact |
| KittenTTS2 | 31.391 | 31.252 | 0.09398 | 0.09357 | 6832 | 6792 | Exact |
| TF-GridNet | 25.053 | 24.982 | 0.14972 | 0.14929 | 850 | 850 | Exact |
| MossFormer2 | 3.886 | 3.832 | 0.02322 | 0.02290 | 1058 | 1058 | Exact |
| RE-USE | 5.345 | 5.318 | 0.03194 | 0.03178 | 3008 | 3008 | Exact |
| Qwen3-ASR | 3.451 | 3.437 | 0.01053 | 0.01049 | 3726 | 3726 | Exact |
| Audio8 ASR | 0.947 | 0.949 | 0.00289 | 0.00290 | 2024 | 2024 | Exact |
| Qwen3-ASR + Forced Aligner | 5.165 | 5.164 | 0.01577 | 0.01576 | 5332 | 5332 | Exact |
| Index-Echo | 22.744 | 22.606 | 0.06943 | 0.06900 | 6274 | 6274 | Exact |
| HeartMuLa (memory saver) | 32.851 | 32.723 | 0.27358 | 0.27251 | 20448 | 20448 | Exact |
