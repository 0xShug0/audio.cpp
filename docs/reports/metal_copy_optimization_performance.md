# Metal Copy Optimization Performance

Measured on Apple M4 Pro with the Metal backend and 8 CPU threads. Baseline is clean `dev` at `3b1bbae`; optimized is the same tree with the Metal copy fast-path patch.

| Model | Request | RTF before | RTF after | Improvement |
|---|---|---:|---:|---:|
| DotTTS SOAR Q8_0 | Request 1 | 2.15469 | 1.62996 | 24.35% |
| DramaBox Q8_0 | Request 1 | 15.5369 | 11.0525 | 28.86% |
| DramaBox Q8_0 | Request 2 | 31.8510 | 18.0591 | 43.30% |
| PocketTTS Q8_0 | Request 1 | 0.072414 | 0.052740 | 27.17% |
| PocketTTS Q8_0 | Request 2 | 0.074366 | 0.054182 | 27.14% |
| OmniVoice Q8_0 | Request 1 | 0.358067 | 0.310377 | 13.32% |
| OmniVoice Q8_0 | Request 2 | 0.396263 | 0.346676 | 12.51% |
| Qwen3-TTS VoiceDesign Q8_0 | Request 1 | 0.543776 | 0.537302 | 1.19% |
| Qwen3-TTS VoiceDesign Q8_0 | Request 2 | 0.546709 | 0.512593 | 6.24% |
| Qwen3-TTS Base Voice Clone Q8_0 | Request 1 | 1.19203 | 1.14700 | 3.78% |
| Qwen3-TTS Base Voice Clone Q8_0 | Request 2 | 0.929893 | 0.863321 | 7.16% |
| IndexTTS2 Q8_0 | Request 1 | 2.337110 | 2.183560 | 6.57% |
