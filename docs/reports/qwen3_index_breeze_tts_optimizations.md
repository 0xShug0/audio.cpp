# Qwen3-TTS, IndexTTS, and BreezeTTS Optimization Results

RTX 5090, CUDA Debug build, Q8 weights, 8 CPU threads, fixed seed. Each build/model ran sequentially in a fresh server with the same mixed-length request sequence and logging enabled. Peak VRAM was sampled continuously across the sequence (5 ms target interval); warm RTF is the median of four identical long-request repeats. Positive RTF changes mean slower.

| Model/path | Before peak (MiB) | Current peak (MiB) | Peak change (MiB) | Before warm RTF | Current warm RTF | RTF change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-TTS Base clone | 6,376 | 5,070 | -1,306 | 0.10238 | 0.10417 | +1.75% |
| Qwen3-TTS CustomVoice | 6,022 | 5,068 | -954 | 0.10207 | 0.10307 | +0.98% |
| Qwen3-TTS VoiceDesign | 6,180 | 5,110 | -1,070 | 0.10334 | 0.10347 | +0.12% |
| IndexTTS2, including emotion request | 6,288 | 6,290 | +2 | 0.14590 | 0.14676 | +0.60% |
| IndexTTS2.5, including emotion request | 5,918 | 5,908 | -10 | 0.08688 | 0.08623 | -0.75% |
| BreezeTTS clone | 6,914 | 6,716 | -198 | 0.22636 | 0.22632 | -0.02% |
| BreezeTTS voice design | 6,696 | 6,702 | +6 | 0.27655 | 0.27631 | -0.09% |

Index warm RTF above measures ordinary synthesis, not the emotion request. Before the first emotion request, the measured ordinary-only peaks were:

| Path | Before peak (MiB) | Current peak (MiB) | Peak change (MiB) |
| --- | ---: | ---: | ---: |
| IndexTTS2 ordinary synthesis | 6,288 | 5,556 | -732 |
| IndexTTS2.5 ordinary synthesis | 5,918 | 5,188 | -730 |

## Qwen3 timing cross-check

The same saved binaries were retested in reversed order (current, then before), without rebuilding or changing code. Four warmed repeats per path:

| Path | Before warm RTF | Current warm RTF | RTF change | Initial A/B change |
| --- | ---: | ---: | ---: | ---: |
| Base clone | 0.10309 | 0.10196 | -1.09% | +1.75% |
| CustomVoice | 0.10215 | 0.10232 | +0.17% | +0.98% |
| VoiceDesign | 0.10190 | 0.10295 | +1.03% | +0.12% |

Measured component medians in the reversed run:

| Path/component | Before (ms) | Current (ms) |
| --- | ---: | ---: |
| Base cached-step compute | 489.660 | 481.306 |
| CustomVoice cached-step compute | 482.797 | 481.691 |
| VoiceDesign cached-step compute | 478.612 | 478.666 |
| VoiceDesign code-predictor compute (unchanged implementation) | 675.911 | 684.647 |

The initial slowdown did not reproduce consistently: Base reversed direction, CustomVoice narrowed to 0.17%, and VoiceDesign's changed cached-step compute differed by only 0.054 ms. Its total slowdown was mainly in the unchanged code predictor. These results support run-to-run timing variation rather than a demonstrated cached-step regression; a specific clock/thermal/scheduling cause was not established. They do not prove that every sub-percent effect is noise.

**Parity-safe in the tested coverage:** all 82 mixed-length A/B output WAV pairs and all 18 reversed-order Qwen3 pairs were byte-identical. This is before/current C++ parity, not a new Python parity claim.
