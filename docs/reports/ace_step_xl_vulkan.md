# XL ACE-Step Vulkan validation

RTX 3090 (24 GiB), Windows Release, NVIDIA Vulkan device 1, eight CPU threads, one session. Candidate: PR #737 at `48156adc`. The control uses the same PR server/framework/build configuration, with only the three ACE-Step model files restored to upstream `ed96b730`; the optimized model files were restored byte-for-byte afterward. No compilation or other GPU workload overlapped these measurements.

Both XL Turbo and XL SFT Q8-DiT packages use bf16 text/planner/VAE. Same supplied long English caption, seed 1234, five-second direct generation and eight diffusion steps. Warm medians are requests 2?8. Each run adds six mixed-history requests: changed caption/lyrics/duration/seed, planner generation, return to the original caption, and guidance=7 for the longer SFT requests. Turbo ignores guidance by design.

## Results

**56/56 original/optimized response artifacts and WAV SHA-256 comparisons are byte-identical**, across both packages and memory saver on/off. All 112 server requests returned HTTP 200. Two direct Vulkan conditioning probes passed ten histories each (20 total): repeated/changed reference bytes, frame growth/shrinkage, graph release, multi-reference bypass and invalid-order rejection. Both CLI/server and the conditioning probe build with Vulkan.

| Model | Memory saver | Original warm (s) | PR warm (s) | Less time | Original extra warm VRAM (GiB) | PR extra warm VRAM (GiB) | Sequence device peak original / PR (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XL Turbo Q8 DiT | true | 0.744 | 0.618 | 16.9% | 2.373 | 0.820 | 12.773 / 12.731 |
| XL SFT Q8 DiT | true | 0.721 | 0.620 | 14.1% | 2.373 | 0.820 | 13.935 / 12.624 |
| XL Turbo Q8 DiT | false | 0.604 | 0.586 | 3.1% | 0.105 | 0.105 | 15.814 / 14.425 |
| XL SFT Q8 DiT | false | 0.620 | 0.588 | 5.1% | 0.105 | 0.105 | 15.706 / 14.425 |

Extra warm VRAM is the request's sampled device peak minus its immediately preceding loaded-idle reading, median of requests 2?8. It excludes the loaded-idle state (weights, backend state and retained caches), not literally every non-weight byte. NVML sampled every 10 ms. Memory saver on saves 65.4% of incremental warm-request peak for both variants. Memory saver off retains graphs: incremental warm memory stays the same, while full mixed-sequence device peak falls by about 1.3?1.4 GiB. Whole-sequence peaks include loaded weights, cold requests and larger graphs; they are not the warm-overhead figure. Timing percentages describe this measured workload, not every prompt or duration.

This validates Vulkan one-session generation/conditioning on these Q8-DiT packages. It does not validate parallel slots, editing routes, longer songs, CPU/Metal or the full repository test suite.

## Local evidence

- `ace_step_xl_vulkan.json` (checked into this report directory; original local name `xl-vulkan-summary.json`): timings, memory, comparison counts and exact executable SHA-256s.
- `vulkan-original-model-manifest.json`: original model source revision and executable hash.
- `test_xl_vulkan.py`: full server matrix and probe invocations.
- `probe-vulkan-xl-turbo-q8dit.log`, `probe-vulkan-xl-sft-q8dit.log`: all 20 conditioning histories.
- Raw configs, requests, traces, results and WAVs in the local test workspace: `outputs/ace-graph-memory-20260929/xl-vulkan-{original,pr}-{mem,nomem}/`.
- Builds: `build_pr_vulkan.cmd`, `build_vulkan_control.py` and associated logs. The latter restores candidate source in a `finally` block and rebuilds it after making the original-model control.

Run the test matrix from the workspace root after building both executables:

```powershell
python outputs/ace-next-20260929/test_xl_vulkan.py
```

The test harness starts hidden server subprocesses and stops its own server above a 23.5 GiB device-use guard. No guards fired; no server was left running after the tests. Cleanup is process termination, not proof of graceful server shutdown.
