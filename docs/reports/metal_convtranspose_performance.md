# Metal ConvTranspose1d Fast Path Validation

## Decision

Enable the ConvTranspose1d col2im fast path on Metal for dilation-1 configurations.

The previous gate allowed the fast path on CUDA and HIP only. The Metal backend supports the same col2im-style execution path for the tested dilation-1 models, and the validation run shows broad end-to-end coverage with a net performance gain.

## Validation Setup

The validation compared two Metal builds:

- Baseline: ConvTranspose1d fast path gated to CUDA/HIP.
- Candidate: ConvTranspose1d fast path gated to CUDA/HIP/Metal.

The same path-test cases, model weights, backend (`metal`), and request inputs were used for the before/after runs. Logs were kept under `logs/metal_conv1d_baseline/` and `logs/metal_conv1d_gate_metal/`.

## Performance Evidence

Across the 19 matched model/path timing pairs with request-level timing:

- 18 paths were faster with the Metal gate enabled.
- 1 path was slower.
- Median request-time change: `-2.25%`.
- Mean request-time change: `-2.27%`.
- Weighted request-time change: `-1.68%`.

Per-model request-time deltas:

| Model | Baseline ms | Metal fast path ms | Delta |
| --- | ---: | ---: | ---: |
| ace_step | 58217.100 | 54403.300 | -6.55% |
| chatterbox | 6031.960 | 6008.330 | -0.39% |
| confucius4_tts | 178370.100 | 176811.100 | -0.87% |
| fish_audio | 63982.500 | 61995.500 | -3.11% |
| heartmula | 144417.000 | 141916.000 | -1.73% |
| higgs_audio_tts | 11148.800 | 10824.000 | -2.91% |
| htdemucs | 6441.160 | 6381.030 | -0.93% |
| index_tts2 | 30265.100 | 30215.100 | -0.17% |
| irodori_tts | 14611.710 | 13890.530 | -4.94% |
| omnivoice | 1471.070 | 1420.400 | -3.44% |
| pocket_tts | 1638.409 | 1530.807 | -6.57% |
| qwen3_tts | 7503.670 | 7341.790 | -2.16% |
| rvc | 20539.200 | 20252.300 | -1.40% |
| seed_vc | 19932.600 | 19327.100 | -3.04% |
| stable_audio | 5839.500 | 5799.860 | -0.68% |
| vevo2 | 29430.990 | 28769.440 | -2.25% |
| vibevoice | 54587.700 | 56827.300 | +4.10% |
| vibevoice_asr | 18110.820 | 17686.210 | -2.34% |
| voxcpm2 | 6132.150 | 5900.760 | -3.77% |

The single slower path was `vibevoice`, at `+4.10%`. The overall result still supports lifting the gate because most affected paths benefit and the weighted aggregate remains faster.

## FishAudio Metal Performance

FishAudio benefits from the ConvTranspose1d gate in its codec path, but the model remains slow on macOS Metal because the request is dominated by autoregressive generation.

For the normal Metal run with the gate enabled:

| Request | Generated frames | Wall ms | AR generate ms | Codec decode ms | Slow step ms/run | Fast step ms/run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| official_auto_voice_english | 150 | 17266.2 | 15331.3 | 1933.9 | 69.303 | 3.099 |
| official_reference_voice_clone | 160 | 24587.4 | 20063.9 | 2061.5 | 85.694 | 3.111 |
| official_inline_control_tag | 155 | 20141.9 | 18142.4 | 1998.7 | 84.153 | 3.114 |

The first request, for example, spends `88.8%` of wall time in AR generation. It runs `150` slow decoder graph steps and `1500` fast decoder graph steps. The codec decode is only about `11.2%` of wall time, so ConvTranspose1d improvements cannot dominate the total FishAudio runtime.

## FishAudio Root Cause on Metal

The FishAudio slow path is Metal backend execution of many small sequential Qwen decoder graphs:

- one slow AR graph run per generated audio frame,
- one fast AR graph run per codec codebook,
- 10 codec codebooks per frame in the tested S2 Pro model.

On the tested Apple M4 Pro machine, ggml Metal reports:

```text
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_device_init: has tensor            = false
```

That means FishAudio runs its matvec-heavy AR workload through ggml's regular Metal kernels rather than the newer tensor API path. This is a ggml Metal backend/device-generation behavior, not a ConvTranspose1d issue.

Forcing the tensor path with `GGML_METAL_TENSOR_ENABLE=1` was also tested. ggml accepted the tensor API on the M4 Pro (`has tensor = true`), but FishAudio did not improve:

| Request | Normal slow step ms/run | Tensor slow step ms/run | Normal fast step ms/run | Tensor fast step ms/run |
| --- | ---: | ---: | ---: | ---: |
| official_auto_voice_english | 69.303 | 70.402 | 3.099 | 3.148 |
| official_reference_voice_clone | 85.694 | 86.711 | 3.111 | 3.149 |
| official_inline_control_tag | 84.153 | 85.242 | 3.114 | 3.150 |

This matches the upstream ggml gate rationale: the tensor API path is currently not expected to help pre-M5/pre-A19 Apple GPUs. FishAudio exposes this limitation more than other models because its hot path is highly sequential and launch-heavy, while many other models amortize Metal overhead over larger dense graphs, convolution blocks, or fewer autoregressive steps.
