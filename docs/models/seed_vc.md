# Seed-VC

Seed-VC is wired as `--family seed_vc` for voice conversion (`vc`) and singing voice conversion (`svc`). Every route takes a source audio file plus a target voice reference.

Common CLI shape:

```bash
audiocpp_cli --task vc --family seed_vc --model models/SeedVC-MLX --backend cuda --task-route <route> --audio source.wav --voice-ref target.wav --out converted.wav
```

## Model

| Field | Value |
|---|---|
| Family | `seed_vc` |
| Model directory | `models/SeedVC-MLX` |
| Tasks | `vc`, `svc` |
| Default `vc` route | `v2_vc` |
| Default `svc` route | `v1_svc` |
| Source audio | `--audio` |
| Target voice or singer | `--voice-ref` |

## V2 Voice Conversion

Use the V2 voice-conversion path for speech voice conversion. This route uses separate intelligibility and speaker-similarity CFG controls.

| Field | Value |
|---|---|
| Task | `vc` |
| Route | `v2_vc` |
| Source audio | Required |
| Target voice reference | Required |
| Pitch controls | Not used |
| Style conversion | `convert_style=true` is parsed but not implemented |

```bash
audiocpp_cli --task vc --family seed_vc --model models/SeedVC-MLX --backend cuda --task-route v2_vc --audio source.wav --voice-ref target.wav --out converted.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--num-inference-steps` | integer | `30` | V2 CFM denoising steps. |
| `--request-option length_adjust=<float>` | float | `1.0` | Stretch or compress generated duration. |
| `--request-option intelligibility_cfg_rate=<float>` | float | `0.7` | CFG strength for source-content intelligibility. |
| `--request-option similarity_cfg_rate=<float>` | float | `0.7` | CFG strength for target-speaker similarity. |
| `--temperature` | float | `1.0` | V2 AR sampling temperature. |
| `--top-p` | `0..1` | `0.9` | V2 AR nucleus sampling. |
| `--repetition-penalty` | positive float | `1.0` | V2 AR repetition penalty. |
| `--seed` | integer | random if omitted | Seed for V2 random sampling/noise. |
| `--request-option anonymization_only=true|false` | bool | `false` | Use randomized voice conditioning instead of target-style conditioning. |
| `--request-option convert_style=true|false` | bool | `false` | Parsed option; `true` is not implemented in the current C++ route. |
| `--request-option noise_file=<path>` | path | empty | Optional deterministic noise input for validation. |

## V1 Whisper + BigVGAN Voice Conversion

Use this V1 route when you want Whisper content features and the BigVGAN vocoder.

| Field | Value |
|---|---|
| Task | `vc` |
| Route | `v1_whisper_bigvgan_vc` |
| Source audio | Required |
| Target voice reference | Required |
| Content path | Whisper |
| Vocoder | BigVGAN |

```bash
audiocpp_cli --task vc --family seed_vc --model models/SeedVC-MLX --backend cuda --task-route v1_whisper_bigvgan_vc --audio source.wav --voice-ref target.wav --out converted.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--num-inference-steps` | integer | `30` | V1 CFM denoising steps. |
| `--request-option length_adjust=<float>` | float | `1.0` | Stretch or compress generated duration. |
| `--request-option inference_cfg_rate=<float>` | float | `0.7` | V1 CFM guidance strength. |
| `--seed` | integer | random if omitted | Seed for V1 random noise. |

## V1 XLSR + HiFT Voice Conversion

Use this V1 route when you want XLSR content features and the HiFT vocoder.

| Field | Value |
|---|---|
| Task | `vc` |
| Route | `v1_xlsr_hift_vc` |
| Source audio | Required |
| Target voice reference | Required |
| Content path | XLSR |
| Vocoder | HiFT |

```bash
audiocpp_cli --task vc --family seed_vc --model models/SeedVC-MLX --backend cuda --task-route v1_xlsr_hift_vc --audio source.wav --voice-ref target.wav --out converted.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--num-inference-steps` | integer | `30` | V1 CFM denoising steps. |
| `--request-option length_adjust=<float>` | float | `1.0` | Stretch or compress generated duration. |
| `--request-option inference_cfg_rate=<float>` | float | `0.7` | V1 CFM guidance strength. |
| `--seed` | integer | random if omitted | Seed for V1 random noise. |

## V1 Singing Voice Conversion

Use this route for singing voice conversion. It supports the V1 F0 and pitch controls.

| Field | Value |
|---|---|
| Task | `svc` |
| Route | `v1_svc` |
| Source singing | Required |
| Target singer reference | Required |
| F0 extraction | Optional through `f0_condition` |
| Pitch controls | `semi_tone_shift`, `auto_f0_adjust` |

```bash
audiocpp_cli --task svc --family seed_vc --model models/SeedVC-MLX --backend cuda --task-route v1_svc --audio singing.wav --voice-ref target.wav --out svc.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--num-inference-steps` | integer | `30` | V1 CFM denoising steps. |
| `--request-option length_adjust=<float>` | float | `1.0` | Stretch or compress generated duration. |
| `--request-option inference_cfg_rate=<float>` | float | `0.7` | V1 CFM guidance strength. |
| `--request-option f0_condition=true|false` | bool | `false` | Enable F0-conditioned conversion. |
| `--request-option auto_f0_adjust=true|false` | bool | `false` | Automatically adjust F0 when F0 conditioning is enabled. |
| `--request-option semi_tone_shift=<n>` | integer | `0` | Shift pitch by semitones when F0 conditioning is enabled. |
| `--seed` | integer | random if omitted | Seed for V1 random noise. |

## Shared Controls

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task-route` | `v2_vc`, `v1_whisper_bigvgan_vc`, `v1_xlsr_hift_vc`, `v1_svc` | `v2_vc` for `vc`; `v1_svc` for `svc` | Select the conversion route. |
| `--audio` | WAV path | required | Source speech or singing audio. |
| `--voice-ref` | WAV path | required | Target voice or singer reference. |
| `--session-option seed_vc.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Weight storage type. |
