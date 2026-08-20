# F5-TTS (community model)

[F5-TTS](https://github.com/SWivid/F5-TTS) is an open-source zero-shot voice-cloning TTS built on a
flow-matching diffusion transformer (DiT) with a ConvNeXt text conditioner and a Vocos vocoder.
This community port also targets the [Habibi-TTS](https://github.com/SWivid/Habibi-TTS) finetune —
a multi-dialect Arabic checkpoint suite (MSA, SAU, UAE, ALG, IRQ, EGY, MAR, OMN, TUN, LEV, SDN, LBY)
from the same authors — which uses the identical architecture, giving Arabic support through the
same family (`habibi` / `habibi_tts` are registered as aliases).

**Status: M4 — inference wired end to end.** The session (`src/community_models/f5_tts/session.cpp`)
runs full synthesis via `f5_synthesize` (text pipeline with Habibi dialect tokens → batched-CFG DiT
Euler sampler → Vocos vocoder) and is served by `audiocpp_server`. Parity vs the reference PyTorch
implementation is covered by golden harnesses (DiT stage taps, batched CFG incl. the null branch,
tokenizer ids) plus a whisper.cpp ASR pronunciation check; see `/mnt/ai/f5-parity/run_all.sh`.

## Milestones

Each milestone is gated on parity against the reference PyTorch implementation (cosine similarity
≥ 0.999 on fixed inputs) plus a listening check, matching the evidence bar described in #54 and
PR #180.

| Milestone | Scope | Status |
|---|---|---|
| M0 | Family registration, model spec, stub session, this doc | done |
| M1 | Weight loading + mel-Vocos decode path (ConvNeXt + iSTFT) | done (mel-corr 0.9963) |
| M2 | DiT forward (RoPE, adaLN) + ConvNeXt text conditioner | done (all stages cosine 1.000000) |
| M3 | CFM sampler (Euler, sway + EPSS, CFG null-branch parity), inference wiring, En/Ar samples | done |
| M4 | Long-form chunking, RTF/VRAM evidence, server wiring | done (0.51x RTF on RTX 3090; GGUF package pending) |

## Server usage

`audiocpp_server.json` entry: family `f5_tts`, model path = checkpoint directory (must contain
exactly one DiT `*.safetensors` + `vocab.txt`), session option `f5_tts.vocos_path` pointing at the
Vocos checkpoint (or place `vocos.safetensors` next to the DiT checkpoint), optional
`f5_tts.dialect` default. Requests take `reference_text` (required), `dialect`, `speed`, `seed`,
`num_inference_steps`, `guidance_scale`, `sway_sampling_coef`.

## Relevant building blocks already in-tree

- Vocos vocoder: `src/models/vevo2/components.cpp`, `src/models/index_tts2/`
- iSTFT: `src/models/miocodec/`, `src/models/seed_vc/`
- Flow matching: `src/models/vevo2/fm.cpp`
- RoPE DiT / adaLN: `src/models/stable_audio/foundation/rf_dit.cpp`

## Checkpoints

| Model | Source | License |
|---|---|---|
| F5-TTS Base (en/zh) | `SWivid/F5-TTS` | cc-by-nc-4.0 |
| Habibi Unified (ar) | `SWivid/Habibi-TTS` | cc-by-nc-sa-4.0 |

Both are non-commercial licenses; keep that in mind before shipping anything built on them.
