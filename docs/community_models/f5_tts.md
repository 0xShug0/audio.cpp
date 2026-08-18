# F5-TTS (community model) — M0 scaffolding

[F5-TTS](https://github.com/SWivid/F5-TTS) is an open-source zero-shot voice-cloning TTS built on a
flow-matching diffusion transformer (DiT) with a ConvNeXt text conditioner and a Vocos vocoder.
This community port also targets the [Habibi-TTS](https://github.com/SWivid/Habibi-TTS) finetune —
a multi-dialect Arabic checkpoint suite (MSA, SAU, UAE, ALG, IRQ, EGY, MAR, OMN, TUN, LEV, SDN, LBY)
from the same authors — which uses the identical architecture, giving Arabic support through the
same family (`habibi` / `habibi_tts` are registered as aliases).

**Status: M0 — scaffolding only.** The family registers and the model loads through the spec-backed
loader, but inference is not implemented yet; running a task fails loudly rather than producing
silence. F5-TTS is on the candidate list in #34 (struck through, "contributions welcome"), and this
draft follows the community-model process from #54 (open early, milestone-gated evidence).

## Milestones

Each milestone is gated on parity against the reference PyTorch implementation (cosine similarity
≥ 0.999 on fixed inputs) plus a listening check, matching the evidence bar described in #54 and
PR #180.

| Milestone | Scope |
|---|---|
| M0 | Family registration, model spec, stub session, this doc (this PR) |
| M1 | Weight loading + mel-Vocos decode path (ConvNeXt + iSTFT) |
| M2 | DiT forward (RoPE, adaLN) + ConvNeXt text conditioner |
| M3 | CFM sampler (Euler, sway sampling), inference wiring, En/Ar samples |
| M4 | Long-form chunking via shared text chunkers, RTF/VRAM evidence, GGUF package |

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
