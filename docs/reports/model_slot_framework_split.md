# Model slot support after framework extraction

PR #715 contains the reusable server/session implementation extracted from
PR #706 onto upstream `77491a33`. PR #706 is its dependent model-support
follow-up. Merge #715 first; this does not merge either PR automatically.

## Review boundary

The framework owns the scheduler, session pool, parallel-session contract,
per-slot dispatch, management/lifecycle guards, shared-weight cache ownership,
tensor-source concurrency protection and generic regression tools/tests.
Its audited admission tables are empty, so existing models keep one slot.

The model follow-up retains the specialized Higgs v3 CUDA/Vulkan adapter,
model-specific weight sharing, stage guards, codec/graph/backend fixes and
offline admission rules. The current table admits 89 CUDA and 75 Vulkan
family/task pairs, each at its own validated two-to-four-slot limit.
Model-capacity assertions now run in `model_slot_admission_test`, separate
from the unchanged common pool/scheduler tests.

All 59 changed model implementation/header files match the integrated
pre-split source (`cbc7a0de`) byte-for-byte after newline normalization.
The common scheduler, runtime, pool, interfaces, shared cache and tensor-source
implementation match #715 exactly. Existing individual model commits are
retained; no force push or history rewrite is needed.

[Model-only comparison](https://github.com/mirek190/audio.cpp/compare/agent/common-slot-framework-20260927...agent/common-model-slots-20260926).
GitHub's main-based #706 diff still includes the shared dependency until #715
is merged and #706 is updated with main. Its fork-based comparison provides
the separate model review meanwhile.

## Fresh validation of the split

Windows/MSVC Release, RTX 3090 24 GiB, eight inference threads, CUDA device 0,
Vulkan device 1. Both full-model CLI/server builds pass. Focused CTests pass
8/8 on CUDA and 9/9 on Vulkan, including model admission; Vulkan adds the
immutable shared-weight backend test. Standalone CPU/CUDA/Vulkan framework
validation is documented separately in [the framework report](common_slot_framework.md).

| Backend | Package | Slots | Cold / warm / reload parity | Observed active slots |
|---|---|---:|---|---:|
| CUDA | Canary 180M Flash Q8_0 | 4 | Exact in all three phases | 4 |
| CUDA | Higgs v3 TTS 4B Q8_0 | 4 | Exact in all three phases | 4 |
| CUDA | BS-RoFormer ep368 Q8_0 | 4 | Exact in all three phases | 4 |
| VULKAN | Canary 180M Flash Q8_0 | 4 | Exact in all three phases | 4 |
| VULKAN | Higgs v3 TTS 4B Q8_0 | 4 | Exact in all three phases | 4 |
| VULKAN | BS-RoFormer ep368 Q8_0 | 4 | Exact in all three phases | 4 |

Each backend/package case first generates three single-slot responses (cold,
warm and repeated warm). A fresh four-slot server then runs four identical
requests concurrently in cold, warm and unload/reload phases. Every phase
observes four active loaded slots, returns HTTP 200, matches the corresponding
single-slot response byte-for-byte after removing only top-level `timing`, and
drains active/queued counters. This compares exact WAV bytes (including both
BS-RoFormer stems), transcript and response metadata. There are 18 serial
references and 72 parallel responses across six cases. All servers remain
healthy and are stopped after their case.

Higgs cold and warm reference outputs are intentionally compared separately;
the known fixed-seed cold/warm difference is not treated as a slot regression.
This is an integration/parity check, not a new speed benchmark or a rerun of
the complete model catalogue. Historical model audits and newer follow-up
reports remain the evidence for the other admitted packages and counts.

The test uses the saved CUDA catalogue's existing request/config fixtures,
without changing generation limits, precision or separation overlap. Local
evidence: `outputs/pr706-framework-split/smoke_models.py`,
`model-smoke-results.json`, `model-smoke-{backend}-{family}/`, and
`ctest-{cuda,vulkan}-models.log`; full build logs are under
`outputs/scheduler-timeout-fix/build-{cuda,vulkan}-model-split.log`.
The checkpoint-backed mixed-input harness remains
`tests/higgs_audio_tts/server_parallel_compat.py`.

## Admission and outstanding limits

The authoritative policy is `app/server/audited_model_slots.h`; survey reports
contain historical failures/capacities and should be read with their follow-ups.
A two-slot pass does not certify three or four, other checkpoints/options,
other tasks, arbitrary CPU execution, or parallel streaming/native batching.
Private mutable state remains per slot; some problematic GPU stages use
model-specific serialization. Multi-slot execution is not a universal speedup.

Ten additional Vulkan packages have passed two-slot audits but remain capped
at one in production pending admission: ACE-Step, Chatterbox, Chatterbox Turbo,
MMS Forced Aligner, Vevo2, AudioSR, MioCodec, MioTTS, NeuTTS and VieNeu v3 Turbo.
Hardware/backend exclusions remain excluded. Remote CI is separate from the
local build, focused-test and checkpoint checks reported above.
