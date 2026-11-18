# granite_speech_2b E2 validation report

Task E2 gate verdict on the E1 artifacts (base `0b03d489`).
Status: **QUALIFIED, NOT RELEASED** — all measured gates PASS, but no publish
path exists in this environment (see §9), so the package stays `experimental`
per the brief's upload-check rule. No flip commit was made.

## 1. Environment (pinned)

| Item | Value |
|---|---|
| Hardware | 24-core x86_64, 31 GiB RAM (WSL2 on Windows host) |
| OS | Ubuntu 24.04.4 LTS, kernel 6.18.58.1-microsoft-standard-WSL2 |
| Compiler | gcc 13.3.0 (`/usr/bin/c++`) |
| Source commit | `0b03d489` (E2 base; working tree adds only the E2 files below) |
| Binary stamp | `audiocpp_cli --version` reports `git: 123e3a85`; the E1 rebuild after `0b03d489` did not re-run configure, so the stamp predates the fix commit. Delta `123e3a85..0b03d489` = scalar-guard warning + shared unit test only (no CLI/runtime change); all runs below use these rebuilt binaries. |
| CMake flags | `-DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=granite_speech_2b,granite5asr -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON` (build dir `/tmp/opencode/granite-e1`) |
| Backends | cpu only (`backends: cpu`). No CUDA/Metal measured; nothing claims them. |
| Model revision | `ibm-granite/granite-speech-5.0-2b` @ `ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7` |
| Threads/device | `--backend cpu --threads 8` everywhere unless noted |

## 2. Artifacts under gate (E1 outputs, unchanged, re-verified by sha256)

| File | Size | SHA-256 |
|---|---|---|
| `build/granite-2b-handoff/dist/granite-speech-5.0-2b-orig.gguf` | 4633736448 | `077a944d8079068403ea987d151fb779a9010c05435e6079079bd2bd25f870b` |
| `build/granite-2b-handoff/dist/granite-speech-5.0-2b-q8_0.gguf` | 3296802048 | `0a8aee12e172b391c5e4d0460b5e53daeaa685d3d6f378f93bd906965a636e` |
| `build/granite-2b-handoff/hf/` (native ckpt) | — | `model.safetensors` + 8 sidecars (E1 staging) |

`build/` is git-ignored; artifacts stay untracked. The release candidate is
`granite-speech-5.0-2b-q8_0.gguf` as staged package `granite_speech_2b_q8_0`.

## 3. Eval harness + manifest (new committed files)

- `tests/granite_speech_2b/eval_metrics.py` — pinned NFKC normalization; WER
  for `en,fr,de,es,pt` (uppercase, hyphen→space, keep Unicode letters +
  apostrophe + space); CER for `ja` (NFKC, strip whitespace, keep
  punctuation/kana); separate punctuation/case pass (NFKC only, exact flag +
  raw char distance, never mixed into gate numbers); Levenshtein with
  sub/del/ins breakdown + denominators. Self-tested (exact-fold 0 errors,
  1-sub scoring, ja CER, fr-as-WER-lang, `xx` rejected).
- `tests/granite_speech_2b/eval_manifest.jsonl` — 10 rows: 5 `ready` (4× en
  LibriSpeech + 1× ja Common Voice, all with audio SHA-256, license, reference
  path, request language), 1 `smoke-only` (IBM `multilingual_sample.wav`,
  no human reference ships with the checkpoint → cross-artifact agreement
  only), 4 `pending` (fr/de/es/pt: no in-tree fixtures; fetch recipe recorded
  per row).
- `tools/asr_wer.py` used unmodified for the en smoke (brief requirement).

## 4. GGUF quality comparison (same PCM, prompts, budgets, scoring)

Commands (per model `<m>` in native/orig/q8_0):

```bash
python3 tools/asr_wer.py --cli /tmp/opencode/granite-e1/bin/audiocpp_cli \
  --family granite_speech_2b --model <m> --backend cpu --threads 8 --label <name>
python3 tests/granite_speech_2b/eval_metrics.py --manifest tests/granite_speech_2b/eval_manifest.jsonl \
  --cli /tmp/opencode/granite-e1/bin/audiocpp_cli --model <m> --backend cpu --threads 8 \
  --label <name> --out /tmp/opencode/e2-eval-<name>.json
# IBM sample (language=en), stdout saved to /tmp/opencode/e2-ibm-<name>.stdout
audiocpp_cli --task asr --mode offline --family granite_speech_2b --model <m> \
  --audio build/granite-2b-handoff/hf/multilingual_sample.wav --backend cpu --threads 8 \
  --request-option language=en
```

Full script: `/tmp/opencode/e2-matrix.sh`. Raw outputs:
`/tmp/opencode/e2-eval-{native,orig,q8_0}.json`, `/tmp/opencode/e2-ibm-*.stdout`.

| Clip (lang, metric, denom) | native | orig | q8_0 | q8_0 Δ vs orig |
|---|---|---|---|---|
| 6930-75918-0000 (en, WER, 8) | 0.00% | 0.00% | 0.00% | +0.00pp, IDENTICAL |
| 6930-75918-0001 14.2 s long slice (en, WER, 43) | 0.00% | 0.00% | 0.00% | +0.00pp, IDENTICAL |
| 7902-96591-0000 (en, WER, 9) | 22.22% | 22.22% | 22.22% | +0.00pp, IDENTICAL |
| 7902-96591-0001 (en, WER, 9) | 0.00% | 0.00% | 0.00% | +0.00pp, IDENTICAL |
| common_voice_ja_20461197 (ja, CER, 16) | 6.25% | 6.25% | 6.25% | +0.00pp, IDENTICAL |
| **Aggregate WER (69 words)** | **2.90%** | **2.90%** | **2.90%** | **+0.00pp** |
| **Aggregate CER (16 chars)** | **6.25%** | **6.25%** | **6.25%** | **+0.00pp** |
| IBM multilingual sample | — | byte-identical to native | byte-identical to native/orig | — |

asr_wer en smoke agrees exactly (2.90% on all three). Zero empty outputs;
no repetitive/truncated outputs on review (longest outputs are complete
sentences; IBM sample is a complete EN+FR passage).

Reference-style notes (NOT model errors, identical across artifacts):
- `7902-96591-0000`: 2/9 = `I'M` vs ref `I AM` (contraction folding: 1 sub + 1 del).
  Reference style expands contractions; the model writes `I'm`.
- ja CER 1/16 = the reference file's stray trailing ASCII `.`
  (`...ます。.`); the hypothesis `日本語で書かれた本を読みます。` matches the
  D/E1 golden exactly.

Gate reading: native≡orig exact on all 5 ready clips + IBM sample → **PASS**.
Q8_0 Δ +0.00pp aggregate and on every per-language/long-audio slice, far
inside the +1pp / +2pp entry bar → **PASS, no tolerance adjustment**.
Verdict: `granite-speech-5.0-2b-q8_0.gguf` ships as `granite_speech_2b_q8_0`;
no higher-precision fallback needed. IBM `multilingual_sample.wav`
transcript (EN passage + FR passage) is byte-identical across all three
local artifacts.

Coverage limit (honest): measured en + ja only; fr/de/es/pt rows are
`pending` (no in-tree fixtures, none fetched). The six-language claim is NOT
validated — release notes must say en/ja measured. Long-audio >60 s and
multi-chunk splitting are NOT covered (longest fixture 14.2 s runs whole).

## 5. IBM official Q8_0 pair (independent comparison, pinned llama.cpp)

Attempt SUCCEEDED (not deferred). Pinned `llama.cpp@52975342`
(master 2026-11-17; `git ls-remote` confirms the SHA, shallow fetch + checkout
clean), CPU-only configure
(`-DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
-DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=OFF`), built target
`llama-mtmd-cli` (the E1-brief guess `llama-speech-cli` does not exist in this
revision; the speech path is `tools/mtmd`, binary `llama-mtmd-cli`).
IBM pair fetched with `hf download` at pinned revision
`945b4581e2b40cd9a3121de265fd8e387bdf782d` (sizes match E1 exactly:
1956155104 + 1159354752 B; blobs verified in
`/tmp/opencode/ibm-pair/.../blobs/`).

One tool adjustment recorded: the first run crashed in
`common_chat_templates_apply_legacy` with
`this custom template is not supported, try using --jinja`; re-ran with
`--jinja` (the tool's own prescribed flag, no model/prompt change) → clean
transcriptions. Prompts are IBM's verbatim card strings (card lines 291/318/415):
`transcribe the speech with proper punctuation and capitalization.` for en,
plus the language-ID variant `The speech is in Japanese. Transcribe it
exactly; do not translate. can you transcribe the speech into a written
format?` for ja (mirrors our `language=ja` hint).

| Clip | IBM pair output | vs our q8_0 (gate metric) |
|---|---|---|
| 6930-75918-0000 (en) | `Concord returned to its place amidst the tents,` | byte-identical, WER 0.00% |
| 6930-75918-0001 long (en) | same words, one comma differs (`flowers of which` vs ours `flowers, of which`) | WER 0.00% both (punct-only diff) |
| 7902-96591-0000 (en) | `I'm from the cutter lying off the coast.` | byte-identical, WER 22.22% |
| 7902-96591-0001 (en) | `Don't cry, he said. I was obliged to come.` | byte-identical, WER 0.00% |
| common_voice_ja_20461197 (ja) | `日本語で書かれた本を読みます。` | byte-identical, CER 6.25% |

Independent-implementation verdict: **AGREE** — zero WER/CER delta vs the
audio.cpp q8_0 candidate on every measured slice. (IBM pair uses the two-file
llama.cpp schema `weights + mmproj`, not the audio.cpp standalone package —
comparison only, no import; the audio.cpp release layout is unchanged.)

## 6. Runtime integration

Script `/tmp/opencode/e2-integration.sh`, outputs `/tmp/opencode/e2-int/`.
Binaries: `/tmp/opencode/granite-e1/bin/*` (E1 flags, rebuilt at E2 HEAD).

- **CLI native EN** (`01-cli-native-en.log`): golden
  `Concord returned to its place amidst the tents,` exit 0.
- **CLI GGUF q8_0 EN + embedded-override JA** (`02-cli-gguf.log`): both golden
  (`Concord…tents,` / `日本語で書かれた本を読みます。`), exit 0. Embedded-spec
  override path works on the release candidate.
- **Server** (native dir, port 8123, `03-*`): `/v1/models` lists `granite2b`
  loaded; JSON EN → golden, RTF 0.65; multipart EN → golden, RTF 0.54;
  multipart auto JA → golden, RTF 0.66.
- **Valid-after-failure** (`04-*`): `language=xx` → clean
  `granite_speech_2b unsupported language hint: 'xx'`; immediate next request
  on the same server → golden, RTF 0.58. No poisoned state.
- **Two sessions, different clips/options vs sequential** (`05-*`): q8_0 CLI
  short/none-mode EN + JA, then EN repeat → `REPEAT IDENTICAL` (cmp clean).
- **Shared-math regression** (`06-ctest-shared.log`): 
  `causal_decoder_chunked_prefill_test` PASS (0.04 s),
  `decoder_packed_projection_test` PASS,
  `gguf_tensor_source_test` PASS (covers the E1 shared-converter scalar fix),
  `granite5asr_ctc_timestamps_test` PASS (0.00 s).
  `model_spec_system_test` FAILS under ctest only with
  `model contract spec not found for family 'miotts'` — the known CWD-based
  dev-build spec-discovery caveat (D report §"Verification"): run from the
  repo root it prints `model_spec_system_test passed`, exit 0. Unrelated to
  g2b; same evidence E1 used. No threshold adjustment.
- **TurboCTC golden** (`test_granite5asr_golden_transcription` binary direct):
  `SKIP: ... requires model weights at '/home/brad/audio.cpp/granite5asr'`
  — no weights in repo, same SKIP precedent as C2/D. Timestamps covered above.
- **d_units** (`07-ctest-dunits.log`): `granite_speech_2b_d_units_test`
  **Passed 72.81 s** (golden en/ja, A/B/A incl. over-budget failure between,
  isolation, VAD planning, native-vs-f32, registry auto-detect).
- **Custom registration** (`08-loaders.txt`): `--list-loaders --json` yields
  exactly `{granite_speech_2b, granite5asr, marblenet_vad, silero_vad}`.
- **Full registration** (`08-full-configure.log`): configure-only
  `-DAUDIOCPP_MODEL_SET=full` OK; `granite_speech_2b` precedes `granite5asr`
  in selected+linked (order coupling preserved). Full compile not run
  (E1 precedent: configure check only).
- **Catalog sync** (`08-sync.log`): `ok: runtime loaders, model_specs, and
  model_manager_v2 are in sync` (one pre-existing warning:
  `voxcpm2.json package 'voxcpm2_safetensors' is format=safetensors`,
  unrelated).
- audio8/qwen3-affected callers: `test_audio8_asr_units` is NOT built in this
  custom composite (`audio8_asr` not in `AUDIOCPP_MODELS`; target guarded by
  `IN_LIST`, CMakeLists.txt:5489), so a follow-up aux build
  (`/tmp/opencode/granite-e2aux`, MODELS=granite_speech_2b,granite5asr,audio8_asr)
  was configured and built: **`test_audio8_asr_units` PASS** (exit 0),
  `engine_model_{audio8_asr,qwen3_asr,confucius4_r2t2}` all compile,
  `test_audio8_asr_golden_transcription` SKIP (no weights, expected).
  The shared decoder paths they exercise (chunked prefill, packed
  projections) are green in the E1 build too. No deferral.

Integration verdict: **PASS** (all runnable items green; two items
deferred/recorded, none failing).

## 7. Performance / memory (CPU-only)

Script `/tmp/opencode/e2-perf.sh`, outputs `/tmp/opencode/e2-perf/`.
Fixed `--backend cpu --threads 8`, release-candidate q8_0 GGUF unless noted.
Machine was quiet (no matrix/integration/build jobs) for all timing runs.

Cold single-shot (`/usr/bin/time -v`, includes process + model load):

| Run | Wall | Peak RSS | `prefill_ms` | `session.wall_ms` |
|---|---|---|---|---|
| q8_0 short (3.5 s) | 4.53 s | 4217520 kB (4.02 GiB) | 823 ms | 2061 ms |
| native short (3.5 s) | 4.49 s | 5521764 kB (5.27 GiB) | 815 ms | 2264 ms |
| q8_0 long (14.2 s) | 8.25 s | 4428084 kB (4.22 GiB) | 1125 ms | 6585 ms |

Warm server (q8_0, port 8124): 3 warm-ups + 20 alternating short/long requests
on the running server (`warm-rows.jsonl`):

- short (3.505 s audio): mean 1.84 s, stdev 0.13 s, range 1.68–2.04 s → RTF ~0.53
- long (14.225 s audio): mean 6.41 s, stdev 0.29 s, range 6.10–6.87 s → RTF ~0.45
- per-request text length constant (47 short / 236 long chars): no drift,
  no growth trend across the 20 calls → peak allocation settles.
- fresh-restart check: short 2202 ms (warm band 1676–2041 ms), long 6636 ms
  (warm band 6095–6871 ms), identical transcripts → reused == fresh sessions.

Warm q8 server process RSS (`/proc` after load + long request): VmHWM
4509368 kB = **4.30 GiB** (VmSize 20.6 GiB virtual: 8 GiB decoder weight-context
reservation + 4×1 GiB arenas + GGUF mapping; resident stays ~4.3 GiB).

Reading:
- First-token latency proxy: `prefill_ms` (time to first logits, greedy
  single-chunk decode) ≈ 0.8 s short / 1.1 s long. Decode tok/s is NOT emitted
  by any timing key; proxy from text length over (session − prefill):
  ~38 chars/s short, ~43 chars/s long. Recorded as proxy with this limitation.
- `min_vram_gb: 8` **confirmed**: worst measured resident 5.27 GiB (native
  single-shot, holds the 4.6 GB BF16 mapping) and 4.30 GiB warm q8 server —
  both well under 8 GB. No CUDA/Metal measured; backend tags stay CPU-only.
- End-to-end RTF on this box: ~0.45–0.65 (warm server 0.45 long / 0.53 short;
  cold single-shot incl. load 0.58 long). Well under the 1.0 community
  guideline on the declared hardware.

## 8. Gate table

| Gate | Bar | Result |
|---|---|---|
| native vs orig exact (stable-margin) | byte-identical transcripts | PASS (5/5 clips + IBM sample) |
| Q8_0 vs orig aggregate | ≤ +1pp | PASS (+0.00pp WER, +0.00pp CER) |
| Q8_0 vs orig per-language / long-audio | ≤ +2pp | PASS (max +0.00pp) |
| Empty/repetitive/truncated review | individual review | PASS (none observed) |
| IBM official Q8_0 pair (independent impl) | 0.00pp WER/CER delta vs candidate | PASS (4/5 byte-identical, long-EN punct-only diff) |
| Runtime integration | §6 all green | PASS (model_spec_system_test CWD caveat recorded, not a failure) |
| Perf/memory CPU | measured, no claimed-backend gap | PASS (RTF ~0.45–0.65, VmHWM 4.30 GiB, min_vram 8 confirmed) |
| fr/de/es/pt + >60 s + multilingual IBM refs | honest pending, not claimed | PENDING (documented, blocks six-language claim only) |
| Upload + flip | credentials-gated | NO PATH → QUALIFIED, experimental kept (§9) |

No threshold adjustment was needed; none is recorded.

## 9. Release decision + operator steps

**Decision: QUALIFIED, NOT RELEASED.** Quality gates pass, but the
experimental→community flip requires a real publish, and there is no publish
path in this environment:

- `HF_TOKEN` unset; `hf auth whoami` → user `eutychus` with **zero org
  memberships** (`whoami-v2`), i.e. no write access to
  `audio-cpp/audio.cpp-gguf`. Remote file still 404s (E1 evidence).
- Per the brief, no fake release was made: status stays `experimental`, the
  staged entry is unchanged, no flip commit exists.

Operator publish + flip (exact steps, for the final review):

```bash
# 1. From a credentialed checkout at this commit, upload the gated artifact:
hf upload audio-cpp/audio.cpp-gguf \
  build/granite-2b-handoff/dist/granite-speech-5.0-2b-q8_0.gguf \
  Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf
# 2. Record the published revision + re-sha256 the remote file, then in
#    model_specs/granite_speech_2b.json replace revision "main" with the
#    published digest, and in ONE commit flip together: status experimental->community,
#    runtime backend tags (cpu only, as measured), model-doc wording (drop
#    experimental notes), README/docs/asr.md rows, WebUI catalog entry.
# 3. Re-run BOTH managers against empty roots + catalog sync + CLI from the
#    installed artifact (E1 §"Managers" commands), update this report's §2
#    with the pinned revision, and confirm min_vram_gb: 8 against §7 RSS.
```

File: `build/granite-2b-handoff/dist/granite-speech-5.0-2b-q8_0.gguf`
(sha256 `0a8aee12e172b391c5e4d0460b5e53daeaa685d3d6f378f93bd906965a636e`),
remote name `Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf`.

## 10. Self-review / concerns

1. Coverage is the binding constraint on the release, not quality: en + ja
   measured, fr/de/es/pt pending, no >60 s audio, no multi-chunk splitting.
   The six-language catalog claim must wait for manifest rows to go `ready`.
2. `tools/asr_wer.py` reference style expands contractions while the model
   writes `I'm`/`Don't` — the only en errors (2/69 words) are this style gap
   plus reference punctuation, present identically in native/orig/q8_0/IBM.
   Not a gate issue; noted so nobody re-litigates the 2.90%.
3. `audiocpp_cli --version` reports `git: 123e3a85` (configure-time stamp;
   E2 rebuild did not re-configure). Source HEAD is `0b03d489`; the delta is
   the scalar-guard warning + shared unit test only. Recorded, no impact.
4. `model_spec_system_test` fails ONLY via ctest from the build dir
   (`miotts` CWD caveat, D-report precedent); passes from the repo root.
5. `test_audio8_asr_units` follow-up: GREEN in the aux build
   (`test_audio8_asr_units` PASS, qwen3/audio8/confucius engine libs compile,
   golden SKIP without weights). No deferral remains.
6. Server `--ui-management`/native-manager real `install` still 404s
   pre-publish (E1 evidence stands); the operator steps in §9 re-verify the
   published path after upload.
7. Decode tok/s has no emitting timing key — reported as a chars/s proxy with
   the limitation stated; first-token latency via `prefill_ms`.
8. No code changes in this task: 3 new files only (harness, manifest, this
   report). No inference-math or session-flow edits, so no re-gate
   of affected stages was triggered.
