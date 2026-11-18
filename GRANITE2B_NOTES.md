# IBM Granite Speech 5.0 2B — audio.cpp porting notes

Family `granite_speech_2b`: offline multilingual ASR (en/fr/de/es/pt/ja),
keyword biasing, and speech translation (en→fr/de/es/pt/it, ja→en) through
the CLI and legacy server. Status: **experimental, QUALIFIED NOT RELEASED**
(see §9). Upstream checkpoint `ibm-granite/granite-speech-5.0-2b`,
revision `ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7`;
`model.safetensors` SHA-256 `41512698…bf6d88`.
Work spans commits `74ead48e..339f47a7` on top of upstream `a6e32214`.

Pinned reference: Transformers 4.57.0, torch 2.8.0 CPU, FP32,
`model.eval()`, `attn_implementation="eager"`, 8 threads.
Smoke prompt: `<|audio|>transcribe the speech with proper punctuation and
capitalization.` Reference tooling: `tests/granite_speech_2b/reference.py`
(trace generator), `compare_trace.py` (candidate gate),
`granite_speech_2b_parity_probe` (C++ stage dumps, same trace keys).

## 1. Architecture map

| IBM component | audio.cpp home | Notes |
|---|---|---|
| Conformer encoder (16 layers, hidden 1024) | `src/community_models/granite_speech_2b/encoder.{cpp,h}` | Per-block graphs; pre-norm Macaron FFN (0.5 residual), Shaw attention in 200-frame blocks, full-sequence conv (GLU, symmetric pad 7, BN eval stats, SiLU), block-8 CTC feedback `h += out_mid(softmax(out(h)))`, raw block-16 states to projector |
| Q-Former projector (2 layers, 3 queries) | `projector.{cpp,h}` | Right-zero-pad to W·15 rows, 15-row windows, learned queries, post-residual LayerNorms (eps 1e-12), absent mask (pads participate), window-major/query-major flatten, final 1024→2048 |
| Granite LLM (40 layers, GQA 4:1, untied LM head) | `decoder.{cpp,h}` + shared `GreedyCausalDecoderRuntime::generate(..., reuse_graphs=true)` | Separate Q/K/V, bias-free, no Q/K norm, RMS eps 1e-5, SiLU MLP; one complete prompt per call, zero retained prefix |
| Frontend (80-bin logmel, 160-wide stacked) | `frontend.{cpp,h}` + shared `NemoMelFrontend` | Centered periodic Hann, reflect pad, HTK, no mel norm, 512/400/160 FFT/win/hop, power spectrum, `DensePowerLog10` floor 1e-10; per-clip max floor over ALL raw frames then `(v/4)+1`; adjacent-pair stacking discarding odd trailing row (not the shared stack path) |
| Tokenizer + prompt | `prompt.{cpp,h}`, `LlamaBpeTokenizer` preset Llama3 | Whole-string encode, whole-sequence decode; single user turn, marker `100352`, BOS/EOS `100257`, pad `100256`; template byte-pinned incl. space before assistant cue |
| Session / loader / spec | `session.{cpp,h}`, `model_specs/granite_speech_2b.json` | `RuntimeSessionBase` + `IOfflineVoiceTaskSession`, `make_granite_speech_2b_loader`, spec v1 `experimental`, `packages: []` until release |

Length contract (checked integer arithmetic, asserted per stage):
`M = floor(N/160)+1` raw mel frames, `T = floor(M/2)` encoder steps,
`W = ceil(T/15)` projector windows, `A = 3·W` audio placeholders (2048-d).
Empty valid buffer → empty transcript; nonempty clip ≤256 post-resample
samples → clean input error (chunk planner merges such tails).

## 2. Numerical contract (do not drift)

- Embedding scale **×12 after** substituting projected audio rows, exactly
  once per generated-token lookup, at the shared embedded-prefill /
  token-lookup boundary.
- Attention scale **0.0078125** to Q·K pre-softmax on every path
  (plain/grouped/flash/static); `1/sqrt(128)` stays the default for others.
- Layer residual **`residual + 0.22·branch`** after attention out-proj and
  after MLP down-proj; preserve accumulation/cast ordering.
- RoPE: 128ch NeoX half-pair, base 10000, factor 4, orig 4096, limit 16384,
  YaRN beta 32/1. **Pass `attn_factor = 1.0`: ggml derives the ~1.13863
  magnitude from `freq_scale` when `ext_factor != 0`; passing it would
  square it** (observed 0.1578 error at position 0).
- Logits: final projection **÷ 8**. Note: the reference hook reads
  `lm_head` output **pre-÷8** (`last_pos_logits == 8·forced_step0`
  bitwise); the probe emits the exact ×8 inverse.
- Gate tolerances: frontend `max_abs 2e-4`; encoder/projector
  normalized RMS `1e-3`; logits `max_abs 2e-2` + normalized RMS `1e-3`.
  Nonfinite values fail. Reference top-1 margin rule: match where the
  reference margin exceeds twice the logit tolerance.
- Language hints: `en/fr/de/es/pt/ja → English/French/German/Spanish/
  Portuguese/Japanese`; blank = auto (no hint). Keyword instruction:
  `transcribe the speech to text. Keywords: k1, k2`. Translation:
  `translate the speech to {T} with proper punctuation and
  capitalization.` (punctuated AST row; the raw row produced all-caps
  output and was rejected). Target codes add `it/zh → Italian/Mandarin`.

## 3. Gotchas encountered (with fixes)

1. **Projector graph node budget (fixed in `339f47a7`).** The per-window
   Q-Former loop added O(100+) nodes/window against a fixed 65536-node
   budget, aborting (`GGML_ASSERT n_nodes < size`) on any span above
   ~60 s — which also made the 120 s default chunk size unusable (it had
   never been exercised; only ≤64 s spans were gated). Budget now scales:
   `8192 + 1024·W`. Regression test: synthetic T=9000 projector run
   (red = same assert, green = pass).
2. **Reference `last_pos_logits` is pre-÷8** (hook placement). Probe emits
   the exact fp32 ×8 inverse; units gate the inverse against the reference
   stage and scaled logits against `forced_step0`.
3. **Reference layer-8 pre/post dumps were byte-identical** (hook saved an
   alias the in-place `+=` feedback then mutated). Fixed with a clone-at-
   hook + re-freeze; probe emits true pre under both names only until such
   a re-freeze — check `task-B2` notes if touching stage dumps.
4. **Rope-dict configs**: `text_config.rope_scaling` may be a plain dict;
   index it defensively — missing keys raise `RuntimeError`, never
   `KeyError` (reference.py `dump_model_config`).
5. **HF mask**: `trace.json` carries no mask array; `input_ids` equality +
   `count == A` + contiguity was accepted as mask agreement (same marker
   tokenization feeds both). C2 generation parity is the tripwire.
6. **Template**: byte-pin the chat template including the space before the
   assistant cue; loader verifies the single-turn shape at load (fail
   closed), not at a call site.
7. **Chunking**: 120 s per-pass cap; tiny tails (≤256 samples) merge into a
   neighbor; last-two rebalance may overshoot the cap by ≤~128 samples
   (comment states the real bound). Zero overlap; VAD-no-speech → empty
   transcript. `merge_tiny_tails` single-tiny-non-VAD input throws.
8. **Registry order coupling**: `granite5asr::can_load` also matches
   `model_type == "granite_speech"`, so the `granite_speech_2b` CMake block
   must precede `granite5asr` for native auto-detect (narrowing granite5asr
   is an other-family change; misroute fails loudly at load).
9. **GGUF installed-dir claiming**: the fallback now rejects directories
   holding native weight files (`*.safetensors`/`*.bin`), so a foreign
   native checkpoint next to a stray GGUF is not claimed (red-green
   covered with a symlinked dist GGUF).
10. **Multipart translation**: the single-file OpenAI-compat shape has no
    options channel, so `target_language` there is a loud 400 pointing at
    JSON `options.target_language` (both serial and parallel handlers,
    lifecycle-pinned). Keywords travel via `prompt|text` on all transports.
11. **Q8_0 conversion**: keep `encoder.*`, `projector.*`, `embed_tokens`,
    `lm_head` in BF16 (quoted `--keep-type`); rank-0 scalars keep native
    storage with a stderr warning (shared converter guard + unit test).
12. **Dev-build spec discovery is CWD-based** (`discover_external_model_spec`
    walks up from CWD): run CLI/tests from the repo root, or install specs,
    or pass `--model-spec-override`. `model_spec_system_test` fails from a
    build dir for unrelated families (pre-existing `miotts` case) and
    passes from the root.
13. **CUDA builds with two toolkits**: if CMake picks a stale distro
    `nvcc` instead of the intended toolkit, the CUDA compiler-ID check
    fails. Point the build at the intended toolkit explicitly (see also
    `docs/build/linux.md`):
    `-DCMAKE_CUDA_COMPILER=<cuda-home>/bin/nvcc
    -DCUDAToolkit_ROOT=<cuda-home>`.
14. **Stale-binary discipline**: always rebuild the exact binary under test;
    two incidents in this history (F server check, projector-fix probe)
    were scratch-dir staleness, not product bugs. Prefer a clean-room
    build dir for release-grade numbers.

## 4. Benchmarks — Monte Cristo ch.1, LibriVox (mp3 → 16 kHz mono WAV)

Excerpt 180.0 s (ch.1 start, incl. LibriVox disclaimer) and full chapter
1179.4 s. Machine: i9-12900K, 31 GB RAM; RTX 3080 12 GB; CUDA 12.9 build.
Wall-clock end-to-end (incl. model load), single runs except where noted.

| Implementation | Clip | Wall | RTF (wall ÷ audio) |
|---|---|---|---|
| Pinned reference (Transformers 4.57, FP32 CPU, eager) | 180 s | 110.3 s (load 1.2 + generate 109.0) | 0.612 (0.606 gen-only) |
| C++ CPU, native weights | 180 s | 135.8 s | 0.754 |
| C++ CPU, Q8_0 GGUF | 180 s | 132.5 s | 0.736 |
| C++ GPU (RTX 3080), native | 180 s | 11.33 s | 0.063 |
| C++ GPU, Q8_0 | 180 s | 11.56 s / 10.50 s repeat | ~0.061 |
| C++ CPU native, chunked | full 1179 s | 993.7 s | 0.843 |
| C++ GPU native, chunked | full | 74.1 s | 0.063 |
| C++ GPU Q8_0, chunked | full | 67.4 s / 68.4 s repeat | ~0.058 |

GPU ≈ 12× CPU on the excerpt, ≈13.4× on the full chapter. C++ CPU runs
≈0.8× reference speed. VRAM peak: native 10.4 GB, Q8_0 9.0 GB (incl.
~2.6 GB desktop baseline). Reference truncated at its 512-token budget;
C++ chunked runs transcribe to the end (disclaimer → CHAPTER II).

Transcript parity (trailing-newline aside): 120 s segment C++ == reference
byte-identical; 180 s stages rel_rms ~1e-6 with matching greedy prefixes;
chunk-1 output deterministic across runs. 180 s full-text differs only
because CLI chunks (120+60 s, 512 budget each) vs single-prompt reference
(512 total). Cross-backend: native CPU == GPU except one `"` token pair
in 17.5k chars; Q8_0 CPU-vs-GPU differs by one char (`F`/`f`).
Native-vs-Q8_0 flips (e.g. `KRi`→`KRI`) are isolated single tokens inside
the E2 quality gates — see §5.

## 5. Quality summary

Eval harness (`eval_metrics.py`: pinned NFKC, WER space-delimited / CER
Japanese, separate punctuation/case pass) + manifest (`eval_manifest.jsonl`).
Native ≡ orig ≡ Q8_0 byte-identical on all ready clips: WER 2.90%, CER
6.25%, Δ +0.00pp vs orig (gates: +1pp aggregate / +2pp per-slice; no
adjustment needed). IBM official Q8_0 pair (pinned llama.cpp): zero
WER/CER delta on 4/5 slices, one comma-only long-EN diff. Runtime
integration green (CLI native/embedded GGUF, server JSON + multipart +
auto-JA + `xx` reject, valid-after-failure, two-session repeat,
TurboCTC golden + timestamps, chunked-prefill + packed-projection
regressions). CPU perf: RTF ~0.45–0.65 short/long, flat over 20 alternating
calls, reused == fresh sessions. Coverage honesty: en+ja measured (+63.9 s
synthetic chunk-requiring slice, byte-identical); fr/de/es/pt natural
audio, >60 s natural recordings, and independent human translation refs
are pending — advertised set is en→fr/de/es/pt/it + ja→en only.
Keyword biasing: P100/R75/F1 85.7, zero insertions, budget documented
(~3.8 tokens/term). Translation corpus chrF 66.0 / BLEU 47.7, 10/12
byte-identical C++≡Python parity; en→zh diverges at token 1 and stays
unadvertised pending root-cause.

## 6. Release state

QUALIFIED, NOT RELEASED. All gates pass, but no publish credentials exist
in the build environment, so status stays `experimental` with a staged
`granite_speech_2b_q8_0` package entry (exact remote filename
`Granite-Speech-5.0-2B-GGUF/granite-speech-5.0-2b-q8_0.gguf`, E2-gated
digest `0a8aee12…a636e`). Operator upload + flip steps (fetch-back
`sha256sum` verify, pin revision, one flip commit: status, backend tags,
docs, catalog) are in `docs/models/granite_speech_2b_validation.md` §9.
Do not flip to community without independent human refs for translation
and a re-run of the manager install path against the published bytes.

## 7. Reproduce (from repo root)

```bash
export GRANITE2B_WORK="$PWD/build/granite-2b-handoff" GRANITE2B_HF="$GRANITE2B_WORK/hf"
# reference env: python3 -m venv + torch 2.8.0 (CPU index) + transformers 4.57.0
python tests/granite_speech_2b/reference.py --model "$GRANITE2B_HF" \
  --audio <wav> --out-dir "$GRANITE2B_WORK/trace/x"
cmake -S . -B <build-dir> -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=granite_speech_2b,granite5asr \
  -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON   # + -DENGINE_ENABLE_CUDA=ON (+ toolchain flags, §3.13) for GPU
cmake --build <build-dir> --parallel 24 --target audiocpp_cli audiocpp_server audiocpp_gguf \
  granite_speech_2b_parity_probe granite_speech_2b_b1_units_test granite_speech_2b_d_units_test \
  granite_speech_2b_f_units_test server_parallel_lifecycle_test
ctest --test-dir <build-dir> -R 'granite_speech_2b|granite5asr_ctc|server_parallel'
<build-dir>/bin/granite_speech_2b_parity_probe --model "$GRANITE2B_HF" \
  --audio <wav> --backend cpu --threads 8 --through generate --out-dir "$GRANITE2B_WORK/trace/y"
python tests/granite_speech_2b/compare_trace.py --reference "$GRANITE2B_WORK/trace/x" --candidate "$GRANITE2B_WORK/trace/y"
```

## 8. File map (new, excluding per-task tests/fixtures)

`src/community_models/granite_speech_2b/{assets,frontend,encoder,projector,decoder,prompt,session}.cpp`,
`include/engine/community_models/granite_speech_2b/*.h`,
`model_specs/granite_speech_2b.json`, `docs/models/{granite_speech_2b,granite_speech_2b_validation}.md`,
`tests/granite_speech_2b/{reference,compare_trace,dump_c2_fixtures,eval_metrics,make_{synth,keyword}_fixture}.py`,
shared-module opt-ins in `framework/modules/{transformers/{decoder,causal_decoder{,_runtime}},positional_modules}`,
converter scalar guard in `framework/assets/tensor_source.cpp`, multipart
400 in `app/server/{runtime,parallel_runtime}.cpp`.
