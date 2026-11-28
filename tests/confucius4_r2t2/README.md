# R2T2 ASR verification

These files verify the `confucius4_r2t2` family against the macOS MPS reference
implementation in the Confucius4-R2T2 repository (see `docs/models/r2t2.md`).

## Files

| File | Purpose |
|---|---|
| `make_golden.py` | Runs the Python reference (`R2T2ASRModel` on MPS) for an audio file and writes offline text plus per-chunk streaming `fixed_text`, `raw_decoded`, and `text` to a golden JSON. |
| `compare.py` | Runs `audiocpp_cli` offline and streaming with `--log-file`, parses the per-chunk trace, and diffs everything against a golden. |
| `test_confucius4_r2t2_transcription.cpp` | Repo-native smoke test: offline + final streaming transcripts against the golden, plus a check that Auto-language deltas form a nonempty prefix of the expected transcript for `assets/resources/sample_16k.wav`. Skips with exit code 125 when the model or audio is missing. Also exposes `--encode <text>` to dump token ids for tokenizer diffing. |
| `golden*.json` | Recorded reference outputs. |

## Goldens

| Golden | Audio | Reference run |
|---|---|---|
| `golden.json` | upstream `resources/test.wav` (Chinese) | MPS fp16, auto language, 320 ms chunks |
| `golden_zh.json` | same | MPS, forced `--language Chinese` |
| `golden_sample16k.json` | repo `assets/resources/sample_16k.wav` (English) | MPS fp16, auto language |
| `golden_sample16k_bf16.json` | same | MPS bf16 (isolates reference dtype) |

Streaming parameters are fixed across goldens so comparisons are deterministic:
`chunk_size_ms=320`, `unfixed_chunk_num=2`, `unfixed_token_num=5`,
`max_new_tokens=32`, `rollback_punctuation=false`.

## Regenerating a golden

The generator defaults to the shared English `assets/resources/sample_16k.wav`,
resolved relative to the script's repository, independently of the working directory.
Both `--model_path` and `--out` are required. Use a new output filename to preserve
recorded goldens. Run from the upstream Confucius4-R2T2 checkout on macOS with
its torch/MPS dependencies (the upstream example uses `uv`):

```bash
cd /path/to/Confucius4-R2T2
PYTHONPATH=. uv run python /path/to/audio.cpp/tests/confucius4_r2t2/make_golden.py \
  --model_path /path/to/checkpoints/r2t2 \
  --out /path/to/generated_sample16k.json \
  > /path/to/generated_sample16k.log 2>&1
```

`golden.json` and `golden_zh.json` require the optional external upstream Chinese
`resources/test.wav`: pass its path explicitly with `--audio` (and
`--language Chinese` for the forced-language case), with a separate output file.
There is no shared Chinese fixture; the existing `a` and `c` clips are English
and cannot substitute for that input.

Existing golden `audio` fields, including contributor-local paths, record historical
provenance, not live input paths. These recorded runs have not been regenerated
or relabeled. Future generation records `audio_sha256`, the SHA256 of the original
WAV bytes before preprocessing, and uses a repo-relative `audio` identity for repo
inputs or an absolute path for external inputs. Reference preprocessing and decoding
math are unchanged. Syntax and help checks do not validate MPS inference or parity;
regeneration still requires the external upstream implementation, checkpoint, and Mac/MPS.

## Comparing

```bash
cd /path/to/audio.cpp
python3 tests/confucius4_r2t2/compare.py \
  --cli build/macos-metal-release/bin/audiocpp_cli \
  --model models/Confucius4-R2T2 \
  --audio assets/resources/sample_16k.wav \
  --golden tests/confucius4_r2t2/golden_sample16k.json \
  --backend metal
```

The comparison checks four things: the offline transcript, the committed delta
stream, every per-chunk committed `fixed_text`, and the final streaming
transcript. Metadata-only rollback prefixes are filtered from the unmodified
reference goldens before comparing committed text; `language` fragments must
not appear in the emitted deltas. See the results table in
`docs/models/r2t2.md` for what is exact and the two documented internal
(non-observable) differences on the English clip.

The same binary can dump the family tokenizer for diffing against Hugging Face:

```bash
build/macos-metal-release/bin/test_confucius4_r2t2_transcription \
  --encode "language English<asr_text>Some text 22,500"
```

## Graph reuse regression

```bash
build/macos-metal-release/bin/test_confucius4_r2t2_graph_reuse --backend metal
```

This model-backed test compares actual encoder capacities after growth and
shrink, checks the Metal attention precision boundary, and compares exact
encoder + exact decoder against reused encoder + reused decoder on real audio
prefixes (automatic language and forced English). Decoder-only synthetic
injection tests remain separate. The embedding RMSE bound is a drift alarm;
observable token equality is checked independently. The test defaults to CPU
and accepts `--model` for a local safetensors directory or GGUF file.
