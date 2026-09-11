# Parakeet TDT multilingual weight-type accuracy harness

Answers one question the other two test tiers cannot: **does changing
`parakeet_tdt.matmul_weight_type` actually change what the model transcribes,
across the languages it claims to support?**

The existing tiers are both too narrow for that:

- `test_golden_transcription.cpp` asserts the decoded text of **one** English
  clip. Greedy/argmax decoding is robust to a lot of numerical drift, so this
  passes under changes that measurably move the encoder — the parity harness's
  README documents a real bug it failed to catch.
- `parity/` compares intermediate encoder activations against NeMo, which is
  the right tool for "is the math right", but it also runs on one clip and
  reports cosine similarity, not anything a user experiences.

Cosine similarity is not a transcription. `enc_out` cosine of 0.9964 for q8_0
sounds harmless; what it actually costs is measured here.

## Why this matters

The completed run covers 120 FLEURS clips across 24 languages. `q8_0`—about
1.79x faster than F32 on the tested CPU—matches the F32 transcript exactly on
91.7% of clips. Its aggregate absolute WER is effectively flat on this sample
(`0.1331` versus F32's `0.1338`), but the 8.3% transcript churn proves that it
is not byte-for-byte interchangeable with F32. On the single English golden
clip it is identical, which is exactly the trap: one clip in one language
cannot characterize a quantization tradeoff.

## Corpus

[FLEURS](https://huggingface.co/datasets/google/fleurs) test split, which is
read speech with human reference transcripts and per-language configs.
`fetch_fleurs.py` pulls N clips for each of the 24 European languages
parakeet-tdt-0.6b-v3 supports that FLEURS covers (it has no Maltese), decodes
them to the 16 kHz mono WAV the engine expects, and writes a manifest with the
reference transcript.

Each FLEURS test parquet is 150–400 MB and only a handful of clips are kept
from each, so the script deletes every parquet immediately after extracting
and records per-language progress in `done_<lang>.json`, making it resumable
and bounded in disk use. Expect the download to take a while; it is entirely
network-bound.

```bash
pip install pyarrow soundfile numpy
python3 tests/parakeet_tdt/multilingual/fetch_fleurs.py 5   # 5 clips/language
```

## Running the comparison

```bash
cmake --build build/<preset> --target parakeet_warm_bench
python3 tests/parakeet_tdt/multilingual/compare_weight_types.py native f16 bf16 q8_0
```

All clips for a given weight type go through a **single** process via
`--audio-sequence`, so the model and its quantized weights are built once
rather than once per clip. The first named type is the reference everything
else is compared against, so put `native` first.

Environment overrides: `FLEURS_DIR` (corpus location, default `tmp/fleurs`),
`PARAKEET_BENCH` (bench binary), `PARAKEET_THREADS`.

The checked-in results under `results/` were produced from the upstream
safetensors package with each storage type selected at session creation. The
standalone-GGUF regression checks cover package loading and the golden paths;
they are not a second 120-clip corpus run.

## Reading the output

Three numbers per weight type:

- **exact vs native** — fraction of clips whose decoded text is byte-identical
  to the f32 run. This is the number that matters for "is this change free".
- **WER vs native** — how far the text actually moved. Distinguishes "differs
  on a few clips by one word" from "differs badly".
- **WER vs FLEURS** — absolute quality against the human reference, so a
  weight type that merely agrees with f32 while both are bad cannot hide, and
  so you can see whether a difference from f32 is actually *worse* or just
  *different*.

Then a per-language WER breakdown, because quantization damage is not spread
evenly — it concentrates in particular languages, and an aggregate hides that.

WER uses word-level Levenshtein over NFKC-normalized, lowercased,
punctuation-stripped text. That normalization is deliberately crude: it is
meant for comparing weight types against each other on identical audio, not
for publishing absolute WER figures comparable to other papers' numbers.
