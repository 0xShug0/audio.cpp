# Parakeet TDT numerical parity harness

A heavier, more granular companion to `test_golden_transcription.cpp`
(the "Tier 1" golden-transcription regression test): instead of asserting
the final decoded text matches, this compares intermediate encoder
activations directly against a real NeMo reference run, stage by stage.

This exists because greedy/argmax decoding can be robust to a fair amount
of numerical drift without the final transcription changing — the golden
transcription test alone did **not** catch a real bug (the depthwise conv's
folded batch-norm bias being silently dropped — `use_bias=false` where it
needed to be `true`) that measurably corrupted every encoder layer's output,
because the greedy decode happened to pick the same tokens anyway on the
one test clip. This harness is what actually found and confirmed that bug.

## What it checks

Three stages, run through two independent code paths (real production
entry points where possible, an isolated single-layer build where not — see
the comment at the top of `dump_cpp_reference.cpp` for why intermediate
production-graph debug taps are deliberately avoided here):

1. `mel_features` — `ParakeetFrontend::extract(...)`, straight through the
   real, unmodified frontend entry point.
2. `layer_0` — `build_encoder_layer(...)` (the exported, real production
   per-layer function — see `encoder.h`), invoked once in a small isolated
   graph, fed NeMo's own captured `pre_encode`/`pos_emb` for layer 0 and the
   real layer-0 weights.
3. `enc_out` — `ParakeetEncoderRuntime::encode(...)`, straight through the
   real, unmodified full-encoder entry point (all 24 layers + positional
   encoding, one large production graph).

`compare_parity.py` computes cosine similarity and a relative-std ratio for
each and fails loudly (naming the stage) if anything drops out of
tolerance. `enc_out`'s tolerance is deliberately looser than the other two
— see the comment above its threshold in `compare_parity.py` for why (short
version: float32 summation order isn't required to match bit-for-bit
between a small isolated single-layer graph and the real ~2M-node
24-layer production graph, and per-op rounding differences compound
through 24 layers of softmax/normalization even when the underlying math is
identical — verified directly by chaining 24 isolated single-layer builds
end to end, which stays at cosine >= 0.999999 the entire way through).

## Setup

### C++ side (no extra dependencies beyond the normal build)

```
cmake --build build/<preset> --target parakeet_parity_dump
```

### Python/NeMo side

NeMo's dependency chain is heavy and somewhat fragile outside a normal
desktop Python environment; a dedicated venv is recommended:

```
python3 -m venv /tmp/parakeet_parity_venv
/tmp/parakeet_parity_venv/bin/pip install torch nemo_toolkit[asr] librosa numpy pysqlite3-binary
```

If your system Python lacks a usable `sqlite3` module (common in minimal
containers — NeMo's dependency chain pulls it in transitively via
IPython's history feature even though nothing here needs it), install
`pysqlite3-binary` and let `dump_nemo_reference.py`'s
`sys.modules['sqlite3'] = __import__('pysqlite3')` shim handle it — it
already does this automatically when a real `sqlite3` import fails.

The first run downloads the `nvidia/parakeet-tdt-0.6b-v3` NeMo checkpoint
(~2.5GB) to the standard HuggingFace cache.

## Running it

```
# 1. NeMo reference dump
/tmp/parakeet_parity_venv/bin/python3 tests/parakeet_tdt/parity/dump_nemo_reference.py \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --output-dir /tmp/parakeet_nemo_dump

# 2. C++ dump (accepts the installed safetensors directory or standalone GGUF).
#    --matmul-weight-type defaults to "native" (F32); pass f16/bf16/q8_0 to
#    numerically quantify the accuracy cost of a reduced-precision weight
#    storage type against the same NeMo reference. --flash-attention 1 swaps
#    the encoder's relative-position self-attention onto
#    ggml_flash_attn_ext_with_bias_mask instead of the default explicit
#    QK^T + soft_max_ext path — see the Performance section in
#    docs/community_models/parakeet_tdt.md for measured numbers (and why it's
#    not the default despite being numerically validated: it measured
#    slower, not faster, on the hardware tested).
build/<preset>/bin/parakeet_parity_dump \
    --model models/parakeet-tdt-0.6b-v3 \
    --audio tests/parakeet_tdt/assets/2086-149220-0033.wav \
    --nemo-dir /tmp/parakeet_nemo_dump \
    --output-dir /tmp/parakeet_cpp_dump \
    --matmul-weight-type native \
    --backend cpu   # or: --backend cuda / vulkan

# 3. Compare (numpy only, no NeMo/torch needed — can run anywhere the two
#    dump directories are available, e.g. copied off a NeMo-enabled machine)
python3 tests/parakeet_tdt/parity/compare_parity.py \
    --nemo-dir /tmp/parakeet_nemo_dump --cpp-dir /tmp/parakeet_cpp_dump
```

### Cross-backend comparison (no NeMo needed)

`--backend` also makes this harness answer a second question the NeMo
comparison cannot: does an accelerator still agree with the CPU? Dump twice
and diff the two `enc_out.npy` files directly. This is the check that catches
a change which is correct on CPU but silently wrong on a GPU — a strided view
that a CPU op materializes and a GPU kernel reads differently, for example.

Expect agreement to float32 accumulation noise, not bit-exactness: different
kernels sum in different orders. Currently CPU vs CUDA is cosine 0.99999988,
relative RMS 3.5e-06, with bit-identical mel features.

Expected output on a healthy build:

```
PASS mel_features: cosine=1.000000 (>= 0.999) ...
PASS layer_0: cosine=1.000000 (>= 0.999) ...
PASS enc_out: cosine=0.97xxxx (>= 0.97) ...
```

## Known limitation

This only covers the offline full-context encoder path with one test clip. It
does not exercise the decoder/joint network numerically; the golden and
streaming transcription tests cover those paths end to end, but not stage by
stage. The implemented streaming mode deliberately re-encodes bounded
bidirectional windows, so its encoder output is not expected to concatenate
into the utterance-wide `enc_out` tensor frame-for-frame. A cache-aware parity
test would require a checkpoint trained for chunked-limited attention; this
checkpoint is not one.
