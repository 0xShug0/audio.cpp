# PocketTTS small official reference

Run `reference.py` with `conda run --no-capture-output -n qwen3-tts python`.
No dependency installation is needed. The harness requires upstream revision
`4cc5753319ceddc5cbf70598d9011f1f22858ba0` and imports from `--upstream`.

Example (paths are supplied by the caller):

```sh
python tests/pocket_tts_small/reference.py --log --mode export \
  --upstream "$UPSTREAM" --config "$CONFIG" --voice "$ALBA_STATE" \
  --text 'On a quiet autumn morning, Alba walked beside the river, listening to birds and watching sunlight ripple across the water as the village slowly woke up.' \
  --output-dir "$OUTPUT" --device cuda --threads 8
```

Use the variant's own `reference.yaml` and `embeddings/alba.safetensors`.
Safetensors metadata includes `sample_rate`, `latent_layout`, `prepared_text`,
and `decoder_calls` (JSON). Exports contain `fixture.safetensors`, float `reference.wav`, `fixture.json`, and
`export.log`. Put all generated artifacts under an ignored output directory.
Only voice tensors are copied from weights; model checkpoints are not exported.

## Teacher-forced final waveform validation

This is official AR generation followed by C++ decoder replay, not independent
stochastic AR parity. No AR step or logits instrumentation is used.

* `latents`: concatenated **denormalized** decoder inputs `[1,T,32]`.
* `decoder.NNNN.latent`: exact per-call input slices. Replay these in order,
  resetting the C++ Mimi session whenever `decoder_calls[].reset` is true.
  Preserve state between calls otherwise. Do **not** renormalize these inputs.
* Apply the official linear fade-in (0 to 1 inclusive over 120 samples at 24 kHz)
  at each reset, truncated to that call's sample length. Concatenate outputs in
  call order. Call sample counts and reset/fade metadata are in `fixture.json`.
* `waveform`: final official concatenated mono float waveform, including fades.
* `text.NNNN.token_ids` and `prepared_text`: actual official conditioner inputs.
* `voice.*`: original variant-specific safetensors keys, unchanged except prefix.

The schema assumes default independent text chunks, without previous-chunk
teacher forcing or decoder warmup. Capture is at the normal decoder method and
text preparation boundaries. Decoder batching may vary with thread scheduling;
replay the exported boundaries instead of assuming one frame per call.

After C++ produces a float mono 24 kHz WAV (or safetensors `waveform`):

```sh
python tests/pocket_tts_small/reference.py --log --mode compare \
  --fixture "$OUTPUT/fixture.safetensors" --candidate "$CPP_WAVEFORM" \
  --output-dir "$COMPARISON_OUTPUT"
```

The comparison rejects unequal sample counts, nonfinite data, and cosine
similarity <= 0.99999. It saves `comparison.json`; export alone does not establish
C++ parity. Keep comparison reports outside Git.

The C++ replay defaults to bounded decoding with `--chunk-frames 4`. Test other
block sizes with the same option, or select `--full-decoder` for the full-utterance
comparison. Streaming replay preserves the exported reset boundaries.

## Separate performance mode

Use the generation arguments with `--mode performance` and repeat `--text` for
at least two distinct meaningful requests of 80+ characters. There is exactly
one excluded pass through the supplied requests, followed by one measured pass, with no
capture wrappers/hooks. Run exclusively: no concurrent measurement or build.
For CPU use `--device cpu --threads N` to fix Torch intra-op threads (interop=1).
CUDA synchronizes before timing, at streamed chunks, and on completion.

`performance.json` includes `session.wall_ms`, `ttfa_ms`, audio duration and
`rtf` (wall seconds / audio seconds). TTFA is measured at the first chunk yielded
by the official API. Upstream starts its producer thread and waits for AR
production before draining its result queue; this can make API TTFA much larger
than actual decoder first-audio readiness. No worker hooks are installed to
change that behavior. Model loading and voice loading are outside timing.
Instrumented export wall time is labeled correctness-only and is not a benchmark.

## Offline batch integration test

Build the model tests and run the real loader/session through the batch interface:

```sh
cmake --build build/debug --parallel --target pocket_tts_small_batch
build/debug/bin/pocket_tts_small_batch --model "$MODEL" --requests "$REQUESTS" \
  --backend cuda --threads 8 --batch-size 4 --out-dir "$OUTPUT" \
  --log --log-file "$LOG"
```

`REQUESTS` is a JSON array of objects with `text`, `voice_id`, and optional
`options` (for example `{"seed":42}`). Voices must be available in that model's
embeddings directory. Use `--sequential` for the ordinary single-request control.
Each process executes one excluded pass through the entire fixture followed by
one measured pass, using the same grouping. WAV files contain only measured outputs;
artifact writing is outside timing. After batch measurements, the test also runs
one ordinary request through the same session to check reuse of the larger
batch graph. This check is outside the measured pass. Run measurements sequentially.

`--session-options` accepts a JSON object of existing session controls for
capacity-policy checks. The generic per-row RoPE check belongs to the existing
`encoder_module_test`; it requires no model weights.

For streaming validation, add `--mode streaming --batch-size 1`. The test uses
the same application streaming runner as the CLI/server. It records
`session.ttfa_ms` at the first nonempty audio event, measured from request start
(after initial model/voice preparation), and checks that incremental audio events
concatenate exactly to the final WAV. Request indices and warmup/measured-pass
markers distinguish the first request from warmed requests. WAV writing is
outside request timing. Streaming batching is not supported.
