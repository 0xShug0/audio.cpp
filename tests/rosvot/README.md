# ROSVOT validation

The port produces offline MIDI and JSON note events. It does not transcribe lyrics
or implement streaming. See [usage](../../docs/models/rosvot.md) for conversion
and inference commands. Conversion keeps the original F32 parameters of ROSVOT,
RWBD and RMVPE in one spec-v1 GGUF.

Reference code: `RickyL-2000/ROSVOT` at
`3c8332bf43adae35f6e4d64971862f2f6139b310`. Checkpoints:
`Soul-AILab/SoulX-Singer-Preprocess`, revision
`83dc50289d22a81b1e9998f5b9e111aef7c1fdcd`, directory `rosvot`.

## Component parity

Generate a frozen-input fixture from the official implementation:

```bash
python tests/rosvot/reference.py \
  --source models/ROSVOT-upstream --checkpoint models/ROSVOT/rosvot \
  --audio singing.wav --output build/logs/rosvot/reference \
  --backend cpu --controlled --log build/logs/rosvot/reference.log
```

Build the manual model test with `ENGINE_BUILD_MODEL_TESTS=ON` and
`AUDIOCPP_MODELS=rosvot`, then run:

```bash
build/debug/bin/rosvot_components \
  --model models/ROSVOT-GGUF/rosvot-f32.gguf \
  --reference build/logs/rosvot/reference/reference.safetensors \
  --backend cpu --log --log-file build/logs/rosvot/components.log
```

The default cosine gate is 0.99999. This tests RWBD logits, ROSVOT frame
features, boundary logits, pooling attention, repeated execution and both
propagated and isolated pitch decoding. Test-only matmul precision controls do
not change production inference.

On the validated CUDA stack, pitch-logit cosine is approximately 0.9999885 even
with frozen Python frame features. This backend drift is accepted explicitly
with `--min-pitch-cosine 0.99998`; all other output gates remain unchanged.
CPU passes the strict default. Small logit differences can move note boundaries,
including splitting consecutive notes of the same pitch.

## Public lifecycle and memory

```bash
python tests/rosvot/run_validation.py \
  --model models/ROSVOT-GGUF/rosvot-f32.gguf \
  --audio singing.wav --other-audio other-singing.wav \
  --backend cuda --output build/logs/rosvot/lifecycle --log
```

This runs requests sequentially through one server session, verifies repeated
and restored outputs, checks invalid options and recovery, and samples process
RSS/VRAM. `--benchmark-only` runs one warmup and two measured requests;
`--errors-only` limits execution to option rejection and recovery. Use eight
threads and the same backend/precision for reference comparisons.

For shared RMVPE changes, the same component tester accepts `--pitch-inputs`
(a JSON array of WAV paths), `--pitch-output` (JSON pitch tracks) and optionally
`--pitch-reference` (an unchanged-component output). It reuses one extractor
across all inputs and requires exact output equality, with no cosine tolerance.
Include silence, short clips, recurrent chunk boundaries, changed audio and
restoration of an earlier input. CUDA long-input checks should cross the
4,096-frame feature-window boundary.

Set `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` for both native and Python
performance measurements. The server's `threads` setting alone does not limit
all frontend OpenMP loops. For sequential A/B comparisons, the server harness
accepts `--server-exe` and `--measured-requests`; each benchmark block includes
two unmeasured warmups by default and at least two measured requests.
Use `--warmup-requests 2 --measured-requests 3` for the recorded comparison.
The Python reference accepts the same two arguments.

The component test selects the optional native CUDA implementation with
`--cuda-native true`. Without it, the unchanged GGML execution mode is tested.
`--native-cell true` independently compares the native GRU cell with the GGML
equations; neither switch changes production task options.
