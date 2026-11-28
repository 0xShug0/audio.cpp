# Granite Speech 2B parity tests

Model-specific tests are built only with `ENGINE_BUILD_MODEL_TESTS=ON` and
`granite_speech5` included in `AUDIOCPP_MODELS`. Synthetic mathematical,
shape, prompt-injection, error and lifecycle checks do not need checkpoints.

Reference WAVs, token dumps and logits are not stored in this repository.
Generate references in an external working directory using the pinned Python
implementation, then provide their location explicitly. Missing files in a
supplied reference directory fail the corresponding assertions; omitting the
reference directory reports that checkpoint-reference checks were not run.

## Reference environment

Use checkpoint revision `ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7`,
Transformers 4.57.0, FP32 eager execution with eight threads. The original
reference used PyTorch 2.8.0; the cleanup checks used PyTorch 2.11.0+cu128.
The scripts read local checkpoints and do not download weights.

```bash
# Tokenizer references only; omit --prompts-only to also generate YaRN and
# >4096-position reference logits through the official model.
python tests/granite_speech5/dump_c2_fixtures.py \
  --model /path/to/checkpoint --fixtures-dir /path/to/references --prompts-only

export GRANITE_SPEECH5_MODEL=/path/to/checkpoint
export GRANITE_SPEECH5_FIXTURES=/path/to/references
export GRANITE_SPEECH5_TRACES=/path/to/traces
```

`c1` and `f` test prompt/token parity; `c2` includes controlled decoder
scaling, cache reuse, YaRN and teacher-forced reference comparisons.
`b1`, `b2` and `d` cover frontend, encoder/projector and session contracts.

## End-to-end and teacher-forced parity

Supply any suitable local mono 16 kHz WAV. Generated traces stay outside Git.
Run the reference and candidate sequentially:

```bash
python tests/granite_speech5/reference.py \
  --model /path/to/checkpoint --audio /path/to/input.wav \
  --out-dir /path/to/python-trace

build/debug/bin/granite_speech5_parity_probe \
  --model /path/to/checkpoint --audio /path/to/input.wav \
  --backend cpu --threads 8 --through generate --eager-attention \
  --out-dir /path/to/cpp-trace --log

python tests/granite_speech5/compare_trace.py \
  --reference /path/to/python-trace --candidate /path/to/cpp-trace
```

`--force-ids` supplies the same teacher-forced continuation to both probes.
`--projector-input` lets the C++ probe consume reference encoder states to
isolate downstream numerical checks. `--decoder-input` injects the Python
projected audio embeddings at the AR boundary for aligned teacher forcing.
Intermediate encoder captures are
explicitly requested by the parity probe; normal transcription only reads
back the final encoder state.

The comparison retains stage-specific shape, finiteness, absolute-error and
relative-RMS checks. Removing stored fixtures does not change those gates.

Use `--eager-attention` for the controlled FP32 comparison against Python eager
attention. This switch exists only in the parity probe. For a CUDA FP32 control,
also set `NVIDIA_TF32_OVERRIDE=0`; production inference retains its normal CUDA
precision policy and grouped flash attention. Omitting the probe switch checks
the production attention lowering and its numerical drift separately.
