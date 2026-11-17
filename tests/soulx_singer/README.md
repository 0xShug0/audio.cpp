# SoulX-Singer validation

The conversion script packages the official SVC or SVS checkpoint using the
framework GGUF converter. See the model usage documents for complete conversion
commands. Use `--type f32` for controlled numerical replay; normal packages use
`--type f16` or `--type q8_0`.

Reference scripts import an upstream source checkout. The tested reference uses
PyTorch 2.11 and Transformers 4.41.2. Reference export requires CUDA; native replay
can use CPU or CUDA. SageAttention is not called.

## Frozen SVC replay

```bash
python tests/soulx_singer/reference.py \
  --upstream models/SoulX-Singer-source \
  --checkpoint models/SoulX-Singer/model-svc.pt \
  --rmvpe models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt \
  --reference reference.wav --audio singing.wav --seconds 8 \
  --output build/logs/soulx-singer/control --mode control --log \
  > soulx-singer-reference.log 2>&1

build/debug/bin/soulx_singer_components \
  --model models/SoulX-Singer/control-f32.gguf \
  --reference build/logs/soulx-singer/control/reference.safetensors \
  --family soulx_singer --backend cpu --mode denoise \
  --out build/logs/soulx-singer/control/cpp.wav \
  --log --log-file build/logs/soulx-singer/control/cpp.log
```

The fixture freezes conditioning, noise, final mel and waveform. `--mode encode`
checks the content/pitch frontend; `--mode decode` isolates the vocoder with the
same Python mel. CPU denoiser/vocoder replay retains the strict default cosine gate.

## Frozen SVS replay

`reference_svs.py` exports the official example with `--control score` or
`--control melody`; `--example en` selects the English multi-phone case.
Defaults select the Mandarin example.

```bash
python tests/soulx_singer/reference_svs.py \
  --upstream models/SoulX-Singer-source \
  --checkpoint models/SoulX-Singer/model.pt \
  --output build/logs/soulx-singer/svs --mode control --control score --log \
  > soulx-singer-svs-reference.log 2>&1

build/debug/bin/soulx_singer_components \
  --model models/SoulX-Singer/svs-control-f32.gguf \
  --reference build/logs/soulx-singer/svs/reference.safetensors \
  --family soulx_singer --control score --backend cuda --mode encode \
  --out build/logs/soulx-singer/svs/cpp.wav \
  --log --log-file build/logs/soulx-singer/svs/cpp.log
```

Encoding checks exact phoneme/pitch/type/frame indices, joint reference/target
conditioning and reference mel. Denoising reuses the frozen conditioning and
noise. GPU vocoder drift was accepted after validation; where explicitly chosen,
`--min-waveform-cosine 0.9975` applies only to the waveform check. It does not lower
the frontend or mel gates, and is not a production runtime option.

## Performance and lifecycle

Reference `--mode performance --repeat 3` runs the official FP16 path with a
warmup and records process peak memory through the existing resource monitor.
`run_validation.py` uses the public server and existing server-test lifecycle
helper. Its request JSON uses the public task request format.

The server checks repeated seeds, changed seeds and CFG restoration. Supplying
`--long-request` and `--alternate-reference` checks changing/restoring shape and
reference conditioning. `--errors-only` checks rejected scalar and control
options without regenerating passing audio. Benchmarks explicitly enable pitch
matching; SVS benchmarks select score control. Model defaults remain upstream
melody control with automatic pitch matching disabled.

The model-dependent executable is gated by `ENGINE_BUILD_MODEL_TESTS` and the
selected model task. Independent DiT and ConvNeXt math tests remain framework
unit tests, with no checkpoint dependency.
