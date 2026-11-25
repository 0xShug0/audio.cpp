# KugelAudio encoder reference comparison

`reference_encoder.py` retains its upstream encoder generation path. Supply
`--upstream`, `--config`, `--encoder`, `--audio` and `--output-dir`; use strict
F32 with TF32 disabled for the documented comparison. Fixtures must be finite,
nonempty 24 kHz audio with unique file stems. Regenerate old reference reports
to include sample-rate metadata.

The native `kugelaudio_reference_encoder_probe` writes `<prefix>.input.f32`,
`<prefix>.mean.f32` and `<prefix>.json`. Choose `CPP/<audio stem>` as each prefix.
Compare without loading a model:

```bash
python tests/kugelaudio/reference_encoder.py --compare --output-dir REF \
  --cpp-dir CPP --backend cuda --max-input-abs-error "$INPUT_BOUND"
```

Feature acceptance follows
[the encoder report](../../docs/reports/kugelaudio_reference_cloning.md):
CPU/Vulkan relative L2 < 1e-5 and cosine > 0.99999999; CUDA relative L2 < 1e-3
and cosine > 0.999999. Prepared-input maximum absolute error requires an explicit
finite nonnegative bound; an observed report maximum is not a validated bound.
Missing/empty artifacts, invalid shapes, nonfinite values, zero feature norms,
input disagreement or feature acceptance failures exit nonzero. Native prepared
input is 24 kHz by its encoder contract; reference audio rate is checked before
inference and recorded in the report.
