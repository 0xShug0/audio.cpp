# Phonon-2

[Fermion Research Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)
is an English ASR derivative of NVIDIA Parakeet TDT 0.6B v3. This native port
uses the existing `parakeet_tdt` frontend, FastConformer encoder, LSTM predictor,
and greedy TDT decoder in audio.cpp. Python is needed only for conversion and
the optional reference comparison, not for CLI/server inference.

## Conversion

Install `huggingface_hub`, `numpy`, `safetensors`, and `zstandard`. Download the
official artifact and original Parakeet sidecars (not the teacher's weights):

```bash
hf download FermionResearch/Phonon-2 \
  --revision 7e153e4054c0a10db6d47d6fff5c062d4c77c154 \
  --local-dir models/Phonon-2-source
hf download nvidia/parakeet-tdt-0.6b-v3 \
  --revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b \
  --include 'config.json' 'processor_config.json' 'tokenizer.json' \
            'tokenizer_config.json' 'generation_config.json' \
  --local-dir models/Phonon-2-source/parakeet-reference

python tools/community_models/convert_phonon2.py \
  --source-dir models/Phonon-2-source \
  --reference-dir models/Phonon-2-source/parakeet-reference \
  --output-dir models/Phonon-2-F32 \
  --converter build/<preset>/bin/audiocpp_gguf \
  --gguf-output models/Phonon-2-GGUF/phonon-2-f32.gguf
```

On Windows, add `.exe` to executable names. The staging directory must not
already exist. Omit the last two arguments to create only the safetensors
package. The converter verifies the pinned archive/container SHA256, frontend
and encoder configuration, and the complete tokenizer vocabulary; it retains
the upstream NOTICE and license files and records tensor hashes/provenance.

The packed archive is about 164 MB, but this initial implementation **expands
the weights**: 699 inference tensors contain 2,508,229,144 bytes of F32 data.
Their reconstructed values are retained exactly, including int6 scale products
that F16 would round. Only 24 unused BatchNorm training counters are omitted.
No packed five-value GGML type/kernel is implemented; Q2/Q4 re-quantization is
not equivalent to Phonon's learned levels. Download size is not runtime VRAM.

For a smaller standalone package, add `--gguf-type bf16` to the conversion
command and use a filename such as `phonon-2-bf16.gguf`. The staging package
remains exact F32; the resulting GGUF rounds the weights to BF16, reducing its
size to about 1.26 GB. Its embedded spec/provenance identifies it as a rounded
derivative, and it does not embed the F32 tensor hashes. It uses the same native
CLI/server commands and reference graph policies as the F32 package. BF16 does
not preserve every trained value; compare transcripts for your workload before
assuming equivalent accuracy. The optional exact-F32 validator below is for the
F32 GGUF, not the BF16 derivative.

`--gguf-type q8_0` also produces a reduced-size derivative (about 931 MB with
the current converter). This is a mixed-precision GGUF: eligible matrix weights
use Q8_0, while the converter retains other tensors in F32/F16 and keeps the
duration-sensitive `joint.head.*` tensors in their original F32 precision. It is a new
quantization of the reconstructed weights, not Phonon's original five-value
storage format. Neither the BF16 nor Q8_0 file implies a proportional reduction
in inference VRAM; backends may allocate conversion/work buffers. Q8 packages
select `audiocpp_cpu_matmul_weight_type=f32`: CPU inference expands their
matmul weights to F32 to avoid additional activation quantization. The file
remains Q8-rounded, but this precision policy increases host RAM usage.
See the [BF16/Q8_0 validation results](../reports/phonon2_validation.md)
for transcript checks, measured numerical differences, file sizes, and memory.

The generated config explicitly selects encoder input scale 1 (the reference's
`scale_input=false`), BatchNorm variance floor 0 (`sqrt(variance + epsilon)`),
and F32 matmul accumulation. Existing Parakeet packages retain their previous
scaling, variance floor, and backend precision defaults. Phonon's CUDA instance
also selects full-F32 matmul inputs: TF32 and additional activation rounding or
quantization are disabled through existing kernels/cuBLAS. The shared CUDA
backend has an opt-in instance policy; its default behavior and kernel bodies
are unchanged. Vulkan kernel bodies are unchanged.

Phonon packages also select `audiocpp_word_timestamp_mode=token_duration`:
word ends follow the last token's predicted duration, punctuation attaches to
the preceding word, bare word markers retain their time, and ends clamp to the
actual audio length. Other Parakeet packages keep their existing timestamp
convention. This fixes formatting; floating-point/quantized duration decisions
can still differ, especially after rounding weights. See the
[reference checks and known timing differences](../reports/phonon2_validation.md).

## Native CLI and server

Build with `parakeet_tdt` enabled (or the full model set). The standalone GGUF
embeds an English-only, offline, experimental model spec and its tokenizer:

CUDA reference inference requires the separate instance precision support in
[PR #861](https://github.com/0xShug0/audio.cpp/pull/861). CPU and Vulkan do not
depend on that API. Without it, the strict CUDA path reports an unsupported
backend policy rather than silently using different arithmetic.

```bash
build/<preset>/bin/audiocpp_cli --task asr --family parakeet_tdt \
  --model models/Phonon-2-GGUF/phonon-2-f32.gguf \
  --backend cuda --audio recording.wav
```

Use `--backend cpu` or `--backend vulkan` with the corresponding build. For a
staging safetensors directory also supply
`--model-spec-override models/Phonon-2-F32/model_spec.json`. Long recordings
can use `--session-option parakeet_tdt.offline_mode=long_form`; this scheduling
is audio.cpp's Parakeet implementation, not Phonon's original segmenter.
Its padded encoder windows can produce different timings from full-context
inference; the current precision checks do not establish original long-audio
segmentation parity.

The standard serial server accepts the GGUF through `/v1/tasks/run` or
`/v1/audio/transcriptions`. Configure a model entry with `id: "phonon2"`,
`family: "parakeet_tdt"`, `path` pointing to the GGUF, `task: "asr"`, and
`mode: "offline"`; then use `model: "phonon2"` in requests. See
[the server API](../../app/server/README.md) for request formats.
No parallel-runtime option is needed. Official Phonon hotword biasing and
native streaming behavior are not ported by this weight-package integration.

## Validation

```bash
python tests/parakeet_tdt/test_phonon2_conversion.py
python tests/parakeet_tdt/test_phonon2_timestamps.py
```

For the optional real-weight check, also install `torch`, `transformers`
(with `ParakeetForTDT`), `soundfile`, and `gguf`, then run:

```bash
python tests/parakeet_tdt/validate_phonon2.py \
  --source-dir models/Phonon-2-source \
  --reference-dir models/Phonon-2-source/parakeet-reference \
  --staging-dir models/Phonon-2-F32 \
  --model models/Phonon-2-GGUF/phonon-2-f32.gguf \
  --cli build/<preset>/bin/audiocpp_cli \
  --parity-dumper build/<preset>/bin/parakeet_parity_dump \
  --backend cuda --device 0 --audio speech.wav silence.wav \
  --output-dir outputs/phonon2-validation
```

Use mono 16 kHz WAV inputs and put speech first. The script verifies the
official reference code hashes, compares every inference tensor through
staging and GGUF, checks every transcript against the published F32 reference,
and compares full-encoder activations for the first input. It downloads nothing
and writes `validation.json`, activation dumps, and native logs. Repeat with
CPU and Vulkan, selecting the appropriate device. Server endpoint/repeat
checks and existing Parakeet regression tests are separate from this script;
see the [local validation report](../reports/phonon2_validation.md).

To include original word timing in the validator, install `fermion-research==0.2.11`
or unpack its wheel and add `--reference-formatter /path/to/fermion/_speech/segment.py`.
The validator verifies this independent formatter's SHA256, uses actual generated
token frames/durations (including blank skips), and compares CLI word timestamps
with at most 1 ms tolerance for the original JSON's rounding. A whole 80 ms frame
shift fails. Without this argument, the report explicitly says that word timestamps
were not checked; text/encoder success does not establish timing parity.

The existing `parakeet_parity_dump` supports `--encoder-only 1` to produce
`mel_features.npy` and `enc_out.npy` without NeMo layer-0 captures. This permits
comparison with the Phonon Transformers reference. Exact weight conversion
does not imply bit-identical floating-point activations across backends or
establish corpus-level WER parity; record measured tolerances and transcripts.

## Attribution

Weights: CC-BY-4.0, Fermion Research, derived from NVIDIA Parakeet TDT 0.6B v3.
Official decoder/reference code: Apache-2.0. Preserve the model's
[NOTICE](https://huggingface.co/FermionResearch/Phonon-2/blob/main/NOTICE) and
weight license when redistributing converted packages.
