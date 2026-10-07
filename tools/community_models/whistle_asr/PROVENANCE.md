# Whistle conversion provenance

Checked on 2026-10-06. This records source inspection, not recognition-quality validation or legal approval.

## Model and extracted assets

Measured source inspection identifies the converter's pinned release as
[`Cactus-Compute/whistle` at `b358ddadd89b7a713b5aa131f23032d3cca1b251`](https://huggingface.co/Cactus-Compute/whistle/tree/b358ddadd89b7a713b5aa131f23032d3cca1b251).
The [model card](https://huggingface.co/Cactus-Compute/whistle/blob/b358ddadd89b7a713b5aa131f23032d3cca1b251/README.md)
declares Apache-2.0. The pinned
[LICENSE](https://huggingface.co/Cactus-Compute/whistle/blob/b358ddadd89b7a713b5aa131f23032d3cca1b251/LICENSE)
contains the Apache License, Version 2.0.
Its measured length is 11,358 bytes and SHA-256 is
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`, matching `convert.py`.

Measured converter inspection identifies these inputs:

- `checkpoints/whistle.safetensors` supplies the published FP32 checkpoint, renamed locally to `whistle.safetensors`.
- `config.json` supplies the model configuration.
- `whistle.cact` supplies the tokenizer blob, 257-by-80 mel filterbank, and two 512-element Hadamard permutations.
- `LICENSE` supplies the upstream license text, not audio.cpp's license.

`convert.py` verifies each source file's length and SHA-256 before conversion.
`extract_assets.py` verifies the four extracted assets' shapes, lengths, and SHA-256 hashes.
The `.cact` file itself has pinned SHA-256
`b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb`.
These checks establish byte provenance, not the training origin or ownership of each asset.

Measured inspection of the pinned repository finds one LICENSE and no NOTICE file or separate asset-license files.
No tokenizer training source, mel-filterbank generation source, or permutation-generation source is identified in that release.
The repository-wide Apache-2.0 declaration is the published licensing evidence for these bundled assets.
Independent third-party provenance remains unverified.
If separate rights or notices apply, which upstream source identifies them?
Resolve that question before asserting independently audited asset rights or publishing a model package.

## Notice preservation and conversion

Measured converter inspection passes the source directory through `--root` and verifies six embedded sidecars plus the embedded model spec.
The sidecars are `LICENSE`, `config.json`, `tokenizer.blob`, `mel_filterbank_80.f32`, `hadamard_perm1.f32`, and `hadamard_perm2.f32`.
The upstream LICENSE therefore remains inside the converted GGUF.
Do not remove that license when extracting or repackaging the model.
For any future redistribution, provide recipients a readable license copy and retain applicable attribution and NOTICE content.
Identify the repackaging as an audio.cpp conversion of the pinned FP32 checkpoint with assets extracted from the pinned `.cact`.
No downloadable package or redistribution approval is claimed here.

## Reference implementations

Measured source inspection identifies the documented reconstruction as
[`mrfakename/whistle-ONNX` at `14fc92cb09ad5c271fe9a4ec4be70274790050d2`](https://huggingface.co/mrfakename/whistle-ONNX/tree/14fc92cb09ad5c271fe9a4ec4be70274790050d2).
Its [model card](https://huggingface.co/mrfakename/whistle-ONNX/blob/14fc92cb09ad5c271fe9a4ec4be70274790050d2/README.md)
declares Apache-2.0. The file catalog contains no LICENSE or NOTICE file.
Inspected `scripts/wt.py`, `scripts/cact.py`, `scripts/tok.py`, and `scripts/weights.py` contain no copyright or license notices.
The card declaration is the available licensing evidence, not a separately verified code-license grant.
The contributor confirms that no port code was copied or adapted from an
upstream or reference implementation. This is contributor-reported provenance,
not an independent code-origin audit. No ONNX reference code or engine binary
is included in this port. The scope of the reference card declaration for its
scripts remains unverified, but this port does not claim or rely on permission
to redistribute those scripts.

Measured reference-source inspection shows that the reconstruction uses deployed, dequantized `.cact` weights.
Matching transcripts against that reference does not establish exact FP32 numerical parity with the published Safetensors checkpoint.
The locally available official `needle.exe` has no recorded source revision.
Its presence is not evidence of a pinned reference-engine revision or of any particular licensing terms for that binary.
No reference engine or ONNX reconstruction is bundled by this converter.
