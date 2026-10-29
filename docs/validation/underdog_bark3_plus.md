# Underdog Bark 3.0 Plus validation

Initial implementation checkpoint, Windows 11 / RTX 3090 24 GiB / eight inference
threads, CPU-only and CUDA/Vulkan Release builds. This Plus extension reuses the
Bark runtime proposed in PR #1130. It changes no shared backend/kernel behavior.

Source revision `d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb`:
`model.bark` is 179,524,845 bytes, SHA-256
`fdd9f1052e4fdead07d71e8cd2138813eb0c209fd6ff8b1b6f47737e1e501857`.
The pinned NVIDIA processor/tokenizer sidecars use revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`.

All **699 inference tensors** decoded by the new converter match the unmodified
official `bark_loader.load_state_dict(..., as_numpy=True)` exactly, before the
documented loader-compatibility transforms. The container has 723 records;
24 training-only counters are excluded. Five weight-free converter tests pass,
covering ANS column classes, integer scale products, LZMA/record bounds, precision
selection and the English embedded contract. Ten focused CTests pass across the
CPU/GPU builds, including both explicit Bark variants and stream/prefix behavior.

The first CUDA mixed-F16 smoke test transcribes LibriSpeech
`2086-149220-0033.wav` (7.435 s) correctly, with warm processing times 34.4 and
26.2 ms. Original PyTorch reference matrix, broader backend/precision checks,
long-file performance/memory and HTTP/SSE validation are still in progress.
This checkpoint does not claim those pending results or Metal/HIP/corpus WER.

Reproduction and license details are in [the model guide](../models/underdog_bark3_plus.md).
