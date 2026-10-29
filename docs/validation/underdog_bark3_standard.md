# Underdog Bark 3.0 validation

Standard Bark 3.0 extends the existing `parakeet_tdt` community family and reuses
the shared Bark runtime/packed converter proposed in PRs #1130 and #1131.
Native CPU-only and combined CUDA/Vulkan Release builds pass. Model data,
reference comparisons, long-file time/memory and server checks are being
validated; this initial draft does not claim the pending results.

Source: `ConwayResearch/Underdog-Bark-3.0`, pinned revision
`85db2ad18ab41e534a222ee253c29191d77ed919`. `model.bark` is 163,398,645 bytes,
SHA-256 `c18de9418d73cbeb80b8c1287b392d480a56e191ef4a81d1baaa1b02e66d92ed`.
The base processor/tokenizer sidecars use NVIDIA revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b` and are checked by hash.

Three profile-isolation tests and five existing packed-reader tests pass.
They cover immutable standard/Plus identities, cross-model staging rejection,
checkpoint pinning, ANS/scale reconstruction and embedded contract metadata.
No shared backend/kernel or parallel-slot behavior is changed.

All 699 decoded inference tensors match the unmodified official loader exactly
before compatibility transforms. Ten focused CTests pass across the CPU/GPU
builds, and the first CPU/F32 nine-request reference sequence passes. The
original checkpoint emits `Yeah.` on the one-second silence fixture; this
behavior is part of the comparison, not silently replaced with an empty result.

[Build, conversion, CLI/server and original-reference commands](../models/underdog_bark3_standard.md).
