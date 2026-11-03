# Bark packed decoder dependencies

`xz/` is the XZ Embedded decoder subset, pinned to
`tukaani-project/xz-embedded@ae63ae3a36ed01724674e8f3d750dc47bf125410`.
Upstream files are unchanged; their 0BSD license is in `xz/COPYING`.
Only single-call LZMA2/XZ decoding with CRC32/CRC64 is built; no encoder or BCJ
filters are included.

The model-local C++ ANS/CDF implementation follows the public
`bamler-lab/constriction` v0.5.0 (`fbd679652d1e0903486883eeb0489ce2f4519c7a`)
`DefaultAnsCoder` and fast categorical quantization algorithms. Its MIT license
is retained in `constriction-LICENSE-MIT`. No Rust build dependency is required.

The container layout and row-level reconstruction follow Conway Research's
`bark_loader.py` (CC-BY-4.0); model packages retain the publisher's NOTICE and
license. Runtime code does not execute that Python file.
