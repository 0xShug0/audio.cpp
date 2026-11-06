# audio.cpp vendoring notes

This directory vendors the unmodified `lib/` sources of
[LZ4](https://github.com/lz4/lz4) release `v1.10.0`
(`lz4-1.10.0.tar.gz`, SHA-256
`537512904744b35e232912055ccf8ec66d768639ff3abe5788d90d792ec5f48b`).
The library files are BSD-2-Clause licensed; see `LICENSE`.

Only `lz4.c`, `lz4frame.c`, `lz4hc.c`, and `xxhash.c` are compiled, and only
when `AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON`. The native package manager uses
the LZ4 frame decoder to decompress Hugging Face Xet xorb chunks.
