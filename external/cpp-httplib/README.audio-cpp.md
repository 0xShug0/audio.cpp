# audio.cpp vendoring notes

This directory vendors the split `cpp-httplib` source used by llama.cpp at
commit `ece963f41b0b02d7a0d61436ae365762c073a4c8`. The upstream license is in
`LICENSE`.

audio.cpp builds it as a static library and enables HTTPS with a pinned,
statically linked BoringSSL release by default. See `CMakeLists.txt` and
`docs/model_manager.md` for the build switches.
