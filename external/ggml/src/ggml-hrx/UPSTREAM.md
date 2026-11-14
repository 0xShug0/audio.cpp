Imported from AMD-Ecosystem/llama.cpp, users/stella/hrx-rfc-v1, commit 46eee5c5a891feb68c76c522d5410eec2d9afaa4.

Local integration retains audio.cpp GGML APIs and removes diagnostic enum cases absent in this revision.

Audio coverage extensions add native F32 NORM and REPEAT plus the existing audio.cpp ROUND_BF16 unary. Error reporting prints the complete HRX failure status. The optional test-hrx-ops target reuses GGML CPU-reference operator tests.
