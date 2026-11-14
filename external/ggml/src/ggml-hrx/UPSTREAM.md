Imported from AMD-Ecosystem/llama.cpp, users/stella/hrx-rfc-v1, commit 46eee5c5a891feb68c76c522d5410eec2d9afaa4.

Local integration retains audio.cpp GGML APIs and removes diagnostic enum cases absent in this revision.

Audio coverage extensions add native F32 NORM and REPEAT plus the existing audio.cpp ROUND_BF16 unary. Error reporting prints the complete HRX failure status. The optional test-hrx-ops target reuses GGML CPU-reference operator tests.

Further audio coverage preserves arbitrary leading strides in existing F32 copies, adds scalar floating-point copies/conversions, F32 FILL and a multidimensional strided RMSNorm route. NONE nodes require no dispatch. Existing tuned normalization and copy routes retain priority.

General floating-point matmul covers strided/grouped batches not handled by existing tuned dense routes. Added native strided F32 binary arithmetic, scaled masked softmax without ALiBi/sinks, and 1D/2D IM2COL. Unsupported mathematical features remain explicitly rejected.
