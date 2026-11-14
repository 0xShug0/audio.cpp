Imported from AMD-Ecosystem/llama.cpp, users/stella/hrx-rfc-v1, commit 46eee5c5a891feb68c76c522d5410eec2d9afaa4.

Local integration retains audio.cpp GGML APIs and removes diagnostic enum cases absent in this revision.

Audio coverage extensions add native F32 NORM and REPEAT plus the existing audio.cpp ROUND_BF16 unary. Error reporting prints the complete HRX failure status. The optional test-hrx-ops target reuses GGML CPU-reference operator tests.

Further audio coverage preserves arbitrary leading strides in existing F32 copies, adds scalar floating-point copies/conversions, F32 FILL and a multidimensional strided RMSNorm route. NONE nodes require no dispatch. Existing tuned normalization and copy routes retain priority.

General floating-point matmul covers strided/grouped batches not handled by existing tuned dense routes. Added native strided F32 binary arithmetic, scaled masked softmax without ALiBi/sinks, and 1D/2D IM2COL. Unsupported mathematical features remain explicitly rejected.

Further coverage includes I32-to-floating copies, F32 SET slices and general-axis floating concatenation using the existing strided conversion kernel, plus native causal masking. Fusion matching rejects consumers whose external side inputs have not yet been computed. These changes remain confined to HRX.

Native F32 online attention retains strided/grouped-query layouts without converting K/V precision; the tuned F16 attention paths retain priority. Added F32 SUM_ROWS and ARGMAX, and I32 indices in standalone SET_ROWS while preserving the I64-only fused routes.

Added contiguous F32 CLAMP, arbitrary-width/batched strided F32 GET_ROWS, exact I32 copies and standalone cache-view layout support. Existing tuned gather and fused cache-update routes retain priority and contracts.
