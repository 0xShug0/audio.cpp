#ifndef GGML_SYCL_FATTN_ONEDNN_HPP
#define GGML_SYCL_FATTN_ONEDNN_HPP

#include "common.hpp"

// Static check: this flash-attention op can run as oneDNN's fused SDPA (F16 KV, no softcap, ALiBi
// or sinks, single stream, query long enough to be worth it). The mask is optional.
bool ggml_sycl_flash_attn_ext_onednn_supported(int device, const ggml_tensor * dst);

#if GGML_SYCL_DNNL
// Runs the SDPA and writes the F32 dst. Falls back to the TILE kernel on any failure.
void ggml_sycl_flash_attn_ext_onednn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
#endif

#endif // GGML_SYCL_FATTN_ONEDNN_HPP
