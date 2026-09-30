#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/transformers/decoder.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/models/higgs_audio_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::higgs_audio_tts {

struct HiggsQwen3DecoderStackWeights {
    std::vector<modules::DecoderLayerWeights> layers;
};

struct HiggsQwen3ARWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_embedding;
    core::TensorValue modality_embedding;
    HiggsQwen3DecoderStackWeights decoder;
    core::TensorValue norm;
    bool packed_qkv = false;
};

HiggsQwen3ARWeights load_higgs_ar_weights(
    const HiggsAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

class HiggsQwen3ARRuntime {
public:
    HiggsQwen3ARRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        assets::TensorStorageType weight_storage_type,
        core::AttentionPreference attention_preference = core::AttentionPreference::Auto);

    const HiggsAssets & assets() const noexcept;
    const HiggsQwen3ARWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int device() const noexcept;
    int threads() const noexcept;
    bool allow_flash_attention() const noexcept;

private:
    std::shared_ptr<const HiggsAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int device_ = 0;
    int threads_ = 1;
    bool allow_flash_attention_ = true;
    std::shared_ptr<const HiggsQwen3ARWeights> weights_;
};

struct HiggsARDecodeInput {
    std::vector<int32_t> last_codes;
    bool use_last_codes = false;
};

struct HiggsARDecodeOutput {
    std::vector<float> codebook_logits;
};

struct HiggsARDecodeTiming {
    double input_upload_ms = 0.0;
    double mask_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    int64_t steps = 0;

    void add(const HiggsARDecodeTiming & other) noexcept;
};

struct HiggsARPrefillInput {
    std::vector<int32_t> text_tokens;
    std::vector<int32_t> fused_code_ids;
    std::vector<float> text_gate;
    std::vector<float> code_gate;
    int64_t steps = 0;
};

struct HiggsARPrefillOutput {
    HiggsARDecodeOutput output;
    runtime::TransformerKVState kv_state;
    bool wrote_cache = false;
};

class HiggsQwen3KVCache {
public:
    HiggsQwen3KVCache(std::shared_ptr<HiggsQwen3ARRuntime> runtime, int64_t cache_steps);
    ~HiggsQwen3KVCache();

    HiggsQwen3KVCache(const HiggsQwen3KVCache &) = delete;
    HiggsQwen3KVCache & operator=(const HiggsQwen3KVCache &) = delete;

    bool can_run(const HiggsQwen3ARRuntime & runtime, int64_t required_steps) const;
    int64_t cache_steps() const;
    int64_t valid_steps() const;
    int64_t current_end() const;
    void reset();
    void retain_prefix(int64_t prefix_steps);
    void import_state(const runtime::TransformerKVState & state);
    void copy_from(const HiggsQwen3KVCache & source);
    runtime::TransformerKVState export_state() const;
    void advance_after_direct_append(int64_t steps);
    const core::TensorValue & key_tensor(size_t layer) const;
    const core::TensorValue & value_tensor(size_t layer) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsQwen3PrefillGraph {
public:
    HiggsQwen3PrefillGraph(
        std::shared_ptr<HiggsQwen3ARRuntime> runtime,
        int64_t prompt_steps,
        int64_t start_step,
        HiggsQwen3KVCache * cache,
        size_t graph_arena_bytes);
    ~HiggsQwen3PrefillGraph();

    HiggsQwen3PrefillGraph(const HiggsQwen3PrefillGraph &) = delete;
    HiggsQwen3PrefillGraph & operator=(const HiggsQwen3PrefillGraph &) = delete;

    bool matches(const HiggsQwen3ARRuntime & runtime, int64_t prompt_steps, int64_t start_step) const;
    HiggsARPrefillOutput run(const HiggsARPrefillInput & input, int64_t start_step = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class HiggsQwen3DecodeGraph {
public:
    HiggsQwen3DecodeGraph(
        std::shared_ptr<HiggsQwen3ARRuntime> runtime,
        int64_t cache_steps,
        HiggsQwen3KVCache & cache,
        size_t graph_arena_bytes);
    ~HiggsQwen3DecodeGraph();

    HiggsQwen3DecodeGraph(const HiggsQwen3DecodeGraph &) = delete;
    HiggsQwen3DecodeGraph & operator=(const HiggsQwen3DecodeGraph &) = delete;

    bool can_run(const HiggsQwen3ARRuntime & runtime, int64_t required_steps) const;
    int64_t cache_steps() const;
    void import_prefill_state(const runtime::TransformerKVState & state);
    void begin_decode_run();
    HiggsARDecodeTiming timing() const;
    void run_step_into(const HiggsARDecodeInput & input, HiggsARDecodeOutput & output, bool log_timing = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::higgs_audio_tts
