#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/dots_tts/types.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::dots_tts {

struct DotsLlmHidden {
    std::vector<float> values;
    int64_t steps = 0;
    int64_t hidden_size = 0;
};

class DotsQwen2State {
public:
    DotsQwen2State();
    ~DotsQwen2State();
    DotsQwen2State(DotsQwen2State &&) noexcept;
    DotsQwen2State & operator=(DotsQwen2State &&) noexcept;
    DotsQwen2State(const DotsQwen2State &) = delete;
    DotsQwen2State & operator=(const DotsQwen2State &) = delete;

    int64_t seq_len() const noexcept;
    int64_t capacity() const noexcept;

private:
    friend class DotsQwen2Component;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DotsQwen2Component {
public:
    static DotsQwen2Component load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        DotsLlmConfig config,
        assets::TensorStorageType weight_storage_type);

    DotsQwen2Component();
    DotsQwen2Component(DotsQwen2Component &&) noexcept;
    DotsQwen2Component & operator=(DotsQwen2Component &&) noexcept;
    DotsQwen2Component(const DotsQwen2Component &) = delete;
    DotsQwen2Component & operator=(const DotsQwen2Component &) = delete;
    ~DotsQwen2Component();

    bool is_loaded() const noexcept;
    DotsQwen2State create_state(int64_t max_sequence_length) const;
    std::vector<float> embed_tokens(const std::vector<int32_t> & token_ids) const;
    DotsLlmHidden prefill_embeddings(
        const std::vector<float> & embeddings,
        int64_t steps,
        DotsQwen2State & state) const;
    DotsLlmHidden decode_embedding(
        const std::vector<float> & embedding,
        DotsQwen2State & state) const;
    float eos_probability(const std::vector<float> & hidden) const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::dots_tts
