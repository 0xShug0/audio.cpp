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

class DotsLlmState {
public:
    DotsLlmState();
    ~DotsLlmState();
    DotsLlmState(DotsLlmState &&) noexcept;
    DotsLlmState & operator=(DotsLlmState &&) noexcept;
    DotsLlmState(const DotsLlmState &) = delete;
    DotsLlmState & operator=(const DotsLlmState &) = delete;

    int64_t seq_len() const noexcept;
    int64_t capacity() const noexcept;

private:
    friend class DotsLlmComponent;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DotsLlmComponent {
public:
    static DotsLlmComponent load_from_tensor_source(
        std::shared_ptr<const assets::TensorSource> source,
        core::BackendConfig backend,
        DotsLlmConfig config,
        assets::TensorStorageType weight_storage_type);

    DotsLlmComponent();
    DotsLlmComponent(DotsLlmComponent &&) noexcept;
    DotsLlmComponent & operator=(DotsLlmComponent &&) noexcept;
    DotsLlmComponent(const DotsLlmComponent &) = delete;
    DotsLlmComponent & operator=(const DotsLlmComponent &) = delete;
    ~DotsLlmComponent();

    bool is_loaded() const noexcept;
    DotsLlmState create_state(int64_t max_sequence_length) const;
    std::vector<float> embed_tokens(const std::vector<int32_t> & token_ids) const;
    DotsLlmHidden prefill_embeddings(
        const std::vector<float> & embeddings,
        int64_t steps,
        DotsLlmState & state) const;
    DotsLlmHidden decode_embedding(
        const std::vector<float> & embedding,
        DotsLlmState & state) const;
    float eos_probability(const std::vector<float> & hidden) const;

    void release_runtime_graphs();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::dots_tts
