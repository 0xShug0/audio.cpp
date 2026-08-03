#pragma once

#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/tokenizer_ar.h"
#include "engine/models/vevo2/types.h"

#include <cstddef>
#include <memory>

namespace engine::assets {
enum class TensorStorageType;
class TensorSource;
}

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::vevo2 {

struct Vevo2ARWeights;
struct Vevo2ARPrefillGraph;
struct Vevo2ARDecodeGraph;

class Vevo2AutoregressiveRuntime final {
public:
    Vevo2AutoregressiveRuntime(
        std::shared_ptr<const Vevo2Assets> assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t prefill_graph_context_bytes,
        size_t decode_graph_context_bytes,
        engine::assets::TensorStorageType weight_storage_type);
    ~Vevo2AutoregressiveRuntime();

    Vevo2TokenSequence generate_content_style(
        const Vevo2PromptParts & prompt,
        const Vevo2GenerationOptions & generation) const;
    int64_t last_prompt_tokens() const noexcept;
    int32_t eos_token_id() const noexcept;
    int32_t pad_token_id() const noexcept;
    const Vevo2ARConfig & config() const noexcept;
    bool weights_uploaded() const noexcept;

private:
    std::shared_ptr<const Vevo2Assets> assets_;
    engine::core::ExecutionContext & execution_context_;
    size_t prefill_graph_context_bytes_ = 0;
    size_t decode_graph_context_bytes_ = 0;
    Vevo2ARTokenizer tokenizer_;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<const Vevo2ARWeights> weights_;
    mutable std::unique_ptr<Vevo2ARPrefillGraph> prefill_graph_;
    mutable std::unique_ptr<Vevo2ARDecodeGraph> decode_graph_;
    mutable int64_t last_prompt_tokens_ = 0;
};

}  // namespace engine::models::vevo2
