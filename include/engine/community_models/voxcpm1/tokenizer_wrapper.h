#pragma once

#include "engine/community_models/voxcpm1/tokenizer_text.h"
#include "engine/community_models/voxcpm1/tokenizer_gguf.h"
#include "engine/community_models/voxcpm1/types.h"

#include <memory>
#include <variant>

namespace engine::community_models::voxcpm1 {

// Wrapper that can hold either VoxCPM1TextTokenizer (JSON-based) or VoxCPM1GgufTokenizer (GGUF-based)
class VoxCPM1TokenizerWrapper {
public:
    VoxCPM1TokenizerWrapper() = default;
    explicit VoxCPM1TokenizerWrapper(std::shared_ptr<const VoxCPM1TextTokenizer> tokenizer)
        : tokenizer_(std::move(tokenizer)) {}
    explicit VoxCPM1TokenizerWrapper(std::shared_ptr<const VoxCPM1GgufTokenizer> tokenizer)
        : tokenizer_(std::move(tokenizer)) {}

    VoxCPM1TextPrompt build_prompt(const std::string & text) const {
        if (std::holds_alternative<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)) {
            return std::get<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)->build_prompt(text);
        } else {
            return std::get<std::shared_ptr<const VoxCPM1GgufTokenizer>>(tokenizer_)->build_prompt(text);
        }
    }

    int32_t audio_start_token_id() const noexcept {
        if (std::holds_alternative<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)) {
            return std::get<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)->audio_start_token_id();
        } else {
            return std::get<std::shared_ptr<const VoxCPM1GgufTokenizer>>(tokenizer_)->audio_start_token_id();
        }
    }

    int32_t audio_end_token_id() const noexcept {
        if (std::holds_alternative<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)) {
            return std::get<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)->audio_end_token_id();
        } else {
            return std::get<std::shared_ptr<const VoxCPM1GgufTokenizer>>(tokenizer_)->audio_end_token_id();
        }
    }

    int32_t reference_audio_start_token_id() const noexcept {
        if (std::holds_alternative<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)) {
            return std::get<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)->reference_audio_start_token_id();
        } else {
            return std::get<std::shared_ptr<const VoxCPM1GgufTokenizer>>(tokenizer_)->reference_audio_start_token_id();
        }
    }

    int32_t reference_audio_end_token_id() const noexcept {
        if (std::holds_alternative<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)) {
            return std::get<std::shared_ptr<const VoxCPM1TextTokenizer>>(tokenizer_)->reference_audio_end_token_id();
        } else {
            return std::get<std::shared_ptr<const VoxCPM1GgufTokenizer>>(tokenizer_)->reference_audio_end_token_id();
        }
    }

    bool empty() const noexcept {
        return std::holds_alternative<std::monostate>(tokenizer_);
    }

private:
    std::variant<
        std::monostate,
        std::shared_ptr<const VoxCPM1TextTokenizer>,
        std::shared_ptr<const VoxCPM1GgufTokenizer>
    > tokenizer_;
};

}  // namespace engine::community_models::voxcpm1