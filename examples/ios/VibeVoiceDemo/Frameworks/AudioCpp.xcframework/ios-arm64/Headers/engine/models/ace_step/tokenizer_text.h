#pragma once

#include "engine/models/ace_step/assets.h"
#include "engine/models/ace_step/types.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::ace_step {

class AceStepTextTokenizer {
public:
    struct Impl;
    enum class ResourceSet {
        TextEncoder,
        Planner,
    };

    explicit AceStepTextTokenizer(
        std::shared_ptr<const AceStepAssets> assets,
        ResourceSet resources = ResourceSet::TextEncoder);

    int32_t pad_token_id() const noexcept;
    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_control_special_tokens = false) const;
    std::optional<int32_t> find_token_id(const std::string & token) const;
    AceStepTokenizedText tokenize_text(
        const std::string & text,
        int64_t max_length) const;
    std::string apply_chat_template(
        const std::string & system_content,
        const std::string & user_content,
        bool add_generation_prompt) const;

private:
    std::shared_ptr<const AceStepAssets> assets_;
    std::shared_ptr<const Impl> impl_;
    ResourceSet resources_ = ResourceSet::TextEncoder;
};

}  // namespace engine::models::ace_step
