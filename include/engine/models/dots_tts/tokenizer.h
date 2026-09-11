#pragma once

#include "engine/models/dots_tts/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::dots_tts {

class DotsTokenizer {
public:
    explicit DotsTokenizer(std::shared_ptr<const DotsAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens = false) const;
    DotsGenerationSchedule build_generation_schedule(
        const std::string & text,
        DotsTemplateName template_name,
        int64_t max_audio_tokens) const;
    DotsGenerationSchedule build_edit_generation_schedule(
        const std::string & source_text,
        const std::string & instruction,
        const std::string & target_text,
        int64_t source_audio_tokens,
        int64_t target_audio_tokens) const;
    int32_t audio_gen_span_id() const noexcept;
    int32_t audio_gen_start_id() const noexcept;
    int32_t audio_gen_end_id() const noexcept;
    int32_t text_cond_end_id() const noexcept;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace engine::models::dots_tts
