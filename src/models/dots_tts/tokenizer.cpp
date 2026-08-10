#include "engine/models/dots_tts/tokenizer.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts {
namespace {

constexpr const char * kAudioGenStartToken = "<|audio_gen_start|>";
constexpr const char * kAudioGenSpanToken = "<|audio_gen_span|>";
constexpr const char * kTextCondEndToken = "<|text_cond_end|>";
constexpr const char * kTtsTemplatePrefix = "[文本]";
constexpr const char * kTtsTemplateAudio = "[文本对应语音]";
constexpr const char * kInstructionTemplatePrefix = "[带指令文本]";
constexpr const char * kTextToAudioTemplatePrefix = "[声音描述]";
constexpr const char * kTextToAudioTemplateAudio = "[描述对应声音]";
constexpr const char * kInterleaveTemplatePrefix = "[流式语音合成]";

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("DotTTS tokenizer missing required token: " + token);
    }
    return *id;
}

}  // namespace

struct DotsTokenizer::Impl {
    explicit Impl(std::shared_ptr<const DotsAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              assets->resources.require_file("vocab"),
              assets->resources.require_file("merges"),
              assets->resources.require_file("tokenizer_config"),
              assets->resources.require_file("tokenizer_json"),
              engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          audio_gen_start_id(require_token_id(tokenizer, kAudioGenStartToken)),
          audio_gen_span_id(require_token_id(tokenizer, kAudioGenSpanToken)),
          text_cond_end_id(require_token_id(tokenizer, kTextCondEndToken)) {}

    std::shared_ptr<const DotsAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t audio_gen_start_id = 0;
    int32_t audio_gen_span_id = 0;
    int32_t text_cond_end_id = 0;
};

DotsTokenizer::DotsTokenizer(std::shared_ptr<const DotsAssets> assets)
    : impl_([&]() {
          if (assets == nullptr) {
              throw std::runtime_error("DotTTS tokenizer requires assets");
          }
          return std::make_shared<Impl>(std::move(assets));
      }()) {}

std::vector<int32_t> DotsTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text, true);
}

std::string DotsTokenizer::decode(const std::vector<int32_t> & token_ids, bool skip_special_tokens) const {
    return impl_->tokenizer.decode(token_ids, skip_special_tokens);
}

DotsGenerationSchedule DotsTokenizer::build_generation_schedule(
    const std::string & text,
    DotsTemplateName template_name,
    int64_t max_audio_tokens) const {
    if (max_audio_tokens <= 0) {
        throw std::runtime_error("DotTTS max_tokens must be positive");
    }
    std::string prefix;
    std::string audio_marker;
    switch (template_name) {
        case DotsTemplateName::Tts:
            prefix = kTtsTemplatePrefix;
            audio_marker = kTtsTemplateAudio;
            break;
        case DotsTemplateName::InstructionTts:
            prefix = kInstructionTemplatePrefix;
            audio_marker = kTtsTemplateAudio;
            break;
        case DotsTemplateName::TextToAudio:
            prefix = kTextToAudioTemplatePrefix;
            audio_marker = kTextToAudioTemplateAudio;
            break;
        case DotsTemplateName::TtsInterleave:
            prefix = kInterleaveTemplatePrefix;
            break;
    }
    DotsGenerationSchedule schedule;
    auto prefix_ids = encode(prefix);
    auto text_ids = encode(text);
    auto audio_ids = audio_marker.empty() ? std::vector<int32_t>{} : encode(audio_marker);
    schedule.token_ids.reserve(prefix_ids.size() + text_ids.size() + audio_ids.size() + static_cast<size_t>(max_audio_tokens) + 2);
    schedule.token_ids.insert(schedule.token_ids.end(), prefix_ids.begin(), prefix_ids.end());
    if (template_name == DotsTemplateName::TtsInterleave) {
        if (max_audio_tokens < static_cast<int64_t>(text_ids.size())) {
            throw std::runtime_error("DotTTS tts_interleave requires max_tokens to be at least the text token count");
        }
        for (const int32_t token_id : text_ids) {
            schedule.token_ids.push_back(token_id);
            schedule.token_ids.push_back(impl_->audio_gen_span_id);
        }
        schedule.token_ids.push_back(impl_->text_cond_end_id);
        schedule.token_ids.insert(
            schedule.token_ids.end(),
            static_cast<size_t>(max_audio_tokens - static_cast<int64_t>(text_ids.size())),
            impl_->audio_gen_span_id);
        schedule.interleave = true;
    } else {
        schedule.token_ids.insert(schedule.token_ids.end(), text_ids.begin(), text_ids.end());
        schedule.token_ids.insert(schedule.token_ids.end(), audio_ids.begin(), audio_ids.end());
        schedule.token_ids.push_back(impl_->audio_gen_start_id);
        schedule.token_ids.insert(schedule.token_ids.end(), static_cast<size_t>(max_audio_tokens), impl_->audio_gen_span_id);
        schedule.interleave = false;
    }
    return schedule;
}

int32_t DotsTokenizer::audio_gen_span_id() const noexcept {
    return impl_->audio_gen_span_id;
}

int32_t DotsTokenizer::audio_gen_start_id() const noexcept {
    return impl_->audio_gen_start_id;
}

int32_t DotsTokenizer::text_cond_end_id() const noexcept {
    return impl_->text_cond_end_id;
}

}  // namespace engine::models::dots_tts
