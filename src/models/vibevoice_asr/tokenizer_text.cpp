#include "engine/models/vibevoice_asr/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::vibevoice_asr {
namespace {

constexpr const char * kSystemPrompt =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

std::shared_ptr<const VibeVoiceAssets> require_assets(std::shared_ptr<const VibeVoiceAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("VibeVoice-ASR tokenizer requires assets");
    }
    return assets;
}

const std::filesystem::path & require_path(
    const std::optional<std::filesystem::path> & path,
    const char * label) {
    if (!path.has_value()) {
        throw std::runtime_error(std::string("VibeVoice-ASR missing tokenizer ") + label);
    }
    return *path;
}

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const char * token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error(std::string("VibeVoice-ASR tokenizer missing token: ") + token);
    }
    return *id;
}

std::string format_seconds(double seconds) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << seconds;
    return stream.str();
}

std::string chat_message(const std::string & role, const std::string & content, bool add_generation_prompt) {
    std::string out = "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }
    return out;
}

}  // namespace

struct VibeVoiceASRTextTokenizer::Impl {
    explicit Impl(const VibeVoiceAssets & assets)
        : tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              require_path(assets.paths.tokenizer_vocab_path, "vocab.json"),
              require_path(assets.paths.tokenizer_merges_path, "merges.txt"),
              require_path(assets.paths.tokenizer_config_path, "tokenizer_config.json"),
              assets.paths.tokenizer_json_path,
              engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          speech_start(require_token_id(tokenizer, "<|object_ref_start|>")),
          speech_end(require_token_id(tokenizer, "<|object_ref_end|>")),
          speech_pad(require_token_id(tokenizer, "<|box_start|>")),
          eos(require_token_id(tokenizer, "<|endoftext|>")),
          pad(require_token_id(tokenizer, "<|image_pad|>")) {}

    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t speech_start = 0;
    int32_t speech_end = 0;
    int32_t speech_pad = 0;
    int32_t eos = 0;
    int32_t pad = 0;
};

VibeVoiceASRTextTokenizer::VibeVoiceASRTextTokenizer(std::shared_ptr<const VibeVoiceAssets> assets)
    : assets_(require_assets(std::move(assets))),
      impl_(std::make_shared<Impl>(*assets_)) {}

std::vector<int32_t> VibeVoiceASRTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text);
}

VibeVoiceASRPrompt VibeVoiceASRTextTokenizer::build_prompt(
    const VibeVoiceASRRequest & request,
    int64_t speech_tokens) const {
    if (speech_tokens <= 0) {
        throw std::runtime_error("VibeVoice-ASR prompt requires positive speech token count");
    }

    const double seconds = static_cast<double>(request.audio.samples.size()) /
        static_cast<double>(request.audio.channels * request.audio.sample_rate);
    std::string suffix;
    if (!request.context.empty()) {
        suffix = "This is a " + format_seconds(seconds) +
            " seconds audio, with extra info: " + request.context +
            "\n\nPlease transcribe it with these keys: Start time, End time, Speaker ID, Content";
    } else {
        suffix = "This is a " + format_seconds(seconds) +
            " seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content";
    }

    std::string user_content = "<|object_ref_start|>";
    for (int64_t i = 0; i < speech_tokens; ++i) {
        user_content += "<|box_start|>";
    }
    user_content += "<|object_ref_end|>\n";
    user_content += suffix;

    VibeVoiceASRPrompt prompt;
    auto system_ids = impl_->tokenizer.encode(chat_message("system", kSystemPrompt, false));
    auto user_ids = impl_->tokenizer.encode(chat_message("user", user_content, false));
    prompt.input_ids.reserve(system_ids.size() + user_ids.size());
    prompt.input_ids.insert(prompt.input_ids.end(), system_ids.begin(), system_ids.end());
    prompt.input_ids.insert(prompt.input_ids.end(), user_ids.begin(), user_ids.end());
    prompt.attention_mask.assign(prompt.input_ids.size(), 1);
    for (size_t i = 0; i < prompt.input_ids.size(); ++i) {
        if (prompt.input_ids[i] == impl_->speech_pad) {
            prompt.speech_positions.push_back(static_cast<int32_t>(i));
        }
    }
    if (static_cast<int64_t>(prompt.speech_positions.size()) != speech_tokens) {
        throw std::runtime_error("VibeVoice-ASR prompt speech token count mismatch");
    }
    prompt.speech_tokens = speech_tokens;
    prompt.audio_seconds = seconds;
    return prompt;
}

std::string VibeVoiceASRTextTokenizer::decode(
    const std::vector<int32_t> & token_ids,
    bool skip_special_tokens) const {
    return impl_->tokenizer.decode(token_ids, skip_special_tokens);
}

int32_t VibeVoiceASRTextTokenizer::eos_id() const noexcept {
    return impl_->eos;
}

int32_t VibeVoiceASRTextTokenizer::pad_id() const noexcept {
    return impl_->pad;
}

int32_t VibeVoiceASRTextTokenizer::speech_pad_id() const noexcept {
    return impl_->speech_pad;
}

}  // namespace engine::models::vibevoice_asr
