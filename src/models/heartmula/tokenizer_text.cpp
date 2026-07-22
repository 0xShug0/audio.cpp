#include "engine/models/heartmula/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"

#include "unicode.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::heartmula {
namespace {

std::shared_ptr<const HeartMuLaAssets> require_assets(std::shared_ptr<const HeartMuLaAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("HeartMuLa text tokenizer requires assets");
    }
    return assets;
}

std::string unicode_lowercase(const std::string & text) {
    std::string out;
    for (auto codepoint : unicode_cpts_from_utf8(text)) {
        codepoint = unicode_tolower(codepoint);
        out += unicode_cpt_to_utf8(codepoint);
    }
    return out;
}

void add_bos_eos(
    std::vector<int32_t> & ids,
    int32_t bos_id,
    int32_t eos_id,
    const char * label) {
    if (ids.empty()) {
        throw std::runtime_error(std::string("HeartMuLa tokenizer produced empty ") + label + " ids");
    }
    if (ids.front() != bos_id) {
        ids.insert(ids.begin(), bos_id);
    }
    if (ids.back() != eos_id) {
        ids.push_back(eos_id);
    }
}

std::string normalize_tags(std::string tags) {
    tags = unicode_lowercase(tags);
    if (tags.size() < 5 || tags.compare(0, 5, "<tag>") != 0) {
        tags = "<tag>" + tags;
    }
    if (tags.size() < 6 || tags.compare(tags.size() - 6, 6, "</tag>") != 0) {
        tags += "</tag>";
    }
    return tags;
}

}  // namespace

struct HeartMuLaTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const HeartMuLaAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              {},
              {},
              {},
              assets->resources.require_file("tokenizer_json"),
              engine::tokenizers::LlamaBpePreTokenizer::Llama3,
          }) {}

    std::shared_ptr<const HeartMuLaAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
};

HeartMuLaTextTokenizer::HeartMuLaTextTokenizer(std::shared_ptr<const HeartMuLaAssets> assets)
    : impl_(std::make_shared<Impl>(require_assets(std::move(assets)))) {}

std::vector<int32_t> HeartMuLaTextTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text);
}

HeartMuLaPromptEncoding HeartMuLaTextTokenizer::encode_prompt(const HeartMuLaPromptRequest & request) const {
    const auto & config = impl_->assets->generation_config;
    const auto & model_config = impl_->assets->mula_config;
    auto tags_ids = encode(normalize_tags(request.tags));
    auto lyrics_ids = encode(unicode_lowercase(request.lyrics));
    add_bos_eos(
        tags_ids,
        static_cast<int32_t>(config.text_bos_id),
        static_cast<int32_t>(config.text_eos_id),
        "tags");
    add_bos_eos(
        lyrics_ids,
        static_cast<int32_t>(config.text_bos_id),
        static_cast<int32_t>(config.text_eos_id),
        "lyrics");

    HeartMuLaPromptEncoding encoding;
    encoding.batch_size = request.options.guidance_scale != 1.0F ? 2 : 1;
    encoding.prompt_len = static_cast<int64_t>(tags_ids.size() + 1 + lyrics_ids.size());
    encoding.parallel_number = model_config.audio_num_codebooks + 1;
    encoding.tags_ids = tags_ids;
    encoding.lyrics_ids = lyrics_ids;

    const auto batch_size = static_cast<size_t>(encoding.batch_size);
    const auto prompt_len = static_cast<size_t>(encoding.prompt_len);
    const auto parallel_number = static_cast<size_t>(encoding.parallel_number);
    const auto total_token_values = batch_size * prompt_len * parallel_number;
    encoding.tokens.assign(total_token_values, config.empty_id);
    encoding.tokens_mask.assign(total_token_values, 0U);
    encoding.muq_embed.assign(batch_size * static_cast<size_t>(model_config.muq_dim), 0.0F);
    encoding.muq_idx.assign(batch_size, static_cast<int64_t>(tags_ids.size()));
    encoding.pos.resize(batch_size * prompt_len);

    const size_t text_lane = parallel_number - 1;
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < tags_ids.size(); ++i) {
            encoding.tokens[(b * prompt_len + i) * parallel_number + text_lane] = tags_ids[i];
        }
        const size_t lyrics_row_offset = tags_ids.size() + 1;
        for (size_t i = 0; i < lyrics_ids.size(); ++i) {
            encoding.tokens[(b * prompt_len + lyrics_row_offset + i) * parallel_number + text_lane] = lyrics_ids[i];
        }
        for (size_t row = 0; row < prompt_len; ++row) {
            encoding.tokens_mask[(b * prompt_len + row) * parallel_number + text_lane] = 1U;
            encoding.pos[b * prompt_len + row] = static_cast<int64_t>(row);
        }
    }
    return encoding;
}

}  // namespace engine::models::heartmula
