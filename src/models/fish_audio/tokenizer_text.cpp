#include "engine/models/fish_audio/tokenizer_text.h"

#include "engine/framework/tokenizers/llama_bpe.h"
#include "engine/framework/runtime/errors.h"
#include <utf8proc.h>

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace engine::models::fish_audio {
namespace {

int32_t require_token_id(const engine::tokenizers::LlamaBpeTokenizer & tokenizer, const std::string & token) {
    const auto id = tokenizer.find_token_id(token);
    if (!id.has_value()) {
        throw std::runtime_error("Fish Audio tokenizer missing token: " + token);
    }
    return *id;
}

}  // namespace

struct FishAudioTextTokenizer::Impl {
    explicit Impl(std::shared_ptr<const FishAudioAssets> input_assets)
        : assets(std::move(input_assets)),
          tokenizer(engine::tokenizers::LlamaBpeTokenizerSpec{
              {},
              {},
              assets->resources.require_file("tokenizer_config"),
              assets->resources.require_file("tokenizer_json"),
              assets->config.drama3 ? engine::tokenizers::LlamaBpePreTokenizer::Qwen35 :
                  engine::tokenizers::LlamaBpePreTokenizer::Qwen2,
          }),
          im_end(require_token_id(tokenizer, "<|im_end|>")),
          semantic_begin(static_cast<int32_t>(assets->config.semantic_start_token_id)),
          semantic_end(static_cast<int32_t>(assets->config.semantic_end_token_id)) {}

    std::shared_ptr<const FishAudioAssets> assets;
    engine::tokenizers::LlamaBpeTokenizer tokenizer;
    int32_t im_end = 0;
    int32_t semantic_begin = 0;
    int32_t semantic_end = 0;
};

FishAudioTextTokenizer::FishAudioTextTokenizer(std::shared_ptr<const FishAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Fish Audio text tokenizer requires assets");
    }
    impl_ = std::make_shared<Impl>(std::move(assets));
}

std::vector<int32_t> FishAudioTextTokenizer::encode(const std::string & text) const {
    if(impl_->assets->config.drama3) {
        utf8proc_uint8_t * output=nullptr;
        const auto size=utf8proc_map(reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
            static_cast<utf8proc_ssize_t>(text.size()),&output,
            static_cast<utf8proc_option_t>(UTF8PROC_STABLE|UTF8PROC_COMPOSE));
        std::unique_ptr<utf8proc_uint8_t,decltype(&std::free)> owned(output,&std::free);
        if(size<0)throw runtime::InvalidRequestError("Drama 3 text normalization: "+std::string(utf8proc_errmsg(size)));
        return impl_->tokenizer.encode(std::string(reinterpret_cast<const char *>(output),static_cast<size_t>(size)),true);
    }
    return impl_->tokenizer.encode(text, true);
}

int32_t FishAudioTextTokenizer::token_id(const std::string & token) const {
    return require_token_id(impl_->tokenizer, token);
}

int32_t FishAudioTextTokenizer::im_end_id() const noexcept {
    return impl_->im_end;
}

int32_t FishAudioTextTokenizer::semantic_begin_id() const noexcept {
    return impl_->semantic_begin;
}

int32_t FishAudioTextTokenizer::semantic_end_id() const noexcept {
    return impl_->semantic_end;
}

}  // namespace engine::models::fish_audio
