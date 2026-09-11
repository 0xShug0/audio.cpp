#include "engine/models/moss/moss_tts_nano/tokenizer_text.h"

#include <stdexcept>
#include <utility>

namespace engine::models::moss_tts_nano {

MossTTSNanoTextTokenizer::MossTTSNanoTextTokenizer(std::shared_ptr<const MossTTSNanoAssets> assets)
    : assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS-TTS-Nano tokenizer requires assets");
    }
    pieces_ = engine::tokenizers::load_sentencepiece_model(assets_->resources.require_file("tokenizer_model"));
}

std::vector<int32_t> MossTTSNanoTextTokenizer::encode(const std::string & text) const {
    return engine::tokenizers::tokenize_sentencepiece(pieces_, text);
}

std::string MossTTSNanoTextTokenizer::decode(const std::vector<int32_t> & token_ids) const {
    return engine::tokenizers::decode_sentencepiece(pieces_, token_ids);
}

}  // namespace engine::models::moss_tts_nano
