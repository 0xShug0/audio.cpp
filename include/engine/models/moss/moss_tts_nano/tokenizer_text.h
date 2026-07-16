#pragma once

#include "engine/framework/tokenizers/sentencepiece.h"
#include "engine/models/moss/moss_tts_nano/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::moss_tts_nano {

class MossTTSNanoTextTokenizer {
public:
    explicit MossTTSNanoTextTokenizer(std::shared_ptr<const MossTTSNanoAssets> assets);

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & token_ids) const;

private:
    std::shared_ptr<const MossTTSNanoAssets> assets_;
    std::vector<engine::tokenizers::SentencePiecePiece> pieces_;
};

}  // namespace engine::models::moss_tts_nano
