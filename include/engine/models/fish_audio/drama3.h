#pragma once

#include "engine/framework/io/json.h"
#include "engine/framework/modules/transformers/qwen35_moe_decoder_runtime.h"
#include "engine/models/fish_audio/assets.h"
#include "engine/models/fish_audio/tokenizer_text.h"

#include <functional>
#include <memory>

namespace engine::models::fish_audio {

struct Drama3Config {
    modules::Qwen35MoeDecoderConfig backbone;
    int64_t shared_layers = 4;
    int64_t backbone_codebooks = 1;
    float tower_rotary_percent = 0.25F;
};

FishAudioConfig parse_drama3_config(const engine::io::json::Value & root);
FishAudioPrompt build_drama3_prompt(
    const FishAudioAssets & assets, const FishAudioTextTokenizer & tokenizer,
    const FishAudioRequest & request, const std::vector<codecs::FishDacCodes> & references,
    const std::optional<FishAudioConversationTurn> & previous_turn);

class Drama3ARRuntime {
public:
    Drama3ARRuntime(std::shared_ptr<const FishAudioAssets> assets, core::BackendConfig backend,
        int threads, size_t graph_arena_bytes, size_t weight_context_bytes,
        assets::TensorStorageType storage_type);
    ~Drama3ARRuntime();
    codecs::FishDacCodes generate(const FishAudioPrompt & prompt,
        const FishAudioGenerationOptions & options,
        const std::function<void(const std::vector<int32_t> &)> & on_frame);
    void release_runtime_graphs();
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::fish_audio
