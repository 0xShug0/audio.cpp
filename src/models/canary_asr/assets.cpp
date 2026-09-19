#include "engine/models/canary_asr/model.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::canary_asr {

int32_t CanaryAssets::special_token(const std::string & text) const {
    for (const auto & piece : vocabulary) {
        if (piece.text == text) {
            return piece.id;
        }
    }
    throw std::runtime_error("Canary tokenizer is missing " + text);
}

std::shared_ptr<const CanaryAssets> load_canary_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<CanaryAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("canary_asr"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    const auto & encoder = config.require("encoder");
    const auto & decoder = config.require("transf_decoder").require("config_dict");
    if (io::json::require_string(config, "prompt_format") != "canary2" ||
        io::json::require_i64(encoder, "d_model") != 512 ||
        io::json::require_i64(encoder, "n_layers") != 17 ||
        io::json::require_i64(encoder, "subsampling_factor") != 8 ||
        io::json::require_i64(encoder, "conv_kernel_size") != 9 ||
        io::json::require_i64(decoder, "hidden_size") != 1024 ||
        io::json::require_i64(decoder, "num_layers") != 4 ||
        !io::json::require_bool(decoder, "pre_ln")) {
        throw std::runtime_error("Canary ASR requires the Canary 180M Flash architecture");
    }
    for (const auto & item : config.require("audio_cpp_tokenizers").as_array()) {
        const auto language = io::json::require_string(item, "language");
        const auto offset = io::json::require_i64(item, "offset");
        auto pieces = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer_" + language));
        if (offset != static_cast<int64_t>(out->vocabulary.size()) ||
            static_cast<int64_t>(pieces.size()) != io::json::require_i64(item, "size")) {
            throw std::runtime_error("Canary tokenizer offsets do not match the aggregate vocabulary");
        }
        for (auto & piece : pieces) {
            piece.id += static_cast<int>(offset);
            out->vocabulary.push_back(std::move(piece));
        }
    }
    if (out->vocabulary.size() != 5248) {
        throw std::runtime_error("Canary vocabulary must contain 5248 entries");
    }
    out->window = out->source->require_f32("preprocessor.featurizer.window", {400});
    out->filterbank = {out->source->require_f32("preprocessor.featurizer.fb", {1, 128, 257}), {128, 257}};
    return out;
}

}  // namespace engine::models::canary_asr
