#include "engine/models/cohere_asr/model.h"

#include "engine/framework/model_spec/package.h"

#include <stdexcept>

namespace engine::models::cohere_asr {

int32_t CohereAssets::special_token(const std::string & text) const {
    for (const auto & piece : vocabulary) {
        if (piece.text == text) {
            return piece.id;
        }
    }
    throw std::runtime_error("Cohere tokenizer is missing " + text);
}

std::shared_ptr<const CohereAssets> load_cohere_assets(const std::filesystem::path & path) {
    auto out = std::make_shared<CohereAssets>();
    out->resources = model_spec::load_resource_bundle(path, model_spec::default_spec_path("cohere_asr"));
    out->source = out->resources.open_tensor_source("weights");
    const auto config = out->resources.parse_json("config");
    const auto & encoder = config.require("encoder");
    const auto & decoder = config.require("transf_decoder").require("config_dict");
    if (io::json::require_string(config, "prompt_format") != "cohere_asr" ||
        io::json::require_i64(encoder, "d_model") != 1280 ||
        io::json::require_i64(encoder, "n_layers") != 48 ||
        io::json::require_i64(encoder, "subsampling_factor") != 8 ||
        io::json::require_i64(encoder, "conv_kernel_size") != 9 ||
        io::json::require_i64(decoder, "hidden_size") != 1024 ||
        io::json::require_i64(decoder, "num_layers") != 8 ||
        !io::json::require_bool(decoder, "pre_ln")) {
        throw std::runtime_error("Cohere ASR requires the cohere-transcribe-03-2026 architecture");
    }
    out->vocabulary = tokenizers::load_sentencepiece_model(out->resources.require_file("tokenizer"));
    if (out->vocabulary.size() != 16384) {
        throw std::runtime_error("Cohere vocabulary must contain 16384 entries");
    }
    // The official Transformers processor builds F32 frontend constants rather than
    // using the checkpoint's BF16 copies of the training preprocessor buffers.
    out->window = audio::get_cached_stft_window({512, 160, 400, true, audio::STFTPadMode::Constant});
    out->filterbank = audio::MelFilterbank().build_sparse({16000, 512, 128, 0.0f, 8000.0f, true});
    return out;
}

}  // namespace engine::models::cohere_asr
