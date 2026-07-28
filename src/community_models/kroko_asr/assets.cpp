#include "engine/community_models/kroko_asr/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace engine::models::kroko_asr {
namespace json = engine::io::json;
namespace {

KrokoASRConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    KrokoASRConfig config;
    config.model_type = json::optional_string(root, "model_type", config.model_type);
    config.variant = json::optional_string(root, "variant", config.variant);
    config.sample_rate = json::optional_i64(root, "sample_rate", config.sample_rate);
    config.feature_dim = json::optional_i64(root, "feature_dim", config.feature_dim);
    config.chunk_size = json::optional_i64(root, "chunk_size", config.chunk_size);
    config.chunk_shift = json::optional_i64(root, "chunk_shift", config.chunk_shift);
    config.subsampling_factor = json::optional_i64(root, "subsampling_factor", config.subsampling_factor);
    config.vocab_size = json::require_i64(root, "vocab_size");
    config.context_size = json::optional_i64(root, "context_size", config.context_size);
    config.blank_id = json::optional_i64(root, "blank_id", config.blank_id);
    config.unk_id = json::optional_i64(root, "unk_id", config.unk_id);
    config.encoder_dims = json::require_i64_array(root, "encoder_dims");
    config.query_head_dims = json::require_i64_array(root, "query_head_dims");
    config.value_head_dims = json::require_i64_array(root, "value_head_dims");
    config.num_heads = json::require_i64_array(root, "num_heads");
    config.num_encoder_layers = json::require_i64_array(root, "num_encoder_layers");
    config.cnn_module_kernels = json::require_i64_array(root, "cnn_module_kernels");
    config.left_context_len = json::require_i64_array(root, "left_context_len");
    config.downsampling_factors = json::require_i64_array(root, "downsampling_factors");
    if (const auto * language = root.find("language"); language != nullptr && language->is_object()) {
        config.language = json::optional_string(*language, "iso", config.language);
    }
    return config;
}

void validate_config(const KrokoASRConfig & config) {
    if (config.model_type != "zipformer2") {
        throw std::runtime_error("Kroko ASR currently supports only Zipformer2 packages");
    }
    if (config.sample_rate != 16000 || config.feature_dim != 80) {
        throw std::runtime_error("Kroko ASR currently expects 16 kHz audio and 80-bin filterbanks");
    }
    if (config.chunk_size <= config.chunk_shift ||
        config.chunk_size != config.chunk_shift + 13 ||
        config.chunk_shift <= 0 ||
        config.chunk_shift % 4 != 0 ||
        config.subsampling_factor != 4) {
        throw std::runtime_error("Kroko ASR streaming chunk metadata is invalid");
    }
    const size_t stacks = config.encoder_dims.size();
    if (stacks == 0 ||
        config.query_head_dims.size() != stacks ||
        config.value_head_dims.size() != stacks ||
        config.num_heads.size() != stacks ||
        config.num_encoder_layers.size() != stacks ||
        config.cnn_module_kernels.size() != stacks ||
        config.left_context_len.size() != stacks ||
        config.downsampling_factors.size() != stacks) {
        throw std::runtime_error("Kroko ASR Zipformer2 stack metadata has inconsistent lengths");
    }
    if (config.downsampling_factors.front() != 1) {
        throw std::runtime_error("Kroko ASR first Zipformer2 stack must run at the base frame rate");
    }
    for (size_t stack = 0; stack < stacks; ++stack) {
        if (config.downsampling_factors[stack] <= 0 ||
            config.left_context_len[stack] * config.downsampling_factors[stack] !=
                config.left_context_len.front()) {
            throw std::runtime_error("Kroko ASR Zipformer2 downsampling metadata is inconsistent");
        }
    }
    if (config.context_size != 2 || config.blank_id != 0 || config.vocab_size <= 3) {
        throw std::runtime_error("Kroko ASR currently expects a context-2 stateless predictor with blank id 0");
    }
}

std::vector<std::string> load_tokens(const std::filesystem::path & path, int64_t vocab_size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open Kroko token table: " + path.string());
    }
    std::vector<std::string> tokens(static_cast<size_t>(vocab_size));
    std::string line;
    int64_t count = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const size_t split = line.find_last_of(' ');
        if (split == std::string::npos) {
            throw std::runtime_error("invalid Kroko token row: " + line);
        }
        const int64_t id = std::stoll(line.substr(split + 1));
        if (id < 0) {
            throw std::runtime_error("Kroko token id is negative");
        }
        // Kroko symbol tables may append k2 disambiguation symbols (#0,
        // #1, ...) after the decoder vocabulary.  They are graph-building
        // helpers and can never be emitted by the joiner.
        if (id >= vocab_size) {
            continue;
        }
        tokens[static_cast<size_t>(id)] = line.substr(0, split);
        ++count;
    }
    if (count != vocab_size) {
        throw std::runtime_error("Kroko token table size does not match vocab_size");
    }
    return tokens;
}

}  // namespace

std::shared_ptr<const KrokoASRAssets> load_kroko_asr_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(
        model_path, "kroko_asr");
    auto result = std::make_shared<KrokoASRAssets>();
    result->config = parse_config(resources);
    validate_config(result->config);
    result->tokens = load_tokens(
        resources.require_file("tokens"),
        result->config.vocab_size);
    result->weights = resources.open_tensor_source("weights");
    result->resources = std::move(resources);
    return result;
}

}  // namespace engine::models::kroko_asr
