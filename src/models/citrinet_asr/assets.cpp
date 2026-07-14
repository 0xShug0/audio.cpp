#include "engine/models/citrinet_asr/assets.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/assets/weight_metadata.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::citrinet_asr {
namespace io = engine::io;
namespace asset_meta = engine::assets;

namespace {

std::vector<JasperBlockConfig> parse_jasper_config(const io::json::Value & value) {
    std::vector<JasperBlockConfig> blocks;
    for (const auto & block_value : value.as_array()) {
        const auto & object = block_value.as_object();
        JasperBlockConfig cfg;
        if (const auto it = object.find("filters"); it != object.end()) {
            cfg.filters = it->second.as_i64();
        }
        if (const auto it = object.find("repeat"); it != object.end()) {
            cfg.repeat = it->second.as_i64();
        }
        if (const auto it = object.find("kernel"); it != object.end()) {
            cfg.kernel = it->second.as_i64();
        }
        if (const auto it = object.find("stride"); it != object.end()) {
            cfg.stride = it->second.as_i64();
        }
        if (const auto it = object.find("dilation"); it != object.end()) {
            cfg.dilation = it->second.as_i64();
        }
        if (const auto it = object.find("dropout"); it != object.end()) {
            cfg.dropout = it->second.as_f32();
        }
        if (const auto it = object.find("residual"); it != object.end()) {
            cfg.residual = it->second.as_bool();
        }
        if (const auto it = object.find("residual_mode"); it != object.end() && !it->second.is_null()) {
            cfg.residual_mode = it->second.as_string();
        }
        if (const auto it = object.find("separable"); it != object.end()) {
            cfg.separable = it->second.as_bool();
        }
        if (const auto it = object.find("se"); it != object.end()) {
            cfg.se = it->second.as_bool();
        }
        if (const auto it = object.find("se_reduction_ratio"); it != object.end()) {
            cfg.se_reduction_ratio = it->second.as_i64();
        }
        blocks.push_back(std::move(cfg));
    }
    return blocks;
}

CitrinetConfig parse_config(const io::json::Value & root) {
    CitrinetConfig cfg;
    cfg.sample_rate = root.require("sample_rate").as_i64();
    cfg.n_mels = root.require("n_mels").as_i64();
    cfg.n_fft = root.require("n_fft").as_i64();
    cfg.hop_length = static_cast<int64_t>(std::llround(root.require("window_stride").as_number() * static_cast<double>(cfg.sample_rate)));
    cfg.win_length = static_cast<int64_t>(std::llround(root.require("window_size").as_number() * static_cast<double>(cfg.sample_rate)));
    cfg.pad_to = root.require("pad_to").as_i64();
    cfg.vocab_size = root.require("vocab_size").as_i64();
    cfg.num_classes = root.require("num_classes").as_i64();
    cfg.blank_id = root.require("blank_id").as_i64();
    cfg.window = root.require("window").as_string();
    cfg.normalize = root.require("normalize").as_string();
    cfg.jasper = parse_jasper_config(root.require("jasper"));
    if (cfg.window != "hann") {
        throw std::runtime_error("unsupported Citrinet window: " + cfg.window);
    }
    if (cfg.normalize != "per_feature") {
        throw std::runtime_error("unsupported Citrinet normalize mode: " + cfg.normalize);
    }
    if (cfg.sample_rate <= 0 || cfg.n_mels <= 0 || cfg.n_fft <= 0 || cfg.hop_length <= 0 || cfg.win_length <= 0) {
        throw std::runtime_error("invalid Citrinet metadata values");
    }
    if (cfg.pad_to <= 0) {
        throw std::runtime_error("invalid Citrinet pad_to metadata");
    }
    if (cfg.vocab_size <= 0 || cfg.num_classes <= 0) {
        throw std::runtime_error("invalid Citrinet vocab metadata");
    }
    if (cfg.blank_id < 0 || cfg.blank_id >= cfg.num_classes) {
        throw std::runtime_error("invalid Citrinet blank_id metadata");
    }
    if (cfg.num_classes != cfg.vocab_size + 1) {
        throw std::runtime_error("Citrinet num_classes must equal vocab_size + 1");
    }
    if (cfg.jasper.empty()) {
        throw std::runtime_error("empty Citrinet jasper config");
    }
    cfg.output_stride = 1;
    for (const auto & block : cfg.jasper) {
        if (block.filters <= 0 || block.repeat <= 0 || block.kernel <= 0 || block.stride <= 0 || block.dilation <= 0) {
            throw std::runtime_error("invalid Citrinet jasper block metadata");
        }
        cfg.output_stride *= block.stride;
    }
    return cfg;
}

CitrinetWeights load_citrinet_weights(engine::assets::ResourceBundle resources) {
    CitrinetWeights weights;
    const auto source = resources.open_tensor_source("weights");
    weights.source = source;
    weights.config = parse_config(resources.parse_json("config"));
    weights.window = source->require_f32("preprocessor.featurizer.window", {weights.config.win_length});
    weights.fb = source->require_f32(
        "preprocessor.featurizer.fb",
        {1, weights.config.n_mels, weights.config.n_fft / 2 + 1});

    const auto tokenizer_model_path = resources.require_file("tokenizer");
    weights.tokenizer_pieces = tokenizers::load_sentencepiece_model(tokenizer_model_path);
    if (static_cast<int64_t>(weights.tokenizer_pieces.size()) != weights.config.vocab_size) {
        throw std::runtime_error(
            "Citrinet tokenizer model has " + std::to_string(weights.tokenizer_pieces.size()) +
            " pieces but the config expects vocab_size " + std::to_string(weights.config.vocab_size) +
            ": " + tokenizer_model_path.string());
    }

    weights.blocks.resize(weights.config.jasper.size());
    int64_t in_channels = weights.config.n_mels;
    for (size_t block_index = 0; block_index < weights.config.jasper.size(); ++block_index) {
        const auto & block_cfg = weights.config.jasper[block_index];
        auto & block = weights.blocks[block_index];
        block.separable = block_cfg.separable;
        if (block_cfg.separable) {
            block.separable_repeats.reserve(static_cast<size_t>(block_cfg.repeat));
            int64_t repeat_in = in_channels;
            for (int64_t repeat = 0; repeat < block_cfg.repeat; ++repeat) {
                const int base = static_cast<int>(repeat * 5);
                const int64_t stride = (repeat + 1 == block_cfg.repeat) ? block_cfg.stride : 1;
                SeparableConvBn layer;
                layer.depthwise = asset_meta::load_conv1d_metadata<Conv1dWeights>(
                    *source,
                    "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(base) + ".conv",
                    stride,
                    block_cfg.dilation,
                    (block_cfg.dilation * (block_cfg.kernel - 1)) / 2,
                    repeat_in,
                    false);
                layer.pointwise = asset_meta::load_conv1d_metadata<Conv1dWeights>(
                    *source,
                    "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(base + 1) + ".conv",
                    1,
                    1,
                    0,
                    1,
                    false);
                layer.bn = asset_meta::load_batch_norm1d_metadata<BatchNorm1dWeights>(*source, "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(base + 2));
                block.separable_repeats.push_back(std::move(layer));
                repeat_in = block_cfg.filters;
            }
        } else {
            block.conv_repeats.reserve(static_cast<size_t>(block_cfg.repeat));
            for (int64_t repeat = 0; repeat < block_cfg.repeat; ++repeat) {
                const int base = static_cast<int>(repeat * 3);
                const int64_t stride = (repeat + 1 == block_cfg.repeat) ? block_cfg.stride : 1;
                ConvBn layer;
                layer.conv = asset_meta::load_conv1d_metadata<Conv1dWeights>(
                    *source,
                    "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(base) + ".conv",
                    stride,
                    block_cfg.dilation,
                    (block_cfg.dilation * (block_cfg.kernel - 1)) / 2,
                    1,
                    false);
                layer.bn = asset_meta::load_batch_norm1d_metadata<BatchNorm1dWeights>(*source, "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(base + 1));
                block.conv_repeats.push_back(std::move(layer));
            }
        }
        block.has_residual = block_cfg.residual;
        if (block.has_residual) {
            const int64_t residual_stride = block_cfg.residual_mode == "stride_add" ? block_cfg.stride : 1;
            block.residual_conv = asset_meta::load_conv1d_metadata<Conv1dWeights>(
                *source,
                "encoder.encoder." + std::to_string(block_index) + ".res.0.0.conv",
                residual_stride,
                1,
                0,
                1,
                false);
            block.residual_bn = asset_meta::load_batch_norm1d_metadata<BatchNorm1dWeights>(*source, "encoder.encoder." + std::to_string(block_index) + ".res.0.1");
        }
        block.has_se = block_cfg.se;
        if (block.has_se) {
            const int se_index = static_cast<int>(block_cfg.repeat == 1 ? 3 : block_cfg.repeat * 5 - 2);
            block.se.fc1 = asset_meta::load_linear_as_conv1d_metadata<Conv1dWeights>(
                *source,
                "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(se_index) + ".fc.0",
                false);
            block.se.fc2 = asset_meta::load_linear_as_conv1d_metadata<Conv1dWeights>(
                *source,
                "encoder.encoder." + std::to_string(block_index) + ".mconv." + std::to_string(se_index) + ".fc.2",
                false);
        }
        in_channels = block_cfg.filters;
    }

    weights.decoder = asset_meta::load_conv1d_metadata<Conv1dWeights>(*source, "decoder.decoder_layers.0", 1, 1, 0, 1, true);
    return weights;
}

std::string checkpoint_cache_key(const std::filesystem::path & checkpoint_path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(checkpoint_path, ec);
    return ec ? checkpoint_path.lexically_normal().string() : canonical.string();
}

}  // namespace

std::shared_ptr<const CitrinetWeights> load_citrinet_weights_cached(const std::filesystem::path & model_path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<const CitrinetWeights>> cache;
    auto resources = engine::assets::load_resource_bundle_from_package_spec(
        model_path,
        engine::assets::default_model_package_spec_path("citrinet_asr"));
    const auto key = checkpoint_cache_key(resources.require_file("weights"));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
    }
    auto loaded = std::make_shared<const CitrinetWeights>(load_citrinet_weights(std::move(resources)));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = loaded;
    }
    return loaded;
}

}  // namespace engine::models::citrinet_asr
