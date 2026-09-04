#include "engine/community_models/vibeasr/assets.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace engine::community_models::vibeasr {
namespace {

// VibeASR's AudioVAEEncoder fixes the stride schedule in code (it is not part of
// the checkpoint), giving a total downsampling factor of 3200 samples per frame.
constexpr int64_t kDownsampleStrides[] = {1, 2, 2, 4, 5, 5, 8};
constexpr size_t kNumStages = sizeof(kDownsampleStrides) / sizeof(kDownsampleStrides[0]);

std::vector<int64_t> require_shape(
    const assets::TensorSource & source,
    const std::string & name,
    size_t expected_rank) {
    auto shape = source.require_metadata(name).shape;
    if (shape.size() != expected_rank) {
        throw std::runtime_error(
            "VibeASR VAE tensor " + name + " has rank " + std::to_string(shape.size()) +
            ", expected " + std::to_string(expected_rank));
    }
    return shape;
}

std::string block_prefix(const std::string & branch, size_t stage, size_t block) {
    return branch + ".stages." + std::to_string(stage) + "." + std::to_string(block);
}

VaeBlockConfig derive_block(const assets::TensorSource & source, const std::string & prefix) {
    VaeBlockConfig block;
    // Depthwise kernel is stored as [channels, 1, kernel_size].
    const auto mixer = require_shape(source, prefix + ".mixer.conv.conv.conv.weight", 3);
    block.channels = mixer[0];
    block.kernel_size = mixer[2];
    if (mixer[1] != 1) {
        throw std::runtime_error("VibeASR VAE mixer conv at " + prefix + " is not depthwise");
    }
    // Linear weights are stored as [out_features, in_features].
    const auto fc1 = require_shape(source, prefix + ".ffn.linear1.weight", 2);
    const auto fc2 = require_shape(source, prefix + ".ffn.linear2.weight", 2);
    block.ffn_hidden = fc1[0];
    if (fc1[1] != block.channels || fc2[0] != block.channels || fc2[1] != block.ffn_hidden) {
        throw std::runtime_error("VibeASR VAE FFN shapes at " + prefix + " are inconsistent");
    }
    return block;
}

VaeBranchConfig derive_branch(const assets::TensorSource & source, const std::string & prefix) {
    VaeBranchConfig branch;
    branch.prefix = prefix;
    branch.total_stride = 1;

    int64_t expected_in_channels = 1;  // raw mono waveform
    for (size_t stage = 0; stage < kNumStages; ++stage) {
        const std::string downsample =
            prefix + ".downsample_layers." + std::to_string(stage) + ".0.conv.conv.weight";
        if (!source.has_tensor(downsample)) {
            throw std::runtime_error("VibeASR VAE checkpoint is missing " + downsample);
        }
        // Conv weight is stored as [out_channels, in_channels, kernel_size].
        const auto shape = require_shape(source, downsample, 3);

        VaeStageConfig config;
        config.out_channels = shape[0];
        config.in_channels = shape[1];
        config.downsample_kernel_size = shape[2];
        config.downsample_stride = kDownsampleStrides[stage];
        if (config.in_channels != expected_in_channels) {
            throw std::runtime_error("VibeASR VAE stage " + std::to_string(stage) + " channel count does not chain");
        }
        if (config.downsample_kernel_size < config.downsample_stride) {
            throw std::runtime_error("VibeASR VAE stage " + std::to_string(stage) + " kernel is shorter than its stride");
        }

        for (size_t block = 0;; ++block) {
            const std::string block_name = block_prefix(prefix, stage, block);
            if (!source.has_tensor(block_name + ".norm.weight")) {
                break;
            }
            auto derived = derive_block(source, block_name);
            if (derived.channels != config.out_channels) {
                throw std::runtime_error("VibeASR VAE block " + block_name + " width does not match its stage");
            }
            config.blocks.push_back(derived);
        }
        if (config.blocks.empty()) {
            throw std::runtime_error("VibeASR VAE stage " + std::to_string(stage) + " has no blocks");
        }

        branch.total_stride *= config.downsample_stride;
        expected_in_channels = config.out_channels;
        branch.stages.push_back(std::move(config));
    }

    const auto head = require_shape(source, prefix + ".head.conv.conv.weight", 3);
    branch.latent_dim = head[0];
    branch.head_kernel_size = head[2];
    if (head[1] != expected_in_channels) {
        throw std::runtime_error("VibeASR VAE head input width does not match the last stage");
    }

    const auto fc1 = require_shape(source, prefix + "_connector.fc1.weight", 2);
    const auto fc2 = require_shape(source, prefix + "_connector.fc2.weight", 2);
    branch.connector_hidden = fc1[0];
    if (fc1[1] != branch.latent_dim || fc2[0] != branch.connector_hidden ||
        fc2[1] != branch.connector_hidden) {
        throw std::runtime_error("VibeASR VAE " + prefix + " connector shapes are inconsistent");
    }
    return branch;
}

}  // namespace

int64_t VaeBranchConfig::frames_for_samples(int64_t num_samples) const {
    // Every stage is a causal conv with left padding kernel_size - stride, so
    // its output length is ggml_calc_conv_output_size() with that padding.
    int64_t length = num_samples;
    for (const auto & stage : stages) {
        const int64_t padding = stage.downsample_kernel_size - stage.downsample_stride;
        length = (length + padding - stage.downsample_kernel_size) / stage.downsample_stride + 1;
        if (length <= 0) {
            return 0;
        }
    }
    return length;
}

VibeASRVaeConfig derive_vae_config(const assets::TensorSource & source) {
    VibeASRVaeConfig config;
    config.acoustic = derive_branch(source, "acoustic");
    config.semantic = derive_branch(source, "semantic");
    if (config.acoustic.connector_hidden != config.semantic.connector_hidden) {
        throw std::runtime_error("VibeASR VAE branches disagree on the connector width");
    }
    return config;
}

std::shared_ptr<const VibeASRVaeAssets> load_vibeasr_vae_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<VibeASRVaeAssets>();
    assets->source = engine::assets::open_tensor_source(model_path);
    assets->config = derive_vae_config(*assets->source);
    return assets;
}

}  // namespace engine::community_models::vibeasr
