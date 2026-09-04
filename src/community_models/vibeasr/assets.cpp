#include "engine/community_models/vibeasr/assets.h"

#include "engine/framework/model_spec/package.h"

#include <gguf.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

// assets::TensorSource exposes tensors, not the GGUF KV block, and the decoder
// geometry lives entirely in the KV block. Reading it directly is what the other
// community entries do (see sense_asr/assets.cpp).
class GgufMetadataReader {
public:
    explicit GgufMetadataReader(const std::filesystem::path & path) {
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx = nullptr;
        gguf_context * gguf = gguf_init_from_file(path.string().c_str(), params);
        if (gguf == nullptr) {
            throw std::runtime_error("Failed to read VibeASR GGUF metadata from " + path.string());
        }
        ctx_.reset(gguf);
    }

    int64_t require_u32(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0) {
            throw std::runtime_error(std::string("VibeASR LM GGUF is missing ") + key);
        }
        return static_cast<int64_t>(gguf_get_val_u32(ctx_.get(), id));
    }

    float require_f32(const char * key) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        if (id < 0) {
            throw std::runtime_error(std::string("VibeASR LM GGUF is missing ") + key);
        }
        return gguf_get_val_f32(ctx_.get(), id);
    }

    std::string kv_str(const char * key, std::string fallback) const {
        const int64_t id = gguf_find_key(ctx_.get(), key);
        return id < 0 ? std::move(fallback) : std::string(gguf_get_val_str(ctx_.get(), id));
    }

private:
    struct GgufDeleter {
        void operator()(gguf_context * ctx) const noexcept {
            if (ctx != nullptr) {
                gguf_free(ctx);
            }
        }
    };

    std::unique_ptr<gguf_context, GgufDeleter> ctx_;
};

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
    return make_vibeasr_vae_assets(engine::assets::open_tensor_source(model_path));
}

std::shared_ptr<const VibeASRVaeAssets> make_vibeasr_vae_assets(
    std::shared_ptr<const assets::TensorSource> source) {
    auto assets = std::make_shared<VibeASRVaeAssets>();
    assets->config = derive_vae_config(*source);
    assets->source = std::move(source);
    return assets;
}

VibeASRLmConfig derive_lm_config(const assets::TensorSource & source) {
    const GgufMetadataReader reader(source.source_path());

    const std::string architecture = reader.kv_str("general.architecture", "");
    if (architecture != "qwen2") {
        throw std::runtime_error(
            "VibeASR LM GGUF declares architecture '" + architecture + "', expected qwen2");
    }

    VibeASRLmConfig config;
    config.vocab_size = reader.require_u32("qwen2.vocab_size");
    config.hidden_size = reader.require_u32("qwen2.embedding_length");
    config.intermediate_size = reader.require_u32("qwen2.feed_forward_length");
    config.num_hidden_layers = reader.require_u32("qwen2.block_count");
    config.num_attention_heads = reader.require_u32("qwen2.attention.head_count");
    config.num_key_value_heads = reader.require_u32("qwen2.attention.head_count_kv");
    config.max_position_embeddings = reader.require_u32("qwen2.context_length");
    // The checkpoint has no attention.key_length: Qwen2 stores the per-head width
    // only as the RoPE dimension count, which for this model equals
    // embedding_length / head_count.
    config.head_dim = reader.require_u32("qwen2.rope.dimension_count");
    config.rms_norm_eps = reader.require_f32("qwen2.attention.layer_norm_rms_epsilon");
    config.rope_theta = reader.require_f32("qwen2.rope.freq_base");

    if (config.head_dim * config.num_attention_heads != config.hidden_size) {
        throw std::runtime_error("VibeASR LM head_dim * head_count does not match embedding_length");
    }
    if (config.num_key_value_heads <= 0 || config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error("VibeASR LM head_count is not a multiple of head_count_kv");
    }
    if (config.num_hidden_layers <= 0) {
        throw std::runtime_error("VibeASR LM declares no layers");
    }

    // Cross-check the metadata against the one tensor whose shape pins both dims.
    const auto embedding = source.require_metadata("token_embd.weight").shape;
    if (embedding.size() != 2 || embedding[0] != config.vocab_size || embedding[1] != config.hidden_size) {
        throw std::runtime_error("VibeASR LM token_embd.weight does not match the declared geometry");
    }
    return config;
}

std::shared_ptr<const VibeASRAssets> load_vibeasr_assets(const std::filesystem::path & model_path) {
    auto assets = std::make_shared<VibeASRAssets>();
    assets->resources = engine::model_spec::load_resource_bundle_for_family(model_path, "vibeasr");

    // A GGUF still carrying the VibeASR fork's type ids (36/37) fails deep inside
    // the reader with an unhelpful message, so name the fix here.
    auto open = [&assets](const char * id) {
        try {
            return assets->resources.open_tensor_source(id);
        } catch (const std::exception & error) {
            throw std::runtime_error(
                std::string("VibeASR could not open the '") + id + "' GGUF (" + error.what() +
                "). If it came straight from huggingface.co/XsquirrelC/VibeVoice-ASR-BitNet, run "
                "tools/community_models/convert_vibeasr_gguf.py --in-place on it first.");
        }
    };

    assets->vae = make_vibeasr_vae_assets(open("vae_weights"));
    assets->lm_weights = open("lm_weights");
    assets->lm = derive_lm_config(*assets->lm_weights);
    return assets;
}

}  // namespace engine::community_models::vibeasr
