#include "engine/community_models/zipvoice/weights.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/io/json.h"

#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace engine::models::zipvoice {
namespace {

std::string read_file(const std::filesystem::path & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::vector<int> require_int_list(const io::json::Value & object, const std::string & key) {
    const auto & array = object.require(key);
    std::vector<int> values;
    for (const auto & item : array.as_array()) {
        values.push_back(static_cast<int>(item.as_number()));
    }
    return values;
}

}  // namespace

ZipVoiceConfig load_zipvoice_config(
    const std::filesystem::path & model_dir,
    const engine::assets::TensorSource * probe,
    const std::filesystem::path * config_path) {
    ZipVoiceConfig config;
    const auto config_path_resolved =
        config_path != nullptr ? *config_path : model_dir / "model.json";
    if (std::filesystem::is_regular_file(config_path_resolved)) {
        const auto root = io::json::parse_file(config_path_resolved);
        const auto & model = root.require("model");
        config.fm_downsampling_factor = require_int_list(model, "fm_decoder_downsampling_factor");
        config.fm_num_layers = require_int_list(model, "fm_decoder_num_layers");
        config.fm_cnn_kernel = require_int_list(model, "fm_decoder_cnn_module_kernel");
        config.fm_feedforward_dim = io::json::require_i32(model, "fm_decoder_feedforward_dim");
        config.fm_num_heads = io::json::require_i32(model, "fm_decoder_num_heads");
        config.fm_dim = io::json::require_i32(model, "fm_decoder_dim");
        config.text_num_layers = io::json::require_i32(model, "text_encoder_num_layers");
        config.text_feedforward_dim = io::json::require_i32(model, "text_encoder_feedforward_dim");
        config.text_cnn_kernel = io::json::require_i32(model, "text_encoder_cnn_module_kernel");
        config.text_num_heads = io::json::require_i32(model, "text_encoder_num_heads");
        config.text_dim = io::json::require_i32(model, "text_encoder_dim");
        config.query_head_dim = io::json::require_i32(model, "query_head_dim");
        config.value_head_dim = io::json::require_i32(model, "value_head_dim");
        config.pos_head_dim = io::json::require_i32(model, "pos_head_dim");
        config.pos_dim = io::json::require_i32(model, "pos_dim");
        config.time_embed_dim = io::json::require_i32(model, "time_embed_dim");
        config.text_embed_dim = io::json::require_i32(model, "text_embed_dim");
        config.feat_dim = io::json::require_i32(model, "feat_dim");
        const auto * feature = root.find("feature");
        if (feature != nullptr) {
            config.sampling_rate = io::json::require_i32(*feature, "sampling_rate");
        }
    } else {
        // Defaults matching the released zipvoice / zipvoice_distill config.
    }
    if (config.fm_downsampling_factor.size() != config.fm_num_layers.size() ||
        config.fm_downsampling_factor.size() != config.fm_cnn_kernel.size()) {
        throw std::runtime_error("zipvoice: stack config length mismatch in model.json");
    }
    if (probe != nullptr) {
        config.guidance_scale_embed = probe->has_tensor("fm_decoder.guidance_scale_embed.weight");
    }
    return config;
}

ZipVoiceWeights load_zipvoice_weights(
    const engine::assets::TensorSource & raw_source,
    const std::string & prefix,
    const ZipVoiceConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type) {
    std::shared_ptr<const engine::assets::TensorSource> source =
        engine::assets::make_prefixed_tensor_source(
            std::shared_ptr<const engine::assets::TensorSource>(
                std::shared_ptr<const engine::assets::TensorSource>(), &raw_source),
            prefix);

    ZipVoiceWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "zipvoice.weights", 2ULL * 1024ULL * 1024ULL * 1024ULL);

    // Dimension arguments are PHYSICAL ggml order (ne0 fastest). -1 accepts
    // any value; non-wildcard entries are validated against the checkpoint
    // metadata (which stores torch order, reversed here).
    const auto t2 = [&](const std::string & name, int64_t d0, int64_t d1) {
        const auto meta = source->require_metadata(name);
        if (meta.shape.size() != 2) {
            throw std::runtime_error("zipvoice: " + name + " must be rank 2");
        }
        if ((d0 >= 0 && d0 != meta.shape[1]) || (d1 >= 0 && d1 != meta.shape[0])) {
            throw std::runtime_error("zipvoice: " + name + " shape mismatch");
        }
        // Linear and embedding weights can retain their GGUF quantization.
        return weights.store->load_tensor(
            *source, name, engine::assets::TensorStorageType::Native, meta.shape);
    };
    const auto t1 = [&](const std::string & name, int64_t d0) {
        const auto meta = source->require_metadata(name);
        if (meta.shape.size() != 1) {
            throw std::runtime_error("zipvoice: " + name + " must be rank 1");
        }
        if (d0 >= 0 && d0 != meta.shape[0]) {
            throw std::runtime_error("zipvoice: " + name + " shape mismatch");
        }
        return weights.store->load_f32_tensor(*source, name, meta.shape);
    };
    const auto t3 = [&](const std::string & name, int64_t d0, int64_t d1, int64_t d2) {
        const auto meta = source->require_metadata(name);
        if (meta.shape.size() != 3) {
            throw std::runtime_error("zipvoice: " + name + " must be rank 3");
        }
        if ((d0 >= 0 && d0 != meta.shape[2]) || (d1 >= 0 && d1 != meta.shape[1]) ||
            (d2 >= 0 && d2 != meta.shape[0])) {
            throw std::runtime_error("zipvoice: " + name + " shape mismatch");
        }
        return weights.store->load_f32_tensor(*source, name, meta.shape);
    };

    const auto load_zipformer = [&](TTSZipformerWeights & w, const std::string & base,
                                    int encoder_dim, int feedforward_dim,
                                    const std::vector<int> & downsampling,
                                    const std::vector<int> & num_layers,
                                    const std::vector<int> & kernels,
                                    int num_heads, bool with_time) {
        w.in_proj_w = t2(base + ".in_proj.weight", -1, encoder_dim);
        w.in_proj_b = t1(base + ".in_proj.bias", encoder_dim);
        w.out_proj_w = t2(base + ".out_proj.weight", -1, -1);
        w.out_proj_b = t1(base + ".out_proj.bias", -1);
        if (with_time) {
            w.time_mlp0_w = t2(base + ".time_embed.0.weight", config.time_embed_dim, config.time_embed_dim * 2);
            w.time_mlp0_b = t1(base + ".time_embed.0.bias", config.time_embed_dim * 2);
            w.time_mlp2_w = t2(base + ".time_embed.2.weight", config.time_embed_dim * 2, config.time_embed_dim);
            w.time_mlp2_b = t1(base + ".time_embed.2.bias", config.time_embed_dim);
            if (config.guidance_scale_embed) {
                w.guidance_embed_w = t2(base + ".guidance_scale_embed.weight", config.time_embed_dim, config.time_embed_dim);
            }
        }
        const size_t num_stacks = downsampling.size();
        w.stacks.resize(num_stacks);
        for (size_t s = 0; s < num_stacks; ++s) {
            auto & stack = w.stacks[s];
            const std::string sp = base + ".encoders." + std::to_string(s);
            stack.layers.resize(static_cast<size_t>(num_layers[s]));
            for (size_t l = 0; l < stack.layers.size(); ++l) {
                auto & layer = stack.layers[l];
                const std::string lp = sp + (downsampling[s] > 1 ? ".encoder" : "") +
                                       ".layers." + std::to_string(l);
                layer.bypass_scale = t1(lp + ".bypass.bypass_scale", encoder_dim);
                layer.bypass_mid_scale = t1(lp + ".bypass_mid.bypass_scale", encoder_dim);
                layer.attn_in_proj_w = t2(lp + ".self_attn_weights.in_proj.weight", encoder_dim, -1);
                layer.attn_in_proj_b = t1(lp + ".self_attn_weights.in_proj.bias", -1);
                layer.linear_pos_w = t2(lp + ".self_attn_weights.linear_pos.weight",
                                        config.pos_dim, num_heads * config.pos_head_dim);
                layer.sa1_in_w = t2(lp + ".self_attn1.in_proj.weight", encoder_dim, num_heads * config.value_head_dim);
                layer.sa1_in_b = t1(lp + ".self_attn1.in_proj.bias", num_heads * config.value_head_dim);
                layer.sa1_out_w = t2(lp + ".self_attn1.out_proj.weight", num_heads * config.value_head_dim, encoder_dim);
                layer.sa1_out_b = t1(lp + ".self_attn1.out_proj.bias", encoder_dim);
                layer.sa2_in_w = t2(lp + ".self_attn2.in_proj.weight", encoder_dim, num_heads * config.value_head_dim);
                layer.sa2_in_b = t1(lp + ".self_attn2.in_proj.bias", num_heads * config.value_head_dim);
                layer.sa2_out_w = t2(lp + ".self_attn2.out_proj.weight", num_heads * config.value_head_dim, encoder_dim);
                layer.sa2_out_b = t1(lp + ".self_attn2.out_proj.bias", encoder_dim);
                const int ff1_hidden = feedforward_dim * 3 / 4;
                const int ff3_hidden = feedforward_dim * 5 / 4;
                layer.ff1_in_w = t2(lp + ".feed_forward1.in_proj.weight", encoder_dim, ff1_hidden);
                layer.ff1_in_b = t1(lp + ".feed_forward1.in_proj.bias", ff1_hidden);
                layer.ff1_out_w = t2(lp + ".feed_forward1.out_proj.weight", ff1_hidden, encoder_dim);
                layer.ff1_out_b = t1(lp + ".feed_forward1.out_proj.bias", encoder_dim);
                layer.ff2_in_w = t2(lp + ".feed_forward2.in_proj.weight", encoder_dim, feedforward_dim);
                layer.ff2_in_b = t1(lp + ".feed_forward2.in_proj.bias", feedforward_dim);
                layer.ff2_out_w = t2(lp + ".feed_forward2.out_proj.weight", feedforward_dim, encoder_dim);
                layer.ff2_out_b = t1(lp + ".feed_forward2.out_proj.bias", encoder_dim);
                layer.ff3_in_w = t2(lp + ".feed_forward3.in_proj.weight", encoder_dim, ff3_hidden);
                layer.ff3_in_b = t1(lp + ".feed_forward3.in_proj.bias", ff3_hidden);
                layer.ff3_out_w = t2(lp + ".feed_forward3.out_proj.weight", ff3_hidden, encoder_dim);
                layer.ff3_out_b = t1(lp + ".feed_forward3.out_proj.bias", encoder_dim);
                const int na_hidden = encoder_dim * 3 / 4;
                layer.na_in_w = t2(lp + ".nonlin_attention.in_proj.weight", encoder_dim, na_hidden * 3);
                layer.na_in_b = t1(lp + ".nonlin_attention.in_proj.bias", na_hidden * 3);
                layer.na_out_w = t2(lp + ".nonlin_attention.out_proj.weight", na_hidden, encoder_dim);
                layer.na_out_b = t1(lp + ".nonlin_attention.out_proj.bias", encoder_dim);
                const int kernel = kernels[s];
                for (int m = 1; m <= 2; ++m) {
                    const std::string mp = lp + ".conv_module" + std::to_string(m);
                    core::TensorValue &in_w = m == 1 ? layer.cm1_in_w : layer.cm2_in_w;
                    core::TensorValue &in_b = m == 1 ? layer.cm1_in_b : layer.cm2_in_b;
                    core::TensorValue &cv_w = m == 1 ? layer.cm1_conv_w : layer.cm2_conv_w;
                    core::TensorValue &cv_b = m == 1 ? layer.cm1_conv_b : layer.cm2_conv_b;
                    core::TensorValue &out_w = m == 1 ? layer.cm1_out_w : layer.cm2_out_w;
                    core::TensorValue &out_b = m == 1 ? layer.cm1_out_b : layer.cm2_out_b;
                    in_w = t2(mp + ".in_proj.weight", encoder_dim, encoder_dim * 2);
                    in_b = t1(mp + ".in_proj.bias", encoder_dim * 2);
                    cv_w = t3(mp + ".depthwise_conv.weight", kernel, 1, encoder_dim);
                    cv_b = t1(mp + ".depthwise_conv.bias", encoder_dim);
                    out_w = t2(mp + ".out_proj.weight", encoder_dim, encoder_dim);
                    out_b = t1(mp + ".out_proj.bias", encoder_dim);
                }
                layer.norm_bias = t1(lp + ".norm.bias", encoder_dim);
                const auto log_scale = source->require_f32(lp + ".norm.log_scale");
                if (log_scale.size() != 1) {
                    throw std::runtime_error("zipvoice: " + lp + ".norm.log_scale must be a scalar");
                }
                layer.norm_log_scale = log_scale[0];
            }
            if (with_time) {
                stack.time_proj_w = t2(sp + (downsampling[s] > 1 ? ".encoder" : "") + ".time_emb.1.weight",
                                        config.time_embed_dim, encoder_dim);
                stack.time_proj_b = t1(sp + (downsampling[s] > 1 ? ".encoder" : "") + ".time_emb.1.bias", encoder_dim);
            }
            if (downsampling[s] > 1) {
                const auto bias = source->require_f32(
                    sp + ".downsample.bias",
                    std::vector<int64_t>{downsampling[s]});
                stack.downsample_bias = bias;
                const float maximum = *std::max_element(bias.begin(), bias.end());
                float sum = 0.0F;
                for (float & value : stack.downsample_bias) {
                    value = std::exp(value - maximum);
                    sum += value;
                }
                for (float & value : stack.downsample_bias) value /= sum;
                stack.out_combiner_scale = t1(sp + ".out_combiner.bypass_scale", encoder_dim);
            }
        }
    };

    load_zipformer(weights.fm_decoder, "fm_decoder", config.fm_dim, config.fm_feedforward_dim,
                   config.fm_downsampling_factor, config.fm_num_layers, config.fm_cnn_kernel,
                   config.fm_num_heads, true);
    load_zipformer(weights.text_encoder, "text_encoder", config.text_dim, config.text_feedforward_dim,
                   {1}, {config.text_num_layers}, {config.text_cnn_kernel},
                   config.text_num_heads, false);
    weights.text_encoder.embed_w = t2("embed.weight", config.text_embed_dim, config.vocab_size);

    weights.store->upload();
    source->release_storage();
    return weights;
}

}  // namespace engine::models::zipvoice
