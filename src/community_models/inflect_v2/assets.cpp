#include "engine/community_models/inflect_v2/assets.h"

#include "engine/framework/io/config.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace engine::models::inflect_v2 {
namespace {

namespace json = engine::io::json;

std::vector<int64_t> i64_array(const json::Value & value, const char * name) {
    const auto out = json::number_array_as<int64_t>(value);
    if (out.empty()) {
        throw std::runtime_error(std::string("Inflect v2 ") + name + " must not be empty");
    }
    for (const int64_t item : out) {
        engine::io::require_positive(item, std::string("Inflect v2 ") + name);
    }
    return out;
}

InflectV2Config parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    if (json::require_string(root, "format") != "inflect_v2_inference_config_v1") {
        throw std::runtime_error("Inflect v2 requires an inflect_v2_inference_config_v1 config");
    }
    const auto & data = root.require("data");
    const auto & model = root.require("model");

    InflectV2Config out;
    out.sample_rate = json::require_i64(data, "sampling_rate");
    out.hop_length = json::require_i64(data, "hop_length");
    out.inter_channels = json::require_i64(model, "inter_channels");
    out.hidden_channels = json::require_i64(model, "hidden_channels");
    out.filter_channels = json::require_i64(model, "filter_channels");
    out.attention_heads = json::require_i64(model, "n_heads");
    out.encoder_layers = json::require_i64(model, "n_layers");
    out.upsample_initial_channels = json::require_i64(model, "upsample_initial_channel");
    const int64_t posterior_layers = json::require_i64(model, "n_layers_q");
    out.upsample_rates = i64_array(model.require("upsample_rates"), "upsample_rates");
    out.upsample_kernel_sizes =
        i64_array(model.require("upsample_kernel_sizes"), "upsample_kernel_sizes");
    out.resblock_kernel_sizes =
        i64_array(model.require("resblock_kernel_sizes"), "resblock_kernel_sizes");
    for (const auto & row : model.require("resblock_dilation_sizes").as_array()) {
        out.resblock_dilations.push_back(i64_array(row, "resblock_dilation_sizes"));
    }

    engine::io::require_positive(out.sample_rate, "Inflect v2 sample rate");
    engine::io::require_positive(out.hop_length, "Inflect v2 hop length");
    engine::io::require_positive(out.inter_channels, "Inflect v2 inter channels");
    engine::io::require_positive(out.hidden_channels, "Inflect v2 hidden channels");
    engine::io::require_positive(out.filter_channels, "Inflect v2 filter channels");
    engine::io::require_positive(out.attention_heads, "Inflect v2 attention heads");
    engine::io::require_divisible(
        out.hidden_channels,
        out.attention_heads,
        "Inflect v2 hidden channels");
    if (out.sample_rate != 24000 || out.hop_length != 256 ||
        !json::require_bool(data, "add_blank") ||
        json::require_i64(data, "n_speakers") != 0 ||
        !json::require_bool(data, "cleaned_text") ||
        out.encoder_layers != 3 || out.attention_heads != 2 ||
        json::require_i64(model, "kernel_size") != 3 ||
        json::require_string(model, "resblock") != "1" ||
        json::require_bool(model, "use_spectral_norm") ||
        json::require_bool(model, "use_sdp") ||
        !json::require_bool(model, "inference_only") ||
        out.upsample_rates != std::vector<int64_t>({8, 8, 2, 2}) ||
        out.upsample_kernel_sizes != std::vector<int64_t>({16, 16, 4, 4}) ||
        out.resblock_kernel_sizes != std::vector<int64_t>({3, 7, 11}) ||
        out.resblock_dilations !=
            std::vector<std::vector<int64_t>>({{1, 3, 5}, {1, 3, 5}, {1, 3, 5}})) {
        throw std::runtime_error("Inflect v2 config uses an unsupported architecture");
    }
    if (out.inter_channels == 192 && out.hidden_channels == 96 &&
        out.filter_channels == 768 && out.upsample_initial_channels == 320 &&
        posterior_layers == 3) {
        out.variant = "micro-v2";
    } else if (
        out.inter_channels == 128 && out.hidden_channels == 72 &&
        out.filter_channels == 384 && out.upsample_initial_channels == 192 &&
        posterior_layers == 2) {
        out.variant = "nano-v2";
    } else {
        throw std::runtime_error("Inflect v2 config does not match Micro v2 or Nano v2");
    }
    return out;
}

using TensorInventory =
    std::unordered_map<std::string, std::vector<int64_t>>;

void add_conv(
    TensorInventory & tensors,
    const std::string & prefix,
    std::vector<int64_t> weight,
    std::vector<int64_t> bias) {
    tensors.emplace(prefix + ".weight", std::move(weight));
    tensors.emplace(prefix + ".bias", std::move(bias));
}

TensorInventory expected_tensors(const InflectV2Config & c) {
    TensorInventory out;
    out.emplace(
        "enc_p.emb.weight",
        std::vector<int64_t>{c.vocab_size, c.hidden_channels});
    for (int layer = 0; layer < c.encoder_layers; ++layer) {
        const std::string attention =
            "enc_p.encoder.attn_layers." + std::to_string(layer);
        out.emplace(
            attention + ".emb_rel_k",
            std::vector<int64_t>{
                1,
                9,
                c.hidden_channels / c.attention_heads,
            });
        out.emplace(
            attention + ".emb_rel_v",
            std::vector<int64_t>{
                1,
                9,
                c.hidden_channels / c.attention_heads,
            });
        for (const char * projection :
             {"conv_q", "conv_k", "conv_v", "conv_o"}) {
            add_conv(
                out,
                attention + "." + projection,
                {c.hidden_channels, c.hidden_channels, 1},
                {c.hidden_channels});
        }
        for (const char * group : {"norm_layers_1", "norm_layers_2"}) {
            const std::string norm =
                "enc_p.encoder." + std::string(group) + "." +
                std::to_string(layer);
            out.emplace(
                norm + ".gamma",
                std::vector<int64_t>{c.hidden_channels});
            out.emplace(
                norm + ".beta",
                std::vector<int64_t>{c.hidden_channels});
        }
        const std::string ffn =
            "enc_p.encoder.ffn_layers." + std::to_string(layer);
        add_conv(
            out,
            ffn + ".conv_1",
            {c.filter_channels, c.hidden_channels, 3},
            {c.filter_channels});
        add_conv(
            out,
            ffn + ".conv_2",
            {c.hidden_channels, c.filter_channels, 3},
            {c.hidden_channels});
    }
    add_conv(
        out,
        "enc_p.proj",
        {2 * c.inter_channels, c.hidden_channels, 1},
        {2 * c.inter_channels});
    add_conv(
        out,
        "dp.conv_1",
        {c.duration_channels, c.hidden_channels, 3},
        {c.duration_channels});
    out.emplace(
        "dp.norm_1.gamma",
        std::vector<int64_t>{c.duration_channels});
    out.emplace(
        "dp.norm_1.beta",
        std::vector<int64_t>{c.duration_channels});
    add_conv(
        out,
        "dp.conv_2",
        {c.duration_channels, c.duration_channels, 3},
        {c.duration_channels});
    out.emplace(
        "dp.norm_2.gamma",
        std::vector<int64_t>{c.duration_channels});
    out.emplace(
        "dp.norm_2.beta",
        std::vector<int64_t>{c.duration_channels});
    add_conv(
        out,
        "dp.proj",
        {1, c.duration_channels, 1},
        {1});

    add_conv(
        out,
        "dec.conv_pre",
        {c.upsample_initial_channels, c.inter_channels, 7},
        {c.upsample_initial_channels});
    int64_t channels = c.upsample_initial_channels;
    for (size_t stage = 0; stage < c.upsample_rates.size(); ++stage) {
        const int64_t output_channels = channels / 2;
        add_conv(
            out,
            "dec.ups." + std::to_string(stage),
            {channels, output_channels, c.upsample_kernel_sizes[stage]},
            {output_channels});
        for (size_t block = 0;
             block < c.resblock_kernel_sizes.size();
             ++block) {
            const size_t block_index =
                stage * c.resblock_kernel_sizes.size() + block;
            for (const char * stack : {"convs1", "convs2"}) {
                for (size_t layer = 0; layer < 3; ++layer) {
                    add_conv(
                        out,
                        "dec.resblocks." +
                            std::to_string(block_index) + "." + stack + "." +
                            std::to_string(layer),
                        {
                            output_channels,
                            output_channels,
                            c.resblock_kernel_sizes[block],
                        },
                        {output_channels});
                }
            }
        }
        channels = output_channels;
    }
    out.emplace(
        "dec.conv_post.weight",
        std::vector<int64_t>{1, channels, 7});

    const int64_t half = c.inter_channels / 2;
    for (int flow = 0; flow < c.flow_count; ++flow) {
        const std::string prefix =
            "flow.flows." + std::to_string(flow * 2);
        add_conv(
            out,
            prefix + ".pre",
            {c.hidden_channels, half, 1},
            {c.hidden_channels});
        for (int layer = 0; layer < c.flow_layers; ++layer) {
            add_conv(
                out,
                prefix + ".enc.in_layers." + std::to_string(layer),
                {2 * c.hidden_channels, c.hidden_channels, 5},
                {2 * c.hidden_channels});
            const int64_t output_channels =
                layer + 1 < c.flow_layers
                ? 2 * c.hidden_channels
                : c.hidden_channels;
            add_conv(
                out,
                prefix + ".enc.res_skip_layers." +
                    std::to_string(layer),
                {output_channels, c.hidden_channels, 1},
                {output_channels});
        }
        add_conv(
            out,
            prefix + ".post",
            {half, c.hidden_channels, 1},
            {half});
    }
    if (out.size() != 302) {
        throw std::runtime_error(
            "Inflect v2 internal tensor inventory is invalid");
    }
    return out;
}

void validate_tensors(const InflectV2Assets & assets) {
    const auto & c = assets.config;
    const auto expected = expected_tensors(c);
    const auto actual = assets.weights->tensors();
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            "Inflect v2 expects exactly 302 tensors, found " +
            std::to_string(actual.size()));
    }
    for (const auto & tensor : actual) {
        const auto found = expected.find(tensor.name);
        if (found == expected.end()) {
            throw std::runtime_error(
                "Inflect v2 has unexpected tensor: " + tensor.name);
        }
        if (tensor.shape != found->second) {
            throw std::runtime_error(
                "Inflect v2 tensor shape mismatch: " + tensor.name);
        }
        if (engine::assets::ggml_type_for_tensor_dtype(tensor.dtype) !=
            GGML_TYPE_F32) {
            throw std::runtime_error(
                "Inflect v2 supports FP32 weights only: " + tensor.name);
        }
    }
}

}  // namespace

std::shared_ptr<const InflectV2Assets> load_inflect_v2_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(
        model_path,
        "inflect_v2");
    InflectV2Assets out;
    out.config = parse_config(resources);
    out.weights = resources.open_tensor_source("weights");
    out.resources = std::move(resources);
    validate_tensors(out);
    return std::make_shared<InflectV2Assets>(std::move(out));
}

}  // namespace engine::models::inflect_v2
