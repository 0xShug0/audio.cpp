#include "engine/community_models/sanotts/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/io/config.h"
#include "engine/framework/model_spec/package.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::sanotts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "sanotts";

SanoTtsConfig parse_nano_config(const engine::io::json::Value & root) {
    SanoTtsConfig out;
    out.voice = root.require("voice").as_string();
    out.vocab_size = root.require("vocab_size").as_i64();
    out.sample_rate = root.require("sample_rate").as_i64();
    out.hop_length = root.require("hop_length").as_i64();
    out.n_fft = root.require("n_fft").as_i64();
    out.mels = root.require("mels").as_i64();
    out.dim = root.require("dim").as_i64();
    out.blocks = root.require("blocks").as_i64();
    out.pw_hidden = root.require("pw_hidden").as_i64();
    out.noise_channels = root.require("noise_channels").as_i64();
    out.dw_kernel = root.require("dw_kernel").as_i64();
    out.embed_kernel = root.require("embed_kernel").as_i64();

    const auto & duration = root.require("duration");
    out.duration_hidden = duration.require("hidden").as_i64();
    out.duration_depth = duration.require("depth").as_i64();
    out.duration_kernel = duration.require("kernel").as_i64();
    out.duration_max_tokens = duration.require("max_tokens").as_i64();
    out.duration_max_frames = duration.require("max_duration").as_i64();

    const auto & acoustic = root.require("acoustic");
    out.acoustic_hidden = acoustic.require("hidden").as_i64();
    out.acoustic_token_depth = acoustic.require("token_depth").as_i64();
    out.acoustic_depth = acoustic.require("depth").as_i64();
    out.acoustic_kernel = acoustic.require("kernel").as_i64();

    for (const auto & [label, value] : std::initializer_list<std::pair<const char *, int64_t>>{
             {"sanoTTS dim", out.dim},
             {"sanoTTS blocks", out.blocks},
             {"sanoTTS pw_hidden", out.pw_hidden},
             {"sanoTTS duration hidden", out.duration_hidden},
             {"sanoTTS acoustic hidden", out.acoustic_hidden},
             {"sanoTTS sample_rate", out.sample_rate},
         }) {
        engine::io::require_positive(value, label);
    }
    return out;
}

/**
 * Fail on a missing or wrongly-shaped tensor at load, not mid-graph.
 *
 * The decoder is noise-fed and ends in an iSTFT, so a weight that is present
 * but wrong in shape tends to produce plausible-sounding audio rather than an
 * obvious failure. Checking the whole inventory up front is what keeps a
 * packaging mistake loud.
 */
void validate_nano_tensors(const SanoTtsAssets & assets) {
    const auto & c = assets.config;
    const auto & weights = *assets.weights;

    std::vector<std::pair<std::string, std::vector<int64_t>>> expected;
    const auto conv = [&](const std::string & name, int64_t out_ch, int64_t in_ch, int64_t k) {
        expected.emplace_back(name + ".weight", std::vector<int64_t>{out_ch, in_ch, k});
        expected.emplace_back(name + ".bias", std::vector<int64_t>{out_ch});
    };
    const auto linear = [&](const std::string & name, int64_t out_ch, int64_t in_ch) {
        expected.emplace_back(name + ".weight", std::vector<int64_t>{out_ch, in_ch});
        expected.emplace_back(name + ".bias", std::vector<int64_t>{out_ch});
    };

    expected.emplace_back("duration.embedding.weight",
                          std::vector<int64_t>{c.vocab_size, c.duration_hidden});
    conv("duration.input_proj", c.duration_hidden, c.duration_hidden + 3, 1);
    for (int64_t b = 0; b < c.duration_depth; ++b) {
        const std::string prefix = "duration.blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.duration_hidden, c.duration_hidden, c.duration_kernel);
        conv(prefix + ".net.2", c.duration_hidden, c.duration_hidden, c.duration_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("duration.output", 1, c.duration_hidden, 1);

    expected.emplace_back("acoustic.embedding.weight",
                          std::vector<int64_t>{c.vocab_size, c.acoustic_hidden});
    conv("acoustic.token_input_proj", c.acoustic_hidden, c.acoustic_hidden + 2, 1);
    for (int64_t b = 0; b < c.acoustic_token_depth; ++b) {
        const std::string prefix = "acoustic.token_blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        conv(prefix + ".net.2", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("acoustic.frame_input_proj", c.acoustic_hidden, c.acoustic_hidden + 3, 1);
    for (int64_t b = 0; b < c.acoustic_depth; ++b) {
        const std::string prefix = "acoustic.frame_blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        conv(prefix + ".net.2", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("acoustic.output", c.mels, c.acoustic_hidden, 1);

    conv("decoder.embed", c.dim, c.mels, c.embed_kernel);
    conv("decoder.noise_adapter", c.dim, c.noise_channels, c.embed_kernel);
    expected.emplace_back("decoder.norm.weight", std::vector<int64_t>{c.dim});
    expected.emplace_back("decoder.norm.bias", std::vector<int64_t>{c.dim});
    for (int64_t b = 0; b < c.blocks; ++b) {
        const std::string prefix = "decoder.blocks." + std::to_string(b);
        conv(prefix + ".dwconv", c.dim, 1, c.dw_kernel);   // groups == dim
        expected.emplace_back(prefix + ".norm.weight", std::vector<int64_t>{c.dim});
        expected.emplace_back(prefix + ".norm.bias", std::vector<int64_t>{c.dim});
        linear(prefix + ".pwconv1", c.pw_hidden, c.dim);
        linear(prefix + ".pwconv2", c.dim, c.pw_hidden);
        expected.emplace_back(prefix + ".gamma", std::vector<int64_t>{c.dim});
    }
    expected.emplace_back("decoder.final_norm.weight", std::vector<int64_t>{c.dim});
    expected.emplace_back("decoder.final_norm.bias", std::vector<int64_t>{c.dim});
    linear("decoder.head", c.n_fft + 2, c.dim);

    for (const auto & [name, shape] : expected) {
        if (!weights.has_tensor(name)) {
            throw std::runtime_error("sanoTTS missing tensor: " + name);
        }
        assets::require_tensor_shape(weights, name, shape);
    }
}


SanoTtsPiperConfig parse_piper_config(const engine::io::json::Value & root) {
    SanoTtsPiperConfig out;
    out.voice = root.require("voice").as_string();
    out.language = root.require("language").as_string();
    out.espeak_voice = root.require("espeak_voice").as_string();
    out.sample_rate = root.require("sample_rate").as_i64();
    out.duration_length_scale =
        static_cast<double>(root.require("duration_length_scale").as_f32());

    const auto & duration = root.require("duration");
    out.duration_vocab = duration.require("vocab_size").as_i64();
    out.duration_hidden = duration.require("hidden").as_i64();
    out.duration_depth = duration.require("depth").as_i64();
    out.duration_kernel = duration.require("kernel").as_i64();
    out.duration_max_tokens = duration.require("max_tokens").as_i64();
    out.duration_max_frames = duration.require("max_duration").as_i64();

    const auto & acoustic = root.require("acoustic");
    out.acoustic_vocab = acoustic.require("vocab_size").as_i64();
    out.acoustic_hidden = acoustic.require("hidden").as_i64();
    out.acoustic_depth = acoustic.require("depth").as_i64();
    out.acoustic_token_depth = acoustic.require("token_depth").as_i64();
    out.acoustic_kernel = acoustic.require("kernel").as_i64();
    out.acoustic_out_channels = acoustic.require("out_channels").as_i64();

    const auto & decoder = root.require("decoder");
    const auto channels = decoder.require("channels").as_array();
    if (channels.size() != 4) {
        throw std::runtime_error("sanoTTS piperlite decoder.channels must have 4 entries");
    }
    for (size_t stage = 0; stage < 4; ++stage) {
        out.channels[stage] = channels[stage].as_i64();
    }
    for (size_t stage = 0; stage < 3; ++stage) {
        const auto & branches =
            decoder.require("stage" + std::to_string(stage) + "_branches").as_array();
        for (const auto & branch : branches) {
            const int64_t index = branch.as_i64();
            if (index < 0 || index > 2) {
                throw std::runtime_error("sanoTTS piperlite branch index out of range");
            }
            out.stage_branches[stage].push_back(index);
        }
        if (out.stage_branches[stage].empty()) {
            throw std::runtime_error("sanoTTS piperlite stage has no branches");
        }
    }
    out.post_filter_channels = decoder.require("post_filter_channels").as_i64();
    out.post_filter_layers = decoder.require("post_filter_layers").as_i64();
    out.post_filter_kernel = decoder.require("post_filter_kernel").as_i64();
    out.post_filter_scale =
        static_cast<double>(decoder.require("post_filter_scale").as_f32());

    for (const auto & [symbol, id] : root.require("phoneme_id_map").as_object()) {
        out.phoneme_id_map.emplace(symbol, static_cast<int32_t>(id.as_i64()));
    }
    if (out.phoneme_id_map.empty()) {
        throw std::runtime_error("sanoTTS piperlite config has an empty phoneme_id_map");
    }

    for (const auto & [label, value] : std::initializer_list<std::pair<const char *, int64_t>>{
             {"sanoTTS piperlite duration vocab", out.duration_vocab},
             {"sanoTTS piperlite duration hidden", out.duration_hidden},
             {"sanoTTS piperlite duration max_tokens", out.duration_max_tokens},
             {"sanoTTS piperlite acoustic vocab", out.acoustic_vocab},
             {"sanoTTS piperlite acoustic hidden", out.acoustic_hidden},
             {"sanoTTS piperlite latent channels", out.acoustic_out_channels},
             {"sanoTTS piperlite sample_rate", out.sample_rate},
         }) {
        engine::io::require_positive(value, label);
    }
    if (out.duration_length_scale <= 0.0) {
        throw std::runtime_error("sanoTTS piperlite duration_length_scale must be positive");
    }
    return out;
}

// PiperResidualBank geometry, fixed by the training code.
constexpr int64_t kBankKernels[3] = {3, 5, 7};

/**
 * The piperlite inventory. Kernel sizes the reference reads from the tensors
 * themselves (decoder pre/post convs) are validated as rank/channels only,
 * by looking the actual kernel up from the source.
 */
void validate_piper_tensors(const SanoTtsAssets & assets) {
    const auto & c = assets.piper;
    const auto & weights = *assets.weights;

    std::vector<std::pair<std::string, std::vector<int64_t>>> expected;
    const auto conv = [&](const std::string & name, int64_t out_ch, int64_t in_ch, int64_t k) {
        expected.emplace_back(name + ".weight", std::vector<int64_t>{out_ch, in_ch, k});
        expected.emplace_back(name + ".bias", std::vector<int64_t>{out_ch});
    };
    const auto conv_any_kernel = [&](const std::string & name, int64_t out_ch, int64_t in_ch) {
        const auto metadata = weights.require_metadata(name + ".weight");
        if (metadata.shape.size() != 3 || metadata.shape[0] != out_ch ||
            metadata.shape[1] != in_ch || metadata.shape[2] < 1 ||
            metadata.shape[2] % 2 == 0) {
            throw std::runtime_error("sanoTTS unexpected shape for tensor: " + name + ".weight");
        }
        expected.emplace_back(name + ".bias", std::vector<int64_t>{out_ch});
    };

    expected.emplace_back("duration.embedding.weight",
                          std::vector<int64_t>{c.duration_vocab, c.duration_hidden});
    conv("duration.input_proj", c.duration_hidden, c.duration_hidden + 3, 1);
    for (int64_t b = 0; b < c.duration_depth; ++b) {
        const std::string prefix = "duration.blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.duration_hidden, c.duration_hidden, c.duration_kernel);
        conv(prefix + ".net.2", c.duration_hidden, c.duration_hidden, c.duration_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("duration.output", 1, c.duration_hidden, 1);

    expected.emplace_back("acoustic.embedding.weight",
                          std::vector<int64_t>{c.acoustic_vocab, c.acoustic_hidden});
    conv("acoustic.token_input_proj", c.acoustic_hidden, c.acoustic_hidden + 2, 1);
    for (int64_t b = 0; b < c.acoustic_token_depth; ++b) {
        const std::string prefix = "acoustic.token_blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        conv(prefix + ".net.2", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("acoustic.frame_input_proj", c.acoustic_hidden, c.acoustic_hidden + 3, 1);
    for (int64_t b = 0; b < c.acoustic_depth; ++b) {
        const std::string prefix = "acoustic.frame_blocks." + std::to_string(b);
        conv(prefix + ".net.0", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        conv(prefix + ".net.2", c.acoustic_hidden, c.acoustic_hidden, c.acoustic_kernel);
        expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
    }
    conv("acoustic.output", c.acoustic_out_channels, c.acoustic_hidden, 1);

    conv_any_kernel("decoder.pre", c.channels[0], c.acoustic_out_channels);
    const int64_t up_kernels[3] = {16, 16, 8};
    for (size_t stage = 0; stage < 3; ++stage) {
        const int64_t in_ch = c.channels[stage];
        const int64_t out_ch = c.channels[stage + 1];
        // ConvTranspose1d stores [in, out, K]
        expected.emplace_back(
            "decoder.up" + std::to_string(stage) + ".weight",
            std::vector<int64_t>{in_ch, out_ch, up_kernels[stage]});
        expected.emplace_back(
            "decoder.up" + std::to_string(stage) + ".bias",
            std::vector<int64_t>{out_ch});
        for (const int64_t branch : c.stage_branches[stage]) {
            const std::string prefix = "decoder.res" + std::to_string(stage) + ".0.blocks." +
                                       std::to_string(branch);
            conv(prefix + ".conv1", out_ch, out_ch, kBankKernels[branch]);
            conv(prefix + ".conv2", out_ch, out_ch, kBankKernels[branch]);
        }
    }
    conv_any_kernel("decoder.post", 1, c.channels[3]);
    if (c.post_filter_channels > 0) {
        conv_any_kernel("decoder.post_filter.in_conv", c.post_filter_channels, 1);
        for (int64_t layer = 0; layer < c.post_filter_layers; ++layer) {
            const std::string prefix = "decoder.post_filter.units." + std::to_string(layer);
            // Unit conv kernels are a property of the checkpoint, not of
            // post_filter_kernel (which sizes only the in/out convs).
            conv_any_kernel(prefix + ".conv1", c.post_filter_channels, c.post_filter_channels);
            conv_any_kernel(prefix + ".conv2", c.post_filter_channels, c.post_filter_channels);
            expected.emplace_back(prefix + ".scale", std::vector<int64_t>{1});
        }
        conv_any_kernel("decoder.post_filter.out_conv", 1, c.post_filter_channels);
    }

    for (const auto & [name, shape] : expected) {
        if (!weights.has_tensor(name)) {
            throw std::runtime_error("sanoTTS missing tensor: " + name);
        }
        assets::require_tensor_shape(weights, name, shape);
    }
}

}  // namespace

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    SanoTtsAssets out;
    const auto root = resources.parse_json("config");
    const auto architecture = root.require("architecture").as_string();
    if (architecture != kFamily) {
        throw std::runtime_error(
            "sanoTTS config.json architecture is '" + architecture + "', expected 'sanotts'");
    }
    const auto * graph = root.find("graph");
    const auto graph_name = graph == nullptr ? std::string("nano") : graph->as_string();
    if (graph_name == "nano") {
        out.graph = SanoTtsGraph::Nano;
        out.config = parse_nano_config(root);
    } else if (graph_name == "piperlite") {
        out.graph = SanoTtsGraph::Piperlite;
        out.piper = parse_piper_config(root);
    } else {
        throw std::runtime_error("sanoTTS config.json has unknown graph '" + graph_name + "'");
    }
    out.weights = resources.open_tensor_source("weights");
    out.resources = std::move(resources);
    if (out.graph == SanoTtsGraph::Nano) {
        validate_nano_tensors(out);
    } else {
        validate_piper_tensors(out);
    }
    return std::make_shared<SanoTtsAssets>(std::move(out));
}

}  // namespace engine::models::sanotts
