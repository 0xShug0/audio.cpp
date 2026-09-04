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

SanoTtsConfig parse_config(const assets::ResourceBundle & resources) {
    const auto root = resources.parse_json("config");
    const auto architecture = root.require("architecture").as_string();
    if (architecture != kFamily) {
        throw std::runtime_error(
            "sanoTTS config.json architecture is '" + architecture + "', expected 'sanotts'");
    }
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
void validate_tensors(const SanoTtsAssets & assets) {
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

}  // namespace

std::shared_ptr<const SanoTtsAssets> load_sanotts_assets(
    const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    SanoTtsAssets out;
    out.config = parse_config(resources);
    out.weights = resources.open_tensor_source("weights");
    out.resources = std::move(resources);
    validate_tensors(out);
    return std::make_shared<SanoTtsAssets>(std::move(out));
}

}  // namespace engine::models::sanotts
