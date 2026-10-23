#include "engine/models/pocket_tts/assets.h"
#include "engine/models/pocket_tts/loader.h"
#include "engine/models/pocket_tts/session.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::pocket_tts {
namespace {

namespace json = engine::io::json;

constexpr const char * kFamily = "gradium_phonon";
constexpr const char * kModelName = "Gradium Phonon";
constexpr const char * kConditionerPrefix = "flow_lm.condition_provider.conditioners.";
// The reference runtime samples at 0.3 and does not read the config's `temp`.
constexpr float kDefaultTemperature = 0.3F;

// Values the reference runtime gives the conditioners a request leaves out.
std::optional<std::string> default_condition_value(const std::string & name) {
    if (name == "num_speakers") {
        return std::string("1");
    }
    if (name == "duration_delta" || name == "padding_bonus") {
        return std::string("0");
    }
    return std::nullopt;
}

std::vector<float> project(const assets::TensorDataF32 & weight, const std::vector<float> & input, const std::string & name) {
    if (weight.shape.rank != 2 || weight.shape.dims[1] != static_cast<int64_t>(input.size())) {
        throw std::runtime_error(name + " does not take a " + std::to_string(input.size()) + "-wide input");
    }
    const int64_t rows = weight.shape.dims[0];
    const int64_t cols = weight.shape.dims[1];
    std::vector<float> output(static_cast<size_t>(rows), 0.0F);
    for (int64_t row = 0; row < rows; ++row) {
        const float * w = weight.values.data() + row * cols;
        float sum = 0.0F;
        for (int64_t col = 0; col < cols; ++col) {
            sum += w[col] * input[static_cast<size_t>(col)];
        }
        output[static_cast<size_t>(row)] = sum;
    }
    return output;
}

std::vector<float> project_if_present(
    const assets::TensorSource & source,
    const std::string & prefix,
    std::vector<float> values,
    int64_t output_dim) {
    if (const auto proj = source.optional_f32_tensor(prefix + "output_proj.weight")) {
        return project(*proj, values, prefix + "output_proj.weight");
    }
    if (static_cast<int64_t>(values.size()) != output_dim) {
        throw std::runtime_error(prefix + " has no output_proj to the flow width");
    }
    return values;
}

// Sum of the checkpoint's conditioners at their default values: a constant the reference adds
// to input_linear(latent) on every generated frame and never to the prompt, so it is exactly
// an input_linear bias.
std::optional<std::vector<float>> summed_conditioners(
    const json::Value & config,
    const assets::TensorSource & source,
    int64_t output_dim) {
    const auto * conditioners = config.find("conditioners");
    if (conditioners == nullptr || conditioners->is_null() || conditioners->as_array().empty()) {
        return std::nullopt;
    }
    std::vector<std::string> summed;
    if (const auto * fuser = config.find("fuser"); fuser != nullptr && fuser->find("sum") != nullptr) {
        for (const auto & name : fuser->require("sum").as_array()) {
            summed.push_back(name.as_string());
        }
    }
    std::vector<float> total(static_cast<size_t>(output_dim), 0.0F);
    for (const auto & conditioner : conditioners->as_array()) {
        const std::string name = json::require_string(conditioner, "name");
        const std::string type = json::require_string(conditioner, "type");
        if (std::find(summed.begin(), summed.end(), name) == summed.end()) {
            throw std::runtime_error(std::string(kModelName) + " conditioner " + name + " is not summed; only summed conditioners are supported");
        }
        const auto value = default_condition_value(name);
        if (!value.has_value()) {
            throw std::runtime_error(std::string(kModelName) + " conditioner " + name + " has no default value");
        }
        const std::string prefix = std::string(kConditionerPrefix) + name + ".";
        std::vector<float> embedding;
        if (type == "lut") {
            const auto & lut = conditioner.require("lut");
            if (json::optional_string(lut, "tokenizer", "noop") != "noop") {
                throw std::runtime_error(std::string(kModelName) + " conditioner " + name + " uses an unsupported tokenizer");
            }
            const auto & possible = lut.require("possible_values").as_array();
            int64_t index = -1;
            for (size_t i = 0; i < possible.size(); ++i) {
                if (possible[i].as_string() == *value) {
                    index = static_cast<int64_t>(i);
                }
            }
            const auto table = source.require_f32_tensor(prefix + "embed.weight");
            if (index < 0 || table.shape.rank != 2 || index >= table.shape.dims[0]) {
                throw std::runtime_error(std::string(kModelName) + " conditioner " + name + " cannot look up " + *value);
            }
            const int64_t dim = table.shape.dims[1];
            embedding.assign(
                table.values.begin() + index * dim,
                table.values.begin() + (index + 1) * dim);
        } else if (type == "continuous") {
            const auto & continuous = conditioner.require("continuous");
            const float position = json::require_f32(continuous, "scale_factor") * std::stof(*value);
            const int64_t dim = json::require_i64(continuous, "dim");
            const float max_period = json::optional_f32(continuous, "max_period", 10000.0F);
            if (dim < 4 || dim % 2 != 0) {
                throw std::runtime_error(std::string(kModelName) + " conditioner " + name + " needs an even dim of at least 4");
            }
            const int64_t half = dim / 2;
            embedding.resize(static_cast<size_t>(dim));
            for (int64_t i = 0; i < half; ++i) {
                const float phase = position / std::pow(max_period, static_cast<float>(i) / static_cast<float>(half - 1));
                embedding[static_cast<size_t>(i)] = std::cos(phase);
                embedding[static_cast<size_t>(half + i)] = std::sin(phase);
            }
        } else {
            throw std::runtime_error(std::string(kModelName) + " conditioner type " + type + " is not supported");
        }
        const auto projected = project_if_present(source, prefix, std::move(embedding), output_dim);
        for (size_t i = 0; i < total.size(); ++i) {
            total[i] += projected[i];
        }
    }
    return total;
}

void reject_unsupported(const json::Value & config) {
    if (json::optional_i64(config, "lsd_decode_steps", 1) != 1) {
        throw std::runtime_error(std::string(kModelName) + " supports only lsd_decode_steps = 1");
    }
    if (const auto * sinks = config.find("prefix_sinks"); sinks != nullptr && !sinks->is_null()) {
        throw std::runtime_error(std::string(kModelName) + " checkpoints with prefix_sinks are not supported");
    }
    if (const auto * voices = config.find("voices"); voices != nullptr && !voices->is_null() && !voices->as_array().empty()) {
        throw std::runtime_error(std::string(kModelName) + " checkpoints with baked-in voices are not supported");
    }
    for (const char * section : {"flow_lm", "mimi"}) {
        if (json::optional_string(config.require(section), "gelu", "erf") != "erf") {
            throw std::runtime_error(std::string(kModelName) + " supports only the erf GELU");
        }
    }
}

PocketTTSModelConfig model_config_from_json(const json::Value & config) {
    const auto & flow = config.require("flow_lm");
    const auto & mimi = config.require("mimi");
    PocketTTSModelConfig out;
    out.default_temperature = kDefaultTemperature;
    out.default_eos_threshold = json::optional_f32(config, "eos_threshold", out.default_eos_threshold);
    out.frames_after_eos_offset = 1;
    out.insert_bos_before_voice = false;
    out.time_embedding_plain_rms = json::optional_string(flow, "time_rms_norm", "rms_var") == "rms";
    out.sample_rate = json::require_i32(mimi, "sample_rate");
    out.frame_rate = json::require_f32(mimi, "frame_rate");
    out.mimi_frame_rate = out.frame_rate;
    out.flow_dim = json::require_i64(flow, "d_model");
    out.flow_heads = json::require_i64(flow, "num_heads");
    out.flow_layers = json::require_i64(flow, "num_layers");
    out.flow_intermediate_size = json::require_i64(flow, "dim_feedforward");
    out.flow_hidden_size = json::require_i64(flow, "flow_dim");
    out.flow_depth = json::require_i64(flow, "flow_depth");
    out.latent_dim = json::require_i64(flow, "ldim");
    out.mimi_seanet_dimension = json::require_i64(mimi, "dimension");
    out.mimi_inner_dim = json::require_i64(mimi, "quantizer_dimension");
    out.mimi_outer_dim = json::require_i64(mimi, "quantizer_output_dimension");
    out.mimi_base_filters = json::require_i64(mimi, "n_filters");
    out.mimi_dim = json::require_i64(mimi, "transformer_d_model");
    out.mimi_heads = json::require_i64(mimi, "transformer_num_heads");
    out.mimi_layers = json::require_i64(mimi, "transformer_num_layers");
    out.mimi_intermediate_size = json::require_i64(mimi, "transformer_dim_feedforward");
    out.mimi_context = json::require_i64(mimi, "transformer_context");
    int64_t hop_length = 1;
    for (const int64_t ratio : json::require_i64_array(mimi, "ratios")) {
        hop_length *= ratio;
    }
    const double encoder_frame_rate = static_cast<double>(out.sample_rate) / static_cast<double>(hop_length);
    out.mimi_encoder_upsample_stride =
        static_cast<int64_t>(std::llround(encoder_frame_rate / static_cast<double>(out.frame_rate)));
    if (out.latent_dim != out.mimi_inner_dim) {
        throw std::runtime_error(std::string(kModelName) + " flow_lm.ldim must match mimi.quantizer_dimension");
    }
    return out;
}

std::string language_list(const json::Value & config) {
    std::string out;
    if (const auto * languages = config.find("languages"); languages != nullptr && languages->is_array()) {
        for (const auto & language : languages->as_array()) {
            out += (out.empty() ? "" : ",") + language.as_string();
        }
    }
    return out.empty() ? std::string("en") : out;
}

std::optional<std::filesystem::path> find_default_voice(const std::filesystem::path & root) {
    for (const auto & candidate : {root / "embeddings" / "default.safetensors", root / "default-voice.safetensors"}) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec)) {
            return candidate;
        }
    }
    return std::nullopt;
}

runtime::SessionOptions as_pocket_tts_session_options(
    const runtime::SessionOptions & options,
    const engine::model_spec::ModelContract & contract) {
    runtime::validate_spec_backed_session_options(options, contract, kFamily, kModelName);
    const std::string prefix = std::string(kFamily) + ".";
    runtime::SessionOptions out = options;
    out.options.clear();
    for (const auto & [key, value] : options.options) {
        if (key.rfind(prefix, 0) == 0) {
            out.options["pocket_tts." + key.substr(prefix.size())] = value;
        } else {
            out.options[key] = value;
        }
    }
    return out;
}

}  // namespace

std::shared_ptr<const PocketTTSAssets> load_gradium_phonon_assets(const std::filesystem::path & model_path) {
    auto resources = engine::model_spec::load_resource_bundle_for_family(model_path, kFamily);
    const auto config = resources.parse_json("config");
    reject_unsupported(config);
    const auto tensor_source = resources.open_tensor_source("weights");

    auto manifest = std::make_shared<PocketTTSAssets>();
    manifest->family = kFamily;
    manifest->model_config = model_config_from_json(config);
    manifest->language = language_list(config);
    manifest->model_weights = tensor_source;
    manifest->voice_asset_root = tensor_source->source_path().parent_path();
    manifest->default_voice_path = find_default_voice(manifest->voice_asset_root);

    const std::string prefix = kConditionerPrefix;
    auto & host = manifest->host_weights;
    host.conditioner_embedding_table = tensor_source->require_f32_tensor(prefix + "transcript_in_segment.embed.weight");
    host.bos_emb = tensor_source->require_f32("flow_lm.bos_emb");
    host.emb_mean = tensor_source->require_f32("flow_lm.emb_mean");
    host.emb_std = tensor_source->require_f32("flow_lm.emb_std");
    host.speaker_proj = tensor_source->require_f32_tensor(prefix + "speaker_wavs.output_proj.weight");
    host.input_linear_bias = summed_conditioners(config, *tensor_source, manifest->model_config.flow_dim);

    manifest->resources = std::move(resources);
    manifest->tokenizer_pieces = tokenizers::load_sentencepiece_model(manifest->resources.require_file("tokenizer"));
    return manifest;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_gradium_phonon_loader() {
    runtime::SpecBackedVoiceModelConfig<PocketTTSAssets> config;
    config.family = kFamily;
    config.load_assets = load_gradium_phonon_assets;
    config.create_session = [](const runtime::TaskSpec & task,
                                const runtime::SessionOptions & options,
                                std::shared_ptr<const PocketTTSAssets> assets,
                                std::shared_ptr<const engine::model_spec::ModelContract> contract) {
        if (task.task != runtime::VoiceTaskKind::Tts) {
            throw std::runtime_error(std::string(kModelName) + " supports only tts: its checkpoints have no encoder to clone a voice");
        }
        auto voice_root = assets->voice_asset_root;
        return std::make_unique<PocketTTSSession>(
            task,
            as_pocket_tts_session_options(options, *contract),
            std::move(assets),
            std::move(voice_root),
            contract);
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::pocket_tts
