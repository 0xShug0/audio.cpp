#include "engine/models/yue2/request.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/binary.h"
#include "engine/framework/runtime/options.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace engine::models::yue2 {
namespace {

std::string request_style(const runtime::TaskRequest & request) {
    if (const auto style = runtime::find_option(request.options, {"style"})) {
        return *style;
    }
    if (request.voice.has_value() && request.voice->style.has_value()) {
        const auto & tags = request.voice->style->tags;
        if (const auto it = tags.find("style"); it != tags.end()) {
            return it->second;
        }
        if (const auto it = tags.find("tags"); it != tags.end()) {
            return it->second;
        }
    }
    return {};
}

std::string request_lyrics(const runtime::TaskRequest & request) {
    if (const auto lyrics = runtime::find_option(request.options, {"lyrics"})) {
        return *lyrics;
    }
    if (request.text_input.has_value()) {
        return request.text_input->text;
    }
    return {};
}

std::string abc_from_options(const std::unordered_map<std::string, std::string> & options) {
    if (const auto abc = runtime::find_option(options, {"abc"})) {
        return *abc;
    }
    if (const auto abc_file = runtime::find_option(options, {"abc_file"})) {
        const std::filesystem::path path(*abc_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 abc_file does not exist: " + path.string());
        }
        return engine::io::read_text_file(path);
    }
    return {};
}

void apply_options(
    Yue2Request & out,
    const std::unordered_map<std::string, std::string> & options) {
    if (const auto cot = runtime::find_option(options, {"cot"})) {
        out.cot = parse_cot_mode(*cot);
    }
    if (const auto seed = runtime::parse_u64_option(options, {"seed"})) {
        out.seed = *seed;
    }
    if (out.seed >= (uint64_t{1} << 63U)) {
        throw std::runtime_error("Yue2 seed must be in [0, 2^63)");
    }
    if (const auto value = runtime::parse_finite_float_option(options, {"guidance_scale", "cfg_scale"})) {
        if (*value < 0.0F || *value > 20.0F) {
            throw std::runtime_error("Yue2 guidance_scale must be in [0,20]");
        }
        out.cfg_scale = *value;
    }
    out.generation.ode_steps =
        runtime::parse_positive_i64_option(options, {"num_inference_steps"}, Yue2GenerationConfig{}.ode_steps);
    if (out.generation.ode_steps <= 0) {
        throw std::runtime_error("Yue2 num_inference_steps must be positive");
    }
    auto apply_sampling = [&options](Yue2SamplingConfig & sampling, const std::string & prefix) {
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_temperature"})) {
            sampling.temperature = *value;
        }
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_top_p"})) {
            sampling.top_p = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_top_k"})) {
            sampling.top_k = *value;
        }
        if (const auto value = runtime::parse_finite_float_option(options, {prefix + "_repetition_penalty"})) {
            sampling.repetition_penalty = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_penalty_window"})) {
            sampling.penalty_window = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_min_tokens"})) {
            sampling.min_tokens = *value;
        }
        if (const auto value = runtime::parse_i64_option(options, {prefix + "_max_tokens"})) {
            sampling.max_tokens = *value;
        }
    };
    auto validate_sampling = [](const Yue2SamplingConfig & sampling, const std::string & prefix) {
        if (sampling.temperature < 0.0F || sampling.temperature > 5.0F ||
            sampling.top_p <= 0.0F || sampling.top_p > 1.0F ||
            sampling.top_k < 1 ||
            sampling.repetition_penalty <= 0.0F ||
            sampling.penalty_window < 1 ||
            sampling.min_tokens < 0 ||
            sampling.max_tokens < sampling.min_tokens) {
            throw std::runtime_error("Yue2 " + prefix + " sampling options are invalid");
        }
    };
    apply_sampling(out.generation.abc, "abc");
    apply_sampling(out.generation.semantic, "semantic");
    validate_sampling(out.generation.abc, "abc");
    validate_sampling(out.generation.semantic, "semantic");
    out.abc = abc_from_options(options);
    if (!out.abc.empty() && out.cot == Yue2CotMode::Off) {
        throw std::runtime_error("Yue2 external ABC requires cot=melody or cot=full");
    }
    if (const auto nar_noise_file = runtime::find_option(options, {"nar_noise_file"})) {
        const std::filesystem::path path(*nar_noise_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 nar_noise_file does not exist: " + path.string());
        }
        out.nar_noise = engine::io::read_f32_file(path);
        if (out.nar_noise.empty()) {
            throw std::runtime_error("Yue2 nar_noise_file is empty: " + path.string());
        }
        if (out.nar_noise.size() % static_cast<size_t>(Yue2ModelConfig{}.latent_dim) != 0) {
            throw std::runtime_error("Yue2 nar_noise_file must contain raw float32 acoustic noise rows with 64 columns");
        }
    }
    if (const auto abc_ids_file = runtime::find_option(options, {"abc_ids_file"})) {
        const std::filesystem::path path(*abc_ids_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 abc_ids_file does not exist: " + path.string());
        }
        out.abc_ids = engine::io::read_i32_file(path);
        if (out.abc_ids.empty()) {
            throw std::runtime_error("Yue2 abc_ids_file is empty: " + path.string());
        }
    }
    if (const auto semantic_tokens_file = runtime::find_option(options, {"semantic_tokens_file"})) {
        const std::filesystem::path path(*semantic_tokens_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 semantic_tokens_file does not exist: " + path.string());
        }
        out.semantic_tokens = engine::io::read_i32_file(path);
        if (out.semantic_tokens.empty()) {
            throw std::runtime_error("Yue2 semantic_tokens_file is empty: " + path.string());
        }
    }
    if (const auto latent_in_file = runtime::find_option(options, {"latent_in_file"})) {
        const std::filesystem::path path(*latent_in_file);
        if (!engine::io::is_existing_file(path)) {
            throw std::runtime_error("Yue2 latent_in_file does not exist: " + path.string());
        }
        out.nar_latents = engine::io::read_f32_file(path);
        if (out.nar_latents.empty()) {
            throw std::runtime_error("Yue2 latent_in_file is empty: " + path.string());
        }
        if (out.nar_latents.size() % static_cast<size_t>(Yue2ModelConfig{}.latent_dim) != 0) {
            throw std::runtime_error("Yue2 latent_in_file must contain raw float32 latent rows with 64 columns");
        }
    }
    if (const auto latent_out_file = runtime::find_option(options, {"latent_out_file"})) {
        out.latent_out_file = *latent_out_file;
    }
    if (const auto abc_ids_out_file = runtime::find_option(options, {"abc_ids_out_file"})) {
        out.abc_ids_out_file = *abc_ids_out_file;
    }
    if (const auto semantic_tokens_out_file = runtime::find_option(options, {"semantic_tokens_out_file"})) {
        out.semantic_tokens_out_file = *semantic_tokens_out_file;
    }
    if (const auto noise_out_file = runtime::find_option(options, {"nar_noise_out_file"})) {
        out.diagnostics.noise_out_file = *noise_out_file;
    }
    if (const auto velocity_out_file = runtime::find_option(options, {"nar_velocity_out_file"})) {
        out.diagnostics.velocity_out_file = *velocity_out_file;
    }
    if (const auto tap_out_file = runtime::find_option(options, {"nar_tap_out_file"})) {
        out.diagnostics.tap_out_file = *tap_out_file;
    }
    if (const auto tap_stage = runtime::find_option(options, {"nar_tap_stage"})) {
        const std::string stage = *tap_stage;
        const auto dot = stage.find('.');
        const std::string base = dot == std::string::npos ? stage : stage.substr(0, dot);
        const std::string part = dot == std::string::npos ? std::string() : stage.substr(dot + 1);
        const bool simple = (base == "embed" || base == "prehead") && part.empty();
        const bool layer = base.rfind("layer", 0) == 0 && base.size() > 5 &&
                           base.find_first_not_of("0123456789", 5) == std::string::npos &&
                           (part.empty() || part == "norm" || part == "attn" || part == "mlp");
        if (!simple && !layer) {
            throw std::runtime_error(
                "Yue2 nar_tap_stage must be 'embed', 'prehead', 'layer<N>', or 'layer<N>.norm|attn|mlp'");
        }
        out.diagnostics.tap_stage = stage;
    }
    if (!out.diagnostics.tap_out_file.empty() && out.diagnostics.tap_stage.empty()) {
        throw std::runtime_error("Yue2 nar_tap_out_file requires nar_tap_stage=embed|prehead");
    }
}

Yue2Request normalize_request(Yue2Request out) {
    if (out.style.empty()) {
        throw std::runtime_error("Yue2 requires non-empty style");
    }
    if (out.lyrics.empty()) {
        throw std::runtime_error("Yue2 requires non-empty lyrics");
    }
    return out;
}

}  // namespace

Yue2Request parse_yue2_request(const runtime::TaskRequest & request, const Yue2GenerationConfig & defaults) {
    if (request.audio_input.has_value()) {
        throw std::runtime_error("Yue2 does not consume audio_input");
    }
    if (!request.input_artifacts.empty()) {
        throw std::runtime_error("Yue2 does not consume input artifacts");
    }
    Yue2Request out;
    out.generation = defaults;
    out.style = request_style(request);
    out.lyrics = request_lyrics(request);
    apply_options(out, request.options);
    return normalize_request(std::move(out));
}

Yue2Request parse_yue2_preparation_request(
    const runtime::SessionPreparationRequest & request,
    const Yue2GenerationConfig & defaults) {
    Yue2Request out;
    out.generation = defaults;
    if (request.text.has_value()) {
        out.lyrics = request.text->text;
    }
    if (const auto style = runtime::find_option(request.options, {"style"})) {
        out.style = *style;
    }
    apply_options(out, request.options);
    return out;
}

}  // namespace engine::models::yue2
