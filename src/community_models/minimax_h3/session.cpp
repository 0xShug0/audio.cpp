#include "engine/community_models/minimax_h3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_h3 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr const char * kFamily = "minimax_h3";

std::shared_ptr<const MiniMaxH3Assets> require_assets(std::shared_ptr<const MiniMaxH3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiniMax-H3 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MiniMax-H3 session requires a model contract");
    }
    return contract;
}

MiniMaxH3DitAccelerationMode parse_dit_acceleration_mode(const std::string & value) {
    if (value == "none" || value == "off" || value == "0") {
        return MiniMaxH3DitAccelerationMode::None;
    }
    if (value == "first_block_cache" || value == "first-block-cache") {
        return MiniMaxH3DitAccelerationMode::FirstBlockCache;
    }
    if (value == "spectrum") {
        return MiniMaxH3DitAccelerationMode::Spectrum;
    }
    throw std::runtime_error("MiniMax-H3 unsupported dit_acceleration mode: " + value);
}

MiniMaxH3SamplerMode parse_sampler_mode(const std::string & value) {
    if (value == "euler") {
        return MiniMaxH3SamplerMode::Euler;
    }
    if (value == "res_multistep" || value == "res-multistep") {
        return MiniMaxH3SamplerMode::ResMultistep;
    }
    if (value == "dpmpp_2m" || value == "dpmpp-2m") {
        return MiniMaxH3SamplerMode::Dpmpp2m;
    }
    if (value == "unipc" || value == "uni_pc" || value == "uni-pc") {
        return MiniMaxH3SamplerMode::UniPC;
    }
    throw std::runtime_error("MiniMax-H3 unsupported sampler mode: " + value);
}

std::unique_ptr<runtime::IVoiceTaskSession> create_minimax_h3_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MiniMaxH3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MiniMaxH3Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

MiniMaxH3Session::MiniMaxH3Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MiniMaxH3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MiniMax-H3 supports offline audio generation sessions");
    }
    weight_context_bytes_ = runtime::parse_size_mb_option(
        this->options().options,
        {"minimax_h3.weight_context_mb"},
        weight_context_bytes_);
    bool mem_saver = true;
    if (const auto value = runtime::find_option(this->options().options, {"minimax_h3.mem_saver", "mem_saver"})) {
        mem_saver = runtime::parse_bool_option(*value, "minimax_h3.mem_saver");
    }
    runtime_ = std::make_unique<MiniMaxH3PipelineRuntime>(
        execution_context(),
        assets_,
        weight_context_bytes_,
        mem_saver);
    mark_prepared();
}

std::string MiniMaxH3Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MiniMaxH3Session::task_kind() const {
    return task_.task;
}

runtime::RunMode MiniMaxH3Session::run_mode() const {
    return task_.mode;
}

void MiniMaxH3Session::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult MiniMaxH3Session::run(const runtime::TaskRequest & request) {
    require_prepared("MiniMax-H3 run");
    const auto wall_start = Clock::now();
    auto generated = runtime_->generate(make_request(request));
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        generated.sample_rate,
        generated.channels,
        std::move(generated.samples),
    };
    if (generated.video.has_value()) {
        runtime::VoiceArtifact artifact;
        artifact.kind = runtime::ArtifactKind::Custom;
        artifact.id = "minimax_h3_video_rgb24";
        artifact.payload = std::move(generated.video->rgb24);
        artifact.meta.emplace("format", "rgb24");
        artifact.meta.emplace("width", std::to_string(generated.video->width));
        artifact.meta.emplace("height", std::to_string(generated.video->height));
        artifact.meta.emplace("frames", std::to_string(generated.video->frames));
        artifact.meta.emplace("fps", std::to_string(generated.video->fps));
        result.output_artifacts.push_back(std::move(artifact));
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

MiniMaxH3GenerateRequest MiniMaxH3Session::make_request(const runtime::TaskRequest & request) const {
    MiniMaxH3GenerateRequest out;
    if (request.text_input.has_value()) {
        out.prompt = request.text_input->text;
    }
    if (out.prompt.empty()) {
        throw std::runtime_error("MiniMax-H3 requires non-empty text_input");
    }
    out.steps = runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(assets_->config.denoise_steps);
    if (out.steps <= 0) {
        throw std::runtime_error("MiniMax-H3 num_inference_steps must be positive");
    }
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(out.seed);
    out.height = runtime::parse_int_option(request.options, {"height"}).value_or(assets_->config.height);
    out.width = runtime::parse_int_option(request.options, {"width"}).value_or(assets_->config.width);
    out.num_frames = runtime::parse_int_option(request.options, {"num_frames"}).value_or(assets_->config.num_frames);
    out.guidance_scale = runtime::parse_float_option(request.options, {"guidance_scale"}).value_or(out.guidance_scale);
    if (const auto value = runtime::find_option(request.options, {"sampler", "minimax_h3.sampler"})) {
        out.sampler = parse_sampler_mode(*value);
    }
    out.flow_shift = runtime::parse_float_option(request.options, {"flow_shift"}).value_or(out.flow_shift);
    out.audio_flow_shift = runtime::parse_float_option(request.options, {"audio_flow_shift"}).value_or(out.audio_flow_shift);
    if (const auto value = runtime::find_option(request.options, {"return_video"})) {
        out.return_video = runtime::parse_bool_option(*value, "return_video");
    }
    if (const auto value = runtime::find_option(request.options, {"text_layerwise", "minimax_h3.text_layerwise"})) {
        out.text_layerwise = runtime::parse_bool_option(*value, "text_layerwise");
    }
    if (const auto value = runtime::find_option(request.options, {"dit_layerwise", "minimax_h3.dit_layerwise"})) {
        out.dit_layerwise = runtime::parse_bool_option(*value, "dit_layerwise");
    }
    out.text_layerwise_batch =
        runtime::parse_int_option(request.options, {"text_layerwise_batch", "minimax_h3.text_layerwise_batch"})
            .value_or(static_cast<int>(out.text_layerwise_batch));
    out.dit_layerwise_batch =
        runtime::parse_int_option(request.options, {"dit_layerwise_batch", "minimax_h3.dit_layerwise_batch"})
            .value_or(static_cast<int>(out.dit_layerwise_batch));
    out.dit_mlp_chunk_tokens =
        runtime::parse_int_option(request.options, {"dit_mlp_chunk_tokens", "minimax_h3.dit_mlp_chunk_tokens"})
            .value_or(static_cast<int>(out.dit_mlp_chunk_tokens));
    if (const auto value = runtime::find_option(request.options, {"dit_acceleration", "minimax_h3.dit_acceleration"})) {
        out.dit_acceleration = parse_dit_acceleration_mode(*value);
    }
    out.first_block_cache_threshold =
        runtime::parse_float_option(request.options, {"first_block_cache_threshold", "minimax_h3.first_block_cache_threshold"})
            .value_or(out.first_block_cache_threshold);
    out.first_block_cache_start_percent =
        runtime::parse_float_option(request.options, {"first_block_cache_start_percent", "minimax_h3.first_block_cache_start_percent"})
            .value_or(out.first_block_cache_start_percent);
    out.first_block_cache_end_percent =
        runtime::parse_float_option(request.options, {"first_block_cache_end_percent", "minimax_h3.first_block_cache_end_percent"})
            .value_or(out.first_block_cache_end_percent);
    const auto start_sigma = runtime::parse_float_option(request.options, {"first_block_cache_start_sigma", "minimax_h3.first_block_cache_start_sigma"});
    const auto end_sigma = runtime::parse_float_option(request.options, {"first_block_cache_end_sigma", "minimax_h3.first_block_cache_end_sigma"});
    if (start_sigma.has_value() || end_sigma.has_value()) {
        if (!start_sigma.has_value() || !end_sigma.has_value()) {
            throw std::runtime_error("MiniMax-H3 first-block cache sigma window requires both start and end sigma");
        }
        out.first_block_cache_start_sigma = *start_sigma;
        out.first_block_cache_end_sigma = *end_sigma;
        out.first_block_cache_sigma_window = true;
    }
    out.first_block_cache_max_consecutive =
        runtime::parse_int_option(request.options, {"first_block_cache_max_consecutive", "minimax_h3.first_block_cache_max_consecutive"})
            .value_or(static_cast<int>(out.first_block_cache_max_consecutive));
    out.spectrum_warmup_steps =
        runtime::parse_int_option(request.options, {"spectrum_warmup_steps", "minimax_h3.spectrum_warmup_steps"})
            .value_or(static_cast<int>(out.spectrum_warmup_steps));
    out.spectrum_initial_window =
        runtime::parse_float_option(request.options, {"spectrum_initial_window", "minimax_h3.spectrum_initial_window"})
            .value_or(out.spectrum_initial_window);
    out.spectrum_flex_window =
        runtime::parse_float_option(request.options, {"spectrum_flex_window", "minimax_h3.spectrum_flex_window"})
            .value_or(out.spectrum_flex_window);
    out.spectrum_degree =
        runtime::parse_int_option(request.options, {"spectrum_degree", "minimax_h3.spectrum_degree"})
            .value_or(static_cast<int>(out.spectrum_degree));
    out.spectrum_history_size =
        runtime::parse_int_option(request.options, {"spectrum_history_size", "minimax_h3.spectrum_history_size"})
            .value_or(static_cast<int>(out.spectrum_history_size));
    out.spectrum_ridge_lambda =
        runtime::parse_float_option(request.options, {"spectrum_ridge_lambda", "minimax_h3.spectrum_ridge_lambda"})
            .value_or(out.spectrum_ridge_lambda);
    if (out.text_layerwise_batch <= 0 || out.dit_layerwise_batch <= 0 || out.dit_mlp_chunk_tokens < 0) {
        throw std::runtime_error("MiniMax-H3 layerwise batch values must be valid");
    }
    if (out.first_block_cache_start_percent < 0.0F || out.first_block_cache_start_percent > 1.0F ||
        out.first_block_cache_end_percent < 0.0F || out.first_block_cache_end_percent > 1.0F ||
        out.first_block_cache_start_percent >= out.first_block_cache_end_percent) {
        throw std::runtime_error("MiniMax-H3 first-block cache percent window must satisfy 0 <= start < end <= 1");
    }
    if (out.first_block_cache_max_consecutive <= 0 || out.spectrum_warmup_steps <= 0 ||
        out.spectrum_degree < 0 || out.spectrum_history_size <= 0) {
        throw std::runtime_error("MiniMax-H3 DiT acceleration options must be valid");
    }
    if (const auto negative = runtime::find_option(request.options, {"negative_prompt", "negative"})) {
        out.negative_prompt = *negative;
    }
    return out;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_minimax_h3_loader() {
    runtime::SpecBackedVoiceModelConfig<MiniMaxH3Assets> config;
    config.family = kFamily;
    config.load_assets = load_minimax_h3_assets;
    config.create_session = create_minimax_h3_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::minimax_h3
