#include "engine/community_models/minimax_music3/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;
constexpr const char * kFamily = "minimax_music3";

std::shared_ptr<const MiniMaxMusic3Assets> require_assets(std::shared_ptr<const MiniMaxMusic3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiniMax-Music3 session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MiniMax-Music3 session requires a model contract");
    }
    return contract;
}

std::unique_ptr<runtime::IVoiceTaskSession> create_minimax_music3_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MiniMaxMusic3Session>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

MiniMaxMusic3Session::MiniMaxMusic3Session(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(std::move(options)),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))) {
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration || task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MiniMax-Music3 supports offline audio generation sessions");
    }
    weight_context_bytes_ = runtime::parse_size_mb_option(
        this->options().options,
        {"minimax_music3.weight_context_mb"},
        weight_context_bytes_);
    bool mem_saver = true;
    if (const auto value = runtime::find_option(this->options().options, {"minimax_music3.mem_saver", "mem_saver"})) {
        mem_saver = runtime::parse_bool_option(*value, "minimax_music3.mem_saver");
    }
    runtime_ = std::make_unique<MiniMaxMusic3PipelineRuntime>(
        execution_context(),
        assets_,
        weight_context_bytes_,
        mem_saver);
    mark_prepared();
}

std::string MiniMaxMusic3Session::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MiniMaxMusic3Session::task_kind() const {
    return task_.task;
}

runtime::RunMode MiniMaxMusic3Session::run_mode() const {
    return task_.mode;
}

void MiniMaxMusic3Session::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::TaskResult MiniMaxMusic3Session::run(const runtime::TaskRequest & request) {
    require_prepared("MiniMax-Music3 run");
    const auto wall_start = Clock::now();
    auto generated = runtime_->generate(make_request(request));
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        generated.sample_rate,
        generated.channels,
        std::move(generated.samples),
    };
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    return result;
}

MiniMaxMusic3GenerateRequest MiniMaxMusic3Session::make_request(const runtime::TaskRequest & request) const {
    MiniMaxMusic3GenerateRequest out;
    if (request.text_input.has_value()) {
        out.caption = request.text_input->text;
    }
    if (out.caption.empty()) {
        throw std::runtime_error("MiniMax-Music3 requires the music description as text_input");
    }
    if (const auto value = runtime::find_option(request.options, {"lyrics"})) {
        out.lyrics = *value;
    }
    if (out.lyrics.empty()) {
        throw std::runtime_error("MiniMax-Music3 requires non-empty lyrics (--request-option lyrics=...)");
    }
    out.audio_duration =
        runtime::parse_float_option(request.options, {"duration_seconds", "audio_duration"}).value_or(out.audio_duration);
    if (out.audio_duration <= 0.0F) {
        throw std::runtime_error("MiniMax-Music3 audio_duration must be positive");
    }
    out.num_inference_steps =
        runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(out.num_inference_steps);
    if (out.num_inference_steps <= 0) {
        throw std::runtime_error("MiniMax-Music3 num_inference_steps must be positive");
    }
    out.guidance_scale = runtime::parse_float_option(request.options, {"guidance_scale"}).value_or(out.guidance_scale);
    out.seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(out.seed);
    return out;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_minimax_music3_loader() {
    runtime::SpecBackedVoiceModelConfig<MiniMaxMusic3Assets> config;
    config.family = kFamily;
    config.load_assets = load_minimax_music3_assets;
    config.create_session = create_minimax_music3_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::minimax_music3
