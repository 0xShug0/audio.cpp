#include "engine/models/universr/session.h"

#include "engine/models/universr/frontend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/sampling/torch_random.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace engine::models::universr {

UniverSRSession::UniverSRSession(runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const UniverSRAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
    runtime::validate_spec_backed_session_options(options, *contract_, "universr", "UniverSR");
    if (task.task != runtime::VoiceTaskKind::SpeechToSpeech || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("UniverSR supports only offline s2s");
    }
    using Storage = assets::TensorStorageType;
    const auto storage = runtime::parse_tensor_storage_option(options.options, "universr.weight_type", Storage::Native,
        {Storage::Native, Storage::F32, Storage::F16, Storage::BF16, Storage::Q8_0,
         Storage::Q4_0, Storage::Q4_1, Storage::Q5_0, Storage::Q5_1,
         Storage::Q2_K, Storage::Q3_K, Storage::Q4_K, Storage::Q5_K, Storage::Q6_K});
    runtime_ = std::make_unique<UniverSRRuntime>(assets_, execution_context(), storage);
}

UniverSRSession::~UniverSRSession() = default;
std::string UniverSRSession::family() const { return "universr"; }
runtime::VoiceTaskKind UniverSRSession::task_kind() const { return task_.task; }
runtime::RunMode UniverSRSession::run_mode() const { return task_.mode; }

void UniverSRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio || request.audio->sample_rate <= 0 || request.audio->channels < 1) {
        throw std::runtime_error("UniverSR requires input audio with a valid sample rate and channel count");
    }
    mark_prepared();
}

runtime::TaskResult UniverSRSession::run(const runtime::TaskRequest & request) {
    require_prepared("UniverSR run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "UniverSR");
    if (!request.audio_input) {
        throw std::runtime_error("UniverSR requires audio_input");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto & input = *request.audio_input;
    const int rate = runtime::parse_int_option(request.options, {"input_sample_rate"}).value_or(input.sample_rate);
    const int steps = runtime::parse_int_option(request.options, {"num_inference_steps"}).value_or(4);
    const std::string method = runtime::find_option(request.options, {"sampler_mode"}).value_or("midpoint");
    const float guidance = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(1.5f);
    const auto seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(42);
    const float chunk_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_duration_sec"}).value_or(0.0f);
    if (steps < 1 || guidance < 0 || (method != "euler" && method != "midpoint" && method != "rk4")) {
        throw std::runtime_error("UniverSR requires positive steps, nonnegative guidance, and euler/midpoint/rk4");
    }
    for (float sample : input.samples) {
        if (!std::isfinite(sample)) {
            throw std::runtime_error("UniverSR input audio contains non-finite samples");
        }
    }
    const auto & execution = execution_context();
    if (input.channels < 1 || input.sample_rate <= 0 || input.samples.empty() || input.samples.size() % input.channels != 0 || chunk_seconds < 0) {
        throw std::runtime_error("UniverSR requires valid audio and a nonnegative chunk duration");
    }
    const size_t total_samples = input.samples.size() / input.channels;
    const double requested_chunk_samples = static_cast<double>(chunk_seconds) * input.sample_rate;
    if (chunk_seconds > 0 && requested_chunk_samples < 1) {
        throw std::runtime_error("UniverSR chunk duration must contain at least one input sample");
    }
    const size_t chunk_samples = chunk_seconds == 0 || requested_chunk_samples >= static_cast<double>(total_samples)
        ? total_samples : static_cast<size_t>(requested_chunk_samples);
    const auto policy = sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(), execution.config().device,
        "universr.noise", "UniverSR", sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    uint64_t noise_offset = 0;
    std::vector<float> waveform;
    for (size_t start = 0; start < total_samples; start += chunk_samples) {
        std::vector<float> segment;
        if (chunk_samples != total_samples) {
            const size_t end = std::min(start + chunk_samples, total_samples);
            segment.assign(input.samples.begin() + start * input.channels, input.samples.begin() + end * input.channels);
        }
        const auto features = analyze_audio(*assets_, chunk_samples == total_samples ? input.samples : segment,
            input.channels, input.sample_rate, rate, execution.config().threads);
        const auto count = static_cast<size_t>(2 * assets_->config.hr_freq_bins * features.frames);
        const auto source_noise = sampling::generate_torch_cuda_tensor_iterator_randn(count, seed, noise_offset, policy);
        noise_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, policy);
        // randn_like preserves the STFT's dense time/frequency/complex memory order.
        std::vector<float> noise(source_noise.size());
        for (int64_t frame = 0; frame < features.frames; ++frame) {
            for (int64_t bin = 0; bin < assets_->config.hr_freq_bins; ++bin) {
                for (int channel = 0; channel < 2; ++channel) {
                    noise[static_cast<size_t>((channel * assets_->config.hr_freq_bins + bin) * features.frames + frame)] =
                        source_noise[static_cast<size_t>((frame * assets_->config.hr_freq_bins + bin) * 2 + channel)];
                }
            }
        }
        runtime_->prepare_condition(features.low_spectrum, features.frames, rate / 1000);
        auto high = runtime_->integrate_flow(noise, steps, method, guidance);
        auto restored = synthesize_audio(*assets_, features, high, execution.config().threads);
        waveform.insert(waveform.end(), restored.begin(), restored.end());
        debug::timing_log_scalar("universr.chunk.index", start / chunk_samples);
    }
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{assets_->config.sample_rate, 1, std::move(waveform)};
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_universr_loader() {
    runtime::SpecBackedVoiceModelConfig<UniverSRAssets> config;
    config.family = "universr";
    config.load_assets = load_universr_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const UniverSRAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<UniverSRSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::universr
