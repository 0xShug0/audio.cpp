#include "engine/models/apollo/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::apollo {

ApolloSession::ApolloSession(runtime::TaskSpec task, runtime::SessionOptions options,
                             std::shared_ptr<const ApolloAssets> assets,
                             std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)) {
    runtime::validate_spec_backed_session_options(options, *contract_, "apollo", "Apollo");
    if (task.task != runtime::VoiceTaskKind::SpeechToSpeech || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Apollo supports only offline s2s");
    }
    using Storage = assets::TensorStorageType;
    const auto storage = runtime::parse_tensor_storage_option(options.options, "apollo.weight_type", Storage::Native,
        {Storage::Native, Storage::F32, Storage::F16, Storage::BF16, Storage::Q8_0,
         Storage::Q4_0, Storage::Q4_1, Storage::Q5_0, Storage::Q5_1,
         Storage::Q2_K, Storage::Q3_K, Storage::Q4_K, Storage::Q5_K, Storage::Q6_K});
    runtime_ = std::make_unique<ApolloRuntime>(assets_, execution_context(), storage);
    assets_->tensors->release_storage();
}

ApolloSession::~ApolloSession() = default;
std::string ApolloSession::family() const { return "apollo"; }
runtime::VoiceTaskKind ApolloSession::task_kind() const { return task_.task; }
runtime::RunMode ApolloSession::run_mode() const { return task_.mode; }

void ApolloSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio || request.audio->sample_rate != assets_->config.sample_rate || request.audio->channels < 1) {
        throw std::runtime_error("Apollo requires 44.1 kHz input audio");
    }
    mark_prepared();
}

runtime::TaskResult ApolloSession::run(const runtime::TaskRequest & request) {
    require_prepared("Apollo run");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "Apollo");
    if (!request.audio_input) {
        throw std::runtime_error("Apollo requires audio_input");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto & input = *request.audio_input;
    if (input.sample_rate != assets_->config.sample_rate || input.channels < 1 ||
        input.samples.size() % input.channels != 0) {
        throw std::runtime_error("Apollo input audio contract mismatch");
    }
    const int64_t samples = static_cast<int64_t>(input.samples.size() / input.channels);
    if (samples <= assets_->config.n_fft / 2) {
        throw std::runtime_error("Apollo input must be longer than the STFT reflection pad");
    }
    const float chunk_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_duration_sec"}).value_or(0.0f);
    const float overlap_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_overlap_sec"}).value_or(1.0f);
    const float pad_seconds = runtime::parse_finite_float_option(request.options, {"edge_pad_duration_sec"}).value_or(0.0f);
    const int64_t chunk_samples = std::llrint(static_cast<double>(chunk_seconds) * input.sample_rate);
    const int64_t overlap_samples = std::llrint(static_cast<double>(overlap_seconds) * input.sample_rate);
    const int64_t pad_samples = std::llrint(static_cast<double>(pad_seconds) * input.sample_rate);
    if (chunk_seconds < 0 || overlap_seconds < 0 || pad_seconds < 0 ||
        (chunk_seconds > 0 && (chunk_samples <= assets_->config.n_fft / 2 || overlap_samples * 2 > chunk_samples))) {
        throw std::runtime_error("Apollo chunk size must exceed the STFT pad and overlap must not exceed half the chunk size");
    }

    const auto planar = audio::deinterleave_to_planar_channels(input.samples, input.channels);
    std::vector<float> output(planar.size());
    for (int channel = 0; channel < input.channels; ++channel) {
        std::vector<float> signal(planar.begin() + channel * samples, planar.begin() + (channel + 1) * samples);
        std::vector<float> restored;
        if (chunk_samples == 0 || samples <= chunk_samples) {
            restored = runtime_->restore(signal);
        } else {
            const int64_t padded_samples = chunk_samples + 2 * pad_samples;
            const audio::AudioChunkSpec copy_spec{padded_samples, padded_samples, audio::AudioChunkPadMode::Zero};
            restored.assign(static_cast<size_t>(samples), 0.0f);
            std::vector<float> counter(static_cast<size_t>(samples), 0.0f);
            std::vector<float> chunk(static_cast<size_t>(padded_samples));
            int64_t index = 0;
            for (int64_t start = 0;; start += chunk_samples - overlap_samples, ++index) {
                const int64_t valid = std::min(chunk_samples, samples - start);
                const bool last = start + chunk_samples >= samples;
                int64_t padded_start = std::max<int64_t>(0, start - pad_samples);
                if (pad_samples && last) {
                    padded_start = std::max<int64_t>(0, samples - padded_samples);
                }
                audio::copy_planar_chunk(chunk, signal, 1, samples,
                    {index, start, valid, padded_start, start - padded_start}, copy_spec);
                auto inferred = runtime_->restore(chunk);
                const int64_t offset = start - padded_start;
                std::vector<float> kept(inferred.begin() + offset, inferred.begin() + offset + valid);
                const int64_t fade = std::min(overlap_samples, valid);
                std::vector<float> window(static_cast<size_t>(valid), 1.0f);
                if (fade > 0) {
                    auto fades = audio::make_linear_fade_window(2 * fade, fade);
                    if (fade == 1) {
                        fades[0] = 0.0f;
                        fades[1] = 0.0f;
                    }
                    if (index > 0) {
                        std::copy_n(fades.begin(), fade, window.begin());
                    }
                    if (!last) {
                        std::copy_n(fades.begin() + fade, fade, window.end() - fade);
                    }
                }
                audio::overlap_add_planar_chunk(restored, counter, kept, 1, samples,
                    {index, start, valid, start, 0}, window, audio::AudioChunkCounterMode::SharedAcrossLanes);
                if (last) {
                    break;
                }
            }
            audio::normalize_overlap_added_planar(restored, counter, 1, samples, audio::AudioChunkCounterMode::SharedAcrossLanes);
        }
        std::copy(restored.begin(), restored.end(), output.begin() + channel * samples);
    }
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{input.sample_rate, input.channels,
        audio::interleave_planar_channels(output, input.channels, samples)};
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_apollo_loader() {
    runtime::SpecBackedVoiceModelConfig<ApolloAssets> config;
    config.family = "apollo";
    config.load_assets = load_apollo_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const ApolloAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<ApolloSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::apollo
