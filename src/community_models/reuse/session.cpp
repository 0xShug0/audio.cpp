#include "engine/community_models/reuse/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace engine::models::reuse {

ReuseSession::ReuseSession(runtime::TaskSpec task, runtime::SessionOptions options,
    std::shared_ptr<const ReuseAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
    : RuntimeSessionBase(options), task_(task), contract_(std::move(contract)) {
    runtime::validate_spec_backed_session_options(options, *contract_, "reuse", "RE-USE");
    if (execution_context().backend_type() == core::BackendType::Hip) {
        throw std::runtime_error("RE-USE is disabled on HIP: this backend has not been validated");
    }
    if (execution_context().backend_type() == core::BackendType::Metal) {
        throw std::runtime_error("RE-USE is disabled on Metal due to GGML kernel support bugs");
    }
    if (task.task != runtime::VoiceTaskKind::SpeechToSpeech || task.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("RE-USE supports only offline s2s");
    }
    const auto storage_option = options.options.find("reuse.weight_type");
    const auto storage = storage_option == options.options.end() ? assets::TensorStorageType::Native
        : assets::parse_tensor_storage_type(storage_option->second);
    runtime_ = std::make_unique<ReuseRuntime>(std::move(assets), execution_context(), storage);
}

ReuseSession::~ReuseSession() = default;
std::string ReuseSession::family() const { return "reuse"; }
runtime::VoiceTaskKind ReuseSession::task_kind() const { return task_.task; }
runtime::RunMode ReuseSession::run_mode() const { return task_.mode; }

void ReuseSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio || request.audio->sample_rate < 8000 || request.audio->sample_rate > 48000 || request.audio->channels < 1) {
        throw std::runtime_error("RE-USE requires 8-48 kHz input audio");
    }
    mark_prepared();
}

runtime::TaskResult ReuseSession::run(const runtime::TaskRequest & request) {
    require_prepared("RE-USE run");
    runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "RE-USE");
    if (!request.audio_input) {
        throw std::runtime_error("RE-USE requires audio_input");
    }
    const auto started = std::chrono::steady_clock::now();
    const auto & input = *request.audio_input;
    if (input.channels < 1 || input.samples.empty() || input.samples.size() % input.channels != 0) {
        throw std::runtime_error("RE-USE input audio shape mismatch");
    }
    const float seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_duration_sec"}).value_or(0);
    const float overlap_seconds = runtime::parse_finite_float_option(request.options, {"audio_chunk_overlap_sec"}).value_or(1);
    const int64_t chunk_samples = std::llrint(static_cast<double>(seconds) * input.sample_rate);
    const int64_t overlap = std::llrint(static_cast<double>(overlap_seconds) * input.sample_rate);
    if (seconds < 0 || overlap_seconds < 0 ||
        (seconds > 0 && (chunk_samples <= input.sample_rate / 50 || overlap >= chunk_samples))) {
        throw std::runtime_error("RE-USE chunk duration must exceed 20 ms and overlap must be shorter than a chunk");
    }
    const auto planar = input.channels == 1 ? std::vector<float>{}
        : audio::deinterleave_to_planar_channels(input.samples, input.channels);
    const int64_t samples = static_cast<int64_t>(input.samples.size() / input.channels);
    std::vector<float> output(planar.size());
    for (int channel = 0; channel < input.channels; ++channel) {
        const auto channel_samples = input.channels == 1 ? std::vector<float>{}
            : std::vector<float>(planar.begin() + channel * samples, planar.begin() + (channel + 1) * samples);
        const auto & signal = input.channels == 1 ? input.samples : channel_samples;
        std::vector<float> restored;
        if (chunk_samples == 0 || samples <= chunk_samples) {
            restored = runtime_->restore(signal, input.sample_rate);
        } else {
            const audio::STFTConfig window_config{chunk_samples, chunk_samples, chunk_samples,
                true, audio::STFTPadMode::Reflect, audio::STFTFamily::Kokoro};
            auto window = audio::get_cached_stft_window(window_config);
            for (auto & value : window) {
                value = std::max(value, 1e-7f);
            }
            restored.assign(samples, 0);
            std::vector<float> counter(samples, 0);
            const audio::AudioChunkSpec copy_spec{chunk_samples, chunk_samples, audio::AudioChunkPadMode::Zero};
            const int64_t stride = chunk_samples - overlap;
            const size_t count = 1 + (samples - chunk_samples + stride - 1) / stride;
            runtime_->process_chunks(count, input.sample_rate,
                [&](size_t index, std::vector<float> & chunk) {
                    chunk.resize(static_cast<size_t>(chunk_samples));
                    const int64_t start = std::min(static_cast<int64_t>(index) * stride, samples - chunk_samples);
                    const audio::AudioChunkSpan span{static_cast<int64_t>(index), start, chunk_samples, start, 0};
                    audio::copy_planar_chunk(chunk, signal, 1, samples, span, copy_spec);
                },
                [&](size_t index, const std::vector<float> & inferred) {
                    const int64_t start = std::min(static_cast<int64_t>(index) * stride, samples - chunk_samples);
                    const audio::AudioChunkSpan span{static_cast<int64_t>(index), start, chunk_samples, start, 0};
                    audio::overlap_add_planar_chunk(restored, counter, inferred, 1, samples, span,
                        window, audio::AudioChunkCounterMode::SharedAcrossLanes);
                });
            audio::normalize_overlap_added_planar(restored, counter, 1, samples, audio::AudioChunkCounterMode::SharedAcrossLanes);
        }
        if (input.channels == 1) {
            output = std::move(restored);
        } else {
            std::copy(restored.begin(), restored.end(), output.begin() + channel * samples);
        }
    }
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{input.sample_rate, input.channels,
        input.channels == 1 ? std::move(output) : audio::interleave_planar_channels(output, input.channels, samples)};
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
    return result;
}

std::vector<runtime::TaskResult> ReuseSession::run_batch(
    const std::vector<runtime::TaskRequest> & requests) {
    std::vector<runtime::TaskResult> results(requests.size());
    run_batch(requests, [&](size_t index, runtime::TaskResult result) {
        results[index] = std::move(result);
    });
    return results;
}

void ReuseSession::run_batch(
    const std::vector<runtime::TaskRequest> & requests,
    const runtime::IBatchedOfflineVoiceTaskSession::ResultCallback & on_result) {
    require_prepared("RE-USE run_batch");
    if (requests.empty()) {
        throw std::runtime_error("RE-USE batch must not be empty");
    }

    struct BatchLane {
        size_t request_index;
        int channel;
        std::vector<float> signal;
    };

    const auto started = std::chrono::steady_clock::now();
    std::vector<std::vector<float>> planar_outputs(requests.size());
    std::map<std::pair<int, int64_t>, std::vector<BatchLane>> groups;

    for (size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const auto & request = requests[request_index];
        runtime::validate_spec_backed_request_options(
            request.options, request.option_arrays, *contract_, "RE-USE");
        if (!request.audio_input) {
            throw std::runtime_error("RE-USE requires audio_input");
        }
        const auto & input = *request.audio_input;
        if (input.channels < 1 || input.samples.empty() || input.samples.size() % input.channels != 0) {
            throw std::runtime_error("RE-USE input audio shape mismatch");
        }
        const float chunk_seconds = runtime::parse_finite_float_option(
            request.options, {"audio_chunk_duration_sec"}).value_or(0);
        if (chunk_seconds != 0.0F) {
            throw std::runtime_error(
                "RE-USE native batching does not support audio_chunk_duration_sec");
        }

        const int64_t samples = static_cast<int64_t>(input.samples.size() / input.channels);
        auto planar = audio::deinterleave_to_planar_channels(input.samples, input.channels);
        planar_outputs[request_index].resize(planar.size());
        auto & group = groups[{input.sample_rate, samples}];
        for (int channel = 0; channel < input.channels; ++channel) {
            const auto begin = planar.begin() + channel * samples;
            group.push_back({request_index, channel, std::vector<float>(begin, begin + samples)});
        }
    }

    for (auto & [shape, lanes] : groups) {
        std::vector<std::vector<float>> signals;
        signals.reserve(lanes.size());
        for (auto & lane : lanes) {
            signals.push_back(std::move(lane.signal));
        }
        auto restored = runtime_->restore_batch(signals, shape.first);
        if (restored.size() != lanes.size()) {
            throw std::runtime_error("RE-USE batch result count mismatch");
        }
        for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
            const auto & lane = lanes[lane_index];
            const int64_t samples = shape.second;
            if (restored[lane_index].size() != static_cast<size_t>(samples)) {
                throw std::runtime_error("RE-USE batch output shape mismatch");
            }
            std::copy(restored[lane_index].begin(), restored[lane_index].end(),
                planar_outputs[lane.request_index].begin() + lane.channel * samples);
        }

        std::vector<size_t> group_requests;
        for (const auto & lane : lanes) {
            if (std::find(group_requests.begin(), group_requests.end(), lane.request_index) == group_requests.end()) {
                group_requests.push_back(lane.request_index);
            }
        }
        for (const size_t request_index : group_requests) {
            const auto & input = *requests[request_index].audio_input;
            const int64_t samples = static_cast<int64_t>(input.samples.size() / input.channels);
            runtime::TaskResult result;
            result.audio_output = runtime::AudioBuffer{
                input.sample_rate,
                input.channels,
                audio::interleave_planar_channels(planar_outputs[request_index], input.channels, samples),
            };
            on_result(request_index, std::move(result));
        }
    }
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
}

std::shared_ptr<runtime::IVoiceModelLoader> make_reuse_loader() {
    runtime::SpecBackedVoiceModelConfig<ReuseAssets> config;
    config.family = "reuse";
    config.load_assets = load_reuse_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const ReuseAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<ReuseSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::reuse
