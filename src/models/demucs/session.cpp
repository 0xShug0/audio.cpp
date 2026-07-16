#include "engine/models/demucs/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::demucs {
namespace {

assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const std::string & key,
    assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    const auto storage_type = assets::parse_tensor_storage_type(it->second);
    validate_demucs_weight_storage_type(storage_type);
    return storage_type;
}

std::pair<float, float> normalize_separator_audio(runtime::AudioBuffer & audio) {
    if (audio.channels <= 0 || audio.samples.empty()) {
        throw std::runtime_error("HTDemucs separator normalization received empty audio");
    }
    const int64_t frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    double mean = 0.0;
    for (int64_t frame = 0; frame < frames; ++frame) {
        double mono = 0.0;
        for (int ch = 0; ch < audio.channels; ++ch) {
            mono += static_cast<double>(audio.samples[static_cast<size_t>(frame * audio.channels + ch)]);
        }
        mono /= static_cast<double>(audio.channels);
        mean += mono;
    }
    mean /= static_cast<double>(frames);

    double variance = 0.0;
    for (int64_t frame = 0; frame < frames; ++frame) {
        double mono = 0.0;
        for (int ch = 0; ch < audio.channels; ++ch) {
            mono += static_cast<double>(audio.samples[static_cast<size_t>(frame * audio.channels + ch)]);
        }
        mono /= static_cast<double>(audio.channels);
        const double delta = mono - mean;
        variance += delta * delta;
    }
    const double denom = frames > 1 ? static_cast<double>(frames - 1) : 1.0;
    const float std = static_cast<float>(std::sqrt(variance / denom) + 1.0e-8);
    const float mean_f32 = static_cast<float>(mean);
#ifdef _OPENMP
    #pragma omp parallel for if(audio.samples.size() >= 1 << 16)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(audio.samples.size()); ++i) {
        audio.samples[static_cast<size_t>(i)] = (audio.samples[static_cast<size_t>(i)] - mean_f32) / std;
    }
    return {mean_f32, std};
}

}  // namespace

HTDemucsSession::HTDemucsSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const HTDemucsAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("HTDemucs session requires assets");
    }
    if (task_.task != runtime::VoiceTaskKind::SourceSeparation) {
        throw std::runtime_error("HTDemucs models only support --task sep");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("HTDemucs models only support offline mode");
    }
    if (assets_->submodels.size() != 1) {
        throw std::runtime_error("HTDemucs native session currently supports only single-model manifests");
    }
    const auto default_weight_storage = execution_context().uses_host_graph_plan()
        ? assets::TensorStorageType::F32
        : (execution_context().backend_type() == core::BackendType::Cuda
            ? assets::TensorStorageType::F16
            : assets::TensorStorageType::Native);
    weight_storage_type_ = option_weight_type(
        RuntimeSessionBase::options(),
        "htdemucs.weight_type",
        default_weight_storage);
    pipeline_ = std::make_unique<HTDemucsPipeline>(assets_->submodels.front(), execution_context(), weight_storage_type_);
    const auto & config = pipeline_->config();
    chunk_size_ = config.segment_samples;
    step_ = static_cast<int64_t>(std::llround(static_cast<double>(chunk_size_) * 0.75));
    if (step_ <= 0 || step_ > chunk_size_) {
        throw std::runtime_error("HTDemucs chunk step is invalid");
    }
    chunk_window_ = engine::audio::make_triangular_overlap_window(chunk_size_);
    chunk_planar_work_.resize(static_cast<size_t>(config.audio_channels * chunk_size_));
    assets_->submodels.front()->tensor_source->release_storage();
}

HTDemucsSession::~HTDemucsSession() = default;

std::string HTDemucsSession::family() const {
    return "htdemucs";
}

runtime::VoiceTaskKind HTDemucsSession::task_kind() const {
    return task_.task;
}

runtime::RunMode HTDemucsSession::run_mode() const {
    return task_.mode;
}

void HTDemucsSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("HTDemucs prepare() requires an audio contract");
    }
    const auto & config = pipeline_->config();
    if (request.audio->sample_rate != config.sample_rate) {
        throw std::runtime_error(
            "HTDemucs prepare() sample rate mismatch: expected " +
            std::to_string(config.sample_rate) + ", got " +
            std::to_string(request.audio->sample_rate));
    }
    const bool mono_compatible = config.audio_channels == 2 && request.audio->channels == 1;
    if (request.audio->channels != config.audio_channels && !mono_compatible) {
        throw std::runtime_error(
            "HTDemucs prepare() channel mismatch: expected " +
            std::to_string(config.audio_channels) + ", got " +
            std::to_string(request.audio->channels));
    }
    mark_prepared();
}

runtime::TaskResult HTDemucsSession::run(const runtime::TaskRequest & request) {
    require_prepared("HTDemucs run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("HTDemucs run() requires audio_input");
    }

    const auto & config = pipeline_->config();
    const auto & request_audio = *request.audio_input;
    const bool original_mono = config.audio_channels == 2 && request_audio.channels == 1;
    if (request_audio.sample_rate != config.sample_rate ||
        (request_audio.channels != config.audio_channels && !original_mono)) {
        throw std::runtime_error("HTDemucs run() audio_input does not match prepared audio contract");
    }

    auto audio = original_mono
        ? runtime::AudioBuffer{
              request_audio.sample_rate,
              config.audio_channels,
              engine::audio::duplicate_mono_to_interleaved_channels(request_audio.samples, config.audio_channels)}
        : request_audio;
    const auto wall_start = std::chrono::steady_clock::now();
    const auto normalize_start = std::chrono::steady_clock::now();
    const auto [global_mean, global_std] = normalize_separator_audio(audio);
    const auto normalize_end = std::chrono::steady_clock::now();
    const int64_t total_frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    const int num_sources = static_cast<int>(config.sources.size());
    result_work_.resize(static_cast<size_t>(num_sources * audio.channels * total_frames));
    std::fill(result_work_.begin(), result_work_.end(), 0.0f);
    counter_work_.resize(static_cast<size_t>(total_frames));
    std::fill(counter_work_.begin(), counter_work_.end(), 0.0f);
    double chunk_copy_ms = 0.0;
    double frontend_ms = 0.0;
    double graph_ms = 0.0;
    double postprocess_ms = 0.0;
    double graph_rebuild_ms = 0.0;
    double merge_ms = 0.0;

    const engine::audio::AudioChunkSpec chunk_spec{
        chunk_size_,
        step_,
        engine::audio::AudioChunkPadMode::Zero,
        engine::audio::AudioChunkTailAlignment::Center,
        0,
    };
    std::vector<float> merge_window(static_cast<size_t>(chunk_size_), 0.0F);
    for (const auto & chunk : engine::audio::plan_audio_chunks(total_frames, chunk_spec)) {
        const auto copy_start = std::chrono::steady_clock::now();
        engine::audio::copy_interleaved_chunk_to_planar(
            chunk_planar_work_,
            audio.samples,
            audio.channels,
            total_frames,
            chunk,
            chunk_spec);
        const auto copy_end = std::chrono::steady_clock::now();
        chunk_copy_ms += debug::elapsed_ms(copy_start, copy_end);
        const auto & chunk_sources = pipeline_->separate_chunk(chunk_planar_work_);
        const auto & timing = pipeline_->last_timing();
        frontend_ms += timing.frontend_ms;
        graph_ms += timing.graph_ms;
        postprocess_ms += timing.postprocess_ms;
        graph_rebuild_ms += timing.graph_rebuild_ms;
        const int64_t trim_offset = chunk.valid_start_in_chunk;
        const auto merge_start = std::chrono::steady_clock::now();
        std::fill(merge_window.begin(), merge_window.end(), 0.0F);
        for (int64_t i = 0; i < chunk.valid_samples; ++i) {
            merge_window[static_cast<size_t>(trim_offset + i)] = chunk_window_[static_cast<size_t>(i)];
        }
        engine::audio::overlap_add_planar_chunk(
            result_work_,
            counter_work_,
            chunk_sources,
            num_sources * audio.channels,
            total_frames,
            chunk,
            merge_window,
            engine::audio::AudioChunkCounterMode::SharedAcrossLanes);
        merge_ms += debug::elapsed_ms(merge_start);
    }

    runtime::TaskResult result;
    for (int source = 0; source < num_sources; ++source) {
        runtime::NamedAudioBuffer named;
        named.id = config.sources[static_cast<size_t>(source)];
        named.audio.sample_rate = audio.sample_rate;
        named.audio.channels = original_mono ? 1 : audio.channels;
        if (original_mono) {
            named.audio.samples.resize(static_cast<size_t>(total_frames));
            for (int64_t i = 0; i < total_frames; ++i) {
                const float denom = counter_work_[static_cast<size_t>(i)] > 1.0e-8f
                    ? counter_work_[static_cast<size_t>(i)]
                    : 1.0f;
                const float * src = result_work_.data() + static_cast<size_t>((source * audio.channels) * total_frames);
                named.audio.samples[static_cast<size_t>(i)] = src[i] / denom * global_std + global_mean;
            }
        } else {
            named.audio.samples.resize(static_cast<size_t>(audio.channels * total_frames));
            for (int64_t i = 0; i < total_frames; ++i) {
                const float denom = counter_work_[static_cast<size_t>(i)] > 1.0e-8f
                    ? counter_work_[static_cast<size_t>(i)]
                    : 1.0f;
                for (int ch = 0; ch < audio.channels; ++ch) {
                    const float * src = result_work_.data() + static_cast<size_t>((source * audio.channels + ch) * total_frames);
                    named.audio.samples[static_cast<size_t>(i * audio.channels + ch)] = src[i] / denom * global_std + global_mean;
                }
            }
        }
        result.named_audio_outputs.push_back(std::move(named));
    }
    debug::timing_log_scalar("htdemucs.normalize_ms", debug::elapsed_ms(normalize_start, normalize_end));
    debug::timing_log_scalar("htdemucs.chunk_copy_ms", chunk_copy_ms);
    debug::timing_log_scalar("htdemucs.frontend_ms", frontend_ms);
    debug::timing_log_scalar("htdemucs.graph.total_ms", graph_ms);
    debug::timing_log_scalar("htdemucs.graph.rebuild_ms", graph_rebuild_ms);
    debug::timing_log_scalar("htdemucs.postprocess_ms", postprocess_ms);
    debug::timing_log_scalar("htdemucs.merge_ms", merge_ms);
    debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::demucs
