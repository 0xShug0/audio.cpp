#include "engine/models/roformer/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::roformer {
namespace {

using Clock = std::chrono::steady_clock;

assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const std::string & key,
    assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    const auto storage_type = assets::parse_tensor_storage_type(it->second);
    validate_roformer_weight_storage_type(storage_type);
    return storage_type;
}

runtime::AudioBuffer derive_instrumental(
    const runtime::AudioBuffer & mixture,
    const runtime::AudioBuffer & vocals) {
    if (mixture.sample_rate != vocals.sample_rate || mixture.channels != vocals.channels || mixture.samples.size() != vocals.samples.size()) {
        throw std::runtime_error("RoFormer residual derivation requires matching mixture and vocals buffers");
    }
    runtime::AudioBuffer out = mixture;
#ifdef _OPENMP
    #pragma omp parallel for if(out.samples.size() >= 1 << 16)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(out.samples.size()); ++i) {
        out.samples[static_cast<size_t>(i)] -= vocals.samples[static_cast<size_t>(i)];
    }
    return out;
}

}  // namespace

RoformerSession::RoformerSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const RoformerAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(std::move(assets)) {
    if (assets_ == nullptr) {
        throw std::runtime_error("RoFormer session requires assets");
    }
    if (task_.task != runtime::VoiceTaskKind::SourceSeparation) {
        throw std::runtime_error("RoFormer models only support --task sep");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("RoFormer models only support offline mode");
    }
    const auto default_weight_storage =
        core::requested_backend_uses_host_graph_plan(RuntimeSessionBase::options().backend)
            ? assets::TensorStorageType::F32
            : assets::TensorStorageType::Native;
    weight_storage_type_ = option_weight_type(
        RuntimeSessionBase::options(),
        std::string(kMelBandRoformerFamily) + ".weight_type",
        default_weight_storage);
    runtime_ = std::make_unique<RoformerRuntime>(assets_, execution_context(), weight_storage_type_);
    const auto & config = runtime_->config();
    if (config.chunk_size <= 0) {
        throw std::runtime_error("RoFormer config chunk_size must be positive");
    }
    if (config.inference_num_overlap <= 0) {
        throw std::runtime_error("RoFormer config num_overlap must be positive");
    }
    chunk_size_ = config.chunk_size;
    const int64_t overlap = config.inference_num_overlap;
    step_ = chunk_size_ / overlap;
    fade_size_ = chunk_size_ / 10;
    border_ = chunk_size_ - step_;
    if (step_ <= 0) {
        throw std::runtime_error("RoFormer chunk step must be positive");
    }
    chunk_window_ = engine::audio::make_linear_fade_window(chunk_size_, fade_size_);
    chunk_planar_work_.resize(static_cast<size_t>(config.channels * chunk_size_));
    assets_->tensor_source->release_storage();
}

RoformerSession::~RoformerSession() = default;

std::string RoformerSession::family() const {
    return std::string(kMelBandRoformerFamily);
}

runtime::VoiceTaskKind RoformerSession::task_kind() const {
    return task_.task;
}

runtime::RunMode RoformerSession::run_mode() const {
    return task_.mode;
}

void RoformerSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("RoFormer prepare() requires an audio contract");
    }
    if (request.audio->sample_rate != assets_->config.sample_rate) {
        throw std::runtime_error(
            "RoFormer prepare() sample rate mismatch: expected " +
            std::to_string(assets_->config.sample_rate) + ", got " +
            std::to_string(request.audio->sample_rate));
    }
    const bool mono_compatible = assets_->config.channels == 2 && request.audio->channels == 1;
    if (request.audio->channels != assets_->config.channels && !mono_compatible) {
        throw std::runtime_error(
            "RoFormer prepare() channel mismatch: expected " +
            std::to_string(assets_->config.channels) + ", got " +
            std::to_string(request.audio->channels));
    }
    mark_prepared();
}

runtime::TaskResult RoformerSession::run(const runtime::TaskRequest & request) {
    require_prepared("RoFormer run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("RoFormer run() requires audio_input");
    }
    const auto total_start = Clock::now();

    const auto & config = runtime_->config();
    const auto & request_audio = *request.audio_input;
    const bool original_mono = config.channels == 2 && request_audio.channels == 1;
    if (request_audio.sample_rate != config.sample_rate ||
        (request_audio.channels != config.channels && !original_mono)) {
        throw std::runtime_error("RoFormer run() audio_input does not match prepared audio contract");
    }
    engine::debug::trace_log_scalar("mel_band_roformer.session.sample_rate", config.sample_rate);
    engine::debug::trace_log_scalar("mel_band_roformer.session.channels", config.channels);
    engine::debug::trace_log_scalar("mel_band_roformer.session.chunk_size", chunk_size_);
    engine::debug::trace_log_scalar("mel_band_roformer.session.step", step_);
    engine::debug::trace_log_scalar("mel_band_roformer.session.original_mono", original_mono);

    const auto prepare_start = Clock::now();
    const auto audio = original_mono
        ? runtime::AudioBuffer{
              request_audio.sample_rate,
              config.channels,
              engine::audio::duplicate_mono_to_interleaved_channels(request_audio.samples, config.channels)}
        : request_audio;
    auto planar = engine::audio::deinterleave_to_planar_channels(audio.samples, audio.channels);
    int64_t total_length = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    bool padded_borders = false;
    if (total_length > 2 * border_ && border_ > 0) {
        std::vector<float> padded(static_cast<size_t>(audio.channels * (total_length + 2 * border_)), 0.0f);
        const engine::audio::AudioChunkSpec border_spec{
            total_length + 2 * border_,
            total_length + 2 * border_,
            engine::audio::AudioChunkPadMode::Reflect,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        };
        const engine::audio::AudioChunkSpan border_span{
            0,
            0,
            total_length + 2 * border_,
            -border_,
            0,
        };
        engine::audio::copy_planar_chunk(
            padded,
            planar,
            audio.channels,
            total_length,
            border_span,
            border_spec);
        planar = std::move(padded);
        total_length += 2 * border_;
        padded_borders = true;
    }
    engine::debug::trace_log_scalar("mel_band_roformer.session.frames", total_length);
    engine::debug::trace_log_scalar("mel_band_roformer.session.padded_borders", padded_borders);
    engine::debug::timing_log_scalar("mel_band_roformer.session.audio_prepare_ms", engine::debug::elapsed_ms(prepare_start));

    result_work_.assign(static_cast<size_t>(audio.channels * total_length), 0.0f);
    counter_work_.assign(static_cast<size_t>(audio.channels * total_length), 0.0f);

    const engine::audio::AudioChunkSpec chunk_spec{
        chunk_size_,
        step_,
        engine::audio::AudioChunkPadMode::Reflect,
        engine::audio::AudioChunkTailAlignment::Start,
        chunk_size_ / 2 + 2,
    };
    const auto chunk_loop_start = Clock::now();
    for (const auto & chunk : engine::audio::plan_audio_chunks(total_length, chunk_spec)) {
        engine::audio::copy_planar_chunk(
            chunk_planar_work_,
            planar,
            audio.channels,
            total_length,
            chunk,
            chunk_spec);
        const auto & vocals_planar = runtime_->separate_chunk(chunk_planar_work_);
        std::vector<float> chunk_window = chunk_window_;
        if (chunk.output_start_sample == 0) {
            std::fill(chunk_window.begin(), chunk_window.begin() + fade_size_, 1.0f);
        } else if (chunk.output_start_sample + chunk_size_ >= total_length) {
            std::fill(chunk_window.end() - fade_size_, chunk_window.end(), 1.0f);
        }

        engine::audio::overlap_add_planar_chunk(
            result_work_,
            counter_work_,
            vocals_planar,
            audio.channels,
            total_length,
            chunk,
            chunk_window,
            engine::audio::AudioChunkCounterMode::PerLane);
    }
    engine::debug::timing_log_scalar("mel_band_roformer.session.chunk_loop_ms", engine::debug::elapsed_ms(chunk_loop_start));

    const auto assemble_start = Clock::now();
    vocals_planar_work_ = result_work_;
    engine::audio::normalize_overlap_added_planar(
        vocals_planar_work_,
        counter_work_,
        audio.channels,
        total_length,
        engine::audio::AudioChunkCounterMode::PerLane);

    int64_t final_frames = total_length;
    if (padded_borders) {
        std::vector<float> cropped(static_cast<size_t>(audio.channels * (total_length - 2 * border_)), 0.0f);
        final_frames = total_length - 2 * border_;
        for (int ch = 0; ch < audio.channels; ++ch) {
            const float * src = vocals_planar_work_.data() + static_cast<size_t>(ch * total_length + border_);
            float * dst = cropped.data() + static_cast<size_t>(ch * final_frames);
            std::copy(src, src + final_frames, dst);
        }
        vocals_planar_work_ = std::move(cropped);
    }

    runtime::AudioBuffer vocals_audio;
    vocals_audio.sample_rate = audio.sample_rate;
    vocals_audio.channels = audio.channels;
    vocals_audio.samples = engine::audio::interleave_planar_channels(vocals_planar_work_, audio.channels, final_frames);
    runtime::AudioBuffer mixture_audio = audio;
    if (static_cast<int64_t>(mixture_audio.samples.size()) != final_frames * audio.channels) {
        throw std::runtime_error("RoFormer output length does not match the original mixture length");
    }
    runtime::AudioBuffer instrumental_audio = derive_instrumental(mixture_audio, vocals_audio);

    if (original_mono) {
        vocals_audio.samples = engine::audio::extract_interleaved_channel(vocals_audio.samples, vocals_audio.channels, 0);
        vocals_audio.channels = 1;
        instrumental_audio.samples =
            engine::audio::extract_interleaved_channel(instrumental_audio.samples, instrumental_audio.channels, 0);
        instrumental_audio.channels = 1;
    }

    runtime::TaskResult result_task;
    result_task.named_audio_outputs.push_back({"vocals", std::move(vocals_audio), {}});
    result_task.named_audio_outputs.push_back({"instrumental", std::move(instrumental_audio), {{"derived", "mixture_minus_vocals"}}});
    engine::debug::timing_log_scalar("mel_band_roformer.session.assemble_ms", engine::debug::elapsed_ms(assemble_start));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start));
    return result_task;
}

}  // namespace engine::models::roformer
