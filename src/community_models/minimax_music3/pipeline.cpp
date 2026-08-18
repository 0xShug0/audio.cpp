#include "engine/community_models/minimax_music3/pipeline.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::minimax_music3 {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kCropLeftLatent = 86;
constexpr int64_t kCropRightLatent = 344 - kCropLeftLatent;

std::vector<int64_t> chunk_starts(int64_t frames, const MiniMaxMusic3Config & config) {
    if (frames <= 0) {
        throw std::runtime_error("MiniMax Music 3 requires positive AR frame count");
    }
    if (frames <= config.chunk_frames) {
        return {0};
    }
    std::vector<int64_t> out;
    for (int64_t start = 0; start < frames - config.chunk_hop_frames; start += config.chunk_hop_frames) {
        out.push_back(start);
    }
    return out;
}

std::vector<float> condition_slice(
    const std::vector<float> & condition,
    int64_t start,
    int64_t frames,
    int64_t dim) {
    if (start < 0 || frames <= 0 || dim <= 0 ||
        static_cast<int64_t>(condition.size()) < (start + frames) * dim) {
        throw std::runtime_error("MiniMax Music 3 condition slice is out of range");
    }
    return std::vector<float>(
        condition.begin() + static_cast<std::ptrdiff_t>(start * dim),
        condition.begin() + static_cast<std::ptrdiff_t>((start + frames) * dim));
}

std::vector<float> crop_interleaved_audio(
    const runtime::AudioBuffer & audio,
    int64_t left_samples,
    int64_t right_samples) {
    if (audio.channels != 2 || audio.sample_rate <= 0 ||
        audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MiniMax Music 3 vocoder returned invalid stereo audio");
    }
    const int64_t frames = static_cast<int64_t>(audio.samples.size()) / audio.channels;
    const int64_t start = std::min(std::max<int64_t>(0, left_samples), frames);
    const int64_t end = std::max(start, frames - std::max<int64_t>(0, right_samples));
    std::vector<float> out(static_cast<size_t>((end - start) * audio.channels));
    std::copy(
        audio.samples.begin() + static_cast<std::ptrdiff_t>(start * audio.channels),
        audio.samples.begin() + static_cast<std::ptrdiff_t>(end * audio.channels),
        out.begin());
    return out;
}

struct DenoisedChunk {
    std::vector<float> latents;
    int64_t latent_frames = 0;
};

}  // namespace

struct MiniMaxMusic3PipelineRuntime::Impl {
    Impl(
        core::ExecutionContext & input_execution,
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool memory_saver)
        : execution(input_execution),
          assets(std::move(input_assets)),
          graph_arena_bytes(graph_arena_bytes),
          weight_context_bytes(weight_context_bytes),
          storage_type(storage_type),
          memory_saver(memory_saver),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "minimax_music3.flow.sampling",
              "MiniMax Music 3 flow",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 pipeline requires assets");
        }
        if (!memory_saver) {
            ar = make_ar();
            condition = make_condition();
            flow = make_flow();
            vocoder = make_vocoder();
        }
    }

    ~Impl() {
        release_runtime_graphs();
    }

    runtime::AudioBuffer generate(const MiniMaxMusic3Request & request) {
        if (request.duration_sec <= 0.0) {
            throw std::runtime_error("MiniMax Music 3 duration_sec must be positive");
        }
        if (request.num_inference_steps <= 0) {
            throw std::runtime_error("MiniMax Music 3 num_inference_steps must be positive");
        }
        if (request.guidance_scale <= 0.0F || request.ar_guidance_scale <= 0.0F) {
            throw std::runtime_error("MiniMax Music 3 guidance scales must be positive");
        }
        if (request.top_k <= 0) {
            throw std::runtime_error("MiniMax Music 3 top_k must be positive");
        }
        const int64_t target_frames = std::min<int64_t>(
            assets->config.max_audio_frames,
            static_cast<int64_t>(request.duration_sec * static_cast<double>(assets->config.frame_rate)));
        uint64_t rng_offset_blocks = 0;
        std::vector<float> frame_hiddens;
        {
            auto & ar_runtime = ensure_ar();
            frame_hiddens = ar_runtime.generate_frame_hiddens(request, target_frames, rng_offset_blocks);
            release_ar_after_phase();
        }
        const int64_t generated_frames =
            static_cast<int64_t>(frame_hiddens.size()) /
            (assets->config.condition.condition_layers * assets->config.qwen.hidden_size);
        if (static_cast<int64_t>(frame_hiddens.size()) !=
            generated_frames * assets->config.condition.condition_layers * assets->config.qwen.hidden_size) {
            throw std::runtime_error("MiniMax Music 3 frame hidden shape mismatch");
        }
        if (generated_frames <= 0) {
            throw std::runtime_error("MiniMax Music 3 AR produced no frames");
        }

        const auto denoise_start = Clock::now();
        const auto starts = chunk_starts(generated_frames, assets->config);
        std::vector<DenoisedChunk> denoised;
        denoised.reserve(starts.size());
        std::vector<float> previous_latent;
        std::vector<float> previous_condition;
        {
            auto & condition_runtime = ensure_condition();
            auto & flow_runtime = ensure_flow();
            for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
                const int64_t start = starts[chunk_index];
                const int64_t end = std::min(start + assets->config.chunk_frames, generated_frames);
                const int64_t frame_count = end - start;
                int64_t condition_frames = 0;
                const auto condition_start = Clock::now();
                auto condition_values = condition_runtime.encode(
                    condition_slice(
                        frame_hiddens,
                        start,
                        frame_count,
                        assets->config.condition.condition_layers * assets->config.qwen.hidden_size),
                    frame_count,
                    condition_frames);
                core::round_f32_to_bf16_in_place(condition_values);
                engine::debug::timing_log_scalar(
                    "minimax_music3.condition.total_ms",
                    engine::debug::elapsed_ms(condition_start, Clock::now()));
                std::vector<float> carry_condition;
                std::vector<float> carry_latent;
                const auto flow_start = Clock::now();
                auto latents = flow_runtime.denoise_chunk(
                    condition_values,
                    condition_frames,
                    previous_latent,
                    previous_condition,
                    request,
                    rng_offset_blocks,
                    sampling_policy,
                    carry_condition,
                    carry_latent);
                engine::debug::timing_log_scalar(
                    "minimax_music3.flow.total_ms",
                    engine::debug::elapsed_ms(flow_start, Clock::now()));
                rng_offset_blocks += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(latents.size()),
                    sampling_policy);
                previous_latent = std::move(carry_latent);
                previous_condition = std::move(carry_condition);
                denoised.push_back({std::move(latents), condition_frames});
            }
            release_flow_after_phase();
        }
        std::vector<runtime::AudioBuffer> chunks;
        chunks.reserve(denoised.size());
        {
            auto & vocoder_runtime = ensure_vocoder();
            for (size_t chunk_index = 0; chunk_index < denoised.size(); ++chunk_index) {
                const auto vocoder_start = Clock::now();
                auto audio = vocoder_runtime.decode(
                    denoised[chunk_index].latents,
                    denoised[chunk_index].latent_frames);
                engine::debug::timing_log_scalar(
                    "minimax_music3.vocoder.total_ms",
                    engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                const int64_t left = chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
                const int64_t right =
                    chunk_index + 1 == denoised.size() ? 0 : kCropRightLatent * assets->config.vocoder.hop_length;
                audio.samples = crop_interleaved_audio(audio, left, right);
                chunks.push_back(std::move(audio));
            }
            release_vocoder_after_phase();
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.flow_vocoder.total_ms",
            engine::debug::elapsed_ms(denoise_start, Clock::now()));

        runtime::AudioBuffer out;
        out.sample_rate = assets->config.vocoder.sample_rate;
        out.channels = 2;
        for (const auto & chunk : chunks) {
            if (chunk.sample_rate != out.sample_rate || chunk.channels != out.channels) {
                throw std::runtime_error("MiniMax Music 3 chunk audio format mismatch");
            }
            out.samples.insert(out.samples.end(), chunk.samples.begin(), chunk.samples.end());
        }
        for (float & sample : out.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return out;
    }

    void release_runtime_graphs() {
        if (ar != nullptr) {
            ar->release_runtime_graphs();
        }
        if (condition != nullptr) {
            condition->release_runtime_graphs();
        }
        if (flow != nullptr) {
            flow->release_runtime_graphs();
        }
        if (vocoder != nullptr) {
            vocoder->release_runtime_graphs();
        }
    }

    MiniMaxMusic3ArRuntime & ensure_ar() {
        if (ar == nullptr) {
            ar = make_ar();
        }
        return *ar;
    }

    MiniMaxMusic3ConditionEncoderRuntime & ensure_condition() {
        if (condition == nullptr) {
            condition = make_condition();
        }
        return *condition;
    }

    MiniMaxMusic3FlowSamplerRuntime & ensure_flow() {
        if (flow == nullptr) {
            flow = make_flow();
        }
        return *flow;
    }

    MiniMaxMusic3VocoderRuntime & ensure_vocoder() {
        if (vocoder == nullptr) {
            vocoder = make_vocoder();
        }
        return *vocoder;
    }

    void release_ar_after_phase() {
        if (ar == nullptr) {
            return;
        }
        ar->release_runtime_graphs();
        if (memory_saver) {
            ar.reset();
        }
    }

    void release_flow_after_phase() {
        if (condition != nullptr) {
            condition->release_runtime_graphs();
        }
        if (flow != nullptr) {
            flow->release_runtime_graphs();
        }
        if (memory_saver) {
            flow.reset();
            condition.reset();
        }
    }

    void release_vocoder_after_phase() {
        if (vocoder == nullptr) {
            return;
        }
        vocoder->release_runtime_graphs();
        if (memory_saver) {
            vocoder.reset();
        }
    }

    std::unique_ptr<MiniMaxMusic3ArRuntime> make_ar() {
        return std::make_unique<MiniMaxMusic3ArRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3ConditionEncoderRuntime> make_condition() {
        return std::make_unique<MiniMaxMusic3ConditionEncoderRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> make_flow() {
        return std::make_unique<MiniMaxMusic3FlowSamplerRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    std::unique_ptr<MiniMaxMusic3VocoderRuntime> make_vocoder() {
        return std::make_unique<MiniMaxMusic3VocoderRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type);
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    size_t graph_arena_bytes = 0;
    size_t weight_context_bytes = 0;
    assets::TensorStorageType storage_type = assets::TensorStorageType::Native;
    bool memory_saver = true;
    std::unique_ptr<MiniMaxMusic3ArRuntime> ar;
    std::unique_ptr<MiniMaxMusic3ConditionEncoderRuntime> condition;
    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> flow;
    std::unique_ptr<MiniMaxMusic3VocoderRuntime> vocoder;
    sampling::TorchCudaSamplingPolicy sampling_policy;
};

MiniMaxMusic3PipelineRuntime::MiniMaxMusic3PipelineRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type,
    bool memory_saver)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          memory_saver)) {}

MiniMaxMusic3PipelineRuntime::~MiniMaxMusic3PipelineRuntime() = default;

runtime::AudioBuffer MiniMaxMusic3PipelineRuntime::generate(const MiniMaxMusic3Request & request) {
    return impl_->generate(request);
}

void MiniMaxMusic3PipelineRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
