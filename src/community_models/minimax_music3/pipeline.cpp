#include "engine/community_models/minimax_music3/pipeline.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
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

// Mirrors the condition encoder's output-frame computation exactly so chunk
// latent sizes (and therefore flow noise RNG offsets) can be planned before
// the encoder runs.
int64_t predicted_condition_frames(const MiniMaxMusic3Config & config, int64_t input_frames) {
    return static_cast<int64_t>(
        static_cast<double>(input_frames) * static_cast<double>(config.condition.output_sample_rate) /
        static_cast<double>(config.condition.input_sample_rate) *
        static_cast<double>(config.condition.input_hop_length) /
        static_cast<double>(config.condition.output_hop_length));
}

struct DenoisedChunk {
    std::vector<float> latents;
    int64_t latent_frames = 0;
};

}  // namespace

void detail::append_cropped_interleaved_audio(
    runtime::AudioBuffer & destination,
    const runtime::AudioBuffer & chunk,
    int64_t left_frames,
    int64_t right_frames) {
    if (destination.channels != 2 || destination.sample_rate <= 0 ||
        chunk.channels != destination.channels || chunk.sample_rate != destination.sample_rate ||
        chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("MiniMax Music 3 chunk audio format mismatch");
    }
    const int64_t frames = static_cast<int64_t>(chunk.samples.size()) / chunk.channels;
    const int64_t start = std::min(std::max<int64_t>(0, left_frames), frames);
    const int64_t end = std::max(start, frames - std::max<int64_t>(0, right_frames));
    destination.samples.insert(
        destination.samples.end(),
        chunk.samples.begin() + static_cast<std::ptrdiff_t>(start * chunk.channels),
        chunk.samples.begin() + static_cast<std::ptrdiff_t>(end * chunk.channels));
}

struct MiniMaxMusic3PipelineRuntime::Impl {
    Impl(
        core::ExecutionContext & input_execution,
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type,
        bool memory_saver,
        bool pipeline_overlap)
        : execution(input_execution),
          assets(std::move(input_assets)),
          graph_arena_bytes(graph_arena_bytes),
          weight_context_bytes(weight_context_bytes),
          storage_type(storage_type),
          memory_saver(memory_saver),
          pipeline_overlap(pipeline_overlap),
          sampling_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(),
              execution.config().device,
              "minimax_music3.flow.sampling",
              "MiniMax Music 3 flow",
              sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 pipeline requires assets");
        }
        if (pipeline_overlap && memory_saver) {
            throw std::runtime_error(
                "MiniMax Music 3 pipeline_overlap requires mem_saver=false (all stages stay resident)");
        }
        if (pipeline_overlap) {
            // The AR stage runs a stream of short kernels that must not queue
            // behind the denoise stream's long GEMM waves, so its lazily
            // created streams get the highest CUDA priority. The env is read
            // by the CUDA backend at stream creation: AR streams materialize
            // during the AR weight upload below, denoise-context streams
            // afterwards at default priority.
            set_stream_priority_env("-5");
            ggml_backend_synchronize(execution.backend());
            set_stream_priority_env(nullptr);
            ar = make_ar();
            overlap_execution = std::make_unique<core::ExecutionContext>(execution.config());
            condition = make_condition();
            flow = make_flow();
            vocoder = make_vocoder();
        } else if (!memory_saver) {
            ar = make_ar();
            condition = make_condition();
            flow = make_flow();
            vocoder = make_vocoder();
        }
    }

    static void set_stream_priority_env(const char * value) {
#ifndef _WIN32
        if (value != nullptr) {
            setenv("GGML_CUDA_STREAM_PRIORITY", value, 1);
        } else {
            unsetenv("GGML_CUDA_STREAM_PRIORITY");
        }
#else
        (void) value;
#endif
    }

    ~Impl() {
        release_runtime_graphs();
    }

    core::ExecutionContext & denoise_execution() {
        return overlap_execution != nullptr ? *overlap_execution : execution;
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
        // Overlap pays off only while the flow tail can hide under AR slack;
        // on long requests the constant SM contention degrades AR far more
        // than the tail saves (measured +70% AR at 60 s), so it is applied
        // to short requests only.
        constexpr int64_t kOverlapMaxFrames = 600;
        if (pipeline_overlap && target_frames <= kOverlapMaxFrames) {
            return generate_overlapped(request, target_frames);
        }
        if (pipeline_overlap) {
            engine::debug::timing_log_scalar("minimax_music3.pipeline.overlap_skipped_long", 1.0);
        }
        return generate_sequential(request, target_frames);
    }

    std::vector<runtime::AudioBuffer> generate_ensemble(
        const MiniMaxMusic3Request & request,
        const std::vector<uint64_t> & take_seeds) {
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
        const int64_t takes = static_cast<int64_t>(take_seeds.size());
        if (takes <= 0) {
            throw std::runtime_error("MiniMax Music 3 ensemble requires at least one take seed");
        }
        const int64_t target_frames = std::min<int64_t>(
            assets->config.max_audio_frames,
            static_cast<int64_t>(request.duration_sec * static_cast<double>(assets->config.frame_rate)));
        std::vector<uint64_t> rng_offsets(static_cast<size_t>(takes), 0);
        std::vector<std::vector<float>> take_hiddens;
        {
            auto & ar_runtime = ensure_ar();
            take_hiddens = ar_runtime.generate_frame_hiddens_ensemble(
                request, target_frames, take_seeds, rng_offsets,
                request.ensemble_prefix_frames);
            release_ar_after_phase();
        }
        const int64_t hidden_frame_width =
            assets->config.condition.condition_layers * assets->config.qwen.hidden_size;
        std::vector<runtime::AudioBuffer> out;
        out.reserve(static_cast<size_t>(takes));
        for (int64_t take = 0; take < takes; ++take) {
            const auto & frame_hiddens = take_hiddens[static_cast<size_t>(take)];
            const int64_t generated_frames =
                static_cast<int64_t>(frame_hiddens.size()) / hidden_frame_width;
            if (static_cast<int64_t>(frame_hiddens.size()) != generated_frames * hidden_frame_width) {
                throw std::runtime_error("MiniMax Music 3 frame hidden shape mismatch");
            }
            if (generated_frames <= 0) {
                throw std::runtime_error("MiniMax Music 3 AR produced no frames");
            }
            MiniMaxMusic3Request take_request = request;
            // With an intro-lock prefix the flow noise is shared (master
            // seed): the locked intro renders identically for every take,
            // and post-fork divergence comes from the conditions themselves.
            take_request.seed = request.ensemble_prefix_frames > 0
                ? take_seeds[0]
                : take_seeds[static_cast<size_t>(take)];
            out.push_back(denoise_and_vocode(
                frame_hiddens, generated_frames, take_request, rng_offsets[static_cast<size_t>(take)]));
        }
        return out;
    }

    // The exact pre-existing sequential pipeline, byte-for-byte.
    runtime::AudioBuffer generate_sequential(const MiniMaxMusic3Request & request, int64_t target_frames) {
        uint64_t rng_offset_blocks = 0;
        std::vector<float> frame_hiddens;
        {
            auto & ar_runtime = ensure_ar();
            frame_hiddens = ar_runtime.generate_frame_hiddens(request, target_frames, rng_offset_blocks);
            release_ar_after_phase();
        }
        const int64_t hidden_frame_width =
            assets->config.condition.condition_layers * assets->config.qwen.hidden_size;
        const int64_t generated_frames =
            static_cast<int64_t>(frame_hiddens.size()) / hidden_frame_width;
        if (static_cast<int64_t>(frame_hiddens.size()) != generated_frames * hidden_frame_width) {
            throw std::runtime_error("MiniMax Music 3 frame hidden shape mismatch");
        }
        if (generated_frames <= 0) {
            throw std::runtime_error("MiniMax Music 3 AR produced no frames");
        }
        return denoise_and_vocode(frame_hiddens, generated_frames, request, rng_offset_blocks);
    }

    runtime::AudioBuffer denoise_and_vocode(
        const std::vector<float> & frame_hiddens,
        int64_t generated_frames,
        const MiniMaxMusic3Request & request,
        uint64_t rng_offset_blocks) {
        const int64_t hidden_frame_width =
            assets->config.condition.condition_layers * assets->config.qwen.hidden_size;
        std::vector<DenoisedChunk> denoised;
        const auto denoise_start = Clock::now();
        {
            const auto starts = chunk_starts(generated_frames, assets->config);
            denoised.reserve(starts.size());
            std::vector<float> previous_latent;
            std::vector<float> previous_condition;
            auto & condition_runtime = ensure_condition();
            auto & flow_runtime = ensure_flow();
            for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
                const int64_t start = starts[chunk_index];
                const int64_t end = std::min(start + assets->config.chunk_frames, generated_frames);
                const int64_t frame_count = end - start;
                int64_t condition_frames = 0;
                const auto condition_start = Clock::now();
                auto condition_values = condition_runtime.encode(
                    frame_hiddens.data() + static_cast<std::ptrdiff_t>(start * hidden_frame_width),
                    static_cast<size_t>(frame_count * hidden_frame_width),
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
                // Carry-free probe: with MM3_FLOW_NO_CARRY=1 every chunk
                // denoises independently (the crop overlap still smooths the
                // seams). If listening/panel accept this, chunks can be
                // denoised in parallel across GPUs.
                static const bool no_carry = [] {
                    const char * env = std::getenv("MM3_FLOW_NO_CARRY");
                    return env != nullptr && env[0] == '1';
                }();
                if (no_carry) {
                    previous_latent.clear();
                    previous_condition.clear();
                } else {
                    previous_latent = std::move(carry_latent);
                    previous_condition = std::move(carry_condition);
                }
                denoised.push_back({std::move(latents), condition_frames});
            }
            release_flow_after_phase();
        }

        runtime::AudioBuffer out;
        out.sample_rate = assets->config.vocoder.sample_rate;
        out.channels = 2;
        size_t output_samples = 0;
        for (size_t chunk_index = 0; chunk_index < denoised.size(); ++chunk_index) {
            const int64_t estimated_decoded_frames =
                denoised[chunk_index].latent_frames * assets->config.vocoder.hop_length;
            const int64_t left =
                chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
            const int64_t right =
                chunk_index + 1 == denoised.size()
                    ? 0
                    : kCropRightLatent * assets->config.vocoder.hop_length;
            const int64_t estimated_kept_frames =
                std::max<int64_t>(0, estimated_decoded_frames - left - right);
            output_samples += static_cast<size_t>(estimated_kept_frames * out.channels);
        }
        out.samples.reserve(output_samples);
        {
            auto & vocoder_runtime = ensure_vocoder();
            for (size_t chunk_index = 0; chunk_index < denoised.size(); ++chunk_index) {
                auto chunk = std::move(denoised[chunk_index]);
                const auto vocoder_start = Clock::now();
                const auto audio = vocoder_runtime.decode(chunk.latents, chunk.latent_frames);
                engine::debug::timing_log_scalar(
                    "minimax_music3.vocoder.total_ms",
                    engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                const int64_t left = chunk_index == 0 ? 0 : kCropLeftLatent * assets->config.vocoder.hop_length;
                const int64_t right =
                    chunk_index + 1 == denoised.size() ? 0 : kCropRightLatent * assets->config.vocoder.hop_length;
                detail::append_cropped_interleaved_audio(out, audio, left, right);
            }
            release_vocoder_after_phase();
        }
        engine::debug::timing_log_scalar(
            "minimax_music3.flow_vocoder.total_ms",
            engine::debug::elapsed_ms(denoise_start, Clock::now()));

        for (float & sample : out.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return out;
    }

    // Overlapped pipeline: AR keeps decoding on the primary context while a
    // worker thread runs condition+flow+vocoder for every chunk whose frames
    // are already available, on a second backend context (its own CUDA
    // stream). Flow noise offsets are planned from the deterministic per-frame
    // RNG consumption; if the AR stage ends early (EOS) or consumption differs
    // from the plan, the worker result is discarded and the exact sequential
    // stage runs instead, so output bytes never depend on the overlap.
    runtime::AudioBuffer generate_overlapped(const MiniMaxMusic3Request & request, int64_t target_frames) {
        const auto & config = assets->config;
        const int64_t hidden_frame_width = config.condition.condition_layers * config.qwen.hidden_size;

        struct ChunkPlan {
            int64_t start = 0;
            int64_t end = 0;
            uint64_t noise_offset_blocks = 0;
        };
        const uint64_t semantic_blocks = sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(config.qwen.vocab_size), sampling_policy);
        const uint64_t depth_blocks =
            sampling::torch_cuda_tensor_iterator_offset_blocks(
                static_cast<uint64_t>(config.depth.audio_vocab_size), sampling_policy) *
            static_cast<uint64_t>(config.depth.codebooks - 1);
        const uint64_t predicted_after_ar =
            static_cast<uint64_t>(target_frames + 1) * (semantic_blocks + depth_blocks);

        const auto starts = chunk_starts(target_frames, config);
        std::vector<ChunkPlan> plans(starts.size());
        {
            uint64_t running = predicted_after_ar;
            for (size_t chunk_index = 0; chunk_index < starts.size(); ++chunk_index) {
                auto & plan = plans[chunk_index];
                plan.start = starts[chunk_index];
                plan.end = std::min(plan.start + config.chunk_frames, target_frames);
                plan.noise_offset_blocks = running;
                const int64_t latent_frames = predicted_condition_frames(config, plan.end - plan.start);
                running += sampling::torch_cuda_tensor_iterator_offset_blocks(
                    static_cast<uint64_t>(config.flow.in_channels * latent_frames),
                    sampling_policy);
            }
        }

        std::vector<float> frame_hiddens;
        frame_hiddens.reserve(static_cast<size_t>(target_frames * hidden_frame_width));
        // The AR stage appends into this reserved buffer without reallocating,
        // so the base pointer is stable for the whole overlapped run; the
        // worker reads only rows already published through rows_done.
        const float * const hidden_rows = frame_hiddens.data();

        std::mutex progress_mutex;
        std::condition_variable progress_cv;
        int64_t rows_done = 0;
        bool ar_finished = false;

        std::atomic<bool> worker_valid{true};
        std::exception_ptr worker_error;
        runtime::AudioBuffer worker_audio;
        worker_audio.sample_rate = config.vocoder.sample_rate;
        worker_audio.channels = 2;

        const auto denoise_start = Clock::now();
        std::thread worker([&]() {
            try {
                std::vector<float> previous_latent;
                std::vector<float> previous_condition;
                auto & condition_runtime = ensure_condition();
                auto & flow_runtime = ensure_flow();
                auto & vocoder_runtime = ensure_vocoder();
                for (size_t chunk_index = 0; chunk_index < plans.size(); ++chunk_index) {
                    const auto & plan = plans[chunk_index];
                    {
                        std::unique_lock<std::mutex> lock(progress_mutex);
                        progress_cv.wait(lock, [&] { return rows_done >= plan.end || ar_finished; });
                        if (rows_done < plan.end) {
                            worker_valid.store(false, std::memory_order_release);
                            return;
                        }
                    }
                    const int64_t frame_count = plan.end - plan.start;
                    int64_t condition_frames = 0;
                    const auto condition_start = Clock::now();
                    auto condition_values = condition_runtime.encode(
                        hidden_rows + static_cast<std::ptrdiff_t>(plan.start * hidden_frame_width),
                        static_cast<size_t>(frame_count * hidden_frame_width),
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
                        plan.noise_offset_blocks,
                        sampling_policy,
                        carry_condition,
                        carry_latent);
                    engine::debug::timing_log_scalar(
                        "minimax_music3.flow.total_ms",
                        engine::debug::elapsed_ms(flow_start, Clock::now()));
                    previous_latent = std::move(carry_latent);
                    previous_condition = std::move(carry_condition);
                    const auto vocoder_start = Clock::now();
                    const auto audio = vocoder_runtime.decode(latents, condition_frames);
                    engine::debug::timing_log_scalar(
                        "minimax_music3.vocoder.total_ms",
                        engine::debug::elapsed_ms(vocoder_start, Clock::now()));
                    const int64_t left =
                        chunk_index == 0 ? 0 : kCropLeftLatent * config.vocoder.hop_length;
                    const int64_t right =
                        chunk_index + 1 == plans.size() ? 0 : kCropRightLatent * config.vocoder.hop_length;
                    detail::append_cropped_interleaved_audio(worker_audio, audio, left, right);
                }
            } catch (...) {
                worker_error = std::current_exception();
                worker_valid.store(false, std::memory_order_release);
            }
        });

        uint64_t rng_offset_blocks = 0;
        std::exception_ptr ar_error;
        try {
            const std::function<void(int64_t)> progress = [&](int64_t frame) {
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    rows_done = frame;
                }
                progress_cv.notify_one();
            };
            auto & ar_runtime = ensure_ar();
            ar_runtime.generate_frame_hiddens_into(
                request, target_frames, rng_offset_blocks, frame_hiddens, &progress);
        } catch (...) {
            ar_error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(progress_mutex);
            ar_finished = true;
        }
        progress_cv.notify_one();
        worker.join();
        if (ar_error != nullptr) {
            std::rethrow_exception(ar_error);
        }
        release_ar_after_phase();

        const int64_t generated_frames =
            static_cast<int64_t>(frame_hiddens.size()) / hidden_frame_width;
        if (static_cast<int64_t>(frame_hiddens.size()) != generated_frames * hidden_frame_width) {
            throw std::runtime_error("MiniMax Music 3 frame hidden shape mismatch");
        }
        if (generated_frames <= 0) {
            throw std::runtime_error("MiniMax Music 3 AR produced no frames");
        }

        const bool plan_held =
            worker_valid.load(std::memory_order_acquire) &&
            worker_error == nullptr &&
            generated_frames == target_frames &&
            rng_offset_blocks == predicted_after_ar;
        if (!plan_held) {
            if (worker_error != nullptr) {
                std::rethrow_exception(worker_error);
            }
            engine::debug::timing_log_scalar("minimax_music3.pipeline.overlap_fallback", 1.0);
            return denoise_and_vocode(frame_hiddens, generated_frames, request, rng_offset_blocks);
        }

        release_flow_after_phase();
        release_vocoder_after_phase();
        engine::debug::timing_log_scalar(
            "minimax_music3.flow_vocoder.total_ms",
            engine::debug::elapsed_ms(denoise_start, Clock::now()));
        for (float & sample : worker_audio.samples) {
            sample = std::clamp(sample, -1.0F, 1.0F);
        }
        return worker_audio;
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
            core::trim_backend_pools(execution.backend());
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
            core::trim_backend_pools(denoise_execution().backend());
        }
    }

    void release_vocoder_after_phase() {
        if (vocoder == nullptr) {
            return;
        }
        vocoder->release_runtime_graphs();
        if (memory_saver) {
            vocoder.reset();
            core::trim_backend_pools(denoise_execution().backend());
        }
    }

    std::unique_ptr<MiniMaxMusic3ArRuntime> make_ar() {
        return std::make_unique<MiniMaxMusic3ArRuntime>(
            assets,
            execution,
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            memory_saver);
    }

    std::unique_ptr<MiniMaxMusic3ConditionEncoderRuntime> make_condition() {
        return std::make_unique<MiniMaxMusic3ConditionEncoderRuntime>(
            assets,
            denoise_execution(),
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            memory_saver);
    }

    std::unique_ptr<MiniMaxMusic3FlowSamplerRuntime> make_flow() {
        return std::make_unique<MiniMaxMusic3FlowSamplerRuntime>(
            assets,
            denoise_execution(),
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            memory_saver);
    }

    std::unique_ptr<MiniMaxMusic3VocoderRuntime> make_vocoder() {
        return std::make_unique<MiniMaxMusic3VocoderRuntime>(
            assets,
            denoise_execution(),
            graph_arena_bytes,
            weight_context_bytes,
            storage_type,
            memory_saver);
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    size_t graph_arena_bytes = 0;
    size_t weight_context_bytes = 0;
    assets::TensorStorageType storage_type = assets::TensorStorageType::Native;
    bool memory_saver = true;
    bool pipeline_overlap = false;
    std::unique_ptr<core::ExecutionContext> overlap_execution;
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
    bool memory_saver,
    bool pipeline_overlap)
    : impl_(std::make_unique<Impl>(
          execution,
          std::move(assets),
          graph_arena_bytes,
          weight_context_bytes,
          storage_type,
          memory_saver,
          pipeline_overlap)) {}

MiniMaxMusic3PipelineRuntime::~MiniMaxMusic3PipelineRuntime() = default;

runtime::AudioBuffer MiniMaxMusic3PipelineRuntime::generate(const MiniMaxMusic3Request & request) {
    return impl_->generate(request);
}

std::vector<runtime::AudioBuffer> MiniMaxMusic3PipelineRuntime::generate_ensemble(
    const MiniMaxMusic3Request & request,
    const std::vector<uint64_t> & take_seeds) {
    return impl_->generate_ensemble(request, take_seeds);
}

void MiniMaxMusic3PipelineRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
