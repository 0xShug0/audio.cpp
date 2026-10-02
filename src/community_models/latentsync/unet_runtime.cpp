#include "engine/community_models/latentsync/unet_runtime.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/sampling/diffusion_math.h"

#include <ggml-alloc.h>
#include <chrono>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace engine::models::latentsync {
namespace {

class LatentSyncUnetGraph {
public:
    LatentSyncUnetGraph(core::ExecutionContext & execution, const LatentSyncUnetConfig & config,
        const LatentSyncUnetWeights & weights, int64_t batch, int64_t frames,
        int64_t height, int64_t width, int64_t audio_tokens, bool cache,
        ggml_gallocr_t allocator, bool replay = false) : backend_(execution.backend()), allocator_(allocator) {
        constexpr size_t nodes = 16384;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("LatentSync U-Net graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "latentsync.unet", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({batch * frames, config.input_channels, height, width}));
        audio_ = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({batch * frames, audio_tokens, config.audio_channels}));
        timestep_ = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({batch, config.channels.front()}));
        ggml_set_input(input_.tensor);
        ggml_set_input(audio_.tensor);
        ggml_set_input(timestep_.tensor);
        if (replay) {
            cache_context_.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
            if (!cache_context_) throw std::runtime_error("LatentSync U-Net cache context allocation failed");
            core::ModuleBuildContext cache_ctx{cache_context_.get(), "latentsync.unet.cache", execution.backend_type()};
            cached_feature_ = core::make_tensor(cache_ctx,
                config.cuda_f16_activations ? GGML_TYPE_F16 : GGML_TYPE_F32,
                core::TensorShape::from_dims({batch * frames, config.channels.front(), height, width}));
            ggml_set_input(cached_feature_.tensor);
            cache_buffer_.reset(ggml_backend_alloc_ctx_tensors(cache_context_.get(), backend_));
            if (!cache_buffer_) throw std::runtime_error("LatentSync U-Net cache buffer allocation failed");
        }
        output_ = LatentSyncUnet(config).build(ctx, input_, audio_, timestep_, frames, weights,
            replay ? &cached_feature_ : nullptr, cache && !replay ? &cached_feature_ : nullptr);
        if (cache && !replay) ggml_set_output(cached_feature_.tensor);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("LatentSync U-Net graph allocation failed");
        }
        debug::timing_log_scalar("latentsync.unet.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(allocator_, 0) / 1048576.0);
    }

    ~LatentSyncUnetGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    void copy_cache_to(LatentSyncUnetGraph & replay) const {
        ggml_backend_tensor_copy(cached_feature_.tensor, replay.cached_feature_.tensor);
    }

    std::vector<std::byte> read_cache() const {
        return core::read_tensor_bytes(cached_feature_.tensor);
    }

    void write_cache(const std::vector<std::byte> & bytes) {
        core::write_tensor_bytes(cached_feature_, bytes);
    }

    std::vector<float> predict(const std::vector<float> & input, const std::vector<float> & audio,
        const std::vector<float> & timestep_features) {
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) {
            throw std::runtime_error("LatentSync U-Net graph allocation failed");
        }
        core::write_tensor_f32(input_, input);
        core::write_tensor_f32(audio_, audio);
        core::write_tensor_f32(timestep_, timestep_features);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LatentSync U-Net graph execution failed");
        }
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buffer_{nullptr, ggml_backend_buffer_free};
    ggml_gallocr_t allocator_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_, audio_, timestep_, output_, cached_feature_;
};

}  // namespace

struct LatentSyncUnetRuntime::Impl {
    core::ExecutionContext & execution;
    LatentSyncUnetConfig config;
    core::BackendWeightStore store;
    LatentSyncUnetWeights weights;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> shape;
    std::unique_ptr<LatentSyncUnetGraph> graph;
    std::unique_ptr<LatentSyncUnetGraph> replay_graph;
    std::array<std::vector<std::byte>, 2> cfg_cache;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        LatentSyncUnetConfig config_in, assets::TensorStorageType type)
        : execution(execution_in), config(config_in),
          store(execution.backend(), execution.backend_type(), "latentsync.unet.weights", 4 * 1024 * 1024),
          weights(load_unet_weights(store, source, config, type)),
          allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())),
              &ggml_gallocr_free) {
        if (config.cuda_f16_activations && execution.backend_type() != core::BackendType::Cuda) {
            throw std::runtime_error("LatentSync U-Net FP16 cuDNN execution requires CUDA");
        }
        if (config.cuda_f16_activations && type != assets::TensorStorageType::F16) {
            throw std::runtime_error("LatentSync U-Net FP16 execution requires FP16 weights");
        }
#ifndef GGML_CUDA_CUDNN_CONV2D
        if (config.cuda_f16_activations) {
            throw std::runtime_error("LatentSync U-Net FP16 cuDNN execution is not enabled in this build");
        }
#endif
        store.upload();
    }
};

LatentSyncUnetRuntime::LatentSyncUnetRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, LatentSyncUnetConfig config, assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(execution, source, config, weight_type)) {}

LatentSyncUnetRuntime::~LatentSyncUnetRuntime() = default;

std::vector<float> LatentSyncUnetRuntime::predict(const std::vector<float> & input,
    const std::vector<float> & audio, const std::vector<float> & timestep_features,
    int64_t batch, int64_t frames, int64_t height, int64_t width, int64_t audio_tokens,
    int64_t cache_step, int64_t cache_slot) {
    const auto start = std::chrono::steady_clock::now();
    if (batch <= 0 || frames <= 0 || height <= 0 || width <= 0 || audio_tokens <= 0) {
        throw std::runtime_error("LatentSync U-Net dimensions must be positive");
    }
    const auto shape = std::make_tuple(batch, frames, height, width, audio_tokens);
    const bool cache = cache_step >= 0;
    if (!impl_->graph || impl_->shape != shape || cache != bool(impl_->replay_graph)) {
        impl_->graph.reset();
        impl_->replay_graph.reset();
        impl_->graph = std::make_unique<LatentSyncUnetGraph>(impl_->execution, impl_->config,
            impl_->weights, batch, frames, height, width, audio_tokens, cache,
            impl_->allocator.get());
        if (cache) {
            impl_->replay_graph = std::make_unique<LatentSyncUnetGraph>(impl_->execution, impl_->config,
                impl_->weights, batch, frames, height, width, audio_tokens, true,
                impl_->allocator.get(), true);
        }
        impl_->shape = shape;
    }
    if (cache_step > 0 && impl_->config.sequential_cfg) {
        impl_->replay_graph->write_cache(impl_->cfg_cache[static_cast<size_t>(cache_slot)]);
    }
    auto & graph = cache_step > 0 ? impl_->replay_graph : impl_->graph;
    auto output = graph->predict(input, audio, timestep_features);
    if (cache_step == 0) {
        if (impl_->config.sequential_cfg) {
            impl_->cfg_cache[static_cast<size_t>(cache_slot)] = impl_->graph->read_cache();
        } else {
            impl_->graph->copy_cache_to(*impl_->replay_graph);
        }
    }
    debug::timing_log_scalar("latentsync.unet.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncUnetRuntime::denoise(const LatentSyncLatentClip & clip,
    int64_t steps, float guidance_scale, LatentSyncDdimConfig schedule_config, int64_t cache_interval) {
    const auto start = std::chrono::steady_clock::now();
    if (clip.frames <= 0 || clip.height <= 0 || clip.width <= 0 || clip.audio_tokens <= 0 ||
        !std::isfinite(guidance_scale) || guidance_scale < 0 || cache_interval < 1) {
        throw std::runtime_error("Invalid LatentSync denoising dimensions or guidance scale");
    }
    const size_t pixels = static_cast<size_t>(clip.height * clip.width);
    const size_t latent_frame = pixels * 4;
    const size_t latent_size = latent_frame * static_cast<size_t>(clip.frames);
    if (clip.noise.size() != latent_size || clip.masked_image.size() != latent_size ||
        clip.reference_image.size() != latent_size || clip.mask.size() != pixels * static_cast<size_t>(clip.frames) ||
        clip.audio.size() != static_cast<size_t>(clip.frames * clip.audio_tokens * impl_->config.audio_channels)) {
        throw std::runtime_error("LatentSync latent clip tensor size mismatch");
    }
    const bool cfg = guidance_scale > 1.0F;
    const bool sequential_cfg = cfg && impl_->config.sequential_cfg;
    const int64_t lanes = cfg && !sequential_cfg ? 2 : 1;
    const size_t input_frame = pixels * 13;
    std::vector<float> input(input_frame * static_cast<size_t>(clip.frames * lanes));
    std::vector<float> audio(clip.audio.size() * static_cast<size_t>(lanes), 0.0F);
    const std::vector<float> unconditioned_audio(sequential_cfg ? clip.audio.size() : 0, 0.0F);
    std::copy(clip.audio.begin(), clip.audio.end(), audio.end() - clip.audio.size());
    for (int64_t lane = 0; lane < lanes; ++lane) {
        for (int64_t frame = 0; frame < clip.frames; ++frame) {
            const size_t base = static_cast<size_t>(lane * clip.frames + frame) * input_frame;
            std::copy_n(clip.mask.begin() + frame * pixels, pixels, input.begin() + base + latent_frame);
            std::copy_n(clip.masked_image.begin() + frame * latent_frame, latent_frame, input.begin() + base + pixels * 5);
            std::copy_n(clip.reference_image.begin() + frame * latent_frame, latent_frame, input.begin() + base + pixels * 9);
        }
    }
    const LatentSyncDdimSchedule schedule(schedule_config, steps);
    auto latents = clip.noise;
    for (size_t i = 0; i < schedule.timesteps().size(); ++i) {
        for (int64_t lane = 0; lane < lanes; ++lane) {
            for (int64_t frame = 0; frame < clip.frames; ++frame) {
                const size_t base = static_cast<size_t>(lane * clip.frames + frame) * input_frame;
                std::copy_n(latents.begin() + frame * latent_frame, latent_frame, input.begin() + base);
            }
        }
        const auto time = latentsync_timestep_features(schedule.timesteps()[i], impl_->config.channels.front(), lanes);
        const int64_t cache_step = cache_interval == 1 ? -1 : static_cast<int64_t>(i) % cache_interval;
        std::vector<float> noise;
        if (sequential_cfg) {
            const auto unconditioned = predict(input, unconditioned_audio, time, 1,
                clip.frames, clip.height, clip.width, clip.audio_tokens, cache_step, 0);
            const auto conditioned = predict(input, audio, time, 1,
                clip.frames, clip.height, clip.width, clip.audio_tokens, cache_step, 1);
            noise = sampling::cfg_guidance(conditioned, unconditioned, guidance_scale);
        } else {
            noise = predict(input, audio, time, lanes, clip.frames, clip.height, clip.width,
                clip.audio_tokens, cache_step);
        }
        if (cfg && !sequential_cfg) {
            const std::vector<float> unconditioned(noise.begin(), noise.begin() + latent_size);
            const std::vector<float> conditioned(noise.begin() + latent_size, noise.end());
            noise = sampling::cfg_guidance(conditioned, unconditioned, guidance_scale);
        }
        schedule.step(latents, noise, i);
    }
    debug::timing_log_scalar("latentsync.denoise.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return latents;
}

}  // namespace engine::models::latentsync
