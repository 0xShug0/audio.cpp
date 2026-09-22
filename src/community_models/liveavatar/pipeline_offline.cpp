#include "pipeline_internal.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::liveavatar {

LiveAvatarVideoResult generate_liveavatar_offline(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    LiveAvatarGenerateShared & shared) {
    const auto & request = shared.request;
    const auto total_start = shared.total_start;
    const auto & text_context = shared.text_context;
    const auto & negative_context = shared.negative_context;
    auto & ref_latents = shared.ref_latents;
    const auto & audio_config = shared.audio_config;
    const auto & audio_buckets = shared.audio_buckets;
    const int64_t latent_target_frames = shared.latent_target_frames;
    auto * impl_ = &state;
    const auto * assets_ = assets.get();
    auto & execution_ = execution;


    impl_->release_vae_weights(execution_);

    engine::core::TensorShape cond_latents_shape = engine::core::TensorShape::from_dims({
        assets_->config.latent_channels,
        latent_target_frames,
        request.height / 8,
        request.width / 8,
    });
    if (cond_latents_shape.dims[1] <= 0) {
        throw std::runtime_error("LiveAvatar condition latent frame count is empty");
    }
    std::vector<float> cond_latents(static_cast<size_t>(cond_latents_shape.num_elements()), 0.0F);
    engine::debug::timing_log_scalar("liveavatar.vae_encode_motion_ms", 0.0);
    engine::debug::timing_log_scalar("liveavatar.vae_encode_condition_ms", 0.0);
    const engine::core::TensorShape latent_shape = cond_latents_shape;
    const auto sigmas = make_flow_sigmas(request.steps, request.shift);
    auto solver_sigmas = sigmas;
    if (!solver_sigmas.empty() && solver_sigmas.back() == 0.0F) {
        solver_sigmas.back() = 0.001F;
    }
    auto latents = make_initial_noise(latent_shape, request.seed);
    scale_values(latents, 1.0F / std::sqrt(1.0F + sigmas.front() * sigmas.front()));
    const auto audio_input = slice_audio_bucket_frames(
        audio_buckets.values,
        audio_buckets.layers,
        audio_buckets.dims,
        audio_buckets.frames,
        0,
        audio_config.batch_frames);
    std::vector<float> negative_audio(audio_input.size(), 0.0F);
    const int64_t denoiser_audio_frames = audio_config.batch_frames;
    const int64_t latent_audio_start_frame = 0;

    const auto condition_start = Clock::now();
    const auto cond_condition = impl_->prepare_denoiser_condition(
        execution_,
        LiveAvatarDenoiserConditionRunInput{&text_context, &audio_input},
        latent_shape.dims[1],
        denoiser_audio_frames,
        latent_audio_start_frame);
    std::optional<LiveAvatarDenoiserPreparedCondition> uncond_condition;
    if (request.guidance_scale > 1.0F) {
        uncond_condition = impl_->prepare_denoiser_condition(
            execution_,
            LiveAvatarDenoiserConditionRunInput{&negative_context, &negative_audio},
            latent_shape.dims[1],
            denoiser_audio_frames,
            latent_audio_start_frame);
    }
    engine::debug::timing_log_scalar("liveavatar.denoiser_condition_ms", engine::debug::elapsed_ms(condition_start));

    const auto denoise_start = Clock::now();
    double denoiser_graph_run_ms = 0.0;
    LiveAvatarDenoiserPreparedCondition cfg_condition;
    LiveAvatarDenoiserStaticCache * static_cache = nullptr;
    if (uncond_condition.has_value() && request.fused_cfg) {
        cfg_condition.projected_text_context = cond_condition.projected_text_context;
        cfg_condition.projected_text_context.insert(
            cfg_condition.projected_text_context.end(),
            uncond_condition->projected_text_context.begin(),
            uncond_condition->projected_text_context.end());
        cfg_condition.encoded_audio_local = cond_condition.encoded_audio_local;
        cfg_condition.encoded_audio_local.insert(
            cfg_condition.encoded_audio_local.end(),
            uncond_condition->encoded_audio_local.begin(),
            uncond_condition->encoded_audio_local.end());
        cfg_condition.encoded_audio_global = cond_condition.encoded_audio_global;
        cfg_condition.encoded_audio_global.insert(
            cfg_condition.encoded_audio_global.end(),
            uncond_condition->encoded_audio_global.begin(),
            uncond_condition->encoded_audio_global.end());
        static_cache = &impl_->prepare_denoiser_static_cache(
            execution_,
            cfg_condition,
            latent_shape,
            2,
            request.sage_attention);
    }
    auto run_model_x0 = [&](const std::vector<float> & sample, float sigma) {
        auto denoiser_sample = sample;
        scale_values(denoiser_sample, std::sqrt(1.0F + sigma * sigma));
        LiveAvatarDenoiserRunInput cond_input;
        cond_input.latent = &denoiser_sample;
        cond_input.ref_latents = &ref_latents;
        cond_input.cond_latents = &cond_latents;
        cond_input.projected_text_context = &cond_condition.projected_text_context;
        cond_input.encoded_audio_local = &cond_condition.encoded_audio_local;
        cond_input.encoded_audio_global = &cond_condition.encoded_audio_global;
        cond_input.timestep = sigma * 1000.0F;
        const auto cond_denoise_run_start = Clock::now();
        std::vector<float> model_output;
        if (uncond_condition.has_value()) {
            if (request.fused_cfg) {
                LiveAvatarDenoiserRunInput cfg_input = cond_input;
                cfg_input.projected_text_context = &cfg_condition.projected_text_context;
                cfg_input.encoded_audio_local = &cfg_condition.encoded_audio_local;
                cfg_input.encoded_audio_global = &cfg_condition.encoded_audio_global;
                cfg_input.guidance_scale = request.guidance_scale;
                if (request.denoiser_layerwise) {
                    model_output = impl_->denoise_layerwise(
                        execution_,
                        cfg_input,
                        latent_shape,
                        request.sage_attention,
                        *static_cache,
                        request.denoiser_layerwise_batch);
                } else {
                    model_output = impl_->denoise(execution_, cfg_input, latent_shape, true, request.sage_attention, static_cache);
                }
                denoiser_graph_run_ms += engine::debug::elapsed_ms(cond_denoise_run_start);
            } else {
                auto cond_output = impl_->denoise(execution_, cond_input, latent_shape, false, request.sage_attention);
                denoiser_graph_run_ms += engine::debug::elapsed_ms(cond_denoise_run_start);
                LiveAvatarDenoiserRunInput uncond_input = cond_input;
                uncond_input.projected_text_context = &uncond_condition->projected_text_context;
                uncond_input.encoded_audio_local = &uncond_condition->encoded_audio_local;
                uncond_input.encoded_audio_global = &uncond_condition->encoded_audio_global;
                const auto uncond_denoise_run_start = Clock::now();
                auto noise_pred_uncond = impl_->denoise(execution_, uncond_input, latent_shape, false, request.sage_attention);
                denoiser_graph_run_ms += engine::debug::elapsed_ms(uncond_denoise_run_start);
                model_output = combine_cfg(cond_output, noise_pred_uncond, request.guidance_scale);
            }
        } else {
            model_output = impl_->denoise(execution_, cond_input, latent_shape, false, request.sage_attention);
            denoiser_graph_run_ms += engine::debug::elapsed_ms(cond_denoise_run_start);
        }
        for (size_t i = 0; i < model_output.size(); ++i) {
            model_output[i] = denoiser_sample[i] - sigma * model_output[i];
        }
        return model_output;
    };

    std::vector<float> last_sample;
    std::vector<float> model_prev;
    for (int64_t step = 0; step < request.steps; ++step) {
        const float sigma = sigmas[static_cast<size_t>(step)];
        if (step == 0) {
            model_prev = run_model_x0(latents, sigma);
        } else {
            auto model_current = run_model_x0(latents, sigma);
            latents = unipc_bh2_order1_step(
                last_sample,
                model_prev,
                &model_current,
                sigmas[static_cast<size_t>(step - 1)],
                sigma);
            model_prev = std::move(model_current);
        }
        last_sample = latents;
        latents = unipc_bh2_order1_step(
            last_sample,
            model_prev,
            nullptr,
            sigma,
            solver_sigmas[static_cast<size_t>(step + 1)]);
    }
    scale_values(latents, 1.0F / unipc_alpha(solver_sigmas.back()));
    engine::debug::timing_log_scalar("liveavatar.denoiser_graph_run_ms", denoiser_graph_run_ms);
    engine::debug::timing_log_scalar("liveavatar.denoise_ms", engine::debug::elapsed_ms(denoise_start));
    impl_->release_denoiser(execution_);

    engine::core::TensorShape decoded_shape;
    const auto decode_start = Clock::now();
    auto decode_latents = latents;
    apply_wan21_latent_process_out(decode_latents, latent_shape);
    const auto decoded = impl_->vae_decode(
        execution_,
        decode_latents,
        latent_shape.dims[0],
        latent_shape.dims[1],
        latent_shape.dims[2],
        latent_shape.dims[3],
        request.vae_decoder_tile_size,
        &decoded_shape);
    engine::debug::timing_log_scalar("liveavatar.vae_decode_ms", engine::debug::elapsed_ms(decode_start));
    int64_t output_frames = std::min<int64_t>(request.frames_per_clip, decoded_shape.dims[1]);
    int64_t start_frame = decoded_shape.dims[1] - output_frames;
    if (output_frames <= 0) {
        throw std::runtime_error("LiveAvatar decoded output is empty");
    }

    LiveAvatarVideoResult result;
    result.width = decoded_shape.dims[3];
    result.height = decoded_shape.dims[2];
    result.frames = output_frames;
    result.fps = assets_->config.fps;
    result.rgb24 = video_to_rgb24(
        decoded,
        decoded_shape.dims[0],
        decoded_shape.dims[1],
        decoded_shape.dims[2],
        decoded_shape.dims[3],
        start_frame,
        output_frames);
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start));
    return result;
}

}  // namespace engine::community_models::liveavatar
