#include "flow_impl.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts::detail {

DotsLatentMatrix decode_next_meanflow(DotFlowSharedRuntime & runtime, const DotsFlowDecodeRequest & request, FlowDecodeCacheState * decode_state) {
    const auto & config = runtime.weights->config;
    if (!config.meanflow.has_value() || !config.meanflow->enabled) {
        throw std::runtime_error("DotTTS MeanFlow decode requested for a non-MeanFlow checkpoint");
    }
    const int64_t hidden_size = config.dit.hidden_size;
    const int64_t latent_patch_size = config.patch_size;
    const int64_t latent_dim = config.latent_dim;
    const int64_t total_len = request.fm_seq_len + latent_patch_size;
    if (request.sequence == nullptr || request.speaker_condition == nullptr) {
        throw std::runtime_error("DotTTS MeanFlow decode request is missing borrowed inputs");
    }
    const auto & sequence = *request.sequence;
    const auto & speaker_condition = *request.speaker_condition;
    if (request.fm_seq_len <= 0 || request.num_inference_steps <= 0 ||
        static_cast<int64_t>(sequence.size()) < request.fm_seq_len * hidden_size ||
        static_cast<int64_t>(speaker_condition.size()) != hidden_size) {
        throw std::runtime_error("DotTTS MeanFlow decode request shape mismatch");
    }
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        runtime.weights->execution_context->backend_type(),
        runtime.weights->execution_context->config().device,
        "dots_tts.flow.rng",
        "DotTTS",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    std::vector<float> z = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
        static_cast<size_t>(latent_patch_size * latent_dim),
        request.seed,
        request.rng_offset_blocks,
        rng_policy,
        engine::sampling::TorchRandnPrecision::BFloat16);
    std::vector<int32_t> positions;
    std::vector<uint8_t> attention_mask;
    const float dt = 1.0F / static_cast<float>(request.num_inference_steps);
    const bool use_duration_embedding = config.meanflow->use_duration_embedding;
    std::vector<float> modulation_timesteps;
    std::vector<float> modulation_durations;
    modulation_timesteps.reserve(static_cast<size_t>(request.num_inference_steps));
    if (use_duration_embedding) {
        modulation_durations.reserve(static_cast<size_t>(request.num_inference_steps));
    }
    for (int64_t step = 0; step < request.num_inference_steps; ++step) {
        modulation_timesteps.push_back(static_cast<float>(step) * dt);
        if (use_duration_embedding) {
            modulation_durations.push_back(dt);
        }
    }
    const auto modulations = run_modulation(runtime, modulation_timesteps, modulation_durations, speaker_condition, request.runtime_stats);
    if (!decode_state || !decode_state->initialized) runtime.dit->reset_prefix_cache();
    if (decode_state) decode_state->initialized = true;
    const int64_t hidden_patch_size = 1;
    const int64_t unit_len = hidden_patch_size + latent_patch_size;
    const int64_t prefix_len = request.fm_seq_len - hidden_patch_size;
    if (prefix_len < 0 || prefix_len % unit_len != 0) {
        throw std::runtime_error("DotTTS MeanFlow DiT expects unit-aligned finalized history");
    }
    const bool use_cached_dit = request.use_cached_dit && decode_state != nullptr && prefix_len >= unit_len;
    int64_t persistent_len = 0;
    if (use_cached_dit) {
        if (request.cache_capacity < prefix_len) {
            throw std::runtime_error("DotTTS MeanFlow DiT cache capacity is smaller than the current prefix");
        }
        const int64_t capacity = resolve_dit_cache_capacity_tokens(request.fm_seq_len, unit_len);
        runtime.dit->reserve_prefix_cache({capacity, request.num_inference_steps, 1, 1,
            unit_len, latent_patch_size, GGML_TYPE_F32});
        persistent_len = prefix_len - unit_len;
        if (persistent_len < 0) {
            throw std::runtime_error("DotTTS MeanFlow DiT persistent prefix is invalid");
        }
        std::vector<float> persistent_sequence;
        if (persistent_len > 0) {
            persistent_sequence.assign(
                sequence.begin(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size));
        }
        for (int64_t step = 0; step < request.num_inference_steps; ++step) {
            if (runtime.dit->prefix_tokens(step) != persistent_len) {
                modules::DitOperationStats stats;
                runtime.dit->prefill_prefix(persistent_sequence, persistent_len, step, &stats);
                if (request.runtime_stats) {
                    request.runtime_stats->velocity_graph_ms += stats.prepare_ms;
                    request.runtime_stats->velocity_input_upload_ms += stats.upload_ms;
                    request.runtime_stats->velocity_compute_ms += stats.compute_ms;
                }
            }
        }
    } else {
        positions = build_decode_positions(total_len, request.fm_seq_len, latent_patch_size);
        attention_mask = build_decode_mask(1, total_len, request.fm_seq_len, latent_patch_size, 1);
    }
    modules::FlowSamplerRuntimeConfig sampler_config;
    sampler_config.label = "dots_tts.meanflow.sampling";
    sampler_config.latent_shape = {latent_patch_size, latent_dim};
    sampler_config.branches = {{"velocity", 1}};
    sampler_config.step_size = dt;
    sampler_config.state_precision = modules::FlowSamplerStatePrecision::Bf16;
    for (int64_t step = 0; step < request.num_inference_steps; ++step) {
        const float t = static_cast<float>(step) * dt;
        sampler_config.schedule.push_back({step, t, t + dt, t, t + dt});
    }
    modules::FlowSamplerSingleBranchDenoiserConfig denoiser;
    denoiser.label = sampler_config.label; denoiser.branch_name = "velocity";
    denoiser.sampler_mode = [](const modules::FlowSamplerStepState &) { return "flow"; };
    denoiser.predict = [&](const modules::FlowSamplerDenoiserInput & input) {
        const auto step = input.state.schedule.index;
        if (use_cached_dit) {
            auto z_projected = run_coordinate_projection(runtime, input.latent, latent_patch_size);
            engine::core::round_f32_to_bf16_in_place(z_projected.values);
            std::vector<float> tail_sequence;
            tail_sequence.reserve(static_cast<size_t>(2 * unit_len * hidden_size));
            tail_sequence.insert(
                tail_sequence.end(),
                sequence.begin() + static_cast<std::ptrdiff_t>(persistent_len * hidden_size),
                sequence.begin() + static_cast<std::ptrdiff_t>(request.fm_seq_len * hidden_size));
            tail_sequence.insert(tail_sequence.end(), z_projected.values.begin(), z_projected.values.end());
            modules::DitOperationStats stats;
            DotsVelocityOutput velocity{runtime.dit->predict_prefix(tail_sequence, persistent_len,
                step, 1, &stats), latent_patch_size, latent_dim};
            if (request.runtime_stats) {
                ++request.runtime_stats->velocity_calls;
                request.runtime_stats->velocity_graph_ms += stats.prepare_ms;
                request.runtime_stats->velocity_input_upload_ms += stats.upload_ms;
                request.runtime_stats->velocity_compute_ms += stats.compute_ms;
                request.runtime_stats->velocity_output_read_ms += stats.read_ms;
            }
            if (velocity.frames != latent_patch_size ||
                static_cast<int64_t>(velocity.values.size()) != latent_patch_size * latent_dim) {
                throw std::runtime_error("DotTTS MeanFlow cached velocity shape mismatch");
            }
            engine::core::round_f32_to_bf16_in_place(velocity.values);
            return velocity.values;
        } else {
            auto z_projected = run_coordinate_projection(runtime, input.latent, latent_patch_size);
            engine::core::round_f32_to_bf16_in_place(z_projected.values);
            std::vector<float> branch_sequence;
            branch_sequence.reserve(static_cast<size_t>(total_len * hidden_size));
            append_branch_sequence(branch_sequence, sequence, z_projected.values, request.fm_seq_len, hidden_size, latent_patch_size);
            auto velocity = run_velocity(runtime, 
                branch_sequence,
                total_len,
                {},
                {},
                {},
                1,
                positions,
                attention_mask,
                &modulations,
                step,
                total_len - latent_patch_size,
                latent_patch_size,
                request.runtime_stats);
            if (velocity.frames != latent_patch_size ||
                static_cast<int64_t>(velocity.values.size()) != latent_patch_size * latent_dim) {
                throw std::runtime_error("DotTTS MeanFlow velocity shape mismatch");
            }
            engine::core::round_f32_to_bf16_in_place(velocity.values);
            return velocity.values;
        }
    };
    modules::FlowSamplerRuntime sampler(std::move(sampler_config),
        modules::make_flow_sampler_single_branch_denoiser(std::move(denoiser)));
    sampler.run_sequence(z);
    DotsLatentMatrix out;
    out.frames = latent_patch_size;
    out.dims = latent_dim;
    out.values = sampler.latent();
    return out;
}

}  // namespace engine::models::dots_tts::detail
