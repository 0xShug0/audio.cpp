#include "engine/community_models/minimax_music3/flow_sampler.h"

#include "engine/framework/core/module.h"
#include "engine/framework/modules/flow_sampler_runtime.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::minimax_music3 {
namespace {

void copy_condition_prefix(
    std::vector<float> & condition,
    const std::vector<float> & previous_condition,
    int64_t overlap,
    int64_t dim) {
    if (overlap <= 0) {
        return;
    }
    if (static_cast<int64_t>(condition.size()) < overlap * dim ||
        static_cast<int64_t>(previous_condition.size()) < overlap * dim) {
        throw std::runtime_error("MiniMax Music 3 overlap condition shape mismatch");
    }
    std::copy(
        previous_condition.begin(),
        previous_condition.begin() + static_cast<std::ptrdiff_t>(overlap * dim),
        condition.begin());
}

void copy_latent_prefix(
    std::vector<float> & latent,
    const std::vector<float> & previous,
    int64_t overlap,
    int64_t frames,
    int64_t channels) {
    if (overlap <= 0) {
        return;
    }
    if (static_cast<int64_t>(latent.size()) != channels * frames ||
        static_cast<int64_t>(previous.size()) < channels * overlap) {
        throw std::runtime_error("MiniMax Music 3 overlap latent shape mismatch");
    }
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            previous.begin() + static_cast<std::ptrdiff_t>(channel * overlap),
            previous.begin() + static_cast<std::ptrdiff_t>((channel + 1) * overlap),
            latent.begin() + static_cast<std::ptrdiff_t>(channel * frames));
    }
}

std::vector<float> latent_tail_window(
    const std::vector<float> & latent,
    int64_t frames,
    int64_t channels,
    int64_t start,
    int64_t end) {
    if (start < 0 || end < start || end > frames ||
        static_cast<int64_t>(latent.size()) != channels * frames) {
        throw std::runtime_error("MiniMax Music 3 latent carry range is invalid");
    }
    std::vector<float> out(static_cast<size_t>(channels * (end - start)));
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            latent.begin() + static_cast<std::ptrdiff_t>(channel * frames + start),
            latent.begin() + static_cast<std::ptrdiff_t>(channel * frames + end),
            out.begin() + static_cast<std::ptrdiff_t>(channel * (end - start)));
    }
    return out;
}

std::vector<float> condition_tail_window(
    const std::vector<float> & condition,
    int64_t frames,
    int64_t dim,
    int64_t start,
    int64_t end) {
    if (start < 0 || end < start || end > frames ||
        static_cast<int64_t>(condition.size()) != frames * dim) {
        throw std::runtime_error("MiniMax Music 3 condition carry range is invalid");
    }
    return std::vector<float>(
        condition.begin() + static_cast<std::ptrdiff_t>(start * dim),
        condition.begin() + static_cast<std::ptrdiff_t>(end * dim));
}

std::vector<float> make_sigmas(int64_t steps) {
    if (steps <= 0) {
        throw std::runtime_error("MiniMax Music 3 num_inference_steps must be positive");
    }
    std::vector<float> sigmas(static_cast<size_t>(steps));
    if (steps == 1) {
        sigmas.front() = 0.0F;
        return sigmas;
    }
    const float end = 1.0F / static_cast<float>(steps);
    for (int64_t i = 0; i < steps; ++i) {
        const float raw_sigma =
            1.0F + (end - 1.0F) * static_cast<float>(i) / static_cast<float>(steps - 1);
        sigmas[static_cast<size_t>(i)] = 1.0F - raw_sigma;
    }
    return sigmas;
}

std::vector<modules::FlowSamplerScheduleStep> make_flow_schedule(int64_t steps) {
    const auto sigmas = make_sigmas(steps);
    std::vector<modules::FlowSamplerScheduleStep> schedule;
    schedule.reserve(static_cast<size_t>(steps));
    for (int64_t step = 0; step < steps; ++step) {
        const float t = sigmas[static_cast<size_t>(step)];
        const float next_t = step + 1 < steps ? sigmas[static_cast<size_t>(step + 1)] : 1.0F;
        schedule.push_back({step, t, next_t, t, next_t});
    }
    return schedule;
}

class MiniMaxMusic3FlowDenoiserRuntime final : public modules::FlowSamplerDenoiserRuntime {
public:
    explicit MiniMaxMusic3FlowDenoiserRuntime(MiniMaxMusic3FlowTransformerRuntime & flow)
        : flow_(flow) {}

    void set_chunk_inputs(
        const std::vector<float> & condition,
        int64_t frames,
        int64_t channels,
        int64_t condition_dim,
        const std::vector<float> & previous_latent,
        std::vector<float> noise_prompt,
        int64_t overlap) {
        if (frames <= 0 || channels <= 0 || condition_dim <= 0) {
            throw std::runtime_error("MiniMax Music 3 flow sampler received invalid chunk shape");
        }
        if (static_cast<int64_t>(condition.size()) != frames * condition_dim) {
            throw std::runtime_error("MiniMax Music 3 flow sampler condition shape mismatch");
        }
        if (overlap < 0 || overlap > frames ||
            (!previous_latent.empty() && static_cast<int64_t>(previous_latent.size()) < overlap * channels)) {
            throw std::runtime_error("MiniMax Music 3 flow sampler overlap shape mismatch");
        }
        condition_ = &condition;
        previous_latent_ = &previous_latent;
        noise_prompt_ = std::move(noise_prompt);
        frames_ = frames;
        channels_ = channels;
        overlap_ = overlap;
        flow_.prepare_chunk_condition(condition, frames);
    }

    void reset_sampler_caches(const std::vector<modules::FlowSamplerCacheState> & caches) override {
        if (!caches.empty()) {
            throw std::runtime_error("MiniMax Music 3 flow sampler does not use cache state");
        }
    }

    std::vector<modules::FlowSamplerCacheUpdate> begin_sampler_sequence(
        const modules::FlowSamplerSequenceState & state) override {
        if (!state.caches.empty()) {
            throw std::runtime_error("MiniMax Music 3 flow sampler does not use cache state");
        }
        if (condition_ == nullptr) {
            throw std::runtime_error("MiniMax Music 3 flow sampler requires chunk inputs");
        }
        active_schedule_steps_ = static_cast<int64_t>(state.schedule.size());
        return {};
    }

    modules::FlowSamplerGraphKey sampler_graph_key(const modules::FlowSamplerStepState & state) override {
        modules::FlowSamplerGraphKey key;
        key.latent_shape = {channels_ * frames_};
        key.branch_count = static_cast<int64_t>(state.branches.size());
        key.schedule_steps = active_schedule_steps_;
        key.sampler_mode = "minimax_music3.flow.cfg";
        return key;
    }

    void rebuild_sampler_graph(
        const modules::FlowSamplerGraphKey &,
        const modules::FlowSamplerStepState &) override {}

    modules::FlowSamplerDenoiserOutput run_sampler_denoiser(
        const modules::FlowSamplerDenoiserInput & input) override {
        if (condition_ == nullptr) {
            throw std::runtime_error("MiniMax Music 3 flow sampler chunk inputs are not set");
        }
        auto denoiser_latent = input.latent;
        if (overlap_ > 0) {
            apply_overlap_prompt(denoiser_latent, input.state.schedule.t);
        }
        const auto branches = flow_.predict_velocity_branches(
            denoiser_latent,
            *condition_,
            frames_,
            input.state.schedule.t);
        const size_t branch_size = static_cast<size_t>(channels_ * frames_);
        if (branches.size() != 2 * branch_size) {
            throw std::runtime_error("MiniMax Music 3 flow velocity shape mismatch");
        }
        modules::FlowSamplerDenoiserOutput output;
        output.predictions.push_back({
            "cond",
            std::vector<float>(branches.begin(), branches.begin() + static_cast<std::ptrdiff_t>(branch_size)),
        });
        output.predictions.push_back({
            "uncond",
            std::vector<float>(branches.begin() + static_cast<std::ptrdiff_t>(branch_size), branches.end()),
        });
        return output;
    }

    void release_sampler_graphs() override {
        flow_.release_runtime_graphs();
    }

private:
    void apply_overlap_prompt(std::vector<float> & latent, float t) const {
        if (previous_latent_ == nullptr || previous_latent_->empty()) {
            return;
        }
        for (int64_t channel = 0; channel < channels_; ++channel) {
            for (int64_t frame = 0; frame < overlap_; ++frame) {
                const size_t index = static_cast<size_t>(channel * frames_ + frame);
                const size_t prompt_index = static_cast<size_t>(channel * overlap_ + frame);
                const float noise = noise_prompt_[prompt_index];
                const float previous = (*previous_latent_)[prompt_index];
                latent[index] = (1.0F - (1.0F - 1.0e-6F) * t) * noise + t * previous;
            }
        }
    }

    MiniMaxMusic3FlowTransformerRuntime & flow_;
    const std::vector<float> * condition_ = nullptr;
    const std::vector<float> * previous_latent_ = nullptr;
    std::vector<float> noise_prompt_;
    int64_t frames_ = 0;
    int64_t channels_ = 0;
    int64_t overlap_ = 0;
    int64_t active_schedule_steps_ = 0;
};

class MiniMaxMusic3FlowUpdateRuntime final : public modules::FlowSamplerUpdateRuntime {
public:
    void set_overlap_inputs(
        const std::vector<float> & previous_latent,
        std::vector<float> noise_prompt,
        int64_t overlap,
        int64_t frames,
        int64_t channels) {
        previous_latent_ = &previous_latent;
        noise_prompt_ = std::move(noise_prompt);
        overlap_ = overlap;
        frames_ = frames;
        channels_ = channels;
    }

    void update_latent(const modules::FlowSamplerUpdateInput & input) override {
        if (input.prediction.size() != input.latent.size()) {
            throw std::runtime_error("MiniMax Music 3 flow update shape mismatch");
        }
        if (overlap_ > 0) {
            apply_overlap_prompt(input.latent, input.state.schedule.t);
        }
        const float dt = input.state.schedule.t_next - input.state.schedule.t;
        for (size_t i = 0; i < input.latent.size(); ++i) {
            input.latent[i] += dt * input.prediction[i];
        }
        core::round_f32_to_bf16_in_place(input.latent);
    }

private:
    void apply_overlap_prompt(std::vector<float> & latent, float t) const {
        if (previous_latent_ == nullptr || previous_latent_->empty()) {
            return;
        }
        for (int64_t channel = 0; channel < channels_; ++channel) {
            for (int64_t frame = 0; frame < overlap_; ++frame) {
                const size_t index = static_cast<size_t>(channel * frames_ + frame);
                const size_t prompt_index = static_cast<size_t>(channel * overlap_ + frame);
                const float noise = noise_prompt_[prompt_index];
                const float previous = (*previous_latent_)[prompt_index];
                latent[index] = (1.0F - (1.0F - 1.0e-6F) * t) * noise + t * previous;
            }
        }
    }

    const std::vector<float> * previous_latent_ = nullptr;
    std::vector<float> noise_prompt_;
    int64_t overlap_ = 0;
    int64_t frames_ = 0;
    int64_t channels_ = 0;
};

}  // namespace

struct MiniMaxMusic3FlowSamplerRuntime::Impl {
    Impl(
        std::shared_ptr<const MiniMaxMusic3Assets> input_assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(input_assets)),
          flow(std::make_unique<MiniMaxMusic3FlowTransformerRuntime>(
              assets,
              execution,
              graph_arena_bytes,
              weight_context_bytes,
              storage_type)) {
        if (assets == nullptr) {
            throw std::runtime_error("MiniMax Music 3 flow sampler requires assets");
        }
    }

    void ensure_sampler(int64_t latent_values, int64_t steps, float guidance_scale) {
        if (sampler != nullptr &&
            denoiser != nullptr &&
            updater != nullptr &&
            sampler_latent_values == latent_values &&
            sampler_steps == steps &&
            sampler_guidance_scale == guidance_scale) {
            return;
        }
        auto new_denoiser = std::make_unique<MiniMaxMusic3FlowDenoiserRuntime>(*flow);
        denoiser = new_denoiser.get();
        auto new_updater = std::make_unique<MiniMaxMusic3FlowUpdateRuntime>();
        updater = new_updater.get();

        modules::FlowSamplerRuntimeConfig config;
        config.label = "MiniMax Music 3 flow sampler";
        config.latent_shape = {latent_values};
        config.schedule = make_flow_schedule(steps);
        config.branches = {{"cond", 1.0F}, {"uncond", 1.0F}};
        config.guidance.mode = modules::FlowSamplerGuidanceMode::ClassifierFree;
        config.guidance.cond_branch = "cond";
        config.guidance.uncond_branch = "uncond";
        config.guidance.scale = guidance_scale;
        config.prediction_type = modules::FlowSamplerPredictionType::Velocity;
        config.update_rule = modules::FlowSamplerUpdateRule::Custom;
        sampler = std::make_unique<modules::FlowSamplerRuntime>(
            std::move(config),
            std::move(new_denoiser),
            std::move(new_updater));
        sampler_latent_values = latent_values;
        sampler_steps = steps;
        sampler_guidance_scale = guidance_scale;
    }

    std::vector<float> denoise_chunk(
        const std::vector<float> & condition_values,
        int64_t frames,
        const std::vector<float> & previous_latent,
        const std::vector<float> & previous_condition,
        const MiniMaxMusic3Request & request,
        uint64_t offset_blocks,
        const sampling::TorchCudaSamplingPolicy & sampling_policy,
        std::vector<float> & carry_condition,
        std::vector<float> & carry_latent) {
        const auto & config = assets->config;
        int64_t overlap = 0;
        auto chunk_condition = condition_values;
        if (!previous_latent.empty()) {
            overlap = std::min<int64_t>(
                static_cast<int64_t>(previous_latent.size()) / config.flow.in_channels,
                frames);
            copy_condition_prefix(chunk_condition, previous_condition, overlap, config.flow.condition_dim);
        }
        auto latents = sampling::generate_torch_cuda_tensor_iterator_randn(
            static_cast<size_t>(config.flow.in_channels * frames),
            request.seed,
            offset_blocks,
            sampling_policy,
            sampling::TorchRandnPrecision::BFloat16);
        auto noise_prompt = latent_tail_window(latents, frames, config.flow.in_channels, 0, overlap);
        ensure_sampler(
            config.flow.in_channels * frames,
            request.num_inference_steps,
            request.guidance_scale);
        denoiser->set_chunk_inputs(
            chunk_condition,
            frames,
            config.flow.in_channels,
            config.flow.condition_dim,
            previous_latent,
            noise_prompt,
            overlap);
        updater->set_overlap_inputs(
            previous_latent,
            std::move(noise_prompt),
            overlap,
            frames,
            config.flow.in_channels);
        sampler->run_sequence(latents);
        latents = sampler->latent();
        if (overlap > 0) {
            copy_latent_prefix(latents, previous_latent, overlap, frames, config.flow.in_channels);
        }
        const int64_t overlap_start = std::max<int64_t>(0, frames - 2 * config.overlap_latent_length);
        const int64_t overlap_end = std::max(overlap_start, frames - config.overlap_latent_length);
        carry_latent = latent_tail_window(latents, frames, config.flow.in_channels, overlap_start, overlap_end);
        carry_condition = condition_tail_window(chunk_condition, frames, config.flow.condition_dim, overlap_start, overlap_end);
        return latents;
    }

    void release_runtime_graphs() {
        if (sampler != nullptr) {
            sampler->release_runtime_graphs();
        } else if (flow != nullptr) {
            flow->release_runtime_graphs();
        }
    }

    std::shared_ptr<const MiniMaxMusic3Assets> assets;
    std::unique_ptr<MiniMaxMusic3FlowTransformerRuntime> flow;
    MiniMaxMusic3FlowDenoiserRuntime * denoiser = nullptr;
    MiniMaxMusic3FlowUpdateRuntime * updater = nullptr;
    std::unique_ptr<modules::FlowSamplerRuntime> sampler;
    int64_t sampler_latent_values = 0;
    int64_t sampler_steps = 0;
    float sampler_guidance_scale = 0.0F;
};

MiniMaxMusic3FlowSamplerRuntime::MiniMaxMusic3FlowSamplerRuntime(
    std::shared_ptr<const MiniMaxMusic3Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          storage_type)) {}

MiniMaxMusic3FlowSamplerRuntime::~MiniMaxMusic3FlowSamplerRuntime() = default;

std::vector<float> MiniMaxMusic3FlowSamplerRuntime::denoise_chunk(
    const std::vector<float> & condition_values,
    int64_t frames,
    const std::vector<float> & previous_latent,
    const std::vector<float> & previous_condition,
    const MiniMaxMusic3Request & request,
    uint64_t offset_blocks,
    const sampling::TorchCudaSamplingPolicy & sampling_policy,
    std::vector<float> & carry_condition,
    std::vector<float> & carry_latent) {
    return impl_->denoise_chunk(
        condition_values,
        frames,
        previous_latent,
        previous_condition,
        request,
        offset_blocks,
        sampling_policy,
        carry_condition,
        carry_latent);
}

void MiniMaxMusic3FlowSamplerRuntime::release_runtime_graphs() {
    if (impl_ != nullptr) {
        impl_->release_runtime_graphs();
    }
}

}  // namespace engine::models::minimax_music3
