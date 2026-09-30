#include "engine/models/stable_audio/foundation/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/stable_audio/request.h"
#include "engine/models/stable_audio/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace engine::models::stable_audio::foundation {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const StableAudioAssets> require_assets(std::shared_ptr<const StableAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Stable Audio Foundation session requires assets");
    }
    return assets;
}

void validate_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

bool all_durations_match(const std::vector<float> & durations) {
    if (durations.empty()) {
        return true;
    }
    for (const float duration : durations) {
        if (std::abs(duration - durations.front()) > 1.0e-6F) {
            return false;
        }
    }
    return true;
}

StableAudioRequest clamp_request_to_model_limit(StableAudioRequest request, const StableAudioConfig & config) {
    if (config.sample_rate <= 0 || config.sample_size <= 0) {
        throw std::runtime_error("Stable Audio Foundation model duration limit is invalid");
    }
    const float max_seconds = static_cast<float>(config.sample_size) / static_cast<float>(config.sample_rate);
    for (float & duration : request.durations_seconds) {
        duration = std::min(duration, max_seconds);
    }
    return request;
}

void clamp_and_truncate_audio(
    runtime::AudioBuffer & audio,
    const StableAudioRequest & request,
    const StableAudioSamplingState & sampling,
    size_t batch_index) {
    const int64_t frames = static_cast<int64_t>(audio.samples.size()) / audio.channels;
    if (batch_index >= static_cast<size_t>(sampling.batch)) {
        throw std::runtime_error("Stable Audio Foundation output batch index is out of range");
    }
    const int64_t downsampling_ratio = frames / sampling.latent_sample_size;
    if (downsampling_ratio <= 0) {
        throw std::runtime_error("Stable Audio Foundation decoded dimensions are not aligned");
    }
    const size_t mask_base = batch_index * static_cast<size_t>(sampling.latent_sample_size);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t latent_t = std::min<int64_t>(frame / downsampling_ratio, sampling.latent_sample_size - 1);
        const float mask = sampling.padding_mask[mask_base + static_cast<size_t>(latent_t)];
        for (int c = 0; c < audio.channels; ++c) {
            float & sample = audio.samples[static_cast<size_t>(frame * audio.channels + c)];
            sample *= mask;
            if (!std::isfinite(sample)) {
                throw std::runtime_error("Stable Audio Foundation decoded audio contains non-finite samples");
            }
            sample = std::max(-1.0F, std::min(1.0F, sample));
        }
    }
    if (!request.truncate_output_to_duration || !all_durations_match(request.durations_seconds)) {
        return;
    }
    const int64_t target_frames = std::min<int64_t>(
        static_cast<int64_t>(request.durations_seconds.front() * static_cast<float>(audio.sample_rate)),
        frames);
    const int64_t target_samples = target_frames * audio.channels;
    if (target_samples >= 0 && target_samples < static_cast<int64_t>(audio.samples.size())) {
        audio.samples.resize(static_cast<size_t>(target_samples));
    }
}

bool is_k_diffusion_sampler(std::string_view sampler_type) {
    return sampler_type == "dpmpp-2m" || sampler_type == "dpmpp-3m-sde";
}

struct FoundationRequest {
    StableAudioRequest audio;
    float sigma_min = 0.01F;
    float sigma_max = 100.0F;
    float rho = 1.0F;
};

void erase_option_aliases(
    std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys) {
    for (const std::string_view key : keys) {
        options.erase(std::string(key));
    }
}

FoundationRequest parse_foundation_request(const runtime::TaskRequest & request, const StableAudioConfig & config) {
    runtime::TaskRequest shared_request = request;
    const auto sampler = runtime::find_option(request.options, {"sampler"});
    erase_option_aliases(shared_request.options, {"sampler"});
    FoundationRequest out;
    out.audio = clamp_request_to_model_limit(parse_stable_audio_request(shared_request), config);
    if (sampler.has_value()) {
        if (*sampler != "pingpong" &&
            *sampler != "euler" &&
            *sampler != "dpmpp-2m" &&
            *sampler != "dpmpp-3m-sde") {
            throw std::runtime_error("Stable Audio Foundation sampler must be pingpong, euler, dpmpp-2m, or dpmpp-3m-sde");
        }
        out.audio.sampler = *sampler;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"sigma_min"})) {
        out.sigma_min = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"sigma_max"})) {
        out.sigma_max = *value;
    }
    if (const auto value = runtime::parse_finite_float_option(request.options, {"rho"})) {
        out.rho = *value;
    }
    return out;
}

void apply_init_audio(
    StableAudioSamplingState & sampling,
    const std::vector<float> & init_latents,
    const StableAudioConfig & config,
    const StableAudioRequest & request,
    float sampler_sigma_max) {
    if (static_cast<int64_t>(init_latents.size()) != config.latent_dim * sampling.latent_sample_size) {
        throw std::runtime_error("Stable Audio Foundation init latent shape mismatch");
    }
    if (is_k_diffusion_sampler(request.sampler)) {
        if (!(sampler_sigma_max > 0.0F)) {
            throw std::runtime_error("Stable Audio Foundation k-diffusion init path requires positive init_noise_level");
        }
        for (int64_t b = 0; b < sampling.batch; ++b) {
            for (int64_t c = 0; c < config.latent_dim; ++c) {
                for (int64_t t = 0; t < sampling.latent_sample_size; ++t) {
                    const size_t dst = static_cast<size_t>((b * config.latent_dim + c) * sampling.latent_sample_size + t);
                    const float init = init_latents[static_cast<size_t>(c * sampling.latent_sample_size + t)];
                    sampling.noise[dst] += init / sampler_sigma_max;
                }
            }
        }
        return;
    }

    const float sigma = request.init_noise_level;
    const float alpha = std::sqrt(std::max(0.0F, 1.0F - sigma * sigma));
    for (int64_t b = 0; b < sampling.batch; ++b) {
        for (int64_t c = 0; c < config.latent_dim; ++c) {
            for (int64_t t = 0; t < sampling.latent_sample_size; ++t) {
                const size_t dst = static_cast<size_t>((b * config.latent_dim + c) * sampling.latent_sample_size + t);
                const float init = init_latents[static_cast<size_t>(c * sampling.latent_sample_size + t)];
                sampling.noise[dst] = init * alpha + sampling.noise[dst] * sigma;
            }
        }
    }
}

}  // namespace

FoundationSession::FoundationSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const StableAudioAssets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))) {
    max_batch_ = runtime::parse_int_option(options.options, {"stable_audio.max_batch"}).value_or(max_batch_);
    if (max_batch_ <= 0) {
        throw std::runtime_error("stable_audio.max_batch must be positive");
    }
    if (const auto it = this->options().options.find("stable_audio.weight_type"); it != this->options().options.end()) {
        weight_storage_type_ = engine::assets::parse_tensor_storage_type(it->second);
        validate_weight_storage(weight_storage_type_, "stable_audio.weight_type");
    }
}

FoundationSession::~FoundationSession() = default;

std::string FoundationSession::family() const {
    return "stable_audio";
}

runtime::VoiceTaskKind FoundationSession::task_kind() const {
    return task_.task;
}

runtime::RunMode FoundationSession::run_mode() const {
    return task_.mode;
}

void FoundationSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!conditioner_inputs_) {
        conditioner_inputs_ = std::make_unique<StableAudioConditionerInputs>(assets_);
    }
    if (!conditioner_runtime_) {
        conditioner_runtime_ = std::make_unique<FoundationConditionerRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_,
            max_batch_);
    }
    if (!rf_dit_) {
        rf_dit_ = std::make_unique<FoundationRFDiTRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_);
    }
    if (!oobleck_) {
        oobleck_ = std::make_unique<OobleckAutoencoderRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_);
    }
    conditioner_runtime_->prepare();
    if (request.text.has_value() || !request.options.empty()) {
        runtime::TaskRequest task_request;
        task_request.text_input = request.text;
        task_request.options = request.options;
        const FoundationRequest parsed = parse_foundation_request(task_request, assets_->config);
        const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
            execution_context().backend_type(),
            execution_context().config().device,
            "stable_audio.rng",
            "Stable Audio Foundation");
        const StableAudioSamplingState sampling = prepare_stable_audio_sampling_state(assets_->config, parsed.audio, rng_policy);
        rf_dit_->prepare(sampling, parsed.audio.guidance_scale);
        oobleck_->prepare_decode(sampling.batch, sampling.latent_sample_size);
    }
    mark_prepared();
}

runtime::TaskResult FoundationSession::run(const runtime::TaskRequest & request) {
    require_prepared("Stable Audio Foundation run");
    if (task_.task != runtime::VoiceTaskKind::AudioGeneration) {
        throw std::runtime_error("Stable Audio Foundation supports only the gen task");
    }
    const auto wall_start = Clock::now();
    const FoundationRequest parsed = parse_foundation_request(request, assets_->config);
    if (parsed.audio.inpaint_audio.has_value() || !parsed.audio.inpaint_regions.empty()) {
        throw std::runtime_error("Stable Audio Foundation inpaint conditioning is not declared by this model config");
    }
    const StableAudioConditioningBatch conditioning = conditioner_inputs_->build(parsed.audio);
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context().backend_type(),
        execution_context().config().device,
        "stable_audio.rng",
        "Stable Audio Foundation");
    const auto sampling_start = Clock::now();
    StableAudioSamplingState sampling = prepare_stable_audio_sampling_state(assets_->config, parsed.audio, rng_policy);
    uint64_t rng_offset_blocks = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(sampling.noise.size()),
        rng_policy);
    float sampler_sigma_max = parsed.sigma_max;
    if (parsed.audio.init_audio.has_value() && is_k_diffusion_sampler(parsed.audio.sampler)) {
        sampler_sigma_max = parsed.audio.init_noise_level;
    }
    if (parsed.audio.init_audio.has_value()) {
        const auto init_latents = oobleck_->encode(
            *parsed.audio.init_audio,
            sampling.audio_sample_size,
            parsed.audio.seed,
            rng_offset_blocks,
            rng_policy);
        apply_init_audio(sampling, init_latents, assets_->config, parsed.audio, sampler_sigma_max);
    }
    engine::debug::timing_log_scalar(
        "stable_audio.foundation.sampling_prepare_ms",
        engine::debug::elapsed_ms(sampling_start, Clock::now()));
    const auto conditioner_start = Clock::now();
    const StableAudioConditioningInputs conditioning_inputs = conditioner_runtime_->encode(conditioning);
    engine::debug::timing_log_scalar(
        "stable_audio.foundation.conditioner_ms",
        engine::debug::elapsed_ms(conditioner_start, Clock::now()));
    const auto rf_start = Clock::now();
    const auto latents = rf_dit_->sample(
        sampling,
        conditioning_inputs,
        parsed.audio.seed,
        rng_offset_blocks,
        rng_policy,
        parsed.audio.sampler,
        parsed.sigma_min,
        sampler_sigma_max,
        parsed.rho,
        parsed.audio.guidance_scale,
        parsed.audio.apg_scale);
    engine::debug::timing_log_scalar(
        "stable_audio.foundation.rf_dit_ms",
        engine::debug::elapsed_ms(rf_start, Clock::now()));
    const auto decode_start = Clock::now();
    auto audio_outputs = oobleck_->decode(latents, sampling.batch, sampling.latent_sample_size);
    engine::debug::timing_log_scalar(
        "stable_audio.foundation.oobleck_decode_ms",
        engine::debug::elapsed_ms(decode_start, Clock::now()));
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(wall_start, Clock::now()));
    if (audio_outputs.empty()) {
        throw std::runtime_error("Stable Audio Foundation Oobleck decode produced no audio");
    }
    runtime::TaskResult out;
    for (size_t i = 0; i < audio_outputs.size(); ++i) {
        clamp_and_truncate_audio(audio_outputs[i], parsed.audio, sampling, i);
    }
    out.audio_output = audio_outputs.front();
    for (size_t i = 0; i < audio_outputs.size(); ++i) {
        out.named_audio_outputs.push_back(runtime::NamedAudioBuffer{
            "audio_" + std::to_string(i),
            std::move(audio_outputs[i]),
            {},
        });
    }
    return out;
}

}  // namespace engine::models::stable_audio::foundation
