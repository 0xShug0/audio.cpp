#include "engine/models/stable_audio/session.h"

#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/stable_audio/request.h"
#include "engine/models/stable_audio/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <stdexcept>
#include <utility>

namespace engine::models::stable_audio {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const StableAudioAssets> require_assets(std::shared_ptr<const StableAudioAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Stable Audio session requires assets");
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
        throw std::runtime_error("Stable Audio model duration limit is invalid");
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
        throw std::runtime_error("Stable Audio output batch index is out of range");
    }
    const int64_t downsampling_ratio = sampling.audio_sample_size / sampling.latent_sample_size;
    if (downsampling_ratio <= 0 || sampling.audio_sample_size % sampling.latent_sample_size != 0) {
        throw std::runtime_error("Stable Audio sampling dimensions are not aligned");
    }
    const size_t mask_base = batch_index * static_cast<size_t>(sampling.latent_sample_size);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t latent_t = std::min<int64_t>(frame / downsampling_ratio, sampling.latent_sample_size - 1);
        const float mask = sampling.padding_mask[mask_base + static_cast<size_t>(latent_t)];
        for (int c = 0; c < audio.channels; ++c) {
            float & sample = audio.samples[static_cast<size_t>(frame * audio.channels + c)];
            sample *= mask;
            if (!std::isfinite(sample)) {
                throw std::runtime_error("Stable Audio decoded audio contains non-finite samples");
            }
            sample = std::max(-1.0F, std::min(1.0F, sample));
        }
    }
    if (!request.truncate_output_to_duration || !all_durations_match(request.durations_seconds)) {
        return;
    }
    const int64_t target_frames = std::min<int64_t>(
        static_cast<int64_t>(request.durations_seconds.front() * static_cast<float>(audio.sample_rate)),
        sampling.audio_sample_size);
    const int64_t target_samples = target_frames * audio.channels;
    if (target_samples >= 0 && target_samples < static_cast<int64_t>(audio.samples.size())) {
        audio.samples.resize(static_cast<size_t>(target_samples));
    }
}

std::vector<float> make_inpaint_mask(
    const StableAudioRequest & request,
    const StableAudioSamplingState & sampling,
    const StableAudioConfig & config) {
    std::vector<float> mask(static_cast<size_t>(sampling.batch * sampling.latent_sample_size), 0.0F);
    if (request.inpaint_regions.empty()) {
        return mask;
    }
    std::vector<float> single(static_cast<size_t>(sampling.latent_sample_size), 1.0F);
    for (const auto & region : request.inpaint_regions) {
        const int64_t start = std::min<int64_t>(
            sampling.latent_sample_size,
            static_cast<int64_t>(region.start_seconds * static_cast<float>(config.sample_rate)) /
                config.downsampling_ratio);
        const int64_t end = std::min<int64_t>(
            sampling.latent_sample_size,
            static_cast<int64_t>(region.end_seconds * static_cast<float>(config.sample_rate)) /
                config.downsampling_ratio);
        for (int64_t t = start; t < end; ++t) {
            single[static_cast<size_t>(t)] = 0.0F;
        }
    }
    float max_seconds = 0.0F;
    for (const float seconds : request.durations_seconds) {
        max_seconds = std::max(max_seconds, seconds);
    }
    const int64_t effective = std::min<int64_t>(
        sampling.latent_sample_size,
        static_cast<int64_t>(max_seconds * static_cast<float>(config.sample_rate)) / config.downsampling_ratio);
    for (int64_t t = effective; t < sampling.latent_sample_size; ++t) {
        single[static_cast<size_t>(t)] = 0.0F;
    }
    for (int64_t b = 0; b < sampling.batch; ++b) {
        std::copy(single.begin(), single.end(), mask.begin() + static_cast<std::ptrdiff_t>(b * sampling.latent_sample_size));
    }
    return mask;
}

void apply_init_audio(
    StableAudioSamplingState & sampling,
    const std::vector<float> & init_latents,
    float init_noise_level,
    const StableAudioConfig & config) {
    if (static_cast<int64_t>(init_latents.size()) != config.latent_dim * sampling.latent_sample_size) {
        throw std::runtime_error("Stable Audio init latent shape mismatch");
    }
    for (int64_t b = 0; b < sampling.batch; ++b) {
        for (int64_t c = 0; c < config.latent_dim; ++c) {
            for (int64_t t = 0; t < sampling.latent_sample_size; ++t) {
                const size_t dst = static_cast<size_t>((b * config.latent_dim + c) * sampling.latent_sample_size + t);
                const float init = init_latents[static_cast<size_t>(c * sampling.latent_sample_size + t)];
                sampling.noise[dst] = init * (1.0F - init_noise_level) + sampling.noise[dst] * init_noise_level;
            }
        }
    }
}

void apply_inpaint_conditioning(
    StableAudioSamplingState & sampling,
    const std::vector<float> & inpaint_latents,
    const std::vector<float> & mask,
    const StableAudioConfig & config) {
    if (static_cast<int64_t>(inpaint_latents.size()) != config.latent_dim * sampling.latent_sample_size ||
        static_cast<int64_t>(mask.size()) != sampling.batch * sampling.latent_sample_size) {
        throw std::runtime_error("Stable Audio inpaint conditioning shape mismatch");
    }
    const int64_t local_dim = config.local_add_cond_dim;
    if (local_dim != config.latent_dim + 1) {
        throw std::runtime_error("Stable Audio inpaint local conditioning dimension mismatch");
    }
    for (int64_t b = 0; b < sampling.batch; ++b) {
        for (int64_t t = 0; t < sampling.latent_sample_size; ++t) {
            const float keep = mask[static_cast<size_t>(b * sampling.latent_sample_size + t)];
            const size_t base = static_cast<size_t>((b * sampling.latent_sample_size + t) * local_dim);
            sampling.local_add_conditioning[base] = keep;
            for (int64_t c = 0; c < config.latent_dim; ++c) {
                sampling.local_add_conditioning[base + static_cast<size_t>(1 + c)] =
                    inpaint_latents[static_cast<size_t>(c * sampling.latent_sample_size + t)] * keep;
            }
        }
    }
}

}  // namespace

StableAudioSession::StableAudioSession(
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
    if (const auto mem_saver = runtime::find_option(this->options().options, {"stable_audio.mem_saver"})) {
        mem_saver_ = runtime::parse_bool_option(*mem_saver, "stable_audio.mem_saver");
    }
    for (const auto & [key, value] : this->options().options) {
        if (key == "stable_audio.max_batch" || key == "stable_audio.weight_type" || key == "stable_audio.mem_saver") {
            (void) value;
            continue;
        }
        if (key.rfind("stable_audio.", 0) == 0) {
            throw std::runtime_error("unknown Stable Audio session option: " + key);
        }
    }
}

StableAudioSession::~StableAudioSession() = default;

std::string StableAudioSession::family() const {
    return "stable_audio";
}

runtime::VoiceTaskKind StableAudioSession::task_kind() const {
    return task_.task;
}

runtime::RunMode StableAudioSession::run_mode() const {
    return task_.mode;
}

void StableAudioSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!conditioner_inputs_) {
        conditioner_inputs_ = std::make_unique<StableAudioConditionerInputs>(assets_);
    }
    if (!conditioner_runtime_) {
        conditioner_runtime_ = std::make_unique<StableAudioConditionerRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_,
            max_batch_);
    }
    if (!rf_dit_) {
        rf_dit_ = std::make_unique<StableAudioRFDiTRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_);
    }
    if (!same_) {
        same_ = std::make_unique<StableAudioSameRuntime>(
            execution_context(),
            assets_,
            weight_storage_type_);
    }
    conditioner_runtime_->prepare();
    if (request.text.has_value() || !request.options.empty()) {
        runtime::TaskRequest task_request;
        task_request.text_input = request.text;
        task_request.options = request.options;
        const StableAudioRequest parsed = clamp_request_to_model_limit(
            parse_stable_audio_request(task_request),
            assets_->config);
        const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
            execution_context().backend_type(),
            execution_context().config().device,
            "stable_audio.rng",
            "Stable Audio");
        const StableAudioSamplingState sampling = prepare_stable_audio_sampling_state(assets_->config, parsed, rng_policy);
        rf_dit_->prepare(sampling, parsed.guidance_scale);
        same_->prepare_decode(sampling.batch, sampling.latent_sample_size, parsed.chunked_decode);
    }
    mark_prepared();
}

runtime::TaskResult StableAudioSession::run(const runtime::TaskRequest & request) {
    require_prepared("Stable Audio run");
    if (!conditioner_inputs_) {
        throw std::runtime_error("Stable Audio conditioner was not prepared");
    }
    if (!conditioner_runtime_) {
        throw std::runtime_error("Stable Audio conditioner runtime was not prepared");
    }
    if (!rf_dit_) {
        throw std::runtime_error("Stable Audio RF DiT runtime was not prepared");
    }
    if (!same_) {
        throw std::runtime_error("Stable Audio SAME runtime was not prepared");
    }
    const auto wall_start = Clock::now();
    const StableAudioRequest parsed = clamp_request_to_model_limit(
        parse_stable_audio_request(request),
        assets_->config);
    const StableAudioConditioningBatch conditioning = conditioner_inputs_->build(parsed);
    const auto rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        execution_context().backend_type(),
        execution_context().config().device,
        "stable_audio.rng",
        "Stable Audio");
    const auto sampling_start = Clock::now();
    StableAudioSamplingState sampling = prepare_stable_audio_sampling_state(assets_->config, parsed, rng_policy);
    uint64_t rng_offset_blocks = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(sampling.noise.size()),
        rng_policy);
    engine::debug::timing_log_scalar(
        "stable_audio.sampling_prepare_ms",
        engine::debug::elapsed_ms(sampling_start, Clock::now()));
    if (parsed.init_audio.has_value()) {
        const auto init_latents = same_->encode(
            *parsed.init_audio,
            sampling.audio_sample_size,
            parsed.seed,
            rng_offset_blocks);
        apply_init_audio(sampling, init_latents, parsed.init_noise_level, assets_->config);
    }
    if (parsed.inpaint_audio.has_value() || !parsed.inpaint_regions.empty()) {
        const auto mask = make_inpaint_mask(parsed, sampling, assets_->config);
        std::vector<float> inpaint_latents(static_cast<size_t>(assets_->config.latent_dim * sampling.latent_sample_size), 0.0F);
        if (parsed.inpaint_audio.has_value()) {
            inpaint_latents = same_->encode(
                *parsed.inpaint_audio,
                sampling.audio_sample_size,
                parsed.seed,
                rng_offset_blocks);
        }
        apply_inpaint_conditioning(sampling, inpaint_latents, mask, assets_->config);
    }
    const auto conditioner_start = Clock::now();
    const StableAudioConditioningInputs conditioning_inputs = conditioner_runtime_->encode(conditioning);
    engine::debug::timing_log_scalar(
        "stable_audio.conditioner_ms",
        engine::debug::elapsed_ms(conditioner_start, Clock::now()));
    if (mem_saver_) {
        const auto release_start = Clock::now();
        conditioner_runtime_->release_runtime_graphs();
        engine::debug::timing_log_scalar(
            "stable_audio.conditioner_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }
    const auto rf_start = Clock::now();
    const auto latents = rf_dit_->sample(
        sampling,
        conditioning_inputs,
        parsed.seed,
        rng_offset_blocks,
        rng_policy,
        parsed.sampler,
        parsed.guidance_scale,
        parsed.apg_scale);
    engine::debug::timing_log_scalar(
        "stable_audio.rf_dit_ms",
        engine::debug::elapsed_ms(rf_start, Clock::now()));
    if (mem_saver_) {
        const auto release_start = Clock::now();
        rf_dit_->release_runtime_graphs();
        engine::debug::timing_log_scalar(
            "stable_audio.rf_dit_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }
    const auto same_start = Clock::now();
    auto audio_outputs = same_->decode(
        latents,
        sampling.batch,
        sampling.latent_sample_size,
        parsed.seed,
        rng_offset_blocks,
        parsed.chunked_decode);
    engine::debug::timing_log_scalar(
        "stable_audio.same_decode_ms",
        engine::debug::elapsed_ms(same_start, Clock::now()));
    if (mem_saver_) {
        const auto release_start = Clock::now();
        same_->release_runtime_graphs();
        engine::debug::timing_log_scalar(
            "stable_audio.same_release.runtime_graphs_ms",
            engine::debug::elapsed_ms(release_start, Clock::now()));
    }
    engine::debug::timing_log_scalar(
        "session.wall_ms",
        engine::debug::elapsed_ms(wall_start, Clock::now()));
    runtime::TaskResult out;
    if (audio_outputs.empty()) {
        throw std::runtime_error("Stable Audio SAME decode produced no audio");
    }
    for (size_t i = 0; i < audio_outputs.size(); ++i) {
        clamp_and_truncate_audio(audio_outputs[i], parsed, sampling, i);
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

}  // namespace engine::models::stable_audio
