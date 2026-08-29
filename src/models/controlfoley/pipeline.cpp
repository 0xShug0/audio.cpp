#include "engine/models/controlfoley/pipeline.h"

#include "engine/framework/codecs/mel_latent_vae44k_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/controlfoley/flow_denoiser.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace engine::models::controlfoley {
namespace {

using Clock = std::chrono::steady_clock;

std::shared_ptr<const ControlFoleyAssets> require_assets(
    std::shared_ptr<const ControlFoleyAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("ControlFoley pipeline requires assets");
    }
    return assets;
}

float positive_float_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    float fallback,
    const char * label) {
    const float value = engine::runtime::parse_positive_finite_float_option(options, keys).value_or(fallback);
    if (value <= 0.0F) {
        throw std::runtime_error(std::string("ControlFoley ") + label + " must be positive");
    }
    return value;
}

int64_t positive_i64_option(
    const std::unordered_map<std::string, std::string> & options,
    std::initializer_list<std::string_view> keys,
    int64_t fallback,
    const char * label) {
    const int64_t value = engine::runtime::parse_i64_option(options, keys).value_or(fallback);
    if (value <= 0) {
        throw std::runtime_error(std::string("ControlFoley ") + label + " must be positive");
    }
    return value;
}

ControlFoleyTemporalShape temporal_shape(float duration_sec, const ControlFoleyConfig & config) {
    const double duration = static_cast<double>(duration_sec);
    ControlFoleyTemporalShape out;
    out.latent = static_cast<int64_t>(std::ceil(
        duration * static_cast<double>(config.sample_rate) /
        static_cast<double>(config.spec_frame_frequency * config.latent_reduction_factor)));
    out.clip = static_cast<int64_t>(duration * 8.0);
    out.visual = static_cast<int64_t>(duration * 4.0);
    const double total_sync_frames = duration * 25.0;
    const double segment_count = std::floor((total_sync_frames - 16.0) / 8.0) + 1.0;
    if (out.latent <= 0 || out.clip <= 0 || out.visual <= 0 || segment_count < 0.0) {
        throw std::runtime_error("ControlFoley duration is too short for the official temporal configuration");
    }
    out.sync = static_cast<int64_t>(segment_count * 16.0 / 2.0);
    if (out.sync <= 0) {
        throw std::runtime_error("ControlFoley duration produced an empty sync sequence");
    }
    return out;
}

ControlFoleyFlowConfig flow_config(
    const ControlFoleyConfig & config,
    const ControlFoleyTemporalShape & shape) {
    ControlFoleyFlowConfig out;
    out.latent_dim = config.latent_dim;
    out.clip_dim = config.clip_dim;
    out.visual_dim = config.visual_dim;
    out.sync_dim = config.sync_dim;
    out.text_dim = config.text_dim;
    out.audio_dim = config.audio_dim;
    out.timbre_dim = config.timbre_dim;
    out.latent_seq_len = shape.latent;
    out.clip_seq_len = shape.clip;
    out.visual_seq_len = shape.visual;
    out.sync_seq_len = shape.sync;
    out.text_seq_len = config.text_seq_len;
    out.audio_seq_len = config.audio_seq_len;
    out.timbre_seq_len = config.timbre_seq_len;
    return out;
}

engine::modules::BigVganVocoderConfig bigvgan_config(
    const ControlFoleyConfig & config,
    engine::assets::TensorStorageType weight_type) {
    engine::modules::BigVganVocoderConfig out;
    out.sampling_rate = config.sample_rate;
    out.num_mels = 128;
    out.n_fft = 2048;
    out.hop_size = 512;
    out.win_size = 2048;
    out.upsample_initial_channel = 1536;
    out.snake_logscale = true;
    out.upsample_rates = {8, 4, 2, 2, 2, 2};
    out.upsample_kernel_sizes = {16, 8, 4, 4, 4, 4};
    out.resblock_kernel_sizes = {3, 7, 11};
    out.weight_storage_type = weight_type;
    return out;
}

void unnormalize_latent(
    std::vector<float> & latent,
    const ControlFoleyAssets & assets,
    int64_t frames) {
    const auto mean = assets.flow_weights->require_f32("latent_mean", {1, 1, assets.config.latent_dim});
    const auto std = assets.flow_weights->require_f32("latent_std", {1, 1, assets.config.latent_dim});
    if (static_cast<int64_t>(latent.size()) != frames * assets.config.latent_dim) {
        throw std::runtime_error("ControlFoley latent unnormalize shape mismatch");
    }
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t dim = 0; dim < assets.config.latent_dim; ++dim) {
            const size_t index = static_cast<size_t>(frame * assets.config.latent_dim + dim);
            latent[index] = latent[index] * std[static_cast<size_t>(dim)] + mean[static_cast<size_t>(dim)];
        }
    }
}

std::vector<float> latent_to_channel_first(
    const std::vector<float> & latent,
    const ControlFoleyAssets & assets,
    int64_t frames) {
    if (static_cast<int64_t>(latent.size()) != frames * assets.config.latent_dim) {
        throw std::runtime_error("ControlFoley latent channel-first reshape mismatch");
    }
    std::vector<float> out(latent.size());
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t dim = 0; dim < assets.config.latent_dim; ++dim) {
            out[static_cast<size_t>(dim * frames + frame)] =
                latent[static_cast<size_t>(frame * assets.config.latent_dim + dim)];
        }
    }
    return out;
}

ControlFoleyFlowConditionInput pack_cfg_conditions(
    const ControlFoleyFlowConditionInput & condition,
    const ControlFoleyFlowConditionInput & empty) {
    if (condition.batch != 1 || empty.batch != 1) {
        throw std::runtime_error("ControlFoley CFG condition packing expects batch-1 inputs");
    }
    ControlFoleyFlowConditionInput out;
    out.batch = 2;
    auto append = [](std::vector<float> & dst, const std::vector<float> & first, const std::vector<float> & second) {
        dst.reserve(first.size() + second.size());
        dst.insert(dst.end(), first.begin(), first.end());
        dst.insert(dst.end(), second.begin(), second.end());
    };
    append(out.clip, condition.clip, empty.clip);
    append(out.visual, condition.visual, empty.visual);
    append(out.sync, condition.sync, empty.sync);
    append(out.text, condition.text, empty.text);
    append(out.audio, condition.audio, empty.audio);
    append(out.timbre, condition.timbre, empty.timbre);
    return out;
}

}  // namespace

ControlFoleyOptions parse_controlfoley_options(
    const std::unordered_map<std::string, std::string> & options) {
    ControlFoleyOptions out;
    out.duration_sec = positive_float_option(options, {"duration_sec"}, 8.0F, "duration_sec");
    out.num_inference_steps = positive_i64_option(options, {"num_inference_steps"}, 25, "num_inference_steps");
    out.guidance_scale = positive_float_option(options, {"guidance_scale"}, 4.5F, "guidance_scale");
    out.seed = engine::runtime::parse_u32_option(options, {"seed"}).value_or(42);
    out.negative_prompt = engine::runtime::find_option(options, {"negative_prompt"}).value_or("");
    if (const auto video = engine::runtime::find_option(options, {"video"}); video.has_value() && !video->empty()) {
        out.video = std::filesystem::path(*video);
    }
    if (const auto value = engine::runtime::find_option(options, {"mask_away_clip"}); value.has_value()) {
        out.mask_away_clip = engine::runtime::parse_bool_option(*value, "mask_away_clip");
    }
    return out;
}

struct ControlFoleyPipelineRuntime::Impl {
    Impl(
        std::shared_ptr<const ControlFoleyAssets> assets_in,
        engine::core::ExecutionContext & execution_in,
        engine::assets::TensorStorageType weight_type_in)
        : assets(require_assets(std::move(assets_in))),
          execution(&execution_in),
          weight_type(weight_type_in),
          vae(
              assets->vae_weights,
              *execution,
              engine::codecs::MelLatentVae44kConfig{},
              engine::codecs::MelLatentVae44kRuntimeOptions{
                  1024ull * 1024ull * 1024ull,
                  1024ull * 1024ull * 1024ull,
                  weight_type,
                  GGML_PREC_DEFAULT,
              }),
          vocoder(engine::modules::BigVganVocoderComponent::load_from_tensor_source(
              assets->bigvgan_weights,
              execution->config(),
              bigvgan_config(assets->config, weight_type))),
          conditioner(assets, *execution, weight_type) {
        rng_policy = engine::sampling::resolve_torch_cuda_sampling_policy(
            execution->backend_type(),
            execution->config().device,
            "controlfoley.rng",
            "ControlFoley");
    }

    engine::runtime::AudioBuffer run(const engine::runtime::TaskRequest & request) {
        const auto options = parse_controlfoley_options(request.options);
        const auto shape = temporal_shape(options.duration_sec, assets->config);
        ensure_flow(shape);

        const auto start = Clock::now();
        ControlFoleyConditioningRequest conditioning_request;
        if (request.text_input.has_value()) {
            conditioning_request.text = request.text_input->text;
        }
        conditioning_request.negative_prompt = options.negative_prompt;
        if (request.audio_input.has_value()) {
            conditioning_request.audio = *request.audio_input;
        }
        conditioning_request.video = options.video;
        conditioning_request.mask_away_clip = options.mask_away_clip;
        const auto condition_build_start = Clock::now();
        auto conditioning = conditioner.build(conditioning_request, shape);
        const auto condition_build_end = Clock::now();
        const auto preprocess_start = Clock::now();
        const bool use_cfg = options.guidance_scale >= 1.0F;
        const auto cond_preprocessed = use_cfg
            ? flow->preprocess_conditions(pack_cfg_conditions(conditioning.condition, conditioning.empty))
            : flow->preprocess_conditions(conditioning.condition);
        const auto preprocess_end = Clock::now();

        const size_t latent_count = static_cast<size_t>(shape.latent * assets->config.latent_dim);
        std::vector<float> latent = engine::sampling::generate_torch_cuda_tensor_iterator_randn(
            latent_count,
            options.seed,
            0,
            rng_policy,
            engine::sampling::TorchRandnPrecision::Float32);

        const auto flow_start = Clock::now();
        const float step_dt = 1.0F / static_cast<float>(options.num_inference_steps);
        std::vector<float> cfg_latent;
        if (use_cfg) {
            cfg_latent.resize(latent.size() * 2);
        }
        const int64_t latent_values = static_cast<int64_t>(latent.size());
        const bool timing_enabled = engine::debug::timing_log_enabled();
        double flow_host_pack_ms = 0.0;
        double flow_host_update_ms = 0.0;
        for (int64_t step = 0; step < options.num_inference_steps; ++step) {
            const float t = static_cast<float>(step) * step_dt;
            if (!use_cfg) {
                const auto cond = flow->predict_flow(latent, {t}, cond_preprocessed);
                if (cond.flow.size() != latent.size()) {
                    throw std::runtime_error("ControlFoley flow output shape mismatch");
                }
                Clock::time_point update_started;
                if (timing_enabled) {
                    update_started = Clock::now();
                }
                for (int64_t i = 0; i < latent_values; ++i) {
                    latent[static_cast<size_t>(i)] += step_dt * cond.flow[static_cast<size_t>(i)];
                }
                if (timing_enabled) {
                    flow_host_update_ms += engine::debug::elapsed_ms(update_started);
                }
            } else {
                Clock::time_point pack_started;
                if (timing_enabled) {
                    pack_started = Clock::now();
                }
#ifdef _OPENMP
#pragma omp parallel for if(latent_values >= 4096)
#endif
                for (int64_t i = 0; i < latent_values; ++i) {
                    cfg_latent[static_cast<size_t>(i)] = latent[static_cast<size_t>(i)];
                    cfg_latent[static_cast<size_t>(latent_values + i)] = latent[static_cast<size_t>(i)];
                }
                if (timing_enabled) {
                    flow_host_pack_ms += engine::debug::elapsed_ms(pack_started);
                }
                const auto cfg = flow->predict_flow(cfg_latent, {t, t}, cond_preprocessed);
                if (cfg.batch != 2 || cfg.flow.size() != cfg_latent.size()) {
                    throw std::runtime_error("ControlFoley CFG flow output shape mismatch");
                }
                Clock::time_point update_started;
                if (timing_enabled) {
                    update_started = Clock::now();
                }
                for (int64_t i = 0; i < latent_values; ++i) {
                    const float guided =
                        options.guidance_scale * cfg.flow[static_cast<size_t>(i)] +
                        (1.0F - options.guidance_scale) * cfg.flow[static_cast<size_t>(latent_values + i)];
                    latent[static_cast<size_t>(i)] += step_dt * guided;
                }
                if (timing_enabled) {
                    flow_host_update_ms += engine::debug::elapsed_ms(update_started);
                }
            }
        }
        const auto flow_end = Clock::now();

        double latent_unnormalize_ms = 0.0;
        double latent_channel_first_ms = 0.0;
        latent_unnormalize_ms = engine::debug::measure_ms([&]() {
            unnormalize_latent(latent, *assets, shape.latent);
        });
        std::vector<float> vae_latent;
        latent_channel_first_ms = engine::debug::measure_ms([&]() {
            vae_latent = latent_to_channel_first(latent, *assets, shape.latent);
        });
        const auto vae_start = Clock::now();
        const auto mel = vae.decode(vae_latent, shape.latent);
        const auto vae_end = Clock::now();
        if (mel.bins != 128 || mel.frames != shape.latent * 2) {
            throw std::runtime_error("ControlFoley VAE output shape mismatch");
        }
        const auto vocoder_start = Clock::now();
        auto vocoded = vocoder.synthesize(mel.values, mel.frames);
        const auto vocoder_end = Clock::now();

        engine::debug::timing_log_scalar(
            "controlfoley.condition_build_ms",
            engine::debug::elapsed_ms(condition_build_start, condition_build_end));
        engine::debug::timing_log_scalar(
            "controlfoley.condition_preprocess_ms",
            engine::debug::elapsed_ms(preprocess_start, preprocess_end));
        engine::debug::timing_log_scalar(
            "controlfoley.flow_ms",
            engine::debug::elapsed_ms(flow_start, flow_end));
        engine::debug::timing_log_scalar(
            "controlfoley.flow_host_pack_ms",
            flow_host_pack_ms);
        engine::debug::timing_log_scalar(
            "controlfoley.flow_host_update_ms",
            flow_host_update_ms);
        engine::debug::timing_log_scalar(
            "controlfoley.latent_unnormalize_ms",
            latent_unnormalize_ms);
        engine::debug::timing_log_scalar(
            "controlfoley.latent_channel_first_ms",
            latent_channel_first_ms);
        engine::debug::timing_log_scalar(
            "controlfoley.vae_ms",
            engine::debug::elapsed_ms(vae_start, vae_end));
        engine::debug::timing_log_scalar(
            "controlfoley.vocoder_ms",
            engine::debug::elapsed_ms(vocoder_start, vocoder_end));
        engine::debug::timing_log_scalar(
            "session.wall_ms",
            engine::debug::elapsed_ms(start, vocoder_end));

        engine::runtime::AudioBuffer out;
        out.sample_rate = static_cast<int>(assets->config.sample_rate);
        out.channels = 1;
        out.samples = std::move(vocoded.waveform);
        return out;
    }

    void ensure_flow(const ControlFoleyTemporalShape & shape) {
        ControlFoleyFlowRuntimeOptions options;
        options.weight_storage_type = weight_type;
        auto config = flow_config(assets->config, shape);
        if (flow == nullptr) {
            flow = std::make_unique<ControlFoleyFlowDenoiserRuntime>(
                assets->flow_weights,
                *execution,
                config,
                options);
        } else {
            flow->update_config(config);
        }
    }

    std::shared_ptr<const ControlFoleyAssets> assets;
    engine::core::ExecutionContext * execution = nullptr;
    engine::assets::TensorStorageType weight_type = engine::assets::TensorStorageType::Native;
    engine::sampling::TorchCudaSamplingPolicy rng_policy;
    std::unique_ptr<ControlFoleyFlowDenoiserRuntime> flow;
    engine::codecs::MelLatentVae44kRuntime vae;
    engine::modules::BigVganVocoderComponent vocoder;
    ControlFoleyConditionerRuntime conditioner;
};

ControlFoleyPipelineRuntime::ControlFoleyPipelineRuntime(
    std::shared_ptr<const ControlFoleyAssets> assets,
    engine::core::ExecutionContext & execution,
    engine::assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, weight_type)) {}

ControlFoleyPipelineRuntime::~ControlFoleyPipelineRuntime() = default;

engine::runtime::AudioBuffer ControlFoleyPipelineRuntime::run(
    const engine::runtime::TaskRequest & request) {
    return impl_->run(request);
}

}  // namespace engine::models::controlfoley
