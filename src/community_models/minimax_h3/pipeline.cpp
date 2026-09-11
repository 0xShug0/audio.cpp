#include "engine/community_models/minimax_h3/pipeline.h"

#include "engine/community_models/minimax_h3/audio_vae_decoder.h"
#include "engine/community_models/minimax_h3/dit_acceleration.h"
#include "engine/community_models/minimax_h3/dit_denoiser.h"
#include "engine/community_models/minimax_h3/generation_plan.h"
#include "engine/community_models/minimax_h3/sampler.h"
#include "engine/community_models/minimax_h3/video_vae_decoder.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/transformers/qwen3_vl_encoder_runtime.h"
#include "engine/framework/tokenizers/llama_bpe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::minimax_h3 {
namespace {

namespace core = engine::core;
namespace tokenizers = engine::tokenizers;
using Clock = std::chrono::steady_clock;

std::shared_ptr<const MiniMaxH3Assets> require_runtime_assets(std::shared_ptr<const MiniMaxH3Assets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MiniMax-H3 runtime requires assets");
    }
    return assets;
}

MiniMaxH3SamplerGraphSpec sampler_graph_spec(
    const MiniMaxH3Config & cfg,
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout) {
    const int64_t video_dim = cfg.video_latents_dim * 4;
    const int64_t audio_rows = cfg.audio_steps * cfg.audio_channels;
    return MiniMaxH3SamplerGraphSpec{
        MiniMaxH3SamplerTargetSpec{
            core::TensorShape::from_dims({layout.total, video_dim}),
            core::TensorShape::from_dims({layout.video_rows, video_dim}),
            layout.video_start,
        },
        MiniMaxH3SamplerTargetSpec{
            core::TensorShape::from_dims({layout.total, cfg.audio_latents_dim}),
            core::TensorShape::from_dims({audio_rows, cfg.audio_latents_dim}),
            layout.audio_start,
        },
    };
}

float sigma_at_denoise_percent(const std::vector<float> & sigmas, float percent) {
    if (sigmas.empty() || !std::isfinite(percent)) {
        throw std::runtime_error("MiniMax-H3 first-block cache window cannot be resolved");
    }
    const float clamped = std::min(1.0F, std::max(0.0F, percent));
    const float position = clamped * static_cast<float>(sigmas.size() - 1);
    const auto lower = static_cast<size_t>(std::floor(position));
    const auto upper = std::min<size_t>(lower + 1, sigmas.size() - 1);
    const float mix = position - static_cast<float>(lower);
    return sigmas[lower] + (sigmas[upper] - sigmas[lower]) * mix;
}

float spectrum_coordinate(float sigma, float sigma_min, float sigma_max) {
    const float span = sigma_max - sigma_min;
    if (!std::isfinite(sigma) || !std::isfinite(span) || span <= 0.0F) {
        throw std::runtime_error("MiniMax-H3 spectrum requires a finite sigma schedule range");
    }
    return std::clamp(2.0F * (sigma - sigma_min) / span - 1.0F, -1.0F, 1.0F);
}

engine::modules::Qwen3VlEncoderRuntimeConfig qwen3_vl_encoder_config(
    const MiniMaxH3Config & cfg,
    size_t weight_context_bytes) {
    engine::modules::Qwen3VlEncoderRuntimeConfig out;
    out.trace_name = "minimax_h3.prompt_encoder";
    out.model_prefix = "model.language_model";
    out.vocab_size = cfg.vocab_size;
    out.weight_context_bytes = weight_context_bytes;
    out.graph_arena_bytes = 512ull * 1024ull * 1024ull;
    out.stack.hidden_size = cfg.prompt_hidden;
    out.stack.num_attention_heads = cfg.prompt_heads;
    out.stack.num_key_value_heads = cfg.prompt_kv_heads;
    out.stack.head_dim = cfg.prompt_head_dim;
    out.stack.intermediate_size = cfg.prompt_intermediate;
    out.stack.layers = cfg.prompt_layers;
    out.stack.rms_norm_eps = cfg.prompt_eps;
    out.stack.rope_theta = cfg.prompt_rope_theta;
    out.stack.attention_precision = GGML_PREC_DEFAULT;
    out.stack.projection_precision = GGML_PREC_DEFAULT;
    out.stack.use_qk_norm = true;
    out.readback_round_type = GGML_TYPE_BF16;
    return out;
}

class ScopedEncoderWeightRelease {
public:
    explicit ScopedEncoderWeightRelease(engine::modules::Qwen3VlEncoderRuntime & runtime) : runtime_(runtime) {}
    ~ScopedEncoderWeightRelease() {
        runtime_.release_weights();
    }

    ScopedEncoderWeightRelease(const ScopedEncoderWeightRelease &) = delete;
    ScopedEncoderWeightRelease & operator=(const ScopedEncoderWeightRelease &) = delete;

private:
    engine::modules::Qwen3VlEncoderRuntime & runtime_;
};

}  // namespace

struct MiniMaxH3PipelineRuntime::Impl {
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer;
    size_t weight_context_bytes = 0;
    bool mem_saver = false;
    std::unique_ptr<engine::modules::Qwen3VlEncoderRuntime> text_encoder;
    std::unique_ptr<MiniMaxH3DitWeightStore> dit_weights;
    std::unique_ptr<AudioVaeWeightStore> audio_vae_weights;
    std::unique_ptr<VideoVaeWeightStore> video_vae_weights;
    VideoVaeDecodeCache video_vae_decode_cache;

    Impl(
        core::ExecutionContext & execution,
        const MiniMaxH3Assets & assets,
        size_t weight_context_bytes_in,
        bool mem_saver_in)
        : weight_context_bytes(weight_context_bytes_in),
          mem_saver(mem_saver_in) {
        tokenizers::LlamaBpeTokenizerSpec spec;
        spec.vocab_path = assets.resources.require_file("vocab");
        spec.merges_path = assets.resources.require_file("merges");
        spec.tokenizer_config_path = assets.resources.require_file("tokenizer_config");
        spec.tokenizer_json_path = assets.resources.require_file("tokenizer_json");
        spec.pre_type = tokenizers::LlamaBpePreTokenizer::Qwen35;
        tokenizer = tokenizers::load_llama_bpe_tokenizer(spec);
        text_encoder = std::make_unique<engine::modules::Qwen3VlEncoderRuntime>(
            execution,
            assets.text_encoder_weights,
            qwen3_vl_encoder_config(assets.config, weight_context_bytes));
    }
};

MiniMaxH3PipelineRuntime::MiniMaxH3PipelineRuntime(
    engine::core::ExecutionContext & execution,
    std::shared_ptr<const MiniMaxH3Assets> assets,
    size_t weight_context_bytes,
    bool mem_saver)
    : execution_(execution),
      assets_(require_runtime_assets(std::move(assets))),
      impl_(std::make_unique<Impl>(execution_, *assets_, weight_context_bytes, mem_saver)) {
}

MiniMaxH3PipelineRuntime::~MiniMaxH3PipelineRuntime() = default;

MiniMaxH3GenerateResult MiniMaxH3PipelineRuntime::generate(const MiniMaxH3GenerateRequest & request) {
    const auto cfg = resolve_generation_config(assets_->config, request);
    const auto total_start = Clock::now();
    std::vector<int32_t> ids;
    if (impl_->tokenizer == nullptr) {
        throw std::runtime_error("MiniMax-H3 requires tokenizer");
    }
    ids = impl_->tokenizer->encode(request.prompt, false);
    if (ids.empty()) {
        throw std::runtime_error("MiniMax-H3 tokenizer produced no prompt tokens");
    }
    const bool cfg_enabled = std::abs(request.guidance_scale - 1.0F) > 1.0e-6F;
    std::vector<int32_t> negative_ids;
    if (cfg_enabled) {
        negative_ids = impl_->tokenizer->encode(request.negative_prompt, false);
        if (negative_ids.empty()) {
            throw std::runtime_error("MiniMax-H3 tokenizer produced no negative prompt tokens");
        }
    }
    engine::debug::trace_log_i32("minimax_h3.prompt_token_ids", {static_cast<int64_t>(ids.size())}, ids);
    engine::debug::trace_log_scalar("minimax_h3.config.text_tokens", static_cast<int64_t>(ids.size()));
    engine::debug::trace_log_scalar("minimax_h3.config.height", cfg.height);
    engine::debug::trace_log_scalar("minimax_h3.config.width", cfg.width);
    engine::debug::trace_log_scalar("minimax_h3.config.num_frames", cfg.num_frames);
    engine::debug::trace_log_scalar("minimax_h3.config.video_latent_t", cfg.video_latent_t);
    engine::debug::trace_log_scalar("minimax_h3.config.video_latent_h", cfg.video_latent_h);
    engine::debug::trace_log_scalar("minimax_h3.config.video_latent_w", cfg.video_latent_w);
    engine::debug::trace_log_scalar("minimax_h3.config.video_patches", cfg.video_patches);
    engine::debug::trace_log_scalar("minimax_h3.config.audio_steps", cfg.audio_steps);
    engine::debug::trace_log_scalar("minimax_h3.config.audio_channels", cfg.audio_channels);
    engine::debug::trace_log_scalar("minimax_h3.config.denoise_steps", cfg.denoise_steps);
    engine::debug::trace_log_scalar("minimax_h3.config.guidance_scale", request.guidance_scale);
    engine::debug::trace_log_scalar("minimax_h3.config.flow_shift", request.flow_shift);
    engine::debug::trace_log_scalar("minimax_h3.config.audio_flow_shift", request.audio_flow_shift);
    engine::debug::trace_log_scalar("minimax_h3.config.sampler", static_cast<int64_t>(request.sampler));
    engine::debug::trace_log_scalar("minimax_h3.config.dit_acceleration", static_cast<int64_t>(request.dit_acceleration));

    const auto prompt_start = Clock::now();
    std::vector<float> prompt_out;
    std::vector<float> negative_prompt_out;
    {
        if (impl_->text_encoder == nullptr) {
            throw std::runtime_error("MiniMax-H3 text encoder runtime is missing");
        }
        ScopedEncoderWeightRelease release_encoder_weights(*impl_->text_encoder);
        engine::modules::Qwen3VlEncoderOptions encoder_options;
        encoder_options.layerwise = request.text_layerwise;
        encoder_options.layerwise_batch = request.text_layerwise_batch;
        prompt_out = impl_->text_encoder->encode_text(ids, encoder_options).hidden;
        if (cfg_enabled) {
            negative_prompt_out = impl_->text_encoder->encode_text(negative_ids, encoder_options).hidden;
        }
    }
    engine::debug::timing_log_scalar("minimax_h3.prompt_encoder_ms", engine::debug::elapsed_ms(prompt_start, Clock::now()));

    auto latents = generate_initial_latents(cfg, execution_, request.seed);
    auto video = std::move(latents.video);
    auto audio = std::move(latents.audio);
    engine::debug::trace_log_f32("minimax_h3.prompt_encoder", {static_cast<int64_t>(prompt_out.size()) / cfg.text_dim, cfg.text_dim}, prompt_out);
    engine::debug::trace_log_f32("minimax_h3.initial_video_rows", {cfg.video_patches, cfg.video_latents_dim * 4}, video);
    engine::debug::trace_log_f32("minimax_h3.initial_audio_rows", {cfg.audio_channels * cfg.audio_steps, cfg.audio_latents_dim}, audio);

    const auto video_sigmas = h3_sigmas(cfg.denoise_steps, request.flow_shift);
    const auto audio_sigmas = h3_sigmas(cfg.denoise_steps, request.audio_flow_shift);
    const auto [video_sigma_min_it, video_sigma_max_it] = std::minmax_element(video_sigmas.begin(), video_sigmas.end());
    if (video_sigma_min_it == video_sigmas.end() || video_sigma_max_it == video_sigmas.end()) {
        throw std::runtime_error("MiniMax-H3 generation requires a non-empty sigma schedule");
    }
    const float video_sigma_min = *video_sigma_min_it;
    const float video_sigma_max = *video_sigma_max_it;
    const auto step_timestep_values = build_step_timestep_values(cfg, audio_sigmas, video_sigmas);
    MiniMaxH3GenerateRequest first_block_request = request;
    if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::FirstBlockCache &&
        !request.first_block_cache_sigma_window) {
        first_block_request.first_block_cache_start_sigma =
            sigma_at_denoise_percent(video_sigmas, request.first_block_cache_start_percent);
        first_block_request.first_block_cache_end_sigma =
            sigma_at_denoise_percent(video_sigmas, request.first_block_cache_end_percent);
    }
    if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::FirstBlockCache) {
        engine::debug::trace_log_scalar("minimax_h3.config.first_block_cache_threshold", first_block_request.first_block_cache_threshold);
        engine::debug::trace_log_scalar("minimax_h3.config.first_block_cache_start_sigma", first_block_request.first_block_cache_start_sigma);
        engine::debug::trace_log_scalar("minimax_h3.config.first_block_cache_end_sigma", first_block_request.first_block_cache_end_sigma);
    }
    std::unique_ptr<MiniMaxH3DitWeightStore> scoped_dit_weights;
    std::unique_ptr<MiniMaxH3DitLayerwiseRuntime> layerwise_dit;
    std::unique_ptr<MiniMaxH3DitLayerwiseRuntime> negative_layerwise_dit;
    MiniMaxH3DitWeightStore * dit_weights = impl_->dit_weights.get();
    if (request.dit_acceleration != MiniMaxH3DitAccelerationMode::None && request.dit_layerwise) {
        throw std::runtime_error("MiniMax-H3 DiT acceleration is only wired for the full DiT graph path");
    }
    if (request.dit_acceleration != MiniMaxH3DitAccelerationMode::None &&
        request.sampler != MiniMaxH3SamplerMode::Euler) {
        throw std::runtime_error("MiniMax-H3 DiT acceleration currently requires the Euler sampler");
    }
    if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::Spectrum && cfg_enabled) {
        throw std::runtime_error("MiniMax-H3 Spectrum acceleration currently requires guidance_scale=1.0");
    }
    if (request.dit_layerwise) {
        const auto weight_start = Clock::now();
        layerwise_dit = std::make_unique<MiniMaxH3DitLayerwiseRuntime>(
            execution_,
            assets_->dit_weights,
                cfg,
                prompt_out,
                impl_->weight_context_bytes,
                request.dit_layerwise_batch,
                request.dit_mlp_chunk_tokens);
        if (cfg_enabled) {
            negative_layerwise_dit = std::make_unique<MiniMaxH3DitLayerwiseRuntime>(
                execution_,
                assets_->dit_weights,
                cfg,
                negative_prompt_out,
                impl_->weight_context_bytes,
                request.dit_layerwise_batch,
                request.dit_mlp_chunk_tokens);
        }
        engine::debug::timing_log_scalar("minimax_h3.dit.weights_load_ms", engine::debug::elapsed_ms(weight_start, Clock::now()));
    } else if (impl_->mem_saver) {
        const auto weight_start = Clock::now();
        scoped_dit_weights = std::make_unique<MiniMaxH3DitWeightStore>(execution_, assets_->dit_weights, impl_->weight_context_bytes);
        engine::debug::timing_log_scalar("minimax_h3.dit.weights_load_ms", engine::debug::elapsed_ms(weight_start, Clock::now()));
        dit_weights = scoped_dit_weights.get();
    } else if (dit_weights == nullptr) {
        const auto weight_start = Clock::now();
        impl_->dit_weights = std::make_unique<MiniMaxH3DitWeightStore>(execution_, assets_->dit_weights, impl_->weight_context_bytes);
        engine::debug::timing_log_scalar("minimax_h3.dit.weights_load_ms", engine::debug::elapsed_ms(weight_start, Clock::now()));
        dit_weights = impl_->dit_weights.get();
    } else {
        engine::debug::timing_log_scalar("minimax_h3.dit.weights_load_ms", 0.0);
    }
    const auto step_timestep_features = cfg.adaln_curve_grid > 0
        ? build_step_adaln_curve_inputs(
              cfg,
              step_timestep_values,
              request.dit_layerwise ? layerwise_dit->adaln_curve_table() : dit_weights->adaln_curve_table)
        : build_step_timestep_features(cfg, audio_sigmas, video_sigmas);
    const auto denoise_start = Clock::now();
    double sampler_cfg_ms = 0.0;
    double sampler_update_ms = 0.0;
    double round_ms = 0.0;
    double positive_input_upload_ms = 0.0;
    double positive_output_read_ms = 0.0;
    double negative_input_upload_ms = 0.0;
    double negative_output_read_ms = 0.0;
    if (request.dit_layerwise) {
        DitGraphResult pred;
        DitGraphResult negative_pred;
        const int64_t timestep_width = cfg.adaln_curve_grid > 0 ? cfg.time_embed_dim : cfg.timestep_input_dim;
        for (int64_t step = 0; step < cfg.denoise_steps; ++step) {
            const size_t timestep_offset = static_cast<size_t>(step * 2 * timestep_width);
            const float next_video = step + 1 == cfg.denoise_steps ? 0.0F : video_sigmas[static_cast<size_t>(step + 1)];
            const float next_audio = step + 1 == cfg.denoise_steps ? 0.0F : audio_sigmas[static_cast<size_t>(step + 1)];
            const float sigma_delta_video = video_sigmas[static_cast<size_t>(step)] - next_video;
            const float sigma_delta_audio = audio_sigmas[static_cast<size_t>(step)] - next_audio;
            const bool split_audio_timestep =
                std::abs(step_timestep_values[static_cast<size_t>(step * 2)] -
                         step_timestep_values[static_cast<size_t>(step * 2 + 1)]) > 1.0e-6F;
            layerwise_dit->run(
                audio,
                video,
                step_timestep_features.data() + timestep_offset,
                static_cast<size_t>(2 * timestep_width),
                sigma_delta_audio,
                sigma_delta_video,
                split_audio_timestep,
                cfg_enabled,
                !cfg_enabled,
                pred);
            if (cfg_enabled) {
                negative_layerwise_dit->run(
                    audio,
                    video,
                    step_timestep_features.data() + timestep_offset,
                    static_cast<size_t>(2 * timestep_width),
                    sigma_delta_audio,
                    sigma_delta_video,
                    split_audio_timestep,
                    true,
                    false,
                    negative_pred);
                const auto cfg_start = Clock::now();
                for (size_t i = 0; i < pred.video.size(); ++i) {
                    pred.video[i] = negative_pred.video[i] + request.guidance_scale * (pred.video[i] - negative_pred.video[i]);
                }
                for (size_t i = 0; i < pred.audio.size(); ++i) {
                    pred.audio[i] = negative_pred.audio[i] + request.guidance_scale * (pred.audio[i] - negative_pred.audio[i]);
                }
                sampler_cfg_ms += engine::debug::elapsed_ms(cfg_start, Clock::now());
            }
            if (step == 0 && cfg_enabled) {
                engine::debug::trace_log_f32("minimax_h3.dit_step0_video", {cfg.video_patches, cfg.video_latents_dim * 4}, pred.video);
                engine::debug::trace_log_f32("minimax_h3.dit_step0_audio", {cfg.audio_channels * cfg.audio_steps, cfg.audio_latents_dim}, pred.audio);
            }
            const auto update_start = Clock::now();
            if (cfg_enabled) {
                for (size_t i = 0; i < pred.video.size(); ++i) {
                    video[i] += pred.video[i] * sigma_delta_video;
                }
                for (size_t i = 0; i < pred.audio.size(); ++i) {
                    audio[i] += pred.audio[i] * sigma_delta_audio;
                }
            } else {
                video.swap(pred.next_video);
                audio.swap(pred.next_audio);
            }
            sampler_update_ms += engine::debug::elapsed_ms(update_start, Clock::now());
            const auto round_start = Clock::now();
            core::round_f32_to_bf16_in_place(video);
            core::round_f32_to_bf16_in_place(audio);
            round_ms += engine::debug::elapsed_ms(round_start, Clock::now());
        }
        positive_input_upload_ms = layerwise_dit->input_upload_ms();
        positive_output_read_ms = layerwise_dit->output_read_ms();
        if (negative_layerwise_dit != nullptr) {
            negative_input_upload_ms = negative_layerwise_dit->input_upload_ms();
            negative_output_read_ms = negative_layerwise_dit->output_read_ms();
        }
    } else {
        std::unique_ptr<MiniMaxH3DitGraph> positive_dit;
        std::unique_ptr<MiniMaxH3DitFirstBlockCacheRuntime> first_block_runtime;
        std::unique_ptr<MiniMaxH3DitFirstBlockCacheRuntime> negative_first_block_runtime;
        std::unique_ptr<MiniMaxH3DitCfgGraph> cfg_dit;
        std::unique_ptr<MiniMaxH3SamplerGraph> sampler;
        MiniMaxH3SamplerOutput sampler_out;
        if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::FirstBlockCache) {
            first_block_runtime = std::make_unique<MiniMaxH3DitFirstBlockCacheRuntime>(*dit_weights, cfg, first_block_request, prompt_out);
            if (cfg_enabled) {
                negative_first_block_runtime = std::make_unique<MiniMaxH3DitFirstBlockCacheRuntime>(*dit_weights, cfg, first_block_request, negative_prompt_out);
            }
        } else if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::Spectrum) {
            if (cfg_enabled) {
                cfg_dit = std::make_unique<MiniMaxH3DitCfgGraph>(*dit_weights, cfg, prompt_out, negative_prompt_out);
            } else {
                positive_dit = std::make_unique<MiniMaxH3DitGraph>(*dit_weights, cfg, prompt_out);
            }
        } else if (cfg_enabled) {
            cfg_dit = std::make_unique<MiniMaxH3DitCfgGraph>(*dit_weights, cfg, prompt_out, negative_prompt_out, false);
            sampler = std::make_unique<MiniMaxH3SamplerGraph>(execution_, sampler_graph_spec(cfg, cfg_dit->layout()), request.sampler);
        } else {
            positive_dit = std::make_unique<MiniMaxH3DitGraph>(*dit_weights, cfg, prompt_out, false);
            sampler = std::make_unique<MiniMaxH3SamplerGraph>(execution_, sampler_graph_spec(cfg, positive_dit->layout()), request.sampler);
        }
        DitGraphResult pred;
        DitGraphResult negative_pred;
        const int64_t timestep_width = cfg.adaln_curve_grid > 0 ? cfg.time_embed_dim : cfg.timestep_input_dim;
        std::unique_ptr<MiniMaxH3SpectrumForecaster> spectrum;
        std::unique_ptr<MiniMaxH3DitFinalGraph> spectrum_final;
        if (request.dit_acceleration == MiniMaxH3DitAccelerationMode::Spectrum) {
            const auto & layout = positive_dit->layout();
            spectrum = std::make_unique<MiniMaxH3SpectrumForecaster>(
                request,
                cfg.denoise_steps,
                static_cast<size_t>(layout.total * cfg.hidden));
            spectrum_final = std::make_unique<MiniMaxH3DitFinalGraph>(*dit_weights, cfg, layout);
        }
        for (int64_t step = 0; step < cfg.denoise_steps; ++step) {
            const size_t timestep_offset = static_cast<size_t>(step * 2 * timestep_width);
            const float next_video = step + 1 == cfg.denoise_steps ? 0.0F : video_sigmas[static_cast<size_t>(step + 1)];
            const float next_audio = step + 1 == cfg.denoise_steps ? 0.0F : audio_sigmas[static_cast<size_t>(step + 1)];
            const float sigma_delta_video = video_sigmas[static_cast<size_t>(step)] - next_video;
            const float sigma_delta_audio = audio_sigmas[static_cast<size_t>(step)] - next_audio;
            const float coordinate = spectrum_coordinate(video_sigmas[static_cast<size_t>(step)], video_sigma_min, video_sigma_max);
            const bool split_audio_timestep =
                std::abs(step_timestep_values[static_cast<size_t>(step * 2)] -
                         step_timestep_values[static_cast<size_t>(step * 2 + 1)]) > 1.0e-6F;
            if (first_block_runtime != nullptr) {
                first_block_runtime->run(
                    audio,
                    video,
                    step_timestep_features.data() + timestep_offset,
                    static_cast<size_t>(2 * timestep_width),
                    video_sigmas[static_cast<size_t>(step)],
                    sigma_delta_audio,
                    sigma_delta_video,
                    split_audio_timestep,
                    pred);
                if (cfg_enabled) {
                    negative_first_block_runtime->run(
                        audio,
                        video,
                        step_timestep_features.data() + timestep_offset,
                        static_cast<size_t>(2 * timestep_width),
                        video_sigmas[static_cast<size_t>(step)],
                        sigma_delta_audio,
                        sigma_delta_video,
                        split_audio_timestep,
                        negative_pred);
                    const auto cfg_start = Clock::now();
                    for (size_t i = 0; i < pred.next_video.size(); ++i) {
                        pred.next_video[i] = negative_pred.next_video[i] + request.guidance_scale * (pred.next_video[i] - negative_pred.next_video[i]);
                    }
                    for (size_t i = 0; i < pred.next_audio.size(); ++i) {
                        pred.next_audio[i] = negative_pred.next_audio[i] + request.guidance_scale * (pred.next_audio[i] - negative_pred.next_audio[i]);
                    }
                    pred.video.resize(pred.next_video.size());
                    pred.audio.resize(pred.next_audio.size());
                    for (size_t i = 0; i < pred.video.size(); ++i) {
                        pred.video[i] = (pred.next_video[i] - video[i]) / sigma_delta_video;
                    }
                    for (size_t i = 0; i < pred.audio.size(); ++i) {
                        pred.audio[i] = (pred.next_audio[i] - audio[i]) / sigma_delta_audio;
                    }
                    sampler_cfg_ms += engine::debug::elapsed_ms(cfg_start, Clock::now());
                }
            } else if (spectrum != nullptr) {
                if (spectrum->should_run_full(step)) {
                    if (cfg_dit != nullptr) {
                        cfg_dit->run(
                            audio,
                            video,
                            step_timestep_features.data() + timestep_offset,
                            static_cast<size_t>(2 * timestep_width),
                            sigma_delta_audio,
                            sigma_delta_video,
                            split_audio_timestep,
                            request.guidance_scale,
                            false,
                            true,
                            true,
                            pred);
                    } else {
                        positive_dit->run(
                            audio,
                            video,
                            step_timestep_features.data() + timestep_offset,
                            static_cast<size_t>(2 * timestep_width),
                            sigma_delta_audio,
                            sigma_delta_video,
                            split_audio_timestep,
                            false,
                            true,
                            true,
                            pred);
                    }
                    const auto spectrum_start = Clock::now();
                    spectrum->update(coordinate, pred.hidden);
                    engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.actual_step", 1.0);
                    engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.state_ms", engine::debug::elapsed_ms(spectrum_start, Clock::now()));
                } else {
                    const auto spectrum_start = Clock::now();
                    spectrum->predict(
                        coordinate,
                        cfg.hidden,
                        positive_dit->layout().audio_start,
                        positive_dit->layout().audio_rows,
                        positive_dit->layout().video_start,
                        positive_dit->layout().video_rows,
                        pred.hidden);
                    spectrum_final->run(
                        pred.hidden,
                        audio,
                        video,
                        step_timestep_features.data() + timestep_offset,
                        static_cast<size_t>(2 * timestep_width),
                        sigma_delta_audio,
                        sigma_delta_video,
                        split_audio_timestep,
                        pred);
                    engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.actual_step", 0.0);
                    engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.state_ms", engine::debug::elapsed_ms(spectrum_start, Clock::now()));
                }
            } else if (cfg_dit != nullptr) {
                cfg_dit->run(
                    audio,
                    video,
                    step_timestep_features.data() + timestep_offset,
                    static_cast<size_t>(2 * timestep_width),
                    sigma_delta_audio,
                    sigma_delta_video,
                    split_audio_timestep,
                    request.guidance_scale,
                    step == 0,
                    false,
                    false,
                    pred);
                sampler->run(
                    MiniMaxH3SamplerInput{
                        cfg_dit->video_state_tensor(),
                        cfg_dit->audio_state_tensor(),
                        cfg_dit->video_logits_tensor(),
                        cfg_dit->audio_logits_tensor(),
                        video_sigmas[static_cast<size_t>(step)],
                        audio_sigmas[static_cast<size_t>(step)],
                        sigma_delta_video,
                        sigma_delta_audio,
                    },
                    sampler_out);
                pred.next_video.swap(sampler_out.next_primary);
                pred.next_audio.swap(sampler_out.next_secondary);
            } else {
                positive_dit->run(
                    audio,
                    video,
                    step_timestep_features.data() + timestep_offset,
                    static_cast<size_t>(2 * timestep_width),
                    sigma_delta_audio,
                    sigma_delta_video,
                    split_audio_timestep,
                    false,
                    false,
                    false,
                    pred);
                sampler->run(
                    MiniMaxH3SamplerInput{
                        positive_dit->video_state_tensor(),
                        positive_dit->audio_state_tensor(),
                        positive_dit->video_logits_tensor(),
                        positive_dit->audio_logits_tensor(),
                        video_sigmas[static_cast<size_t>(step)],
                        audio_sigmas[static_cast<size_t>(step)],
                        sigma_delta_video,
                        sigma_delta_audio,
                    },
                    sampler_out);
                pred.next_video.swap(sampler_out.next_primary);
                pred.next_audio.swap(sampler_out.next_secondary);
            }
            if (step == 0 && cfg_enabled) {
                engine::debug::trace_log_f32("minimax_h3.dit_step0_video", {cfg.video_patches, cfg.video_latents_dim * 4}, pred.video);
                engine::debug::trace_log_f32("minimax_h3.dit_step0_audio", {cfg.audio_channels * cfg.audio_steps, cfg.audio_latents_dim}, pred.audio);
            }
            const auto update_start = Clock::now();
            video.swap(pred.next_video);
            audio.swap(pred.next_audio);
            sampler_update_ms += engine::debug::elapsed_ms(update_start, Clock::now());
            const auto round_start = Clock::now();
            core::round_f32_to_bf16_in_place(video);
            core::round_f32_to_bf16_in_place(audio);
            round_ms += engine::debug::elapsed_ms(round_start, Clock::now());
        }
        if (positive_dit != nullptr) {
            positive_input_upload_ms = positive_dit->input_upload_ms();
            positive_output_read_ms = positive_dit->output_read_ms();
        }
        if (first_block_runtime != nullptr) {
            positive_input_upload_ms = first_block_runtime->input_upload_ms();
            positive_output_read_ms = first_block_runtime->output_read_ms();
            if (negative_first_block_runtime != nullptr) {
                negative_input_upload_ms = negative_first_block_runtime->input_upload_ms();
                negative_output_read_ms = negative_first_block_runtime->output_read_ms();
            }
        }
        if (cfg_dit != nullptr) {
            positive_input_upload_ms = cfg_dit->input_upload_ms();
            positive_output_read_ms = cfg_dit->output_read_ms();
        }
        if (sampler != nullptr) {
            engine::debug::timing_log_scalar("minimax_h3.sampler.device_copy_ms", sampler->device_copy_ms());
            engine::debug::timing_log_scalar("minimax_h3.sampler.graph_compute_total_ms", sampler->graph_compute_ms());
            engine::debug::timing_log_scalar("minimax_h3.sampler.history_update_ms", sampler->history_update_ms());
            engine::debug::timing_log_scalar("minimax_h3.sampler.output_read_ms", sampler->output_read_ms());
        }
        if (first_block_runtime != nullptr) {
            engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.full_steps", first_block_runtime->full_steps());
            engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.cached_steps", first_block_runtime->cached_steps());
            if (negative_first_block_runtime != nullptr) {
                engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.negative.full_steps", negative_first_block_runtime->full_steps());
                engine::debug::timing_log_scalar("minimax_h3.dit.first_block_cache.negative.cached_steps", negative_first_block_runtime->cached_steps());
            }
        }
        if (spectrum != nullptr) {
            engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.full_steps", spectrum->full_steps());
            engine::debug::timing_log_scalar("minimax_h3.dit.spectrum.forecast_steps", spectrum->forecast_steps());
        }
    }
    engine::debug::timing_log_scalar("minimax_h3.dit.input_upload_ms", positive_input_upload_ms + negative_input_upload_ms);
    engine::debug::timing_log_scalar("minimax_h3.dit.output_read_ms", positive_output_read_ms + negative_output_read_ms);
    engine::debug::timing_log_scalar("minimax_h3.sampler_cfg_ms", sampler_cfg_ms);
    engine::debug::timing_log_scalar("minimax_h3.sampler_update_ms", sampler_update_ms);
    engine::debug::timing_log_scalar("minimax_h3.bf16_round_ms", round_ms);
    engine::debug::trace_log_f32("minimax_h3.denoise_audio_rows", {cfg.audio_channels * cfg.audio_steps, cfg.audio_latents_dim}, audio);
    engine::debug::timing_log_scalar("minimax_h3.denoise_ms", engine::debug::elapsed_ms(denoise_start, Clock::now()));

    const auto decode_start = Clock::now();
    std::vector<float> wav;
    std::unique_ptr<AudioVaeWeightStore> scoped_audio_weights;
    AudioVaeWeightStore * audio_weights = impl_->audio_vae_weights.get();
    if (impl_->mem_saver) {
        const auto audio_weight_start = Clock::now();
        scoped_audio_weights = std::make_unique<AudioVaeWeightStore>(execution_, assets_->audio_vae_weights, cfg, impl_->weight_context_bytes);
        engine::debug::timing_log_scalar("minimax_h3.audio_vae.weights_load_ms", engine::debug::elapsed_ms(audio_weight_start, Clock::now()));
        audio_weights = scoped_audio_weights.get();
    } else if (audio_weights == nullptr) {
        const auto audio_weight_start = Clock::now();
        impl_->audio_vae_weights = std::make_unique<AudioVaeWeightStore>(execution_, assets_->audio_vae_weights, cfg, impl_->weight_context_bytes);
        engine::debug::timing_log_scalar("minimax_h3.audio_vae.weights_load_ms", engine::debug::elapsed_ms(audio_weight_start, Clock::now()));
        audio_weights = impl_->audio_vae_weights.get();
    } else {
        engine::debug::timing_log_scalar("minimax_h3.audio_vae.weights_load_ms", 0.0);
    }
    wav = run_audio_vae_decode_graph(*audio_weights, cfg, audio);
    engine::debug::timing_log_scalar("minimax_h3.audio_decode_ms", engine::debug::elapsed_ms(decode_start, Clock::now()));
    std::optional<MiniMaxH3VideoFrames> video_frames;
    if (request.return_video) {
        const auto video_decode_start = Clock::now();
        std::unique_ptr<VideoVaeWeightStore> scoped_video_weights;
        VideoVaeWeightStore * video_weights = impl_->video_vae_weights.get();
        if (impl_->mem_saver || video_weights == nullptr) {
            const auto video_weight_start = Clock::now();
            scoped_video_weights = std::make_unique<VideoVaeWeightStore>(execution_, assets_->video_vae_weights, cfg, impl_->weight_context_bytes);
            engine::debug::timing_log_scalar("minimax_h3.video_vae.weights_load_ms", engine::debug::elapsed_ms(video_weight_start, Clock::now()));
            video_weights = scoped_video_weights.get();
            if (!impl_->mem_saver) {
                impl_->video_vae_weights = std::move(scoped_video_weights);
                video_weights = impl_->video_vae_weights.get();
            }
        } else {
            engine::debug::timing_log_scalar("minimax_h3.video_vae.weights_load_ms", 0.0);
        }
        if (impl_->mem_saver) {
            VideoVaeDecodeCache scoped_video_decode_cache;
            video_frames = run_video_vae_decode_graph(*video_weights, cfg, video, scoped_video_decode_cache);
        } else {
            video_frames = run_video_vae_decode_graph(*video_weights, cfg, video, impl_->video_vae_decode_cache);
        }
        engine::debug::timing_log_scalar("minimax_h3.video_decode_total_ms", engine::debug::elapsed_ms(video_decode_start, Clock::now()));
    }
    engine::debug::timing_log_scalar("minimax_h3.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return {cfg.sample_rate, static_cast<int>(cfg.audio_channels), std::move(wav), std::move(video_frames)};
}

}  // namespace engine::models::minimax_h3
