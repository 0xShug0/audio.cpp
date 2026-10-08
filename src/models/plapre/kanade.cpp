#include "engine/models/plapre/kanade.h"

#include "engine/framework/core/attention_fallback.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/convnext_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::plapre {
namespace {

struct KanadeTransformerLayer {
    modules::LinearWeights q, k, v, out;
    modules::LinearWeights gate, up, down;
    modules::NormWeights attention_norm, ffn_norm;
    modules::LinearWeights attention_condition, ffn_condition;
};

struct KanadeTransformerStack {
    KanadeTransformerConfig config;
    int64_t output_dim;
    bool conditioned;
    bool bf16;
    std::vector<KanadeTransformerLayer> layers;
    modules::NormWeights final_norm;
    modules::LinearWeights final_condition, output_projection;

    KanadeTransformerStack(core::BackendWeightStore & store, const assets::TensorSource & source,
        KanadeTransformerConfig config_in, const std::string & prefix, bool conditioned_in, bool bf16_in,
        int64_t output_dim_in = 0)
        : config(std::move(config_in)), output_dim(output_dim_in), conditioned(conditioned_in), bf16(bf16_in) {
        const auto linear = [&](const std::string & name, int64_t out, int64_t in, bool bias) {
            auto weights = modules::binding::linear_from_source(store, source, name,
                bf16 ? assets::TensorStorageType::BF16 : assets::TensorStorageType::F32, out, in, false);
            if (bias) {
                auto values = source.require_f32(name + ".bias", {out});
                if (bf16) core::round_f32_to_bf16_in_place(values);
                weights.bias = store.make_f32(core::TensorShape::from_dims({out}), std::move(values));
            }
            return weights;
        };
        for (int64_t i = 0; i < config.layers; ++i) {
            const auto p = prefix + ".layers." + std::to_string(i);
            KanadeTransformerLayer layer;
            layer.q = linear(p + ".attention.wq", config.hidden, config.hidden, false);
            layer.k = linear(p + ".attention.wk", config.hidden, config.hidden, false);
            layer.v = linear(p + ".attention.wv", config.hidden, config.hidden, false);
            layer.out = linear(p + ".attention.wo", config.hidden, config.hidden, false);
            layer.gate = linear(p + ".feed_forward.w1", config.intermediate, config.hidden, false);
            layer.up = linear(p + ".feed_forward.w3", config.intermediate, config.hidden, false);
            layer.down = linear(p + ".feed_forward.w2", config.hidden, config.intermediate, false);
            if (conditioned) {
                layer.attention_condition = linear(p + ".attention_norm.condition_proj.1",
                    3 * config.hidden, config.speaker_dim, true);
                layer.ffn_condition = linear(p + ".ffn_norm.condition_proj.1",
                    3 * config.hidden, config.speaker_dim, true);
            } else {
                layer.attention_norm = modules::binding::norm_from_named_source(store, source,
                    p + ".attention_norm.weight", p + ".attention_norm.bias");
                layer.ffn_norm = modules::binding::norm_from_named_source(store, source,
                    p + ".ffn_norm.weight", p + ".ffn_norm.bias");
            }
            layers.push_back(std::move(layer));
        }
        if (conditioned) {
            final_condition = linear(prefix + ".norm.condition_proj.1", 2 * config.hidden, config.speaker_dim, true);
            output_projection = linear(prefix + ".output_proj", output_dim, config.hidden, true);
        } else {
            final_norm = modules::binding::norm_from_named_source(store, source,
                prefix + ".norm.weight", prefix + ".norm.bias");
        }
    }

    core::TensorValue build(core::ModuleBuildContext & ctx, core::TensorValue input,
        const core::TensorValue & positions, const core::TensorValue & mask,
        const core::TensorValue & speaker, bool flash) const {
        const auto round = [&](core::TensorValue x) {
            return bf16 ? modules::Bf16RoundModule().build(ctx, x) : x;
        };
        const auto project = [&](core::TensorValue x, const modules::LinearWeights & weights) {
            const auto in = weights.weight.shape.at(1);
            const auto out = weights.weight.shape.at(0);
            return round(modules::LinearModule({in, out, weights.bias.has_value()}).build(ctx, round(x), weights));
        };
        const auto speaker_condition = conditioned ? modules::SiluModule().build(ctx, speaker) : core::TensorValue{};
        const auto modulate = [&](core::TensorValue x, const modules::LinearWeights & weights,
                                  core::TensorValue * gate) {
            x = modules::LayerNormModule({config.hidden, config.norm_eps, false, false}).build(ctx, x, {});
            const auto params = project(speaker_condition, weights);
            auto shift = modules::SliceModule({2, 0, config.hidden}).build(ctx, params);
            auto scale = modules::SliceModule({2, config.hidden, config.hidden}).build(ctx, params);
            scale = round(modules::ScaleBiasModule({1.0f, 1.0f}).build(ctx, scale));
            x = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, scale));
            x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, shift));
            if (gate) {
                *gate = modules::SliceModule({2, 2 * config.hidden, config.hidden}).build(ctx, params);
            }
            return x;
        };
        const int64_t frames = input.shape.at(1);
        const int64_t dim = config.hidden / config.heads;
        const auto heads_shape = core::TensorShape::from_dims({1, frames, config.heads, dim});
        const modules::RoPEModule rope({dim, GGML_ROPE_TYPE_NORMAL, config.rope_theta});
        const modules::LayerNormModule norm({config.hidden, config.norm_eps, true, true});
        for (const auto & layer : layers) {
            core::TensorValue gate;
            auto h = conditioned ? modulate(input, layer.attention_condition, &gate) : norm.build(ctx, input, layer.attention_norm);
            auto q = core::reshape_tensor(ctx, project(h, layer.q), heads_shape);
            auto k = core::reshape_tensor(ctx, project(h, layer.k), heads_shape);
            auto v = core::reshape_tensor(ctx, project(h, layer.v), heads_shape);
            q = round(rope.build(ctx, q, positions));
            k = round(rope.build(ctx, k, positions));
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            h = modules::ScaledDotProductAttentionModule({dim, flash
                ? modules::ScaledDotProductAttentionLowering::Flash
                : modules::ScaledDotProductAttentionLowering::Explicit}).build(ctx, q, k, v, mask);
            h = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, h), input.shape);
            h = project(h, layer.out);
            if (conditioned) h = round(modules::MulModule().build(ctx, h, modules::RepeatModule({h.shape}).build(ctx, gate)));
            input = modules::AddModule().build(ctx, input, h);
            if (conditioned) input = round(input);
            h = conditioned ? modulate(input, layer.ffn_condition, &gate) : norm.build(ctx, input, layer.ffn_norm);
            auto activated = round(modules::SiluModule().build(ctx, project(h, layer.gate)));
            h = round(modules::MulModule().build(ctx, activated, project(h, layer.up)));
            h = project(h, layer.down);
            if (conditioned) h = round(modules::MulModule().build(ctx, h, modules::RepeatModule({h.shape}).build(ctx, gate)));
            input = modules::AddModule().build(ctx, input, h);
            if (conditioned) input = round(input);
        }
        return conditioned ? project(modulate(input, final_condition, nullptr), output_projection)
                           : norm.build(ctx, input, final_norm);
    }
};

}  // namespace

struct KanadeAudioEncoder::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source,
        std::shared_ptr<const assets::TensorSource> wavlm_source,
        KanadeEncoderConfig config, core::ExecutionContext & execution)
        : source_(std::move(source)), config_(std::move(config)), execution_(execution),
          bf16_(execution.backend_type() == core::BackendType::Cuda),
          store_(execution.backend(), execution.backend_type(), "plapre.kanade.encoder.weights", 4 * 1024 * 1024),
          local_(store_, *source_, config_.local, "local_encoder", false, bf16_),
          wavlm_(modules::WavlmEncoderComponent::load_from_tensor_source(*wavlm_source, execution.config(), config_.wavlm)) {
        const auto storage = bf16_ ? assets::TensorStorageType::BF16 : assets::TensorStorageType::F32;
        const auto conv = [&](const std::string & name, int64_t out, int64_t in, int64_t kernel) {
            auto weights = modules::binding::conv1d_from_source(store_, *source_, name, storage, out, in, kernel, true);
            if (bf16_) {
                auto bias = source_->require_f32(name + ".bias", {out});
                core::round_f32_to_bf16_in_place(bias);
                weights.bias = store_.make_f32(core::TensorShape::from_dims({out}), std::move(bias));
            }
            return weights;
        };
        const auto linear = [&](const std::string & name, int64_t out, int64_t in) {
            auto weights = modules::binding::linear_from_source(store_, *source_, name, storage, out, in, false);
            auto bias = source_->require_f32(name + ".bias", {out});
            if (bf16_) core::round_f32_to_bf16_in_place(bias);
            weights.bias = store_.make_f32(core::TensorShape::from_dims({out}), std::move(bias));
            return weights;
        };
        const auto norm = [&](const std::string & name) {
            return modules::binding::norm_from_named_source(store_, *source_, name + ".weight", name + ".bias");
        };
        const auto hidden = config_.local.hidden;
        downsample_ = conv("conv_downsample", hidden, hidden, config_.downsample_factor);
        quantizer_in_ = linear("local_quantizer.proj_in", static_cast<int64_t>(config_.levels.size()), hidden);
        const auto channels = config_.global_channels;
        global_embed_ = conv("global_encoder.backbone.embed", channels, hidden, 7);
        global_norm_ = norm("global_encoder.backbone.norm");
        global_final_norm_ = norm("global_encoder.backbone.final_layer_norm");
        for (int64_t i = 0; i < config_.global_layers; ++i) {
            const auto p = "global_encoder.backbone.convnext." + std::to_string(i);
            modules::ConvNeXt1dWeights block;
            block.depthwise = modules::binding::depthwise_conv1d_from_source(store_, *source_,
                p + ".dwconv", storage, channels, 7, true);
            if (bf16_) {
                auto bias = source_->require_f32(p + ".dwconv.bias", {channels});
                core::round_f32_to_bf16_in_place(bias);
                block.depthwise.bias = store_.make_f32(core::TensorShape::from_dims({channels}), std::move(bias));
            }
            block.norm = norm(p + ".norm");
            block.expansion = linear(p + ".pwconv1", config_.global_intermediate, channels);
            block.projection = linear(p + ".pwconv2", channels, config_.global_intermediate);
            block.gamma = store_.load_f32_tensor(*source_, p + ".gamma", {channels});
            blocks_.push_back(std::move(block));
        }
        attention_in_ = conv("global_encoder.pooling.attn.0", config_.attention_channels, channels, 1);
        attention_out_ = conv("global_encoder.pooling.attn.2", channels, config_.attention_channels, 1);
        pool_projection_ = linear("global_encoder.pooling.proj", config_.local.speaker_dim, channels * 2);
        pool_norm_ = norm("global_encoder.pooling.norm");
        store_.upload();
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);
        graph_ = nullptr;
        plan_.reset();
        if (allocator_) ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
        if (context_) ggml_free(context_);
        context_ = nullptr;
    }

    void build(int64_t frames) {
        release_graph();
        const auto started = std::chrono::steady_clock::now();
        const auto & c = config_;
        const auto hidden = c.local.hidden;
        const auto channels = c.global_channels;
        context_ = ggml_init({4 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{context_, "plapre.kanade.encoder", execution_.backend_type()};
        const auto round = [&](core::TensorValue value) {
            if (!bf16_) return value;
            return modules::Bf16RoundModule().build(ctx, value);
        };
        const auto linear = [&](core::TensorValue x, const modules::LinearWeights & weights) {
            return round(modules::LinearModule({weights.weight.shape.at(1), weights.weight.shape.at(0), true})
                .build(ctx, round(x), weights));
        };
        const auto transpose = [&](core::TensorValue x) {
            return modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        };
        local_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, hidden}));
        global_input_ = core::make_tensor(ctx, GGML_TYPE_F32, local_input_.shape);
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({frames}));
        mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({frames, frames}));
        for (const auto & input : {local_input_, global_input_, positions_, mask_}) ggml_set_input(input.tensor);
        const bool flash = execution_.backend_type() != core::BackendType::Cpu && core::resolve_flash_attention(
            execution_.backend(), hidden / c.local.heads, core::AttentionPreference::Auto);
        auto x = local_.build(ctx, local_input_, positions_, mask_, {}, flash);
        x = transpose(x);
        x = modules::Conv1dModule({hidden, hidden, c.downsample_factor, static_cast<int>(c.downsample_factor),
            0, 1, true, bf16_}).build(ctx, x, downsample_);
        local_output_ = linear(transpose(x), quantizer_in_);
        ggml_set_output(local_output_.tensor);

        x = modules::Conv1dModule({hidden, channels, 7, 1, 3, 1, true, bf16_})
            .build(ctx, transpose(global_input_), global_embed_);
        x = transpose(x);
        x = modules::LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, global_norm_);
        for (const auto & block : blocks_) {
            x = modules::build_convnext1d(ctx, x, block, bf16_);
        }
        x = modules::LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, global_final_norm_);
        x = transpose(x);
        auto alpha = modules::Conv1dModule({channels, c.attention_channels, 1, 1, 0, 1, true, bf16_})
            .build(ctx, x, attention_in_);
        alpha = round(modules::TanhModule().build(ctx, alpha));
        alpha = modules::Conv1dModule({c.attention_channels, channels, 1, 1, 0, 1, true, bf16_})
            .build(ctx, alpha, attention_out_);
        alpha = modules::SoftmaxModule().build(ctx, alpha);
        const auto moments = modules::WeightedStatisticsModule({2, 1e-4f, 1e4f}).build(ctx, x, alpha);
        auto stats = modules::ConcatModule({1}).build(ctx, moments.mean, moments.stddev);
        stats = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, stats),
            core::TensorShape::from_dims({1, 1, channels * 2}));
        speaker_output_ = modules::LayerNormModule({c.local.speaker_dim, 1e-5f, true, true})
            .build(ctx, linear(stats, pool_projection_), pool_norm_);
        ggml_set_output(speaker_output_.tensor);
        graph_ = ggml_new_graph_custom(context_, 8192, false);
        ggml_build_forward_expand(graph_, local_output_.tensor);
        ggml_build_forward_expand(graph_, speaker_output_.tensor);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("Kanade encoder graph allocation failed");
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        frames_ = frames;
        debug::timing_log_context_reservation("plapre.kanade.encoder.graph", context_);
        debug::timing_log_scalar("plapre.kanade.encoder.graph.build_ms", debug::elapsed_ms(started));
    }

    KanadeAudioFeatures encode(const std::vector<float> & audio) {
        if (audio.empty()) throw std::runtime_error("Plapre reference audio is empty");
        const auto started = std::chrono::steady_clock::now();
        // Plapre passes channel-first mono to Kanade; its current framing pads 300 samples per side.
        std::vector<float> padded(audio.size() + 600, 0.0f);
        std::copy(audio.begin(), audio.end(), padded.begin() + 300);
        audio::TorchaudioSincHannResampleOptions resampling;
        resampling.accumulation = audio::TorchaudioSincHannAccumulation::Float32;
        auto samples = audio::resample_mono_torchaudio_sinc_hann(padded, 24000, 16000, resampling);
        auto layers = config_.local_ssl_layers;
        layers.insert(layers.end(), config_.global_ssl_layers.begin(), config_.global_ssl_layers.end());
        std::sort(layers.begin(), layers.end());
        layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
        const auto ssl = wavlm_.encode_layers(samples, 1, static_cast<int64_t>(samples.size()), layers);
        if (!graph_ || frames_ != ssl.tokens) build(ssl.tokens);
        const auto hidden = config_.local.hidden;
        std::vector<float> local(static_cast<size_t>(ssl.tokens * hidden), 0.0f), global(local.size(), 0.0f);
        for (size_t i = 0; i < layers.size(); ++i) {
            if (std::find(config_.local_ssl_layers.begin(), config_.local_ssl_layers.end(), layers[i]) != config_.local_ssl_layers.end()) {
                for (size_t j = 0; j < local.size(); ++j) local[j] += ssl.hidden_states[i][j] / config_.local_ssl_layers.size();
            }
            if (std::find(config_.global_ssl_layers.begin(), config_.global_ssl_layers.end(), layers[i]) != config_.global_ssl_layers.end()) {
                for (size_t j = 0; j < global.size(); ++j) global[j] += ssl.hidden_states[i][j] / config_.global_ssl_layers.size();
            }
        }
        if (config_.normalize_ssl_features) {
            for (int64_t c = 0; c < hidden; ++c) {
                double sum = 0, variance = 0;
                for (int64_t t = 0; t < ssl.tokens; ++t) sum += local[t * hidden + c];
                const auto mean = static_cast<float>(sum / ssl.tokens);
                for (int64_t t = 0; t < ssl.tokens; ++t) {
                    const double diff = local[t * hidden + c] - mean;
                    variance += diff * diff;
                }
                const auto stddev = static_cast<float>(std::sqrt(variance / (ssl.tokens - 1))) + 1e-8f;
                for (int64_t t = 0; t < ssl.tokens; ++t) local[t * hidden + c] = (local[t * hidden + c] - mean) / stddev;
            }
        }
        core::write_tensor_f32(local_input_, local);
        core::write_tensor_f32(global_input_, global);
        std::vector<int32_t> positions(static_cast<size_t>(frames_));
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(positions_.tensor, positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(frames_ * frames_),
            ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        for (int64_t q = 0; q < frames_; ++q) {
            for (int64_t k = std::max<int64_t>(0, q - config_.local.window_size / 2);
                k < std::min(frames_, q + config_.local.window_size / 2 + 1); ++k) mask[q * frames_ + k] = ggml_fp32_to_fp16(0.f);
        }
        ggml_backend_tensor_set(mask_.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        if (core::compute_graph(execution_, graph_, plan_, "plapre.kanade.encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Kanade encoder execution failed");
        }
        const auto latent = core::read_tensor_f32(local_output_.tensor);
        KanadeAudioFeatures out;
        out.speaker = core::read_tensor_f32(speaker_output_.tensor);
        const auto dimensions = config_.levels.size();
        out.codes.resize(latent.size() / dimensions);
        for (size_t t = 0; t < out.codes.size(); ++t) {
            int64_t code = 0, basis = 1;
            for (size_t d = 0; d < dimensions; ++d) {
                const auto level = config_.levels[d];
                const float half = static_cast<float>(level - 1) * 0.999f / 2.f;
                const float offset = level % 2 == 0 ? 0.5f : 0.f;
                const float bounded = std::tanh(latent[t * dimensions + d] + std::tan(offset / half)) * half - offset;
                code += (static_cast<int64_t>(std::nearbyint(bounded)) + level / 2) * basis;
                basis *= level;
            }
            out.codes[t] = static_cast<int32_t>(code);
        }
        debug::timing_log_scalar("plapre.kanade.encoder.wall_ms", debug::elapsed_ms(started));
        return out;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    KanadeEncoderConfig config_;
    core::ExecutionContext & execution_;
    bool bf16_;
    core::BackendWeightStore store_;
    KanadeTransformerStack local_;
    modules::WavlmEncoderComponent wavlm_;
    modules::Conv1dWeights downsample_, global_embed_, attention_in_, attention_out_;
    modules::LinearWeights quantizer_in_, pool_projection_;
    modules::NormWeights global_norm_, global_final_norm_, pool_norm_;
    std::vector<modules::ConvNeXt1dWeights> blocks_;
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue local_input_, global_input_, positions_, mask_, local_output_, speaker_output_;
    int64_t frames_ = 0;
};

KanadeAudioEncoder::KanadeAudioEncoder(std::shared_ptr<const assets::TensorSource> source,
    std::shared_ptr<const assets::TensorSource> wavlm_source,
    KanadeEncoderConfig config, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(wavlm_source), std::move(config), execution)) {}
KanadeAudioEncoder::~KanadeAudioEncoder() = default;
KanadeAudioFeatures KanadeAudioEncoder::encode(const std::vector<float> & audio) {
    return impl_->encode(audio);
}

struct KanadeMelDecoder::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source, KanadeMelConfig config,
        core::ExecutionContext & execution)
        : source_(std::move(source)), config_(std::move(config)), execution_(execution),
          bf16_(config_.bf16_autocast && execution.backend_type() == core::BackendType::Cuda),
          flash_(execution.backend_type() != core::BackendType::Cpu &&
              (execution.backend_type() != core::BackendType::Cuda || bf16_) && core::resolve_flash_attention(
              execution.backend(), config_.hidden / config_.heads, core::AttentionPreference::Auto)),
          store_(execution.backend(), execution.backend_type(), "plapre.kanade.mel.weights", 4 * 1024 * 1024),
          prenet_(store_, *source_, config_, "mel_prenet", false, bf16_),
          decoder_(store_, *source_, config_, "mel_decoder", true, bf16_, config_.mel_bins) {
        quantizer_out_ = modules::binding::linear_from_source(store_, *source_, "local_quantizer.proj_out",
            assets::TensorStorageType::F32, config_.hidden, static_cast<int64_t>(config_.levels.size()), true);
        // The transposed convolution has no overlapping taps (kernel == stride).
        upsample_ = modules::binding::conv_transpose1d_from_source(store_, *source_, "mel_conv_upsample",
            bf16_ ? assets::TensorStorageType::BF16 : assets::TensorStorageType::F32,
            config_.hidden, config_.hidden, config_.upsample_factor, false);
        auto upsample_bias = source_->require_f32("mel_conv_upsample.bias", {config_.hidden});
        if (bf16_) core::round_f32_to_bf16_in_place(upsample_bias);
        upsample_.bias = store_.make_f32(core::TensorShape::from_dims({config_.hidden}), std::move(upsample_bias));
        for (int64_t i = 0; i < config_.postnet_layers; ++i) {
            const auto p = "mel_postnet.convolutions." + std::to_string(i);
            const auto in = i == 0 ? config_.mel_bins : config_.postnet_channels;
            const auto out = i + 1 == config_.postnet_layers ? config_.mel_bins : config_.postnet_channels;
            postnet_.push_back(modules::binding::conv1d_from_source(store_, *source_, p + ".0",
                bf16_ ? assets::TensorStorageType::BF16 : assets::TensorStorageType::F32,
                out, in, config_.postnet_kernel, true));
            postnet_norm_.push_back(modules::binding::norm_from_named_source(store_, *source_,
                p + ".1.norm.weight", p + ".1.norm.bias"));
        }
        store_.upload();
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);
        graph_ = nullptr;
        plan_.reset();
        if (allocator_) ggml_gallocr_free(allocator_);
        allocator_ = nullptr;
        if (context_) ggml_free(context_);
        context_ = nullptr;
    }

    void build(int64_t frames) {
        release_graph();
        const auto started = std::chrono::steady_clock::now();
        const auto & c = config_;
        const auto mel_frames = frames * c.upsample_factor;
        context_ = ggml_init({4 * 1024 * 1024, nullptr, true});
        core::ModuleBuildContext ctx{context_, "plapre.kanade.mel", execution_.backend_type()};
        code_input_ = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, frames, static_cast<int64_t>(c.levels.size())}));
        speaker_input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, c.speaker_dim}));
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({mel_frames}));
        short_mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({frames, frames}));
        long_mask_ = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({mel_frames, mel_frames}));
        for (const auto & input : {code_input_, speaker_input_, positions_, short_mask_, long_mask_}) ggml_set_input(input.tensor);
        // These inputs are uploaded once; preserve their storage between decodes.
        for (const auto & constant : {positions_, short_mask_, long_mask_}) ggml_set_output(constant.tensor);
        const auto short_positions = modules::SliceModule({0, 0, frames}).build(ctx, positions_);
        auto x = modules::LinearModule({static_cast<int64_t>(c.levels.size()), c.hidden, true})
            .build(ctx, code_input_, quantizer_out_);
        x = prenet_.build(ctx, x, short_positions, short_mask_, speaker_input_, flash_);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        if (bf16_) x = modules::Bf16RoundModule().build(ctx, x);
        x = modules::ConvTranspose1dModule({c.hidden, c.hidden, c.upsample_factor,
            static_cast<int>(c.upsample_factor), 0, 1, false}).build(ctx, x, upsample_);
        if (bf16_) x = modules::Bf16RoundModule().build(ctx, x);
        const auto bias = core::reshape_tensor(ctx, *upsample_.bias, core::TensorShape::from_dims({1, c.hidden, 1}));
        x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, bias));
        if (bf16_) x = modules::Bf16RoundModule().build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = decoder_.build(ctx, x, positions_, long_mask_, speaker_input_, flash_);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        const auto residual = x;
        for (size_t i = 0; i < postnet_.size(); ++i) {
            const auto in = i == 0 ? c.mel_bins : c.postnet_channels;
            const auto out = i + 1 == postnet_.size() ? c.mel_bins : c.postnet_channels;
            x = modules::Conv1dModule({in, out, c.postnet_kernel, 1,
                static_cast<int>(c.postnet_kernel / 2), 1, true, bf16_}).build(ctx, x, postnet_[i]);
            x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
            x = modules::LayerNormModule({out, c.norm_eps, true, true}).build(ctx, x, postnet_norm_[i]);
            if (i + 1 < postnet_.size()) x = modules::TanhModule().build(ctx, x);
            x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        }
        output_ = modules::AddModule().build(ctx, residual, x);
        output_ = core::ensure_backend_addressable_layout(ctx, output_);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_, 8192, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("Kanade mel graph allocation failed");
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<int32_t> positions(static_cast<size_t>(mel_frames));
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(positions_.tensor, positions.data(), 0, positions.size() * sizeof(int32_t));
        for (const auto & mask : {short_mask_, long_mask_}) {
            const auto length = mask.shape.at(0);
            std::vector<ggml_fp16_t> values(static_cast<size_t>(length * length), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int64_t q = 0; q < length; ++q) {
                for (int64_t k = std::max<int64_t>(0, q - c.window_size / 2);
                    k < std::min(length, q + c.window_size / 2 + 1); ++k) values[q * length + k] = ggml_fp32_to_fp16(0.0f);
            }
            ggml_backend_tensor_set(mask.tensor, values.data(), 0, values.size() * sizeof(ggml_fp16_t));
        }
        frames_ = frames;
        debug::timing_log_scalar("plapre.kanade.mel.graph.build_ms", debug::elapsed_ms(started));
    }

    std::vector<float> decode(const std::vector<int32_t> & codes, const std::vector<float> & speaker) {
        const auto started = std::chrono::steady_clock::now();
        if (codes.empty() || speaker.size() != static_cast<size_t>(config_.speaker_dim)) {
            throw std::runtime_error("Kanade decoding requires audio codes and a speaker embedding");
        }
        if (!graph_ || frames_ != static_cast<int64_t>(codes.size())) build(static_cast<int64_t>(codes.size()));
        std::vector<float> values;
        values.reserve(codes.size() * config_.levels.size());
        for (auto code : codes) {
            if (code < 0) throw std::runtime_error("Kanade audio code is outside the codebook");
            for (const auto level : config_.levels) {
                // Kanade centers on floor(level / 2), not endpoints [-1, 1].
                const auto half = level / 2;
                values.push_back(static_cast<float>(code % level - half) / static_cast<float>(half));
                code /= level;
            }
            if (code != 0) throw std::runtime_error("Kanade audio code is outside the codebook");
        }
        core::write_tensor_f32(code_input_, values);
        core::write_tensor_f32(speaker_input_, speaker);
        if (core::compute_graph(execution_, graph_, plan_, "plapre.kanade.mel") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Kanade mel graph execution failed");
        }
        auto output = core::read_tensor_f32(output_.tensor);
        debug::timing_log_scalar("plapre.kanade.mel.wall_ms", debug::elapsed_ms(started));
        return output;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    KanadeMelConfig config_;
    core::ExecutionContext & execution_;
    bool bf16_, flash_;
    core::BackendWeightStore store_;
    KanadeTransformerStack prenet_, decoder_;
    modules::LinearWeights quantizer_out_;
    modules::ConvTranspose1dWeights upsample_;
    std::vector<modules::Conv1dWeights> postnet_;
    std::vector<modules::NormWeights> postnet_norm_;
    ggml_context * context_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    core::HostGraphPlan plan_;
    core::TensorValue code_input_, speaker_input_, positions_, short_mask_, long_mask_, output_;
    int64_t frames_ = 0;
};

KanadeMelDecoder::KanadeMelDecoder(std::shared_ptr<const assets::TensorSource> source,
    KanadeMelConfig config, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), std::move(config), execution)) {}
KanadeMelDecoder::~KanadeMelDecoder() = default;
std::vector<float> KanadeMelDecoder::decode(const std::vector<int32_t> & codes, const std::vector<float> & speaker) {
    return impl_->decode(codes, speaker);
}

}  // namespace engine::models::plapre
