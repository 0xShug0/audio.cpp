#include "engine/framework/conditioners/musicgen_style_conditioner_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/positional_encoding.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::conditioners {
namespace {

namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_config(const MusicGenStyleConfig & config) {
    if (config.hidden_size <= 0 || config.mert_hidden_size <= 0 || config.output_dim <= 0 ||
        config.transformer_layers <= 0 || config.transformer_heads <= 0 ||
        config.transformer_intermediate_size <= 0 || config.codebook_size <= 0 ||
        config.eval_codebooks != 1 || config.downsample_factor <= 0) {
        throw std::runtime_error("MusicGen style config contains unsupported dimensions");
    }
    if (config.hidden_size % config.transformer_heads != 0) {
        throw std::runtime_error("MusicGen style hidden size must be divisible by head count");
    }
}

modules::AttentionWeights style_attention_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const MusicGenStyleConfig & config,
    const MusicGenStyleRuntimeOptions & options,
    int64_t layer) {
    const std::string prefix = "transformer.layers." + std::to_string(layer) + ".self_attn";
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        prefix + ".in_proj_weight",
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        prefix + ".out_proj.weight",
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    return weights;
}

struct StyleLayerWeights {
    modules::NormWeights norm1;
    modules::AttentionWeights attention;
    modules::NormWeights norm2;
    modules::FeedForwardWeights feed_forward;
};

struct StyleWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights embed;
    std::vector<StyleLayerWeights> layers;
    modules::BatchNorm1dEvalWeights batch_norm;
    modules::LinearWeights output_proj;
    std::vector<float> codebook;
};

std::vector<float> batch_norm_scale(
    const assets::TensorSource & source,
    const MusicGenStyleConfig & config) {
    const auto running_var = source.require_f32("batch_norm.running_var", {config.hidden_size});
    std::vector<float> scale(static_cast<size_t>(config.hidden_size));
    for (int64_t channel = 0; channel < config.hidden_size; ++channel) {
        scale[static_cast<size_t>(channel)] =
            1.0F / std::sqrt(running_var[static_cast<size_t>(channel)] + config.batch_norm_eps);
    }
    return scale;
}

std::vector<float> batch_norm_bias(
    const assets::TensorSource & source,
    const MusicGenStyleConfig & config,
    const std::vector<float> & scale) {
    const auto running_mean = source.require_f32("batch_norm.running_mean", {config.hidden_size});
    std::vector<float> bias(static_cast<size_t>(config.hidden_size));
    for (int64_t channel = 0; channel < config.hidden_size; ++channel) {
        bias[static_cast<size_t>(channel)] = -running_mean[static_cast<size_t>(channel)] * scale[static_cast<size_t>(channel)];
    }
    return bias;
}

StyleWeights load_style_weights(
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    const MusicGenStyleConfig & config,
    const MusicGenStyleRuntimeOptions & options) {
    StyleWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        "framework.musicgen_style.weights",
        options.style_weight_context_bytes);
    auto & store = *weights.store;
    weights.embed = {
        store.load_tensor(source, "embed.weight", options.weight_storage_type, {config.hidden_size, config.mert_hidden_size}),
        store.load_f32_tensor(source, "embed.bias", {config.hidden_size})};
    weights.layers.reserve(static_cast<size_t>(config.transformer_layers));
    for (int64_t layer = 0; layer < config.transformer_layers; ++layer) {
        const std::string prefix = "transformer.layers." + std::to_string(layer);
        weights.layers.push_back({
            binding::norm_from_source(store, source, prefix + ".norm1", config.hidden_size),
            style_attention_weights(store, source, config, options, layer),
            binding::norm_from_source(store, source, prefix + ".norm2", config.hidden_size),
            {
                store.load_tensor(
                    source,
                    prefix + ".linear1.weight",
                    options.weight_storage_type,
                    {config.transformer_intermediate_size, config.hidden_size}),
                std::nullopt,
                store.load_tensor(
                    source,
                    prefix + ".linear2.weight",
                    options.weight_storage_type,
                    {config.hidden_size, config.transformer_intermediate_size}),
                std::nullopt,
            }});
    }
    const auto bn_scale = batch_norm_scale(source, config);
    const auto bn_bias = batch_norm_bias(source, config, bn_scale);
    weights.batch_norm = {
        store.make_f32(core::TensorShape::from_dims({config.hidden_size}), bn_scale),
        store.make_f32(core::TensorShape::from_dims({config.hidden_size}), bn_bias)};
    weights.output_proj = {
        store.load_tensor(source, "output_proj.weight", options.weight_storage_type, {config.output_dim, config.hidden_size}),
        store.load_f32_tensor(source, "output_proj.bias", {config.output_dim})};
    weights.codebook = source.require_f32("rvq.vq.layers.0._codebook.embed", {config.codebook_size, config.hidden_size});
    store.upload();
    return weights;
}

core::TensorValue build_style_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & positional_encoding,
    const StyleWeights & weights,
    const MusicGenStyleConfig & config,
    const MusicGenStyleRuntimeOptions & options) {
    auto x = modules::LinearModule({config.mert_hidden_size, config.hidden_size, true, options.projection_precision})
        .build(ctx, input, weights.embed);
    x = modules::PositionalEncodingModule({config.hidden_size}).build(ctx, x, {positional_encoding});
    for (const auto & layer : weights.layers) {
        const auto residual_attn = x;
        auto y = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true, false})
            .build(ctx, x, layer.norm1);
        y = modules::SelfAttentionModule({
                config.hidden_size,
                config.transformer_heads,
                false,
                options.projection_precision,
                options.attention_precision,
                modules::AttentionPrefixCacheLayout::SequenceHeads,
                true,
                false,
            })
            .build(ctx, y, layer.attention);
        x = modules::ResidualAddModule().build(ctx, y, residual_attn);

        const auto residual_ff = x;
        y = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true, false})
            .build(ctx, x, layer.norm2);
        y = modules::FeedForwardModule({
                config.hidden_size,
                config.transformer_intermediate_size,
                false,
                modules::GeluApproximation::ExactErf,
                options.projection_precision,
            })
            .build(ctx, y, layer.feed_forward);
        x = modules::ResidualAddModule().build(ctx, y, residual_ff);
    }
    auto bct = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    bct = modules::BatchNorm1dEvalModule({config.hidden_size}).build(ctx, bct, weights.batch_norm);
    return core::ensure_backend_addressable_layout(ctx, modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, bct));
}

int64_t downsampled_tokens(int64_t tokens, int64_t factor) {
    if (tokens <= 0 || factor <= 0) {
        throw std::runtime_error("MusicGen style downsample dimensions must be positive");
    }
    return (tokens + factor - 1) / factor;
}

std::vector<float> quantize_and_downsample(
    const std::vector<float> & values,
    const std::vector<float> & codebook,
    const MusicGenStyleConfig & config,
    int64_t batch,
    int64_t tokens) {
    const int64_t out_tokens = downsampled_tokens(tokens, config.downsample_factor);
    std::vector<float> output(static_cast<size_t>(batch * out_tokens * config.hidden_size), 0.0F);
    std::vector<float> input_norm(static_cast<size_t>(batch * tokens), 0.0F);
    std::vector<float> code_norm(static_cast<size_t>(config.codebook_size), 0.0F);
    for (int64_t index = 0; index < batch * tokens; ++index) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(index * config.hidden_size);
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            const float value = values[base + static_cast<size_t>(hidden)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        input_norm[static_cast<size_t>(index)] = static_cast<float>(sum);
    }
    for (int64_t code = 0; code < config.codebook_size; ++code) {
        double sum = 0.0;
        const size_t base = static_cast<size_t>(code * config.hidden_size);
        for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
            const float value = codebook[base + static_cast<size_t>(hidden)];
            sum += static_cast<double>(value) * static_cast<double>(value);
        }
        code_norm[static_cast<size_t>(code)] = static_cast<float>(sum);
    }
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t out_t = 0; out_t < out_tokens; ++out_t) {
            const int64_t t = out_t * config.downsample_factor;
            const size_t input_base = static_cast<size_t>((b * tokens + t) * config.hidden_size);
            int64_t best_code = 0;
            float best_score = -std::numeric_limits<float>::infinity();
            for (int64_t code = 0; code < config.codebook_size; ++code) {
                const size_t code_base = static_cast<size_t>(code * config.hidden_size);
                double dot = 0.0;
                for (int64_t hidden = 0; hidden < config.hidden_size; ++hidden) {
                    dot += static_cast<double>(values[input_base + static_cast<size_t>(hidden)]) *
                        static_cast<double>(codebook[code_base + static_cast<size_t>(hidden)]);
                }
                const float score = -input_norm[static_cast<size_t>(b * tokens + t)] +
                    2.0F * static_cast<float>(dot) - code_norm[static_cast<size_t>(code)];
                if (score > best_score) {
                    best_score = score;
                    best_code = code;
                }
            }
            const size_t output_base = static_cast<size_t>((b * out_tokens + out_t) * config.hidden_size);
            const size_t code_base = static_cast<size_t>(best_code * config.hidden_size);
            std::copy_n(codebook.data() + code_base, static_cast<size_t>(config.hidden_size), output.data() + output_base);
        }
    }
    return output;
}

struct StyleGraph {
    explicit StyleGraph(ggml_backend_t owner_backend_in)
        : owner_backend(owner_backend_in) {}

    std::unique_ptr<ggml_context, GgmlContextDeleter> context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_t owner_backend = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue input;
    core::TensorValue positional_encoding;
    core::TensorValue output;
    int64_t batch = 0;
    int64_t tokens = 0;

    ~StyleGraph() {
        if (owner_backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(owner_backend, graph);
            graph = nullptr;
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
    }
};

struct ProjectionGraph {
    explicit ProjectionGraph(ggml_backend_t owner_backend_in)
        : owner_backend(owner_backend_in) {}

    std::unique_ptr<ggml_context, GgmlContextDeleter> context;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_backend_t owner_backend = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue input;
    core::TensorValue output;
    int64_t batch = 0;
    int64_t tokens = 0;

    ~ProjectionGraph() {
        if (owner_backend != nullptr && graph != nullptr) {
            core::release_backend_graph_resources(owner_backend, graph);
            graph = nullptr;
        }
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
    }
};

std::unique_ptr<StyleGraph> build_style_graph(
    core::ExecutionContext & execution,
    const StyleWeights & weights,
    const MusicGenStyleConfig & config,
    const MusicGenStyleRuntimeOptions & options,
    int64_t batch,
    int64_t tokens) {
    auto out = std::make_unique<StyleGraph>(execution.backend());
    ggml_init_params params{options.graph_arena_bytes, nullptr, true};
    out->context.reset(ggml_init(params));
    if (out->context == nullptr) {
        throw std::runtime_error("failed to initialize MusicGen style graph context");
    }
    core::ModuleBuildContext ctx{out->context.get(), "framework.musicgen_style", execution.backend_type()};
    out->batch = batch;
    out->tokens = tokens;
    out->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, tokens, config.mert_hidden_size}));
    out->positional_encoding = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({tokens, config.hidden_size}));
    out->output = build_style_transformer(ctx, out->input, out->positional_encoding, weights, config, options);
    out->graph = ggml_new_graph_custom(out->context.get(), 262144, false);
    ggml_build_forward_expand(out->graph, out->output.tensor);
    out->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    if (out->gallocr == nullptr || !ggml_gallocr_reserve(out->gallocr, out->graph) ||
        !ggml_gallocr_alloc_graph(out->gallocr, out->graph)) {
        throw std::runtime_error("failed to allocate MusicGen style graph");
    }
    std::vector<float> encoding(static_cast<size_t>(tokens * config.hidden_size), 0.0F);
    const int64_t half_dim = config.hidden_size / 2;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t dim = 0; dim < half_dim; ++dim) {
            const float phase = static_cast<float>(t) /
                std::pow(10000.0F, static_cast<float>(dim) / static_cast<float>(half_dim - 1));
            encoding[static_cast<size_t>(t * config.hidden_size + dim)] = std::cos(phase);
            encoding[static_cast<size_t>(t * config.hidden_size + half_dim + dim)] = std::sin(phase);
        }
    }
    core::write_tensor_f32(out->positional_encoding, encoding);
    return out;
}

std::unique_ptr<ProjectionGraph> build_projection_graph(
    core::ExecutionContext & execution,
    const StyleWeights & weights,
    const MusicGenStyleConfig & config,
    const MusicGenStyleRuntimeOptions & options,
    int64_t batch,
    int64_t tokens) {
    auto out = std::make_unique<ProjectionGraph>(execution.backend());
    ggml_init_params params{options.graph_arena_bytes / 4, nullptr, true};
    out->context.reset(ggml_init(params));
    if (out->context == nullptr) {
        throw std::runtime_error("failed to initialize MusicGen style projection graph context");
    }
    core::ModuleBuildContext ctx{out->context.get(), "framework.musicgen_style.project", execution.backend_type()};
    out->batch = batch;
    out->tokens = tokens;
    out->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch, tokens, config.hidden_size}));
    out->output = core::ensure_backend_addressable_layout(
        ctx,
        modules::LinearModule({config.hidden_size, config.output_dim, true, options.projection_precision})
            .build(ctx, out->input, weights.output_proj));
    out->graph = ggml_new_graph_custom(out->context.get(), 4096, false);
    ggml_build_forward_expand(out->graph, out->output.tensor);
    out->gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    if (out->gallocr == nullptr || !ggml_gallocr_reserve(out->gallocr, out->graph) ||
        !ggml_gallocr_alloc_graph(out->gallocr, out->graph)) {
        throw std::runtime_error("failed to allocate MusicGen style projection graph");
    }
    return out;
}

}  // namespace

MusicGenStyleConfig MusicGenStyleConfig::mert_default() {
    MusicGenStyleConfig config;
    config.mert.hidden_size = 768;
    config.mert.intermediate_size = 3072;
    config.mert.num_hidden_layers = 12;
    config.mert.output_hidden_layer = 12;
    config.mert.num_attention_heads = 12;
    config.mert.conv_in_channels = 1;
    config.mert.num_conv_pos_embeddings = 128;
    config.mert.num_conv_pos_embedding_groups = 16;
    config.mert.conv_dim = {512, 512, 512, 512, 512, 512, 512};
    config.mert.conv_kernel = {10, 3, 3, 3, 3, 2, 2};
    config.mert.conv_stride = {5, 2, 2, 2, 2, 2, 2};
    config.mert.layer_norm_eps = 1.0e-5F;
    config.mert.apply_positional_embedding = true;
    config.mert.apply_encoder_input_layer_norm = true;
    config.mert.apply_final_layer_norm = false;
    config.mert.materialize_output = true;
    config.mert.release_graph_after_encode = true;
    config.mert.final_projection_size = 0;
    config.mert.feature_extractor_norm = modules::HubertFeatureExtractorNorm::FirstLayerGroupNorm;
    config.mert.encoder_layer_norm_order = modules::HubertEncoderLayerNormOrder::PostNorm;
    return config;
}

struct MusicGenStyleConditionerRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> mert_source,
        std::shared_ptr<const assets::TensorSource> style_source,
        core::ExecutionContext & execution_ref,
        MusicGenStyleConfig config_value,
        MusicGenStyleRuntimeOptions options_value)
        : execution(execution_ref),
          config(std::move(config_value)),
          options(options_value),
          mert(modules::HubertEncoderComponent::load_from_tensor_source(
              std::move(mert_source),
              execution_ref.config(),
              config.mert,
              mert_binding(options_value))),
          weights(load_style_weights(*style_source, execution_ref, config, options_value)) {
        validate_config(config);
    }

    static modules::HubertEncoderWeightBinding mert_binding(const MusicGenStyleRuntimeOptions & options) {
        modules::HubertEncoderWeightBinding binding;
        binding.conv_storage_type = options.weight_storage_type;
        binding.positional_conv_storage_type = options.weight_storage_type;
        binding.projection_storage_type = options.weight_storage_type;
        binding.attention_storage_type = options.weight_storage_type;
        binding.feed_forward_storage_type = options.weight_storage_type;
        binding.layer.post_attention_layer_norm = "layer_norm";
        return binding;
    }

    core::ExecutionContext & execution;
    MusicGenStyleConfig config;
    MusicGenStyleRuntimeOptions options;
    modules::HubertEncoderComponent mert;
    StyleWeights weights;
    std::unique_ptr<StyleGraph> style_graph;
    std::unique_ptr<ProjectionGraph> projection_graph;
};

MusicGenStyleConditionerRuntime::MusicGenStyleConditionerRuntime(
    std::shared_ptr<const assets::TensorSource> mert_source,
    std::shared_ptr<const assets::TensorSource> style_source,
    core::ExecutionContext & execution,
    MusicGenStyleConfig config,
    MusicGenStyleRuntimeOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(mert_source),
          std::move(style_source),
          execution,
          std::move(config),
          options)) {}

MusicGenStyleConditionerRuntime::~MusicGenStyleConditionerRuntime() = default;

MusicGenStyleEmbedding MusicGenStyleConditionerRuntime::encode_audio(
    const std::vector<float> & waveform,
    int64_t batch,
    int64_t samples,
    size_t) {
    if (impl_ == nullptr) {
        throw std::runtime_error("MusicGen style runtime is not initialized");
    }
    const auto mert_hidden = impl_->mert.encode(waveform, batch, samples);
    if (mert_hidden.hidden_size != impl_->config.mert_hidden_size) {
        throw std::runtime_error("MusicGen style MERT hidden size mismatch");
    }
    if (impl_->style_graph == nullptr || impl_->style_graph->batch != batch ||
        impl_->style_graph->tokens != mert_hidden.tokens) {
        impl_->style_graph = build_style_graph(
            impl_->execution,
            impl_->weights,
            impl_->config,
            impl_->options,
            batch,
            mert_hidden.tokens);
    }
    auto & style_graph = *impl_->style_graph;
    core::write_tensor_f32(style_graph.input, mert_hidden.hidden_states);
    const ggml_status style_status = core::compute_backend_graph(
        impl_->execution.backend(),
        style_graph.graph,
        nullptr,
        "framework.musicgen_style");
    if (style_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MusicGen style graph compute failed");
    }
    auto style_values = core::read_tensor_f32(style_graph.output.tensor);
    auto quantized = quantize_and_downsample(
        style_values,
        impl_->weights.codebook,
        impl_->config,
        batch,
        mert_hidden.tokens);
    const int64_t projected_tokens = downsampled_tokens(mert_hidden.tokens, impl_->config.downsample_factor);
    if (impl_->projection_graph == nullptr || impl_->projection_graph->batch != batch ||
        impl_->projection_graph->tokens != projected_tokens) {
        impl_->projection_graph = build_projection_graph(
            impl_->execution,
            impl_->weights,
            impl_->config,
            impl_->options,
            batch,
            projected_tokens);
    }
    auto & projection_graph = *impl_->projection_graph;
    core::write_tensor_f32(projection_graph.input, quantized);
    const ggml_status projection_status = core::compute_backend_graph(
        impl_->execution.backend(),
        projection_graph.graph,
        nullptr,
        "framework.musicgen_style.projection");
    if (projection_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MusicGen style projection graph compute failed");
    }
    auto values = core::read_tensor_f32(projection_graph.output.tensor);
    impl_->style_graph.reset();
    impl_->projection_graph.reset();
    return {
        batch,
        projected_tokens,
        impl_->config.output_dim,
        std::move(values)};
}

void MusicGenStyleConditionerRuntime::release_runtime_graphs() {
    if (impl_ == nullptr) {
        return;
    }
    impl_->mert.release_runtime_graph();
    impl_->style_graph.reset();
    impl_->projection_graph.reset();
}

}  // namespace engine::conditioners
