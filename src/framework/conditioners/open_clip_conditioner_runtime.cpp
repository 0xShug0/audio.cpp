#include "engine/framework/conditioners/open_clip_conditioner_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <cstddef>
#include <cstdint>
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

std::string join_name(const OpenClipTextConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

std::string join_name(const OpenClipImageConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

struct OpenClipTextLayerWeights {
    modules::NormWeights ln_1;
    modules::AttentionWeights attention;
    modules::NormWeights ln_2;
    modules::FeedForwardWeights mlp;
};

struct OpenClipTextWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue token_embedding;
    core::TensorValue positional_embedding;
    std::vector<OpenClipTextLayerWeights> layers;
    modules::NormWeights ln_final;
};

struct OpenClipImageWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights patch_embedding;
    core::TensorValue class_embedding;
    core::TensorValue positional_embedding;
    modules::NormWeights ln_pre;
    std::vector<OpenClipTextLayerWeights> layers;
    modules::NormWeights ln_post;
    core::TensorValue projection;
};

void validate_config(const OpenClipTextConfig & config) {
    if (config.vocab_size <= 0 || config.context_length <= 0 || config.hidden_size <= 0 ||
        config.layers <= 0 || config.heads <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("OpenCLIP text config dimensions must be positive");
    }
    if (config.hidden_size % config.heads != 0) {
        throw std::runtime_error("OpenCLIP text hidden size must be divisible by head count");
    }
}

void validate_config(const OpenClipImageConfig & config) {
    if (config.image_channels <= 0 || config.image_size <= 0 || config.patch_size <= 0 ||
        config.hidden_size <= 0 || config.output_dim <= 0 || config.layers <= 0 ||
        config.heads <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("OpenCLIP image config dimensions must be positive");
    }
    if (config.hidden_size % config.heads != 0) {
        throw std::runtime_error("OpenCLIP image hidden size must be divisible by head count");
    }
    if (config.image_size < config.patch_size) {
        throw std::runtime_error("OpenCLIP image size must be at least one patch");
    }
}

modules::AttentionWeights load_open_clip_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const OpenClipTextConfig & config,
    const std::string & prefix,
    const OpenClipRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        join_name(config, prefix + ".attn.in_proj_weight"),
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        join_name(config, prefix + ".attn.in_proj_bias"),
        {3 * config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        join_name(config, prefix + ".attn.out_proj.weight"),
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.out_bias = store.load_f32_tensor(
        source,
        join_name(config, prefix + ".attn.out_proj.bias"),
        {config.hidden_size});
    return weights;
}

modules::AttentionWeights load_open_clip_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const OpenClipImageConfig & config,
    const std::string & prefix,
    const OpenClipRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        join_name(config, prefix + ".attn.in_proj_weight"),
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        join_name(config, prefix + ".attn.in_proj_bias"),
        {3 * config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        join_name(config, prefix + ".attn.out_proj.weight"),
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.out_bias = store.load_f32_tensor(
        source,
        join_name(config, prefix + ".attn.out_proj.bias"),
        {config.hidden_size});
    return weights;
}

OpenClipTextLayerWeights load_text_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const OpenClipTextConfig & config,
    int64_t layer,
    const OpenClipRuntimeOptions & options) {
    const std::string prefix = "transformer.resblocks." + std::to_string(layer);
    return {
        binding::norm_from_source(store, source, join_name(config, prefix + ".ln_1"), config.hidden_size),
        load_open_clip_attention(store, source, config, prefix, options),
        binding::norm_from_source(store, source, join_name(config, prefix + ".ln_2"), config.hidden_size),
        {
            store.load_tensor(
                source,
                join_name(config, prefix + ".mlp.c_fc.weight"),
                options.weight_storage_type,
                {config.intermediate_size, config.hidden_size}),
            store.load_f32_tensor(source, join_name(config, prefix + ".mlp.c_fc.bias"), {config.intermediate_size}),
            store.load_tensor(
                source,
                join_name(config, prefix + ".mlp.c_proj.weight"),
                options.weight_storage_type,
                {config.hidden_size, config.intermediate_size}),
            store.load_f32_tensor(source, join_name(config, prefix + ".mlp.c_proj.bias"), {config.hidden_size}),
        },
    };
}

OpenClipTextLayerWeights load_image_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const OpenClipImageConfig & config,
    int64_t layer,
    const OpenClipRuntimeOptions & options) {
    const std::string prefix = "visual.transformer.resblocks." + std::to_string(layer);
    return {
        binding::norm_from_source(store, source, join_name(config, prefix + ".ln_1"), config.hidden_size),
        load_open_clip_attention(store, source, config, prefix, options),
        binding::norm_from_source(store, source, join_name(config, prefix + ".ln_2"), config.hidden_size),
        {
            store.load_tensor(
                source,
                join_name(config, prefix + ".mlp.c_fc.weight"),
                options.weight_storage_type,
                {config.intermediate_size, config.hidden_size}),
            store.load_f32_tensor(source, join_name(config, prefix + ".mlp.c_fc.bias"), {config.intermediate_size}),
            store.load_tensor(
                source,
                join_name(config, prefix + ".mlp.c_proj.weight"),
                options.weight_storage_type,
                {config.hidden_size, config.intermediate_size}),
            store.load_f32_tensor(source, join_name(config, prefix + ".mlp.c_proj.bias"), {config.hidden_size}),
        },
    };
}

OpenClipTextWeights load_text_weights(
    const assets::TensorSource & source,
    const OpenClipTextConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const OpenClipRuntimeOptions & options) {
    OpenClipTextWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.open_clip.text.weights",
        options.weight_context_bytes);
    weights.token_embedding = weights.store->load_tensor(
        source,
        join_name(config, "token_embedding.weight"),
        options.weight_storage_type,
        {config.vocab_size, config.hidden_size});
    weights.positional_embedding = weights.store->load_f32_tensor(
        source,
        join_name(config, "positional_embedding"),
        {config.context_length, config.hidden_size});
    weights.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.layers.push_back(load_text_layer(*weights.store, source, config, layer, options));
    }
    weights.ln_final = binding::norm_from_source(
        *weights.store,
        source,
        join_name(config, "ln_final"),
        config.hidden_size);
    weights.store->upload();
    return weights;
}

OpenClipImageWeights load_image_weights(
    const assets::TensorSource & source,
    const OpenClipImageConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const OpenClipRuntimeOptions & options) {
    OpenClipImageWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.open_clip.image.weights",
        options.weight_context_bytes);
    weights.patch_embedding = {
        weights.store->load_tensor(
            source,
            join_name(config, "visual.conv1.weight"),
            options.weight_storage_type,
            {config.hidden_size, config.image_channels, config.patch_size, config.patch_size}),
        std::nullopt};
    weights.class_embedding = weights.store->load_f32_tensor(
        source,
        join_name(config, "visual.class_embedding"),
        {config.hidden_size});
    const int64_t grid = config.image_size / config.patch_size;
    weights.positional_embedding = weights.store->load_f32_tensor(
        source,
        join_name(config, "visual.positional_embedding"),
        {grid * grid + 1, config.hidden_size});
    weights.ln_pre = binding::norm_from_source(
        *weights.store,
        source,
        join_name(config, "visual.ln_pre"),
        config.hidden_size);
    weights.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.layers.push_back(load_image_layer(*weights.store, source, config, layer, options));
    }
    weights.ln_post = binding::norm_from_source(
        *weights.store,
        source,
        join_name(config, "visual.ln_post"),
        config.hidden_size);
    weights.projection = weights.store->load_tensor(
        source,
        join_name(config, "visual.proj"),
        options.weight_storage_type,
        {config.hidden_size, config.output_dim});
    weights.store->upload();
    return weights;
}

core::TensorValue l2_normalize_last(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, contiguous.tensor), contiguous.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({static_cast<int>(contiguous.shape.rank - 1)}).build(ctx, squared);
    auto norm = core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, ggml_sqrt(ctx.ggml, sum.tensor), 1.0F, 1.0e-12F),
        sum.shape,
        GGML_TYPE_F32);
    auto norm_shape = contiguous.shape;
    norm_shape.dims[norm_shape.rank - 1] = 1;
    norm = core::reshape_tensor(ctx, norm, norm_shape);
    auto repeated = modules::RepeatModule({contiguous.shape}).build(ctx, norm);
    return core::wrap_tensor(ggml_div(ctx.ggml, contiguous.tensor, repeated.tensor), contiguous.shape, GGML_TYPE_F32);
}

core::TensorValue build_text_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & token_ids,
    const OpenClipTextWeights & weights,
    const OpenClipTextConfig & config,
    const OpenClipRuntimeOptions & options) {
    auto hidden = modules::EmbeddingModule({config.vocab_size, config.hidden_size})
                      .build(ctx, token_ids, weights.token_embedding);
    auto positions = core::reshape_tensor(ctx, weights.positional_embedding, core::TensorShape::from_dims({1, config.context_length, config.hidden_size}));
    positions = modules::RepeatModule({hidden.shape}).build(ctx, positions);
    hidden = modules::AddModule().build(ctx, hidden, positions);

    const modules::SelfAttentionModule attention({
        config.hidden_size,
        config.heads,
        true,
        options.projection_precision,
        options.attention_precision,
        modules::AttentionPrefixCacheLayout::SequenceHeads,
        true,
        config.causal_text_attention,
    });
    const modules::FeedForwardModule feed_forward({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::Quick,
        options.projection_precision,
    });
    const modules::ResidualAddModule residual_add;
    for (const auto & layer : weights.layers) {
        auto attn_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                              .build(ctx, hidden, layer.ln_1);
        auto attn_out = attention.build(ctx, attn_input, layer.attention);
        hidden = residual_add.build(ctx, hidden, attn_out);
        auto ff_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                            .build(ctx, hidden, layer.ln_2);
        auto ff_out = feed_forward.build(ctx, ff_input, layer.mlp);
        hidden = residual_add.build(ctx, hidden, ff_out);
    }

    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, weights.ln_final);
    if (config.normalize_output) {
        hidden = l2_normalize_last(ctx, hidden);
    }
    return core::ensure_backend_addressable_layout(ctx, hidden);
}

core::TensorValue build_image_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & images,
    const OpenClipImageWeights & weights,
    const OpenClipImageConfig & config,
    const OpenClipRuntimeOptions & options) {
    const int64_t grid = config.image_size / config.patch_size;
    auto patches = modules::Conv2dModule({
        config.image_channels,
        config.hidden_size,
        config.patch_size,
        config.patch_size,
        static_cast<int>(config.patch_size),
        static_cast<int>(config.patch_size),
        0,
        0,
        1,
        1,
        false,
    }).build(ctx, images, weights.patch_embedding);
    patches = core::reshape_tensor(ctx, patches, core::TensorShape::from_dims({images.shape.dims[0], config.hidden_size, grid * grid}));
    patches = modules::TransposeModule({{0, 2, 1, 3}, patches.shape.rank}).build(ctx, patches);
    patches = core::ensure_backend_addressable_layout(ctx, patches);

    auto class_token = core::reshape_tensor(ctx, weights.class_embedding, core::TensorShape::from_dims({1, 1, config.hidden_size}));
    class_token = modules::RepeatModule({core::TensorShape::from_dims({images.shape.dims[0], 1, config.hidden_size})}).build(ctx, class_token);
    auto hidden = modules::ConcatModule({1}).build(ctx, class_token, patches);

    auto positions = core::reshape_tensor(ctx, weights.positional_embedding, core::TensorShape::from_dims({1, grid * grid + 1, config.hidden_size}));
    positions = modules::RepeatModule({hidden.shape}).build(ctx, positions);
    hidden = modules::AddModule().build(ctx, hidden, positions);
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, weights.ln_pre);

    modules::AttentionConfig image_attention_config{
        config.hidden_size,
        config.heads,
        true,
        options.projection_precision,
        options.attention_precision,
        modules::AttentionPrefixCacheLayout::SequenceHeads,
        true,
        false,
    };
    image_attention_config.use_flash_attention = true;
    const modules::SelfAttentionModule attention(image_attention_config);
    const modules::FeedForwardModule feed_forward({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::Quick,
        options.projection_precision,
    });
    const modules::ResidualAddModule residual_add;
    for (const auto & layer : weights.layers) {
        auto attn_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                              .build(ctx, hidden, layer.ln_1);
        auto attn_out = attention.build(ctx, attn_input, layer.attention);
        hidden = residual_add.build(ctx, hidden, attn_out);
        auto ff_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                            .build(ctx, hidden, layer.ln_2);
        auto ff_out = feed_forward.build(ctx, ff_input, layer.mlp);
        hidden = residual_add.build(ctx, hidden, ff_out);
    }

    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, weights.ln_post);
    auto pooled = modules::SliceModule({1, 0, 1}).build(ctx, hidden);
    pooled = core::ensure_backend_addressable_layout(ctx, pooled);
    pooled = core::reshape_tensor(ctx, pooled, core::TensorShape::from_dims({images.shape.dims[0], config.hidden_size}));
    auto projection = modules::TransposeModule({{1, 0, 2, 3}, weights.projection.shape.rank}).build(ctx, weights.projection);
    projection = core::ensure_backend_addressable_layout(ctx, projection);
    pooled = modules::LinearModule({config.hidden_size, config.output_dim, false, options.projection_precision})
                 .build(ctx, pooled, {projection, std::nullopt});
    if (config.normalize_output) {
        pooled = l2_normalize_last(ctx, pooled);
    }
    return core::ensure_backend_addressable_layout(ctx, pooled);
}

}  // namespace

struct OpenClipConditionerRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> input_source,
        core::ExecutionContext & input_execution,
        OpenClipTextConfig input_config,
        OpenClipRuntimeOptions input_options,
        OpenClipImageConfig input_image_config)
        : source(std::move(input_source)),
          execution(input_execution),
          backend(input_execution.backend()),
          backend_type(input_execution.backend_type()),
          config(std::move(input_config)),
          image_config(std::move(input_image_config)),
          options(input_options) {
        if (source == nullptr) {
            throw std::runtime_error("OpenCLIP conditioner runtime requires a tensor source");
        }
        if (backend == nullptr) {
            throw std::runtime_error("OpenCLIP conditioner runtime requires an initialized backend");
        }
        if (options.load_text) {
            validate_config(config);
            weights = std::make_shared<OpenClipTextWeights>(
                load_text_weights(*source, config, backend, backend_type, options));
        }
        if (options.load_image) {
            validate_config(image_config);
            image_weights = std::make_shared<OpenClipImageWeights>(
                load_image_weights(*source, image_config, backend, backend_type, options));
        }
    }

    struct TextGraph {
        TextGraph(const Impl & owner, int64_t input_batch, int64_t input_tokens)
            : batch(input_batch),
              tokens(input_tokens),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("OpenCLIP text failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.open_clip.text", owner.backend_type};
            token_ids = core::make_tensor(build, GGML_TYPE_I32, core::TensorShape::from_dims({batch, tokens}));
            if (owner.weights == nullptr) {
                throw std::runtime_error("OpenCLIP text weights are not loaded");
            }
            output = build_text_encoder(build, token_ids, *owner.weights, owner.config, owner.options);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("OpenCLIP text failed to allocate graph");
            }
        }

        ~TextGraph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        TextGraph(const TextGraph &) = delete;
        TextGraph & operator=(const TextGraph &) = delete;

        int64_t batch = 0;
        int64_t tokens = 0;
        ggml_backend_t owner_backend = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue token_ids;
        core::TensorValue output;
    };

    struct ImageGraph {
        ImageGraph(const Impl & owner, int64_t input_batch, int64_t input_height, int64_t input_width)
            : batch(input_batch),
              height(input_height),
              width(input_width),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("OpenCLIP image failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.open_clip.image", owner.backend_type};
            images = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, owner.image_config.image_channels, height, width}));
            if (owner.image_weights == nullptr) {
                throw std::runtime_error("OpenCLIP image weights are not loaded");
            }
            output = build_image_encoder(build, images, *owner.image_weights, owner.image_config, owner.options);
            graph = ggml_new_graph_custom(ctx.get(), 262144, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("OpenCLIP image failed to allocate graph");
            }
        }

        ~ImageGraph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        ImageGraph(const ImageGraph &) = delete;
        ImageGraph & operator=(const ImageGraph &) = delete;

        int64_t batch = 0;
        int64_t height = 0;
        int64_t width = 0;
        ggml_backend_t owner_backend = nullptr;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
        ggml_cgraph * graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;
        core::TensorValue images;
        core::TensorValue output;
    };

    TextGraph & text_graph_for_shape(int64_t batch, int64_t tokens) {
        if (text_graph == nullptr || text_graph->batch != batch || text_graph->tokens != tokens) {
            text_graph = std::make_unique<TextGraph>(*this, batch, tokens);
        }
        return *text_graph;
    }

    ImageGraph & image_graph_for_shape(int64_t batch, int64_t height, int64_t width) {
        if (image_graph == nullptr || image_graph->batch != batch || image_graph->height != height || image_graph->width != width) {
            image_graph = std::make_unique<ImageGraph>(*this, batch, height, width);
        }
        return *image_graph;
    }

    OpenClipTextHidden encode_text(const std::vector<int32_t> & token_ids, int64_t batch, int64_t tokens) {
        if (!options.load_text) {
            throw std::runtime_error("OpenCLIP text path is not enabled");
        }
        if (batch <= 0 || tokens <= 0 || tokens > config.context_length ||
            static_cast<int64_t>(token_ids.size()) != batch * tokens) {
            throw std::runtime_error("OpenCLIP text token id shape mismatch");
        }
        auto & active_graph = text_graph_for_shape(batch, tokens);
        const double upload_ms = debug::measure_ms([&]() {
            core::write_tensor_i32(active_graph.token_ids, token_ids);
        });
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms = debug::measure_ms([&]() {
            core::set_backend_threads(execution.backend(), execution.config().threads);
            status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.open_clip.text");
            ggml_backend_synchronize(execution.backend());
        });
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OpenCLIP text graph compute failed");
        }
        OpenClipTextHidden hidden;
        hidden.batch = batch;
        hidden.tokens = tokens;
        hidden.hidden = config.hidden_size;
        const double read_ms = debug::measure_ms([&]() {
            core::read_tensor_float_into(active_graph.output.tensor, hidden.values);
        });
        if (static_cast<int64_t>(hidden.values.size()) != batch * tokens * config.hidden_size) {
            throw std::runtime_error("OpenCLIP text output shape mismatch");
        }
        debug::timing_log_scalar("framework.open_clip.text.upload_ms", upload_ms);
        debug::timing_log_scalar("framework.open_clip.text.compute_ms", compute_ms);
        debug::timing_log_scalar("framework.open_clip.text.read_ms", read_ms);
        return hidden;
    }

    OpenClipImageEmbedding encode_image(const std::vector<float> & images, int64_t batch, int64_t height, int64_t width) {
        if (!options.load_image) {
            throw std::runtime_error("OpenCLIP image path is not enabled");
        }
        if (batch <= 0 || height != image_config.image_size || width != image_config.image_size ||
            static_cast<int64_t>(images.size()) != batch * image_config.image_channels * height * width) {
            throw std::runtime_error("OpenCLIP image input shape mismatch");
        }
        auto & active_graph = image_graph_for_shape(batch, height, width);
        const double upload_ms = debug::measure_ms([&]() {
            core::write_tensor_f32(active_graph.images, images);
        });
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms = debug::measure_ms([&]() {
            core::set_backend_threads(execution.backend(), execution.config().threads);
            status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.open_clip.image");
            ggml_backend_synchronize(execution.backend());
        });
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OpenCLIP image graph compute failed");
        }
        OpenClipImageEmbedding embedding;
        embedding.batch = batch;
        embedding.features = image_config.output_dim;
        const double read_ms = debug::measure_ms([&]() {
            core::read_tensor_float_into(active_graph.output.tensor, embedding.values);
        });
        if (static_cast<int64_t>(embedding.values.size()) != batch * image_config.output_dim) {
            throw std::runtime_error("OpenCLIP image output shape mismatch");
        }
        debug::timing_log_scalar("framework.open_clip.image.upload_ms", upload_ms);
        debug::timing_log_scalar("framework.open_clip.image.compute_ms", compute_ms);
        debug::timing_log_scalar("framework.open_clip.image.read_ms", read_ms);
        return embedding;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & execution;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    OpenClipTextConfig config;
    OpenClipImageConfig image_config;
    OpenClipRuntimeOptions options;
    std::shared_ptr<OpenClipTextWeights> weights;
    std::shared_ptr<OpenClipImageWeights> image_weights;
    std::unique_ptr<TextGraph> text_graph;
    std::unique_ptr<ImageGraph> image_graph;
};

OpenClipConditionerRuntime::OpenClipConditionerRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    OpenClipTextConfig config,
    OpenClipRuntimeOptions options,
    OpenClipImageConfig image_config)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options, std::move(image_config))) {}

OpenClipConditionerRuntime::~OpenClipConditionerRuntime() = default;
OpenClipConditionerRuntime::OpenClipConditionerRuntime(OpenClipConditionerRuntime &&) noexcept = default;
OpenClipConditionerRuntime & OpenClipConditionerRuntime::operator=(OpenClipConditionerRuntime &&) noexcept = default;

OpenClipTextHidden OpenClipConditionerRuntime::encode_text(const std::vector<int32_t> & token_ids, int64_t batch, int64_t tokens) {
    return impl_->encode_text(token_ids, batch, tokens);
}

OpenClipImageEmbedding OpenClipConditionerRuntime::encode_image(const std::vector<float> & images, int64_t batch, int64_t height, int64_t width) {
    return impl_->encode_image(images, batch, height, width);
}

void OpenClipConditionerRuntime::release_runtime_graphs() {
    impl_->text_graph.reset();
    impl_->image_graph.reset();
}

}  // namespace engine::conditioners
