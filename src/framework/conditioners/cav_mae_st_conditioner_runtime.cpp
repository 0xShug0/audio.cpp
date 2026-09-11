#include "engine/framework/conditioners/cav_mae_st_conditioner_runtime.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conv_modules.h"
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

std::string tensor_name(const CavMaeStVisualConfig & config, const std::string & name) {
    if (config.tensor_prefix.empty()) {
        return name;
    }
    return config.tensor_prefix + "." + name;
}

void validate_config(const CavMaeStVisualConfig & config) {
    if (config.image_channels <= 0 || config.image_size <= 0 || config.patch_size <= 0 ||
        config.hidden_size <= 0 || config.visual_layers < 0 || config.shared_layers < 0 ||
        config.heads <= 0 || config.intermediate_size <= 0) {
        throw std::runtime_error("CAV-MAE-ST visual config dimensions must be valid");
    }
    if (config.hidden_size % config.heads != 0) {
        throw std::runtime_error("CAV-MAE-ST hidden size must be divisible by head count");
    }
    if (config.image_size % config.patch_size != 0) {
        throw std::runtime_error("CAV-MAE-ST image size must be divisible by patch size");
    }
}

struct CavMaeBlockWeights {
    modules::NormWeights norm1;
    modules::NormWeights norm2;
    modules::AttentionWeights attention;
    modules::FeedForwardWeights mlp;
};

struct CavMaeStWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv2dWeights patch_embedding;
    core::TensorValue pos_embed_v;
    core::TensorValue modality_v;
    std::vector<CavMaeBlockWeights> visual_blocks;
    std::vector<CavMaeBlockWeights> shared_visual_blocks;
    modules::NormWeights norm_v;
};

modules::AttentionWeights load_attention(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const CavMaeStVisualConfig & config,
    const std::string & prefix,
    const CavMaeStRuntimeOptions & options) {
    modules::AttentionWeights weights;
    weights.qkv_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".attn.qkv.weight"),
        options.weight_storage_type,
        {3 * config.hidden_size, config.hidden_size});
    weights.qkv_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".attn.qkv.bias"),
        {3 * config.hidden_size});
    weights.out_weight = store.load_tensor(
        source,
        tensor_name(config, prefix + ".attn.proj.weight"),
        options.weight_storage_type,
        {config.hidden_size, config.hidden_size});
    weights.out_bias = store.load_f32_tensor(
        source,
        tensor_name(config, prefix + ".attn.proj.bias"),
        {config.hidden_size});
    return weights;
}

CavMaeBlockWeights load_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const CavMaeStVisualConfig & config,
    const std::string & prefix,
    const std::string & norm_suffix,
    const CavMaeStRuntimeOptions & options) {
    return {
        binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm1" + norm_suffix), config.hidden_size),
        binding::norm_from_source(store, source, tensor_name(config, prefix + ".norm2" + norm_suffix), config.hidden_size),
        load_attention(store, source, config, prefix, options),
        {
            store.load_tensor(
                source,
                tensor_name(config, prefix + ".mlp.fc1.weight"),
                options.weight_storage_type,
                {config.intermediate_size, config.hidden_size}),
            store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc1.bias"), {config.intermediate_size}),
            store.load_tensor(
                source,
                tensor_name(config, prefix + ".mlp.fc2.weight"),
                options.weight_storage_type,
                {config.hidden_size, config.intermediate_size}),
            store.load_f32_tensor(source, tensor_name(config, prefix + ".mlp.fc2.bias"), {config.hidden_size}),
        },
    };
}

CavMaeStWeights load_weights(
    const assets::TensorSource & source,
    const CavMaeStVisualConfig & config,
    ggml_backend_t backend,
    core::BackendType backend_type,
    const CavMaeStRuntimeOptions & options) {
    CavMaeStWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "framework.cav_mae_st.visual.weights",
        options.weight_context_bytes);
    weights.patch_embedding = {
        weights.store->load_tensor(
            source,
            tensor_name(config, "patch_embed_v.proj.weight"),
            options.weight_storage_type,
            {config.hidden_size, config.image_channels, config.patch_size, config.patch_size}),
        weights.store->load_f32_tensor(
            source,
            tensor_name(config, "patch_embed_v.proj.bias"),
            {config.hidden_size})};
    const int64_t grid = config.image_size / config.patch_size;
    weights.pos_embed_v = weights.store->load_f32_tensor(
        source,
        tensor_name(config, "pos_embed_v"),
        {1, grid * grid, config.hidden_size});
    weights.modality_v = weights.store->load_f32_tensor(
        source,
        tensor_name(config, "modality_v"),
        {1, 1, config.hidden_size});
    weights.visual_blocks.reserve(static_cast<size_t>(config.visual_layers));
    for (int64_t layer = 0; layer < config.visual_layers; ++layer) {
        weights.visual_blocks.push_back(load_block(
            *weights.store,
            source,
            config,
            "blocks_v." + std::to_string(layer),
            "",
            options));
    }
    weights.shared_visual_blocks.reserve(static_cast<size_t>(config.shared_layers));
    for (int64_t layer = 0; layer < config.shared_layers; ++layer) {
        weights.shared_visual_blocks.push_back(load_block(
            *weights.store,
            source,
            config,
            "blocks_u." + std::to_string(layer),
            "_v",
            options));
    }
    weights.norm_v = binding::norm_from_source(
        *weights.store,
        source,
        tensor_name(config, "norm_v"),
        config.hidden_size);
    weights.store->upload();
    return weights;
}

core::TensorValue apply_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const CavMaeBlockWeights & weights,
    const CavMaeStVisualConfig & config,
    const CavMaeStRuntimeOptions & options) {
    const modules::SelfAttentionModule attention({
        config.hidden_size,
        config.heads,
        true,
        options.projection_precision,
        options.attention_precision,
        modules::AttentionPrefixCacheLayout::SequenceHeads,
        true,
        false,
    });
    const modules::FeedForwardModule feed_forward({
        config.hidden_size,
        config.intermediate_size,
        true,
        modules::GeluApproximation::ExactErf,
        options.projection_precision,
    });
    const modules::ResidualAddModule residual_add;

    auto attn_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                          .build(ctx, input, weights.norm1);
    auto hidden = residual_add.build(ctx, input, attention.build(ctx, attn_input, weights.attention));
    auto ff_input = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                        .build(ctx, hidden, weights.norm2);
    return residual_add.build(ctx, hidden, feed_forward.build(ctx, ff_input, weights.mlp));
}

core::TensorValue build_visual_encoder(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & images,
    const CavMaeStWeights & weights,
    const CavMaeStVisualConfig & config,
    const CavMaeStRuntimeOptions & options) {
    const int64_t grid = config.image_size / config.patch_size;
    auto hidden = modules::Conv2dModule({
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
        true,
    }).build(ctx, images, weights.patch_embedding);
    hidden = core::reshape_tensor(ctx, hidden, core::TensorShape::from_dims({images.shape.dims[0], config.hidden_size, grid * grid}));
    hidden = modules::TransposeModule({{0, 2, 1, 3}, hidden.shape.rank}).build(ctx, hidden);
    hidden = core::ensure_backend_addressable_layout(ctx, hidden);

    auto pos_embed = modules::RepeatModule({hidden.shape}).build(ctx, weights.pos_embed_v);
    auto modality = modules::RepeatModule({hidden.shape}).build(ctx, weights.modality_v);
    hidden = modules::AddModule().build(ctx, hidden, pos_embed);
    hidden = modules::AddModule().build(ctx, hidden, modality);

    for (const auto & block : weights.visual_blocks) {
        hidden = apply_block(ctx, hidden, block, config, options);
    }
    for (const auto & block : weights.shared_visual_blocks) {
        hidden = apply_block(ctx, hidden, block, config, options);
    }
    hidden = modules::LayerNormModule({config.hidden_size, config.layer_norm_eps, true, true})
                 .build(ctx, hidden, weights.norm_v);
    return core::ensure_backend_addressable_layout(ctx, hidden);
}

}  // namespace

struct CavMaeStConditionerRuntime::Impl {
    Impl(
        std::shared_ptr<const assets::TensorSource> input_source,
        core::ExecutionContext & input_execution,
        CavMaeStVisualConfig input_config,
        CavMaeStRuntimeOptions input_options)
        : source(std::move(input_source)),
          execution(input_execution),
          backend(input_execution.backend()),
          backend_type(input_execution.backend_type()),
          config(std::move(input_config)),
          options(input_options) {
        if (source == nullptr) {
            throw std::runtime_error("CAV-MAE-ST runtime requires a tensor source");
        }
        if (backend == nullptr) {
            throw std::runtime_error("CAV-MAE-ST runtime requires an initialized backend");
        }
        validate_config(config);
        weights = std::make_shared<CavMaeStWeights>(
            load_weights(*source, config, backend, backend_type, options));
    }

    struct VisualGraph {
        VisualGraph(const Impl & owner, int64_t input_batch, int64_t input_height, int64_t input_width)
            : batch(input_batch),
              height(input_height),
              width(input_width),
              owner_backend(owner.backend) {
            ggml_init_params params{owner.options.graph_arena_bytes, nullptr, true};
            ctx.reset(ggml_init(params));
            if (ctx == nullptr) {
                throw std::runtime_error("CAV-MAE-ST visual failed to create graph context");
            }
            core::ModuleBuildContext build{ctx.get(), "framework.cav_mae_st.visual", owner.backend_type};
            images = core::make_tensor(
                build,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({batch, owner.config.image_channels, height, width}));
            output = build_visual_encoder(build, images, *owner.weights, owner.config, owner.options);
            graph = ggml_new_graph_custom(ctx.get(), 65536, false);
            ggml_set_output(output.tensor);
            ggml_build_forward_expand(graph, output.tensor);
            gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
            if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
                throw std::runtime_error("CAV-MAE-ST visual failed to allocate graph");
            }
        }

        ~VisualGraph() {
            if (owner_backend != nullptr && graph != nullptr) {
                core::release_backend_graph_resources(owner_backend, graph);
                graph = nullptr;
            }
            if (gallocr != nullptr) {
                ggml_gallocr_free(gallocr);
                gallocr = nullptr;
            }
        }

        VisualGraph(const VisualGraph &) = delete;
        VisualGraph & operator=(const VisualGraph &) = delete;

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

    VisualGraph & graph_for_shape(int64_t batch, int64_t height, int64_t width) {
        if (graph == nullptr || graph->batch != batch || graph->height != height || graph->width != width) {
            graph = std::make_unique<VisualGraph>(*this, batch, height, width);
        }
        return *graph;
    }

    CavMaeStVisualFeatures encode_visual(const std::vector<float> & images, int64_t batch, int64_t height, int64_t width) {
        if (batch <= 0 || height != config.image_size || width != config.image_size ||
            static_cast<int64_t>(images.size()) != batch * config.image_channels * height * width) {
            throw std::runtime_error("CAV-MAE-ST visual input shape mismatch");
        }
        auto & active_graph = graph_for_shape(batch, height, width);
        core::write_tensor_f32(active_graph.images, images);
        core::set_backend_threads(execution.backend(), execution.config().threads);
        const ggml_status status = core::compute_backend_graph(execution.backend(), active_graph.graph, nullptr, "framework.cav_mae_st.visual");
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CAV-MAE-ST visual graph compute failed");
        }
        CavMaeStVisualFeatures features;
        features.batch = batch;
        features.patches = (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
        features.hidden = config.hidden_size;
        core::read_tensor_float_into(active_graph.output.tensor, features.values);
        if (static_cast<int64_t>(features.values.size()) != features.batch * features.patches * features.hidden) {
            throw std::runtime_error("CAV-MAE-ST visual output shape mismatch");
        }
        return features;
    }

    std::shared_ptr<const assets::TensorSource> source;
    core::ExecutionContext & execution;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    CavMaeStVisualConfig config;
    CavMaeStRuntimeOptions options;
    std::shared_ptr<CavMaeStWeights> weights;
    std::unique_ptr<VisualGraph> graph;
};

CavMaeStConditionerRuntime::CavMaeStConditionerRuntime(
    std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution,
    CavMaeStVisualConfig config,
    CavMaeStRuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), options)) {}

CavMaeStConditionerRuntime::~CavMaeStConditionerRuntime() = default;
CavMaeStConditionerRuntime::CavMaeStConditionerRuntime(CavMaeStConditionerRuntime &&) noexcept = default;
CavMaeStConditionerRuntime & CavMaeStConditionerRuntime::operator=(CavMaeStConditionerRuntime &&) noexcept = default;

CavMaeStVisualFeatures CavMaeStConditionerRuntime::encode_visual(
    const std::vector<float> & images,
    int64_t batch,
    int64_t height,
    int64_t width) {
    return impl_->encode_visual(images, batch, height, width);
}

void CavMaeStConditionerRuntime::release_runtime_graphs() {
    impl_->graph.reset();
}

}  // namespace engine::conditioners
