#include "engine/community_models/mira_tts/processor.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::community_models::mira_tts {
namespace {

namespace binding = modules::binding;

struct ContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) ggml_free(context);
    }
};

struct ConvNeXtWeights {
    modules::Conv1dWeights depthwise;
    modules::NormWeights norm;
    modules::LinearWeights first;
    modules::LinearWeights second;
    core::TensorValue gamma;
};

struct PlainStageWeights {
    modules::Conv1dWeights embed;
    modules::NormWeights norm;
    std::vector<ConvNeXtWeights> blocks;
    modules::NormWeights final_norm;
};

struct ConditionalNormWeights {
    modules::LinearWeights scale;
    modules::LinearWeights shift;
};

struct ConditionalBlockWeights {
    modules::Conv1dWeights depthwise;
    ConditionalNormWeights norm;
    modules::LinearWeights first;
    modules::LinearWeights second;
    core::TensorValue gamma;
};

struct ProcessorWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue speech_codebook;
    modules::Conv1dWeights speech_projection;
    modules::LinearWeights speech_linear;
    core::TensorValue context_codebook;
    modules::LinearWeights context_project_out;
    modules::LinearWeights speaker_project;
    std::vector<PlainStageWeights> downsample;
    modules::Conv1dWeights backbone_embed;
    ConditionalNormWeights backbone_norm;
    std::vector<ConditionalBlockWeights> backbone_blocks;
    modules::NormWeights final_norm;
    modules::LinearWeights output_linear;
};

modules::Conv1dWeights load_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel,
    int64_t groups = 1) {
    modules::Conv1dWeights out;
    out.weight = store.load_tensor(
        source, prefix + ".weight", storage_type,
        {out_channels, in_channels / groups, kernel});
    out.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return out;
}

modules::LinearWeights load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features,
    bool bias = true) {
    return binding::linear_from_source(
        store, source, prefix, storage_type, out_features, in_features, bias);
}

modules::NormWeights load_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t hidden) {
    return {
        store.load_f32_tensor(source, prefix + ".weight", {hidden}),
        store.load_f32_tensor(source, prefix + ".bias", {hidden})};
}

ConvNeXtWeights load_plain_block(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType linear_type,
    assets::TensorStorageType conv_type) {
    ConvNeXtWeights out;
    out.depthwise = load_conv(store, source, prefix + ".dwconv", conv_type, 384, 384, 7, 384);
    out.norm = load_norm(store, source, prefix + ".norm", 384);
    out.first = load_linear(store, source, prefix + ".pwconv1", linear_type, 2048, 384);
    out.second = load_linear(store, source, prefix + ".pwconv2", linear_type, 384, 2048);
    out.gamma = store.load_f32_tensor(source, prefix + ".gamma", {384});
    return out;
}

ConditionalNormWeights load_cond_norm(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type) {
    return {
        load_linear(store, source, prefix + ".scale", storage_type, 384, 1024),
        load_linear(store, source, prefix + ".shift", storage_type, 384, 1024)};
}

ProcessorWeights load_weights(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t context_bytes,
    assets::TensorStorageType linear_type,
    assets::TensorStorageType conv_type) {
    ProcessorWeights out;
    out.store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(),
        "mira_tts.processor.weights", context_bytes);
    const auto & source = *assets.processor_weights;
    out.speech_codebook = out.store->load_tensor(
        source, "quantizer.codebook.weight", assets::TensorStorageType::F32, {8192, 8});
    out.speech_projection = load_conv(
        *out.store, source, "quantizer.out_project", conv_type, 1024, 8, 1);
    out.speech_linear = load_linear(
        *out.store, source, "prenet.linear_pre", linear_type, 384, 1024);
    out.context_codebook = out.store->load_tensor(
        source, "speaker_encoder.context_codebook", assets::TensorStorageType::F32,
        {4096, 6});
    out.context_project_out = load_linear(
        *out.store, source, "speaker_encoder.quantizer.project_out",
        linear_type, 128, 6);
    out.speaker_project = load_linear(
        *out.store, source, "speaker_encoder.project", linear_type, 1024, 4096);
    for (int stage = 0; stage < 2; ++stage) {
        const std::string prefix = "prenet.downsample." + std::to_string(stage) + ".1";
        PlainStageWeights item;
        item.embed = load_conv(*out.store, source, prefix + ".embed", conv_type, 384, 384, 7);
        item.norm = load_norm(*out.store, source, prefix + ".norm", 384);
        for (int block = 0; block < 2; ++block) {
            item.blocks.push_back(load_plain_block(
                *out.store, source,
                prefix + ".convnext." + std::to_string(block),
                linear_type, conv_type));
        }
        item.final_norm = load_norm(*out.store, source, prefix + ".final_layer_norm", 384);
        out.downsample.push_back(std::move(item));
    }
    out.backbone_embed = load_conv(
        *out.store, source, "prenet.vocos_backbone.embed", conv_type, 384, 384, 7);
    out.backbone_norm = load_cond_norm(
        *out.store, source, "prenet.vocos_backbone.norm", linear_type);
    for (int block = 0; block < 12; ++block) {
        const std::string prefix =
            "prenet.vocos_backbone.convnext." + std::to_string(block);
        ConditionalBlockWeights item;
        item.depthwise = load_conv(
            *out.store, source, prefix + ".dwconv", conv_type, 384, 384, 7, 384);
        item.norm = load_cond_norm(*out.store, source, prefix + ".norm", linear_type);
        item.first = load_linear(*out.store, source, prefix + ".pwconv1", linear_type, 2048, 384);
        item.second = load_linear(*out.store, source, prefix + ".pwconv2", linear_type, 384, 2048);
        item.gamma = out.store->load_f32_tensor(source, prefix + ".gamma", {384});
        out.backbone_blocks.push_back(std::move(item));
    }
    out.final_norm = load_norm(
        *out.store, source, "prenet.vocos_backbone.final_layer_norm", 384);
    out.output_linear = load_linear(
        *out.store, source, "prenet.linear", linear_type, 1024, 384);
    out.store->upload();
    return out;
}

core::TensorValue linear(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::LinearWeights & weights,
    int64_t in_features,
    int64_t out_features) {
    return modules::LinearModule({in_features, out_features, weights.bias.has_value()})
        .build(ctx, input, weights);
}

core::TensorValue conv(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::Conv1dWeights & weights,
    int64_t channels,
    int64_t kernel,
    int64_t groups = 1) {
    if (groups == channels) {
        return modules::DepthwiseConv1dModule({
            channels, kernel, 1, static_cast<int>(kernel / 2), 1, true})
            .build(ctx, input, {weights.weight, weights.bias});
    }
    if (groups != 1) {
        throw std::runtime_error("MiraTTS processor only supports regular or depthwise convolution");
    }
    return modules::Conv1dModule({
        channels, channels, kernel, 1, static_cast<int>(kernel / 2), 1, true})
        .build(ctx, input, weights);
}

core::TensorValue layer_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const modules::NormWeights & weights) {
    return modules::LayerNormModule({384, 1.0e-5F, true, true})
        .build(ctx, input, weights);
}

core::TensorValue scale_last(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & scale) {
    auto shaped = core::reshape_tensor(
        ctx, scale, core::TensorShape::from_dims({1, 1, 384}));
    auto repeated = core::wrap_tensor(
        ggml_repeat(ctx.ggml, shaped.tensor, input.tensor), input.shape, GGML_TYPE_F32);
    return modules::MulModule{}.build(ctx, input, repeated);
}

core::TensorValue plain_block(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const ConvNeXtWeights & weights) {
    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    x = conv(ctx, x, weights.depthwise, 384, 7, 384);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = layer_norm(ctx, x, weights.norm);
    x = linear(ctx, x, weights.first, 384, 2048);
    x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
    x = linear(ctx, x, weights.second, 2048, 384);
    x = scale_last(ctx, x, weights.gamma);
    return modules::AddModule{}.build(ctx, input, x);
}

core::TensorValue conditional_norm(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const core::TensorValue & condition,
    const ConditionalNormWeights & weights) {
    auto normalized = modules::LayerNormModule({384, 1.0e-5F, false, false})
        .build(ctx, input, {});
    auto scale = linear(ctx, condition, weights.scale, 1024, 384);
    auto shift = linear(ctx, condition, weights.shift, 1024, 384);
    scale = core::reshape_tensor(ctx, scale, core::TensorShape::from_dims({1, 1, 384}));
    shift = core::reshape_tensor(ctx, shift, core::TensorShape::from_dims({1, 1, 384}));
    auto scale_rep = core::wrap_tensor(
        ggml_repeat(ctx.ggml, scale.tensor, normalized.tensor), normalized.shape, GGML_TYPE_F32);
    auto shift_rep = core::wrap_tensor(
        ggml_repeat(ctx.ggml, shift.tensor, normalized.tensor), normalized.shape, GGML_TYPE_F32);
    auto x = modules::MulModule{}.build(ctx, normalized, scale_rep);
    return modules::AddModule{}.build(ctx, x, shift_rep);
}

}  // namespace

struct MiraAcousticProcessor::Impl {
    Impl(
        const MiraTTSAssets & assets,
        core::ExecutionContext & execution_in,
        size_t weight_bytes,
        size_t graph_bytes,
        assets::TensorStorageType linear_type,
        assets::TensorStorageType conv_type)
        : execution(execution_in),
          graph_context_bytes(graph_bytes),
          weights(load_weights(
              assets, execution_in, weight_bytes, linear_type, conv_type)) {}

    std::vector<float> process(
        const std::vector<int32_t> & speech_codes,
        const std::vector<int32_t> & context_codes) {
        if (speech_codes.empty()) throw std::runtime_error("MiraTTS processor requires speech codes");
        if (context_codes.size() != 32) throw std::runtime_error("MiraTTS processor requires 32 context codes");
        const int64_t frames = static_cast<int64_t>(speech_codes.size());
        ggml_init_params params{graph_context_bytes, nullptr, true};
        std::unique_ptr<ggml_context, ContextDeleter> context(ggml_init(params));
        if (!context) throw std::runtime_error("failed to create MiraTTS processor graph context");
        auto * speech = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, frames, 1);
        auto * speaker = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 32, 1);
        ggml_set_input(speech);
        ggml_set_input(speaker);
        core::ModuleBuildContext build{context.get(), "mira_tts.processor"};

        auto speech_ids = core::wrap_tensor(
            speech, core::TensorShape::from_dims({1, frames}), GGML_TYPE_I32);
        auto x = modules::EmbeddingModule({8192, 8}).build(
            build, speech_ids, weights.speech_codebook);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
        x = modules::Conv1dModule({8, 1024, 1, 1, 0, 1, true})
            .build(build, x, weights.speech_projection);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
        x = linear(build, x, weights.speech_linear, 1024, 384);

        auto context_ids = core::wrap_tensor(
            speaker, core::TensorShape::from_dims({1, 32}), GGML_TYPE_I32);
        auto condition = modules::EmbeddingModule({4096, 6}).build(
            build, context_ids, weights.context_codebook);
        condition = linear(build, condition, weights.context_project_out, 6, 128);
        // The exported processor flattens [B, 128, 32], not [B, 32, 128].
        // Preserve that channel-major speaker-conditioning order.
        condition = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, condition);
        condition = core::reshape_tensor(
            build,
            core::ensure_backend_addressable_layout(build, condition),
            core::TensorShape::from_dims({1, 4096}));
        condition = linear(build, condition, weights.speaker_project, 4096, 1024);

        for (const auto & stage : weights.downsample) {
            x = core::wrap_tensor(ggml_scale(build.ggml, x.tensor, 3.0F), x.shape, x.type);
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
            x = conv(build, x, stage.embed, 384, 7);
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
            x = layer_norm(build, x, stage.norm);
            for (const auto & block : stage.blocks) x = plain_block(build, x, block);
            x = layer_norm(build, x, stage.final_norm);
        }

        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
        x = conv(build, x, weights.backbone_embed, 384, 7);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
        x = conditional_norm(build, x, condition, weights.backbone_norm);
        for (const auto & block : weights.backbone_blocks) {
            auto residual = x;
            auto hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
            hidden = conv(build, hidden, block.depthwise, 384, 7, 384);
            hidden = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, hidden);
            hidden = conditional_norm(build, hidden, condition, block.norm);
            hidden = linear(build, hidden, block.first, 384, 2048);
            hidden = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(build, hidden);
            hidden = linear(build, hidden, block.second, 2048, 384);
            hidden = scale_last(build, hidden, block.gamma);
            x = modules::AddModule{}.build(build, residual, hidden);
        }
        x = layer_norm(build, x, weights.final_norm);
        x = linear(build, x, weights.output_linear, 384, 1024);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, x);
        auto cond_bct = core::reshape_tensor(
            build, condition, core::TensorShape::from_dims({1, 1024, 1}));
        cond_bct = core::wrap_tensor(
            ggml_repeat(build.ggml, cond_bct.tensor, x.tensor), x.shape, GGML_TYPE_F32);
        x = modules::AddModule{}.build(build, x, cond_bct);
        x = core::ensure_backend_addressable_layout(build, x);
        ggml_set_output(x.tensor);
        auto * graph = ggml_new_graph_custom(context.get(), 65536, false);
        ggml_build_forward_expand(graph, x.tensor);
        auto allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(execution.backend()));
        if (allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
            if (allocator != nullptr) ggml_gallocr_free(allocator);
            throw std::runtime_error("failed to allocate MiraTTS processor graph");
        }
        ggml_backend_tensor_set(speech, speech_codes.data(), 0, speech_codes.size() * sizeof(int32_t));
        ggml_backend_tensor_set(speaker, context_codes.data(), 0, context_codes.size() * sizeof(int32_t));
        core::set_backend_threads(execution.backend(), std::max(1, execution.config().threads));
        const auto status = core::compute_backend_graph(execution.backend(), graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) {
            core::release_backend_graph_resources(execution.backend(), graph);
            ggml_gallocr_free(allocator);
            throw std::runtime_error("MiraTTS processor graph compute failed");
        }
        std::vector<float> output(static_cast<size_t>(1024 * frames));
        ggml_backend_tensor_get(x.tensor, output.data(), 0, output.size() * sizeof(float));
        core::release_backend_graph_resources(execution.backend(), graph);
        ggml_gallocr_free(allocator);
        return output;
    }

    core::ExecutionContext & execution;
    size_t graph_context_bytes;
    ProcessorWeights weights;
};

MiraAcousticProcessor::MiraAcousticProcessor(
    const MiraTTSAssets & assets,
    core::ExecutionContext & execution,
    size_t weight_context_bytes,
    size_t graph_context_bytes,
    assets::TensorStorageType linear_storage_type,
    assets::TensorStorageType conv_storage_type)
    : impl_(std::make_unique<Impl>(
          assets, execution, weight_context_bytes, graph_context_bytes,
          linear_storage_type, conv_storage_type)) {}

MiraAcousticProcessor::~MiraAcousticProcessor() = default;

std::vector<float> MiraAcousticProcessor::process(
    const std::vector<int32_t> & speech_codes,
    const std::vector<int32_t> & context_codes) {
    return impl_->process(speech_codes, context_codes);
}

}  // namespace engine::community_models::mira_tts
