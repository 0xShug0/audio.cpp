#include "engine/models/owsm_ctc/model.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/greedy_decode.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::owsm_ctc {
namespace {

using core::TensorShape;
using core::TensorValue;
constexpr size_t kGraphNodes = 65536;
constexpr float kLayerNormEps = 1.0e-12F;

struct Graph {
    core::ExecutionContext & execution;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan plan;

    Graph(core::ExecutionContext & execution, size_t context_bytes) : execution(execution) {
        context = ggml_init({context_bytes, nullptr, true});
        if (!context) {
            throw std::runtime_error("OWSM-CTC v4 graph context allocation failed");
        }
        graph = ggml_new_graph_custom(context, kGraphNodes, false);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
    }
    ~Graph() {
        plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
        ggml_gallocr_free(allocator);
        ggml_free(context);
    }
    void allocate() {
        core::validate_backend_graph_supported(execution.backend(), graph, "OWSM-CTC v4");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("OWSM-CTC v4 graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "OWSM-CTC v4") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OWSM-CTC v4 graph execution failed");
        }
    }
};

TensorValue build_ebranchformer_layer(
    core::ModuleBuildContext & ctx,
    TensorValue input,
    const OWSMCTCV4EBranchformerLayerWeights & weights,
    int64_t hidden_size,
    int64_t num_heads,
    int64_t intermediate_size,
    const TensorValue & half_scale,
    const TensorValue * memory = nullptr,
    const TensorValue * memory_mask = nullptr) {
    const modules::LayerNormModule norm({hidden_size, kLayerNormEps});
    const modules::LayerScaleModule scale;
    const modules::ResidualAddModule residual;

    auto transformed = norm.build(ctx, input, weights.macaron_norm);
    transformed = modules::LinearModule({hidden_size, intermediate_size, true}).build(
        ctx, transformed, {weights.macaron_ffn.fc1_weight, weights.macaron_ffn.fc1_bias});
    transformed = modules::SiluModule().build(ctx, transformed);
    transformed = modules::LinearModule({intermediate_size, hidden_size, true}).build(
        ctx, transformed, {weights.macaron_ffn.fc2_weight, weights.macaron_ffn.fc2_bias});
    transformed = scale.build(ctx, transformed, {half_scale});
    auto x = residual.build(ctx, input, transformed);

    auto attention_input = norm.build(ctx, x, weights.attention_norm);
    modules::AttentionConfig attention_config{hidden_size, num_heads, true};
    attention_config.use_packed_qkv = true;
    auto attention = modules::SelfAttentionModule(attention_config).build(ctx, attention_input, weights.attention);

    auto cgmlp = norm.build(ctx, x, weights.cgmlp_norm);
    cgmlp = modules::LinearModule({hidden_size, intermediate_size, true})
        .build(ctx, cgmlp, weights.cgmlp.input_projection);
    cgmlp = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, cgmlp);
    auto value = modules::SliceModule({2, 0, intermediate_size / 2}).build(ctx, cgmlp);
    auto gate = modules::SliceModule({2, intermediate_size / 2, intermediate_size / 2}).build(ctx, cgmlp);
    gate = modules::LayerNormModule({intermediate_size / 2, kLayerNormEps})
        .build(ctx, gate, weights.cgmlp.gate_norm);
    gate = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, gate);
    gate = modules::DepthwiseConv1dModule({intermediate_size / 2, 31, 1, 15, 1, true})
        .build(ctx, gate, weights.cgmlp.gate_conv);
    gate = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, gate);
    cgmlp = modules::MulModule().build(ctx, value, gate);
    cgmlp = modules::LinearModule({intermediate_size / 2, hidden_size, true})
        .build(ctx, cgmlp, weights.cgmlp.output_projection);

    auto merged = modules::ConcatModule({2}).build(ctx, attention, cgmlp);
    auto convolved = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, merged);
    convolved = modules::DepthwiseConv1dModule({hidden_size * 2, 31, 1, 15, 1, true})
        .build(ctx, convolved, weights.merge_conv);
    convolved = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, convolved);
    merged = modules::AddModule().build(ctx, merged, convolved);
    merged = modules::LinearModule({hidden_size * 2, hidden_size, true})
        .build(ctx, merged, weights.merge_projection);
    x = residual.build(ctx, x, merged);

    transformed = norm.build(ctx, x, weights.final_ffn_norm);
    transformed = modules::LinearModule({hidden_size, intermediate_size, true}).build(
        ctx, transformed, {weights.final_ffn.fc1_weight, weights.final_ffn.fc1_bias});
    transformed = modules::SiluModule().build(ctx, transformed);
    transformed = modules::LinearModule({intermediate_size, hidden_size, true}).build(
        ctx, transformed, {weights.final_ffn.fc2_weight, weights.final_ffn.fc2_bias});
    transformed = scale.build(ctx, transformed, {half_scale});
    x = residual.build(ctx, x, transformed);
    if (memory) {
        transformed = norm.build(ctx, x, weights.cross_attention_norm);
        modules::AttentionConfig cross_config{hidden_size, num_heads, true};
        cross_config.use_packed_kv = true;
        transformed = modules::CrossAttentionModule(cross_config)
            .build(ctx, transformed, *memory, weights.cross_attention, *memory_mask);
        x = residual.build(ctx, x, transformed);
    }
    return norm.build(ctx, x, weights.output_norm);
}

}  // namespace

struct OWSMCTCV4EBranchformerRuntime::Graphs {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer{
        nullptr, ggml_backend_buffer_free};
    Graph encoder;
    TensorValue features;
    TensorValue prefix;
    TensorValue prompt;
    TensorValue positions;
    TensorValue prompt_position;
    TensorValue memory_mask;
    TensorValue tokens;

    Graphs(core::ExecutionContext & execution, const OWSMCTCV4Config & config,
           const OWSMCTCV4Weights & weights) : encoder(execution, 32 * 1024 * 1024) {
        state_context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!state_context) {
            throw std::runtime_error("OWSM-CTC v4 state context allocation failed");
        }
        core::ModuleBuildContext ctx{};
        ctx.ggml = state_context.get();
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "owsm_ctc.ctc_encoder";
        const auto d = config.hidden_size;
        const auto p = config.prompt_hidden_size;
        const auto frames = config.encoder_frames + 2;
        features = core::make_tensor(ctx, GGML_TYPE_F32,
            TensorShape::from_dims({1, config.frontend_frames, 128}));
        prefix = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 2}));
        prompt = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        positions = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, d}));
        prompt_position = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, p}));
        memory_mask = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        for (auto input : {features, prefix, prompt, positions, prompt_position, memory_mask}) {
            ggml_set_input(input.tensor);
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_context.get(), execution.backend()));
        if (!state_buffer) {
            throw std::runtime_error("OWSM-CTC v4 state allocation failed");
        }
        ctx.ggml = encoder.context;

        const modules::EmbeddingModule embedding({config.vocabulary_size, p});
        auto memory = embedding.build(ctx, prompt, {weights.embedding});
        memory = modules::LayerScaleModule().build(ctx, memory, {weights.prompt_scale});
        memory = modules::AddModule().build(ctx, memory, prompt_position);
        modules::AttentionConfig prompt_attention{p, config.prompt_num_heads, true};
        prompt_attention.use_packed_qkv = true;
        modules::FeedForwardConfig prompt_ff{p, config.prompt_intermediate_size, true};
        prompt_ff.activation = modules::FeedForwardActivation::Relu;
        const modules::LayerNormModule prompt_norm({p, kLayerNormEps});
        for (const auto & layer : weights.prompt_encoder) {
            auto x = prompt_norm.build(ctx, memory, layer.norm1);
            x = modules::SelfAttentionModule(prompt_attention).build(ctx, x, layer.self_attention);
            memory = modules::ResidualAddModule().build(ctx, memory, x);
            x = prompt_norm.build(ctx, memory, layer.norm2);
            x = modules::FeedForwardModule(prompt_ff).build(ctx, x, layer.feed_forward);
            memory = modules::ResidualAddModule().build(ctx, memory, x);
        }
        memory = prompt_norm.build(ctx, memory, weights.prompt_norm);
        memory = modules::LinearModule({p, d, true}).build(ctx, memory, weights.prompt_projection);

        auto x = core::reshape_tensor(ctx, features,
            TensorShape::from_dims({1, 1, config.frontend_frames, 128}));
        x = modules::Conv2dModule({1, d, 3, 3, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.conv0);
        x = modules::ReluModule().build(ctx, x);
        for (const auto * conv : {&weights.subsampling.conv1, &weights.subsampling.conv2}) {
            const auto output_frames = (x.shape.dims[2] - 3) / 2 + 1;
            const bool align_cpu = execution.backend_type() == core::BackendType::Cpu;
            // Align the im2col row count for CPU GEMM; discard only added output rows.
            const auto padded_frames = align_cpu ? ((output_frames + 15) / 16) * 16 : output_frames;
            if (padded_frames != output_frames) {
                x = modules::Pad2dModule({0, 0, 0, 2 * (padded_frames - output_frames)})
                    .build(ctx, x);
            }
            x = modules::Conv2dModule({d, d, 3, 3, 2, 2, 0, 0, 1, 1, true}).build(ctx, x, *conv);
            if (padded_frames != output_frames) {
                x = modules::SliceModule({2, 0, output_frames}).build(ctx, x);
            }
            x = modules::ReluModule().build(ctx, x);
        }
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, TensorShape::from_dims({1, config.encoder_frames, d * 15}));
        x = modules::LinearModule({d * 15, d, true}).build(ctx, x, weights.subsampling.projection);
        auto prefix_embedding = embedding.build(ctx, prefix, {weights.embedding});
        prefix_embedding = modules::LinearModule({p, d, true})
            .build(ctx, prefix_embedding, weights.prefix_projection);
        x = modules::ConcatModule({1}).build(ctx, prefix_embedding, x);
        x = modules::LayerScaleModule().build(ctx, x, {weights.embedding_scale});
        x = modules::AddModule().build(ctx, x, positions);
        const auto output_rows = weights.output.weight.shape.dims[0];
        const modules::LinearModule ctc_head({d, output_rows, true});
        for (int64_t index = 0; index < config.encoder_layers; ++index) {
            const bool cross = std::find(config.cross_attention_layers.begin(),
                config.cross_attention_layers.end(), index) != config.cross_attention_layers.end();
            x = build_ebranchformer_layer(ctx, x, weights.encoder[static_cast<size_t>(index)],
                d, config.num_heads, config.intermediate_size, weights.half_scale,
                cross ? &memory : nullptr, cross ? &memory_mask : nullptr);
            if (std::find(config.interctc_layers.begin(), config.interctc_layers.end(), index + 1) !=
                config.interctc_layers.end()) {
                auto logits = ctc_head.build(ctx, x, weights.output);
                if (output_rows != config.vocabulary_size) {
                    logits = modules::SliceModule({2, 0, config.vocabulary_size}).build(ctx, logits);
                }
                auto probabilities = modules::SoftmaxModule().build(ctx, logits);
                const auto conditioning_width = weights.ctc_conditioning.weight.shape.last_dim();
                if (conditioning_width != config.vocabulary_size) {
                    probabilities = core::reshape_tensor(ctx, probabilities,
                        TensorShape::from_dims({1, 1, frames, config.vocabulary_size}));
                    probabilities = modules::Pad2dModule({0, conditioning_width - config.vocabulary_size, 0, 0})
                        .build(ctx, probabilities);
                    probabilities = core::reshape_tensor(ctx, probabilities,
                        TensorShape::from_dims({1, frames, conditioning_width}));
                }
                auto conditioning = modules::LinearModule({conditioning_width, d, true})
                    .build(ctx, probabilities, weights.ctc_conditioning);
                x = modules::AddModule().build(ctx, x, conditioning);
            }
        }
        x = modules::LayerNormModule({d, kLayerNormEps}).build(ctx, x, weights.encoder_norm);
        auto logits = ctc_head.build(ctx, x, weights.output);
        if (output_rows != config.vocabulary_size) {
            logits = modules::SliceModule({2, 0, config.vocabulary_size}).build(ctx, logits);
        }
        tokens = sampling::GreedyDecodeModule().build(ctx, logits);
        ggml_set_output(tokens.tensor);
        ggml_build_forward_expand(encoder.graph, tokens.tensor);
        encoder.allocate();
        std::vector<float> pe(static_cast<size_t>(frames * d));
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t channel = 0; channel < d; channel += 2) {
                const auto phase = static_cast<double>(t) *
                    std::exp(static_cast<double>(channel) * -(std::log(10000.0) / static_cast<double>(d)));
                pe[static_cast<size_t>(t * d + channel)] = static_cast<float>(std::sin(phase));
                pe[static_cast<size_t>(t * d + channel + 1)] = static_cast<float>(std::cos(phase));
            }
        }
        core::write_tensor_f32(positions, pe);
        std::vector<float> prompt_pe(static_cast<size_t>(p), 0.0F);
        for (int64_t channel = 1; channel < p; channel += 2) {
            prompt_pe[static_cast<size_t>(channel)] = 1.0F;
        }
        core::write_tensor_f32(prompt_position, prompt_pe);
        core::write_tensor_i32(memory_mask, std::vector<int32_t>{1});
    }
};

OWSMCTCV4EBranchformerRuntime::OWSMCTCV4EBranchformerRuntime(const OWSMCTCV4Assets & assets, const OWSMCTCV4Weights & weights,
                                  core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution) {}

OWSMCTCV4EBranchformerRuntime::~OWSMCTCV4EBranchformerRuntime() = default;

std::vector<int32_t> OWSMCTCV4EBranchformerRuntime::frame_tokens(
    const std::vector<float> & samples, int32_t language, int32_t task) {
    if (samples.empty() || samples.size() > static_cast<size_t>(assets_.config.max_audio_samples)) {
        throw std::runtime_error("OWSM-CTC v4 requires between one sample and 30 seconds of 16 kHz audio");
    }
    if (!graphs_) {
        graphs_ = std::make_unique<Graphs>(execution_, assets_.config, weights_);
    }
    const auto started = std::chrono::steady_clock::now();
    auto padded = samples;
    padded.resize(static_cast<size_t>(assets_.config.max_audio_samples), 0.0F);
    auto features = assets_.frontend->extract_mono(
        padded, static_cast<size_t>(std::max<int64_t>(1, execution_.config().threads)));
    if (features.frames != assets_.config.frontend_frames || features.mel_bins != 128) {
        throw std::runtime_error("OWSM-CTC v4 frontend produced unexpected feature geometry");
    }
    for (int64_t frame = 0; frame < features.frames; ++frame) {
        for (int64_t mel = 0; mel < features.mel_bins; ++mel) {
            auto & value = features.values[static_cast<size_t>(frame * features.mel_bins + mel)];
            value = (value - assets_.feature_mean[static_cast<size_t>(mel)]) /
                assets_.feature_std[static_cast<size_t>(mel)];
        }
    }
    debug::timing_log_scalar("owsm_ctc.frontend_ms", debug::elapsed_ms(started));
    core::write_tensor_f32(graphs_->features, features.values);
    core::write_tensor_i32(graphs_->prefix, std::vector<int32_t>{language, task});
    core::write_tensor_i32(graphs_->prompt, std::vector<int32_t>{assets_.token_id("<na>")});
    const auto encoder_started = std::chrono::steady_clock::now();
    graphs_->encoder.compute();
    auto tokens = core::read_tensor_i32(graphs_->tokens.tensor);
    debug::timing_log_scalar("owsm_ctc.encoder_ms", debug::elapsed_ms(encoder_started));
    return tokens;
}

}  // namespace engine::models::owsm_ctc
