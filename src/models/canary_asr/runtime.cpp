#include "engine/models/canary_asr/model.h"
#include "engine/framework/audio/nemo_mel_frontend.h"

#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/attention/relative_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/bounded_static_kv_decode.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine::models::canary_asr {
namespace {

using core::TensorShape;
using core::TensorValue;
constexpr int64_t kCacheSteps = 1024;
constexpr size_t kGraphNodes = 65536;

struct Graph {
    core::ExecutionContext & execution;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan plan;

    explicit Graph(core::ExecutionContext & execution) : execution(execution) {
        context = ggml_init({32 * 1024 * 1024, nullptr, true});
        if (!context) {
            throw std::runtime_error("Canary graph context allocation failed");
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
        if (execution.backend_type() == core::BackendType::Cpu) {
            // Fold explicit parameter repeats into the CPU kernels' native broadcasting.
            auto options = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Other);
            options.backend = runtime::GraphOptimizationBackend::Cpu;
            options.fold_broadcast_repeats = true;
            runtime::optimize_graph(*graph, options);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "Canary");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("Canary graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "Canary") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Canary graph execution failed");
        }
    }
};

}  // namespace

struct CanaryRuntime::Graphs {
    core::ExecutionContext & execution;
    int64_t frames;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer{nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> keys, values;
    std::vector<modules::CrossAttentionKeyValue> cross;
    runtime::TransformerKVCache cache;
    runtime::BoundedStaticKVDecodeCursor cursor;
    Graph encoder, decoder;
    TensorValue features, pos, mask, keep, stage1_keep, stage2_keep;
    TensorValue token, position, slot, causal_mask, memory_mask, logits;

    Graphs(core::ExecutionContext & execution, const CanaryWeights & weights, int64_t frames)
        : execution(execution), frames(frames), encoder(execution), decoder(execution) {
        state_context.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!state_context) {
            throw std::runtime_error("Canary cache context allocation failed");
        }
        core::ModuleBuildContext state_ctx{ };
        state_ctx.ggml = state_context.get();
        state_ctx.backend_type = execution.backend_type();
        features = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames * 8, 128}));
        pos = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 2 * frames - 1, 512}));
        mask = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({frames, frames}));
        keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames}));
        stage1_keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames * 4}));
        stage2_keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames * 2}));
        token = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        position = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        slot = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        causal_mask = core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({1, kCacheSteps}));
        memory_mask = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames}));
        for (size_t layer = 0; layer < weights.decoder.size(); ++layer) {
            keys.push_back(core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({1, kCacheSteps, 8, 128})));
            values.push_back(core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({1, kCacheSteps, 8, 128})));
            cross.push_back({
                core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 8, frames, 128})),
                core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 8, frames, 128}))});
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_context.get(), execution.backend()));
        if (!state_buffer) {
            throw std::runtime_error("Canary KV allocation failed");
        }
        cache = runtime::TransformerKVCache(kCacheSteps, 1024, keys, values, {true, false});
        build_encoder(weights);
        build_decoder(weights);
    }

    ~Graphs() {
        // Graphs are destroyed before their external KV storage.
        ggml_backend_synchronize(execution.backend());
    }

    void build_encoder(const CanaryWeights & w) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = encoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "canary.encoder";
        for (auto input : {features, pos, mask, keep, stage1_keep, stage2_keep}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::DepthwiseConvSubsamplingModule({128, 512, 256})
            .build(ctx, features, w.subsampling, {stage1_keep, stage2_keep, keep});
        modules::ConformerBlockConfig encoder_config{512, 8, 2048, 9};
        encoder_config.contiguous_glu_gate = true;
        for (const auto & layer : w.encoder) {
            x = modules::RelativeConformerBlockModule(encoder_config)
                .build(ctx, x, pos, layer, mask, keep, keep);
        }
        x = modules::LinearModule({512, 1024, true}).build(ctx, x, w.encoder_out);
        modules::AttentionConfig cross_config{1024, 8, true};
        cross_config.use_packed_kv = true;
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            const auto projected = modules::CrossAttentionModule(cross_config)
                .build_key_value(ctx, x, w.decoder[i].cross_attention);
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, projected.key.tensor, cross[i].key.tensor));
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, projected.value.tensor, cross[i].value.tensor));
        }
        encoder.allocate();
        std::vector<float> positions(static_cast<size_t>((2 * frames - 1) * 512));
        for (int64_t p = 0; p < 2 * frames - 1; ++p) {
            for (int64_t i = 0; i < 256; ++i) {
                const double phase = (frames - 1 - p) * std::pow(10000.0, -2.0 * i / 512.0);
                positions[static_cast<size_t>(p * 512 + 2 * i)] = static_cast<float>(std::sin(phase));
                positions[static_cast<size_t>(p * 512 + 2 * i + 1)] = static_cast<float>(std::cos(phase));
            }
        }
        core::write_tensor_f32(pos, positions);
    }

    void build_decoder(const CanaryWeights & w) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = decoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "canary.decoder";
        for (auto input : {token, position, slot, causal_mask, memory_mask}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::EmbeddingModule({5248, 1024}).build(ctx, token, w.embedding);
        auto p = modules::EmbeddingModule({1024, 1024}).build(ctx, position, w.positions);
        x = modules::AddModule().build(ctx, x, p);
        x = modules::LayerNormModule({1024}).build(ctx, x, w.embedding_norm);
        modules::TransformerDecoderBlockConfig decoder_config{1024, 8, 4096};
        decoder_config.activation = modules::FeedForwardActivation::Relu;
        decoder_config.use_packed_qkv = true;
        decoder_config.use_packed_kv = true;
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            x = modules::TransformerDecoderBlockModule(decoder_config).build_cached_tail(ctx, x, w.decoder[i],
                keys[i], values[i], slot, causal_mask, cross[i], memory_mask);
        }
        x = modules::LayerNormModule({1024}).build(ctx, x, w.decoder_norm);
        logits = modules::LinearModule({1024, 5248, true}).build(ctx, x, w.head);
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(decoder.graph, logits.tensor);
        decoder.allocate();
    }
};

CanaryRuntime::CanaryRuntime(const CanaryAssets & assets, const CanaryWeights & weights, core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution) {}

CanaryRuntime::~CanaryRuntime() = default;

CanaryFrontendFeatures extract_canary_frontend(
    const std::vector<float> & samples,
    const CanaryAssets & assets,
    size_t threads) {
    auto features = assets.frontend->extract_mono(
        samples, {true, audio::ValidFrameRule::FloorHops}, threads);
    return {std::move(features.values), features.raw_frames, features.valid_frames};
}

std::vector<int32_t> CanaryRuntime::transcribe(const std::vector<float> & samples,
    const std::vector<int32_t> & prompt, int64_t max_tokens) {
    if (samples.size() < 320 || samples.size() > 40 * 16000) {
        throw std::runtime_error("Canary chunks must contain between 20 ms and 40 seconds of audio");
    }
    const auto features = extract_canary_frontend(samples, assets_, static_cast<size_t>(execution_.config().threads));
    const int64_t valid = features.valid_frames;
    const int64_t raw_frames = features.raw_frames;
    const int64_t required = (raw_frames + 7) / 8;
    if (!graphs_ || graphs_->frames < required) {
        graphs_ = std::make_unique<Graphs>(execution_, weights_, ((required + 31) / 32) * 32);
    }
    auto & g = *graphs_;
    std::vector<float> input(static_cast<size_t>(g.frames * 8 * 128), 0.0f);
    for (int64_t t = 0; t < valid; ++t) {
        for (int64_t m = 0; m < 128; ++m) {
            input[static_cast<size_t>(t * 128 + m)] = features.values[static_cast<size_t>(m * raw_frames + t)];
        }
    }
    const int64_t valid_encoded = (valid + 7) / 8;
    std::vector<int32_t> keep(static_cast<size_t>(g.frames), 0);
    std::fill_n(keep.begin(), valid_encoded, 1);
    std::vector<int32_t> mask1(static_cast<size_t>(g.frames * 4), 0), mask2(static_cast<size_t>(g.frames * 2), 0);
    std::fill_n(mask1.begin(), (valid + 1) / 2, 1);
    std::fill_n(mask2.begin(), (valid + 3) / 4, 1);
    std::vector<float> mask(static_cast<size_t>(g.frames * g.frames), -std::numeric_limits<float>::infinity());
    for (int64_t q = 0; q < g.frames; ++q) {
        std::fill_n(mask.begin() + q * g.frames, valid_encoded, 0.0f);
    }
    core::write_tensor_f32(g.features, input);
    core::write_tensor_f32(g.mask, mask);
    core::write_tensor_i32(g.keep, keep);
    core::write_tensor_i32(g.stage1_keep, mask1);
    core::write_tensor_i32(g.stage2_keep, mask2);
    g.encoder.compute();
    core::write_tensor_i32(g.memory_mask, keep);
    g.cache.clear_on_backend();
    g.cursor.reset_to_empty(kCacheSteps);
    std::vector<float> causal(static_cast<size_t>(kCacheSteps), -std::numeric_limits<float>::infinity());
    std::vector<int32_t> generated;
    int32_t next = 0;
    const int32_t eos = assets_.special_token("<|endoftext|>");
    const int32_t pad = assets_.special_token("<pad>");
    const int64_t limit = max_tokens > 0 ? max_tokens : valid_encoded + 50;
    if (prompt.empty() || static_cast<int64_t>(prompt.size()) + limit > kCacheSteps) {
        throw std::runtime_error("Canary prompt and generation exceed the 1024-token context");
    }
    std::vector<float> logits;
    for (int64_t step = 0; step < static_cast<int64_t>(prompt.size()) + limit - 1; ++step) {
        const auto cursor = g.cursor.next_step();
        const int32_t position = static_cast<int32_t>(cursor.position);
        const int32_t input_token = step < static_cast<int64_t>(prompt.size()) ? prompt[step] : next;
        causal[static_cast<size_t>(position)] = input_token == pad ? -std::numeric_limits<float>::infinity() : 0.0f;
        core::write_tensor_i32(g.token, &input_token, 1);
        core::write_tensor_i32(g.position, &position, 1);
        core::write_tensor_i32(g.slot, &cursor.cache_slot, 1);
        core::write_tensor_f16(g.causal_mask, causal);
        g.decoder.compute();
        g.cache.advance_after_direct_append(1);
        g.cursor.advance_after_direct_append(1);
        if (step + 1 < static_cast<int64_t>(prompt.size())) {
            continue;
        }
        core::read_tensor_f32_into(g.logits.tensor, logits);
        next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (!std::isfinite(logits[static_cast<size_t>(next)])) {
            throw std::runtime_error("Canary decoder produced non-finite logits");
        }
        if (next == eos || next == pad) {
            return generated;
        }
        generated.push_back(next);
    }
    throw std::runtime_error("Canary reached max_tokens before EOS");
}

}  // namespace engine::models::canary_asr
