#include "engine/models/cohere_asr/model.h"
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
#include "engine/framework/sampling/hf_sampler.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <stdexcept>

namespace engine::models::cohere_asr {
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
            throw std::runtime_error("Cohere graph context allocation failed");
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
        } else if (execution.backend_type() == core::BackendType::Cuda) {
            runtime::optimize_graph(*graph, runtime::GraphOptimizationBackend::Gpu);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "Cohere");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("Cohere graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "Cohere") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Cohere graph execution failed");
        }
    }
};

}  // namespace

struct CohereRuntime::Graphs {
    core::ExecutionContext & execution;
    int64_t frames;
    int64_t batch;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer{nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> keys, values;
    std::vector<TensorValue> projected_positions;
    std::vector<modules::CrossAttentionKeyValue> cross;
    runtime::BoundedStaticKVDecodeCursor cursor;
    Graph encoder, decoder;
    TensorValue features, pos, mask, keep, stage1_keep, stage2_keep;
    TensorValue token, position, slot, causal_mask, memory_mask, cross_mask, logits;

    Graphs(core::ExecutionContext & execution, const CohereWeights & weights, int64_t frames, int64_t batch)
        : execution(execution), frames(frames), batch(batch), encoder(execution), decoder(execution) {
        state_context.reset(ggml_init({1024 * 1024, nullptr, true}));
        if (!state_context) {
            throw std::runtime_error("Cohere cache context allocation failed");
        }
        core::ModuleBuildContext state_ctx{ };
        state_ctx.ggml = state_context.get();
        state_ctx.backend_type = execution.backend_type();
        features = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, frames * 8, 128}));
        pos = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 2 * frames - 1, 1280}));
        mask = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 1, frames, frames}));
        keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, frames}));
        stage1_keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, frames * 4}));
        stage2_keep = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, frames * 2}));
        token = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, 1}));
        position = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, 1}));
        slot = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch}));
        causal_mask = core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({1, kCacheSteps}));
        memory_mask = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, frames}));
        if (execution.backend_type() == core::BackendType::Cuda) {
            cross_mask = core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({batch, 1, 1, frames}));
            for (size_t layer = 0; layer < weights.encoder.size(); ++layer) {
                projected_positions.push_back(core::make_tensor(state_ctx, GGML_TYPE_F32,
                    TensorShape::from_dims({1, 2 * frames - 1, 1280})));
            }
        }
        const auto cross_type = execution.backend_type() == core::BackendType::Cuda ? GGML_TYPE_F16 : GGML_TYPE_F32;
        for (size_t layer = 0; layer < weights.decoder.size(); ++layer) {
            keys.push_back(core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({batch, kCacheSteps, 8, 128})));
            values.push_back(core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({batch, kCacheSteps, 8, 128})));
            cross.push_back({
                core::make_tensor(state_ctx, cross_type, TensorShape::from_dims({batch, 8, frames, 128})),
                core::make_tensor(state_ctx, cross_type, TensorShape::from_dims({batch, 8, frames, 128}))});
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_context.get(), execution.backend()));
        if (!state_buffer) {
            throw std::runtime_error("Cohere KV allocation failed");
        }
        initialize_positions(weights);
        build_encoder(weights);
        build_decoder(weights);
    }

    ~Graphs() {
        // Graphs are destroyed before their external KV storage.
        ggml_backend_synchronize(execution.backend());
    }

    void initialize_positions(const CohereWeights & w) {
        std::vector<float> positions(static_cast<size_t>((2 * frames - 1) * 1280));
        for (int64_t p = 0; p < 2 * frames - 1; ++p) {
            for (int64_t i = 0; i < 640; ++i) {
                const double phase = (frames - 1 - p) * std::pow(10000.0, -2.0 * i / 1280.0);
                positions[static_cast<size_t>(p * 1280 + 2 * i)] = static_cast<float>(std::sin(phase));
                positions[static_cast<size_t>(p * 1280 + 2 * i + 1)] = static_cast<float>(std::cos(phase));
            }
        }
        core::write_tensor_f32(pos, positions);
        if (projected_positions.empty()) {
            return;
        }
        // These projections depend only on weights and the cached graph's frame count.
        Graph projection(execution);
        core::ModuleBuildContext ctx{};
        ctx.ggml = projection.context;
        ctx.backend_type = execution.backend_type();
        for (size_t i = 0; i < w.encoder.size(); ++i) {
            const auto p = modules::LinearModule({1280, 1280, false}).build(ctx, pos,
                {w.encoder[i].self_attention.pos_weight, std::nullopt});
            ggml_build_forward_expand(projection.graph,
                ggml_cpy(ctx.ggml, p.tensor, projected_positions[i].tensor));
        }
        projection.allocate();
        projection.compute();
        ggml_backend_synchronize(execution.backend());
    }

    void build_encoder(const CohereWeights & w) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = encoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "cohere.encoder";
        for (auto input : {features, pos, mask, keep, stage1_keep, stage2_keep}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::DepthwiseConvSubsamplingModule({128, 1280, 256})
            .build(ctx, features, w.subsampling, {stage1_keep, stage2_keep, keep});
        modules::ConformerBlockConfig encoder_config{1280, 8, 5120, 9};
        encoder_config.contiguous_glu_gate = true;
        for (size_t i = 0; i < w.encoder.size(); ++i) {
            x = modules::RelativeConformerBlockModule(encoder_config)
                .build(ctx, x, pos, w.encoder[i], mask, keep, keep,
                    projected_positions.empty() ? std::nullopt : std::optional<TensorValue>(projected_positions[i]));
        }
        x = modules::LinearModule({1280, 1024, true}).build(ctx, x, w.encoder_out);
        modules::AttentionConfig cross_config{1024, 8, true};
        cross_config.use_packed_kv = true;
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            const auto projected = modules::CrossAttentionModule(cross_config)
                .build_key_value(ctx, x, w.decoder[i].cross_attention);
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, projected.key.tensor, cross[i].key.tensor));
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, projected.value.tensor, cross[i].value.tensor));
        }
        encoder.allocate();
    }

    void build_decoder(const CohereWeights & w) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = decoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "cohere.decoder";
        for (auto input : {token, position, slot, causal_mask, memory_mask}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::EmbeddingModule({16384, 1024}).build(ctx, token, w.embedding);
        auto p = modules::EmbeddingModule({1024, 1024}).build(ctx, position, w.positions);
        x = modules::AddModule().build(ctx, x, p);
        x = modules::LayerNormModule({1024}).build(ctx, x, w.embedding_norm);
        modules::TransformerDecoderBlockConfig decoder_config{1024, 8, 4096};
        decoder_config.activation = modules::FeedForwardActivation::Relu;
        decoder_config.use_packed_qkv = true;
        decoder_config.use_packed_kv = true;
        const bool flash_cross = execution.backend_type() == core::BackendType::Cuda;
        decoder_config.use_flash_cross_attention = flash_cross;
        if (flash_cross) {
            ggml_set_input(cross_mask.tensor);
        }
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            x = modules::TransformerDecoderBlockModule(decoder_config).build_cached_tail(ctx, x, w.decoder[i],
                keys[i], values[i], slot, causal_mask, cross[i], flash_cross ? cross_mask : memory_mask);
        }
        x = modules::LayerNormModule({1024}).build(ctx, x, w.decoder_norm);
        logits = modules::LinearModule({1024, 16384, true}).build(ctx, x, w.head);
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(decoder.graph, logits.tensor);
        decoder.allocate();
    }
};

CohereRuntime::CohereRuntime(const CohereAssets & assets, const CohereWeights & weights, core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution), graphs_(3) {}

CohereRuntime::~CohereRuntime() = default;

namespace {

audio::AudioTensor extract_features(const std::vector<float> & samples, const CohereAssets & assets, size_t threads) {
    if (samples.size() < 320 || samples.size() > 35 * 16000) {
        throw std::runtime_error("Cohere chunks must contain between 20 ms and 35 seconds of audio");
    }
    auto features = assets.frontend->extract_mono(
        samples, {true, audio::ValidFrameRule::FloorHops}, threads);
    return {std::move(features.values), {1, features.feature_size, features.frames}};
}

}  // namespace

audio::AudioTensor extract_cohere_frontend(
    const std::vector<float> & samples,
    const CohereAssets & assets,
    size_t threads) {
    return extract_features(samples, assets, threads);
}

std::vector<std::vector<int32_t>> CohereRuntime::transcribe(
    const std::vector<std::vector<float>> & samples, const std::vector<int32_t> & prompt, int64_t max_tokens) {
    const int64_t batch = static_cast<int64_t>(samples.size());
    if (batch < 1 || batch > 8 || prompt.empty() || max_tokens < 1 ||
        static_cast<int64_t>(prompt.size()) + max_tokens > kCacheSteps) {
        throw std::runtime_error("Cohere requires 1-8 chunks and a prompt/generation fitting the 1024-token context");
    }
    std::vector<audio::AudioTensor> features(static_cast<size_t>(batch));
    const size_t threads = static_cast<size_t>(execution_.config().threads);
    const size_t workers = execution_.backend_type() == core::BackendType::Cuda
        ? std::min(static_cast<size_t>(batch), std::max(size_t{1}, threads)) : 1;
    const auto extract = [&](size_t worker) {
        for (size_t b = worker; b < samples.size(); b += workers) {
            features[b] = extract_cohere_frontend(samples[b], assets_, std::max(size_t{1}, threads / workers));
        }
    };
    std::vector<std::future<void>> pending;
    for (size_t worker = 1; worker < workers; ++worker) {
        pending.push_back(std::async(std::launch::async, extract, worker));
    }
    extract(0);
    for (auto & task : pending) {
        task.get();
    }
    int64_t required = 0;
    for (const auto & feature : features) {
        required = std::max(required, (feature.shape[2] + 7) / 8);
    }
    const int64_t frames = ((required + 31) / 32) * 32;
    const auto key = std::make_pair(batch, frames);
    auto * entry = graphs_.find(key);
    if (!entry) {
        graphs_.put(key, std::make_unique<Graphs>(execution_, weights_, frames, batch));
        entry = graphs_.find(key);
    }
    auto & g = **entry;
    std::vector<float> input(static_cast<size_t>(batch * frames * 8 * 128), 0.0f);
    std::vector<int32_t> keep(static_cast<size_t>(batch * frames), 0);
    std::vector<int32_t> mask1(static_cast<size_t>(batch * frames * 4), 0);
    std::vector<int32_t> mask2(static_cast<size_t>(batch * frames * 2), 0);
    std::vector<float> mask(static_cast<size_t>(batch * frames * frames), -std::numeric_limits<float>::infinity());
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t valid = static_cast<int64_t>(samples[b].size()) / 160;
        const int64_t valid_encoded = (valid + 7) / 8;
        const auto & mel = features[b];
        for (int64_t t = 0; t < valid; ++t) {
            for (int64_t m = 0; m < 128; ++m) {
                input[static_cast<size_t>((b * frames * 8 + t) * 128 + m)] =
                    mel.values[static_cast<size_t>(m * mel.shape[2] + t)];
            }
        }
        std::fill_n(keep.begin() + b * frames, valid_encoded, 1);
        std::fill_n(mask1.begin() + b * frames * 4, (valid + 1) / 2, 1);
        std::fill_n(mask2.begin() + b * frames * 2, (valid + 3) / 4, 1);
        for (int64_t q = 0; q < frames; ++q) {
            std::fill_n(mask.begin() + (b * frames + q) * frames, valid_encoded, 0.0f);
        }
    }
    core::write_tensor_f32(g.features, input);
    core::write_tensor_f32(g.mask, mask);
    core::write_tensor_i32(g.keep, keep);
    core::write_tensor_i32(g.stage1_keep, mask1);
    core::write_tensor_i32(g.stage2_keep, mask2);
    g.encoder.compute();
    core::write_tensor_i32(g.memory_mask, keep);
    if (execution_.backend_type() == core::BackendType::Cuda) {
        std::vector<float> cross_mask(keep.size());
        std::transform(keep.begin(), keep.end(), cross_mask.begin(), [](int32_t valid) {
            return valid ? 0.0f : -std::numeric_limits<float>::infinity();
        });
        core::write_tensor_f16(g.cross_mask, cross_mask);
    }
    for (size_t layer = 0; layer < g.keys.size(); ++layer) {
        ggml_backend_tensor_memset(g.keys[layer].tensor, 0, 0, ggml_nbytes(g.keys[layer].tensor));
        ggml_backend_tensor_memset(g.values[layer].tensor, 0, 0, ggml_nbytes(g.values[layer].tensor));
    }
    g.cursor.reset_to_empty(kCacheSteps);
    std::vector<float> causal(static_cast<size_t>(kCacheSteps), -std::numeric_limits<float>::infinity());
    std::vector<std::vector<int32_t>> generated(static_cast<size_t>(batch));
    std::vector<int32_t> next(static_cast<size_t>(batch), 0), tokens(batch), positions(batch), slots(batch);
    std::vector<bool> finished(static_cast<size_t>(batch), false);
    const int32_t eos = assets_.special_token("<|endoftext|>");
    std::vector<float> logits;
    for (int64_t step = 0; step < static_cast<int64_t>(prompt.size()) + max_tokens - 1; ++step) {
        const auto cursor = g.cursor.next_step();
        const int32_t position = static_cast<int32_t>(cursor.position);
        causal[static_cast<size_t>(position)] = 0.0f;
        for (int64_t b = 0; b < batch; ++b) {
            tokens[b] = step < static_cast<int64_t>(prompt.size()) ? prompt[step] : next[b];
            positions[b] = position;
            slots[b] = static_cast<int32_t>(b * kCacheSteps + cursor.cache_slot);
        }
        core::write_tensor_i32(g.token, tokens);
        core::write_tensor_i32(g.position, positions);
        core::write_tensor_i32(g.slot, slots);
        core::write_tensor_f16(g.causal_mask, causal);
        g.decoder.compute();
        g.cursor.advance_after_direct_append(1);
        if (step + 1 < static_cast<int64_t>(prompt.size())) {
            continue;
        }
        core::read_tensor_f32_into(g.logits.tensor, logits);
        for (int64_t b = 0; b < batch; ++b) {
            if (finished[b]) {
                continue;
            }
            const auto * row = logits.data() + b * 16384;
            next[b] = sampling::HfLogitsProcessor::argmax(row, 16384, "Cohere");
            if (!std::isfinite(row[next[b]])) {
                throw std::runtime_error("Cohere decoder produced non-finite logits");
            }
            if (next[b] == eos) {
                finished[b] = true;
            } else {
                generated[b].push_back(next[b]);
            }
        }
        if (std::all_of(finished.begin(), finished.end(), [](bool done) { return done; })) {
            return generated;
        }
    }
    throw std::runtime_error("Cohere reached max_tokens before EOS");
}

}  // namespace engine::models::cohere_asr
