#include "engine/models/owsm/model.h"

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
#include "engine/framework/runtime/bounded_static_kv_decode.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::owsm {
namespace {

using core::TensorShape;
using core::TensorValue;
constexpr int64_t kCacheSteps = 512;
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
            throw std::runtime_error("OWSM v4 graph context allocation failed");
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
        core::validate_backend_graph_supported(execution.backend(), graph, "OWSM v4");
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("OWSM v4 graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "OWSM v4") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("OWSM v4 graph execution failed");
        }
    }
};

}  // namespace

struct OWSMV4Runtime::Graphs {
    core::ExecutionContext & execution;
    const OWSMV4Config & config;
    const int64_t cache_steps;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state_context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> state_buffer{
        nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> keys;
    std::vector<TensorValue> values;
    std::vector<modules::CrossAttentionKeyValue> cross;
    runtime::TransformerKVCache cache;
    runtime::BoundedStaticKVDecodeCursor cursor;
    Graph encoder;
    Graph decoder;
    TensorValue features;
    TensorValue encoder_positions;
    TensorValue token;
    TensorValue decoder_position;
    TensorValue slot;
    TensorValue causal_mask;
    TensorValue memory_mask;
    TensorValue logits;

    Graphs(core::ExecutionContext & execution, const OWSMV4Config & config,
           const OWSMV4Weights & weights, int64_t cache_steps)
        : execution(execution), config(config), cache_steps(cache_steps), encoder(execution, 32 * 1024 * 1024),
          decoder(execution, 16 * 1024 * 1024) {
        state_context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!state_context) {
            throw std::runtime_error("OWSM v4 state context allocation failed");
        }
        core::ModuleBuildContext state_ctx{};
        state_ctx.ggml = state_context.get();
        state_ctx.backend_type = execution.backend_type();
        const auto d = config.hidden_size;
        const auto heads = config.num_heads;
        const auto head_dim = d / heads;
        features = core::make_tensor(state_ctx, GGML_TYPE_F32,
            TensorShape::from_dims({1, config.frontend_frames, 128}));
        encoder_positions = core::make_tensor(state_ctx, GGML_TYPE_F32,
            TensorShape::from_dims({1, config.encoder_frames, d}));
        token = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        decoder_position = core::make_tensor(state_ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, d}));
        slot = core::make_tensor(state_ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        causal_mask = core::make_tensor(state_ctx, GGML_TYPE_F16, TensorShape::from_dims({1, cache_steps}));
        memory_mask = core::make_tensor(state_ctx, GGML_TYPE_I32,
            TensorShape::from_dims({1, config.encoder_frames}));
        for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
            keys.push_back(core::make_tensor(state_ctx, GGML_TYPE_F32,
                TensorShape::from_dims({1, cache_steps, heads, head_dim})));
            values.push_back(core::make_tensor(state_ctx, GGML_TYPE_F32,
                TensorShape::from_dims({1, cache_steps, heads, head_dim})));
            cross.push_back({
                core::make_tensor(state_ctx, GGML_TYPE_F32,
                    TensorShape::from_dims({1, heads, config.encoder_frames, head_dim})),
                core::make_tensor(state_ctx, GGML_TYPE_F32,
                    TensorShape::from_dims({1, heads, config.encoder_frames, head_dim}))});
        }
        state_buffer.reset(ggml_backend_alloc_ctx_tensors(state_context.get(), execution.backend()));
        if (!state_buffer) {
            throw std::runtime_error("OWSM v4 state allocation failed");
        }
        cache = runtime::TransformerKVCache(cache_steps, d, keys, values, {true, false});
        build_encoder(weights);
        build_decoder(weights);
        write_positions();
    }

    ~Graphs() {
        ggml_backend_synchronize(execution.backend());
    }

    void build_encoder(const OWSMV4Weights & weights) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = encoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "owsm.encoder";
        ggml_set_input(features.tensor);
        ggml_set_input(encoder_positions.tensor);
        auto x = core::reshape_tensor(ctx, features,
            TensorShape::from_dims({1, 1, config.frontend_frames, 128}));
        x = modules::Conv2dModule({1, config.hidden_size, 3, 3, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.conv0);
        x = modules::ReluModule().build(ctx, x);
        x = modules::Conv2dModule({config.hidden_size, config.hidden_size, 3, 3, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.conv1);
        x = modules::ReluModule().build(ctx, x);
        x = modules::Conv2dModule({config.hidden_size, config.hidden_size, 3, 3, 2, 2, 0, 0, 1, 1, true})
            .build(ctx, x, weights.subsampling.conv2);
        x = modules::ReluModule().build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x,
            TensorShape::from_dims({1, config.encoder_frames, config.hidden_size * 15}));
        x = modules::LinearModule({config.hidden_size * 15, config.hidden_size, true})
            .build(ctx, x, weights.subsampling.projection);
        x = modules::LayerScaleModule().build(ctx, x, {weights.embedding_scale});
        x = modules::AddModule().build(ctx, x, encoder_positions);
        for (const auto & layer : weights.encoder) {
            x = modules::EBranchformerBlockModule({config.hidden_size,
                config.num_heads, config.intermediate_size, kLayerNormEps})
                .build(ctx, x, layer, weights.half_scale);
        }
        x = modules::LayerNormModule({config.hidden_size, kLayerNormEps})
            .build(ctx, x, weights.encoder_norm);
        modules::AttentionConfig cross_config{config.hidden_size, config.num_heads, true};
        cross_config.use_packed_kv = true;
        for (size_t layer = 0; layer < weights.decoder.size(); ++layer) {
            const auto projected = modules::CrossAttentionModule(cross_config)
                .build_key_value(ctx, x, weights.decoder[layer].cross_attention);
            ggml_build_forward_expand(encoder.graph,
                ggml_cpy(ctx.ggml, projected.key.tensor, cross[layer].key.tensor));
            ggml_build_forward_expand(encoder.graph,
                ggml_cpy(ctx.ggml, projected.value.tensor, cross[layer].value.tensor));
        }
        encoder.allocate();
    }

    void build_decoder(const OWSMV4Weights & weights) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = decoder.context;
        ctx.backend_type = execution.backend_type();
        ctx.module_instance_name = "owsm.decoder";
        for (const auto input : {token, decoder_position, slot, causal_mask, memory_mask}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::EmbeddingModule({config.vocabulary_size, config.hidden_size})
            .build(ctx, token, weights.embedding);
        x = modules::LayerScaleModule().build(ctx, x, {weights.embedding_scale});
        x = modules::AddModule().build(ctx, x, decoder_position);
        modules::TransformerDecoderBlockConfig decoder_config{
            config.hidden_size, config.num_heads, config.intermediate_size};
        decoder_config.eps = kLayerNormEps;
        decoder_config.activation = modules::FeedForwardActivation::Relu;
        decoder_config.use_packed_qkv = true;
        decoder_config.use_packed_kv = true;
        for (size_t layer = 0; layer < weights.decoder.size(); ++layer) {
            x = modules::TransformerDecoderBlockModule(decoder_config).build_cached_tail(
                ctx, x, weights.decoder[layer], keys[layer], values[layer], slot,
                causal_mask, cross[layer], memory_mask);
        }
        x = modules::LayerNormModule({config.hidden_size, kLayerNormEps})
            .build(ctx, x, weights.decoder_norm);
        logits = modules::LinearModule({config.hidden_size, config.vocabulary_size, true})
            .build(ctx, x, weights.output);
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(decoder.graph, logits.tensor);
        decoder.allocate();
    }

    void write_positions() {
        const auto d = config.hidden_size;
        std::vector<float> encoder_values(static_cast<size_t>(config.encoder_frames * d));
        for (int64_t position = 0; position < config.encoder_frames; ++position) {
            for (int64_t channel = 0; channel < d; channel += 2) {
                const auto phase = static_cast<double>(position) *
                    std::exp(static_cast<double>(channel) * -(std::log(10000.0) / static_cast<double>(d)));
                encoder_values[static_cast<size_t>(position * d + channel)] = static_cast<float>(std::sin(phase));
                encoder_values[static_cast<size_t>(position * d + channel + 1)] = static_cast<float>(std::cos(phase));
            }
        }
        core::write_tensor_f32(encoder_positions, encoder_values);
    }
};

OWSMV4Runtime::OWSMV4Runtime(
    const OWSMV4Assets & assets,
    const OWSMV4Weights & weights,
    core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution) {}

OWSMV4Runtime::~OWSMV4Runtime() = default;

OWSMV4DecodeResult OWSMV4Runtime::decode(
    const std::vector<float> & samples,
    const std::vector<int32_t> & prompt,
    bool predict_timestamps,
    int64_t max_tokens,
    int64_t beam_size) {
    if (samples.empty() || samples.size() > static_cast<size_t>(assets_.config.max_audio_samples)) {
        throw std::runtime_error("OWSM v4 requires between one sample and 30 seconds of 16 kHz audio");
    }
    if (prompt.empty()) {
        throw std::runtime_error("OWSM v4 decoder prompt must not be empty");
    }
    const auto limit = max_tokens > 0 ? std::min(max_tokens, assets_.config.max_decode_tokens)
                                      : assets_.config.max_decode_tokens;
    const auto required_steps = static_cast<int64_t>(prompt.size()) + limit;
    if (!graphs_ || graphs_->cache_steps < required_steps) {
        const auto cache_steps = ((required_steps + kCacheSteps - 1) / kCacheSteps) * kCacheSteps;
        graphs_.reset();
        graphs_ = std::make_unique<Graphs>(execution_, assets_.config, weights_, cache_steps);
    }
    auto & graphs = *graphs_;
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> padded = samples;
    padded.resize(static_cast<size_t>(assets_.config.max_audio_samples), 0.0F);
    auto features = assets_.frontend->extract_mono(
        padded, static_cast<size_t>(std::max<int64_t>(1, execution_.config().threads)));
    if (features.frames != assets_.config.frontend_frames || features.mel_bins != 128) {
        throw std::runtime_error("OWSM v4 frontend produced unexpected feature geometry");
    }
    for (int64_t frame = 0; frame < features.frames; ++frame) {
        for (int64_t mel = 0; mel < features.mel_bins; ++mel) {
            auto & value = features.values[static_cast<size_t>(frame * features.mel_bins + mel)];
            value = (value - assets_.feature_mean[static_cast<size_t>(mel)]) /
                assets_.feature_std[static_cast<size_t>(mel)];
        }
    }
    debug::timing_log_scalar("owsm.frontend_ms", debug::elapsed_ms(started));
    core::write_tensor_f32(graphs.features, features.values);
    const auto encoder_started = std::chrono::steady_clock::now();
    graphs.encoder.compute();
    debug::timing_log_scalar("owsm.encoder_ms", debug::elapsed_ms(encoder_started));

    graphs.cache.clear_on_backend();
    graphs.cursor.reset_to_empty(graphs.cache_steps);
    core::write_tensor_i32(graphs.memory_mask,
        std::vector<int32_t>(static_cast<size_t>(assets_.config.encoder_frames), 1));
    std::vector<float> causal(static_cast<size_t>(graphs.cache_steps), -std::numeric_limits<float>::infinity());
    std::vector<float> position(static_cast<size_t>(assets_.config.hidden_size));
    std::vector<float> logits;
    std::vector<int32_t> generated;
    const auto decoder_started = std::chrono::steady_clock::now();
    int32_t next = 0;
    int32_t detected_language_id = -1;
    std::string detected_language;
    const auto compute_step = [&](int32_t input) {
        const auto cursor = graphs.cursor.next_step();
        causal[static_cast<size_t>(cursor.position)] = 0.0F;
        for (int64_t channel = 0; channel < assets_.config.hidden_size; channel += 2) {
            const auto phase = static_cast<double>(cursor.position) *
                std::exp(static_cast<double>(channel) *
                    -(std::log(10000.0) / static_cast<double>(assets_.config.hidden_size)));
            position[static_cast<size_t>(channel)] = static_cast<float>(std::sin(phase));
            position[static_cast<size_t>(channel + 1)] = static_cast<float>(std::cos(phase));
        }
        core::write_tensor_i32(graphs.token, &input, 1);
        core::write_tensor_f32(graphs.decoder_position, position);
        core::write_tensor_i32(graphs.slot, &cursor.cache_slot, 1);
        core::write_tensor_f16(graphs.causal_mask, causal);
        graphs.decoder.compute();
        graphs.cache.advance_after_direct_append(1);
        graphs.cursor.advance_after_direct_append(1);
    };
    const auto constrain_logits = [&](const std::vector<int32_t> & tokens) {
        logits[static_cast<size_t>(assets_.config.blank_id)] = -std::numeric_limits<float>::infinity();
        if (!predict_timestamps) {
            std::fill(logits.begin() + assets_.config.first_timestamp_id,
                logits.begin() + assets_.config.last_timestamp_id + 1,
                -std::numeric_limits<float>::infinity());
        } else {
            const auto first = assets_.config.first_timestamp_id;
            const auto last = assets_.config.last_timestamp_id;
            const auto eos = assets_.config.eos_id;
            size_t timestamp_count = 0;
            int32_t previous_timestamp = first;
            for (const auto id : tokens) {
                if (id >= first && id <= last) {
                    ++timestamp_count;
                    previous_timestamp = id;
                }
            }
            for (int32_t id = 0; id < assets_.config.vocabulary_size; ++id) {
                bool allowed;
                if (tokens.empty()) {
                    allowed = id >= first && id <= last;
                } else if (timestamp_count % 2 != 0) {
                    allowed = id != eos && !(id >= first && id <= previous_timestamp);
                } else {
                    allowed = id == eos || (id >= previous_timestamp && id <= last);
                }
                if (!allowed) {
                    logits[static_cast<size_t>(id)] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    };
    for (int64_t step = 0; step < static_cast<int64_t>(prompt.size()) + limit - 1; ++step) {
        auto input = step < static_cast<int64_t>(prompt.size()) ? prompt[static_cast<size_t>(step)] : next;
        if (input == -1) {
            input = detected_language_id;
        }
        compute_step(input);
        if (step + 1 < static_cast<int64_t>(prompt.size())) {
            if (prompt[static_cast<size_t>(step + 1)] == -1) {
                core::read_tensor_f32_into(graphs.logits.tensor, logits);
                const auto first = assets_.token_id("<abk>");
                const auto last = assets_.token_id("<zul>");
                detected_language_id = static_cast<int32_t>(
                    std::max_element(logits.begin() + first, logits.begin() + last + 1) - logits.begin());
                const auto & token = assets_.sentencepiece[static_cast<size_t>(detected_language_id + 1)].text;
                detected_language = token.substr(1, token.size() - 2);
            }
            continue;
        }
        core::read_tensor_f32_into(graphs.logits.tensor, logits);
        if (beam_size > 1) {
            struct Hypothesis {
                std::vector<int32_t> tokens;
                float score = 0.0F;
                std::shared_ptr<const runtime::TransformerKVState> state;
            };
            std::vector<Hypothesis> active(1);
            Hypothesis best_ended;
            best_ended.score = -std::numeric_limits<float>::infinity();
            std::vector<int32_t> ranked(logits.size());
            std::iota(ranked.begin(), ranked.end(), 0);
            for (int64_t depth = 0; depth < limit; ++depth) {
                std::vector<Hypothesis> candidates;
                for (const auto & parent : active) {
                    if (depth > 0) {
                        graphs.cache.import_state(*parent.state);
                        graphs.cursor.import_state(*parent.state, graphs.cache_steps);
                        compute_step(parent.tokens.back());
                        core::read_tensor_f32_into(graphs.logits.tensor, logits);
                    }
                    const auto maximum = *std::max_element(logits.begin(), logits.end());
                    double sum = 0.0;
                    for (const auto value : logits) {
                        sum += std::exp(static_cast<double>(value - maximum));
                    }
                    const auto log_sum = static_cast<float>(std::log(sum));
                    for (auto & value : logits) {
                        value = (value - maximum) - log_sum;
                    }
                    constrain_logits(parent.tokens);
                    std::partial_sort(ranked.begin(), ranked.begin() + beam_size, ranked.end(),
                        [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
                    auto state = std::make_shared<runtime::TransformerKVState>(graphs.cache.export_state());
                    for (int64_t rank = 0; rank < beam_size; ++rank) {
                        const auto id = ranked[static_cast<size_t>(rank)];
                        if (!std::isfinite(logits[static_cast<size_t>(id)])) {
                            continue;
                        }
                        auto tokens = parent.tokens;
                        tokens.push_back(id);
                        candidates.push_back({std::move(tokens), parent.score + logits[id], state});
                    }
                }
                const auto count = std::min(candidates.size(), static_cast<size_t>(beam_size));
                std::partial_sort(candidates.begin(), candidates.begin() + count, candidates.end(),
                    [](const Hypothesis & a, const Hypothesis & b) { return a.score > b.score; });
                candidates.resize(count);
                active.clear();
                for (auto & candidate : candidates) {
                    const bool ended = candidate.tokens.back() == assets_.config.eos_id;
                    if (ended || depth + 1 == limit) {
                        if (ended) {
                            candidate.tokens.pop_back();
                        }
                        if (candidate.score > best_ended.score) {
                            best_ended = std::move(candidate);
                            best_ended.state.reset();
                        }
                    } else {
                        active.push_back(std::move(candidate));
                    }
                }
                if (active.empty() || best_ended.score >= active.front().score) {
                    break;
                }
            }
            if (!std::isfinite(best_ended.score)) {
                throw std::runtime_error("OWSM v4 beam search produced no finite hypothesis");
            }
            debug::timing_log_scalar("owsm.decoder_ms", debug::elapsed_ms(decoder_started));
            return {std::move(best_ended.tokens), std::move(detected_language)};
        }
        constrain_logits(generated);
        next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (!std::isfinite(logits[static_cast<size_t>(next)])) {
            throw std::runtime_error("OWSM v4 decoder produced no finite token");
        }
        if (next == assets_.config.eos_id) {
            debug::timing_log_scalar("owsm.decoder_ms", debug::elapsed_ms(decoder_started));
            return {std::move(generated), std::move(detected_language)};
        }
        generated.push_back(next);
    }
    debug::timing_log_scalar("owsm.decoder_ms", debug::elapsed_ms(decoder_started));
    return {std::move(generated), std::move(detected_language)};
}

}  // namespace engine::models::owsm
