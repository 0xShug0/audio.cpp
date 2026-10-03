#include "engine/models/crisperwhisper/model.h"

#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/bounded_static_kv_decode.h"
#include "engine/framework/sampling/hf_sampler.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace engine::models::crisperwhisper {

std::unique_ptr<CrisperWhisperWeights> load_weights(
    const CrisperWhisperAssets & assets, core::ExecutionContext & execution, assets::TensorStorageType type) {
    namespace binding = modules::binding;
    auto out = std::make_unique<CrisperWhisperWeights>();
    out->store = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
        "crisperwhisper.weights", 4 * 1024 * 1024);
    auto & store = *out->store;
    const auto & source = *assets.source;
    const int64_t d = assets.encoder.n_audio_state;
    const auto norm = [&](const std::string & name) {
        return binding::norm_from_named_source(store, source, name + ".weight", name + ".bias");
    };
    const auto linear = [&](const std::string & name, bool bias = true) {
        return binding::linear_from_named_source(store, source, name + ".weight",
            bias ? std::optional<std::string>(name + ".bias") : std::nullopt, type);
    };
    auto & e = out->encoder;
    e.conv1 = binding::conv1d_from_source(store, source, "model.encoder.conv1", assets::TensorStorageType::F32,
        d, assets.encoder.n_mels, 3, true);
    e.conv2 = binding::conv1d_from_source(store, source, "model.encoder.conv2", assets::TensorStorageType::F32,
        d, d, 3, true);
    e.positional_embedding = store.load_f32_tensor(source, "model.encoder.embed_positions.weight",
        {assets.encoder.n_audio_ctx, d});
    e.final_norm = norm("model.encoder.layer_norm");
    for (int64_t i = 0; i < assets.encoder.n_audio_layer; ++i) {
        const auto p = "model.encoder.layers." + std::to_string(i);
        modules::WhisperEncoderLayerWeights layer;
        layer.attention_norm = norm(p + ".self_attn_layer_norm");
        layer.attention = {linear(p + ".self_attn.q_proj"), linear(p + ".self_attn.k_proj", false),
                           linear(p + ".self_attn.v_proj"), linear(p + ".self_attn.out_proj")};
        layer.mlp_norm = norm(p + ".final_layer_norm");
        const auto fc1 = linear(p + ".fc1"), fc2 = linear(p + ".fc2");
        layer.mlp = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        e.layers.push_back(std::move(layer));
    }
    out->embedding = store.load_tensor(source, "model.decoder.embed_tokens.weight", type, {assets.vocabulary_size, d});
    out->positions = store.load_f32_tensor(source, "model.decoder.embed_positions.weight", {assets.max_positions, d});
    out->decoder_norm = norm("model.decoder.layer_norm");
    const auto attention = [&](const std::string & prefix, bool self) {
        modules::AttentionWeights w;
        std::vector<modules::PackedLinearSource> projections;
        if (self) {
            projections.push_back({prefix + ".q_proj.weight", std::nullopt, d});
        } else {
            const auto q = linear(prefix + ".q_proj");
            w.q_weight = q.weight;
            w.q_bias = q.bias;
        }
        projections.push_back({prefix + ".k_proj.weight", std::nullopt, d});
        projections.push_back({prefix + ".v_proj.weight", std::nullopt, d});
        w.qkv_weight = modules::PackedLinearWeightsBuilder({d, projections, false}).build(store, source, type).weight;
        // Whisper has no key bias. Pack a zero key lane with the trained Q/V biases.
        std::vector<float> biases(projections.size() * static_cast<size_t>(d), 0.0f);
        if (self) {
            const auto q = source.require_f32_tensor(prefix + ".q_proj.bias", {d});
            std::copy(q.values.begin(), q.values.end(), biases.begin());
        }
        const auto v = source.require_f32_tensor(prefix + ".v_proj.bias", {d});
        std::copy(v.values.begin(), v.values.end(), biases.end() - d);
        const auto bias_size = static_cast<int64_t>(biases.size());
        w.qkv_bias = store.make_f32(core::TensorShape::from_dims({bias_size}), std::move(biases));
        const auto o = linear(prefix + ".out_proj");
        w.out_weight = o.weight;
        w.out_bias = o.bias;
        return w;
    };
    for (int64_t i = 0; i < assets.decoder_layers; ++i) {
        const auto p = "model.decoder.layers." + std::to_string(i);
        modules::TransformerDecoderBlockWeights layer;
        layer.norm1 = norm(p + ".self_attn_layer_norm");
        layer.norm2 = norm(p + ".encoder_attn_layer_norm");
        layer.norm3 = norm(p + ".final_layer_norm");
        layer.self_attention = attention(p + ".self_attn", true);
        layer.cross_attention = attention(p + ".encoder_attn", false);
        const auto fc1 = linear(p + ".fc1"), fc2 = linear(p + ".fc2");
        layer.feed_forward = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
        out->decoder.push_back(std::move(layer));
    }
    store.upload();
    return out;
}

namespace {

using core::TensorShape;
using core::TensorValue;

struct WhisperGraph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    ggml_cgraph * graph = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan plan;

    explicit WhisperGraph(core::ExecutionContext & execution) : execution(execution) {
        context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!context) {
            throw std::runtime_error("CrisperWhisper graph context allocation failed");
        }
        graph = ggml_new_graph_custom(context.get(), 8192, false);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
    }
    ~WhisperGraph() {
        plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }
    void allocate() {
        core::validate_backend_graph_supported(execution.backend(), graph, "CrisperWhisper");
        if (!ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("CrisperWhisper graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }
    void compute() {
        if (core::compute_graph(execution, graph, plan, "CrisperWhisper") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CrisperWhisper graph execution failed");
        }
    }
};

}  // namespace

struct CrisperWhisperEncoderDecoderRuntime::Graphs {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> state{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> keys, values;
    std::vector<modules::CrossAttentionKeyValue> cross;
    TensorValue features, token, position, slot, causal_mask, memory_mask, logits;
    std::vector<TensorValue> alignment;
    WhisperGraph encoder, decoder;
    runtime::BoundedStaticKVDecodeCursor cursor;

    Graphs(const CrisperWhisperAssets & a, const CrisperWhisperWeights & w, core::ExecutionContext & execution)
        : execution(execution), encoder(execution), decoder(execution) {
        state.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!state) {
            throw std::runtime_error("CrisperWhisper state context allocation failed");
        }
        core::ModuleBuildContext ctx{};
        ctx.ggml = state.get();
        ctx.backend_type = execution.backend_type();
        const auto d = a.encoder.n_audio_state;
        const auto frames = a.encoder.n_audio_ctx;
        const auto h = a.decoder_heads;
        features = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, a.encoder.n_mels, 2 * frames}));
        token = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        position = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        slot = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        causal_mask = core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, a.max_positions}));
        memory_mask = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames}));
        for (int64_t i = 0; i < a.decoder_layers; ++i) {
            keys.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, a.max_positions, h, d / h})));
            values.push_back(core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, a.max_positions, h, d / h})));
            cross.push_back({core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, frames, d / h})),
                             core::make_tensor(ctx, GGML_TYPE_F16, TensorShape::from_dims({1, h, frames, d / h}))});
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(state.get(), execution.backend()));
        if (!buffer) {
            throw std::runtime_error("CrisperWhisper state allocation failed");
        }
        core::write_tensor_i32(memory_mask, std::vector<int32_t>(static_cast<size_t>(frames), 1));
        ctx.ggml = encoder.context.get();
        ctx.module_instance_name = "crisperwhisper.encoder";
        ggml_set_input(features.tensor);
        auto encoded = modules::WhisperEmbeddingModule(a.encoder).build(ctx, features, w.encoder);
        modules::AttentionConfig cross_config{d, h, true};
        cross_config.use_packed_kv = true;
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            const auto kv = modules::CrossAttentionModule(cross_config).build_key_value(ctx, encoded, w.decoder[i].cross_attention);
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, kv.key.tensor, cross[i].key.tensor));
            ggml_build_forward_expand(encoder.graph, ggml_cpy(ctx.ggml, kv.value.tensor, cross[i].value.tensor));
        }
        encoder.allocate();

        ctx.ggml = decoder.context.get();
        ctx.module_instance_name = "crisperwhisper.decoder";
        for (auto input : {token, position, slot, causal_mask, memory_mask}) {
            ggml_set_input(input.tensor);
        }
        auto x = modules::EmbeddingModule({a.vocabulary_size, d}).build(ctx, token, w.embedding);
        const auto p = modules::EmbeddingModule({a.max_positions, d}).build(ctx, position, w.positions);
        x = modules::AddModule().build(ctx, x, p);
        modules::AttentionConfig self_config{d, h, true};
        self_config.use_packed_qkv = true;
        self_config.causal = true;
        const modules::LayerNormModule norm({d, 1e-5f, true, true});
        const modules::FeedForwardModule ffn({d, a.decoder_ffn, true, modules::GeluApproximation::ExactErf});
        alignment.resize(a.alignment_heads.size());
        for (size_t i = 0; i < w.decoder.size(); ++i) {
            const auto & layer = w.decoder[i];
            auto y = norm.build(ctx, x, layer.norm1);
            y = modules::SelfAttentionModule(self_config).build_cached_tail(ctx, y, layer.self_attention,
                keys[i], values[i], slot, causal_mask).output;
            x = modules::AddModule().build(ctx, x, y);
            y = norm.build(ctx, x, layer.norm2);
            TensorValue probabilities;
            const bool capture = std::any_of(a.alignment_heads.begin(), a.alignment_heads.end(),
                [&](const auto & head) { return head.first == static_cast<int64_t>(i); });
            y = modules::CrossAttentionModule(cross_config).build_cached(ctx, y, cross[i],
                layer.cross_attention, memory_mask, nullptr, capture ? &probabilities : nullptr);
            if (capture) {
                for (size_t k = 0; k < a.alignment_heads.size(); ++k) {
                    if (a.alignment_heads[k].first == static_cast<int64_t>(i)) {
                        alignment[k] = modules::SliceModule({1, a.alignment_heads[k].second, 1}).build(ctx, probabilities);
                        ggml_set_output(alignment[k].tensor);
                    }
                }
            }
            x = modules::AddModule().build(ctx, x, y);
            y = ffn.build(ctx, norm.build(ctx, x, layer.norm3), layer.feed_forward);
            x = modules::AddModule().build(ctx, x, y);
        }
        x = norm.build(ctx, x, w.decoder_norm);
        logits = modules::LinearModule({d, a.vocabulary_size, false}).build(ctx, x, {w.embedding, std::nullopt});
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(decoder.graph, logits.tensor);
        for (const auto & head : alignment) {
            ggml_build_forward_expand(decoder.graph, head.tensor);
        }
        decoder.allocate();
    }
};

CrisperWhisperEncoderDecoderRuntime::CrisperWhisperEncoderDecoderRuntime(
    const CrisperWhisperAssets & assets, const CrisperWhisperWeights & weights, core::ExecutionContext & execution)
    : assets_(assets), weights_(weights), execution_(execution),
      frontend_({16000, 400, 160, assets.encoder.n_mels, audio::STFTFamily::Kokoro}) {}

CrisperWhisperEncoderDecoderRuntime::~CrisperWhisperEncoderDecoderRuntime() = default;

CrisperWhisperDecodeResult CrisperWhisperEncoderDecoderRuntime::transcribe(
    const std::vector<float> & samples, const std::vector<int32_t> & prompt, int64_t max_tokens,
    const std::string & language, double recovery_tail_sec, const std::vector<int32_t> & sibling_prompt) {
    const auto max_samples = static_cast<size_t>(assets_.encoder.n_audio_ctx * 320);
    if (samples.empty() || samples.size() > max_samples) {
        throw std::runtime_error("CrisperWhisper encoder input must be non-empty and at most 30 seconds");
    }
    if (prompt.empty() || prompt.size() >= static_cast<size_t>(assets_.max_positions) || max_tokens < 1) {
        throw std::runtime_error("CrisperWhisper invalid decoder prompt or token limit");
    }
    if (!graphs_) {
        graphs_ = std::make_unique<Graphs>(assets_, weights_, execution_);
    }
    auto & g = *graphs_;
    std::vector<float> padded = samples;
    padded.resize(max_samples, 0.0f);
    CrisperWhisperDecodeResult result;
    result.features = frontend_.compute(padded, execution_.config().threads);
    core::write_tensor_f32(g.features, result.features.values);
    g.encoder.compute();
    std::vector<float> logits, attention;
    const auto * active_prompt = &prompt;
    const auto decode_from = [&](size_t keep, std::optional<int32_t> ban_first, size_t min_new_tokens = 0,
                                 float temperature = 0.0f, uint32_t seed = 0) {
        auto prefix = *active_prompt;
        const auto steps = std::min<int64_t>(assets_.max_positions,
            static_cast<int64_t>(prefix.size()) + max_tokens - 1);
        sampling::HfSamplingOptions sampling_options;
        sampling_options.temperature = temperature;
        sampling::HfSamplerScratch scratch;
        std::mt19937 rng(seed);
        prefix.insert(prefix.end(), result.tokens.begin(), result.tokens.begin() + keep);
        result.tokens.resize(keep);
        result.attention.resize(keep * assets_.encoder.n_audio_ctx);
        result.reached_eos = false;
        result.stop_probability = -1.0;
        for (size_t i = 0; i < g.keys.size(); ++i) {
            ggml_backend_tensor_memset(g.keys[i].tensor, 0, 0, ggml_nbytes(g.keys[i].tensor));
            ggml_backend_tensor_memset(g.values[i].tensor, 0, 0, ggml_nbytes(g.values[i].tensor));
        }
        g.cursor.reset_to_empty(assets_.max_positions);
        std::vector<float> mask(static_cast<size_t>(assets_.max_positions), -std::numeric_limits<float>::infinity());
        int32_t next = 0;
        for (int64_t step = 0; step < steps; ++step) {
            const auto cursor = g.cursor.next_step();
            mask[static_cast<size_t>(cursor.position)] = 0.0f;
            core::write_tensor_i32(g.token, {step < static_cast<int64_t>(prefix.size()) ? prefix[step] : next});
            core::write_tensor_i32(g.position, {static_cast<int32_t>(cursor.position)});
            core::write_tensor_i32(g.slot, {static_cast<int32_t>(cursor.cache_slot)});
            core::write_tensor_f16(g.causal_mask, mask);
            g.decoder.compute();
            g.cursor.advance_after_direct_append(1);
            if (step + 1 < static_cast<int64_t>(prefix.size())) {
                continue;
            }
            core::read_tensor_f32_into(g.logits.tensor, logits);
            for (const auto id : assets_.suppress) {
                logits[id] = -std::numeric_limits<float>::infinity();
            }
            if (ban_first && step + 1 == static_cast<int64_t>(prefix.size())) {
                logits[*ban_first] = -std::numeric_limits<float>::infinity();
            }
            if (result.tokens.size() < min_new_tokens) {
                logits[assets_.eos] = -std::numeric_limits<float>::infinity();
            }
            next = temperature > 0.0f ? sampling::HfSampler{}.sample(
                logits, {}, sampling_options, scratch, rng, nullptr, "CrisperWhisper") :
                sampling::HfLogitsProcessor::argmax(logits.data(), logits.size(), "CrisperWhisper");
            if (!std::isfinite(logits[next])) {
                throw std::runtime_error("CrisperWhisper decoder returned non-finite logits");
            }
            if (next == assets_.eos) {
                result.reached_eos = true;
                double sum = 0.0;
                for (const auto logit : logits) {
                    sum += std::exp(static_cast<double>(logit - logits[next]));
                }
                result.stop_probability = 1.0 / sum;
                break;
            }
            result.tokens.push_back(next);
            const auto offset = result.attention.size();
            result.attention.resize(offset + assets_.encoder.n_audio_ctx, 0.0f);
            for (const auto & head : g.alignment) {
                core::read_tensor_f32_into(head.tensor, attention);
                for (size_t f = 0; f < attention.size(); ++f) {
                    result.attention[offset + f] += attention[f] / static_cast<float>(g.alignment.size());
                }
            }
        }
    };
    decode_from(0, std::nullopt);
    // Match the official Transformers repair policy: keep one repetition,
    // ban the loop starter for one step, and retry at most three times.
    for (int attempt = 0; attempt < 3; ++attempt) {
        std::optional<std::pair<size_t, size_t>> loop;
        for (size_t start = 0; start < result.tokens.size() && !loop; ++start) {
            for (size_t n = 1; n <= 24; ++n) {
                const size_t repeats = n <= 2 ? 8 : n == 3 ? 4 : 3;
                if (start + n * repeats > result.tokens.size()) {
                    continue;
                }
                bool matches = true;
                for (size_t r = 1; r < repeats && matches; ++r) {
                    matches = std::equal(result.tokens.begin() + start, result.tokens.begin() + start + n,
                        result.tokens.begin() + start + r * n);
                }
                if (matches) {
                    loop = {start, n};
                    break;
                }
            }
        }
        if (!loop) {
            break;
        }
        const auto [start, length] = *loop;
        debug::log_message(debug::LogLevel::Info, "crisperwhisper",
            "repair=" + std::to_string(attempt + 1) + " loop_start=" + std::to_string(start) +
            " ngram=" + std::to_string(length));
        decode_from(start + length, result.tokens[start]);
    }
    std::vector<float> energy;
    double threshold = 0.0, energy_range = 0.0;
    if (!sibling_prompt.empty() || recovery_tail_sec > 0.0) {
        const auto frames = static_cast<size_t>(result.features.frames);
        const auto bins = static_cast<size_t>(result.features.mel_bins);
        energy.resize(frames, 0.0f);
        for (size_t f = 0; f < frames; ++f) {
            for (size_t m = 0; m < bins; ++m) {
                energy[f] += result.features.values[m * frames + f];
            }
            energy[f] /= static_cast<float>(bins);
        }
        auto sorted = energy;
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&](double q) {
            const auto index = q * (frames - 1);
            const auto lo = static_cast<size_t>(index);
            return sorted[lo] + (sorted[std::min(lo + 1, frames - 1)] - sorted[lo]) * (index - lo);
        };
        const double floor = percentile(0.1), peak = percentile(0.95);
        energy_range = peak - floor;
        threshold = floor + 0.2 * energy_range;
    }
    if (!sibling_prompt.empty() && energy_range >= 1e-3) {
        const double speech_seconds = 0.01 * std::count_if(energy.begin(), energy.end(),
            [&](float value) { return value > threshold; });
        const auto word_count = [&](const std::vector<int32_t> & tokens) {
            std::istringstream text(assets_.decode_text(tokens));
            size_t count = 0;
            for (std::string word; text >> word;) {
                ++count;
            }
            return count;
        };
        size_t best_count = word_count(result.tokens);
        if (speech_seconds >= 5.0 && best_count < 0.5 * speech_seconds) {
            // Confirm sparse output against the other transcription mode before
            // sampling. All retries reuse the encoder output and decoder graph.
            CrisperWhisperDecodeResult best;
            best.tokens = std::move(result.tokens);
            best.attention = std::move(result.attention);
            best.reached_eos = result.reached_eos;
            best.stop_probability = result.stop_probability;
            active_prompt = &sibling_prompt;
            decode_from(0, std::nullopt);
            const auto reference_count = word_count(result.tokens);
            active_prompt = &prompt;
            if (reference_count >= 8 && best_count < 0.5 * reference_count) {
                bool recovered = false;
                uint32_t seed = 0;
                for (float temperature : {0.4f, 0.6f, 0.8f, 1.0f}) {
                    for (int draw = 0; draw < 3 && !recovered; ++draw) {
                        decode_from(0, std::nullopt, 0, temperature, seed++);
                        bool loop = false;
                        for (size_t n = 1; n <= 5 && !loop; ++n) {
                            for (size_t start = 0; start + n * 8 <= result.tokens.size() && !loop; ++start) {
                                loop = true;
                                for (size_t repeat = 1; repeat < 8 && loop; ++repeat) {
                                    loop = std::equal(result.tokens.begin() + start,
                                        result.tokens.begin() + start + n,
                                        result.tokens.begin() + start + repeat * n);
                                }
                            }
                        }
                        if (loop) {
                            continue;
                        }
                        const auto count = word_count(result.tokens);
                        recovered = count >= 0.5 * speech_seconds;
                        if (count > best_count || recovered) {
                            best_count = count;
                            best.tokens = std::move(result.tokens);
                            best.attention = std::move(result.attention);
                            best.reached_eos = result.reached_eos;
                            best.stop_probability = result.stop_probability;
                        }
                    }
                    if (recovered) {
                        break;
                    }
                }
                debug::log_message(debug::LogLevel::Info, "crisperwhisper",
                    "coverage_recovery attempts=" + std::to_string(seed) +
                    " words=" + std::to_string(best_count));
            }
            result.tokens = std::move(best.tokens);
            result.attention = std::move(best.attention);
            result.reached_eos = best.reached_eos;
            result.stop_probability = best.stop_probability;
        }
    }
    result.words = align_words(assets_, result, language);
    if (recovery_tail_sec > 0.0 && result.reached_eos && result.stop_probability < 0.7) {
        double last_end = -1.0;
        for (const auto & word : result.words) {
            last_end = std::max(last_end, word.end);
        }
        size_t active = 0;
        if (last_end >= 0.0 && energy_range >= 1e-3) {
            for (size_t f = static_cast<size_t>(last_end / 0.01); f < energy.size(); ++f) {
                active += energy[f] > threshold;
            }
        }
        if (static_cast<double>(active) * 0.01 >= recovery_tail_sec) {
            auto old_tokens = std::move(result.tokens);
            auto old_attention = std::move(result.attention);
            const auto old_probability = result.stop_probability;
            decode_from(0, std::nullopt, old_tokens.size() + 1);
            const bool accept = result.reached_eos && result.stop_probability >= 0.9 &&
                result.tokens.size() > old_tokens.size();
            debug::log_message(debug::LogLevel::Info, "crisperwhisper",
                std::string("early_eot_recovery=") + (accept ? "accepted" : "rejected") +
                " stop_probability=" + std::to_string(old_probability) +
                " retry_probability=" + std::to_string(result.stop_probability));
            if (accept) {
                result.words = align_words(assets_, result, language);
            } else {
                result.tokens = std::move(old_tokens);
                result.attention = std::move(old_attention);
                result.reached_eos = true;
                result.stop_probability = old_probability;
            }
        }
    }
    return result;
}

}  // namespace engine::models::crisperwhisper
