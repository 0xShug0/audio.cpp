// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Adapted for audio.cpp from NVIDIA/NeMo-Speech.cpp, revision 66d50d91b75a406becd95fbaf39d2c9c13e95d7c.
#include "engine/community_models/nemotron_transcribe/ctc.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <sentencepiece_processor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace engine::models::nemotron_transcribe {
namespace {

constexpr int kClassChunk = 4096;
constexpr int kProjectionFrames = 1024;
constexpr size_t kProjectionElements = 1 << 22;

void validate_config(const CtcConfig & c) {
    if (c.dimension <= 0 || c.heads <= 0 || c.dimension % c.heads || c.layers <= 0 ||
        c.ff_hidden <= 0 || c.upsampling <= 0 || c.upsampling > 16 || c.vocabulary < 2 ||
        c.blank != c.vocabulary - 1 || c.rotary_dims < 2 || c.rotary_dims % 2 ||
        c.rotary_dims > c.dimension / c.heads || !std::isfinite(c.residual_scale) ||
        !std::isfinite(c.norm_epsilon) || c.norm_epsilon <= 0.0f ||
        c.dimension > std::numeric_limits<int>::max() / std::max(3, c.upsampling)) {
        throw std::invalid_argument("invalid Nemotron CTC head configuration");
    }
}

int ctc_frames(int encoder_frames, int upsampling) {
    if (encoder_frames <= 0 || encoder_frames > std::numeric_limits<int>::max() / upsampling) {
        throw std::invalid_argument("invalid Nemotron CTC encoder frame count");
    }
    return encoder_frames * upsampling;
}

void validate_columns(const std::vector<int> & columns, int vocabulary) {
    if (columns.empty()) throw std::invalid_argument("CTC projection needs class IDs");
    for (int id : columns) {
        if (id < 0 || id >= vocabulary) throw std::invalid_argument("invalid CTC class ID");
    }
}

int validate_evidence(const CtcEvidence & evidence, const CtcConfig & c) {
    const int frames = ctc_frames(evidence.encoder_frames, c.upsampling);
    if (evidence.dimension != c.dimension || evidence.upsampling != c.upsampling ||
        evidence.vocabulary != c.vocabulary || evidence.hidden.size() != size_t(frames) * c.dimension ||
        evidence.logsumexp.size() != size_t(frames) || evidence.maximum.size() != size_t(frames)) {
        throw std::invalid_argument("CTC evidence dimensions do not match the head");
    }
    return frames;
}

core::TensorValue value(ggml_tensor * tensor) {
    return core::wrap_tensor(tensor, core::TensorShape::from_dims(
        {tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]}), tensor->type);
}

struct LayerWeights {
    modules::NormWeights norm1, norm2, q_norm, k_norm;
    modules::LinearWeights qkv, out, ff_in, ff_out;
};

struct Weights {
    modules::NormWeights embed, final;
    modules::LinearWeights upsample, classifier;
    std::vector<LayerWeights> layers;
};

Weights load_weights(core::BackendWeightStore & store, const assets::TensorSource & source,
    const CtcConfig & c) {
    namespace binding = modules::binding;
    constexpr auto type = assets::TensorStorageType::F32;
    Weights weights;
    if (c.upsampling > 1) {
        weights.upsample = binding::linear_from_source(store, source, "ctc.upsample", type,
            c.upsampling * c.dimension, c.dimension, true);
    }
    weights.embed = binding::norm_from_source(store, source, "ctc.transformer.embed_norm", c.dimension);
    weights.final = binding::norm_from_source(store, source, "ctc.transformer.final_norm", c.dimension);
    for (int i = 0; i < c.layers; ++i) {
        const auto prefix = "ctc.transformer.layers." + std::to_string(i);
        LayerWeights layer;
        layer.norm1 = binding::norm_from_source(store, source, prefix + ".norm1", c.dimension);
        layer.norm2 = binding::norm_from_source(store, source, prefix + ".norm2", c.dimension);
        layer.qkv = binding::linear_from_source(store, source, prefix + ".attn.w_qkv", type,
            3 * c.dimension, c.dimension, c.qkv_bias);
        layer.out = binding::linear_from_source(store, source, prefix + ".attn.out_proj", type,
            c.dimension, c.dimension, true);
        if (c.qk_norm) {
            layer.q_norm = binding::norm_from_source(store, source, prefix + ".attn.q_norm", c.dimension / c.heads);
            layer.k_norm = binding::norm_from_source(store, source, prefix + ".attn.k_norm", c.dimension / c.heads);
        }
        layer.ff_in = binding::linear_from_source(store, source, prefix + ".ffn.net.0", type,
            c.ff_hidden, c.dimension, true);
        layer.ff_out = binding::linear_from_source(store, source, prefix + ".ffn.net.3", type,
            c.dimension, c.ff_hidden, true);
        weights.layers.push_back(std::move(layer));
    }
    weights.classifier = binding::linear_from_source(store, source, "ctc.classifier", type,
        c.vocabulary, c.dimension, true);
    return weights;
}

// All graphs use the framework backend, host plan, and CUDA capture cleanup. The
// shape caches are model-local and retain only one shape of each graph kind.
struct Graph {
    core::ExecutionContext & execution;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * maximum = nullptr;
    core::HostGraphPlan plan;
    int frames = 0;
    int columns = 0;

    explicit Graph(core::ExecutionContext & execution_in, size_t nodes = 4096)
        : execution(execution_in) {
        context.reset(ggml_init({nodes * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("Nemotron CTC graph context allocation failed");
        graph = ggml_new_graph_custom(context.get(), nodes, false);
    }

    ~Graph() {
        plan.reset();
        core::release_backend_graph_resources(execution.backend(), graph, true);
    }

    void allocate() {
        ggml_set_input(input);
        if (positions) ggml_set_input(positions);
        if (ids) ggml_set_input(ids);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (maximum) {
            ggml_set_output(maximum);
            ggml_build_forward_expand(graph, maximum);
        }
        core::validate_backend_graph_supported(execution.backend(), graph, "nemotron.ctc");
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("Nemotron CTC graph allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
    }

    void run(const float * rows, size_t count) {
        ggml_backend_tensor_set(input, rows, 0, count * sizeof(float));
        if (core::compute_graph(execution, graph, plan, "nemotron.ctc") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Nemotron CTC graph execution failed");
        }
        ggml_backend_synchronize(execution.backend());
    }
};

ggml_tensor * norm(core::ModuleBuildContext & ctx, ggml_tensor * x,
    const modules::NormWeights & weights, const CtcConfig & c) {
    // Preserve the Q/K strided views, as NeMo-Speech.cpp does. Normalization is
    // over each head's ne0 before partial NeoX RoPE.
    return modules::LayerNormModule({x->ne[0], c.norm_epsilon, true, true, true})
        .build(ctx, value(x), weights).tensor;
}

ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x,
    const modules::LinearWeights & weights) {
    return modules::LinearModule({weights.weight.tensor->ne[0], weights.weight.tensor->ne[1],
        weights.bias.has_value(), GGML_PREC_F32}).build(ctx, value(x), weights).tensor;
}

std::unique_ptr<Graph> bridge_graph(core::ExecutionContext & execution, const Weights & weights,
    const CtcConfig & c, int encoder_frames) {
    auto g = std::make_unique<Graph>(execution, std::max<size_t>(4096, 512 + c.layers * 192));
    g->frames = encoder_frames;
    auto * ctx = g->context.get();
    core::ModuleBuildContext build{ctx, "nemotron.ctc.bridge", execution.backend_type()};
    const int T = ctc_frames(encoder_frames, c.upsampling), D = c.dimension, H = c.heads, K = D / H;
    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, encoder_frames);
    g->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    auto * x = g->input;
    if (c.upsampling > 1) x = ggml_reshape_2d(ctx, linear(build, x, weights.upsample), D, T);
    auto * h = norm(build, x, weights.embed, c);
    for (const auto & layer : weights.layers) {
        auto * qkv = linear(build, norm(build, h, layer.norm1, c), layer.qkv);
        const auto split = [&](int part) {
            return ggml_view_3d(ctx, qkv, K, H, T, size_t(K) * sizeof(float), qkv->nb[1],
                size_t(part) * D * sizeof(float));
        };
        auto * q = split(0);
        auto * k = split(1);
        auto * v = split(2);
        if (c.qk_norm) {
            q = norm(build, q, layer.q_norm, c);
            k = norm(build, k, layer.k_norm, c);
        }
        q = ggml_rope_ext(ctx, q, g->positions, nullptr, c.rotary_dims, GGML_ROPE_TYPE_NEOX,
            0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, g->positions, nullptr, c.rotary_dims, GGML_ROPE_TYPE_NEOX,
            0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        // Match the reference's F16 K/V attention inputs and F32 accumulation.
        auto * attended = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3),
            ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16),
            ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16), nullptr,
            1.0f / std::sqrt(float(K)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
        h = ggml_add(ctx, h, linear(build, ggml_reshape_2d(ctx, attended, D, T), layer.out));
        auto * ff = linear(build, norm(build, h, layer.norm2, c), layer.ff_in);
        h = ggml_add(ctx, h, linear(build, ggml_gelu_erf(ctx, ff), layer.ff_out));
    }
    g->output = ggml_add(ctx, x, ggml_scale(ctx, norm(build, h, weights.final, c), c.residual_scale));
    g->allocate();
    return g;
}

ggml_tensor * row_max(ggml_context * ctx, ggml_tensor * x) {
    const int64_t n = x->ne[0], rows = x->ne[1];
    auto * column = ggml_reshape_2d(ctx, ggml_argmax(ctx, x), 1, rows);
    return ggml_reshape_2d(ctx,
        ggml_get_rows(ctx, ggml_reshape_3d(ctx, x, 1, n, rows), column), 1, rows);
}

std::unique_ptr<Graph> normalizer_graph(core::ExecutionContext & execution, const Weights & weights,
    const CtcConfig & c, int frames) {
    auto g = std::make_unique<Graph>(execution);
    g->frames = frames;
    auto * ctx = g->context.get();
    auto * w = weights.classifier.weight.tensor;
    auto * b = weights.classifier.bias->tensor;
    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dimension, frames);
    ggml_tensor * peaks = nullptr, * sums = nullptr;
    // Retain chunk summaries, not full-vocabulary logits. GGML's allocator can
    // reuse one [frames, <=4096] logits buffer between these graph branches.
    for (int first = 0; first < c.vocabulary; first += kClassChunk) {
        const int count = std::min(kClassChunk, c.vocabulary - first);
        auto * logits = ggml_mul_mat(ctx,
            ggml_view_2d(ctx, w, c.dimension, count, w->nb[1], size_t(first) * w->nb[1]), g->input);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        logits = ggml_add(ctx, logits, ggml_view_1d(ctx, b, count, size_t(first) * b->nb[0]));
        auto * peak = row_max(ctx, logits);
        auto * sum = ggml_sum_rows(ctx, ggml_exp(ctx, ggml_sub(ctx, logits, peak)));
        peaks = peaks ? ggml_concat(ctx, peaks, peak, 0) : peak;
        sums = sums ? ggml_concat(ctx, sums, sum, 0) : sum;
    }
    g->maximum = row_max(ctx, peaks);
    g->output = ggml_add(ctx, ggml_log(ctx, ggml_sum_rows(ctx,
        ggml_mul(ctx, sums, ggml_exp(ctx, ggml_sub(ctx, peaks, g->maximum))))), g->maximum);
    g->allocate();
    return g;
}

std::unique_ptr<Graph> projection_graph(core::ExecutionContext & execution, const Weights & weights,
    const CtcConfig & c, int frames, int columns) {
    auto g = std::make_unique<Graph>(execution);
    g->frames = frames;
    g->columns = columns;
    auto * ctx = g->context.get();
    g->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.dimension, frames);
    g->ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, columns);
    auto * selected = ggml_get_rows(ctx, weights.classifier.weight.tensor, g->ids);
    auto * bias = ggml_get_rows(ctx,
        ggml_reshape_2d(ctx, weights.classifier.bias->tensor, 1, c.vocabulary), g->ids);
    auto * logits = ggml_mul_mat(ctx, selected, g->input);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    g->output = ggml_add(ctx, logits, ggml_reshape_1d(ctx, bias, columns));
    g->allocate();
    return g;
}

}  // namespace

struct CtcTimestampHead::Impl {
    CtcConfig config;
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    Weights weights;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> tokenizer;
    std::unique_ptr<Graph> bridge_cache, normalizer_cache, projection_cache;
    std::mutex mutex;

    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution_in,
        CtcConfig config_in, const std::string & serialized_tokenizer)
        : config(std::move(config_in)), execution(execution_in),
          store(execution.backend(), execution.backend_type(), "nemotron.ctc.weights",
              (size_t(config.layers) * 24 + 16) * ggml_tensor_overhead() + 65536) {
        if (!source) throw std::invalid_argument("missing Nemotron CTC tensor source");
        weights = load_weights(store, *source, config);
        store.upload();
        if (!serialized_tokenizer.empty()) {
            tokenizer = std::make_unique<sentencepiece::SentencePieceProcessor>();
            const auto status = tokenizer->LoadFromSerializedProto(serialized_tokenizer);
            if (!status.ok()) throw std::runtime_error("invalid CTC SentencePiece model: " + status.ToString());
            if (tokenizer->GetPieceSize() != config.blank) {
                throw std::runtime_error("CTC SentencePiece vocabulary does not match blank ID");
            }
        }
    }

    std::vector<float> bridge(const float * states, int frames) {
        const int count = ctc_frames(frames, config.upsampling);
        if (!states) throw std::invalid_argument("missing CTC ASR states");
        if (!bridge_cache || bridge_cache->frames != frames) {
            bridge_cache.reset();
            bridge_cache = bridge_graph(execution, weights, config, frames);
        }
        std::vector<int32_t> positions(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) positions[i] = i;
        ggml_backend_tensor_set(bridge_cache->positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        bridge_cache->run(states, size_t(frames) * config.dimension);
        return core::read_tensor_f32(bridge_cache->output);
    }

    std::vector<float> project(const float * hidden, int frames, const std::vector<int> & columns) {
        if (!hidden || frames <= 0) throw std::invalid_argument("invalid CTC projection states");
        validate_columns(columns, config.vocabulary);
        if (columns.size() > size_t(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("too many CTC projection columns");
        }
        const int tile = std::min<int>(kProjectionFrames,
            std::max<size_t>(1, kProjectionElements / columns.size()));
        std::vector<float> out(size_t(frames) * columns.size());
        for (int first = 0; first < frames; first += tile) {
            const int count = std::min(tile, frames - first);
            if (!projection_cache || projection_cache->frames != count ||
                projection_cache->columns != int(columns.size())) {
                projection_cache.reset();
                projection_cache = projection_graph(execution, weights, config, count, int(columns.size()));
            }
            ggml_backend_tensor_set(projection_cache->ids, columns.data(), 0, columns.size() * sizeof(int32_t));
            projection_cache->run(hidden + size_t(first) * config.dimension, size_t(count) * config.dimension);
            ggml_backend_tensor_get(projection_cache->output, out.data() + size_t(first) * columns.size(),
                0, size_t(count) * columns.size() * sizeof(float));
        }
        return out;
    }
};

CtcTimestampHead::CtcTimestampHead(std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution, CtcConfig config, const std::string & serialized_tokenizer) {
    validate_config(config);
    impl_ = std::make_unique<Impl>(std::move(source), execution, std::move(config), serialized_tokenizer);
}
CtcTimestampHead::~CtcTimestampHead() = default;
const CtcConfig & CtcTimestampHead::config() const noexcept { return impl_->config; }

std::vector<float> CtcTimestampHead::bridge(const float * states, int frames) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->bridge(states, frames);
}

std::vector<float> CtcTimestampHead::project(const float * hidden, int frames, const std::vector<int> & columns) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->project(hidden, frames, columns);
}

CtcEvidence CtcTimestampHead::prepare(const float * states, int encoder_frames, const CtcOptions & options) {
    const auto & c = impl_->config;
    const int frames = ctc_frames(encoder_frames, c.upsampling);
    if (!states || options.head_left_context < 0 || options.head_right_context < 0) {
        throw std::invalid_argument("invalid Nemotron CTC head windows");
    }
    CtcEvidence evidence;
    evidence.encoder_frames = encoder_frames;
    evidence.dimension = c.dimension;
    evidence.upsampling = c.upsampling;
    evidence.vocabulary = c.vocabulary;
    evidence.hidden.resize(size_t(frames) * c.dimension);
    evidence.logsumexp.resize(static_cast<size_t>(frames));
    evidence.maximum.resize(static_cast<size_t>(frames));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto & window : ::engine::models::nemotron_transcribe::timestamps::head_windows(encoder_frames, options.head_window,
             options.head_left_context, options.head_right_context)) {
        const auto h = impl_->bridge(states + size_t(window.context_start) * c.dimension,
            window.context_end - window.context_start);
        const int first = window.start * c.upsampling, count = (window.end - window.start) * c.upsampling;
        const float * kept = h.data() + size_t(window.start - window.context_start) * c.upsampling * c.dimension;
        std::copy_n(kept, size_t(count) * c.dimension, evidence.hidden.data() + size_t(first) * c.dimension);
        if (!impl_->normalizer_cache || impl_->normalizer_cache->frames != count) {
            impl_->normalizer_cache.reset();
            impl_->normalizer_cache = normalizer_graph(impl_->execution, impl_->weights, c, count);
        }
        auto & g = *impl_->normalizer_cache;
        g.run(kept, size_t(count) * c.dimension);
        ggml_backend_tensor_get(g.output, evidence.logsumexp.data() + first, 0, size_t(count) * sizeof(float));
        ggml_backend_tensor_get(g.maximum, evidence.maximum.data() + first, 0, size_t(count) * sizeof(float));
    }
    return evidence;
}

std::vector<float> CtcTimestampHead::emissions(const CtcEvidence & evidence,
    const std::vector<int> & columns, const CtcOptions & options) {
    const auto & c = impl_->config;
    const int frames = validate_evidence(evidence, c);
    auto out = project(evidence.hidden.data(), frames, columns);
    for (int frame = 0; frame < frames; ++frame) {
        if (!std::isfinite(evidence.logsumexp[frame])) throw std::runtime_error("non-finite CTC normalizer");
        for (size_t column = 0; column < columns.size(); ++column) {
            auto & emission = out[size_t(frame) * columns.size() + column];
            emission -= evidence.logsumexp[frame];
            if (!std::isfinite(emission)) throw std::runtime_error("non-finite CTC emission");
        }
    }
    if (options.bf16_emissions) core::round_f32_to_bf16_in_place(out);
    return out;
}

CtcTimestampResult CtcTimestampHead::timestamps(const CtcEvidence & evidence,
    const std::string & transcript, const CtcSpeakerEvidence & speakers, const CtcOptions & options) {
    namespace ts = ::engine::models::nemotron_transcribe::timestamps;
    CtcTimestampResult result;
    const auto fail = [&](const char * reason, const std::string & error) {
        result.status = "unavailable";
        result.failure_reason = reason;
        result.error = error;
        result.alignment.words.clear();
        result.alignment.ctc_step = 0.0;
    };
    try {
        if (!std::isfinite(speakers.duration_seconds) || speakers.duration_seconds <= 0.0 ||
            !std::isfinite(speakers.time_offset) || speakers.speakers < 0 ||
            speakers.probability_frames < 0 || speakers.activity10_frames < 0 ||
            (speakers.probabilities && (speakers.probability_frames <= 0 || speakers.speakers <= 0)) ||
            (speakers.activity10 && (speakers.activity10_frames <= 0 || speakers.speakers <= 0))) {
            throw std::invalid_argument("invalid CTC speaker evidence or duration");
        }
        if (speakers.activity10) {
            result.alignment.diarization = ts::diarization_segments(speakers.activity10,
                speakers.activity10_frames, speakers.speakers, 0.01, speakers.duration_seconds, speakers.time_offset);
            for (auto & segment : result.alignment.diarization) {
                segment.start = ts::round_decimals(segment.start, 3);
                segment.end = ts::round_decimals(segment.end, 3);
            }
        }
        result.cleaned_transcript = ts::clean_sot_transcript(transcript, options.special_tokens);
        if (result.cleaned_transcript.empty()) {
            fail("empty_transcript", "empty transcript");
            return result;
        }
        if (!impl_->tokenizer) {
            fail("ctc_tokenizer_unavailable", "CTC SentencePiece model is unavailable");
            return result;
        }
        if (evidence.encoder_frames <= 0 || evidence.hidden.empty()) {
            fail("ctc_evidence_unavailable", "CTC ASR evidence is unavailable");
            return result;
        }
        const int frames = validate_evidence(evidence, impl_->config);
        result.targets = ts::alignment_targets(*impl_->tokenizer, result.cleaned_transcript,
            impl_->config.blank, frames);
        auto probs = result.targets.streams.empty() ? std::vector<float>(frames, 0.0f) :
            emissions(evidence, result.targets.columns, options);
        std::vector<float> activity;
        if (speakers.probabilities) {
            activity.resize(size_t(evidence.encoder_frames) * speakers.speakers);
            for (int t = 0; t < evidence.encoder_frames; ++t) {
                std::copy_n(speakers.probabilities + size_t(std::min(t, speakers.probability_frames - 1)) * speakers.speakers,
                    speakers.speakers, activity.data() + size_t(t) * speakers.speakers);
            }
        }
        ts::AlignmentInputs inputs;
        inputs.log_probs = probs.data();
        inputs.frames = frames;
        inputs.speaker_probs = activity.empty() ? nullptr : activity.data();
        inputs.speaker_frames = evidence.encoder_frames;
        inputs.speakers = speakers.speakers;
        inputs.activity10 = speakers.activity10;
        inputs.label_frames = speakers.activity10 ? speakers.activity10_frames : 0;
        inputs.label_speakers = speakers.speakers;
        inputs.label_seconds = 0.01;
        inputs.duration = speakers.duration_seconds;
        inputs.time_offset = speakers.time_offset;
        ts::AlignmentConfig align_config;
        align_config.max_word_frames = 12 * impl_->config.upsampling;
        result.alignment = ts::align_record(result.targets, inputs, align_config);
        result.alignment.words = ts::public_words(result.alignment.words);
        for (auto & segment : result.alignment.diarization) {
            segment.start = ts::round_decimals(segment.start, 3);
            segment.end = ts::round_decimals(segment.end, 3);
        }
        if (!result.alignment.failed_streams.empty()) {
            result.status = result.alignment.words.empty() ? "unavailable" : "partial";
            result.failure_reason = "ctc_alignment_failed";
            result.error = "one or more speaker streams have no legal CTC alignment";
        } else {
            result.status = "aligned";
        }
        result.model_name = impl_->config.model_name;
    } catch (const std::exception & e) {
        fail("ctc_alignment_failed", e.what());
    }
    return result;
}

}  // namespace engine::models::nemotron_transcribe
