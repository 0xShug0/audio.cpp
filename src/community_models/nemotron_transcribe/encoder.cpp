// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PEE architecture and host Sortformer cache update adapted from NVIDIA's
// NeMo-Speech.cpp speechlm/pee implementation, revision 66d50d91.
// Graph ownership, weight loading, and execution use audio.cpp's framework.
#include "engine/community_models/nemotron_transcribe/encoder.h"

#include "engine/framework/core/backend_weight_store.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::nemotron_transcribe {
namespace {

constexpr size_t kGraphNodes = 32768;

struct LinearWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};
struct NormWeights {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
};
struct LayerWeights {
    NormWeights norm1, norm2, q_norm, k_norm;
    LinearWeights qkv, attention_output, ff_in, ff_out;
};
struct ExpertWeights {
    LinearWeights pre_encode;
    NormWeights embed_norm, final_norm;
    std::vector<LayerWeights> layers;
};

LinearWeights load_linear(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int in, int out, bool bias, assets::TensorStorageType storage) {
    LinearWeights result;
    result.weight = store.load_tensor(source, prefix + ".weight", storage, {out, in}).tensor;
    if (bias) result.bias = store.load_f32_tensor(source, prefix + ".bias", {out}).tensor;
    return result;
}

NormWeights load_norm(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int size) {
    return {store.load_f32_tensor(source, prefix + ".weight", {size}).tensor,
            store.load_f32_tensor(source, prefix + ".bias", {size}).tensor};
}

ExpertWeights load_expert(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, const RopeExpertConfig & config, assets::TensorStorageType storage) {
    ExpertWeights result;
    const int d = config.hidden_size;
    result.pre_encode = load_linear(store, source, prefix + ".pre_encode.proj",
        config.feature_size * config.subsampling_factor, d, false, storage);
    if (config.pre_block_norm) result.embed_norm = load_norm(store, source, prefix + ".embed_norm", d);
    result.final_norm = load_norm(store, source, prefix + ".final_norm", d);
    for (int i = 0; i < config.layers; ++i) {
        const std::string p = prefix + ".layers." + std::to_string(i);
        LayerWeights layer;
        layer.norm1 = load_norm(store, source, p + ".norm1", d);
        layer.norm2 = load_norm(store, source, p + ".norm2", d);
        layer.qkv = load_linear(store, source, p + ".attn.w_qkv", d, 3 * d, config.qkv_bias, storage);
        layer.attention_output = load_linear(store, source, p + ".attn.out_proj", d, d, true, storage);
        if (config.qk_norm) {
            layer.q_norm = load_norm(store, source, p + ".attn.q_norm", d / config.heads);
            layer.k_norm = load_norm(store, source, p + ".attn.k_norm", d / config.heads);
        }
        layer.ff_in = load_linear(store, source, p + ".ffn.net.0", d, config.intermediate_size, true, storage);
        layer.ff_out = load_linear(store, source, p + ".ffn.net.3", config.intermediate_size, d, true, storage);
        result.layers.push_back(layer);
    }
    return result;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const LinearWeights & weights) {
    auto * result = ggml_mul_mat(ctx, weights.weight, x);
    // Explicit F32 accumulation matches the upstream speech encoder.
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return weights.bias == nullptr ? result : ggml_add(ctx, result, weights.bias);
}

ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, const NormWeights & weights, float epsilon) {
    // Q/K split views retain the QKV frame stride; Vulkan NORM needs contiguous input.
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    auto * result = ggml_norm(ctx, x, epsilon);
    return ggml_add(ctx, ggml_mul(ctx, result, weights.weight), weights.bias);
}

ggml_tensor * expert_rows(ggml_context * ctx, ggml_tensor * rows, ggml_tensor * positions,
    const ExpertWeights & weights, const RopeExpertConfig & config, bool flash) {
    ggml_tensor * x = rows;
    if (config.pre_block_norm) x = norm(ctx, x, weights.embed_norm, config.layer_norm_epsilon);
    const int d = config.hidden_size;
    const int dk = d / config.heads;
    const int64_t frames = x->ne[1];
    for (const auto & layer : weights.layers) {
        auto * qkv = linear(ctx, norm(ctx, x, layer.norm1, config.layer_norm_epsilon), layer.qkv);
        const auto split = [&](int offset) {
            return ggml_view_3d(ctx, qkv, dk, config.heads, frames,
                static_cast<size_t>(dk) * sizeof(float), qkv->nb[1],
                static_cast<size_t>(offset * d) * sizeof(float));
        };
        auto * q = split(0);
        auto * k = split(1);
        auto * v = split(2);
        if (config.qk_norm) {
            q = norm(ctx, q, layer.q_norm, config.layer_norm_epsilon);
            k = norm(ctx, k, layer.k_norm, config.layer_norm_epsilon);
        }
        q = ggml_rope_ext(ctx, q, positions, nullptr, config.rotary_dimensions,
            GGML_ROPE_TYPE_NEOX, 0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, positions, nullptr, config.rotary_dimensions,
            GGML_ROPE_TYPE_NEOX, 0, config.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        auto * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
        auto * kp = ggml_cast(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), GGML_TYPE_F16);
        auto * vp = ggml_cast(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), GGML_TYPE_F16);
        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
        ggml_tensor * attention;
        if (flash) {
            attention = ggml_flash_attn_ext(ctx, qp, kp, vp, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attention, GGML_PREC_F32);
        } else {
            auto * scores = ggml_mul_mat(ctx, kp, qp);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            auto * probabilities = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
            auto * value_rows = ggml_cont(ctx, ggml_permute(ctx, vp, 1, 0, 2, 3));
            auto * attended = ggml_mul_mat(ctx, value_rows, probabilities);
            ggml_mul_mat_set_prec(attended, GGML_PREC_F32);
            attention = ggml_cont(ctx, ggml_permute(ctx, attended, 0, 2, 1, 3));
        }
        auto * merged = ggml_reshape_2d(ctx, attention, d, frames);
        x = ggml_add(ctx, x, linear(ctx, merged, layer.attention_output));
        auto * hidden = linear(ctx, norm(ctx, x, layer.norm2, config.layer_norm_epsilon), layer.ff_in);
        x = ggml_add(ctx, x, linear(ctx, ggml_gelu_erf(ctx, hidden), layer.ff_out));
    }
    return norm(ctx, x, weights.final_norm, config.layer_norm_epsilon);
}

// One reusable graph per stage. No backend allocations survive encoder teardown.
struct Graph {
    ggml_backend_t backend = nullptr;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * previous = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * speaker_probs = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * raw = nullptr;
    int64_t frames = 0;
    int64_t previous_frames = 0;

    explicit Graph(const core::ExecutionContext & execution) : backend(execution.backend()) {
        const size_t bytes = ggml_tensor_overhead() * kGraphNodes * 2 + ggml_graph_overhead_custom(kGraphNodes, false);
        context = ggml_init({bytes, nullptr, true});
        if (context == nullptr) throw std::runtime_error("Nemotron Transcribe graph context allocation failed");
    }
    ~Graph() {
        if (plan != nullptr) core::free_backend_graph_plan(backend, plan);
        if (graph != nullptr) core::release_backend_graph_resources(backend, graph, true);
        if (allocator != nullptr) ggml_gallocr_free(allocator);
        if (context != nullptr) ggml_free(context);
    }
    void allocate(const core::ExecutionContext & execution, const std::vector<ggml_tensor *> & outputs) {
        graph = ggml_new_graph_custom(context, kGraphNodes, false);
        for (auto * tensor : outputs) {
            ggml_set_output(tensor);
            ggml_build_forward_expand(graph, tensor);
        }
        core::validate_backend_graph_supported(backend, graph, "Nemotron Transcribe encoder");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("Nemotron Transcribe graph tensor allocation failed");
        }
        if (execution.uses_host_graph_plan()) {
            plan = core::create_backend_graph_plan_if_host(backend, graph);
            if (plan == nullptr) throw std::runtime_error("Nemotron Transcribe CPU graph plan failed");
        }
        if (positions != nullptr) {
            std::vector<int32_t> values(static_cast<size_t>(positions->ne[0]));
            std::iota(values.begin(), values.end(), 0);
            ggml_backend_tensor_set(positions, values.data(), 0, values.size() * sizeof(int32_t));
        }
    }
    void run() {
        if (core::compute_backend_graph(backend, graph, plan, "Nemotron Transcribe encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Nemotron Transcribe graph execution failed");
        }
    }
};

void upload(ggml_tensor * tensor, const std::vector<float> & values) {
    if (values.size() != static_cast<size_t>(ggml_nelements(tensor))) {
        throw std::runtime_error("Nemotron Transcribe graph input shape mismatch");
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

std::vector<int> topk(const std::vector<float> & values, int count) {
    if (values.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Nemotron Transcribe speaker cache has too many score entries");
    }
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    count = std::clamp(count, 0, static_cast<int>(indices.size()));
    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(), [&](int a, int b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    });
    indices.resize(static_cast<size_t>(count));
    return indices;
}

struct SpeakerState {
    std::vector<float> cache, cache_probs, fifo, fifo_probs;
    bool compressed = false;
};

// NeMo SortformerModules._compress_spkcache, deterministic inference ordering.
// Product scores avoid backend-dependent log rounding at near ties.
void compress_cache(SpeakerState & state, int dimension, int speakers,
    const SpeakerCacheConfig & config, const std::vector<float> & silence) {
    const int n = static_cast<int>(state.cache.size() / dimension);
    const int per_speaker = config.cache_frames / speakers - config.silence_frames_per_speaker;
    const auto bounded_count = [&](float rate, int maximum) {
        if (static_cast<double>(per_speaker) * rate > std::numeric_limits<float>::max()) return maximum;
        const float count = std::floor(per_speaker * rate);
        return count >= maximum ? maximum : static_cast<int>(count);
    };
    const int strong = bounded_count(config.strong_boost_rate, n);
    const int weak = bounded_count(config.weak_boost_rate, n);
    // An unachievable threshold must stay above n rather than overflowing an int.
    const int min_positive = bounded_count(config.min_positive_score_rate, n + 1);
    const int nt = n + config.silence_frames_per_speaker;
    std::vector<std::vector<float>> scores(static_cast<size_t>(speakers),
        std::vector<float>(static_cast<size_t>(nt), std::numeric_limits<float>::infinity()));
    for (int t = 0; t < n; ++t) {
        const float * p = state.cache_probs.data() + static_cast<size_t>(t) * speakers;
        for (int s = 0; s < speakers; ++s) {
            float product = 1.0f;
            for (int j = 0; j < speakers; ++j) {
                product *= j == s ? 1.0f : std::max(1.0f - p[j], config.prediction_score_threshold);
            }
            scores[s][t] = product * std::max(p[s], config.prediction_score_threshold) * 2.0f;
        }
    }
    for (int s = 0; s < speakers; ++s) {
        int n_positive = 0;
        for (int t = 0; t < n; ++t) {
            if (!(state.cache_probs[static_cast<size_t>(t) * speakers + s] > 0.5f)) scores[s][t] = 0.0f;
            n_positive += scores[s][t] > 1.0f;
        }
        if (n_positive >= min_positive) {
            for (int t = 0; t < n; ++t) if (!(scores[s][t] > 1.0f)) scores[s][t] = 0.0f;
        }
        for (int t = config.cache_frames; t < n; ++t) scores[s][t] *= std::exp(config.latest_score_boost);
        std::vector<float> row(scores[s].begin(), scores[s].begin() + n);
        for (int t : topk(row, strong)) scores[s][t] *= 4.0f;
        row.assign(scores[s].begin(), scores[s].begin() + n);
        for (int t : topk(row, weak)) scores[s][t] *= 2.0f;
    }
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(speakers) * nt);
    for (const auto & row : scores) flat.insert(flat.end(), row.begin(), row.end());
    auto selected = topk(flat, config.cache_frames);
    const int placeholder = static_cast<int>(flat.size());
    for (int & index : selected) if (flat[index] == 0.0f) index = placeholder;
    std::sort(selected.begin(), selected.end());
    std::vector<float> embeddings(static_cast<size_t>(config.cache_frames) * dimension);
    std::vector<float> probabilities(static_cast<size_t>(config.cache_frames) * speakers, 0.0f);
    for (int i = 0; i < config.cache_frames; ++i) {
        const int t = selected[i] == placeholder ? -1 : selected[i] % nt;
        float * destination = embeddings.data() + static_cast<size_t>(i) * dimension;
        if (t < 0 || t >= n) {
            std::copy(silence.begin(), silence.end(), destination);
        } else {
            std::copy_n(state.cache.data() + static_cast<size_t>(t) * dimension, dimension, destination);
            std::copy_n(state.cache_probs.data() + static_cast<size_t>(t) * speakers, speakers,
                probabilities.data() + static_cast<size_t>(i) * speakers);
        }
    }
    state.cache.swap(embeddings);
    state.cache_probs.swap(probabilities);
    state.compressed = true;
}

}  // namespace

void validate_encoder_config(const EncoderConfig & config) {
    // Each block needs fewer than 64 nodes; reserve the remaining nodes for the
    // stacking, head and fusion graph before accepting metadata-driven depths.
    constexpr int maximum_layers = static_cast<int>((kGraphNodes - 256) / 64);
    for (const auto * expert : {&config.asr, &config.diar}) {
        if (expert->feature_size <= 0 || expert->hidden_size <= 0 || expert->heads <= 0 ||
            expert->hidden_size % expert->heads != 0 || expert->layers < 0 || expert->layers > maximum_layers ||
            expert->hidden_size > std::numeric_limits<int>::max() / 3 || expert->intermediate_size <= 0 ||
            expert->subsampling_factor <= 0 || expert->rotary_dimensions <= 0 ||
            expert->feature_size > std::numeric_limits<int>::max() / expert->subsampling_factor ||
            expert->rotary_dimensions % 2 != 0 || expert->rotary_dimensions > expert->hidden_size / expert->heads ||
            !(expert->layer_norm_epsilon > 0.0f) || !std::isfinite(expert->layer_norm_epsilon) ||
            !(expert->rope_base > 0.0f) || !std::isfinite(expert->rope_base)) {
            throw std::runtime_error("Nemotron Transcribe expert configuration is invalid");
        }
    }
    const auto & cache = config.speaker_cache;
    if (config.asr.feature_size != config.diar.feature_size ||
        config.asr.subsampling_factor != config.diar.subsampling_factor ||
        config.upsample_factor != config.diar.subsampling_factor || config.projection_dim <= 0 ||
        config.speaker_count <= 0 || config.speaker_hidden_size <= 0 || config.online_inference_length <= 0 ||
        config.chunk_left_context < 0 || config.chunk_right_context < 0 ||
        !(config.speaker_activity_threshold >= 0.0f && config.speaker_activity_threshold <= 1.0f) ||
        !(config.maximum_audio_seconds > 0.0) || !std::isfinite(config.maximum_audio_seconds) ||
        config.maximum_audio_seconds > 3600.0 ||
        cache.cache_frames <= 0 || cache.fifo_frames < 0 ||
        cache.update_period <= 0 || cache.silence_frames_per_speaker < 0 ||
        cache.cache_frames / config.speaker_count <= cache.silence_frames_per_speaker ||
        !(cache.prediction_score_threshold > 0.0f && cache.prediction_score_threshold <= 1.0f) ||
        !(cache.strong_boost_rate >= 0.0f) || !std::isfinite(cache.strong_boost_rate) ||
        !(cache.weak_boost_rate >= 0.0f) || !std::isfinite(cache.weak_boost_rate) ||
        !(cache.min_positive_score_rate >= 0.0f) || !std::isfinite(cache.min_positive_score_rate) ||
        !std::isfinite(cache.latest_score_boost)) {
        throw std::runtime_error("Nemotron Transcribe encoder configuration is invalid");
    }
    if (config.speaker_hidden_size > std::numeric_limits<int>::max() / config.upsample_factor) {
        throw std::runtime_error("Nemotron Transcribe speaker upsample dimensions overflow");
    }
    const uint64_t conv_out = static_cast<uint64_t>(config.speaker_hidden_size) * config.upsample_factor;
    if (static_cast<uint64_t>(config.speaker_hidden_size) >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / (sizeof(float) * 3 * conv_out)) {
        throw std::runtime_error("Nemotron Transcribe speaker convolution size overflows");
    }
    if (cache.latest_score_boost < std::log(static_cast<double>(std::numeric_limits<float>::denorm_min())) ||
        cache.latest_score_boost > std::log(static_cast<double>(std::numeric_limits<float>::max()) / 16.0)) {
        throw std::runtime_error("Nemotron Transcribe speaker score boost is outside its finite range");
    }
    const int64_t mel_frames = static_cast<int64_t>(std::ceil(config.maximum_audio_seconds * 100.0));
    const int64_t encoder_frames = (mel_frames + config.asr.subsampling_factor - 1) / config.asr.subsampling_factor;
    if (cache.cache_frames < encoder_frames && static_cast<uint64_t>(config.speaker_count) *
        static_cast<uint64_t>(encoder_frames + cache.silence_frames_per_speaker) >
        static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Nemotron Transcribe speaker score indices overflow");
    }
}

std::vector<EncoderWindow> plan_encoder_windows(int64_t mel_frames, const EncoderConfig & config) {
    validate_encoder_config(config);
    if (mel_frames <= 0 || static_cast<double>(mel_frames) / 100.0 > config.maximum_audio_seconds) {
        throw std::runtime_error("Nemotron Transcribe mel duration is outside the configured limit");
    }
    const int64_t sub = config.asr.subsampling_factor;
    const int64_t chunk = static_cast<int64_t>(config.online_inference_length) * sub;
    if (mel_frames <= chunk) return {{0, mel_frames, 0, (mel_frames + sub - 1) / sub, 0}};
    const int64_t left = static_cast<int64_t>(config.chunk_left_context) * sub;
    const int64_t right = static_cast<int64_t>(config.chunk_right_context) * sub;
    std::vector<EncoderWindow> windows;
    for (int64_t start = 0; start < mel_frames; start += chunk) {
        const int64_t end = std::min(start + chunk, mel_frames);
        EncoderWindow window;
        window.mel_begin = std::max<int64_t>(0, start - left);
        window.mel_end = std::min(end + right, mel_frames);
        window.left_drop = (start - window.mel_begin) / sub;
        window.right_mel_frames = window.mel_end - end;
        window.core_frames = (end + sub - 1) / sub - (start + sub - 1) / sub;
        windows.push_back(window);
    }
    return windows;
}

struct ParallelExpertEncoder::Impl {
    const core::ExecutionContext & execution;
    EncoderConfig config;
    std::unique_ptr<core::BackendWeightStore> store;
    ExpertWeights asr, diar;
    LinearWeights encoder_projection, head_hidden, speaker_head, projection;
    ggml_tensor * upsample_weight = nullptr;
    ggml_tensor * upsample_bias = nullptr;
    NormWeights asr_norm, diar_norm;
    ggml_tensor * kernel = nullptr;
    std::vector<float> silence;
    std::unique_ptr<Graph> pre_graph, speaker_graph, asr_graph;
    std::mutex mutex;

    Impl(const core::ExecutionContext & e, const assets::TensorSource & source, EncoderConfig c,
        assets::TensorStorageType storage) : execution(e), config(std::move(c)) {
        validate_encoder_config(config);
        const size_t tensors = (static_cast<size_t>(config.asr.layers) + config.diar.layers) * 24 + 128;
        store = std::make_unique<core::BackendWeightStore>(execution.backend(), execution.backend_type(),
            "nemotron_transcribe.encoder", ggml_tensor_overhead() * tensors * 2);
        asr = load_expert(*store, source, "asr", config.asr, storage);
        diar = load_expert(*store, source, "diar", config.diar, storage);
        const int h = config.speaker_hidden_size;
        encoder_projection = load_linear(*store, source, "diar.head.encoder_proj", config.diar.hidden_size, h, true, storage);
        upsample_weight = store->load_tensor(source, "diar.head.subpixel_upsample.weight", assets::TensorStorageType::F16,
            {h * config.upsample_factor, h, 3}).tensor;
        upsample_bias = store->load_f32_tensor(source, "diar.head.subpixel_upsample.bias", {h * config.upsample_factor}).tensor;
        head_hidden = load_linear(*store, source, "diar.head.first_hidden_to_hidden", h, h, true, storage);
        speaker_head = load_linear(*store, source, "diar.head.single_hidden_to_spks", h, config.speaker_count, true, storage);
        projection = load_linear(*store, source, "fusion.proj", config.asr.hidden_size, config.projection_dim, true, storage);
        asr_norm = load_norm(*store, source, "fusion.asr_norm", config.asr.hidden_size);
        diar_norm = load_norm(*store, source, "fusion.diar_norm", config.speaker_count);
        kernel = store->load_f32_tensor(source, "fusion.diar_kernel", {config.speaker_count, config.asr.hidden_size}).tensor;
        // Required for multi-window identity continuity. Never silently reset
        // the diarizer between windows when an older conversion omits this.
        silence = source.require_f32("diar.head.learnable_sil_emb", {config.diar.hidden_size});
        store->upload();
    }

    std::vector<float> pre_encode(const std::vector<float> & stacked, int64_t frames) {
        if (!pre_graph || pre_graph->frames != frames) {
            pre_graph.reset();
            auto graph = std::make_unique<Graph>(execution);
            graph->frames = frames;
            graph->input = ggml_new_tensor_2d(graph->context, GGML_TYPE_F32,
                config.diar.feature_size * config.diar.subsampling_factor, frames);
            ggml_set_input(graph->input);
            graph->output = linear(graph->context, graph->input, diar.pre_encode);
            graph->allocate(execution, {graph->output});
            pre_graph = std::move(graph);
        }
        upload(pre_graph->input, stacked);
        pre_graph->run();
        return core::read_tensor_f32(pre_graph->output);
    }

    std::vector<float> speaker_probabilities(const std::vector<float> & embeddings,
        const SpeakerState & state, int64_t frames) {
        const int64_t previous = static_cast<int64_t>((state.cache.size() + state.fifo.size()) / config.diar.hidden_size);
        const int64_t length = previous + frames;
        if (!speaker_graph || speaker_graph->frames != length) {
            speaker_graph.reset();
            auto graph = std::make_unique<Graph>(execution);
            graph->frames = length;
            auto * ctx = graph->context;
            graph->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.diar.hidden_size, length);
            if (config.diar.layers > 0) graph->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length);
            ggml_set_input(graph->input);
            // Positions are constant inputs and kept live between graph runs.
            if (graph->positions != nullptr) {
                ggml_set_input(graph->positions);
                ggml_set_output(graph->positions);
            }
            auto * rows = expert_rows(ctx, graph->input, graph->positions, diar, config.diar, config.use_flash_attention);
            auto * projected = linear(ctx, rows, encoder_projection);
            auto * time_major = ggml_cont(ctx, ggml_transpose(ctx, projected));
            auto * upsampled = ggml_conv_1d(ctx, upsample_weight, time_major, 1, 1, 1);
            upsampled = ggml_add(ctx, upsampled, ggml_reshape_2d(ctx, upsample_bias, 1,
                config.speaker_hidden_size * config.upsample_factor));
            auto * channels = ggml_cont(ctx, ggml_transpose(ctx, upsampled));
            auto * native = ggml_reshape_2d(ctx, channels, config.speaker_hidden_size, length * config.upsample_factor);
            auto * hidden = linear(ctx, ggml_relu(ctx, native), head_hidden);
            graph->output = ggml_sigmoid(ctx, linear(ctx, ggml_relu(ctx, hidden), speaker_head));
            graph->allocate(execution, {graph->output});
            speaker_graph = std::move(graph);
        }
        std::vector<float> rows;
        rows.reserve(static_cast<size_t>(length) * config.diar.hidden_size);
        rows.insert(rows.end(), state.cache.begin(), state.cache.end());
        rows.insert(rows.end(), state.fifo.begin(), state.fifo.end());
        rows.insert(rows.end(), embeddings.begin(), embeddings.end());
        upload(speaker_graph->input, rows);
        speaker_graph->run();
        return core::read_tensor_f32(speaker_graph->output);
    }

    std::pair<std::vector<float>, std::vector<float>> asr_forward(
        const std::vector<float> & stacked, const std::vector<float> & probabilities, int64_t frames) {
        if (!asr_graph || asr_graph->frames != frames) {
            asr_graph.reset();
            auto graph = std::make_unique<Graph>(execution);
            graph->frames = frames;
            auto * ctx = graph->context;
            graph->input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                config.asr.feature_size * config.asr.subsampling_factor, frames);
            if (config.asr.layers > 0) graph->positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames);
            graph->speaker_probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.speaker_count, frames);
            for (auto * tensor : {graph->input, graph->positions, graph->speaker_probs}) {
                if (tensor != nullptr) ggml_set_input(tensor);
            }
            if (graph->positions != nullptr) ggml_set_output(graph->positions);
            auto * encoded = linear(ctx, graph->input, asr.pre_encode);
            graph->raw = expert_rows(ctx, encoded, graph->positions, asr, config.asr, config.use_flash_attention);
            auto * binary = ggml_step(ctx, ggml_scale_bias(ctx, graph->speaker_probs, 1.0f, -config.speaker_activity_threshold));
            auto * speaker_features = norm(ctx, binary, diar_norm, config.asr.layer_norm_epsilon);
            auto * transposed_kernel = ggml_cont(ctx, ggml_transpose(ctx, kernel));
            auto * infusion = ggml_mul_mat(ctx, transposed_kernel, speaker_features);
            ggml_mul_mat_set_prec(infusion, GGML_PREC_F32);
            auto * fused = ggml_add(ctx, norm(ctx, graph->raw, asr_norm, config.asr.layer_norm_epsilon), infusion);
            graph->output = linear(ctx, fused, projection);
            graph->allocate(execution, {graph->raw, graph->output});
            asr_graph = std::move(graph);
        }
        upload(asr_graph->input, stacked);
        upload(asr_graph->speaker_probs, probabilities);
        asr_graph->run();
        return {core::read_tensor_f32(asr_graph->raw), core::read_tensor_f32(asr_graph->output)};
    }

    void update_speakers(SpeakerState & state, const std::vector<float> & embeddings,
        const std::vector<float> & probabilities, int64_t left, int64_t core) {
        const int d = config.diar.hidden_size;
        const int speakers = config.speaker_count;
        const int64_t cached = static_cast<int64_t>(state.cache.size() / d);
        const int64_t fifo = static_cast<int64_t>(state.fifo.size() / d);
        state.fifo_probs.assign(probabilities.begin() + cached * speakers,
            probabilities.begin() + (cached + fifo) * speakers);
        state.fifo.insert(state.fifo.end(), embeddings.begin() + left * d, embeddings.begin() + (left + core) * d);
        state.fifo_probs.insert(state.fifo_probs.end(), probabilities.begin() + (cached + fifo + left) * speakers,
            probabilities.begin() + (cached + fifo + left + core) * speakers);
        const auto & geometry = config.speaker_cache;
        if (fifo + core <= geometry.fifo_frames) return;
        const int64_t pop = std::min(std::max<int64_t>(geometry.update_period, core - geometry.fifo_frames + fifo), fifo + core);
        if (!state.compressed) {
            state.cache_probs.assign(probabilities.begin(), probabilities.begin() + cached * speakers);
        }
        state.cache.insert(state.cache.end(), state.fifo.begin(), state.fifo.begin() + pop * d);
        state.cache_probs.insert(state.cache_probs.end(), state.fifo_probs.begin(), state.fifo_probs.begin() + pop * speakers);
        state.fifo.erase(state.fifo.begin(), state.fifo.begin() + pop * d);
        state.fifo_probs.erase(state.fifo_probs.begin(), state.fifo_probs.begin() + pop * speakers);
        if (static_cast<int64_t>(state.cache.size() / d) > geometry.cache_frames) {
            compress_cache(state, d, speakers, geometry, silence);
        }
    }

    EncoderOutput encode(const MelFeatures & features, const std::function<bool()> & should_stop) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto check_cancelled = [&] {
            if (should_stop && should_stop()) throw std::runtime_error("Nemotron Transcribe encoder cancelled");
        };
        check_cancelled();
        if (features.bins != config.asr.feature_size || features.frames <= 0 ||
            // Both experts consume the released model's 10 ms frontend grid.
            static_cast<double>(features.frames) / 100.0 > config.maximum_audio_seconds ||
            features.values.size() != static_cast<size_t>(features.frames * features.bins) ||
            !std::isfinite(features.duration_seconds) || features.duration_seconds < 0.0 ||
            features.duration_seconds > config.maximum_audio_seconds ||
            std::any_of(features.values.begin(), features.values.end(), [](float x) { return !std::isfinite(x); })) {
            throw std::runtime_error("Nemotron Transcribe mel input or duration is invalid");
        }
        const auto normalized = normalize_asr_features(features);
        const auto windows = plan_encoder_windows(features.frames, config);
        const int64_t sub = config.asr.subsampling_factor;
        const int speakers = config.speaker_count;
        EncoderOutput result;
        result.frames = (features.frames + sub - 1) / sub;
        result.native_speaker_frames = features.frames;
        result.embedding_dim = config.projection_dim;
        result.asr_dim = config.asr.hidden_size;
        result.speaker_count = speakers;
        result.duration_seconds = features.duration_seconds;
        result.audio_embeddings.reserve(static_cast<size_t>(result.frames) * config.projection_dim);
        result.asr_states.reserve(static_cast<size_t>(result.frames) * config.asr.hidden_size);
        result.speaker_probs80.reserve(static_cast<size_t>(result.frames) * speakers);
        result.speaker_probs10.reserve(static_cast<size_t>(result.frames * sub) * speakers);
        SpeakerState state;
        for (size_t i = 0; i < windows.size(); ++i) {
            check_cancelled();
            const auto & window = windows[i];
            const int64_t length = window.mel_end - window.mel_begin;
            const int64_t frames = (length + sub - 1) / sub;
            std::vector<float> raw(static_cast<size_t>(frames * sub) * features.bins, 0.0f), asr_input(raw.size(), 0.0f);
            std::copy_n(features.values.begin() + window.mel_begin * features.bins, length * features.bins, raw.begin());
            std::copy_n(normalized.begin() + window.mel_begin * features.bins, length * features.bins, asr_input.begin());
            const auto embeddings = pre_encode(raw, frames);
            check_cancelled();
            const int64_t previous = static_cast<int64_t>((state.cache.size() + state.fifo.size()) / config.diar.hidden_size);
            const auto p10 = speaker_probabilities(embeddings, state, frames);
            check_cancelled();
            const int64_t total = previous + frames;
            std::vector<float> probabilities(static_cast<size_t>(total) * speakers, 0.0f);
            // The single-window fused path masks the last incomplete 80 ms
            // cell. Streaming Sortformer uses its native plain downsample mean.
            for (int64_t t = 0; t < total; ++t) {
                const int64_t valid = windows.size() == 1 ? std::min(sub, length - t * sub) : sub;
                for (int64_t step = 0; step < valid; ++step) {
                    for (int s = 0; s < speakers; ++s) {
                        probabilities[t * speakers + s] += p10[(t * sub + step) * speakers + s] / static_cast<float>(valid);
                    }
                }
            }
            const int64_t right = (window.right_mel_frames + sub - 1) / sub;
            const int64_t speaker_core = std::max<int64_t>(0, frames - window.left_drop - right);
            std::vector<float> window_probs(static_cast<size_t>(frames) * speakers, 0.0f);
            for (int64_t t = 0; t < window.core_frames; ++t) {
                if (speaker_core == 0) {
                    result.speaker_probs10.insert(result.speaker_probs10.end(), static_cast<size_t>(sub) * speakers, 0.0f);
                    continue;
                }
                const int64_t source = previous + window.left_drop + std::min(t, speaker_core - 1);
                std::copy_n(probabilities.begin() + source * speakers, speakers,
                    window_probs.begin() + (window.left_drop + t) * speakers);
                result.speaker_probs10.insert(result.speaker_probs10.end(), p10.begin() + source * sub * speakers,
                    p10.begin() + (source + 1) * sub * speakers);
            }
            auto [states, projected] = asr_forward(asr_input, window_probs, frames);
            check_cancelled();
            result.asr_states.insert(result.asr_states.end(), states.begin() + window.left_drop * config.asr.hidden_size,
                states.begin() + (window.left_drop + window.core_frames) * config.asr.hidden_size);
            result.audio_embeddings.insert(result.audio_embeddings.end(), projected.begin() + window.left_drop * config.projection_dim,
                projected.begin() + (window.left_drop + window.core_frames) * config.projection_dim);
            result.speaker_probs80.insert(result.speaker_probs80.end(), window_probs.begin() + window.left_drop * speakers,
                window_probs.begin() + (window.left_drop + window.core_frames) * speakers);
            if (i + 1 < windows.size()) update_speakers(state, embeddings, probabilities, window.left_drop, speaker_core);
        }
        result.speaker_probs10.resize(static_cast<size_t>(result.native_speaker_frames) * speakers);
        if (result.asr_states.size() != static_cast<size_t>(result.frames) * result.asr_dim ||
            result.audio_embeddings.size() != static_cast<size_t>(result.frames) * result.embedding_dim ||
            result.speaker_probs80.size() != static_cast<size_t>(result.frames) * speakers) {
            throw std::runtime_error("Nemotron Transcribe window stitching geometry mismatch");
        }
        return result;
    }
};

ParallelExpertEncoder::ParallelExpertEncoder(const core::ExecutionContext & execution,
    const assets::TensorSource & source, EncoderConfig config, assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(execution, source, std::move(config), storage)) {}
ParallelExpertEncoder::~ParallelExpertEncoder() = default;
EncoderOutput ParallelExpertEncoder::encode(const MelFeatures & features, const std::function<bool()> & should_stop) {
    return impl_->encode(features, should_stop);
}
const EncoderConfig & ParallelExpertEncoder::config() const noexcept { return impl_->config; }

}  // namespace engine::models::nemotron_transcribe
