// YourMT3 architecture and event decoding adapted from the YourMT3 Authors
// (2024), Apache-2.0; see tests/mulacover/LICENSE.yourmt3.
#include "engine/models/mulacover/transcriber.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_kv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace engine::models::mulacover {
namespace {
constexpr int64_t kSamples = 32767, kFrames = 110, kChannels = 13;
constexpr int64_t kMaxTokens = 256, kVocabulary = 596;
using core::TensorShape;
using core::TensorValue;

struct PerceiverAttentionWeights {
    modules::NormWeights query_norm, input_norm;
    modules::LinearWeights query, key, value, output;
};
struct PerceiverLayerWeights {
    PerceiverAttentionWeights attention;
    modules::NormWeights norm;
    modules::LinearWeights router;
    TensorValue w1, gate, w2;
};
struct PerceiverBlockWeights {
    PerceiverAttentionWeights spectral;
    std::array<PerceiverLayerWeights, 2> local, temporal;
};
struct ConvBlockWeights {
    modules::Conv2dWeights conv1, conv2, shortcut;
    modules::BatchNorm2dEvalWeights bn1, bn2;
};
struct T5AttentionWeights {
    modules::NormWeights norm;
    modules::LinearWeights q, k, v, o;
};
struct T5LayerWeights {
    T5AttentionWeights self, cross;
    modules::NormWeights ff_norm;
    modules::GatedFeedForwardWeights ff;
    TensorValue self_key, self_value, cross_key, cross_value;
};

// Fixed-shape graph lifetime follows the framework backend/gallocr pattern.
// Each of the two instances is constructed once by the owning session.
struct Graph {
    core::ExecutionContext & execution;
    std::string name;
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    core::HostGraphPlan plan;

    Graph(core::ExecutionContext & owner, const char * label) : execution(owner), name(label) {
        context = ggml_init({128ull * 1024 * 1024, nullptr, true});
        if (!context) throw std::runtime_error("YourMT3 graph context allocation failed");
        graph = ggml_new_graph_custom(context, 32768, false);
    }
    ~Graph() {
        release_workspace();
        if (context) ggml_free(context);
    }
    void allocate() {
        if (allocator) return;
        core::validate_backend_graph_supported(execution.backend(), graph, "mulacover.transcriber");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!allocator || !ggml_gallocr_reserve(allocator, graph) ||
            !ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("YourMT3 graph buffer allocation failed");
        }
        core::prepare_host_graph_plan(execution, graph, plan);
        debug::timing_log_scalar(name + ".graph_buffer_mb",
            static_cast<double>(ggml_gallocr_get_buffer_size(allocator, 0)) / (1024 * 1024));
    }
    void release_workspace() {
        plan.reset();
        if (!allocator) return;
        core::release_backend_graph_resources(execution.backend(), graph, true);
        ggml_gallocr_free(allocator);
        allocator = nullptr;
        for (auto * tensor = ggml_get_first_tensor(context); tensor;
            tensor = ggml_get_next_tensor(context, tensor)) {
            tensor->data = nullptr;
            tensor->buffer = nullptr;
            tensor->extra = nullptr;
        }
        debug::timing_log_scalar(name + ".graph_buffer_mb", 0.0);
    }
    void compute(const char * label) {
        if (core::compute_graph(execution, graph, plan, label) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label) + " graph compute failed");
        }
    }
};

struct NoteEvent {
    double time;
    int program, pitch, velocity;
    bool drum;
};
struct SegmentEvents {
    std::vector<NoteEvent> events;
    std::vector<std::pair<int, int>> ties, last_activity;
    double start;
};
}  // namespace

class Transcriber::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source,
         const core::BackendConfig & backend, assets::TensorStorageType storage)
        : execution_(backend), store_(execution_.backend(), execution_.backend_type(),
              "mulacover.transcriber.weights", 8 * 1024 * 1024),
          encoder_graph_(execution_, "mulacover.transcriber.encoder"),
          decoder_graph_(execution_, "mulacover.transcriber.decoder") {
        if (!source) throw std::invalid_argument("YourMT3 requires a tensor source");
        load_weights(*source, storage);
        store_.upload();
        core::set_backend_threads(execution_.backend(), std::max(1, backend.threads));
        build_encoder();
        build_decoder();
    }

    std::vector<TranscribedNote> transcribe(const std::vector<float> & mono, int sample_rate) {
        const auto begin = std::chrono::steady_clock::now();
        if (mono.empty() || sample_rate <= 0 ||
            !std::all_of(mono.begin(), mono.end(), [](float x) { return std::isfinite(x); })) {
            throw std::invalid_argument("YourMT3 audio must contain finite nonempty mono samples and a positive sample rate");
        }
        const auto waveform = sample_rate == 16000 ? mono : audio::resample_mono_torchaudio_sinc_hann(
            mono, sample_rate, 16000, audio::torchaudio_sinc_hann_float32_options());
        const size_t segments = (waveform.size() + kSamples - 1) / kSamples;
        std::array<std::vector<SegmentEvents>, kChannels> events;
        for (size_t segment = 0; segment < segments; ++segment) {
            std::vector<float> samples(kSamples, 0.0f);
            const size_t offset = segment * kSamples;
            std::copy_n(waveform.begin() + offset, std::min<size_t>(kSamples, waveform.size() - offset), samples.begin());
            encoder_graph_.allocate();
            core::write_tensor_f32(spectrogram_input_, frontend(samples));
            encoder_graph_.compute("mulacover.transcriber.encoder");
            // Cross-attention K/V live in the persistent weight/cache store.
            // The large encoder scratch is not needed by token decoding or AR.
            encoder_graph_.release_workspace();
            // The fixed masked tail is still read by attention's matrix multiply.
            // Zero it so earlier requests (or uninitialized buffers) cannot inject NaNs.
            for (auto & layer : decoder_) {
                ggml_backend_tensor_memset(layer.self_key.tensor, 0, 0, ggml_nbytes(layer.self_key.tensor));
                ggml_backend_tensor_memset(layer.self_value.tensor, 0, 0, ggml_nbytes(layer.self_value.tensor));
            }
            std::array<std::vector<int32_t>, kChannels> tokens;
            std::vector<int32_t> input(kChannels, 0), slots(kChannels), position(1);
            std::array<bool, kChannels> finished{};
            for (int step = 0; step < kMaxTokens; ++step) {
                position[0] = step;
                for (int c = 0; c < kChannels; ++c) slots[c] = c * kMaxTokens + step;
                std::vector<float> mask(kMaxTokens, -std::numeric_limits<float>::infinity());
                std::fill_n(mask.begin(), step + 1, 0.0f);
                core::write_tensor_i32(decoder_ids_, input);
                core::write_tensor_i32(decoder_position_, position);
                core::write_tensor_i32(decoder_slots_, slots);
                core::write_tensor_f32(decoder_mask_, mask);
                decoder_graph_.compute("mulacover.transcriber.decoder");
                input = core::read_tensor_i32(decoder_output_);
                for (int c = 0; c < kChannels; ++c) {
                    // Match task_cond_dec_generate: the first prediction does not
                    // update unfinished_sequences; subsequent EOS predictions do.
                    tokens[c].push_back(finished[c] ? 0 : input[c]);
                    if (step > 0 && input[c] == 1) finished[c] = true;
                }
                if (std::all_of(finished.begin(), finished.end(), [](bool x) { return x; })) break;
            }
            const double start = static_cast<double>(segment * kSamples) / 16000.0;
            for (int c = 0; c < kChannels; ++c) events[c].push_back(detokenize(tokens[c], start));
        }
        std::vector<TranscribedNote> notes;
        for (auto & channel : events) {
            auto decoded = merge_events(channel);
            notes.insert(notes.end(), decoded.begin(), decoded.end());
        }
        // mix_notes: sort including offset before trimming across decoder channels.
        std::stable_sort(notes.begin(), notes.end(), [](const auto & a, const auto & b) {
            return std::make_tuple(a.onset_seconds, a.is_drum, a.program, a.pitch, a.duration) <
                   std::make_tuple(b.onset_seconds, b.is_drum, b.program, b.pitch, b.duration);
        });
        trim_notes(notes);
        for (auto & note : notes) if (!note.is_drum && note.duration < 0.01) note.duration = 0.01;
        debug::timing_log_scalar("mulacover.transcriber.wall_ms", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
        return notes;
    }

private:
    void load_weights(const assets::TensorSource & source, assets::TensorStorageType storage) {
        for (int i = 0; i < 3; ++i) {
            auto & block = conv_[i];
            const std::string p = "pre_encoder.0.blocks." + std::to_string(i);
            block.conv1 = modules::binding::conv2d_from_source(store_, source, p + ".conv1", storage, 128, i ? 128 : 1, 3, 3, false);
            block.conv2 = modules::binding::conv2d_from_source(store_, source, p + ".conv2", storage, 128, 128, 3, 3, false);
            if (!i) block.shortcut = modules::binding::conv2d_from_source(store_, source, p + ".shortcut", storage, 128, 1, 1, 1, true);
            const auto bn1 = modules::binding::batch_norm_eval_from_source(store_, source, p + ".bn1", 128, 1e-5f);
            const auto bn2 = modules::binding::batch_norm_eval_from_source(store_, source, p + ".bn2", 128, 1e-5f);
            block.bn1 = {bn1.scale, bn1.bias};
            block.bn2 = {bn2.scale, bn2.bias};
        }
        latents_ = store_.load_f32_tensor(source, "encoder.latent_array.latents", {26, 128});
        // inference.py passes the NUMBER OF HEADS (8), not head_dim (16),
        // to get_rotary_emb. With partial PE this rotates only four coordinates.
        const auto pixel_freqs = source.require_f32("encoder.rotary_emb_sca.freqs", {2});
        const auto local_freqs = source.require_f32("encoder.rotary_emb_latent.freqs", {2});
        const auto temporal_freqs = source.require_f32("encoder.rotary_emb_temporal.freqs", {2});
        for (int kind = 0; kind < 2; ++kind) {
            const int length = kind ? 110 : 26;
            const auto & freqs = kind ? temporal_freqs : local_freqs;
            for (int trig = 0; trig < 2; ++trig) {
                std::vector<float> values(length * 2);
                for (int t = 0; t < length; ++t) {
                    const float pos = kind ? static_cast<float>(t) : -1.0f + 2.0f * t / (length - 1);
                    for (int d = 0; d < 2; ++d) values[t * 2 + d] = trig ? std::sin(pos * freqs[d]) : std::cos(pos * freqs[d]);
                }
                rotary_[kind][trig] = store_.make_f32(TensorShape::from_dims({1, 1, length, 2}), std::move(values));
            }
        }
        // The spectral and local rotary modules share positions, but keep their
        // saved frequencies separate; checkpoints contain each module's parameter.
        for (int trig = 0; trig < 2; ++trig) {
            std::vector<float> values(26 * 2);
            for (int t = 0; t < 26; ++t) for (int d = 0; d < 2; ++d) {
                const float angle = (-1.0f + 2.0f * t / 25.0f) * pixel_freqs[d];
                values[t * 2 + d] = trig ? std::sin(angle) : std::cos(angle);
            }
            spectral_rotary_[trig] = store_.make_f32(TensorShape::from_dims({1, 1, 26, 2}), std::move(values));
        }
        for (int i = 0; i < 3; ++i) {
            auto & block = perceiver_[i];
            const std::string p = "encoder.blocks." + std::to_string(i);
            block.spectral = load_perceiver_attention(source, storage, p + ".spectral_cross_attention", true);
            for (int j = 0; j < 2; ++j) {
                block.local[j] = load_perceiver_layer(source, storage, p + ".local_transformer." + std::to_string(j));
                block.temporal[j] = load_perceiver_layer(source, storage, p + ".temporal_transformer." + std::to_string(j));
            }
        }
        projection_ = modules::binding::linear_from_source(store_, source, "pre_decoder.0.linear", storage, 512, 256, false);
        embedding_ = store_.load_tensor(source, "embed_tokens.weight", storage, {kVocabulary, 512});
        // Despite tie_word_embeddings=True, inference.py constructs two modules
        // and does not alias their parameters. Load the actual saved lm_head.
        lm_head_ = modules::binding::linear_from_source(store_, source, "lm_head.lm_head", storage, kVocabulary, 512, false);
        const std::string dp = "decoder.decoder";
        positions_ = store_.load_f32_tensor(source, dp + ".additive_pe.weight", {1034, 512});
        final_norm_ = modules::binding::norm_weight_from_source(store_, source, dp + ".final_layer_norm", 512);
        for (int i = 0; i < 8; ++i) {
            auto & layer = decoder_[i];
            const std::string p = dp + ".block." + std::to_string(i) + ".layer.";
            for (int cross = 0; cross < 2; ++cross) {
                auto & a = cross ? layer.cross : layer.self;
                const std::string prefix = p + std::to_string(cross);
                a.norm = modules::binding::norm_weight_from_source(store_, source, prefix + ".layer_norm", 512);
                const std::string ap = prefix + (cross ? ".EncDecAttention" : ".SelfAttention");
                a.q = modules::binding::linear_from_source(store_, source, ap + ".q", storage, 384, 512, false);
                a.k = modules::binding::linear_from_source(store_, source, ap + ".k", storage, 384, 512, false);
                a.v = modules::binding::linear_from_source(store_, source, ap + ".v", storage, 384, 512, false);
                a.o = modules::binding::linear_from_source(store_, source, ap + ".o", storage, 512, 384, false);
            }
            layer.ff_norm = modules::binding::norm_weight_from_source(store_, source, p + "2.layer_norm", 512);
            layer.ff.gate_proj = modules::binding::linear_from_source(store_, source, p + "2.DenseReluDense.wi_0", storage, 1024, 512, false);
            layer.ff.up_proj = modules::binding::linear_from_source(store_, source, p + "2.DenseReluDense.wi_1", storage, 1024, 512, false);
            layer.ff.down_proj = modules::binding::linear_from_source(store_, source, p + "2.DenseReluDense.wo", storage, 512, 1024, false);
            layer.self_key = store_.make_f32(TensorShape::from_dims({13, 256, 6, 64}), std::vector<float>(13 * 256 * 384, 0.0f));
            layer.self_value = store_.make_f32(TensorShape::from_dims({13, 256, 6, 64}), std::vector<float>(13 * 256 * 384, 0.0f));
            layer.cross_key = store_.make_f32(TensorShape::from_dims({13, 6, 110, 64}), std::vector<float>(13 * 110 * 384, 0.0f));
            layer.cross_value = store_.make_f32(TensorShape::from_dims({13, 6, 110, 64}), std::vector<float>(13 * 110 * 384, 0.0f));
        }
        score_one_ = store_.make_f32(TensorShape::from_dims({1}), {1.0f});
    }

    PerceiverAttentionWeights load_perceiver_attention(const assets::TensorSource & source,
        assets::TensorStorageType storage, const std::string & p, bool cross) {
        PerceiverAttentionWeights w;
        w.query_norm = modules::binding::norm_from_source(store_, source, p + ".self.layernorm1", 128);
        if (cross) w.input_norm = modules::binding::norm_from_source(store_, source, p + ".self.layernorm2", 128);
        w.query = modules::binding::linear_from_source(store_, source, p + ".self.query", storage, 128, 128, true);
        w.key = modules::binding::linear_from_source(store_, source, p + ".self.key", storage, 128, 128, true);
        w.value = modules::binding::linear_from_source(store_, source, p + ".self.value", storage, 128, 128, true);
        w.output = modules::binding::linear_from_source(store_, source, p + ".output.dense", storage, 128, 128, true);
        return w;
    }

    PerceiverLayerWeights load_perceiver_layer(const assets::TensorSource & source,
        assets::TensorStorageType storage, const std::string & p) {
        PerceiverLayerWeights w;
        w.attention = load_perceiver_attention(source, storage, p + ".attention", false);
        w.norm = modules::binding::norm_from_source(store_, source, p + ".layernorm", 128);
        w.router = modules::binding::linear_from_source(store_, source, p + ".mlp.gate", storage, 8, 128, false);
        for (int matrix = 0; matrix < 3; ++matrix) {
            const std::string name = matrix == 0 ? "w1" : matrix == 1 ? "gate" : "w2";
            std::vector<core::BackendWeightStore::TensorPart> parts;
            for (int expert = 0; expert < 8; ++expert) parts.push_back({
                p + ".mlp.experts." + std::to_string(expert) + "." + name + ".weight",
                matrix == 2 ? std::vector<int64_t>{128, 512} : std::vector<int64_t>{512, 128}});
            auto packed = store_.load_tensor(source, parts, storage);
            // The packed source concatenates the output rows. Its contiguous
            // storage becomes [expert,out,in] in the graph's weight view.
            if (matrix == 0) w.w1 = packed;
            else if (matrix == 1) w.gate = packed;
            else w.w2 = packed;
        }
        return w;
    }

    TensorValue build_perceiver_attention(core::ModuleBuildContext & ctx, const TensorValue & hidden,
        const TensorValue * inputs, const PerceiverAttentionWeights & w,
        const std::array<TensorValue, 2> & rotary) {
        const auto qn = modules::LayerNormModule({128, 1e-5f, true, true}).build(ctx, hidden, w.query_norm);
        const auto kvn = inputs ? modules::LayerNormModule({128, 1e-5f, true, true}).build(ctx, *inputs, w.input_norm) : qn;
        auto q = modules::LinearModule({128, 128, true, GGML_PREC_DEFAULT}).build(ctx, qn, w.query);
        auto k = modules::LinearModule({128, 128, true, GGML_PREC_DEFAULT}).build(ctx, kvn, w.key);
        auto v = modules::LinearModule({128, 128, true, GGML_PREC_DEFAULT}).build(ctx, kvn, w.value);
        const int64_t batch = hidden.shape.dims[0], queries = hidden.shape.dims[1], keys = kvn.shape.dims[1];
        q = core::reshape_tensor(ctx, q, TensorShape::from_dims({batch, queries, 8, 16}));
        k = core::reshape_tensor(ctx, k, TensorShape::from_dims({batch, keys, 8, 16}));
        v = core::reshape_tensor(ctx, v, TensorShape::from_dims({batch, keys, 8, 16}));
        q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
        k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
        v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
        // Apply adjacent-pair rotary math with framework tensor modules to the first four
        // coordinates only. Pixel positions span [-1,1]; temporal positions are
        // integer indices. The reference deliberately leaves every key unrotated.
        auto first = modules::SliceModule({3, 0, 4}).build(ctx, q);
        auto tail = modules::SliceModule({3, 4, 12}).build(ctx, q);
        first = core::wrap_tensor(ggml_cont(ctx.ggml, first.tensor), first.shape);
        const int64_t rows = batch * 8 * queries;
        const auto pair_shape = TensorShape::from_dims({rows, 2, 1});
        first = core::reshape_tensor(ctx, first, TensorShape::from_dims({rows, 2, 2}));
        auto even = modules::SliceModule({2, 0, 1}).build(ctx, first);
        auto odd = modules::SliceModule({2, 1, 1}).build(ctx, first);
        std::array<TensorValue, 2> trig;
        for (int i = 0; i < 2; ++i) {
            auto expanded = modules::RepeatModule({TensorShape::from_dims({batch, 8, queries, 2})}).build(ctx, rotary[i]);
            trig[i] = core::reshape_tensor(ctx, expanded, pair_shape);
        }
        auto even_cos = modules::MulModule().build(ctx, even, trig[0]);
        auto odd_sin = modules::MulModule().build(ctx, odd, trig[1]);
        auto odd_cos = modules::MulModule().build(ctx, odd, trig[0]);
        auto even_sin = modules::MulModule().build(ctx, even, trig[1]);
        auto re = core::wrap_tensor(ggml_sub(ctx.ggml, even_cos.tensor, odd_sin.tensor), pair_shape);
        auto im = modules::AddModule().build(ctx, odd_cos, even_sin);
        auto rotated = modules::ConcatModule({2}).build(ctx, re, im);
        rotated = core::reshape_tensor(ctx, rotated, TensorShape::from_dims({batch, 8, queries, 4}));
        q = modules::ConcatModule({3}).build(ctx, rotated, tail);
        auto attended = modules::ScaledDotProductAttentionModule({16,
            modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_DEFAULT}).build(ctx, q, k, v);
        attended = core::wrap_tensor(ggml_cont(ctx.ggml, attended.tensor), attended.shape);
        attended = core::reshape_tensor(ctx, attended, hidden.shape);
        auto result = modules::LinearModule({128, 128, true, GGML_PREC_DEFAULT}).build(ctx, attended, w.output);
        return modules::ResidualAddModule().build(ctx, result, hidden);
    }

    TensorValue build_perceiver_layer(core::ModuleBuildContext & ctx, const TensorValue & input,
        const PerceiverLayerWeights & w, const std::array<TensorValue, 2> & rotary) {
        const auto hidden = build_perceiver_attention(ctx, input, nullptr, w.attention, rotary);
        auto normalized = modules::LayerNormModule({128, 1e-5f, true, true}).build(ctx, hidden, w.norm);
        const int64_t tokens = normalized.shape.num_elements() / 128;
        normalized = core::reshape_tensor(ctx, normalized, TensorShape::from_dims({tokens, 128}));
        auto logits = modules::LinearModule({128, 8, false, GGML_PREC_DEFAULT}).build(ctx, normalized, w.router);
        auto * probabilities = ggml_soft_max(ctx.ggml, logits.tensor);
        auto * ids = ggml_top_k(ctx.ggml, probabilities, 2);
        // Gather the top two probabilities independently for each token, then
        // renormalize their sum exactly as MixtralSparseMoeBlock.forward.
        auto * probabilities_by_token = ggml_reshape_3d(ctx.ggml, probabilities, 1, 8, tokens);
        auto * selected = ggml_get_rows(ctx.ggml, probabilities_by_token, ids);
        selected = ggml_reshape_2d(ctx.ggml, selected, 2, tokens);
        selected = ggml_div(ctx.ggml, selected, ggml_sum_rows(ctx.ggml, selected));
        selected = ggml_reshape_3d(ctx.ggml, selected, 1, 2, tokens);
        auto * rows = ggml_reshape_3d(ctx.ggml, normalized.tensor, 128, 1, tokens);
        auto * w1 = ggml_reshape_3d(ctx.ggml, w.w1.tensor, 128, 512, 8);
        auto * gate = ggml_reshape_3d(ctx.ggml, w.gate.tensor, 128, 512, 8);
        auto * w2 = ggml_reshape_3d(ctx.ggml, w.w2.tensor, 512, 128, 8);
        auto * up = ggml_mul_mat_id(ctx.ggml, w1, rows, ids);
        auto * gated = ggml_mul_mat_id(ctx.ggml, gate, rows, ids);
        auto * activated = ggml_mul(ctx.ggml, ggml_silu(ctx.ggml, up), gated);
        auto * down = ggml_mul_mat_id(ctx.ggml, w2, activated, ids);
        down = ggml_mul(ctx.ggml, down, selected);
        // [dim,topk,tokens] -> sum over topk -> [dim,tokens].
        auto * sum = ggml_sum_rows(ctx.ggml, ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, down, 1, 0, 2, 3)));
        sum = ggml_reshape_3d(ctx.ggml, sum, 128, hidden.shape.dims[1], hidden.shape.dims[0]);
        return modules::ResidualAddModule().build(ctx, core::wrap_tensor(sum, hidden.shape), hidden);
    }

    void build_encoder() {
        auto * g = encoder_graph_.context;
        core::ModuleBuildContext ctx{g, "mulacover.transcriber.encoder", execution_.backend_type()};
        spectrogram_input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, 110, 1024}));
        ggml_set_input(spectrogram_input_.tensor);
        auto features = spectrogram_input_;
        for (int i = 0; i < 3; ++i) {
            const auto residual = i ? features : modules::Conv2dModule({1, 128, 1, 1}).build(ctx, features, conv_[i].shortcut);
            modules::Conv2dConfig cfg;
            cfg.in_channels = i ? 128 : 1;
            cfg.out_channels = 128;
            cfg.kernel_height = cfg.kernel_width = 3;
            cfg.padding_height = cfg.padding_width = 1;
            cfg.use_bias = false;
            features = modules::Conv2dModule(cfg).build(ctx, features, conv_[i].conv1);
            features = modules::BatchNorm2dEvalModule({128}).build(ctx, features, conv_[i].bn1);
            features = modules::ReluModule().build(ctx, features);
            cfg.in_channels = 128;
            features = modules::Conv2dModule(cfg).build(ctx, features, conv_[i].conv2);
            features = modules::BatchNorm2dEvalModule({128}).build(ctx, features, conv_[i].bn2);
            features = modules::ResidualAddModule().build(ctx, features, residual);
            features = modules::ReluModule().build(ctx, features);
            auto shape = features.shape;
            shape.dims[3] /= 2;
            features = core::wrap_tensor(ggml_pool_2d(g, features.tensor, GGML_OP_POOL_AVG, 2, 1, 2, 1, 0, 0), shape);
        }
        // NCHW -> [time, channel, frequency]: spectral attention attends to
        // channels with frequency as the embedding, not the other way around.
        features = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, features);
        features = core::wrap_tensor(ggml_cont(g, features.tensor), features.shape);
        features = core::reshape_tensor(ctx, features, TensorShape::from_dims({110, 128, 128}));
        auto hidden = core::reshape_tensor(ctx, latents_, TensorShape::from_dims({1, 26, 128}));
        hidden = modules::RepeatModule({TensorShape::from_dims({110, 26, 128})}).build(ctx, hidden);
        for (auto & block : perceiver_) {
            hidden = build_perceiver_attention(ctx, hidden, &features, block.spectral, spectral_rotary_);
            for (auto & layer : block.local) hidden = build_perceiver_layer(ctx, hidden, layer, rotary_[0]);
            hidden = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, hidden);
            hidden = core::wrap_tensor(ggml_cont(g, hidden.tensor), hidden.shape);
            for (auto & layer : block.temporal) hidden = build_perceiver_layer(ctx, hidden, layer, rotary_[1]);
            hidden = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, hidden);
            hidden = core::wrap_tensor(ggml_cont(g, hidden.tensor), hidden.shape);
        }
        // Each decoder channel sees a contiguous pair of Perceiver latents.
        hidden = core::reshape_tensor(ctx, hidden, TensorShape::from_dims({110, 13, 256}));
        hidden = modules::TransposeModule({{1, 0, 2, 3}, 3}).build(ctx, hidden);
        hidden = modules::LinearModule({256, 512, false, GGML_PREC_DEFAULT}).build(ctx, hidden, projection_);
        for (auto & layer : decoder_) {
            auto k = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, hidden, layer.cross.k);
            auto v = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, hidden, layer.cross.v);
            k = core::reshape_tensor(ctx, k, TensorShape::from_dims({13, 110, 6, 64}));
            v = core::reshape_tensor(ctx, v, TensorShape::from_dims({13, 110, 6, 64}));
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            ggml_build_forward_expand(encoder_graph_.graph, ggml_cpy(g, k.tensor, layer.cross_key.tensor));
            ggml_build_forward_expand(encoder_graph_.graph, ggml_cpy(g, v.tensor, layer.cross_value.tensor));
        }
    }

    void build_decoder() {
        auto * g = decoder_graph_.context;
        core::ModuleBuildContext ctx{g, "mulacover.transcriber.decoder", execution_.backend_type()};
        decoder_ids_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({13}));
        decoder_position_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        decoder_slots_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({13}));
        decoder_mask_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, 1, 256}));
        for (const auto & input : {decoder_ids_, decoder_position_, decoder_slots_, decoder_mask_}) ggml_set_input(input.tensor);
        auto * tokens = ggml_get_rows(g, embedding_.tensor, decoder_ids_.tensor);
        auto * pos = ggml_get_rows(g, positions_.tensor, decoder_position_.tensor);
        auto hidden = core::wrap_tensor(ggml_reshape_3d(g, ggml_add(g, tokens, pos), 512, 1, 13),
            TensorShape::from_dims({13, 1, 512}));
        const modules::RMSNormModule norm({512, 1e-6f, true, false});
        const modules::ScaledDotProductAttentionModule attention({64,
            modules::ScaledDotProductAttentionLowering::Explicit, GGML_PREC_DEFAULT});
        const modules::FastKVSetRowsModule set_rows;
        for (auto & layer : decoder_) {
            auto n = norm.build(ctx, hidden, layer.self.norm);
            auto q = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, n, layer.self.q);
            auto k = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, n, layer.self.k);
            auto v = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, n, layer.self.v);
            q = core::reshape_tensor(ctx, q, TensorShape::from_dims({13, 1, 6, 64}));
            k = core::reshape_tensor(ctx, k, TensorShape::from_dims({13, 1, 6, 64}));
            v = core::reshape_tensor(ctx, v, TensorShape::from_dims({13, 1, 6, 64}));
            auto kc = set_rows.build(ctx, layer.self_key, k, decoder_slots_);
            auto vc = set_rows.build(ctx, layer.self_value, v, decoder_slots_);
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            kc = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, kc);
            vc = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, vc);
            auto a = attention.build(ctx, q, kc, vc, decoder_mask_, score_one_);
            a = core::wrap_tensor(ggml_cont(g, a.tensor), a.shape);
            a = core::reshape_tensor(ctx, a, TensorShape::from_dims({13, 1, 384}));
            a = modules::LinearModule({384, 512, false, GGML_PREC_DEFAULT}).build(ctx, a, layer.self.o);
            hidden = modules::ResidualAddModule().build(ctx, hidden, a);
            n = norm.build(ctx, hidden, layer.cross.norm);
            q = modules::LinearModule({512, 384, false, GGML_PREC_DEFAULT}).build(ctx, n, layer.cross.q);
            q = core::reshape_tensor(ctx, q, TensorShape::from_dims({13, 1, 6, 64}));
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            a = attention.build(ctx, q, layer.cross_key, layer.cross_value, std::nullopt, score_one_);
            a = core::wrap_tensor(ggml_cont(g, a.tensor), a.shape);
            a = core::reshape_tensor(ctx, a, TensorShape::from_dims({13, 1, 384}));
            a = modules::LinearModule({384, 512, false, GGML_PREC_DEFAULT}).build(ctx, a, layer.cross.o);
            hidden = modules::ResidualAddModule().build(ctx, hidden, a);
            n = norm.build(ctx, hidden, layer.ff_norm);
            a = modules::GatedFeedForwardModule({512, 1024, false,
                modules::GatedFeedForwardActivation::Gelu, modules::GeluApproximation::Tanh,
                GGML_PREC_DEFAULT}).build(ctx, n, layer.ff);
            hidden = modules::ResidualAddModule().build(ctx, hidden, a);
        }
        hidden = norm.build(ctx, hidden, final_norm_);
        hidden = modules::ScaleBiasModule({1.0f / std::sqrt(512.0f), 0.0f}).build(ctx, hidden);
        auto logits = modules::LinearModule({512, kVocabulary, false, GGML_PREC_DEFAULT}).build(ctx, hidden, lm_head_);
        decoder_output_ = ggml_argmax(g, ggml_reshape_2d(g, logits.tensor, kVocabulary, 13));
        ggml_set_output(decoder_output_);
        ggml_build_forward_expand(decoder_graph_.graph, decoder_output_);
        decoder_graph_.allocate();
    }

    std::vector<float> frontend(const std::vector<float> & samples) const {
        audio::STFTConfig cfg;
        cfg.n_fft = cfg.win_length = 2048;
        cfg.hop_length = 300;
        cfg.family = audio::STFTFamily::PeriodicF32;
        const auto spec = audio::STFT().compute_magnitude(samples, audio::get_cached_stft_window(cfg),
            1, kSamples, cfg, std::max(1, execution_.config().threads));
        // Framework STFT stores [batch,freq,time]; Conv2d expects time-major
        // rows with contiguous frequency, with DC removed before normalization.
        if (spec.values.size() != 1025 * kFrames) throw std::runtime_error("YourMT3 STFT shape mismatch");
        std::vector<float> result(kFrames * 1024);
        for (int t = 0; t < kFrames; ++t) for (int f = 0; f < 1024; ++f) {
            result[t * 1024 + f] = std::log(spec.values[(f + 1) * kFrames + t] + 1e-5f);
        }
        const auto extrema = std::minmax_element(result.begin(), result.end());
        const float low = *extrema.first, range = *extrema.second - low + 0.008f;
        for (auto & x : result) x = (x - low) / range;
        return result;
    }

    SegmentEvents detokenize(const std::vector<int32_t> & tokens, double start) const {
        // FastCodec's fixed ranges: PAD/EOS/UNK 0..2, shift 3..208,
        // pitch 209..336, velocity 337..338, tie 339, program 340..467,
        // drum 468..595. Shift values are absolute within the segment.
        SegmentEvents result;
        result.start = start;
        int program = -1, velocity = -1;
        size_t first_event = tokens.size();
        bool has_tie = false;
        std::map<std::string, int> errors;
        for (size_t i = 0; i < tokens.size(); ++i) {
            const int token = tokens[i];
            if (token == 339) { first_event = i + 1; has_tie = true; break; }
            if (token >= 3 && token <= 208) break;
            if (token >= 340 && token <= 467) program = token - 340;
            else if (token >= 209 && token <= 336) {
                if (program < 0) ++errors["missing_program_in_tie"];
                else {
                    result.ties.emplace_back(program, token - 209);
                    result.last_activity.emplace_back(program, token - 209);
                }
            }
        }
        if (!has_tie) {
            ++errors["missing_tie"];
            result.ties.clear();
            result.last_activity.clear();
        }
        // Python round() uses nearest-even for the segment's centisecond grid.
        const double tick_value = start * 100.0;
        const double floor_tick = std::floor(tick_value);
        const double fraction = tick_value - floor_tick;
        const double start_tick = fraction < 0.5 ? floor_tick : fraction > 0.5 ? floor_tick + 1 :
            (std::fmod(floor_tick, 2.0) == 0.0 ? floor_tick : floor_tick + 1);
        double time = start_tick / 100.0;
        for (size_t i = first_event; i < tokens.size(); ++i) {
            const int token = tokens[i];
            if (token < 0 || token >= kVocabulary) throw std::runtime_error("YourMT3 predicted a token outside its vocabulary");
            if (token == 1) break;
            if (token <= 2) continue;
            if (token <= 208) {
                const int shift = token - 3;
                if (!shift) ++errors["shift_out_of_range"];
                else time = (start_tick + shift) / 100.0;
            } else if (token <= 336) {
                const int pitch = token - 209;
                if (program < 0) { ++errors["missing_program"]; continue; }
                if (velocity < 0) { ++errors["missing_velocity"]; continue; }
                const std::pair<int, int> key{program, pitch};
                if (velocity == 1) result.last_activity.push_back(key);
                else {
                    const auto found = std::find(result.last_activity.begin(), result.last_activity.end(), key);
                    if (found == result.last_activity.end()) { ++errors["note_off_without_note_on"]; continue; }
                    result.last_activity.erase(found);
                }
                result.events.push_back({time, program, pitch, velocity, false});
            } else if (token <= 338) velocity = token - 337;
            else if (token == 339) ++errors[time == start_tick / 100.0 ? "multi_tie_type_1" : "multi_tie_type_2"];
            else if (token <= 467) program = token - 340;
            else result.events.push_back({time, 128, token - 468, 1, true});
        }
        std::stable_sort(result.events.begin(), result.events.end(), [](const auto & a, const auto & b) {
            return std::tie(a.time, a.drum, a.program, a.velocity, a.pitch) <
                   std::tie(b.time, b.drum, b.program, b.velocity, b.pitch);
        });
        for (const auto & error : errors) debug::timing_log_scalar(
            ("mulacover.transcriber.decode." + error.first).c_str(), error.second);
        return result;
    }

    std::vector<TranscribedNote> merge_events(const std::vector<SegmentEvents> & segments) const {
        std::vector<NoteEvent> merged;
        const SegmentEvents * previous = nullptr;
        int missing_ties = 0, missing_onsets = 0, long_notes = 0;
        for (const auto & segment : segments) {
            if (previous) for (const auto & key : previous->last_activity) {
                if (std::find(segment.ties.begin(), segment.ties.end(), key) == segment.ties.end()) {
                    merged.push_back({segment.start, key.first, key.second, 0, false});
                    ++missing_ties;
                }
            }
            merged.insert(merged.end(), segment.events.begin(), segment.events.end());
            previous = &segment;
        }
        // note_event2note uses a different ordering than event2note_event:
        // pitch precedes velocity, so note-offs precede same-pitch onsets.
        std::stable_sort(merged.begin(), merged.end(), [](const auto & a, const auto & b) {
            return std::tie(a.time, a.drum, a.pitch, a.velocity, a.program) <
                   std::tie(b.time, b.drum, b.pitch, b.velocity, b.program);
        });
        std::map<std::pair<int, int>, NoteEvent> active;
        std::vector<TranscribedNote> notes;
        for (const auto & event : merged) {
            if (event.drum) {
                notes.push_back({event.time, event.pitch, 0.01, 128, true});
                continue;
            }
            const std::pair<int, int> key{event.pitch, event.program};
            const auto found = active.find(key);
            if (found != active.end()) {
                notes.push_back({found->second.time, event.pitch, event.time - found->second.time, event.program, false});
                active.erase(found);
            } else if (!event.velocity) ++missing_onsets;
            if (event.velocity) active.emplace(key, event);
        }
        // Upstream gives unresolved notes a 10 ms duration, not the segment end.
        for (const auto & entry : active) {
            const auto & event = entry.second;
            notes.push_back({event.time, event.pitch, 0.01, event.program, false});
        }
        for (auto & note : notes) {
            if (note.duration > 10.0) { note.duration = 0.01; ++long_notes; }
            if (!note.is_drum && note.duration < 0.01) note.duration = 0.01;
        }
        std::stable_sort(notes.begin(), notes.end(), [](const auto & a, const auto & b) {
            return std::tie(a.onset_seconds, a.is_drum, a.program, a.pitch) <
                   std::tie(b.onset_seconds, b.is_drum, b.program, b.pitch);
        });
        trim_notes(notes);
        debug::timing_log_scalar("mulacover.transcriber.decode.missing_segment_tie", missing_ties);
        debug::timing_log_scalar("mulacover.transcriber.decode.onset_not_found", missing_onsets);
        debug::timing_log_scalar("mulacover.transcriber.decode.long_note", long_notes);
        return notes;
    }

    static void trim_notes(std::vector<TranscribedNote> & notes) {
        // trim_overlapping_notes: truncate each same-(pitch,program,is_drum)
        // predecessor at the next onset, then discard zero-length notes.
        std::map<std::tuple<int, int, bool>, size_t> previous;
        for (size_t i = 0; i < notes.size(); ++i) {
            const auto & note = notes[i];
            const auto key = std::make_tuple(note.pitch, note.program, note.is_drum);
            const auto found = previous.find(key);
            if (found != previous.end()) {
                auto & preceding = notes[found->second];
                if (preceding.onset_seconds + preceding.duration > note.onset_seconds) {
                    preceding.duration = note.onset_seconds - preceding.onset_seconds;
                }
            }
            previous[key] = i;
        }
        notes.erase(std::remove_if(notes.begin(), notes.end(), [](const auto & note) {
            return note.duration <= 0;
        }), notes.end());
    }

    core::ExecutionContext execution_;
    core::BackendWeightStore store_;
    std::array<ConvBlockWeights, 3> conv_;
    std::array<PerceiverBlockWeights, 3> perceiver_;
    std::array<T5LayerWeights, 8> decoder_;
    TensorValue latents_, embedding_, positions_, score_one_;
    modules::LinearWeights projection_, lm_head_;
    modules::NormWeights final_norm_;
    std::array<std::array<TensorValue, 2>, 2> rotary_;
    std::array<TensorValue, 2> spectral_rotary_;
    Graph encoder_graph_, decoder_graph_;
    TensorValue spectrogram_input_, decoder_ids_, decoder_position_, decoder_slots_, decoder_mask_;
    ggml_tensor * decoder_output_ = nullptr;
};

Transcriber::Transcriber(std::shared_ptr<const assets::TensorSource> source,
    const core::BackendConfig & backend, assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(source), backend, storage_type)) {}
Transcriber::~Transcriber() = default;
std::vector<TranscribedNote> Transcriber::transcribe(const std::vector<float> & mono, int sample_rate) {
    return impl_->transcribe(mono, sample_rate);
}

}  // namespace engine::models::mulacover
