#include "engine/community_models/voice_isolator/network.h"

#include "engine/community_models/voice_isolator/attention_window.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/mamba_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::voice_isolator {
namespace {

using core::TensorShape;
using core::TensorValue;
using Storage = assets::TensorStorageType;

constexpr float kNormEps = 1.0e-5F;
constexpr float kRopeTheta = 10000.0F;
constexpr int kGraphNodes = 16384;
// Offline slices (see NetworkGraph::sliced): about this many tokens each, and at
// most this many per block, which bounds the graph's nodes for long chunks.
constexpr int64_t kSliceTokens = 4096;
constexpr int64_t kMaxSlices = 8;

struct AttentionBlockWeights {
    modules::NormWeights time_norm1, time_norm2, band_norm1, band_norm2;
    modules::LinearWeights time_qkv, time_proj, band_qkv, band_proj;
    modules::FeedForwardWeights time_ffn, band_ffn;
};

struct BidirectionalMambaWeights {
    modules::Mamba1Weights forward;
    std::optional<modules::Mamba1Weights> backward;
    modules::LinearWeights output_proj;
    modules::NormWeights norm;
};

struct BlockWeights {
    BlockKind kind = BlockKind::Attention;
    AttentionBlockWeights attention;
    BidirectionalMambaWeights time_mamba, freq_mamba;
};

// The per-band layers, packed: one batched matmul over every band, each band's
// weights zero-padded to the widest band.
struct EncoderWeights {
    TensorValue norm_weight, norm_bias, valid, inverse_count, weight;
    // Linear bias plus band embedding, plus the look-ahead embedding of a
    // streaming position; keyed by look-ahead.
    std::map<int64_t, TensorValue> bias;
};

struct DecoderWeights {
    TensorValue film_scale, film_shift, norm_weight, norm_bias, hidden_weight, hidden_bias, out_weight, out_bias;
};

modules::NormWeights load_norm(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t width) {
    return {store.load_tensor(source, prefix + ".weight", Storage::F32, {width}),
        store.load_tensor(source, prefix + ".bias", Storage::F32, {width})};
}

modules::LinearWeights load_linear(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & weight, const std::string & bias, int64_t in, int64_t out, Storage storage) {
    return {store.load_tensor(source, weight, storage, {out, in}),
        store.load_tensor(source, bias, Storage::F32, {out})};
}

modules::LinearWeights load_linear(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & prefix, int64_t in, int64_t out, Storage storage) {
    return load_linear(store, source, prefix + ".weight", prefix + ".bias", in, out, storage);
}

modules::FeedForwardWeights load_ffn(core::BackendWeightStore & store, const assets::TensorSource & source,
    const std::string & fc1, const std::string & fc2, int64_t width, Storage storage) {
    const auto first = load_linear(store, source, fc1, width, 4 * width, storage);
    const auto second = load_linear(store, source, fc2, 4 * width, width, storage);
    return {first.weight, first.bias, second.weight, second.bias};
}

Storage weight_storage(Storage storage, ggml_backend_t backend) {
    if (storage != Storage::Native) {
        return storage;
    }
    // BF16 halves weight traffic, but without a native BF16 dot product it runs slower than F32.
    return core::cpu_backend_has_feature(backend, "AVX512_BF16") ? Storage::BF16 : Storage::F32;
}

ggml_type attention_cache_type(Storage storage, core::BackendType backend) {
    if (storage == Storage::Native) {
        storage = backend == core::BackendType::Cpu ? Storage::Q8_0 : Storage::F16;
    }
    const ggml_type type = assets::ggml_type_for_tensor_storage(storage);
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q5_0 &&
        type != GGML_TYPE_Q4_0) {
        throw std::runtime_error("Voice Isolator attention caches are F32, F16, Q8_0, Q5_0 or Q4_0");
    }
    return type;
}

modules::Mamba1Config mamba_config(const Config & config) {
    return {config.width, config.mamba_inner(), config.mamba_state, config.mamba_conv, config.dt_rank()};
}

}  // namespace

struct Network::State {
    Config config;
    BandLayout layout;
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    EncoderWeights encoder;
    std::vector<BlockWeights> blocks;
    DecoderWeights decoder;
    ggml_type cache_type = GGML_TYPE_F32;
    int64_t rebase_frames = 0;
    // Backbones share the backend, which runs one graph at a time.
    std::mutex compute;

    State(const Config & config_in, const assets::TensorSource & source, core::ExecutionContext & execution_in,
        const NetworkOptions & options)
        : config(config_in),
          layout(make_band_layout(config_in)),
          execution(execution_in),
          store(execution.backend(), execution.backend_type(), "voice_isolator.weights", 8ull * 1024ull * 1024ull),
          cache_type(attention_cache_type(options.attention_cache_storage, execution.backend_type())),
          rebase_frames(options.rebase_frames) {
        if (config.head_dim() % ggml_blck_size(cache_type) != 0) {
            throw std::runtime_error("Voice Isolator attention heads of " + std::to_string(config.head_dim()) +
                " values do not fill whole " + ggml_type_name(cache_type) + " blocks");
        }
        const auto weights = weight_storage(options.weight_storage, execution.backend());
        load_encoder(source, weights);
        load_blocks(source, weights);
        load_decoder(source, weights);
        store.upload();
        source.release_storage();
        core::set_backend_threads(execution.backend(), execution.config().threads);
    }

    void load_encoder(const assets::TensorSource & source, Storage storage) {
        const int64_t bands = layout.bands();
        const int64_t inputs = layout.encoder_inputs();
        const int64_t width = config.width;
        std::vector<float> norm_weight(bands * inputs, 0.0F), norm_bias(bands * inputs, 0.0F);
        std::vector<float> valid(bands * inputs, 0.0F), inverse_count(bands), weight(bands * width * inputs, 0.0F);
        std::vector<float> bias(bands * width);
        const auto embedding = source.require_f32("band_embedding", {bands, width});
        for (int64_t band = 0; band < bands; ++band) {
            const auto [first, last] = layout.edges[static_cast<size_t>(band)];
            const int64_t n = 2 * (last - first);
            const auto prefix = "encoders." + std::to_string(band);
            const auto ln_weight = source.require_f32(prefix + ".0.weight", {n});
            const auto ln_bias = source.require_f32(prefix + ".0.bias", {n});
            const auto linear = source.require_f32(prefix + ".1.weight", {width, n});
            const auto linear_bias = source.require_f32(prefix + ".1.bias", {width});
            for (int64_t i = 0; i < n; ++i) {
                norm_weight[band * inputs + i] = ln_weight[i];
                norm_bias[band * inputs + i] = ln_bias[i];
                valid[band * inputs + i] = 1.0F;
            }
            inverse_count[band] = 1.0F / static_cast<float>(n);
            for (int64_t o = 0; o < width; ++o) {
                std::copy_n(linear.begin() + o * n, n, weight.begin() + (band * width + o) * inputs);
                bias[band * width + o] = linear_bias[o] + embedding[band * width + o];
            }
        }
        const auto per_input = TensorShape::from_dims({bands, 1, inputs});
        encoder.norm_weight = store.make_f32(per_input, std::move(norm_weight));
        encoder.norm_bias = store.make_f32(per_input, std::move(norm_bias));
        encoder.valid = store.make_f32(per_input, std::move(valid));
        encoder.inverse_count = store.make_f32(TensorShape::from_dims({bands, 1, 1}), std::move(inverse_count));
        encoder.weight = store.make_from_f32(TensorShape::from_dims({bands, width, inputs}), storage, std::move(weight));
        const auto bias_shape = TensorShape::from_dims({bands, 1, width});
        const auto & choices = config.lookahead_choices;
        if (choices.empty()) {
            encoder.bias.emplace(config.lookahead_frames, store.make_f32(bias_shape, bias));
            return;
        }
        const auto lookahead = source.require_f32("lookahead_embedding.weight", {static_cast<int64_t>(choices.size()), width});
        for (size_t index = 0; index < choices.size(); ++index) {
            auto shifted = bias;
            for (int64_t band = 0; band < bands; ++band) {
                for (int64_t o = 0; o < width; ++o) {
                    shifted[band * width + o] += lookahead[static_cast<int64_t>(index) * width + o];
                }
            }
            encoder.bias.emplace(choices[index], store.make_f32(bias_shape, std::move(shifted)));
        }
    }

    BidirectionalMambaWeights load_bidirectional(const assets::TensorSource & source, const std::string & prefix,
        bool causal, Storage storage) {
        const auto mamba = mamba_config(config);
        const int64_t width = config.width;
        BidirectionalMambaWeights weights;
        weights.forward = modules::binding::mamba1(store, source, prefix + ".forward_blocks", mamba, storage);
        if (!causal) {
            weights.backward = modules::binding::mamba1(store, source, prefix + ".backward_blocks", mamba, storage);
        }
        weights.output_proj = load_linear(store, source, prefix + ".output_proj", causal ? width : 2 * width, width, storage);
        weights.norm = load_norm(store, source, prefix + ".norm", width);
        return weights;
    }

    void load_blocks(const assets::TensorSource & source, Storage storage) {
        const int64_t width = config.width;
        for (size_t index = 0; index < config.blocks.size(); ++index) {
            const auto prefix = "blocks." + std::to_string(index);
            BlockWeights block;
            block.kind = config.blocks[index];
            if (block.kind == BlockKind::Attention) {
                const auto time = prefix + ".time_transformer";
                const auto band = prefix + ".band_transformer";
                auto & w = block.attention;
                w.time_norm1 = load_norm(store, source, time + ".norm1", width);
                w.time_qkv = load_linear(store, source, time + ".attn.qkv", width, 3 * width, storage);
                w.time_proj = load_linear(store, source, time + ".attn.proj", width, width, storage);
                w.time_norm2 = load_norm(store, source, time + ".norm2", width);
                w.time_ffn = load_ffn(store, source, time + ".ffn.0", time + ".ffn.3", width, storage);
                w.band_norm1 = load_norm(store, source, band + ".norm1", width);
                w.band_qkv = load_linear(store, source, band + ".self_attn.in_proj_weight",
                    band + ".self_attn.in_proj_bias", width, 3 * width, storage);
                w.band_proj = load_linear(store, source, band + ".self_attn.out_proj", width, width, storage);
                w.band_norm2 = load_norm(store, source, band + ".norm2", width);
                w.band_ffn = load_ffn(store, source, band + ".linear1", band + ".linear2", width, storage);
            } else {
                block.time_mamba = load_bidirectional(source, prefix + ".time_mamba", config.causal, storage);
                block.freq_mamba = load_bidirectional(source, prefix + ".freq_mamba", false, storage);
            }
            blocks.push_back(std::move(block));
        }
    }

    void load_decoder(const assets::TensorSource & source, Storage storage) {
        const int64_t bands = layout.bands();
        const int64_t width = config.width;
        const int64_t outputs = layout.decoder_outputs();
        // The condition layer reads constant controls (1, 0): a fixed scale and shift.
        const auto condition = source.require_f32("denoise_decoder.condition.weight", {2 * width, 2});
        const auto condition_bias = source.require_f32("denoise_decoder.condition.bias", {2 * width});
        std::vector<float> scale(width), shift(width);
        for (int64_t j = 0; j < width; ++j) {
            scale[j] = 1.0F + condition[j * 2] + condition_bias[j];
            shift[j] = condition[(width + j) * 2] + condition_bias[width + j];
        }
        std::vector<float> norm_weight(bands * width), norm_bias(bands * width), hidden_weight(bands * width * width);
        std::vector<float> hidden_bias(bands * width), out_weight(bands * outputs * width, 0.0F), out_bias(bands * outputs, 0.0F);
        for (int64_t band = 0; band < bands; ++band) {
            const auto [first, last] = layout.edges[static_cast<size_t>(band)];
            const int64_t n = 4 * (last - first);
            const auto prefix = "denoise_decoder.heads." + std::to_string(band);
            const auto ln_weight = source.require_f32(prefix + ".0.weight", {width});
            const auto ln_bias = source.require_f32(prefix + ".0.bias", {width});
            const auto hidden = source.require_f32(prefix + ".1.weight", {width, width});
            const auto hidden_b = source.require_f32(prefix + ".1.bias", {width});
            const auto out = source.require_f32(prefix + ".3.weight", {n, width});
            const auto out_b = source.require_f32(prefix + ".3.bias", {n});
            std::copy(ln_weight.begin(), ln_weight.end(), norm_weight.begin() + band * width);
            std::copy(ln_bias.begin(), ln_bias.end(), norm_bias.begin() + band * width);
            std::copy(hidden.begin(), hidden.end(), hidden_weight.begin() + band * width * width);
            std::copy(hidden_b.begin(), hidden_b.end(), hidden_bias.begin() + band * width);
            std::copy(out.begin(), out.end(), out_weight.begin() + band * outputs * width);
            std::copy(out_b.begin(), out_b.end(), out_bias.begin() + band * outputs);
        }
        const auto vector = TensorShape::from_dims({width});
        const auto per_band = TensorShape::from_dims({bands, 1, width});
        decoder.film_scale = store.make_f32(vector, std::move(scale));
        decoder.film_shift = store.make_f32(vector, std::move(shift));
        decoder.norm_weight = store.make_f32(per_band, std::move(norm_weight));
        decoder.norm_bias = store.make_f32(per_band, std::move(norm_bias));
        decoder.hidden_weight = store.make_from_f32(TensorShape::from_dims({bands, width, width}), storage, std::move(hidden_weight));
        decoder.hidden_bias = store.make_f32(per_band, std::move(hidden_bias));
        decoder.out_weight = store.make_from_f32(TensorShape::from_dims({bands, outputs, width}), storage, std::move(out_weight));
        decoder.out_bias = store.make_f32(TensorShape::from_dims({bands, 1, outputs}), std::move(out_bias));
    }
};

namespace {

using State = Network::State;

// A stream's carried state: per attention block the time keys and values by
// slot, per Mamba block the time scan's convolution and recurrent state.
struct StreamTensors {
    std::vector<TensorValue> keys, values;
    std::vector<modules::Mamba1State> mamba;
};

// Index rows uploaded once into a graph set's persistent storage: the set's
// graphs share their scratch memory, so nothing uploaded once may live there.
class IndexTable {
public:
    ggml_tensor * add(core::PreparedGraph & prepared, std::vector<int32_t> values, int64_t ne0, int64_t ne1 = 1) {
        auto * tensor = ggml_new_tensor_2d(prepared.persistent_build_context().ggml, GGML_TYPE_I32, ne0, ne1);
        values_.emplace_back(tensor, std::move(values));
        return tensor;
    }

    void upload() const {
        for (const auto & [tensor, values] : values_) {
            ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(int32_t));
        }
    }

private:
    std::vector<std::pair<ggml_tensor *, std::vector<int32_t>>> values_;
};

// What a network graph indexes with: an identity at least as long as its
// frames and bands; per frame its bands reversed ([bands, >= frames]); offline,
// per band its frames reversed.
struct GraphIndices {
    ggml_tensor * identity = nullptr;
    ggml_tensor * band_reverse = nullptr;
    ggml_tensor * time_reverse = nullptr;
};

core::PreparedGraphConfig graph_config(const char * label, size_t graphs, size_t persistent_tensors) {
    return {label, graphs * (kGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphNodes, false)),
        kGraphNodes, persistent_tensors * ggml_tensor_overhead()};
}

// One graph of a PreparedGraph set.
class GraphBase {
protected:
    GraphBase(const State & state, core::PreparedGraph & prepared, core::PreparedGraph::GraphId id)
        : state_(state), prepared_(prepared), id_(id), ctx_(prepared.build_context()) {}

    ggml_tensor * input(ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1) {
        auto * tensor = ggml_new_tensor_3d(ctx_.ggml, type, ne0, ne1, ne2);
        ggml_set_input(tensor);
        return tensor;
    }

    void finish(std::vector<ggml_tensor *> outputs) {
        auto * graph = prepared_.graph(id_);
        for (auto * output : outputs) {
            ggml_set_output(output);
            ggml_build_forward_expand(graph, output);
        }
        // State is committed after every reader of the old state.
        for (auto * update : updates_) {
            ggml_build_forward_expand(graph, update);
        }
    }

    void compute() { prepared_.compute(id_); }

    const State & state_;
    core::PreparedGraph & prepared_;
    core::PreparedGraph::GraphId id_;
    core::ModuleBuildContext & ctx_;
    std::vector<ggml_tensor *> updates_;
};

std::vector<int32_t> identity(int64_t count) {
    std::vector<int32_t> values(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    return values;
}

// Index rows [length, sequences] reversing each sequence.
std::vector<int32_t> reversal(int64_t length, int64_t sequences) {
    std::vector<int32_t> values(static_cast<size_t>(length * sequences));
    for (int64_t s = 0; s < sequences; ++s) {
        for (int64_t t = 0; t < length; ++t) {
            values[static_cast<size_t>(s * length + t)] = static_cast<int32_t>(length - 1 - t);
        }
    }
    return values;
}

// The network over a block of frames. Offline: every frame sees the whole
// block. Streaming: the time attention reads and extends the stream's cache
// and the time Mamba carries its state.
class NetworkGraph : public GraphBase {
public:
    NetworkGraph(const State & state, core::PreparedGraph & prepared, core::PreparedGraph::GraphId id, int64_t frames,
        int64_t lookahead, const GraphIndices & indices, const StreamTensors * stream, int64_t capacity)
        : GraphBase(state, prepared, id), frames_(frames), stream_(stream), capacity_(capacity) {
        const auto & layout = state.layout;
        const int64_t bands = layout.bands();
        packed_ = input(GGML_TYPE_F32, layout.encoder_inputs(), frames, bands);
        identity_ = indices.identity;
        band_reverse_ = indices.band_reverse;
        if (stream != nullptr) {
            positions_ = wrap(input(GGML_TYPE_I32, frames), {frames});
            slots_ = input(GGML_TYPE_I32, frames);
            mask_ = input(GGML_TYPE_F16, capacity, frames);
        } else {
            positions_ = ids(frames);
            time_reverse_ = indices.time_reverse;
        }
        auto x = encode(lookahead);
        size_t attention = 0;
        size_t mamba = 0;
        for (const auto & block : state.blocks) {
            if (block.kind == BlockKind::Attention) {
                const size_t index = attention++;
                x = sliced(x, [&](const TensorValue & part) { return time_attention(part, block.attention, index); });
                x = to_time_layout(sliced(to_band_layout(x),
                    [&](const TensorValue & part) { return band_attention(part, block.attention); }));
            } else {
                const auto carried = stream_state(mamba++);
                x = sliced(x, [&](const TensorValue & part) {
                    return add(part, bidirectional(part, block.time_mamba, time_reverse_, carried));
                });
                x = to_time_layout(sliced(to_band_layout(x), [&](const TensorValue & part) {
                    return add(part, bidirectional(part, block.freq_mamba, band_reverse_, std::nullopt));
                }));
            }
        }
        raw_ = decode(x).tensor;
        finish({raw_});
    }

    void run(const float * packed, float * raw, const AttentionWindow::Step * step = nullptr,
        const std::vector<ggml_fp16_t> * mask = nullptr) {
        ggml_backend_tensor_set(packed_, packed, 0, ggml_nbytes(packed_));
        if (stream_ != nullptr) {
            ggml_backend_tensor_set(positions_.tensor, step->positions.data(), 0, ggml_nbytes(positions_.tensor));
            ggml_backend_tensor_set(slots_, step->slots.data(), 0, ggml_nbytes(slots_));
            ggml_backend_tensor_set(mask_, mask->data(), 0, ggml_nbytes(mask_));
        }
        compute();
        ggml_backend_tensor_get(raw_, raw, 0, ggml_nbytes(raw_));
    }

private:
    static TensorValue wrap(ggml_tensor * tensor, std::initializer_list<int64_t> dims) {
        return core::wrap_tensor(tensor, TensorShape::from_dims(dims), tensor->type);
    }

    // Features in time layout: [bands, frames, width] (sequences along time).
    TensorValue time_value(ggml_tensor * tensor) const {
        return wrap(tensor, {state_.layout.bands(), frames_, state_.config.width});
    }

    // Features in band layout: [frames, bands, width] (sequences across bands).
    TensorValue band_value(ggml_tensor * tensor) const {
        return wrap(tensor, {frames_, state_.layout.bands(), state_.config.width});
    }

    TensorValue to_band_layout(const TensorValue & x) {
        return band_value(ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml, x.tensor, 0, 2, 1, 3)));
    }

    TensorValue to_time_layout(const TensorValue & x) {
        return time_value(ggml_cont(ctx_.ggml, ggml_permute(ctx_.ggml, x.tensor, 0, 2, 1, 3)));
    }

    TensorValue ids(int64_t count) const {
        return wrap(ggml_view_1d(ctx_.ggml, identity_, count, 0), {count});
    }

    // Offline, a block runs over slices of its sequences (the outer axis) so that
    // only one slice's wide intermediates, the 4x feed-forward and the Mamba
    // projections, are alive at a time.
    template <typename Block>
    TensorValue sliced(const TensorValue & x, Block block) {
        auto * g = ctx_.ggml;
        auto * t = x.tensor;
        const int64_t total = t->ne[2];
        const int64_t rows = std::max({int64_t{1}, kSliceTokens / t->ne[1], (total + kMaxSlices - 1) / kMaxSlices});
        if (stream_ != nullptr || rows >= total) {
            return block(x);
        }
        // Each slice reads only its own rows, so its output can replace them.
        auto * joined = t;
        for (int64_t first = 0; first < total; first += rows) {
            const int64_t count = std::min(rows, total - first);
            auto * part = ggml_view_3d(g, t, t->ne[0], t->ne[1], count, t->nb[1], t->nb[2], first * t->nb[2]);
            joined = ggml_set_inplace(g, joined, block(wrap(part, {count, t->ne[1], t->ne[0]})).tensor, joined->nb[1],
                joined->nb[2], joined->nb[3], first * joined->nb[2]);
        }
        return core::wrap_tensor(joined, x.shape);
    }

    TensorValue add(const TensorValue & a, const TensorValue & b) {
        return core::wrap_tensor(ggml_add(ctx_.ggml, a.tensor, b.tensor), a.shape);
    }

    std::optional<modules::Mamba1State> stream_state(size_t index) const {
        if (stream_ == nullptr) {
            return std::nullopt;
        }
        return stream_->mamba[index];
    }

    TensorValue encode(int64_t lookahead) {
        const auto & w = state_.encoder;
        auto * g = ctx_.ggml;
        auto * x = packed_;
        // LayerNorm over each band's own bins: the padding stays out of the statistics.
        auto * mean = ggml_mul(g, ggml_sum_rows(g, x), w.inverse_count.tensor);
        auto * centered = ggml_mul(g, ggml_sub(g, x, mean), w.valid.tensor);
        auto * variance = ggml_mul(g, ggml_sum_rows(g, ggml_sqr(g, centered)), w.inverse_count.tensor);
        auto * normalized = ggml_div(g, centered, ggml_sqrt(g, ggml_scale_bias(g, variance, 1.0F, kNormEps)));
        normalized = ggml_add(g, ggml_mul(g, normalized, w.norm_weight.tensor), w.norm_bias.tensor);
        const auto bias = w.bias.find(lookahead);
        if (bias == w.bias.end()) {
            throw std::runtime_error("Voice Isolator look-ahead " + std::to_string(lookahead) + " was not trained");
        }
        return time_value(ggml_add(g, ggml_mul_mat(g, w.weight.tensor, normalized), bias->second.tensor));
    }

    TensorValue decode(const TensorValue & x) {
        const auto & w = state_.decoder;
        auto * g = ctx_.ggml;
        auto * h = ggml_add(g, ggml_mul(g, x.tensor, w.film_scale.tensor), w.film_shift.tensor);
        h = ggml_norm(g, h, kNormEps);
        h = ggml_add(g, ggml_mul(g, h, w.norm_weight.tensor), w.norm_bias.tensor);
        h = ggml_add(g, ggml_mul_mat(g, w.hidden_weight.tensor, h), w.hidden_bias.tensor);
        const auto gelu = ctx_.backend_type == core::BackendType::Cpu
            ? modules::GeluApproximation::ApproxErf : modules::GeluApproximation::ExactErf;
        h = modules::GeluModule({gelu}).build(ctx_, time_value(h)).tensor;
        return wrap(ggml_add(g, ggml_mul_mat(g, w.out_weight.tensor, h), w.out_bias.tensor),
            {state_.layout.bands(), frames_, state_.layout.decoder_outputs()});
    }

    // output - (output . value / |value|^2) value, per head and token: removes
    // each token's component along its own value.
    ggml_tensor * exclusive(ggml_tensor * output, ggml_tensor * value) {
        auto * g = ctx_.ggml;
        auto * v = ggml_cont(g, value);
        auto * dot = ggml_sum_rows(g, ggml_mul(g, output, v));
        auto * norm = ggml_clamp(g, ggml_sum_rows(g, ggml_sqr(g, v)), 1.0e-6F, FLT_MAX);
        return ggml_sub(g, output, ggml_mul(g, v, ggml_div(g, dot, norm)));
    }

    TensorValue transformer_tail(const TensorValue & x, ggml_tensor * attended, int64_t width,
        const modules::LinearWeights & proj, const modules::NormWeights & norm2,
        const modules::FeedForwardWeights & ffn) {
        auto * o = ggml_reshape_3d(ctx_.ggml, attended, width, x.tensor->ne[1], x.tensor->ne[2]);
        auto y = add(x, modules::LinearModule({width, width, true}).build(ctx_, core::wrap_tensor(o, x.shape), proj));
        auto normed = modules::LayerNormModule({width, kNormEps}).build(ctx_, y, norm2);
        const auto gelu = ctx_.backend_type == core::BackendType::Cpu
            ? modules::GeluApproximation::ApproxErf : modules::GeluApproximation::ExactErf;
        return add(y, modules::FeedForwardModule({width, 4 * width, true, gelu})
            .build(ctx_, normed, ffn));
    }

    // Self attention along time within each band, rotary positions; causal
    // streams read the cache under the window mask.
    TensorValue time_attention(const TensorValue & x, const AttentionBlockWeights & w, size_t index) {
        auto * g = ctx_.ggml;
        const auto & c = state_.config;
        const int64_t width = c.width, heads = c.num_heads, dim = c.head_dim();
        const int64_t frames = x.tensor->ne[1], bands = x.tensor->ne[2];
        auto h = modules::LayerNormModule({width, kNormEps}).build(ctx_, x, w.time_norm1);
        auto * qkv = modules::LinearModule({width, 3 * width, true}).build(ctx_, h, w.time_qkv).tensor;
        auto part = [&](int64_t i) {
            return ggml_view_4d(g, qkv, dim, heads, frames, bands, dim * sizeof(float), qkv->nb[1], qkv->nb[2],
                i * width * sizeof(float));
        };
        const modules::RoPEModule rope({dim, GGML_ROPE_TYPE_NEOX, kRopeTheta});
        auto * q = rope.build(ctx_, wrap(part(0), {bands, frames, heads, dim}), positions_).tensor;
        auto * k = rope.build(ctx_, wrap(part(1), {bands, frames, heads, dim}), positions_).tensor;
        auto * v = part(2);
        TensorValue keys, values;
        std::optional<TensorValue> mask;
        if (stream_ != nullptr) {
            // The caches are head-major, [dim, slots, heads * bands]: each head's
            // keys lie contiguous for the attention to stream through.
            auto head_rows = [&](ggml_tensor * t) {
                auto * framed = ggml_cont(g, ggml_permute(g, t, 0, 2, 1, 3));
                return ggml_reshape_3d(g, framed, dim, frames, heads * bands);
            };
            auto heads_view = [&](ggml_tensor * cache, ggml_tensor * t) {
                auto * written = ggml_set_rows(g, cache, head_rows(t), slots_);
                return wrap(ggml_view_4d(g, written, dim, capacity_, heads, bands, written->nb[1], written->nb[2],
                    written->nb[2] * heads, 0), {bands, heads, capacity_, dim});
            };
            keys = heads_view(stream_->keys[index].tensor, k);
            values = heads_view(stream_->values[index].tensor, v);
            mask = core::wrap_tensor(mask_, TensorShape::from_dims({1, 1, frames, capacity_}), GGML_TYPE_F16);
        } else {
            keys = wrap(ggml_permute(g, k, 0, 2, 1, 3), {bands, heads, frames, dim});
            values = wrap(ggml_permute(g, v, 0, 2, 1, 3), {bands, heads, frames, dim});
        }
        auto attended = modules::ScaledDotProductAttentionModule({dim,
            ctx_.backend_type == core::BackendType::WebGPU
                ? modules::ScaledDotProductAttentionLowering::Flash
                : modules::ScaledDotProductAttentionLowering::FlashPreserveViews, GGML_PREC_F32,
            modules::AttentionCausality::NonCausal})
            .build(ctx_, wrap(ggml_permute(g, q, 0, 2, 1, 3), {bands, heads, frames, dim}), keys, values, mask).tensor;
        if (c.exclusive_time) {
            attended = exclusive(attended, v);
        }
        return transformer_tail(x, attended, width, w.time_proj, w.time_norm2, w.time_ffn);
    }

    // Self attention across the bands of each frame.
    TensorValue band_attention(const TensorValue & x, const AttentionBlockWeights & w) {
        auto * g = ctx_.ggml;
        const auto & c = state_.config;
        const int64_t width = c.width, heads = c.num_heads, dim = c.head_dim();
        const int64_t bands = x.tensor->ne[1], frames = x.tensor->ne[2];
        auto h = modules::LayerNormModule({width, kNormEps}).build(ctx_, x, w.band_norm1);
        auto * qkv = modules::LinearModule({width, 3 * width, true}).build(ctx_, h, w.band_qkv).tensor;
        auto part = [&](int64_t i) {
            return ggml_view_4d(g, qkv, dim, heads, bands, frames, dim * sizeof(float), qkv->nb[1], qkv->nb[2],
                i * width * sizeof(float));
        };
        auto heads_of = [&](ggml_tensor * t) { return wrap(ggml_permute(g, t, 0, 2, 1, 3), {frames, heads, bands, dim}); };
        auto attended = modules::ScaledDotProductAttentionModule({dim,
            ctx_.backend_type == core::BackendType::WebGPU
                ? modules::ScaledDotProductAttentionLowering::Flash
                : modules::ScaledDotProductAttentionLowering::FlashPreserveViews, GGML_PREC_F32,
            modules::AttentionCausality::NonCausal})
            .build(ctx_, heads_of(part(0)), heads_of(part(1)), heads_of(part(2))).tensor;
        if (c.exclusive_band) {
            attended = exclusive(attended, part(2));
        }
        return transformer_tail(x, attended, width, w.band_proj, w.band_norm2, w.band_ffn);
    }

    // Forward scan, plus the backward scan of the reversed sequences when the
    // weights have one, projected together and normalized. reverse: index rows
    // [length, >= sequences], each reversing a sequence.
    TensorValue bidirectional(const TensorValue & x, const BidirectionalMambaWeights & w, ggml_tensor * reverse,
        const std::optional<modules::Mamba1State> & carried) {
        auto * g = ctx_.ggml;
        const int64_t width = state_.config.width;
        const int64_t sequences = x.tensor->ne[2];
        const auto sequence_ids = ids(sequences);
        const modules::Mamba1Module mamba(mamba_config(state_.config));
        const auto preserve_range = [&](const TensorValue & output) {
            if (ctx_.backend_type != core::BackendType::Metal || state_.config.causal) return;
            // LinearModule wraps its projection in a reshape and optional bias.
            auto * projection = output.tensor;
            while (projection->op == GGML_OP_RESHAPE || projection->op == GGML_OP_ADD) {
                projection = projection->src[0];
            }
            GGML_ASSERT(projection->op == GGML_OP_MUL_MAT);
            // Offline scan outputs exceed FP16 range before normalization.
            if (projection->src[0]->type != GGML_TYPE_F32) {
                projection->src[0] = ggml_cast(g, projection->src[0], GGML_TYPE_F32);
            }
            ggml_mul_mat_set_lowering(projection, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
        };
        auto forward = mamba.build(ctx_, x, w.forward, sequence_ids, carried);
        preserve_range(forward.output);
        if (forward.next_state.has_value()) {
            updates_.push_back(ggml_cpy(g, forward.next_state->conv.tensor, carried->conv.tensor));
            updates_.push_back(ggml_cpy(g, forward.next_state->ssm.tensor, carried->ssm.tensor));
        }
        auto * joined = ggml_add(g, forward.output.tensor, x.tensor);
        if (w.backward.has_value()) {
            if (reverse == nullptr) {
                throw std::runtime_error("Voice Isolator streams need a causal checkpoint");
            }
            reverse = ggml_view_2d(g, reverse, reverse->ne[0], sequences, reverse->nb[1], 0);
            auto reversed = core::wrap_tensor(ggml_get_rows(g, x.tensor, reverse), x.shape);
            auto backward_output = mamba.build(ctx_, reversed, *w.backward, sequence_ids).output;
            preserve_range(backward_output);
            auto * backward = ggml_get_rows(g, backward_output.tensor, reverse);
            joined = ggml_concat(g, joined, ggml_add(g, backward, x.tensor), 0);
        }
        const auto shape = x.shape.with_last_dim(joined->ne[0]);
        auto projected = modules::LinearModule({joined->ne[0], width, true})
            .build(ctx_, core::wrap_tensor(joined, shape), w.output_proj);
        preserve_range(projected);
        return modules::LayerNormModule({width, kNormEps}).build(ctx_, projected, w.norm);
    }

    int64_t frames_ = 0;
    const StreamTensors * stream_ = nullptr;
    int64_t capacity_ = 0;
    ggml_tensor * packed_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * slots_ = nullptr;
    ggml_tensor * band_reverse_ = nullptr;
    ggml_tensor * time_reverse_ = nullptr;
    ggml_tensor * raw_ = nullptr;
    ggml_tensor * identity_ = nullptr;
    TensorValue positions_;
};

// Rotates every cached key of a stream back by a runtime number of positions,
// in place, a span of slots at a time so the F32 working copies stay small.
class RebaseGraph : public GraphBase {
public:
    RebaseGraph(const State & state, core::PreparedGraph & prepared, core::PreparedGraph::GraphId id,
        const StreamTensors & stream, ggml_tensor * identity)
        : GraphBase(state, prepared, id) {
        auto * g = ctx_.ggml;
        const int64_t dim = state.config.head_dim();
        const auto & first = stream.keys.front().tensor;
        const int64_t slots = first->ne[1], rows = first->ne[2];
        shift_ = input(GGML_TYPE_I32, rows);
        const auto positions = core::wrap_tensor(shift_, TensorShape::from_dims({rows}), GGML_TYPE_I32);
        const modules::RoPEModule rope({dim, GGML_ROPE_TYPE_NEOX, kRopeTheta});
        constexpr int64_t span = 128;
        std::vector<ggml_tensor *> outputs;
        for (const auto & cache : stream.keys) {
            auto * keys = cache.tensor;
            for (int64_t start = 0; start < slots; start += span) {
                const int64_t count = std::min(span, slots - start);
                auto * part = ggml_cast(g, ggml_view_3d(g, keys, dim, count, rows, keys->nb[1], keys->nb[2],
                    start * keys->nb[1]), GGML_TYPE_F32);
                auto rotated = rope.build(ctx_, core::wrap_tensor(part, TensorShape::from_dims({1, rows, count, dim}),
                    GGML_TYPE_F32), positions);
                outputs.push_back(ggml_set_rows(g, keys, rotated.tensor,
                    ggml_view_1d(g, identity, count, static_cast<size_t>(start) * sizeof(int32_t))));
            }
        }
        finish(outputs);
    }

    void run(int64_t shift) {
        const std::vector<int32_t> positions(static_cast<size_t>(shift_->ne[0]), static_cast<int32_t>(-shift));
        ggml_backend_tensor_set(shift_, positions.data(), 0, ggml_nbytes(shift_));
        compute();
    }

private:
    ggml_tensor * shift_ = nullptr;
};

class OfflineBackbone final : public Backbone {
public:
    explicit OfflineBackbone(std::shared_ptr<State> state) : state_(std::move(state)) {}

    void run(const float * packed, int64_t frames, float * raw) override {
        std::lock_guard<std::mutex> lock(state_->compute);
        // Chunks of one clip share a length: one graph is kept.
        if (!graph_ || frames != frames_) {
            graph_.reset();
            prepared_.reset();
            prepared_ = std::make_unique<core::PreparedGraph>(state_->execution, graph_config("voice_isolator.offline", 1, 3));
            const int64_t bands = state_->layout.bands();
            const int64_t length = std::max(bands, frames);
            IndexTable table;
            const GraphIndices indices{table.add(*prepared_, identity(length), length),
                table.add(*prepared_, reversal(bands, frames), bands, frames),
                table.add(*prepared_, reversal(frames, bands), frames, bands)};
            prepared_->allocate_persistent();
            table.upload();
            graph_ = std::make_unique<NetworkGraph>(*state_, *prepared_, 0, frames, state_->config.lookahead_frames, indices,
                nullptr, 0);
            prepared_->allocate();
            frames_ = frames;
        }
        graph_->run(packed, raw);
    }

    void reset() override {}

private:
    std::shared_ptr<State> state_;
    std::unique_ptr<core::PreparedGraph> prepared_;
    std::unique_ptr<NetworkGraph> graph_;
    int64_t frames_ = 0;
};

class StreamingBackbone final : public Backbone {
public:
    StreamingBackbone(std::shared_ptr<State> state, int64_t lookahead, int64_t block_frames)
        : state_(std::move(state)),
          lookahead_(lookahead),
          block_frames_(block_frames),
          window_(make_window(*state_, block_frames)),
          prepared_(state_->execution, graph_config("voice_isolator.stream", static_cast<size_t>(block_frames) + 1,
              2 * state_->blocks.size() + 2)) {
        const auto indices = allocate();
        // Every block size a push can produce, built now so no push pays for one.
        for (int64_t count = 1; count <= block_frames_; ++count) {
            const auto id = count == 1 ? 0 : prepared_.add_graph("frames_" + std::to_string(count));
            graphs_[count] = std::make_unique<NetworkGraph>(*state_, prepared_, id, count, lookahead_, indices, &tensors_,
                window_.capacity());
        }
        if (!tensors_.keys.empty()) {
            rebase_ = std::make_unique<RebaseGraph>(*state_, prepared_, prepared_.add_graph("rebase"), tensors_, indices.identity);
        }
        prepared_.allocate();
    }

    void run(const float * packed, int64_t frames, float * raw) override {
        const auto & layout = state_->layout;
        std::lock_guard<std::mutex> lock(state_->compute);
        for (int64_t first = 0; first < frames; first += block_frames_) {
            const int64_t count = std::min(block_frames_, frames - first);
            // The block's rows of the band-major input and output.
            std::vector<float> block_in(static_cast<size_t>(layout.bands() * count * layout.encoder_inputs()));
            std::vector<float> block_out(static_cast<size_t>(layout.bands() * count * layout.decoder_outputs()));
            for (int64_t band = 0; band < layout.bands(); ++band) {
                std::copy_n(packed + (band * frames + first) * layout.encoder_inputs(), count * layout.encoder_inputs(),
                    block_in.begin() + band * count * layout.encoder_inputs());
            }
            run_block(block_in.data(), count, block_out.data());
            for (int64_t band = 0; band < layout.bands(); ++band) {
                std::copy_n(block_out.begin() + band * count * layout.decoder_outputs(), count * layout.decoder_outputs(),
                    raw + (band * frames + first) * layout.decoder_outputs());
            }
        }
    }

    void reset() override {
        window_.reset();
        prepared_.clear_persistent();
        indices_.upload();
    }

private:
    static AttentionWindow make_window(const State & state, int64_t block_frames) {
        const auto & c = state.config;
        const int64_t retained = c.attention_window_frames > 0
            ? c.attention_window_frames
            : std::max<int64_t>(1, std::llround(c.chunk_seconds * static_cast<double>(c.sample_rate) /
                static_cast<double>(c.hop_length)));
        return AttentionWindow(c.attention_window_frames, retained, block_frames,
            std::max(state.rebase_frames, retained + block_frames));
    }

    // The stream's state and index rows, in the graph set's persistent storage.
    GraphIndices allocate() {
        const auto & c = state_->config;
        const int64_t bands = state_->layout.bands();
        auto & ctx = prepared_.persistent_build_context();
        auto make = [&](ggml_type type, std::initializer_list<int64_t> dims) {
            return core::make_tensor(ctx, type, TensorShape::from_dims(dims));
        };
        const int64_t rows = c.num_heads * bands;
        for (const auto & block : state_->blocks) {
            if (block.kind == BlockKind::Attention) {
                tensors_.keys.push_back(make(state_->cache_type, {rows, window_.capacity(), c.head_dim()}));
                tensors_.values.push_back(make(state_->cache_type, {rows, window_.capacity(), c.head_dim()}));
            } else {
                tensors_.mamba.push_back({make(GGML_TYPE_F32, {bands, c.mamba_inner(), c.mamba_conv - 1}),
                    make(GGML_TYPE_F32, {bands, c.mamba_inner(), 1, c.mamba_state})});
            }
        }
        const int64_t length = std::max({window_.capacity(), bands, block_frames_});
        const GraphIndices indices{indices_.add(prepared_, identity(length), length),
            indices_.add(prepared_, reversal(bands, block_frames_), bands, block_frames_)};
        prepared_.allocate_persistent();
        reset();
        return indices;
    }

    void run_block(const float * packed, int64_t count, float * raw) {
        const auto step = window_.advance(count);
        if (step.rebase > 0 && rebase_) {
            rebase_->run(step.rebase);
        }
        mask_.resize(static_cast<size_t>(count * window_.capacity()));
        window_.fill_mask(step, mask_.data());
        mask_f16_.resize(mask_.size());
        ggml_fp32_to_fp16_row(mask_.data(), mask_f16_.data(), static_cast<int64_t>(mask_.size()));
        graphs_.at(count)->run(packed, raw, &step, &mask_f16_);
    }

    std::shared_ptr<State> state_;
    int64_t lookahead_ = 0;
    int64_t block_frames_ = 1;
    AttentionWindow window_;
    core::PreparedGraph prepared_;
    IndexTable indices_;
    StreamTensors tensors_;
    std::map<int64_t, std::unique_ptr<NetworkGraph>> graphs_;
    std::unique_ptr<RebaseGraph> rebase_;
    std::vector<float> mask_;
    std::vector<ggml_fp16_t> mask_f16_;
};

}  // namespace

Network::Network(const Config & config, const assets::TensorSource & source, core::ExecutionContext & execution,
    const NetworkOptions & options)
    : state_(std::make_shared<State>(config, source, execution, options)) {}

Network::~Network() = default;

const Config & Network::config() const noexcept {
    return state_->config;
}

const BandLayout & Network::layout() const noexcept {
    return state_->layout;
}

std::unique_ptr<Backbone> Network::offline_backbone() const {
    if (state_->config.causal) {
        throw std::runtime_error("This Voice Isolator checkpoint only streams");
    }
    return std::make_unique<OfflineBackbone>(state_);
}

std::unique_ptr<Backbone> Network::streaming_backbone(int64_t lookahead_frames, int64_t block_frames) const {
    if (!state_->config.causal) {
        throw std::runtime_error("This Voice Isolator checkpoint has no streaming mode: it was trained offline only");
    }
    if (block_frames < 1) {
        throw std::runtime_error("Voice Isolator streams process at least one frame per block");
    }
    return std::make_unique<StreamingBackbone>(state_, lookahead_frames, block_frames);
}

}  // namespace engine::models::voice_isolator
