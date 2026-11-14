#include "engine/community_models/zonos3/codec_decoder.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"

#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <complex>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos3 {

namespace detail {

// One decode stream: device state (MOSS K/V rings, ConvNeXt histories) in its own context + buffer, plus the host
// iSTFT / mixer / resampler holdback.
struct Zonos3DecoderState {
    std::array<double, 6> timing_ms{};  // graph/input preparation, compute, readback, waveform head, resample, flush
    uint64_t id = 0;  // decode graphs bind the device tensors of one stream
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> keys, values;  // per MOSS layer, stage-major: F32 [head_dim, cap_s, n_head]
    std::vector<ggml_tensor *> history;       // per ConvNeXt block (4 semantic, 8 head): F32 [C, 6]
    int64_t frames_done = 0;  // 15 Hz frames decoded
    // iSTFT per branch: overlap-add and window envelope of padded positions [ola_base, ..).
    std::array<std::vector<float>, 2> ola, env;
    std::array<int64_t, 2> ola_base{};
    int64_t head_frames = 0;  // 240 Hz head frames synthesized
    bool head_done = false;   // flushed
    // Mixer input per branch, offset by the FIR half-width h: index j holds branch sample j - h; mix_in[c][0] has
    // absolute index mix_base, mix_avail[c] indices exist, mix_next is the next output sample.
    std::array<std::vector<float>, 2> mix_in;
    std::array<int64_t, 2> mix_avail{};
    int64_t mix_base = 0;
    int64_t mix_next = 0;
    // Resampler: retained internal samples (rs_buf[0] is absolute index rs_base), samples received, next block.
    std::vector<float> rs_buf;
    int64_t rs_base = 0;
    int64_t rs_in = 0;
    int64_t rs_next_block = 0;
};

}  // namespace detail

namespace {

using core::TensorValue;
using State = detail::Zonos3DecoderState;

constexpr int kStages = 5;
constexpr int kMaxChunk = 32;     // frames per graph call
constexpr int kCachedGraphs = 4;  // decode graphs kept (one per chunk length)
constexpr int kGraphNodes = 8192;
constexpr int kConvKernel = 7;

struct StageHp {
    int in_dim = 0, out_dim = 0, d_model = 0, n_head = 0, n_layer = 0, n_ff = 0, context = 0, patch = 0;
};

struct ConvNextWeights {
    TensorValue dw;  // logical {K, C}: tap-major
    TensorValue dw_bias, norm_weight, norm_bias, pw1, pw1_bias, pw2, pw2_bias;
};

struct MossWeights {
    TensorValue norm1_weight, norm1_bias, qkv, wo, attn_scale, norm2_weight, norm2_bias, ffn1, ffn2, ffn_scale;
};

struct StageWeights {
    TensorValue in, out;  // out absent for the last stage
    std::vector<MossWeights> layers;
};

struct StageInputs {
    ggml_tensor * pos = nullptr;    // I32 [T]
    ggml_tensor * slots = nullptr;  // I32 [T]
    ggml_tensor * mask = nullptr;   // F16 [cap, T]
    int cpf = 1;
    int columns = 0;
};

struct DecodeGraph {
    int frames = 0;
    uint64_t stream = 0;  // id of the stream whose state tensors the graph reads and writes
    uint64_t last_use = 0;
    std::unique_ptr<core::PreparedGraph> prepared;
    std::array<ggml_tensor *, 10> ids{};
    std::vector<StageInputs> stages;
    std::array<ggml_tensor *, 2> spec{};  // [n_fft + 2, 16 * frames]
};

TensorValue matrix(ggml_tensor * x) {
    return core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[1], x->ne[0]}));
}

[[noreturn]] void reject(const std::string & why) {
    throw std::runtime_error("zonos3: decoder GGUF is not a ZONOS3 Mini iSTFT decoder (mossdual_istft): " + why);
}

}  // namespace

class Zonos3CodecDecoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : source_(std::move(source)), execution_(execution) {
        if (!source_) throw std::runtime_error("zonos3: decoder tensor source is missing");
        read_metadata();
        load_weights();
        compute_ring_capacities();
        for (int i = 0; i < 2; ++i) irfft_[static_cast<size_t>(i)] = audio::get_real_fft_plan(static_cast<size_t>(n_fft_[static_cast<size_t>(i)]));
    }

    ~Impl() { graphs_.clear(); }

    int sample_rate() const { return sample_rate_; }
    int samples_per_frame() const { return hop_; }
    float headroom_db() const { return headroom_db_; }

    // Allocates a stream's zeroed K/V rings and ConvNeXt histories on the decoder backend.
    void open(State & st) {
        size_t tensors = static_cast<size_t>(sem_depth_ + head_depth_);
        for (const auto & hp : stages_) tensors += 2 * static_cast<size_t>(hp.n_layer);
        st.context.reset(ggml_init({(tensors + 1) * ggml_tensor_overhead(), nullptr, true}));
        if (!st.context) throw std::runtime_error("zonos3: cannot allocate decoder stream metadata");
        auto * g = st.context.get();
        for (size_t s = 0; s < stages_.size(); ++s) {
            const auto & hp = stages_[s];
            const int hd = hp.d_model / hp.n_head;
            for (int l = 0; l < hp.n_layer; ++l) {
                st.keys.push_back(ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, caps_[s], hp.n_head));
                st.values.push_back(ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, caps_[s], hp.n_head));
            }
        }
        for (int b = 0; b < sem_depth_; ++b) st.history.push_back(ggml_new_tensor_2d(g, GGML_TYPE_F32, semantic_dim_, kConvKernel - 1));
        for (int b = 0; b < head_depth_; ++b) {
            st.history.push_back(ggml_new_tensor_2d(g, GGML_TYPE_F32, stages_.back().d_model, kConvKernel - 1));
        }
        st.buffer.reset(ggml_backend_alloc_ctx_tensors(g, execution_.backend()));
        if (!st.buffer) throw std::runtime_error("zonos3: cannot allocate decoder stream state");
        ggml_backend_buffer_clear(st.buffer.get(), 0);  // masked ring rows are read too: no NaN garbage
        st.id = ++next_stream_id_;
    }

    std::unique_ptr<State> acquire() {
        auto state = std::make_unique<State>();
        if (idle_state_) {
            state->id = idle_state_->id;
            state->context = std::move(idle_state_->context);
            state->buffer = std::move(idle_state_->buffer);
            state->keys = std::move(idle_state_->keys);
            state->values = std::move(idle_state_->values);
            state->history = std::move(idle_state_->history);
            idle_state_.reset();
            ggml_backend_buffer_clear(state->buffer.get(), 0);
        } else {
            open(*state);
        }
        return state;
    }

    void close(std::unique_ptr<State> state) {
        const std::array<const char *, 6> keys{"zonos3.codec.prepare_ms", "zonos3.codec.compute_ms",
            "zonos3.codec.readback_ms", "zonos3.codec.head_ms", "zonos3.codec.resample_ms", "zonos3.codec.flush_ms"};
        for (size_t i = 0; i < keys.size(); ++i) debug::timing_log_scalar(keys[i], state->timing_ms[i]);
        // Keep one idle device state and its graph bindings for the next request.
        // Other live streams continue to own independent state.
        if (idle_state_) {
            const auto id = idle_state_->id;
            graphs_.erase(std::remove_if(graphs_.begin(), graphs_.end(),
                [id](const auto & graph) { return graph->stream == id; }), graphs_.end());
        }
        idle_state_ = std::move(state);
    }

    std::vector<float> decode(State & st, const int32_t * frames, int n) {
        if (st.head_done) throw std::runtime_error("zonos3: decoder stream is already flushed");
        if (n < 0 || (n > 0 && frames == nullptr)) throw std::runtime_error("zonos3: invalid decoder frame block");
        for (int64_t i = 0; i < static_cast<int64_t>(n) * n_codebooks_; ++i) {
            if (frames[i] < 0 || frames[i] >= codebook_size_) {
                throw std::runtime_error("zonos3: decoder code " + std::to_string(frames[i]) + " is outside 0.." +
                                         std::to_string(codebook_size_ - 1));
            }
        }
        std::vector<float> audio;
        std::vector<float> internal;
        for (int f = 0; f < n; f += kMaxChunk) {
            const int count = std::min(kMaxChunk, n - f);
            internal.clear();
            decode_chunk(st, frames + static_cast<size_t>(f) * static_cast<size_t>(n_codebooks_), count, internal);
            st.frames_done += count;
            st.rs_buf.insert(st.rs_buf.end(), internal.begin(), internal.end());
            st.rs_in += static_cast<int64_t>(internal.size());
            const auto resample_started = std::chrono::steady_clock::now();
            resample(st, false, audio);
            st.timing_ms[4] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - resample_started).count();
        }
        return audio;
    }

    std::vector<float> flush(State & st) {
        const auto flush_started = std::chrono::steady_clock::now();
        std::vector<float> internal;
        head_flush(st, internal);
        st.rs_buf.insert(st.rs_buf.end(), internal.begin(), internal.end());
        st.rs_in += static_cast<int64_t>(internal.size());
        std::vector<float> audio;
        resample(st, true, audio);
        st.timing_ms[5] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - flush_started).count();
        return audio;
    }

private:
    // ---- metadata (zac.cpp:96-163) ----
    void read_metadata() {
        const assets::GgufMetadata meta(source_->source_path());
        if (meta.require_string("general.architecture") != "mossdual_istft") reject("general.architecture");
        auto u32 = [&](const char * key) {
            const auto value = meta.require_int((std::string("mossdual_istft.") + key).c_str());
            if (value <= 0 || value > std::numeric_limits<int>::max()) reject(std::string(key) + " must be positive");
            return static_cast<int>(value);
        };
        auto f32 = [&](const char * key) {
            const float value = meta.require_f32((std::string("mossdual_istft.") + key).c_str());
            if (!(value > 0.0f)) reject(std::string(key) + " must be positive");
            return value;
        };
        auto array = [&](const char * key, int minimum) {
            const auto values = meta.require_int_array((std::string("mossdual_istft.") + key).c_str());
            if (values.size() != kStages) reject(std::string(key) + " must have 5 entries");
            std::vector<int> out;
            for (const auto value : values) {
                if (value < minimum || value > std::numeric_limits<int>::max()) reject(std::string(key) + " is out of range");
                out.push_back(static_cast<int>(value));
            }
            return out;
        };
        sample_rate_ = u32("sample_rate");
        internal_rate_ = u32("internal_sample_rate");
        const int frame_rate = u32("frame_rate");
        hop_ = u32("hop");
        internal_hop_ = u32("internal_hop");
        n_codebooks_ = u32("n_codebooks");
        codebook_size_ = u32("codebook_size");
        latent_dim_ = u32("latent_dim");
        semantic_dim_ = u32("semantic_dim");
        sem_depth_ = u32("sem_head.depth");
        rs_orig_ = u32("resample.orig");
        rs_new_ = u32("resample.new");
        rs_width_ = u32("resample.width");
        ln_eps_ = f32("ln_eps");
        convnext_eps_ = f32("convnext_eps");
        const auto in_dim = array("stage.in_dim", 1);
        const auto out_dim = array("stage.out_dim", 0);
        const auto d_model = array("stage.d_model", 1);
        const auto n_head = array("stage.n_head", 1);
        const auto n_layer = array("stage.n_layer", 1);
        const auto n_ff = array("stage.n_ff", 1);
        const auto context = array("stage.context", 1);
        const auto patch = array("stage.patch", 1);
        head_depth_ = u32("head.depth");
        const int head_hidden = u32("head.hidden");
        head_hop_ = u32("head.hop");
        fir_kernel_ = u32("head.fir_kernel");
        const int head_step = u32("head.step");
        max_log_mag_ = f32("head.max_log_magnitude");
        headroom_db_ = f32("headroom_db");
        const auto n_fft = meta.require_int_array("mossdual_istft.head.n_fft");
        if (n_fft.size() != 2) reject("head.n_fft must have 2 entries");
        if (meta.require_string("mossdual_istft.head.variant").empty()) reject("head.variant is empty");
        n_fft_ = {static_cast<int>(n_fft[0]), static_cast<int>(n_fft[1])};
        int up = head_hop_;
        for (const int p : patch) up *= p;
        bool consistent = sample_rate_ == 44100 && internal_rate_ == 44160 && frame_rate == 15 && hop_ == 2940 &&
                          internal_hop_ == 2944 && n_codebooks_ == 10 && codebook_size_ == 4096 &&
                          static_cast<int64_t>(internal_hop_) * rs_new_ == static_cast<int64_t>(hop_) * rs_orig_ &&
                          patch == std::vector<int>({2, 2, 2, 2, 1}) && out_dim[kStages - 1] == 0 && up == internal_hop_ &&
                          16 * head_hop_ == internal_hop_ && in_dim[0] == latent_dim_ && n_fft_[0] == 736 &&
                          n_fft_[1] == 2208 && n_fft_[0] % head_hop_ == 0 && n_fft_[1] % head_hop_ == 0 &&
                          fir_kernel_ % 2 == 1 && head_hidden > 0 && head_step > 0;
        for (int s = 0; s < kStages; ++s) {
            consistent = consistent && d_model[s] % n_head[s] == 0 && (d_model[s] / n_head[s]) % 2 == 0;
            if (s + 1 < kStages) consistent = consistent && in_dim[s + 1] * patch[s] == out_dim[s];
            stages_.push_back({in_dim[s], out_dim[s], d_model[s], n_head[s], n_layer[s], n_ff[s], context[s], patch[s]});
        }
        if (!consistent) reject("inconsistent dimensions");
        head_hidden_ = head_hidden;
    }

    // ---- weights (zac.cpp:210-332) ----
    void load_weights() {
        const auto & source = *source_;
        is_gpu_ = execution_.backend_type() != core::BackendType::Cpu;
        // Matrices run F16 on GPUs and F32 on the CPU (zac.cpp:227); everything else loads as F32.
        const auto mat_storage = is_gpu_ ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32;
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos3.decoder.weights", 16 * 1024 * 1024);
        auto plain = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return store_->load_f32_tensor(source, name, shape);
        };
        auto mat = [&](const std::string & name, int64_t out, int64_t in) {
            consumed_.insert(name);
            return store_->load_tensor(source, name, mat_storage, {out, in});
        };
        auto host = [&](const std::string & name, std::vector<int64_t> shape) {
            consumed_.insert(name);
            return source.require_f32(name, shape);
        };
        auto convnext = [&](const std::string & p, int64_t channels) {
            ConvNextWeights w;
            w.dw = store_->make_f32(core::TensorShape::from_dims({channels, 1, kConvKernel}),
                                   host(p + ".dw.weight", {channels, kConvKernel}));
            w.dw_bias = plain(p + ".dw.bias", {channels});
            w.norm_weight = plain(p + ".norm.weight", {channels});
            w.norm_bias = plain(p + ".norm.bias", {channels});
            w.pw1 = mat(p + ".pw1.weight", head_hidden_, channels);
            w.pw1_bias = plain(p + ".pw1.bias", {head_hidden_});
            w.pw2 = mat(p + ".pw2.weight", channels, head_hidden_);
            w.pw2_bias = plain(p + ".pw2.bias", {channels});
            return w;
        };
        sem_table_ = plain("sem.table", {codebook_size_, semantic_dim_});
        sem_bias_ = plain("sem.bias", {semantic_dim_});
        for (int c = 1; c < n_codebooks_; ++c) {
            res_tables_.push_back(plain("res." + std::to_string(c - 1) + ".table", {codebook_size_, latent_dim_}));
            // Per-codebook biases serve ZONOS3's partial-frame decode only; complete frames use their sum res.bias.
            consumed_.insert("res." + std::to_string(c - 1) + ".bias");
        }
        res_bias_ = plain("res.bias", {latent_dim_});
        for (int b = 0; b < sem_depth_; ++b) sem_blocks_.push_back(convnext("sem_head." + std::to_string(b), semantic_dim_));
        sem_out_ = mat("sem_head.out.weight", latent_dim_, semantic_dim_);
        sem_out_bias_ = plain("sem_head.out.bias", {latent_dim_});
        for (int s = 0; s < kStages; ++s) {
            const auto & hp = stages_[static_cast<size_t>(s)];
            const std::string p = "stage." + std::to_string(s);
            StageWeights w;
            w.in = mat(p + ".in.weight", hp.d_model, hp.in_dim);
            if (hp.out_dim > 0) w.out = mat(p + ".out.weight", hp.out_dim, hp.d_model);
            const int64_t d = hp.d_model;
            for (int l = 0; l < hp.n_layer; ++l) {
                const std::string q = p + ".l" + std::to_string(l);
                MossWeights m;
                m.norm1_weight = plain(q + ".norm1.weight", {d});
                m.norm1_bias = plain(q + ".norm1.bias", {d});
                m.qkv = mat(q + ".qkv.weight", 3 * d, d);
                m.wo = mat(q + ".wo.weight", d, d);
                m.attn_scale = plain(q + ".attn_scale", {d});
                m.norm2_weight = plain(q + ".norm2.weight", {d});
                m.norm2_bias = plain(q + ".norm2.bias", {d});
                m.ffn1 = mat(q + ".ffn1.weight", hp.n_ff, d);
                m.ffn2 = mat(q + ".ffn2.weight", d, hp.n_ff);
                m.ffn_scale = plain(q + ".ffn_scale", {d});
                w.layers.push_back(std::move(m));
            }
            stage_weights_.push_back(std::move(w));
        }
        const int64_t d4 = stages_.back().d_model;
        for (int b = 0; b < head_depth_; ++b) head_blocks_.push_back(convnext("head." + std::to_string(b), d4));
        head_norm_weight_ = plain("head.norm.weight", {d4});
        head_norm_bias_ = plain("head.norm.bias", {d4});
        for (int i = 0; i < 2; ++i) {
            const auto idx = static_cast<size_t>(i);
            const std::string p = "head.out." + std::to_string(i);
            head_out_[idx] = mat(p + ".weight", n_fft_[idx] + 2, d4);
            head_out_bias_[idx] = plain(p + ".bias", {n_fft_[idx] + 2});
            window_[idx] = host("head.window." + std::to_string(i), {n_fft_[idx]});
        }
        mixer_ = host("head.mixer.weight", {2, fir_kernel_});                            // [2][K]
        rs_kernel_ = host("resample.kernel", {rs_new_, 2 * rs_width_ + rs_orig_});      // [new][2W + orig]
        const int kernel_size = 2 * rs_width_ + rs_orig_;
        const int grouped_taps = kernel_size / 8 * 8;
        rs_support_.reserve(static_cast<size_t>(rs_new_));
        for (int phase = 0; phase < rs_new_; ++phase) {
            const auto * row = rs_kernel_.data() + static_cast<size_t>(phase) * kernel_size;
            int begin = 0;
            int end = kernel_size;
            while (begin < end && row[begin] == 0.0f) ++begin;
            while (end > begin && row[end - 1] == 0.0f) --end;
            // Keep each nonzero tap in its original accumulator lane. The
            // scalar tail below is still evaluated in its original order.
            rs_support_.emplace_back(std::min(begin / 8 * 8, grouped_taps),
                                     std::min((end + 7) / 8 * 8, grouped_taps));
        }
        rs_window_.resize(static_cast<size_t>(kernel_size));
        rs_output_.resize(static_cast<size_t>(rs_new_));
        for (const auto & tensor : source.tensors()) {
            if (!consumed_.count(tensor.name)) reject("unmapped tensor '" + tensor.name + "'");
        }
        store_->upload();
        source_->release_storage();
    }

    // K/V ring rows per stage (zac.cpp:924-941): distinct rows for any chunk's window [p0 - context + 1, p0 + T - 1]
    // with T <= 32 frames of columns.
    void compute_ring_capacities() {
        int cpf = 1;
        for (const auto & hp : stages_) {
            caps_.push_back(hp.context - 1 + kMaxChunk * cpf);
            cpf *= hp.patch;
        }
    }

    // ---- graph (zac.cpp:709-813, 1111-1195) ----
    ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight,
                         const TensorValue * bias = nullptr) const {
        modules::LinearConfig config;
        config.in_features = weight.shape.dims[1];
        config.out_features = weight.shape.dims[0];
        config.use_bias = bias != nullptr;
        config.precision = GGML_PREC_F32;
        modules::LinearWeights weights{weight, std::nullopt};
        if (bias) weights.bias = *bias;
        return modules::LinearModule(config).build(ctx, matrix(x), weights).tensor;
    }

    static ggml_tensor * layer_norm(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & weight,
                                    const TensorValue & bias, float eps) {
        modules::NormConfig config;
        config.hidden_size = weight.shape.dims[0];
        config.eps = eps;
        modules::NormWeights weights;
        weights.weight = weight;
        weights.bias = bias;
        return modules::LayerNormModule(config).build(ctx, matrix(x), weights).tensor;
    }

    static ggml_tensor * gelu(core::ModuleBuildContext & ctx, ggml_tensor * x) {
        return modules::GeluModule(modules::GeluConfig{modules::GeluApproximation::ExactErf}).build(ctx, matrix(x)).tensor;
    }

    // Causal ConvNeXt block: depthwise k7 over [history | x], LayerNorm, pw1, exact GELU, pw2, residual. The last
    // K-1 input columns become the next chunk's history.
    ggml_tensor * convnext_block(core::ModuleBuildContext & ctx, ggml_cgraph * graph, ggml_tensor * x,
                                 const ConvNextWeights & w, ggml_tensor * history) const {
        auto * g = ctx.ggml;
        const int64_t C = x->ne[0];
        const int64_t T = x->ne[1];
        auto * xf = ggml_concat(g, history, x, 1);  // [C, K - 1 + T]
        ggml_build_forward_expand(graph, ggml_cpy(g, ggml_view_2d(g, xf, C, kConvKernel - 1, xf->nb[1],
                                                                  static_cast<size_t>(T) * xf->nb[1]), history));
        modules::DepthwiseConv1dConfig config;
        config.channels = C;
        config.kernel_size = kConvKernel;
        config.use_bias = false;
        auto * h = modules::DepthwiseConv1dModule(config).build_time_major(ctx,
            core::wrap_tensor(ggml_reshape_3d(g, xf, C, T + kConvKernel - 1, 1),
                              core::TensorShape::from_dims({1, T + kConvKernel - 1, C})),
            {w.dw, std::nullopt}).tensor;
        h = ggml_reshape_2d(g, h, C, T);
        h = ggml_add(g, h, w.dw_bias.tensor);
        h = layer_norm(ctx, h, w.norm_weight, w.norm_bias, convnext_eps_);
        h = gelu(ctx, linear(ctx, h, w.pw1, &w.pw1_bias));
        h = linear(ctx, h, w.pw2, &w.pw2_bias);
        return ggml_add(g, x, h);
    }

    // Pre-LN windowed-causal attention over the stream's K/V ring plus a GELU FFN, each with a channel scale.
    ggml_tensor * moss_layer(core::ModuleBuildContext & ctx, ggml_tensor * x, const MossWeights & w, const StageHp & hp,
                             const StageInputs & in, ggml_tensor * key_ring, ggml_tensor * value_ring) const {
        auto * g = ctx.ggml;
        const int d = hp.d_model;
        const int nh = hp.n_head;
        const int hd = d / nh;
        const int64_t T = x->ne[1];
        auto * h = layer_norm(ctx, x, w.norm1_weight, w.norm1_bias, ln_eps_);
        auto * qkv = linear(ctx, h, w.qkv);  // [3d, T]: q | k | v, head-major
        auto section = [&](int s) {
            return ggml_cont(g, ggml_view_3d(g, qkv, hd, nh, T, static_cast<size_t>(hd) * sizeof(float), qkv->nb[1],
                                             static_cast<size_t>(s) * static_cast<size_t>(d) * sizeof(float)));
        };
        modules::RoPEConfig rope_config;
        rope_config.dimensions = hd;
        rope_config.mode = GGML_ROPE_TYPE_NORMAL;
        rope_config.theta = 10000.0f;
        const modules::RoPEModule rope(rope_config);
        const auto positions = core::wrap_tensor(in.pos, core::TensorShape::from_dims({T}), GGML_TYPE_I32);
        auto rotate = [&](ggml_tensor * z) {
            return rope.build(ctx, core::wrap_tensor(z, core::TensorShape::from_dims({1, T, nh, hd})), positions).tensor;
        };
        auto * q = rotate(section(0));                                                        // [hd, nh, T]
        auto * k = ggml_cont(g, ggml_permute(g, rotate(section(1)), 0, 2, 1, 3));             // [hd, T, nh]
        auto * v = ggml_cont(g, ggml_permute(g, section(2), 0, 2, 1, 3));
        // Attention reads the set_rows results, so ggml orders it after this chunk's writes.
        auto * keys = ggml_set_rows(g, key_ring, k, in.slots);                                // [hd, cap, nh]
        auto * values = ggml_set_rows(g, value_ring, v, in.slots);
        auto * qp = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));                             // [hd, T, nh]
        const auto query_shape = core::TensorShape::from_dims({1, nh, T, hd});
        const auto cache_shape = core::TensorShape::from_dims({1, nh, key_ring->ne[1], hd});
        const auto context = modules::ScaledDotProductAttentionModule({hd}).build(ctx,
            core::wrap_tensor(qp, query_shape), core::wrap_tensor(keys, cache_shape),
            core::wrap_tensor(values, cache_shape),
            core::wrap_tensor(in.mask, core::TensorShape::from_dims({T, key_ring->ne[1]}), GGML_TYPE_F16));
        auto * o = ggml_reshape_2d(g, ggml_cont(g, context.tensor), d, T);
        x = ggml_add(g, x, ggml_mul(g, linear(ctx, o, w.wo), w.attn_scale.tensor));
        h = layer_norm(ctx, x, w.norm2_weight, w.norm2_bias, ln_eps_);
        h = linear(ctx, gelu(ctx, linear(ctx, h, w.ffn1)), w.ffn2);
        return ggml_add(g, x, ggml_mul(g, h, w.ffn_scale.tensor));
    }

    // Patch up h: out[ch, t*h + j] = in[ch*h + j, t].
    static ggml_tensor * patch_up(ggml_context * g, ggml_tensor * x, int h) {
        const int64_t C = x->ne[0];
        const int64_t T = x->ne[1];
        auto * y = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, x, h, C / h, T), 1, 0, 2, 3));
        return ggml_reshape_2d(g, y, C / h, h * T);
    }

    std::unique_ptr<DecodeGraph> build_graph(const State & st, int frames) {
        auto out = std::make_unique<DecodeGraph>();
        out->frames = frames;
        out->stream = st.id;
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        out->prepared = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos3.decoder", metadata, kGraphNodes, 64 * ggml_tensor_overhead()});
        auto * persistent = out->prepared->persistent_build_context().ggml;
        for (int c = 0; c < n_codebooks_; ++c) {
            out->ids[static_cast<size_t>(c)] = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, frames);
            ggml_set_input(out->ids[static_cast<size_t>(c)]);
        }
        int cpf = 1;
        for (int s = 0; s < kStages; ++s) {
            StageInputs si;
            si.cpf = cpf;
            si.columns = frames * cpf;
            si.pos = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, si.columns);
            si.slots = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, si.columns);
            si.mask = ggml_new_tensor_2d(persistent, GGML_TYPE_F16, caps_[static_cast<size_t>(s)], si.columns);
            for (auto * t : {si.pos, si.slots, si.mask}) ggml_set_input(t);
            out->stages.push_back(si);
            cpf *= stages_[static_cast<size_t>(s)].patch;
        }
        out->prepared->allocate_persistent();
        out->prepared->clear_persistent();

        auto & ctx = out->prepared->build_context();
        auto * g = ctx.ggml;
        auto * graph = out->prepared->graph();
        size_t history = 0;
        auto * x = ggml_add(g, ggml_get_rows(g, sem_table_.tensor, out->ids[0]), sem_bias_.tensor);  // [768, n]
        for (const auto & block : sem_blocks_) x = convnext_block(ctx, graph, x, block, st.history[history++]);
        x = linear(ctx, x, sem_out_, &sem_out_bias_);
        for (int c = 1; c < n_codebooks_; ++c) {
            x = ggml_add(g, x, ggml_get_rows(g, res_tables_[static_cast<size_t>(c - 1)].tensor, out->ids[static_cast<size_t>(c)]));
        }
        x = ggml_add(g, x, res_bias_.tensor);  // [latent, n]
        size_t layer = 0;
        for (int s = 0; s < kStages; ++s) {
            const auto & hp = stages_[static_cast<size_t>(s)];
            const auto & w = stage_weights_[static_cast<size_t>(s)];
            x = linear(ctx, x, w.in);
            for (const auto & moss : w.layers) {
                x = moss_layer(ctx, x, moss, hp, out->stages[static_cast<size_t>(s)], st.keys[layer], st.values[layer]);
                ++layer;
            }
            if (hp.out_dim > 0) x = linear(ctx, x, w.out);
            if (hp.patch > 1) x = patch_up(g, x, hp.patch);
        }
        for (const auto & block : head_blocks_) x = convnext_block(ctx, graph, x, block, st.history[history++]);
        x = layer_norm(ctx, x, head_norm_weight_, head_norm_bias_, convnext_eps_);
        for (size_t i = 0; i < 2; ++i) {
            out->spec[i] = linear(ctx, x, head_out_[i], &head_out_bias_[i]);
            ggml_set_output(out->spec[i]);
            ggml_build_forward_expand(graph, out->spec[i]);
        }
        // Every MUL_MAT accumulates in F32 (zac.cpp:1174-1177), whatever chunk width selects which kernel.
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
        }
        out->prepared->allocate();
        return out;
    }

    DecodeGraph & graph_for(const State & st, int frames) {
        ++use_clock_;
        for (auto & graph : graphs_) {
            if (graph->frames == frames && graph->stream == st.id) {
                graph->last_use = use_clock_;
                return *graph;
            }
        }
        if (graphs_.size() >= kCachedGraphs) {
            graphs_.erase(std::min_element(graphs_.begin(), graphs_.end(),
                                           [](const auto & a, const auto & b) { return a->last_use < b->last_use; }));
        }
        graphs_.push_back(build_graph(st, frames));
        graphs_.back()->last_use = use_clock_;
        return *graphs_.back();
    }

    // Frames [frames_done, frames_done + n) through the graph, then the host iSTFT/mixer (zac.cpp:1216-1366).
    void decode_chunk(State & st, const int32_t * frames, int n, std::vector<float> & internal) {
        const auto prepare_started = std::chrono::steady_clock::now();
        auto & graph = graph_for(st, n);
        std::vector<int32_t> column(static_cast<size_t>(n));
        for (int c = 0; c < n_codebooks_; ++c) {
            for (int i = 0; i < n; ++i) {
                column[static_cast<size_t>(i)] = frames[static_cast<size_t>(i) * static_cast<size_t>(n_codebooks_) + static_cast<size_t>(c)];
            }
            ggml_backend_tensor_set(graph.ids[static_cast<size_t>(c)], column.data(), 0, column.size() * sizeof(int32_t));
        }
        const auto open = ggml_fp32_to_fp16(0.0f);
        const auto blocked = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        std::vector<int32_t> pos, slots;
        std::vector<ggml_fp16_t> mask;
        for (size_t s = 0; s < graph.stages.size(); ++s) {
            const auto & si = graph.stages[s];
            const int64_t cap = caps_[s];
            const int64_t context = stages_[s].context;
            const int64_t p0 = st.frames_done * si.cpf;
            const int64_t p_last = p0 + si.columns - 1;
            pos.resize(static_cast<size_t>(si.columns));
            slots.resize(static_cast<size_t>(si.columns));
            for (int i = 0; i < si.columns; ++i) {
                pos[static_cast<size_t>(i)] = static_cast<int32_t>(p0 + i);
                slots[static_cast<size_t>(i)] = static_cast<int32_t>((p0 + i) % cap);
            }
            ggml_backend_tensor_set(si.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(si.slots, slots.data(), 0, slots.size() * sizeof(int32_t));
            // Ring row j holds the latest position written there; keep keys with 0 <= q - k < context.
            mask.resize(static_cast<size_t>(cap) * static_cast<size_t>(si.columns));
            for (int64_t j = 0; j < cap; ++j) {
                const int64_t kp = p_last - (((p_last - j) % cap) + cap) % cap;
                for (int i = 0; i < si.columns; ++i) {
                    const int64_t qp = p0 + i;
                    mask[static_cast<size_t>(i * cap + j)] = (kp >= 0 && kp <= qp && qp - kp < context) ? open : blocked;
                }
            }
            ggml_backend_tensor_set(si.mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        const auto compute_started = std::chrono::steady_clock::now();
        st.timing_ms[0] += std::chrono::duration<double, std::milli>(compute_started - prepare_started).count();
        graph.prepared->compute();
        const auto readback_started = std::chrono::steady_clock::now();
        st.timing_ms[1] += std::chrono::duration<double, std::milli>(readback_started - compute_started).count();
        const int64_t columns = static_cast<int64_t>(n) * (internal_hop_ / head_hop_);
        for (size_t i = 0; i < 2; ++i) {
            spec_[i].resize(static_cast<size_t>(n_fft_[i] + 2) * static_cast<size_t>(columns));
            ggml_backend_tensor_get(graph.spec[i], spec_[i].data(), 0, spec_[i].size() * sizeof(float));
        }
        const auto head_started = std::chrono::steady_clock::now();
        st.timing_ms[2] += std::chrono::duration<double, std::milli>(head_started - readback_started).count();
        head_push(st, columns, internal);
        st.timing_ms[3] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - head_started).count();
    }

    // ---- host iSTFT + FIR mixer (zac.cpp:960-1078) ----
    // Streaming torch.istft(center=True, length=frames*hop) per branch, then the 2->1 mixer Conv1d (K taps, zero
    // padding (K-1)/2). Frame t of branch i covers padded positions [t*hop, t*hop + n); sample m is position m + n/2.
    void mix_emit(State & st, std::vector<float> & out) const {
        const int K = fir_kernel_;
        const int64_t avail = std::min(st.mix_avail[0], st.mix_avail[1]);
        const int64_t end = avail - K + 1;
        if (end <= st.mix_next) return;
        const size_t count = static_cast<size_t>(end - st.mix_next);
        const size_t o0 = out.size();
        out.resize(o0 + count, 0.0f);
        float * y = out.data() + o0;
        for (size_t c = 0; c < 2; ++c) {
            const float * w = mixer_.data() + c * static_cast<size_t>(K);
            const float * s = st.mix_in[c].data() + (st.mix_next - st.mix_base);
            for (int k = 0; k < K; ++k) {
                const float wk = w[k];
                for (size_t i = 0; i < count; ++i) y[i] += wk * s[i + static_cast<size_t>(k)];
            }
        }
        st.mix_next = end;
        for (auto & in : st.mix_in) in.erase(in.begin(), in.begin() + (st.mix_next - st.mix_base));
        st.mix_base = st.mix_next;
    }

    // Moves finished overlap-add positions [ola_base, pf) of branch i into the mixer input, dropping the centre pad.
    void ola_finalize(State & st, size_t i, int64_t pf) const {
        const int64_t pad = n_fft_[i] / 2;
        const int64_t count = pf - st.ola_base[i];
        if (count <= 0) return;
        for (int64_t j = 0; j < count; ++j) {
            if (st.ola_base[i] + j < pad) continue;
            st.mix_in[i].push_back(st.ola[i][static_cast<size_t>(j)] / st.env[i][static_cast<size_t>(j)]);
            ++st.mix_avail[i];
        }
        st.ola[i].erase(st.ola[i].begin(), st.ola[i].begin() + count);
        st.env[i].erase(st.env[i].begin(), st.env[i].begin() + count);
        st.ola_base[i] = pf;
    }

    // Adds `columns` head frames from spec_ (per column: logmag[0:bins], phase[bins:2*bins]) and appends the
    // finished 44160 Hz samples.
    void head_push(State & st, int64_t columns, std::vector<float> & out) {
        if (st.head_frames == 0) {
            const int h = (fir_kernel_ - 1) / 2;
            for (size_t i = 0; i < 2; ++i) {
                st.mix_in[i].assign(static_cast<size_t>(h), 0.0f);
                st.mix_avail[i] = h;
            }
            st.mix_base = 0;
            st.mix_next = 0;
        }
        const int64_t hop = head_hop_;
        for (size_t i = 0; i < 2; ++i) {
            const int n = n_fft_[i];
            const int bins = n / 2 + 1;
            spectrum_.resize(static_cast<size_t>(columns) * static_cast<size_t>(bins));
            frames_.resize(static_cast<size_t>(columns) * static_cast<size_t>(n));
#ifdef _OPENMP
#pragma omp parallel for num_threads(execution_.config().threads)
#endif
            for (int64_t c = 0; c < columns; ++c) {
                const float * logmag = spec_[i].data() + static_cast<size_t>(c) * static_cast<size_t>(n + 2);
                const float * phase = logmag + bins;
                auto * bin = spectrum_.data() + static_cast<size_t>(c) * static_cast<size_t>(bins);
                for (int k = 0; k < bins; ++k) {
                    const float a = std::exp(std::min(logmag[k], max_log_mag_));
                    // The inverse real FFT ignores the imaginary parts of the DC and Nyquist bins.
                    bin[k] = (k == 0 || k == n / 2) ? std::complex<float>(a * std::cos(phase[k]), 0.0f)
                                                    : std::complex<float>(a * std::cos(phase[k]), a * std::sin(phase[k]));
                }
            }
            irfft_[i]->inverse({static_cast<size_t>(columns), static_cast<size_t>(n)},
                               {static_cast<std::ptrdiff_t>(bins * sizeof(std::complex<float>)), sizeof(std::complex<float>)},
                               {static_cast<std::ptrdiff_t>(n * sizeof(float)), sizeof(float)}, 1, spectrum_.data(),
                               frames_.data(), 1.0f / static_cast<float>(n), static_cast<size_t>(execution_.config().threads));
            const float * win = window_[i].data();
            for (int64_t c = 0; c < columns; ++c) {
                const int64_t t = st.head_frames + c;
                const size_t need = static_cast<size_t>(t * hop + n - st.ola_base[i]);
                if (st.ola[i].size() < need) {
                    st.ola[i].resize(need, 0.0f);
                    st.env[i].resize(need, 0.0f);
                }
                float * ola = st.ola[i].data() + (t * hop - st.ola_base[i]);
                float * env = st.env[i].data() + (t * hop - st.ola_base[i]);
                const float * frame = frames_.data() + static_cast<size_t>(c) * static_cast<size_t>(n);
                for (int j = 0; j < n; ++j) {
                    ola[j] += win[j] * frame[j];
                    env[j] += win[j] * win[j];
                }
            }
        }
        st.head_frames += columns;
        for (size_t i = 0; i < 2; ++i) ola_finalize(st, i, st.head_frames * hop);
        mix_emit(st, out);
    }

    // End of stream: every sample m < L = head_frames*hop is final; the mixer's right padding is h zeros.
    void head_flush(State & st, std::vector<float> & out) const {
        if (st.head_done || st.head_frames == 0) return;
        st.head_done = true;
        const int h = (fir_kernel_ - 1) / 2;
        for (size_t i = 0; i < 2; ++i) {
            ola_finalize(st, i, st.ola_base[i] + static_cast<int64_t>(st.ola[i].size()));
            const int64_t length = st.head_frames * head_hop_;
            const int64_t keep = h + length - st.mix_base;
            st.mix_in[i].resize(static_cast<size_t>(keep), 0.0f);
            st.mix_in[i].resize(static_cast<size_t>(keep + h), 0.0f);
            st.mix_avail[i] = st.mix_base + keep + h;
        }
        mix_emit(st, out);
    }

    // ---- 44160 -> 44100 resampler (zac.cpp:1368-1402) ----
    // out[b*new + i] = sum_k K[i][k] * x[b*orig + k - width], x = 0 outside [0, rs_in). Without `flush` block b waits
    // for its last input; `flush` zero-extends and truncates the total to ceil(new * rs_in / orig).
    void resample(State & st, bool flush, std::vector<float> & audio) {
        const int O = rs_orig_;
        const int N = rs_new_;
        const int W = rs_width_;
        const int KL = 2 * W + O;
        const int64_t total = flush ? (static_cast<int64_t>(N) * st.rs_in + O - 1) / O : -1;
        auto & window = rs_window_;
        auto & y = rs_output_;
        for (;; ++st.rs_next_block) {
            const int64_t b = st.rs_next_block;
            const int64_t x0 = b * O - W;
            if (flush ? b * N >= total : x0 + KL > st.rs_in) break;
            for (int k = 0; k < KL; ++k) {
                const int64_t j = x0 + k;
                window[static_cast<size_t>(k)] = (j < 0 || j >= st.rs_in) ? 0.0f : st.rs_buf[static_cast<size_t>(j - st.rs_base)];
            }
            for (int i = 0; i < N; ++i) {
                const float * row = rs_kernel_.data() + static_cast<size_t>(i) * static_cast<size_t>(KL);
                std::array<float, 8> acc{};
                const auto support = rs_support_[static_cast<size_t>(i)];
                for (int k = support.first; k < support.second; k += 8) {
                    for (int l = 0; l < 8; ++l) acc[static_cast<size_t>(l)] += row[k + l] * window[static_cast<size_t>(k + l)];
                }
                float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                for (int k = KL / 8 * 8; k < KL; ++k) sum += row[k] * window[static_cast<size_t>(k)];
                y[static_cast<size_t>(i)] = sum;
            }
            const int64_t emit = flush ? std::min<int64_t>(N, total - b * N) : N;
            audio.insert(audio.end(), y.begin(), y.begin() + emit);
        }
        // Drop samples no later block reads.
        const int64_t keep_from = std::min(st.rs_in, std::max<int64_t>(0, st.rs_next_block * O - W));
        if (keep_from > st.rs_base) {
            st.rs_buf.erase(st.rs_buf.begin(), st.rs_buf.begin() + (keep_from - st.rs_base));
            st.rs_base = keep_from;
        }
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    bool is_gpu_ = false;

    int sample_rate_ = 0, internal_rate_ = 0, hop_ = 0, internal_hop_ = 0;
    int n_codebooks_ = 0, codebook_size_ = 0, latent_dim_ = 0, semantic_dim_ = 0, sem_depth_ = 0;
    int head_depth_ = 0, head_hidden_ = 0, head_hop_ = 0, fir_kernel_ = 0;
    int rs_orig_ = 0, rs_new_ = 0, rs_width_ = 0;
    float ln_eps_ = 1e-5f, convnext_eps_ = 1e-6f, max_log_mag_ = 0.0f, headroom_db_ = 0.0f;
    std::array<int, 2> n_fft_{};
    std::vector<StageHp> stages_;

    std::unique_ptr<core::BackendWeightStore> store_;
    std::set<std::string> consumed_;
    TensorValue sem_table_, sem_bias_, res_bias_, sem_out_, sem_out_bias_, head_norm_weight_, head_norm_bias_;
    std::vector<TensorValue> res_tables_;
    std::vector<ConvNextWeights> sem_blocks_, head_blocks_;
    std::vector<StageWeights> stage_weights_;
    std::array<TensorValue, 2> head_out_, head_out_bias_;
    std::array<std::vector<float>, 2> window_;
    std::vector<float> mixer_;      // [2][fir_kernel]
    std::vector<float> rs_kernel_;  // [new][2 * width + orig]
    std::vector<std::pair<int, int>> rs_support_;
    std::vector<float> rs_window_, rs_output_;
    std::array<std::shared_ptr<audio::RealFFTPlan>, 2> irfft_;

    std::vector<int> caps_;  // K/V ring rows per stage
    std::unique_ptr<State> idle_state_;
    uint64_t next_stream_id_ = 0;

    std::vector<std::unique_ptr<DecodeGraph>> graphs_;
    uint64_t use_clock_ = 0;
    std::array<std::vector<float>, 2> spec_;  // host copies of the spectral projections
    std::vector<std::complex<float>> spectrum_;
    std::vector<float> frames_;
};

Zonos3CodecDecoder::Zonos3CodecDecoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), execution)) {}

Zonos3CodecDecoder::~Zonos3CodecDecoder() = default;

int Zonos3CodecDecoder::sample_rate() const { return impl_->sample_rate(); }

int Zonos3CodecDecoder::samples_per_frame() const { return impl_->samples_per_frame(); }

float Zonos3CodecDecoder::headroom_db() const { return impl_->headroom_db(); }

Zonos3DecoderStream::Zonos3DecoderStream(Zonos3CodecDecoder & decoder)
    : decoder_(decoder), state_(decoder_.impl_->acquire()) {}

Zonos3DecoderStream::~Zonos3DecoderStream() { decoder_.impl_->close(std::move(state_)); }

std::vector<float> Zonos3DecoderStream::decode(const int32_t * frames, int n) {
    return decoder_.impl_->decode(*state_, frames, n);
}

std::vector<float> Zonos3DecoderStream::flush() { return decoder_.impl_->flush(*state_); }

}  // namespace engine::models::zonos3
