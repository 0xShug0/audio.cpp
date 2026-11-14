// MECT-B2 speaker encoder, ported from ZONOS3 servers/ggml/src/spk-encoder.cpp (MIT).
#include "engine/community_models/zonos3/speaker_encoder.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos3 {
namespace {

constexpr int kHidden = 64;
constexpr int kFeat = 2560;
constexpr int kMels = 80;
constexpr int kHeads = 4;
constexpr int kExperts = 4;
constexpr int kPosKernel = 113;
constexpr int kEmbedding = 192;
constexpr int kSampleRate = 44100;
constexpr int kFrameLength16k = 400;
constexpr int64_t kMaxSamples16k = 60 * 16000;
constexpr size_t kGraphNodes = 32768;
constexpr float kLnEps = 1e-6f;
constexpr float kPoolVarFloor = 1e-7f;

// MECT_B2 (backbones/mect.py:759-780): per block (cnn depth, out channels, stride F, stride T, expansion).
struct BlockConfig {
    int cnn, out_ch, s_f, s_t, expansion;
};

const std::vector<std::vector<BlockConfig>> & mect_b2() {
    static const std::vector<std::vector<BlockConfig>> config = [] {
        std::vector<std::vector<BlockConfig>> out(4);
        out[0].assign(2, {3, 32, 1, 1, 4});
        out[1] = {{3, 64, 2, 1, 2}};
        out[1].insert(out[1].end(), 3, {3, 64, 1, 2, 2});
        out[2] = {{3, 128, 2, 1, 2}};
        out[2].insert(out[2].end(), 4, {3, 128, 1, 2, 2});
        out[3] = {{3, 256, 2, 1, 1}};
        out[3].insert(out[3].end(), 2, {2, 256, 1, 2, 1});
        return out;
    }();
    return config;
}

std::string block_name(int stage, int k) { return "fe.b" + std::to_string(stage + 1) + "." + std::to_string(k) + "."; }

core::TensorValue matrix(ggml_tensor * x) { return core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[1], x->ne[0]})); }

}  // namespace

class Zonos3SpeakerEncoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : source_(std::move(source)), execution_(execution) {
        if (!source_) throw std::runtime_error("zonos3: speaker encoder GGUF is missing");
        validate_metadata();
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos3.speaker_encoder.weights", 4 * 1024 * 1024);
        for (const auto & tensor : source_->tensors()) {
            if (tensor.name.rfind("frontend.", 0) == 0) continue;  // host frontend tables; the framework rebuilds them
            weights_[tensor.name] = store_->load_tensor(*source_, tensor.name, assets::TensorStorageType::F32, tensor.shape);
        }
        for (const char * name : {"fe.head.conv.weight", "fe.dense_w", "pool.l1.weight", "bn.scale", "linear.weight"}) {
            if (!weights_.count(name)) throw std::runtime_error(std::string("zonos3: speaker encoder GGUF lacks ") + name);
        }
        store_->upload();
        source_->release_storage();
    }

    std::vector<float> embed(const std::vector<float> & mono_44k) const {
        std::vector<float> audio(mono_44k.begin(),
                                 mono_44k.begin() + static_cast<std::ptrdiff_t>(std::min<size_t>(mono_44k.size(), 60 * kSampleRate)));
        auto wav16 = audio::resample_mono_torchaudio_sinc_hann(audio, kSampleRate, 16000);
        if (static_cast<int64_t>(wav16.size()) > kMaxSamples16k) wav16.resize(static_cast<size_t>(kMaxSamples16k));
        if (wav16.size() < static_cast<size_t>(kFrameLength16k)) {
            throw runtime::InvalidRequestError("zonos3: reference audio is shorter than 25 ms");
        }
        audio::KaldiFbankOptions options;
        options.sample_rate = 16000;
        options.num_mels = kMels;
        options.frame_length_ms = 25.0f;
        options.frame_shift_ms = 10.0f;
        options.window_type = audio::KaldiFbankWindowType::Hamming;
        options.lfr_m = 1;
        options.lfr_n = 1;
        options.preemphasis = 0.97f;
        options.low_frequency = 20.0f;
        options.high_frequency = 0.0f;
        options.remove_dc_offset = true;
        options.upscale_samples = true;
        const auto fbank = audio::extract_kaldi_fbank(wav16, options);
        const int T = fbank.frames;
        if (T <= 0 || fbank.feature_dim != kMels) throw std::runtime_error("zonos3: speaker fbank has an unexpected shape");
        // Per-utterance CMN, transposed to the network's time-fastest layout [80][T].
        std::vector<float> x0(static_cast<size_t>(T) * kMels);
        for (int b = 0; b < kMels; ++b) {
            double mean = 0.0;
            for (int t = 0; t < T; ++t) mean += fbank.values[static_cast<size_t>(t) * kMels + static_cast<size_t>(b)];
            const float fmean = static_cast<float>(mean / T);
            for (int t = 0; t < T; ++t) {
                x0[static_cast<size_t>(b) * static_cast<size_t>(T) + static_cast<size_t>(t)] =
                    fbank.values[static_cast<size_t>(t) * kMels + static_cast<size_t>(b)] - fmean;
            }
        }
        if (!graph_ || frames_ != T) build_graph(T);
        ggml_backend_tensor_set(input_, x0.data(), 0, x0.size() * sizeof(float));
        graph_->compute();
        std::vector<float> embedding(kEmbedding);
        ggml_backend_tensor_get(output_, embedding.data(), 0, embedding.size() * sizeof(float));
        for (float v : embedding) {
            if (!std::isfinite(v)) throw std::runtime_error("zonos3: speaker encoder produced a non-finite embedding");
        }
        return embedding;
    }

private:
    void validate_metadata() const {
        const assets::GgufMetadata meta(source_->source_path());
        if (meta.require_string("general.architecture") != "mect_spk" || meta.require_string("mect_spk.backbone") != "MECT_B2") {
            throw std::runtime_error("zonos3: speaker encoder GGUF must be a mect_spk MECT_B2 model");
        }
        const std::pair<const char *, int64_t> ints[] = {
            {"mect_spk.sample_rate", 16000}, {"mect_spk.max_seconds", 60}, {"mect_spk.embedding_length", kEmbedding},
            {"mect_spk.n_mels", kMels}, {"mect_spk.frame_length", 400}, {"mect_spk.frame_shift", 160},
            {"mect_spk.n_fft", 512}, {"mect_spk.n_heads", kHeads}, {"mect_spk.n_experts", kExperts},
            {"mect_spk.pos_kernel", kPosKernel}};
        for (const auto & [key, expected] : ints) {
            if (meta.require_int(key) != expected) throw std::runtime_error(std::string("zonos3: speaker encoder GGUF: unsupported ") + key);
        }
        const std::pair<const char *, float> floats[] = {
            {"mect_spk.preemphasis", 0.97f}, {"mect_spk.waveform_scale", 32768.0f},
            {"mect_spk.ln_eps", kLnEps}, {"mect_spk.pool_var_floor", kPoolVarFloor}};
        for (const auto & [key, expected] : floats) {
            if (meta.require_f32(key) != expected) throw std::runtime_error(std::string("zonos3: speaker encoder GGUF: unsupported ") + key);
        }
    }

    ggml_tensor * W(const std::string & name) const {
        const auto it = weights_.find(name);
        if (it == weights_.end()) throw std::runtime_error("zonos3: speaker encoder GGUF lacks " + name);
        return it->second.tensor;
    }
    bool has(const std::string & name) const { return weights_.count(name) != 0; }
    int count_slots(const std::string & prefix) const {
        int a = 0;
        while (has(prefix + ".w" + std::to_string(a))) ++a;
        return a;
    }

    struct Builder {
        const Impl & m;
        core::ModuleBuildContext & ctx;
        ggml_context * c;

        ggml_tensor * chan(ggml_tensor * v) const { return ggml_reshape_4d(c, v, 1, 1, v->ne[0], 1); }
        ggml_tensor * bn4(ggml_tensor * x, const std::string & p) const {
            return ggml_add(c, ggml_mul(c, x, chan(m.W(p + ".scale"))), chan(m.W(p + ".shift")));
        }
        ggml_tensor * conv(ggml_tensor * x, const std::string & p, int s_t, int s_f, bool bias, int pad = 0) const {
            ggml_tensor * y = ggml_conv_2d_direct(c, m.W(p + ".weight"), x, s_t, s_f, pad, pad, 1, 1);
            return bias ? ggml_add(c, y, chan(m.W(p + ".bias"))) : y;
        }
        // torch Conv2d(3x3, pad 1, groups=G, bias=False), weight split into `a` input slots.
        ggml_tensor * grouped3x3(ggml_tensor * x, const std::string & p, int s_t, int s_f) const {
            const int a = m.count_slots(p);
            const int64_t T = x->ne[0], F = x->ne[1], IC = x->ne[2];
            const int64_t oc = m.W(p + ".w0")->ne[3], G = IC / a, b = oc / G;
            ggml_tensor * x4 = ggml_reshape_4d(c, x, T, F, a, G);
            ggml_tensor * y = nullptr;
            for (int j = 0; j < a; ++j) {
                ggml_tensor * xj = x;
                if (a > 1 || b > 1) {
                    ggml_tensor * slot = ggml_view_4d(c, x4, T, F, 1, G, x4->nb[1], x4->nb[2], x4->nb[3],
                                                      static_cast<size_t>(j) * x4->nb[2]);
                    ggml_tensor * shape_like = ggml_new_tensor_4d(c, GGML_TYPE_F32, T, F, b, G);
                    xj = ggml_reshape_3d(c, ggml_repeat(c, slot, shape_like), T, F, oc);
                }
                ggml_tensor * term = ggml_conv_2d_dw_direct(c, m.W(p + ".w" + std::to_string(j)), xj, s_t, s_f, 1, 1, 1, 1);
                y = y ? ggml_add(c, y, term) : term;
            }
            return y;
        }
        ggml_tensor * res_block(ggml_tensor * x, const std::string & q, int s_f, int s_t) const {
            ggml_tensor * t = grouped3x3(x, q + "conv1", s_t, s_f);
            t = ggml_relu(c, bn4(conv(t, q + "pw1", 1, 1, true), q + "bn1"));
            t = grouped3x3(t, q + "conv2", 1, 1);
            t = ggml_relu(c, bn4(conv(t, q + "pw2", 1, 1, true), q + "bn2"));
            ggml_tensor * sc = x;
            if (m.has(q + "sc.weight")) sc = bn4(conv(x, q + "sc", s_t, s_f, false), q + "sc.bn");
            return ggml_add(c, t, sc);
        }
        ggml_tensor * linear(ggml_tensor * x, const std::string & p, bool bias = true) const {
            const auto & w = m.weights_.at(p + ".weight");
            modules::LinearWeights weights{w, std::nullopt};
            if (bias) weights.bias = m.weights_.at(p + ".bias");
            return modules::LinearModule({w.shape.at(1), w.shape.at(0), bias, GGML_PREC_F32}).build(ctx, matrix(x), weights).tensor;
        }
        ggml_tensor * layer_norm(ggml_tensor * x, const std::string & p) const {
            return modules::LayerNormModule({kHidden, kLnEps, true, true})
                .build(ctx, matrix(x), {m.weights_.at(p + ".weight"), m.weights_.at(p + ".bias")})
                .tensor;
        }
        // NewGELU (tanh form) exactly as spk-encoder.cpp builds it.
        ggml_tensor * new_gelu(ggml_tensor * x) const {
            ggml_tensor * x3 = ggml_mul(c, ggml_mul(c, x, x), x);
            ggml_tensor * inner = ggml_scale(c, ggml_add(c, x, ggml_scale(c, x3, 0.044715f)), 0.7978845608028654f);
            return ggml_scale(c, ggml_mul(c, x, ggml_scale_bias(c, ggml_tanh(c, inner), 1.0f, 1.0f)), 0.5f);
        }
        // [T,F,C] -> [F*C, T] with k = f*C + c.
        ggml_tensor * flatten(ggml_tensor * x) const {
            const int64_t T = x->ne[0], F = x->ne[1], C = x->ne[2];
            return ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, x, 2, 1, 0, 3)), F * C, T);
        }
        // [F*C, T] -> [T, F, C]
        ggml_tensor * unflatten(ggml_tensor * x, int64_t F, int64_t C) const {
            return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, x, C, F, x->ne[1]), 2, 1, 0, 3));
        }
        ggml_tensor * mul_mat(ggml_tensor * a, ggml_tensor * b) const {
            ggml_tensor * out = ggml_mul_mat(c, a, b);
            ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            return out;
        }
        ggml_tensor * attention(ggml_tensor * h, const std::string & p) const {
            const int64_t T = h->ne[1];
            const int hd = kHidden / kHeads;
            ggml_tensor * q = ggml_scale(c, linear(h, p + "attn.q"), 1.0f / std::sqrt(static_cast<float>(hd)));
            ggml_tensor * k = linear(h, p + "attn.k");
            ggml_tensor * v = linear(h, p + "attn.v");
            auto heads = [&](ggml_tensor * z) {  // [64,T] -> [hd,T,nh]
                return ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, z, hd, kHeads, T), 0, 2, 1, 3));
            };
            ggml_tensor * qp = heads(q);
            ggml_tensor * kp = heads(k);
            ggml_tensor * vt = ggml_cont(c, ggml_permute(c, heads(v), 1, 0, 2, 3));  // [T,hd,nh]
            ggml_tensor * o = nullptr;
            for (int64_t a = 0; a < T; a += 512) {
                const int64_t nq = std::min<int64_t>(512, T - a);
                ggml_tensor * qb = ggml_view_3d(c, qp, hd, nq, kHeads, qp->nb[1], qp->nb[2], static_cast<size_t>(a) * qp->nb[1]);
                ggml_tensor * score = ggml_soft_max(c, mul_mat(kp, qb));  // [T,nq,nh]
                ggml_tensor * ob = mul_mat(vt, score);                    // [hd,nq,nh]
                ob = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, ob, 0, 2, 1, 3)), kHidden, nq);
                o = o ? ggml_concat(c, o, ob, 1) : ob;
            }
            return linear(o, p + "attn.o");
        }
        // Dense-softmax token MoE: every expert runs, mixed by the softmax gate.
        ggml_tensor * moe_ffn(ggml_tensor * h, const std::string & p) const {
            ggml_tensor * r = ggml_soft_max(c, linear(h, p + "moe.gate", false));  // [4,T]
            ggml_tensor * e = new_gelu(linear(h, p + "moe.experts", false));       // [256,T]
            ggml_tensor * mix = nullptr;
            for (int i = 0; i < kExperts; ++i) {
                ggml_tensor * ei = ggml_view_2d(c, e, kHidden, e->ne[1], e->nb[1], static_cast<size_t>(i) * kHidden * sizeof(float));
                ggml_tensor * ri = ggml_view_2d(c, r, 1, r->ne[1], r->nb[1], static_cast<size_t>(i) * sizeof(float));
                ggml_tensor * term = ggml_mul(c, ei, ri);
                mix = mix ? ggml_add(c, mix, term) : term;
            }
            return linear(new_gelu(mix), p + "ffn_out");
        }
        ggml_tensor * pos_conv(ggml_tensor * h, const std::string & p) const {
            const int pad = kPosKernel / 2;
            const int64_t T = h->ne[1];
            ggml_tensor * hp = ggml_pad_ext(c, h, 0, 0, pad, pad, 0, 0, 0, 0);  // [64, T+2*pad]
            ggml_tensor * wt = ggml_cont(c, ggml_transpose(c, m.W(p + "pos.dw.weight")));  // [K,64] -> [64,K]
            ggml_tensor * acc = nullptr;
            for (int k = 0; k < kPosKernel; ++k) {
                ggml_tensor * term = ggml_mul(c, ggml_view_2d(c, hp, kHidden, T, hp->nb[1], static_cast<size_t>(k) * hp->nb[1]),
                                              ggml_view_2d(c, wt, kHidden, 1, wt->nb[1], static_cast<size_t>(k) * wt->nb[1]));
                acc = acc ? ggml_add(c, acc, term) : term;
            }
            acc = ggml_add(c, acc, m.W(p + "pos.dw.bias"));
            acc = ggml_add(c, ggml_mul(c, acc, m.W(p + "pos.bn.scale")), m.W(p + "pos.bn.shift"));
            acc = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, matrix(acc)).tensor;
            return linear(acc, p + "pos.pw");
        }
        // One MECT block. x: [T,F,C] spatial and xf its flatten [2560,T]; returns spatial and flat outputs.
        std::pair<ggml_tensor *, ggml_tensor *> block(ggml_tensor * x, ggml_tensor * xf, const BlockConfig & b,
                                                      const std::string & p) const {
            const int64_t T = x->ne[0];
            ggml_tensor * y = x;
            for (int r = 0; r < b.cnn; ++r) {
                y = res_block(y, p + "cnn." + std::to_string(r) + ".", r == 0 ? b.s_f : 1, r == 0 ? b.s_t : 1);
            }
            if (m.has(p + "reduce.weight")) y = bn4(conv(y, p + "reduce", 1, 1, true), p + "reduce.bn");
            const int64_t F2 = y->ne[1], C2 = y->ne[2];
            ggml_tensor * x2 = flatten(y);  // [2560,T']
            ggml_tensor * h = layer_norm(linear(x2, p + "rproj"), p + "rproj.ln");
            h = ggml_add(c, h, pos_conv(h, p));
            h = layer_norm(ggml_add(c, h, attention(h, p)), p + "ln1");
            h = layer_norm(ggml_add(c, h, moe_ffn(h, p)), p + "ln2");
            ggml_tensor * e = ggml_add(c, linear(h, p + "eproj"), x2);  // [2560,T']
            if (b.s_t > 1) {
                const int64_t Tp = e->ne[1];
                ggml_tensor * e3 = ggml_reshape_3d(c, e, kFeat, 1, Tp);
                ggml_tensor * rep = ggml_repeat(c, e3, ggml_new_tensor_3d(c, GGML_TYPE_F32, kFeat, b.s_t, Tp));
                e = ggml_reshape_2d(c, rep, kFeat, static_cast<int64_t>(b.s_t) * Tp);
                e = ggml_view_2d(c, e, kFeat, T, e->nb[1], 0);
            }
            ggml_tensor * out = ggml_add(c, e, xf);  // [2560,T]
            return {unflatten(out, F2, C2), out};
        }
    };

    void build_graph(int T) const {
        graph_.reset();
        frames_ = 0;
        auto owner = std::make_unique<core::PreparedGraph>(execution_, core::PreparedGraphConfig{
            "zonos3.speaker_encoder", ggml_graph_overhead_custom(kGraphNodes, false) + 2 * kGraphNodes * ggml_tensor_overhead(),
            kGraphNodes, 4 * ggml_tensor_overhead()});
        auto & prepared = *owner;
        ggml_tensor * input = ggml_new_tensor_4d(prepared.persistent_build_context().ggml, GGML_TYPE_F32, T, kMels, 1, 1);
        prepared.allocate_persistent();
        auto & ctx = prepared.build_context();
        ggml_context * c = ctx.ggml;
        const Builder g{*this, ctx, c};

        ggml_tensor * x = g.bn4(g.conv(input, "fe.head.conv", 1, 1, true, 1), "fe.head.bn");
        ggml_tensor * xf = g.flatten(x);
        std::vector<ggml_tensor *> feats;
        const auto & stages = mect_b2();
        for (int s = 0; s < static_cast<int>(stages.size()); ++s) {
            for (int k = 0; k < static_cast<int>(stages[static_cast<size_t>(s)].size()); ++k) {
                const auto r = g.block(x, xf, stages[static_cast<size_t>(s)][static_cast<size_t>(k)], block_name(s, k));
                x = r.first;
                xf = r.second;
            }
            feats.push_back(xf);
        }
        ggml_tensor * dense = W("fe.dense_w");
        ggml_tensor * fused = nullptr;
        for (int s = 0; s < 4; ++s) {
            ggml_tensor * ws = ggml_view_2d(c, dense, kFeat, 1, dense->nb[1], static_cast<size_t>(s) * dense->nb[1]);
            ggml_tensor * term = ggml_mul(c, feats[static_cast<size_t>(s)], ws);
            fused = fused ? ggml_add(c, fused, term) : term;
        }
        // Attentive statistics pooling over time.
        ggml_tensor * att = g.linear(ggml_tanh(c, g.linear(fused, "pool.l1")), "pool.l2");  // [2560,T]
        ggml_tensor * alpha = ggml_soft_max(c, ggml_cont(c, ggml_transpose(c, att)));     // [T,2560]
        ggml_tensor * xt = ggml_cont(c, ggml_transpose(c, fused));                        // [T,2560]
        const auto shape = core::TensorShape::from_dims({kFeat, T});
        const auto statistics = modules::WeightedStatisticsModule({-1, kPoolVarFloor, 3.0e38f}).build(
            ctx, core::wrap_tensor(xt, shape), core::wrap_tensor(alpha, shape));
        auto * mean = statistics.mean.tensor;
        auto * stdv = statistics.stddev.tensor;
        ggml_tensor * pooled = ggml_concat(c, ggml_reshape_2d(c, mean, kFeat, 1), ggml_reshape_2d(c, stdv, kFeat, 1), 0);
        ggml_tensor * pooled_bn = ggml_add(c, ggml_mul(c, pooled, W("bn.scale")), W("bn.shift"));
        ggml_tensor * emb = g.linear(pooled_bn, "linear", false);  // [192,1]
        ggml_set_output(emb);
        ggml_build_forward_expand(prepared.graph(), emb);
        prepared.allocate();
        input_ = input;
        output_ = emb;
        frames_ = T;
        graph_ = std::move(owner);
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::map<std::string, core::TensorValue> weights_;
    mutable std::unique_ptr<core::PreparedGraph> graph_;
    mutable ggml_tensor * input_ = nullptr;
    mutable ggml_tensor * output_ = nullptr;
    mutable int frames_ = 0;
};

Zonos3SpeakerEncoder::Zonos3SpeakerEncoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), execution)) {}

Zonos3SpeakerEncoder::~Zonos3SpeakerEncoder() = default;

std::vector<float> Zonos3SpeakerEncoder::embed(const std::vector<float> & mono_44k) const { return impl_->embed(mono_44k); }

}  // namespace engine::models::zonos3
