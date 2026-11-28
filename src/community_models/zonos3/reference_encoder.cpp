#include "engine/community_models/zonos3/reference_encoder.h"

#include "engine/community_models/zonos3/ssl_teacher.h"
#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"

#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::zonos3 {
namespace {

using core::TensorValue;

constexpr int kLane = 300;           // 15 Hz frames per lane
constexpr int kContext = 150;        // left-context frames of continuation lanes
constexpr int kHop = 2940;           // 44.1 kHz samples per frame
constexpr int kInternalHop = 2944;   // 44.16 kHz samples per frame
constexpr int kCodebooks = 10;
constexpr int kCodebookSize = 4096;
constexpr int kLatent = 1024;
constexpr int kSemantic = 768;
constexpr int kSemanticDepth = 4;
constexpr int kMaxSeconds = 600;
constexpr int kLaneSamples = kLane * kInternalHop;
constexpr int64_t kPingPongElements = static_cast<int64_t>(64) * kLaneSamples;  // largest activation: [64, 883200]
constexpr int kGraphNodes = 8192;
constexpr int kAttentionBlock = 512;

struct StageParams {
    int in_dim = 0, out_dim = 0, d_model = 0, n_head = 0, n_layer = 0, n_ff = 0, context = 0, patch = 0;
};

// Encoder Conv1d as one BF16 GEMM over im2col windows: weight rows [OC][IC * K], tap fastest.
struct ConvWeights {
    TensorValue weight;  // BF16 {OC, IC * K}
    TensorValue bias;    // F32 {OC, 1} (ggml [1, OC]), BF16-rounded values
    int64_t in = 0, out = 0, kernel = 0;
};

struct Quantizer {
    int input_dim = 0;
    int code_dim = 8;
    int size = kCodebookSize;
    std::vector<float> in_weight, in_bias, out_weight, out_bias, codebook, normalized, norm_squared;
};

TensorValue tv(ggml_tensor * x) {
    const int rank = ggml_n_dims(x);
    core::TensorShape shape;
    shape.rank = static_cast<size_t>(rank);
    for (int i = 0; i < rank; ++i) shape.dims[static_cast<size_t>(i)] = x->ne[rank - 1 - i];
    return core::wrap_tensor(x, shape);
}

ggml_tensor * mul_mat_f32(ggml_context * g, ggml_tensor * a, ggml_tensor * b) {
    auto * out = ggml_mul_mat(g, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

// sin(v)^2 by Cody-Waite reduction and the Cephes sinf/cosf minimax polynomials, as ZONOS3's CPU Snake computes it.
inline void sin2_f32(int n, float * y, const float * v) {
    for (int i = 0; i < n; ++i) {
        const float q = rintf(v[i] * 0.63661977236758134f);
        float r = v[i] - q * 1.5703125f;
        r -= q * 4.837512969970703125e-4f;
        r -= q * 7.54978995489188216e-8f;
        const float r2 = r * r;
        const float s = r + r * r2 * ((-1.9515295891e-4f * r2 + 8.3321608736e-3f) * r2 - 1.6666654611e-1f);
        const float c = 1.0f - 0.5f * r2 + r2 * r2 * ((2.443315711809948e-5f * r2 - 1.388731625493765e-3f) * r2 + 4.166664568298827e-2f);
        const float t = (static_cast<int>(q) & 1) ? c : s;
        y[i] = t * t;
    }
}

// snake(x) = x + inv_alpha * sin(alpha * x)^2 over the columns of a contiguous [C, T] tensor.
void snake_cpu(ggml_tensor * dst, const ggml_tensor * x, const ggml_tensor * alpha, const ggml_tensor * inv_alpha, int ith,
               int nth, void *) {
    const int64_t C = x->ne[0];
    const int64_t T = ggml_nrows(x);
    const auto * a = static_cast<const float *>(alpha->data);
    const auto * ia = static_cast<const float *>(inv_alpha->data);
    float buf[256];
    for (int64_t t = T * ith / nth, t1 = T * (ith + 1) / nth; t < t1; ++t) {
        const float * xs = static_cast<const float *>(x->data) + t * C;
        float * ys = static_cast<float *>(dst->data) + t * C;
        for (int64_t c0 = 0; c0 < C; c0 += 256) {
            const int n = static_cast<int>(std::min<int64_t>(256, C - c0));
            for (int i = 0; i < n; ++i) buf[i] = a[c0 + i] * xs[c0 + i];
            sin2_f32(n, buf, buf);
            for (int i = 0; i < n; ++i) ys[c0 + i] = xs[c0 + i] + buf[i] * ia[c0 + i];
        }
    }
}

bool autocast_parameter(const std::string & name) {
    return name.find("norm") == std::string::npos &&
           (name.find(".weight") != std::string::npos || name.find(".bias") != std::string::npos);
}

}  // namespace

class Zonos3ReferenceEncoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> encoder, std::shared_ptr<const assets::TensorSource> teacher,
         core::ExecutionContext & execution)
        : source_(std::move(encoder)), execution_(execution) {
        validate_metadata();
        load_weights();
        teacher_ = std::make_unique<Zonos3SslTeacher>(std::move(teacher), execution_);
        allocate_ping_pong();
    }

    std::vector<int32_t> encode(const std::vector<float> & audio) const {
        const int64_t n64 = static_cast<int64_t>(audio.size());
        if (n64 <= 0 || n64 > static_cast<int64_t>(kMaxSeconds) * 44100 ||
            !std::all_of(audio.begin(), audio.end(), [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("zonos3: the reference encoder takes finite, non-empty 44.1 kHz mono audio of at most 600 s");
        }
        const int n = static_cast<int>(n64);
        const int frames = (n + kHop - 1) / kHop;
        auto internal = resample(audio);
        internal.resize(static_cast<size_t>(frames) * kInternalHop, 0.0f);
        std::vector<int32_t> result(static_cast<size_t>(frames) * kCodebooks);
        std::vector<float> lane_audio(static_cast<size_t>(kLaneSamples));
        std::vector<float> teacher_audio(static_cast<size_t>(kLane) * kHop);
        for (int start = 0; start < frames;) {
            const int context = start == 0 ? 0 : kContext;
            const int count = std::min(frames - start, kLane - context);
            const int left = start - context;
            std::fill(lane_audio.begin(), lane_audio.end(), 0.0f);
            std::fill(teacher_audio.begin(), teacher_audio.end(), 0.0f);
            std::copy_n(internal.data() + static_cast<size_t>(left) * kInternalHop,
                        static_cast<size_t>(context + count) * kInternalHop, lane_audio.data());
            const int teacher_start = left * kHop;
            const int teacher_count = std::min(n - teacher_start, (context + count) * kHop);
            std::copy_n(audio.data() + teacher_start, teacher_count, teacher_audio.data());

            int ssl_frames = 0;
            const auto ssl = teacher_->encode(teacher_audio, ssl_frames);
            const auto acoustic = acoustic_lane(lane_audio);
            const auto pooled = pool_ssl(ssl, ssl_frames);
            std::vector<float> q1, prediction, qr;
            std::vector<int32_t> ids;
            const auto semantic = semantic_module(pooled, true);
            quantize(rvq_[0], semantic, q1, ids);
            prediction = semantic_module(q1, false);
            for (int t = 0; t < count; ++t) result[static_cast<size_t>(start + t) * kCodebooks] = ids[static_cast<size_t>(context + t)];
            std::vector<float> residual(acoustic.size());
            for (size_t i = 0; i < residual.size(); ++i) residual[i] = acoustic[i] - prediction[i];
            for (int c = 1; c < kCodebooks; ++c) {
                quantize(rvq_[static_cast<size_t>(c)], residual, qr, ids);
                for (int t = 0; t < count; ++t) {
                    result[static_cast<size_t>(start + t) * kCodebooks + static_cast<size_t>(c)] = ids[static_cast<size_t>(context + t)];
                }
                for (size_t i = 0; i < residual.size(); ++i) residual[i] -= qr[i];
            }
            start += count;
        }
        return result;
    }

private:
    // ---- loading ------------------------------------------------------------------------------------------------

    void validate_metadata() {
        const assets::GgufMetadata meta(source_->source_path());
        auto fail = [](const std::string & what) {
            throw std::runtime_error("zonos3: zac-encoder GGUF: " + what);
        };
        if (meta.require_string("general.architecture") != "mossdual_encoder") fail("general.architecture must be mossdual_encoder");
        const std::array<std::pair<const char *, int64_t>, 15> ints{{
            {"sample_rate", 44100}, {"internal_sample_rate", 44160}, {"frame_rate", 15}, {"hop", kHop},
            {"internal_hop", kInternalHop}, {"n_codebooks", kCodebooks}, {"codebook_size", kCodebookSize},
            {"latent_dim", kLatent}, {"semantic_dim", kSemantic}, {"semantic_depth", kSemanticDepth},
            {"lane_frames", kLane}, {"context_frames", kContext}, {"resample.orig", 735}, {"resample.new", 736},
            {"resample.width", 17}}};
        for (const auto & [key, expected] : ints) {
            const std::string full = std::string("mossdual_encoder.") + key;
            if (meta.require_int(full.c_str()) != expected) fail(full + " must be " + std::to_string(expected));
        }
        if (meta.require_f32("mossdual_encoder.window_ratio") != 2.0f) fail("mossdual_encoder.window_ratio must be 2");
        ln_eps_ = meta.require_f32("mossdual_encoder.ln_eps");
        convnext_eps_ = meta.require_f32("mossdual_encoder.convnext_eps");
        if (!(ln_eps_ > 0.0f) || !(convnext_eps_ > 0.0f)) fail("normalization epsilons must be positive");
        std::array<std::vector<int64_t>, 8> arrays;
        const std::array<const char *, 8> keys{"in_dim", "out_dim", "d_model", "n_head", "n_layer", "n_ff", "context", "patch"};
        for (size_t i = 0; i < keys.size(); ++i) {
            arrays[i] = meta.require_int_array((std::string("mossdual_encoder.stage.") + keys[i]).c_str());
            if (arrays[i].size() != 5) fail(std::string("mossdual_encoder.stage.") + keys[i] + " must hold 5 stages");
        }
        for (size_t s = 0; s < 5; ++s) {
            StageParams hp;
            hp.in_dim = static_cast<int>(arrays[0][s]);
            hp.out_dim = static_cast<int>(arrays[1][s]);
            hp.d_model = static_cast<int>(arrays[2][s]);
            hp.n_head = static_cast<int>(arrays[3][s]);
            hp.n_layer = static_cast<int>(arrays[4][s]);
            hp.n_ff = static_cast<int>(arrays[5][s]);
            hp.context = static_cast<int>(arrays[6][s]);
            hp.patch = static_cast<int>(arrays[7][s]);
            const bool valid = hp.in_dim > 0 && hp.out_dim > 0 && hp.d_model > 0 && hp.n_head > 0 && hp.n_layer > 0 &&
                               hp.n_ff > 0 && hp.context > 0 && hp.patch == (s == 0 ? 23 : 2) &&
                               hp.d_model % hp.n_head == 0 && hp.d_model / hp.n_head == 64 &&
                               (s == 0 ? hp.in_dim == 256 * 23 : hp.in_dim == stages_.back().out_dim * hp.patch);
            if (!valid) fail("stage " + std::to_string(s) + " dimensions are not supported");
            stages_.push_back(hp);
        }
        if (stages_.back().out_dim != kLatent) fail("the last stage must output the 1024-wide latent");
    }

    // Logical shape (reverse of ggml ne) check against a ggml-order list.
    void require_ne(const std::string & name, std::initializer_list<int64_t> ne) const {
        auto shape = source_->require_metadata(name).shape;
        std::vector<int64_t> expected(ne.begin(), ne.end());
        while (expected.size() > 1 && expected.back() == 1) expected.pop_back();
        std::reverse(expected.begin(), expected.end());
        if (shape != expected) throw std::runtime_error("zonos3: zac-encoder GGUF: " + name + " has an unexpected shape");
    }

    void load_weights() {
        const auto & source = *source_;
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos3.reference_encoder.weights", 4 * 1024 * 1024);
        // Shape checks (ggml ne order), mirroring ZONOS3's loader.
        auto conv_ok = [&](const std::string & p, int64_t ic, int64_t oc, int64_t k) {
            require_ne(p + ".weight", {ic, oc, k});
            require_ne(p + ".bias", {1, oc});
        };
        auto snake_ok = [&](const std::string & p, int64_t c) {
            require_ne(p + ".alpha", {1, c});
            require_ne(p + ".inv_alpha", {1, c});
        };
        conv_ok("enc.in", 1, 64, 7);
        for (int b = 0; b < 2; ++b) {
            const std::string p = "enc.b" + std::to_string(b);
            const int64_t C = b == 0 ? 64 : 128;
            snake_ok(p + ".snake", C);
            conv_ok(p + ".conv", C, 2 * C, b == 0 ? 4 : 8);
            for (int r = 0; r < 3; ++r) {
                const std::string q = p + ".res" + std::to_string(r);
                snake_ok(q + ".snake1", C);
                snake_ok(q + ".snake2", C);
                conv_ok(q + ".conv1", C, C, 7);
                conv_ok(q + ".conv2", C, C, 1);
            }
        }
        for (int s = 0; s < 5; ++s) {
            const auto & hp = stages_[static_cast<size_t>(s)];
            const std::string p = "stage." + std::to_string(s);
            require_ne(p + ".in.weight", {hp.in_dim, hp.d_model});
            require_ne(p + ".out.weight", {hp.d_model, hp.out_dim});
            for (int l = 0; l < hp.n_layer; ++l) {
                const std::string q = p + ".l" + std::to_string(l);
                for (const char * k : {"norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn_scale", "ffn_scale"}) {
                    require_ne(q + "." + k, {hp.d_model});
                }
                require_ne(q + ".qkv.weight", {hp.d_model, 3 * hp.d_model});
                require_ne(q + ".wo.weight", {hp.d_model, hp.d_model});
                require_ne(q + ".ffn1.weight", {hp.d_model, hp.n_ff});
                require_ne(q + ".ffn2.weight", {hp.n_ff, hp.d_model});
            }
        }
        for (const char * branch : {"sem_enc", "sem_head"}) {
            for (int b = 0; b < kSemanticDepth; ++b) {
                const std::string p = std::string(branch) + "." + std::to_string(b);
                require_ne(p + ".dw.weight", {kSemantic, 7});
                require_ne(p + ".dw.bias", {kSemantic});
                require_ne(p + ".norm.weight", {kSemantic});
                require_ne(p + ".norm.bias", {kSemantic});
                require_ne(p + ".pw1.weight", {kSemantic, 2048});
                require_ne(p + ".pw1.bias", {2048});
                require_ne(p + ".pw2.weight", {2048, kSemantic});
                require_ne(p + ".pw2.bias", {kSemantic});
            }
        }
        require_ne("sem_enc.in.weight", {kLatent, kSemantic});
        require_ne("sem_enc.in.bias", {kSemantic});
        require_ne("sem_head.out.weight", {kSemantic, kLatent});
        require_ne("sem_head.out.bias", {kLatent});
        require_ne("resample.kernel", {2 * 17 + 735, 736});
        for (int c = 0; c < kCodebooks; ++c) {
            const std::string p = "rvq." + std::to_string(c);
            const int64_t C = c == 0 ? kSemantic : kLatent;
            require_ne(p + ".in_proj.weight", {C, 8});
            require_ne(p + ".in_proj.bias", {8});
            require_ne(p + ".out_proj.weight", {8, C});
            require_ne(p + ".out_proj.bias", {C});
            require_ne(p + ".codebook", {8, kCodebookSize});
        }

        // Precision follows ZONOS3's CUDA-autocast emulation: linear matrices are BF16 operands, convolutions
        // become BF16 GEMM rows, other autocast parameters hold BF16-rounded values in F32, norms, Snake and layer
        // scales stay F32, and the RVQ / resampler tables stay on the host.
        for (const auto & meta : source.tensors()) {
            if (meta.dtype != "f32") throw std::runtime_error("zonos3: zac-encoder GGUF: " + meta.name + " must be F32");
            const std::string & name = meta.name;
            if (name == "resample.kernel") {
                resample_kernel_ = source.require_f32(name);
                continue;
            }
            if (name.rfind("rvq.", 0) == 0) continue;  // read per quantizer below
            const bool stage_matrix = name.rfind("stage.", 0) == 0 && name.find(".weight") != std::string::npos &&
                                      name.find("norm") == std::string::npos && meta.shape.size() == 2;
            const bool linear_matrix = meta.shape.size() == 2 &&
                (stage_matrix || name == "sem_enc.in.weight" || name == "sem_head.out.weight" ||
                 name.find(".pw1.weight") != std::string::npos || name.find(".pw2.weight") != std::string::npos);
            if (linear_matrix) {
                weights_[name] = store_->load_tensor(source, name, assets::TensorStorageType::BF16, meta.shape);
                continue;
            }
            // Conv1d kernels (ggml [IC, OC, K]; K = 1 drops to two dims): every enc.* weight except Snake's.
            if (name.rfind("enc.", 0) == 0 && name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
                convs_[name.substr(0, name.size() - 7)].weight = conv_weight(name, meta.shape);
                continue;
            }
            auto values = source.require_f32(name);
            if (autocast_parameter(name)) core::round_f32_to_bf16_in_place(values);
            weights_[name] = make_f32(meta.shape, std::move(values));
        }
        for (auto & [prefix, conv] : convs_) {
            conv.bias = weights_.at(prefix + ".bias");
            conv.out = conv.weight.shape.dims[0];
        }
        for (int c = 0; c < kCodebooks; ++c) {
            const std::string p = "rvq." + std::to_string(c);
            Quantizer q;
            q.input_dim = c == 0 ? kSemantic : kLatent;
            q.in_weight = source.require_f32(p + ".in_proj.weight");
            q.in_bias = source.require_f32(p + ".in_proj.bias");
            q.out_weight = source.require_f32(p + ".out_proj.weight");
            q.out_bias = source.require_f32(p + ".out_proj.bias");
            q.codebook = source.require_f32(p + ".codebook");
            q.normalized.resize(q.codebook.size());
            q.norm_squared.resize(static_cast<size_t>(q.size));
            for (int k = 0; k < q.size; ++k) {
                const float * row = q.codebook.data() + static_cast<size_t>(k) * q.code_dim;
                float norm = 0;
                for (int d = 0; d < q.code_dim; ++d) norm += row[d] * row[d];
                norm = std::max(std::sqrt(norm), 1e-12f);
                float squared = 0;
                for (int d = 0; d < q.code_dim; ++d) {
                    const float v = row[d] / norm;
                    q.normalized[static_cast<size_t>(k) * q.code_dim + static_cast<size_t>(d)] = v;
                    squared += v * v;
                }
                q.norm_squared[static_cast<size_t>(k)] = squared;
            }
            rvq_.push_back(std::move(q));
        }
        store_->upload();
        source_->release_storage();
    }

    TensorValue make_f32(const std::vector<int64_t> & shape, std::vector<float> values) {
        core::TensorShape s;
        s.rank = shape.size();
        for (size_t i = 0; i < shape.size(); ++i) s.dims[i] = shape[i];
        return store_->make_f32(s, std::move(values));
    }

    // Conv1d weight ggml [IC, OC, K] -> BF16 GEMM rows [OC][IC * K] (PyTorch [OC, IC, K] order, tap fastest),
    // matching im2col's column order.
    TensorValue conv_weight(const std::string & name, const std::vector<int64_t> & shape) {
        if (shape.size() != 2 && shape.size() != 3) throw std::runtime_error("zonos3: zac-encoder GGUF: " + name + " is not a Conv1d kernel");
        const int64_t K = shape.size() == 3 ? shape[0] : 1;
        const int64_t OC = shape[shape.size() - 2];
        const int64_t IC = shape.back();
        const auto w = source_->require_f32(name);
        std::vector<float> rows(w.size());
        for (int64_t oc = 0; oc < OC; ++oc) {
            for (int64_t ic = 0; ic < IC; ++ic) {
                for (int64_t k = 0; k < K; ++k) {
                    rows[static_cast<size_t>((oc * IC + ic) * K + k)] = w[static_cast<size_t>((k * OC + oc) * IC + ic)];
                }
            }
        }
        auto value = store_->make_from_f32(core::TensorShape::from_dims({OC, IC * K}), assets::TensorStorageType::BF16,
                                           std::move(rows));
        const std::string prefix = name.substr(0, name.size() - std::string(".weight").size());
        convs_[prefix].in = IC;
        convs_[prefix].kernel = K;
        return value;
    }

    const TensorValue & weight(const std::string & name) const {
        const auto it = weights_.find(name);
        if (it == weights_.end()) throw std::runtime_error("zonos3: zac-encoder GGUF: missing tensor " + name);
        return it->second;
    }

    void allocate_ping_pong() {
        ping_pong_context_.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!ping_pong_context_) throw std::runtime_error("zonos3: cannot allocate reference encoder activation metadata");
        for (auto *& buffer : ping_pong_) buffer = ggml_new_tensor_1d(ping_pong_context_.get(), GGML_TYPE_F32, kPingPongElements);
        ping_pong_buffer_.reset(ggml_backend_alloc_ctx_tensors(ping_pong_context_.get(), execution_.backend()));
        if (!ping_pong_buffer_) throw std::runtime_error("zonos3: cannot allocate reference encoder activations");
    }

    // ---- graph pieces -------------------------------------------------------------------------------------------

    static ggml_tensor * round(core::ModuleBuildContext & ctx, ggml_tensor * x) {
        return modules::Bf16RoundModule().build(ctx, tv(x)).tensor;
    }

    static ggml_tensor * add_bias(ggml_context * g, ggml_tensor * y, const TensorValue & bias) {
        return ggml_add(g, y, ggml_reshape_2d(g, bias.tensor, y->ne[0], 1));
    }

    // BF16 GEMM without bias, F32 accumulation.
    static ggml_tensor * linear(core::ModuleBuildContext & ctx, ggml_tensor * x, const TensorValue & w) {
        modules::LinearConfig config;
        config.in_features = w.shape.dims[1];
        config.out_features = w.shape.dims[0];
        config.use_bias = false;
        config.precision = GGML_PREC_F32;
        return modules::LinearModule(config).build(ctx, tv(x), {w, std::nullopt}).tensor;
    }

    ggml_tensor * layer_norm(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p, float eps) const {
        modules::NormConfig config;
        config.hidden_size = x->ne[0];
        config.eps = eps;
        return modules::LayerNormModule(config).build(ctx, tv(x), {weight(p + ".weight"), weight(p + ".bias")}).tensor;
    }

    ggml_tensor * gelu(core::ModuleBuildContext & ctx, ggml_tensor * x) const {
        return round(ctx, modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, tv(x)).tensor);
    }

    // The input, the GEMM output and the biased output are BF16 (cuDNN returns the BF16 convolution first, then
    // adds the bias in BF16). Causal padding (K - 1) * d + 1 - stride on the left.
    ggml_tensor * conv(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p, int dilation, int stride) const {
        auto * g = ctx.ggml;
        const auto & c = convs_.at(p);
        x = round(ctx, x);
        ggml_tensor * columns = x;
        if (c.kernel > 1 || stride != 1) {
            const int left = static_cast<int>((c.kernel - 1) * dilation + 1 - stride);
            // Keep time contiguous: CUDA padding dispatches the other axes through grid Y/Z.
            auto * input = ggml_cont(g, ggml_transpose(g, x));  // [T, IC]
            input = ggml_pad_ext(g, input, left, 0, 0, 0, 0, 0, 0, 0);  // [T + left, IC]
            auto * kernel = ggml_reshape_3d(g, c.weight.tensor, c.kernel, c.in, c.out);  // im2col reads its shape
            // F32 im2col keeps the already-rounded BF16 inputs exact; the GEMM narrows them to BF16.
            auto * windows = ggml_im2col(g, kernel, input, stride, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
            columns = ggml_reshape_2d(g, windows, c.in * c.kernel, windows->ne[1]);
        }
        auto * y = mul_mat_f32(g, c.weight.tensor, columns);
        return round(ctx, add_bias(g, round(ctx, y), c.bias));
    }

    ggml_tensor * snake(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p) const {
        auto * g = ctx.ggml;
        auto * alpha = weight(p + ".alpha").tensor;
        auto * inv_alpha = weight(p + ".inv_alpha").tensor;
        if (execution_.backend_type() == core::BackendType::Cpu) {
            return ggml_map_custom3(g, ggml_is_contiguous(x) ? x : ggml_cont(g, x), alpha, inv_alpha, snake_cpu,
                                    GGML_N_TASKS_MAX, nullptr);
        }
        const int64_t C = x->ne[0];
        auto * s2 = ggml_sqr(g, ggml_sin(g, ggml_mul(g, x, ggml_reshape_2d(g, alpha, C, 1))));
        return ggml_add(g, x, ggml_mul(g, s2, ggml_reshape_2d(g, inv_alpha, C, 1)));
    }

    // Residual unit: snake -> conv k7 (dilation d) -> snake -> conv k1, plus the input; the sum is BF16.
    ggml_tensor * residual_unit(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p, int dilation) const {
        auto * y = conv(ctx, snake(ctx, x, p + ".snake1"), p + ".conv1", dilation, 1);
        y = conv(ctx, snake(ctx, y, p + ".snake2"), p + ".conv2", 1, 1);
        return round(ctx, ggml_add(ctx.ggml, x, y));
    }

    // Causal ConvNeXt: depthwise k7 over the left-padded BF16 input, LayerNorm, pw1, exact GELU, pw2, residual.
    ggml_tensor * convnext(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p, bool round_output) const {
        auto * g = ctx.ggml;
        const int64_t C = x->ne[0];
        const int64_t T = x->ne[1];
        const auto & dw = weight(p + ".dw.weight");  // ggml [C, K]
        const int K = static_cast<int>(dw.tensor->ne[1]);
        auto * xf = ggml_cont(g, ggml_transpose(g, round(ctx, x)));
        xf = ggml_pad_ext(g, xf, K - 1, 0, 0, 0, 0, 0, 0, 0);
        xf = ggml_cont(g, ggml_transpose(g, xf));  // [C, T + K - 1]
        ggml_tensor * h = nullptr;
        for (int k = 0; k < K; ++k) {
            auto * term = ggml_mul(g, ggml_view_2d(g, xf, C, T, xf->nb[1], static_cast<size_t>(k) * xf->nb[1]),
                                   ggml_view_1d(g, dw.tensor, C, static_cast<size_t>(k) * static_cast<size_t>(C) * sizeof(float)));
            h = h ? ggml_add(g, h, term) : term;
        }
        h = round(ctx, ggml_add(g, round(ctx, h), weight(p + ".dw.bias").tensor));
        h = round(ctx, layer_norm(ctx, h, p + ".norm", convnext_eps_));
        h = gelu(ctx, round(ctx, ggml_add(g, linear(ctx, h, weight(p + ".pw1.weight")), weight(p + ".pw1.bias").tensor)));
        h = round(ctx, ggml_add(g, linear(ctx, h, weight(p + ".pw2.weight")), weight(p + ".pw2.bias").tensor));
        auto * y = ggml_add(g, x, h);
        return round_output ? round(ctx, y) : y;
    }

    // MOSS layer: pre-LN band-masked attention (adjacent-pair RoPE, 512-query blocks) and an exact-GELU FFN, each
    // with a layer scale. qkv, roped q/k, the attention output, wo, ffn1, GELU and ffn2 are BF16; the residual
    // stream, LayerNorm and attention stay F32.
    ggml_tensor * moss(core::ModuleBuildContext & ctx, ggml_tensor * x, const std::string & p, const StageParams & hp,
                       ggml_tensor * pos, ggml_tensor * mask, bool first) const {
        auto * g = ctx.ggml;
        const int d = hp.d_model;
        const int nh = hp.n_head;
        const int hd = d / nh;
        const int64_t T = x->ne[1];
        auto * h = layer_norm(ctx, x, p + ".norm1", ln_eps_);
        if (first) h = round(ctx, h);
        auto * qkv = round(ctx, linear(ctx, round(ctx, h), weight(p + ".qkv.weight")));  // [3d, T]
        auto section = [&](int s) {
            return ggml_cont(g, ggml_view_3d(g, qkv, hd, nh, T, static_cast<size_t>(hd) * sizeof(float), qkv->nb[1],
                                             static_cast<size_t>(s) * static_cast<size_t>(d) * sizeof(float)));
        };
        modules::RoPEConfig rope_config;
        rope_config.dimensions = hd;
        rope_config.mode = GGML_ROPE_TYPE_NORMAL;
        rope_config.theta = 10000.0f;
        const modules::RoPEModule rope_module(rope_config);
        const auto positions = core::wrap_tensor(pos, core::TensorShape::from_dims({T}), GGML_TYPE_I32);
        auto rope = [&](ggml_tensor * z) {
            auto value = core::wrap_tensor(z, core::TensorShape::from_dims({1, T, nh, hd}));
            return round(ctx, rope_module.build(ctx, value, positions).tensor);
        };
        auto * q = rope(section(0));                                                 // [hd, nh, T]
        auto * kb = ggml_cont(g, ggml_permute(g, rope(section(1)), 0, 2, 1, 3));    // [hd, T, nh]
        auto * vb = ggml_cont(g, ggml_permute(g, section(2), 0, 2, 1, 3));
        auto * qp = ggml_cont(g, ggml_permute(g, q, 0, 2, 1, 3));                   // [hd, T, nh]
        ggml_tensor * o = nullptr;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        for (int64_t a = 0; a < T; a += kAttentionBlock) {
            const int64_t end = std::min<int64_t>(a + kAttentionBlock, T);
            const int64_t left = std::max<int64_t>(0, a - hp.context + 1);
            const int64_t nk = end - left;
            const int64_t nq = end - a;
            auto * qb = ggml_view_3d(g, qp, hd, nq, nh, qp->nb[1], qp->nb[2], static_cast<size_t>(a) * qp->nb[1]);
            auto * kk = ggml_view_3d(g, kb, hd, nk, nh, kb->nb[1], kb->nb[2], static_cast<size_t>(left) * kb->nb[1]);
            auto * vv = ggml_view_3d(g, vb, hd, nk, nh, vb->nb[1], vb->nb[2], static_cast<size_t>(left) * vb->nb[1]);
            auto * mb = ggml_cont(g, ggml_view_2d(g, mask, nk, nq, mask->nb[1],
                                                  static_cast<size_t>(left) * mask->nb[0] + static_cast<size_t>(a) * mask->nb[1]));
            auto * score = ggml_soft_max_ext(g, mul_mat_f32(g, kk, qb), mb, scale, 0.0f);
            auto * vt = ggml_cont(g, ggml_permute(g, vv, 1, 0, 2, 3));
            auto * ob = mul_mat_f32(g, vt, score);
            ob = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, ob, 0, 2, 1, 3)), d, nq);
            o = o ? ggml_concat(g, o, ob, 1) : ob;
        }
        o = round(ctx, o);
        o = ggml_mul(g, round(ctx, linear(ctx, o, weight(p + ".wo.weight"))), weight(p + ".attn_scale").tensor);
        x = ggml_add(g, x, o);
        h = layer_norm(ctx, x, p + ".norm2", ln_eps_);
        h = gelu(ctx, round(ctx, linear(ctx, round(ctx, h), weight(p + ".ffn1.weight"))));
        h = round(ctx, linear(ctx, h, weight(p + ".ffn2.weight")));
        return ggml_add(g, x, ggml_mul(g, h, weight(p + ".ffn_scale").tensor));
    }

    // ---- ping-pong execution ------------------------------------------------------------------------------------

    using Builder = std::function<ggml_tensor *(core::ModuleBuildContext &, ggml_tensor *)>;

    struct Activations {
        int current = 0;
        int64_t channels = 0;
        int64_t frames = 0;
    };

    void reset(Activations & state, int64_t C, int64_t T, const float * values) const {
        if (C * T > kPingPongElements) throw std::runtime_error("zonos3: reference encoder activation exceeds its buffer");
        state = {0, C, T};
        ggml_backend_tensor_set(ping_pong_[0], values, 0, static_cast<size_t>(C * T) * sizeof(float));
    }

    // One op group as its own graph: reads the current ping-pong buffer, writes the other.
    void run(Activations & state, const Builder & build) const {
        const size_t metadata = ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead();
        core::PreparedGraph graph(execution_, core::PreparedGraphConfig{"zonos3.reference_encoder", metadata, kGraphNodes, 0});
        auto & ctx = graph.build_context();
        auto * g = ctx.ggml;
        auto * x = ggml_view_2d(g, ping_pong_[state.current], state.channels, state.frames,
                                static_cast<size_t>(state.channels) * sizeof(float), 0);
        auto * y = build(ctx, x);
        const int64_t C = y->ne[0];
        const int64_t T = y->ne[1];
        if (C * T > kPingPongElements || y->ne[2] != 1 || y->ne[3] != 1) {
            throw std::runtime_error("zonos3: reference encoder activation exceeds its buffer");
        }
        auto * dst = ggml_view_2d(g, ping_pong_[1 - state.current], C, T, static_cast<size_t>(C) * sizeof(float), 0);
        ggml_build_forward_expand(graph.graph(), ggml_cpy(g, y, dst));
        graph.allocate();
        graph.compute();
        state = {1 - state.current, C, T};
    }

    std::vector<float> host(const Activations & state) const {
        std::vector<float> out(static_cast<size_t>(state.channels * state.frames));
        ggml_backend_tensor_get(ping_pong_[state.current], out.data(), 0, out.size() * sizeof(float));
        return out;
    }

    // Stage inputs (positions 0..T-1 and the T x T band mask) for every layer of one MOSS stage.
    struct StageInputs {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
        ggml_tensor * positions = nullptr;
        ggml_tensor * mask = nullptr;
    };

    StageInputs stage_inputs(int T, int context) const {
        StageInputs in;
        in.context.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        if (!in.context) throw std::runtime_error("zonos3: cannot allocate reference encoder stage metadata");
        in.positions = ggml_new_tensor_1d(in.context.get(), GGML_TYPE_I32, T);
        in.mask = ggml_new_tensor_2d(in.context.get(), GGML_TYPE_F32, T, T);
        in.buffer.reset(ggml_backend_alloc_ctx_tensors(in.context.get(), execution_.backend()));
        if (!in.buffer) throw std::runtime_error("zonos3: cannot allocate reference encoder stage inputs");
        std::vector<int32_t> positions(static_cast<size_t>(T));
        std::iota(positions.begin(), positions.end(), 0);
        std::vector<float> mask(static_cast<size_t>(T) * static_cast<size_t>(T), -std::numeric_limits<float>::infinity());
        for (int q = 0; q < T; ++q) {
            for (int k = std::max(0, q - context + 1); k <= q; ++k) mask[static_cast<size_t>(q) * static_cast<size_t>(T) + static_cast<size_t>(k)] = 0.0f;
        }
        ggml_backend_tensor_set(in.positions, positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(in.mask, mask.data(), 0, mask.size() * sizeof(float));
        return in;
    }

    // ---- encoder stages -----------------------------------------------------------------------------------------

    std::vector<float> resample(const std::vector<float> & audio) const {
        constexpr int orig = 735;
        constexpr int target = 736;
        constexpr int width = 17;
        constexpr int K = 2 * width + orig;
        const auto n = static_cast<int64_t>(audio.size());
        const int64_t count = (n * target + orig - 1) / orig;
        std::vector<float> out(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) {
            const int64_t block = i / target;
            const int64_t phase = i % target;
            const int64_t start = block * orig - width;
            const float * kernel = resample_kernel_.data() + static_cast<size_t>(phase) * K;
            float value = 0;
            for (int64_t k = std::max<int64_t>(0, -start), end = std::min<int64_t>(K, n - start); k < end; ++k) {
                value += kernel[k] * audio[static_cast<size_t>(start + k)];
            }
            out[static_cast<size_t>(i)] = value;
        }
        return out;
    }

    std::vector<float> acoustic_lane(const std::vector<float> & audio) const {
        Activations state;
        reset(state, 1, kLaneSamples, audio.data());
        run(state, [&](auto & ctx, auto * x) { return conv(ctx, x, "enc.in", 1, 1); });
        for (int b = 0; b < 2; ++b) {
            const std::string p = "enc.b" + std::to_string(b);
            for (int r = 0; r < 3; ++r) {
                const int dilation = r == 0 ? 1 : r == 1 ? 3 : 9;
                run(state, [&](auto & ctx, auto * x) { return residual_unit(ctx, x, p + ".res" + std::to_string(r), dilation); });
            }
            run(state, [&](auto & ctx, auto * x) { return conv(ctx, snake(ctx, x, p + ".snake"), p + ".conv", 1, b == 0 ? 2 : 4); });
        }
        for (size_t s = 0; s < stages_.size(); ++s) {
            const auto & hp = stages_[s];
            const std::string p = "stage." + std::to_string(s);
            run(state, [&](auto & ctx, auto * x) {
                // Patch down: out[ch * h + j, t] = in[ch, t * h + j].
                auto * g = ctx.ggml;
                const int64_t C = x->ne[0];
                const int64_t T = x->ne[1];
                const int h = hp.patch;
                x = ggml_reshape_3d(g, ggml_cont(g, x), C, h, T / h);
                x = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, x, 1, 0, 2, 3)), C * h, T / h);
                return round(ctx, linear(ctx, round(ctx, x), weight(p + ".in.weight")));
            });
            const auto inputs = stage_inputs(static_cast<int>(state.frames), hp.context);
            for (int l = 0; l < hp.n_layer; ++l) {
                run(state, [&](auto & ctx, auto * x) {
                    return moss(ctx, x, p + ".l" + std::to_string(l), hp, inputs.positions, inputs.mask, l == 0);
                });
            }
            run(state, [&](auto & ctx, auto * x) { return round(ctx, linear(ctx, round(ctx, x), weight(p + ".out.weight"))); });
        }
        if (state.frames != kLane || state.channels != kLatent) {
            throw std::runtime_error("zonos3: reference encoder acoustic lane produced an unexpected shape");
        }
        return host(state);
    }

    // encode: the semantic ConvNeXt encoder over pooled SSL features; otherwise the semantic head over RVQ-0's output.
    std::vector<float> semantic_module(const std::vector<float> & input, bool encode) const {
        Activations state;
        reset(state, encode ? kLatent : kSemantic, kLane, input.data());
        run(state, [&](auto & ctx, auto * x) {
            auto * g = ctx.ggml;
            if (encode) {
                // WNConv1d: the unbiased output rounds before the BF16 bias add.
                x = round(ctx, ggml_add(g, round(ctx, linear(ctx, round(ctx, x), weight("sem_enc.in.weight"))),
                                        weight("sem_enc.in.bias").tensor));
            }
            for (int b = 0; b < kSemanticDepth; ++b) {
                x = convnext(ctx, x, std::string(encode ? "sem_enc." : "sem_head.") + std::to_string(b), encode);
            }
            if (!encode) {
                x = round(ctx, ggml_add(g, round(ctx, linear(ctx, round(ctx, x), weight("sem_head.out.weight"))),
                                        weight("sem_head.out.bias").tensor));
            }
            return x;
        });
        return host(state);
    }

    // Average SSL frames into the 300 lane frames over windows of window_ratio (2) frames.
    static std::vector<float> pool_ssl(const std::vector<float> & ssl, int T) {
        constexpr int C = 1024;
        constexpr float window_ratio = 2.0f;
        std::vector<float> prefix(static_cast<size_t>(T + 1) * C, 0.0f);
        std::vector<float> pooled(static_cast<size_t>(kLane) * C);
        for (int t = 0; t < T; ++t) {
            for (int c = 0; c < C; ++c) {
                prefix[static_cast<size_t>(t + 1) * C + c] = prefix[static_cast<size_t>(t) * C + c] + ssl[static_cast<size_t>(t) * C + c];
            }
        }
        const float ratio = static_cast<float>(T) / kLane;
        const float half = 0.5f * window_ratio * ratio;
        for (int t = 0; t < kLane; ++t) {
            const float center = (static_cast<float>(t) + 0.5f) * ratio;
            const float left = std::clamp(center - half, 0.0f, static_cast<float>(T));
            const float right = std::clamp(center + half, 0.0f, static_cast<float>(T));
            const int li = static_cast<int>(std::floor(left));
            const int ri = static_cast<int>(std::floor(right));
            const float lf = left - static_cast<float>(li);
            const float rf = right - static_cast<float>(ri);
            for (int c = 0; c < C; ++c) {
                const float a = prefix[static_cast<size_t>(li) * C + c] + ssl[static_cast<size_t>(std::min(li, T - 1)) * C + c] * lf;
                const float b = prefix[static_cast<size_t>(ri) * C + c] + ssl[static_cast<size_t>(std::min(ri, T - 1)) * C + c] * rf;
                pooled[static_cast<size_t>(t) * C + c] = (b - a) / (right - left);
            }
        }
        return pooled;
    }

    // Cosine RVQ in F32: L2-normalized projection against the normalized codebook, first index on ties; the output
    // keeps the straight-through expression z_e + (c - z_e).
    static void quantize(const Quantizer & q, const std::vector<float> & input, std::vector<float> & quantized,
                         std::vector<int32_t> & ids) {
        const int C = q.input_dim;
        const int D = q.code_dim;
        const int T = static_cast<int>(input.size()) / C;
        quantized.resize(input.size());
        ids.resize(static_cast<size_t>(T));
        std::vector<float> projected(static_cast<size_t>(D));
        std::vector<float> normalized(static_cast<size_t>(D));
        std::vector<float> lookup(static_cast<size_t>(D));
        for (int t = 0; t < T; ++t) {
            const float * row = input.data() + static_cast<size_t>(t) * C;
            float norm = 0;
            for (int d = 0; d < D; ++d) {
                float value = 0;
                const float * w = q.in_weight.data() + static_cast<size_t>(d) * C;
                for (int c = 0; c < C; ++c) value += row[c] * w[c];
                value += q.in_bias[static_cast<size_t>(d)];
                projected[static_cast<size_t>(d)] = value;
                norm += value * value;
            }
            norm = std::max(std::sqrt(norm), 1e-12f);
            float encoding_squared = 0;
            for (int d = 0; d < D; ++d) {
                normalized[static_cast<size_t>(d)] = projected[static_cast<size_t>(d)] / norm;
                encoding_squared += normalized[static_cast<size_t>(d)] * normalized[static_cast<size_t>(d)];
            }
            float best = std::numeric_limits<float>::infinity();
            int id = 0;
            for (int k = 0; k < q.size; ++k) {
                float dot = 0;
                const float * code = q.normalized.data() + static_cast<size_t>(k) * D;
                for (int d = 0; d < D; ++d) dot += normalized[static_cast<size_t>(d)] * code[d];
                const float distance = (encoding_squared - 2.0f * dot) + q.norm_squared[static_cast<size_t>(k)];
                if (distance < best) {
                    best = distance;
                    id = k;
                }
            }
            ids[static_cast<size_t>(t)] = id;
            for (int d = 0; d < D; ++d) {
                const float ze = projected[static_cast<size_t>(d)];
                lookup[static_cast<size_t>(d)] = ze + (q.codebook[static_cast<size_t>(id) * D + static_cast<size_t>(d)] - ze);
            }
            float * out = quantized.data() + static_cast<size_t>(t) * C;
            for (int c = 0; c < C; ++c) {
                float value = 0;
                const float * w = q.out_weight.data() + static_cast<size_t>(c) * D;
                for (int d = 0; d < D; ++d) value += lookup[static_cast<size_t>(d)] * w[d];
                out[c] = value + q.out_bias[static_cast<size_t>(c)];
            }
        }
        if (!std::all_of(quantized.begin(), quantized.end(), [](float v) { return std::isfinite(v); })) {
            throw std::runtime_error("zonos3: reference encoder RVQ produced non-finite values");
        }
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    std::unique_ptr<Zonos3SslTeacher> teacher_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::unordered_map<std::string, TensorValue> weights_;
    std::unordered_map<std::string, ConvWeights> convs_;
    std::vector<StageParams> stages_;
    std::vector<Quantizer> rvq_;
    std::vector<float> resample_kernel_;  // [736][769]
    float ln_eps_ = 1e-5f;
    float convnext_eps_ = 1e-6f;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ping_pong_context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> ping_pong_buffer_{nullptr, ggml_backend_buffer_free};
    std::array<ggml_tensor *, 2> ping_pong_{};
};

Zonos3ReferenceEncoder::Zonos3ReferenceEncoder(std::shared_ptr<const assets::TensorSource> zac_encoder,
                                               std::shared_ptr<const assets::TensorSource> ssl_teacher,
                                               core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(zac_encoder), std::move(ssl_teacher), execution)) {}

Zonos3ReferenceEncoder::~Zonos3ReferenceEncoder() = default;

std::vector<int32_t> Zonos3ReferenceEncoder::encode(const std::vector<float> & mono_44k) const {
    return impl_->encode(mono_44k);
}

}  // namespace engine::models::zonos3
