#include "engine/community_models/kale_tts/runtime.h"

#include "engine/community_models/kale_tts/source_filter.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/sampling/noise.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <string>
#include <unordered_map>

namespace engine::models::kale_tts {
namespace {

namespace core = engine::core;

constexpr float kLeak = 0.2F;
constexpr size_t kGraphNodes = 1 << 17;

struct ContextDeleter {
    void operator()(ggml_context * c) const noexcept {
        if (c != nullptr) ggml_free(c);
    }
};

// One graph: inputs in their own context (kept on the backend), everything else allocated by gallocr.
struct Graph {
    ggml_backend_t backend = nullptr;
    std::unique_ptr<ggml_context, ContextDeleter> inputs;
    std::unique_ptr<ggml_context, ContextDeleter> nodes;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<std::pair<ggml_tensor *, std::vector<char>>> uploads;

    explicit Graph(ggml_backend_t b) : backend(b) {
        inputs.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
        nodes.reset(ggml_init({kGraphNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphNodes, false), nullptr, true}));
        if (!inputs || !nodes) throw std::runtime_error("kale-tts: failed to create graph contexts");
    }
    ~Graph() {
        if (allocator != nullptr) ggml_gallocr_free(allocator);
        if (input_buffer != nullptr) ggml_backend_buffer_free(input_buffer);
    }
    ggml_context * ctx() const { return nodes.get(); }

    ggml_tensor * f32(const std::vector<float> & values, int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor * t = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, ne0, ne1);
        ggml_set_input(t);
        std::vector<char> bytes(values.size() * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        uploads.emplace_back(t, std::move(bytes));
        return t;
    }
    ggml_tensor * i32(const std::vector<int32_t> & values) {
        ggml_tensor * t = ggml_new_tensor_1d(inputs.get(), GGML_TYPE_I32, static_cast<int64_t>(values.size()));
        ggml_set_input(t);
        std::vector<char> bytes(values.size() * sizeof(int32_t));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        uploads.emplace_back(t, std::move(bytes));
        return t;
    }

    void run(const std::vector<ggml_tensor *> & outputs, const char * label) {
        graph = ggml_new_graph_custom(ctx(), kGraphNodes, false);
        for (ggml_tensor * out : outputs) {
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
        }
        input_buffer = ggml_backend_alloc_ctx_tensors(inputs.get(), backend);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (input_buffer == nullptr || allocator == nullptr || !ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error(std::string("kale-tts: failed to allocate the ") + label + " graph");
        }
        core::validate_backend_graph_supported(backend, graph, label);
        for (auto & [tensor, bytes] : uploads) {
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
        }
        ggml_backend_graph_plan_t plan = core::create_backend_graph_plan_if_host(backend, graph);
        const auto status = core::compute_backend_graph(backend, graph, plan, label);
        core::free_backend_graph_plan(backend, plan);
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("kale-tts: the ") + label + " graph failed");
        }
    }

    static std::vector<float> read(ggml_tensor * t) {
        std::vector<float> out(static_cast<size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        return out;
    }
};

// Layers. Two layouts: "ct" = ne0 channels, ne1 time (rows of features: linear, norms, attention, LSTM) and
// "tc" = ne0 time, ne1 channels (convolutions, instance norm).
class Builder {
public:
    Builder(Graph & g, const std::unordered_map<std::string, ggml_tensor *> & w) : g_(g), c_(g.ctx()), w_(w) {}

    ggml_tensor * W(const std::string & name) const {
        const auto found = w_.find(name);
        if (found == w_.end()) throw std::runtime_error("kale-tts: missing tensor " + name);
        return found->second;
    }
    bool has(const std::string & name) const { return w_.count(name) != 0; }

    ggml_tensor * tr(ggml_tensor * x) const { return ggml_cont(c_, ggml_transpose(c_, x)); }

    ggml_tensor * linear(ggml_tensor * x, const std::string & name, bool bias = true) const {
        ggml_tensor * y = ggml_mul_mat(c_, W(name + ".weight"), x);
        return bias ? ggml_add(c_, y, W(name + ".bias")) : y;
    }

    ggml_tensor * layer_norm(ggml_tensor * x, const std::string & name, float eps, bool bias = true) const {
        ggml_tensor * y = ggml_mul(c_, ggml_norm(c_, x, eps), W(name + ".weight"));
        return bias ? ggml_add(c_, y, W(name + ".bias")) : y;
    }

    // per-channel vector [C] -> [1, C] for tc tensors
    ggml_tensor * column(ggml_tensor * v) const { return ggml_reshape_2d(c_, v, 1, ggml_nelements(v)); }

    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int stride, int pad, bool bias = true) const {
        ggml_tensor * k = W(name + ".weight");
        ggml_tensor * y = ggml_conv_1d(c_, k, x, stride, pad, 1);
        y = ggml_reshape_2d(c_, y, y->ne[0], y->ne[1]);
        return bias ? ggml_add(c_, y, column(W(name + ".bias"))) : y;
    }

    // x[t + offset] with zeros outside, tc layout
    ggml_tensor * shifted(ggml_tensor * x, int64_t offset) const {
        const int64_t T = x->ne[0];
        if (offset == 0) return x;
        const int64_t n = std::min<int64_t>(std::llabs(offset), T);
        ggml_tensor * zeros = ggml_scale(c_, ggml_cont(c_, ggml_view_2d(c_, x, n, x->ne[1], x->nb[1], 0)), 0.0F);
        if (n == T) return ggml_scale(c_, x, 0.0F);
        if (offset > 0) {
            ggml_tensor * body = ggml_view_2d(c_, x, T - n, x->ne[1], x->nb[1], static_cast<size_t>(n) * x->nb[0]);
            return ggml_concat(c_, ggml_cont(c_, body), ggml_cont(c_, zeros), 0);
        }
        ggml_tensor * body = ggml_view_2d(c_, x, T - n, x->ne[1], x->nb[1], 0);
        return ggml_concat(c_, ggml_cont(c_, zeros), ggml_cont(c_, body), 0);
    }

    // depthwise kernel tap k of a [C, 1, K] weight, as [1, C]
    ggml_tensor * tap(ggml_tensor * k, int64_t i) const {
        return ggml_cont(c_, ggml_view_2d(c_, k, 1, k->ne[2], k->nb[2], static_cast<size_t>(i) * k->nb[0]));
    }

    ggml_tensor * depthwise(ggml_tensor * x, const std::string & name) const {
        ggml_tensor * k = W(name + ".weight");
        const int64_t K = k->ne[0];
        ggml_tensor * y = nullptr;
        for (int64_t i = 0; i < K; ++i) {
            ggml_tensor * term = ggml_mul(c_, shifted(x, i - K / 2), tap(k, i));
            y = y == nullptr ? term : ggml_add(c_, y, term);
        }
        return ggml_add(c_, y, column(W(name + ".bias")));
    }

    // depthwise ConvTranspose1d(k=3, stride=2, padding=1, output_padding=1): T -> 2T
    ggml_tensor * pool(ggml_tensor * x, const std::string & name) const {
        ggml_tensor * k = W(name + ".weight");
        ggml_tensor * even = ggml_mul(c_, x, tap(k, 1));
        ggml_tensor * odd = ggml_add(c_, ggml_mul(c_, x, tap(k, 2)), ggml_mul(c_, shifted(x, 1), tap(k, 0)));
        ggml_tensor * both = ggml_concat(c_, ggml_reshape_3d(c_, ggml_cont(c_, even), 1, x->ne[0], x->ne[1]),
                                         ggml_reshape_3d(c_, ggml_cont(c_, odd), 1, x->ne[0], x->ne[1]), 0);
        ggml_tensor * y = ggml_reshape_2d(c_, ggml_cont(c_, both), 2 * x->ne[0], x->ne[1]);
        return ggml_add(c_, y, column(W(name + ".bias")));
    }

    ggml_tensor * upsample(ggml_tensor * x) const {
        ggml_tensor * x3 = ggml_reshape_3d(c_, ggml_cont(c_, x), 1, x->ne[0], x->ne[1]);
        return ggml_reshape_2d(c_, ggml_cont(c_, ggml_concat(c_, x3, x3, 0)), 2 * x->ne[0], x->ne[1]);
    }

    // instance norm over time of a tc tensor, from sums: Metal's fused norm kernel mis-handles these rows (time lengths
    // that are not a multiple of 4, followed by per-channel scales)
    ggml_tensor * instance_norm(ggml_tensor * x, float eps) const {
        const float inv = 1.0F / static_cast<float>(x->ne[0]);
        ggml_tensor * centred = ggml_sub(c_, x, ggml_scale(c_, ggml_sum_rows(c_, x), inv));
        ggml_tensor * var = ggml_scale_bias(c_, ggml_sum_rows(c_, ggml_sqr(c_, centred)), inv, eps);
        return ggml_div(c_, centred, ggml_sqrt(c_, var));
    }

    // AdaIN on tc: instance norm (with the students' learned scale / shift) then (1 + gamma) x + beta from the style
    ggml_tensor * adain(ggml_tensor * x, ggml_tensor * style, const std::string & name) const {
        const int64_t C = x->ne[1];
        ggml_tensor * h = linear(style, name + ".fc");
        ggml_tensor * gamma = column(ggml_cont(c_, ggml_view_1d(c_, h, C, 0)));
        ggml_tensor * beta = column(ggml_cont(c_, ggml_view_1d(c_, h, C, static_cast<size_t>(C) * sizeof(float))));
        ggml_tensor * y = instance_norm(x, 1e-5F);
        if (has(name + ".norm.weight")) {
            y = ggml_add(c_, ggml_mul(c_, y, column(W(name + ".norm.weight"))), column(W(name + ".norm.bias")));
        }
        return ggml_add(c_, ggml_add(c_, y, ggml_mul(c_, y, gamma)), beta);
    }

    ggml_tensor * adain_block(ggml_tensor * x, ggml_tensor * style, const std::string & name, bool up) const {
        ggml_tensor * r = ggml_leaky_relu(c_, adain(x, style, name + ".norm1"), kLeak, false);
        if (up) r = pool(r, name + ".pool");
        r = conv(r, name + ".conv1", 1, 1);
        r = ggml_leaky_relu(c_, adain(r, style, name + ".norm2"), kLeak, false);
        r = conv(r, name + ".conv2", 1, 1);
        ggml_tensor * s = up ? upsample(x) : x;
        if (has(name + ".conv1x1.weight")) s = conv(s, name + ".conv1x1", 1, 0, false);
        return ggml_scale(c_, ggml_add(c_, r, s), 1.0F / std::sqrt(2.0F));
    }

    // AdaLayerNorm on ct: layer norm over channels, then (1 + gamma) x + beta
    ggml_tensor * ada_layer_norm(ggml_tensor * x, ggml_tensor * style, const std::string & name) const {
        const int64_t C = x->ne[0];
        ggml_tensor * h = linear(style, name + ".fc");
        ggml_tensor * gamma = ggml_view_1d(c_, h, C, 0);
        ggml_tensor * beta = ggml_view_1d(c_, h, C, static_cast<size_t>(C) * sizeof(float));
        ggml_tensor * y = ggml_norm(c_, x, 1e-5F);
        return ggml_add(c_, ggml_add(c_, y, ggml_mul(c_, y, gamma)), beta);
    }

    // bidirectional LSTM on ct: [In, T] -> [2H, T], PyTorch gate order i, f, g, o
    ggml_tensor * bilstm(ggml_tensor * x, const std::string & name, ggml_tensor * zeros) const {
        std::vector<ggml_tensor *> out;
        for (const std::string suffix : {"", "_reverse"}) {
            ggml_tensor * wh = W(name + ".weight_hh_l0" + suffix);
            const int64_t H = wh->ne[0];
            ggml_tensor * xp = ggml_add(c_, ggml_add(c_, ggml_mul_mat(c_, W(name + ".weight_ih_l0" + suffix), x), W(name + ".bias_ih_l0" + suffix)),
                                        W(name + ".bias_hh_l0" + suffix));
            const int64_t T = x->ne[1];
            ggml_tensor * h = ggml_view_2d(c_, zeros, H, 1, zeros->nb[1], 0);
            ggml_tensor * cell = h;
            std::vector<ggml_tensor *> steps(static_cast<size_t>(T));
            const bool reverse = !std::string(suffix).empty();
            for (int64_t s = 0; s < T; ++s) {
                const int64_t t = reverse ? T - 1 - s : s;
                ggml_tensor * gates = ggml_add(c_, ggml_view_2d(c_, xp, 4 * H, 1, xp->nb[1], static_cast<size_t>(t) * xp->nb[1]), ggml_mul_mat(c_, wh, h));
                auto part = [&](int64_t k) { return ggml_view_2d(c_, gates, H, 1, gates->nb[1], static_cast<size_t>(k * H) * sizeof(float)); };
                ggml_tensor * i = ggml_sigmoid(c_, part(0));
                ggml_tensor * f = ggml_sigmoid(c_, part(1));
                ggml_tensor * z = ggml_tanh(c_, part(2));
                ggml_tensor * o = ggml_sigmoid(c_, part(3));
                cell = ggml_add(c_, ggml_mul(c_, f, cell), ggml_mul(c_, i, z));
                h = ggml_mul(c_, o, ggml_tanh(c_, cell));
                steps[static_cast<size_t>(t)] = h;
            }
            while (steps.size() > 1) {
                std::vector<ggml_tensor *> merged;
                for (size_t k = 0; k < steps.size(); k += 2) {
                    merged.push_back(k + 1 < steps.size() ? ggml_concat(c_, steps[k], steps[k + 1], 1) : steps[k]);
                }
                steps.swap(merged);
            }
            out.push_back(steps[0]);
        }
        return ggml_concat(c_, out[0], out[1], 0);
    }

    // multi-head attention on ct [C, T]; mask [T, T] or null
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t heads, ggml_tensor * mask, ggml_tensor * positions,
                            float rope_theta) const {
        const int64_t C = q->ne[0];
        const int64_t T = q->ne[1];
        const int64_t D = C / heads;
        q = ggml_reshape_3d(c_, ggml_cont(c_, q), D, heads, T);
        k = ggml_reshape_3d(c_, ggml_cont(c_, k), D, heads, T);
        v = ggml_reshape_3d(c_, ggml_cont(c_, v), D, heads, T);
        if (positions != nullptr) {
            q = ggml_rope_ext(c_, q, positions, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
            k = ggml_rope_ext(c_, k, positions, nullptr, static_cast<int>(D), GGML_ROPE_TYPE_NEOX, 0, rope_theta, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F);
        }
        ggml_tensor * qh = ggml_permute(c_, q, 0, 2, 1, 3);                    // [D, T, H]
        ggml_tensor * kh = ggml_cont(c_, ggml_permute(c_, k, 0, 2, 1, 3));
        ggml_tensor * scores = ggml_scale(c_, ggml_mul_mat(c_, kh, qh), 1.0F / std::sqrt(static_cast<float>(D)));   // [Tk, Tq, H]
        if (mask != nullptr) scores = ggml_add(c_, scores, mask);
        scores = ggml_soft_max(c_, scores);
        ggml_tensor * vh = ggml_cont(c_, ggml_permute(c_, v, 1, 2, 0, 3));     // [T, D, H]
        ggml_tensor * out = ggml_mul_mat(c_, vh, scores);                        // [D, Tq, H]
        out = ggml_cont(c_, ggml_permute(c_, out, 0, 2, 1, 3));                  // [D, H, T]
        return ggml_reshape_2d(c_, out, C, T);
    }

    ggml_context * c() const { return c_; }

private:
    Graph & g_;
    ggml_context * c_;
    const std::unordered_map<std::string, ggml_tensor *> & w_;
};

}  // namespace

struct KaleRuntime::State {
    // The text side and the pitch / loudness heads are small and sequential (LSTMs over tokens and frames): they run
    // on the CPU backend, where their duration rounding matches the reference graphs. The decoder graph runs on the
    // session's backend.
    struct Device {
        ggml_backend_t backend = nullptr;
        std::shared_ptr<core::BackendWeightStore> store;
        std::unordered_map<std::string, ggml_tensor *> weights;
    };
    std::shared_ptr<const KaleAssets> assets;
    Device cpu, main;
    bool same = true;

    void load(Device & d, core::BackendConfig config) {
        d.backend = core::init_backend(config);
        core::set_backend_threads(d.backend, config.threads);
        const auto & source = *assets->weights;
        const auto tensors = source.tensors();
        d.store = std::make_shared<core::BackendWeightStore>(d.backend, core::backend_type(d.backend), "kale_tts.weights",
                                                             (tensors.size() + 8) * ggml_tensor_overhead());
        for (const auto & t : tensors) {
            d.weights.emplace(t.name, d.store->load_tensor(source, t.name, assets::TensorStorageType::F32, t.shape).tensor);
        }
        d.store->upload();
    }

    ~State() {
        for (Device * d : {&main, &cpu}) {
            d->store.reset();
            if (d->backend != nullptr && (d == &cpu || !same)) ggml_backend_free(d->backend);
        }
    }
};

KaleRuntime::KaleRuntime(std::shared_ptr<const KaleAssets> assets, core::BackendConfig backend_config) : state_(std::make_unique<State>()) {
    State & s = *state_;
    s.assets = std::move(assets);
    backend_config.threads = std::max(1, backend_config.threads);
    core::BackendConfig cpu_config = backend_config;
    cpu_config.type = core::BackendType::Cpu;
    cpu_config.impl.clear();
    s.load(s.cpu, cpu_config);
    s.same = backend_config.type == core::BackendType::Cpu;
    if (s.same) {
        s.main.backend = s.cpu.backend;
        s.main.weights = s.cpu.weights;
    } else {
        s.load(s.main, backend_config);
    }
}

KaleRuntime::~KaleRuntime() = default;

std::vector<float> KaleRuntime::synthesize(const KaleRequest & r) {
    State & s = *state_;
    const KaleConfig & c = s.assets->config;
    if (r.voice == nullptr || r.ids.empty() || static_cast<int64_t>(r.ids.size()) > c.max_tokens) {
        throw std::runtime_error("kale-tts: a sentence needs between 1 and " + std::to_string(c.max_tokens) + " phonemes");
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<int32_t> ids = {0};
    ids.insert(ids.end(), r.ids.begin(), r.ids.end());
    ids.push_back(0);
    const auto T = static_cast<int64_t>(ids.size());
    std::vector<int32_t> positions(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    std::vector<float> style;
    if (c.ettin) {
        const int64_t row = std::min<int64_t>(static_cast<int64_t>(r.ids.size()), r.voice->style_rows) - 1;
        style.assign(r.voice->style.begin() + row * 256 + 128, r.voice->style.begin() + row * 256 + 256);
    } else {
        style = r.voice->prosody;
    }
    const int64_t hd = c.hidden_dim;
    const int64_t width = std::max<int64_t>(hd, 512);
    const std::vector<float> zero_values(static_cast<size_t>(width), 0.0F);

    // 1. text side, durations, text encoder
    std::vector<float> durations, d_values, t_en_values;
    {
        Graph g(s.cpu.backend);
        Builder b(g, s.cpu.weights);
        ggml_context * x = g.ctx();
        ggml_tensor * tokens = g.i32(ids);
        ggml_tensor * pos = g.i32(positions);
        ggml_tensor * sv = g.f32(style, c.style_dim);
        ggml_tensor * zeros = g.f32(zero_values, width);
        ggml_tensor * d_en = nullptr;
        if (!c.ettin) {
            ggml_tensor * e = ggml_add(x, ggml_get_rows(x, b.W("bert.embeddings.word_embeddings.weight"), tokens),
                                       ggml_get_rows(x, b.W("bert.embeddings.position_embeddings.weight"), pos));
            ggml_tensor * type0 = ggml_view_1d(x, b.W("bert.embeddings.token_type_embeddings.weight"), c.albert_embedding, 0);
            e = b.layer_norm(ggml_add(x, e, type0), "bert.embeddings.LayerNorm", 1e-12F);
            ggml_tensor * h = b.linear(e, "bert.encoder.embedding_hidden_mapping_in");
            const std::string L = "bert.encoder.albert_layer_groups.0.albert_layers.0.";
            for (int64_t layer = 0; layer < c.albert_layers; ++layer) {
                ggml_tensor * a = b.attention(b.linear(h, L + "attention.query"), b.linear(h, L + "attention.key"),
                                              b.linear(h, L + "attention.value"), c.albert_heads, nullptr, nullptr, 0.0F);
                ggml_tensor * attended = b.layer_norm(ggml_add(x, b.linear(a, L + "attention.dense"), h), L + "attention.LayerNorm", 1e-12F);
                ggml_tensor * f = b.linear(ggml_gelu(x, b.linear(attended, L + "ffn")), L + "ffn_output");
                h = b.layer_norm(ggml_add(x, f, attended), L + "full_layer_layer_norm", 1e-12F);
            }
            d_en = b.linear(h, "bert_encoder");
        } else {
            const auto S = static_cast<int64_t>(r.ettin_ids.size());
            std::vector<int32_t> spos(static_cast<size_t>(S));
            std::vector<float> local(static_cast<size_t>(S * S));
            for (int64_t i = 0; i < S; ++i) {
                spos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
                for (int64_t j = 0; j < S; ++j) {
                    local[static_cast<size_t>(i * S + j)] = std::llabs(i - j) <= c.ettin_local_window / 2 ? 0.0F : -INFINITY;
                }
            }
            ggml_tensor * sub = g.i32(r.ettin_ids);
            ggml_tensor * sub_pos = g.i32(spos);
            ggml_tensor * mask = g.f32(local, S, S);
            ggml_tensor * h = b.layer_norm(ggml_get_rows(x, b.W("ettin.embeddings.tok_embeddings.weight"), sub), "ettin.embeddings.norm", 1e-5F, false);
            ggml_tensor * previous = h;
            const int64_t E = c.ettin_hidden;
            for (int64_t layer = 0; layer < c.ettin_layers; ++layer) {
                const std::string L = "ettin.layers." + std::to_string(layer) + ".";
                previous = h;
                ggml_tensor * n = layer == 0 ? h : b.layer_norm(h, L + "attn_norm", 1e-5F, false);
                ggml_tensor * qkv = b.linear(n, L + "attn.Wqkv", false);
                auto part = [&](int64_t k) { return ggml_view_2d(x, qkv, E, S, qkv->nb[1], static_cast<size_t>(k * E) * sizeof(float)); };
                const bool global = layer % c.ettin_global_every == 0;
                ggml_tensor * a = b.attention(part(0), part(1), part(2), c.ettin_heads, global ? nullptr : mask, sub_pos, c.ettin_rope_theta);
                h = ggml_add(x, h, b.linear(a, L + "attn.Wo", false));
                ggml_tensor * wi = b.linear(b.layer_norm(h, L + "mlp_norm", 1e-5F, false), L + "mlp.Wi", false);
                const int64_t I = c.ettin_intermediate;
                ggml_tensor * input = ggml_cont(x, ggml_view_2d(x, wi, I, S, wi->nb[1], 0));
                ggml_tensor * gate = ggml_cont(x, ggml_view_2d(x, wi, I, S, wi->nb[1], static_cast<size_t>(I) * sizeof(float)));
                h = ggml_add(x, h, b.linear(ggml_mul(x, ggml_gelu_erf(x, input), gate), L + "mlp.Wo", false));
            }
            ggml_tensor * words = ggml_scale(x, ggml_add(x, h, previous), 0.5F);                      // [E, S]
            ggml_tensor * wavg = g.f32(r.ettin_wavg, S, r.ettin_words);
            ggml_tensor * wfeat = ggml_mul_mat(x, b.tr(words), wavg);                                  // [E, W]
            std::vector<int32_t> widx = {r.word_index.front()};
            widx.insert(widx.end(), r.word_index.begin(), r.word_index.end());
            widx.push_back(r.word_index.back());
            ggml_tensor * word_of = g.i32(widx);
            ggml_tensor * st = ggml_get_rows(x, b.W("student.emb.weight"), tokens);                   // [Cs, T]
            for (int64_t k = 0; k < c.student_conv_blocks; ++k) {
                const std::string L = "student.convs." + std::to_string(k) + ".";
                ggml_tensor * y = b.tr(b.conv(b.tr(st), L + "conv", 1, 2));
                st = ggml_add(x, st, ggml_gelu_erf(x, b.layer_norm(y, L + "norm", 1e-5F)));
            }
            ggml_tensor * w = b.linear(ggml_gelu_erf(x, b.linear(ggml_get_rows(x, wfeat, word_of), "student.wproj.0")), "student.wproj.2");
            st = ggml_add(x, ggml_add(x, st, w), ggml_get_rows(x, b.W("student.pos.weight"), pos));
            const int64_t Cs = c.student_channels;
            for (int64_t k = 0; k < c.student_transformer_blocks; ++k) {
                const std::string L = "student.tf." + std::to_string(k) + ".";
                ggml_tensor * qkv = b.linear(b.layer_norm(st, L + "n1", 1e-5F), L + "qkv");
                auto part = [&](int64_t j) { return ggml_view_2d(x, qkv, Cs, T, qkv->nb[1], static_cast<size_t>(j * Cs) * sizeof(float)); };
                st = ggml_add(x, st, b.linear(b.attention(part(0), part(1), part(2), c.student_heads, nullptr, nullptr, 0.0F), L + "o"));
                ggml_tensor * f = b.linear(ggml_gelu_erf(x, b.linear(b.layer_norm(st, L + "n2", 1e-5F), L + "ff.0")), L + "ff.2");
                st = ggml_add(x, st, f);
            }
            d_en = b.linear(st, "student.out");
        }
        // duration encoder: [LSTM, AdaLayerNorm] x n_layer, the voice's style next to every step
        ggml_tensor * sx = ggml_repeat(x, ggml_reshape_2d(x, sv, c.style_dim, 1), ggml_new_tensor_2d(x, GGML_TYPE_F32, c.style_dim, T));
        ggml_tensor * dx = ggml_concat(x, d_en, sx, 0);
        for (int64_t layer = 0; layer < c.n_layer; ++layer) {
            dx = b.bilstm(dx, "predictor.text_encoder.lstms." + std::to_string(2 * layer), zeros);
            dx = b.ada_layer_norm(dx, sv, "predictor.text_encoder.lstms." + std::to_string(2 * layer + 1));
            dx = ggml_concat(x, dx, sx, 0);
        }
        ggml_tensor * logits = b.linear(b.bilstm(dx, "predictor.lstm", zeros), "predictor.duration_proj.linear_layer");
        ggml_tensor * dur = ggml_sum_rows(x, ggml_sigmoid(x, logits));                          // [1, T]
        // text encoder
        ggml_tensor * te = ggml_get_rows(x, b.W("text_encoder.embedding.weight"), tokens);
        for (int64_t k = 0; k < c.n_layer; ++k) {
            const std::string L = "text_encoder.cnn." + std::to_string(k) + ".";
            te = b.tr(b.conv(b.tr(te), L + "0", 1, 2));
            te = ggml_leaky_relu(x, b.layer_norm(te, L + "1", 1e-5F), kLeak, false);
        }
        te = b.bilstm(te, "text_encoder.lstm", zeros);
        g.run({dur, dx, te}, "kale_tts.text");
        durations = Graph::read(dur);
        d_values = Graph::read(dx);
        t_en_values = Graph::read(te);
    }

    // durations -> the token of every frame
    std::vector<int32_t> frame_token;
    for (int64_t t = 0; t < T; ++t) {
        const auto n = std::max<int64_t>(1, static_cast<int64_t>(std::nearbyint(durations[static_cast<size_t>(t)] / r.speed)));
        frame_token.insert(frame_token.end(), static_cast<size_t>(n), static_cast<int32_t>(t));
    }
    const auto F = static_cast<int64_t>(frame_token.size());

    // 2. pitch and loudness
    std::vector<float> f0, loud;
    {
        Graph g(s.cpu.backend);
        Builder b(g, s.cpu.weights);
        ggml_context * x = g.ctx();
        ggml_tensor * d = g.f32(d_values, hd + c.style_dim, T);
        ggml_tensor * idx = g.i32(frame_token);
        ggml_tensor * sv = g.f32(style, c.style_dim);
        ggml_tensor * zeros = g.f32(zero_values, width);
        ggml_tensor * shared = b.tr(b.bilstm(ggml_get_rows(x, d, idx), "predictor.shared", zeros));   // [F, hd]
        auto branch = [&](const std::string & name) {
            ggml_tensor * y = b.adain_block(shared, sv, "predictor." + name + ".0", false);
            y = b.adain_block(y, sv, "predictor." + name + ".1", true);
            y = b.adain_block(y, sv, "predictor." + name + ".2", false);
            return b.conv(y, "predictor." + name + "_proj", 1, 0);
        };
        ggml_tensor * pitch = branch("F0");
        ggml_tensor * energy = branch("N");
        g.run({pitch, energy}, "kale_tts.prosody");
        f0 = Graph::read(pitch);
        loud = Graph::read(energy);
    }

    // end clamp: in the last frames of the sentence, voiced pitch is held at or above the voice's floor; then pitch below
    // the voice's unvoicing floor is unvoiced
    {
        const KaleVoice & v = *r.voice;
        int64_t last = -1;
        int64_t voiced = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(f0.size()); ++i) {
            if (f0[static_cast<size_t>(i)] > v.clamp_threshold) {
                last = i;
                ++voiced;
            }
        }
        if (voiced >= 5) {
            for (int64_t i = 0; i < static_cast<int64_t>(f0.size()); ++i) {
                float & p = f0[static_cast<size_t>(i)];
                if (p > v.clamp_threshold && static_cast<float>(i) >= static_cast<float>(last) - v.clamp_tail && i <= last) {
                    p = std::max(p, v.pitch_floor);
                }
            }
        }
        for (float & p : f0) {
            if (p < v.unvoice_hz) p = 0.0F;
        }
    }

    // 3. decoder trunk and the head's network part
    std::vector<float> head, burst;
    {
        Graph g(s.main.backend);
        Builder b(g, s.main.weights);
        ggml_context * x = g.ctx();
        ggml_tensor * te = g.f32(t_en_values, hd, T);
        ggml_tensor * idx = g.i32(frame_token);
        ggml_tensor * pitch = g.f32(f0, static_cast<int64_t>(f0.size()), 1);
        ggml_tensor * energy = g.f32(loud, static_cast<int64_t>(loud.size()), 1);
        ggml_tensor * sound = g.f32(r.voice->sound, c.voice_sound_dim);
        ggml_tensor * asr = b.tr(ggml_get_rows(x, te, idx));                                     // [F, hd]
        ggml_tensor * f0c = b.conv(pitch, "decoder.F0_conv", 2, 1);                              // [F, 1]
        ggml_tensor * nc = b.conv(energy, "decoder.N_conv", 2, 1);
        ggml_tensor * h = b.adain_block(ggml_concat(x, ggml_concat(x, asr, f0c, 1), nc, 1), sound, "decoder.encode", false);
        ggml_tensor * res = b.conv(asr, "decoder.asr_res.0", 1, 0);
        for (int k = 0; k < 4; ++k) {
            h = ggml_concat(x, ggml_concat(x, ggml_concat(x, h, res, 1), f0c, 1), nc, 1);
            h = b.adain_block(h, sound, "decoder.decode." + std::to_string(k), k == 3);
        }
        // head network: input conv, ConvNeXt blocks, norm, the two output layers
        ggml_tensor * z = b.layer_norm(b.tr(b.conv(h, "decoder.head.inp", 1, 3)), "decoder.head.norm0", 1e-5F);   // [D, 2F]
        for (int64_t k = 0; k < c.head_blocks; ++k) {
            const std::string L = "decoder.head.blocks." + std::to_string(k) + ".";
            ggml_tensor * y = b.layer_norm(b.tr(b.depthwise(b.tr(z), L + "dw")), L + "norm", 1e-5F);
            y = b.linear(ggml_gelu_erf(x, b.linear(y, L + "pw1")), L + "pw2");
            z = ggml_add(x, z, ggml_mul(x, y, b.W(L + "gamma")));
        }
        ggml_tensor * hn = b.layer_norm(z, "decoder.head.norm1", 1e-5F);
        ggml_tensor * out = b.linear(hn, "decoder.head.out");
        ggml_tensor * out_b = b.linear(hn, "decoder.head.out_b");
        g.run({out, out_b}, "kale_tts.decoder");
        head = Graph::read(out);
        burst = Graph::read(out_b);
    }

    // 4. the waveform: sine source + noise through the network's filter
    const auto n = static_cast<int64_t>(f0.size());
    SourceFilterConfig sf;
    sf.harmonics = c.head_harmonics;
    sf.mod = c.head_mod;
    sf.points = c.head_points;
    sf.burst_points = c.head_burst_points;
    const std::vector<float> noise = r.noise != nullptr ? *r.noise : engine::sampling::generate_normal_noise(static_cast<size_t>(n * sf.f0_hop), r.seed);
    std::vector<float> wave = source_filter(sf, head, burst, f0, noise);
    engine::debug::timing_log_scalar("kale_tts.synthesize_ms", engine::debug::elapsed_ms(start));
    return wave;
}

}  // namespace engine::models::kale_tts
