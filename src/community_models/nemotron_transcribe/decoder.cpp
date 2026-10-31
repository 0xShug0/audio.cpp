#include "engine/community_models/nemotron_transcribe/decoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/disk_expert_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/kv_cache.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace engine::models::nemotron_transcribe {
namespace {

constexpr size_t kGraphNodes = 8192;
constexpr size_t kGraphArenaBytes = 16 * 1024 * 1024;
constexpr size_t kWeightArenaBytes = 4 * 1024 * 1024;

struct ContextDeleter {
    void operator()(ggml_context * value) const { ggml_free(value); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); }
};
struct BufferDeleter {
    void operator()(ggml_backend_buffer_t value) const { ggml_backend_buffer_free(value); }
};
using Context = std::unique_ptr<ggml_context, ContextDeleter>;
using Buffer = std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, BufferDeleter>;
using Allocator = std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, AllocatorDeleter>;

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

int64_t checked_elements(std::initializer_list<int64_t> dimensions) {
    const auto limit = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
        std::numeric_limits<size_t>::max()) / sizeof(float);
    uint64_t elements = 1;
    for (auto dimension : dimensions) {
        if (dimension <= 0 || static_cast<uint64_t>(dimension) > limit / elements) {
            throw std::runtime_error("Nemotron-H decoder dimensions exceed addressable tensor sizes");
        }
        elements *= static_cast<uint64_t>(dimension);
    }
    return static_cast<int64_t>(elements);
}

int64_t expert_row_width(int64_t logical, int64_t alignment) {
    const auto physical = ((logical + alignment - 1) / alignment) * alignment;
    if (physical > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("Nemotron-H padded expert row exceeds GGML kernel limits");
    return physical;
}

void validate_config(const DecoderConfig & c) {
    // Budget at most 64 graph tensors per block, leaving room for inputs/head.
    if (c.hidden_size <= 0 || c.vocab_size <= 0 || c.vocab_size > std::numeric_limits<int32_t>::max() ||
        c.block_pattern.empty() || c.block_pattern.size() > (kGraphNodes - 64) / 64 ||
        c.attention_heads <= 0 || c.attention_kv_heads <= 0 || c.attention_head_dim <= 0 ||
        c.attention_heads % c.attention_kv_heads != 0 || c.mamba_heads <= 0 || c.mamba_head_dim <= 0 ||
        c.mamba_state_size <= 0 || c.mamba_groups <= 0 || c.mamba_heads % c.mamba_groups != 0 ||
        c.conv_kernel < 2 || c.experts <= 0 || c.experts > std::numeric_limits<int>::max() ||
        c.experts_per_token <= 0 || c.experts_per_token > c.experts ||
        c.expert_width <= 0 || c.shared_expert_width <= 0 || c.max_context <= 0 ||
        c.prefill_batch_size <= 0 || c.prefill_batch_size > c.max_context ||
        (c.expert_row_alignment != 1 && c.expert_row_alignment != 256) ||
        !std::isfinite(c.rms_norm_eps) || c.rms_norm_eps <= 0.0F ||
        !std::isfinite(c.routed_scaling_factor) || c.routed_scaling_factor <= 0.0F ||
        c.block_pattern.find_first_not_of("MEA") != std::string::npos) {
        throw std::runtime_error("Nemotron-H invalid decoder configuration");
    }
    // GGML's SSM kernels use int-sized axes. Also reject oversized tensor
    // products before channel arithmetic, ggml_new_tensor or cache rounding.
    const auto max_axis = std::numeric_limits<int32_t>::max();
    for (auto axis : {c.hidden_size, c.vocab_size, c.attention_heads, c.attention_kv_heads,
             c.attention_head_dim, c.mamba_heads, c.mamba_head_dim, c.mamba_state_size,
             c.mamba_groups, c.conv_kernel, c.experts, c.experts_per_token,
             c.expert_width, c.shared_expert_width, c.max_context, c.prefill_batch_size}) {
        if (axis > max_axis) throw std::runtime_error("Nemotron-H decoder axis exceeds GGML kernel limits");
    }
    const auto inner = checked_elements({c.mamba_heads, c.mamba_head_dim});
    const auto grouped_state = checked_elements({c.mamba_groups, c.mamba_state_size});
    const auto q_width = checked_elements({c.attention_heads, c.attention_head_dim});
    const auto kv_width = checked_elements({c.attention_kv_heads, c.attention_head_dim});
    const auto expert_input = expert_row_width(c.hidden_size, c.expert_row_alignment);
    const auto expert_hidden = expert_row_width(c.expert_width, c.expert_row_alignment);
    if (inner > max_axis || grouped_state > (max_axis - inner) / 2 || q_width > max_axis || kv_width > max_axis) {
        throw std::runtime_error("Nemotron-H decoder projected axis exceeds GGML kernel limits");
    }
    const auto conv_dim = inner + 2 * grouped_state;
    if (inner > max_axis - conv_dim || c.mamba_heads > max_axis - inner - conv_dim ||
        c.conv_kernel - 1 > max_axis - c.prefill_batch_size) {
        throw std::runtime_error("Nemotron-H decoder projected axis exceeds GGML kernel limits");
    }
    for (auto elements : {
             checked_elements({c.vocab_size, c.hidden_size}),
             checked_elements({inner + conv_dim + c.mamba_heads, c.hidden_size}),
             checked_elements({inner + conv_dim + c.mamba_heads, c.prefill_batch_size}),
             checked_elements({inner, c.hidden_size}), checked_elements({conv_dim, c.conv_kernel}),
             checked_elements({conv_dim, c.conv_kernel - 1 + c.prefill_batch_size}),
             checked_elements({inner, c.mamba_state_size + c.prefill_batch_size}),
             checked_elements({q_width, c.hidden_size}), checked_elements({kv_width, c.max_context}),
             checked_elements({c.max_context, c.prefill_batch_size}),
             checked_elements({c.experts, c.expert_width, c.hidden_size}),
             checked_elements({c.experts, c.expert_width, expert_input}),
             checked_elements({c.experts, c.hidden_size, expert_hidden}),
             checked_elements({c.experts, c.prefill_batch_size}),
             checked_elements({c.shared_expert_width, c.hidden_size}),
             checked_elements({c.experts_per_token, std::max(c.expert_width, c.hidden_size), c.prefill_batch_size})}) {
        (void)elements;
    }
    checked_elements({c.experts_per_token, std::max(expert_input, expert_hidden), c.prefill_batch_size});
}

struct LayerWeights {
    ggml_tensor * norm = nullptr;
    ggml_tensor * ssm_in = nullptr;
    ggml_tensor * conv = nullptr;
    ggml_tensor * conv_bias = nullptr;
    ggml_tensor * dt_bias = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * d = nullptr;
    ggml_tensor * ssm_norm = nullptr;
    ggml_tensor * ssm_out = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * attention_out = nullptr;
    ggml_tensor * router = nullptr;
    ggml_tensor * router_bias = nullptr;
    ggml_tensor * experts_up = nullptr;
    ggml_tensor * experts_down = nullptr;
    ggml_tensor * shared_up = nullptr;
    ggml_tensor * shared_down = nullptr;
};

struct Weights {
    ggml_tensor * embedding = nullptr;
    ggml_tensor * final_norm = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<LayerWeights> layers;
    core::DiskExpertWeightStore * disk_store = nullptr;
    std::vector<size_t> disk_groups;
};

// Caches have their own allocation so all bounded prefill/decode graphs share
// the same state without transferring recurrent tensors through host memory.
class DecoderState {
public:
    DecoderState(const DecoderConfig & c, core::ExecutionContext & execution, int64_t capacity)
        : capacity_(capacity) {
        context_.reset(ggml_init({kWeightArenaBytes, nullptr, true}));
        if (!context_) throw std::runtime_error("Nemotron-H cache context allocation failed");
        auto * g = context_.get();
        const int64_t inner = c.mamba_heads * c.mamba_head_dim;
        const int64_t conv_dim = inner + 2 * c.mamba_groups * c.mamba_state_size;
        std::vector<core::TensorValue> keys;
        std::vector<core::TensorValue> values;
        conv_states_.resize(c.block_pattern.size(), nullptr);
        ssm_states_.resize(c.block_pattern.size(), nullptr);
        key_states_.resize(c.block_pattern.size(), nullptr);
        value_states_.resize(c.block_pattern.size(), nullptr);
        for (size_t i = 0; i < c.block_pattern.size(); ++i) {
            if (c.block_pattern[i] == 'M') {
                conv_states_[i] = ggml_new_tensor_3d(g, GGML_TYPE_F32, c.conv_kernel - 1, conv_dim, 1);
                ssm_states_[i] = ggml_new_tensor_4d(g, GGML_TYPE_F32,
                    c.mamba_state_size, c.mamba_head_dim, c.mamba_heads, 1);
            } else if (c.block_pattern[i] == 'A') {
                const auto width = c.attention_kv_heads * c.attention_head_dim;
                key_states_[i] = ggml_new_tensor_2d(g, GGML_TYPE_F16, width, capacity);
                value_states_[i] = ggml_new_tensor_2d(g, GGML_TYPE_F16, width, capacity);
                const auto shape = core::TensorShape::from_dims({capacity, width});
                keys.push_back(core::wrap_tensor(key_states_[i], shape, GGML_TYPE_F16));
                values.push_back(core::wrap_tensor(value_states_[i], shape, GGML_TYPE_F16));
            }
        }
        if (c.block_pattern.find_first_of("MA") != std::string::npos) {
            buffer_.reset(ggml_backend_alloc_ctx_tensors(g, execution.backend()));
            if (!buffer_) throw std::runtime_error("Nemotron-H recurrent and attention cache allocation failed");
        }
        runtime::TransformerKVCacheOptions options;
        options.allow_f16_storage = true;
        options.lazy_import_scratch = true;
        kv_ = runtime::TransformerKVCache(capacity, c.attention_kv_heads * c.attention_head_dim,
            std::move(keys), std::move(values), options);
        reset();
    }

    void reset() {
        // Includes all recurrent and convolution state, as well as masked KV.
        if (buffer_) ggml_backend_buffer_clear(buffer_.get(), 0);
        kv_.retain_prefix(0);
        processed_ = 0;
    }

    void advance(int64_t steps) {
        kv_.advance_after_direct_append(steps);
        processed_ += steps;
    }

    int64_t capacity() const { return capacity_; }
    int64_t processed() const { return processed_; }
    ggml_tensor * conv(size_t layer) const { return conv_states_[layer]; }
    ggml_tensor * ssm(size_t layer) const { return ssm_states_[layer]; }
    ggml_tensor * key(size_t layer) const { return key_states_[layer]; }
    ggml_tensor * value(size_t layer) const { return value_states_[layer]; }

private:
    int64_t capacity_;
    int64_t processed_ = 0;
    Context context_;
    Buffer buffer_;
    runtime::TransformerKVCache kv_;
    std::vector<ggml_tensor *> conv_states_;
    std::vector<ggml_tensor *> ssm_states_;
    std::vector<ggml_tensor *> key_states_;
    std::vector<ggml_tensor *> value_states_;
};

ggml_tensor * rms_norm(ggml_context * g, ggml_tensor * x, ggml_tensor * w, float eps) {
    return ggml_mul(g, ggml_rms_norm(g, x, eps), w);
}

ggml_tensor * build_mamba(ggml_context * g, ggml_cgraph * graph, ggml_tensor * input,
    const LayerWeights & w, const DecoderConfig & c, DecoderState & state, size_t layer,
    ggml_tensor * sequence_id, int64_t steps) {
    const int64_t inner = c.mamba_heads * c.mamba_head_dim;
    const int64_t grouped_state = c.mamba_groups * c.mamba_state_size;
    const int64_t conv_dim = inner + 2 * grouped_state;
    auto * projected = ggml_mul_mat(g, w.ssm_in, input);
    auto * z = ggml_view_4d(g, projected, c.mamba_head_dim, c.mamba_heads, steps, 1,
        c.mamba_head_dim * projected->nb[0], projected->nb[1], projected->nb[2], 0);
    auto * xbc = ggml_view_3d(g, projected, conv_dim, steps, 1,
        projected->nb[1], projected->nb[2], inner * projected->nb[0]);
    auto * dt = ggml_view_3d(g, projected, c.mamba_heads, steps, 1,
        projected->nb[1], projected->nb[2], (2 * inner + 2 * grouped_state) * projected->nb[0]);

    auto * window = ggml_concat(g, state.conv(layer), ggml_transpose(g, xbc), 0);
    auto * tail = ggml_view_3d(g, window, c.conv_kernel - 1, conv_dim, 1,
        window->nb[1], window->nb[2], steps * window->nb[0]);
    ggml_build_forward_expand(graph, ggml_cpy(g, tail, state.conv(layer)));
    xbc = ggml_ssm_conv(g, window, w.conv);
    if (w.conv_bias) xbc = ggml_add(g, xbc, w.conv_bias);
    xbc = ggml_silu(g, xbc);

    auto * x = ggml_view_4d(g, xbc, c.mamba_head_dim, c.mamba_heads, steps, 1,
        c.mamba_head_dim * xbc->nb[0], xbc->nb[1], xbc->nb[2], 0);
    auto * b = ggml_view_4d(g, xbc, c.mamba_state_size, c.mamba_groups, steps, 1,
        c.mamba_state_size * xbc->nb[0], xbc->nb[1], xbc->nb[2], inner * xbc->nb[0]);
    auto * cc = ggml_view_4d(g, xbc, c.mamba_state_size, c.mamba_groups, steps, 1,
        c.mamba_state_size * xbc->nb[0], xbc->nb[1], xbc->nb[2], (inner + grouped_state) * xbc->nb[0]);
    dt = ggml_add(g, ggml_cont(g, dt), w.dt_bias);
    // Stock seven-argument GGML scan returns y followed by the final F32 state.
    // GGUF ssm_a already contains -exp(A_log), so no second transform belongs here.
    auto * scan = ggml_ssm_scan(g, state.ssm(layer), x, dt, w.a, b, cc, sequence_id);
    const int64_t state_elements = c.mamba_state_size * inner;
    auto * next_state = ggml_view_1d(g, scan, state_elements,
        static_cast<size_t>(inner * steps) * sizeof(float));
    ggml_build_forward_expand(graph, ggml_cpy(g, next_state,
        ggml_reshape_1d(g, state.ssm(layer), state_elements)));
    auto * y = ggml_view_4d(g, scan, c.mamba_head_dim, c.mamba_heads, steps, 1,
        static_cast<size_t>(c.mamba_head_dim) * sizeof(float),
        static_cast<size_t>(inner) * sizeof(float), static_cast<size_t>(inner * steps) * sizeof(float), 0);
    y = ggml_add(g, y, ggml_mul(g, x, w.d));
    y = ggml_swiglu_split(g, ggml_cont(g, z), y);
    y = ggml_reshape_4d(g, y, inner / c.mamba_groups, c.mamba_groups, steps, 1);
    y = rms_norm(g, y, w.ssm_norm, c.rms_norm_eps);
    return ggml_mul_mat(g, w.ssm_out, ggml_reshape_2d(g, y, inner, steps));
}

ggml_tensor * build_attention(ggml_context * g, ggml_tensor * input, const LayerWeights & w,
    const DecoderConfig & c, DecoderState & state, size_t layer, ggml_tensor * slots,
    ggml_tensor * mask, int64_t steps) {
    auto * q = ggml_mul_mat(g, w.q, input);
    auto * k = ggml_mul_mat(g, w.k, input);
    auto * v = ggml_mul_mat(g, w.v, input);
    k = ggml_set_rows(g, state.key(layer), k, slots);
    v = ggml_set_rows(g, state.value(layer), v, slots);
    q = ggml_cont(g, ggml_permute(g,
        ggml_reshape_3d(g, q, c.attention_head_dim, c.attention_heads, steps), 0, 2, 1, 3));
    k = ggml_permute(g, ggml_reshape_3d(g, k,
        c.attention_head_dim, c.attention_kv_heads, state.capacity()), 0, 2, 1, 3);
    v = ggml_permute(g, ggml_reshape_3d(g, v,
        c.attention_head_dim, c.attention_kv_heads, state.capacity()), 0, 2, 1, 3);
    auto * attended = ggml_flash_attn_ext(g, q, k, v, mask,
        1.0F / std::sqrt(static_cast<float>(c.attention_head_dim)), 0.0F, 0.0F);
    ggml_flash_attn_ext_set_prec(attended, GGML_PREC_F32);
    attended = ggml_reshape_2d(g, attended, c.attention_head_dim * c.attention_heads, steps);
    return ggml_mul_mat(g, w.attention_out, attended);
}

struct Routing {
    ggml_tensor * ids;
    ggml_tensor * scales;
};

Routing build_routing(ggml_context * g, ggml_tensor * input, const LayerWeights & w,
    const DecoderConfig & c, int64_t steps) {
    auto * logits = ggml_mul_mat(g, w.router, input);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    auto * probabilities = ggml_sigmoid(g, logits);
    auto * selected_ids = ggml_argsort_top_k(g, ggml_add(g, probabilities, w.router_bias),
        static_cast<int>(c.experts_per_token));
    // Correction bias affects ranking only. Gather the original probabilities.
    auto * selected = ggml_get_rows(g,
        ggml_reshape_3d(g, probabilities, 1, c.experts, steps), selected_ids);
    selected = ggml_reshape_2d(g, selected, c.experts_per_token, steps);
    auto * denominator = ggml_clamp(g, ggml_sum_rows(g, selected), 6.103515625e-5F,
        std::numeric_limits<float>::infinity());
    selected = ggml_scale(g, ggml_div(g, selected, denominator), c.routed_scaling_factor);
    return {selected_ids, selected};
}

ggml_tensor * build_experts(ggml_context * g, ggml_tensor * input, ggml_tensor * ids,
    const LayerWeights & w) {
    auto * expanded = input;
    if (w.experts_up->ne[0] > expanded->ne[0]) {
        expanded = ggml_pad(g, expanded, static_cast<int>(w.experts_up->ne[0] - expanded->ne[0]), 0, 0, 0);
    }
    auto * up = ggml_mul_mat_id(g, w.experts_up, expanded, ids);
    auto * activated = ggml_sqr(g, ggml_relu(g, up));
    if (w.experts_down->ne[0] > activated->ne[0]) {
        activated = ggml_pad(g, activated, static_cast<int>(w.experts_down->ne[0] - activated->ne[0]), 0, 0, 0);
    }
    auto * routed = ggml_mul_mat_id(g, w.experts_down, activated, ids);
    return routed;
}

ggml_tensor * build_moe(ggml_context * g, ggml_tensor * input, const LayerWeights & w,
    const DecoderConfig & c, int64_t steps) {
    const auto routing = build_routing(g, input, w, c, steps);
    auto * expanded = ggml_reshape_3d(g, input, c.hidden_size, 1, steps);
    auto * routed = build_experts(g, expanded, routing.ids, w);
    routed = ggml_mul(g, routed, ggml_reshape_3d(g, routing.scales, 1, c.experts_per_token, steps));
    routed = ggml_cont(g, ggml_permute(g, routed, 1, 0, 2, 3));
    routed = ggml_reshape_2d(g, ggml_sum_rows(g, routed), c.hidden_size, steps);
    auto * shared = ggml_mul_mat(g, w.shared_up, input);
    shared = ggml_sqr(g, ggml_relu(g, shared));
    shared = ggml_mul_mat(g, w.shared_down, shared);
    return ggml_add(g, routed, shared);
}

class DecoderGraph {
public:
    DecoderGraph(const DecoderConfig & c, const Weights & w, DecoderState & state,
        core::ExecutionContext & execution, int64_t steps, bool prompt)
        : execution_(execution), state_(state), steps_(steps), prompt_(prompt), hidden_size_(c.hidden_size), weights_(w) {
        context_.reset(ggml_init({kGraphArenaBytes, nullptr, true}));
        if (!context_) throw std::runtime_error("Nemotron-H graph context allocation failed");
        auto * g = context_.get();
        graph_ = ggml_new_graph_custom(g, kGraphNodes, false);
        if (w.disk_store) {
            offload_inputs_.reset(ggml_init({64 * ggml_tensor_overhead(), nullptr, true}));
            if (!offload_inputs_) throw std::runtime_error("Nemotron-H offload input context allocation failed");
        }
        auto * inputs = w.disk_store ? offload_inputs_.get() : g;
        ids_ = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, steps);
        ggml_set_input(ids_);
        if (c.block_pattern.find('M') != std::string::npos) {
            sequence_id_ = ggml_new_tensor_1d(inputs, GGML_TYPE_I32, 1);
            ggml_set_input(sequence_id_);
        }
        if (c.block_pattern.find('A') != std::string::npos) {
            slots_ = ggml_new_tensor_1d(inputs, GGML_TYPE_I64, steps);
            mask_ = ggml_new_tensor_2d(inputs, GGML_TYPE_F16, state.capacity(), steps);
            ggml_set_input(slots_);
            ggml_set_input(mask_);
        }
        auto * cur = ggml_get_rows(g, w.embedding, ids_);
        if (prompt) {
            audio_ = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, c.hidden_size, steps);
            text_mask_ = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, 1, steps);
            ggml_set_input(audio_);
            ggml_set_input(text_mask_);
            cur = ggml_add(g, ggml_mul(g, cur, text_mask_), audio_);
        }
        if (mask_) {
            mask_scratch_.resize(static_cast<size_t>(state.capacity() * steps));
            slots_scratch_.resize(static_cast<size_t>(steps));
        }
        if (prompt) {
            audio_scratch_.resize(static_cast<size_t>(c.hidden_size * steps));
            text_mask_scratch_.resize(static_cast<size_t>(steps));
        }
        if (w.disk_store) {
            build_offload(c, w, cur);
            return;
        }
        for (size_t i = 0; i < c.block_pattern.size(); ++i) {
            const auto & layer = w.layers[i];
            auto * residual = cur;
            cur = rms_norm(g, cur, layer.norm, c.rms_norm_eps);
            switch (c.block_pattern[i]) {
                case 'M': cur = build_mamba(g, graph_, cur, layer, c, state, i, sequence_id_, steps); break;
                case 'A': cur = build_attention(g, cur, layer, c, state, i, slots_, mask_, steps); break;
                case 'E': cur = build_moe(g, cur, layer, c, steps); break;
            }
            cur = ggml_add(g, cur, residual);
        }
        // All tokens update state, but only the last row needs the large head.
        cur = ggml_view_2d(g, cur, c.hidden_size, 1, cur->nb[1], (steps - 1) * cur->nb[1]);
        cur = rms_norm(g, cur, w.final_norm, c.rms_norm_eps);
        logits_ = ggml_mul_mat(g, w.output, cur);
        ggml_set_output(logits_);
        ggml_build_forward_expand(graph_, logits_);
        core::validate_backend_graph_supported(execution.backend(), graph_, "Nemotron-H decoder");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("Nemotron-H decoder graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    ~DecoderGraph() {
        plan_.reset();
        if (offload_graphs_.empty()) core::release_backend_graph_resources(execution_.backend(), graph_, true);
        for (auto & entry : offload_plans_) entry.second.reset();
        for (auto * graph : offload_graphs_) core::release_backend_graph_resources(execution_.backend(), graph, true);
    }

    int64_t steps() const { return steps_; }

    void run_prompt(const std::vector<int32_t> & ids, const std::vector<float> & audio,
        int64_t first_audio, int64_t audio_frames, int64_t offset) {
        if (!prompt_) throw std::runtime_error("Nemotron-H prompt graph mismatch");
        std::fill(audio_scratch_.begin(), audio_scratch_.end(), 0.0F);
        std::fill(text_mask_scratch_.begin(), text_mask_scratch_.end(), 1.0F);
        for (int64_t row = 0; row < steps_; ++row) {
            const int64_t audio_row = offset + row - first_audio;
            if (audio_row >= 0 && audio_row < audio_frames) {
                text_mask_scratch_[static_cast<size_t>(row)] = 0.0F;
                std::copy_n(audio.data() + audio_row * hidden_size_, hidden_size_,
                    audio_scratch_.data() + row * hidden_size_);
            }
        }
        ggml_backend_tensor_set(audio_, audio_scratch_.data(), 0, audio_scratch_.size() * sizeof(float));
        ggml_backend_tensor_set(text_mask_, text_mask_scratch_.data(), 0, text_mask_scratch_.size() * sizeof(float));
        run(ids.data() + offset);
    }

    void run_token(int32_t token) {
        if (prompt_ || steps_ != 1) throw std::runtime_error("Nemotron-H token graph mismatch");
        run(&token);
    }

    std::vector<float> logits() const { return core::read_tensor_f32(logits_); }


private:
    void build_offload(const DecoderConfig & c, const Weights & weights,
        ggml_tensor * embeddings) {
        const auto h = c.hidden_size, k = c.experts_per_token;
        const auto first = c.block_pattern.find('E');
        slots_count_ = weights.layers.at(first).experts_up->ne[2];
        expert_rows_ = std::min<int64_t>(steps_, 32);
        while ((expert_rows_ & (expert_rows_ - 1)) != 0) ++expert_rows_;
        const auto rows = slots_count_ * expert_rows_;
        auto * persistent = offload_inputs_.get();
        hidden_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps_);
        x_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps_);
        shared_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, steps_);
        scales_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, k, steps_);
        routing_ids_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, k, steps_);
        expert_output_ = ggml_new_tensor_2d(persistent, GGML_TYPE_F32, h, k * steps_ + rows);
        source_rows_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I32, rows);
        target_rows_ = ggml_new_tensor_1d(persistent, GGML_TYPE_I64, rows);
        slot_ids_ = ggml_new_tensor_2d(persistent, GGML_TYPE_I32, 1, rows);
        for (auto * tensor : {hidden_, x_, shared_, scales_, routing_ids_, expert_output_, source_rows_, target_rows_, slot_ids_})
            ggml_set_input(tensor);
        offload_buffer_.reset(ggml_backend_alloc_ctx_tensors(persistent, execution_.backend()));
        if (!offload_buffer_) throw std::runtime_error("Nemotron-H offload input allocation failed");
        ggml_backend_buffer_clear(offload_buffer_.get(), 0);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!allocator_) throw std::runtime_error("Nemotron-H offload workspace allocation failed");
        auto * g = context_.get();
        ggml_cgraph * largest = nullptr;
        size_t maximum = 0;
        const auto add = [&](ggml_cgraph * graph) {
            core::validate_backend_graph_supported(execution_.backend(), graph, "Nemotron-H offload");
            size_t bytes = 0;
            ggml_gallocr_reserve_n_size(allocator_.get(), graph, nullptr, nullptr, &bytes);
            if (!largest || bytes > maximum) { largest = graph; maximum = bytes; }
            offload_graphs_.push_back(graph);
        };
        ggml_build_forward_expand(graph_, ggml_cpy(g, embeddings, x_));
        add(graph_);
        for (size_t i = 0; i < c.block_pattern.size(); ++i) {
            const auto & layer = weights.layers[i];
            auto * graph = ggml_new_graph_custom(g, 128, false);
            auto * normalized = rms_norm(g, x_, layer.norm, c.rms_norm_eps);
            if (c.block_pattern[i] == 'E') {
                const auto routing = build_routing(g, normalized, layer, c, steps_);
                auto * shared = ggml_mul_mat(g, layer.shared_up, normalized);
                shared = ggml_sqr(g, ggml_relu(g, shared));
                shared = ggml_mul_mat(g, layer.shared_down, shared);
                for (const auto & copy : {std::make_pair(normalized, hidden_), std::make_pair(shared, shared_),
                         std::make_pair(routing.scales, scales_),
                         std::make_pair(ggml_cast(g, routing.ids, GGML_TYPE_F32), routing_ids_)})
                    ggml_build_forward_expand(graph, ggml_cpy(g, copy.first, copy.second));
            } else {
                auto * output = c.block_pattern[i] == 'M'
                    ? build_mamba(g, graph, normalized, layer, c, state_, i, sequence_id_, steps_)
                    : build_attention(g, normalized, layer, c, state_, i, slots_, mask_, steps_);
                ggml_build_forward_expand(graph, ggml_cpy(g, ggml_add(g, output, x_), x_));
            }
            layer_graphs_.push_back(graph);
            add(graph);
            std::vector<ggml_cgraph *> experts;
            if (c.block_pattern[i] == 'E') {
                for (int64_t count = 1; count <= expert_rows_; count *= 2) {
                    auto * expert_graph = ggml_new_graph_custom(g, 64, false);
                    const auto n = slots_count_ * count;
                    auto * source = ggml_view_1d(g, source_rows_, n, 0);
                    auto * target = ggml_view_1d(g, target_rows_, n, 0);
                    auto * ids = ggml_view_2d(g, slot_ids_, 1, n, slot_ids_->nb[1], 0);
                    auto * input = ggml_reshape_3d(g, ggml_get_rows(g, hidden_, source), h, 1, n);
                    auto * output = build_experts(g, input, ids, layer);
                    ggml_build_forward_expand(expert_graph, ggml_set_rows(g, expert_output_,
                        ggml_reshape_2d(g, output, h, n), target));
                    experts.push_back(expert_graph);
                    add(expert_graph);
                }
            }
            expert_graphs_.push_back(std::move(experts));
        }
        finish_ = ggml_new_graph_custom(g, 64, false);
        auto * experts = ggml_view_3d(g, expert_output_, h, k, steps_,
            h * sizeof(float), h * k * sizeof(float), 0);
        auto * routed = ggml_mul(g, experts, ggml_reshape_3d(g, scales_, 1, k, steps_));
        routed = ggml_cont(g, ggml_permute(g, routed, 1, 0, 2, 3));
        routed = ggml_reshape_2d(g, ggml_sum_rows(g, routed), h, steps_);
        ggml_build_forward_expand(finish_, ggml_cpy(g, ggml_add(g, ggml_add(g, routed, shared_), x_), x_));
        add(finish_);
        final_ = ggml_new_graph_custom(g, 64, false);
        auto * last = ggml_view_2d(g, x_, h, 1, x_->nb[1], (steps_ - 1) * x_->nb[1]);
        logits_ = ggml_mul_mat(g, weights.output, rms_norm(g, last, weights.final_norm, c.rms_norm_eps));
        ggml_set_output(logits_);
        ggml_build_forward_expand(final_, logits_);
        add(final_);
        if (!ggml_gallocr_reserve(allocator_.get(), largest))
            throw std::runtime_error("Nemotron-H offload workspace reservation failed");
        for (auto * graph : offload_graphs_) {
            if (!ggml_gallocr_reserve(allocator_.get(), graph) || !ggml_gallocr_alloc_graph(allocator_.get(), graph))
                throw std::runtime_error("Nemotron-H offload graph allocation failed");
            core::prepare_host_graph_plan(execution_, graph, offload_plans_[graph]);
        }
        debug::timing_log_scalar("nemotron_transcribe.decoder.offload_workspace_mb",
            ggml_gallocr_get_buffer_size(allocator_.get(), 0) / (1024.0 * 1024.0));
        routing_scratch_.resize(static_cast<size_t>(k * steps_));
        by_expert_.resize(static_cast<size_t>(c.experts));
        source_scratch_.resize(static_cast<size_t>(rows));
        target_scratch_.resize(static_cast<size_t>(rows));
        slot_scratch_.resize(static_cast<size_t>(rows));
    }

    void run_offload() {
        const auto compute = [&](ggml_cgraph * graph) {
            if (core::compute_graph(execution_, graph, offload_plans_.at(graph), "nemotron_transcribe.decoder.offload") != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Nemotron-H offload graph execution failed");
        };
        compute(graph_);
        const auto k = scales_->ne[0];
        for (size_t layer = 0; layer < layer_graphs_.size(); ++layer) {
            compute(layer_graphs_[layer]);
            if (expert_graphs_[layer].empty()) continue;
            ggml_backend_tensor_get(routing_ids_, routing_scratch_.data(), 0, routing_scratch_.size() * sizeof(float));
            for (auto & pairs : by_expert_) pairs.clear();
            for (size_t pair = 0; pair < routing_scratch_.size(); ++pair)
                by_expert_.at(static_cast<size_t>(routing_scratch_[pair])).push_back(static_cast<int32_t>(pair));
            std::vector<int32_t> experts;
            for (size_t expert = 0; expert < by_expert_.size(); ++expert)
                if (!by_expert_[expert].empty()) experts.push_back(static_cast<int32_t>(expert));
            std::stable_sort(experts.begin(), experts.end(), [&](int32_t a, int32_t b) {
                return by_expert_[a].size() > by_expert_[b].size();
            });
            for (size_t first = 0; first < experts.size(); first += slots_count_) {
                const std::vector<int32_t> group(experts.begin() + first,
                    experts.begin() + std::min(experts.size(), first + static_cast<size_t>(slots_count_)));
                const auto mapped = weights_.disk_store->stage(weights_.disk_groups[layer], group);
                size_t maximum = 0;
                for (auto expert : group) maximum = std::max(maximum, by_expert_[expert].size());
                for (size_t begin = 0; begin < maximum; begin += expert_rows_) {
                    size_t count = 1, bucket = 0;
                    while (count < std::min(maximum - begin, static_cast<size_t>(expert_rows_))) { count *= 2; ++bucket; }
                    const auto rows = count * slots_count_;
                    std::fill_n(source_scratch_.begin(), rows, 0);
                    for (size_t row = 0; row < rows; ++row) {
                        target_scratch_[row] = k * steps_ + static_cast<int64_t>(row);
                        slot_scratch_[row] = static_cast<int32_t>(row / count);
                    }
                    for (size_t e = 0; e < group.size(); ++e) {
                        const auto & pairs = by_expert_[group[e]];
                        for (size_t row = begin; row < std::min(pairs.size(), begin + count); ++row) {
                            const auto target = mapped[e] * count + row - begin;
                            source_scratch_[target] = static_cast<int32_t>(pairs[row] / k);
                            target_scratch_[target] = pairs[row];
                        }
                    }
                    ggml_backend_tensor_set(source_rows_, source_scratch_.data(), 0, rows * sizeof(int32_t));
                    ggml_backend_tensor_set(target_rows_, target_scratch_.data(), 0, rows * sizeof(int64_t));
                    ggml_backend_tensor_set(slot_ids_, slot_scratch_.data(), 0, rows * sizeof(int32_t));
                    compute(expert_graphs_[layer][bucket]);
                }
            }
            compute(finish_);
        }
        compute(final_);
    }

    void run(const int32_t * ids) {
        const int64_t offset = state_.processed();
        if (offset + steps_ > state_.capacity()) throw std::runtime_error("Nemotron-H cache capacity exceeded");
        ggml_backend_tensor_set(ids_, ids, 0, static_cast<size_t>(steps_) * sizeof(int32_t));
        if (sequence_id_) {
            const int32_t sequence_id = 0;
            ggml_backend_tensor_set(sequence_id_, &sequence_id, 0, sizeof(sequence_id));
        }
        if (mask_) {
            const auto negative_infinity = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
            for (int64_t row = 0; row < steps_; ++row) {
                slots_scratch_[static_cast<size_t>(row)] = offset + row;
                auto * mask_row = mask_scratch_.data() + row * state_.capacity();
                std::fill_n(mask_row, offset + row + 1, ggml_fp32_to_fp16(0.0F));
                std::fill(mask_row + offset + row + 1, mask_row + state_.capacity(), negative_infinity);
            }
            ggml_backend_tensor_set(slots_, slots_scratch_.data(), 0, slots_scratch_.size() * sizeof(int64_t));
            ggml_backend_tensor_set(mask_, mask_scratch_.data(), 0, mask_scratch_.size() * sizeof(ggml_fp16_t));
        }
        if (!offload_graphs_.empty()) run_offload();
        else if (core::compute_graph(execution_, graph_, plan_, "nemotron_transcribe.decoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Nemotron-H decoder graph execution failed");
        }
        state_.advance(steps_);
    }

    core::ExecutionContext & execution_;
    DecoderState & state_;
    int64_t steps_;
    bool prompt_;
    int64_t hidden_size_;
    Context context_;
    Allocator allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * sequence_id_ = nullptr;
    ggml_tensor * slots_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * audio_ = nullptr;
    ggml_tensor * text_mask_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    const Weights & weights_;
    Context offload_inputs_;
    Buffer offload_buffer_;
    std::vector<ggml_cgraph *> offload_graphs_, layer_graphs_;
    std::vector<std::vector<ggml_cgraph *>> expert_graphs_;
    std::map<ggml_cgraph *, core::HostGraphPlan> offload_plans_;
    ggml_cgraph * finish_ = nullptr, * final_ = nullptr;
    ggml_tensor * hidden_ = nullptr, * x_ = nullptr, * shared_ = nullptr, * scales_ = nullptr;
    ggml_tensor * routing_ids_ = nullptr, * expert_output_ = nullptr;
    ggml_tensor * source_rows_ = nullptr, * target_rows_ = nullptr, * slot_ids_ = nullptr;
    int64_t slots_count_ = 0, expert_rows_ = 0;
    std::vector<float> routing_scratch_;
    std::vector<std::vector<int32_t>> by_expert_;
    std::vector<int32_t> source_scratch_, slot_scratch_;
    std::vector<int64_t> target_scratch_;
    std::vector<ggml_fp16_t> mask_scratch_;
    std::vector<int64_t> slots_scratch_;
    std::vector<float> audio_scratch_;
    std::vector<float> text_mask_scratch_;
};

}  // namespace

void validate_generation_options(const GenerationOptions & options, int64_t max_context) {
    if (max_context <= 0 || options.max_new_tokens <= 0 || options.max_new_tokens > max_context ||
        !std::isfinite(options.temperature) || (options.do_sample && options.temperature <= 0.0F) ||
        options.top_k < 0 || !std::isfinite(options.top_p) || options.top_p < 0.0F || options.top_p > 1.0F ||
        !std::isfinite(options.repetition_penalty) || options.repetition_penalty <= 0.0F) {
        throw std::runtime_error("Nemotron-H invalid generation options");
    }
}

struct NemotronHDecoderRuntime::Impl {
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution,
        DecoderConfig config, assets::TensorStorageType storage, DecoderOptions options)
        : source_(std::move(source)), execution_(execution), config_(std::move(config)),
          store_(execution.backend(), execution.backend_type(), "nemotron_transcribe.decoder.weights", kWeightArenaBytes) {
        validate_config(config_);
        if (!source_) throw std::runtime_error("Nemotron-H weights are missing");
        const auto & c = config_;
        const auto & s = *source_;
        if (options.disk_experts) {
            if (storage != assets::TensorStorageType::Native)
                throw std::runtime_error("Nemotron-H disk_experts requires native weight storage");
            if (c.block_pattern.find('E') == std::string::npos ||
                options.expert_cache_size < c.experts_per_token || options.expert_cache_size > c.experts)
                throw std::runtime_error("Nemotron-H expert cache must cover top-k and not exceed the expert count");
            disk_store_ = std::make_unique<core::DiskExpertWeightStore>(source_->source_path(), options.pin_experts,
                static_cast<size_t>(options.disk_expert_read_threads));
            weights_.disk_store = disk_store_.get();
        }
        weights_.disk_groups.resize(c.block_pattern.size());
        const int64_t inner = c.mamba_heads * c.mamba_head_dim;
        const int64_t conv_dim = inner + 2 * c.mamba_groups * c.mamba_state_size;
        weights_.embedding = store_.load_tensor(s, "token_embd.weight", storage, {c.vocab_size, c.hidden_size}).tensor;
        weights_.final_norm = store_.load_f32_tensor(s, "output_norm.weight", {c.hidden_size}).tensor;
        weights_.output = s.has_tensor("output.weight")
            ? store_.load_tensor(s, "output.weight", storage, {c.vocab_size, c.hidden_size}).tensor
            : weights_.embedding;
        for (size_t i = 0; i < c.block_pattern.size(); ++i) {
            const auto prefix = "blk." + std::to_string(i) + ".";
            LayerWeights layer;
            layer.norm = store_.load_f32_tensor(s, prefix + "attn_norm.weight", {c.hidden_size}).tensor;
            if (c.block_pattern[i] == 'M') {
                layer.ssm_in = store_.load_tensor(s, prefix + "ssm_in.weight", storage,
                    {inner + conv_dim + c.mamba_heads, c.hidden_size}).tensor;
                // SSM convolution and scan kernels require F32 parameters.
                layer.conv = store_.load_f32_tensor(s, prefix + "ssm_conv1d.weight", {conv_dim, c.conv_kernel}).tensor;
                if (s.has_tensor(prefix + "ssm_conv1d.bias")) {
                    layer.conv_bias = store_.load_f32_tensor(s, prefix + "ssm_conv1d.bias", {conv_dim}).tensor;
                }
                layer.dt_bias = store_.load_f32_tensor(s, prefix + "ssm_dt.bias", {c.mamba_heads}).tensor;
                layer.a = store_.load_f32_tensor(s, prefix + "ssm_a", {c.mamba_heads, 1}).tensor;
                layer.d = store_.load_f32_tensor(s, prefix + "ssm_d", {c.mamba_heads, 1}).tensor;
                layer.ssm_norm = store_.load_f32_tensor(s, prefix + "ssm_norm.weight", {c.mamba_groups, inner / c.mamba_groups}).tensor;
                layer.ssm_out = store_.load_tensor(s, prefix + "ssm_out.weight", storage, {c.hidden_size, inner}).tensor;
            } else if (c.block_pattern[i] == 'A') {
                layer.q = store_.load_tensor(s, prefix + "attn_q.weight", storage,
                    {c.attention_heads * c.attention_head_dim, c.hidden_size}).tensor;
                layer.k = store_.load_tensor(s, prefix + "attn_k.weight", storage,
                    {c.attention_kv_heads * c.attention_head_dim, c.hidden_size}).tensor;
                layer.v = store_.load_tensor(s, prefix + "attn_v.weight", storage,
                    {c.attention_kv_heads * c.attention_head_dim, c.hidden_size}).tensor;
                layer.attention_out = store_.load_tensor(s, prefix + "attn_output.weight", storage,
                    {c.hidden_size, c.attention_heads * c.attention_head_dim}).tensor;
            } else {
                layer.router = store_.load_f32_tensor(s, prefix + "ffn_gate_inp.weight", {c.experts, c.hidden_size}).tensor;
                layer.router_bias = store_.load_f32_tensor(s, prefix + "exp_probs_b.bias", {c.experts}).tensor;
                if (disk_store_) {
                    if (s.require_metadata(prefix + "ffn_up_exps.weight").shape !=
                            std::vector<int64_t>{c.experts, c.expert_width, expert_row_width(c.hidden_size, c.expert_row_alignment)} ||
                        s.require_metadata(prefix + "ffn_down_exps.weight").shape !=
                            std::vector<int64_t>{c.experts, c.hidden_size, expert_row_width(c.expert_width, c.expert_row_alignment)})
                        throw std::runtime_error("Nemotron-H offloaded expert bank shape mismatch");
                    const auto group = disk_store_->add_group(
                        {prefix + "ffn_up_exps.weight", prefix + "ffn_down_exps.weight"},
                        execution.backend(), options.expert_cache_size);
                    weights_.disk_groups[i] = group;
                    layer.experts_up = disk_store_->tensor(group, 0).tensor;
                    layer.experts_down = disk_store_->tensor(group, 1).tensor;
                } else {
                    layer.experts_up = store_.load_tensor(s, prefix + "ffn_up_exps.weight", storage,
                        {c.experts, c.expert_width, expert_row_width(c.hidden_size, c.expert_row_alignment)}).tensor;
                    layer.experts_down = store_.load_tensor(s, prefix + "ffn_down_exps.weight", storage,
                        {c.experts, c.hidden_size, expert_row_width(c.expert_width, c.expert_row_alignment)}).tensor;
                }
                layer.shared_up = store_.load_tensor(s, prefix + "ffn_up_shexp.weight", storage,
                    {c.shared_expert_width, c.hidden_size}).tensor;
                layer.shared_down = store_.load_tensor(s, prefix + "ffn_down_shexp.weight", storage,
                    {c.hidden_size, c.shared_expert_width}).tensor;
            }
            weights_.layers.push_back(layer);
        }
        store_.upload();
        if (disk_store_) {
            const auto stats = disk_store_->statistics();
            debug::timing_log_scalar("nemotron_transcribe.decoder.expert_cache_mb", stats.buffer_bytes / (1024.0 * 1024.0));
            debug::timing_log_scalar("nemotron_transcribe.decoder.host_experts_mb", stats.host_buffer_bytes / (1024.0 * 1024.0));
        }
    }

    GenerationResult generate(const std::vector<int32_t> & before, const std::vector<float> & audio,
        int64_t frames, const std::vector<int32_t> & after, const GenerationOptions & options) {
        const auto & c = config_;
        if (frames < 0 || frames > c.max_context ||
            audio.size() % static_cast<size_t>(c.hidden_size) != 0 ||
            audio.size() / static_cast<size_t>(c.hidden_size) != static_cast<size_t>(frames)) {
            throw std::runtime_error("Nemotron-H audio embeddings must have shape [frames, hidden_size]");
        }
        validate_generation_options(options, c.max_context);
        if (before.size() > static_cast<size_t>(c.max_context) || after.size() > static_cast<size_t>(c.max_context)) {
            throw std::runtime_error("Nemotron-H invalid prompt length");
        }
        const int64_t steps = static_cast<int64_t>(before.size()) + frames + static_cast<int64_t>(after.size());
        if (steps == 0 || steps > c.max_context - options.max_new_tokens) {
            throw std::runtime_error("Nemotron-H prompt plus max_new_tokens exceeds context or prompt is empty");
        }
        auto validate_ids = [&](const std::vector<int32_t> & ids) {
            for (int32_t token : ids) {
                if (token < 0 || token >= c.vocab_size) throw std::runtime_error("Nemotron-H token ID out of vocabulary");
            }
        };
        validate_ids(before);
        validate_ids(after);
        validate_ids(options.stop_token_ids);
        if (std::any_of(audio.begin(), audio.end(), [](float value) { return !std::isfinite(value); })) {
            throw std::runtime_error("Nemotron-H audio embeddings contain non-finite values");
        }
        GenerationResult out;
        out.prompt_tokens = steps;
        if (options.cancelled && options.cancelled()) {
            out.termination = TerminationReason::Cancelled;
            return out;
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        prepare(steps + options.max_new_tokens);
        state_->reset();
        // Match the other offloaded MoE runtimes: a new request starts from
        // canonical slot placement, independent of the previous routing history.
        if (disk_store_) disk_store_->invalidate_cache();
        out.cache_capacity = state_->capacity();
        std::vector<int32_t> prompt = before;
        prompt.insert(prompt.end(), static_cast<size_t>(frames), 0);
        prompt.insert(prompt.end(), after.begin(), after.end());
        std::vector<float> logits;
        for (int64_t offset = 0; offset < steps;) {
            if (options.cancelled && options.cancelled()) {
                out.termination = TerminationReason::Cancelled;
                out.prefill_ms = elapsed_ms(prefill_start);
                return out;
            }
            const auto count = std::min(c.prefill_batch_size, steps - offset);
            auto & graph = prompt_graph(count);
            graph.run_prompt(prompt, audio, static_cast<int64_t>(before.size()), frames, offset);
            offset += count;
            out.processed_prompt_tokens = offset;
            if (offset == steps) logits = graph.logits();
        }
        out.prefill_ms = elapsed_ms(prefill_start);
        const auto decode_start = std::chrono::steady_clock::now();
        sampling::HfSamplingOptions sampling_options;
        sampling_options.do_sample = options.do_sample;
        sampling_options.temperature = options.temperature;
        sampling_options.top_k = options.top_k;
        sampling_options.top_p = options.top_p;
        sampling_options.repetition_penalty = options.repetition_penalty;
        sampling::HfSampler sampler;
        sampling::HfSamplerScratch scratch;
        std::seed_seq seed{static_cast<uint32_t>(options.seed), static_cast<uint32_t>(options.seed >> 32)};
        std::mt19937 rng(seed);
        auto history = before;
        history.insert(history.end(), after.begin(), after.end());
        for (int64_t i = 0; i < options.max_new_tokens; ++i) {
            if (options.cancelled && options.cancelled()) {
                out.termination = TerminationReason::Cancelled;
                break;
            }
            const auto token = sampler.sample(logits, history, sampling_options, scratch, rng, nullptr, "Nemotron-H");
            if (options.return_last_logits) out.last_logits = logits;
            if (std::find(options.stop_token_ids.begin(), options.stop_token_ids.end(), token) != options.stop_token_ids.end()) {
                out.termination = TerminationReason::StopToken;
                out.stop_token_id = token;
                break;
            }
            out.token_ids.push_back(token);
            history.push_back(token);
            if (options.on_token) options.on_token(token);
            if (i + 1 < options.max_new_tokens) {
                if (options.cancelled && options.cancelled()) {
                    out.termination = TerminationReason::Cancelled;
                    break;
                }
                decode_->run_token(token);
                logits = decode_->logits();
            }
        }
        out.decode_ms = elapsed_ms(decode_start);
        return out;
    }

private:
    void prepare(int64_t required) {
        if (state_ && state_->capacity() >= required &&
            state_->capacity() <= std::max<int64_t>(128, 2 * required)) return;
        // Drop graphs before the state they reference. At most a full chunk,
        // one remainder shape and a single-token graph remain resident.
        tail_.reset();
        prefill_.reset();
        decode_.reset();
        state_.reset();
        const int64_t capacity = std::min(config_.max_context, ((required + 127) / 128) * 128);
        auto state = std::make_unique<DecoderState>(config_, execution_, capacity);
        auto decode = std::make_unique<DecoderGraph>(config_, weights_, *state, execution_, 1, false);
        state_ = std::move(state);
        decode_ = std::move(decode);
    }

    DecoderGraph & prompt_graph(int64_t count) {
        auto & slot = count == config_.prefill_batch_size ? prefill_ : tail_;
        if (!slot || slot->steps() != count) {
            slot = std::make_unique<DecoderGraph>(config_, weights_, *state_, execution_, count, true);
        }
        return *slot;
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    DecoderConfig config_;
    core::BackendWeightStore store_;
    std::unique_ptr<core::DiskExpertWeightStore> disk_store_;
    Weights weights_;
    std::unique_ptr<DecoderState> state_;
    std::unique_ptr<DecoderGraph> decode_;
    std::unique_ptr<DecoderGraph> prefill_;
    std::unique_ptr<DecoderGraph> tail_;
};

NemotronHDecoderRuntime::NemotronHDecoderRuntime(std::shared_ptr<const assets::TensorSource> source,
    core::ExecutionContext & execution, DecoderConfig config, assets::TensorStorageType storage, DecoderOptions options)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config), storage, options)) {}

NemotronHDecoderRuntime::~NemotronHDecoderRuntime() = default;

GenerationResult NemotronHDecoderRuntime::generate(const std::vector<int32_t> & before_audio,
    const std::vector<float> & audio_embeddings, int64_t audio_frames,
    const std::vector<int32_t> & after_audio, const GenerationOptions & options) {
    return impl_->generate(before_audio, audio_embeddings, audio_frames, after_audio, options);
}

}  // namespace engine::models::nemotron_transcribe
