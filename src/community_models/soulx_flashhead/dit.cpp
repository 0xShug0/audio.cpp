#include "engine/community_models/soulx_flashhead/dit.h"
#include "graph.h"
#include <cmath>

namespace engine::community_models::soulx_flashhead {
namespace {
class DenoiserGraph final : public Graph {
  public:
    DenoiserGraph(core::ExecutionContext & execution, Weights & weights, const Config & config, int64_t height,
                  int64_t width)
        : Graph(execution, "soulx_flashhead.dit"), weights(weights),
          condition_cache(execution.backend(), execution.config().threads, "soulx_flashhead.dit.condition", 128 * 1024) {
        auto & c = config;
        const int64_t frames = 32 / c.stride[0] + 1;
        const int64_t gh = height / c.patch[1], gw = width / c.patch[2];
        const int64_t steps = frames * gh * gw, hd = c.dim / c.heads;
        if (c.patch[0] != 1 || height % c.patch[1] || width % c.patch[2])
            throw std::runtime_error("FlashHead latent grid must fit the configured patch size");
        auto x = input("noisy", {c.latent_channels, frames, height, width});
        const auto reference = input("reference", {c.latent_channels, frames, height, width});
        x = modules::ConcatModule({0}).build(ctx, x, reference);
        modules::Conv3dConfig patch;
        patch.in_channels = c.in_channels;
        patch.out_channels = c.dim;
        patch.kernel_depth = c.patch[0];
        patch.kernel_height = c.patch[1];
        patch.kernel_width = c.patch[2];
        patch.stride_depth = c.patch[0];
        patch.stride_height = c.patch[1];
        patch.stride_width = c.patch[2];
        modules::Conv3dWeights pw{weights.get("patch_embedding.weight"), weights.get("patch_embedding.bias")};
        x = modules::Conv3dModule(patch).build(ctx, x, pw);
        x = modules::TransposeModule({{1, 2, 3, 0}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({steps, c.dim})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        auto time = input("time", {1, c.freq_dim});
        time = linear(time, "time_embedding.0");
        time = modules::SiluModule().build(ctx, time);
        time = linear(time, "time_embedding.2");
        auto t_mod = modules::SiluModule().build(ctx, time);
        t_mod = linear(t_mod, "time_projection.1");
        t_mod = modules::ReshapeModule({TensorShape::from_dims({6, c.dim})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, t_mod));

        // Audio windows are flattened in (window, hidden-layer, channel) order.
        auto first = input("audio_first", {1, 5 * 12 * 768});
        auto rest = input("audio_rest", {frames - 1, (5 + c.stride[0] - 1) * 12 * 768});
        first = modules::ReluModule().build(ctx, linear(first, "audio_proj.proj1"));
        rest = modules::ReluModule().build(ctx, linear(rest, "audio_proj.proj1_vf"));
        auto audio = modules::ConcatModule({0}).build(ctx, first, rest);
        audio = modules::ReluModule().build(ctx, linear(audio, "audio_proj.proj2"));
        audio = linear(audio, "audio_proj.proj3");
        audio = modules::ReshapeModule({TensorShape::from_dims({frames, 32, c.dim})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, audio));
        audio = norm(audio, "audio_proj.norm", 1.e-5F, true);

        // The conditioner writes persistent leaves; the step graph never follows its producers.
        condition_graph = ggml_new_graph_custom(ctx.ggml, 32768, false);
        const auto kv_shape = TensorShape::from_dims({frames, 32, c.dim});
        const std::vector<float> zeros(static_cast<size_t>(kv_shape.num_elements()), 0.0F);
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto prefix = "blocks." + std::to_string(i) + ".cross_attn";
            const auto k = norm(linear(audio, prefix + ".k"), prefix + ".norm_k", c.eps, true, true);
            const auto v = linear(audio, prefix + ".v");
            auto key = condition_cache.make_f32(kv_shape, zeros);
            auto value = condition_cache.make_f32(kv_shape, zeros);
            ggml_set_name(key.tensor, (prefix + ".cached_k").c_str());
            ggml_set_name(value.tensor, (prefix + ".cached_v").c_str());
            cross_kv.emplace_back(key, value);
            ggml_build_forward_expand(condition_graph, ggml_cpy(ctx.ggml, k.tensor, key.tensor));
            ggml_build_forward_expand(condition_graph, ggml_cpy(ctx.ggml, v.tensor, value.tensor));
        }
        condition_cache.ensure_uploaded();
        core::validate_backend_graph_supported(execution.backend(), condition_graph, "soulx_flashhead.dit.condition");
        condition_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!condition_allocator || !ggml_gallocr_alloc_graph(condition_allocator.get(), condition_graph))
            throw std::runtime_error("Cannot allocate FlashHead audio conditioning graph");
        core::prepare_host_graph_plan(execution, condition_graph, condition_plan);
        debug::timing_log_scalar("soulx_flashhead.dit.condition.buffer_mb",
                                ggml_gallocr_get_buffer_size(condition_allocator.get(), 0) / (1024. * 1024.));
        debug::timing_log_scalar("soulx_flashhead.dit.condition.cache_buffer_mb",
                                ggml_backend_buffer_get_size(cross_kv.front().first.tensor->buffer) / (1024. * 1024.));

        std::vector<float> cos(static_cast<size_t>(steps * c.heads * hd / 2)), sin(cos.size());
        const int64_t pairs = hd / 2, axis_pairs = pairs / 3;
        for (int64_t p = 0; p < steps; ++p)
            for (int64_t h = 0; h < c.heads; ++h)
                for (int64_t d = 0; d < pairs; ++d) {
                    const int64_t axis = d < pairs - 2 * axis_pairs ? 0 : (d < pairs - axis_pairs ? 1 : 2);
                    const int64_t start = axis == 0 ? 0 : (axis == 1 ? pairs - 2 * axis_pairs : pairs - axis_pairs);
                    const int64_t count = axis == 0 ? pairs - 2 * axis_pairs : axis_pairs;
                    const int64_t pos = axis == 0 ? p / (gh * gw) : (axis == 1 ? p / gw % gh : p % gw);
                    const double angle = pos / std::pow(10000., static_cast<double>(d - start) / count);
                    const size_t i = static_cast<size_t>((p * c.heads + h) * pairs + d);
                    cos[i] = std::cos(angle);
                    sin[i] = std::sin(angle);
                }
        const auto cosine = constant("rope_cos", {1, steps, c.heads, pairs}, std::move(cos));
        const auto sine = constant("rope_sin", {1, steps, c.heads, pairs}, std::move(sin));
        const int64_t pair_count = steps * c.heads * pairs;
        const auto cos_pairs = core::reshape_tensor(ctx, cosine, TensorShape::from_dims({1, pair_count, 1}));
        const auto sin_pairs = core::reshape_tensor(ctx, sine, TensorShape::from_dims({1, pair_count, 1}));
        const auto rotary = [&](TensorValue v) {
            if (!core::uses_ggml_cuda_or_hip_backend(ctx.backend_type)) {
                v = modules::ReshapeModule({TensorShape::from_dims({steps, c.heads, pairs, 2})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
                v = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, v);
                v = modules::ReshapeModule({TensorShape::from_dims({1, steps, c.heads, hd})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
                v = modules::SplitRoPEModule({hd}).build(ctx, v, cosine, sine);
                v = modules::ReshapeModule({TensorShape::from_dims({steps, c.heads, 2, pairs})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
                v = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, v);
                v = modules::ReshapeModule({TensorShape::from_dims({steps, c.dim})})
                        .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
                return v;
            }
            // Flatten (step, head, pair) in the same order as the RoPE tables.
            v = modules::ReshapeModule({TensorShape::from_dims({1, pair_count, 2})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
            const auto even = modules::SliceModule({2, 0, 1}).build(ctx, v);
            const auto odd = modules::SliceModule({2, 1, 1}).build(ctx, v);
            v = modules::RopeInterleavedPairsModule().build(ctx, even, odd, cos_pairs, sin_pairs);
            v = modules::ReshapeModule({TensorShape::from_dims({steps, c.dim})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
            return v;
        };
        bool used_sage_attention = false, used_flash_attention = false;
        const auto attention = [&](TensorValue q, TensorValue k, TensorValue v, int64_t batch, int64_t nq, int64_t nk) {
            q = modules::ReshapeModule({TensorShape::from_dims({batch, nq, c.heads, hd})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, q));
            k = modules::ReshapeModule({TensorShape::from_dims({batch, nk, c.heads, hd})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, k));
            v = modules::ReshapeModule({TensorShape::from_dims({batch, nk, c.heads, hd})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, v));
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            TensorValue y;
            if (ctx.backend_type == core::BackendType::Cuda && (hd == 64 || hd == 128)) {
                auto * q_f16 = ggml_cast(ctx.ggml, q.tensor, GGML_TYPE_F16);
                auto * k_f16 = ggml_cast(ctx.ggml, k.tensor, GGML_TYPE_F16);
                auto * v_f16 = ggml_cast(ctx.ggml, v.tensor, GGML_TYPE_F16);
                auto * sage = ggml_sage_attn2(ctx.ggml, q_f16, k_f16, v_f16,
                                            1.0F / std::sqrt(static_cast<float>(hd)), false);
                if (ggml_backend_supports_op(execution.backend(), sage)) {
                    y = core::wrap_tensor(ggml_cast(ctx.ggml, sage, GGML_TYPE_F32),
                                          TensorShape::from_dims({batch, nq, c.heads, hd}), GGML_TYPE_F32);
                    used_sage_attention = true;
                }
            }
            if (!y.valid()) {
                y = modules::ScaledDotProductAttentionModule({hd, modules::ScaledDotProductAttentionLowering::Flash})
                        .build(ctx, q, k, v);
                used_flash_attention = true;
            }
            y = modules::ReshapeModule({TensorShape::from_dims({batch * nq, c.dim})})
                    .build(ctx, core::ensure_backend_addressable_layout(ctx, y));
            return y;
        };
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto prefix = "blocks." + std::to_string(i);
            auto modulation = modules::ReshapeModule({TensorShape::from_dims({6, c.dim})})
                                  .build(ctx, weights.get(prefix + ".modulation"));
            modulation = modules::AddModule().build(ctx, modulation, t_mod);
            const auto part = [&](int index) { return modules::SliceModule({0, index, 1}).build(ctx, modulation); };
            const auto modulate = [&](TensorValue v, int shift, int scale) {
                const auto s = modules::ScaleBiasModule({1, 1}).build(ctx, part(scale));
                v = modules::MulModule(/*allow_broadcast=*/true).build(ctx, v, s);
                return modules::AddModule(false, true).build(ctx, v, part(shift));
            };
            auto y = modulate(norm(x, "", c.eps, false), 0, 1);
            auto q = rotary(norm(linear(y, prefix + ".self_attn.q"), prefix + ".self_attn.norm_q", c.eps, true, true));
            auto k = rotary(norm(linear(y, prefix + ".self_attn.k"), prefix + ".self_attn.norm_k", c.eps, true, true));
            auto v = linear(y, prefix + ".self_attn.v");
            y = linear(attention(q, k, v, 1, steps, steps), prefix + ".self_attn.o");
            y = modules::MulModule(/*allow_broadcast=*/true).build(ctx, y, part(2));
            x = modules::AddModule().build(ctx, x, y);
            y = norm(x, prefix + ".norm3", c.eps, true);
            q = norm(linear(y, prefix + ".cross_attn.q"), prefix + ".cross_attn.norm_q", c.eps, true, true);
            k = cross_kv[static_cast<size_t>(i)].first;
            v = cross_kv[static_cast<size_t>(i)].second;
            y = linear(attention(q, k, v, frames, gh * gw, 32), prefix + ".cross_attn.o");
            x = modules::AddModule().build(ctx, x, y);
            y = modulate(norm(x, "", c.eps, false), 3, 4);
            y = linear(y, prefix + ".ffn.0");
            y = modules::GeluModule({modules::GeluApproximation::Tanh}).build(ctx, y);
            y = linear(y, prefix + ".ffn.2");
            y = modules::MulModule(/*allow_broadcast=*/true).build(ctx, y, part(5));
            x = modules::AddModule().build(ctx, x, y);
        }
        debug::trace_log_scalar("soulx_flashhead.dit.attention.lowering",
                                std::string_view(used_sage_attention
                                                     ? (used_flash_attention ? "sage_attn2+flash" : "sage_attn2")
                                                     : "flash"));
        auto hm =
            modules::ReshapeModule({TensorShape::from_dims({2, c.dim})}).build(ctx, weights.get("head.modulation"));
        hm = modules::AddModule(false, true).build(ctx, hm, time);
        auto shift = modules::SliceModule({0, 0, 1}).build(ctx, hm);
        auto scale = modules::SliceModule({0, 1, 1}).build(ctx, hm);
        scale = modules::ScaleBiasModule({1, 1}).build(ctx, scale);
        x = norm(x, "", c.eps, false);
        x = modules::MulModule(/*allow_broadcast=*/true).build(ctx, x, scale);
        x = modules::AddModule(false, true).build(ctx, x, shift);
        x = linear(x, "head.head");
        x = modules::ReshapeModule(
                {TensorShape::from_dims({frames * gh, gw, c.patch[1], c.patch[2] * c.latent_channels})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, x);
        x = modules::ReshapeModule({TensorShape::from_dims({frames, height, width, c.latent_channels})})
                .build(ctx, core::ensure_backend_addressable_layout(ctx, x));
        x = modules::TransposeModule({{3, 0, 1, 2}, 4}).build(ctx, x);
        finish(core::ensure_backend_addressable_layout(ctx, x));
    }

    ~DenoiserGraph() override {
        condition_plan.reset();
        host_plan.reset();
        core::release_backend_graph_resources(execution.backend(), condition_graph, true);
        // Release step captures before the persistent K/V member storage is destroyed.
        if (graph) {
            core::release_backend_graph_resources(execution.backend(), graph, true);
            graph = nullptr;
        }
    }

    void prepare_audio(const std::vector<float> & first, const std::vector<float> & rest) {
        const auto upload_started = std::chrono::steady_clock::now();
        core::write_tensor_f32(inputs.at("audio_first"), first);
        core::write_tensor_f32(inputs.at("audio_rest"), rest);
        debug::timing_log_scalar("soulx_flashhead.dit.condition.input_upload_ms", debug::elapsed_ms(upload_started));
        const auto compute_started = std::chrono::steady_clock::now();
        if (core::compute_graph(execution, condition_graph, condition_plan, "soulx_flashhead.dit.condition") !=
            GGML_STATUS_SUCCESS)
            throw std::runtime_error("FlashHead audio conditioning graph execution failed");
        debug::timing_log_scalar("soulx_flashhead.dit.condition.graph.compute_ms", debug::elapsed_ms(compute_started));
    }

  private:
    TensorValue linear(const TensorValue & x, const std::string & prefix) {
        const auto w = weights.get(prefix + ".weight");
        modules::LinearWeights values{w, weights.get(prefix + ".bias")};
        auto y = modules::LinearModule({w.shape.dims[1], w.shape.dims[0], true, GGML_PREC_F32}).build(ctx, x, values);
        return y;
    }
    TensorValue norm(const TensorValue & x, const std::string & prefix, float eps, bool affine, bool rms = false) {
        modules::NormConfig config{x.shape.last_dim(), eps, affine, affine && !rms};
        modules::NormWeights values;
        if (affine)
            values.weight = weights.get(prefix + ".weight");
        if (affine && !rms)
            values.bias = weights.get(prefix + ".bias");
        auto y = rms ? modules::RMSNormModule(config).build(ctx, x, values)
                     : modules::LayerNormModule(config).build(ctx, x, values);
        return y;
    }
    Weights & weights;
    core::ConstantTensorCache condition_cache;
    std::vector<std::pair<TensorValue, TensorValue>> cross_kv;
    ggml_cgraph * condition_graph = nullptr;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> condition_allocator{nullptr, ggml_gallocr_free};
    core::HostGraphPlan condition_plan;
};
} // namespace

struct Denoiser::State {
    core::ExecutionContext & execution;
    const Assets & assets;
    Weights weights;
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<DenoiserGraph>> graphs;
    DenoiserGraph * prepared_graph = nullptr;
    State(core::ExecutionContext & execution, const Assets & assets)
        : execution(execution), assets(assets), weights(execution, assets.dit, "soulx_flashhead.dit.weights") {}
};
Denoiser::Denoiser(core::ExecutionContext & execution, const Assets & assets)
    : state_(std::make_unique<State>(execution, assets)) {}
Denoiser::~Denoiser() = default;
void Denoiser::prepare_audio(const std::vector<float> & audio, int64_t height, int64_t width) {
    const auto started = std::chrono::steady_clock::now();
    auto & s = *state_;
    s.prepared_graph = nullptr;
    if (audio.size() != 33 * 5 * 12 * 768)
        throw std::runtime_error("FlashHead audio context shape mismatch");
    auto & graph = s.graphs[{height, width}];
    if (!graph) {
        const auto build_started = std::chrono::steady_clock::now();
        graph = std::make_unique<DenoiserGraph>(s.execution, s.weights, s.assets.config, height, width);
        debug::timing_log_scalar("soulx_flashhead.dit.build_ms", debug::elapsed_ms(build_started));
    }
    const auto & c = s.assets.config;
    const int64_t frames = 32 / c.stride[0] + 1;
    const auto pack_started = std::chrono::steady_clock::now();
    std::vector<float> first(audio.begin(), audio.begin() + 5 * 12 * 768), rest;
    rest.reserve(static_cast<size_t>((frames - 1) * (c.stride[0] + 4) * 12 * 768));
    for (int64_t latent = 0; latent < frames - 1; ++latent) {
        for (int64_t frame = 0; frame < c.stride[0]; ++frame) {
            const int64_t begin = frame == 0 ? 0 : 2;
            const int64_t end = frame + 1 == c.stride[0] ? 5 : 3;
            for (int64_t window = begin; window < end; ++window) {
                const auto offset = ((1 + latent * c.stride[0] + frame) * 5 + window) * 12 * 768;
                rest.insert(rest.end(), audio.begin() + offset, audio.begin() + offset + 12 * 768);
            }
        }
    }
    debug::timing_log_scalar("soulx_flashhead.dit.condition.input_pack_ms", debug::elapsed_ms(pack_started));
    graph->prepare_audio(first, rest);
    s.prepared_graph = graph.get();
    debug::timing_log_scalar("soulx_flashhead.dit.condition.wall_ms", debug::elapsed_ms(started));
}
std::vector<float> Denoiser::predict(const std::vector<float> & noisy, const std::vector<float> & reference,
                                     float timestep, int64_t height, int64_t width) {
    auto & s = *state_;
    const auto graph = s.graphs.find({height, width});
    if (graph == s.graphs.end() || graph->second.get() != s.prepared_graph)
        throw std::runtime_error("FlashHead DiT requires prepare_audio for the requested latent shape");
    const auto & c = s.assets.config;
    std::vector<float> time(static_cast<size_t>(c.freq_dim));
    // The official embedding casts the timestep to BF16 before float64 sin/cos.
    std::vector<float> rounded_time{timestep};
    core::round_f32_to_bf16_in_place(rounded_time);
    for (int64_t i = 0; i < c.freq_dim / 2; ++i) {
        const double angle = rounded_time[0] / std::pow(10000., static_cast<double>(i) / (c.freq_dim / 2));
        time[static_cast<size_t>(i)] = std::cos(angle);
        time[static_cast<size_t>(i + c.freq_dim / 2)] = std::sin(angle);
    }
    core::round_f32_to_bf16_in_place(time);
    return graph->second->run({{"noisy", &noisy}, {"reference", &reference}, {"time", &time}});
}
} // namespace engine::community_models::soulx_flashhead
