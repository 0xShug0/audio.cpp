#include "engine/models/hviske_asr_v6/encoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/packed_linear_weights.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace engine::models::hviske_asr_v6 {
namespace {
constexpr size_t kContextBytes = 4 * 1024 * 1024;
struct ContextDeleter {
    void operator()(ggml_context * p) const { ggml_free(p); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t p) const { ggml_gallocr_free(p); }
};
}  // namespace

struct HviskeV6WhisperRoPEEncoderRuntime::Impl {
    Impl(std::shared_ptr<const HviskeV6Assets> assets, core::ExecutionContext & execution,
        assets::TensorStorageType storage)
        : assets_(std::move(assets)), execution_(execution),
          store_(execution.backend(), execution.backend_type(), "hviske_asr_v6.encoder.weights", kContextBytes) {
        const auto & c = assets_->config.encoder;
        const auto & source = *assets_->weights;
        conv1_ = modules::binding::conv1d_from_source(store_, source, "encoder.conv1", storage,
            c.hidden_size, c.mel_bins, 3, true);
        conv2_ = modules::binding::conv1d_from_source(store_, source, "encoder.conv2", storage,
            c.hidden_size, c.hidden_size, 3, true);
        for (int64_t i = 0; i < c.layers; ++i) {
            const auto p = "encoder.layers." + std::to_string(i);
            modules::WhisperEncoderLayerWeights layer;
            layer.attention_norm = modules::binding::norm_from_named_source(store_, source,
                p + ".self_attn_layer_norm.weight", p + ".self_attn_layer_norm.bias");
            layer.mlp_norm = modules::binding::norm_from_named_source(store_, source,
                p + ".final_layer_norm.weight", p + ".final_layer_norm.bias");
            const auto q_type = source.require_metadata(p + ".self_attn.q_proj.weight").dtype;
            const bool pack = storage != assets::TensorStorageType::Native ||
                (q_type == source.require_metadata(p + ".self_attn.k_proj.weight").dtype &&
                 q_type == source.require_metadata(p + ".self_attn.v_proj.weight").dtype);
            if (pack) {
                packed_qkv_.push_back(modules::PackedLinearWeightsBuilder({c.hidden_size, {
                    {p + ".self_attn.q_proj.weight", std::nullopt, c.hidden_size},
                    {p + ".self_attn.k_proj.weight", std::nullopt, c.hidden_size},
                    {p + ".self_attn.v_proj.weight", std::nullopt, c.hidden_size}}, false})
                    .build(store_, source, storage));
                auto bias = source.require_f32(p + ".self_attn.q_proj.bias", {c.hidden_size});
                bias.resize(2 * c.hidden_size, 0.0f);
                const auto value_bias = source.require_f32(p + ".self_attn.v_proj.bias", {c.hidden_size});
                bias.insert(bias.end(), value_bias.begin(), value_bias.end());
                packed_qkv_.back()->bias = store_.make_f32(core::TensorShape::from_dims({3 * c.hidden_size}), std::move(bias));
            } else {
                packed_qkv_.push_back(std::nullopt);
                layer.attention.query = modules::binding::linear_from_source(store_, source,
                    p + ".self_attn.q_proj", storage, c.hidden_size, c.hidden_size, true);
                layer.attention.key = modules::binding::linear_from_source(store_, source,
                    p + ".self_attn.k_proj", storage, c.hidden_size, c.hidden_size, false);
                layer.attention.value = modules::binding::linear_from_source(store_, source,
                    p + ".self_attn.v_proj", storage, c.hidden_size, c.hidden_size, true);
            }
            layer.attention.out = modules::binding::linear_from_source(store_, source,
                p + ".self_attn.out_proj", storage, c.hidden_size, c.hidden_size, true);
            const auto fc1 = modules::binding::linear_from_source(store_, source,
                p + ".fc1", storage, c.intermediate_size, c.hidden_size, true);
            const auto fc2 = modules::binding::linear_from_source(store_, source,
                p + ".fc2", storage, c.hidden_size, c.intermediate_size, true);
            layer.mlp = {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
            layers_.push_back(std::move(layer));
        }
        final_norm_ = modules::binding::norm_from_named_source(store_, source,
            "encoder.layer_norm.weight", "encoder.layer_norm.bias");
        const auto stacked = c.hidden_size * assets_->config.frame_stack;
        projector_norm_ = modules::binding::norm_weight_from_source(store_, source, "norm", stacked);
        projector_ = modules::binding::linear_from_source(store_, source, "proj", storage,
            assets_->config.decoder.hidden_size, stacked, false);
        store_.upload();
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_);
        graph_ = nullptr;
        plan_.reset();
        allocator_.reset();
        context_.reset();
    }

    void build(int64_t frames) {
        release_graph();
        const auto started = std::chrono::steady_clock::now();
        const auto & c = assets_->config.encoder;
        frames_ = frames;
        const auto encoded_frames = (frames + 1) / 2;
        const auto dim = c.hidden_size / c.heads;
        context_.reset(ggml_init({kContextBytes, nullptr, true}));
        core::ModuleBuildContext ctx{context_.get(), "hviske_asr_v6.encoder", execution_.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, c.mel_bins, frames}));
        positions_ = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({encoded_frames}));
        ggml_set_input(input_.tensor);
        ggml_set_input(positions_.tensor);
        ggml_set_output(positions_.tensor);
        const modules::GeluModule gelu({modules::GeluApproximation::ExactErf});
        auto x = modules::Conv1dModule({c.mel_bins, c.hidden_size, 3, 1, 1, 1, true}).build(ctx, input_, conv1_);
        x = gelu.build(ctx, x);
        x = modules::Conv1dModule({c.hidden_size, c.hidden_size, 3, 2, 1, 1, true}).build(ctx, x, conv2_);
        x = gelu.build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        const modules::LayerNormModule norm({c.hidden_size, 1.0e-5f, true, true});
        const modules::RoPEModule rope({c.rotary_dim, GGML_ROPE_TYPE_NEOX, c.rope_theta});
        for (size_t i = 0; i < layers_.size(); ++i) {
            const auto & layer = layers_[i];
            auto h = norm.build(ctx, x, layer.attention_norm);
            core::TensorValue q, k, v;
            if (packed_qkv_[i]) {
                auto qkv = modules::LinearModule({c.hidden_size, 3 * c.hidden_size, true}).build(ctx, h, *packed_qkv_[i]);
                q = modules::SliceModule({2, 0, c.hidden_size}).build(ctx, qkv);
                k = modules::SliceModule({2, c.hidden_size, c.hidden_size}).build(ctx, qkv);
                v = modules::SliceModule({2, 2 * c.hidden_size, c.hidden_size}).build(ctx, qkv);
                q = core::ensure_backend_addressable_layout(ctx, q);
                k = core::ensure_backend_addressable_layout(ctx, k);
                v = core::ensure_backend_addressable_layout(ctx, v);
            } else {
                q = modules::LinearModule({c.hidden_size, c.hidden_size, true}).build(ctx, h, layer.attention.query);
                k = modules::LinearModule({c.hidden_size, c.hidden_size, false}).build(ctx, h, layer.attention.key);
                v = modules::LinearModule({c.hidden_size, c.hidden_size, true}).build(ctx, h, layer.attention.value);
            }
            const auto heads = core::TensorShape::from_dims({1, encoded_frames, c.heads, dim});
            q = rope.build(ctx, core::reshape_tensor(ctx, q, heads), positions_);
            k = rope.build(ctx, core::reshape_tensor(ctx, k, heads), positions_);
            v = core::reshape_tensor(ctx, v, heads);
            q = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, q);
            k = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, k);
            v = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, v);
            h = modules::ScaledDotProductAttentionModule({dim, modules::ScaledDotProductAttentionLowering::Flash})
                .build(ctx, q, k, v);
            h = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, h),
                core::TensorShape::from_dims({1, encoded_frames, c.hidden_size}));
            h = modules::LinearModule({c.hidden_size, c.hidden_size, true}).build(ctx, h, layer.attention.out);
            x = modules::AddModule().build(ctx, x, h);
            h = norm.build(ctx, x, layer.mlp_norm);
            h = modules::FeedForwardModule({c.hidden_size, c.intermediate_size, true,
                modules::GeluApproximation::ExactErf}).build(ctx, h, layer.mlp);
            x = modules::AddModule().build(ctx, x, h);
        }
        x = norm.build(ctx, x, final_norm_);
        const auto stack = assets_->config.frame_stack;
        const auto tokens = encoded_frames / stack;
        if (tokens * stack != encoded_frames) {
            x = modules::SliceModule({1, 0, tokens * stack}).build(ctx, x);
        }
        x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x),
            core::TensorShape::from_dims({1, tokens, stack * c.hidden_size}));
        x = modules::RMSNormModule({stack * c.hidden_size, 1.0e-6f, true, false}).build(ctx, x, projector_norm_);
        output_ = modules::LinearModule({stack * c.hidden_size, assets_->config.decoder.hidden_size, false})
            .build(ctx, x, projector_);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(ctx.ggml, 8192, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) {
            throw std::runtime_error("Hviske v6 encoder graph allocation failed");
        }
        core::prepare_host_graph_plan(execution_, graph_, plan_);
        std::vector<int32_t> positions(static_cast<size_t>(encoded_frames));
        std::iota(positions.begin(), positions.end(), 0);
        ggml_backend_tensor_set(positions_.tensor, positions.data(), 0, positions.size() * sizeof(int32_t));
        debug::timing_log_scalar("hviske_asr_v6.encoder.graph.build_ms", debug::elapsed_ms(started));
    }

    std::vector<float> encode(const HviskeV6Features & features) {
        if (!graph_ || frames_ != features.frames) {
            build(features.frames);
        }
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(input_.tensor, features.values.data(), 0, features.values.size() * sizeof(float));
        if (core::compute_graph(execution_, graph_, plan_, "hviske_asr_v6.encoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Hviske v6 encoder execution failed");
        }
        const auto & c = assets_->config;
        const auto valid_encoded = (features.valid_frames + 1) / 2;
        const auto tokens = std::clamp<int64_t>((valid_encoded + c.frame_stack - 1) / c.frame_stack,
            1, ((features.frames + 1) / 2) / c.frame_stack);
        std::vector<float> output(static_cast<size_t>(tokens * c.decoder.hidden_size));
        ggml_backend_tensor_get(output_.tensor, output.data(), 0, output.size() * sizeof(float));
        debug::timing_log_scalar("hviske_asr_v6.encoder.wall_ms", debug::elapsed_ms(started));
        return output;
    }

    std::shared_ptr<const HviskeV6Assets> assets_;
    core::ExecutionContext & execution_;
    core::BackendWeightStore store_;
    modules::Conv1dWeights conv1_, conv2_;
    std::vector<modules::WhisperEncoderLayerWeights> layers_;
    std::vector<std::optional<modules::LinearWeights>> packed_qkv_;
    modules::NormWeights final_norm_, projector_norm_;
    modules::LinearWeights projector_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    core::TensorValue input_, positions_, output_;
    ggml_cgraph * graph_ = nullptr;
    int64_t frames_ = 0;
};

HviskeV6WhisperRoPEEncoderRuntime::HviskeV6WhisperRoPEEncoderRuntime(
    std::shared_ptr<const HviskeV6Assets> assets, core::ExecutionContext & execution,
    assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(std::move(assets), execution, storage)) {}
HviskeV6WhisperRoPEEncoderRuntime::~HviskeV6WhisperRoPEEncoderRuntime() = default;
std::vector<float> HviskeV6WhisperRoPEEncoderRuntime::encode(const HviskeV6Features & features) {
    return impl_->encode(features);
}

}  // namespace engine::models::hviske_asr_v6
