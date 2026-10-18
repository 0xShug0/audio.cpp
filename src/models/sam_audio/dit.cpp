#include "engine/models/sam_audio/dit.h"

#include "engine/framework/io/json.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/sampling/diffusion_math.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace engine::models::sam_audio {
using core::TensorShape;
using core::TensorValue;

DiTModule::DiTModule(const assets::TensorSource & source, core::ExecutionContext & execution,
                     const std::filesystem::path & config)
    : store_(execution.backend(), execution.backend_type(), "sam_audio.dit.weights", 4 * 1024 * 1024) {
    const auto json = io::json::parse_file(config).require("transformer");
    dim_ = io::json::require_i64(json, "dim");
    heads_ = io::json::require_i64(json, "n_heads");
    layers_ = io::json::require_i64(json, "n_layers");
    frequency_dim_ = io::json::require_i64(json, "frequency_embedding_dim");
    max_positions_ = io::json::require_i64(json, "max_positions");
    norm_eps_ = io::json::require_f32(json, "norm_eps");
    rope_theta_ = std::max(10000.0f, 2.0f * io::json::require_f32(json, "max_positions"));
    qk_norm_ = io::json::require_bool(json, "qk_norm");
    context_norm_ = io::json::require_bool(json, "context_norm");
    for (const auto & tensor : source.tensors()) {
        if (tensor.name.rfind("transformer.", 0) == 0) {
            weights_.emplace(tensor.name.substr(12), store_.load_tensor(
                source, tensor.name, assets::TensorStorageType::Native, tensor.shape));
        } else if (tensor.name.rfind("proj.", 0) == 0 || tensor.name.rfind("memory_proj.", 0) == 0 ||
                   tensor.name.rfind("align_masked_video.", 0) == 0 || tensor.name.rfind("embed_anchors.", 0) == 0) {
            weights_.emplace(tensor.name, store_.load_tensor(
                source, tensor.name, assets::TensorStorageType::Native, tensor.shape));
        }
    }
    ones_ = store_.make_f32(TensorShape::from_dims({1, 1, dim_}), std::vector<float>(dim_, 1.0f));
    store_.upload();
}

TensorValue DiTModule::align_inputs(core::ModuleBuildContext & ctx, const TensorValue & concatenated_audio,
                                  const TensorValue & video, const TensorValue & aligned_anchor_ids) const {
    auto x = modules::LinearModule({concatenated_audio.shape.last_dim(), dim_, true, GGML_PREC_F32})
        .build(ctx, concatenated_audio, {weights_.at("proj.weight"), weights_.at("proj.bias")});
    auto visual = modules::Conv1dModule({video.shape.dims[1], dim_, 1, 1, 0, 1, true}).build(ctx, video,
        {weights_.at("align_masked_video.conv.weight"), weights_.at("align_masked_video.conv.bias")});
    visual = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, visual);
    visual = modules::LayerNormModule({dim_, 1e-5f, true, true}).build(ctx, visual,
        {weights_.at("align_masked_video.layer_norm.weight"), weights_.at("align_masked_video.layer_norm.bias")});
    auto gate = modules::TanhModule().build(ctx, weights_.at("align_masked_video.gate"));
    gate = core::reshape_tensor(ctx, gate, TensorShape::from_dims({1, 1, 1}));
    visual = modules::MulModule().build(ctx, visual, modules::RepeatModule({visual.shape}).build(ctx, gate));
    x = modules::AddModule().build(ctx, x, visual);
    const auto & embeddings = weights_.at("embed_anchors.embed.weight");
    auto anchors = modules::EmbeddingModule({embeddings.shape.dims[0], embeddings.shape.dims[1]})
        .build(ctx, aligned_anchor_ids, embeddings);
    anchors = modules::LinearModule({embeddings.shape.dims[1], dim_, false, GGML_PREC_F32})
        .build(ctx, anchors, {weights_.at("embed_anchors.proj.weight"), std::nullopt});
    gate = modules::TanhModule().build(ctx, weights_.at("embed_anchors.gate"));
    gate = core::reshape_tensor(ctx, gate, TensorShape::from_dims({1, 1, 1}));
    anchors = modules::MulModule().build(ctx, anchors, modules::RepeatModule({anchors.shape}).build(ctx, gate));
    return modules::AddModule().build(ctx, x, anchors);
}

TensorValue DiTModule::project_text(core::ModuleBuildContext & ctx, const TensorValue & text) const {
    return modules::LinearModule({text.shape.last_dim(), dim_, true, GGML_PREC_F32})
        .build(ctx, text, {weights_.at("memory_proj.weight"), weights_.at("memory_proj.bias")});
}

TensorValue DiTModule::build(core::ModuleBuildContext & ctx, const TensorValue & input,
                            const TensorValue & memory, const TensorValue & time_embedding,
                            const TensorValue & positions) const {
    const int64_t head_dim = dim_ / heads_;
    auto linear = [&](const TensorValue & x, const std::string & name) {
        const auto & weight = weights_.at(name + ".weight");
        const auto bias = weights_.find(name + ".bias");
        return modules::LinearModule({weight.shape.dims[1], weight.shape.dims[0], bias != weights_.end(), GGML_PREC_F32})
            .build(ctx, x, {weight, bias != weights_.end() ? std::optional<TensorValue>(bias->second) : std::nullopt});
    };
    auto projection = [&](const TensorValue & x, const std::string & name) {
        auto gate = modules::SiluModule().build(ctx, linear(x, name + ".w1"));
        auto up = linear(x, name + ".w3");
        return linear(modules::MulModule().build(ctx, gate, up), name + ".w2");
    };
    auto norm = [&](const TensorValue & x, const std::string & name) {
        return modules::RMSNormModule({x.shape.last_dim(), norm_eps_, true, false})
            .build(ctx, x, {weights_.at(name + ".weight"), std::nullopt});
    };
    auto modulate = [&](const TensorValue & x, const TensorValue & shift, const TensorValue & scale) {
        const auto factor = modules::AddModule().build(ctx, ones_, scale);
        auto y = modules::MulModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, factor));
        return modules::AddModule().build(ctx, y, modules::RepeatModule({x.shape}).build(ctx, shift));
    };
    auto attention = [&](const TensorValue & x, const TensorValue & context, const std::string & name, bool rotary) {
        auto heads = [&](const TensorValue & value, const std::string & norm_name) {
            // Upstream interleaves head indices within channel pairs: B,T,D,H rather than B,T,H,D.
            auto y = core::reshape_tensor(ctx, value, TensorShape::from_dims({1, value.shape.dims[1], head_dim, heads_}));
            y = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx, y);
            y = core::ensure_backend_addressable_layout(ctx, y);
            if (qk_norm_ && !norm_name.empty()) y = norm(y, name + norm_name);
            if (rotary && !norm_name.empty())
                y = modules::RoPEModule({head_dim, GGML_ROPE_TYPE_NORMAL, rope_theta_}).build(ctx, y, positions);
            return modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, y);
        };
        auto q = heads(linear(x, name + ".wq"), ".q_norm");
        auto k = heads(linear(context, name + ".wk"), ".k_norm");
        auto v = heads(linear(context, name + ".wv"), "");
        auto y = modules::ScaledDotProductAttentionModule({head_dim,
            modules::ScaledDotProductAttentionLowering::Flash, GGML_PREC_F32}).build(ctx, q, k, v);
        y = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, y),
                                TensorShape::from_dims({1, x.shape.dims[1], dim_}));
        return linear(y, name + ".wo");
    };

    auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
    const auto residual = x;
    for (int block = 1; block <= 2; ++block) {
        const std::string prefix = "x_embedder.block.block" + std::to_string(block);
        x = modules::GroupNormModule({dim_, 1, 1e-5f, true, true}).build(ctx, x,
            {weights_.at(prefix + ".groupnorm.weight"), weights_.at(prefix + ".groupnorm.bias")});
        x = modules::SiluModule().build(ctx, x);
        x = modules::Conv1dModule({dim_, dim_, 3, 1, 1, 1, true}).build(ctx, x,
            {weights_.at(prefix + ".project.weight"), weights_.at(prefix + ".project.bias")});
    }
    x = modules::AddModule().build(ctx, x, residual);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = core::ensure_backend_addressable_layout(ctx, x);

    auto time = projection(time_embedding, "t_embedder.projection");
    auto time_block = linear(modules::SiluModule().build(ctx, time), "t_block");
    time_block = core::reshape_tensor(ctx, time_block, TensorShape::from_dims({1, 6, dim_}));
    auto y = context_norm_ ? norm(memory, "y_embedder.norm") : memory;
    y = projection(y, "y_embedder.projection");
    for (int64_t layer = 0; layer < layers_; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer);
        auto table = core::reshape_tensor(ctx, weights_.at(prefix + ".scale_shift_table"), TensorShape::from_dims({1, 6, dim_}));
        auto biases = modules::AddModule().build(ctx, table, time_block);
        auto shift_a = modules::SliceModule({1, 0, 1}).build(ctx, biases);
        auto scale_a = modules::SliceModule({1, 1, 1}).build(ctx, biases);
        auto gate_a = modules::SliceModule({1, 2, 1}).build(ctx, biases);
        auto shift_f = modules::SliceModule({1, 3, 1}).build(ctx, biases);
        auto scale_f = modules::SliceModule({1, 4, 1}).build(ctx, biases);
        auto gate_f = modules::SliceModule({1, 5, 1}).build(ctx, biases);
        auto h = modulate(norm(x, prefix + ".attention_norm"), shift_a, scale_a);
        h = attention(h, h, prefix + ".attention", true);
        h = modules::MulModule().build(ctx, h, modules::RepeatModule({h.shape}).build(ctx, gate_a));
        x = modules::AddModule().build(ctx, x, h);
        x = modules::AddModule().build(ctx, x, attention(x, y, prefix + ".cross_attention", false));
        h = modulate(norm(x, prefix + ".ffn_norm"), shift_f, scale_f);
        const auto & gate = weights_.at(prefix + ".feed_forward.w1.weight");
        h = modules::GatedFeedForwardModule({dim_, gate.shape.dims[0], false,
            modules::GatedFeedForwardActivation::Silu, modules::GeluApproximation::Tanh, GGML_PREC_F32})
            .build(ctx, h, {{gate, std::nullopt}, {weights_.at(prefix + ".feed_forward.w3.weight"), std::nullopt},
                           {weights_.at(prefix + ".feed_forward.w2.weight"), std::nullopt}});
        h = modules::MulModule().build(ctx, h, modules::RepeatModule({h.shape}).build(ctx, gate_f));
        x = modules::AddModule().build(ctx, x, h);
    }
    auto table = core::reshape_tensor(ctx, weights_.at("final_layer_scale_shift_table"), TensorShape::from_dims({1, 2, dim_}));
    time = core::reshape_tensor(ctx, time, TensorShape::from_dims({1, 1, dim_}));
    auto biases = modules::AddModule().build(ctx, table, modules::RepeatModule({table.shape}).build(ctx, time));
    auto shift = modules::SliceModule({1, 0, 1}).build(ctx, biases);
    auto scale = modules::SliceModule({1, 1, 1}).build(ctx, biases);
    x = modulate(norm(x, "norm"), shift, scale);
    return linear(x, "output");
}

namespace {
class DiTGraph {
public:
    DiTGraph(core::ExecutionContext & execution, const DiTModule & module, int64_t frames, int64_t tokens,
             int64_t audio_channels, int64_t text_channels, int64_t video_channels,
             ggml_gallocr_t shared_allocator = nullptr, bool reserve_only = false)
        : backend_(execution.backend()), frames_(frames), audio_channels_(audio_channels), dim_(module.dim()),
          frequency_dim_(module.frequency_dim()), projection_(static_cast<size_t>(frames * audio_channels * 3), 0.0f) {
        constexpr size_t nodes = 32768;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio DiT graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.dit", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, frames, audio_channels * 3}));
        text_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, tokens, text_channels}));
        video_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, video_channels, frames}));
        anchors_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, frames}));
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({frames}));
        time_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, frequency_dim_}));
        memory_time_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, dim_}));
        for (auto value : {input_, text_, video_, anchors_, positions, time_, memory_time_}) ggml_set_input(value.tensor);
        for (auto value : {text_, video_, anchors_, positions}) ggml_set_output(value.tensor);
        auto aligned = module.align_inputs(ctx, input_, video_, anchors_);
        auto memory = module.project_text(ctx, text_);
        memory = modules::AddModule().build(ctx, memory, modules::RepeatModule({memory.shape}).build(ctx, memory_time_));
        output_ = module.build(ctx, aligned, memory, time_, positions);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        runtime::optimize_graph(*graph_, runtime::GraphOptimizationBackend::Gpu);
        if (!shared_allocator) allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        auto * allocator = shared_allocator ? shared_allocator : allocator_.get();
        if (!allocator || !ggml_gallocr_reserve(allocator, graph_))
            throw std::runtime_error("SAM Audio DiT graph allocation failed");
        if (reserve_only) return;
        if (!ggml_gallocr_alloc_graph(allocator, graph_))
            throw std::runtime_error("SAM Audio DiT graph allocation failed");
        debug::timing_log_scalar("sam_audio.dit.graph_buffer_mb", ggml_gallocr_get_buffer_size(allocator, 0) / 1048576.0);
        std::vector<int32_t> indices(frames);
        std::iota(indices.begin(), indices.end(), 0);
        core::write_tensor_i32(positions, indices);
    }

    ~DiTGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    void condition(const DiTConditioning & conditioning) {
        for (int64_t frame = 0; frame < frames_; ++frame)
            std::copy_n(conditioning.audio.begin() + frame * audio_channels_, audio_channels_,
                        projection_.begin() + frame * audio_channels_ * 3 + audio_channels_ * 2);
        core::write_tensor_f32(text_, conditioning.text);
        core::write_tensor_f32(video_, conditioning.video);
        core::write_tensor_i32(anchors_, conditioning.anchors);
    }

    std::vector<float> evaluate(const std::vector<float> & state, float t) {
        for (int64_t frame = 0; frame < frames_; ++frame)
            std::copy_n(state.begin() + frame * audio_channels_, audio_channels_, projection_.begin() + frame * audio_channels_ * 3);
        core::write_tensor_f32(input_, projection_);
        for (const auto & value : {time_, memory_time_}) {
            const int64_t dim = value.shape.last_dim();
            std::vector<float> embedding(dim);
            for (int64_t i = 0; i < dim / 2; ++i) {
                const float angle = t * std::exp(-std::log(10000.0f) * i / (dim / 2));
                embedding[i] = std::cos(angle);
                embedding[i + dim / 2] = std::sin(angle);
            }
            core::write_tensor_f32(value, embedding);
        }
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM Audio DiT graph execution failed");
        return core::read_tensor_f32(output_.tensor);
    }

private:
    ggml_backend_t backend_;
    int64_t frames_, audio_channels_, dim_, frequency_dim_;
    std::vector<float> projection_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    TensorValue input_, text_, video_, anchors_, time_, memory_time_, output_;
};
}  // namespace

struct DiTRuntime::Impl {
    core::ExecutionContext & execution;
    DiTModule module;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> bounded_allocator{nullptr, ggml_gallocr_free};
    std::unique_ptr<DiTGraph> graph;
    bool memory_bounded = false;
    int64_t frames = 0, tokens = 0;
    int64_t audio_channels, text_channels, video_channels;

    Impl(const assets::TensorSource & source, core::ExecutionContext & execution_, const std::filesystem::path & config)
        : execution(execution_), module(source, execution, config) {
        const auto json = io::json::parse_file(config);
        audio_channels = io::json::require_i64(json.require("transformer"), "out_channels");
        text_channels = io::json::require_i64(json.require("text_encoder"), "dim");
        video_channels = io::json::require_i64(json.require("vision_encoder"), "dim");
    }
};

DiTRuntime::DiTRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution,
                       const std::filesystem::path & config, bool memory_bounded)
    : impl_(std::make_unique<Impl>(*source, execution, config)) {
    impl_->memory_bounded = memory_bounded;
}
DiTRuntime::~DiTRuntime() = default;

std::vector<float> DiTRuntime::sample(const DiTConditioning & conditioning, const std::vector<float> & noise, int steps) {
    if (steps <= 0 || conditioning.frames <= 0 || conditioning.tokens <= 0)
        throw std::runtime_error("SAM Audio DiT requires positive steps, frames, and text token counts");
    if (noise.size() != static_cast<size_t>(conditioning.frames * impl_->audio_channels) ||
        conditioning.audio.size() != noise.size() ||
        conditioning.text.size() != static_cast<size_t>(conditioning.tokens * impl_->text_channels) ||
        conditioning.video.size() != static_cast<size_t>(conditioning.frames * impl_->video_channels) ||
        conditioning.anchors.size() != static_cast<size_t>(conditioning.frames))
        throw std::runtime_error("SAM Audio DiT conditioning shape mismatch");
    const auto start = std::chrono::steady_clock::now();
    if (impl_->memory_bounded) {
        if (conditioning.frames > impl_->module.max_positions())
            throw std::runtime_error("SAM Audio bounded mode exceeds the model's maximum audio position count");
        if (!impl_->bounded_allocator) {
            impl_->bounded_allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->execution.backend())));
            if (!impl_->bounded_allocator) throw std::runtime_error("SAM Audio DiT allocator creation failed");
            // Reserve the model's finite context capacity, not a longer padded input:
            // every real graph still evaluates only the original recording and text.
            DiTGraph capacity(impl_->execution, impl_->module, impl_->module.max_positions(), 512,
                impl_->audio_channels, impl_->text_channels, impl_->video_channels, impl_->bounded_allocator.get(), true);
        }
    }
    if (!impl_->graph || impl_->frames != conditioning.frames || impl_->tokens != conditioning.tokens) {
        impl_->graph.reset();
        impl_->graph = std::make_unique<DiTGraph>(impl_->execution, impl_->module, conditioning.frames, conditioning.tokens,
                                                impl_->audio_channels, impl_->text_channels, impl_->video_channels,
                                                impl_->bounded_allocator.get());
        impl_->frames = conditioning.frames;
        impl_->tokens = conditioning.tokens;
    }
    impl_->graph->condition(conditioning);
    auto state = noise;
    const float dt = 1.0f / steps;
    for (int step = 0; step < steps; ++step) {
        const float t = static_cast<float>(step) / steps;
        auto first = impl_->graph->evaluate(state, t);
        // The framework helper subtracts velocity; SAM's ODE advances from t=0 to t=1.
        auto middle = sampling::euler_step(state, first, -dt * 0.5f);
        auto second = impl_->graph->evaluate(middle, t + dt * 0.5f);
        sampling::euler_step_in_place(state, second, -dt);
    }
    debug::timing_log_scalar("sam_audio.dit.wall_ms", debug::elapsed_ms(start));
    return state;
}

}  // namespace engine::models::sam_audio
