#include "engine/models/sam_audio/vision.h"
#include "engine/models/sam_audio/frontend.h"

#include "engine/framework/debug/profiler.h"

#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <cmath>

namespace engine::models::sam_audio {
using core::TensorShape;
using core::TensorValue;

PECoreVisionEncoderModule::PECoreVisionEncoderModule(const assets::TensorSource & source, core::ExecutionContext & execution)
    : store_(execution.backend(), execution.backend_type(), "sam_audio.vision.weights", 4 * 1024 * 1024) {
    const std::string prefix = "vision_encoder.model.visual.";
    for (const auto & tensor : source.tensors()) {
        if (tensor.name.rfind(prefix, 0) == 0)
            weights_.emplace(tensor.name.substr(prefix.size()), store_.load_tensor(
                source, tensor.name, assets::TensorStorageType::Native, tensor.shape));
    }
    std::vector<int32_t> x(577, 0), y(577, 0);
    for (int32_t i = 0; i < 576; ++i) {
        x[i + 1] = i % 24 + 1;
        y[i + 1] = i / 24 + 1;
    }
    x_positions_ = store_.make_tensor(TensorShape::from_dims({577}), GGML_TYPE_I32, x.data(), x.size() * sizeof(int32_t));
    y_positions_ = store_.make_tensor(TensorShape::from_dims({577}), GGML_TYPE_I32, y.data(), y.size() * sizeof(int32_t));
    store_.upload();
}

TensorValue PECoreVisionEncoderModule::build(core::ModuleBuildContext & ctx, const TensorValue & pixels,
                                       std::map<std::string, TensorValue> * boundaries) const {
    const int64_t batch = pixels.shape.dims[0];
    core::validate_shape(pixels, TensorShape::from_dims({batch, 3, 336, 336}), "PE-Core-L14 pixels");
    constexpr int64_t dim = 1024, tokens = 577, heads = 16, head_dim = 64;
    auto x = modules::Conv2dModule({3, dim, 14, 14, 14, 14, 0, 0, 1, 1, false}).build(
        ctx, pixels, {weights_.at("conv1.weight"), std::nullopt});
    if (boundaries) boundaries->emplace("conv1", x);
    x = modules::TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, x);
    x = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({batch, 576, dim}));
    auto cls = core::reshape_tensor(ctx, weights_.at("class_embedding"), TensorShape::from_dims({1, 1, dim}));
    cls = modules::RepeatModule({TensorShape::from_dims({batch, 1, dim})}).build(ctx, cls);
    x = modules::ConcatModule({1}).build(ctx, cls, x);
    auto pos = core::reshape_tensor(ctx, weights_.at("positional_embedding"), TensorShape::from_dims({1, tokens, dim}));
    x = modules::AddModule().build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, pos));
    x = modules::LayerNormModule({dim, 1e-5f, true, true}).build(ctx, x,
        {weights_.at("ln_pre.weight"), weights_.at("ln_pre.bias")});
    if (boundaries) boundaries->emplace("ln_pre", x);
    for (int layer = 0; layer < 24; ++layer) {
        const auto name = "transformer.resblocks." + std::to_string(layer);
        auto y = modules::LayerNormModule({dim, 1e-5f, true, true}).build(ctx, x,
            {weights_.at(name + ".ln_1.weight"), weights_.at(name + ".ln_1.bias")});
        const auto qkv = modules::LinearModule({dim, 3 * dim, true, GGML_PREC_F32}).build(ctx, y,
            {weights_.at(name + ".attn.in_proj_weight"), weights_.at(name + ".attn.in_proj_bias")});
        TensorValue attention_inputs[3];
        for (int part = 0; part < 3; ++part) {
            y = modules::SliceModule({2, part * dim, dim}).build(ctx, qkv);
            y = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, y),
                TensorShape::from_dims({batch, tokens, heads, head_dim}));
            if (part < 2) {
                // PE assigns adjacent rotary pairs to x, then y coordinates.
                auto horizontal = modules::SliceModule({3, 0, head_dim / 2}).build(ctx, y);
                auto vertical = modules::SliceModule({3, head_dim / 2, head_dim / 2}).build(ctx, y);
                horizontal = modules::RoPEModule({head_dim / 2}).build(ctx, horizontal, x_positions_);
                vertical = modules::RoPEModule({head_dim / 2}).build(ctx, vertical, y_positions_);
                y = modules::ConcatModule({3}).build(ctx, horizontal, vertical);
            }
            attention_inputs[part] = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, y);
        }
        y = modules::ScaledDotProductAttentionModule({head_dim, modules::ScaledDotProductAttentionLowering::Flash,
            GGML_PREC_F32}).build(ctx, attention_inputs[0], attention_inputs[1], attention_inputs[2]);
        y = core::reshape_tensor(ctx, core::ensure_backend_addressable_layout(ctx, y), TensorShape::from_dims({batch, tokens, dim}));
        y = modules::LinearModule({dim, dim, true, GGML_PREC_F32}).build(ctx, y,
            {weights_.at(name + ".attn.out_proj.weight"), weights_.at(name + ".attn.out_proj.bias")});
        x = modules::AddModule().build(ctx, x, y);
        y = modules::LayerNormModule({dim, 1e-5f, true, true}).build(ctx, x,
            {weights_.at(name + ".ln_2.weight"), weights_.at(name + ".ln_2.bias")});
        y = modules::FeedForwardModule({dim, 4 * dim, true, modules::GeluApproximation::ExactErf, GGML_PREC_F32})
            .build(ctx, y, {weights_.at(name + ".mlp.c_fc.weight"), weights_.at(name + ".mlp.c_fc.bias"),
                           weights_.at(name + ".mlp.c_proj.weight"), weights_.at(name + ".mlp.c_proj.bias")});
        x = modules::AddModule().build(ctx, x, y);
        if (boundaries) boundaries->emplace(name, x);
    }
    x = modules::LayerNormModule({dim, 1e-5f, true, true}).build(ctx, x,
        {weights_.at("ln_post.weight"), weights_.at("ln_post.bias")});
    if (boundaries) boundaries->emplace("ln_post", x);
    modules::AttentionWeights pool;
    TensorValue * matrices[] = {&pool.q_weight, &pool.k_weight, &pool.v_weight};
    std::optional<TensorValue> * biases[] = {&pool.q_bias, &pool.k_bias, &pool.v_bias};
    for (int part = 0; part < 3; ++part) {
        *matrices[part] = modules::SliceModule({0, part * dim, dim}).build(ctx, weights_.at("attn_pool.attn.in_proj_weight"));
        *biases[part] = modules::SliceModule({0, part * dim, dim}).build(ctx, weights_.at("attn_pool.attn.in_proj_bias"));
    }
    pool.out_weight = weights_.at("attn_pool.attn.out_proj.weight");
    pool.out_bias = weights_.at("attn_pool.attn.out_proj.bias");
    auto query = modules::RepeatModule({TensorShape::from_dims({batch, 1, dim})}).build(ctx, weights_.at("attn_pool.probe"));
    modules::AttentionConfig attention{dim, 8, true, GGML_PREC_F32, GGML_PREC_F32};
    attention.use_flash_attention = true;
    x = modules::CrossAttentionModule(attention).build(ctx, query, x, pool);
    auto y = modules::LayerNormModule({dim, 1e-5f, true, true}).build(ctx, x,
        {weights_.at("attn_pool.layernorm.weight"), weights_.at("attn_pool.layernorm.bias")});
    y = modules::FeedForwardModule({dim, 4 * dim, true, modules::GeluApproximation::ExactErf, GGML_PREC_F32})
        .build(ctx, y, {weights_.at("attn_pool.mlp.c_fc.weight"), weights_.at("attn_pool.mlp.c_fc.bias"),
                       weights_.at("attn_pool.mlp.c_proj.weight"), weights_.at("attn_pool.mlp.c_proj.bias")});
    x = modules::AddModule().build(ctx, x, y);
    if (boundaries) boundaries->emplace("attn_pool", x);
    auto projection = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx, weights_.at("proj"));
    projection = core::ensure_backend_addressable_layout(ctx, projection);
    x = modules::LinearModule({dim, dim, false, GGML_PREC_F32}).build(ctx, x, {projection, std::nullopt});
    return core::reshape_tensor(ctx, x, TensorShape::from_dims({batch, dim}));
}

namespace {
class PECoreVisionEncoderGraph {
public:
    PECoreVisionEncoderGraph(core::ExecutionContext & execution, const PECoreVisionEncoderModule & module, int64_t batch)
        : backend_(execution.backend()) {
        context_.reset(ggml_init({32 * 1024 * 1024, nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio vision context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.vision", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 3, 336, 336}));
        ggml_set_input(input_.tensor);
        output_ = module.build(ctx, input_);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        auto optimization = runtime::graph_optimization_options_for_backend(execution.backend_type() == core::BackendType::Cpu
            ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        // Preserve materialization owners referenced by sliced rotary/projection views.
        optimization.fold_identity_materializations = false;
        runtime::optimize_graph(*graph_, optimization);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("SAM Audio vision graph allocation failed");
    }
    ~PECoreVisionEncoderGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    std::vector<float> run(const std::vector<float> & pixels) {
        const auto start = std::chrono::steady_clock::now();
        core::write_tensor_f32(input_, pixels);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM Audio vision graph execution failed");
        auto result = core::read_tensor_f32(output_.tensor);
        for (size_t offset = 0; offset < result.size(); offset += 1024) {
            double norm = 0;
            for (size_t i = offset; i < offset + 1024; ++i) norm += double(result[i]) * result[i];
            norm = std::max(1e-12, std::sqrt(norm));
            for (size_t i = offset; i < offset + 1024; ++i) result[i] /= norm;
        }
        debug::timing_log_scalar("sam_audio.vision.wall_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    TensorValue input_, output_;
};
}  // namespace

struct PECoreVisionEncoder::Impl {
    core::ExecutionContext & execution;
    PECoreVisionEncoderModule module;
    std::unique_ptr<PECoreVisionEncoderGraph> graph;
    int64_t batch = 0;
    Impl(const assets::TensorSource & source, core::ExecutionContext & context)
        : execution(context), module(source, context) {}
};

PECoreVisionEncoder::PECoreVisionEncoder(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(*source, execution)) {}
PECoreVisionEncoder::~PECoreVisionEncoder() = default;

std::vector<float> PECoreVisionEncoder::encode_video(const SourceVideo & video, int64_t audio_frames,
                                               int64_t hop, int sample_rate) {
    const auto selected = select_video_frames(video, audio_frames, hop, sample_rate);
    auto unique = selected;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    const size_t batch = std::min<size_t>(32, unique.size());
    constexpr size_t frame_values = 3 * 336 * 336;
    std::vector<float> pixels(batch * frame_values), embeddings(unique.size() * 1024);
    for (size_t offset = 0; offset < unique.size(); offset += batch) {
        const size_t count = std::min(batch, unique.size() - offset);
        for (size_t i = 0; i < count; ++i) {
            const auto frame = prepare_video_frame(video.frames[unique[offset + i]].rgb, video.width, video.height);
            std::copy(frame.begin(), frame.end(), pixels.begin() + i * frame_values);
        }
        // Pad the last batch with a valid frame to retain one graph shape.
        for (size_t i = count; i < batch; ++i)
            std::copy_n(pixels.begin(), frame_values, pixels.begin() + i * frame_values);
        const auto encoded = encode(pixels, batch);
        std::copy_n(encoded.begin(), count * 1024, embeddings.begin() + offset * 1024);
    }
    std::vector<float> result(audio_frames * 1024);
    for (int64_t t = 0; t < audio_frames; ++t) {
        const size_t index = std::lower_bound(unique.begin(), unique.end(), selected[t]) - unique.begin();
        for (int64_t c = 0; c < 1024; ++c) result[c * audio_frames + t] = embeddings[index * 1024 + c];
    }
    return result;
}

std::vector<float> PECoreVisionEncoder::encode(const std::vector<float> & pixels, int64_t batch) {
    if (batch <= 0 || pixels.size() != static_cast<size_t>(batch * 3 * 336 * 336))
        throw std::runtime_error("SAM Audio vision expects [batch,3,336,336] normalized pixels");
    if (!impl_->graph || impl_->batch != batch) {
        impl_->graph.reset();
        impl_->graph = std::make_unique<PECoreVisionEncoderGraph>(impl_->execution, impl_->module, batch);
        impl_->batch = batch;
    }
    return impl_->graph->run(pixels);
}

}  // namespace engine::models::sam_audio
