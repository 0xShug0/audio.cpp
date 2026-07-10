#include "engine/models/index_tts2/semantic_codec.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::index_tts2 {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

constexpr int64_t kHidden = 1024;
constexpr int64_t kVocosDim = 384;
constexpr int64_t kVocosIntermediate = 2048;
constexpr int64_t kVocosLayers = 12;
constexpr int64_t kConvNeXtKernel = 7;
constexpr int64_t kCodebookSize = 8192;
constexpr int64_t kCodebookDim = 8;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> normalized_codebook(const std::vector<float> & codebook) {
    std::vector<float> out(codebook.size(), 0.0F);
    for (int64_t row = 0; row < kCodebookSize; ++row) {
        double norm = 0.0;
        for (int64_t dim = 0; dim < kCodebookDim; ++dim) {
            const float value = codebook[static_cast<size_t>(row * kCodebookDim + dim)];
            norm += static_cast<double>(value) * static_cast<double>(value);
        }
        const float inv_norm = 1.0F / static_cast<float>(std::sqrt(norm));
        for (int64_t dim = 0; dim < kCodebookDim; ++dim) {
            const size_t index = static_cast<size_t>(row * kCodebookDim + dim);
            out[index] = codebook[index] * inv_norm;
        }
    }
    return out;
}

core::TensorValue div(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & lhs,
    const core::TensorValue & rhs) {
    core::validate_shape(rhs, lhs.shape, "Div rhs");
    return core::wrap_tensor(ggml_div(ctx.ggml, lhs.tensor, rhs.tensor), lhs.shape, GGML_TYPE_F32);
}

core::TensorValue vocos_backbone(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const IndexTTS2VocosBackboneWeights & weights) {
    auto x = modules::Conv1dModule({input_bct.shape.dims[1], kVocosDim, kConvNeXtKernel, 1, 3, 1, true})
                 .build(ctx, input_bct, weights.embed);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-6F, true, true}).build(ctx, x, weights.norm);
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    for (const auto & block : weights.blocks) {
        const auto residual = x;
        x = modules::DepthwiseConv1dModule({kVocosDim, kConvNeXtKernel, 1, 3, 1, true}).build(ctx, x, block.depthwise);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::LayerNormModule({x.shape.last_dim(), 1.0e-6F, true, true}).build(ctx, x, block.norm);
        x = modules::LinearModule({kVocosDim, kVocosIntermediate, true}).build(ctx, x, block.pointwise_in);
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({kVocosIntermediate, kVocosDim, true}).build(ctx, x, block.pointwise_out);
        const auto gamma = modules::RepeatModule({x.shape}).build(
            ctx,
            core::reshape_tensor(ctx, block.gamma, core::TensorShape::from_dims({1, 1, kVocosDim})));
        x = modules::MulModule{}.build(ctx, x, gamma);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = modules::AddModule{}.build(ctx, residual, x);
    }
    x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    return modules::LayerNormModule({x.shape.last_dim(), 1.0e-6F, true, true}).build(ctx, x, weights.final_norm);
}

core::TensorValue quantizer_in_project(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const IndexTTS2SemanticCodecWeights & weights) {
    return modules::Conv1dModule({kHidden, kCodebookDim, 1, 1, 0, 1, true}).build(ctx, input_bct, weights.quantizer_in);
}

core::TensorValue quantizer_out_project(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input_bct,
    const IndexTTS2SemanticCodecWeights & weights) {
    return modules::Conv1dModule({kCodebookDim, kHidden, 1, 1, 0, 1, true}).build(ctx, input_bct, weights.quantizer_out);
}

core::TensorValue normalize_code_latents(core::ModuleBuildContext & ctx, const core::TensorValue & latents_bdt) {
    const auto squared = core::wrap_tensor(ggml_sqr(ctx.ggml, latents_bdt.tensor), latents_bdt.shape, GGML_TYPE_F32);
    auto sum = modules::ReduceSumModule({1}).build(ctx, squared);
    sum = core::wrap_tensor(ggml_sqrt(ctx.ggml, sum.tensor), sum.shape, GGML_TYPE_F32);
    return div(ctx, latents_bdt, modules::RepeatModule({latents_bdt.shape}).build(ctx, sum));
}

core::TensorValue argmax_last_dim(core::ModuleBuildContext & ctx, const core::TensorValue & logits) {
    auto flat = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, logits),
        core::TensorShape::from_dims({logits.shape.num_elements() / logits.shape.last_dim(), logits.shape.last_dim()}));
    auto argmax = core::wrap_tensor(
        ggml_argmax(ctx.ggml, flat.tensor),
        core::TensorShape::from_dims({flat.shape.dims[0]}),
        GGML_TYPE_I32);
    return core::reshape_tensor(ctx, argmax, core::TensorShape::from_dims({logits.shape.dims[0], logits.shape.dims[1]}));
}

core::TensorValue embed_codes_bct(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & codes_bt,
    const IndexTTS2SemanticCodecWeights & weights) {
    auto emb_btd = modules::EmbeddingModule({kCodebookSize, kCodebookDim}).build(ctx, codes_bt, weights.codebook);
    auto emb_bdt = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, emb_btd);
    return quantizer_out_project(ctx, emb_bdt, weights);
}

std::vector<float> fuse_weight_norm_conv1d(
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    const auto g = source.require_f32(prefix + ".weight_g", {out_channels, 1, 1});
    const auto v = source.require_f32(prefix + ".weight_v", {out_channels, in_channels, kernel_size});
    std::vector<float> weight(v.size(), 0.0F);
    for (int64_t out = 0; out < out_channels; ++out) {
        double norm = 0.0;
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                const float value = v[static_cast<size_t>((out * in_channels + in) * kernel_size + k)];
                norm += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        const float scale = g[static_cast<size_t>(out)] / static_cast<float>(std::sqrt(norm));
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                const size_t index = static_cast<size_t>((out * in_channels + in) * kernel_size + k);
                weight[index] = v[index] * scale;
            }
        }
    }
    return weight;
}

engine::modules::Conv1dWeights load_weight_norm_conv1d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    engine::modules::Conv1dWeights weights;
    weights.weight = store.make_from_f32(
        engine::core::TensorShape::from_dims({out_channels, in_channels, kernel_size}),
        storage_type,
        fuse_weight_norm_conv1d(source, prefix, out_channels, in_channels, kernel_size));
    weights.bias = store.load_f32_tensor(source, prefix + ".bias", {out_channels});
    return weights;
}

IndexTTS2VocosConvNeXtBlockWeights load_convnext_block(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2VocosConvNeXtBlockWeights block;
    block.depthwise = binding::depthwise_conv1d_from_source(
        store,
        source,
        prefix + ".dwconv",
        conv_storage_type,
        kVocosDim,
        kConvNeXtKernel,
        true);
    block.norm = binding::norm_from_source(store, source, prefix + ".norm", kVocosDim);
    block.pointwise_in = binding::linear_from_source(
        store,
        source,
        prefix + ".pwconv1",
        matmul_storage_type,
        kVocosIntermediate,
        kVocosDim,
        true);
    block.pointwise_out = binding::linear_from_source(
        store,
        source,
        prefix + ".pwconv2",
        matmul_storage_type,
        kVocosDim,
        kVocosIntermediate,
        true);
    block.gamma = store.load_f32_tensor(source, prefix + ".gamma", {kVocosDim});
    return block;
}

IndexTTS2VocosBackboneWeights load_vocos_backbone(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t input_channels,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type) {
    IndexTTS2VocosBackboneWeights backbone;
    backbone.embed = binding::conv1d_from_source(
        store,
        source,
        prefix + ".embed",
        conv_storage_type,
        kVocosDim,
        input_channels,
        kConvNeXtKernel,
        true);
    backbone.norm = binding::norm_from_source(store, source, prefix + ".norm", kVocosDim);
    backbone.blocks.reserve(static_cast<size_t>(kVocosLayers));
    for (int64_t i = 0; i < kVocosLayers; ++i) {
        backbone.blocks.push_back(load_convnext_block(
            store,
            source,
            prefix + ".convnext." + std::to_string(i),
            matmul_storage_type,
            conv_storage_type));
    }
    backbone.final_norm = binding::norm_from_source(store, source, prefix + ".final_layer_norm", kVocosDim);
    return backbone;
}

}  // namespace

class IndexTTS2SemanticCodecRuntime::QuantizeGraph {
public:
    QuantizeGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2SemanticCodecWeights> weights,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 semantic codec quantize graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 semantic codec quantize graph requires weights");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 semantic codec quantize graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 semantic codec quantize input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.semantic_codec.quantize", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.semantic_codec.quantize.inputs",
            execution_.backend_type()};

        semantic_ = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames_, kHidden})).tensor;
        ggml_set_input(semantic_);
        auto x = core::wrap_tensor(semantic_, core::TensorShape::from_dims({1, frames_, kHidden}), GGML_TYPE_F32);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = vocos_backbone(ctx, x, weights_->encoder_backbone);
        x = modules::LinearModule({kVocosDim, kHidden, true}).build(ctx, x, weights_->encoder_projection);
        x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        auto latents = normalize_code_latents(ctx, quantizer_in_project(ctx, x, *weights_));
        const auto latents_btd = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, latents);
        auto logits = modules::LinearModule({kCodebookDim, kCodebookSize, false})
                          .build(ctx, latents_btd, {weights_->normalized_codebook, std::nullopt});
        codes_ = argmax_last_dim(ctx, logits).tensor;
        auto embedding = embed_codes_bct(
            ctx,
            core::wrap_tensor(codes_, core::TensorShape::from_dims({1, frames_}), GGML_TYPE_I32),
            *weights_);
        embedding_ = core::ensure_backend_addressable_layout(ctx, embedding).tensor;
        ggml_set_output(codes_);
        ggml_set_output(embedding_);

        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(65536, frames_ * 2048 + 4096)), false);
        ggml_build_forward_expand(graph_, codes_);
        ggml_build_forward_expand(graph_, embedding_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 semantic codec quantize input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 semantic codec quantize graph");
        }
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.quantize.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.semantic_codec.quantize.frames", frames_);
    }

    ~QuantizeGraph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    IndexTTS2SemanticCodecOutput run(const IndexTTS2SemanticEmbedding & semantic) {
        if (semantic.frames != frames_ || semantic.dims != kHidden) {
            throw std::runtime_error("IndexTTS2 semantic codec semantic input shape does not match prepared graph");
        }
        if (static_cast<int64_t>(semantic.values.size()) != frames_ * kHidden) {
            throw std::runtime_error("IndexTTS2 semantic codec semantic input value count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(semantic_, semantic.values.data(), 0, semantic.values.size() * sizeof(float));
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.quantize.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.quantize.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 semantic codec quantize graph compute failed");
        }

        IndexTTS2SemanticCodecOutput output;
        output.frames = frames_;
        output.dims = kHidden;
        output.codes.resize(static_cast<size_t>(frames_));
        output.embedding_channel_first.resize(static_cast<size_t>(kHidden * frames_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(codes_, output.codes.data(), 0, output.codes.size() * sizeof(int32_t));
        ggml_backend_tensor_get(
            embedding_,
            output.embedding_channel_first.data(),
            0,
            output.embedding_channel_first.size() * sizeof(float));
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.quantize.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2SemanticCodecWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * semantic_ = nullptr;
    ggml_tensor * codes_ = nullptr;
    ggml_tensor * embedding_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

class IndexTTS2SemanticCodecRuntime::CodesGraph {
public:
    CodesGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const IndexTTS2SemanticCodecWeights> weights,
        int64_t frames,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          frames_(frames) {
        if (frames_ <= 0) {
            throw std::runtime_error("IndexTTS2 semantic codec code graph requires positive frame count");
        }
        if (weights_ == nullptr) {
            throw std::runtime_error("IndexTTS2 semantic codec code graph requires weights");
        }

        const auto build_start = Clock::now();
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 semantic codec code graph context");
        }
        ggml_init_params input_params{16ull * 1024ull * 1024ull, nullptr, true};
        input_ctx_.reset(ggml_init(input_params));
        if (input_ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize IndexTTS2 semantic codec code input context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "index_tts2.semantic_codec.codes", execution_.backend_type()};
        core::ModuleBuildContext input_ctx{
            input_ctx_.get(),
            "index_tts2.semantic_codec.codes.inputs",
            execution_.backend_type()};
        codes_ = core::make_tensor(input_ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1, frames_})).tensor;
        ggml_set_input(codes_);
        auto embedding = embed_codes_bct(
            ctx,
            core::wrap_tensor(codes_, core::TensorShape::from_dims({1, frames_}), GGML_TYPE_I32),
            *weights_);
        embedding_ = core::ensure_backend_addressable_layout(ctx, embedding).tensor;
        ggml_set_output(embedding_);

        graph_ = ggml_new_graph_custom(ctx_.get(), static_cast<size_t>(std::max<int64_t>(4096, frames_ * 64 + 1024)), false);
        ggml_build_forward_expand(graph_, embedding_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        if (input_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate IndexTTS2 semantic codec code input buffer");
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            clear_graph();
            throw std::runtime_error("failed to allocate IndexTTS2 semantic codec code graph");
        }
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.codes.graph.build_ms",
            engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("index_tts2.semantic_codec.codes.frames", frames_);
    }

    ~CodesGraph() {
        clear_graph();
    }

    int64_t frames() const noexcept {
        return frames_;
    }

    IndexTTS2SemanticCodecOutput run(const std::vector<int32_t> & codes) {
        if (static_cast<int64_t>(codes.size()) != frames_) {
            throw std::runtime_error("IndexTTS2 semantic codec code count does not match prepared graph");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(codes_, codes.data(), 0, codes.size() * sizeof(int32_t));
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.codes.input_upload_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));

        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        timing_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.codes.graph.compute_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("IndexTTS2 semantic codec code graph compute failed");
        }

        IndexTTS2SemanticCodecOutput output;
        output.frames = frames_;
        output.dims = kHidden;
        output.codes = codes;
        output.embedding_channel_first.resize(static_cast<size_t>(kHidden * frames_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(
            embedding_,
            output.embedding_channel_first.data(),
            0,
            output.embedding_channel_first.size() * sizeof(float));
        debug::timing_log_scalar(
            "index_tts2.semantic_codec.codes.output_read_ms",
            engine::debug::elapsed_ms(timing_start, Clock::now()));
        return output;
    }

private:
    void clear_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(execution_.backend(), graph_);
            graph_ = nullptr;
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (input_buffer_ != nullptr) {
            ggml_backend_buffer_free(input_buffer_);
            input_buffer_ = nullptr;
        }
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const IndexTTS2SemanticCodecWeights> weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * codes_ = nullptr;
    ggml_tensor * embedding_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
};

std::shared_ptr<const IndexTTS2SemanticCodecWeights> load_index_tts2_semantic_codec_weights(
    const IndexTTS2Assets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes) {
    if (assets.semantic_codec_weights == nullptr) {
        throw std::runtime_error("IndexTTS2 semantic codec requires tensor source");
    }
    auto weights = std::make_shared<IndexTTS2SemanticCodecWeights>();
    weights->store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "index_tts2.semantic_codec.weights",
        weight_context_bytes);

    const auto & source = *assets.semantic_codec_weights;
    weights->encoder_backbone = load_vocos_backbone(
        *weights->store,
        source,
        "encoder.0",
        kHidden,
        matmul_storage_type,
        conv_storage_type);
    weights->encoder_projection = binding::linear_from_source(
        *weights->store,
        source,
        "encoder.1",
        matmul_storage_type,
        kHidden,
        kVocosDim,
        true);
    weights->quantizer_in = load_weight_norm_conv1d(
        *weights->store,
        source,
        "quantizer.quantizers.0.in_project",
        conv_storage_type,
        kCodebookDim,
        kHidden,
        1);
    const auto codebook = source.require_f32("quantizer.quantizers.0.codebook.weight", {kCodebookSize, kCodebookDim});
    weights->codebook = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({kCodebookSize, kCodebookDim}),
        matmul_storage_type,
        codebook);
    weights->normalized_codebook = weights->store->make_from_f32(
        engine::core::TensorShape::from_dims({kCodebookSize, kCodebookDim}),
        matmul_storage_type,
        normalized_codebook(codebook));
    weights->quantizer_out = load_weight_norm_conv1d(
        *weights->store,
        source,
        "quantizer.quantizers.0.out_project",
        conv_storage_type,
        kHidden,
        kCodebookDim,
        1);
    weights->decoder_backbone = load_vocos_backbone(
        *weights->store,
        source,
        "decoder.0",
        kHidden,
        matmul_storage_type,
        conv_storage_type);
    weights->decoder_projection = binding::linear_from_source(
        *weights->store,
        source,
        "decoder.1",
        matmul_storage_type,
        kHidden,
        kVocosDim,
        true);

    weights->store->upload();
    assets.semantic_codec_weights->release_storage();
    return weights;
}

IndexTTS2SemanticCodecRuntime::IndexTTS2SemanticCodecRuntime(
    std::shared_ptr<const IndexTTS2Assets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("IndexTTS2 semantic codec runtime requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("IndexTTS2 semantic codec graph arena must be non-zero");
    }
    weights_ = load_index_tts2_semantic_codec_weights(
        *assets_,
        execution.backend(),
        execution.backend_type(),
        matmul_storage_type,
        conv_storage_type,
        weight_context_bytes);
}

IndexTTS2SemanticCodecRuntime::~IndexTTS2SemanticCodecRuntime() = default;

void IndexTTS2SemanticCodecRuntime::prepare_quantize(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 semantic codec runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("IndexTTS2 semantic codec quantize prepare requires positive frames");
    }
    if (quantize_graph_ != nullptr && quantize_graph_->frames() == frames) {
        return;
    }
    quantize_graph_.reset();
    quantize_graph_ = std::make_unique<QuantizeGraph>(*execution_, weights_, frames, graph_arena_bytes_);
}

void IndexTTS2SemanticCodecRuntime::prepare_codes(int64_t frames) {
    if (execution_ == nullptr) {
        throw std::runtime_error("IndexTTS2 semantic codec runtime execution context is missing");
    }
    if (frames <= 0) {
        throw std::runtime_error("IndexTTS2 semantic codec code prepare requires positive frames");
    }
    if (codes_graph_ != nullptr && codes_graph_->frames() == frames) {
        return;
    }
    codes_graph_.reset();
    codes_graph_ = std::make_unique<CodesGraph>(*execution_, weights_, frames, graph_arena_bytes_);
}

IndexTTS2SemanticCodecOutput IndexTTS2SemanticCodecRuntime::quantize(const IndexTTS2SemanticEmbedding & semantic) {
    if (quantize_graph_ == nullptr || quantize_graph_->frames() != semantic.frames) {
        throw std::runtime_error("IndexTTS2 semantic codec quantize graph was not prepared for this reference length");
    }
    return quantize_graph_->run(semantic);
}

IndexTTS2SemanticCodecOutput IndexTTS2SemanticCodecRuntime::codes_to_embedding(
    const std::vector<int32_t> & codes,
    int64_t frames) {
    if (codes_graph_ == nullptr || codes_graph_->frames() != frames) {
        throw std::runtime_error("IndexTTS2 semantic codec code graph was not prepared for this generation length");
    }
    return codes_graph_->run(codes);
}

void IndexTTS2SemanticCodecRuntime::release_graphs() {
    quantize_graph_.reset();
    codes_graph_.reset();
}

}  // namespace engine::models::index_tts2
