#include "engine/models/miocodec/components.h"

#include "graph_ops.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/core/constant_tensor_cache.h"

#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::miocodec::graphs {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int64_t conv1d_output_frames(const engine::modules::Conv1dConfig & config, int64_t input_frames) {
    return (input_frames + 2 * config.padding - config.dilation * (config.kernel_size - 1) - 1) / config.stride + 1;
}

int64_t conv_transpose1d_output_frames(const engine::modules::ConvTranspose1dConfig & config, int64_t input_frames) {
    return (input_frames - 1) * config.stride - 2 * config.padding + config.dilation * (config.kernel_size - 1) + 1;
}

}  // namespace

class MioCodecGlobalEncoderGraph final {
public:
    MioCodecGlobalEncoderGraph(
        std::shared_ptr<const MioCodecWeights> weights,
        int64_t ssl_frames,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes)
        : weights_(std::move(weights)),
          ssl_frames_(ssl_frames) {
        backend_ = execution_context.backend();
        compute_threads_ = std::max(1, execution_context.config().threads);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MioCodec global encoder graph context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "miocodec.global_encoder", execution_context.backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, ssl_frames_, 768}));
        input_ = input.tensor;
        const auto output = global_encoder(build_ctx, input, weights_->global_encoder);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MioCodec global encoder graph");
        }
    }

    ~MioCodecGlobalEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    int64_t ssl_frames() const noexcept {
        return ssl_frames_;
    }

    MioCodecGlobalEmbedding encode(const std::vector<float> & ssl_features) const {
        if (static_cast<int64_t>(ssl_features.size()) != ssl_frames_ * 768) {
            throw std::runtime_error("MioCodec global encoder input shape mismatch");
        }
        auto timing_start = Clock::now();
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, ssl_frames_, 768}), GGML_TYPE_F32), ssl_features);
        engine::debug::timing_log_scalar("miocodec.global_encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start));
        timing_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MioCodec global encoder graph execution failed");
        }
        engine::debug::timing_log_scalar("miocodec.global_encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start));
        MioCodecGlobalEmbedding out;
        out.dim = output_->ne[0];
        timing_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("miocodec.global_encoder.output_read_ms", engine::debug::elapsed_ms(timing_start));
        return out;
    }

private:
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::shared_ptr<const MioCodecWeights> weights_;
    int64_t ssl_frames_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

class MioCodecContentEncoderGraph final {
public:
    MioCodecContentEncoderGraph(
        std::shared_ptr<const MioCodecWeights> weights,
        int64_t ssl_frames,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes)
        : weights_(std::move(weights)),
          ssl_frame_capacity_(ssl_frames),
          content_frame_capacity_(conv1d_output_frames(weights_->conv_downsample.config, ssl_frames)) {
        backend_ = execution_context.backend();
        compute_threads_ = std::max(1, execution_context.config().threads);
        constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            compute_threads_,
            "miocodec.content_encoder.constants",
            constant_context_bytes);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MioCodec content encoder graph context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "miocodec.content_encoder", execution_context.backend_type()};
        input_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, ssl_frame_capacity_, 768}));
        attention_mask_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, ssl_frame_capacity_, ssl_frame_capacity_}));
        ggml_set_input(input_.tensor);
        ggml_set_input(attention_mask_.tensor);
        constants_->begin_graph();
        auto x = transformer(build_ctx, *constants_, input_, weights_->local_encoder, std::nullopt, attention_mask_);
        const auto token_mask = make_ones_i32(
            *constants_,
            core::TensorShape::from_dims({1, ssl_frame_capacity_}));
        x = engine::modules::MaskingModule().build(build_ctx, x, token_mask);
        x = engine::modules::Conv1dModule(weights_->conv_downsample.config).build(
            build_ctx,
            engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x),
            weights_->conv_downsample.weights);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
        x = fsq_quantized(build_ctx, *constants_, x, weights_->local_quantizer);
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        constants_->finish_graph();
        engine::debug::timing_log_scalar("miocodec.content_encoder.constants.upload_ms", constants_->ensure_uploaded());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MioCodec content encoder graph");
        }
    }

    ~MioCodecContentEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    int64_t ssl_frame_capacity() const noexcept {
        return ssl_frame_capacity_;
    }

    MioCodecContentEmbedding encode(const std::vector<float> & ssl_features, int64_t ssl_frames) const {
        if (ssl_frames <= 0 || ssl_frames > ssl_frame_capacity_ ||
            static_cast<int64_t>(ssl_features.size()) != ssl_frames * 768) {
            throw std::runtime_error("MioCodec content encoder input shape mismatch");
        }
        const int64_t content_frames = conv1d_output_frames(weights_->conv_downsample.config, ssl_frames);
        if (content_frames <= 0 || content_frames > content_frame_capacity_) {
            throw std::runtime_error("MioCodec content encoder output frame count exceeds graph capacity");
        }
        auto timing_start = Clock::now();
        core::write_tensor_f32(input_, ssl_features);
        core::write_tensor_f32(
            attention_mask_,
            local_attention_mask(
                1,
                ssl_frame_capacity_,
                weights_->local_encoder.window_size,
                ssl_frames));
        engine::debug::timing_log_scalar("miocodec.content_encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start));
        timing_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MioCodec content encoder graph execution failed");
        }
        engine::debug::timing_log_scalar("miocodec.content_encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start));
        MioCodecContentEmbedding out;
        out.frames = content_frames;
        out.dim = output_->ne[0];
        timing_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        out.values.resize(static_cast<size_t>(out.frames * out.dim));
        engine::debug::timing_log_scalar("miocodec.content_encoder.output_read_ms", engine::debug::elapsed_ms(timing_start));
        return out;
    }

private:
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    std::shared_ptr<const MioCodecWeights> weights_;
    int64_t ssl_frame_capacity_ = 0;
    int64_t content_frame_capacity_ = 0;
    core::TensorValue input_;
    core::TensorValue attention_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

class MioCodecWaveDecoderGraph final {
public:
    MioCodecWaveDecoderGraph(
        std::shared_ptr<const MioCodecWeights> weights,
        int64_t content_frames,
        int64_t stft_frames,
        core::ExecutionContext & execution_context,
        size_t graph_arena_bytes,
        size_t constant_context_bytes)
        : weights_(std::move(weights)),
          content_frame_capacity_(content_frames),
          stft_frame_capacity_(stft_frames) {
        backend_ = execution_context.backend();
        compute_threads_ = std::max(1, execution_context.config().threads);
        int64_t stage_frames = stft_frame_capacity_;
        upsample_stage_frame_capacities_.reserve(weights_->wave_upsampler.stages.size());
        for (const auto & stage : weights_->wave_upsampler.stages) {
            stage_frames = conv_transpose1d_output_frames(stage.upsample.config, stage_frames);
            upsample_stage_frame_capacities_.push_back(stage_frames);
        }
        output_frame_capacity_ = stage_frames;
        constants_ = std::make_unique<core::ConstantTensorCache>(
            backend_,
            compute_threads_,
            "miocodec.wave_decoder.constants",
            constant_context_bytes);
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize MioCodec wave decoder graph context");
        }
        core::ModuleBuildContext build_ctx{ctx_.get(), "miocodec.wave_decoder", execution_context.backend_type()};
        content_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, content_frame_capacity_, 768}));
        condition_ = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, 128}));
        prenet_attention_mask_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, content_frame_capacity_, content_frame_capacity_}));
        decoder_attention_mask_ = core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, 1, stft_frame_capacity_, stft_frame_capacity_}));
        ggml_set_input(content_.tensor);
        ggml_set_input(condition_.tensor);
        ggml_set_input(prenet_attention_mask_.tensor);
        ggml_set_input(decoder_attention_mask_.tensor);
        constants_->begin_graph();
        const auto content_token_mask = make_ones_i32(
            *constants_,
            core::TensorShape::from_dims({1, content_frame_capacity_}));
        const auto interpolation = make_interpolation_constants(
            *constants_,
            content_frame_capacity_,
            stft_frame_capacity_,
            weights_->wave_conv_upsample);
        const auto stft_token_mask = make_ones_i32(
            *constants_,
            core::TensorShape::from_dims({1, stft_frame_capacity_}));
        const auto stft_time = make_full_time_mask(*constants_, stft_frame_capacity_, 512 / 32);
        std::vector<core::TensorValue> upsample_time_masks;
        std::vector<core::TensorValue> upsample_inv_group_counts;
        upsample_time_masks.reserve(upsample_stage_frame_capacities_.size());
        upsample_inv_group_counts.reserve(upsample_stage_frame_capacities_.size());
        for (size_t index = 0; index < upsample_stage_frame_capacities_.size(); ++index) {
            const int64_t frames = upsample_stage_frame_capacities_[index];
            const int64_t channels = 512 / (int64_t{1} << (static_cast<int64_t>(index) + 1));
            const auto mask = make_full_time_mask(
                *constants_,
                frames,
                channels / std::min<int64_t>(32, channels));
            upsample_time_masks.push_back(mask.mask);
            upsample_inv_group_counts.push_back(mask.inv_valid_group_count);
        }
        auto x = transformer(build_ctx, *constants_, content_, weights_->wave_prenet, std::nullopt, prenet_attention_mask_);
        x = engine::modules::MaskingModule().build(build_ctx, x, content_token_mask);
        x = engine::modules::ConvTranspose1dModule(weights_->wave_conv_upsample.config).build(
            build_ctx,
            engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x),
            weights_->wave_conv_upsample.weights);
        x = dynamic_linear_interpolate_bct(
            build_ctx,
            x,
            interpolation.left_indices,
            interpolation.right_indices,
            interpolation.right_weight);
        x = mask_bct(build_ctx, x, stft_time.mask);
        x = resnet_stack(build_ctx, x, weights_->wave_prior_net, 32, stft_time.mask, stft_time.inv_valid_group_count);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x);
        x = transformer(build_ctx, *constants_, x, weights_->wave_decoder, condition_, decoder_attention_mask_);
        x = engine::modules::MaskingModule().build(build_ctx, x, stft_token_mask);
        x = resnet_stack(
            build_ctx,
            engine::modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(build_ctx, x),
            weights_->wave_post_net,
            32,
            stft_time.mask,
            stft_time.inv_valid_group_count);
        x = upsampler(build_ctx, x, weights_->wave_upsampler, 32, &upsample_time_masks, &upsample_inv_group_counts);
        x = engine::modules::LinearModule({512, 394, weights_->istft_head.output_projection.bias.has_value()}).build(
            build_ctx,
            x,
            weights_->istft_head.output_projection);
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 32768, false);
        ggml_build_forward_expand(graph_, output_);
        constants_->finish_graph();
        engine::debug::timing_log_scalar("miocodec.wave_decoder.constants.upload_ms", constants_->ensure_uploaded());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate MioCodec wave decoder graph");
        }
    }

    ~MioCodecWaveDecoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    int64_t content_frame_capacity() const noexcept {
        return content_frame_capacity_;
    }

    int64_t stft_frame_capacity() const noexcept {
        return stft_frame_capacity_;
    }

    MioCodecWaveHead decode(const MioCodecContentEmbedding & content, const MioCodecGlobalEmbedding & condition, int64_t stft_frames) const {
        if (content.frames <= 0 || content.frames > content_frame_capacity_ || content.dim != 768 ||
            static_cast<int64_t>(content.values.size()) != content.frames * 768) {
            throw std::runtime_error("MioCodec wave decoder content input shape mismatch");
        }
        if (stft_frames <= 0 || stft_frames > stft_frame_capacity_) {
            throw std::runtime_error("MioCodec wave decoder STFT frame count exceeds graph capacity");
        }
        if (condition.dim != 128 || static_cast<int64_t>(condition.values.size()) != 128) {
            throw std::runtime_error("MioCodec wave decoder condition input shape mismatch");
        }
        const int64_t actual_output_frames = upsample_output_frames(stft_frames);
        if (actual_output_frames <= 0 || actual_output_frames > output_frame_capacity_) {
            throw std::runtime_error("MioCodec wave decoder output frame count exceeds graph capacity");
        }
        auto timing_start = Clock::now();
        core::write_tensor_f32(content_, content.values);
        core::write_tensor_f32(condition_, condition.values);
        core::write_tensor_f32(
            prenet_attention_mask_,
            local_attention_mask(
                1,
                content_frame_capacity_,
                weights_->wave_prenet.window_size,
                content.frames));
        core::write_tensor_f32(
            decoder_attention_mask_,
            local_attention_mask(
                1,
                stft_frame_capacity_,
                weights_->wave_decoder.window_size,
                stft_frames));
        engine::debug::timing_log_scalar("miocodec.wave_decoder.input_upload_ms", engine::debug::elapsed_ms(timing_start));
        timing_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MioCodec wave decoder graph execution failed");
        }
        engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start));
        MioCodecWaveHead out;
        out.frames = actual_output_frames;
        out.bins = output_->ne[0];
        timing_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        out.values.resize(static_cast<size_t>(out.frames * out.bins));
        engine::debug::timing_log_scalar("miocodec.wave_decoder.output_read_ms", engine::debug::elapsed_ms(timing_start));
        return out;
    }

private:
    int64_t upsample_output_frames(int64_t stft_frames) const {
        int64_t frames = stft_frames;
        for (const auto & stage : weights_->wave_upsampler.stages) {
            frames = conv_transpose1d_output_frames(stage.upsample.config, frames);
        }
        return frames;
    }

    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<core::ConstantTensorCache> constants_;
    std::shared_ptr<const MioCodecWeights> weights_;
    int64_t content_frame_capacity_ = 0;
    int64_t stft_frame_capacity_ = 0;
    int64_t output_frame_capacity_ = 0;
    std::vector<int64_t> upsample_stage_frame_capacities_;
    core::TensorValue content_;
    core::TensorValue condition_;
    core::TensorValue prenet_attention_mask_;
    core::TensorValue decoder_attention_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_gallocr_t gallocr_ = nullptr;
};

}  // namespace engine::models::miocodec::graphs

namespace engine::models::miocodec {

using Clock = std::chrono::steady_clock;

MioCodecContentEmbedding content_embedding_from_tokens(
    const MioCodecWeights & weights,
    const std::vector<int32_t> & content_tokens) {
    if (content_tokens.empty()) {
        throw std::runtime_error("MioCodec content token decode requires at least one token");
    }
    const auto & table = weights.content_token_embeddings;
    if (table.codebook_size <= 0 || table.dim <= 0 ||
        static_cast<int64_t>(table.values.size()) != table.codebook_size * table.dim) {
        throw std::runtime_error("MioCodec content token embedding table is invalid");
    }
    MioCodecContentEmbedding out;
    out.frames = static_cast<int64_t>(content_tokens.size());
    out.dim = table.dim;
    out.values.resize(static_cast<size_t>(out.frames * out.dim), 0.0F);
    for (int64_t frame = 0; frame < out.frames; ++frame) {
        const int32_t token = content_tokens[static_cast<size_t>(frame)];
        if (token < 0 || token >= table.codebook_size) {
            throw std::runtime_error("MioCodec content token is out of range");
        }
        const float * src = table.values.data() + static_cast<size_t>(token) * static_cast<size_t>(table.dim);
        float * dst = out.values.data() + static_cast<size_t>(frame) * static_cast<size_t>(out.dim);
        std::copy_n(src, static_cast<size_t>(out.dim), dst);
    }
    return out;
}

MioCodecGlobalEncoderRuntime::MioCodecGlobalEncoderRuntime(
    std::shared_ptr<const MioCodecAssets> assets,
    std::shared_ptr<const MioCodecWeights> weights,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {}

MioCodecGlobalEncoderRuntime::~MioCodecGlobalEncoderRuntime() = default;

void MioCodecGlobalEncoderRuntime::ensure_graph(int64_t ssl_frames) const {
    if (graph_ != nullptr && graph_->ssl_frames() == ssl_frames) {
        engine::debug::timing_log_scalar("miocodec.global_encoder.graph.rebuilt", false);
        engine::debug::timing_log_scalar("miocodec.global_encoder.graph.reused", true);
        engine::debug::timing_log_scalar("miocodec.global_encoder.graph.build_ms", 0.0);
        return;
    }
    const auto build_start = Clock::now();
    graph_ = std::make_unique<graphs::MioCodecGlobalEncoderGraph>(
        weights_,
        ssl_frames,
        *execution_context_,
        graph_arena_bytes_);
    engine::debug::timing_log_scalar("miocodec.global_encoder.graph.rebuilt", true);
    engine::debug::timing_log_scalar("miocodec.global_encoder.graph.reused", false);
    engine::debug::timing_log_scalar("miocodec.global_encoder.graph.build_ms", engine::debug::elapsed_ms(build_start));
}

MioCodecGlobalEmbedding MioCodecGlobalEncoderRuntime::encode(
    const std::vector<float> & ssl_features,
    int64_t ssl_frames) const {
    ensure_graph(ssl_frames);
    return graph_->encode(ssl_features);
}

MioCodecContentEncoderRuntime::MioCodecContentEncoderRuntime(
    std::shared_ptr<const MioCodecAssets> assets,
    std::shared_ptr<const MioCodecWeights> weights,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t constant_context_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes),
      constant_context_bytes_(constant_context_bytes) {}

MioCodecContentEncoderRuntime::~MioCodecContentEncoderRuntime() = default;

void MioCodecContentEncoderRuntime::ensure_graph(int64_t ssl_frames) const {
    if (graph_ != nullptr && graph_->ssl_frame_capacity() == ssl_frames) {
        engine::debug::timing_log_scalar("miocodec.content_encoder.graph.rebuilt", false);
        engine::debug::timing_log_scalar("miocodec.content_encoder.graph.reused", true);
        engine::debug::timing_log_scalar("miocodec.content_encoder.graph.build_ms", 0.0);
        return;
    }
    const auto build_start = Clock::now();
    graph_ = std::make_unique<graphs::MioCodecContentEncoderGraph>(
        weights_,
        ssl_frames,
        *execution_context_,
        graph_arena_bytes_,
        constant_context_bytes_);
    engine::debug::timing_log_scalar("miocodec.content_encoder.graph.rebuilt", true);
    engine::debug::timing_log_scalar("miocodec.content_encoder.graph.reused", false);
    engine::debug::timing_log_scalar("miocodec.content_encoder.graph.build_ms", engine::debug::elapsed_ms(build_start));
}

MioCodecContentEmbedding MioCodecContentEncoderRuntime::encode(
    const std::vector<float> & ssl_features,
    int64_t ssl_frames) const {
    ensure_graph(ssl_frames);
    return graph_->encode(ssl_features, ssl_frames);
}

MioCodecWaveDecoderRuntime::MioCodecWaveDecoderRuntime(
    std::shared_ptr<const MioCodecAssets> assets,
    std::shared_ptr<const MioCodecWeights> weights,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t constant_context_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes),
      constant_context_bytes_(constant_context_bytes) {}

MioCodecWaveDecoderRuntime::~MioCodecWaveDecoderRuntime() = default;

void MioCodecWaveDecoderRuntime::ensure_graph(int64_t content_frames, int64_t stft_frames) const {
    if (graph_ != nullptr &&
        graph_->content_frame_capacity() == content_frames &&
        graph_->stft_frame_capacity() == stft_frames) {
        engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.rebuilt", false);
        engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.reused", true);
        engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.build_ms", 0.0);
        return;
    }
    const auto build_start = Clock::now();
    graph_ = std::make_unique<graphs::MioCodecWaveDecoderGraph>(
        weights_,
        content_frames,
        stft_frames,
        *execution_context_,
        graph_arena_bytes_,
        constant_context_bytes_);
    engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.rebuilt", true);
    engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.reused", false);
    engine::debug::timing_log_scalar("miocodec.wave_decoder.graph.build_ms", engine::debug::elapsed_ms(build_start));
}

MioCodecWaveHead MioCodecWaveDecoderRuntime::decode(
    const MioCodecContentEmbedding & content,
    const MioCodecGlobalEmbedding & condition,
    int64_t stft_frames) const {
    ensure_graph(content.frames, stft_frames);
    return graph_->decode(content, condition, stft_frames);
}

MioCodecWaveHead MioCodecWaveDecoderRuntime::decode_tokens(
    const std::vector<int32_t> & content_tokens,
    const MioCodecGlobalEmbedding & condition,
    int64_t stft_frames) const {
    const auto embedding_start = Clock::now();
    const auto content = content_embedding_from_tokens(*weights_, content_tokens);
    engine::debug::timing_log_scalar(
        "miocodec.content_tokens.embedding_ms",
        engine::debug::elapsed_ms(embedding_start));
    return decode(content, condition, stft_frames);
}

}  // namespace engine::models::miocodec
