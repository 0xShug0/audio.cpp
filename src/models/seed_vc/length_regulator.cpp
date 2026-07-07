#include "engine/models/seed_vc/length_regulator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace engine::models::seed_vc {
namespace {

const engine::core::TensorValue & require_tensor(
    const SeedVcWeightBundle & weights,
    const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("Seed-VC missing length regulator tensor: " + name);
    }
    return it->second;
}

const engine::core::TensorValue * find_tensor(
    const SeedVcWeightBundle & weights,
    const std::string & name) {
    const auto it = weights.tensors.find(name);
    return it == weights.tensors.end() ? nullptr : &it->second;
}

void validate_token_ids(
    const std::vector<int32_t> & token_ids,
    int64_t expected_count,
    int64_t codebook_size) {
    if (expected_count <= 0) {
        throw std::runtime_error("Seed-VC length regulator requires non-empty token ids");
    }
    if (static_cast<int64_t>(token_ids.size()) != expected_count) {
        throw std::runtime_error("Seed-VC length regulator token count mismatch");
    }
    for (const int32_t token : token_ids) {
        if (token < 0 || token >= codebook_size) {
            throw std::runtime_error("Seed-VC length regulator token id is outside the codebook");
        }
    }
}

std::vector<float> build_sequence_mask(
    const std::vector<int64_t> & lengths,
    int64_t output_tokens,
    int64_t channels) {
    if (output_tokens <= 0 || channels <= 0) {
        throw std::runtime_error("Seed-VC length regulator mask dimensions must be positive");
    }
    std::vector<float> mask(static_cast<size_t>(lengths.size() * output_tokens * channels), 0.0F);
    for (size_t batch = 0; batch < lengths.size(); ++batch) {
        const int64_t valid = lengths[batch];
        if (valid <= 0 || valid > output_tokens) {
            throw std::runtime_error("Seed-VC length regulator output length is out of range");
        }
        for (int64_t token = 0; token < valid; ++token) {
            const size_t base = (batch * static_cast<size_t>(output_tokens) + static_cast<size_t>(token)) *
                static_cast<size_t>(channels);
            std::fill(mask.begin() + static_cast<std::ptrdiff_t>(base),
                      mask.begin() + static_cast<std::ptrdiff_t>(base + static_cast<size_t>(channels)),
                      1.0F);
        }
    }
    return mask;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue apply_channel_affine(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int64_t channels) {
    engine::core::TensorShape broadcast_shape = {};
    broadcast_shape.rank = input.shape.rank;
    for (size_t i = 0; i < broadcast_shape.rank; ++i) {
        broadcast_shape.dims[i] = 1;
    }
    broadcast_shape.dims[1] = channels;
    auto weight_view = engine::core::reshape_tensor(
        ctx,
        weight,
        broadcast_shape);
    auto bias_view = engine::core::reshape_tensor(
        ctx,
        bias,
        broadcast_shape);
    auto weight_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, weight_view.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto bias_rep = engine::core::wrap_tensor(
        ggml_repeat(ctx.ggml, bias_view.tensor, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    auto scaled = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, input.tensor, weight_rep.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, scaled.tensor, bias_rep.tensor),
        input.shape,
        GGML_TYPE_F32);
}

engine::core::TensorValue group_norm_1_group(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int64_t channels) {
    engine::core::validate_shape(
        input,
        engine::core::TensorShape::from_dims({input.shape.dims[0], channels, input.shape.dims[2]}),
        "Seed-VC CFM length regulator group norm input");
    const auto input4 = engine::core::reshape_tensor(
        ctx,
        contiguous(ctx, input),
        engine::core::TensorShape::from_dims({input.shape.dims[0], channels, 1, input.shape.dims[2]}));
    const auto normalized = engine::core::wrap_tensor(
        ggml_group_norm(ctx.ggml, input4.tensor, 1, 1.0e-5F),
        input4.shape,
        GGML_TYPE_F32);
    const auto affine = apply_channel_affine(ctx, normalized, weight, bias, channels);
    return engine::core::reshape_tensor(ctx, affine, input.shape);
}

engine::core::TensorValue mish(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    const auto softplus = engine::core::wrap_tensor(
        ggml_softplus(ctx.ggml, input.tensor),
        input.shape,
        GGML_TYPE_F32);
    const auto tanh = engine::core::wrap_tensor(
        ggml_tanh(ctx.ggml, softplus.tensor),
        input.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, input.tensor, tanh.tensor),
        input.shape,
        GGML_TYPE_F32);
}

struct CfmLengthRegulatorWeights {
    engine::core::TensorValue embedding;
    std::array<engine::modules::Conv1dWeights, 4> convs;
    std::array<engine::core::TensorValue, 4> norm_weights;
    std::array<engine::core::TensorValue, 4> norm_biases;
};

struct V1LengthRegulatorWeights {
    engine::modules::LinearWeights content_projection;
    std::optional<engine::core::TensorValue> f0_embedding;
    std::optional<engine::core::TensorValue> f0_mask;
    std::array<engine::modules::Conv1dWeights, 4> convs;
    std::array<engine::core::TensorValue, 4> norm_weights;
    std::array<engine::core::TensorValue, 4> norm_biases;
    engine::modules::Conv1dWeights output_projection;
};

CfmLengthRegulatorWeights load_cfm_weights(
    const SeedVcWeightBundle & weights,
    const std::string & prefix,
    int64_t codebook_size,
    int64_t channels) {
    CfmLengthRegulatorWeights out;
    out.embedding = require_tensor(weights, prefix + ".embedding.weight");
    engine::core::validate_shape(
        out.embedding,
        engine::core::TensorShape::from_dims({codebook_size, channels}),
        "Seed-VC CFM length regulator embedding");
    for (int layer = 0; layer < 4; ++layer) {
        const int conv_index = layer * 3;
        const int norm_index = conv_index + 1;
        out.convs[static_cast<size_t>(layer)] = engine::modules::Conv1dWeights{
            require_tensor(weights, prefix + ".model." + std::to_string(conv_index) + ".weight"),
            require_tensor(weights, prefix + ".model." + std::to_string(conv_index) + ".bias")};
        engine::core::validate_shape(
            out.convs[static_cast<size_t>(layer)].weight,
            engine::core::TensorShape::from_dims({channels, channels, 3}),
            "Seed-VC CFM length regulator conv weight");
        engine::core::validate_shape(
            *out.convs[static_cast<size_t>(layer)].bias,
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC CFM length regulator conv bias");
        out.norm_weights[static_cast<size_t>(layer)] =
            require_tensor(weights, prefix + ".model." + std::to_string(norm_index) + ".weight");
        out.norm_biases[static_cast<size_t>(layer)] =
            require_tensor(weights, prefix + ".model." + std::to_string(norm_index) + ".bias");
        engine::core::validate_shape(
            out.norm_weights[static_cast<size_t>(layer)],
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC CFM length regulator norm weight");
        engine::core::validate_shape(
            out.norm_biases[static_cast<size_t>(layer)],
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC CFM length regulator norm bias");
    }
    return out;
}

V1LengthRegulatorWeights load_v1_weights(
    const SeedVcWeightBundle & weights,
    const std::string & prefix,
    int64_t input_channels,
    int64_t channels,
    int64_t f0_bins) {
    V1LengthRegulatorWeights out;
    out.content_projection = engine::modules::LinearWeights{
        require_tensor(weights, prefix + ".content_in_proj.weight"),
        require_tensor(weights, prefix + ".content_in_proj.bias")};
    engine::core::validate_shape(
        out.content_projection.weight,
        engine::core::TensorShape::from_dims({channels, input_channels}),
        "Seed-VC V1 length regulator content projection weight");
    engine::core::validate_shape(
        *out.content_projection.bias,
        engine::core::TensorShape::from_dims({channels}),
        "Seed-VC V1 length regulator content projection bias");
    if (const auto * f0_embedding = find_tensor(weights, prefix + ".f0_embedding.weight")) {
        out.f0_embedding = *f0_embedding;
        engine::core::validate_shape(
            *out.f0_embedding,
            engine::core::TensorShape::from_dims({f0_bins, channels}),
            "Seed-VC V1 length regulator f0 embedding");
        out.f0_mask = require_tensor(weights, prefix + ".f0_mask");
        engine::core::validate_shape(
            *out.f0_mask,
            engine::core::TensorShape::from_dims({1, channels}),
            "Seed-VC V1 length regulator f0 mask");
    }
    for (int layer = 0; layer < 4; ++layer) {
        const int conv_index = layer * 3;
        const int norm_index = conv_index + 1;
        out.convs[static_cast<size_t>(layer)] = engine::modules::Conv1dWeights{
            require_tensor(weights, prefix + ".model." + std::to_string(conv_index) + ".weight"),
            require_tensor(weights, prefix + ".model." + std::to_string(conv_index) + ".bias")};
        engine::core::validate_shape(
            out.convs[static_cast<size_t>(layer)].weight,
            engine::core::TensorShape::from_dims({channels, channels, 3}),
            "Seed-VC V1 length regulator conv weight");
        engine::core::validate_shape(
            *out.convs[static_cast<size_t>(layer)].bias,
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC V1 length regulator conv bias");
        out.norm_weights[static_cast<size_t>(layer)] =
            require_tensor(weights, prefix + ".model." + std::to_string(norm_index) + ".weight");
        out.norm_biases[static_cast<size_t>(layer)] =
            require_tensor(weights, prefix + ".model." + std::to_string(norm_index) + ".bias");
        engine::core::validate_shape(
            out.norm_weights[static_cast<size_t>(layer)],
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC V1 length regulator norm weight");
        engine::core::validate_shape(
            out.norm_biases[static_cast<size_t>(layer)],
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC V1 length regulator norm bias");
    }
    out.output_projection = engine::modules::Conv1dWeights{
        require_tensor(weights, prefix + ".model.12.weight"),
        require_tensor(weights, prefix + ".model.12.bias")};
    engine::core::validate_shape(
        out.output_projection.weight,
        engine::core::TensorShape::from_dims({channels, channels, 1}),
        "Seed-VC V1 length regulator output projection weight");
    engine::core::validate_shape(
        *out.output_projection.bias,
        engine::core::TensorShape::from_dims({channels}),
        "Seed-VC V1 length regulator output projection bias");
    return out;
}

std::vector<int32_t> f0_to_coarse_ids(
    const std::vector<float> & f0,
    int64_t f0_bins) {
    constexpr double f0_max = 1100.0;
    constexpr double f0_min = 50.0;
    const double f0_mel_min = 1127.0 * std::log(1.0 + f0_min / 700.0);
    const double f0_mel_max = 1127.0 * std::log(1.0 + f0_max / 700.0);
    const double a = static_cast<double>(f0_bins - 2) / (f0_mel_max - f0_mel_min);
    const double b = f0_mel_min * a - 1.0;
    std::vector<int32_t> ids(f0.size(), 0);
    for (size_t index = 0; index < f0.size(); ++index) {
        const float value = f0[index];
        double mel = 1127.0 * std::log(1.0 + static_cast<double>(value) / 700.0);
        if (mel > 0.0) {
            mel = mel * a - b;
        }
        int64_t coarse = static_cast<int64_t>(std::llround(mel));
        coarse = coarse * (coarse > 0);
        coarse = coarse + ((coarse < 1) ? 1 : 0);
        coarse = coarse * (coarse < f0_bins);
        coarse = coarse + ((coarse >= f0_bins) ? (f0_bins - 1) : 0);
        ids[index] = static_cast<int32_t>(coarse);
    }
    return ids;
}

class DiscreteLengthRegulatorRunner {
public:
    DiscreteLengthRegulatorRunner(
        engine::core::ExecutionContext & execution_context,
        engine::core::TensorValue embedding_weight,
        int64_t codebook_size,
        int64_t channels)
        : execution_context_(execution_context),
          embedding_weight_(embedding_weight),
          codebook_size_(codebook_size),
          channels_(channels) {}

    ~DiscreteLengthRegulatorRunner() {
        release_graph();
    }

    SeedVcLengthRegulatorOutput run(
        const std::vector<int32_t> & token_ids,
        int64_t batch,
        int64_t tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch <= 0 || tokens <= 0) {
            throw std::runtime_error("Seed-VC length regulator batch/tokens must be positive");
        }
        validate_token_ids(token_ids, batch * tokens, codebook_size_);
        ensure_graph(batch, tokens);
        engine::core::write_tensor_i32(input_, token_ids);
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC length regulator");
        }
        SeedVcLengthRegulatorOutput out;
        out.values = engine::core::read_tensor_f32(output_.tensor);
        out.batch = batch;
        out.tokens = tokens;
        out.channels = channels_;
        return out;
    }

private:
    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        graph_ = nullptr;
        input_ = {};
        output_ = {};
        graph_batch_ = 0;
        graph_tokens_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t tokens) {
        if (ggml_ != nullptr && graph_batch_ == batch && graph_tokens_ == tokens) {
            return;
        }
        release_graph();

        struct ggml_init_params params = {};
        params.mem_size = 32ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to allocate Seed-VC length regulator graph context");
        }

        engine::core::ModuleBuildContext ctx;
        ctx.ggml = ggml_;
        ctx.backend_type = execution_context_.backend_type();
        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batch, tokens}));
        output_ = engine::modules::EmbeddingModule({codebook_size_, channels_})
            .build(ctx, input_, embedding_weight_);
        output_ = engine::core::wrap_tensor(
            ggml_cont(ctx.ggml, output_.tensor),
            output_.shape,
            output_.type);

        graph_ = ggml_new_graph_custom(ggml_, 4096, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC length regulator graph tensors");
        }
        graph_batch_ = batch;
        graph_tokens_ = tokens;
    }

    engine::core::ExecutionContext & execution_context_;
    engine::core::TensorValue embedding_weight_;
    int64_t codebook_size_ = 0;
    int64_t channels_ = 0;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    engine::core::TensorValue output_;
    int64_t graph_batch_ = 0;
    int64_t graph_tokens_ = 0;
};

class CfmLengthRegulatorRunner {
public:
    CfmLengthRegulatorRunner(
        engine::core::ExecutionContext & execution_context,
        CfmLengthRegulatorWeights weights,
        int64_t codebook_size,
        int64_t channels)
        : execution_context_(execution_context),
          weights_(weights),
          codebook_size_(codebook_size),
          channels_(channels) {}

    ~CfmLengthRegulatorRunner() {
        release_graph();
    }

    SeedVcLengthRegulatorOutput run(
        const std::vector<int32_t> & token_ids,
        const std::vector<int64_t> & output_lengths,
        int64_t batch,
        int64_t tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch <= 0 || tokens <= 0) {
            throw std::runtime_error("Seed-VC CFM length regulator batch/tokens must be positive");
        }
        if (static_cast<int64_t>(output_lengths.size()) != batch) {
            throw std::runtime_error("Seed-VC CFM length regulator length count mismatch");
        }
        const int64_t output_tokens = *std::max_element(output_lengths.begin(), output_lengths.end());
        validate_token_ids(token_ids, batch * tokens, codebook_size_);
        ensure_graph(batch, tokens, output_tokens);
        engine::core::write_tensor_i32(input_, token_ids);
        engine::core::write_tensor_f32(mask_, build_sequence_mask(output_lengths, output_tokens, channels_));
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC CFM length regulator");
        }
        SeedVcLengthRegulatorOutput out;
        out.values = engine::core::read_tensor_f32(output_.tensor);
        out.batch = batch;
        out.tokens = output_tokens;
        out.channels = channels_;
        return out;
    }

private:
    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        graph_ = nullptr;
        input_ = {};
        mask_ = {};
        output_ = {};
        graph_batch_ = 0;
        graph_tokens_ = 0;
        graph_output_tokens_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t tokens, int64_t output_tokens) {
        if (ggml_ != nullptr && graph_batch_ == batch && graph_tokens_ == tokens &&
            graph_output_tokens_ == output_tokens) {
            return;
        }
        release_graph();

        struct ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to allocate Seed-VC CFM length regulator graph context");
        }

        engine::core::ModuleBuildContext ctx;
        ctx.ggml = ggml_;
        ctx.backend_type = execution_context_.backend_type();
        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({batch, tokens}));
        mask_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, output_tokens, channels_}));

        auto x = engine::modules::EmbeddingModule({codebook_size_, channels_})
            .build(ctx, input_, weights_.embedding);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = engine::modules::Interpolate1dModule({output_tokens, engine::modules::Interpolate1dMode::Nearest})
            .build(ctx, x);
        for (size_t layer = 0; layer < weights_.convs.size(); ++layer) {
            x = engine::modules::Conv1dModule({channels_, channels_, 3, 1, 1, 1, true})
                .build(ctx, x, weights_.convs[layer]);
            x = group_norm_1_group(ctx, x, weights_.norm_weights[layer], weights_.norm_biases[layer], channels_);
            x = mish(ctx, x);
        }
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = contiguous(ctx, x);
        output_ = engine::core::wrap_tensor(
            ggml_mul(ctx.ggml, x.tensor, mask_.tensor),
            x.shape,
            GGML_TYPE_F32);

        graph_ = ggml_new_graph_custom(ggml_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC CFM length regulator graph tensors");
        }
        graph_batch_ = batch;
        graph_tokens_ = tokens;
        graph_output_tokens_ = output_tokens;
    }

    engine::core::ExecutionContext & execution_context_;
    CfmLengthRegulatorWeights weights_;
    int64_t codebook_size_ = 0;
    int64_t channels_ = 0;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    engine::core::TensorValue mask_;
    engine::core::TensorValue output_;
    int64_t graph_batch_ = 0;
    int64_t graph_tokens_ = 0;
    int64_t graph_output_tokens_ = 0;
};

class V1LengthRegulatorRunner {
public:
    V1LengthRegulatorRunner(
        engine::core::ExecutionContext & execution_context,
        V1LengthRegulatorWeights weights,
        int64_t input_channels,
        int64_t channels,
        int64_t f0_bins)
        : execution_context_(execution_context),
          weights_(weights),
          input_channels_(input_channels),
          channels_(channels),
          f0_bins_(f0_bins) {}

    ~V1LengthRegulatorRunner() {
        release_graph();
    }

    SeedVcLengthRegulatorOutput run(const SeedVcV1LengthRegulatorInput & request) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (request.batch <= 0 || request.tokens <= 0) {
            throw std::runtime_error("Seed-VC V1 length regulator batch/tokens must be positive");
        }
        if (static_cast<int64_t>(request.content.size()) != request.batch * request.tokens * input_channels_) {
            throw std::runtime_error("Seed-VC V1 length regulator content size mismatch");
        }
        if (static_cast<int64_t>(request.output_lengths.size()) != request.batch) {
            throw std::runtime_error("Seed-VC V1 length regulator length count mismatch");
        }
        if (request.has_f0 && request.f0_tokens <= 0) {
            throw std::runtime_error("Seed-VC V1 length regulator f0 token count must be positive");
        }
        if (request.has_f0 && static_cast<int64_t>(request.f0.size()) != request.batch * request.f0_tokens) {
            throw std::runtime_error("Seed-VC V1 length regulator f0 size mismatch");
        }
        const int64_t output_tokens = *std::max_element(request.output_lengths.begin(), request.output_lengths.end());
        ensure_graph(request.batch, request.tokens, output_tokens, request.has_f0 ? request.f0_tokens : 0);
        engine::core::write_tensor_f32(content_, request.content);
        engine::core::write_tensor_f32(mask_, build_sequence_mask(request.output_lengths, output_tokens, channels_));
        if (request.has_f0) {
            engine::core::write_tensor_i32(f0_ids_, f0_to_coarse_ids(request.f0, f0_bins_));
        }
        if (engine::core::compute_backend_graph(execution_context_.backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC V1 length regulator");
        }
        SeedVcLengthRegulatorOutput out;
        out.values = engine::core::read_tensor_f32(output_.tensor);
        out.batch = request.batch;
        out.tokens = output_tokens;
        out.channels = channels_;
        return out;
    }

private:
    void release_graph() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }
        if (ggml_ != nullptr) {
            ggml_free(ggml_);
            ggml_ = nullptr;
        }
        graph_ = nullptr;
        content_ = {};
        f0_ids_ = {};
        mask_ = {};
        output_ = {};
        graph_batch_ = 0;
        graph_tokens_ = 0;
        graph_output_tokens_ = 0;
        graph_f0_tokens_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t tokens, int64_t output_tokens, int64_t f0_tokens) {
        const bool has_f0 = f0_tokens > 0;
        if (ggml_ != nullptr && graph_batch_ == batch && graph_tokens_ == tokens &&
            graph_output_tokens_ == output_tokens && graph_f0_tokens_ == f0_tokens) {
            return;
        }
        release_graph();

        struct ggml_init_params params = {};
        params.mem_size = 256ull * 1024ull * 1024ull;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to allocate Seed-VC V1 length regulator graph context");
        }

        engine::core::ModuleBuildContext ctx;
        ctx.ggml = ggml_;
        ctx.backend_type = execution_context_.backend_type();
        content_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, tokens, input_channels_}));
        mask_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, output_tokens, channels_}));

        auto x = engine::modules::LinearModule({input_channels_, channels_, true})
            .build(ctx, content_, weights_.content_projection);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = engine::modules::Interpolate1dModule({output_tokens, engine::modules::Interpolate1dMode::Nearest})
            .build(ctx, x);
        if (has_f0) {
            if (!weights_.f0_embedding.has_value()) {
                throw std::runtime_error("Seed-VC V1 length regulator request has F0 but weights are not F0-conditioned");
            }
            f0_ids_ = engine::core::make_tensor(
                ctx,
                GGML_TYPE_I32,
                engine::core::TensorShape::from_dims({batch, f0_tokens}));
            auto f0 = engine::modules::EmbeddingModule({f0_bins_, channels_})
                .build(ctx, f0_ids_, *weights_.f0_embedding);
            f0 = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, f0);
            f0 = engine::modules::Interpolate1dModule({output_tokens, engine::modules::Interpolate1dMode::Nearest})
                .build(ctx, f0);
            x = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, contiguous(ctx, x).tensor, contiguous(ctx, f0).tensor),
                x.shape,
                GGML_TYPE_F32);
        } else if (weights_.f0_mask.has_value()) {
            auto f0_mask = engine::core::reshape_tensor(
                ctx,
                *weights_.f0_mask,
                engine::core::TensorShape::from_dims({1, channels_, 1}));
            auto f0_mask_rep = engine::core::wrap_tensor(
                ggml_repeat(ctx.ggml, f0_mask.tensor, contiguous(ctx, x).tensor),
                x.shape,
                GGML_TYPE_F32);
            x = engine::core::wrap_tensor(
                ggml_add(ctx.ggml, contiguous(ctx, x).tensor, f0_mask_rep.tensor),
                x.shape,
                GGML_TYPE_F32);
        }
        for (size_t layer = 0; layer < weights_.convs.size(); ++layer) {
            x = engine::modules::Conv1dModule({channels_, channels_, 3, 1, 1, 1, true})
                .build(ctx, x, weights_.convs[layer]);
            x = group_norm_1_group(ctx, x, weights_.norm_weights[layer], weights_.norm_biases[layer], channels_);
            x = mish(ctx, x);
        }
        x = engine::modules::Conv1dModule({channels_, channels_, 1, 1, 0, 1, true})
            .build(ctx, x, weights_.output_projection);
        x = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
        x = contiguous(ctx, x);
        output_ = engine::core::wrap_tensor(
            ggml_mul(ctx.ggml, x.tensor, mask_.tensor),
            x.shape,
            GGML_TYPE_F32);

        graph_ = ggml_new_graph_custom(ggml_, 32768, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC V1 length regulator graph tensors");
        }
        graph_batch_ = batch;
        graph_tokens_ = tokens;
        graph_output_tokens_ = output_tokens;
        graph_f0_tokens_ = f0_tokens;
    }

    engine::core::ExecutionContext & execution_context_;
    V1LengthRegulatorWeights weights_;
    int64_t input_channels_ = 0;
    int64_t channels_ = 0;
    int64_t f0_bins_ = 0;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue content_;
    engine::core::TensorValue f0_ids_;
    engine::core::TensorValue mask_;
    engine::core::TensorValue output_;
    int64_t graph_batch_ = 0;
    int64_t graph_tokens_ = 0;
    int64_t graph_output_tokens_ = 0;
    int64_t graph_f0_tokens_ = 0;
};

}  // namespace

struct SeedVcDiscreteLengthRegulator::State {
    std::unique_ptr<DiscreteLengthRegulatorRunner> runner;
};

struct SeedVcCfmLengthRegulator::State {
    std::unique_ptr<CfmLengthRegulatorRunner> runner;
};

struct SeedVcV1LengthRegulator::State {
    std::unique_ptr<V1LengthRegulatorRunner> runner;
};

SeedVcDiscreteLengthRegulator::SeedVcDiscreteLengthRegulator(
    std::shared_ptr<const SeedVcWeightBundle> weights,
    std::string prefix)
    : weights_(std::move(weights)),
      prefix_(std::move(prefix)),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr || weights_->execution_context == nullptr) {
        throw std::runtime_error("Seed-VC length regulator requires loaded weights");
    }
    const std::string embedding_name = prefix_ + ".embedding.weight";
    const auto & embedding = require_tensor(*weights_, embedding_name);
    if (embedding.shape.rank != 2) {
        throw std::runtime_error("Seed-VC length regulator embedding weight must be rank 2");
    }
    codebook_size_ = embedding.shape.dims[0];
    channels_ = embedding.shape.dims[1];
    state_->runner = std::make_unique<DiscreteLengthRegulatorRunner>(
        *weights_->execution_context,
        embedding,
        codebook_size_,
        channels_);
}

SeedVcDiscreteLengthRegulator::~SeedVcDiscreteLengthRegulator() = default;
SeedVcDiscreteLengthRegulator::SeedVcDiscreteLengthRegulator(SeedVcDiscreteLengthRegulator &&) noexcept = default;
SeedVcDiscreteLengthRegulator & SeedVcDiscreteLengthRegulator::operator=(SeedVcDiscreteLengthRegulator &&) noexcept = default;

int64_t SeedVcDiscreteLengthRegulator::codebook_size() const noexcept {
    return codebook_size_;
}

int64_t SeedVcDiscreteLengthRegulator::channels() const noexcept {
    return channels_;
}

SeedVcLengthRegulatorOutput SeedVcDiscreteLengthRegulator::run(
    const std::vector<int32_t> & token_ids,
    int64_t batch,
    int64_t tokens) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC length regulator is not initialized");
    }
    return state_->runner->run(token_ids, batch, tokens);
}

SeedVcCfmLengthRegulator::SeedVcCfmLengthRegulator(
    std::shared_ptr<const SeedVcWeightBundle> weights,
    std::string prefix)
    : weights_(std::move(weights)),
      prefix_(std::move(prefix)),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr || weights_->execution_context == nullptr) {
        throw std::runtime_error("Seed-VC CFM length regulator requires loaded weights");
    }
    const std::string embedding_name = prefix_ + ".embedding.weight";
    const auto & embedding = require_tensor(*weights_, embedding_name);
    if (embedding.shape.rank != 2) {
        throw std::runtime_error("Seed-VC CFM length regulator embedding weight must be rank 2");
    }
    codebook_size_ = embedding.shape.dims[0];
    channels_ = embedding.shape.dims[1];
    auto cfm_weights = load_cfm_weights(*weights_, prefix_, codebook_size_, channels_);
    state_->runner = std::make_unique<CfmLengthRegulatorRunner>(
        *weights_->execution_context,
        cfm_weights,
        codebook_size_,
        channels_);
}

SeedVcCfmLengthRegulator::~SeedVcCfmLengthRegulator() = default;
SeedVcCfmLengthRegulator::SeedVcCfmLengthRegulator(SeedVcCfmLengthRegulator &&) noexcept = default;
SeedVcCfmLengthRegulator & SeedVcCfmLengthRegulator::operator=(SeedVcCfmLengthRegulator &&) noexcept = default;

int64_t SeedVcCfmLengthRegulator::codebook_size() const noexcept {
    return codebook_size_;
}

int64_t SeedVcCfmLengthRegulator::channels() const noexcept {
    return channels_;
}

SeedVcLengthRegulatorOutput SeedVcCfmLengthRegulator::run(
    const std::vector<int32_t> & token_ids,
    const std::vector<int64_t> & output_lengths,
    int64_t batch,
    int64_t tokens) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC CFM length regulator is not initialized");
    }
    return state_->runner->run(token_ids, output_lengths, batch, tokens);
}

SeedVcV1LengthRegulator::SeedVcV1LengthRegulator(
    std::shared_ptr<const SeedVcWeightBundle> weights,
    std::string prefix)
    : weights_(std::move(weights)),
      prefix_(std::move(prefix)),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr || weights_->execution_context == nullptr) {
        throw std::runtime_error("Seed-VC V1 length regulator requires loaded weights");
    }
    const std::string projection_name = prefix_ + ".content_in_proj.weight";
    const auto & projection = require_tensor(*weights_, projection_name);
    if (projection.shape.rank != 2) {
        throw std::runtime_error("Seed-VC V1 length regulator content projection weight must be rank 2");
    }
    channels_ = projection.shape.dims[0];
    input_channels_ = projection.shape.dims[1];
    const std::string f0_embedding_name = prefix_ + ".f0_embedding.weight";
    if (const auto * f0_embedding = find_tensor(*weights_, f0_embedding_name)) {
        if (f0_embedding->shape.rank != 2 || f0_embedding->shape.dims[1] != channels_) {
            throw std::runtime_error("Seed-VC V1 length regulator f0 embedding shape mismatch");
        }
        f0_bins_ = f0_embedding->shape.dims[0];
    }
    auto v1_weights = load_v1_weights(*weights_, prefix_, input_channels_, channels_, f0_bins_);
    state_->runner = std::make_unique<V1LengthRegulatorRunner>(
        *weights_->execution_context,
        v1_weights,
        input_channels_,
        channels_,
        f0_bins_);
}

SeedVcV1LengthRegulator::~SeedVcV1LengthRegulator() = default;
SeedVcV1LengthRegulator::SeedVcV1LengthRegulator(SeedVcV1LengthRegulator &&) noexcept = default;
SeedVcV1LengthRegulator & SeedVcV1LengthRegulator::operator=(SeedVcV1LengthRegulator &&) noexcept = default;

int64_t SeedVcV1LengthRegulator::channels() const noexcept {
    return channels_;
}

int64_t SeedVcV1LengthRegulator::input_channels() const noexcept {
    return input_channels_;
}

int64_t SeedVcV1LengthRegulator::f0_bins() const noexcept {
    return f0_bins_;
}

SeedVcLengthRegulatorOutput SeedVcV1LengthRegulator::run(
    const SeedVcV1LengthRegulatorInput & input) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC V1 length regulator is not initialized");
    }
    return state_->runner->run(input);
}

}  // namespace engine::models::seed_vc
