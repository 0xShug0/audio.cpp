#include "engine/models/seed_vc/astral_quantizer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::seed_vc {
namespace {

const engine::core::TensorValue & require_tensor(
    const SeedVcWeightBundle & weights,
    const std::string & name) {
    const auto it = weights.tensors.find(name);
    if (it == weights.tensors.end()) {
        throw std::runtime_error("Seed-VC missing ASTRAL tensor: " + name);
    }
    return it->second;
}

engine::core::TensorValue contiguous(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value) {
    return engine::core::ensure_backend_addressable_layout(ctx, value);
}

engine::core::TensorValue repeat_like(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & value,
    const engine::core::TensorValue & like) {
    return engine::modules::RepeatModule({like.shape}).build(ctx, value);
}

engine::core::TensorValue transpose_bct_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input) {
    return engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, input);
}

engine::core::TensorValue channel_layer_norm_bct(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_bct,
    const engine::core::TensorValue & weight,
    const engine::core::TensorValue & bias,
    int64_t channels) {
    engine::core::validate_shape(
        weight,
        engine::core::TensorShape::from_dims({channels}),
        "Seed-VC ASTRAL ConvNeXt norm weight");
    engine::core::validate_shape(
        bias,
        engine::core::TensorShape::from_dims({channels}),
        "Seed-VC ASTRAL ConvNeXt norm bias");
    auto x = contiguous(ctx, transpose_bct_btc(ctx, input_bct));
    const auto mean = engine::modules::ReduceMeanModule({2}).build(ctx, x);
    const auto centered = engine::core::wrap_tensor(
        ggml_sub(ctx.ggml, x.tensor, repeat_like(ctx, mean, x).tensor),
        x.shape,
        GGML_TYPE_F32);
    const auto squared = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, centered.tensor, centered.tensor),
        centered.shape,
        GGML_TYPE_F32);
    auto variance = engine::modules::ReduceMeanModule({2}).build(ctx, squared);
    variance = engine::core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, variance.tensor, 1.0F, 1.0e-6F),
        variance.shape,
        GGML_TYPE_F32);
    const auto sqrt_var = engine::core::wrap_tensor(
        ggml_sqrt(ctx.ggml, repeat_like(ctx, variance, x).tensor),
        x.shape,
        GGML_TYPE_F32);
    auto normalized = engine::core::wrap_tensor(
        ggml_div(ctx.ggml, centered.tensor, sqrt_var.tensor),
        x.shape,
        GGML_TYPE_F32);
    const auto weight_view = engine::core::reshape_tensor(
        ctx,
        weight,
        engine::core::TensorShape::from_dims({1, 1, channels}));
    const auto bias_view = engine::core::reshape_tensor(
        ctx,
        bias,
        engine::core::TensorShape::from_dims({1, 1, channels}));
    normalized = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, normalized.tensor, repeat_like(ctx, weight_view, normalized).tensor),
        normalized.shape,
        GGML_TYPE_F32);
    normalized = engine::core::wrap_tensor(
        ggml_add(ctx.ggml, normalized.tensor, repeat_like(ctx, bias_view, normalized).tensor),
        normalized.shape,
        GGML_TYPE_F32);
    return transpose_bct_btc(ctx, normalized);
}

engine::core::TensorValue grn_btc(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input_btc,
    const engine::core::TensorValue & gamma,
    const engine::core::TensorValue & beta,
    int64_t channels) {
    engine::core::validate_shape(
        gamma,
        engine::core::TensorShape::from_dims({1, 1, channels}),
        "Seed-VC ASTRAL GRN gamma");
    engine::core::validate_shape(
        beta,
        engine::core::TensorShape::from_dims({1, 1, channels}),
        "Seed-VC ASTRAL GRN beta");
    const auto x = contiguous(ctx, input_btc);
    const auto squared = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, x.tensor, x.tensor),
        x.shape,
        GGML_TYPE_F32);
    auto sum_time = engine::modules::ReduceSumModule({1}).build(ctx, squared);
    auto gx = engine::core::wrap_tensor(
        ggml_sqrt(ctx.ggml, sum_time.tensor),
        sum_time.shape,
        GGML_TYPE_F32);
    auto gx_mean = engine::modules::ReduceMeanModule({2}).build(ctx, gx);
    gx_mean = engine::core::wrap_tensor(
        ggml_scale_bias(ctx.ggml, gx_mean.tensor, 1.0F, 1.0e-6F),
        gx_mean.shape,
        GGML_TYPE_F32);
    const auto nx = engine::core::wrap_tensor(
        ggml_div(ctx.ggml, repeat_like(ctx, gx, x).tensor, repeat_like(ctx, gx_mean, x).tensor),
        x.shape,
        GGML_TYPE_F32);
    const auto scaled = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, x.tensor, nx.tensor),
        x.shape,
        GGML_TYPE_F32);
    auto out = engine::core::wrap_tensor(
        ggml_mul(ctx.ggml, scaled.tensor, repeat_like(ctx, gamma, x).tensor),
        x.shape,
        GGML_TYPE_F32);
    out = engine::core::wrap_tensor(
        ggml_add(ctx.ggml, out.tensor, repeat_like(ctx, beta, x).tensor),
        x.shape,
        GGML_TYPE_F32);
    return engine::core::wrap_tensor(
        ggml_add(ctx.ggml, out.tensor, x.tensor),
        x.shape,
        GGML_TYPE_F32);
}

struct AstralBlockWeights {
    engine::modules::DepthwiseConv1dWeights dwconv;
    engine::core::TensorValue norm_weight;
    engine::core::TensorValue norm_bias;
    engine::modules::LinearWeights pwconv1;
    engine::core::TensorValue grn_gamma;
    engine::core::TensorValue grn_beta;
    engine::modules::LinearWeights pwconv2;
};

struct AstralWeights {
    engine::modules::Conv1dWeights input_projection;
    std::vector<AstralBlockWeights> blocks;
    engine::modules::LinearWeights quantizer_project_in;
};

AstralWeights load_astral_weights(
    const SeedVcWeightBundle & weights,
    const std::string & prefix,
    int64_t input_channels,
    int64_t channels,
    int64_t intermediate_channels,
    int64_t blocks,
    int64_t code_dim) {
    const std::string root = prefix.empty() ? std::string() : prefix + ".";
    AstralWeights out;
    out.input_projection = engine::modules::Conv1dWeights{
        require_tensor(weights, root + "encoder.input_projection.weight"),
        require_tensor(weights, root + "encoder.input_projection.bias")};
    engine::core::validate_shape(
        out.input_projection.weight,
        engine::core::TensorShape::from_dims({channels, input_channels, 1}),
        "Seed-VC ASTRAL input projection weight");
    engine::core::validate_shape(
        *out.input_projection.bias,
        engine::core::TensorShape::from_dims({channels}),
        "Seed-VC ASTRAL input projection bias");
    out.blocks.reserve(static_cast<size_t>(blocks));
    for (int64_t block = 0; block < blocks; ++block) {
        const std::string base = root + "encoder.blocks." + std::to_string(block);
        AstralBlockWeights item;
        item.dwconv = engine::modules::DepthwiseConv1dWeights{
            require_tensor(weights, base + ".dwconv.weight"),
            require_tensor(weights, base + ".dwconv.bias")};
        engine::core::validate_shape(
            item.dwconv.weight,
            engine::core::TensorShape::from_dims({channels, 1, 7}),
            "Seed-VC ASTRAL depthwise conv weight");
        engine::core::validate_shape(
            *item.dwconv.bias,
            engine::core::TensorShape::from_dims({channels}),
            "Seed-VC ASTRAL depthwise conv bias");
        item.norm_weight = require_tensor(weights, base + ".norm.weight");
        item.norm_bias = require_tensor(weights, base + ".norm.bias");
        item.pwconv1 = engine::modules::LinearWeights{
            require_tensor(weights, base + ".pwconv1.weight"),
            require_tensor(weights, base + ".pwconv1.bias")};
        engine::core::validate_shape(
            item.pwconv1.weight,
            engine::core::TensorShape::from_dims({intermediate_channels, channels}),
            "Seed-VC ASTRAL pwconv1 weight");
        item.grn_gamma = require_tensor(weights, base + ".grn.gamma");
        item.grn_beta = require_tensor(weights, base + ".grn.beta");
        item.pwconv2 = engine::modules::LinearWeights{
            require_tensor(weights, base + ".pwconv2.weight"),
            require_tensor(weights, base + ".pwconv2.bias")};
        engine::core::validate_shape(
            item.pwconv2.weight,
            engine::core::TensorShape::from_dims({channels, intermediate_channels}),
            "Seed-VC ASTRAL pwconv2 weight");
        out.blocks.push_back(item);
    }
    out.quantizer_project_in = engine::modules::LinearWeights{
        require_tensor(weights, root + "quantizer.project_in.weight"),
        require_tensor(weights, root + "quantizer.project_in.bias")};
    engine::core::validate_shape(
        out.quantizer_project_in.weight,
        engine::core::TensorShape::from_dims({code_dim, channels}),
        "Seed-VC ASTRAL quantizer project_in weight");
    return out;
}

engine::core::TensorValue build_astral_projected(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & hidden_btc,
    const AstralWeights & weights,
    int64_t input_channels,
    int64_t channels,
    int64_t intermediate_channels) {
    auto x = transpose_bct_btc(ctx, hidden_btc);
    x = engine::modules::Conv1dModule({input_channels, channels, 1, 1, 0, 1, true})
            .build(ctx, x, weights.input_projection);
    for (const auto & block : weights.blocks) {
        const auto residual = x;
        x = engine::modules::DepthwiseConv1dModule({channels, 7, 1, 3, 1, true})
                .build(ctx, x, block.dwconv);
        x = channel_layer_norm_bct(ctx, x, block.norm_weight, block.norm_bias, channels);
        x = contiguous(ctx, transpose_bct_btc(ctx, x));
        x = engine::modules::LinearModule({channels, intermediate_channels, true})
                .build(ctx, x, block.pwconv1);
        x = engine::modules::GeluModule({engine::modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = grn_btc(ctx, x, block.grn_gamma, block.grn_beta, intermediate_channels);
        x = engine::modules::LinearModule({intermediate_channels, channels, true})
                .build(ctx, x, block.pwconv2);
        x = transpose_bct_btc(ctx, x);
        x = engine::core::wrap_tensor(
            ggml_add(ctx.ggml, contiguous(ctx, x).tensor, contiguous(ctx, residual).tensor),
            x.shape,
            GGML_TYPE_F32);
    }
    x = contiguous(ctx, transpose_bct_btc(ctx, x));
    return engine::modules::LinearModule({channels, weights.quantizer_project_in.weight.shape.dims[0], true})
        .build(ctx, x, weights.quantizer_project_in);
}

std::vector<int32_t> pack_bsq_indices(
    const std::vector<float> & projected,
    int64_t batch,
    int64_t tokens,
    int64_t code_dim) {
    std::vector<int32_t> indices(static_cast<size_t>(batch * tokens), 0);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < tokens; ++t) {
            int32_t value = 0;
            const size_t base = static_cast<size_t>((b * tokens + t) * code_dim);
            for (int64_t bit = 0; bit < code_dim; ++bit) {
                if (projected[base + static_cast<size_t>(bit)] > 0.0F) {
                    value += static_cast<int32_t>(int64_t{1} << (code_dim - bit - 1));
                }
            }
            indices[static_cast<size_t>(b * tokens + t)] = value;
        }
    }
    return indices;
}

class AstralRunner {
public:
    AstralRunner(
        const SeedVcWeightBundle & weights,
        std::string prefix,
        int64_t input_channels,
        int64_t channels,
        int64_t intermediate_channels,
        int64_t blocks,
        int64_t code_dim)
        : source_(weights),
          graph_weights_(load_astral_weights(
              weights,
              prefix,
              input_channels,
              channels,
              intermediate_channels,
              blocks,
              code_dim)),
          input_channels_(input_channels),
          channels_(channels),
          intermediate_channels_(intermediate_channels),
          code_dim_(code_dim) {
        if (source_.execution_context == nullptr) {
            throw std::runtime_error("Seed-VC ASTRAL runner requires execution context");
        }
    }

    ~AstralRunner() {
        release_graph();
    }

    std::vector<float> run_projected(
        const std::vector<float> & hidden_states,
        int64_t batch,
        int64_t tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (batch <= 0 || tokens <= 0) {
            throw std::runtime_error("Seed-VC ASTRAL requires positive batch and token count");
        }
        if (static_cast<int64_t>(hidden_states.size()) != batch * tokens * input_channels_) {
            throw std::runtime_error("Seed-VC ASTRAL hidden state size mismatch");
        }
        ensure_graph(batch, tokens);
        engine::core::write_tensor_f32(input_, hidden_states);
        if (engine::core::compute_backend_graph(source_.execution_context->backend(), graph_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed for Seed-VC ASTRAL");
        }
        return engine::core::read_tensor_f32(output_.tensor);
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
        batch_ = 0;
        tokens_ = 0;
    }

    void ensure_graph(int64_t batch, int64_t tokens) {
        if (ggml_ != nullptr && batch_ == batch && tokens_ == tokens) {
            return;
        }
        release_graph();
        ggml_init_params params{
            1024ull * 1024ull * 512ull,
            nullptr,
            true,
        };
        ggml_ = ggml_init(params);
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize Seed-VC ASTRAL graph context");
        }
        engine::core::ModuleBuildContext ctx{
            ggml_,
            "seed_vc.astral",
            source_.execution_context->backend_type()};
        input_ = engine::core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({batch, tokens, input_channels_}));
        output_ = build_astral_projected(
            ctx,
            input_,
            graph_weights_,
            input_channels_,
            channels_,
            intermediate_channels_);
        graph_ = ggml_new_graph_custom(ggml_, 65536, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(source_.execution_context->backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate Seed-VC ASTRAL graph tensors");
        }
        batch_ = batch;
        tokens_ = tokens;
    }

    const SeedVcWeightBundle & source_;
    AstralWeights graph_weights_;
    int64_t input_channels_ = 0;
    int64_t channels_ = 0;
    int64_t intermediate_channels_ = 0;
    int64_t code_dim_ = 0;
    std::mutex mutex_;
    ggml_context * ggml_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    engine::core::TensorValue input_;
    engine::core::TensorValue output_;
    int64_t batch_ = 0;
    int64_t tokens_ = 0;
};

}  // namespace

struct SeedVcAstralQuantizer::State {
    std::unique_ptr<AstralRunner> runner;
};

SeedVcAstralQuantizer::SeedVcAstralQuantizer(
    std::shared_ptr<const SeedVcWeightBundle> weights,
    std::string prefix,
    int64_t input_channels,
    int64_t channels,
    int64_t intermediate_channels,
    int64_t blocks,
    int64_t codebook_size)
    : weights_(std::move(weights)),
      prefix_(std::move(prefix)),
      input_channels_(input_channels),
      channels_(channels),
      intermediate_channels_(intermediate_channels),
      blocks_(blocks),
      code_dim_(static_cast<int64_t>(std::llround(std::log2(static_cast<double>(codebook_size))))),
      codebook_size_(codebook_size),
      state_(std::make_shared<State>()) {
    if (weights_ == nullptr) {
        throw std::runtime_error("Seed-VC ASTRAL requires weights");
    }
    if (input_channels_ <= 0 || channels_ <= 0 || intermediate_channels_ <= 0 || blocks_ <= 0 ||
        codebook_size_ <= 0 || (int64_t{1} << code_dim_) != codebook_size_) {
        throw std::runtime_error("Seed-VC ASTRAL config is invalid");
    }
    state_->runner = std::make_unique<AstralRunner>(
        *weights_,
        prefix_,
        input_channels_,
        channels_,
        intermediate_channels_,
        blocks_,
        code_dim_);
}

SeedVcAstralQuantizer::~SeedVcAstralQuantizer() = default;
SeedVcAstralQuantizer::SeedVcAstralQuantizer(SeedVcAstralQuantizer &&) noexcept = default;
SeedVcAstralQuantizer & SeedVcAstralQuantizer::operator=(SeedVcAstralQuantizer &&) noexcept = default;

int64_t SeedVcAstralQuantizer::input_channels() const noexcept {
    return input_channels_;
}

int64_t SeedVcAstralQuantizer::channels() const noexcept {
    return channels_;
}

int64_t SeedVcAstralQuantizer::code_dim() const noexcept {
    return code_dim_;
}

int64_t SeedVcAstralQuantizer::codebook_size() const noexcept {
    return codebook_size_;
}

SeedVcAstralQuantizerOutput SeedVcAstralQuantizer::run(
    const std::vector<float> & hidden_states,
    int64_t batch,
    int64_t tokens) const {
    if (state_ == nullptr || state_->runner == nullptr) {
        throw std::runtime_error("Seed-VC ASTRAL is not initialized");
    }
    const auto projected = state_->runner->run_projected(hidden_states, batch, tokens);
    SeedVcAstralQuantizerOutput out;
    out.indices = pack_bsq_indices(projected, batch, tokens, code_dim_);
    out.batch = batch;
    out.tokens = tokens;
    out.code_dim = code_dim_;
    out.codebook_size = codebook_size_;
    return out;
}

}  // namespace engine::models::seed_vc
