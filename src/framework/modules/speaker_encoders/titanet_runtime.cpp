#include "engine/framework/modules/speaker_encoders/titanet_runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/optimizations/fast_conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/speaker_encoders/titanet_speaker.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::modules::titanet {

struct BackendConv1dWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t stride = 1;
    int64_t dilation = 1;
    int64_t padding = 0;
    int64_t groups = 1;
    bool use_bias = false;
    modules::Conv1dWeights conv;
    modules::DepthwiseConv1dWeights depthwise;
};

struct BackendSeparableConvBn {
    BackendConv1dWeights depthwise;
    BackendConv1dWeights pointwise;
    modules::BatchNorm1dEvalWeights bn;
    int64_t channels = 0;
};

struct BackendSqueezeExciteWeights {
    modules::Conv1dWeights fc0;
    modules::Conv1dWeights fc2;
    int64_t channels = 0;
    int64_t hidden = 0;
};

struct BackendJasperBlockWeights {
    std::vector<BackendSeparableConvBn> repeats;
    bool has_residual = false;
    BackendConv1dWeights residual_conv;
    modules::BatchNorm1dEvalWeights residual_bn;
    int64_t residual_channels = 0;
    BackendSqueezeExciteWeights se;
};

struct BackendAttentionPoolWeights {
    BackendConv1dWeights tdnn_conv;
    BackendConv1dWeights tdnn_x_conv;
    BackendConv1dWeights tdnn_stats_conv;
    modules::BatchNorm1dEvalWeights tdnn_bn;
    int64_t tdnn_channels = 0;
    BackendConv1dWeights out_conv;
};

struct BackendEmbeddingHeadWeights {
    modules::BatchNorm1dEvalWeights bn;
    int64_t channels = 0;
    BackendConv1dWeights conv;
};

struct TitaNetBackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<BackendJasperBlockWeights> blocks;
    BackendAttentionPoolWeights pool;
    BackendEmbeddingHeadWeights emb;
    core::TensorValue pool_eps;
};

namespace {

constexpr float kPreemph = 0.97f;
constexpr float kNormEps = 1.0e-5f;
constexpr float kBatchNormEps = 1.0e-3f;
constexpr float kPoolStatsEps = 1.0e-10f;

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

core::TensorValue store_weight(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & tensor,
    assets::TensorStorageType storage_type) {
    return store.load_tensor_as_shape(
        source,
        tensor.weight_name,
        storage_type,
        tensor.weight_source_shape,
        core::TensorShape::from_dims({tensor.out_channels, tensor.in_channels / tensor.groups, tensor.kernel}));
}

core::TensorValue store_f32(
    core::BackendWeightStore & store,
    const core::TensorShape & shape,
    const std::vector<float> & values) {
    return store.make_f32(shape, values);
}

BackendConv1dWeights make_backend_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & conv,
    assets::TensorStorageType storage_type) {
    BackendConv1dWeights weights;
    weights.in_channels = conv.in_channels;
    weights.out_channels = conv.out_channels;
    weights.kernel = conv.kernel;
    weights.stride = conv.stride;
    weights.dilation = conv.dilation;
    weights.padding = conv.padding;
    weights.groups = conv.groups;
    weights.use_bias = conv.use_bias;
    weights.conv.weight = store_weight(store, source, conv, storage_type);
    if (conv.use_bias) {
        weights.conv.bias = store.load_f32_tensor(source, *conv.bias_name, {conv.out_channels});
    }
    weights.depthwise.weight = weights.conv.weight;
    weights.depthwise.bias = weights.conv.bias;
    return weights;
}

std::vector<float> read_f32_tensor_values(
    const assets::TensorSource & source,
    const std::string & name,
    const std::vector<int64_t> & expected_shape) {
    const auto tensor = source.require_f32_tensor(name);
    if (tensor.shape.rank != expected_shape.size()) {
        throw std::runtime_error("TitaNet tensor rank mismatch: " + name);
    }
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        if (tensor.shape.dims[i] != expected_shape[i]) {
            throw std::runtime_error("TitaNet tensor shape mismatch: " + name);
        }
    }
    return tensor.values;
}

std::pair<BackendConv1dWeights, BackendConv1dWeights> make_backend_split_pool_conv(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const Conv1dWeights & conv,
    int64_t feature_channels,
    assets::TensorStorageType storage_type) {
    if (conv.kernel != 1 || conv.stride != 1 || conv.padding != 0 || conv.dilation != 1 ||
        conv.groups != 1 || conv.in_channels != feature_channels * 3) {
        throw std::runtime_error("TitaNet attentive pooling projection has unsupported shape");
    }
    auto values = read_f32_tensor_values(source, conv.weight_name, conv.weight_source_shape);
    std::vector<float> x_values(static_cast<size_t>(conv.out_channels * feature_channels), 0.0f);
    std::vector<float> stats_values(static_cast<size_t>(conv.out_channels * feature_channels * 2), 0.0f);
    for (int64_t out = 0; out < conv.out_channels; ++out) {
        for (int64_t in = 0; in < conv.in_channels; ++in) {
            const float value = values[static_cast<size_t>(out * conv.in_channels + in)];
            if (in < feature_channels) {
                x_values[static_cast<size_t>(out * feature_channels + in)] = value;
            } else {
                stats_values[static_cast<size_t>(out * feature_channels * 2 + (in - feature_channels))] = value;
            }
        }
    }

    BackendConv1dWeights x_conv;
    x_conv.in_channels = feature_channels;
    x_conv.out_channels = conv.out_channels;
    x_conv.kernel = 1;
    x_conv.stride = 1;
    x_conv.dilation = 1;
    x_conv.padding = 0;
    x_conv.groups = 1;
    x_conv.use_bias = false;
    x_conv.conv.weight = store.make_from_f32(
        core::TensorShape::from_dims({conv.out_channels, feature_channels, 1}),
        storage_type,
        std::move(x_values));
    x_conv.depthwise.weight = x_conv.conv.weight;

    BackendConv1dWeights stats_conv;
    stats_conv.in_channels = feature_channels * 2;
    stats_conv.out_channels = conv.out_channels;
    stats_conv.kernel = 1;
    stats_conv.stride = 1;
    stats_conv.dilation = 1;
    stats_conv.padding = 0;
    stats_conv.groups = 1;
    stats_conv.use_bias = conv.use_bias;
    stats_conv.conv.weight = store.make_from_f32(
        core::TensorShape::from_dims({conv.out_channels, feature_channels * 2, 1}),
        storage_type,
        std::move(stats_values));
    if (conv.use_bias) {
        stats_conv.conv.bias = store.load_f32_tensor(source, *conv.bias_name, {conv.out_channels});
    }
    stats_conv.depthwise.weight = stats_conv.conv.weight;
    stats_conv.depthwise.bias = stats_conv.conv.bias;
    return {std::move(x_conv), std::move(stats_conv)};
}

modules::BatchNorm1dEvalWeights bind_batch_norm_weights(
    const BatchNorm1dWeights & bn,
    core::BackendWeightStore & store) {
    std::vector<float> scale(static_cast<size_t>(bn.channels), 0.0f);
    std::vector<float> bias(static_cast<size_t>(bn.channels), 0.0f);
    for (int64_t i = 0; i < bn.channels; ++i) {
        const float inv_std = 1.0f / std::sqrt(bn.running_var[static_cast<size_t>(i)] + kBatchNormEps);
        scale[static_cast<size_t>(i)] = bn.weight[static_cast<size_t>(i)] * inv_std;
        bias[static_cast<size_t>(i)] = bn.bias[static_cast<size_t>(i)] - bn.running_mean[static_cast<size_t>(i)] * scale[static_cast<size_t>(i)];
    }
    return {
        store_f32(store, core::TensorShape::from_dims({bn.channels}), scale),
        store_f32(store, core::TensorShape::from_dims({bn.channels}), bias),
    };
}

modules::Conv1dWeights make_backend_se_weight(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & name,
    int64_t out_channels,
    int64_t in_channels,
    assets::TensorStorageType storage_type) {
    return {
        store.load_tensor_as_shape(
            source,
            name,
            storage_type,
            {out_channels, in_channels},
            core::TensorShape::from_dims({out_channels, in_channels, 1})),
        std::nullopt};
}

std::shared_ptr<const TitaNetBackendWeights> load_backend_weights(
    const TitaNetWeights & weights,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    auto out = std::make_shared<TitaNetBackendWeights>();
    out->store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "titanet_spk.weights", 256ull * 1024ull * 1024ull);
    auto & store = *out->store;
    if (weights.source == nullptr) {
        throw std::runtime_error("TitaNet weights require a tensor source");
    }
    const auto & source = *weights.source;
    out->blocks.reserve(weights.blocks.size());
    for (const auto & block : weights.blocks) {
        BackendJasperBlockWeights dst;
        dst.has_residual = block.has_residual;
        dst.repeats.reserve(block.repeats.size());
        for (const auto & repeat : block.repeats) {
            dst.repeats.push_back({
                make_backend_conv(store, source, repeat.depthwise, storage_type),
                make_backend_conv(store, source, repeat.pointwise, storage_type),
                bind_batch_norm_weights(repeat.bn, store),
                repeat.bn.channels,
            });
        }
        if (block.has_residual) {
            dst.residual_conv = make_backend_conv(store, source, block.residual_conv, storage_type);
            dst.residual_bn = bind_batch_norm_weights(block.residual_bn, store);
            dst.residual_channels = block.residual_bn.channels;
        }
        dst.se.fc0 = make_backend_se_weight(store, source, block.se.fc0_weight_name, block.se.hidden, block.se.channels, storage_type);
        dst.se.fc2 = make_backend_se_weight(store, source, block.se.fc2_weight_name, block.se.channels, block.se.hidden, storage_type);
        dst.se.channels = block.se.channels;
        dst.se.hidden = block.se.hidden;
        out->blocks.push_back(std::move(dst));
    }
    if (engine::core::uses_host_graph_plan(backend)) {
        auto [pool_x_conv, pool_stats_conv] = make_backend_split_pool_conv(
            store,
            source,
            weights.pool.tdnn_conv,
            weights.pool.tdnn_conv.in_channels / 3,
            storage_type);
        out->pool.tdnn_x_conv = std::move(pool_x_conv);
        out->pool.tdnn_stats_conv = std::move(pool_stats_conv);
    } else {
        out->pool.tdnn_conv = make_backend_conv(store, source, weights.pool.tdnn_conv, storage_type);
    }
    out->pool.tdnn_bn = bind_batch_norm_weights(weights.pool.tdnn_bn, store);
    out->pool.tdnn_channels = weights.pool.tdnn_bn.channels;
    out->pool.out_conv = make_backend_conv(store, source, weights.pool.out_conv, storage_type);
    out->emb.bn = bind_batch_norm_weights(weights.emb.bn, store);
    out->emb.channels = weights.emb.bn.channels;
    out->emb.conv = make_backend_conv(store, source, weights.emb.conv, storage_type);
    out->pool_eps = store_f32(store, core::TensorShape::from_dims({1, 1, 1}), std::vector<float>{kPoolStatsEps});
    store.upload();
    weights.source->release_storage();
    return out;
}

std::shared_ptr<const TitaNetWeights> require_weights(std::shared_ptr<const TitaNetWeights> weights) {
    if (weights == nullptr) {
        throw std::runtime_error("TitaNet runtime requires weights");
    }
    return weights;
}

core::TensorValue batch_norm_1d(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const modules::BatchNorm1dEvalWeights & bn,
    int64_t channels) {
    return modules::BatchNorm1dEvalModule({channels}).build(
        ctx,
        x,
        bn);
}

core::TensorValue conv1d(core::ModuleBuildContext & ctx, const core::TensorValue & x, const BackendConv1dWeights & conv) {
    if (conv.groups == conv.in_channels && conv.out_channels == conv.in_channels) {
        return modules::DepthwiseConv1dModule({
            conv.in_channels,
            conv.kernel,
            static_cast<int>(conv.stride),
            static_cast<int>(conv.padding),
            static_cast<int>(conv.dilation),
            conv.use_bias,
        }).build(ctx, x, conv.depthwise);
    }
    if (conv.groups != 1) {
        throw std::runtime_error("TitaNet only supports regular or depthwise Conv1d groups");
    }
    return modules::FastConv1dModule({
        conv.in_channels,
        conv.out_channels,
        conv.kernel,
        static_cast<int>(conv.stride),
        static_cast<int>(conv.padding),
        static_cast<int>(conv.dilation),
        conv.use_bias,
    }, modules::FastConv1dKind::MinittsFast1dIm2col).build(ctx, x, conv.conv);
}

core::TensorValue separable_conv_bn(core::ModuleBuildContext & ctx, core::TensorValue x, const BackendSeparableConvBn & layer) {
    x = conv1d(ctx, x, layer.depthwise);
    x = conv1d(ctx, x, layer.pointwise);
    x = batch_norm_1d(ctx, x, layer.bn, layer.channels);
    return x;
}

core::TensorValue concat_channels(core::ModuleBuildContext & ctx, const core::TensorValue & a, const core::TensorValue & b) {
    return modules::ConcatModule({1}).build(ctx, a, b);
}

core::TensorValue squeeze_excite(core::ModuleBuildContext & ctx, const core::TensorValue & x, const BackendSqueezeExciteWeights & se) {
    return modules::SqueezeExcite1dModule({se.channels, se.hidden, false}).build(
        ctx,
        x,
        {se.fc0, se.fc2});
}

core::TensorValue jasper_block(core::ModuleBuildContext & ctx, core::TensorValue x, const BackendJasperBlockWeights & block) {
    auto residual = x;
    for (size_t i = 0; i < block.repeats.size(); ++i) {
        x = separable_conv_bn(ctx, x, block.repeats[i]);
        if (i + 1 != block.repeats.size()) {
            x = modules::ReluModule{}.build(ctx, x);
        }
    }
    x = squeeze_excite(ctx, x, block.se);
    if (block.has_residual) {
        auto res = conv1d(ctx, residual, block.residual_conv);
        res = batch_norm_1d(ctx, res, block.residual_bn, block.residual_channels);
        x = modules::ResidualAddModule{}.build(ctx, x, res);
    }
    x = modules::ReluModule{}.build(ctx, x);
    return x;
}

core::TensorValue attentive_pool(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & x,
    const BackendAttentionPoolWeights & pool,
    core::TensorValue eps) {
    auto mean = modules::ReduceMeanModule({static_cast<int>(x.shape.rank - 1)}).build(ctx, x);
    auto mean_rep = modules::RepeatModule({x.shape}).build(ctx, mean);
    auto centered = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean_rep.tensor), x.shape, GGML_TYPE_F32);
    auto centered_sq = modules::MulModule{}.build(ctx, centered, centered);
    auto var = modules::ReduceMeanModule({static_cast<int>(centered_sq.shape.rank - 1)}).build(ctx, centered_sq);
    auto eps_stats = modules::RepeatModule({var.shape}).build(ctx, eps);
    auto std = modules::SqrtModule{}.build(ctx, modules::AddModule{}.build(ctx, var, eps_stats));
    core::TensorValue attn;
    if (core::uses_host_graph_plan(ctx.backend_type)) {
        auto stats = concat_channels(ctx, mean, std);
        attn = conv1d(ctx, x, pool.tdnn_x_conv);
        auto stats_attn = conv1d(ctx, stats, pool.tdnn_stats_conv);
        auto stats_attn_rep = modules::RepeatModule({attn.shape}).build(ctx, stats_attn);
        attn = modules::AddModule{}.build(ctx, attn, stats_attn_rep);
    } else {
        auto std_rep = modules::RepeatModule({x.shape}).build(ctx, std);
        auto pooled_features = concat_channels(ctx, concat_channels(ctx, x, mean_rep), std_rep);
        attn = conv1d(
            ctx,
            pooled_features,
            pool.tdnn_conv);
    }
    attn = modules::ReluModule{}.build(ctx, attn);
    attn = batch_norm_1d(ctx, attn, pool.tdnn_bn, pool.tdnn_channels);
    attn = modules::TanhModule{}.build(ctx, attn);
    attn = conv1d(ctx, attn, pool.out_conv);
    auto alpha = modules::SoftmaxModule{}.build(ctx, attn);
    auto weighted_x = modules::MulModule{}.build(ctx, x, alpha);
    auto mu = modules::ReduceSumModule({static_cast<int>(weighted_x.shape.rank - 1)}).build(ctx, weighted_x);
    auto mu_rep = modules::RepeatModule({x.shape}).build(ctx, mu);
    auto diff = core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mu_rep.tensor), x.shape, GGML_TYPE_F32);
    auto diff_sq = modules::MulModule{}.build(ctx, diff, diff);
    auto weighted_var = modules::MulModule{}.build(ctx, diff_sq, alpha);
    auto sigma = modules::SqrtModule{}.build(
        ctx,
        modules::AddModule{}.build(
            ctx,
            modules::ReduceSumModule({static_cast<int>(weighted_var.shape.rank - 1)}).build(ctx, weighted_var),
            modules::RepeatModule({mean.shape}).build(ctx, eps)));
    return concat_channels(ctx, mu, sigma);
}

core::TensorValue build_titanet_graph(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TitaNetBackendWeights & weights) {
    auto x = input;
    for (size_t i = 0; i < weights.blocks.size(); ++i) {
        x = jasper_block(ctx, x, weights.blocks[i]);
    }
    x = attentive_pool(ctx, x, weights.pool, weights.pool_eps);
    x = batch_norm_1d(ctx, x, weights.emb.bn, weights.emb.channels);
    x = conv1d(ctx, x, weights.emb.conv);
    return x;
}

}  // namespace

class TitaNetRuntime::Graph {
  public:
    Graph(
        std::shared_ptr<const TitaNetWeights> weights,
        std::shared_ptr<const TitaNetBackendWeights> backend_weights,
        int64_t frames,
        core::ExecutionContext & execution_context)
        : weights_(std::move(weights)),
          backend_weights_(std::move(backend_weights)),
          frames_(frames),
          backend_(execution_context.backend()),
          compute_threads_(std::max(1, execution_context.config().threads)) {
        const auto build_start = Clock::now();
        if (frames_ <= 0) {
            throw std::runtime_error("invalid TitaNet graph shape");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("TitaNet execution backend is not initialized");
        }
        ggml_init_params params{256ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml context");
        }
        core::ModuleBuildContext build_ctx{
            ctx_.get(),
            "titanet_spk",
            execution_context.backend_type(),
        };
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.n_mels, frames_}));
        input_ = input.tensor;
        output_ = build_titanet_graph(build_ctx, input, *backend_weights_).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 8192, false);
        ggml_build_forward_expand(graph_, output_);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate graph");
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            const auto plan_start = Clock::now();
            plan_ = engine::core::create_backend_graph_plan_if_host(backend_, graph_);
            plan_create_ms_ = engine::debug::elapsed_ms(plan_start, Clock::now());
            if (plan_ == nullptr) {
                throw std::runtime_error("failed to create TitaNet graph plan");
            }
        }
        const auto build_end = Clock::now();
        debug::timing_log_scalar("titanet.graph.build_ms", engine::debug::elapsed_ms(build_start, build_end));
        debug::timing_log_scalar("titanet.graph.plan_create_ms", plan_create_ms_);
    }

    ~Graph() {
        if (plan_ != nullptr) {
            engine::core::free_backend_graph_plan(backend_, plan_);
        }
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    bool matches(const TitaNetWeights & weights, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && frames_ == frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    std::vector<float> run(const std::vector<float> & features) {
        if (features.size() != static_cast<size_t>(frames_ * weights_->config.n_mels)) {
            throw std::runtime_error("feature tensor size mismatch");
        }
        if (channels_first_.size() != features.size()) {
            channels_first_.resize(features.size());
        }
        for (int64_t t = 0; t < frames_; ++t) {
            for (int64_t c = 0; c < weights_->config.n_mels; ++c) {
                channels_first_[static_cast<size_t>(t + frames_ * c)] = features[static_cast<size_t>(t * weights_->config.n_mels + c)];
            }
        }
        ggml_backend_tensor_set(input_, channels_first_.data(), 0, channels_first_.size() * sizeof(float));
        core::set_backend_threads(backend_, compute_threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_, plan_);
        ggml_backend_synchronize(backend_);
        const auto compute_end = Clock::now();
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("TitaNet graph compute failed");
        }
        const size_t values = ggml_nelements(output_);
        std::vector<float> out(values, 0.0f);
        ggml_backend_tensor_get(output_, out.data(), 0, out.size() * sizeof(float));
        debug::timing_log_scalar("titanet.graph.compute_ms", engine::debug::elapsed_ms(compute_start, compute_end));
        return out;
    }

  private:
    std::shared_ptr<const TitaNetWeights> weights_;
    std::shared_ptr<const TitaNetBackendWeights> backend_weights_;
    int64_t frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int compute_threads_ = 1;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_backend_graph_plan_t plan_ = nullptr;
    double plan_create_ms_ = 0.0;
    std::vector<float> channels_first_;
};

namespace {

std::vector<float> to_time_major_features(const engine::audio::AudioTensor & features) {
    if (features.shape.size() != 3 || features.shape[0] != 1) {
        throw std::runtime_error("expected single-batch audio feature tensor");
    }
    const int64_t feature_dim = features.shape[1];
    const int64_t frames = features.shape[2];
    std::vector<float> out(static_cast<size_t>(frames * feature_dim), 0.0f);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (int64_t feature = 0; feature < feature_dim; ++feature) {
            out[static_cast<size_t>(frame * feature_dim + feature)] =
                features.values[static_cast<size_t>(feature * frames + frame)];
        }
    }
    return out;
}

std::vector<float> compute_titanet_features(const std::vector<float> & waveform, const TitaNetWeights & weights) {
    const auto start = Clock::now();
    if (waveform.empty()) {
        throw std::runtime_error("waveform is empty");
    }
    const auto & cfg = weights.config;
    if (weights.window.size() != static_cast<size_t>(cfg.win_length)) {
        throw std::runtime_error("unexpected window size in checkpoint");
    }
    if (weights.fb.size() != static_cast<size_t>(cfg.n_mels * (cfg.n_fft / 2 + 1))) {
        throw std::runtime_error("unexpected mel filterbank size in checkpoint");
    }
    std::vector<float> x = waveform;
    engine::audio::apply_preemphasis_in_place(x, cfg.preemph);
    const engine::audio::STFTConfig stft_config{
        cfg.n_fft,
        cfg.hop_length,
        cfg.win_length,
        true,
        engine::audio::STFTPadMode::Constant,
        engine::audio::STFTFamily::Default,
    };
    engine::audio::AudioTensor magnitude;
    const auto stft_start = Clock::now();
    magnitude = engine::audio::STFT().compute_magnitude(
        x,
        weights.window,
        1,
        static_cast<int64_t>(x.size()),
        stft_config);
    debug::timing_log_scalar("audio.stft_ms", engine::debug::elapsed_ms(stft_start, Clock::now()));
    auto sparse_fb = engine::audio::MelFilterbank().prepare_sparse(
        engine::audio::AudioTensor{weights.fb, {cfg.n_mels, cfg.n_fft / 2 + 1}});
    auto log_mel = engine::audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values,
        1,
        magnitude.shape[1],
        magnitude.shape[2],
        magnitude.shape[2],
        sparse_fb);
    constexpr float kLogZeroGuard = 5.960464477539063e-8f;
    for (float & value : log_mel.values) {
        value = std::log(value + kLogZeroGuard);
    }
    const int64_t frames = log_mel.shape[2];
    auto normalized = engine::audio::FeatureNormalizer().compute(
        log_mel.values,
        std::vector<int64_t>{frames},
        1,
        cfg.n_mels,
        frames,
        engine::audio::FeatureNormalizeType::PerFeature);
    auto features = to_time_major_features(normalized.normalized);
    debug::timing_log_scalar("titanet.features_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return features;
}

std::vector<float> normalize_classifier_weights(const TitaNetWeights & weights) {
    const size_t class_count = static_cast<size_t>(weights.num_classes);
    const size_t embedding_size = static_cast<size_t>(weights.config.embedding_size);
    if (weights.classifier_weight.size() != class_count * embedding_size) {
        throw std::runtime_error("classifier weight size mismatch");
    }
    std::vector<float> normalized(weights.classifier_weight.size(), 0.0f);
    for (size_t cls = 0; cls < class_count; ++cls) {
        double w_norm_sq = 0.0;
        const size_t base = cls * embedding_size;
        for (size_t i = 0; i < embedding_size; ++i) {
            const float w = weights.classifier_weight[base + i];
            w_norm_sq += static_cast<double>(w) * static_cast<double>(w);
        }
        const double inv_norm = 1.0 / std::sqrt(std::max(w_norm_sq, static_cast<double>(kNormEps)));
        for (size_t i = 0; i < embedding_size; ++i) {
            normalized[base + i] = static_cast<float>(static_cast<double>(weights.classifier_weight[base + i]) * inv_norm);
        }
    }
    return normalized;
}

std::vector<float> classify_embedding(
    const TitaNetWeights & weights,
    const std::vector<float> & normalized_classifier_weight,
    const std::vector<float> & embedding) {
    const auto start = Clock::now();
    if (embedding.size() != static_cast<size_t>(weights.config.embedding_size)) {
        throw std::runtime_error("embedding size mismatch");
    }
    const size_t class_count = static_cast<size_t>(weights.num_classes);
    const size_t embedding_size = static_cast<size_t>(weights.config.embedding_size);
    if (normalized_classifier_weight.size() != class_count * embedding_size) {
        throw std::runtime_error("normalized classifier weight size mismatch");
    }
    float emb_norm_sq = 0.0f;
    for (float v : embedding) {
        emb_norm_sq += v * v;
    }
    const float inv_emb_norm = 1.0f / std::sqrt(std::max(emb_norm_sq, kNormEps));
    std::vector<float> logits(class_count, 0.0f);
    for (size_t cls = 0; cls < class_count; ++cls) {
        float dot = 0.0f;
        const size_t base = cls * embedding_size;
        for (size_t i = 0; i < embedding_size; ++i) {
            dot += embedding[i] * normalized_classifier_weight[base + i];
        }
        logits[cls] = dot * inv_emb_norm;
    }
    debug::timing_log_scalar("titanet.classifier_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return logits;
}

}  // namespace

std::vector<float> embed_runtime_audio(
    TitaNetRuntime & runtime,
    const TitaNetWeights & weights,
    const runtime::AudioBuffer & audio_buffer) {
    const auto mono = engine::audio::convert_interleaved_audio_to_mono_linear_resampled(
        audio_buffer.samples,
        audio_buffer.sample_rate,
        audio_buffer.channels,
        static_cast<int>(weights.config.sample_rate));
    const auto features = compute_titanet_features(mono, weights);
    const int64_t frames = static_cast<int64_t>(features.size() / static_cast<size_t>(weights.config.n_mels));
    return runtime.embed_features(features, frames);
}

TitaNetClassifyResult classify_runtime_audio(
    TitaNetRuntime & runtime,
    const TitaNetWeights & weights,
    const std::vector<float> & normalized_classifier_weight,
    const std::vector<std::string> & labels,
    const runtime::AudioBuffer & audio) {
    const auto start = Clock::now();
    if (labels.size() != static_cast<size_t>(weights.num_classes)) {
        throw std::runtime_error("label count mismatch");
    }
    auto embedding = embed_runtime_audio(runtime, weights, audio);
    auto logits = classify_embedding(weights, normalized_classifier_weight, embedding);
    const auto best = std::max_element(logits.begin(), logits.end());
    const int index = static_cast<int>(std::distance(logits.begin(), best));
    TitaNetClassifyResult out;
    out.label = labels[static_cast<size_t>(index)];
    out.index = index;
    out.score = *best;
    out.embedding = std::move(embedding);
    out.logits = std::move(logits);
    debug::timing_log_scalar("titanet.classify_audio_ms", engine::debug::elapsed_ms(start, Clock::now()));
    return out;
}

TitaNetRuntime::TitaNetRuntime(
    std::shared_ptr<const TitaNetWeights> weights,
    std::vector<std::string> labels,
    core::ExecutionContext & execution_context,
    assets::TensorStorageType weight_storage_type)
    : weights_(require_weights(std::move(weights))),
      backend_weights_(load_backend_weights(*weights_, execution_context.backend(), execution_context.backend_type(), weight_storage_type)),
      labels_(std::move(labels)),
      normalized_classifier_weight_(normalize_classifier_weights(*weights_)),
      execution_context_(&execution_context) {}

TitaNetRuntime::~TitaNetRuntime() = default;

TitaNetRuntime::Graph & TitaNetRuntime::ensure_graph(int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("frames must be positive");
    }
    if (execution_context_ == nullptr) {
        throw std::runtime_error("TitaNet execution context is not initialized");
    }
    if (!graph_ || !graph_->matches(*weights_, frames, execution_context_->backend(), execution_context_->config().threads)) {
        graph_ = std::make_unique<Graph>(weights_, backend_weights_, frames, *execution_context_);
    }
    return *graph_;
}

std::vector<float> TitaNetRuntime::embed_features(const std::vector<float> & features, int64_t frames) {
    if (frames <= 0) {
        throw std::runtime_error("frames must be positive");
    }
    if (features.size() != static_cast<size_t>(frames * weights_->config.n_mels)) {
        throw std::runtime_error("feature tensor size mismatch");
    }
    return ensure_graph(frames).run(features);
}

std::vector<float> TitaNetRuntime::embed_audio(const runtime::AudioBuffer & audio) {
    return embed_runtime_audio(*this, *weights_, audio);
}

TitaNetClassifyResult TitaNetRuntime::classify_audio(const runtime::AudioBuffer & audio) {
    return classify_runtime_audio(*this, *weights_, normalized_classifier_weight_, labels_, audio);
}

}  // namespace engine::modules::titanet
