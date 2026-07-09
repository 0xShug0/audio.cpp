#include "engine/models/higgs_audio_stt/audio_encoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include "../../framework/modules/attention/attention_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::higgs_audio_stt {

namespace assets = engine::assets;
namespace ai = engine::modules::attention::internal;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultAudioWeightContextBytes = 2048ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct Conv1dWeightsData {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct LinearWeightsData {
    core::TensorValue weight;
    core::TensorValue bias;
};

struct AudioLayerWeights {
    core::TensorValue self_attn_norm_weight;
    core::TensorValue self_attn_norm_bias;
    core::TensorValue q_proj_weight;
    core::TensorValue q_proj_bias;
    core::TensorValue k_proj_weight;
    core::TensorValue v_proj_weight;
    core::TensorValue v_proj_bias;
    core::TensorValue out_proj_weight;
    core::TensorValue out_proj_bias;
    core::TensorValue final_norm_weight;
    core::TensorValue final_norm_bias;
    core::TensorValue fc1_weight;
    core::TensorValue fc1_bias;
    core::TensorValue fc2_weight;
    core::TensorValue fc2_bias;
};

struct HiggsAudioSTTAudioEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    Conv1dWeightsData conv1;
    Conv1dWeightsData conv2;
    core::TensorValue embed_positions;
    std::vector<AudioLayerWeights> layers;
    core::TensorValue layer_norm_weight;
    core::TensorValue layer_norm_bias;
    core::TensorValue temporal_weight;
    core::TensorValue temporal_bias;
    LinearWeightsData linear1;
    LinearWeightsData linear2;
};

Conv1dWeightsData load_conv1d(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_size) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, {out_channels, in_channels, kernel_size}),
        store.load_f32_tensor(source, prefix + ".bias", {out_channels}),
    };
}

LinearWeightsData load_linear(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    int64_t out_features,
    int64_t in_features) {
    return {
        store.load_tensor(source, prefix + ".weight", storage_type, {out_features, in_features}),
        store.load_f32_tensor(source, prefix + ".bias", {out_features}),
    };
}

std::shared_ptr<const HiggsAudioSTTAudioEncoderWeights> load_weights(
    const HiggsAudioSTTAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.audio_encoder;
    const auto & text_config = assets.config.text_decoder;
    const auto & source = *assets.model_weights;
    auto weights = std::make_shared<HiggsAudioSTTAudioEncoderWeights>();
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "higgs_audio_stt.audio_encoder.weights",
        kDefaultAudioWeightContextBytes);
    weights->store = store;
    weights->conv1 = load_conv1d(*store, source, "audio_tower.conv1", storage_type, config.d_model, config.num_mel_bins, 3);
    weights->conv2 = load_conv1d(*store, source, "audio_tower.conv2", storage_type, config.d_model, config.d_model, 3);
    weights->embed_positions = store->load_f32_tensor(
        source,
        "audio_tower.embed_positions.weight",
        {config.max_source_positions, config.d_model});
    weights->layers.reserve(static_cast<size_t>(config.encoder_layers));
    for (int64_t i = 0; i < config.encoder_layers; ++i) {
        const std::string prefix = "audio_tower.layers." + std::to_string(i);
        AudioLayerWeights w;
        w.self_attn_norm_weight = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.weight", {config.d_model});
        w.self_attn_norm_bias = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.bias", {config.d_model});
        w.q_proj_weight = store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.d_model, config.d_model});
        w.q_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.q_proj.bias", {config.d_model});
        w.k_proj_weight = store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.d_model, config.d_model});
        w.v_proj_weight = store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.d_model, config.d_model});
        w.v_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.v_proj.bias", {config.d_model});
        w.out_proj_weight = store->load_tensor(source, prefix + ".self_attn.out_proj.weight", storage_type, {config.d_model, config.d_model});
        w.out_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.out_proj.bias", {config.d_model});
        w.final_norm_weight = store->load_f32_tensor(source, prefix + ".final_layer_norm.weight", {config.d_model});
        w.final_norm_bias = store->load_f32_tensor(source, prefix + ".final_layer_norm.bias", {config.d_model});
        w.fc1_weight = store->load_tensor(source, prefix + ".fc1.weight", storage_type, {config.encoder_ffn_dim, config.d_model});
        w.fc1_bias = store->load_f32_tensor(source, prefix + ".fc1.bias", {config.encoder_ffn_dim});
        w.fc2_weight = store->load_tensor(source, prefix + ".fc2.weight", storage_type, {config.d_model, config.encoder_ffn_dim});
        w.fc2_bias = store->load_f32_tensor(source, prefix + ".fc2.bias", {config.d_model});
        weights->layers.push_back(std::move(w));
    }
    weights->layer_norm_weight = store->load_f32_tensor(source, "audio_tower.layer_norm.weight", {config.d_model});
    weights->layer_norm_bias = store->load_f32_tensor(source, "audio_tower.layer_norm.bias", {config.d_model});
    weights->temporal_weight = store->load_tensor(
        source,
        "audio_encoder_proj.temporal.weight",
        storage_type,
        {config.d_model, 1, 3});
    weights->temporal_bias = store->load_f32_tensor(source, "audio_encoder_proj.temporal.bias", {config.d_model});
    weights->linear1 = load_linear(*store, source, "audio_encoder_proj.linear1", storage_type, text_config.hidden_size, config.d_model);
    weights->linear2 = load_linear(*store, source, "audio_encoder_proj.linear2", storage_type, text_config.hidden_size, text_config.hidden_size);
    store->upload();
    return weights;
}

core::TensorValue audio_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioLayerWeights & weights,
    const HiggsAudioSTTAudioEncoderConfig & config,
    const core::TensorValue & attention_mask) {
    const int64_t head_dim = config.d_model / config.encoder_attention_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    const modules::LinearModule q_proj({config.d_model, config.d_model, true});
    const modules::LinearModule k_proj({config.d_model, config.d_model, false});
    const modules::LinearModule v_proj({config.d_model, config.d_model, true});
    const modules::LinearModule out_proj({config.d_model, config.d_model, true});
    const modules::MatMulModule matmul;

    auto q = q_proj.build(ctx, input, {weights.q_proj_weight, weights.q_proj_bias});
    q = core::wrap_tensor(ggml_scale(ctx.ggml, q.tensor, scale), q.shape, GGML_TYPE_F32);
    q = ai::reshape_heads(ctx, core::ensure_backend_addressable_layout(ctx, q), config.encoder_attention_heads, head_dim);
    auto k = ai::reshape_heads(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k_proj.build(ctx, input, {weights.k_proj_weight, std::nullopt})),
        config.encoder_attention_heads,
        head_dim);
    auto v = ai::reshape_heads(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v_proj.build(ctx, input, {weights.v_proj_weight, weights.v_proj_bias})),
        config.encoder_attention_heads,
        head_dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto scores = matmul.build(ctx, q_heads, modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(ggml_soft_max_ext(ctx.ggml, scores.tensor, attention_mask.tensor, 1.0F, 0.0F), scores.shape, GGML_TYPE_F32);
    auto context = matmul.build(ctx, attn, v_heads);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.d_model}));
    return out_proj.build(ctx, context, {weights.out_proj_weight, weights.out_proj_bias});
}

core::TensorValue audio_encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioLayerWeights & weights,
    const HiggsAudioSTTAudioEncoderConfig & config,
    const core::TensorValue & attention_mask) {
    const modules::LayerNormModule norm({config.d_model, 1.0e-5F, true, true});
    auto attn_in = norm.build(ctx, input, {weights.self_attn_norm_weight, weights.self_attn_norm_bias});
    auto attn = audio_self_attention(ctx, attn_in, weights, config, attention_mask);
    auto x = modules::AddModule().build(ctx, input, attn);
    auto ff_in = norm.build(ctx, x, {weights.final_norm_weight, weights.final_norm_bias});
    auto ff = modules::LinearModule({config.d_model, config.encoder_ffn_dim, true}).build(ctx, ff_in, {weights.fc1_weight, weights.fc1_bias});
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule({config.encoder_ffn_dim, config.d_model, true}).build(ctx, ff, {weights.fc2_weight, weights.fc2_bias});
    return modules::AddModule().build(ctx, x, ff);
}

core::TensorValue avg_pool_time_2x(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    if (input.shape.rank != 3) {
        throw std::runtime_error("Higgs Audio STT avg_pool_time_2x expects [batch, time, hidden]");
    }
    const int64_t batch = input.shape.dims[0];
    const int64_t pooled_time = (input.shape.dims[1] - 2) / 2 + 1;
    const int64_t hidden = input.shape.dims[2];
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    if (input.shape.dims[1] != pooled_time * 2) {
        x = modules::SliceModule({1, 0, pooled_time * 2}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
    }
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, pooled_time, 2, hidden}));
    auto pooled = modules::ReduceMeanModule({2}).build(ctx, x);
    return core::reshape_tensor(ctx, pooled, core::TensorShape::from_dims({batch, pooled_time, hidden}));
}

std::vector<float> attention_mask_values(
    int64_t batch,
    int64_t tokens,
    const std::vector<int32_t> & feature_mask) {
    if (static_cast<int64_t>(feature_mask.size()) != batch * tokens * 2) {
        throw std::runtime_error("Higgs Audio STT feature mask size does not match conv input");
    }
    std::vector<float> values(static_cast<size_t>(batch * tokens * tokens), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t src_offset = b * tokens * 2;
        int64_t valid = 0;
        for (int64_t i = 0; i < tokens * 2; ++i) {
            if (feature_mask[static_cast<size_t>(src_offset + i)] != 0) {
                ++valid;
            }
        }
        const int64_t valid_tokens = higgs_audio_stt_conv_frames(valid);
        for (int64_t row = 0; row < tokens; ++row) {
            for (int64_t col = 0; col < valid_tokens; ++col) {
                values[static_cast<size_t>((b * tokens + row) * tokens + col)] = 1.0F;
            }
        }
    }
    return values;
}

class HiggsAudioSTTAudioEncoderGraph {
public:
    HiggsAudioSTTAudioEncoderGraph(
        std::shared_ptr<const HiggsAudioSTTAssets> assets,
        std::shared_ptr<const HiggsAudioSTTAudioEncoderWeights> weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t chunks,
        int64_t frames)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          chunks_(chunks),
          frames_(frames) {
        if (assets_ == nullptr || weights_ == nullptr) {
            throw std::runtime_error("Higgs Audio STT audio encoder graph requires assets and weights");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Higgs Audio STT audio encoder backend is not initialized");
        }
        if (chunks_ <= 0 || frames_ <= 0) {
            throw std::runtime_error("Higgs Audio STT audio encoder graph requires positive shape");
        }
        const auto build_start = Clock::now();
        const auto & config = assets_->config.audio_encoder;
        const auto & text_config = assets_->config.text_decoder;
        conv_tokens_ = higgs_audio_stt_conv_frames(frames_);
        output_tokens_ = higgs_audio_stt_audio_encoder_token_count(frames_, assets_->config.projector_temporal_downsample);
        if (conv_tokens_ > config.max_source_positions) {
            throw std::runtime_error("Higgs Audio STT audio encoder conv token count exceeds max_source_positions");
        }
        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Higgs Audio STT audio encoder graph context");
        }

        core::ModuleBuildContext ctx{ctx_.get(), "higgs_audio_stt.audio_encoder", backend_type_};
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({chunks_, config.num_mel_bins, frames_}));
        input_ = input.tensor;
        auto x = modules::Conv1dModule({config.num_mel_bins, config.d_model, 3, 1, 1, 1, true})
                     .build(ctx, input, {weights_->conv1.weight, weights_->conv1.bias});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::Conv1dModule({config.d_model, config.d_model, 3, 2, 1, 1, true})
                .build(ctx, x, {weights_->conv2.weight, weights_->conv2.bias});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x);
        auto pos = modules::SliceModule({0, 0, conv_tokens_}).build(ctx, weights_->embed_positions);
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, conv_tokens_, config.d_model}));
        if (chunks_ > 1) {
            pos = modules::RepeatModule({core::TensorShape::from_dims({chunks_, conv_tokens_, config.d_model})}).build(ctx, pos);
        }
        x = modules::AddModule().build(ctx, x, pos);
        auto attention_mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({chunks_, 1, conv_tokens_, conv_tokens_}));
        attention_mask_ = attention_mask.tensor;
        for (const auto & layer : weights_->layers) {
            x = audio_encoder_layer(ctx, x, layer, config, attention_mask);
        }
        x = modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x);
        x = avg_pool_time_2x(ctx, modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x));
        x = modules::LayerNormModule({config.d_model, 1.0e-5F, true, true})
                .build(ctx, x, {weights_->layer_norm_weight, weights_->layer_norm_bias});
        x = modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x);
        x = modules::DepthwiseConv1dModule({
                config.d_model,
                3,
                static_cast<int>(assets_->config.projector_temporal_downsample),
                1,
                1,
                true,
            }).build(ctx, x, {weights_->temporal_weight, weights_->temporal_bias});
        if (x.shape.dims[2] != output_tokens_) {
            throw std::runtime_error("Higgs Audio STT projector temporal output token count mismatch");
        }
        x = modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x);
        x = modules::LinearModule({config.d_model, text_config.hidden_size, true}).build(
            ctx,
            x,
            {weights_->linear1.weight, weights_->linear1.bias});
        x = modules::ReluModule().build(ctx, x);
        x = modules::LinearModule({text_config.hidden_size, text_config.hidden_size, true}).build(
            ctx,
            x,
            {weights_->linear2.weight, weights_->linear2.bias});
        output_ = x.tensor;
        output_dim_ = text_config.hidden_size;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_, graph_) || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Higgs Audio STT audio encoder graph");
        }
        debug::timing_log_scalar("higgs_audio_stt.audio_encoder.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("higgs_audio_stt.audio_encoder.chunks", chunks_);
        debug::trace_log_scalar("higgs_audio_stt.audio_encoder.frames", frames_);
    }

    ~HiggsAudioSTTAudioEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const HiggsAudioSTTAudioEncoderWeights & weights, int64_t chunks, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && chunks_ == chunks && frames_ == frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    HiggsAudioSTTAudioEmbeddings run(const HiggsAudioSTTAudioFeatures & features) {
        const auto & config = assets_->config.audio_encoder;
        if (features.chunks != chunks_ || features.mel_bins != config.num_mel_bins || features.frames != frames_) {
            throw std::runtime_error("Higgs Audio STT audio encoder feature shape mismatch");
        }
        if (static_cast<int64_t>(features.values.size()) != chunks_ * config.num_mel_bins * frames_) {
            throw std::runtime_error("Higgs Audio STT audio encoder feature value count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, features.values.data(), 0, features.values.size() * sizeof(float));
        const auto mask = attention_mask_values(chunks_, conv_tokens_, features.attention_mask);
        ggml_backend_tensor_set(attention_mask_, mask.data(), 0, mask.size() * sizeof(float));
        debug::timing_log_scalar("higgs_audio_stt.audio_encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(backend_, compute_threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("higgs_audio_stt.audio_encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Higgs Audio STT audio encoder graph compute failed");
        }
        std::vector<float> padded(static_cast<size_t>(chunks_ * output_tokens_ * output_dim_));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, padded.data(), 0, padded.size() * sizeof(float));
        debug::timing_log_scalar("higgs_audio_stt.audio_encoder.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));

        HiggsAudioSTTAudioEmbeddings out;
        out.tokens = features.encoder_tokens;
        out.hidden_size = output_dim_;
        out.values.reserve(static_cast<size_t>(out.tokens * out.hidden_size));
        for (int64_t b = 0; b < chunks_; ++b) {
            const int64_t valid_tokens = features.encoder_tokens_per_chunk[static_cast<size_t>(b)];
            if (valid_tokens > output_tokens_) {
                throw std::runtime_error("Higgs Audio STT valid audio token count exceeds graph output");
            }
            const auto begin = padded.begin() + static_cast<std::ptrdiff_t>(b * output_tokens_ * output_dim_);
            out.values.insert(out.values.end(), begin, begin + static_cast<std::ptrdiff_t>(valid_tokens * output_dim_));
        }
        return out;
    }

private:
    std::shared_ptr<const HiggsAudioSTTAssets> assets_;
    std::shared_ptr<const HiggsAudioSTTAudioEncoderWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t chunks_ = 0;
    int64_t frames_ = 0;
    int64_t conv_tokens_ = 0;
    int64_t output_tokens_ = 0;
    int64_t output_dim_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

HiggsAudioSTTAudioEncoderRuntime::HiggsAudioSTTAudioEncoderRuntime(
    std::shared_ptr<const HiggsAudioSTTAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Higgs Audio STT audio encoder requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Higgs Audio STT audio encoder graph arena must be non-zero");
    }
    weights_ = load_weights(*assets_, execution.backend(), execution.backend_type(), weight_storage_type);
}

HiggsAudioSTTAudioEncoderRuntime::~HiggsAudioSTTAudioEncoderRuntime() = default;

HiggsAudioSTTAudioEmbeddings HiggsAudioSTTAudioEncoderRuntime::encode(const HiggsAudioSTTAudioFeatures & features) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Higgs Audio STT audio encoder execution context is null");
    }
    if (features.chunks <= 0 || features.frames <= 0 || features.encoder_tokens <= 0) {
        throw std::runtime_error("Higgs Audio STT audio encoder features are empty");
    }
    if (features.encoder_tokens_per_chunk.size() != static_cast<size_t>(features.chunks)) {
        throw std::runtime_error("Higgs Audio STT audio encoder chunk token metadata mismatch");
    }
    const int threads = std::max(1, execution_->config().threads);
    if (graph_ == nullptr || !graph_->matches(*weights_, features.chunks, features.frames, execution_->backend(), threads)) {
        graph_ = std::make_unique<HiggsAudioSTTAudioEncoderGraph>(
            assets_,
            weights_,
            *execution_,
            graph_arena_bytes_,
            features.chunks,
            features.frames);
    } else {
        debug::timing_log_scalar("higgs_audio_stt.audio_encoder.graph.build_ms", 0.0);
        debug::trace_log_scalar("higgs_audio_stt.audio_encoder.chunks", features.chunks);
        debug::trace_log_scalar("higgs_audio_stt.audio_encoder.frames", features.frames);
    }
    auto out = graph_->run(features);
    if (out.tokens != features.encoder_tokens) {
        throw std::runtime_error("Higgs Audio STT audio encoder output token count mismatch");
    }
    return out;
}

}  // namespace engine::models::higgs_audio_stt
