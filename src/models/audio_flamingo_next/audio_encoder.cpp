#include "engine/models/audio_flamingo_next/audio_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace engine::models::audio_flamingo_next {

namespace assets = engine::assets;
namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

constexpr size_t kAudioWeightContextBytes = 4ull * 1024ull * 1024ull;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
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

struct AFNextAudioEncoderWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
    core::TensorValue embed_positions;
    std::vector<AudioLayerWeights> layers;
    core::TensorValue layer_norm_weight;
    core::TensorValue layer_norm_bias;
};

std::shared_ptr<const AFNextAudioEncoderWeights> load_weights(
    const AFNextAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config.audio_encoder;
    const auto & source = *assets.model_weights;
    auto weights = std::make_shared<AFNextAudioEncoderWeights>();
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "audio_flamingo_next.audio_encoder.weights",
        kAudioWeightContextBytes);
    weights->store = store;
    weights->conv1 = modules::binding::conv1d_from_source(*store, source, "audio_tower.conv1", storage_type, config.hidden_size, config.num_mel_bins, 3, true);
    weights->conv2 = modules::binding::conv1d_from_source(*store, source, "audio_tower.conv2", storage_type, config.hidden_size, config.hidden_size, 3, true);
    weights->embed_positions = store->load_f32_tensor(
        source,
        "audio_tower.embed_positions.weight",
        {config.max_source_positions, config.hidden_size});
    weights->layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string prefix = "audio_tower.layers." + std::to_string(i);
        AudioLayerWeights w;
        w.self_attn_norm_weight = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.weight", {config.hidden_size});
        w.self_attn_norm_bias = store->load_f32_tensor(source, prefix + ".self_attn_layer_norm.bias", {config.hidden_size});
        w.q_proj_weight = store->load_tensor(source, prefix + ".self_attn.q_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
        w.q_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.q_proj.bias", {config.hidden_size});
        w.k_proj_weight = store->load_tensor(source, prefix + ".self_attn.k_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
        w.v_proj_weight = store->load_tensor(source, prefix + ".self_attn.v_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
        w.v_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.v_proj.bias", {config.hidden_size});
        w.out_proj_weight = store->load_tensor(source, prefix + ".self_attn.out_proj.weight", storage_type, {config.hidden_size, config.hidden_size});
        w.out_proj_bias = store->load_f32_tensor(source, prefix + ".self_attn.out_proj.bias", {config.hidden_size});
        w.final_norm_weight = store->load_f32_tensor(source, prefix + ".final_layer_norm.weight", {config.hidden_size});
        w.final_norm_bias = store->load_f32_tensor(source, prefix + ".final_layer_norm.bias", {config.hidden_size});
        w.fc1_weight = store->load_tensor(source, prefix + ".fc1.weight", storage_type, {config.intermediate_size, config.hidden_size});
        w.fc1_bias = store->load_f32_tensor(source, prefix + ".fc1.bias", {config.intermediate_size});
        w.fc2_weight = store->load_tensor(source, prefix + ".fc2.weight", storage_type, {config.hidden_size, config.intermediate_size});
        w.fc2_bias = store->load_f32_tensor(source, prefix + ".fc2.bias", {config.hidden_size});
        weights->layers.push_back(std::move(w));
    }
    weights->layer_norm_weight = store->load_f32_tensor(source, "audio_tower.layer_norm.weight", {config.hidden_size});
    weights->layer_norm_bias = store->load_f32_tensor(source, "audio_tower.layer_norm.bias", {config.hidden_size});
    store->upload();
    return weights;
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    const auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    return core::reshape_tensor(ctx, contiguous, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

core::TensorValue af_next_self_attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioLayerWeights & weights,
    const AFNextAudioEncoderConfig & config,
    const core::TensorValue & attention_mask) {
    const int64_t head_dim = config.hidden_size / config.num_attention_heads;
    const modules::LinearModule q_proj({config.hidden_size, config.hidden_size, true});
    const modules::LinearModule k_proj({config.hidden_size, config.hidden_size, false});
    const modules::LinearModule v_proj({config.hidden_size, config.hidden_size, true});
    const modules::LinearModule out_proj({config.hidden_size, config.hidden_size, true});

    auto q = q_proj.build(ctx, input, {weights.q_proj_weight, weights.q_proj_bias});
    q = reshape_heads(ctx, q, config.num_attention_heads, head_dim);
    auto k = reshape_heads(ctx, k_proj.build(ctx, input, {weights.k_proj_weight, std::nullopt}), config.num_attention_heads, head_dim);
    auto v = reshape_heads(ctx, v_proj.build(ctx, input, {weights.v_proj_weight, weights.v_proj_bias}), config.num_attention_heads, head_dim);
    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        head_dim, modules::ScaledDotProductAttentionLowering::Flash})
        .build(ctx, q_heads, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.hidden_size}));
    return out_proj.build(ctx, context, {weights.out_proj_weight, weights.out_proj_bias});
}

core::TensorValue audio_encoder_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const AudioLayerWeights & weights,
    const AFNextAudioEncoderConfig & config,
    const core::TensorValue & attention_mask) {
    const modules::LayerNormModule norm({config.hidden_size, 1.0e-5F, true, true});
    auto attn_in = norm.build(ctx, input, {weights.self_attn_norm_weight, weights.self_attn_norm_bias});
    auto attn = af_next_self_attention(ctx, attn_in, weights, config, attention_mask);
    auto x = modules::AddModule().build(ctx, input, attn);
    auto ff_in = norm.build(ctx, x, {weights.final_norm_weight, weights.final_norm_bias});
    auto ff = modules::LinearModule({config.hidden_size, config.intermediate_size, true}).build(
        ctx,
        ff_in,
        {weights.fc1_weight, weights.fc1_bias});
    ff = modules::GeluModule().build(ctx, ff);
    ff = modules::LinearModule({config.intermediate_size, config.hidden_size, true}).build(
        ctx,
        ff,
        {weights.fc2_weight, weights.fc2_bias});
    return modules::AddModule().build(ctx, x, ff);
}

core::TensorValue avg_pool_time_2x(core::ModuleBuildContext & ctx, const core::TensorValue & input) {
    if (input.shape.rank != 3 || input.shape.dims[1] % 2 != 0) {
        throw std::runtime_error("Audio Flamingo Next avg_pool_time_2x expects [batch, even_time, hidden]");
    }
    const int64_t batch = input.shape.dims[0];
    const int64_t pooled_time = input.shape.dims[1] / 2;
    const int64_t hidden = input.shape.dims[2];
    auto x = core::ensure_backend_addressable_layout(ctx, input);
    x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({batch, pooled_time, 2, hidden}));
    auto pooled = modules::ReduceMeanModule({2}).build(ctx, x);
    return core::reshape_tensor(ctx, pooled, core::TensorShape::from_dims({batch, pooled_time, hidden}));
}

std::vector<float> attention_mask_values(
    int64_t batch,
    int64_t tokens,
    const std::vector<int32_t> & feature_mask) {
    if (static_cast<int64_t>(feature_mask.size()) != batch * tokens * 2) {
        throw std::runtime_error("Audio Flamingo Next feature mask size does not match conv input");
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
        const int64_t valid_tokens = af_next_conv2_length(valid);
        for (int64_t row = 0; row < tokens; ++row) {
            for (int64_t col = valid_tokens; col < tokens; ++col) {
                values[static_cast<size_t>((b * tokens + row) * tokens + col)] = -INFINITY;
            }
        }
    }
    return values;
}

class AFNextAudioEncoderGraph {
public:
    AFNextAudioEncoderGraph(
        std::shared_ptr<const AFNextAssets> assets,
        std::shared_ptr<const AFNextAudioEncoderWeights> weights,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        int64_t batch,
        int64_t frames)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          compute_threads_(std::max(1, execution.config().threads)),
          batch_(batch),
          frames_(frames) {
        if (assets_ == nullptr || weights_ == nullptr) {
            throw std::runtime_error("Audio Flamingo Next audio encoder graph requires assets and weights");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("Audio Flamingo Next audio encoder backend is not initialized");
        }
        const auto build_start = Clock::now();
        const auto & config = assets_->config.audio_encoder;
        if (frames_ != config.max_source_positions * 2 || batch_ <= 0) {
            throw std::runtime_error("Audio Flamingo Next audio encoder graph expects 30-second feature windows");
        }
        const int64_t conv_tokens = af_next_conv2_length(frames_);
        const int64_t pooled_tokens = (conv_tokens - 2) / 2 + 1;

        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize Audio Flamingo Next audio encoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "audio_flamingo_next.audio_encoder", backend_type_};
        auto input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({batch_, config.num_mel_bins, frames_}));
        input_ = input.tensor;
        auto x = modules::Conv1dModule({config.num_mel_bins, config.hidden_size, 3, 1, 1, 1, true})
                     .build(ctx, input, {weights_->conv1.weight, weights_->conv1.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::Conv1dModule({config.hidden_size, config.hidden_size, 3, 2, 1, 1, true})
                .build(ctx, x, {weights_->conv2.weight, weights_->conv2.bias});
        x = modules::GeluModule().build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1}, x.shape.rank}).build(ctx, x);
        auto pos = weights_->embed_positions;
        pos = core::reshape_tensor(ctx, pos, core::TensorShape::from_dims({1, conv_tokens, config.hidden_size}));
        if (batch_ > 1) {
            pos = modules::RepeatModule({core::TensorShape::from_dims({batch_, conv_tokens, config.hidden_size})}).build(ctx, pos);
        }
        x = modules::AddModule().build(ctx, x, pos);
        auto attention_mask = core::make_tensor(ctx, GGML_TYPE_F16, core::TensorShape::from_dims({batch_, 1, conv_tokens, conv_tokens}));
        attention_mask_ = attention_mask.tensor;
        // Preserve cached input storage across graph evaluations.
        ggml_set_input(attention_mask_);
        ggml_set_output(attention_mask_);
        for (const auto & layer : weights_->layers) {
            x = audio_encoder_layer(ctx, x, layer, config, attention_mask);
        }
        x = avg_pool_time_2x(ctx, x);
        x = modules::LayerNormModule({config.hidden_size, 1.0e-5F, true, true})
                .build(ctx, x, {weights_->layer_norm_weight, weights_->layer_norm_bias});
        output_ = x.tensor;
        output_tokens_ = pooled_tokens;
        output_dim_ = config.hidden_size;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("failed to allocate Audio Flamingo Next audio encoder graph");
        }
        debug::timing_log_scalar("audio_flamingo_next.audio_encoder.graph.build_ms", engine::debug::elapsed_ms(build_start, Clock::now()));
        debug::trace_log_scalar("audio_flamingo_next.audio_encoder.batch", batch_);
        debug::trace_log_scalar("audio_flamingo_next.audio_encoder.frames", frames_);
    }

    ~AFNextAudioEncoderGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(const AFNextAudioEncoderWeights & weights, int64_t batch, int64_t frames, ggml_backend_t backend, int threads) const {
        return weights_.get() == &weights && batch_ == batch && frames_ == frames && backend_ == backend && compute_threads_ == std::max(1, threads);
    }

    AFNextAudioEncoderOutput run(const AFNextAudioFeatures & features) {
        const auto & config = assets_->config.audio_encoder;
        if (features.batch != batch_ || features.frames != frames_ || features.mel_bins != config.num_mel_bins) {
            throw std::runtime_error("Audio Flamingo Next audio encoder feature shape mismatch");
        }
        if (static_cast<int64_t>(features.values.size()) != batch_ * config.num_mel_bins * frames_) {
            throw std::runtime_error("Audio Flamingo Next audio encoder feature value count mismatch");
        }
        auto timing_start = Clock::now();
        ggml_backend_tensor_set(input_, features.values.data(), 0, features.values.size() * sizeof(float));
        if (cached_feature_mask_ != features.attention_mask) {
            const auto mask = attention_mask_values(batch_, af_next_conv2_length(frames_), features.attention_mask);
            std::vector<ggml_fp16_t> mask_f16(mask.size());
            ggml_fp32_to_fp16_row(mask.data(), mask_f16.data(), mask.size());
            ggml_backend_tensor_set(attention_mask_, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));
            cached_feature_mask_ = features.attention_mask;
        }
        debug::timing_log_scalar("audio_flamingo_next.audio_encoder.input_upload_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        core::set_backend_threads(backend_, compute_threads_);
        timing_start = Clock::now();
        const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("audio_flamingo_next.audio_encoder.graph.compute_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Audio Flamingo Next audio encoder graph compute failed");
        }
        AFNextAudioEncoderOutput out;
        out.batch = batch_;
        out.tokens = output_tokens_;
        out.hidden_size = output_dim_;
        out.values.resize(static_cast<size_t>(out.batch * out.tokens * out.hidden_size));
        timing_start = Clock::now();
        ggml_backend_tensor_get(output_, out.values.data(), 0, out.values.size() * sizeof(float));
        debug::timing_log_scalar("audio_flamingo_next.audio_encoder.output_read_ms", engine::debug::elapsed_ms(timing_start, Clock::now()));
        return out;
    }

private:
    std::shared_ptr<const AFNextAssets> assets_;
    std::shared_ptr<const AFNextAudioEncoderWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int compute_threads_ = 1;
    int64_t batch_ = 0;
    int64_t frames_ = 0;
    int64_t output_tokens_ = 0;
    int64_t output_dim_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * attention_mask_ = nullptr;
    std::vector<int32_t> cached_feature_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

AFNextAudioEncoderRuntime::AFNextAudioEncoderRuntime(
    std::shared_ptr<const AFNextAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    assets::TensorStorageType weight_storage_type)
    : assets_(std::move(assets)),
      execution_(&execution),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr) {
        throw std::runtime_error("Audio Flamingo Next audio encoder requires assets");
    }
    if (graph_arena_bytes_ == 0) {
        throw std::runtime_error("Audio Flamingo Next audio encoder graph arena must be non-zero");
    }
    weights_ = load_weights(*assets_, execution.backend(), execution.backend_type(), weight_storage_type);
}

AFNextAudioEncoderRuntime::~AFNextAudioEncoderRuntime() = default;

AFNextAudioEncoderOutput AFNextAudioEncoderRuntime::encode(const AFNextAudioFeatures & features) {
    if (execution_ == nullptr) {
        throw std::runtime_error("Audio Flamingo Next audio encoder execution context is null");
    }
    const int threads = std::max(1, execution_->config().threads);
    if (graph_ == nullptr || !graph_->matches(*weights_, features.batch, features.frames, execution_->backend(), threads)) {
        graph_ = std::make_unique<AFNextAudioEncoderGraph>(
            assets_,
            weights_,
            *execution_,
            graph_arena_bytes_,
            features.batch,
            features.frames);
    } else {
        debug::timing_log_scalar("audio_flamingo_next.audio_encoder.graph.build_ms", 0.0);
        debug::trace_log_scalar("audio_flamingo_next.audio_encoder.batch", features.batch);
        debug::trace_log_scalar("audio_flamingo_next.audio_encoder.frames", features.frames);
    }
    return graph_->run(features);
}

}  // namespace engine::models::audio_flamingo_next
