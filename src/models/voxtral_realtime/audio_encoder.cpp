#include "engine/models/voxtral_realtime/audio_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/transformers/qwen_causal_decoder.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::voxtral_realtime {
namespace {

namespace modules = engine::modules;
using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GgmlBackendBufferDeleter {
    void operator()(ggml_backend_buffer_t buffer) const noexcept {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct AudioLayerWeights {
    core::TensorValue attn_norm;
    core::TensorValue q_weight;
    core::TensorValue q_bias;
    core::TensorValue k_weight;
    core::TensorValue v_weight;
    core::TensorValue v_bias;
    core::TensorValue o_weight;
    core::TensorValue o_bias;
    core::TensorValue final_norm;
    core::TensorValue gate_weight;
    core::TensorValue up_weight;
    core::TensorValue down_weight;
    core::TensorValue down_bias;
};

struct AudioWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue conv1_weight;
    core::TensorValue conv1_bias;
    core::TensorValue conv2_weight;
    core::TensorValue conv2_bias;
    std::vector<AudioLayerWeights> layers;
    core::TensorValue norm;
    core::TensorValue projector1_weight;
    core::TensorValue projector2_weight;
};

assets::TensorStorageType normalize_weight_storage(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
        // Requantized from the source GGUF at load time. Tensors whose rows are not a whole
        // number of blocks (the 3-wide conv kernels) fall back to F32 in the weight store.
        case assets::TensorStorageType::Q4_0:
        case assets::TensorStorageType::Q4_K:
        case assets::TensorStorageType::Q5_K:
        case assets::TensorStorageType::Q6_K:
            return storage_type;
        default:
            throw std::runtime_error(
                "VoxTral audio_encoder_weight_type supports only native, f32, f16, bf16, q4_0, q4_k, q5_k, q6_k, and q8_0");
    }
}

std::shared_ptr<AudioWeights> load_weights(
    const VoxtralRealtimeAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType storage_type) {
    const auto & config = assets.config;
    const auto & audio = config.audio;
    const auto & source = *assets.model_weights;
    auto weights = std::make_shared<AudioWeights>();
    weights->store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "voxtral_realtime.audio_encoder.weights",
        weight_context_bytes);
    auto & store = *weights->store;
    const auto resolved = normalize_weight_storage(storage_type);
    weights->conv1_weight = store.load_tensor(source, "audio_tower.embedder.conv1.weight", resolved,
                                              {audio.hidden_size, config.frontend.feature_size, 3});
    weights->conv1_bias = store.load_f32_tensor(source, "audio_tower.embedder.conv1.bias", {audio.hidden_size});
    weights->conv2_weight = store.load_tensor(source, "audio_tower.embedder.conv2.weight", resolved,
                                              {audio.hidden_size, audio.hidden_size, 3});
    weights->conv2_bias = store.load_f32_tensor(source, "audio_tower.embedder.conv2.bias", {audio.hidden_size});
    weights->layers.reserve(static_cast<size_t>(audio.num_hidden_layers));
    for (int64_t layer = 0; layer < audio.num_hidden_layers; ++layer) {
        const std::string prefix = "audio_tower.layers." + std::to_string(layer);
        AudioLayerWeights w;
        w.attn_norm = store.load_f32_tensor(source, prefix + ".self_attn_layer_norm.weight", {audio.hidden_size});
        w.q_weight = store.load_tensor(source, prefix + ".self_attn.q_proj.weight", resolved,
                                       {audio.num_attention_heads * audio.head_dim, audio.hidden_size});
        w.q_bias = store.load_f32_tensor(source, prefix + ".self_attn.q_proj.bias",
                                         {audio.num_attention_heads * audio.head_dim});
        w.k_weight = store.load_tensor(source, prefix + ".self_attn.k_proj.weight", resolved,
                                       {audio.num_key_value_heads * audio.head_dim, audio.hidden_size});
        w.v_weight = store.load_tensor(source, prefix + ".self_attn.v_proj.weight", resolved,
                                       {audio.num_key_value_heads * audio.head_dim, audio.hidden_size});
        w.v_bias = store.load_f32_tensor(source, prefix + ".self_attn.v_proj.bias",
                                         {audio.num_key_value_heads * audio.head_dim});
        w.o_weight = store.load_tensor(source, prefix + ".self_attn.o_proj.weight", resolved,
                                       {audio.hidden_size, audio.num_attention_heads * audio.head_dim});
        w.o_bias = store.load_f32_tensor(source, prefix + ".self_attn.o_proj.bias", {audio.hidden_size});
        w.final_norm = store.load_f32_tensor(source, prefix + ".final_layer_norm.weight", {audio.hidden_size});
        w.gate_weight = store.load_tensor(source, prefix + ".mlp.gate_proj.weight", resolved,
                                          {audio.intermediate_size, audio.hidden_size});
        w.up_weight = store.load_tensor(source, prefix + ".mlp.up_proj.weight", resolved,
                                        {audio.intermediate_size, audio.hidden_size});
        w.down_weight = store.load_tensor(source, prefix + ".mlp.down_proj.weight", resolved,
                                          {audio.hidden_size, audio.intermediate_size});
        w.down_bias = store.load_f32_tensor(source, prefix + ".mlp.down_proj.bias", {audio.hidden_size});
        weights->layers.push_back(std::move(w));
    }
    weights->norm = store.load_f32_tensor(source, "audio_tower.norm.weight", {audio.hidden_size});
    weights->projector1_weight = store.load_tensor(
        source,
        "multi_modal_projector.linear_1.weight",
        resolved,
        {config.hidden_size, audio.hidden_size * config.downsample_factor});
    weights->projector2_weight = store.load_tensor(
        source,
        "multi_modal_projector.linear_2.weight",
        resolved,
        {config.hidden_size, config.hidden_size});
    const auto upload_start = Clock::now();
    store.upload();
    engine::debug::timing_log_scalar(
        "voxtral_realtime.audio_encoder.weights.upload_ms",
        engine::debug::elapsed_ms(upload_start));
    return weights;
}

core::TensorValue build_audio_attention(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeAudioConfig & config,
    const AudioLayerWeights & weights,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask) {
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * config.head_dim, true});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * config.head_dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * config.head_dim, true});
    const modules::LinearModule o_proj({config.num_attention_heads * config.head_dim, config.hidden_size, true});
    auto q = q_proj.build(ctx, input, {weights.q_weight, weights.q_bias});
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads, config.head_dim}));
    auto k = k_proj.build(ctx, input, {weights.k_weight, std::nullopt});
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    auto v = v_proj.build(ctx, input, {weights.v_weight, weights.v_bias});
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    const modules::RoPEModule rope({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);
    q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    k = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    v = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = modules::ScaledDotProductAttentionModule({
        config.head_dim,
        modules::ScaledDotProductAttentionLowering::Flash,
        GGML_PREC_F32,
    }).build(ctx, q, k, v, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * config.head_dim}));
    return o_proj.build(ctx, context, {weights.o_weight, weights.o_bias});
}

core::TensorValue set_audio_kv_rows(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & cache,
    const core::TensorValue & rows,
    const core::TensorValue & row_indices) {
    core::validate_rank_between(cache, 4, 4, "cache");
    core::validate_rank_between(rows, 4, 4, "rows");
    // ggml_set_rows converts on write, so an f16 cache still takes f32 rows straight from RoPE.
    if ((cache.type != GGML_TYPE_F32 && cache.type != GGML_TYPE_F16) || rows.type != GGML_TYPE_F32) {
        throw std::runtime_error("VoxTral audio streaming cache update requires f32 rows and an f32 or f16 cache");
    }
    if (row_indices.type != GGML_TYPE_I32 || row_indices.shape.rank != 1 ||
        row_indices.shape.dims[0] != rows.shape.dims[1]) {
        throw std::runtime_error("VoxTral audio streaming cache update row-index shape mismatch");
    }
    if (cache.shape.dims[0] != 1 || rows.shape.dims[0] != 1 ||
        cache.shape.dims[2] != rows.shape.dims[2] ||
        cache.shape.dims[3] != rows.shape.dims[3]) {
        throw std::runtime_error("VoxTral audio streaming cache update tensor shape mismatch");
    }
    const int64_t cache_steps = cache.shape.dims[1];
    const int64_t current_steps = rows.shape.dims[1];
    const int64_t row_elems = cache.shape.dims[2] * cache.shape.dims[3];
    auto flat_cache = core::reshape_tensor(ctx, cache, core::TensorShape::from_dims({cache_steps, row_elems}));
    auto flat_rows = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, rows),
        core::TensorShape::from_dims({current_steps, row_elems}));
    auto * updated = ggml_set_rows(ctx.ggml, flat_cache.tensor, flat_rows.tensor, row_indices.tensor);
    auto flat_updated = core::wrap_tensor(updated, flat_cache.shape, cache.type);
    return core::reshape_tensor(ctx, flat_updated, cache.shape);
}

core::TensorValue build_audio_attention_streaming_static(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeAudioConfig & config,
    const AudioLayerWeights & weights,
    const core::TensorValue & input,
    const core::TensorValue & positions,
    const core::TensorValue & attention_mask,
    const core::TensorValue & key_cache,
    const core::TensorValue & value_cache,
    const core::TensorValue & cache_slots) {
    const modules::LinearModule q_proj({config.hidden_size, config.num_attention_heads * config.head_dim, true});
    const modules::LinearModule k_proj({config.hidden_size, config.num_key_value_heads * config.head_dim, false});
    const modules::LinearModule v_proj({config.hidden_size, config.num_key_value_heads * config.head_dim, true});
    const modules::LinearModule o_proj({config.num_attention_heads * config.head_dim, config.hidden_size, true});
    auto q = q_proj.build(ctx, input, {weights.q_weight, weights.q_bias});
    q = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, q),
        core::TensorShape::from_dims({q.shape.dims[0], q.shape.dims[1], config.num_attention_heads, config.head_dim}));
    auto k = k_proj.build(ctx, input, {weights.k_weight, std::nullopt});
    k = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, k),
        core::TensorShape::from_dims({k.shape.dims[0], k.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    auto v = v_proj.build(ctx, input, {weights.v_weight, weights.v_bias});
    v = core::reshape_tensor(
        ctx,
        core::ensure_backend_addressable_layout(ctx, v),
        core::TensorShape::from_dims({v.shape.dims[0], v.shape.dims[1], config.num_key_value_heads, config.head_dim}));
    const modules::RoPEModule rope({config.head_dim, GGML_ROPE_TYPE_NEOX, config.rope_theta});
    q = rope.build(ctx, q, positions);
    k = rope.build(ctx, k, positions);
    q = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    q = core::ensure_backend_addressable_layout(ctx, q);
    k = core::ensure_backend_addressable_layout(ctx, k);
    v = core::ensure_backend_addressable_layout(ctx, v);
    auto updated_key = set_audio_kv_rows(ctx, key_cache, k, cache_slots);
    auto updated_value = set_audio_kv_rows(ctx, value_cache, v, cache_slots);
    // Feed the sliding-window cache to flash attention as a strided view. Materializing the
    // transpose would copy the full 750-step K and V caches once per layer per 80 ms chunk.
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_key.shape.rank}).build(ctx, updated_key);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, updated_value.shape.rank}).build(ctx, updated_value);
    auto context = modules::ScaledDotProductAttentionModule({
        config.head_dim,
        modules::ScaledDotProductAttentionLowering::FlashPreserveViews,
        GGML_PREC_F32,
    }).build(ctx, q, k_heads, v_heads, attention_mask);
    context = core::ensure_backend_addressable_layout(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], config.num_attention_heads * config.head_dim}));
    return o_proj.build(ctx, context, {weights.o_weight, weights.o_bias});
}

core::TensorValue build_audio_mlp(
    core::ModuleBuildContext & ctx,
    const VoxtralRealtimeAudioConfig & config,
    const AudioLayerWeights & weights,
    const core::TensorValue & input) {
    const modules::LinearModule gate({config.hidden_size, config.intermediate_size, false});
    const modules::LinearModule up({config.hidden_size, config.intermediate_size, false});
    const modules::LinearModule down({config.intermediate_size, config.hidden_size, true});
    auto gated = modules::SiluModule().build(ctx, gate.build(ctx, input, {weights.gate_weight, std::nullopt}));
    auto up_value = up.build(ctx, input, {weights.up_weight, std::nullopt});
    auto hidden = modules::MulModule().build(ctx, gated, up_value);
    return down.build(ctx, hidden, {weights.down_weight, weights.down_bias});
}

std::vector<ggml_fp16_t> make_causal_sliding_mask(int64_t steps, int64_t window) {
    std::vector<ggml_fp16_t> values(static_cast<size_t>(steps * steps), ggml_fp32_to_fp16(-INFINITY));
    for (int64_t row = 0; row < steps; ++row) {
        const int64_t begin = std::max<int64_t>(0, row - window + 1);
        for (int64_t col = begin; col <= row; ++col) {
            values[static_cast<size_t>(row * steps + col)] = ggml_fp32_to_fp16(0.0F);
        }
    }
    return values;
}

class AudioGraph {
public:
    AudioGraph(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        std::shared_ptr<const AudioWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        int64_t feature_frames,
        size_t arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          feature_frames_(feature_frames) {
        const auto build_start = Clock::now();
        if (feature_frames_ <= 0) {
            throw std::runtime_error("VoxTral audio encoder graph requires positive feature frames");
        }
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral audio encoder graph context");
        }
        const auto & config = assets_->config;
        core::ModuleBuildContext ctx{ctx_.get(), "voxtral_realtime.audio_encoder", backend_type_};
        auto input_value = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.frontend.feature_size, feature_frames_}));
        input_ = input_value.tensor;
        auto x = input_value;
        x = modules::StreamingConv1dModule({
                config.frontend.feature_size,
                config.audio.hidden_size,
                3,
                1,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {weights_->conv1_weight, weights_->conv1_bias});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::StreamingConv1dModule({
                config.audio.hidden_size,
                config.audio.hidden_size,
                3,
                2,
                1,
                true,
                modules::StreamingPadMode::Constant,
            }).build(ctx, x, {weights_->conv2_weight, weights_->conv2_bias});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        const int64_t encoder_steps = x.shape.dims[1];
        if (encoder_steps % config.downsample_factor != 0) {
            throw std::runtime_error("VoxTral audio encoder steps must be divisible by downsample_factor");
        }
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, encoder_steps);
        mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, encoder_steps, encoder_steps, 1, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({encoder_steps}), GGML_TYPE_I32);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, encoder_steps, encoder_steps}), GGML_TYPE_F16);
        for (const auto & layer : weights_->layers) {
            auto attn_in = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                               .build(ctx, x, {layer.attn_norm, std::nullopt});
            auto attn = build_audio_attention(ctx, config.audio, layer, attn_in, positions, mask);
            x = modules::AddModule().build(ctx, x, attn);
            auto mlp_in = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                              .build(ctx, x, {layer.final_norm, std::nullopt});
            x = modules::AddModule().build(ctx, x, build_audio_mlp(ctx, config.audio, layer, mlp_in));
        }
        x = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                .build(ctx, x, {weights_->norm, std::nullopt});
        const int64_t tokens = encoder_steps / config.downsample_factor;
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, tokens, config.audio.hidden_size * config.downsample_factor}));
        x = modules::LinearModule({config.audio.hidden_size * config.downsample_factor, config.hidden_size, false})
                .build(ctx, x, {weights_->projector1_weight, std::nullopt});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                .build(ctx, x, {weights_->projector2_weight, std::nullopt});
        output_ = x.tensor;
        tokens_ = tokens;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate VoxTral audio encoder graph");
        }
        const auto positions_values = modules::qwen_position_ids(encoder_steps);
        ggml_backend_tensor_set(positions_, positions_values.data(), 0, positions_values.size() * sizeof(int32_t));
        auto mask_values = make_causal_sliding_mask(encoder_steps, config.audio.sliding_window);
        ggml_backend_tensor_set(mask_, mask_values.data(), 0, mask_values.size() * sizeof(ggml_fp16_t));
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.graph_build_ms",
            engine::debug::elapsed_ms(build_start));
    }

    ~AudioGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t feature_frames) const {
        return feature_frames_ == feature_frames;
    }

    VoxtralRealtimeAudioEmbeddings run(const VoxtralRealtimeFeatures & features) {
        const auto total_start = Clock::now();
        const auto & config = assets_->config;
        if (features.frames != feature_frames_ || features.mel_bins != config.frontend.feature_size) {
            throw std::runtime_error("VoxTral audio encoder feature shape mismatch");
        }
        const auto upload_start = Clock::now();
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, config.frontend.feature_size, feature_frames_}), GGML_TYPE_F32), features.values);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.input_upload_ms",
            engine::debug::elapsed_ms(upload_start));
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.graph_compute_ms",
            engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VoxTral audio encoder graph compute failed");
        }
        VoxtralRealtimeAudioEmbeddings out;
        const auto read_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.output_read_ms",
            engine::debug::elapsed_ms(read_start));
        out.tokens = tokens_;
        out.hidden_size = config.hidden_size;
        engine::debug::timing_log_scalar("voxtral_realtime.audio_encoder.tokens", out.tokens);
        engine::debug::timing_log_scalar("voxtral_realtime.audio_encoder.total_ms", engine::debug::elapsed_ms(total_start));
        return out;
    }

private:
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    std::shared_ptr<const AudioWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    int64_t feature_frames_ = 0;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
};

class StreamingAudioCache {
public:
    StreamingAudioCache(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        ggml_backend_t backend,
        int64_t cache_steps)
        : assets_(std::move(assets)),
          backend_(backend),
          cache_steps_(cache_steps) {
        if (assets_ == nullptr || backend_ == nullptr || cache_steps_ <= 0) {
            throw std::runtime_error("VoxTral streaming audio cache requires valid assets, backend, and shape");
        }
        const auto & config = assets_->config;
        ggml_init_params params{16ull * 1024ull * 1024ull, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral streaming audio cache context");
        }
        conv1_cache_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 2, config.frontend.feature_size, 1);
        conv2_cache_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, 1, config.audio.hidden_size, 1);
        key_caches_.reserve(static_cast<size_t>(config.audio.num_hidden_layers));
        value_caches_.reserve(static_cast<size_t>(config.audio.num_hidden_layers));
        // The sliding-window cache is read in full by flash attention on every 80 ms chunk and is
        // never exported, so it is held in F16: half the read traffic, and the type Metal's flash
        // attention kernels prefer.
        for (int64_t layer = 0; layer < config.audio.num_hidden_layers; ++layer) {
            key_caches_.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(
                    ctx_.get(),
                    GGML_TYPE_F16,
                    config.audio.head_dim,
                    config.audio.num_key_value_heads,
                    cache_steps_,
                    1),
                core::TensorShape::from_dims({1, cache_steps_, config.audio.num_key_value_heads, config.audio.head_dim}),
                GGML_TYPE_F16));
            value_caches_.push_back(core::wrap_tensor(
                ggml_new_tensor_4d(
                    ctx_.get(),
                    GGML_TYPE_F16,
                    config.audio.head_dim,
                    config.audio.num_key_value_heads,
                    cache_steps_,
                    1),
                core::TensorShape::from_dims({1, cache_steps_, config.audio.num_key_value_heads, config.audio.head_dim}),
                GGML_TYPE_F16));
        }
        state_buffer_.reset(ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_));
        if (state_buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate VoxTral streaming audio cache tensors");
        }
        zero_all();
    }

    ggml_tensor * conv1_cache() const noexcept {
        return conv1_cache_;
    }

    ggml_tensor * conv2_cache() const noexcept {
        return conv2_cache_;
    }

    const core::TensorValue & key_cache(size_t layer) const {
        return key_caches_.at(layer);
    }

    const core::TensorValue & value_cache(size_t layer) const {
        return value_caches_.at(layer);
    }

    int64_t cache_steps() const noexcept {
        return cache_steps_;
    }

    void zero_all() {
        zero_convs();
        const size_t cache_bytes = ggml_nbytes(key_caches_.front().tensor);
        zero_scratch_.assign(cache_bytes, 0);
        for (size_t layer = 0; layer < key_caches_.size(); ++layer) {
            ggml_backend_tensor_set(key_caches_[layer].tensor, zero_scratch_.data(), 0, cache_bytes);
            ggml_backend_tensor_set(value_caches_[layer].tensor, zero_scratch_.data(), 0, cache_bytes);
        }
        ggml_backend_synchronize(backend_);
    }

    void zero_convs() {
        const auto & config = assets_->config;
        std::vector<float> conv1_zero(static_cast<size_t>(2 * config.frontend.feature_size), 0.0F);
        std::vector<float> conv2_zero(static_cast<size_t>(config.audio.hidden_size), 0.0F);
        ggml_backend_tensor_set(conv1_cache_, conv1_zero.data(), 0, conv1_zero.size() * sizeof(float));
        ggml_backend_tensor_set(conv2_cache_, conv2_zero.data(), 0, conv2_zero.size() * sizeof(float));
    }

private:
    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    int64_t cache_steps_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, GgmlBackendBufferDeleter> state_buffer_;
    ggml_tensor * conv1_cache_ = nullptr;
    ggml_tensor * conv2_cache_ = nullptr;
    std::vector<core::TensorValue> key_caches_;
    std::vector<core::TensorValue> value_caches_;
    std::vector<unsigned char> zero_scratch_;
};

class StreamingAudioGraph {
public:
    StreamingAudioGraph(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        std::shared_ptr<const AudioWeights> weights,
        ggml_backend_t backend,
        core::BackendType backend_type,
        int threads,
        StreamingAudioCache & cache,
        int64_t feature_frames,
        size_t arena_bytes)
        : assets_(std::move(assets)),
          weights_(std::move(weights)),
          backend_(backend),
          backend_type_(backend_type),
          threads_(threads),
          cache_(cache),
          feature_frames_(feature_frames) {
        const auto build_start = Clock::now();
        const auto & config = assets_->config;
        cache_steps_ = cache_.cache_steps();
        if (feature_frames_ <= 0 || cache_steps_ <= 0) {
            throw std::runtime_error("VoxTral streaming audio encoder graph requires positive dimensions");
        }
        ggml_init_params params{arena_bytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize VoxTral streaming audio encoder graph context");
        }
        core::ModuleBuildContext ctx{ctx_.get(), "voxtral_realtime.audio_encoder.streaming", backend_type_};
        auto input_value = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.frontend.feature_size, feature_frames_}));
        input_ = input_value.tensor;
        auto conv1_cache = core::wrap_tensor(cache_.conv1_cache(), core::TensorShape::from_dims({1, config.frontend.feature_size, 2}), GGML_TYPE_F32);
        auto conv2_cache = core::wrap_tensor(cache_.conv2_cache(), core::TensorShape::from_dims({1, config.audio.hidden_size, 1}), GGML_TYPE_F32);
        auto conv1_input = modules::ConcatModule({2}).build(ctx, conv1_cache, input_value);
        auto x = modules::Conv1dModule({
                config.frontend.feature_size,
                config.audio.hidden_size,
                3,
                1,
                0,
                1,
                true,
            }).build(ctx, conv1_input, {weights_->conv1_weight, weights_->conv1_bias});
        next_conv1_cache_ = core::ensure_backend_addressable_layout(
            ctx,
            modules::SliceModule({2, input_value.shape.dims[2] - 2, 2}).build(ctx, input_value))
                                .tensor;
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        auto conv2_input = modules::ConcatModule({2}).build(ctx, conv2_cache, x);
        next_conv2_cache_ = core::ensure_backend_addressable_layout(
            ctx,
            modules::SliceModule({2, x.shape.dims[2] - 1, 1}).build(ctx, x))
                                .tensor;
        x = modules::Conv1dModule({
                config.audio.hidden_size,
                config.audio.hidden_size,
                3,
                2,
                0,
                1,
                true,
            }).build(ctx, conv2_input, {weights_->conv2_weight, weights_->conv2_bias});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::TransposeModule({{0, 2, 1, 3}, x.shape.rank}).build(ctx, x);
        current_steps_ = x.shape.dims[1];
        if (current_steps_ % config.downsample_factor != 0) {
            throw std::runtime_error(
                "VoxTral streaming audio encoder steps must be divisible by downsample_factor; feature_frames=" +
                std::to_string(feature_frames_) + " encoder_steps=" + std::to_string(current_steps_) +
                " downsample_factor=" + std::to_string(config.downsample_factor));
        }
        positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, current_steps_);
        cache_slots_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, current_steps_);
        mask_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F16, cache_steps_, current_steps_, 1, 1);
        auto positions = core::wrap_tensor(positions_, core::TensorShape::from_dims({current_steps_}), GGML_TYPE_I32);
        auto cache_slots = core::wrap_tensor(cache_slots_, core::TensorShape::from_dims({current_steps_}), GGML_TYPE_I32);
        auto mask = core::wrap_tensor(mask_, core::TensorShape::from_dims({1, 1, current_steps_, cache_steps_}), GGML_TYPE_F16);
        for (size_t layer_index = 0; layer_index < weights_->layers.size(); ++layer_index) {
            const auto & layer = weights_->layers[layer_index];
            auto attn_in = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                               .build(ctx, x, {layer.attn_norm, std::nullopt});
            auto attn = build_audio_attention_streaming_static(
                ctx,
                config.audio,
                layer,
                attn_in,
                positions,
                mask,
                cache_.key_cache(layer_index),
                cache_.value_cache(layer_index),
                cache_slots);
            x = modules::AddModule().build(ctx, x, attn);
            auto mlp_in = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                              .build(ctx, x, {layer.final_norm, std::nullopt});
            x = modules::AddModule().build(ctx, x, build_audio_mlp(ctx, config.audio, layer, mlp_in));
        }
        x = modules::RMSNormModule({config.audio.hidden_size, config.audio.rms_norm_eps, true, false})
                .build(ctx, x, {weights_->norm, std::nullopt});
        tokens_ = current_steps_ / config.downsample_factor;
        x = core::ensure_backend_addressable_layout(ctx, x);
        x = core::reshape_tensor(ctx, x, core::TensorShape::from_dims({1, tokens_, config.audio.hidden_size * config.downsample_factor}));
        x = modules::LinearModule({config.audio.hidden_size * config.downsample_factor, config.hidden_size, false})
                .build(ctx, x, {weights_->projector1_weight, std::nullopt});
        x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
        x = modules::LinearModule({config.hidden_size, config.hidden_size, false})
                .build(ctx, x, {weights_->projector2_weight, std::nullopt});
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
        // Expand the encoder body first so the conv-cache writes land after every node that reads
        // the caches: the backend runs nodes in graph order, so this ordering is what keeps the
        // in-graph update from clobbering the current chunk's own left context.
        ggml_build_forward_expand(graph_, output_);
        ggml_build_forward_expand(
            graph_,
            ggml_cpy(ctx_.get(), next_conv1_cache_, cache_.conv1_cache()));
        ggml_build_forward_expand(
            graph_,
            ggml_cpy(ctx_.get(), next_conv2_cache_, cache_.conv2_cache()));
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_.get(), graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            throw std::runtime_error("failed to allocate VoxTral streaming audio encoder graph");
        }
        mask_values_.assign(static_cast<size_t>(current_steps_ * cache_steps_), ggml_fp32_to_fp16(-INFINITY));
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.stream.graph_build_ms",
            engine::debug::elapsed_ms(build_start));
    }

    ~StreamingAudioGraph() {
        engine::core::release_backend_graph_resources(backend_, graph_);
    }

    bool matches(int64_t feature_frames) const {
        return feature_frames_ == feature_frames;
    }

    VoxtralRealtimeAudioEmbeddings run(
        const VoxtralRealtimeFeatures & features,
        VoxtralRealtimeAudioEncoderStreamState & state) {
        const auto & config = assets_->config;
        if (features.frames != feature_frames_ || features.mel_bins != config.frontend.feature_size) {
            throw std::runtime_error("VoxTral streaming audio encoder feature shape mismatch");
        }
        const auto upload_start = Clock::now();
        auto input_value = core::wrap_tensor(input_, core::TensorShape::from_dims({1, config.frontend.feature_size, feature_frames_}), GGML_TYPE_F32);
        core::write_tensor_f32(input_value, features.values);
        if (state.seen_encoder_steps == 0) {
            cache_.zero_convs();
        }
        write_positions(state.seen_encoder_steps);
        write_cache_slots(state.seen_encoder_steps);
        write_mask(state.seen_encoder_steps);
        upload_ms_ += engine::debug::elapsed_ms(upload_start);
        core::set_backend_threads(backend_, threads_);
        const auto compute_start = Clock::now();
        const ggml_status status = core::compute_backend_graph(backend_, graph_);
        ggml_backend_synchronize(backend_);
        compute_ms_ += engine::debug::elapsed_ms(compute_start);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("VoxTral streaming audio encoder graph compute failed");
        }
        state.first_chunk = false;
        state.seen_encoder_steps += current_steps_;
        state.cached_encoder_steps = std::min<int64_t>(cache_steps_, state.cached_encoder_steps + current_steps_);
        VoxtralRealtimeAudioEmbeddings out;
        const auto readback_start = Clock::now();
        out.values = core::read_tensor_f32(output_);
        readback_ms_ += engine::debug::elapsed_ms(readback_start);
        out.tokens = tokens_;
        out.hidden_size = config.hidden_size;
        ++runs_;
        return out;
    }

    void log_timings(const char * prefix) const {
        const std::string base = std::string("voxtral_realtime.audio_encoder.stream.") + prefix;
        engine::debug::timing_log_scalar(base + ".runs", runs_);
        engine::debug::timing_log_scalar(base + ".upload_ms", upload_ms_);
        engine::debug::timing_log_scalar(base + ".compute_ms", compute_ms_);
        engine::debug::timing_log_scalar(base + ".readback_ms", readback_ms_);
    }

private:
    void write_positions(int64_t start) {
        std::vector<int32_t> positions(static_cast<size_t>(current_steps_));
        for (int64_t i = 0; i < current_steps_; ++i) {
            positions[static_cast<size_t>(i)] = static_cast<int32_t>(start + i);
        }
        ggml_backend_tensor_set(positions_, positions.data(), 0, positions.size() * sizeof(int32_t));
    }

    void write_cache_slots(int64_t start) {
        std::vector<int32_t> slots(static_cast<size_t>(current_steps_));
        for (int64_t i = 0; i < current_steps_; ++i) {
            slots[static_cast<size_t>(i)] = static_cast<int32_t>((start + i) % cache_steps_);
        }
        ggml_backend_tensor_set(cache_slots_, slots.data(), 0, slots.size() * sizeof(int32_t));
    }

    void write_mask(int64_t start) {
        const auto masked = ggml_fp32_to_fp16(-INFINITY);
        const auto visible = ggml_fp32_to_fp16(0.0F);
        std::fill(mask_values_.begin(), mask_values_.end(), masked);
        for (int64_t row = 0; row < current_steps_; ++row) {
            const int64_t absolute_step = start + row;
            const int64_t visible_begin = std::max<int64_t>(
                0,
                absolute_step - assets_->config.audio.sliding_window + 1);
            for (int64_t step = visible_begin; step <= absolute_step; ++step) {
                const int64_t slot = step % cache_steps_;
                mask_values_[static_cast<size_t>(row * cache_steps_ + slot)] = visible;
            }
        }
        ggml_backend_tensor_set(mask_, mask_values_.data(), 0, mask_values_.size() * sizeof(ggml_fp16_t));
    }

    std::shared_ptr<const VoxtralRealtimeAssets> assets_;
    std::shared_ptr<const AudioWeights> weights_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    StreamingAudioCache & cache_;
    int64_t feature_frames_ = 0;
    int64_t current_steps_ = 0;
    int64_t cache_steps_ = 0;
    int64_t tokens_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * next_conv1_cache_ = nullptr;
    ggml_tensor * next_conv2_cache_ = nullptr;
    ggml_tensor * positions_ = nullptr;
    ggml_tensor * cache_slots_ = nullptr;
    ggml_tensor * mask_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<ggml_fp16_t> mask_values_;
    ggml_cgraph * graph_ = nullptr;
    int64_t runs_ = 0;
    double upload_ms_ = 0.0;
    double compute_ms_ = 0.0;
    double readback_ms_ = 0.0;
};

}  // namespace

struct VoxtralRealtimeAudioEncoderRuntime::Impl {
    Impl(
        std::shared_ptr<const VoxtralRealtimeAssets> assets,
        core::ExecutionContext & execution,
        size_t graph_arena_bytes,
        size_t weight_context_bytes,
        assets::TensorStorageType storage_type)
        : assets(std::move(assets)),
          backend(execution.backend()),
          backend_type(execution.backend_type()),
          threads(std::max(1, execution.config().threads)),
          graph_arena_bytes(graph_arena_bytes) {
        if (this->assets == nullptr) {
            throw std::runtime_error("VoxTral audio encoder requires assets");
        }
        weights = load_weights(*this->assets, backend, backend_type, weight_context_bytes, storage_type);
        stream_cache = std::make_unique<StreamingAudioCache>(
            this->assets,
            backend,
            this->assets->config.audio.sliding_window);
    }

    VoxtralRealtimeAudioEmbeddings encode(const VoxtralRealtimeFeatures & features) {
        const auto total_start = Clock::now();
        bool rebuilt = false;
        if (graph == nullptr || !graph->matches(features.frames)) {
            rebuilt = true;
            graph = std::make_unique<AudioGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                features.frames,
                graph_arena_bytes);
        }
        auto out = graph->run(features);
        engine::debug::timing_log_scalar("voxtral_realtime.audio_encoder.graph_rebuilt", rebuilt);
        engine::debug::timing_log_scalar(
            "voxtral_realtime.audio_encoder.encode_ms",
            engine::debug::elapsed_ms(total_start));
        return out;
    }

    VoxtralRealtimeAudioEncoderStreamState make_stream_state() const {
        return VoxtralRealtimeAudioEncoderStreamState{};
    }

    VoxtralRealtimeAudioEmbeddings encode_stream_chunk(
        const VoxtralRealtimeFeatures & features,
        VoxtralRealtimeAudioEncoderStreamState & state) {
        const bool first_chunk = state.first_chunk;
        std::unique_ptr<StreamingAudioGraph> & graph_slot = first_chunk ? first_stream_graph : steady_stream_graph;
        if (!first_chunk && graph_slot != nullptr && !graph_slot->matches(features.frames)) {
            throw std::runtime_error("VoxTral steady streaming audio chunk shape changed");
        }
        if (graph_slot == nullptr || !graph_slot->matches(features.frames)) {
            graph_slot = std::make_unique<StreamingAudioGraph>(
                assets,
                weights,
                backend,
                backend_type,
                threads,
                *stream_cache,
                features.frames,
                graph_arena_bytes);
        }
        return graph_slot->run(features, state);
    }

    void log_stream_timings() const {
        if (first_stream_graph != nullptr) {
            first_stream_graph->log_timings("first");
        }
        if (steady_stream_graph != nullptr) {
            steady_stream_graph->log_timings("steady");
        }
    }

    std::shared_ptr<const VoxtralRealtimeAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;
    std::shared_ptr<const AudioWeights> weights;
    std::unique_ptr<StreamingAudioCache> stream_cache;
    std::unique_ptr<AudioGraph> graph;
    std::unique_ptr<StreamingAudioGraph> first_stream_graph;
    std::unique_ptr<StreamingAudioGraph> steady_stream_graph;
};

VoxtralRealtimeAudioEncoderRuntime::VoxtralRealtimeAudioEncoderRuntime(
    std::shared_ptr<const VoxtralRealtimeAssets> assets,
    core::ExecutionContext & execution,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>(
          std::move(assets),
          execution,
          graph_arena_bytes,
          weight_context_bytes,
          weight_storage_type)) {}

VoxtralRealtimeAudioEncoderRuntime::~VoxtralRealtimeAudioEncoderRuntime() = default;

VoxtralRealtimeAudioEmbeddings VoxtralRealtimeAudioEncoderRuntime::encode(const VoxtralRealtimeFeatures & features) {
    return impl_->encode(features);
}

VoxtralRealtimeAudioEncoderStreamState VoxtralRealtimeAudioEncoderRuntime::make_stream_state() const {
    return impl_->make_stream_state();
}

VoxtralRealtimeAudioEmbeddings VoxtralRealtimeAudioEncoderRuntime::encode_stream_chunk(
    const VoxtralRealtimeFeatures & features,
    VoxtralRealtimeAudioEncoderStreamState & state) {
    return impl_->encode_stream_chunk(features, state);
}

void VoxtralRealtimeAudioEncoderRuntime::log_stream_timings() const {
    impl_->log_stream_timings();
}

}  // namespace engine::models::voxtral_realtime
