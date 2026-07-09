#include "engine/models/ace_step/cover_tokenizer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/attention/qwen_decoder.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "helper_utils.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

class GgmlContextDeleter {
public:
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

modules::QwenDecoderLayerWeights load_pooler_layer_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    assets::TensorStorageType storage_type,
    const AceStepDiffusionConfig & config) {
    const int64_t dim = ace_step_diffusion_attention_head_dim(config, "ACE-Step cover tokenizer");
    modules::QwenDecoderLayerWeights layer;
    layer.input_norm.weight = store.load_f32_tensor(source, prefix + ".input_layernorm.weight", {config.hidden_size});
    layer.post_norm.weight = store.load_f32_tensor(source, prefix + ".post_attention_layernorm.weight", {config.hidden_size});
    layer.q_norm.weight = store.load_f32_tensor(source, prefix + ".self_attn.q_norm.weight", {dim});
    layer.k_norm.weight = store.load_f32_tensor(source, prefix + ".self_attn.k_norm.weight", {dim});
    layer.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.num_attention_heads * dim, config.hidden_size});
    layer.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    layer.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.num_key_value_heads * dim, config.hidden_size});
    layer.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.hidden_size, config.num_attention_heads * dim});
    layer.mlp.gate_proj = {
        store.load_tensor(source, prefix + ".mlp.gate_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
        std::nullopt,
    };
    layer.mlp.up_proj = {
        store.load_tensor(source, prefix + ".mlp.up_proj.weight", storage_type, {config.intermediate_size, config.hidden_size}),
        std::nullopt,
    };
    layer.mlp.down_proj = {
        store.load_tensor(source, prefix + ".mlp.down_proj.weight", storage_type, {config.hidden_size, config.intermediate_size}),
        std::nullopt,
    };
    return layer;
}

std::shared_ptr<const AceStepCoverTokenizerWeights> load_cover_tokenizer_weights(
    ggml_backend_t backend,
    core::BackendType backend_type,
    const AceStepAssets & assets,
    assets::TensorStorageType storage_type) {
    auto store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "ace_step.cover_tokenizer.weights",
        256ull * 1024ull * 1024ull);
    const auto & config = assets.config.diffusion;
    const auto & source = *assets.dit_weights;
    auto weights = std::make_shared<AceStepCoverTokenizerWeights>();
    weights->store = store;
    weights->audio_acoustic_proj = {
        store->load_tensor(source, "tokenizer.audio_acoustic_proj.weight", storage_type, {config.hidden_size, config.latent_channels}),
        store->load_tensor(source, "tokenizer.audio_acoustic_proj.bias", assets::TensorStorageType::F32, {config.hidden_size}),
    };
    weights->attention_pooler_embed_tokens = {
        store->load_tensor(source, "tokenizer.attention_pooler.embed_tokens.weight", storage_type, {config.hidden_size, config.hidden_size}),
        store->load_tensor(source, "tokenizer.attention_pooler.embed_tokens.bias", assets::TensorStorageType::F32, {config.hidden_size}),
    };
    weights->attention_pooler_special_token_host = source.require_f32(
        "tokenizer.attention_pooler.special_token",
        {1, 1, config.hidden_size});
    weights->attention_pooler_layers.layers.reserve(static_cast<size_t>(config.num_attention_pooler_hidden_layers));
    for (int64_t i = 0; i < config.num_attention_pooler_hidden_layers; ++i) {
        weights->attention_pooler_layers.layers.push_back(load_pooler_layer_weights(
            *store,
            source,
            "tokenizer.attention_pooler.layers." + std::to_string(i),
            storage_type,
            config));
    }
    weights->attention_pooler_norm = store->load_f32_tensor(
        source,
        "tokenizer.attention_pooler.norm.weight",
        {config.hidden_size});
    weights->quantizer_project_in = {
        store->load_tensor(source, "tokenizer.quantizer.project_in.weight", storage_type, {config.fsq_dim, config.hidden_size}),
        store->load_tensor(source, "tokenizer.quantizer.project_in.bias", assets::TensorStorageType::F32, {config.fsq_dim}),
    };
    store->upload();
    assets.dit_weights->release_storage();
    return weights;
}

std::vector<int32_t> fsq_indices_from_projected(
    const AceStepFsqQuantizerTable & table,
    const std::vector<float> & values,
    int64_t rows) {
    const int64_t dim = static_cast<int64_t>(table.levels.size());
    if (rows < 0 || static_cast<int64_t>(values.size()) != rows * dim) {
        throw std::runtime_error("ACE-Step cover tokenizer FSQ input shape is invalid");
    }
    std::vector<int32_t> indices(static_cast<size_t>(rows), 0);
    for (int64_t r = 0; r < rows; ++r) {
        int64_t index = 0;
        for (int64_t d = 0; d < dim; ++d) {
            const int64_t level = table.levels[static_cast<size_t>(d)];
            const float clamp_value = 1.0F + 1.0F / static_cast<float>(level - 1);
            const float soft_clamped =
                std::tanh(values[static_cast<size_t>(r * dim + d)] / clamp_value) * clamp_value;
            const float hard_clamped = std::clamp(soft_clamped, -1.0F, 1.0F);
            const float bracket =
                std::floor((static_cast<float>(level - 1) * (hard_clamped + 1.0F) * 0.5F) + 0.5F);
            const int64_t component = std::clamp<int64_t>(
                static_cast<int64_t>(bracket),
                0,
                level - 1);
            index += component * table.basis[static_cast<size_t>(d)];
        }
        indices[static_cast<size_t>(r)] = static_cast<int32_t>(index);
    }
    return indices;
}

}  // namespace

class AceStepCoverTokenizerRuntime::Impl {
public:
    static constexpr int64_t kMaxGraphCodeCapacity = 512;

    class Graph {
    public:
        Graph(
            std::shared_ptr<const AceStepAssets> assets,
            ggml_backend_t backend,
            int threads,
            std::shared_ptr<const AceStepCoverTokenizerWeights> weights,
            int64_t code_capacity)
            : assets_(std::move(assets)),
              backend_(backend),
              threads_(threads),
              weights_(std::move(weights)),
              code_capacity_(code_capacity) {
            if (backend_ == nullptr) {
                throw std::runtime_error("ACE-Step cover tokenizer backend initialization failed");
            }
            if (code_capacity_ <= 0) {
                throw std::runtime_error("ACE-Step cover tokenizer requires positive code capacity");
            }
            build();
        }

        ~Graph() {
            if (backend_ != nullptr && graph_ != nullptr) {
                engine::core::release_backend_graph_resources(backend_, graph_);
            }
            if (buffer_ != nullptr) {
                ggml_backend_buffer_free(buffer_);
            }
        }

        bool can_run(int64_t code_count) const noexcept {
            return code_capacity_ >= code_count;
        }

        int64_t code_capacity() const noexcept {
            return code_capacity_;
        }

        std::vector<float> project(
            const AceStepLatents & latents,
            int64_t code_offset,
            int64_t code_count,
            const std::vector<float> & silence_latent,
            int64_t silence_frames,
            int64_t silence_channels,
            std::vector<float> & input_buffer) const {
            const auto total_start = Clock::now();
            const auto & config = assets_->config.diffusion;
            if (code_count <= 0 || code_count > code_capacity_) {
                throw std::runtime_error("ACE-Step cover tokenizer chunk exceeds graph capacity");
            }
            if (latents.channels != config.latent_channels ||
                silence_channels != config.latent_channels ||
                silence_frames <= 0 ||
                static_cast<int64_t>(silence_latent.size()) < silence_frames * silence_channels) {
                throw std::runtime_error("ACE-Step cover tokenizer latent shape is invalid");
            }

            const auto input_start = Clock::now();
            input_buffer.assign(
                static_cast<size_t>(code_capacity_ * config.pool_window_size * config.latent_channels),
                0.0F);
            for (int64_t code = 0; code < code_count; ++code) {
                for (int64_t patch = 0; patch < config.pool_window_size; ++patch) {
                    const int64_t src_frame = (code_offset + code) * config.pool_window_size + patch;
                    const int64_t silence_frame = std::min<int64_t>(patch, silence_frames - 1);
                    const float * src =
                        src_frame < latents.frames
                            ? latents.values.data() + static_cast<size_t>(src_frame * latents.channels)
                            : silence_latent.data() + static_cast<size_t>(silence_frame * silence_channels);
                    float * dst = input_buffer.data() + static_cast<size_t>(
                        (code * config.pool_window_size + patch) * config.latent_channels);
                    std::copy_n(src, static_cast<size_t>(config.latent_channels), dst);
                }
            }
            core::write_tensor_f32(input_value_, input_buffer);
            core::set_backend_threads(backend_, threads_);
            engine::debug::timing_log_scalar(
                "ace_step.cover_tokenizer.chunk.input_upload_ms",
                engine::debug::elapsed_ms(input_start, Clock::now()));

            const auto compute_start = Clock::now();
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step cover tokenizer graph compute failed");
            }
            engine::debug::timing_log_scalar(
                "ace_step.cover_tokenizer.chunk.graph.compute_ms",
                engine::debug::elapsed_ms(compute_start, Clock::now()));

            const auto output_start = Clock::now();
            std::vector<float> values = core::read_tensor_f32(output_);
            values.resize(static_cast<size_t>(code_count * config.fsq_dim));
            engine::debug::timing_log_scalar(
                "ace_step.cover_tokenizer.chunk.output_read_ms",
                engine::debug::elapsed_ms(output_start, Clock::now()));
            engine::debug::timing_log_scalar(
                "ace_step.cover_tokenizer.chunk.total_ms",
                engine::debug::elapsed_ms(total_start, Clock::now()));
            return values;
        }

    private:
        void build() {
            const auto & config = assets_->config.diffusion;
            const int64_t patch_tokens = config.pool_window_size + 1;
            ggml_init_params params{128ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step cover tokenizer ggml context initialization failed");
            }

            input_ = ggml_new_tensor_3d(
                ctx_.get(),
                GGML_TYPE_F32,
                config.latent_channels,
                config.pool_window_size,
                code_capacity_);
            special_token_ = ggml_new_tensor_3d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, 1, 1);
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, patch_tokens);
            full_attention_mask_ =
                ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, patch_tokens, patch_tokens, 1, 1);
            sliding_attention_mask_ =
                ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, patch_tokens, patch_tokens, 1, 1);

            input_value_ = core::wrap_tensor(
                input_,
                core::TensorShape::from_dims({code_capacity_, config.pool_window_size, config.latent_channels}),
                GGML_TYPE_F32);
            core::ModuleBuildContext build_ctx{ctx_.get()};
            auto special_token = core::wrap_tensor(
                special_token_,
                core::TensorShape::from_dims({1, 1, config.hidden_size}),
                GGML_TYPE_F32);
            auto positions = core::wrap_tensor(
                positions_,
                core::TensorShape::from_dims({patch_tokens}),
                GGML_TYPE_I32);
            auto full_attention_mask = core::wrap_tensor(
                full_attention_mask_,
                core::TensorShape::from_dims({1, 1, patch_tokens, patch_tokens}),
                GGML_TYPE_F32);
            auto sliding_attention_mask = core::wrap_tensor(
                sliding_attention_mask_,
                core::TensorShape::from_dims({1, 1, patch_tokens, patch_tokens}),
                GGML_TYPE_F32);

            auto x = modules::LinearModule({config.latent_channels, config.hidden_size, true}).build(
                build_ctx,
                input_value_,
                weights_->audio_acoustic_proj);
            x = modules::LinearModule({config.hidden_size, config.hidden_size, true}).build(
                build_ctx,
                x,
                weights_->attention_pooler_embed_tokens);
            auto special_tokens = modules::RepeatModule({
                core::TensorShape::from_dims({code_capacity_, 1, config.hidden_size})})
                                      .build(build_ctx, special_token);
            x = modules::ConcatModule({1}).build(build_ctx, special_tokens, x);
            modules::QwenDecoderLayerConfig layer_config;
            layer_config.hidden_size = config.hidden_size;
            layer_config.num_attention_heads = config.num_attention_heads;
            layer_config.num_key_value_heads = config.num_key_value_heads;
            layer_config.head_dim = config.head_dim;
            layer_config.intermediate_size = config.intermediate_size;
            layer_config.rms_norm_eps = config.rms_norm_eps;
            layer_config.rope_theta = config.rope_theta;
            layer_config.attention_precision = GGML_PREC_F32;
            layer_config.projection_precision = GGML_PREC_F32;
            const modules::QwenDecoderLayerModule layer_module(layer_config);
            for (int64_t i = 0; i < config.num_attention_pooler_hidden_layers; ++i) {
                const auto & layer = weights_->attention_pooler_layers.layers[static_cast<size_t>(i)];
                const auto & mask =
                    (config.layer_types[static_cast<size_t>(i)] == "sliding_attention")
                        ? sliding_attention_mask
                        : full_attention_mask;
                x = layer_module.build(build_ctx, x, positions, layer, std::nullopt, std::nullopt, mask).output;
            }
            x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(
                build_ctx,
                x,
                {weights_->attention_pooler_norm, std::nullopt});
            x = modules::SliceModule({1, 0, 1}).build(build_ctx, x);
            x = modules::LinearModule({config.hidden_size, config.fsq_dim, true}).build(
                build_ctx,
                x,
                weights_->quantizer_project_in);
            output_ = x.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            ggml_build_forward_expand(graph_, output_);
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step cover tokenizer backend buffer allocation failed");
            }

            std::vector<int32_t> position_values(static_cast<size_t>(patch_tokens), 0);
            for (int64_t i = 0; i < patch_tokens; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(
                positions_,
                position_values.data(),
                0,
                position_values.size() * sizeof(int32_t));
            const std::vector<float> full_mask_values(static_cast<size_t>(patch_tokens * patch_tokens), 0.0F);
            ggml_backend_tensor_set(
                full_attention_mask_,
                full_mask_values.data(),
                0,
                full_mask_values.size() * sizeof(float));
            const std::vector<float> sliding_mask_values =
                ace_step_bidirectional_sliding_mask_values(
                    patch_tokens,
                    config.sliding_window,
                    "ACE-Step cover tokenizer");
            ggml_backend_tensor_set(
                sliding_attention_mask_,
                sliding_mask_values.data(),
                0,
                sliding_mask_values.size() * sizeof(float));
            ggml_backend_tensor_set(
                special_token_,
                weights_->attention_pooler_special_token_host.data(),
                0,
                weights_->attention_pooler_special_token_host.size() * sizeof(float));
        }

        std::shared_ptr<const AceStepAssets> assets_;
        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::shared_ptr<const AceStepCoverTokenizerWeights> weights_;
        int64_t code_capacity_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * special_token_ = nullptr;
        ggml_tensor * positions_ = nullptr;
        ggml_tensor * full_attention_mask_ = nullptr;
        ggml_tensor * sliding_attention_mask_ = nullptr;
        core::TensorValue input_value_;
        ggml_tensor * output_ = nullptr;
        ggml_cgraph * graph_ = nullptr;
        ggml_backend_buffer_t buffer_ = nullptr;
    };

    Impl(
        core::ExecutionContext & execution,
        std::shared_ptr<const AceStepAssets> assets,
        assets::TensorStorageType storage_type)
        : assets_(std::move(assets)),
          backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          storage_type_(storage_type),
          quantizer_(ace_step_build_fsq_quantizer_table(
              assets_->config.diffusion,
              "ACE-Step native cover tokenizer")) {
        if (assets_ == nullptr) {
            throw std::runtime_error("ACE-Step cover tokenizer requires assets");
        }
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step cover tokenizer backend initialization failed");
        }
    }

    std::vector<int32_t> encode_audio_codes(
        const AceStepLatents & latents,
        const std::vector<float> & silence_latent,
        int64_t silence_frames,
        int64_t silence_channels) const {
        const auto total_start = Clock::now();
        const auto & config = assets_->config.diffusion;
        if (latents.frames <= 0 || latents.channels != config.latent_channels) {
            throw std::runtime_error("ACE-Step cover tokenizer requires positive latent frames and matching channels");
        }
        ensure_weights();
        const int64_t code_count = (latents.frames + config.pool_window_size - 1) / config.pool_window_size;
        const int64_t graph_capacity = std::min(code_count, kMaxGraphCodeCapacity);
        const auto ensure_graph_start = Clock::now();
        if (!graph_ || !graph_->can_run(graph_capacity)) {
            graph_.reset();
            graph_ = std::make_unique<Graph>(
                assets_,
                backend_,
                threads_,
                weights_,
                graph_capacity);
        }
        engine::debug::timing_log_scalar(
            "ace_step.cover_tokenizer.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_graph_start, Clock::now()));

        std::vector<int32_t> audio_codes;
        audio_codes.reserve(static_cast<size_t>(code_count));
        int64_t offset = 0;
        while (offset < code_count) {
            const int64_t chunk_codes = std::min(graph_->code_capacity(), code_count - offset);
            const std::vector<float> projected = graph_->project(
                latents,
                offset,
                chunk_codes,
                silence_latent,
                silence_frames,
                silence_channels,
                input_buffer_);
            const std::vector<int32_t> chunk_indices = fsq_indices_from_projected(
                quantizer_,
                projected,
                chunk_codes);
            audio_codes.insert(audio_codes.end(), chunk_indices.begin(), chunk_indices.end());
            offset += chunk_codes;
        }
        engine::debug::timing_log_scalar(
            "ace_step.cover_tokenizer.total_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        return audio_codes;
    }

    void release_runtime_graphs() const {
        graph_.reset();
        input_buffer_.clear();
    }

private:
    void ensure_weights() const {
        if (!weights_) {
            const auto start = Clock::now();
            weights_ = load_cover_tokenizer_weights(backend_, backend_type_, *assets_, storage_type_);
            engine::debug::timing_log_scalar(
                "ace_step.cover_tokenizer.load_weights_ms",
                engine::debug::elapsed_ms(start, Clock::now()));
        }
    }

    std::shared_ptr<const AceStepAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    assets::TensorStorageType storage_type_ = assets::TensorStorageType::Native;
    AceStepFsqQuantizerTable quantizer_;
    mutable std::shared_ptr<const AceStepCoverTokenizerWeights> weights_;
    mutable std::unique_ptr<Graph> graph_;
    mutable std::vector<float> input_buffer_;
};

AceStepCoverTokenizerRuntime::AceStepCoverTokenizerRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepAssets> assets,
    assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), storage_type)) {
}

AceStepCoverTokenizerRuntime::~AceStepCoverTokenizerRuntime() = default;

std::vector<int32_t> AceStepCoverTokenizerRuntime::encode_audio_codes(
    const AceStepLatents & latents,
    const std::vector<float> & silence_latent,
    int64_t silence_frames,
    int64_t silence_channels) const {
    return impl_->encode_audio_codes(latents, silence_latent, silence_frames, silence_channels);
}

void AceStepCoverTokenizerRuntime::release_runtime_graphs() const {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::ace_step
