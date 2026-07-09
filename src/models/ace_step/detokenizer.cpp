#include "engine/models/ace_step/detokenizer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/attention/qwen_decoder.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/structural_modules.h"
#include "helper_utils.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::ace_step {
namespace {

namespace modules = engine::modules;

using Clock = std::chrono::steady_clock;

std::vector<float> fsq_codes_from_indices(const AceStepFsqQuantizerTable & table, const int32_t * indices, size_t count) {
    const size_t dim = table.levels.size();
    std::vector<float> codes(count * dim, 0.0F);
    for (size_t n = 0; n < count; ++n) {
        int64_t index = std::max<int64_t>(0, indices[n]);
        for (size_t d = 0; d < dim; ++d) {
            const int64_t level = table.levels[d];
            const int64_t level_index = (index / table.basis[d]) % level;
            codes[n * dim + d] = static_cast<float>(level_index) * (2.0F / static_cast<float>(level - 1)) - 1.0F;
        }
    }
    return codes;
}

std::vector<float> apply_linear(
    const std::vector<float> & input,
    int64_t rows,
    int64_t in_dim,
    int64_t out_dim,
    const std::vector<float> & weight,
    const std::vector<float> & bias) {
    std::vector<float> output(static_cast<size_t>(rows * out_dim), 0.0F);
    for (int64_t r = 0; r < rows; ++r) {
        const float * src = input.data() + static_cast<size_t>(r * in_dim);
        float * dst = output.data() + static_cast<size_t>(r * out_dim);
        for (int64_t o = 0; o < out_dim; ++o) {
            const float * w = weight.data() + static_cast<size_t>(o * in_dim);
            float sum = bias[static_cast<size_t>(o)];
            for (int64_t i = 0; i < in_dim; ++i) {
                sum += src[i] * w[i];
            }
            dst[o] = sum;
        }
    }
    return output;
}

class GgmlContextDeleter {
public:
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

}  // namespace

class AceStepAudioDetokenizerRuntime::Impl {
public:
    static constexpr int64_t kMaxGraphCodeCapacity = 512;

    class Graph {
    public:
        Graph(
            std::shared_ptr<const AceStepAssets> assets,
            ggml_backend_t backend,
            int threads,
            std::shared_ptr<const AceStepDetokenizerWeights> weights,
            AceStepFsqQuantizerTable quantizer,
            int64_t code_capacity)
            : backend_(backend),
              threads_(threads),
              assets_(std::move(assets)),
              weights_(std::move(weights)),
              quantizer_(std::move(quantizer)),
              code_capacity_(code_capacity) {
            if (backend_ == nullptr) {
                throw std::runtime_error("ACE-Step detokenizer backend initialization failed");
            }
            if (code_capacity_ <= 0) {
                throw std::runtime_error("ACE-Step detokenizer requires positive code capacity");
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

        AceStepLatents decode_audio_codes(const int32_t * audio_code_ids, int64_t code_count, std::vector<float> & expanded) const {
            const auto total_start = Clock::now();
            const auto & config = assets_->config.diffusion;
            if (code_count <= 0) {
                return {};
            }
            if (code_count > code_capacity_) {
                throw std::runtime_error("ACE-Step audio-code sequence exceeds native detokenizer capacity");
            }

            const auto quantize_start = Clock::now();
            const std::vector<float> codes = fsq_codes_from_indices(
                quantizer_,
                audio_code_ids,
                static_cast<size_t>(code_count));
            const std::vector<float> quantized = apply_linear(
                codes,
                code_count,
                config.fsq_dim,
                config.hidden_size,
                weights_->quantizer_project_out_weight,
                weights_->quantizer_project_out_bias);
            engine::debug::timing_log_scalar(
                "ace_step.detokenizer.chunk.quantize_project_ms",
                engine::debug::elapsed_ms(quantize_start, Clock::now()));

            const auto input_start = Clock::now();
            expanded.assign(static_cast<size_t>(code_capacity_ * config.hidden_size), 0.0F);
            for (int64_t t = 0; t < code_count; ++t) {
                const float * src = quantized.data() + t * static_cast<size_t>(config.hidden_size);
                float * dst = expanded.data() + t * static_cast<size_t>(config.hidden_size);
                std::copy(src, src + static_cast<size_t>(config.hidden_size), dst);
            }

            core::write_tensor_f32(input_value_, expanded);
            core::set_backend_threads(backend_, threads_);
            engine::debug::timing_log_scalar(
                "ace_step.detokenizer.chunk.input_upload_ms",
                engine::debug::elapsed_ms(input_start, Clock::now()));
            const auto compute_start = Clock::now();
            const ggml_status status = engine::core::compute_backend_graph(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("ACE-Step detokenizer graph compute failed");
            }
            engine::debug::timing_log_scalar(
                "ace_step.detokenizer.chunk.graph.compute_ms",
                engine::debug::elapsed_ms(compute_start, Clock::now()));

            const auto output_start = Clock::now();
            std::vector<float> values = core::read_tensor_f32(output_);
            AceStepLatents latents;
            latents.frames = code_count * config.pool_window_size;
            latents.channels = config.latent_channels;
            latents.values.assign(
                values.begin(),
                values.begin() + static_cast<std::ptrdiff_t>(latents.frames * latents.channels));
            engine::debug::timing_log_scalar(
                "ace_step.detokenizer.chunk.output_read_ms",
                engine::debug::elapsed_ms(output_start, Clock::now()));
            engine::debug::timing_log_scalar("ace_step.detokenizer.chunk.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
            return latents;
        }

    private:
        void build() {
            const auto & config = assets_->config.diffusion;
            ggml_init_params params{96ull * 1024ull * 1024ull, nullptr, true};
            ctx_.reset(ggml_init(params));
            if (ctx_ == nullptr) {
                throw std::runtime_error("ACE-Step detokenizer ggml context initialization failed");
            }

            input_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, config.hidden_size, code_capacity_);
            input_value_ = core::wrap_tensor(
                input_,
                core::TensorShape::from_dims({code_capacity_, config.hidden_size}),
                GGML_TYPE_F32);
            special_tokens_ = ggml_new_tensor_3d(
                ctx_.get(),
                GGML_TYPE_F32,
                config.hidden_size,
                config.pool_window_size,
                1);
            positions_ = ggml_new_tensor_1d(ctx_.get(), GGML_TYPE_I32, config.pool_window_size);
            full_attention_mask_ = ggml_new_tensor_4d(
                ctx_.get(),
                GGML_TYPE_F32,
                config.pool_window_size,
                config.pool_window_size,
                1,
                1);
            sliding_attention_mask_ = ggml_new_tensor_4d(
                ctx_.get(),
                GGML_TYPE_F32,
                config.pool_window_size,
                config.pool_window_size,
                1,
                1);

            core::ModuleBuildContext build_ctx{ctx_.get()};
            auto special_tokens = core::wrap_tensor(
                special_tokens_,
                core::TensorShape::from_dims({1, config.pool_window_size, config.hidden_size}),
                GGML_TYPE_F32);
            auto positions = core::wrap_tensor(
                positions_,
                core::TensorShape::from_dims({config.pool_window_size}),
                GGML_TYPE_I32);
            auto full_attention_mask = core::wrap_tensor(
                full_attention_mask_,
                core::TensorShape::from_dims({1, 1, config.pool_window_size, config.pool_window_size}),
                GGML_TYPE_F32);
            auto sliding_attention_mask = core::wrap_tensor(
                sliding_attention_mask_,
                core::TensorShape::from_dims({1, 1, config.pool_window_size, config.pool_window_size}),
                GGML_TYPE_F32);

            auto x = modules::LinearModule(
                         {config.hidden_size, config.hidden_size, true})
                         .build(
                build_ctx,
                input_value_,
                weights_->embed_tokens);
            x = modules::ReshapeModule({core::TensorShape::from_dims({code_capacity_, 1, config.hidden_size})}).build(build_ctx, x);
            x = modules::RepeatModule({core::TensorShape::from_dims({code_capacity_, config.pool_window_size, config.hidden_size})}).build(
                build_ctx,
                x);
            auto repeated_special_tokens = modules::RepeatModule({
                core::TensorShape::from_dims({code_capacity_, config.pool_window_size, config.hidden_size})})
                                               .build(build_ctx, special_tokens);
            x = modules::AddModule{}.build(build_ctx, x, repeated_special_tokens);
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
                const auto & layer = weights_->layers.layers[static_cast<size_t>(i)];
                const auto & mask =
                    (config.layer_types[static_cast<size_t>(i)] == "sliding_attention")
                    ? sliding_attention_mask
                    : full_attention_mask;
                x = layer_module.build(build_ctx, x, positions, layer, std::nullopt, std::nullopt, mask).output;
            }
            x = modules::RMSNormModule({config.hidden_size, config.rms_norm_eps, true, false}).build(
                build_ctx,
                x,
                {weights_->norm, std::nullopt});
            x = modules::LinearModule(
                    {config.hidden_size, config.latent_channels, true})
                    .build(build_ctx, x, weights_->proj_out);
            output_ = x.tensor;
            ggml_set_output(output_);
            graph_ = ggml_new_graph_custom(ctx_.get(), 65536, false);
            ggml_build_forward_expand(graph_, output_);
            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), backend_);
            if (buffer_ == nullptr) {
                throw std::runtime_error("ACE-Step detokenizer backend buffer allocation failed");
            }

            std::vector<int32_t> position_values(static_cast<size_t>(config.pool_window_size), 0);
            for (int64_t i = 0; i < config.pool_window_size; ++i) {
                position_values[static_cast<size_t>(i)] = static_cast<int32_t>(i);
            }
            ggml_backend_tensor_set(
                positions_,
                position_values.data(),
                0,
                position_values.size() * sizeof(int32_t));

            const std::vector<float> full_mask_values(
                static_cast<size_t>(config.pool_window_size * config.pool_window_size),
                0.0F);
            ggml_backend_tensor_set(
                full_attention_mask_,
                full_mask_values.data(),
                0,
                full_mask_values.size() * sizeof(float));
            const std::vector<float> sliding_mask_values =
                ace_step_bidirectional_sliding_mask_values(
                    config.pool_window_size,
                    config.sliding_window,
                    "ACE-Step detokenizer");
            ggml_backend_tensor_set(
                sliding_attention_mask_,
                sliding_mask_values.data(),
                0,
                sliding_mask_values.size() * sizeof(float));
            ggml_backend_tensor_set(
                special_tokens_,
                weights_->special_tokens_host.data(),
                0,
                weights_->special_tokens_host.size() * sizeof(float));

        }

        ggml_backend_t backend_ = nullptr;
        int threads_ = 1;
        std::shared_ptr<const AceStepAssets> assets_;
        std::shared_ptr<const AceStepDetokenizerWeights> weights_;
        AceStepFsqQuantizerTable quantizer_;
        int64_t code_capacity_ = 0;
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
        ggml_tensor * input_ = nullptr;
        ggml_tensor * special_tokens_ = nullptr;
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
        std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime)
        : backend_(execution.backend()),
          threads_(std::max(1, execution.config().threads)),
          assets_(std::move(assets)),
          weights_(dit_weights_runtime->detokenizer_weights()),
          quantizer_(ace_step_build_fsq_quantizer_table(
              assets_->config.diffusion,
              "ACE-Step native detokenizer")) {
        if (backend_ == nullptr) {
            throw std::runtime_error("ACE-Step detokenizer backend initialization failed");
        }
    }

    AceStepLatents decode_audio_codes(const std::vector<int32_t> & audio_code_ids) const {
        const auto total_start = Clock::now();
        const auto ensure_start = Clock::now();
        const int64_t code_count = static_cast<int64_t>(audio_code_ids.size());
        if (code_count <= 0) {
            return {};
        }
        const int64_t graph_capacity = std::min(code_count, kMaxGraphCodeCapacity);
        if (!graph_ || !graph_->can_run(graph_capacity)) {
            graph_.reset();
            graph_ = std::make_unique<Graph>(
                assets_,
                backend_,
                threads_,
                weights_,
                quantizer_,
                graph_capacity);
        }
        engine::debug::timing_log_scalar(
            "ace_step.detokenizer.graph.ensure_ms",
            engine::debug::elapsed_ms(ensure_start, Clock::now()));
        const auto & config = assets_->config.diffusion;
        AceStepLatents out;
        out.frames = code_count * config.pool_window_size;
        out.channels = config.latent_channels;
        out.values.reserve(static_cast<size_t>(out.frames * out.channels));
        int64_t offset = 0;
        while (offset < code_count) {
            const int64_t chunk_codes = std::min(graph_->code_capacity(), code_count - offset);
            AceStepLatents chunk = graph_->decode_audio_codes(
                audio_code_ids.data() + offset,
                chunk_codes,
                expanded_);
            out.values.insert(out.values.end(), chunk.values.begin(), chunk.values.end());
            offset += chunk_codes;
        }
        engine::debug::timing_log_scalar(
            "ace_step.detokenizer.decode_audio_codes_ms",
            engine::debug::elapsed_ms(total_start, Clock::now()));
        engine::debug::timing_log_scalar("ace_step.detokenizer.total_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
        return out;
    }

    void release_runtime_graphs() const {
        graph_.reset();
        expanded_.clear();
    }

private:
    ggml_backend_t backend_ = nullptr;
    int threads_ = 1;
    std::shared_ptr<const AceStepAssets> assets_;
    std::shared_ptr<const AceStepDetokenizerWeights> weights_;
    AceStepFsqQuantizerTable quantizer_;
    mutable std::unique_ptr<Graph> graph_;
    mutable std::vector<float> expanded_;
};

AceStepAudioDetokenizerRuntime::AceStepAudioDetokenizerRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const AceStepAssets> assets,
    std::shared_ptr<const AceStepDitWeightsRuntime> dit_weights_runtime)
    : assets_(std::move(assets)),
      impl_(std::make_unique<Impl>(execution, assets_, std::move(dit_weights_runtime))) {
    if (assets_ == nullptr) {
        throw std::runtime_error("ACE-Step detokenizer runtime requires assets");
    }
}

AceStepAudioDetokenizerRuntime::~AceStepAudioDetokenizerRuntime() = default;

AceStepLatents AceStepAudioDetokenizerRuntime::decode_audio_codes(const std::vector<int32_t> & audio_code_ids) const {
    return impl_->decode_audio_codes(audio_code_ids);
}

void AceStepAudioDetokenizerRuntime::release_runtime_graphs() const {
    impl_->release_runtime_graphs();
}

}  // namespace engine::models::ace_step
