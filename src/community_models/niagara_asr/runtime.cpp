#include "engine/community_models/niagara_asr/runtime.h"

#include "engine/community_models/niagara_asr/encoder.h"
#include "engine/community_models/niagara_asr/weights.h"
#include "engine/framework/core/backend.h"

#include "ggml-alloc.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::community_models::niagara_asr {
namespace assets = engine::assets;
namespace core = engine::core;
namespace {

using Clock = std::chrono::steady_clock;

const NiagaraAsrAssets & require_runtime_assets(const std::shared_ptr<const NiagaraAsrAssets> & assets) {
    if (assets == nullptr || assets->source == nullptr) {
        throw std::runtime_error("Niagara runtime requires assets and tensor source");
    }
    return *assets;
}

}  // namespace

struct NiagaraRuntime::Impl {
    Impl(
        std::shared_ptr<const NiagaraAsrAssets> assets,
        core::ExecutionContext & execution,
        assets::TensorStorageType weight_storage_type,
        size_t graph_arena_bytes,
        size_t weight_context_bytes)
        : assets(std::move(assets)),
          execution(&execution),
          weights(load_niagara_weights(require_runtime_assets(this->assets), execution, weight_storage_type, weight_context_bytes)),
          graph_arena_bytes(graph_arena_bytes) {}

    std::shared_ptr<const NiagaraAsrAssets> assets;
    core::ExecutionContext * execution = nullptr;
    std::shared_ptr<const NiagaraWeights> weights;
    size_t graph_arena_bytes = 0;
};

NiagaraRuntime::NiagaraRuntime(
    std::shared_ptr<const NiagaraAsrAssets> assets,
    core::ExecutionContext & execution,
    assets::TensorStorageType weight_storage_type,
    size_t graph_arena_bytes,
    size_t weight_context_bytes)
    : impl_(new Impl(std::move(assets), execution, weight_storage_type, graph_arena_bytes, weight_context_bytes)) {}

NiagaraRuntime::~NiagaraRuntime() = default;

NiagaraInferenceResult NiagaraRuntime::infer(const NiagaraFeatures & features) {
    if (features.frames <= 0 || features.feature_dim != 80 ||
        static_cast<int64_t>(features.values.size()) != features.frames * features.feature_dim) {
        throw std::runtime_error("Niagara runtime received invalid frontend features");
    }

    const auto started = Clock::now();
    ggml_init_params params{
        impl_->graph_arena_bytes,
        nullptr,
        true,
    };
    ggml_context * ggml_ctx = ggml_init(params);
    if (ggml_ctx == nullptr) {
        throw std::runtime_error("failed to allocate Niagara runtime graph context");
    }

    ggml_gallocr_t gallocr = nullptr;
    try {
        core::ModuleBuildContext ctx{ggml_ctx, "niagara_asr.encoder", impl_->execution->backend_type()};
        auto input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, features.frames, features.feature_dim}));
        auto logits = build_niagara_encoder_logits(ctx, input, *impl_->weights, impl_->assets->config);
        logits = core::ensure_backend_addressable_layout(ctx, logits);
        ggml_set_output(logits.tensor);

        ggml_cgraph * graph = ggml_new_graph_custom(ggml_ctx, 65536, false);
        ggml_build_forward_expand(graph, logits.tensor);

        const auto backend = impl_->execution->backend();
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error("failed to allocate Niagara runtime graph");
        }

        core::write_tensor_f32(input, features.values);
        const ggml_status status = core::compute_backend_graph(backend, graph, nullptr, "Niagara ASR encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Niagara runtime graph compute failed");
        }

        NiagaraInferenceResult result;
        result.frames = logits.shape.dims[1];
        result.vocab_size = logits.shape.dims[2];
        result.logits = core::read_tensor_f32(logits.tensor);
        result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        ggml_gallocr_free(gallocr);
        ggml_free(ggml_ctx);
        return result;
    } catch (...) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        ggml_free(ggml_ctx);
        throw;
    }
}

}  // namespace engine::community_models::niagara_asr
