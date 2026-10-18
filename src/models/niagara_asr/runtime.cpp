#include "engine/models/niagara_asr/runtime.h"

#include "engine/models/niagara_asr/encoder.h"
#include "engine/models/niagara_asr/weights.h"
#include "engine/framework/core/backend.h"

#include "ggml-alloc.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace engine::models::niagara_asr {
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

    ~Impl() {
        release_graph();
    }

    void release_graph() {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
            gallocr = nullptr;
        }
        if (ggml_ctx != nullptr) {
            ggml_free(ggml_ctx);
            ggml_ctx = nullptr;
        }
        graph = nullptr;
        input = {};
        logits = {};
        frames = 0;
        feature_dim = 0;
    }

    void ensure_graph(int64_t requested_frames, int64_t requested_feature_dim) {
        if (ggml_ctx != nullptr && frames == requested_frames && feature_dim == requested_feature_dim) {
            return;
        }
        release_graph();
        ggml_init_params params{
            graph_arena_bytes,
            nullptr,
            true,
        };
        ggml_ctx = ggml_init(params);
        if (ggml_ctx == nullptr) {
            throw std::runtime_error("failed to allocate Niagara runtime graph context");
        }

        core::ModuleBuildContext ctx{ggml_ctx, "niagara_asr.encoder", execution->backend_type()};
        input = core::make_tensor(
            ctx,
            GGML_TYPE_F32,
            core::TensorShape::from_dims({1, requested_frames, requested_feature_dim}));
        logits = build_niagara_encoder_logits(ctx, input, *weights, assets->config);
        logits = core::ensure_backend_addressable_layout(ctx, logits);
        ggml_set_output(logits.tensor);

        graph = ggml_new_graph_custom(ggml_ctx, 65536, false);
        ggml_build_forward_expand(graph, logits.tensor);

        const auto backend = execution->backend();
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            release_graph();
            throw std::runtime_error("failed to allocate Niagara runtime graph");
        }
        frames = requested_frames;
        feature_dim = requested_feature_dim;
    }

    std::shared_ptr<const NiagaraAsrAssets> assets;
    core::ExecutionContext * execution = nullptr;
    std::shared_ptr<const NiagaraWeights> weights;
    size_t graph_arena_bytes = 0;
    ggml_context * ggml_ctx = nullptr;
    ggml_gallocr_t gallocr = nullptr;
    ggml_cgraph * graph = nullptr;
    core::TensorValue input;
    core::TensorValue logits;
    int64_t frames = 0;
    int64_t feature_dim = 0;
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
    impl_->ensure_graph(features.frames, features.feature_dim);
    core::write_tensor_f32(impl_->input, features.values);
    const ggml_status status =
        core::compute_backend_graph(impl_->execution->backend(), impl_->graph, nullptr, "Niagara ASR encoder");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Niagara runtime graph compute failed");
    }

    NiagaraInferenceResult result;
    result.frames = impl_->logits.shape.dims[1];
    result.vocab_size = impl_->logits.shape.dims[2];
    result.logits = core::read_tensor_f32(impl_->logits.tensor);
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    return result;
}

}  // namespace engine::models::niagara_asr
