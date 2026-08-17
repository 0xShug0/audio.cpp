#include "engine/community_models/moss_voicegen/heads.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::moss_voicegen {
namespace {

namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void validate_weight_storage_type(assets::TensorStorageType storage_type) {
    switch (storage_type) {
        case assets::TensorStorageType::Native:
        case assets::TensorStorageType::F32:
        case assets::TensorStorageType::F16:
        case assets::TensorStorageType::BF16:
        case assets::TensorStorageType::Q8_0:
            return;
        default:
            throw std::runtime_error(
                "MOSS-VoiceGenerator heads weight_type supports only native, f32, f16, bf16, and q8_0");
    }
}

}  // namespace

struct MossVoiceGenHeadsRuntime::Impl {
    std::shared_ptr<const MossVoiceGenAssets> assets;
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    size_t graph_arena_bytes = 0;

    std::shared_ptr<core::BackendWeightStore> store;
    core::TensorValue text_head;
    std::vector<core::TensorValue> audio_heads;

    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * hidden_input = nullptr;
    ggml_tensor * text_output = nullptr;
    std::vector<ggml_tensor *> audio_outputs;

    ~Impl() {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

MossVoiceGenHeadsRuntime::MossVoiceGenHeadsRuntime(
    std::shared_ptr<const MossVoiceGenAssets> assets,
    core::ExecutionContext & execution_context,
    size_t graph_arena_bytes,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type)
    : impl_(std::make_unique<Impl>()) {
    if (assets == nullptr || assets->model_weights == nullptr) {
        throw std::runtime_error("MOSS-VoiceGenerator heads require assets and model weights");
    }
    validate_weight_storage_type(weight_storage_type);
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS-VoiceGenerator heads backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->graph_arena_bytes = graph_arena_bytes;

    const auto & config = assets->config;
    const auto & source = *assets->model_weights;
    const int64_t hidden_size = config.backbone.hidden_size;
    const int64_t audio_head_size = config.audio_vocab_size + 1;

    impl_->store = std::make_shared<core::BackendWeightStore>(
        impl_->backend,
        impl_->backend_type,
        "moss_voicegen.heads.weights",
        weight_context_bytes);
    impl_->text_head = impl_->store->load_tensor(
        source, "lm_heads.0.weight", weight_storage_type, {config.backbone.vocab_size, hidden_size});
    impl_->audio_heads.reserve(static_cast<size_t>(config.num_codebooks));
    for (int64_t codebook = 0; codebook < config.num_codebooks; ++codebook) {
        // Head 0 is the text head, so codebook i lives at head i + 1.
        impl_->audio_heads.push_back(impl_->store->load_tensor(
            source,
            "lm_heads." + std::to_string(codebook + 1) + ".weight",
            weight_storage_type,
            {audio_head_size, hidden_size}));
    }
    impl_->store->upload();

    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    impl_->graph_ctx.reset(ggml_init(params));
    if (impl_->graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS-VoiceGenerator heads graph context");
    }
    ggml_context * gctx = impl_->graph_ctx.get();
    core::ModuleBuildContext ctx{gctx, "moss_voicegen.heads", impl_->backend_type};

    auto hidden = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_size}));
    ggml_set_input(hidden.tensor);

    impl_->graph = ggml_new_graph_custom(gctx, 4096, false);

    auto text_logits = modules::LinearModule({hidden_size, config.backbone.vocab_size, false, GGML_PREC_F32})
                           .build(ctx, hidden, {impl_->text_head, std::nullopt});
    text_logits = core::ensure_backend_addressable_layout(ctx, text_logits);
    ggml_set_output(text_logits.tensor);
    ggml_build_forward_expand(impl_->graph, text_logits.tensor);

    impl_->audio_outputs.reserve(impl_->audio_heads.size());
    const modules::LinearModule audio_head_module({hidden_size, audio_head_size, false, GGML_PREC_F32});
    for (const auto & head : impl_->audio_heads) {
        auto logits = audio_head_module.build(ctx, hidden, {head, std::nullopt});
        logits = core::ensure_backend_addressable_layout(ctx, logits);
        ggml_set_output(logits.tensor);
        ggml_build_forward_expand(impl_->graph, logits.tensor);
        impl_->audio_outputs.push_back(logits.tensor);
    }

    impl_->buffer = ggml_backend_alloc_ctx_tensors(gctx, impl_->backend);
    if (impl_->buffer == nullptr) {
        throw std::runtime_error("failed to allocate MOSS-VoiceGenerator heads graph");
    }
    impl_->hidden_input = hidden.tensor;
    impl_->text_output = text_logits.tensor;
    impl_->assets = std::move(assets);
}

MossVoiceGenHeadsRuntime::~MossVoiceGenHeadsRuntime() = default;

void MossVoiceGenHeadsRuntime::evaluate(
    const std::vector<float> & hidden_state,
    MossVoiceGenStepLogits & out) const {
    auto & impl = *impl_;
    const auto & config = impl.assets->config;
    const int64_t hidden_size = config.backbone.hidden_size;
    if (static_cast<int64_t>(hidden_state.size()) != hidden_size) {
        throw std::runtime_error("MOSS-VoiceGenerator heads input does not match hidden_size");
    }

    ggml_backend_tensor_set(impl.hidden_input, hidden_state.data(), 0, hidden_state.size() * sizeof(float));
    core::set_backend_threads(impl.backend, impl.threads);
    const ggml_status status = ggml_backend_graph_compute(impl.backend, impl.graph);
    ggml_backend_synchronize(impl.backend);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("MOSS-VoiceGenerator heads graph compute failed");
    }

    out.text.resize(static_cast<size_t>(config.backbone.vocab_size));
    ggml_backend_tensor_get(impl.text_output, out.text.data(), 0, out.text.size() * sizeof(float));
    out.audio.resize(impl.audio_outputs.size());
    const auto audio_head_size = static_cast<size_t>(config.audio_vocab_size + 1);
    for (size_t codebook = 0; codebook < impl.audio_outputs.size(); ++codebook) {
        out.audio[codebook].resize(audio_head_size);
        ggml_backend_tensor_get(
            impl.audio_outputs[codebook],
            out.audio[codebook].data(),
            0,
            out.audio[codebook].size() * sizeof(float));
    }
}

}  // namespace engine::models::moss_voicegen
