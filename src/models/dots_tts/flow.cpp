#include "flow_impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::dots_tts::detail {

modules::DitRuntimeConfig dit_runtime_config(const DotsConfig & config,
    core::BackendType backend, assets::TensorStorageType storage) {
    modules::DitRuntimeConfig runtime;
    runtime.label = "dots_tts.flow";
    runtime.weight_type = storage;
    runtime.graph_metadata_bytes = kLargeGraphContextBytes;
    runtime.guidance = modules::DitGuidance::Linear;
    runtime.modulation_storage_type = GGML_TYPE_BF16;
    auto & model = runtime.model;
    model.input_channels = config.dit.hidden_size;
    model.output_channels = config.latent_dim;
    model.hidden_size = config.dit.hidden_size;
    model.intermediate_size = config.dit.ffn_hidden_size;
    model.layers = config.dit.num_layers;
    model.norm_eps = kLayerNormEps;
    model.modulation = modules::DitModulation::PerLayer;
    model.global_dim = config.dit.hidden_size;
    model.time_features = kTimestepFrequencyDim;
    auto & attention = model.self_attention;
    attention.heads = config.dit.num_heads;
    attention.kv_heads = config.dit.num_heads;
    attention.head_dim = head_dim(config.dit);
    attention.qk_norm = modules::DitQKNorm::Heads;
    attention.norm_eps = kTorchBFloat16Eps;
    attention.rotary = {config.dit.rotary_bias ? attention.head_dim : 0,
        GGML_ROPE_TYPE_NEOX, config.dit.rotary_theta};
    attention.lowering = backend == core::BackendType::Cuda
        ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews
        : modules::ScaledDotProductAttentionLowering::Explicit;
    attention.precision = GGML_PREC_DEFAULT;
    return runtime;
}

modules::DitWeightNames dit_weight_names(const DotsConfig & config) {
    modules::DitWeightNames names;
    const std::string root = "velocity_field_predictor.";
    names.input = {root + "input_layer.weight", root + "input_layer.bias"};
    for (const auto * index : {"0", "2"}) {
        const auto prefix = root + "time_embedder.mlp." + index;
        names.time_embedding.push_back({prefix + ".weight", prefix + ".bias"});
        if (config.meanflow && config.meanflow->enabled && config.meanflow->use_duration_embedding) {
            const auto duration = root + "duration_embedder.mlp." + index;
            names.duration_embedding.push_back({duration + ".weight", duration + ".bias"});
        }
    }
    const auto layer = root + "blocks.{layer}.";
    names.layer.modulation = {layer + "adaLN_modulation.1.weight", layer + "adaLN_modulation.1.bias"};
    auto & attention = names.layer.self_attention;
    for (const auto * projection : {"q_proj", "k_proj", "v_proj"})
        attention.qkv_parts.push_back({layer + "attn." + projection + ".weight", {}});
    attention.query_norm.weight = layer + "attn.q_norm.weight";
    attention.key_norm.weight = layer + "attn.k_norm.weight";
    attention.output = {layer + "attn.o_proj.weight", layer + "attn.o_proj.bias"};
    names.layer.feed_forward_in = {layer + "ffn.fc1.weight", layer + "ffn.fc1.bias"};
    names.layer.feed_forward_out = {layer + "ffn.fc2.weight", layer + "ffn.fc2.bias"};
    names.final_modulation = {root + "output_layer.adaLN_modulation.1.weight", root + "output_layer.adaLN_modulation.1.bias"};
    names.output = {root + "output_layer.linear.weight", root + "output_layer.linear.bias"};
    return names;
}

void GgmlContextDeleter::operator()(ggml_context * ctx) const noexcept {
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

void GgmlGallocrDeleter::operator()(ggml_gallocr_t alloc) const noexcept {
    if (alloc != nullptr) {
        ggml_gallocr_free(alloc);
    }
}

int64_t head_dim(const DotsTransformerConfig & config) {
    if (config.num_heads <= 0 || config.hidden_size <= 0 || config.hidden_size % config.num_heads != 0) {
        throw std::runtime_error("DotTTS DiT attention config is invalid");
    }
    return config.hidden_size / config.num_heads;
}

ggml_type cached_dit_kv_type(core::BackendType backend_type) {
    return backend_type == core::BackendType::Cuda ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

void validate_config(const DotsConfig & config) {
    if (config.dit.hidden_size <= 0 || config.dit.ffn_hidden_size <= 0 || config.dit.num_layers <= 0) {
        throw std::runtime_error("DotTTS DiT config contains non-positive dimensions");
    }
    if (!config.dit.modulation || config.dit.norm_layer != "RMSNorm" || config.dit.qkv_bias || !config.dit.qk_norm || !config.dit.rotary_bias) {
        throw std::runtime_error("DotTTS DiT checkpoint layout is not supported by the native component");
    }
    if (config.latent_dim <= 0 || config.llm.hidden_size <= 0 || config.campplus_embedding_size <= 0) {
        throw std::runtime_error("DotTTS flow projection config contains invalid dimensions");
    }
}

std::shared_ptr<DotFlowWeights> load_weights(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("DotTTS flow component requires tensor source");
    }
    validate_config(config);
    auto weights = std::make_shared<DotFlowWeights>();
    weights->config = config;
    weights->execution_context = std::make_shared<core::ExecutionContext>(backend);
    weights->store = std::make_shared<core::BackendWeightStore>(
        weights->execution_context->backend(),
        weights->execution_context->backend_type(),
        "dots_tts.flow.projections.weights",
        kWeightContextBytes);
    weights->hidden_proj = {
        weights->store->load_tensor(*source, "hidden_proj.weight", storage_type, {config.dit.hidden_size, config.llm.hidden_size}),
        weights->store->load_f32_tensor(*source, "hidden_proj.bias", {config.dit.hidden_size}),
    };
    weights->latent_proj = {
        weights->store->load_tensor(*source, "latent_proj.weight", storage_type, {config.dit.hidden_size, config.latent_dim}),
        weights->store->load_f32_tensor(*source, "latent_proj.bias", {config.dit.hidden_size}),
    };
    weights->coordinate_proj = {
        weights->store->load_tensor(*source, "coordinate_proj.weight", storage_type, {config.dit.hidden_size, config.latent_dim}),
        weights->store->load_f32_tensor(*source, "coordinate_proj.bias", {config.dit.hidden_size}),
    };
    weights->xvec_proj = {
        weights->store->load_tensor(*source, "xvec_proj.0.weight", storage_type, {config.dit.hidden_size, config.campplus_embedding_size}),
        weights->store->load_f32_tensor(*source, "xvec_proj.0.bias", {config.dit.hidden_size}),
    };
    weights->xvec_norm = {
        weights->store->load_f32_tensor(*source, "xvec_proj.1.weight", {config.dit.hidden_size}),
        weights->store->load_f32_tensor(*source, "xvec_proj.1.bias", {config.dit.hidden_size}),
    };
    weights->store->upload();
    return weights;
}

std::vector<uint8_t> build_decode_mask(int64_t batch_size, int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size, int64_t hidden_patch_size) {
    std::vector<uint8_t> mask(static_cast<size_t>(batch_size * total_len * total_len), 0);
    const int64_t latent_start = total_len - latent_patch_size;
    const int64_t block_start = fm_seq_len - hidden_patch_size;
    if (block_start < 0) {
        throw std::runtime_error("DotTTS flow decode mask requires a current hidden patch");
    }
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t base = batch * total_len * total_len;
        if (block_start > 0) {
            for (int64_t row = 0; row < block_start; ++row) {
                for (int64_t col = 0; col <= row; ++col) {
                    mask[static_cast<size_t>(base + row * total_len + col)] = 1;
                }
            }
        }
        for (int64_t row = block_start; row < fm_seq_len; ++row) {
            for (int64_t col = 0; col < fm_seq_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
            for (int64_t col = latent_start; col < total_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
        }
        for (int64_t row = latent_start; row < total_len; ++row) {
            for (int64_t col = 0; col < fm_seq_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
            for (int64_t col = latent_start; col < total_len; ++col) {
                mask[static_cast<size_t>(base + row * total_len + col)] = 1;
            }
        }
    }
    return mask;
}

std::vector<int32_t> build_decode_positions(int64_t total_len, int64_t fm_seq_len, int64_t latent_patch_size) {
    std::vector<int32_t> positions(static_cast<size_t>(total_len), 0);
    for (int64_t i = 0; i < fm_seq_len; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    const int64_t latent_start = total_len - latent_patch_size;
    for (int64_t i = 0; i < latent_patch_size; ++i) {
        positions[static_cast<size_t>(latent_start + i)] = static_cast<int32_t>(fm_seq_len + i);
    }
    return positions;
}

int64_t resolve_generate_length_bucket(int64_t requested_patches) {
    if (requested_patches <= 0) {
        throw std::runtime_error("DotTTS DiT cache bucket requires a positive patch count");
    }
    constexpr int64_t kBuckets[] = {64, 128, 256, 512};
    for (const int64_t bucket : kBuckets) {
        if (requested_patches <= bucket) {
            return bucket;
        }
    }
    throw std::runtime_error("DotTTS DiT cache request exceeds the largest supported generation bucket");
}

int64_t resolve_dit_cache_capacity_tokens(int64_t fm_seq_len, int64_t unit_len) {
    if (fm_seq_len <= 0 || unit_len <= 0) {
        throw std::runtime_error("DotTTS DiT cache capacity request is invalid");
    }
    const int64_t requested_patches = std::max<int64_t>(1, (fm_seq_len + unit_len - 1) / unit_len);
    return resolve_generate_length_bucket(requested_patches) * unit_len;
}

void append_branch_sequence(
    std::vector<float> & out,
    const std::vector<float> & prefix,
    const std::vector<float> & noise_projection,
    int64_t fm_seq_len,
    int64_t hidden_size,
    int64_t latent_patch_size) {
    const int64_t total_len = fm_seq_len + latent_patch_size;
    if (static_cast<int64_t>(prefix.size()) < fm_seq_len * hidden_size ||
        static_cast<int64_t>(noise_projection.size()) != latent_patch_size * hidden_size) {
        throw std::runtime_error("DotTTS flow branch sequence shape mismatch");
    }
    out.insert(out.end(), prefix.begin(), prefix.begin() + static_cast<std::ptrdiff_t>(fm_seq_len * hidden_size));
    out.insert(out.end(), noise_projection.begin(), noise_projection.end());
    if (static_cast<int64_t>(out.size()) % (total_len * hidden_size) != 0) {
        throw std::runtime_error("DotTTS flow branch sequence assembly failed");
    }
}

class LinearProjectionRunner {
public:
    LinearProjectionRunner(
        std::shared_ptr<const DotFlowWeights> weights,
        const modules::LinearWeights DotFlowWeights::* projection,
        int64_t in_features,
        int64_t out_features,
        const char * label)
        : weights_(std::move(weights)),
          projection_(projection),
          in_features_(in_features),
          out_features_(out_features),
          label_(label) {}

    ~LinearProjectionRunner() { release_graph(); }

    DotsProjectedSequence run(const std::vector<float> & input, int64_t steps) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (steps <= 0 || static_cast<int64_t>(input.size()) != steps * in_features_) {
            throw std::runtime_error(std::string(label_) + " input shape mismatch");
        }
        ensure_graph(steps);
        ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, label_) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label_) + " graph compute failed");
        }
        DotsProjectedSequence out;
        out.steps = steps;
        out.hidden_size = out_features_;
        out.values = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
        steps_ = 0;
    }

private:
    void ensure_graph(int64_t steps) {
        if (ggml_ != nullptr && steps_ == steps) {
            return;
        }
        release_graph();
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error(std::string("failed to initialize ") + label_ + " graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), label_, weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, steps, in_features_}));
        input_ = input.tensor;
        auto output = modules::LinearModule({in_features_, out_features_, true}).build(build_ctx, input, weights_.get()->*projection_);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, label_);
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error(std::string("failed to allocate ") + label_ + " graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
        steps_ = steps;
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    const modules::LinearWeights DotFlowWeights::* projection_ = nullptr;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    const char * label_ = nullptr;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    int64_t steps_ = 0;
};

class SpeakerProjectionRunner {
public:
    explicit SpeakerProjectionRunner(std::shared_ptr<const DotFlowWeights> weights)
        : weights_(std::move(weights)) {}

    ~SpeakerProjectionRunner() { release_graph(); }

    DotsProjectedSequence run(const std::vector<float> & speaker) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto & config = weights_->config;
        if (static_cast<int64_t>(speaker.size()) != config.campplus_embedding_size) {
            throw std::runtime_error("DotTTS speaker projection input shape mismatch");
        }
        ensure_graph();
        ggml_backend_tensor_set(input_, speaker.data(), 0, speaker.size() * sizeof(float));
        if (core::compute_graph(*weights_->execution_context, graph_, plan_, "dots_tts.flow.speaker_proj") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DotTTS speaker projection graph compute failed");
        }
        DotsProjectedSequence out;
        out.steps = 1;
        out.hidden_size = config.dit.hidden_size;
        out.values = core::read_tensor_f32(output_);
        return out;
    }

    void release_graph() {
        if (graph_ != nullptr) {
            core::release_backend_graph_resources(weights_->execution_context->backend(), graph_);
        }
        gallocr_.reset();
        ggml_.reset();
        plan_.reset();
        graph_ = nullptr;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure_graph() {
        if (ggml_ != nullptr) {
            return;
        }
        ggml_init_params params{kSmallGraphContextBytes, nullptr, true};
        ggml_.reset(ggml_init(params));
        if (ggml_ == nullptr) {
            throw std::runtime_error("failed to initialize DotTTS speaker projection graph context");
        }
        core::ModuleBuildContext build_ctx{ggml_.get(), "dots_tts.flow.speaker_proj", weights_->execution_context->backend_type()};
        auto input = core::make_tensor(build_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, weights_->config.campplus_embedding_size}));
        input_ = input.tensor;
        auto output = modules::LinearModule({weights_->config.campplus_embedding_size, weights_->config.dit.hidden_size, true})
                          .build(build_ctx, input, weights_->xvec_proj);
        output = modules::LayerNormModule({weights_->config.dit.hidden_size, kLayerNormEps, true, true})
                     .build(build_ctx, output, weights_->xvec_norm);
        output_ = output.tensor;
        graph_ = ggml_new_graph_custom(ggml_.get(), 65536, false);
        ggml_build_forward_expand(graph_, output_);
        core::validate_backend_graph_supported(weights_->execution_context->backend(), graph_, "dots_tts.flow.speaker_proj");
        gallocr_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(weights_->execution_context->backend())));
        if (gallocr_ == nullptr || !ggml_gallocr_reserve(gallocr_.get(), graph_) || !ggml_gallocr_alloc_graph(gallocr_.get(), graph_)) {
            release_graph();
            throw std::runtime_error("failed to allocate DotTTS speaker projection graph memory");
        }
        core::prepare_host_graph_plan(*weights_->execution_context, graph_, plan_);
    }

    std::shared_ptr<const DotFlowWeights> weights_;
    std::mutex mutex_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml_;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr_;
    ggml_cgraph * graph_ = nullptr;
    core::HostGraphPlan plan_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

DotFlowSharedRuntime::DotFlowSharedRuntime(std::shared_ptr<const DotFlowWeights> weights,
    const assets::TensorSource & source, assets::TensorStorageType storage)
    : weights(std::move(weights)),
      hidden_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::hidden_proj,
          this->weights->config.llm.hidden_size,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.hidden_proj")),
      latent_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::latent_proj,
          this->weights->config.latent_dim,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.latent_proj")),
      coordinate_projection(std::make_unique<LinearProjectionRunner>(
          this->weights,
          &DotFlowWeights::coordinate_proj,
          this->weights->config.latent_dim,
          this->weights->config.dit.hidden_size,
          "dots_tts.flow.coordinate_proj")),
      speaker_projection(std::make_unique<SpeakerProjectionRunner>(this->weights)),
      dit(std::make_unique<modules::DitRuntime>(*this->weights->execution_context, source,
          dit_runtime_config(this->weights->config, this->weights->execution_context->backend_type(), storage),
          dit_weight_names(this->weights->config))) {}

DotFlowSharedRuntime::~DotFlowSharedRuntime() = default;

std::unique_ptr<DotFlowSharedRuntime> load_flow_shared_runtime(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType storage_type) {
    auto weights = load_weights(source, backend, std::move(config), storage_type);
    return std::make_unique<DotFlowSharedRuntime>(std::move(weights), *source, storage_type);
}

DotsProjectedSequence run_hidden_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & hidden, int64_t steps) {
    return runtime.hidden_projection->run(hidden, steps);
}

DotsProjectedSequence run_latent_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames) {
    return runtime.latent_projection->run(latents, frames);
}

DotsProjectedSequence run_coordinate_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & latents, int64_t frames) {
    return runtime.coordinate_projection->run(latents, frames);
}

DotsProjectedSequence run_speaker_projection(DotFlowSharedRuntime & runtime, const std::vector<float> & speaker) {
    return runtime.speaker_projection->run(speaker);
}

DotModulationOutput run_modulation(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    DotsFlowRuntimeStats * runtime_stats) {
    modules::DitOperationStats stats;
    const auto bank = runtime.dit->prepare_modulations(timesteps, durations, speaker_condition, &stats);
    if (runtime_stats) {
        runtime_stats->modulation_graph_ms += stats.prepare_ms;
        runtime_stats->modulation_input_upload_ms += stats.upload_ms;
        runtime_stats->modulation_compute_ms += stats.compute_ms;
    }
    return bank;
}

DotsVelocityOutput run_velocity(
    DotFlowSharedRuntime & runtime,
    const std::vector<float> & sequence,
    int64_t steps,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    int64_t batch_size,
    const std::vector<int32_t> & positions,
    const std::vector<uint8_t> & attention_mask,
    const DotModulationOutput * modulations,
    int64_t modulation_row_start,
    int64_t output_start,
    int64_t output_length,
    DotsFlowRuntimeStats * runtime_stats,
    std::optional<float> guidance) {
    modules::DitSequenceRequest request{sequence};
    request.frames = steps; request.batch = batch_size;
    request.timesteps = &timesteps; request.durations = &durations; request.global = &speaker_condition;
    request.positions = &positions; request.attention_mask = &attention_mask;
    request.modulation_start = modulations ? modulation_row_start : -1;
    request.output_start = output_start; request.output_tokens = output_length;
    request.guidance = guidance;
    modules::DitOperationStats stats;
    auto values = runtime.dit->predict_sequence(request, &stats);
    if (runtime_stats) {
        ++runtime_stats->velocity_calls;
        runtime_stats->velocity_graph_ms += stats.prepare_ms;
        runtime_stats->velocity_input_upload_ms += stats.upload_ms;
        runtime_stats->velocity_compute_ms += stats.compute_ms;
        runtime_stats->velocity_output_read_ms += stats.read_ms;
    }
    return {std::move(values), output_length > 0 ? output_length : steps, runtime.weights->config.latent_dim};
}

void release_flow_graphs(DotFlowSharedRuntime & runtime) {
    runtime.hidden_projection->release_graph();
    runtime.latent_projection->release_graph();
    runtime.coordinate_projection->release_graph();
    runtime.speaker_projection->release_graph();
    runtime.dit->release_graphs();
}

}  // namespace engine::models::dots_tts::detail

namespace engine::models::dots_tts {

struct DotsFlowDecodeState::Impl {
    detail::FlowDecodeCacheState state;
};

DotsFlowDecodeState::DotsFlowDecodeState()
    : impl_(std::make_unique<Impl>()) {}

DotsFlowDecodeState::~DotsFlowDecodeState() = default;
DotsFlowDecodeState::DotsFlowDecodeState(DotsFlowDecodeState &&) noexcept = default;
DotsFlowDecodeState & DotsFlowDecodeState::operator=(DotsFlowDecodeState &&) noexcept = default;

struct DotsFlowComponent::Impl {
    explicit Impl(std::unique_ptr<detail::DotFlowSharedRuntime> runtime)
        : runtime(std::move(runtime)) {}

    std::unique_ptr<detail::DotFlowSharedRuntime> runtime;
};

DotsFlowComponent DotsFlowComponent::load_from_tensor_source(
    std::shared_ptr<const assets::TensorSource> source,
    core::BackendConfig backend,
    DotsConfig config,
    assets::TensorStorageType weight_storage_type) {
    DotsFlowComponent component;
    component.impl_ = std::make_unique<Impl>(detail::load_flow_shared_runtime(
        std::move(source),
        backend,
        std::move(config),
        weight_storage_type));
    return component;
}

DotsFlowComponent::DotsFlowComponent() = default;
DotsFlowComponent::~DotsFlowComponent() = default;
DotsFlowComponent::DotsFlowComponent(DotsFlowComponent &&) noexcept = default;
DotsFlowComponent & DotsFlowComponent::operator=(DotsFlowComponent &&) noexcept = default;

bool DotsFlowComponent::is_loaded() const noexcept {
    return impl_ != nullptr && impl_->runtime != nullptr;
}

DotsProjectedSequence DotsFlowComponent::project_llm_hidden(const std::vector<float> & hidden, int64_t steps) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_hidden_projection(*impl_->runtime, hidden, steps);
}

DotsProjectedSequence DotsFlowComponent::project_latents(const std::vector<float> & latents, int64_t frames) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_latent_projection(*impl_->runtime, latents, frames);
}

DotsProjectedSequence DotsFlowComponent::project_speaker(const std::vector<float> & speaker) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_speaker_projection(*impl_->runtime, speaker);
}

DotsVelocityOutput DotsFlowComponent::predict_velocity(
    const std::vector<float> & sequence,
    int64_t steps,
    const std::vector<float> & timesteps,
    const std::vector<float> & durations,
    const std::vector<float> & speaker_condition,
    int64_t batch_size,
    const std::vector<int32_t> & positions,
    const std::vector<uint8_t> & attention_mask) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    return detail::run_velocity(
        *impl_->runtime,
        sequence,
        steps,
        timesteps,
        durations,
        speaker_condition,
        batch_size,
        positions,
        attention_mask);
}

DotsLatentMatrix DotsFlowComponent::decode_next_soar(const DotsFlowDecodeRequest & request) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    auto * state = request.decode_state != nullptr ? &request.decode_state->impl_->state : nullptr;
    return detail::decode_next_soar(*impl_->runtime, request, state);
}

DotsLatentMatrix DotsFlowComponent::decode_next_meanflow(const DotsFlowDecodeRequest & request) const {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        throw std::runtime_error("DotTTS flow component is not initialized");
    }
    auto * state = request.decode_state != nullptr ? &request.decode_state->impl_->state : nullptr;
    return detail::decode_next_meanflow(*impl_->runtime, request, state);
}

void DotsFlowComponent::release_decode_cache() {
    if (impl_) impl_->runtime->dit->release_prefix_buffers();
}

void DotsFlowComponent::release_runtime_graphs() {
    if (impl_ == nullptr || impl_->runtime == nullptr) {
        return;
    }
    detail::release_flow_graphs(*impl_->runtime);
}

}  // namespace engine::models::dots_tts
