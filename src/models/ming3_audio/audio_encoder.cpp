#include "engine/models/ming3_audio/audio_encoder.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <stdexcept>

namespace engine::models::ming3_audio {

struct AudioEncoder::Impl {
    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
         AudioConfig config, int64_t hidden_size)
        : execution(execution), source(std::move(source)), config(std::move(config)), hidden_size(hidden_size),
          store(execution.backend(), execution.backend_type(), "ming3_audio.encoder.weights", 4 * 1024 * 1024) {
        namespace binding = modules::binding;
        const auto native = assets::TensorStorageType::Native;
        const auto & c = this->config.encoder;
        const std::string prefix = "model.audio";
        weights.conv1 = binding::conv1d_from_source(store, *this->source, prefix + ".conv1", native,
            c.n_audio_state, c.n_mels, 3, true);
        weights.conv2 = binding::conv1d_from_source(store, *this->source, prefix + ".conv2", native,
            c.n_audio_state, c.n_audio_state, 3, true);
        weights.positional_embedding = store.load_f32_tensor(*this->source, prefix + ".positional_embedding",
            {c.n_audio_ctx, c.n_audio_state});
        for (int64_t i = 0; i < c.n_audio_layer; ++i) {
            const auto p = prefix + ".blocks." + std::to_string(i);
            modules::WhisperEncoderLayerWeights layer;
            layer.attention_norm = binding::norm_from_source(store, *this->source, p + ".attn_ln", c.n_audio_state);
            layer.mlp_norm = binding::norm_from_source(store, *this->source, p + ".mlp_ln", c.n_audio_state);
            layer.attention.query = binding::linear_from_source(store, *this->source, p + ".attn.query", native, c.n_audio_state, c.n_audio_state, true);
            layer.attention.key = binding::linear_from_source(store, *this->source, p + ".attn.key", native, c.n_audio_state, c.n_audio_state, false);
            layer.attention.value = binding::linear_from_source(store, *this->source, p + ".attn.value", native, c.n_audio_state, c.n_audio_state, true);
            layer.attention.out = binding::linear_from_source(store, *this->source, p + ".attn.out", native, c.n_audio_state, c.n_audio_state, true);
            layer.mlp.fc1_weight = store.load_tensor(*this->source, p + ".mlp.0.weight", native, {c.n_audio_state * 4, c.n_audio_state});
            layer.mlp.fc1_bias = store.load_f32_tensor(*this->source, p + ".mlp.0.bias", {c.n_audio_state * 4});
            layer.mlp.fc2_weight = store.load_tensor(*this->source, p + ".mlp.2.weight", native, {c.n_audio_state, c.n_audio_state * 4});
            layer.mlp.fc2_bias = store.load_f32_tensor(*this->source, p + ".mlp.2.bias", {c.n_audio_state});
            weights.layers.push_back(std::move(layer));
        }
        weights.final_norm = binding::norm_from_source(store, *this->source, prefix + ".ln_post", c.n_audio_state);
        int64_t channels = c.n_audio_state;
        for (size_t i = 0; i < this->config.projector_kernels.size(); ++i) {
            projector.push_back(binding::conv1d_from_source(store, *this->source, "linear_proj_audio." + std::to_string(i * 2),
                native, hidden_size, channels, this->config.projector_kernels[i], true));
            channels = hidden_size;
        }
        projection = binding::linear_from_source(store, *this->source,
            "linear_proj_audio." + std::to_string(projector.size() * 2 + 1), native, hidden_size, hidden_size, true);
        store.upload();
    }

    ~Impl() { release_graph(); }

    void release_graph() {
        core::release_backend_graph_resources(execution.backend(), graph, true);
        plan.reset();
        if (allocator) ggml_gallocr_free(allocator);
        if (context) ggml_free(context);
        allocator = nullptr;
        context = nullptr;
        graph = nullptr;
    }

    void build(int64_t frames) {
        release_graph();
        constexpr size_t nodes = 16384;
        context = ggml_init({ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true});
        if (!context) throw std::runtime_error("Ming3 Audio failed to allocate encoder graph metadata");
        core::ModuleBuildContext ctx{context, "ming3_audio.encoder", execution.backend_type()};
        graph = ggml_new_graph_custom(context, nodes, false);
        input = core::make_tensor(ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, config.encoder.n_mels, frames}));
        ggml_set_input(input.tensor);
        auto x = modules::WhisperEmbeddingModule(config.encoder).build(ctx, input, weights);
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        int64_t channels = config.encoder.n_audio_state;
        for (size_t i = 0; i < projector.size(); ++i) {
            const auto kernel = config.projector_kernels[i];
            x = modules::Conv1dModule({channels, hidden_size, kernel,
                static_cast<int>(config.projector_strides[i]), static_cast<int>(kernel / 2), 1, true})
                .build(ctx, x, projector[i]);
            x = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, x);
            channels = hidden_size;
        }
        x = modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
        x = modules::LinearModule({hidden_size, hidden_size, true}).build(ctx, x, projection);
        if (config.normalize_embeddings) x = core::wrap_tensor(ggml_l2_norm(ctx.ggml, x.tensor, 1e-12f), x.shape);
        output = core::ensure_backend_addressable_layout(ctx, x);
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph, output.tensor);
        core::validate_backend_graph_supported(execution.backend(), graph, "ming3_audio.encoder");
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("Ming3 Audio encoder allocation failed");
        core::prepare_host_graph_plan(execution, graph, plan);
        graph_frames = frames;
        debug::timing_log_scalar("ming3_audio.encoder.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(allocator, 0) / (1024.0 * 1024.0));
    }

    std::vector<float> encode(const std::vector<float> & mel, int64_t frames) {
        const auto bins = config.encoder.n_mels;
        if (frames <= 0 || mel.size() != static_cast<size_t>(frames * bins)) {
            throw std::runtime_error("Ming3 Audio log-mel must contain [frames, mel_bins]");
        }
        const auto padded_frames = std::min(frames, config.chunk_frames);
        if (!graph || graph_frames != padded_frames) build(padded_frames);
        std::vector<float> result;
        for (int64_t offset = 0; offset < frames; offset += config.chunk_frames) {
            const auto count = std::min(config.chunk_frames, frames - offset);
            std::vector<float> chunk(bins * padded_frames, 0.0f);
            for (int64_t t = 0; t < count; ++t) {
                for (int64_t bin = 0; bin < bins; ++bin) chunk[bin * padded_frames + t] = mel[(offset + t) * bins + bin];
            }
            core::round_f32_to_bf16_in_place(chunk);
            core::write_tensor_f32(input, chunk);
            if (core::compute_graph(execution, graph, plan, "ming3_audio.encoder") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Ming3 Audio encoder execution failed");
            }
            const auto encoded = core::read_tensor_f32(output.tensor);
            int64_t valid = (count + 1) / 2;
            for (size_t i = 0; i < projector.size(); ++i) {
                const auto kernel = config.projector_kernels[i];
                valid = (valid + 2 * (kernel / 2) - kernel) / config.projector_strides[i] + 1;
            }
            result.insert(result.end(), encoded.begin(), encoded.begin() + valid * hidden_size);
        }
        return result;
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const assets::TensorSource> source;
    AudioConfig config;
    int64_t hidden_size;
    core::BackendWeightStore store;
    modules::WhisperEmbeddingWeights weights;
    std::vector<modules::Conv1dWeights> projector;
    modules::LinearWeights projection;
    ggml_context * context = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    core::TensorValue input;
    core::TensorValue output;
    int64_t graph_frames = 0;
};

AudioEncoder::AudioEncoder(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> source,
                           AudioConfig config, int64_t hidden_size)
    : impl_(std::make_unique<Impl>(execution, std::move(source), std::move(config), hidden_size)) {}
AudioEncoder::~AudioEncoder() = default;
std::vector<float> AudioEncoder::encode(const std::vector<float> & log_mel, int64_t frames) { return impl_->encode(log_mel, frames); }

}  // namespace engine::models::ming3_audio
