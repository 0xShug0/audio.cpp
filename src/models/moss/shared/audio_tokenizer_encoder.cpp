#include "engine/models/moss/shared/audio_tokenizer_encoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/models/moss/shared/audio_tokenizer_quantizer.h"
#include "engine/models/moss/shared/audio_tokenizer_transformer.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::moss {
namespace {

namespace cd = codec_detail;

cd::TransformerSpec to_transformer_spec(const AudioTokenizerTransformerStage & stage) {
    return {
        stage.input_dimension,
        stage.output_dimension,
        stage.model_dimension,
        stage.num_heads,
        stage.num_layers,
        stage.feedforward_dimension,
        stage.context_window,
        stage.patch_size,
    };
}

// PatchedPretransform (encode/downsample): [1, l, d] -> [1, l/patch, d*patch].
// Packs `patch` consecutive frames into the feature dim, matching
// x.reshape(b, d, -1, h).permute(0, 1, 3, 2).reshape(b, d * h, -1) (conv layout),
// i.e. output feature (d_idx*patch + h_idx) at time lt = input feature d_idx at
// time lt*patch + h_idx. This is the exact inverse of the decoder's upsample.
core::TensorValue patch_downsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t total_length = contiguous.shape.dims[1];
    const int64_t channels = contiguous.shape.dims[2];
    const int64_t length = total_length / patch;
    auto reshaped = engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, patch, channels}),
    }).build(ctx, contiguous);
    auto transposed = engine::modules::TransposeModule({{0, 1, 3, 2}, reshaped.shape.rank}).build(ctx, reshaped);
    return engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, channels * patch}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, transposed));
}

}  // namespace

struct MossAudioTokenizerEncoder::Impl {
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    int64_t samples_per_frame = 3840;
    size_t graph_arena_bytes = 0;
    AudioTokenizerConfig config;
    std::unique_ptr<MossAudioTokenizerQuantizer> quantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<cd::TransformerWeights> transformers;
};

MossAudioTokenizerEncoder::MossAudioTokenizerEncoder(
    const assets::TensorSource & source,
    core::ExecutionContext & execution_context,
    int64_t num_quantizers,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    AudioTokenizerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec encoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->config = config;
    impl_->samples_per_frame = config.samples_per_frame;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->quantizer = std::make_unique<MossAudioTokenizerQuantizer>(source, num_quantizers, config.quantizer);

    cd::CodecWeights weights(source);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss.audio_tokenizer.encoder", weight_context_bytes);
    impl_->transformers.reserve(config.encoder_stages.size());
    for (size_t index = 0; index < config.encoder_stages.size(); ++index) {
        const int64_t module_index =
            config.encoder_module_start + static_cast<int64_t>(index) * config.encoder_module_stride;
        impl_->transformers.push_back(cd::load_transformer(
            *impl_->store, weights, to_transformer_spec(config.encoder_stages[index]), "encoder", module_index));
    }
    impl_->store->upload();
}

MossAudioTokenizerEncoder::~MossAudioTokenizerEncoder() = default;

std::vector<std::vector<int32_t>> MossAudioTokenizerEncoder::encode(
    const std::vector<std::vector<float>> & channels) const {
    if (channels.size() != 2) {
        throw std::runtime_error("MOSS codec encoder requires stereo (2-channel) input");
    }
    if (channels[0].size() != channels[1].size()) {
        throw std::runtime_error("MOSS codec encoder channels must have equal length");
    }
    const int64_t raw_per_channel = static_cast<int64_t>(channels[0].size());
    if (raw_per_channel <= 0) {
        throw std::runtime_error("MOSS codec encoder requires a non-empty waveform");
    }

    // Pad each channel up to a multiple of the downsample rate for the encoder graph,
    // but keep the official valid code length as floor(valid_samples / samples_per_frame).
    // MossAudioTokenizerPatchedPretransform pads the tensor and propagates input_lengths
    // with integer division, then slices audio_codes to audio_codes_lengths.
    const int64_t frames = (raw_per_channel + impl_->samples_per_frame - 1) / impl_->samples_per_frame;
    const int64_t valid_frames = raw_per_channel / impl_->samples_per_frame;
    const int64_t per_channel = frames * impl_->samples_per_frame;
    const int64_t interleaved = per_channel * 2;
    std::vector<float> waveform(static_cast<size_t>(interleaved), 0.0F);
#ifdef _OPENMP
#pragma omp parallel for if(raw_per_channel >= 4096)
#endif
    for (int64_t i = 0; i < raw_per_channel; ++i) {
        waveform[static_cast<size_t>(2 * i)] = channels[0][static_cast<size_t>(i)];
        waveform[static_cast<size_t>(2 * i + 1)] = channels[1][static_cast<size_t>(i)];
    }

    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, cd::GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS codec encoder graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss.audio_tokenizer.encode", impl_->backend_type};

    // Input is the interleaved waveform as a single-feature stream [1, L, 1].
    auto input_tensor =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, interleaved, 1}));
    ggml_set_input(input_tensor.tensor);

    struct StageInput {
        ggml_tensor * positions;
        std::vector<int32_t> position_host;
        ggml_tensor * mask;
        std::vector<float> mask_host;
    };
    std::vector<StageInput> stage_inputs;
    stage_inputs.reserve(impl_->transformers.size());

    auto hidden = input_tensor;
    int64_t steps = interleaved;
    for (const auto & transformer : impl_->transformers) {
        // Downsample first (encoder order), then run the transformer at the
        // reduced frame count.
        hidden = patch_downsample(ctx, hidden, transformer.spec.patch);
        steps /= transformer.spec.patch;

        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(positions.tensor);
        auto mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
        ggml_set_input(mask.tensor);

        StageInput stage;
        stage.positions = positions.tensor;
        stage.position_host.resize(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) {
            stage.position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        stage.mask = mask.tensor;
        stage.mask_host = cd::causal_context_mask(steps, transformer.spec.context);
        stage_inputs.push_back(std::move(stage));

        hidden = cd::run_transformer(ctx, hidden, transformer, positions, mask, steps);
    }
    if (impl_->config.encoder_final_patch > 1) {
        hidden = patch_downsample(ctx, hidden, impl_->config.encoder_final_patch);
        steps /= impl_->config.encoder_final_patch;
    }

    hidden = core::ensure_backend_addressable_layout(ctx, hidden);
    ggml_set_output(hidden.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 131072, false);
    ggml_build_forward_expand(graph, hidden.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS codec encoder forward graph");
    }

    ggml_backend_tensor_set(input_tensor.tensor, waveform.data(), 0, waveform.size() * sizeof(float));
    for (const auto & stage : stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            stage.mask, stage.mask_host.data(), 0, stage.mask_host.size() * sizeof(float));
    }

    core::set_backend_threads(impl_->backend, impl_->threads);
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS codec encoder forward graph compute failed");
    }

    // hidden is [1, frames, code_dim] feature-last; ggml memory order is
    // channel-fastest, i.e. flat[frame * code_dim + channel] -- exactly the
    // layout MossAudioTokenizerQuantizer::encode expects.
    std::vector<float> latent(static_cast<size_t>(steps * cd::kCodeDim));
    ggml_backend_tensor_get(hidden.tensor, latent.data(), 0, latent.size() * sizeof(float));
    ggml_gallocr_free(gallocr);

    if (valid_frames <= 0) {
        throw std::runtime_error("MOSS codec encoder input is shorter than one codec frame");
    }
    latent.resize(static_cast<size_t>(valid_frames * cd::kCodeDim));
    return impl_->quantizer->encode(latent, valid_frames);
}

}  // namespace engine::models::moss
