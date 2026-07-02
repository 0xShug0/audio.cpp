#include "engine/models/moss_tts_local/codec_decoder.h"

#include "engine/framework/core/module.h"
#include "engine/models/moss_tts_local/codec_quantizer.h"
#include "engine/models/moss_tts_local/codec_transformer.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

namespace cd = codec_detail;

// Decoder ProjectedTransformer stages, mirroring the v2 codec config.json
// decoder_kwargs (modules decoder.0/2/4/6/8/10). context is the local-attention
// window at that stage's frame rate; the patch that follows expands the frame
// rate (the last one by 240).
constexpr cd::TransformerSpec kDecoderSpecs[] = {
    {cd::kCodeDim, 1280, 1280, 20, 32, 5120, 125, 2},
    {640, 768, 768, 12, 12, 3072, 250, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 240, 768, 12, 12, 3072, 400, 240},
};

constexpr size_t kNumTransformers = sizeof(kDecoderSpecs) / sizeof(kDecoderSpecs[0]);

// PatchedPretransform (decode/upsample): [1, l, d*patch] -> [1, l*patch, d].
// Each frame is unpacked into `patch` consecutive frames along time, matching
// x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l * h).
core::TensorValue patch_upsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = cd::ensure_contiguous(ctx, input);
    const int64_t length = contiguous.shape.dims[1];
    const int64_t packed = contiguous.shape.dims[2];
    const int64_t channels = packed / patch;
    ggml_tensor * reshaped = ggml_reshape_3d(ctx.ggml, contiguous.tensor, patch, channels, length);
    ggml_tensor * permuted = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, reshaped, 1, 0, 2, 3));
    ggml_tensor * unpacked = ggml_reshape_2d(ctx.ggml, permuted, channels, length * patch);
    return core::wrap_tensor(
        unpacked, core::TensorShape::from_dims({1, length * patch, channels}), GGML_TYPE_F32);
}

}  // namespace

struct MossCodecDecoder::Impl {
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    size_t graph_arena_bytes = 0;
    std::unique_ptr<MossCodecDequantizer> dequantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<cd::TransformerWeights> transformers;
};

MossCodecDecoder::MossCodecDecoder(
    const std::filesystem::path & codec_dir,
    core::ExecutionContext & execution_context,
    int64_t num_quantizers,
    size_t weight_context_bytes,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec decoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->dequantizer = std::make_unique<MossCodecDequantizer>(codec_dir, num_quantizers);

    cd::CodecShards shards(codec_dir);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss_tts_local.codec.decoder", weight_context_bytes);
    impl_->transformers.reserve(kNumTransformers);
    for (size_t index = 0; index < kNumTransformers; ++index) {
        impl_->transformers.push_back(cd::load_transformer(
            *impl_->store, shards, kDecoderSpecs[index], "decoder", static_cast<int64_t>(2 * index)));
    }
    impl_->store->upload();
}

MossCodecDecoder::~MossCodecDecoder() = default;

int64_t MossCodecDecoder::sampling_rate() const noexcept {
    return 48000;
}

std::vector<std::vector<float>> MossCodecDecoder::decode(
    const std::vector<std::vector<int32_t>> & codes) const {
    const int64_t frames = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec decoder requires a non-empty code sequence");
    }

    // Codes -> continuous latent [code_dim, frames] (channel-major), transposed
    // into the feature-last [1, frames, code_dim] layout the decoder expects.
    const auto latent = impl_->dequantizer->decode(codes);
    std::vector<float> latent_input(static_cast<size_t>(frames * cd::kCodeDim));
    for (int64_t channel = 0; channel < cd::kCodeDim; ++channel) {
        for (int64_t step = 0; step < frames; ++step) {
            latent_input[static_cast<size_t>(step * cd::kCodeDim + channel)] =
                latent[static_cast<size_t>(channel * frames + step)];
        }
    }

    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, cd::GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS codec decoder graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_local.codec.decode", impl_->backend_type};

    auto latent_tensor =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, cd::kCodeDim}));
    ggml_set_input(latent_tensor.tensor);

    struct StageInput {
        ggml_tensor * positions;
        std::vector<int32_t> position_host;
        ggml_tensor * mask;
        std::vector<float> mask_host;
    };
    std::vector<StageInput> stage_inputs;
    stage_inputs.reserve(impl_->transformers.size());

    auto hidden = latent_tensor;
    int64_t steps = frames;
    for (const auto & transformer : impl_->transformers) {
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
        hidden = patch_upsample(ctx, hidden, transformer.spec.patch);
        steps *= transformer.spec.patch;
    }

    hidden = cd::ensure_contiguous(ctx, hidden);
    ggml_set_output(hidden.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 131072, false);
    ggml_build_forward_expand(graph, hidden.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS codec decoder forward graph");
    }

    ggml_backend_tensor_set(
        latent_tensor.tensor, latent_input.data(), 0, latent_input.size() * sizeof(float));
    for (const auto & stage : stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            stage.mask, stage.mask_host.data(), 0, stage.mask_host.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS codec decoder forward graph compute failed");
    }

    const int64_t interleaved = steps;  // frames * 3840 * 2 (stereo interleaved)
    std::vector<float> flat(static_cast<size_t>(interleaved));
    ggml_backend_tensor_get(hidden.tensor, flat.data(), 0, flat.size() * sizeof(float));
    ggml_gallocr_free(gallocr);

    // De-interleave the jointly-processed stream back into left/right channels
    // (channel 0 = even samples, channel 1 = odd samples).
    const int64_t per_channel = frames * cd::kSamplesPerFrame;
    std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(per_channel)));
    for (int64_t i = 0; i < per_channel; ++i) {
        stereo[0][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i)];
        stereo[1][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i + 1)];
    }
    return stereo;
}

}  // namespace engine::models::moss_tts_local
