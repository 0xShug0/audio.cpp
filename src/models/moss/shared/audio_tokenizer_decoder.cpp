#include "engine/models/moss/shared/audio_tokenizer_decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/module.h"
#include "engine/framework/debug/profiler.h"
#include "engine/models/moss/shared/audio_tokenizer_quantizer.h"
#include "engine/models/moss/shared/audio_tokenizer_transformer.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace engine::models::moss {
namespace {

namespace cd = codec_detail;

constexpr int64_t kAttentionQueryChunk = 1500;

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

// PatchedPretransform (decode/upsample): [1, l, d*patch] -> [1, l*patch, d].
// Each frame is unpacked into `patch` consecutive frames along time, matching
// x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l * h).
core::TensorValue patch_upsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = core::ensure_backend_addressable_layout(ctx, input);
    const int64_t length = contiguous.shape.dims[1];
    const int64_t packed = contiguous.shape.dims[2];
    const int64_t channels = packed / patch;
    auto reshaped = engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length, channels, patch}),
    }).build(ctx, contiguous);
    auto transposed = engine::modules::TransposeModule({{0, 1, 3, 2}, reshaped.shape.rank}).build(ctx, reshaped);
    return engine::modules::ReshapeModule({
        core::TensorShape::from_dims({1, length * patch, channels}),
    }).build(ctx, core::ensure_backend_addressable_layout(ctx, transposed));
}

}  // namespace

struct MossAudioTokenizerDecoder::Impl {
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    int64_t sampling_rate = 48000;
    size_t graph_arena_bytes = 0;
    AudioTokenizerConfig config;
    std::unique_ptr<MossAudioTokenizerQuantizer> dequantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<cd::TransformerWeights> transformers;
};

MossAudioTokenizerDecoder::MossAudioTokenizerDecoder(
    const assets::TensorSource & source,
    core::ExecutionContext & execution_context,
    int64_t num_quantizers,
    size_t weight_context_bytes,
    size_t graph_arena_bytes,
    AudioTokenizerConfig config)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec decoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->threads = execution_context.config().threads;
    impl_->config = config;
    impl_->sampling_rate = config.sampling_rate;
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->dequantizer = std::make_unique<MossAudioTokenizerQuantizer>(source, num_quantizers, config.quantizer);

    cd::CodecWeights weights(source);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss.audio_tokenizer.decoder", weight_context_bytes);
    impl_->transformers.reserve(config.decoder_stages.size());
    for (size_t index = 0; index < config.decoder_stages.size(); ++index) {
        const int64_t module_index =
            config.decoder_module_start + static_cast<int64_t>(index) * config.decoder_module_stride;
        impl_->transformers.push_back(cd::load_transformer(
            *impl_->store, weights, to_transformer_spec(config.decoder_stages[index]), "decoder", module_index));
    }
    impl_->store->upload();
}

MossAudioTokenizerDecoder::~MossAudioTokenizerDecoder() = default;

int64_t MossAudioTokenizerDecoder::sampling_rate() const noexcept {
    return impl_->sampling_rate;
}

std::vector<std::vector<float>> MossAudioTokenizerDecoder::decode(
    const std::vector<std::vector<int32_t>> & codes) const {
    const int64_t frames = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec decoder requires a non-empty code sequence");
    }

    // Codes -> continuous latent [code_dim, frames] (channel-major), transposed
    // into the feature-last [1, frames, code_dim] layout the decoder expects.
    double dequant_ms = 0.0;
    double latent_pack_ms = 0.0;
    double graph_build_ms = 0.0;
    double input_upload_ms = 0.0;
    double graph_compute_ms = 0.0;
    double output_read_ms = 0.0;
    double deinterleave_ms = 0.0;
    const bool collect_timing = engine::debug::timing_log_enabled();
    std::vector<float> latent;
    if (collect_timing) {
        dequant_ms = engine::debug::measure_ms([&]() {
            latent = impl_->dequantizer->decode(codes);
        });
    } else {
        latent = impl_->dequantizer->decode(codes);
    }
    std::vector<float> latent_input;
    if (collect_timing) {
        latent_pack_ms = engine::debug::measure_ms([&]() {
            latent_input.resize(static_cast<size_t>(frames * cd::kCodeDim));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * cd::kCodeDim >= 4096)
#endif
            for (int64_t channel = 0; channel < cd::kCodeDim; ++channel) {
                for (int64_t step = 0; step < frames; ++step) {
                    latent_input[static_cast<size_t>(step * cd::kCodeDim + channel)] =
                        latent[static_cast<size_t>(channel * frames + step)];
                }
            }
        });
    } else {
        latent_input.resize(static_cast<size_t>(frames * cd::kCodeDim));
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if(frames * cd::kCodeDim >= 4096)
#endif
        for (int64_t channel = 0; channel < cd::kCodeDim; ++channel) {
            for (int64_t step = 0; step < frames; ++step) {
                latent_input[static_cast<size_t>(step * cd::kCodeDim + channel)] =
                    latent[static_cast<size_t>(channel * frames + step)];
            }
        }
    }

    const auto graph_build_start = std::chrono::steady_clock::now();
    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, cd::GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS codec decoder graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss.audio_tokenizer.decode", impl_->backend_type};

    auto latent_tensor =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, cd::kCodeDim}));
    ggml_set_input(latent_tensor.tensor);

    struct StageInput {
        struct MaskInput {
            ggml_tensor * tensor;
            std::vector<float> host;
        };

        ggml_tensor * positions;
        std::vector<int32_t> position_host;
        std::vector<MaskInput> masks;
    };
    std::vector<StageInput> stage_inputs;
    stage_inputs.reserve(impl_->transformers.size());

    auto hidden = latent_tensor;
    int64_t steps = frames;
    if (impl_->config.decoder_initial_patch > 1) {
        hidden = patch_upsample(ctx, hidden, impl_->config.decoder_initial_patch);
        steps *= impl_->config.decoder_initial_patch;
    }
    for (const auto & transformer : impl_->transformers) {
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(positions.tensor);
        const bool use_windowed_attention = steps > kAttentionQueryChunk;
        core::TensorValue mask;
        if (!use_windowed_attention) {
            mask = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
            ggml_set_input(mask.tensor);
        }

        StageInput stage;
        stage.positions = positions.tensor;
        stage.position_host.resize(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) {
            stage.position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        std::vector<cd::AttentionWindow> windows;
        if (use_windowed_attention) {
            for (int64_t query_start = 0; query_start < steps; query_start += kAttentionQueryChunk) {
                const int64_t query_steps = std::min<int64_t>(kAttentionQueryChunk, steps - query_start);
                const int64_t key_start = std::max<int64_t>(0, query_start - transformer.spec.context + 1);
                const int64_t key_steps = query_start + query_steps - key_start;
                auto window_mask = core::make_tensor(
                    ctx,
                    GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 1, query_steps, key_steps}));
                ggml_set_input(window_mask.tensor);
                stage.masks.push_back(StageInput::MaskInput{
                    window_mask.tensor,
                    cd::causal_context_mask_window(
                        query_start,
                        query_steps,
                        key_start,
                        key_steps,
                        transformer.spec.context),
                });
                windows.push_back(cd::AttentionWindow{
                    query_start,
                    query_steps,
                    key_start,
                    key_steps,
                    window_mask,
                });
            }
        } else {
            stage.masks.push_back(StageInput::MaskInput{mask.tensor, cd::causal_context_mask(steps, transformer.spec.context)});
        }
        stage_inputs.push_back(std::move(stage));

        hidden = windows.empty()
            ? cd::run_transformer(ctx, hidden, transformer, positions, mask, steps)
            : cd::run_transformer(ctx, hidden, transformer, positions, windows.front().mask, steps, &windows);
        hidden = patch_upsample(ctx, hidden, transformer.spec.patch);
        steps *= transformer.spec.patch;
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
        throw std::runtime_error("failed to allocate MOSS codec decoder forward graph");
    }
    if (collect_timing) {
        graph_build_ms = engine::debug::elapsed_ms(graph_build_start);
    }

    const auto upload_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(
        latent_tensor.tensor, latent_input.data(), 0, latent_input.size() * sizeof(float));
    for (const auto & stage : stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        for (const auto & mask : stage.masks) {
            ggml_backend_tensor_set(mask.tensor, mask.host.data(), 0, mask.host.size() * sizeof(float));
        }
    }
    if (collect_timing) {
        input_upload_ms = engine::debug::elapsed_ms(upload_start);
    }

    const auto compute_start = std::chrono::steady_clock::now();
    core::set_backend_threads(impl_->backend, impl_->threads);
    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (collect_timing) {
        graph_compute_ms = engine::debug::elapsed_ms(compute_start);
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS codec decoder forward graph compute failed");
    }

    const int64_t interleaved = steps;  // frames * 3840 * 2 (stereo interleaved)
    std::vector<float> flat(static_cast<size_t>(interleaved));
    const auto read_start = std::chrono::steady_clock::now();
    ggml_backend_tensor_get(hidden.tensor, flat.data(), 0, flat.size() * sizeof(float));
    ggml_gallocr_free(gallocr);
    if (collect_timing) {
        output_read_ms = engine::debug::elapsed_ms(read_start);
    }

    // De-interleave the jointly-processed stream back into left/right channels
    // (channel 0 = even samples, channel 1 = odd samples).
    const int64_t per_channel = frames * cd::kSamplesPerFrame;
    std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(per_channel)));
    if (collect_timing) {
        deinterleave_ms = engine::debug::measure_ms([&]() {
#ifdef _OPENMP
#pragma omp parallel for if(per_channel >= 4096)
#endif
            for (int64_t i = 0; i < per_channel; ++i) {
                stereo[0][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i)];
                stereo[1][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i + 1)];
            }
        });
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.dequant_ms", dequant_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.latent_pack_ms", latent_pack_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.graph_build_ms", graph_build_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.input_upload_ms", input_upload_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.graph_compute_ms", graph_compute_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.output_read_ms", output_read_ms);
        engine::debug::timing_log_scalar("moss.audio_tokenizer.decode.deinterleave_ms", deinterleave_ms);
    } else {
#ifdef _OPENMP
#pragma omp parallel for if(per_channel >= 4096)
#endif
        for (int64_t i = 0; i < per_channel; ++i) {
            stereo[0][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i)];
            stereo[1][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i + 1)];
        }
    }
    return stereo;
}

}  // namespace engine::models::moss
