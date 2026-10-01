#include "engine/community_models/latentsync/whisper_runtime.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::latentsync {

struct LatentSyncWhisperRuntime::Impl {
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    modules::WhisperEmbeddingWeights weights;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    core::TensorValue input, output;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        modules::WhisperEmbeddingConfig config, assets::TensorStorageType type)
        : execution(execution_in),
          store(execution.backend(), execution.backend_type(), "latentsync.whisper.weights", 4 * 1024 * 1024),
          weights(load_whisper_encoder_weights(store, source, config, type)) {
        store.upload();
        constexpr size_t nodes = 4096;
        context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context) throw std::runtime_error("LatentSync Whisper graph context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "latentsync.whisper", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, config.n_mels, config.n_audio_ctx * 2}));
        ggml_set_input(input.tensor);
        output = LatentSyncWhisperEncoder(config).build(ctx, input, weights);
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), nodes, false);
        ggml_build_forward_expand(graph, output.tensor);
        allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("LatentSync Whisper graph allocation failed");
        debug::timing_log_scalar("latentsync.whisper.graph_buffer_mb", ggml_gallocr_get_buffer_size(allocator.get(), 0) / 1048576.0);
        audio::MelSpectrogramFrontendConfig mel;
        mel.sample_rate = 16000;
        mel.n_fft = mel.win_length = 400;
        mel.hop_length = 160;
        mel.n_mels = 80;
        mel.stft_center = true;
        mel.waveform_padding = audio::MelWaveformPadding::None;
        mel.spectrum_mode = audio::MelSpectrumMode::PowerDuringProjection;
        mel.value_transform = audio::MelValueTransform::Log10;
        mel.log_floor = 1e-10;
        mel.drop_last_frames = 1;
        mel.log_dynamic_range = 8;
        mel.log_shift = 4;
        mel.log_divisor = 4;
        frontend = audio::get_cached_mel_spectrogram_frontend(mel);
    }
    ~Impl() { core::release_backend_graph_resources(execution.backend(), graph, true); }
};

LatentSyncWhisperRuntime::LatentSyncWhisperRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, modules::WhisperEmbeddingConfig config, assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(execution, source, config, weight_type)) {}
LatentSyncWhisperRuntime::~LatentSyncWhisperRuntime() = default;

std::vector<float> LatentSyncWhisperRuntime::encode_mel(const std::vector<float> & mel) {
    const auto start = std::chrono::steady_clock::now();
    core::write_tensor_f32(impl_->input, mel);
    if (core::compute_backend_graph(impl_->execution.backend(), impl_->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LatentSync Whisper execution failed");
    }
    auto output = core::read_tensor_f32(impl_->output.tensor);
    debug::timing_log_scalar("latentsync.whisper.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncWhisperRuntime::encode_audio(const std::vector<float> & mono_16khz, size_t threads) {
    const auto start = std::chrono::steady_clock::now();
    const auto mel = impl_->frontend->extract_mono(mono_16khz, threads);
    if (mel.frames < 2) throw std::runtime_error("LatentSync audio is too short for Whisper conditioning");
    constexpr int64_t window = 3000, row = 5 * 384;
    std::vector<float> output;
    output.reserve(static_cast<size_t>(mel.frames / 2) * row);
    std::vector<float> segment(80 * window);
    for (int64_t offset = 0; offset < mel.frames; offset += window) {
        std::fill(segment.begin(), segment.end(), 0.0F);
        const int64_t count = std::min(window, mel.frames - offset);
        for (int64_t bin = 0; bin < 80; ++bin) {
            std::copy_n(mel.values.begin() + bin * mel.frames + offset, count, segment.begin() + bin * window);
        }
        const auto embeddings = encode_mel(segment);
        output.insert(output.end(), embeddings.begin(), embeddings.begin() + count / 2 * row);
    }
    debug::timing_log_scalar("latentsync.audio_conditioning.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> latentsync_audio_windows(const std::vector<float> & features,
    int64_t first_frame, int64_t frames, double fps) {
    constexpr int64_t row = 5 * 384;
    if (features.empty() || features.size() % row || first_frame < 0 || frames <= 0 || !std::isfinite(fps) || fps <= 0) {
        throw std::runtime_error("Invalid LatentSync audio window dimensions");
    }
    const int64_t length = features.size() / row;
    std::vector<float> output(static_cast<size_t>(frames) * 10 * row);
    for (int64_t frame = 0; frame < frames; ++frame) {
        const int64_t center = static_cast<int64_t>((first_frame + frame) * 50 / fps);
        for (int64_t i = 0; i < 10; ++i) {
            const int64_t index = std::clamp<int64_t>(center - 4 + i, 0, length - 1);
            std::copy_n(features.begin() + index * row, row, output.begin() + (frame * 10 + i) * row);
        }
    }
    return output;
}

}  // namespace engine::models::latentsync
