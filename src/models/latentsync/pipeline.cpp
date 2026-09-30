#include "engine/models/latentsync/pipeline.h"
#include "engine/models/latentsync/face_runtime.h"
#include "engine/models/latentsync/face_restore.h"
#include "engine/models/latentsync/whisper_runtime.h"
#include "engine/models/latentsync/sd_vae_runtime.h"
#include "engine/models/latentsync/unet_runtime.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::latentsync {

struct LatentSyncPipeline::Impl {
    core::ExecutionContext & execution;
    LatentSyncFaceRuntime face;
    LatentSyncWhisperRuntime whisper;
    LatentSyncSdVaeRuntime vae;
    LatentSyncUnetRuntime unet;
    sampling::TorchCudaSamplingPolicy random_policy;
    Impl(core::ExecutionContext & execution_in, const LatentSyncPipelineSources & source,
        assets::TensorStorageType type, bool vae_cudnn_f16)
        : execution(execution_in),
          face(execution, *source.detector, source.detector_config, *source.landmarks, source.landmark_config, type),
          whisper(execution, *source.whisper, {80, 1500, 384, 6, 4}, type),
          vae(execution, *source.vae, vae_cudnn_f16 ? assets::TensorStorageType::F16 : type,
              512, vae_cudnn_f16, vae_cudnn_f16), unet(execution, *source.unet, {}, type),
          random_policy(sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(), execution.config().device,
              "latentsync.sampling", "LatentSync", sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {}
};

LatentSyncPipeline::LatentSyncPipeline(core::ExecutionContext & execution, const LatentSyncPipelineSources & sources,
    assets::TensorStorageType weight_type, bool vae_cudnn_f16)
    : impl_(std::make_unique<Impl>(execution, sources, weight_type, vae_cudnn_f16)) {}
LatentSyncPipeline::~LatentSyncPipeline() = default;

void LatentSyncPipeline::run(const std::vector<std::vector<uint8_t>> & frames, int height, int width,
    const std::vector<float> & audio_16khz, const std::vector<float> & mask,
    const LatentSyncGenerationOptions & options,
    const std::function<void(size_t, const std::vector<uint8_t> &)> & emit) {
    const auto start = std::chrono::steady_clock::now();
    constexpr size_t pixels = 512 * 512, latent_pixels = 64 * 64, latent_frame = 4 * latent_pixels;
    if (frames.empty() || height <= 0 || width <= 0 || mask.size() != pixels * 3 || !emit ||
        options.frames_per_chunk <= 0 || options.frames_per_chunk > 24 || !std::isfinite(options.fps) || options.fps <= 0 ||
        options.steps <= 0 || !std::isfinite(options.guidance_scale) || options.guidance_scale < 0) {
        throw std::runtime_error("Invalid LatentSync pipeline input or generation options");
    }
    const size_t frame_pixels = static_cast<size_t>(height) * width;
    for (const auto & frame : frames) if (frame.size() != frame_pixels * 3) throw std::runtime_error("LatentSync video frame size mismatch");
    const auto features = impl_->whisper.encode_audio(audio_16khz, impl_->execution.config().threads);
    const int64_t feature_length = features.size() / (5 * 384);
    // Match feature2chunks: it appends the first frame whose center is beyond the feature sequence.
    size_t output_frames = 0;
    do { ++output_frames; } while (int64_t((output_frames - 1) * 50 / options.fps) <= feature_length);
    const size_t unique_frames = std::min(frames.size(), output_frames);
    impl_->face.reset_alignment();
    std::vector<LatentSyncAlignedFace> aligned;
    aligned.reserve(unique_frames);
    std::vector<float> original(frame_pixels * 3);
    const auto alignment_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < unique_frames; ++i) {
        for (size_t p = 0; p < frame_pixels; ++p) for (size_t c = 0; c < 3; ++c) original[c * frame_pixels + p] = frames[i][p * 3 + c];
        aligned.push_back(impl_->face.align_frame(original, height, width));
    }
    debug::timing_log_scalar("latentsync.face_alignment.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - alignment_start).count());
    double restoration_ms = 0;
    uint64_t offset = 0;
    const auto normal = [&](size_t count) {
        auto values = sampling::generate_torch_cuda_tensor_iterator_randn(count, options.seed, offset, impl_->random_policy);
        offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, impl_->random_policy);
        return values;
    };
    const auto initial_noise = normal(latent_frame);
    const auto frame_index = [&](size_t index) {
        const size_t cycle = index / frames.size(), position = index % frames.size();
        return cycle % 2 ? frames.size() - 1 - position : position;
    };
    for (size_t first = 0; first < output_frames; first += options.frames_per_chunk) {
        const size_t count = std::min<size_t>(options.frames_per_chunk, output_frames - first);
        LatentSyncLatentClip clip;
        clip.frames = count; clip.height = clip.width = 64;
        clip.audio = latentsync_audio_windows(features, first, count, options.fps);
        clip.noise.reserve(count * latent_frame);
        clip.mask.reserve(count * latent_pixels);
        const auto masked_noise = normal(count * latent_frame);
        const auto reference_noise = normal(count * latent_frame);
        for (size_t i = 0; i < count; ++i) {
            clip.noise.insert(clip.noise.end(), initial_noise.begin(), initial_noise.end());
            for (int y = 0; y < 64; ++y) for (int x = 0; x < 64; ++x) clip.mask.push_back(mask[y * 8 * 512 + x * 8]);
        }
        for (size_t i = 0; i < count; ++i) {
            const auto & face = aligned[frame_index(first + i)];
            std::vector<float> masked(pixels * 3);
            for (size_t p = 0; p < pixels * 3; ++p) masked[p] = face.pixels[p] * mask[p];
            const std::vector<float> noise(masked_noise.begin() + i * latent_frame, masked_noise.begin() + (i + 1) * latent_frame);
            const auto encoded = impl_->vae.encode(masked, noise);
            clip.masked_image.insert(clip.masked_image.end(), encoded.begin(), encoded.end());
        }
        for (size_t i = 0; i < count; ++i) {
            const auto & face = aligned[frame_index(first + i)];
            const std::vector<float> noise(reference_noise.begin() + i * latent_frame, reference_noise.begin() + (i + 1) * latent_frame);
            const auto encoded = impl_->vae.encode(face.pixels, noise);
            clip.reference_image.insert(clip.reference_image.end(), encoded.begin(), encoded.end());
        }
        const auto latents = impl_->unet.denoise(clip, options.steps, options.guidance_scale, {}, options.cache_interval);
        for (size_t i = 0; i < count; ++i) {
            const std::vector<float> latent(latents.begin() + i * latent_frame, latents.begin() + (i + 1) * latent_frame);
            auto decoded = impl_->vae.decode(latent);
            const size_t index = frame_index(first + i);
            const auto restoration_start = std::chrono::steady_clock::now();
            for (size_t p = 0; p < decoded.size(); ++p) decoded[p] = decoded[p] * (1 - mask[p]) + aligned[index].pixels[p] * mask[p];
            for (size_t p = 0; p < frame_pixels; ++p) for (size_t c = 0; c < 3; ++c) original[c * frame_pixels + p] = frames[index][p * 3 + c];
            const auto restored = latentsync_restore_face(original, height, width, decoded, aligned[index].transform,
                impl_->execution.config().threads);
            std::vector<uint8_t> output(frame_pixels * 3);
            for (size_t p = 0; p < frame_pixels; ++p) for (size_t c = 0; c < 3; ++c) output[p * 3 + c] = static_cast<uint8_t>(restored[c * frame_pixels + p]);
            emit(first + i, output);
            restoration_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - restoration_start).count();
        }
    }
    debug::timing_log_scalar("latentsync.face_restoration.wall_ms", restoration_ms);
    debug::timing_log_scalar("latentsync.pipeline_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace engine::models::latentsync
