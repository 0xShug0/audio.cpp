#include "engine/community_models/latentsync/pipeline.h"
#include "engine/community_models/latentsync/face_runtime.h"
#include "engine/community_models/latentsync/face_restore.h"
#include "engine/community_models/latentsync/whisper_runtime.h"
#include "engine/community_models/latentsync/sd_vae_runtime.h"
#include "engine/community_models/latentsync/unet_runtime.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace engine::models::latentsync {

struct LatentSyncPipeline::Impl {
    core::ExecutionContext & execution;
    LatentSyncPipelineSources sources;
    assets::TensorStorageType weight_type;
    bool vae_cudnn_f16;
    bool unet_cudnn_f16;
    bool mem_saver;
    std::unique_ptr<LatentSyncFaceRuntime> face;
    std::unique_ptr<LatentSyncWhisperRuntime> whisper;
    std::unique_ptr<LatentSyncSdVaeRuntime> vae;
    std::unique_ptr<LatentSyncUnetRuntime> unet;
    sampling::TorchCudaSamplingPolicy random_policy;
    Impl(core::ExecutionContext & execution_in, const LatentSyncPipelineSources & source,
        assets::TensorStorageType type, bool vae_cudnn_f16_in,
        bool unet_cudnn_f16_in, bool mem_saver_in)
        : execution(execution_in), sources(source), weight_type(type),
          vae_cudnn_f16(vae_cudnn_f16_in), unet_cudnn_f16(unet_cudnn_f16_in),
          mem_saver(mem_saver_in),
          random_policy(sampling::resolve_torch_cuda_sampling_policy(execution.backend_type(), execution.config().device,
              "latentsync.sampling", "LatentSync", sampling::TorchCudaSamplingPolicyFailureMode::FallbackToDefault)) {
        if (!mem_saver) {
            face = std::make_unique<LatentSyncFaceRuntime>(execution, *sources.detector,
                sources.detector_config, *sources.landmarks, sources.landmark_config, weight_type);
            whisper = std::make_unique<LatentSyncWhisperRuntime>(execution, *sources.whisper,
                modules::WhisperEmbeddingConfig{80, 1500, 384, 6, 4}, weight_type);
            vae = std::make_unique<LatentSyncSdVaeRuntime>(execution, *sources.vae,
                vae_cudnn_f16 ? assets::TensorStorageType::F16 : weight_type,
                512, vae_cudnn_f16, vae_cudnn_f16, true);
            unet = std::make_unique<LatentSyncUnetRuntime>(execution, *sources.unet,
                LatentSyncUnetConfig(unet_cudnn_f16), weight_type);
        }
    }
};

LatentSyncPipeline::LatentSyncPipeline(core::ExecutionContext & execution, const LatentSyncPipelineSources & sources,
    assets::TensorStorageType weight_type, bool vae_cudnn_f16, bool unet_cudnn_f16, bool mem_saver)
    : impl_(std::make_unique<Impl>(execution, sources, weight_type, vae_cudnn_f16, unet_cudnn_f16, mem_saver)) {}
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
    std::vector<float> features;
    if (impl_->mem_saver) {
        {
            LatentSyncWhisperRuntime whisper(impl_->execution, *impl_->sources.whisper,
                {80, 1500, 384, 6, 4}, impl_->weight_type);
            features = whisper.encode_audio(audio_16khz, impl_->execution.config().threads);
        }
        core::trim_backend_pools(impl_->execution.backend());
    } else {
        features = impl_->whisper->encode_audio(audio_16khz, impl_->execution.config().threads);
    }
    const int64_t feature_length = features.size() / (5 * 384);
    // Match feature2chunks: it appends the first frame whose center is beyond the feature sequence.
    size_t output_frames = 0;
    do { ++output_frames; } while (int64_t((output_frames - 1) * 50 / options.fps) <= feature_length);
    const size_t unique_frames = std::min(frames.size(), output_frames);
    std::vector<LatentSyncAlignedFace> aligned;
    aligned.reserve(unique_frames);
    std::vector<float> original(frame_pixels * 3);
    const auto alignment_start = std::chrono::steady_clock::now();
    const auto align_frames = [&](LatentSyncFaceRuntime & face) {
        face.reset_alignment();
        for (size_t i = 0; i < unique_frames; ++i) {
            for (size_t p = 0; p < frame_pixels; ++p) {
                for (size_t c = 0; c < 3; ++c) original[c * frame_pixels + p] = frames[i][p * 3 + c];
            }
            aligned.push_back(face.align_frame(original, height, width));
        }
    };
    if (impl_->mem_saver) {
        {
            LatentSyncFaceRuntime face(impl_->execution, *impl_->sources.detector,
                impl_->sources.detector_config, *impl_->sources.landmarks,
                impl_->sources.landmark_config, impl_->weight_type);
            align_frames(face);
        }
        core::trim_backend_pools(impl_->execution.backend());
    } else {
        align_frames(*impl_->face);
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

    struct PreparedChunk {
        size_t first = 0;
        LatentSyncLatentClip clip;
        std::vector<float> latents;
    };
    const auto prepare_chunk = [&](LatentSyncSdVaeRuntime & vae, size_t first) {
        const size_t count = std::min<size_t>(options.frames_per_chunk, output_frames - first);
        PreparedChunk prepared;
        prepared.first = first;
        auto & clip = prepared.clip;
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
            const auto encoded = vae.encode(masked, noise);
            clip.masked_image.insert(clip.masked_image.end(), encoded.begin(), encoded.end());
        }
        for (size_t i = 0; i < count; ++i) {
            const auto & face = aligned[frame_index(first + i)];
            const std::vector<float> noise(reference_noise.begin() + i * latent_frame, reference_noise.begin() + (i + 1) * latent_frame);
            const auto encoded = vae.encode(face.pixels, noise);
            clip.reference_image.insert(clip.reference_image.end(), encoded.begin(), encoded.end());
        }
        return prepared;
    };
    const auto restore_chunk = [&](LatentSyncSdVaeRuntime & vae, PreparedChunk & prepared) {
        const size_t first = prepared.first;
        const size_t count = static_cast<size_t>(prepared.clip.frames);
        std::vector<std::vector<float>> decoded_faces(count);
        for (size_t i = 0; i < count; ++i) {
            const std::vector<float> latent(prepared.latents.begin() + i * latent_frame,
                prepared.latents.begin() + (i + 1) * latent_frame);
            decoded_faces[i] = vae.decode(latent);
        }
        const auto restoration_start = std::chrono::steady_clock::now();
        std::vector<std::vector<uint8_t>> restored_frames(count);
        const int restoration_workers = std::min<int>(impl_->execution.config().threads, count);
#ifdef _OPENMP
#pragma omp parallel for num_threads(restoration_workers) if(restoration_workers > 1) schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
            const size_t index = frame_index(first + i);
            auto & decoded = decoded_faces[i];
            for (size_t p = 0; p < decoded.size(); ++p) decoded[p] = decoded[p] * (1 - mask[p]) + aligned[index].pixels[p] * mask[p];
            std::vector<float> source(frame_pixels * 3);
            for (size_t p = 0; p < frame_pixels; ++p) for (size_t c = 0; c < 3; ++c) source[c * frame_pixels + p] = frames[index][p * 3 + c];
            const auto restored = latentsync_restore_face(source, height, width, decoded,
                aligned[index].transform, restoration_workers > 1 ? 1 : impl_->execution.config().threads);
            auto & output = restored_frames[i];
            output.resize(frame_pixels * 3);
            for (size_t p = 0; p < frame_pixels; ++p) for (size_t c = 0; c < 3; ++c) output[p * 3 + c] = static_cast<uint8_t>(restored[c * frame_pixels + p]);
        }
        for (size_t i = 0; i < count; ++i) emit(first + i, restored_frames[i]);
        restoration_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - restoration_start).count();
    };

    if (!impl_->mem_saver) {
        for (size_t first = 0; first < output_frames; first += options.frames_per_chunk) {
            auto prepared = prepare_chunk(*impl_->vae, first);
            prepared.latents = impl_->unet->denoise(prepared.clip, options.steps,
                options.guidance_scale, {}, options.cache_interval);
            restore_chunk(*impl_->vae, prepared);
        }
    } else {
        std::vector<PreparedChunk> prepared_chunks;
        prepared_chunks.reserve((output_frames + options.frames_per_chunk - 1) / options.frames_per_chunk);
        {
            LatentSyncSdVaeRuntime vae(impl_->execution, *impl_->sources.vae,
                impl_->vae_cudnn_f16 ? assets::TensorStorageType::F16 : impl_->weight_type,
                512, impl_->vae_cudnn_f16, impl_->vae_cudnn_f16, true);
            for (size_t first = 0; first < output_frames; first += options.frames_per_chunk) {
                prepared_chunks.push_back(prepare_chunk(vae, first));
            }
        }
        core::trim_backend_pools(impl_->execution.backend());
        {
            LatentSyncUnetRuntime unet(impl_->execution, *impl_->sources.unet,
                LatentSyncUnetConfig(impl_->unet_cudnn_f16, true), impl_->weight_type);
            for (auto & prepared : prepared_chunks) {
                prepared.latents = unet.denoise(prepared.clip, options.steps,
                    options.guidance_scale, {}, options.cache_interval);
            }
        }
        core::trim_backend_pools(impl_->execution.backend());
        {
            LatentSyncSdVaeRuntime vae(impl_->execution, *impl_->sources.vae,
                impl_->vae_cudnn_f16 ? assets::TensorStorageType::F16 : impl_->weight_type,
                512, impl_->vae_cudnn_f16, impl_->vae_cudnn_f16, true);
            for (auto & prepared : prepared_chunks) restore_chunk(vae, prepared);
        }
        core::trim_backend_pools(impl_->execution.backend());
    }
    debug::timing_log_scalar("latentsync.face_restoration.wall_ms", restoration_ms);
    debug::timing_log_scalar("latentsync.pipeline_ms", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
}

}  // namespace engine::models::latentsync
