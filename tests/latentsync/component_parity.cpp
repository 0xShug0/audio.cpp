#include "engine/community_models/latentsync/whisper_encoder.h"
#include "engine/community_models/latentsync/whisper_runtime.h"
#include "engine/community_models/latentsync/pipeline.h"
#include "engine/community_models/latentsync/video.h"
#include "engine/community_models/latentsync/sd_vae.h"
#include "engine/community_models/latentsync/sd_vae_runtime.h"
#include "engine/community_models/latentsync/unet_attention.h"
#include "engine/community_models/latentsync/unet.h"
#include "engine/community_models/latentsync/unet_runtime.h"
#include "engine/community_models/latentsync/sampling.h"
#include "engine/community_models/latentsync/face_landmarks.h"
#include "engine/community_models/latentsync/scrfd.h"
#include "engine/community_models/latentsync/face_runtime.h"
#include "engine/community_models/latentsync/face_alignment.h"
#include "engine/community_models/latentsync/face_restore.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/model_spec/schema.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

int main(int argc, char ** argv) {
    using namespace engine;
    try {
        if (argc != 5 || std::string(argv[4]) != "--log") {
            throw std::runtime_error("usage: latentsync_component_parity <fixture-dir or video-path> <cpu|cuda|vulkan> <component> --log");
        }
        debug::configure_logging({true});
        const std::filesystem::path root(argv[1]);
        core::BackendConfig backend;
        const std::string name(argv[2]);
        const std::string component(argv[3]);
        if (component == "spec") {
            const auto start = std::chrono::steady_clock::now();
            model_spec::validate_spec(io::json::parse_file(root), root.string());
            std::cout << "spec validation passed\n";
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return 0;
        }
        if (component == "video") {
            const auto start = std::chrono::steady_clock::now();
            const auto video = models::latentsync::load_latentsync_video(root, 25);
            const size_t pixels = static_cast<size_t>(video.height) * video.width * 3;
            for (const auto & frame : video.frames) {
                if (frame.size() != pixels) throw std::runtime_error("Video frame size mismatch");
            }
            std::cout << "width=" << video.width << " height=" << video.height << " frames=" << video.frames.size() << '\n';
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return 0;
        }
        if (component == "restoration") {
            const auto start = std::chrono::steady_clock::now();
            const auto reference = assets::open_tensor_source(root / "reference.safetensors");
            const auto image = reference->require_f32_tensor("rgb");
            const auto face = reference->require_f32("restore_face");
            const auto matrices = reference->require_f32("matrices");
            models::latentsync::LatentSyncAffine matrix;
            std::copy_n(matrices.begin(), 6, matrix.begin());
            const auto resize_start = std::chrono::steady_clock::now();
            const auto resized = models::latentsync::latentsync_resize_bicubic(face, 3, 512, 512, 560, 420);
            const auto resize_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - resize_start).count();
            const float det = matrix[0] * matrix[4] - matrix[1] * matrix[3];
            models::latentsync::LatentSyncAffine inverse{
                matrix[4] / det, -matrix[1] / det, 0,
                -matrix[3] / det, matrix[0] / det, 0};
            inverse[2] = -inverse[0] * matrix[2] - inverse[1] * matrix[5];
            inverse[5] = -inverse[3] * matrix[2] - inverse[4] * matrix[5];
            const auto face_warp_start = std::chrono::steady_clock::now();
            const auto warped_face = models::latentsync::latentsync_warp_affine(
                resized, 3, 560, 420, inverse, image.shape.dims[2], image.shape.dims[3], 127, 8);
            const auto face_warp_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - face_warp_start).count();
            const auto mask_warp_start = std::chrono::steady_clock::now();
            const auto warped_mask = models::latentsync::latentsync_warp_affine(
                std::vector<float>(560 * 420, 1), 1, 560, 420, inverse,
                image.shape.dims[2], image.shape.dims[3], 0, 8);
            const auto mask_warp_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - mask_warp_start).count();
            std::cout << "restoration resize_ms=" << resize_ms
                      << " face_warp_ms=" << face_warp_ms
                      << " mask_warp_ms=" << mask_warp_ms
                      << " warped_values=" << (warped_face.size() + warped_mask.size()) << '\n';
            std::vector<float> restored;
            for (const int threads : {1, 8}) {
                const auto restore_start = std::chrono::steady_clock::now();
                const auto output = models::latentsync::latentsync_restore_face(image.values, image.shape.dims[2], image.shape.dims[3], face, matrix, threads);
                const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - restore_start).count();
                std::cout << "restoration threads=" << threads << " wall_ms=" << wall_ms << '\n';
                if (threads == 1) restored = output;
                else if (restored != output) throw std::runtime_error("Parallel restoration changed pixels");
            }
            std::vector<std::vector<float>> batch(8);
            const auto batch_start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for num_threads(8) schedule(static)
#endif
            for (int i = 0; i < static_cast<int>(batch.size()); ++i) {
                batch[i] = models::latentsync::latentsync_restore_face(
                    image.values, image.shape.dims[2], image.shape.dims[3], face, matrix, 1);
            }
            const auto batch_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - batch_start).count();
            for (const auto & output : batch) {
                if (output != restored) throw std::runtime_error("Batched restoration changed pixels");
            }
            std::cout << "restoration frame_batch=8 wall_ms=" << batch_ms
                      << " per_frame_ms=" << batch_ms / batch.size() << '\n';
            bool passed = true;
            for (int stage = 0; stage < 2; ++stage) {
                const auto expected = reference->require_f32(stage ? "restored" : "restore_resized");
                const auto & actual = stage ? restored : resized;
                double squared = 0, maximum = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const double error = double(actual[i]) - expected[i];
                    squared += error * error;
                    maximum = std::max(maximum, std::abs(error));
                }
                const double rmse = std::sqrt(squared / actual.size());
                std::cout << "stage=" << stage << " rmse=" << rmse << " max_abs=" << maximum << '\n';
                passed = passed && rmse < (stage ? 0.2 : 0.0001) && maximum < (stage ? 2 : 0.001);
            }
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return passed ? 0 : 1;
        }
        if (component == "alignment") {
            const auto start = std::chrono::steady_clock::now();
            const auto reference = assets::open_tensor_source(root / "reference.safetensors");
            const auto landmarks = reference->require_f32("landmarks");
            const auto points = reference->require_f32("points");
            const auto matrices = reference->require_f32("matrices");
            const auto image = reference->require_f32_tensor("image");
            const auto expected_warp = reference->require_f32("warped");
            models::latentsync::LatentSyncAffine first_matrix;
            std::copy_n(matrices.begin(), 6, first_matrix.begin());
            const auto warp_start = std::chrono::steady_clock::now();
            const auto warped = models::latentsync::latentsync_warp_affine(image.values, 3,
                image.shape.dims[2], image.shape.dims[3], first_matrix, 560, 420, 127, 8);
            const auto warp_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - warp_start).count();
            double warp_squared = 0, warp_maximum = 0;
            for (size_t i = 0; i < warped.size(); ++i) {
                const double error = double(warped[i]) - expected_warp[i];
                warp_squared += error * error;
                warp_maximum = std::max(warp_maximum, std::abs(error));
            }
            const double warp_rmse = std::sqrt(warp_squared / warped.size());
            std::cout << "warp_rmse=" << warp_rmse << " warp_max_abs=" << warp_maximum << '\n';
            const auto resize_start = std::chrono::steady_clock::now();
            const auto resized = models::latentsync::latentsync_resize_lanczos4(
                expected_warp, 3, 560, 420, 512, 512, 8);
            const auto resize_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - resize_start).count();
            const auto expected_resize = reference->require_f32("resized");
            double resize_squared = 0, resize_maximum = 0;
            for (size_t i = 0; i < resized.size(); ++i) {
                const double error = double(resized[i]) - expected_resize[i];
                resize_squared += error * error;
                resize_maximum = std::max(resize_maximum, std::abs(error));
            }
            const double resize_rmse = std::sqrt(resize_squared / resized.size());
            std::cout << "resize_rmse=" << resize_rmse << " resize_max_abs=" << resize_maximum << '\n';
            models::latentsync::LatentSyncFaceAlignment alignment;
            double maximum = 0;
            for (int repeat = 0; repeat < 2; ++repeat) {
                alignment.reset();
                for (size_t frame = 0; frame < matrices.size() / 6; ++frame) {
                    const std::vector<float> input(landmarks.begin() + frame * 212, landmarks.begin() + (frame + 1) * 212);
                    const auto three = models::latentsync::latentsync_alignment_points(input);
                    for (size_t i = 0; i < 6; ++i) {
                        if (three[i / 2][i % 2] != points[frame * 6 + i]) throw std::runtime_error("Alignment landmark selection mismatch");
                    }
                    const auto matrix = alignment.transform(three);
                    for (size_t i = 0; i < 6; ++i) maximum = std::max(maximum, std::abs(double(matrix[i]) - matrices[frame * 6 + i]));
                }
            }
            std::cout << "alignment_max_abs=" << maximum
                      << " warp_ms=" << warp_ms << " resize_ms=" << resize_ms << '\n';
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return maximum < 0.001 && warp_rmse < 0.01 && warp_maximum < 0.1 && resize_rmse < 0.01 && resize_maximum < 0.1 ? 0 : 1;
        }
        if (component == "ddim") {
            const auto start = std::chrono::steady_clock::now();
            const auto reference = assets::open_tensor_source(root / "reference.safetensors");
            auto sample = reference->require_f32("initial");
            const auto noises = reference->require_f32("noises");
            const auto expected = reference->require_f32("outputs");
            const auto time_features = reference->require_f32("features");
            const models::latentsync::LatentSyncDdimSchedule schedule({}, 20);
            double max_step_error = 0, max_feature_error = 0;
            for (size_t i = 0; i < schedule.timesteps().size(); ++i) {
                const std::vector<float> noise(noises.begin() + i * sample.size(), noises.begin() + (i + 1) * sample.size());
                schedule.step(sample, noise, i);
                for (size_t j = 0; j < sample.size(); ++j) {
                    max_step_error = std::max(max_step_error, std::abs(double(sample[j]) - expected[i * sample.size() + j]));
                }
                const auto features = models::latentsync::latentsync_timestep_features(schedule.timesteps()[i], 320, 1);
                for (size_t j = 0; j < features.size(); ++j) {
                    max_feature_error = std::max(max_feature_error, std::abs(double(features[j]) - time_features[i * features.size() + j]));
                }
            }
            std::cout << std::setprecision(10) << "DDIM max_abs=" << max_step_error
                      << " timestep_features max_abs=" << max_feature_error << '\n';
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            return max_step_error < 1e-4 && max_feature_error < 1e-4 ? 0 : 1;
        }
        if (component != "whisper" && component != "whisper-runtime" && component != "whisper-audio" && component != "vae-encoder" && component != "vae-decoder" &&
            component != "vae-encoder-runtime" && component != "vae-decoder-runtime" &&
            component != "vae-encoder-runtime-cudnn-f16" && component != "vae-decoder-runtime-cudnn-f16" &&
            component != "vae-production-encoder" && component != "vae-production-decoder" &&
            component != "vae-production-encoder-cudnn" && component != "vae-production-decoder-cudnn" &&
            component != "vae-production-encoder-cudnn-f16" && component != "vae-production-decoder-cudnn-f16" &&
            component != "spatial" && component != "temporal" && component != "unet" && component != "unet-runtime" && component != "denoise" && component != "landmarks" && component != "scrfd" && component != "face-runtime" && component != "face-frame" && component != "pipeline") {
            throw std::runtime_error("unknown component");
        }
        if (name == "cuda") backend.type = core::BackendType::Cuda;
        else if (name == "vulkan") backend.type = core::BackendType::Vulkan;
        else if (name != "cpu") throw std::runtime_error("unknown backend");
        backend.threads = 8;
        core::ExecutionContext execution(backend);
        if (component == "vae-production-encoder" || component == "vae-production-decoder" ||
            component == "vae-production-encoder-cudnn" || component == "vae-production-decoder-cudnn" ||
            component == "vae-production-encoder-cudnn-f16" || component == "vae-production-decoder-cudnn-f16") {
            const auto resources = model_spec::load_resource_bundle_for_family(root, "latentsync");
            const auto source = resources.open_tensor_source("weights");
            const bool cudnn = component.find("-cudnn") != std::string::npos;
            const bool f16 = component.find("-cudnn-f16") != std::string::npos;
            models::latentsync::LatentSyncSdVaeRuntime runtime(execution, *source, assets::TensorStorageType::F16, 512, cudnn, f16);
            const bool encoder = component.find("encoder") != std::string::npos;
            const size_t elements = encoder ? 3ull * 512 * 512 : 4ull * 64 * 64;
            const std::vector<float> input(elements, encoder ? 0.5F : 0.1F);
            auto output = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            const auto start = std::chrono::steady_clock::now();
            const int iterations = encoder ? 36 : 18;
            for (int i = 0; i < iterations; ++i) {
                output = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            }
            const double wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            debug::timing_log_scalar("session.wall_ms", wall_ms);
            std::cout << "vae=" << (encoder ? "encoder" : "decoder") << " batch=1 iterations=" << iterations
                      << " wall_ms=" << wall_ms << " output_size=" << output.size() << '\n';
            if (f16) {
                std::vector<float> varied_input(elements);
                for (size_t i = 0; i < elements; ++i) {
                    varied_input[i] = encoder ? float(i % 251) / 125.0F - 1.0F :
                        0.1F * std::sin(float(i % 4096) * 0.013F);
                }
                const auto candidate = encoder ? runtime.encode_moments(varied_input) : runtime.decode(varied_input);
                models::latentsync::LatentSyncSdVaeRuntime baseline(execution, *source, assets::TensorStorageType::F16);
                const auto reference = encoder ? baseline.encode_moments(varied_input) : baseline.decode(varied_input);
                double squared_error = 0.0, squared_reference = 0.0, dot = 0.0, squared_candidate = 0.0;
                float max_abs = 0.0F;
                for (size_t i = 0; i < candidate.size(); ++i) {
                    const double difference = double(candidate[i]) - reference[i];
                    squared_error += difference * difference;
                    squared_reference += double(reference[i]) * reference[i];
                    squared_candidate += double(candidate[i]) * candidate[i];
                    dot += double(reference[i]) * candidate[i];
                    max_abs = std::max(max_abs, float(std::abs(difference)));
                }
                std::cout << "relative_rmse=" << std::sqrt(squared_error / squared_reference)
                          << " cosine=" << dot / std::sqrt(squared_reference * squared_candidate)
                          << " max_abs=" << max_abs << '\n';
            }
            return 0;
        }
        if (component == "pipeline") {
            const auto config = io::json::parse_file(root / "config.json");
            models::latentsync::LatentSyncPipelineSources sources;
            sources.unet = assets::open_tensor_source(config.require("unet").as_string());
            sources.vae = assets::open_tensor_source(config.require("vae").as_string());
            sources.whisper = assets::open_tensor_source(config.require("whisper").as_string());
            sources.detector = assets::open_tensor_source(config.require("detector").as_string());
            sources.landmarks = assets::open_tensor_source(config.require("landmarks").as_string());
            sources.detector_config = config.require("detector_config").as_string();
            sources.landmark_config = config.require("landmark_config").as_string();
            const auto input = assets::open_tensor_source(root / "input.safetensors");
            const auto images = input->require_f32_tensor("frames");
            const int height = images.shape.dims[1], width = images.shape.dims[2];
            const size_t frame_size = size_t(height) * width * 3;
            std::vector<std::vector<uint8_t>> frames(images.shape.dims[0], std::vector<uint8_t>(frame_size));
            for (size_t i = 0; i < frames.size(); ++i) for (size_t j = 0; j < frame_size; ++j) frames[i][j] = uint8_t(images.values[i * frame_size + j]);
            models::latentsync::LatentSyncPipeline pipeline(execution, sources, assets::TensorStorageType::F32);
            std::ofstream output(root / "output.rgb", std::ios::binary);
            if (!output) throw std::runtime_error("Cannot open pipeline output");
            size_t emitted = 0;
            pipeline.run(frames, height, width, input->require_f32("audio"), input->require_f32("mask"), {},
                [&](size_t index, const std::vector<uint8_t> & pixels) {
                    if (index != emitted++) throw std::runtime_error("Unexpected pipeline frame order");
                    output.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
                    if (!output) throw std::runtime_error("Cannot write pipeline output");
                });
            std::cout << "output_frames=" << emitted << " width=" << width << " height=" << height << '\n';
            return emitted == 18 ? 0 : 1;
        }
        if (component == "face-runtime" || component == "face-frame") {
            const auto detector = assets::open_tensor_source(root / "scrfd/weights.safetensors");
            const auto landmark = assets::open_tensor_source(root / "landmarks/weights.safetensors");
            const auto detector_reference = assets::open_tensor_source(root / "scrfd/reference.safetensors");
            const auto landmark_reference = assets::open_tensor_source(root / "landmarks/reference.safetensors");
            models::latentsync::LatentSyncFaceRuntime runtime(execution, *detector, root / "scrfd/config.json",
                *landmark, root / "landmarks/config.json", assets::TensorStorageType::F32);
            if (component == "face-frame") {
                const auto reference = assets::open_tensor_source(root / "alignment/reference.safetensors");
                const auto image = reference->require_f32_tensor("rgb");
                const auto expected = reference->require_f32("aligned");
                const auto expected_points = reference->require_f32("landmarks");
                const auto start = std::chrono::steady_clock::now();
                const auto face = runtime.align_frame(image.values, image.shape.dims[2], image.shape.dims[3]);
                double squared = 0, maximum = 0, landmark_max = 0;
                for (size_t i = 0; i < expected.size(); ++i) {
                    const double error = double(face.pixels[i]) - expected[i];
                    squared += error * error;
                    maximum = std::max(maximum, std::abs(error));
                }
                for (size_t i = 0; i < face.landmarks.size(); ++i) landmark_max = std::max(landmark_max, std::abs(double(face.landmarks[i]) - expected_points[i]));
                const double rmse = std::sqrt(squared / expected.size());
                std::cout << "aligned_rmse=" << rmse << " max_abs=" << maximum << " landmark_max_pixels=" << landmark_max << '\n';
                debug::timing_log_scalar("session.wall_ms",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                return rmse < 0.05 && landmark_max < 3 ? 0 : 1;
            }
            const auto image = detector_reference->require_f32("input");
            const auto crop = landmark_reference->require_f32("input");
            const auto expected_faces = detector_reference->require_f32("detections");
            const auto expected_landmarks = landmark_reference->require_f32("output");
            const auto scale = detector_reference->require_f32("resize_scale")[0];
            std::vector<float> previous;
            for (int iteration = 0; iteration < 3; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                const auto faces = runtime.detect(image, scale);
                const auto points = runtime.landmarks(crop);
                if (faces.size() * 15 != expected_faces.size() || points.size() != expected_landmarks.size()) {
                    throw std::runtime_error("Face runtime output count mismatch");
                }
                std::vector<float> actual;
                double maximum = 0;
                for (size_t i = 0; i < faces.size(); ++i) {
                    const auto & face = faces[i];
                    actual.insert(actual.end(), face.box.begin(), face.box.end());
                    actual.push_back(face.score);
                    actual.insert(actual.end(), face.landmarks.begin(), face.landmarks.end());
                }
                for (size_t i = 0; i < actual.size(); ++i) maximum = std::max(maximum, std::abs(double(actual[i]) - expected_faces[i]));
                double landmark_error = 0;
                for (size_t i = 0; i < points.size(); ++i) landmark_error = std::max(landmark_error, std::abs(double(points[i]) - expected_landmarks[i]));
                actual.insert(actual.end(), points.begin(), points.end());
                if (maximum >= 0.5 || landmark_error >= 0.001 || (iteration && actual != previous)) {
                    throw std::runtime_error("Face runtime parity or repeated-call mismatch");
                }
                previous = std::move(actual);
                std::cout << "iteration=" << iteration << " face_max_abs=" << maximum << " landmark_max_abs=" << landmark_error << '\n';
                debug::timing_log_scalar("session.wall_ms",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            }
            return 0;
        }
        const auto config = io::json::parse_file(root / "config.json");
        modules::WhisperEmbeddingConfig dimensions;
        if (component == "whisper" || component == "whisper-runtime" || component == "whisper-audio") {
            dimensions.n_mels = config.require("n_mels").as_i64();
            dimensions.n_audio_ctx = config.require("n_audio_ctx").as_i64();
            dimensions.n_audio_state = config.require("n_audio_state").as_i64();
            dimensions.n_audio_head = config.require("n_audio_head").as_i64();
            dimensions.n_audio_layer = config.require("n_audio_layer").as_i64();
        }
        const auto source = assets::open_tensor_source(root / "weights.safetensors");
        const auto reference = assets::open_tensor_source(root / "reference.safetensors");
        const bool whisper = component == "whisper" || component == "whisper-runtime" || component == "whisper-audio";
        const auto input_data = reference->require_f32_tensor(whisper ? "mel" : "input");
        const auto expected = reference->require_f32(component == "whisper-audio" ? "audio_features" : (whisper ? "embeddings" : "output"));
        std::vector<float> actual;
        if (component == "whisper-audio") {
            models::latentsync::LatentSyncWhisperRuntime runtime(execution, *source, dimensions, assets::TensorStorageType::F32);
            const auto start = std::chrono::steady_clock::now();
            actual = runtime.encode_audio(reference->require_f32("audio"), 8);
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        } else if (component == "whisper-runtime") {
            models::latentsync::LatentSyncWhisperRuntime runtime(execution, *source, dimensions, assets::TensorStorageType::F32);
            const auto start = std::chrono::steady_clock::now();
            actual = runtime.encode_mel(input_data.values);
            const auto repeated = runtime.encode_mel(input_data.values);
            auto changed_input = input_data.values;
            for (auto & value : changed_input) value *= 0.5F;
            const auto changed = runtime.encode_mel(changed_input);
            const auto restored = runtime.encode_mel(input_data.values);
            if (actual != repeated || actual != restored || actual == changed) throw std::runtime_error("Whisper graph input/repeat mismatch");
            const auto selected = models::latentsync::latentsync_audio_windows(expected, 0, 751, 25);
            if (selected != reference->require_f32("windows")) throw std::runtime_error("Whisper frame-window indexing mismatch");
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        } else if (component == "vae-encoder-runtime" || component == "vae-decoder-runtime" ||
                   component == "vae-encoder-runtime-cudnn-f16" || component == "vae-decoder-runtime-cudnn-f16") {
            const bool f16 = component.find("-cudnn-f16") != std::string::npos;
            models::latentsync::LatentSyncSdVaeRuntime runtime(execution, *source,
                f16 ? assets::TensorStorageType::F16 : assets::TensorStorageType::F32, 512, f16, f16);
            auto input = input_data.values;
            const bool encoder = component.find("vae-encoder") == 0;
            if (!encoder) for (auto & value : input) value *= 0.18215F;
            const auto start = std::chrono::steady_clock::now();
            actual = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            const auto repeated = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            for (auto & value : input) value = -value;
            const auto changed = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            for (auto & value : input) value = -value;
            const auto restored = encoder ? runtime.encode_moments(input) : runtime.decode(input);
            if (actual != repeated || actual != restored || actual == changed) throw std::runtime_error("VAE cached graph input/repeat mismatch");
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        } else if (component == "denoise") {
            models::latentsync::LatentSyncUnetRuntime runtime(execution, *source, {}, assets::TensorStorageType::F32);
            models::latentsync::LatentSyncLatentClip clip;
            clip.frames = config.require("frames").as_i64();
            clip.height = input_data.shape.dims[2];
            clip.width = input_data.shape.dims[3];
            clip.noise = input_data.values;
            clip.mask = reference->require_f32("mask");
            clip.masked_image = reference->require_f32("masked_image");
            clip.reference_image = reference->require_f32("reference_image");
            clip.audio = reference->require_f32("audio");
            const auto start = std::chrono::steady_clock::now();
            const auto cache = config.find("cache_interval");
            const int64_t interval = cache ? cache->as_i64() : 1;
            const auto steps = config.require("steps").as_i64();
            const auto guidance = static_cast<float>(config.require("guidance").as_number());
            actual = runtime.denoise(clip, steps, guidance, {}, interval);
            if (interval > 1) {
                const models::latentsync::LatentSyncDdimSchedule schedule({}, steps);
                const int64_t lanes = guidance > 1 ? 2 : 1;
                std::vector<float> model_audio(clip.audio.size() * lanes, 0.0F);
                std::copy(clip.audio.begin(), clip.audio.end(), model_audio.end() - clip.audio.size());
                for (int64_t i = 0; i < steps; ++i) {
                    const auto expected_step = reference->require_f32("step_output." + std::to_string(i));
                    const auto predicted = runtime.predict(reference->require_f32("step_input." + std::to_string(i)),
                        model_audio, models::latentsync::latentsync_timestep_features(schedule.timesteps()[i], 320, lanes),
                        lanes, clip.frames, clip.height, clip.width, clip.audio_tokens, i % interval);
                    double squared = 0, norm = 0;
                    for (size_t j = 0; j < predicted.size(); ++j) {
                        squared += std::pow(double(predicted[j]) - expected_step[j], 2);
                        norm += double(expected_step[j]) * expected_step[j];
                    }
                    std::cout << "step=" << i << " relative_error=" << std::sqrt(squared / norm) << '\n';
                }
                for (auto & value : clip.noise) value = -value;
                const auto changed = runtime.denoise(clip, steps, guidance, {}, interval);
                for (auto & value : clip.noise) value = -value;
                const auto restored = runtime.denoise(clip, steps, guidance, {}, interval);
                if (changed == actual || restored != actual) throw std::runtime_error("DeepCache leaked between clips");
            }
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        } else if (component == "unet-runtime") {
            const auto weight_type = config.find("weight_type");
            const auto type = weight_type ? assets::parse_tensor_storage_type(weight_type->as_string()) : assets::TensorStorageType::F32;
            models::latentsync::LatentSyncUnetConfig unet_config;
            if (const auto fast = config.find("cuda_f16_activations")) unet_config.cuda_f16_activations = fast->as_bool();
            models::latentsync::LatentSyncUnetRuntime runtime(execution, *source, unet_config, type);
            const auto audio_data = reference->require_f32_tensor("audio");
            const auto time_data = reference->require_f32_tensor("timestep_features");
            const auto frames = config.require("frames").as_i64();
            const auto batch = time_data.shape.dims[0];
            const auto height = input_data.shape.dims[2], width = input_data.shape.dims[3];
            const auto tokens = audio_data.shape.dims[1];
            const auto start = std::chrono::steady_clock::now();
            actual = runtime.predict(input_data.values, audio_data.values, time_data.values, batch, frames, height, width, tokens);
            if (const auto runs = config.find("benchmark_runs")) {
                const int64_t count = runs->as_i64();
                if (count <= 0) throw std::runtime_error("benchmark_runs must be positive");
                const auto bench_start = std::chrono::steady_clock::now();
                for (int64_t i = 0; i < count; ++i) {
                    actual = runtime.predict(input_data.values, audio_data.values, time_data.values,
                        batch, frames, height, width, tokens);
                }
                const double wall_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - bench_start).count();
                debug::timing_log_scalar("session.wall_ms", wall_ms);
                std::cout << "unet_benchmark_runs=" << count << " wall_ms=" << wall_ms << '\n';
                double dot = 0.0, xx = 0.0, yy = 0.0, squared_error = 0.0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    const double x = actual[i], y = expected[i];
                    dot += x * y;
                    xx += x * x;
                    yy += y * y;
                    squared_error += (x - y) * (x - y);
                }
                std::cout << "cosine=" << dot / std::sqrt(xx * yy)
                          << " relative_rmse=" << std::sqrt(squared_error / yy) << '\n';
                return 0;
            }
            const auto repeated = runtime.predict(input_data.values, audio_data.values, time_data.values, batch, frames, height, width, tokens);
            if (actual != repeated) throw std::runtime_error("Repeated U-Net graph changed output");
            auto changed_input = input_data.values;
            for (auto & value : changed_input) value *= -1;
            const auto changed = runtime.predict(changed_input, audio_data.values, time_data.values, batch, frames, height, width, tokens);
            if (actual == changed) throw std::runtime_error("U-Net graph ignored updated input");
            const auto restored = runtime.predict(input_data.values, audio_data.values, time_data.values, batch, frames, height, width, tokens);
            if (actual != restored) throw std::runtime_error("U-Net graph retained stale input");
            if (frames > 1) {
                std::vector<float> short_input(input_data.values.begin(), input_data.values.begin() + 13 * height * width);
                std::vector<float> short_audio(audio_data.values.begin(), audio_data.values.begin() + tokens * 384);
                std::vector<float> short_time(time_data.values.begin(), time_data.values.begin() + 320);
                runtime.predict(short_input, short_audio, short_time, 1, 1, height, width, tokens);
                const auto rebuilt = runtime.predict(input_data.values, audio_data.values, time_data.values, batch, frames, height, width, tokens);
                if (actual != rebuilt) throw std::runtime_error("U-Net graph shape replacement changed output");
            }
            debug::timing_log_scalar("session.wall_ms",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            std::cout << "Repeated, changed-input, and shape-replacement checks passed\n";
        } else {
        core::BackendWeightStore store(execution.backend(), backend.type,
            "latentsync." + component + ".weights", 4 * 1024 * 1024);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({static_cast<size_t>(component == "unet" ? 16 : 4) * 1024 * 1024, nullptr, true}), ggml_free);
        if (!context) throw std::runtime_error("graph context allocation failed");
        core::ModuleBuildContext build{context.get(), "latentsync.component", backend.type};
        const auto input = core::make_tensor(build, GGML_TYPE_F32, input_data.shape);
        std::optional<core::TensorValue> audio;
        std::optional<core::TensorValue> timestep;
        core::TensorValue output;
        if (component == "whisper") {
            const auto weights = models::latentsync::load_whisper_encoder_weights(
                store, *source, dimensions, assets::TensorStorageType::F32);
            output = models::latentsync::LatentSyncWhisperEncoder(dimensions).build(build, input, weights);
        } else if (component == "unet") {
            const models::latentsync::LatentSyncUnetConfig unet_config;
            const auto audio_data = reference->require_f32_tensor("audio");
            const auto time_data = reference->require_f32_tensor("timestep_features");
            audio = core::make_tensor(build, GGML_TYPE_F32, audio_data.shape);
            timestep = core::make_tensor(build, GGML_TYPE_F32, time_data.shape);
            const auto weights = models::latentsync::load_unet_weights(
                store, *source, unet_config, assets::TensorStorageType::F32);
            output = models::latentsync::LatentSyncUnet(unet_config).build(build, input, *audio,
                *timestep, config.require("frames").as_i64(), weights);
        } else if (component == "spatial" || component == "temporal") {
            const auto prefix = config.require("prefix").as_string();
            const auto channels = input_data.shape.dims[1];
            const models::latentsync::LatentSyncUnetAttention attention;
            if (component == "spatial") {
                const auto audio_data = reference->require_f32_tensor("audio");
                audio = core::make_tensor(build, GGML_TYPE_F32, audio_data.shape);
                const auto weights = models::latentsync::load_spatial_transformer_weights(
                    store, *source, prefix, channels, audio_data.shape.dims[2], assets::TensorStorageType::F32);
                output = attention.build_spatial(build, input, *audio, weights);
            } else {
                const auto weights = models::latentsync::load_temporal_transformer_weights(
                    store, *source, prefix, channels, 24, assets::TensorStorageType::F32);
                output = attention.build_temporal(build, input, config.require("frames").as_i64(), weights);
            }
        } else if (component == "scrfd") {
            const auto weights = models::latentsync::load_scrfd_weights(
                store, *source, root / "config.json", assets::TensorStorageType::F32);
            const auto heads = models::latentsync::LatentSyncScrfdDetector().build(build, input, weights);
            for (const auto & head : heads) {
                const auto flat = core::reshape_tensor(build, core::ensure_backend_addressable_layout(build, head),
                    core::TensorShape::from_dims({head.shape.num_elements()}));
                output = output.tensor ? modules::ConcatModule({0}).build(build, output, flat) : flat;
            }
        } else if (component == "landmarks") {
            const auto weights = models::latentsync::load_landmark_weights(
                store, *source, root / "config.json", assets::TensorStorageType::F32);
            output = models::latentsync::LatentSyncDepthwiseCnnLandmarks().build(build, input, weights);
        } else {
            models::latentsync::LatentSyncSdVaeConfig vae_config;
            const auto weights = models::latentsync::load_sd_vae_weights(
                store, *source, vae_config, component == "vae-encoder", assets::TensorStorageType::F32);
            const models::latentsync::LatentSyncSdVae vae(vae_config);
            output = component == "vae-encoder" ? vae.build_encoder(build, input, weights)
                                                 : vae.build_decoder(build, input, weights);
        }
        store.upload();
        ggml_set_input(input.tensor);
        if (audio) ggml_set_input(audio->tensor);
        if (timestep) ggml_set_input(timestep->tensor);
        ggml_set_output(output.tensor);
        auto * graph = ggml_new_graph_custom(context.get(), component == "unet" ? 16384 : 4096, false);
        ggml_build_forward_expand(graph, output.tensor);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())), ggml_gallocr_free);
        if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) {
            throw std::runtime_error("graph allocation failed");
        }
        ggml_backend_tensor_set(input.tensor, input_data.values.data(), 0, input_data.values.size() * sizeof(float));
        if (audio) {
            const auto values = reference->require_f32("audio");
            ggml_backend_tensor_set(audio->tensor, values.data(), 0, values.size() * sizeof(float));
        }
        if (timestep) {
            const auto values = reference->require_f32("timestep_features");
            ggml_backend_tensor_set(timestep->tensor, values.data(), 0, values.size() * sizeof(float));
        }
        const auto start = std::chrono::steady_clock::now();
        const auto status = core::compute_backend_graph(execution.backend(), graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("component graph execution failed");
        debug::timing_log_scalar("session.wall_ms",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        if (output.shape.num_elements() != static_cast<int64_t>(expected.size())) {
            throw std::runtime_error("conditioning output shape mismatch");
        }
        actual.resize(expected.size());
        ggml_backend_tensor_get(output.tensor, actual.data(), 0, actual.size() * sizeof(float));
        core::release_backend_graph_resources(execution.backend(), graph, true);
        }
        if (actual.size() != expected.size()) throw std::runtime_error("Component output size mismatch");
        bool passed = true;
        if (component == "scrfd") {
            std::array<std::vector<float>, 9> heads;
            size_t offset = 0;
            for (size_t i = 0; i < heads.size(); ++i) {
                const auto count = reference->require_f32("head_" + std::to_string(i)).size();
                heads[i].assign(actual.begin() + offset, actual.begin() + offset + count);
                offset += count;
            }
            const auto faces = models::latentsync::decode_scrfd_faces(heads, reference->require_f32("resize_scale")[0]);
            const auto detections = reference->require_f32("detections");
            if (faces.size() * 15 != detections.size()) throw std::runtime_error("SCRFD face count differs from reference");
            double maximum = 0;
            for (size_t i = 0; i < faces.size(); ++i) {
                for (size_t j = 0; j < 15; ++j) {
                    const float value = j < 4 ? faces[i].box[j] : (j == 4 ? faces[i].score : faces[i].landmarks[j - 5]);
                    maximum = std::max(maximum, std::abs(double(value) - detections[i * 15 + j]));
                }
            }
            std::cout << "faces=" << faces.size() << " detection_max_abs=" << maximum << '\n';
            passed = maximum < 0.5;
        }
        const size_t groups = component == "scrfd" ? 9 : ((component == "vae-encoder" ||
            component == "vae-encoder-runtime" || component == "vae-encoder-runtime-cudnn-f16") ? 2 : 1);
        size_t offset = 0;
        for (size_t group = 0; group < groups; ++group) {
            const size_t count = component == "scrfd"
                ? reference->require_f32("head_" + std::to_string(group)).size()
                : actual.size() / groups;
            double dot = 0, xx = 0, yy = 0, squared_error = 0, maximum = 0;
            for (size_t i = offset; i < offset + count; ++i) {
                const double x = actual[i], y = expected[i];
                dot += x * y;
                xx += x * x;
                yy += y * y;
                squared_error += (x - y) * (x - y);
                maximum = std::max(maximum, std::abs(x - y));
            }
            const double cosine = dot / std::sqrt(xx * yy);
            std::cout << std::setprecision(10) << "group=" << group << " cosine=" << cosine
                      << " rmse=" << std::sqrt(squared_error / count)
                      << " max_abs=" << maximum << '\n';
            passed = passed && std::isfinite(cosine) && cosine >= 0.999;
            offset += count;
        }
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
