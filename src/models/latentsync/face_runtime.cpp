#include "engine/models/latentsync/face_runtime.h"
#include "engine/models/latentsync/face_landmarks.h"
#include "engine/framework/debug/trace.h"

#include <ggml-alloc.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::latentsync {

struct LatentSyncFaceRuntime::Impl {
    struct Graph {
        ggml_backend_t backend;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
        ggml_cgraph * graph = nullptr;
        core::TensorValue input;
        std::vector<core::TensorValue> outputs;

        explicit Graph(ggml_backend_t backend_in) : backend(backend_in) {}
        ~Graph() { core::release_backend_graph_resources(backend, graph, true); }
    };

    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    LatentSyncScrfdWeights detector_weights;
    LatentSyncLandmarkWeights landmark_weights;
    Graph detector, landmark;
    LatentSyncFaceAlignment alignment;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & detector_source,
        const std::filesystem::path & detector_config, const assets::TensorSource & landmark_source,
        const std::filesystem::path & landmark_config, assets::TensorStorageType type)
        : execution(execution_in),
          store(execution.backend(), execution.backend_type(), "latentsync.face.weights", 4 * 1024 * 1024),
          detector_weights(load_scrfd_weights(store, detector_source, detector_config, type)),
          landmark_weights(load_landmark_weights(store, landmark_source, landmark_config, type)),
          detector(execution.backend()), landmark(execution.backend()) {
        store.upload();
        for (auto * item : {&detector, &landmark}) {
            constexpr size_t nodes = 4096;
            item->context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
            if (!item->context) throw std::runtime_error("LatentSync face graph context allocation failed");
            const bool is_detector = item == &detector;
            core::ModuleBuildContext ctx{item->context.get(), "latentsync.face", execution.backend_type()};
            const int size = is_detector ? 512 : 192;
            item->input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 3, size, size}));
            ggml_set_input(item->input.tensor);
            if (is_detector) item->outputs = LatentSyncScrfdDetector().build(ctx, item->input, detector_weights);
            else item->outputs = {LatentSyncDepthwiseCnnLandmarks().build(ctx, item->input, landmark_weights)};
            item->graph = ggml_new_graph_custom(item->context.get(), nodes, false);
            for (const auto & output : item->outputs) {
                ggml_set_output(output.tensor);
                ggml_build_forward_expand(item->graph, output.tensor);
            }
            item->allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            if (!item->allocator || !ggml_gallocr_alloc_graph(item->allocator.get(), item->graph)) {
                throw std::runtime_error("LatentSync face graph allocation failed");
            }
            debug::timing_log_scalar(is_detector ? "latentsync.detector.graph_buffer_mb" : "latentsync.landmarks.graph_buffer_mb",
                ggml_gallocr_get_buffer_size(item->allocator.get(), 0) / 1048576.0);
        }
    }
};

LatentSyncFaceRuntime::LatentSyncFaceRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & detector_source, const std::filesystem::path & detector_config,
    const assets::TensorSource & landmark_source, const std::filesystem::path & landmark_config,
    assets::TensorStorageType weight_type)
    : impl_(std::make_unique<Impl>(execution, detector_source, detector_config,
        landmark_source, landmark_config, weight_type)) {}

LatentSyncFaceRuntime::~LatentSyncFaceRuntime() = default;

std::vector<LatentSyncFaceDetection> LatentSyncFaceRuntime::detect(const std::vector<float> & image, float resize_scale) {
    const auto start = std::chrono::steady_clock::now();
    auto & graph = impl_->detector;
    core::write_tensor_f32(graph.input, image);
    if (core::compute_backend_graph(graph.backend, graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LatentSync face detector execution failed");
    }
    std::array<std::vector<float>, 9> heads;
    for (size_t i = 0; i < heads.size(); ++i) heads[i] = core::read_tensor_f32(graph.outputs[i].tensor);
    auto faces = decode_scrfd_faces(heads, resize_scale);
    debug::timing_log_scalar("latentsync.detector.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return faces;
}

std::vector<float> LatentSyncFaceRuntime::landmarks(const std::vector<float> & image) {
    const auto start = std::chrono::steady_clock::now();
    auto & graph = impl_->landmark;
    core::write_tensor_f32(graph.input, image);
    if (core::compute_backend_graph(graph.backend, graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LatentSync landmark execution failed");
    }
    auto output = core::read_tensor_f32(graph.outputs.front().tensor);
    debug::timing_log_scalar("latentsync.landmarks.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

LatentSyncAlignedFace LatentSyncFaceRuntime::align_frame(const std::vector<float> & rgb, int height, int width) {
    if (height <= 0 || width <= 0 || rgb.size() != static_cast<size_t>(height) * width * 3) {
        throw std::runtime_error("Invalid LatentSync video frame dimensions");
    }
    const double ratio = 512.0 / std::max(height, width);
    const int resized_height = int(height * ratio), resized_width = int(width * ratio);
    std::vector<float> detector_input(3 * 512 * 512, -127.5F / 128);
    for (int y = 0; y < resized_height; ++y) {
        const double sy = (y + 0.5) * height / resized_height - 0.5;
        const int y0 = int(std::floor(sy));
        const float wy = float(sy - y0);
        for (int x = 0; x < resized_width; ++x) {
            const double sx = (x + 0.5) * width / resized_width - 0.5;
            const int x0 = int(std::floor(sx));
            const float wx = float(sx - x0);
            for (int c = 0; c < 3; ++c) {
                float value = 0;
                for (int yy = 0; yy < 2; ++yy) for (int xx = 0; xx < 2; ++xx) {
                    value += rgb[(static_cast<size_t>(c) * height + std::clamp(y0 + yy, 0, height - 1)) * width + std::clamp(x0 + xx, 0, width - 1)] *
                        (xx ? wx : 1 - wx) * (yy ? wy : 1 - wy);
                }
                detector_input[(c * 512 + y) * 512 + x] = (std::nearbyint(value) - 127.5F) / 128;
            }
        }
    }
    const auto faces = detect(detector_input, float(resized_height) / height);
    const LatentSyncFaceDetection * selected = nullptr;
    int64_t max_area = 0;
    for (const auto & face : faces) {
        const int w = int(face.box[2]) - int(face.box[0]), h = int(face.box[3]) - int(face.box[1]);
        if (w < 50 || h < 80 || float(w) / h < 0.2F || float(w) / h > 1.5F) continue;
        const int64_t area = int64_t(w) * h;
        if (area > max_area) { selected = &face; max_area = area; }
    }
    if (!selected) throw std::runtime_error("LatentSync face not detected");
    const auto & box = selected->box;
    const double scale = 192.0 / (std::max(box[2] - box[0], box[3] - box[1]) * 1.5);
    const double center_x = (box[0] + box[2]) * 0.5, center_y = (box[1] + box[3]) * 0.5;
    // OpenCV's landmark crop uses 5-bit interpolation fractions and a 10-bit affine accumulator.
    std::vector<float> crop(3 * 192 * 192);
    for (int y = 0; y < 192; ++y) {
        const int sy = (int(std::nearbyint((y / scale + center_y - 96 / scale) * 1024)) + 16) >> 5;
        const int y0 = sy >> 5;
        const float wy = (sy & 31) / 32.0F;
        for (int x = 0; x < 192; ++x) {
            const int sx = (int(std::nearbyint((center_x - 96 / scale) * 1024)) + 16 + int(std::nearbyint(x / scale * 1024))) >> 5;
            const int x0 = sx >> 5;
            const float wx = (sx & 31) / 32.0F;
            for (int c = 0; c < 3; ++c) {
                float value = 0;
                for (int yy = 0; yy < 2; ++yy) for (int xx = 0; xx < 2; ++xx) {
                    const int ix = x0 + xx, iy = y0 + yy;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        value += rgb[(static_cast<size_t>(c) * height + iy) * width + ix] * (xx ? wx : 1 - wx) * (yy ? wy : 1 - wy);
                    }
                }
                crop[(c * 192 + y) * 192 + x] = std::floor(value + 0.5F);
            }
        }
    }
    LatentSyncAlignedFace output;
    output.landmarks = landmarks(crop);
    for (size_t i = 0; i < output.landmarks.size(); ++i) {
        output.landmarks[i] = float(((output.landmarks[i] + 1) * 96 - 96) / scale + (i % 2 ? center_y : center_x));
    }
    output.transform = impl_->alignment.transform(latentsync_alignment_points(output.landmarks));
    auto warped = latentsync_warp_affine(rgb, 3, height, width, output.transform, impl_->alignment.crop_height(), impl_->alignment.crop_width(), 127);
    for (auto & value : warped) value = std::floor(std::clamp(value, 0.0F, 255.0F));
    output.pixels = latentsync_resize_lanczos4(warped, 3, impl_->alignment.crop_height(), impl_->alignment.crop_width(), 512, 512);
    for (auto & value : output.pixels) value = std::nearbyint(std::clamp(value, 0.0F, 255.0F)) / 127.5F - 1;
    return output;
}

void LatentSyncFaceRuntime::reset_alignment() { impl_->alignment.reset(); }

}  // namespace engine::models::latentsync
