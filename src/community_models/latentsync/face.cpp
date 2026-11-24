#include "engine/community_models/latentsync/face.h"

#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <ggml-alloc.h>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace engine::models::latentsync {

LatentSyncFacePoints latentsync_alignment_points(const std::vector<float> & landmarks) {
    if (landmarks.size() != 212) throw std::runtime_error("LatentSync alignment requires 106 landmarks");
    constexpr int left[] = {43, 48, 49, 51, 50};
    constexpr int right[] = {101, 102, 103, 104, 105};
    constexpr int nose[] = {74, 77, 83, 86};
    LatentSyncFacePoints points{};
    for (int d = 0; d < 2; ++d) {
        for (const int i : left) points[0][d] += std::nearbyint(landmarks[i * 2 + d]);
        for (const int i : right) points[1][d] += std::nearbyint(landmarks[i * 2 + d]);
        for (const int i : nose) points[2][d] += std::nearbyint(landmarks[i * 2 + d]);
        points[0][d] = std::nearbyint(points[0][d] / 5);
        points[1][d] = std::nearbyint(points[1][d] / 5);
        points[2][d] = std::nearbyint(points[2][d] / 4);
    }
    return points;
}

LatentSyncFaceAlignment::LatentSyncFaceAlignment(int resolution) {
    if (resolution <= 0) throw std::runtime_error("LatentSync face resolution must be positive");
    const double ratio = resolution / 256.0 * 2.8;
    target_ = {{{float(17 * ratio), float(20 * ratio)},
                {float(58 * ratio), float(20 * ratio)},
                {float(37.5 * ratio), float(40 * ratio)}}};
    width_ = int(75 * ratio);
    height_ = int(100 * ratio);
}

LatentSyncAffine LatentSyncFaceAlignment::transform(const LatentSyncFacePoints & points, bool smooth) {
    std::array<float, 2> center{}, target_center{};
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            center[d] += points[i][d] / 3;
            target_center[d] += target_[i][d] / 3;
        }
    }
    auto source = points;
    auto target = target_;
    float source_variance = 0, target_variance = 0;
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            source[i][d] -= center[d];
            target[i][d] -= target_center[d];
            source_variance += source[i][d] * source[i][d];
            target_variance += target[i][d] * target[i][d];
        }
    }
    const float s1 = std::sqrt(source_variance / 5);
    const float s2 = std::sqrt(target_variance / 5);
    if (!(s1 > 0)) throw std::runtime_error("Degenerate LatentSync face alignment points");
    float dot = 0, cross = 0;
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 2; ++d) {
            source[i][d] /= s1;
            target[i][d] /= s2;
        }
        dot += source[i][0] * target[i][0] + source[i][1] * target[i][1];
        cross += source[i][0] * target[i][1] - source[i][1] * target[i][0];
    }
    // Proper 2D Procrustes rotation, equivalent to the reference's determinant-corrected SVD.
    const float norm = std::hypot(dot, cross);
    if (!(norm > 0)) throw std::runtime_error("Indeterminate LatentSync face rotation");
    const float a = s2 / s1 * dot / norm;
    const float b = s2 / s1 * cross / norm;
    LatentSyncAffine matrix{a, -b, target_center[0] - a * center[0] + b * center[1],
                           b, a, target_center[1] - b * center[0] - a * center[1]};
    if (smooth) {
        std::array<float, 2> bias;
        for (int d = 0; d < 2; ++d) {
            bias[d] = target[2][d] - source[2][d];
            if (previous_bias_) bias[d] = (*previous_bias_)[d] * 0.2F + bias[d] * 0.8F;
            matrix[d * 3 + 2] += bias[d];
        }
        previous_bias_ = bias;
    }
    return matrix;
}

void LatentSyncFaceAlignment::reset() { previous_bias_.reset(); }

std::vector<float> latentsync_warp_affine(const std::vector<float> & image,
    int channels, int height, int width, const LatentSyncAffine & matrix,
    int output_height, int output_width, float fill, [[maybe_unused]] int threads) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) {
        throw std::runtime_error("Invalid LatentSync affine image dimensions");
    }
    const double determinant = double(matrix[0]) * matrix[4] - double(matrix[1]) * matrix[3];
    if (!std::isfinite(determinant) || determinant == 0) throw std::runtime_error("Singular LatentSync affine transform");
    const double a = matrix[4] / determinant;
    const double b = -matrix[1] / determinant;
    const double c = -matrix[3] / determinant;
    const double d = matrix[0] / determinant;
    std::vector<float> output(static_cast<size_t>(channels) * output_height * output_width);
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            const double dx = x - matrix[2], dy = y - matrix[5];
            const double sx = a * dx + b * dy, sy = c * dx + d * dy;
            const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            const float wx = float(sx - x0), wy = float(sy - y0);
            for (int channel = 0; channel < channels; ++channel) {
                float value = 0;
                for (int yy = 0; yy < 2; ++yy) {
                    for (int xx = 0; xx < 2; ++xx) {
                        const int ix = x0 + xx, iy = y0 + yy;
                        const float pixel = ix >= 0 && ix < width && iy >= 0 && iy < height
                            ? image[(static_cast<size_t>(channel) * height + iy) * width + ix] : fill;
                        value += pixel * (xx ? wx : 1 - wx) * (yy ? wy : 1 - wy);
                    }
                }
                output[(static_cast<size_t>(channel) * output_height + y) * output_width + x] = value;
            }
        }
    }
    return output;
}

std::vector<float> latentsync_resize_lanczos4(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width,
    [[maybe_unused]] int threads) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) {
        throw std::runtime_error("Invalid LatentSync resize dimensions");
    }
    struct Taps { int start; std::array<float, 8> weights; };
    std::array<std::vector<Taps>, 2> axes;
    constexpr double pi = 3.14159265358979323846;
    for (int axis = 0; axis < 2; ++axis) {
        const int source_size = axis ? height : width;
        const int target_size = axis ? output_height : output_width;
        axes[axis].resize(target_size);
        for (int i = 0; i < target_size; ++i) {
            const double position = (i + 0.5) * source_size / target_size - 0.5;
            auto & taps = axes[axis][i];
            taps.start = int(std::floor(position)) - 3;
            double sum = 0;
            for (int j = 0; j < 8; ++j) {
                const double distance = position - (taps.start + j);
                const double weight = std::abs(distance) < 1e-12 ? 1 :
                    std::sin(pi * distance) * std::sin(pi * distance / 4) / (pi * pi * distance * distance / 4);
                taps.weights[j] = float(weight);
                sum += weight;
            }
            for (auto & weight : taps.weights) weight /= float(sum);
        }
    }
    std::vector<float> horizontal(static_cast<size_t>(channels) * height * output_width);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                float value = 0;
                const auto & taps = axes[0][x];
                for (int j = 0; j < 8; ++j) {
                    value += image[(static_cast<size_t>(c) * height + y) * width + std::clamp(taps.start + j, 0, width - 1)] * taps.weights[j];
                }
                horizontal[(static_cast<size_t>(c) * height + y) * output_width + x] = value;
            }
        }
    }
    std::vector<float> output(static_cast<size_t>(channels) * output_height * output_width);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (int c = 0; c < channels; ++c) {
        for (int y = 0; y < output_height; ++y) {
            const auto & taps = axes[1][y];
            for (int x = 0; x < output_width; ++x) {
                float value = 0;
                for (int j = 0; j < 8; ++j) {
                    value += horizontal[(static_cast<size_t>(c) * height + std::clamp(taps.start + j, 0, height - 1)) * output_width + x] * taps.weights[j];
                }
                output[(static_cast<size_t>(c) * output_height + y) * output_width + x] = value;
            }
        }
    }
    return output;
}

namespace {

std::vector<float> erode_square(const std::vector<float> & image, int height, int width, int size,
    [[maybe_unused]] int threads) {
    if (size <= 0) throw std::runtime_error("LatentSync restoration mask is too small");
    auto input = image;
    std::vector<float> output(image.size());
    const int anchor = size / 2;
    for (int axis = 0; axis < 2; ++axis) {
        const int lines = axis ? width : height, length = axis ? height : width;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
        for (int line = 0; line < lines; ++line) {
            std::deque<std::pair<int, float>> queue;
            for (int position = -anchor; position < length + size - 1 - anchor; ++position) {
                const float value = position >= 0 && position < length
                    ? input[axis ? position * width + line : line * width + position]
                    : std::numeric_limits<float>::infinity();
                while (!queue.empty() && queue.back().second >= value) queue.pop_back();
                queue.emplace_back(position, value);
                while (queue.front().first <= position - size) queue.pop_front();
                const int target = position - (size - 1 - anchor);
                if (target >= 0 && target < length) output[axis ? target * width + line : line * width + target] = queue.front().second;
            }
        }
        input.swap(output);
    }
    return input;
}

std::vector<float> gaussian_mask(const std::vector<float> & image, int height, int width, int radius,
    [[maybe_unused]] int threads) {
    const float sigma = 0.3F * (radius - 1) + 0.8F;
    std::vector<float> kernel(radius * 2 + 1);
    float total = 0;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = std::exp(-float(i * i) / (2 * sigma * sigma));
        total += kernel[i + radius];
    }
    for (auto & value : kernel) value /= total;
    auto input = image;
    std::vector<float> output(image.size());
    for (int axis = 0; axis < 2; ++axis) {
        const int length = axis ? height : width;
        if (radius >= length) throw std::runtime_error("LatentSync restoration blur exceeds frame dimensions");
        if (axis == 0) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    float value = 0;
                    for (int i = -radius; i <= radius; ++i) {
                        int source = x + i;
                        if (source < 0) source = -source;
                        if (source >= width) source = 2 * width - 2 - source;
                        value += input[y * width + source] * kernel[i + radius];
                    }
                    output[y * width + x] = value;
                }
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
            for (int y = 0; y < height; ++y) {
                auto * destination = output.data() + static_cast<size_t>(y) * width;
                std::fill_n(destination, width, 0.0F);
                for (int i = -radius; i <= radius; ++i) {
                    int source = y + i;
                    if (source < 0) source = -source;
                    if (source >= height) source = 2 * height - 2 - source;
                    const auto * row = input.data() + static_cast<size_t>(source) * width;
                    const float weight = kernel[i + radius];
                    for (int x = 0; x < width; ++x) destination[x] += row[x] * weight;
                }
            }
        }
        input.swap(output);
    }
    return input;
}

}  // namespace

std::vector<float> latentsync_resize_bicubic(const std::vector<float> & image,
    int channels, int height, int width, int output_height, int output_width) {
    if (channels <= 0 || height <= 0 || width <= 0 || output_height <= 0 || output_width <= 0 ||
        image.size() != static_cast<size_t>(channels) * height * width) throw std::runtime_error("Invalid LatentSync bicubic resize dimensions");
    auto input = image;
    for (int axis = 0; axis < 2; ++axis) {
        const int source_size = axis ? height : width, target_size = axis ? output_height : output_width;
        const int other = axis ? output_width : height;
        const double scale = double(source_size) / target_size;
        const double filter_scale = std::max(1.0, scale);
        const double support = 2 * filter_scale;
        std::vector<float> output(static_cast<size_t>(channels) * other * target_size);
        for (int target = 0; target < target_size; ++target) {
            const double center = (target + 0.5) * scale;
            const int begin = std::max(0, int(center - support + 0.5));
            const int end = std::min(source_size, int(center + support + 0.5));
            std::vector<float> weights(end - begin);
            double sum = 0;
            for (int source = begin; source < end; ++source) {
                const double x = std::abs((source + 0.5 - center) / filter_scale);
                const double value = x < 1 ? ((1.5 * x - 2.5) * x * x + 1) :
                    (x < 2 ? ((-0.5 * x + 2.5) * x - 4) * x + 2 : 0);
                weights[source - begin] = float(value);
                sum += value;
            }
            for (auto & weight : weights) weight /= float(sum);
            for (int c = 0; c < channels; ++c) for (int j = 0; j < other; ++j) {
                float value = 0;
                for (int source = begin; source < end; ++source) {
                    const size_t index = axis ? (static_cast<size_t>(c) * height + source) * output_width + j
                                              : (static_cast<size_t>(c) * height + j) * width + source;
                    value += input[index] * weights[source - begin];
                }
                const size_t index = axis ? (static_cast<size_t>(c) * target_size + target) * other + j
                                          : (static_cast<size_t>(c) * other + j) * target_size + target;
                output[index] = value;
            }
        }
        input = std::move(output);
    }
    return input;
}

std::vector<float> latentsync_restore_face(const std::vector<float> & original,
    int height, int width, const std::vector<float> & face, const LatentSyncAffine & transform, int threads) {
    if (height <= 0 || width <= 0 || original.size() != static_cast<size_t>(height) * width * 3) {
        throw std::runtime_error("Invalid LatentSync restoration frame");
    }
    const auto resized = latentsync_resize_bicubic(face, 3, 512, 512, 560, 420);
    const float det = transform[0] * transform[4] - transform[1] * transform[3];
    if (!std::isfinite(det) || det == 0) throw std::runtime_error("Singular LatentSync restoration transform");
    LatentSyncAffine inverse{transform[4] / det, -transform[1] / det, 0, -transform[3] / det, transform[0] / det, 0};
    inverse[2] = -inverse[0] * transform[2] - inverse[1] * transform[5];
    inverse[5] = -inverse[3] * transform[2] - inverse[4] * transform[5];
    auto warped = latentsync_warp_affine(resized, 3, 560, 420, inverse, height, width, 127, threads);
    auto mask = latentsync_warp_affine(std::vector<float>(560 * 420, 1), 1, 560, 420, inverse, height, width, 0, threads);
    mask = erode_square(mask, height, width, 2, threads);
    const double area = std::accumulate(mask.begin(), mask.end(), 0.0);
    const int edge = int(std::sqrt(area)) / 20;
    const auto center = erode_square(mask, height, width, edge * 2, threads);
    const auto soft = gaussian_mask(center, height, width, edge, threads);
    const size_t pixels = static_cast<size_t>(height) * width;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) if(threads > 1) schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(warped.size()); ++i) {
        const size_t pixel = i % pixels;
        const float value = std::clamp(warped[i] / 2 + 0.5F, 0.0F, 1.0F) * 255;
        warped[i] = std::floor(std::clamp(soft[pixel] * mask[pixel] * value + (1 - soft[pixel]) * original[i], 0.0F, 255.0F));
    }
    return warped;
}

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
    modules::ScrfdWeights detector_weights;
    modules::Landmark106Weights landmark_weights;
    Graph detector, landmark;
    LatentSyncFaceAlignment alignment;

    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & detector_source,
        const std::filesystem::path & detector_config, const assets::TensorSource & landmark_source,
        const std::filesystem::path & landmark_config, assets::TensorStorageType type)
        : execution(execution_in),
          store(execution.backend(), execution.backend_type(), "latentsync.face.weights", 4 * 1024 * 1024),
          detector_weights(modules::load_scrfd_weights(store, detector_source, detector_config, type)),
          landmark_weights(modules::load_landmark106_weights(store, landmark_source, landmark_config, type)),
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
            if (is_detector) item->outputs = modules::ScrfdDetectorModule().build(ctx, item->input, detector_weights);
            else item->outputs = {modules::Landmark106Module().build(ctx, item->input, landmark_weights)};
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

std::vector<modules::ScrfdFaceDetection> LatentSyncFaceRuntime::detect(const std::vector<float> & image, float resize_scale) {
    const auto start = std::chrono::steady_clock::now();
    auto & graph = impl_->detector;
    core::write_tensor_f32(graph.input, image);
    if (core::compute_backend_graph(graph.backend, graph.graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("LatentSync face detector execution failed");
    }
    std::array<std::vector<float>, 9> heads;
    for (size_t i = 0; i < heads.size(); ++i) heads[i] = core::read_tensor_f32(graph.outputs[i].tensor);
    auto faces = modules::decode_scrfd_faces(heads, resize_scale);
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
    const modules::ScrfdFaceDetection * selected = nullptr;
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
    const int threads = impl_->execution.config().threads;
    auto warped = latentsync_warp_affine(rgb, 3, height, width, output.transform,
        impl_->alignment.crop_height(), impl_->alignment.crop_width(), 127, threads);
    for (auto & value : warped) value = std::floor(std::clamp(value, 0.0F, 255.0F));
    output.pixels = latentsync_resize_lanczos4(warped, 3, impl_->alignment.crop_height(),
        impl_->alignment.crop_width(), 512, 512, threads);
    for (auto & value : output.pixels) value = std::nearbyint(std::clamp(value, 0.0F, 255.0F)) / 127.5F - 1;
    return output;
}

void LatentSyncFaceRuntime::reset_alignment() { impl_->alignment.reset(); }

}  // namespace engine::models::latentsync
