#include "engine/models/controlfoley/video_processing.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/ggml/examples/stb_image.h"

namespace engine::models::controlfoley::video {
namespace {

constexpr double kDefaultFrameDirectoryFps = 30.0;

bool is_supported_image_file(const std::filesystem::path & path) {
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
        extension == ".bmp" || extension == ".tga";
}

std::string frame_directory_help() {
    return "ControlFoley frame-directory fallback expects a directory containing "
           "decoded image frames (*.png/*.jpg/*.jpeg/*.bmp/*.tga) and optional "
           "timestamps.txt with one timestamp per frame";
}

std::vector<std::filesystem::path> list_frame_paths(const std::filesystem::path & path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("ControlFoley video path does not exist: " + path.string());
    }
    if (!std::filesystem::is_directory(path)) {
        throw std::runtime_error(frame_directory_help() + ": " + path.string());
    }

    std::vector<std::filesystem::path> out;
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_regular_file() && is_supported_image_file(entry.path())) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) {
        throw std::runtime_error(frame_directory_help() + ": " + path.string());
    }
    return out;
}

std::vector<double> read_timestamps(
    const std::filesystem::path & directory,
    size_t frame_count) {
    const auto path = directory / "timestamps.txt";
    std::vector<double> out;
    if (std::filesystem::exists(path)) {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("ControlFoley failed to read video timestamps: " + path.string());
        }
        std::string line;
        while (std::getline(stream, line)) {
            const size_t split = line.find(',');
            if (split != std::string::npos) {
                line.resize(split);
            }
            if (line.empty()) {
                continue;
            }
            out.push_back(std::stod(line));
        }
        if (out.size() != frame_count) {
            throw std::runtime_error("ControlFoley video timestamps count does not match frame count: " + path.string());
        }
        return out;
    }

    out.resize(frame_count);
    for (size_t i = 0; i < frame_count; ++i) {
        out[i] = static_cast<double>(i) / kDefaultFrameDirectoryFps;
    }
    return out;
}

struct ImageBuffer {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<uint8_t> rgb;
};

ImageBuffer load_rgb_image(const std::filesystem::path & path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char * data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (data == nullptr || width <= 0 || height <= 0) {
        if (data != nullptr) {
            stbi_image_free(data);
        }
        throw std::runtime_error("ControlFoley failed to load video frame image: " + path.string());
    }
    ImageBuffer out;
    out.width = width;
    out.height = height;
    out.rgb.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    stbi_image_free(data);
    return out;
}

struct Contributor {
    int64_t index = 0;
    double weight = 0.0;
};

double cubic_weight(double x) {
    constexpr double a = -0.75;
    x = std::abs(x);
    if (x <= 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    }
    return 0.0;
}

std::vector<std::array<Contributor, 4>> make_cubic_contributors(
    int64_t input_size,
    int64_t output_size) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::runtime_error("ControlFoley video resize shape is invalid");
    }
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    std::vector<std::array<Contributor, 4>> out(static_cast<size_t>(output_size));
    for (int64_t dst = 0; dst < output_size; ++dst) {
        const double center = (static_cast<double>(dst) + 0.5) * scale - 0.5;
        const int64_t start = static_cast<int64_t>(std::floor(center)) - 1;
        double sum = 0.0;
        for (int tap = 0; tap < 4; ++tap) {
            const int64_t src = std::clamp<int64_t>(start + tap, 0, input_size - 1);
            const double weight = cubic_weight(center - static_cast<double>(start + tap));
            out[static_cast<size_t>(dst)][static_cast<size_t>(tap)] = {src, weight};
            sum += weight;
        }
        if (sum != 0.0 && std::isfinite(sum)) {
            for (auto & contributor : out[static_cast<size_t>(dst)]) {
                contributor.weight /= sum;
            }
        }
    }
    return out;
}

ImageBuffer resize_rgb_bicubic(
    int64_t source_width,
    int64_t source_height,
    const io::RgbVideoFrame & source,
    int64_t width,
    int64_t height,
    const std::vector<std::array<Contributor, 4>> & x_contributors,
    const std::vector<std::array<Contributor, 4>> & y_contributors) {
    if (source_width == width && source_height == height) {
        ImageBuffer out;
        out.width = width;
        out.height = height;
        out.rgb.resize(static_cast<size_t>(height * width * 3));
        for (int64_t y = 0; y < height; ++y)
            std::copy_n(source.rgb.get() + y * source.stride, width * 3, out.rgb.data() + y * width * 3);
        return out;
    }

    std::vector<float> temp(static_cast<size_t>(source_height * width * 3));
    for (int64_t y = 0; y < source_height; ++y) {
        for (int64_t x = 0; x < width; ++x) {
            const auto & contributors = x_contributors[static_cast<size_t>(x)];
            for (int64_t c = 0; c < 3; ++c) {
                double value = 0.0;
                for (const auto & contributor : contributors) {
                    const int64_t src = y * source.stride + contributor.index * 3 + c;
                    value += static_cast<double>(source.rgb.get()[static_cast<size_t>(src)]) * contributor.weight;
                }
                temp[static_cast<size_t>((y * width + x) * 3 + c)] = static_cast<float>(value);
            }
        }
    }

    ImageBuffer out;
    out.width = width;
    out.height = height;
    out.rgb.resize(static_cast<size_t>(height * width * 3));
    for (int64_t y = 0; y < height; ++y) {
        const auto & contributors = y_contributors[static_cast<size_t>(y)];
        for (int64_t x = 0; x < width; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                double value = 0.0;
                for (const auto & contributor : contributors) {
                    const int64_t src = (contributor.index * width + x) * 3 + c;
                    value += static_cast<double>(temp[static_cast<size_t>(src)]) * contributor.weight;
                }
                value = std::clamp(value, 0.0, 255.0);
                out.rgb[static_cast<size_t>((y * width + x) * 3 + c)] = static_cast<uint8_t>(std::lround(value));
            }
        }
    }
    return out;
}

struct ResizePlan {
    ResizePlan(
        int64_t source_width,
        int64_t source_height,
        int64_t size,
        bool center_crop)
        : source_width(source_width),
          source_height(source_height),
          target_size(size),
          center_crop(center_crop) {
        if (center_crop) {
            const double ratio = static_cast<double>(size) / static_cast<double>(std::min(source_width, source_height));
            resized_width = static_cast<int64_t>(std::llround(static_cast<double>(source_width) * ratio));
            resized_height = static_cast<int64_t>(std::llround(static_cast<double>(source_height) * ratio));
        } else {
            resized_width = size;
            resized_height = size;
        }
        x_contributors = make_cubic_contributors(source_width, resized_width);
        y_contributors = make_cubic_contributors(source_height, resized_height);
    }

    ImageBuffer resize(const io::RgbVideoFrame & source) const {
        auto resized = resize_rgb_bicubic(
            source_width,
            source_height,
            source,
            resized_width,
            resized_height,
            x_contributors,
            y_contributors);
        if (!center_crop) {
            return resized;
        }

        const int64_t crop_x = std::max<int64_t>(0, (resized.width - target_size) / 2);
        const int64_t crop_y = std::max<int64_t>(0, (resized.height - target_size) / 2);
        ImageBuffer out;
        out.width = target_size;
        out.height = target_size;
        out.rgb.resize(static_cast<size_t>(target_size * target_size * 3));
        for (int64_t y = 0; y < target_size; ++y) {
            const auto * src = resized.rgb.data() + static_cast<size_t>(((crop_y + y) * resized.width + crop_x) * 3);
            auto * dst = out.rgb.data() + static_cast<size_t>(y * target_size * 3);
            std::memcpy(dst, src, static_cast<size_t>(target_size * 3));
        }
        return out;
    }

    int64_t source_width = 0;
    int64_t source_height = 0;
    int64_t target_size = 0;
    int64_t resized_width = 0;
    int64_t resized_height = 0;
    bool center_crop = false;
    std::vector<std::array<Contributor, 4>> x_contributors;
    std::vector<std::array<Contributor, 4>> y_contributors;
};

io::RgbVideo load_frame_directory(const std::filesystem::path & path) {
    const auto frame_paths = list_frame_paths(path);
    const auto timestamps = read_timestamps(path, frame_paths.size());
    io::RgbVideo out;
    out.frames.reserve(frame_paths.size());
    for (size_t i = 0; i < frame_paths.size(); ++i) {
        auto image = load_rgb_image(frame_paths[i]);
        if (i == 0) {
            out.width = image.width;
            out.height = image.height;
        } else if (out.width != image.width || out.height != image.height) {
            throw std::runtime_error("ControlFoley video frame directory has inconsistent frame sizes");
        }
        io::RgbVideoFrame frame;
        frame.time = timestamps[i];
        auto pixels = std::make_shared<std::vector<uint8_t>>(std::move(image.rgb));
        frame.stride = out.width * 3;
        frame.rgb = std::shared_ptr<const uint8_t>(pixels, pixels->data());
        out.frames.push_back(std::move(frame));
    }
    return out;
}

}  // namespace

io::RgbVideo load_video_frames(const std::filesystem::path & path, double max_duration_sec) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("ControlFoley video path does not exist: " + path.string());
    }
    if (std::filesystem::is_directory(path)) {
        return load_frame_directory(path);
    }
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "ControlFoley video path is neither a regular video file nor " + frame_directory_help() + ": " +
            path.string());
    }
    return io::load_video_rgb(path, max_duration_sec);
}

std::vector<int64_t> select_source_indices(
    const io::RgbVideo & video,
    float duration_sec,
    int64_t fps) {
    const int64_t expected_frames = static_cast<int64_t>(duration_sec * static_cast<float>(fps));
    if (expected_frames <= 0) {
        throw std::runtime_error("ControlFoley video is too short for the requested duration");
    }
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(expected_frames));
    double next_timestamp = 0.0;
    const double frame_interval = 1.0 / static_cast<double>(fps);
    for (size_t index = 0; index < video.frames.size(); ++index) {
        while (video.frames[index].time >= next_timestamp && static_cast<int64_t>(out.size()) < expected_frames) {
            out.push_back(static_cast<int64_t>(index));
            next_timestamp += frame_interval;
        }
        if (static_cast<int64_t>(out.size()) >= expected_frames) {
            break;
        }
    }
    if (static_cast<int64_t>(out.size()) < expected_frames) {
        throw std::runtime_error("ControlFoley video is too short for the requested duration");
    }
    return out;
}

void write_normalized_frame(
    const std::vector<uint8_t> & rgb,
    int64_t written,
    int64_t size,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std,
    std::vector<float> & out) {
    for (int64_t y = 0; y < size; ++y) {
        for (int64_t x = 0; x < size; ++x) {
            for (int64_t c = 0; c < 3; ++c) {
                const int64_t src = (y * size + x) * 3 + c;
                const int64_t dst = ((written * 3 + c) * size + y) * size + x;
                const float value = static_cast<float>(rgb[static_cast<size_t>(src)]) / 255.0F;
                out[static_cast<size_t>(dst)] = (value - mean[static_cast<size_t>(c)]) / std[static_cast<size_t>(c)];
            }
        }
    }
}

ResizedFrameCache::ResizedFrameCache(
    const io::RgbVideo & video,
    int64_t size,
    bool center_crop)
    : video_(&video),
      size_(size),
      center_crop_(center_crop) {
    if (size_ <= 0) {
        throw std::runtime_error("ControlFoley video target size is invalid");
    }
}

std::vector<float> ResizedFrameCache::extract(
    float duration_sec,
    int64_t fps,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std,
    std::string_view trace_name) {
    std::vector<int64_t> indices;
    const double select_ms = engine::debug::measure_ms([&]() {
        indices = select_source_indices(*video_, duration_sec, fps);
    });
    std::vector<float> out(static_cast<size_t>(indices.size() * 3 * size_ * size_));
    const ResizePlan resize_plan(video_->width, video_->height, size_, center_crop_);
    double resize_ms = 0.0;
    double write_ms = 0.0;
    std::vector<int64_t> miss_indices;
    std::unordered_map<int64_t, size_t> planned_misses;
    planned_misses.reserve(indices.size());
    for (const auto index : indices) {
        if (cache_.find(index) == cache_.end() && planned_misses.find(index) == planned_misses.end()) {
            planned_misses.emplace(index, miss_indices.size());
            miss_indices.push_back(index);
        }
    }
    const int64_t cache_misses = static_cast<int64_t>(miss_indices.size());
    std::vector<ImageBuffer> resized_misses(miss_indices.size());
    resize_ms = engine::debug::measure_ms([&]() {
#ifdef _OPENMP
#pragma omp parallel for if(static_cast<int64_t>(miss_indices.size()) >= 8)
#endif
        for (int64_t miss = 0; miss < static_cast<int64_t>(miss_indices.size()); ++miss) {
            const int64_t index = miss_indices[static_cast<size_t>(miss)];
            resized_misses[static_cast<size_t>(miss)] =
                resize_plan.resize(video_->frames[static_cast<size_t>(index)]);
        }
    });
    for (size_t miss = 0; miss < miss_indices.size(); ++miss) {
        cache_.emplace(miss_indices[miss], std::move(resized_misses[miss].rgb));
    }
    for (size_t written = 0; written < indices.size(); ++written) {
        const auto cached = cache_.find(indices[written]);
        if (cached == cache_.end()) {
            throw std::runtime_error("ControlFoley resized video frame cache miss after resize planning");
        }
        write_ms += engine::debug::measure_ms([&]() {
            write_normalized_frame(cached->second, static_cast<int64_t>(written), size_, mean, std, out);
        });
    }
    if (!trace_name.empty()) {
        const std::string prefix(trace_name);
        engine::debug::timing_log_scalar(prefix + ".select_ms", select_ms);
        engine::debug::timing_log_scalar(prefix + ".resize_ms", resize_ms);
        engine::debug::timing_log_scalar(prefix + ".write_ms", write_ms);
        engine::debug::timing_log_scalar(prefix + ".cache_misses", cache_misses);
    }
    return out;
}

std::vector<float> extract_video_frames(
    const io::RgbVideo & video,
    float duration_sec,
    int64_t fps,
    int64_t size,
    bool center_crop,
    const std::array<float, 3> & mean,
    const std::array<float, 3> & std) {
    ResizedFrameCache cache(video, size, center_crop);
    return cache.extract(duration_sec, fps, mean, std);
}

}  // namespace engine::models::controlfoley::video
