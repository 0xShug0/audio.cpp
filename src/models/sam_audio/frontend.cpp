#include "engine/models/sam_audio/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "../../../external/ggml/examples/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace engine::models::sam_audio {
namespace {
struct FilterSpan {
    int start;
    std::vector<float> weights;
};

std::vector<FilterSpan> resize_filter(int input, int output) {
    const float scale = static_cast<float>(input) / output;
    const float support = 2.0f * std::max(scale, 1.0f);
    const float inverse = scale >= 1.0f ? 1.0f / scale : 1.0f;
    std::vector<FilterSpan> spans;
    spans.reserve(output);
    for (int i = 0; i < output; ++i) {
        const float center = scale * (i + 0.5f);
        const int start = std::max(static_cast<int>(center - support + 0.5f), 0);
        const int end = std::min(static_cast<int>(center + support + 0.5f), input);
        FilterSpan span{start, {}};
        float sum = 0;
        for (int j = start; j < end; ++j) {
            const float x = std::abs((j - center + 0.5f) * inverse);
            const float weight = x < 1.0f ? (1.5f * x - 2.5f) * x * x + 1.0f :
                x < 2.0f ? (((x - 5.0f) * x + 8.0f) * x - 4.0f) * -0.5f : 0.0f;
            span.weights.push_back(weight);
            sum += weight;
        }
        for (auto & weight : span.weights) weight /= sum;
        spans.push_back(std::move(span));
    }
    return spans;
}
}  // namespace

std::vector<size_t> select_video_frames(const SourceVideo & video, int64_t frames,
                                       int64_t hop, int sample_rate) {
    if (video.frames.empty() || frames <= 0 || hop <= 0 || sample_rate <= 0)
        throw std::runtime_error("SAM Audio video alignment requires frames and a positive audio time base");
    std::vector<size_t> selected(frames);
    for (int64_t t = 0; t < frames; ++t) {
        // Python creates audio timestamps in F32, then compares with F64 video PTS.
        const double timestamp = static_cast<float>(t * hop) / static_cast<float>(sample_rate);
        size_t best = 0;
        for (size_t i = 1; i < video.frames.size(); ++i)
            if (std::abs(video.frames[i].time - timestamp) < std::abs(video.frames[best].time - timestamp)) best = i;
        selected[t] = best;
    }
    return selected;
}

std::vector<float> load_reference_image(const std::filesystem::path & path) {
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> image(
        stbi_load(path.string().c_str(), &width, &height, &channels, 3), stbi_image_free);
    if (!image) throw std::runtime_error("SAM Audio cannot read reference image: " + path.string());
    const std::vector<uint8_t> rgb(image.get(), image.get() + static_cast<size_t>(width) * height * 3);
    return prepare_video_frame(rgb, width, height);
}

std::vector<float> prepare_video_frame(const std::vector<uint8_t> & rgb, int width, int height) {
    if (width <= 0 || height <= 0 || rgb.size() != static_cast<size_t>(width) * height * 3)
        throw std::runtime_error("SAM Audio video frame has an invalid RGB shape");
    constexpr int size = 336;
    const auto horizontal = resize_filter(width, size);
    const auto vertical = resize_filter(height, size);
    std::vector<float> rows(static_cast<size_t>(height) * size * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < size; ++x) {
            const auto & span = horizontal[x];
            for (int c = 0; c < 3; ++c) {
                float value = 0;
                for (size_t i = 0; i < span.weights.size(); ++i)
                    value += rgb[(static_cast<size_t>(y) * width + span.start + i) * 3 + c] * span.weights[i];
                rows[(static_cast<size_t>(y) * size + x) * 3 + c] = value;
            }
        }
    }
    std::vector<float> result(3 * size * size);
    for (int y = 0; y < size; ++y) {
        const auto & span = vertical[y];
        for (int x = 0; x < size; ++x) {
            for (int c = 0; c < 3; ++c) {
                float value = 0;
                for (size_t i = 0; i < span.weights.size(); ++i)
                    value += rows[((span.start + i) * size + x) * 3 + c] * span.weights[i];
                // torchvision converts the resized float tensor back to uint8 first.
                value = std::nearbyint(std::clamp(value, 0.0f, 255.0f));
                result[(c * size + y) * size + x] = (value / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return result;
}

}  // namespace engine::models::sam_audio
