#include "pipeline_internal.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/codecs/wan_video_vae_runtime.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/speech_encoders/hubert_encoder.h"
#include "wan_s2v_audio_conditioner.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <ggml-alloc.h>
#include <ggml.h>

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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::liveavatar {

using Clock = std::chrono::steady_clock;

constexpr int32_t kT5PadTokenId = 0;
constexpr int32_t kT5EosTokenId = 1;
constexpr std::array<float, 16> kWan21LatentsMean = {
    -0.7571F, -0.7089F, -0.9113F, 0.1075F, -0.1745F, 0.9653F, -0.1517F, 1.5508F,
    0.4134F, -0.0715F, 0.5517F, -0.3632F, -0.1922F, -0.9497F, 0.2503F, -0.2921F,
};
constexpr std::array<float, 16> kWan21LatentsStd = {
    2.8184F, 1.4541F, 2.3275F, 2.6558F, 1.2196F, 1.7708F, 2.6052F, 2.0743F,
    3.2687F, 2.1526F, 2.8652F, 1.5579F, 1.6382F, 1.1253F, 2.8251F, 1.9160F,
};
constexpr const char * kDefaultNegativePrompt =
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

engine::modules::T5BaseEncoderConfig liveavatar_text_encoder_config() {
    engine::modules::T5BaseEncoderConfig config;
    config.hidden_size = 4096;
    config.layers = 24;
    config.attention_heads = 64;
    config.head_dim = 64;
    config.intermediate_size = 10240;
    config.vocab_size = 256384;
    config.rms_norm_eps = 1.0e-6F;
    config.relative_attention_num_buckets = 32;
    config.relative_attention_max_distance = 128;
    config.feed_forward_kind = engine::modules::T5BaseFeedForwardKind::GatedGeluTanh;
    config.shared_relative_position_bias = false;
    return config;
}

std::shared_ptr<const LiveAvatarAssets> require_assets(std::shared_ptr<const LiveAvatarAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("LiveAvatar pipeline requires assets");
    }
    return assets;
}

std::vector<float> normalize_wav2vec2_input(std::vector<float> values) {
    if (values.size() <= 1) {
        return values;
    }
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double variance_sum = 0.0;
    for (const float value : values) {
        const double centered = static_cast<double>(value) - mean;
        variance_sum += centered * centered;
    }
    const double variance = variance_sum / static_cast<double>(values.size() - 1);
    const float scale = static_cast<float>(1.0 / std::sqrt(variance + 1.0e-7));
    for (float & value : values) {
        value = (value - static_cast<float>(mean)) * scale;
    }
    return values;
}

std::vector<float> slice_audio_bucket_frames(
    const std::vector<float> & values,
    int64_t layers,
    int64_t dims,
    int64_t frames,
    int64_t start,
    int64_t count) {
    if (layers <= 0 || dims <= 0 || frames <= 0 || count < 0 ||
        start < 0 || start + count > frames) {
        throw std::runtime_error("LiveAvatar audio bucket slice is invalid");
    }
    if (static_cast<int64_t>(values.size()) != layers * dims * frames) {
        throw std::runtime_error("LiveAvatar audio bucket size mismatch");
    }
    std::vector<float> out(static_cast<size_t>(layers * dims * count));
    for (int64_t layer = 0; layer < layers; ++layer) {
        for (int64_t dim = 0; dim < dims; ++dim) {
            for (int64_t frame = 0; frame < count; ++frame) {
                const size_t src = static_cast<size_t>(
                    ((layer * dims + dim) * frames) + start + frame);
                const size_t dst = static_cast<size_t>(
                    ((layer * dims + dim) * count) + frame);
                out[dst] = values[src];
            }
        }
    }
    return out;
}

struct RgbImage {
    int64_t width = 0;
    int64_t height = 0;
    std::vector<uint8_t> pixels;
};

RgbImage load_rgb_image(const std::filesystem::path & path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char * data = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (data == nullptr || width <= 0 || height <= 0) {
        if (data != nullptr) {
            stbi_image_free(data);
        }
        throw std::runtime_error("LiveAvatar failed to load reference image: " + path.string());
    }
    RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(data, data + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    stbi_image_free(data);
    return image;
}

std::vector<float> resize_center_crop_bilinear_to_vae_input(
    const RgbImage & source,
    int64_t target_height,
    int64_t target_width) {
    if (target_height <= 0 || target_width <= 0) {
        throw std::runtime_error("LiveAvatar target image size is invalid");
    }
    int64_t crop_x = 0;
    int64_t crop_y = 0;
    int64_t crop_width = source.width;
    int64_t crop_height = source.height;
    const double old_aspect = static_cast<double>(source.width) / static_cast<double>(source.height);
    const double new_aspect = static_cast<double>(target_width) / static_cast<double>(target_height);
    if (old_aspect > new_aspect) {
        crop_x = static_cast<int64_t>(std::llround(
            (static_cast<double>(source.width) - static_cast<double>(source.width) * (new_aspect / old_aspect)) / 2.0));
        crop_width = source.width - crop_x * 2;
    } else if (old_aspect < new_aspect) {
        crop_y = static_cast<int64_t>(std::llround(
            (static_cast<double>(source.height) - static_cast<double>(source.height) * (old_aspect / new_aspect)) / 2.0));
        crop_height = source.height - crop_y * 2;
    }
    if (crop_width <= 0 || crop_height <= 0) {
        throw std::runtime_error("LiveAvatar reference image crop is empty");
    }
    std::vector<float> out(static_cast<size_t>(3 * target_height * target_width));
    for (int64_t y = 0; y < target_height; ++y) {
        const double src_y_f = (static_cast<double>(y) + 0.5) *
            static_cast<double>(crop_height) / static_cast<double>(target_height) - 0.5;
        const int64_t y0_raw = static_cast<int64_t>(std::floor(src_y_f));
        const int64_t y1_raw = y0_raw + 1;
        const double wy = src_y_f - static_cast<double>(y0_raw);
        const int64_t y0 = crop_y + std::clamp<int64_t>(y0_raw, 0, crop_height - 1);
        const int64_t y1 = crop_y + std::clamp<int64_t>(y1_raw, 0, crop_height - 1);
        for (int64_t x = 0; x < target_width; ++x) {
            const double src_x_f = (static_cast<double>(x) + 0.5) *
                static_cast<double>(crop_width) / static_cast<double>(target_width) - 0.5;
            const int64_t x0_raw = static_cast<int64_t>(std::floor(src_x_f));
            const int64_t x1_raw = x0_raw + 1;
            const double wx = src_x_f - static_cast<double>(x0_raw);
            const int64_t x0 = crop_x + std::clamp<int64_t>(x0_raw, 0, crop_width - 1);
            const int64_t x1 = crop_x + std::clamp<int64_t>(x1_raw, 0, crop_width - 1);
            for (int64_t c = 0; c < 3; ++c) {
                const auto at = [&](int64_t yy, int64_t xx) -> double {
                    return static_cast<double>(
                        source.pixels[static_cast<size_t>((yy * source.width + xx) * 3 + c)]);
                };
                const double top = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                const double bottom = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                const double value = top * (1.0 - wy) + bottom * wy;
                const int64_t dst = (c * target_height + y) * target_width + x;
                out[static_cast<size_t>(dst)] = static_cast<float>(value / 127.5 - 1.0);
            }
        }
    }
    return out;
}

std::vector<float> slice_video_frames(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start,
    int64_t length) {
    if (start < 0 || length <= 0 || start + length > frames ||
        static_cast<int64_t>(values.size()) != channels * frames * height * width) {
        throw std::runtime_error("LiveAvatar video slice shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * length * height * width));
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < length; ++t) {
            const auto src = values.data() + static_cast<size_t>(((c * frames + start + t) * height) * width);
            auto * dst = out.data() + static_cast<size_t>(((c * length + t) * height) * width);
            std::copy(src, src + height * width, dst);
        }
    }
    return out;
}

std::vector<float> slice_video_time(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t frame_start,
    int64_t frame_count) {
    if (channels <= 0 || frames <= 0 || height <= 0 || width <= 0 ||
        frame_start < 0 || frame_count <= 0 || frame_start + frame_count > frames ||
        static_cast<int64_t>(values.size()) != channels * frames * height * width) {
        throw std::runtime_error("LiveAvatar temporal slice shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * frame_count * height * width));
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < frame_count; ++t) {
            const auto * src = values.data() +
                static_cast<size_t>(((c * frames + frame_start + t) * height) * width);
            auto * dst = out.data() +
                static_cast<size_t>(((c * frame_count + t) * height) * width);
            std::copy(src, src + height * width, dst);
        }
    }
    return out;
}

std::vector<float> repeat_video_frame(
    const std::vector<float> & values,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t frames) {
    if (channels <= 0 || height <= 0 || width <= 0 || frames <= 0 ||
        static_cast<int64_t>(values.size()) != channels * height * width) {
        throw std::runtime_error("LiveAvatar repeated video frame shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * frames * height * width));
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
            const auto * src = values.data() + static_cast<size_t>(c * height * width);
            auto * dst = out.data() + static_cast<size_t>(((c * frames + t) * height) * width);
            std::copy(src, src + height * width, dst);
        }
    }
    return out;
}

std::vector<float> slice_video_spatial(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t y_start,
    int64_t tile_height,
    int64_t x_start,
    int64_t tile_width) {
    if (channels <= 0 || frames <= 0 || height <= 0 || width <= 0 ||
        y_start < 0 || x_start < 0 || tile_height <= 0 || tile_width <= 0 ||
        y_start + tile_height > height || x_start + tile_width > width ||
        static_cast<int64_t>(values.size()) != channels * frames * height * width) {
        throw std::runtime_error("LiveAvatar spatial slice shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * frames * tile_height * tile_width));
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t y = 0; y < tile_height; ++y) {
                const auto * src = values.data() +
                    static_cast<size_t>(((c * frames + t) * height + y_start + y) * width + x_start);
                auto * dst = out.data() +
                    static_cast<size_t>(((c * frames + t) * tile_height + y) * tile_width);
                std::copy(src, src + tile_width, dst);
            }
        }
    }
    return out;
}

void append_video_time(
    std::vector<float> & dst,
    int64_t channels,
    int64_t current_frames,
    int64_t append_frames,
    int64_t height,
    int64_t width,
    const std::vector<float> & values) {
    if (append_frames <= 0 ||
        static_cast<int64_t>(dst.size()) != channels * current_frames * height * width ||
        static_cast<int64_t>(values.size()) != channels * append_frames * height * width) {
        throw std::runtime_error("LiveAvatar video concat shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(channels * (current_frames + append_frames) * height * width));
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < current_frames; ++t) {
            const auto * src = dst.data() + static_cast<size_t>(((c * current_frames + t) * height) * width);
            auto * target = out.data() + static_cast<size_t>(((c * (current_frames + append_frames) + t) * height) * width);
            std::copy(src, src + height * width, target);
        }
        for (int64_t t = 0; t < append_frames; ++t) {
            const auto * src = values.data() + static_cast<size_t>(((c * append_frames + t) * height) * width);
            auto * target = out.data() + static_cast<size_t>(((c * (current_frames + append_frames) + current_frames + t) * height) * width);
            std::copy(src, src + height * width, target);
        }
    }
    dst = std::move(out);
}

void copy_video_time(
    std::vector<float> & dst,
    int64_t channels,
    int64_t dst_frames,
    int64_t height,
    int64_t width,
    int64_t dst_frame_start,
    const std::vector<float> & values,
    int64_t value_frames) {
    if (channels <= 0 || dst_frames <= 0 || height <= 0 || width <= 0 ||
        dst_frame_start < 0 || value_frames <= 0 || dst_frame_start + value_frames > dst_frames ||
        static_cast<int64_t>(dst.size()) != channels * dst_frames * height * width ||
        static_cast<int64_t>(values.size()) != channels * value_frames * height * width) {
        throw std::runtime_error("LiveAvatar temporal copy shape mismatch");
    }
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t t = 0; t < value_frames; ++t) {
            const auto * src = values.data() + static_cast<size_t>(((c * value_frames + t) * height) * width);
            auto * target = dst.data() +
                static_cast<size_t>(((c * dst_frames + dst_frame_start + t) * height) * width);
            std::copy(src, src + height * width, target);
        }
    }
}

std::vector<float> concat_video_time(
    const std::vector<float> & lhs,
    int64_t channels,
    int64_t lhs_frames,
    int64_t height,
    int64_t width,
    const std::vector<float> & rhs,
    int64_t rhs_frames) {
    if (channels <= 0 || lhs_frames <= 0 || rhs_frames <= 0 || height <= 0 || width <= 0 ||
        static_cast<int64_t>(lhs.size()) != channels * lhs_frames * height * width ||
        static_cast<int64_t>(rhs.size()) != channels * rhs_frames * height * width) {
        throw std::runtime_error("LiveAvatar video concat shape mismatch");
    }
    auto out = lhs;
    append_video_time(out, channels, lhs_frames, rhs_frames, height, width, rhs);
    return out;
}

std::vector<int32_t> tokenize_prompt(
    const std::vector<engine::tokenizers::SentencePiecePiece> & pieces,
    const std::string & text,
    int64_t text_len,
    int64_t * token_count) {
    if (text_len <= 0) {
        throw std::runtime_error("LiveAvatar text length must be positive");
    }
    auto ids = engine::tokenizers::tokenize_sentencepiece(pieces, text);
    ids.push_back(kT5EosTokenId);
    if (static_cast<int64_t>(ids.size()) > text_len) {
        ids.resize(static_cast<size_t>(text_len));
        ids.back() = kT5EosTokenId;
    }
    if (token_count != nullptr) {
        *token_count = static_cast<int64_t>(ids.size());
    }
    ids.resize(static_cast<size_t>(text_len), kT5PadTokenId);
    return ids;
}

std::vector<float> make_flow_sigmas(int64_t steps, float shift) {
    if (steps <= 0 || shift <= 0.0F) {
        throw std::runtime_error("LiveAvatar flow schedule requires positive steps and shift");
    }

    const int64_t schedule_steps = steps + 1;
    std::vector<float> sigmas;
    sigmas.reserve(static_cast<size_t>(schedule_steps + 1));
    const double stride = 1000.0 / static_cast<double>(schedule_steps);
    for (int64_t i = 0; i < schedule_steps; ++i) {
        const int64_t index = 999 - static_cast<int64_t>(i * stride);
        const float t = static_cast<float>(index + 1) / 1000.0F;
        sigmas.push_back(shift * t / (1.0F + (shift - 1.0F) * t));
    }
    sigmas.push_back(0.0F);
    sigmas.erase(sigmas.end() - 2);
    return sigmas;
}

float unipc_alpha(float sigma) {
    return 1.0F / std::sqrt(1.0F + sigma * sigma);
}

float unipc_std(float sigma) {
    return sigma / std::sqrt(1.0F + sigma * sigma);
}

void scale_values(std::vector<float> & values, float scale) {
    for (float & value : values) {
        value *= scale;
    }
}

void flow_euler_step_in_place(
    std::vector<float> & sample,
    const std::vector<float> & model_output,
    float sigma,
    float sigma_next) {
    if (sample.size() != model_output.size()) {
        throw std::runtime_error("LiveAvatar FlowMatch Euler step shape mismatch");
    }
    const float dt = sigma_next - sigma;
    for (size_t i = 0; i < sample.size(); ++i) {
        sample[i] += model_output[i] * dt;
    }
}

void apply_wan21_latent_process_in(std::vector<float> & values, const engine::core::TensorShape & shape) {
    if (shape.rank != 4 || shape.dims[0] != static_cast<int64_t>(kWan21LatentsMean.size()) ||
        static_cast<int64_t>(values.size()) != shape.num_elements()) {
        throw std::runtime_error("LiveAvatar latent process_in shape mismatch");
    }
    const int64_t channel_stride = shape.dims[1] * shape.dims[2] * shape.dims[3];
    for (int64_t c = 0; c < shape.dims[0]; ++c) {
        const float mean = kWan21LatentsMean[static_cast<size_t>(c)];
        const float std = kWan21LatentsStd[static_cast<size_t>(c)];
        const size_t begin = static_cast<size_t>(c * channel_stride);
        const size_t end = begin + static_cast<size_t>(channel_stride);
        for (size_t i = begin; i < end; ++i) {
            values[i] = (values[i] - mean) / std;
        }
    }
}

void apply_wan21_latent_process_out(std::vector<float> & values, const engine::core::TensorShape & shape) {
    if (shape.rank != 4 || shape.dims[0] != static_cast<int64_t>(kWan21LatentsMean.size()) ||
        static_cast<int64_t>(values.size()) != shape.num_elements()) {
        throw std::runtime_error("LiveAvatar latent process_out shape mismatch");
    }
    const int64_t channel_stride = shape.dims[1] * shape.dims[2] * shape.dims[3];
    for (int64_t c = 0; c < shape.dims[0]; ++c) {
        const float mean = kWan21LatentsMean[static_cast<size_t>(c)];
        const float std = kWan21LatentsStd[static_cast<size_t>(c)];
        const size_t begin = static_cast<size_t>(c * channel_stride);
        const size_t end = begin + static_cast<size_t>(channel_stride);
        for (size_t i = begin; i < end; ++i) {
            values[i] = values[i] * std + mean;
        }
    }
}

std::vector<float> unipc_bh2_order1_step(
    const std::vector<float> & sample,
    const std::vector<float> & model_prev,
    const std::vector<float> * model_current,
    float sigma_prev,
    float sigma) {
    if (sample.size() != model_prev.size() || (model_current != nullptr && sample.size() != model_current->size())) {
        throw std::runtime_error("LiveAvatar UniPC update shape mismatch");
    }
    const float alpha_t = unipc_alpha(sigma);
    const float h =
        (std::log(unipc_alpha(sigma)) - std::log(unipc_std(sigma))) -
        (std::log(unipc_alpha(sigma_prev)) - std::log(unipc_std(sigma_prev)));
    const float hh = -h;
    const float h_phi_1 = std::expm1(hh);
    const float b_h = std::expm1(hh);
    std::vector<float> out(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        out[i] = (unipc_std(sigma) / unipc_std(sigma_prev)) * sample[i] - alpha_t * h_phi_1 * model_prev[i];
    }
    if (model_current != nullptr) {
        const float correction = alpha_t * b_h * 0.5F;
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] -= correction * ((*model_current)[i] - model_prev[i]);
        }
    }
    return out;
}

std::vector<float> combine_cfg(
    const std::vector<float> & cond,
    const std::vector<float> & uncond,
    float guidance_scale) {
    if (cond.size() != uncond.size()) {
        throw std::runtime_error("LiveAvatar CFG output shape mismatch");
    }
    std::vector<float> out(cond.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = uncond[i] + guidance_scale * (cond[i] - uncond[i]);
    }
    return out;
}

std::vector<float> make_initial_noise(const engine::core::TensorShape & shape, uint64_t seed) {
    return engine::sampling::generate_torch_cuda_randn(static_cast<size_t>(shape.num_elements()), seed);
}

std::vector<std::byte> video_to_rgb24(
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t start_frame,
    int64_t output_frames) {
    if (channels != 3 || frames <= 0 || height <= 0 || width <= 0 ||
        start_frame < 0 || output_frames <= 0 || start_frame + output_frames > frames ||
        static_cast<int64_t>(values.size()) != channels * frames * height * width) {
        throw std::runtime_error("LiveAvatar decoded video shape mismatch");
    }
    std::vector<std::byte> out(static_cast<size_t>(output_frames * height * width * 3));
    for (int64_t t = 0; t < output_frames; ++t) {
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                const int64_t dst = ((t * height + y) * width + x) * 3;
                for (int64_t c = 0; c < 3; ++c) {
                    const int64_t src = ((c * frames + start_frame + t) * height + y) * width + x;
                    const float normalized = std::clamp((values[static_cast<size_t>(src)] + 1.0F) * 0.5F, 0.0F, 1.0F);
                    out[static_cast<size_t>(dst + c)] =
                        static_cast<std::byte>(static_cast<uint8_t>(std::lround(normalized * 255.0F)));
                }
            }
        }
    }
    return out;
}

bool same_shape(const engine::core::TensorShape & lhs, const engine::core::TensorShape & rhs) noexcept {
    if (lhs.rank != rhs.rank) {
        return false;
    }
    for (size_t i = 0; i < lhs.rank; ++i) {
        if (lhs.dims[i] != rhs.dims[i]) {
            return false;
        }
    }
    return true;
}

struct LiveAvatarPipelineState::Data {
    Data(
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::core::ExecutionContext & execution,
        bool denoiser_weight_streaming)
        : assets_(require_assets(std::move(assets))),
          text_config(liveavatar_text_encoder_config()) {
        if (assets_->config.text_dim != text_config.hidden_size) {
            throw std::runtime_error("LiveAvatar text encoder config does not match model config");
        }
        audio_binding = {};
        audio_binding.feature_extractor_layers = "wav2vec2.feature_extractor.conv_layers";
        audio_binding.feature_projection_layer_norm = "wav2vec2.feature_projection.layer_norm";
        audio_binding.feature_projection_projection = "wav2vec2.feature_projection.projection";
        audio_binding.positional_conv = "wav2vec2.encoder.pos_conv_embed.conv";
        audio_binding.encoder_layer_norm = "wav2vec2.encoder.layer_norm";
        audio_binding.encoder_layers = "wav2vec2.encoder.layers";
        audio_config = {};
        audio_config.record_final_layer_after_final_norm = true;
        audio_encoder = engine::modules::HubertEncoderComponent::load_from_tensor_source(
            assets_->audio_encoder_weights,
            execution.config(),
            audio_config,
            audio_binding);
        assets_->audio_encoder_weights->release_storage();
        text_encoder = std::make_unique<LiveAvatarTextEncoderRuntime>(execution, assets_, text_config);
        vae = std::make_unique<LiveAvatarVAERuntime>(assets_);
        denoiser = std::make_unique<LiveAvatarDenoiserRuntime>(assets_, denoiser_weight_streaming);
    }

    void release_vae_weights(engine::core::ExecutionContext & execution) {
        vae->release(execution);
    }

    void release_denoiser(engine::core::ExecutionContext & execution) {
        denoiser->release(execution);
    }

    std::shared_ptr<const LiveAvatarAssets> assets_;
    engine::modules::T5BaseEncoderConfig text_config;
    engine::modules::HubertEncoderConfig audio_config;
    engine::modules::HubertEncoderWeightBinding audio_binding;
    engine::modules::HubertEncoderComponent audio_encoder;
    std::unique_ptr<LiveAvatarTextEncoderRuntime> text_encoder;
    std::unique_ptr<LiveAvatarVAERuntime> vae;
    std::unique_ptr<LiveAvatarDenoiserRuntime> denoiser;
};

LiveAvatarPipelineState::LiveAvatarPipelineState(
    std::shared_ptr<const LiveAvatarAssets> assets,
    engine::core::ExecutionContext & execution,
    bool denoiser_weight_streaming)
    : data_(std::make_unique<Data>(std::move(assets), execution, denoiser_weight_streaming)) {}

LiveAvatarPipelineState::~LiveAvatarPipelineState() = default;

std::vector<std::vector<float>> LiveAvatarPipelineState::encode_text_batch(
    engine::core::ExecutionContext & execution,
    const std::vector<std::vector<int32_t>> & input_ids,
    const std::vector<int64_t> & token_counts) {
    if (!data_->text_encoder) {
        data_->text_encoder = std::make_unique<LiveAvatarTextEncoderRuntime>(
            execution,
            data_->assets_,
            data_->text_config);
    }
    const auto out = data_->text_encoder->encode_batch(input_ids, token_counts);
    data_->text_encoder.reset();
    return out;
}

LiveAvatarPreparedAudio LiveAvatarPipelineState::prepare_audio_buckets(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & audio_hubert_input,
    int64_t batch_frames,
    int64_t audio_layers) {
    if (data_->audio_encoder.weights() == nullptr) {
        data_->audio_encoder = engine::modules::HubertEncoderComponent::load_from_tensor_source(
            data_->assets_->audio_encoder_weights,
            execution.config(),
            data_->audio_config,
            data_->audio_binding);
        data_->assets_->audio_encoder_weights->release_storage();
    }
    WanS2VAudioConditionerConfig audio_config;
    audio_config.batch_frames = batch_frames;
    std::vector<int64_t> layers(static_cast<size_t>(audio_layers));
    std::iota(layers.begin(), layers.end(), 0);
    const auto hubert_layers = data_->audio_encoder.encode_layers(
        audio_hubert_input,
        1,
        static_cast<int64_t>(audio_hubert_input.size()),
        layers);
    LiveAvatarPreparedAudio prepared;
    prepared.hubert_layer_stack.reserve(static_cast<size_t>(
        hubert_layers.hidden_states.size() * hubert_layers.tokens * hubert_layers.hidden_size));
    for (const auto & layer : hubert_layers.hidden_states) {
        prepared.hubert_layer_stack.insert(prepared.hubert_layer_stack.end(), layer.begin(), layer.end());
    }
    prepared.buckets = wan_s2v_prepare_audio_encoder_output(
        wan_s2v_audio_feature_from_hubert_layers(hubert_layers),
        audio_config);
    data_->audio_encoder.release_runtime_graph();
    data_->audio_encoder = engine::modules::HubertEncoderComponent{};
    return prepared;
}

LiveAvatarDenoiserWeights & LiveAvatarPipelineState::ensure_denoiser_weights(engine::core::ExecutionContext & execution) {
    return data_->denoiser->ensure_weights(execution);
}

void LiveAvatarPipelineState::release_vae_weights(engine::core::ExecutionContext & execution) {
    data_->release_vae_weights(execution);
}

void LiveAvatarPipelineState::release_denoiser(engine::core::ExecutionContext & execution) {
    data_->release_denoiser(execution);
}

void LiveAvatarPipelineState::release_denoiser_condition_graph() {
    data_->denoiser->release_condition_graph();
}

LiveAvatarDenoiserPreparedCondition LiveAvatarPipelineState::prepare_denoiser_condition(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserConditionRunInput & input,
    int64_t target_frames,
    int64_t audio_frames,
    int64_t latent_audio_start_frame) {
    return data_->denoiser->prepare_condition(execution, input, target_frames, audio_frames, latent_audio_start_frame);
}

LiveAvatarDenoiserStaticCache & LiveAvatarPipelineState::prepare_denoiser_static_cache(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserPreparedCondition & condition,
    const engine::core::TensorShape & latent_shape,
    int64_t lanes,
    bool use_sage_attention) {
    return data_->denoiser->prepare_static_cache(execution, condition, latent_shape, lanes, use_sage_attention);
}

std::vector<float> LiveAvatarPipelineState::denoise(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserRunInput & input,
    const engine::core::TensorShape & latent_shape,
    bool cfg_enabled,
    bool use_sage_attention,
    const LiveAvatarDenoiserStaticCache * static_cache) {
    return data_->denoiser->denoise(
        execution,
        input,
        latent_shape,
        cfg_enabled,
        use_sage_attention,
        static_cache);
}

std::vector<float> LiveAvatarPipelineState::denoise_layerwise(
    engine::core::ExecutionContext & execution,
    const LiveAvatarDenoiserRunInput & input,
    const engine::core::TensorShape & latent_shape,
    bool use_sage_attention,
    const LiveAvatarDenoiserStaticCache & static_cache,
    int64_t layer_batch) {
    return data_->denoiser->denoise_layerwise(execution, input, latent_shape, use_sage_attention, static_cache, layer_batch);
}

std::vector<float> LiveAvatarPipelineState::vae_encode_raw(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    engine::core::TensorShape * output_shape) {
    return data_->vae->encode_raw(execution, values, channels, frames, height, width, output_shape);
}

std::vector<float> LiveAvatarPipelineState::vae_encode_cached(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t chunk_size,
    bool cache_f16,
    engine::core::TensorShape * output_shape) {
    return data_->vae->encode_cached(execution, values, channels, frames, height, width, chunk_size, cache_f16, output_shape);
}

std::vector<float> LiveAvatarPipelineState::vae_decode(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t tile_size,
    engine::core::TensorShape * output_shape) {
    return data_->vae->decode(execution, values, channels, frames, height, width, tile_size, output_shape);
}

LiveAvatarGenerateShared prepare_liveavatar_generate_shared(
    LiveAvatarPipelineState & state,
    engine::core::ExecutionContext & execution,
    const std::shared_ptr<const LiveAvatarAssets> & assets,
    const LiveAvatarGenerateRequest & request) {
    auto * impl_ = &state;
    const auto * assets_ = assets.get();
    auto & execution_ = execution;
    const auto total_start = Clock::now();
    if (request.prompt.empty()) {
        throw std::runtime_error("LiveAvatar requires a non-empty prompt");
    }
    if (request.reference_image_path.empty()) {
        throw std::runtime_error("LiveAvatar requires reference_image_path");
    }
    if (request.audio.samples.empty()) {
        throw std::runtime_error("LiveAvatar requires audio input");
    }
    if (request.height <= 0 || request.width <= 0 || request.height % 16 != 0 || request.width % 16 != 0) {
        throw std::runtime_error("LiveAvatar height and width must be positive multiples of 16");
    }
    if (request.frames_per_clip <= 0) {
        throw std::runtime_error("LiveAvatar num_frames must be positive");
    }
    if (request.blockwise_generation && request.frames_per_clip % 4 != 0) {
        throw std::runtime_error("LiveAvatar num_frames must be a multiple of 4 in blockwise mode");
    }
    if (request.max_clips <= 0) {
        throw std::runtime_error("LiveAvatar num_clips must be positive");
    }
    if (request.denoiser_layerwise_batch <= 0) {
        throw std::runtime_error("LiveAvatar denoiser_layerwise_batch must be positive");
    }
    if (request.vae_encoder_chunk_size <= 0) {
        throw std::runtime_error("LiveAvatar vae_encoder_chunk_size must be positive");
    }
    if (request.target_cache_blocks < 0) {
        throw std::runtime_error("LiveAvatar target_cache_blocks must be non-negative");
    }
    if (!request.blockwise_generation && request.denoiser_layerwise && (!request.fused_cfg || request.guidance_scale <= 1.0F)) {
        throw std::runtime_error("LiveAvatar denoiser_layerwise requires fused_cfg=true and guidance_scale > 1");
    }
    if (request.blockwise_generation && request.guidance_scale != 0.0F) {
        throw std::runtime_error("LiveAvatar blockwise mode requires guidance_scale=0");
    }
    engine::debug::trace_log_scalar("liveavatar.input.reference_image_path", request.reference_image_path);
    engine::debug::trace_log_scalar("liveavatar.input.audio_sample_rate", request.audio.sample_rate);
    engine::debug::trace_log_scalar("liveavatar.input.audio_channels", request.audio.channels);
    engine::debug::trace_log_scalar("liveavatar.input.audio_samples", static_cast<int64_t>(request.audio.samples.size()));
    engine::debug::trace_log_scalar("liveavatar.input.seed", request.seed);
    engine::debug::trace_log_scalar("liveavatar.input.fused_cfg", request.fused_cfg);
    engine::debug::trace_log_scalar("liveavatar.input.sage_attention", request.sage_attention);
    engine::debug::trace_log_scalar("liveavatar.input.memory_saver", request.memory_saver);
    engine::debug::trace_log_scalar("liveavatar.input.denoiser_layerwise", request.denoiser_layerwise);
    engine::debug::trace_log_scalar("liveavatar.input.denoiser_layerwise_batch", request.denoiser_layerwise_batch);
    engine::debug::trace_log_scalar("liveavatar.input.vae_encoder_chunk_size", request.vae_encoder_chunk_size);
    engine::debug::trace_log_scalar("liveavatar.input.target_cache_blocks", request.target_cache_blocks);
    engine::debug::trace_log_scalar("liveavatar.input.vae_cache_f16", request.vae_cache_f16);

    int64_t token_count = 0;
    const auto input_ids = tokenize_prompt(
        assets_->tokenizer_pieces,
        request.prompt,
        assets_->config.text_len,
        &token_count);
    int64_t negative_token_count = 0;
    const std::string negative_prompt =
        request.negative_prompt.empty() ? std::string(kDefaultNegativePrompt) : request.negative_prompt;
    const auto negative_ids = tokenize_prompt(
        assets_->tokenizer_pieces,
        negative_prompt,
        assets_->config.text_len,
        &negative_token_count);

    const auto text_start = Clock::now();
    const auto text_contexts = impl_->encode_text_batch(execution_, {input_ids, negative_ids}, {token_count, negative_token_count});
    const auto text_context = text_contexts[0];
    const auto negative_context = text_contexts[1];
    engine::core::trim_backend_pools(execution_.backend());
    engine::debug::timing_log_scalar("liveavatar.text_encoder_ms", engine::debug::elapsed_ms(text_start));

    const auto audio_start = Clock::now();
    auto audio_mono_16k = engine::audio::mixdown_interleaved_to_mono_average(
        request.audio.samples,
        request.audio.channels);
    if (request.audio.sample_rate != 16000) {
        audio_mono_16k = engine::audio::resample_mono_torchaudio_sinc_hann(
            audio_mono_16k,
            request.audio.sample_rate,
            16000);
    }
    const auto audio_hubert_input = normalize_wav2vec2_input(audio_mono_16k);
    const int64_t latent_target_frames = (request.frames_per_clip - 1) / 4 + 1;
    WanS2VAudioConditionerConfig audio_config;
    audio_config.batch_frames = latent_target_frames * 4;
    const auto prepared_audio = impl_->prepare_audio_buckets(
        execution_,
        audio_hubert_input,
        audio_config.batch_frames,
        assets_->config.audio_layers);
    engine::core::trim_backend_pools(execution_.backend());
    engine::debug::timing_log_scalar("liveavatar.audio_encoder_ms", engine::debug::elapsed_ms(audio_start));

    const auto image_start = Clock::now();
    const auto reference_image_input = resize_center_crop_bilinear_to_vae_input(
        load_rgb_image(request.reference_image_path),
        request.height,
        request.width);
    engine::debug::timing_log_scalar("liveavatar.reference_image_ms", engine::debug::elapsed_ms(image_start));

    engine::core::TensorShape ref_latents_shape;
    const auto ref_vae_start = Clock::now();
    auto ref_latents = impl_->vae_encode_raw(
        execution_,
        reference_image_input,
        3,
        1,
        request.height,
        request.width,
        &ref_latents_shape);
    apply_wan21_latent_process_in(ref_latents, ref_latents_shape);
    engine::debug::timing_log_scalar("liveavatar.vae_encode_ref_ms", engine::debug::elapsed_ms(ref_vae_start));

    LiveAvatarGenerateShared shared{
        request,
        total_start,
        text_context,
        negative_context,
        reference_image_input,
        std::move(ref_latents),
        audio_config,
        prepared_audio.buckets,
        latent_target_frames,
    };

    return shared;
}

}  // namespace engine::community_models::liveavatar
