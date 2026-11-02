#include "engine/community_models/soulx_flashhead/pipeline.h"
#include "engine/community_models/soulx_flashhead/dit.h"
#include "engine/community_models/soulx_flashhead/audio.h"
#include "engine/community_models/soulx_flashhead/color.h"
#include "engine/community_models/soulx_flashhead/portrait.h"
#include "engine/community_models/soulx_flashhead/vae.h"
#include "engine/framework/modules/speech_encoders/wav2vec2_encoder.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/framework/debug/profiler.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::community_models::soulx_flashhead {
struct Pipeline::State {
    core::ExecutionContext & execution;
    std::shared_ptr<const Assets> assets;
    Denoiser dit;
    VideoVae vae;
    AudioConditioner audio;
    ColorMatcher colors;
    GenerateOptions options;
    std::vector<float> portrait, reference, motion, ring, pending, video;
    uint64_t random_offset = 0;
    int64_t index = 0;
    bool active = false;
    sampling::TorchCudaSamplingPolicy random_policy;
    State(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets, bool cuda_cudnn)
        : execution(execution), assets(std::move(assets)), dit(execution, *this->assets), vae(execution, *this->assets, cuda_cudnn),
          audio(execution.config(), *this->assets), colors(execution) {
        core::set_backend_round_bf16_fusion(execution.backend(), true);
        random_policy = sampling::resolve_torch_cuda_sampling_policy(
            execution.backend_type(), execution.config().device, "soulx_flashhead", "SoulX-FlashHead");
    }
    std::vector<float> noise(size_t count) {
        const auto started = std::chrono::steady_clock::now();
        auto values = sampling::generate_torch_cuda_tensor_iterator_randn(
            count, options.seed, random_offset, random_policy, sampling::TorchRandnPrecision::BFloat16);
        random_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, random_policy);
        debug::timing_log_scalar("soulx_flashhead.noise.wall_ms", debug::elapsed_ms(started));
        return values;
    }
    VideoChunk generate(const std::vector<float> & samples) {
        const auto started = std::chrono::steady_clock::now();
        std::rotate(ring.begin(), ring.begin() + static_cast<std::ptrdiff_t>(samples.size()), ring.end());
        std::copy(samples.begin(), samples.end(), ring.end() - static_cast<std::ptrdiff_t>(samples.size()));
        const auto audio_started = std::chrono::steady_clock::now();
        auto context = audio.encode(ring);
        debug::timing_log_scalar("soulx_flashhead.audio.wall_ms", debug::elapsed_ms(audio_started));
        core::round_f32_to_bf16_in_place(context);
        auto z = noise(reference.size());
        const auto & c = assets->config;
        const int64_t latent_frames = 32 / c.stride[0] + 1, lh = options.height / c.stride[1],
                      lw = options.width / c.stride[2];
        const int64_t motion_frames = c.stride[0] + 1;
        const int64_t motion_latents = motion.size() / (c.latent_channels * lh * lw);
        std::vector<float> times;
        for (int64_t i = 0; i < options.steps; ++i) {
            const float t = options.steps == 2   ? 1000 - i * 500
                            : options.steps == 4 ? 1000 - i * 250
                                                 : 1000 - i * 999.F / std::max<int64_t>(1, options.steps - 1);
            const float u = t / 1000;
            times.push_back(5000 * u / (1 + 4 * u));
        }
        times.push_back(0);
        const auto condition = [&]() {
            for (int64_t ch = 0; ch < c.latent_channels; ++ch)
                std::copy_n(motion.begin() + ch * motion_latents * lh * lw,
                            static_cast<size_t>(motion_latents * lh * lw), z.begin() + ch * latent_frames * lh * lw);
        };
        if (c.variant != "pretrained")
            dit.prepare_audio(context, lh, lw);
        for (int64_t step = 0; step < options.steps; ++step) {
            condition();
            if (c.variant == "pretrained")
                dit.prepare_audio(context, lh, lw);
            const auto flow = dit.predict(z, reference, times[static_cast<size_t>(step)], lh, lw);
            if (c.variant == "pretrained") {
                dit.prepare_audio(std::vector<float>(context.size(), 0), lh, lw);
                const auto unconditioned = dit.predict(z, reference, times[static_cast<size_t>(step)], lh, lw);
                std::vector<float> dt{(times[static_cast<size_t>(step)] - times[static_cast<size_t>(step + 1)]) / 1000};
                core::round_f32_to_bf16_in_place(dt);
                auto guided = flow;
                for (size_t j = 0; j < z.size(); ++j)
                    guided[j] -= unconditioned[j];
                core::round_f32_to_bf16_in_place(guided);
                for (auto & value : guided)
                    value *= options.audio_guidance_scale;
                core::round_f32_to_bf16_in_place(guided);
                for (size_t j = 0; j < z.size(); ++j)
                    guided[j] += unconditioned[j];
                core::round_f32_to_bf16_in_place(guided);
                for (auto & value : guided)
                    value *= dt[0];
                core::round_f32_to_bf16_in_place(guided);
                for (size_t j = 0; j < z.size(); ++j)
                    z[j] -= guided[j];
                core::round_f32_to_bf16_in_place(z);
                continue;
            }
            auto epsilon = noise(z.size());
            const auto update_started = std::chrono::steady_clock::now();
            std::vector<float> t{times[static_cast<size_t>(step)] / 1000, times[static_cast<size_t>(step + 1)] / 1000};
            core::round_f32_to_bf16_in_place(t);
            std::vector<float> prediction(z.size());
            for (size_t j = 0; j < z.size(); ++j)
                prediction[j] = flow[j] * t[0];
            core::round_f32_to_bf16_in_place(prediction);
            for (size_t j = 0; j < z.size(); ++j)
                prediction[j] = z[j] - prediction[j];
            core::round_f32_to_bf16_in_place(prediction);
            std::vector<float> next_scale{1 - t[1]};
            core::round_f32_to_bf16_in_place(next_scale);
            for (size_t j = 0; j < z.size(); ++j) {
                prediction[j] *= next_scale[0];
                epsilon[j] *= t[1];
            }
            core::round_f32_to_bf16_in_place(prediction);
            core::round_f32_to_bf16_in_place(epsilon);
            for (size_t j = 0; j < z.size(); ++j)
                z[j] = prediction[j] + epsilon[j];
            core::round_f32_to_bf16_in_place(z);
            debug::timing_log_scalar("soulx_flashhead.denoise_update.wall_ms", debug::elapsed_ms(update_started));
        }
        condition();
        const auto decoded = vae.decode_device(z, latent_frames, lh, lw);
        const size_t pixels = static_cast<size_t>(options.height * options.width);
        const auto corrected = colors.match_device(decoded, portrait, options.height, options.width,
                                                     options.color_strength, motion_frames);
        // Copy the small motion latents to the host; video pixels stay on the backend until delivery.
        motion = vae.encode_device(corrected.motion, motion_frames, options.height, options.width);
        const auto read_started = std::chrono::steady_clock::now();
        const size_t output_frames = static_cast<size_t>(33 - motion_frames);
        const size_t channel_values = output_frames * pixels;
        video.resize(3 * channel_values);
        for (size_t channel = 0; channel < 3; ++channel)
            ggml_backend_tensor_get(corrected.video.tensor, video.data() + channel * channel_values,
                                    channel * channel_values * sizeof(float),
                                    channel_values * sizeof(float));
        debug::timing_log_scalar("soulx_flashhead.video.output_read_ms", debug::elapsed_ms(read_started));
        VideoChunk out{options.height, options.width, 33 - motion_frames, index++, {}};
        out.rgb.resize(static_cast<size_t>(out.frames) * pixels * 3);
        const auto pack_started = std::chrono::steady_clock::now();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(execution.config().threads) if(execution.config().threads > 1) schedule(static)
#endif
        for (int64_t t = 0; t < out.frames; ++t)
            for (size_t p = 0; p < pixels; ++p)
                for (size_t ch = 0; ch < 3; ++ch) {
                    const auto value = video[(ch * output_frames + t) * pixels + p];
                    out.rgb[(t * pixels + p) * 3 + ch] =
                        static_cast<std::byte>(static_cast<uint8_t>(std::clamp((value + 1) * 127.5F, 0.F, 255.F)));
                }
        debug::timing_log_scalar("soulx_flashhead.video.pack_ms", debug::elapsed_ms(pack_started));
        debug::timing_log_scalar("soulx_flashhead.chunk.wall_ms", debug::elapsed_ms(started));
        return out;
    }
};
Pipeline::Pipeline(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets, bool cuda_cudnn)
    : state_(std::make_unique<State>(execution, std::move(assets), cuda_cudnn)) {}
Pipeline::~Pipeline() = default;
void Pipeline::begin(const GenerateOptions & options) {
    reset();
    auto & s = *state_;
    const auto & c = s.assets->config;
    if (options.height <= 0 || options.width <= 0 || options.height % c.stride[1] || options.width % c.stride[2] ||
        options.steps < 1 || !std::isfinite(options.audio_guidance_scale) || options.audio_guidance_scale < 0 ||
        !std::isfinite(options.color_strength) || options.color_strength < 0 || options.color_strength > 1)
        throw std::runtime_error("Invalid FlashHead generation options");
    s.options = options;
    s.vae.reset(options.seed);
    s.portrait = load_portrait(options.reference_image, options.height, options.width);
    const size_t pixels = static_cast<size_t>(options.height * options.width);
    std::vector<float> repeated(3 * 33 * pixels);
    for (size_t ch = 0; ch < 3; ++ch)
        for (size_t t = 0; t < 33; ++t)
            std::copy_n(s.portrait.begin() + ch * pixels, pixels, repeated.begin() + (ch * 33 + t) * pixels);
    s.reference = s.vae.encode(repeated, 33, options.height, options.width);
    const int64_t lh = options.height / c.stride[1], lw = options.width / c.stride[2], lf = 32 / c.stride[0] + 1;
    s.motion.resize(static_cast<size_t>(c.latent_channels * lh * lw));
    for (int64_t ch = 0; ch < c.latent_channels; ++ch)
        std::copy_n(s.reference.begin() + ch * lf * lh * lw, static_cast<size_t>(lh * lw),
                    s.motion.begin() + ch * lh * lw);
    s.ring.assign(8 * 16000, 0);
    s.active = true;
}
std::vector<VideoChunk> Pipeline::push(const std::vector<float> & mono16k, bool final) {
    auto & s = *state_;
    if (!s.active)
        throw std::runtime_error("FlashHead stream is not active");
    s.pending.insert(s.pending.end(), mono16k.begin(), mono16k.end());
    const size_t block = static_cast<size_t>(32 - s.assets->config.stride[0]) * 16000 / 25;
    if (final && !s.pending.empty())
        s.pending.resize((s.pending.size() + block - 1) / block * block, 0);
    std::vector<VideoChunk> out;
    while (s.pending.size() >= block) {
        std::vector<float> input(s.pending.begin(), s.pending.begin() + static_cast<std::ptrdiff_t>(block));
        out.push_back(s.generate(input));
        s.pending.erase(s.pending.begin(), s.pending.begin() + static_cast<std::ptrdiff_t>(block));
    }
    if (final)
        s.active = false;
    return out;
}
void Pipeline::reset() {
    auto & s = *state_;
    s.active = false;
    s.pending.clear();
    s.ring.clear();
    s.motion.clear();
    s.random_offset = 0;
    s.index = 0;
}
} // namespace engine::community_models::soulx_flashhead
