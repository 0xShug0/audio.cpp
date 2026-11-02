#include "engine/community_models/soulx_flashhead/assets.h"
#include "engine/community_models/soulx_flashhead/dit.h"
#include "engine/community_models/soulx_flashhead/audio.h"
#include "engine/community_models/soulx_flashhead/vae.h"
#include "engine/community_models/soulx_flashhead/color.h"
#include "engine/community_models/soulx_flashhead/portrait.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

int main(int argc, char ** argv) {
    try {
        if (argc != 6 && argc != 7)
            throw std::runtime_error("Usage: flashhead_components MODEL.gguf REFERENCE_DIR encode|decode|dit "
                                     "cpu|cuda|vulkan LOG [PREPARED_SOURCE]");
        engine::debug::configure_logging({true, argv[5]});
        namespace fh = engine::community_models::soulx_flashhead;
        auto assets = std::make_shared<fh::Assets>(*fh::load_assets(argv[1]));
        if (argc == 7) {
            const auto root = std::filesystem::path(argv[6]);
            std::ifstream stream(root / "config.json");
            const std::string text((std::istreambuf_iterator<char>(stream)), {});
            const auto json = engine::io::json::parse(text);
            assets->config = fh::Config::from_json(json, engine::io::json::require_string(json, "variant"));
            assets->dit = engine::assets::open_tensor_source(root / "dit.safetensors");
            assets->vae = engine::assets::open_tensor_source(root / "vae.safetensors");
            assets->audio = engine::assets::open_tensor_source(root / "audio.safetensors");
        }
        engine::core::BackendConfig config;
        std::string mode = argv[3], backend = argv[4];
        if (backend == "cuda")
            config.type = engine::core::BackendType::Cuda;
        else if (backend == "vulkan")
            config.type = engine::core::BackendType::Vulkan;
        else if (backend == "cpu")
            config.type = engine::core::BackendType::Cpu;
        else
            throw std::runtime_error("Unknown backend");
        config.threads = 8;
        engine::core::ExecutionContext execution(config);
        engine::core::set_backend_round_bf16_fusion(execution.backend(), true);
        const auto source =
            engine::assets::open_tensor_source(std::filesystem::path(argv[2]) / (mode + ".safetensors"));
        auto read = [&](const std::string & name) {
            return source->require_f32(name, source->require_metadata(name).shape);
        };
        std::vector<float> actual;
        const auto started = std::chrono::steady_clock::now();
        if (mode == "portrait") {
            const auto shape = source->require_metadata("output").shape;
            actual = fh::load_portrait(std::filesystem::path(argv[2]) / "portrait.png", shape[1], shape[2]);
            if (actual != read("output"))
                throw std::runtime_error("Portrait normalization differs from upstream");
        } else if (mode == "color") {
            fh::ColorMatcher colors(execution);
            actual = colors.match(read("video"), read("portrait"), 512, 512, read("strength")[0]);
            for (int repeat = 0; repeat < 2; ++repeat) {
                const auto next = colors.match(read("video"), read("portrait"), 512, 512, read("strength")[0]);
                if (next != actual)
                    throw std::runtime_error("Repeated color correction changed identical inputs");
            }
            engine::core::ConstantTensorCache inputs(execution.backend(), config.threads, "color.fixture", 4096);
            const auto video = read("video"), portrait = read("portrait");
            const auto device_video = inputs.make_f32(engine::core::TensorShape::from_dims({3, 33, 512, 512}), video);
            inputs.ensure_uploaded();
            for (const int tail_frames : {9, 5}) {
              for (const float strength : {read("strength")[0], 0.F}) {
                const size_t output_frames = 33 - tail_frames;
                const auto corrected = colors.match_device(device_video, portrait, 512, 512, strength, tail_frames);
                std::vector<float> delivered, motion;
                engine::core::read_tensor_float_into(corrected.video.tensor, delivered);
                engine::core::read_tensor_float_into(corrected.motion.tensor, motion);
                const auto & full = strength == 0 ? video : actual;
                const size_t pixels = 512 * 512;
                if (delivered.size() != 3 * output_frames * pixels || motion.size() != 3 * tail_frames * pixels)
                    throw std::runtime_error("Invalid trimmed color output size");
                for (size_t channel = 0; channel < 3; ++channel) {
                    if (!std::equal(delivered.begin() + channel * output_frames * pixels,
                                    delivered.begin() + (channel + 1) * output_frames * pixels,
                                    full.begin() + (channel * 33 + tail_frames) * pixels) ||
                        !std::equal(motion.begin() + channel * tail_frames * pixels,
                                    motion.begin() + (channel + 1) * tail_frames * pixels,
                                    full.begin() + (channel * 33 + output_frames) * pixels))
                        throw std::runtime_error("Trimming context changed delivered video or motion tail");
                }
              }
            }
        } else if (mode == "audio") {
            fh::AudioConditioner audio(config, *assets);
            actual = audio.encode(read("wave"));
        } else if (mode.rfind("dit", 0) == 0) {
            fh::Denoiser dit(execution, *assets);
            dit.prepare_audio(read("context"), 512 / assets->config.stride[1], 512 / assets->config.stride[2]);
            const auto cold = dit.predict(read("x"), read("y"), read("timestep")[0],
                                          512 / assets->config.stride[1], 512 / assets->config.stride[2]);
            actual = dit.predict(read("x"), read("y"), read("timestep")[0],
                                 512 / assets->config.stride[1], 512 / assets->config.stride[2]);
            if (actual != cold)
                throw std::runtime_error("Repeated DiT execution changed identical inputs");
            actual = dit.predict(read("x"), read("y"), read("timestep")[0],
                                 512 / assets->config.stride[1], 512 / assets->config.stride[2]);
            if (actual != cold)
                throw std::runtime_error("Captured DiT execution changed identical inputs");
        } else {
            fh::VideoVae vae(execution, *assets);
            vae.reset(42);
            if (mode == "encode")
                actual = vae.encode(read("pixels"), 33, 512, 512);
            else if (mode == "decode") {
                actual = vae.decode(read("latents"), 32 / assets->config.stride[0] + 1, 512 / assets->config.stride[1],
                                    512 / assets->config.stride[2]);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto next = vae.decode(read("latents"), 32 / assets->config.stride[0] + 1,
                                                 512 / assets->config.stride[1], 512 / assets->config.stride[2]);
                    if (next != actual)
                        throw std::runtime_error("Repeated VAE decoding changed identical inputs");
                }
            } else
                throw std::runtime_error("Unknown component");
        }
        const auto expected = read("output");
        if (actual.size() != expected.size())
            throw std::runtime_error("Component output size mismatch");
        double dot = 0, aa = 0, bb = 0, mse = 0, max_error = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]))
                throw std::runtime_error("Non-finite component output");
            dot += actual[i] * static_cast<double>(expected[i]);
            aa += actual[i] * static_cast<double>(actual[i]);
            bb += expected[i] * static_cast<double>(expected[i]);
            double error = actual[i] - static_cast<double>(expected[i]);
            mse += error * error;
            max_error = std::max(max_error, std::abs(error));
        }
        const double cosine = dot / std::sqrt(aa * bb);
        std::cout << mode << " cosine=" << cosine << " rmse=" << std::sqrt(mse / actual.size())
                  << " max_abs=" << max_error << '\n';
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        engine::debug::trace_log_f32("soulx_flashhead." + mode + ".actual", {static_cast<int64_t>(actual.size())},
                                     actual);
        engine::debug::trace_log_f32("soulx_flashhead." + mode + ".expected", {static_cast<int64_t>(expected.size())},
                                     expected);
        if (mode == "decode" || mode == "color") {
            double rgb_dot = 0, rgb_aa = 0, rgb_bb = 0, mae = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double a = static_cast<uint8_t>(std::clamp((actual[i] + 1) * 127.5F, 0.F, 255.F));
                const double b = static_cast<uint8_t>(std::clamp((expected[i] + 1) * 127.5F, 0.F, 255.F));
                rgb_dot += a * b;
                rgb_aa += a * a;
                rgb_bb += b * b;
                mae += std::abs(a - b);
            }
            const double rgb_cosine = rgb_dot / std::sqrt(rgb_aa * rgb_bb);
            std::cout << "forced-latent RGB cosine=" << std::setprecision(12) << rgb_cosine
                      << " mean_pixel_error=" << mae / actual.size() << '\n';
            return rgb_cosine > .9999 ? 0 : 1;
        }
        // Video kernels use reduced-precision attention and fused F32 normalization.
        const double minimum_cosine = mode == "audio" ? .99999 : .999;
        return cosine > minimum_cosine ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
