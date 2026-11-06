// Manual isolated HeartCodec parity; deliberately not registered with CTest.
// Usage: mulacover_codec MODEL REF.safetensors cpu|cuda|vulkan --log LOG
//        [--min-waveform-cosine VALUE] (default: 0.99999)
// MODEL: packaged nine-namespace MuLaCover GGUF.
// Fixture tensors (logical contiguous layouts; no normalization/alignment):
// codec_codes I32[B,8,T] -> conditioning F32[B,2T,512]
// flow_input F32[B,frames,1024], flow_timestep F32[B]
//     -> flow_output F32[B,frames,256]
// scalar_latents F32[B,128,frames] -> scalar_waveform F32[B,1,samples]
// Scalar B=2 already represents separated audio channels. Component batches
// and frame counts are independent. Each API consumes official fixture inputs,
// never the output of another native component. No AR or sampling is involved.
#include "engine/models/mulacover/assets.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/codecs/heartcodec_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Numerical failures are returned so all three boundaries are reported.
// Invalid shapes and non-finite values are fixture/execution errors and throw.
bool compare(const engine::assets::TensorSource & reference, const std::string & name,
             const std::vector<float> & actual, const std::vector<int64_t> & shape,
             double minimum_cosine = 0.99999) {
    const auto expected = reference.require_f32(name, shape);
    if (actual.size() != expected.size()) throw std::runtime_error(name + " payload size mismatch");
    double dot = 0, aa = 0, bb = 0, max_abs = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error(name + " contains non-finite values");
        dot += a * b;
        aa += a * a;
        bb += b * b;
        max_abs = std::max(max_abs, std::abs(a - b));
    }
    // Zero energy has no defined cosine and therefore cannot pass the gate.
    const bool defined = aa > 0 && bb > 0;
    const double cosine = defined ? dot / std::sqrt(aa * bb) : 0;
    const bool passed = defined && cosine > minimum_cosine;
    std::ostringstream metrics;
    metrics << std::setprecision(17) << "validation.tensor=" << name
            << " cosine=" << cosine << " max_abs_error=" << max_abs
            << " minimum_cosine=" << minimum_cosine
            << " cosine_defined=" << (defined ? "true" : "false")
            << " validation.status=" << (passed ? "passed" : "failed");
    engine::debug::log_message(metrics.str());
    std::cout << metrics.str() << '\n';
    return passed;
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    bool logging = false;
    try {
        if ((argc != 6 && argc != 8) || std::string(argv[4]) != "--log" || std::string(argv[5]).empty() ||
            (argc == 8 && std::string(argv[6]) != "--min-waveform-cosine"))
            throw std::runtime_error("Usage: mulacover_codec MODEL REF.safetensors cpu|cuda|vulkan --log LOG [--min-waveform-cosine VALUE]");
        engine::debug::configure_logging({true, argv[5]});
        logging = true;
        double minimum_waveform_cosine = 0.99999;
        if (argc == 8) {
            size_t consumed = 0;
            minimum_waveform_cosine = std::stod(argv[7], &consumed);
            if (consumed != std::strlen(argv[7]) || !std::isfinite(minimum_waveform_cosine) ||
                minimum_waveform_cosine < 0 || minimum_waveform_cosine >= 1)
                throw std::runtime_error("--min-waveform-cosine must be finite and in [0,1)");
        }
        engine::core::BackendConfig backend;
        const std::string backend_name = argv[3];
        if (backend_name == "cpu") backend.type = engine::core::BackendType::Cpu;
        else if (backend_name == "cuda") backend.type = engine::core::BackendType::Cuda;
        else if (backend_name == "vulkan") backend.type = engine::core::BackendType::Vulkan;
        else throw std::runtime_error("Unknown backend: " + backend_name);
        backend.threads = 8;
        engine::debug::log_message("validation.backend=" + backend_name + " validation.weight_type=f32");
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        for (const auto * name : {"conditioning", "flow_input", "flow_timestep", "flow_output", "scalar_latents", "scalar_waveform"}) {
            const auto metadata = reference->require_metadata(name);
            if (metadata.dtype != "F32") throw std::runtime_error(std::string(name) + " must be F32");
        }
        const auto raw_codes = reference->require_tensor_data("codec_codes");
        const auto & code_shape = raw_codes.metadata.shape;
        if (raw_codes.metadata.dtype != "I32" || code_shape.size() != 3 ||
            code_shape[0] <= 0 || code_shape[1] != 8 || code_shape[2] <= 0 ||
            raw_codes.bytes.size() != static_cast<size_t>(code_shape[0] * 8 * code_shape[2]) * sizeof(int32_t))
            throw std::runtime_error("codec_codes must be I32[B,8,T], B,T > 0");
        std::vector<int32_t> codes(raw_codes.bytes.size() / sizeof(int32_t));
        std::memcpy(codes.data(), raw_codes.bytes.data(), raw_codes.bytes.size());
        const auto flow_shape = reference->require_metadata("flow_input").shape;
        if (flow_shape.size() != 3 || flow_shape[0] <= 0 || flow_shape[1] <= 0 || flow_shape[2] != 1024)
            throw std::runtime_error("flow_input must be F32[B,frames,1024], B,frames > 0");
        const auto flow_input = reference->require_f32("flow_input", flow_shape);
        const auto timesteps = reference->require_f32("flow_timestep", {flow_shape[0]});
        const auto scalar_shape = reference->require_metadata("scalar_latents").shape;
        if (scalar_shape.size() != 3 || scalar_shape[0] != 2 || scalar_shape[1] != 128 || scalar_shape[2] <= 0)
            throw std::runtime_error("scalar_latents must be F32[2,128,frames], frames > 0 (channels already separated)");
        const auto latents = reference->require_f32("scalar_latents", scalar_shape);
        for (const auto * values : {&flow_input, &timesteps, &latents})
            for (float value : *values)
                if (!std::isfinite(value)) throw std::runtime_error("Non-finite codec fixture input");

        const auto assets = engine::models::mulacover::load_mulacover_assets(argv[1]);
        engine::core::ExecutionContext execution(backend);
        engine::codecs::HeartCodecRuntime codec(assets->codec_weights, execution, assets->codec_config,
            {8ull << 20, 32ull << 20, 32ull << 20, 32ull << 20,
             engine::assets::TensorStorageType::F32});
        const auto conditioning = codec.build_conditioning(codes, code_shape[0], code_shape[2]);
        if (conditioning.batch_size != code_shape[0] || conditioning.frames != 2 * code_shape[2] || conditioning.channels != 512)
            throw std::runtime_error("Native conditioning must be [B,2T,512]");
        const bool conditioning_passed = compare(*reference, "conditioning", conditioning.values,
            {conditioning.batch_size, conditioning.frames, conditioning.channels});

        const auto flow = codec.estimate_flow(flow_input, flow_shape[0], flow_shape[1], timesteps);
        if (flow.batch_size != flow_shape[0] || flow.frames != flow_shape[1] || flow.channels != 256)
            throw std::runtime_error("Native flow_output must be [B,frames,256]");
        const bool flow_passed = compare(*reference, "flow_output", flow.values,
            {flow.batch_size, flow.frames, flow.channels});

        const auto scalar = codec.decode_scalar_latents(latents, scalar_shape[0], scalar_shape[2]);
        if (scalar.batch_size != scalar_shape[0] || scalar.channels != 1 || scalar.samples <= 0)
            throw std::runtime_error("Native scalar_waveform must be [B,1,samples]");
        const bool scalar_passed = compare(*reference, "scalar_waveform", scalar.values,
            {scalar.batch_size, scalar.channels, scalar.samples});

        // Replay the codec ODE from the official initial noise and conditioning.
        // This isolates flow accumulation from AR, sampling and native codebook
        // lookup. Compare latents before the scalar decoder's discrete rounding.
        if (flow_shape[0] != 2 || flow_shape[1] != scalar_shape[2])
            throw std::runtime_error("Flow replay requires CFG batch 2 and matching scalar frames");
        const auto steps_data = reference->require_tensor_data("flow_steps");
        int64_t steps = 0;
        if (steps_data.metadata.dtype != "I64" || steps_data.bytes.size() != sizeof(steps))
            throw std::runtime_error("flow_steps must be an I64 scalar");
        std::memcpy(&steps, steps_data.bytes.data(), sizeof(steps));
        const float guidance = reference->require_f32("flow_guidance", {})[0];
        if (steps <= 0 || !(guidance > 1)) throw std::runtime_error("Invalid CFG flow replay settings");
        const int64_t frames = flow_shape[1];
        const size_t batch_stride = static_cast<size_t>(frames * 1024);
        const size_t latent_stride = static_cast<size_t>(frames * 256);
        auto replay_input = flow_input;
        std::vector<float> solved(latent_stride);
        for (int64_t frame = 0; frame < frames; ++frame)
            std::copy_n(flow_input.begin() + frame * 1024, 256, solved.begin() + frame * 256);
        float t = 0, dt = 1.0F / static_cast<float>(steps);
        for (int64_t step = 1; step <= steps; ++step) {
            const auto estimate = step == 1 ? flow : codec.estimate_flow(replay_input, 2, frames, {t, t});
            for (int64_t frame = 0; frame < frames; ++frame) {
                for (int64_t channel = 0; channel < 256; ++channel) {
                    const size_t index = static_cast<size_t>(frame * 256 + channel);
                    const float uncond = estimate.values[index], cond = estimate.values[latent_stride + index];
                    solved[index] += dt * (uncond + guidance * (cond - uncond));
                    replay_input[static_cast<size_t>(frame * 1024 + channel)] = solved[index];
                    replay_input[batch_stride + static_cast<size_t>(frame * 1024 + channel)] = solved[index];
                }
            }
            t += dt;
            if (step < steps) dt = static_cast<float>(step + 1) / static_cast<float>(steps) - t;
        }
        std::vector<float> replay_latents(latent_stride);
        size_t different_bins = 0;
        for (int64_t stream = 0; stream < 2; ++stream)
            for (int64_t channel = 0; channel < 128; ++channel)
                for (int64_t frame = 0; frame < frames; ++frame) {
                    const size_t index = static_cast<size_t>((stream * 128 + channel) * frames + frame);
                    replay_latents[index] = solved[static_cast<size_t>(frame * 256 + stream * 128 + channel)];
                    different_bins += std::nearbyint(replay_latents[index] * 9) != std::nearbyint(latents[index] * 9);
                }
        const bool replay_passed = compare(*reference, "scalar_latents", replay_latents, scalar_shape);
        engine::debug::timing_log_scalar("validation.scalar_quantizer.different_bins", different_bins);
        engine::debug::timing_log_scalar("validation.scalar_quantizer.total_bins", replay_latents.size());
        const auto replay_audio = codec.decode_scalar_latents(replay_latents, 2, frames);
        const bool replay_audio_passed = compare(*reference, "scalar_waveform", replay_audio.values,
            {replay_audio.batch_size, replay_audio.channels, replay_audio.samples}, minimum_waveform_cosine);
        const auto repeated_conditioning = codec.build_conditioning(codes, code_shape[0], code_shape[2]);
        if (repeated_conditioning.values != conditioning.values)
            throw std::runtime_error("Codec conditioning changed after flow/scalar workspace reuse");
        engine::debug::log_message("validation.codec_phase_reuse=passed");

        if (!conditioning_passed || !flow_passed || !scalar_passed || !replay_passed || !replay_audio_passed)
            throw std::runtime_error("Codec parity failed (see component and final-waveform metrics)");
        engine::debug::log_message("validation.status=passed");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        if (logging) {
            engine::debug::log_message(engine::debug::LogLevel::Error, "validation", error.what());
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
