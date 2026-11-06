// Manual teacher-forced final-waveform parity; no CTest registration.
// Usage: mulacover_waveform MODEL REFERENCE.safetensors cpu|cuda|vulkan OUTPUT.wav
//        --log LOG [--weight-type native|f32] [--min-waveform-cosine VALUE]
// Defaults: F32 weights, minimum cosine 0.99999.
// MODEL is the packaged nine-namespace MuLaCover GGUF.
// Fixture: prompt_embeddings F32[B,P,3072], qwen_embedding F32[B,1024],
// pianoroll F32[B,S,256], drum_pianoroll F32[B,S,128], chord F32[B,S,12],
// codes I32[8,T], waveform F32[channels,samples], seed I64 scalar.
// B is 1 or 2, with the conditional row first. Optional codec_randn_offset
// I64 scalar is the captured Philox offset after official sampler calls.
// Sampling defaults: temperature=1, top_k=250, CFG=1.5 for B=2 (1 for B=1);
// codec duration=29.76, steps=10, guidance=1.25. Fixtures must use these settings.
#include "engine/models/mulacover/assets.h"
#include "engine/models/mulacover/conditioning.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/codecs/heartcodec_runtime.h"
#include "engine/models/mulacover/ar.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    bool logging = false;
    try {
        if (argc < 7 || argc % 2 == 0 || std::string(argv[5]) != "--log") {
            throw std::runtime_error("Usage: mulacover_waveform MODEL REFERENCE.safetensors cpu|cuda|vulkan OUTPUT.wav --log LOG [--weight-type native|f32] [--min-waveform-cosine VALUE]");
        }
        if (std::string(argv[6]).empty()) throw std::runtime_error("--log requires a file path");
        engine::debug::configure_logging({true, argv[6]});
        logging = true;
        std::string weight_type = "f32";
        double minimum_waveform_cosine = 0.99999;
        for (int i = 7; i < argc; i += 2) {
            const std::string option = argv[i];
            if (option == "--weight-type") weight_type = argv[i + 1];
            else if (option == "--min-waveform-cosine") {
                size_t consumed = 0;
                minimum_waveform_cosine = std::stod(argv[i + 1], &consumed);
                if (consumed != std::strlen(argv[i + 1]) || !std::isfinite(minimum_waveform_cosine) ||
                    minimum_waveform_cosine < 0 || minimum_waveform_cosine >= 1)
                    throw std::runtime_error("--min-waveform-cosine must be finite and in [0,1)");
            } else throw std::runtime_error("Unknown waveform validation option: " + option);
        }
        engine::assets::TensorStorageType storage;
        if (weight_type == "native") storage = engine::assets::TensorStorageType::Native;
        else if (weight_type == "f32") storage = engine::assets::TensorStorageType::F32;
        else throw std::runtime_error("--weight-type must be native|f32");
        engine::core::BackendConfig backend;
        const std::string backend_name = argv[3];
        if (backend_name == "cpu") backend.type = engine::core::BackendType::Cpu;
        else if (backend_name == "cuda") backend.type = engine::core::BackendType::Cuda;
        else if (backend_name == "vulkan") backend.type = engine::core::BackendType::Vulkan;
        else throw std::runtime_error("Unknown backend: " + backend_name);
        backend.threads = 8;
        engine::debug::log_message("validation.backend=" + backend_name + " validation.weight_type=" + weight_type);
        engine::debug::timing_log_scalar("validation.waveform.minimum_cosine", minimum_waveform_cosine);

        namespace mc = engine::models::mulacover;
        const auto reference = engine::assets::open_tensor_source(argv[2]);
        for (const auto * name : {"prompt_embeddings", "qwen_embedding", "pianoroll", "drum_pianoroll", "chord", "waveform"}) {
            if (reference->require_metadata(name).dtype != "F32") throw std::runtime_error(std::string(name) + " must be F32");
        }
        const auto prompt_shape = reference->require_metadata("prompt_embeddings").shape;
        if (prompt_shape.size() != 3 || (prompt_shape[0] != 1 && prompt_shape[0] != 2) ||
            prompt_shape[1] <= 0 || prompt_shape[2] != 3072) {
            throw std::runtime_error("prompt_embeddings must be [1|2,P,3072], P > 0");
        }
        const int64_t batch = prompt_shape[0];
        mc::PromptEmbeddings prompt;
        prompt.batch_size = batch;
        prompt.steps = prompt_shape[1];
        prompt.dims = prompt_shape[2];
        prompt.values = reference->require_f32("prompt_embeddings", prompt_shape);
        const auto symbolic_shape = reference->require_metadata("pianoroll").shape;
        if (symbolic_shape.size() != 3 || symbolic_shape[0] != batch ||
            symbolic_shape[1] <= 0 || symbolic_shape[2] != 256) {
            throw std::runtime_error("pianoroll must be [B,S,256], S > 0");
        }
        mc::SymbolicInputs symbolic;
        symbolic.length = symbolic_shape[1];
        symbolic.pianoroll = reference->require_f32("pianoroll", {batch, symbolic.length, 256});
        symbolic.drum_pianoroll = reference->require_f32("drum_pianoroll", {batch, symbolic.length, 128});
        symbolic.chord = reference->require_f32("chord", {batch, symbolic.length, 12});
        auto style = reference->require_f32("qwen_embedding", {batch, 1024});
        // prepare creates CFG rows itself, using -1 raw features for the unconditional row.
        symbolic.pianoroll.resize(static_cast<size_t>(symbolic.length * 256));
        symbolic.drum_pianoroll.resize(static_cast<size_t>(symbolic.length * 128));
        symbolic.chord.resize(static_cast<size_t>(symbolic.length * 12));
        style.resize(1024);

        const auto raw_codes = reference->require_tensor_data("codes");
        const auto & code_shape = raw_codes.metadata.shape;
        if (raw_codes.metadata.dtype != "I32" || code_shape.size() != 2 ||
            code_shape[0] != 8 || code_shape[1] <= 0 ||
            raw_codes.bytes.size() != static_cast<size_t>(8 * code_shape[1]) * sizeof(int32_t)) {
            throw std::runtime_error("codes must be I32[8,T], T > 0");
        }
        const int64_t frames = code_shape[1];
        std::vector<int32_t> codebook_major(raw_codes.bytes.size() / sizeof(int32_t));
        std::memcpy(codebook_major.data(), raw_codes.bytes.data(), raw_codes.bytes.size());
        std::vector<int32_t> teacher_codes(codebook_major.size());
        for (int64_t t = 0; t < frames; ++t)
            for (int64_t q = 0; q < 8; ++q)
                teacher_codes[static_cast<size_t>(t * 8 + q)] = codebook_major[static_cast<size_t>(q * frames + t)];
        const auto waveform_shape = reference->require_metadata("waveform").shape;
        if (waveform_shape.size() != 2 || waveform_shape[0] <= 0 || waveform_shape[1] <= 0)
            throw std::runtime_error("waveform must be [channels,samples]");
        const auto expected = engine::audio::interleave_planar_channels(
            reference->require_f32("waveform", waveform_shape), static_cast<int>(waveform_shape[0]), waveform_shape[1]);
        for (const auto * name : {"seed", "codec_randn_offset"}) {
            if (std::string(name) == "codec_randn_offset" && !reference->has_tensor(name)) continue;
            const auto meta = reference->require_metadata(name);
            if (meta.dtype != "I64" || !meta.shape.empty() || reference->require_i64_scalar(name) < 0)
                throw std::runtime_error(std::string(name) + " must be a nonnegative I64 scalar");
        }
        const auto seed = static_cast<uint64_t>(reference->require_i64_scalar("seed"));
        const auto assets = mc::load_mulacover_assets(argv[1]);
        engine::core::ExecutionContext execution(backend);
        mc::AutoregressiveRuntime mula(execution, *assets, storage);
        mula.prepare(symbolic, style, batch == 2);
        engine::codecs::HeartCodecRuntime codec(assets->codec_weights, execution, assets->codec_config,
            {8ull << 20, 32ull << 20, 8ull << 20, 32ull << 20, storage});
        mc::GenerationOptions options;
        options.top_k = 250;
        options.guidance_scale = batch == 2 ? 1.5F : 1.0F;
        const auto generated = mula.generate(prompt, options, seed, frames, &teacher_codes);
        if (generated.frames != frames || generated.codebooks != 8 || generated.codes != teacher_codes)
            throw std::runtime_error("Teacher-forced generated codes differ from reference");
        uint64_t codec_offset = generated.rng_offset_blocks;
        if (reference->has_tensor("codec_randn_offset"))
            codec_offset = static_cast<uint64_t>(reference->require_i64_scalar("codec_randn_offset"));
        engine::debug::timing_log_scalar("validation.generated_codec_randn_offset", generated.rng_offset_blocks);
        engine::debug::timing_log_scalar("validation.codec_randn_offset", codec_offset);
        engine::codecs::HeartCodecDecodeOptions codec_options;
        const auto codec_stride = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(static_cast<int64_t>(codec_options.chunk_duration_seconds * 25.0F) *
                assets->codec_config.out_channels), generated.rng_policy);
        engine::debug::timing_log_scalar("validation.codec_randn_call_offset_blocks", codec_stride);
        const auto decoded = codec.detokenize_codes(generated.codes, generated.frames, generated.codebooks,
            codec_options, seed, codec_offset, codec_stride);
        if (decoded.batch_size != 1 || decoded.channels != waveform_shape[0] ||
            decoded.samples != waveform_shape[1] || decoded.values.size() != expected.size())
            throw std::runtime_error("Final waveform shape differs from reference");
        double dot = 0, aa = 0, bb = 0, max_abs = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            const double a = decoded.values[i], b = expected[i];
            if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("Non-finite final waveform");
            dot += a * b; aa += a * a; bb += b * b;
            max_abs = std::max(max_abs, std::abs(a - b));
        }
        if (aa == 0 || bb == 0) throw std::runtime_error("Waveform cosine is undefined for zero-energy audio");
        const double cosine = dot / std::sqrt(aa * bb);
        std::ostringstream metrics;
        metrics.precision(12);
        metrics << "validation.waveform.cosine=" << cosine << " validation.waveform.max_abs_error=" << max_abs;
        engine::debug::log_message(metrics.str());
        engine::audio::WavWriteOptions wav_options;
        wav_options.format = engine::audio::WavSampleFormat::Float32;
        engine::audio::write_wav(argv[4], assets->codec_config.sample_rate,
            static_cast<int>(decoded.channels), decoded.values, wav_options);
        if (!(cosine > minimum_waveform_cosine)) throw std::runtime_error("Final waveform parity failed: cosine below the requested gate");
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
