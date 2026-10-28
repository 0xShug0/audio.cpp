// The codec decoder (codec.h).
//
// - FSQ latents: the value table of the real levels [8, 7, 6, 6], codebook-major order, and rejection of
//   EOS and malformed frames.
// - Synthetic package (F32 mode): project_out, from_quantizer, decoder stack and spectrum against the host
//   fp64 reference, over 200 frames (two chunk graphs and a wrap of the K/V ring), and the waveform
//   against lfm2_audio_istft of the fp64 spectrum. Streaming pushes of 1/5/7/130/57 frames give HF's
//   chunk sizes and the offline samples. Native mode against the same reference.
//
//   lfm3_audio_codec_test [--backend cpu|cuda|metal|best] [--threads N]
//
// A requested backend that is not available exits 77.
#include "engine/community_models/lfm2_audio/detokenizer.h"
#include "engine/community_models/lfm3_audio/codec.h"
#include "engine/framework/core/backend.h"
#include "lfm3_audio_reference.h"
#include "lfm3_audio_test_package.h"
#include "test_assert.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace core = engine::core;
namespace ref = lfm3_audio_test;
using namespace engine::community_models::lfm3_audio;
using engine::test::require;
using ref::rel_error;
using ref::report;
using ref::Values;

constexpr double kWaveTolerance = 1e-4;
// The waveform amplifies spectrum errors through exp(log_mag) and the phases, about 10x for random codes (the
// fixtures' bf16 floors: 0.83% at lin, 9.9% at the rand200 wave), as the CPU wave tolerance allows 10x too.
constexpr double kGpuWaveTolerance = 1e-2;
// Native mode on the synthetic package: the same BF16 weights as the reference, F16 K/V and flash attention.
constexpr double kNativeTolerance = 2e-2;

void test_fsq() {
    const std::vector<int64_t> levels = {8, 7, 6, 6};
    const std::map<int32_t, std::vector<float>> table = {
        {0, {-1.0f, -1.0f, -1.0f, -1.0f}},
        {1, {-0.75f, -1.0f, -1.0f, -1.0f}},
        {7, {0.75f, -1.0f, -1.0f, -1.0f}},
        {8, {-1.0f, -2.0f / 3.0f, -1.0f, -1.0f}},
        {55, {0.75f, 1.0f, -1.0f, -1.0f}},
        {56, {-1.0f, -1.0f, -2.0f / 3.0f, -1.0f}},
        {335, {0.75f, 1.0f, 2.0f / 3.0f, -1.0f}},
        {336, {-1.0f, -1.0f, -1.0f, -2.0f / 3.0f}},
        {2015, {0.75f, 1.0f, 2.0f / 3.0f, 2.0f / 3.0f}},
    };
    for (const auto & [code, expected] : table) {
        require(lfm3_fsq_latents({{code}}, levels) == expected, "FSQ value of code " + std::to_string(code));
    }

    // Codebook-major: frame f is [cb0 d0..d3, cb1 d0..d3, ...].
    const auto two = lfm3_fsq_latents({{0, 2015}, {7, 336}}, levels);
    const std::vector<float> expected = {-1.0f,  -1.0f, -1.0f, -1.0f, 0.75f, 1.0f,  2.0f / 3.0f, 2.0f / 3.0f,
                                         0.75f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,       -2.0f / 3.0f};
    require(two == expected, "FSQ latents are codebook-major");

    ref::require_throws_with([&] { (void)lfm3_fsq_latents({{2016}}, levels); }, "EOS", "the EOS code");
    ref::require_throws([&] { (void)lfm3_fsq_latents({{-1}}, levels); }, "a negative code");
    ref::require_throws([&] { (void)lfm3_fsq_latents({{0, 1}, {0}}, levels); }, "frames of different sizes");
    std::printf("PASS fsq table and order\n");
}

std::vector<std::vector<int32_t>> random_frames(int64_t count, int64_t codebooks, int64_t codes, uint64_t seed) {
    lfm2_audio_test::Random random(seed);
    std::vector<std::vector<int32_t>> out(static_cast<size_t>(count), std::vector<int32_t>(static_cast<size_t>(codebooks)));
    for (auto & frame : out) {
        for (auto & code : frame) {
            code = std::min<int32_t>(static_cast<int32_t>((random.uniform(1.0f) + 1.0f) * 0.5f * static_cast<float>(codes)),
                                     static_cast<int32_t>(codes - 1));
        }
    }
    return out;
}

struct Reference {
    Values project_out, from_quantizer, decoder, spectrum;
    std::vector<float> wave;
};

Reference codec_reference(const engine::assets::TensorSource & source, const Lfm3CodecConfig & config,
                          const std::vector<std::vector<int32_t>> & frames) {
    const auto read = [&](const std::string & name) { return ref::to_values(source.require_f32(name)); };
    const int64_t d = config.stack.hidden;
    const int64_t latent = config.codebooks * static_cast<int64_t>(config.fsq_levels.size());
    const auto count = static_cast<int64_t>(frames.size());
    const int64_t steps = count * config.upsample;

    Values latents;
    for (const auto & frame : frames) {
        for (const int32_t code : frame) {
            int64_t basis = 1;
            for (const auto level : config.fsq_levels) {
                latents.push_back(static_cast<double>((code / basis) % level - level / 2) / static_cast<double>(level / 2));
                basis *= level;
            }
        }
    }

    Reference out;
    out.project_out =
        ref::linear_bias(latents, read("quantizer.project_out.weight"), read("quantizer.project_out.bias"), latent, config.quantizer_dim);
    out.from_quantizer =
        ref::linear_bias(out.project_out, read("from_quantizer.weight"), read("from_quantizer.bias"), config.quantizer_dim, d);
    Values x(static_cast<size_t>(steps * d));
    for (int64_t p = 0; p < steps; ++p) {
        std::copy_n(out.from_quantizer.begin() + (p / config.upsample) * d, d, x.begin() + p * d);
    }
    out.decoder = ref::stack(config.stack, ref::read_ref_stack(source, "decoder.lfm.", config.stack), x, ref::positions_from(0, steps));
    out.spectrum = ref::linear_bias(out.decoder, read("decoder.lin.weight"), read("decoder.lin.bias"), d, config.n_fft + 2);
    out.wave = engine::community_models::lfm2_audio::lfm2_audio_istft(std::vector<float>(out.spectrum.begin(), out.spectrum.end()), steps,
                                                                      source.require_f32("decoder.istft.window"), config.hop);
    return out;
}

void test_synthetic(core::ExecutionContext & execution) {
    const auto package = ref::write_package("lfm3_audio_codec_test");
    const auto assets = load_lfm3_audio_assets(package.root);
    const auto & config = assets->codec;
    int64_t codes = 1;
    for (const auto level : config.fsq_levels) {
        codes *= level;
    }
    // 200 frames = 1200 steps: a 128 + 72 frame split and a wrap of the 1024-row K/V ring.
    const auto frames = random_frames(200, config.codebooks, codes, 7);
    const auto expected = codec_reference(*assets->codec_weights, config, frames);
    const bool gpu = execution.backend_type() != core::BackendType::Cpu;
    const double stage_tol = ref::reference_tolerance(execution);
    const double wave_tol = gpu ? kGpuWaveTolerance : kWaveTolerance;

    Lfm3CodecDecoder codec(assets->codec_weights, config, execution, {Lfm3WeightType::F32, true});
    ref::require_throws_with([&] { (void)codec.push(frames); }, "not been started", "push before start_stream");

    codec.start_stream();
    Lfm3CodecTaps taps;
    const auto spectrum = codec.spectrum(frames, &taps);
    report("synthetic f32 project_out", ref::rel_error(taps.project_out, expected.project_out), stage_tol);
    report("synthetic f32 from_quantizer", ref::rel_error(taps.from_quantizer, expected.from_quantizer), stage_tol);
    report("synthetic f32 decoder", ref::rel_error(taps.decoder, expected.decoder), stage_tol);
    report("synthetic f32 spectrum", ref::rel_error(spectrum, expected.spectrum), stage_tol);

    const auto offline = codec.decode(frames);
    require(offline.size() == expected.wave.size(), "offline length = hop per step");
    report("synthetic f32 wave", ref::rel_error(offline, expected.wave), wave_tol);

    // Streaming: each push returns every sample its rows complete, HF's chunk output (first chunk drops the
    // pad, later chunks hold back 2 * pad); finish() gives the rest.
    const int64_t pad = (config.n_fft - config.hop) / 2;
    codec.start_stream();
    std::vector<float> streamed;
    int64_t pushed = 0;
    for (const int64_t count : {1, 5, 7, 130, 57}) {
        const std::vector<std::vector<int32_t>> part(frames.begin() + pushed, frames.begin() + pushed + count);
        const auto samples = codec.push(part);
        pushed += count;
        streamed.insert(streamed.end(), samples.begin(), samples.end());
        require(static_cast<int64_t>(streamed.size()) == pushed * config.upsample * config.hop - pad,
                "streamed samples after " + std::to_string(pushed) + " frames");
    }
    const auto rest = codec.finish();
    require(static_cast<int64_t>(rest.size()) == pad, "finish() returns the last pad samples");
    streamed.insert(streamed.end(), rest.begin(), rest.end());
    report("synthetic f32 stream vs offline", rel_error(streamed, offline), gpu ? kGpuWaveTolerance : stage_tol);
    ref::require_throws_with([&] { (void)codec.push(frames); }, "not been started", "push after finish");

    codec.start_stream();
    ref::require_throws_with([&] { (void)codec.push({std::vector<int32_t>(static_cast<size_t>(config.codebooks), static_cast<int32_t>(codes))}); },
                             "EOS", "an EOS frame");

    Lfm3CodecDecoder native(assets->codec_weights, config, execution, {Lfm3WeightType::Native, true});
    native.start_stream();
    report("synthetic native spectrum", ref::rel_error(native.spectrum(frames), expected.spectrum), kNativeTolerance);
}

}  // namespace

int main(int argc, char ** argv) {
    int status = 0;
    const auto execution = ref::open_backend(argc, argv, status);
    if (!execution) {
        return status;
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        test_fsq();
        test_synthetic(*execution);
        std::printf("PASS: lfm3_audio codec (%.1f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
