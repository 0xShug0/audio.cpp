// Numerical parity harness for the Echo-TTS DiT graph.
//
// Not registered with add_test: it needs the 5.5 GB GGUF and a reference dump
// from the upstream PyTorch implementation, so it is driven by hand, the same
// way dots_tts_vocoder_parity is.
//
//   python3 tools/community_models/echo_tts_reference.py --speaker ref.wav
//       --force-dtype float32 --full-blocks -o echo_ref.npz
//   python3 tools/community_models/echo_tts_pack_reference.py echo_ref.npz
//       -o echo_ref.bin
//   ./echo_tts_dit_parity --model /path/to/Echo-TTS-GGUF --reference echo_ref.bin
//
// --force-dtype float32 is load-bearing, not a nicety. ggml accumulates in F32
// whatever the stored weight type is, so a float32 reference is the like-for-like
// comparison even against an F16 GGUF. A bfloat16 reference -- upstream's default
// -- scores the 40-step trajectory at 0.905 and looks like a defect in this port;
// the same run against float32 scores 0.9995.
//
// Two checks, deliberately separate:
//
//   denoiser  feeds the reference's own x and t through one conditional
//             forward, removing the sampler from the comparison entirely.
//
//             It does NOT isolate the DiT blocks by themselves. The reference
//             text ids and speaker latents are injected, but prepare_conditioning()
//             then runs our own text encoder, speaker encoder and KV
//             projections, so a difference here could originate in any of
//             them. It is a combined conditioning-plus-denoiser comparison,
//             which is still enough to catch a wrong block -- nothing is being
//             compared against itself -- but not enough to localise one.
//
//   sampler   runs the full 40-step trajectory. Run it twice: once from the
//             reference's own initial noise, which removes the RNG from the
//             comparison, and once from our seeded draw.
//
//             Neither is expected to be exact, and the dominant term is NOT the
//             RNG. The reference defaults to bfloat16 while our GGUF is F16;
//             the two round differently, and dual CFG at 3.0/8.0 amplifies the
//             per-step difference every step. Dumping the reference with
//             --force-dtype float16 moves the 40-step cosine from 0.905 to
//             0.977 and the denoiser probe from 0.999977 to 0.999999. Compare
//             like dtypes or expect the gap. Cosine, never equality.

#include "engine/community_models/echo_tts/config.h"
#include "engine/community_models/echo_tts/dit.h"
#include "engine/community_models/echo_tts/sampler.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/model_spec/package.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// --- reference bundle ------------------------------------------------------

struct Tensor {
    bool is_int = false;
    std::vector<float> f32;
    std::vector<int32_t> i32;

    int64_t size() const { return is_int ? static_cast<int64_t>(i32.size())
                                         : static_cast<int64_t>(f32.size()); }
};

class ReferenceBundle {
public:
    // The largest bundle the packer emits with --blocks is 24 x 640 x 2048
    // floats in one entry; these caps sit well above that and well below
    // anything that could exhaust memory.
    static constexpr int32_t kMaxEntries = 4096;
    static constexpr int32_t kMaxNameLength = 1024;
    static constexpr int64_t kMaxElements = 1LL << 32;

    explicit ReferenceBundle(const std::filesystem::path & path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open reference bundle: " + path.string());
        }
        char magic[8] = {};
        in.read(magic, 8);
        if (std::memcmp(magic, "ECHOPAR1", 8) != 0) {
            throw std::runtime_error("not an ECHOPAR1 bundle: " + path.string());
        }
        // Every length below is signed on the wire and comes from a file this
        // process did not write. Validate before it reaches resize(), or a
        // negative value becomes a huge size_t and a malformed header becomes
        // an enormous allocation instead of a format error.
        const int32_t count = read_i32(in);
        if (count < 0 || count > kMaxEntries) {
            throw std::runtime_error("reference bundle declares an implausible entry count");
        }
        for (int32_t i = 0; i < count; ++i) {
            const int32_t name_len = read_i32(in);
            if (name_len < 0 || name_len > kMaxNameLength) {
                throw std::runtime_error("reference bundle has an implausible tensor name length");
            }
            std::string name(static_cast<size_t>(name_len), '\0');
            in.read(name.data(), name_len);
            const int32_t dtype = read_i32(in);
            if (dtype != 0 && dtype != 1) {
                throw std::runtime_error("reference bundle has an unknown dtype tag");
            }
            const int64_t elements = read_i64(in);
            if (elements < 0 || elements > kMaxElements) {
                throw std::runtime_error("reference bundle declares an implausible element count");
            }
            if (!in) {
                throw std::runtime_error("truncated reference bundle header at entry " + name);
            }

            Tensor tensor;
            tensor.is_int = dtype == 1;
            if (tensor.is_int) {
                tensor.i32.resize(static_cast<size_t>(elements));
                in.read(reinterpret_cast<char *>(tensor.i32.data()), elements * 4);
            } else {
                tensor.f32.resize(static_cast<size_t>(elements));
                in.read(reinterpret_cast<char *>(tensor.f32.data()), elements * 4);
            }
            if (!in) {
                throw std::runtime_error("truncated reference bundle at entry " + name);
            }
            entries_.emplace(std::move(name), std::move(tensor));
        }
    }

    const Tensor & at(const std::string & name) const {
        const auto it = entries_.find(name);
        if (it == entries_.end()) {
            throw std::runtime_error("reference bundle has no tensor named " + name);
        }
        return it->second;
    }

private:
    // The packer emits little-endian and documents that audio.cpp targets only
    // little-endian hosts, so these native reads are correct here. They would
    // need byte-swapping on a big-endian build.
    static int32_t read_i32(std::istream & in) {
        int32_t value = 0;
        in.read(reinterpret_cast<char *>(&value), 4);
        return value;
    }
    static int64_t read_i64(std::istream & in) {
        int64_t value = 0;
        in.read(reinterpret_cast<char *>(&value), 8);
        return value;
    }

    std::map<std::string, Tensor> entries_;
};

// --- metrics ---------------------------------------------------------------

struct Metrics {
    double cosine = 0.0;
    double max_abs_error = 0.0;
    double rms_error = 0.0;
};

// Cosine over the flattened tensors, reported with max-absolute-error beside it.
// Cosine alone hides a uniform scale error; max-abs alone is dominated by one
// outlier. The pair is what the port's own gate is written against.
Metrics compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            "size mismatch: actual=" + std::to_string(actual.size()) +
            " expected=" + std::to_string(expected.size()));
    }
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double sq = 0.0;
    Metrics metrics;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i];
        const double b = expected[i];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        const double diff = std::fabs(a - b);
        sq += diff * diff;
        metrics.max_abs_error = std::max(metrics.max_abs_error, diff);
    }
    const double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
    metrics.cosine = denom > 0.0 ? dot / denom : 0.0;
    metrics.rms_error = std::sqrt(sq / static_cast<double>(actual.size()));
    return metrics;
}

// Cosine alone is not a gate: `actual = 1000 * expected` scores a perfect 1.0
// while being catastrophically wrong in amplitude. The max-absolute error is
// what closes that hole, so both must hold for a PASS.
bool report(const std::string & label, const Metrics & m, double gate, double max_abs_gate) {
    const bool pass = m.cosine >= gate && m.max_abs_error <= max_abs_gate;
    std::cout << std::left << std::setw(10) << label
              << " cosine=" << std::fixed << std::setprecision(9) << m.cosine
              << "  max_abs=" << std::setprecision(6) << m.max_abs_error
              << "  rms=" << m.rms_error
              << "  gate=" << std::setprecision(3) << gate
              << "/" << max_abs_gate
              << (pass ? "  PASS" : "  FAIL") << "\n";
    return pass;
}

// --- arg parsing -----------------------------------------------------------

std::string arg_value(int argc, char ** argv, const std::string & name,
                      const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("echo_tts_dit_parity supports cuda, vulkan, cpu, or best");
}

bool has_flag(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

std::vector<float> to_float(const Tensor & tensor) {
    if (!tensor.is_int) {
        return tensor.f32;
    }
    std::vector<float> out(tensor.i32.size());
    std::transform(tensor.i32.begin(), tensor.i32.end(), out.begin(),
                   [](int32_t v) { return static_cast<float>(v); });
    return out;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model_path = arg_value(argc, argv, "--model", "");
    const std::filesystem::path reference_path = arg_value(argc, argv, "--reference", "");
    const std::string backend_name = arg_value(argc, argv, "--backend", "cuda");
    const double denoiser_gate = std::stod(arg_value(argc, argv, "--denoiser-gate", "0.999"));
    const double sampler_gate = std::stod(arg_value(argc, argv, "--sampler-gate", "0.999"));
    // Amplitude gates, deliberately loose relative to the cosine gate: they
    // exist to catch a scale error that cosine cannot see, not to re-litigate
    // the rounding difference the cosine gate already bounds.
    const double denoiser_max_abs =
        std::stod(arg_value(argc, argv, "--denoiser-max-abs", "0.25"));
    const double sampler_max_abs =
        std::stod(arg_value(argc, argv, "--sampler-max-abs", "4.0"));
    const bool skip_sampler = has_flag(argc, argv, "--skip-sampler");

    if (model_path.empty() || reference_path.empty()) {
        std::cerr << "usage: echo_tts_dit_parity --model <gguf dir> --reference <echo_ref.bin>\n"
                  << "       [--backend cuda|vulkan|cpu|best] [--denoiser-gate 0.999]\n"
                  << "       [--sampler-gate 0.999] [--denoiser-max-abs 0.25]\n"
                  << "       [--sampler-max-abs 4.0] [--skip-sampler]\n";
        return 2;
    }

    const ReferenceBundle reference(reference_path);

    engine::core::BackendConfig backend_config;
    backend_config.type = parse_backend(backend_name);
    engine::core::ExecutionContext execution(backend_config);

    auto bundle = engine::model_spec::load_resource_bundle_for_family(model_path, "echo_tts");
    auto dit_weights = bundle.open_tensor_source("dit_weights");

    engine::models::echo_tts::EchoTtsConfig config;
    config.validate();

    engine::models::echo_tts::EchoDitRuntime dit(
        config, *dit_weights, "", execution, engine::assets::TensorStorageType::Native);

    // Inject the reference's own conditioning rather than recomputing it, so
    // this measures the DiT and not the speaker encoder feeding it.
    engine::models::echo_tts::EchoConditioning conditioning;
    const auto & text_ids = reference.at("text.input_ids");
    conditioning.text_input_ids = text_ids.i32;
    conditioning.text_mask = to_float(reference.at("text.mask"));
    conditioning.text_length = text_ids.size();

    const auto & speaker_latent = reference.at("speaker.latent");
    conditioning.speaker_latent = speaker_latent.f32;
    conditioning.speaker_mask = to_float(reference.at("speaker.mask"));
    conditioning.speaker_frames = speaker_latent.size() / config.latent_size;

    std::cout << "text_length=" << conditioning.text_length
              << " speaker_frames=" << conditioning.speaker_frames << "\n";

    dit.prepare_conditioning(conditioning);

    bool ok = true;

    // 1. Fixed-timestep denoiser probe.
    {
        const auto & x_input = reference.at("dit.x_input");
        const auto t = static_cast<float>(reference.at("dit.t").f32.at(0));
        const auto predicted = dit.denoise_once(x_input.f32, t);
        std::cout << "denoiser probe at t=" << std::fixed << std::setprecision(4) << t << "\n";
        ok &= report("denoiser", compare(predicted, reference.at("dit.v_pred").f32), denoiser_gate,
                     denoiser_max_abs);
    }

    // 2. Sampler driven from the reference's OWN initial noise.
    //
    // This is the discriminator. Check 3 below runs the sampler from its own
    // seeded draw, which cannot be bit-identical to CUDA's. If that one
    // diverges while this one holds, the divergence is the RNG plus the
    // trajectory's sensitivity to it, not a defect in the integration. If this
    // one also diverges, the sampler itself is wrong.
    if (!skip_sampler) {
        engine::models::echo_tts::EchoSamplerOptions options;
        options.num_steps = static_cast<int>(reference.at("config.steps").i32.at(0));
        options.sequence_length = reference.at("config.sequence_length").i32.at(0);
        options.window_pinned = true;

        auto denoise = [&dit](const std::vector<float> & x, float t, int lanes) {
            return dit.denoise_once(x, t, lanes);
        };
        const auto latent = engine::models::echo_tts::run_euler_sampler(
            options,
            options.sequence_length,
            config.latent_size,
            reference.at("sampler.initial_noise").f32,
            denoise);
        std::cout << "sampler, reference initial noise injected\n";
        ok &= report("injected", compare(latent, reference.at("sampler.latent").f32), sampler_gate,
                     sampler_max_abs);
    }

    // 3. Full sampler trajectory from our own seeded noise. Expected to be
    // close, not exact: see header.
    if (!skip_sampler) {
        engine::models::echo_tts::EchoSamplerOptions options;
        options.num_steps = static_cast<int>(reference.at("config.steps").i32.at(0));
        options.sequence_length = reference.at("config.sequence_length").i32.at(0);
        options.seed = reference.at("config.seed").i32.at(0);
        options.window_pinned = true;
        const auto latent = dit.sample(options);
        std::cout << "sampler steps=" << options.num_steps
                  << " sequence_length=" << options.sequence_length
                  << " seed=" << options.seed << "\n";
        ok &= report("sampler", compare(latent, reference.at("sampler.latent").f32), sampler_gate,
                     sampler_max_abs);
    }

    std::cout << (ok ? "echo_tts_dit_parity: ok\n" : "echo_tts_dit_parity: FAILED\n");
    return ok ? 0 : 1;
} catch (const std::exception & ex) {
    std::cerr << "echo_tts_dit_parity: " << ex.what() << "\n";
    return 1;
}
