// Stage 4a: the DiT's input stages -- time embedding, text projection, audio
// embedding -- against the reference's own taps.
//
// Diffed separately rather than as one number, because a single end-to-end figure
// over a 30-block backbone says "wrong" without saying where.

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/models/auk/dit.h"
#include "engine/models/auk/sampler.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

int int_arg(int argc, char ** argv, const std::string & name, int fallback) {
    return std::stoi(arg_value(argc, argv, name, std::to_string(fallback)));
}

std::vector<float> read_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const auto bytes = static_cast<std::streamoff>(input.tellg());
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

bool report(const char * label, const std::vector<float> & ours, const std::vector<float> & reference, double gate) {
    if (ours.size() != reference.size()) {
        std::cerr << "FAIL " << label << ": " << ours.size() << " values against " << reference.size() << "\n";
        return false;
    }
    double dot = 0.0, ln = 0.0, rn = 0.0, worst = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        dot += double(ours[i]) * double(reference[i]);
        ln += double(ours[i]) * double(ours[i]);
        rn += double(reference[i]) * double(reference[i]);
        worst = std::max(worst, std::fabs(double(ours[i]) - double(reference[i])));
    }
    const double cos = (ln == 0.0 || rn == 0.0) ? 0.0 : dot / (std::sqrt(ln) * std::sqrt(rn));
    const bool ok = cos >= gate;
    std::cout << std::fixed << std::setprecision(8)
              << (ok ? "  ok   " : "  FAIL ") << std::left << std::setw(16) << label
              << " cosine=" << cos << " max_abs_diff=" << worst << "\n";
    return ok;
}

}  // namespace

int main(int argc, char ** argv) try {
    const std::filesystem::path model = arg_value(argc, argv, "--model", "");
    const std::filesystem::path dir = arg_value(argc, argv, "--fixtures", "");
    if (model.empty() || dir.empty()) {
        std::cerr << "usage: auk_dit_parity --model <auk.gguf> --fixtures <dir with stage4_*.f32>\n";
        return 2;
    }
    const double gate = std::stod(arg_value(argc, argv, "--min-cosine", "0.9999"));

    engine::core::BackendConfig backend;
    // ⚠ Selectable, and it matters: a CUDA BUILD does not mean a CUDA RUN. Hardcoding
    // Cpu here made a GPU build reproduce the CPU numbers exactly, which reads as
    // "the backends agree" when it actually means the GPU never ran.
    const std::string backend_name = arg_value(argc, argv, "--backend", "cpu");
    backend.type = backend_name == "cuda"  ? engine::core::BackendType::Cuda
                 : backend_name == "best"  ? engine::core::BackendType::BestAvailable
                                           : engine::core::BackendType::Cpu;
    backend.threads = int_arg(argc, argv, "--threads", 1);
    engine::core::ExecutionContext execution(backend);

    engine::models::auk::AukDitConfig config;
    auto source = engine::assets::open_tensor_source(model);
    engine::models::auk::AukDit dit(config, *source, execution);

    engine::models::auk::AukDitInputs inputs;
    inputs.text = read_f32(dir / "stage4_text.f32");
    inputs.reference = read_f32(dir / "stage4_ref.f32");
    inputs.noised = read_f32(dir / "stage4_x.f32");
    inputs.timestep = read_f32(dir / "stage4_t.f32").at(0);
    inputs.text_tokens = static_cast<int64_t>(inputs.text.size()) / config.text_hidden_dim;
    inputs.ref_frames = static_cast<int64_t>(inputs.reference.size()) / config.latent_dim;
    inputs.gen_frames = static_cast<int64_t>(inputs.noised.size()) / config.latent_dim;
    std::cout << "loaded tensors=" << dit.loaded_tensor_count()
              << " text_tokens=" << inputs.text_tokens << " ref_frames=" << inputs.ref_frames
              << " gen_frames=" << inputs.gen_frames << " t=" << inputs.timestep << "\n";

    const auto out = dit.forward(inputs);
    bool ok = true;
    ok &= report("time_embed", out.time_embedding, read_f32(dir / "stage4_time_embed.f32"), gate);
    ok &= report("txt_norm", out.projected_text, read_f32(dir / "stage4_txt_norm.f32"), gate);
    ok &= report("audio_embed", out.embedded_audio, read_f32(dir / "stage4_audio_embed.f32"), gate);
    ok &= report("double0", out.first_double, read_f32(dir / "stage4_double0.f32"), gate);
    ok &= report("single0", out.first_single, read_f32(dir / "stage4_single0.f32"), gate);
    ok &= report("v_pred", out.velocity, read_f32(dir / "stage4_v_pred.f32"), gate);

    // Stage 5: the Euler/CFG sampler, driven from the reference's OWN y0 so a
    // difference here is the integration and not an unreproducible RNG.
    const std::filesystem::path y0_path = dir / "stage5_y0.f32";
    if (std::filesystem::exists(y0_path)) {
        engine::models::auk::AukSamplerOptions options;
        options.steps = int_arg(argc, argv, "--steps", 4);
        options.cfg_strength = static_cast<float>(std::stod(arg_value(argc, argv, "--cfg", "2.0")));
        const auto grid = engine::models::auk::build_time_grid(options.steps, options.use_sway, options.sway_coef);
        std::cout << "sampler: steps=" << options.steps << " cfg=" << options.cfg_strength
                  << " grid=[" << grid.front() << " .. " << grid.back() << "]\n";
        const auto sampled = engine::models::auk::sample_latents(dit, inputs, read_f32(y0_path), options);
        ok &= report("sampled", std::vector<float>(
                         sampled.latents.begin() + inputs.ref_frames * config.latent_dim,
                         sampled.latents.end()),
                     read_f32(dir / "stage5_sampled.f32"), gate);
        ok &= report("assembled", sampled.latents, read_f32(dir / "stage5_assembled.f32"), gate);
    }

    if (!ok) return 1;
    std::cout << "auk dit forward OK\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
}
