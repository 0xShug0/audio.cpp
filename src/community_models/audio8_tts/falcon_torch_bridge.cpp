#include "engine/community_models/audio8_tts/falcon_torch_bridge.h"

#include "engine/framework/audio/wav_reader.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace engine::models::audio8_tts {
namespace fs = std::filesystem;

bool is_falcon_backbone(const Audio8TtsAssets & assets) noexcept {
    const auto & cfg = assets.config.text;
    if (cfg.slow_backbone == "falcon_h1") return true;
    // Fallback: detect by tensor name when config omits slow_backbone (e.g. older 0.6B check)
    if (assets.model_weights && assets.model_weights->has_tensor("slow.embed_tokens.weight")) return true;
    return false;
}

static std::string shell_escape(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

runtime::AudioBuffer generate_audio_via_torch_falcon(
    const Audio8TtsAssets & assets,
    const Audio8TtsRequest & request,
    const std::vector<Audio8TtsCodes> & /*reference_codes*/,
    const std::optional<Audio8TtsConversationTurn> & /*previous_turn*/) {
    // This bridge replicates the HF generate path for Falcon-H1 so 0.1B is usable
    // before native Mamba is landed. It shells out to falcon_bridge.py using the
    // golden modeling_arktts.py implementation.
    const fs::path model_root = assets.resources.model_root();
    const fs::path bridge_py = fs::path(__FILE__).parent_path() / "falcon_bridge.py";

    // Prepare temp dir
    char tmp_template[] = "/tmp/audio8_falcon_XXXXXX";
    char * tmpdir_c = mkdtemp(tmp_template);
    if (!tmpdir_c) throw std::runtime_error("falcon bridge mkdtemp failed");
    fs::path tmpdir(tmpdir_c);
    fs::path out_wav = tmpdir / "out.wav";
    std::string ref_wav_arg;
    std::string ref_text_arg;
    fs::path ref_wav_path;

    // Handle voice cloning references: only first reference for now (chunked path handles one per request)
    if (!request.references.empty()) {
        const auto & ref = request.references.front();
        if (ref.audio.has_value()) {
            ref_wav_path = tmpdir / "ref.wav";
            const auto & ab = *ref.audio;
            // Write reference wav via raw f32 + soundfile to avoid huge command line
            fs::path raw_path = tmpdir / "ref.f32";
            {
                std::ofstream ofs(raw_path, std::ios::binary);
                if (!ofs) throw std::runtime_error("falcon bridge failed to open raw ref file");
                ofs.write(reinterpret_cast<const char*>(ab.samples.data()), ab.samples.size() * sizeof(float));
            }
            std::string py = "import soundfile as sf, numpy as np; "
                             "raw='" + raw_path.string() + "'; "
                             "wav='" + ref_wav_path.string() + "'; "
                             "sr=" + std::to_string(ab.sample_rate) + "; "
                             "samples=np.fromfile(raw, dtype=np.float32); "
                             "sf.write(wav, samples, sr)";
            std::string cmd = "/workspace/.torch_venv/bin/python -c " + shell_escape(py) + " 2>&1";
            int rc = std::system(cmd.c_str());
            if (rc != 0) {
                throw std::runtime_error("falcon bridge failed to write reference wav");
            }
            ref_wav_arg = " --reference-audio " + shell_escape(ref_wav_path.string());
            ref_text_arg = " --reference-text " + shell_escape(ref.text);
        }
    }

    std::string python = "/workspace/.torch_venv/bin/python";
    // Prefer torch venv python which has transformers + torch
    if (!fs::exists(python)) python = "python3";

    std::string cmd = shell_escape(python) + " " + shell_escape(bridge_py.string()) +
                      " --model " + shell_escape(model_root.string()) +
                      " --text " + shell_escape(request.text) +
                      " --out " + shell_escape(out_wav.string()) +
                      " --max-new-tokens " + std::to_string(request.generation.max_new_tokens) +
                      " --temperature " + std::to_string(request.generation.temperature) +
                      " --top-p " + std::to_string(request.generation.top_p) +
                      " --top-k " + std::to_string(request.generation.top_k) +
                      " --seed " + std::to_string(request.generation.seed) +
                      ref_wav_arg + ref_text_arg +
                      " 2>&1";

    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("falcon bridge python generate failed (rc=" + std::to_string(rc) + ") cmd: " + cmd);
    }
    if (!fs::exists(out_wav)) {
        throw std::runtime_error("falcon bridge did not produce wav: " + out_wav.string());
    }
    auto wav = engine::audio::read_wav_f32(out_wav);
    // Clean up
    std::error_code ec;
    fs::remove_all(tmpdir, ec);

    runtime::AudioBuffer out;
    out.sample_rate = wav.sample_rate;
    out.channels = wav.channels;
    out.samples = std::move(wav.samples);
    return out;
}

}  // namespace engine::models::audio8_tts
