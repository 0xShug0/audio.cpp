// granite_speech_2b parity probe (Task B2).
//
// Runs the B1 frontend plus the B2 Conformer encoder and Q-Former projector on
// CPU (FP32, eager math, threads from --threads) and emits a trace directory
// with the same stage names, file names, shapes, and trace.json keys as
// tests/granite_speech_2b/reference.py, so tests/granite_speech_2b/compare_trace.py
// is the acceptance gate.
//
// Usage:
//   granite_speech_2b_parity_probe --model DIR --audio WAV \
//     --backend cpu --threads 8 --through frontend|encoder|projector \
//     --out-dir DIR [--projector-input FILE]
//
// --projector-input feeds a direct encoder_last-shaped [T,1024] fixture into
// the projector (letting decoder work proceed with reference encoder states);
// the frontend/encoder still run on --audio for the earlier stages and for
// span/length metadata, and --through is then limited to projector.

#include "engine/community_models/granite_speech_2b/assets.h"
#include "engine/community_models/granite_speech_2b/encoder.h"
#include "engine/community_models/granite_speech_2b/frontend.h"
#include "engine/community_models/granite_speech_2b/projector.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace g2b = engine::community_models::granite_speech_2b;

constexpr const char * kPinnedRevision = "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7";
// Pinned model.safetensors SHA-256, identical to MODEL_SHA256 in
// tests/granite_speech_2b/reference.py and to build/granite-2b-handoff/
// checkpoint.sha256. A mismatch is a hard FAIL (no trace emitted).
constexpr const char * kPinnedModelSha256 =
    "4151269881d2dd825d45ca75d6552b2a20e6e39d0934ed84c4db826af8bf6d88";
constexpr const char * kDefaultPrompt =
    "<|audio|>transcribe the speech with proper punctuation and capitalization.";

// --- minimal SHA-256 (FIPS 180-4) for trace hashes ---

class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        state_ = {0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul,
                  0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul};
        total_ = 0;
        buffered_ = 0;
    }
    void update(const void * data, size_t bytes) {
        const auto * begin = static_cast<const uint8_t *>(data);
        total_ += bytes;
        while (bytes > 0) {
            const size_t room = 64 - buffered_;
            const size_t take = bytes < room ? bytes : room;
            std::memcpy(block_ + buffered_, begin, take);
            begin += take;
            bytes -= take;
            buffered_ += take;
            if (buffered_ == 64) {
                compress(block_);
                buffered_ = 0;
            }
        }
    }
    std::string hexdigest() {
        const uint64_t bit_total = total_ * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (buffered_ != 56) {
            update(&zero, 1);
        }
        uint8_t length[8];
        for (int i = 0; i < 8; ++i) {
            length[i] = static_cast<uint8_t>((bit_total >> ((7 - i) * 8)) & 0xff);
        }
        update(length, 8);
        std::string out;
        char buf[9];
        for (uint32_t word : state_) {
            std::snprintf(buf, sizeof(buf), "%08x", word);
            out += buf;
        }
        reset();
        return out;
    }

private:
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void compress(const uint8_t * block) {
        static constexpr uint32_t k[64] = {
            0x428a2f98ul, 0x71374491ul, 0xb5c0fbcful, 0xe9b5dba5ul, 0x3956c25bul, 0x59f111f1ul, 0x923f82a4ul, 0xab1c5ed5ul,
            0xd807aa98ul, 0x12835b01ul, 0x243185beul, 0x550c7dc3ul, 0x72be5d74ul, 0x80deb1feul, 0x9bdc06a7ul, 0xc19bf174ul,
            0xe49b69c1ul, 0xefbe4786ul, 0x0fc19dc6ul, 0x240ca1ccul, 0x2de92c6ful, 0x4a7484aaul, 0x5cb0a9dcul, 0x76f988daul,
            0x983e5152ul, 0xa831c66dul, 0xb00327c8ul, 0xbf597fc7ul, 0xc6e00bf3ul, 0xd5a79147ul, 0x06ca6351ul, 0x14292967ul,
            0x27b70a85ul, 0x2e1b2138ul, 0x4d2c6dfcul, 0x53380d13ul, 0x650a7354ul, 0x766a0abbul, 0x81c2c92eul, 0x92722c85ul,
            0xa2bfe8a1ul, 0xa81a664bul, 0xc24b8b70ul, 0xc76c51a3ul, 0xd192e819ul, 0xd6990624ul, 0xf40e3585ul, 0x106aa070ul,
            0x19a4c116ul, 0x1e376c08ul, 0x2748774cul, 0x34b0bcb5ul, 0x391c0cb3ul, 0x4ed8aa4aul, 0x5b9cca4ful, 0x682e6ff3ul,
            0x748f82eeul, 0x78a5636ful, 0x84c87814ul, 0x8cc70208ul, 0x90befffaul, 0xa4506cebul, 0xbef9a3f7ul, 0xc67178f2ul};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
                (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
                (static_cast<uint32_t>(block[i * 4 + 2]) << 8) | static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
    std::array<uint32_t, 8> state_;
    uint64_t total_ = 0;
    uint8_t block_[64];
    size_t buffered_ = 0;
};

std::string sha256_file(const std::filesystem::path & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("parity probe cannot open file: " + path.string());
    }
    Sha256 hash;
    std::vector<char> chunk(1 << 20);
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto got = in.gcount();
        if (got > 0) {
            hash.update(chunk.data(), static_cast<size_t>(got));
        }
    }
    return hash.hexdigest();
}

// --- CLI ---

std::string arg_value(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return {};
}

bool has_flag(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

// --- tiny JSON writer ---

std::string json_escape(const std::string & text) {
    std::string out = "\"";
    for (char ch : text) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += ch;
                }
        }
    }
    out += "\"";
    return out;
}

std::string shape_json(const std::vector<int64_t> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    return out + "]";
}

void write_f32(const std::filesystem::path & path, const std::vector<float> & values) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("parity probe cannot write: " + path.string());
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<float> read_f32(const std::filesystem::path & path, size_t expect_elements) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("parity probe cannot open projector input: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto bytes = in.tellg();
    if (bytes < 0 || static_cast<size_t>(bytes) != expect_elements * sizeof(float)) {
        throw std::runtime_error(
            "parity probe projector input size " + std::to_string(bytes) + " bytes does not match " +
            std::to_string(expect_elements) + " expected float elements");
    }
    in.seekg(0, std::ios::beg);
    std::vector<float> values(expect_elements);
    in.read(reinterpret_cast<char *>(values.data()), bytes);
    return values;
}

int run(int argc, char ** argv) {
    const std::string model = arg_value(argc, argv, "--model");
    const std::string audio_path = arg_value(argc, argv, "--audio");
    const std::string backend = arg_value(argc, argv, "--backend");
    const std::string threads_text = arg_value(argc, argv, "--threads");
    const std::string through = arg_value(argc, argv, "--through");
    const std::string out_dir = arg_value(argc, argv, "--out-dir");
    const std::string projector_input = arg_value(argc, argv, "--projector-input");
    if (model.empty() || audio_path.empty() || backend.empty() || threads_text.empty() || through.empty() ||
        out_dir.empty() || has_flag(argc, argv, "--help")) {
        std::cerr << "usage: granite_speech_2b_parity_probe --model DIR --audio WAV "
                     "--backend cpu --threads 8 --through frontend|encoder|projector "
                     "--out-dir DIR [--projector-input FILE]\n";
        return 2;
    }
    if (backend != "cpu") {
        throw std::runtime_error(
            "parity probe only wires the cpu backend (advertise only what works): " + backend);
    }
    const int threads = std::stoi(threads_text);
    if (threads <= 0) {
        throw std::runtime_error("parity probe requires a positive thread count");
    }
    if (through != "frontend" && through != "encoder" && through != "projector") {
        throw std::runtime_error(
            "parity probe through must be frontend|encoder|projector in Task B2 (prefill|generate belong to Task C): " +
            through);
    }
    if (!projector_input.empty() && through != "projector") {
        throw std::runtime_error("parity probe --projector-input limits --through to projector");
    }
    const bool want_encoder = through == "encoder" || through == "projector";
    const bool want_projector = through == "projector";

    const auto audio_sha = sha256_file(audio_path);
    const auto model_sha = sha256_file(std::filesystem::path(model) / "model.safetensors");
    if (model_sha != kPinnedModelSha256) {
        throw std::runtime_error(
            "parity probe model.safetensors sha256 " + model_sha + " != pinned " + kPinnedModelSha256 +
            "; refusing to emit a trace");
    }

    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(audio_path));
    if (wav.samples.empty()) {
        throw std::runtime_error("parity probe audio has no samples: " + audio_path);
    }
    engine::runtime::AudioBuffer buffer;
    buffer.sample_rate = wav.sample_rate;
    buffer.channels = wav.channels;
    buffer.samples = wav.samples;
    const auto num_samples = static_cast<int64_t>(wav.samples.size());

    const auto assets = g2b::load_granite_speech_2b_assets(model);

    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{engine::core::BackendType::Cpu, 0, threads});

    const g2b::GraniteSpeech2BFrontend frontend(assets);
    const auto features = frontend.extract(buffer);

    g2b::GraniteSpeech2BEncoderOutputs encoder_out;
    g2b::GraniteSpeech2BProjectorOutputs projector_out;
    if (want_encoder) {
        g2b::GraniteSpeech2BEncoderRuntime encoder(
            assets, execution, engine::assets::TensorStorageType::F32);
        encoder_out = encoder.encode(features.stacked, features.encoder_steps);
    }
    if (want_projector) {
        g2b::GraniteSpeech2BProjectorRuntime projector(
            assets, execution, engine::assets::TensorStorageType::F32);
        // All graph uploads are done; the reopenable source may drop storage.
        assets->source->release_storage();
        if (!projector_input.empty()) {
            const auto fixture = read_f32(
                projector_input,
                static_cast<size_t>(features.encoder_steps * g2b::kGraniteSpeech2BProjectorHidden));
            projector_out = projector.project(fixture, features.encoder_steps);
        } else {
            projector_out = projector.project(encoder_out.last, features.encoder_steps);
        }
    } else if (want_encoder) {
        assets->source->release_storage();
    }

    std::filesystem::create_directories(out_dir);
    write_f32(std::filesystem::path(out_dir) / "mel.f32", features.mel);
    write_f32(std::filesystem::path(out_dir) / "stacked.f32", features.stacked);
    if (want_encoder) {
        write_f32(std::filesystem::path(out_dir) / "input_proj.f32", encoder_out.input_proj);
        write_f32(std::filesystem::path(out_dir) / "encoder_block1.f32", encoder_out.block1);
        write_f32(
            std::filesystem::path(out_dir) / "encoder_layer8_pre_feedback.f32",
            encoder_out.layer8_pre_feedback);
        write_f32(
            std::filesystem::path(out_dir) / "encoder_layer8_post_feedback.f32",
            encoder_out.layer8_post_feedback);
        write_f32(std::filesystem::path(out_dir) / "encoder_last.f32", encoder_out.last);
    }
    if (want_projector) {
        write_f32(std::filesystem::path(out_dir) / "projected.f32", projector_out.projected);
    }

    // trace.json mirrors reference.py keys; thresholds are the reference
    // PIPELINE entries (the gate reads them from the reference side).
    const auto & lengths = features.lengths;
    std::ostringstream json;
    json << "{\n";
    json << "  \"family\": \"granite_speech_2b\",\n";
    json << "  \"revision\": \"" << kPinnedRevision << "\",\n";
    json << "  \"model_sha256\": \"" << model_sha << "\",\n";
    json << "  \"audio_sha256\": \"" << audio_sha << "\",\n";
    json << "  \"runtime\": {\"impl\": \"audio.cpp parity probe\", \"threads\": " << threads << "},\n";
    json << "  \"backend\": \"cpu\",\n";
    json << "  \"attn_implementation\": \"eager\",\n";
    json << "  \"dtype\": \"float32\",\n";
    json << "  \"threads\": " << threads << ",\n";
    json << "  \"through\": " << json_escape(through) << ",\n";
    json << "  \"sample_rate\": 16000,\n";
    json << "  \"audio\": {\"path\": " << json_escape(audio_path) << ", \"num_samples\": " << num_samples
         << ", \"span\": [0, " << num_samples << "]},\n";
    json << "  \"lengths\": {\"mel_frames\": " << lengths.mel_frames << ", \"encoder_steps\": "
         << lengths.encoder_steps << ", \"windows\": " << lengths.projector_windows << ", \"audio_tokens\": "
         << lengths.audio_placeholders << ", \"prompt_len\": 0, \"generated_len\": 0},\n";
    json << "  \"prompt\": " << json_escape(kDefaultPrompt) << ",\n";
    json << "  \"language_hint\": null,\n";
    json << "  \"input_ids\": [],\n";
    json << "  \"max_tokens\": 512,\n";
    json << "  \"generated_ids\": [],\n";
    json << "  \"observed_stop\": \"\",\n";
    json << "  \"eos_token_id\": " << assets->config.text.eos_token_id << ",\n";
    json << "  \"text\": \"\",\n";
    json << "  \"pinned\": true,\n";
    json << "  \"allow_unpinned\": false,\n";
    const auto & enc = assets->config.encoder;
    const auto & proj = assets->config.projector;
    const auto & txt = assets->config.text;
    json << "  \"model_config\": {\n";
    json << "    \"audio_token_id\": " << txt.audio_token_id << ",\n";
    json << "    \"encoder\": {\"input_dim\": " << enc.input_dim << ", \"hidden_dim\": " << enc.hidden_dim
         << ", \"num_layers\": " << enc.num_layers << ", \"num_heads\": " << enc.num_heads << ", \"dim_head\": "
         << enc.dim_head << ", \"context_size\": " << enc.context_size << ", \"output_dim\": " << enc.output_dim
         << "},\n";
    json << "    \"projector\": {\"window_size\": " << proj.window_size
         << ", \"downsample_rate\": " << proj.downsample_rate << ", \"hidden_size\": " << proj.hidden_size
         << ", \"qformer_layers\": " << proj.num_hidden_layers << ", \"qformer_heads\": " << proj.num_attention_heads
         << "},\n";
    json << "    \"text\": {\"num_layers\": " << txt.num_hidden_layers << ", \"hidden_size\": " << txt.hidden_size
         << ", \"num_attention_heads\": " << txt.num_attention_heads << ", \"num_key_value_heads\": "
         << txt.num_key_value_heads << ", \"intermediate_size\": " << txt.intermediate_size << ", \"vocab_size\": "
         << txt.vocab_size << ", \"bos_token_id\": " << txt.bos_token_id << ", \"eos_token_id\": "
         << txt.eos_token_id << ", \"pad_token_id\": " << txt.pad_token_id << ", \"embedding_multiplier\": "
         << txt.embedding_multiplier << ", \"attention_multiplier\": " << txt.attention_multiplier
         << ", \"logits_scaling\": " << txt.logits_scaling << ", \"residual_multiplier\": "
         << txt.residual_multiplier << ", \"rope_theta\": " << txt.rope_theta << ", \"rope_scaling\": {\"factor\": "
         << txt.rope_factor << ", \"original_max_position_embeddings\": "
         << txt.rope_original_max_position_embeddings << ", \"rope_type\": " << json_escape(txt.rope_type)
         << "}},\n";
    json << "    \"frontend\": {\"sample_rate\": " << assets->config.frontend.sample_rate << "}\n";
    json << "  },\n";
    json << "  \"load_info\": {\"missing\": [], \"unexpected\": [], \"mismatched\": [], \"errors\": []},\n";
    json << "  \"stages\": [\n";
    const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, const char *>> stages = {
        {"mel", "mel.f32", {lengths.mel_frames, 80}, "\"max_abs\": 0.0002, \"rel_rms\": 0.001"},
        {"stacked", "stacked.f32", {lengths.encoder_steps, 160}, "\"max_abs\": 0.0002, \"rel_rms\": 0.001"},
        {"input_proj", "input_proj.f32", {lengths.encoder_steps, 1024}, "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"encoder_block1",
         "encoder_block1.f32",
         {lengths.encoder_steps, 1024},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"encoder_layer8_pre_feedback",
         "encoder_layer8_pre_feedback.f32",
         {lengths.encoder_steps, 1024},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"encoder_layer8_post_feedback",
         "encoder_layer8_post_feedback.f32",
         {lengths.encoder_steps, 1024},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"encoder_last",
         "encoder_last.f32",
         {lengths.encoder_steps, 1024},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"projected",
         "projected.f32",
         {lengths.audio_placeholders, 2048},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
    };
    size_t stage_count = through == "frontend" ? 2 : (through == "encoder" ? 7 : 8);
    for (size_t i = 0; i < stage_count; ++i) {
        const auto & [name, file, shape, thresholds] = stages[i];
        json << "    {\"name\": " << json_escape(name) << ", \"file\": " << json_escape(file) << ", \"shape\": "
             << shape_json(shape) << ", \"dtype\": \"float32\", " << thresholds << "}"
             << (i + 1 == stage_count ? "\n" : ",\n");
    }
    json << "  ],\n";
    json << "  \"cutoffs\": {\"frontend\": \"stacked\", \"encoder\": \"encoder_last\", "
            "\"projector\": \"projected\", \"prefill\": \"last_pos_logits\", "
            "\"generate\": \"forced_step7\"}\n";
    json << "}\n";
    std::ofstream trace_out(std::filesystem::path(out_dir) / "trace.json", std::ios::binary | std::ios::trunc);
    if (!trace_out) {
        throw std::runtime_error("parity probe cannot write trace.json");
    }
    trace_out << json.str();
    std::cout << "wrote " << stage_count << " stages through " << through << " to " << out_dir << "\n";
    return 0;
}
}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & error) {
        std::cerr << "granite_speech_2b_parity_probe failed: " << error.what() << "\n";
        return 1;
    }
}
