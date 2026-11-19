// granite_speech5 parity probe.
//
// Runs the B1 frontend plus the B2 Conformer encoder and Q-Former projector on
// CPU (FP32, eager math, threads from --threads) and, for --through
// prefill|generate, the C2 Granite text decoder (device embedding lookup,
// host audio injection, bounded-block prefill, forced single-token decode).
// Emits a trace directory with the same stage names, file names, shapes, and
// trace.json keys as tests/granite_speech5/reference.py, so
// tests/granite_speech5/compare_trace.py is the acceptance gate.
//
// Usage:
//
//   granite_speech5_parity_probe --model DIR --audio WAV --backend cpu
//     --threads 8 --through frontend|encoder|projector|prefill|generate
//     --out-dir DIR [--projector-input FILE] [--language CODE] [--force-ids FILE]
//
// --projector-input feeds a direct encoder_last-shaped [T,1024] fixture into
// the projector (letting decoder work proceed with reference encoder states);
// the frontend/encoder still run on --audio for the earlier stages and for
// span/length metadata, and --through is then limited to
// projector|prefill|generate.
// --language selects the IBM source-language hint (reference.py --language
// choices); without it the default transcription prompt is used.
// --force-ids takes a JSON list of token IDs as the teacher-forced
// continuation for generate (default: this run's greedy output, mirroring
// reference.py --force-ids).

#include "engine/community_models/granite_speech5/assets.h"
#include "decoder_parity.h"
#include "engine/community_models/granite_speech5/encoder.h"
#include "engine/community_models/granite_speech5/frontend.h"
#include "engine/community_models/granite_speech5/projector.h"
#include "engine/community_models/granite_speech5/prompt.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace g2b = engine::community_models::granite_speech5;

constexpr const char * kPinnedRevision = "ec49ca6b1a5e7e9176127e09cb303ab78cbacdf7";
// Pinned model.safetensors SHA-256, identical to MODEL_SHA256 in
// tests/granite_speech5/reference.py. A mismatch fails without emitting traces.
constexpr const char * kPinnedModelSha256 =
    "4151269881d2dd825d45ca75d6552b2a20e6e39d0934ed84c4db826af8bf6d88";
constexpr const char * kDefaultPrompt =
    "<|audio|>transcribe the speech with proper punctuation and capitalization.";
// Forced-step bound mirroring reference.py N_FORCED_STEPS, and the emitted
// max_tokens bound on the (never materialized here) continuation; the
// generate cutoff derives from both (see the cutoffs emission below).
constexpr int kProbeForcedSteps = 8;
constexpr int kProbeMaxTokens = 512;

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

// Minimal JSON reader for a flat list of integer token IDs: "[1, 2, 3]".
// Only what --force-ids needs; anything else throws.
std::vector<int32_t> read_id_list(const std::filesystem::path & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("parity probe cannot open --force-ids file: " + path.string());
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<int32_t> out;
    std::string token;
    bool have_bracket = false;
    for (char ch : text) {
        if (ch == '[') {
            if (have_bracket) {
                throw std::runtime_error("parity probe --force-ids must be a single flat list");
            }
            have_bracket = true;
        } else if (ch == ']') {
            if (!have_bracket) {
                throw std::runtime_error("parity probe --force-ids has an unmatched ']'");
            }
            have_bracket = false;
            if (!token.empty()) {
                out.push_back(static_cast<int32_t>(std::stol(token)));
                token.clear();
            }
        } else if (ch == ',') {
            if (!token.empty()) {
                out.push_back(static_cast<int32_t>(std::stol(token)));
                token.clear();
            }
        } else if (ch == '-' || (ch >= '0' && ch <= '9')) {
            token += ch;
        } else if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            if (!token.empty()) {
                out.push_back(static_cast<int32_t>(std::stol(token)));
                token.clear();
            }
        } else {
            throw std::runtime_error(
                std::string("parity probe --force-ids has an unexpected character: ") + ch);
        }
    }
    if (have_bracket || !token.empty()) {
        throw std::runtime_error("parity probe --force-ids is not a complete [...] list");
    }
    return out;
}

std::string ids_json(const std::vector<int32_t> & ids) {
    std::string out = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(ids[i]);
    }
    return out + "]";
}

int run(int argc, char ** argv) {
    const std::string model = arg_value(argc, argv, "--model");
    const std::string audio_path = arg_value(argc, argv, "--audio");
    const std::string backend = arg_value(argc, argv, "--backend");
    const std::string threads_text = arg_value(argc, argv, "--threads");
    const std::string through = arg_value(argc, argv, "--through");
    const std::string out_dir = arg_value(argc, argv, "--out-dir");
    const std::string projector_input = arg_value(argc, argv, "--projector-input");
    const std::string language = arg_value(argc, argv, "--language");
    const std::string force_ids = arg_value(argc, argv, "--force-ids");
    const std::string decoder_input = arg_value(argc, argv, "--decoder-input");
    if (model.empty() || audio_path.empty() || backend.empty() || threads_text.empty() || through.empty() ||
        out_dir.empty() || has_flag(argc, argv, "--help")) {
        std::cerr << "usage: granite_speech5_parity_probe --model DIR --audio WAV --backend cpu|cuda|vulkan\n"
                     "  --threads 8 --through frontend|encoder|projector|prefill|generate --out-dir DIR\n"
                     "  [--projector-input FILE] [--decoder-input FILE] [--eager-attention] [--language CODE] [--force-ids FILE]\n";
        return 2;
    }
    engine::core::BackendType backend_type;
    if (backend == "cpu") backend_type = engine::core::BackendType::Cpu;
    else if (backend == "cuda") backend_type = engine::core::BackendType::Cuda;
    else if (backend == "vulkan") backend_type = engine::core::BackendType::Vulkan;
    else throw std::runtime_error("unsupported probe backend: " + backend);
    const int threads = std::stoi(threads_text);
    if (threads <= 0) {
        throw std::runtime_error("parity probe requires a positive thread count");
    }
    if (through != "frontend" && through != "encoder" && through != "projector" && through != "prefill" &&
        through != "generate") {
        throw std::runtime_error(
            "parity probe through must be frontend|encoder|projector|prefill|generate: " + through);
    }
    if (!projector_input.empty() && through != "projector" && through != "prefill" && through != "generate") {
        throw std::runtime_error("parity probe --projector-input limits --through to projector|prefill|generate");
    }
    const bool want_encoder = through != "frontend";
    const bool want_projector = through != "frontend" && through != "encoder";
    const bool want_prefill = through == "prefill" || through == "generate";
    const bool want_generate = through == "generate";

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

    const auto assets = g2b::load_granite_speech5_assets(model);

    engine::core::ExecutionContext execution(
        engine::core::BackendConfig{backend_type, 0, threads});

    const g2b::GraniteSpeech5Frontend frontend(assets);
    const auto features = frontend.extract(buffer);

    g2b::GraniteSpeech5EncoderOutputs encoder_out;
    g2b::GraniteSpeech5ProjectorOutputs projector_out;
    if (want_encoder) {
        g2b::GraniteSpeech5EncoderRuntime encoder(
            assets, execution, engine::assets::TensorStorageType::F32);
        encoder_out = encoder.encode(features.stacked, features.encoder_steps, true);
    }
    if (want_projector) {
        g2b::GraniteSpeech5ProjectorRuntime projector(
            assets, execution, engine::assets::TensorStorageType::F32);
        if (!want_prefill) {
            // All graph uploads are done; the reopenable source may drop storage.
            assets->source->release_storage();
        }
        if (!projector_input.empty()) {
            const auto fixture = read_f32(
                projector_input,
                static_cast<size_t>(features.encoder_steps * g2b::kGraniteSpeech5ProjectorHidden));
            projector_out = projector.project(fixture, features.encoder_steps);
        } else {
            projector_out = projector.project(encoder_out.last, features.encoder_steps);
        }
    } else if (want_encoder) {
        assets->source->release_storage();
    }

    // C2 text decoder (prefill|generate): prompt from the C1 builder over the
    // projector outputs, then block prefill (+ forced single-token decode and
    // greedy continuation for generate).
    g2b::GraniteSpeech5Decoder::PrefillResult decoder_prefill;
    std::vector<std::vector<float>> forced_steps;
    std::vector<int32_t> greedy_ids;
    std::vector<int32_t> forced_continuation;
    g2b::GraniteSpeech5PromptIds expanded_prompt;
    std::string prompt_text = kDefaultPrompt;
    std::string language_hint;
    std::string generated_text;
    if (want_prefill) {
        const auto tokenizer = g2b::load_granite_speech5_tokenizer(assets->resources);
        const g2b::GraniteSpeech5PromptBuilder builder(tokenizer);
        const std::string normalized =
            language.empty() ? std::string() : g2b::granite_speech5_normalize_language_code(language);
        if (!language.empty()) {
            language_hint = normalized;
        }
        auto prompt = builder.build_decoder_prompt(normalized, projector_out.audio_placeholders,
                                                   projector_out.projected);
        expanded_prompt.input_ids = prompt.input_ids;
        expanded_prompt.audio_positions = prompt.injection.positions;
        expanded_prompt.audio_tokens = prompt.injection.tokens;
        const auto budget = g2b::granite_speech5_prompt_budget(expanded_prompt);
        g2b::granite_speech5_require_prompt_fits(budget);
        prompt_text = g2b::granite_speech5_render_single_turn_prompt(
            g2b::granite_speech5_build_user_content(normalized));
        if (!decoder_input.empty()) {
            prompt.injection.values = read_f32(
                decoder_input, static_cast<size_t>(prompt.injection.tokens * 2048));
        }
        g2b::GraniteSpeech5Decoder decoder(assets->source, execution, has_flag(argc, argv, "--eager-attention"));
        if (!force_ids.empty()) {
            forced_continuation = read_id_list(force_ids);
        }
        decoder_prefill = decoder.prefill(prompt, kProbeMaxTokens);
        assets->source->release_storage();
        if (want_generate) {
            // forced_step{k} predicts continuation[k]: step 0 is the prefill
            // last-position distribution; step k>=1 feeds continuation[k-1]
            // (exactly what generate() does per step). Mirrors
            // reference.py N_FORCED_STEPS.
            auto build_forced_steps = [&](const std::vector<float> & first_logits) {
                forced_steps.clear();
                const size_t n_steps =
                    std::min<size_t>(kProbeForcedSteps, forced_continuation.size());
                forced_steps.reserve(n_steps);
                if (n_steps > 0) {
                    forced_steps.push_back(first_logits);
                    for (size_t k = 1; k < n_steps; ++k) {
                        forced_steps.push_back(decoder.decode_token(forced_continuation[k - 1]));
                    }
                }
            };
            if (!forced_continuation.empty()) {
                // --force-ids path: teacher-force the given continuation on
                // the prefill cache above.
                build_forced_steps(decoder_prefill.last_logits);
            }
            // Fresh full prefill inside; no forced decode above leaks into
            // the greedy continuation.
            greedy_ids = decoder.generate_greedy(prompt, kProbeMaxTokens);
            if (force_ids.empty()) {
                // Default path: the continuation is this run's greedy output,
                // so it only exists now. Re-prefill (generate_greedy consumed
                // its own cache) and teacher-force it, which must reproduce
                // the greedy path exactly.
                forced_continuation = greedy_ids;
                build_forced_steps(decoder.prefill(prompt, kProbeMaxTokens).last_logits);
                for (size_t k = 0; k < forced_steps.size(); ++k) {
                    int32_t top = 0;
                    for (size_t i = 1; i < forced_steps[k].size(); ++i) {
                        if (forced_steps[k][i] > forced_steps[k][top]) {
                            top = static_cast<int32_t>(i);
                        }
                    }
                    if (top != greedy_ids[k]) {
                        throw std::runtime_error(
                            "parity probe forced replay diverges at step " + std::to_string(k));
                    }
                }
            }
        }
        generated_text = want_generate
            ? g2b::granite_speech5_decode_generated(*tokenizer, greedy_ids)
            : std::string();
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
    if (want_prefill) {
        write_f32(std::filesystem::path(out_dir) / "merged_prefill.f32", decoder_prefill.merged);
        write_f32(std::filesystem::path(out_dir) / "scaled_prefill.f32", decoder_prefill.scaled);
        // Reference last_pos_logits.f32 is the raw lm_head output: the
        // reference hooks language_model.lm_head itself, which fires BEFORE
        // GraniteForCausalLM divides by logits_scaling (verified bitwise:
        // last_pos_logits == 8 * forced_step0 on all fixtures). The C++
        // decoder carries the post-scale logits (identical to forced_step0),
        // so this stage emits the exact x8 inverse (exact in fp32).
        auto raw_logits = decoder_prefill.last_logits;
        for (float & value : raw_logits) {
            value *= 8.0F;
        }
        write_f32(std::filesystem::path(out_dir) / "last_pos_logits.f32", raw_logits);
    }
    if (want_generate) {
        for (size_t k = 0; k < forced_steps.size(); ++k) {
            write_f32(
                std::filesystem::path(out_dir) / ("forced_step" + std::to_string(k) + ".f32"),
                forced_steps[k]);
        }
    }

    // trace.json mirrors reference.py keys; thresholds are the reference
    // PIPELINE entries (the gate reads them from the reference side).
    const auto & lengths = features.lengths;
    const int64_t prompt_len =
        want_prefill ? static_cast<int64_t>(expanded_prompt.input_ids.size()) : 0;
    const int64_t generated_len =
        want_generate ? static_cast<int64_t>(greedy_ids.size()) : 0;
    const std::string observed_stop =
        !want_generate ? "" : (generated_len < kProbeMaxTokens ? "eos" : "budget");
    std::ostringstream json;
    json << "{\n";
    json << "  \"family\": \"granite_speech5\",\n";
    json << "  \"revision\": \"" << kPinnedRevision << "\",\n";
    json << "  \"model_sha256\": \"" << model_sha << "\",\n";
    json << "  \"audio_sha256\": \"" << audio_sha << "\",\n";
    json << "  \"runtime\": {\"impl\": \"audio.cpp parity probe\", \"threads\": " << threads << "},\n";
    json << "  \"backend\": " << json_escape(backend) << ",\n";
    json << "  \"attn_implementation\": " << json_escape(has_flag(argc, argv, "--eager-attention") ? "eager" : "flash") << ",\n";
    json << "  \"dtype\": \"float32\",\n";
    json << "  \"threads\": " << threads << ",\n";
    json << "  \"decoder_input_override\": " << json_escape(decoder_input) << ",\n";
    json << "  \"through\": " << json_escape(through) << ",\n";
    json << "  \"sample_rate\": 16000,\n";
    json << "  \"audio\": {\"path\": " << json_escape(audio_path) << ", \"num_samples\": " << num_samples
         << ", \"span\": [0, " << num_samples << "]},\n";
    json << "  \"lengths\": {\"mel_frames\": " << lengths.mel_frames << ", \"encoder_steps\": "
         << lengths.encoder_steps << ", \"windows\": " << lengths.projector_windows << ", \"audio_tokens\": "
         << lengths.audio_placeholders << ", \"prompt_len\": " << prompt_len << ", \"generated_len\": "
         << generated_len << "},\n";
    json << "  \"prompt\": " << json_escape(prompt_text) << ",\n";
    json << "  \"language_hint\": " << (language_hint.empty() ? "null" : json_escape(language_hint)) << ",\n";
    json << "  \"input_ids\": " << (want_prefill ? ids_json(expanded_prompt.input_ids) : "[]") << ",\n";
    json << "  \"max_tokens\": " << kProbeMaxTokens << ",\n";
    json << "  \"generated_ids\": " << (want_generate ? ids_json(greedy_ids) : "[]") << ",\n";
    json << "  \"observed_stop\": " << json_escape(observed_stop) << ",\n";
    json << "  \"eos_token_id\": " << assets->config.text.eos_token_id << ",\n";
    json << "  \"text\": " << json_escape(generated_text) << ",\n";
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
         << "}}\n";
    json << "  },\n";
    // load_info key names mirror from_pretrained(output_loading_info=True).
    json << "  \"load_info\": {\"missing_keys\": [], \"unexpected_keys\": [], "
            "\"mismatched_keys\": [], \"error_msgs\": []},\n";
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
        {"merged_prefill",
         "merged_prefill.f32",
         {prompt_len, 2048},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"scaled_prefill",
         "scaled_prefill.f32",
         {prompt_len, 2048},
         "\"max_abs\": null, \"rel_rms\": 0.001"},
        {"last_pos_logits",
         "last_pos_logits.f32",
         {assets->config.text.vocab_size},
         "\"max_abs\": 0.02, \"rel_rms\": 0.001"},
    };
    // Forced cached steps (generate only): forced_step{k} predicts
    // continuation[k] through the single-token decode cache.
    std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, const char *>> all_stages = stages;
    for (size_t k = 0; k < forced_steps.size(); ++k) {
        all_stages.push_back(
            {"forced_step" + std::to_string(k),
             "forced_step" + std::to_string(k) + ".f32",
             {assets->config.text.vocab_size},
             "\"max_abs\": 0.02, \"rel_rms\": 0.001"});
    }
    size_t stage_count = 0;
    if (through == "frontend") {
        stage_count = 2;
    } else if (through == "encoder") {
        stage_count = 7;
    } else if (through == "projector") {
        stage_count = 8;
    } else if (through == "prefill") {
        stage_count = 11;
    } else {
        stage_count = all_stages.size();
    }
    for (size_t i = 0; i < stage_count; ++i) {
        const auto & [name, file, shape, thresholds] = all_stages[i];
        json << "    {\"name\": " << json_escape(name) << ", \"file\": " << json_escape(file) << ", \"shape\": "
             << shape_json(shape) << ", \"dtype\": \"float32\", " << thresholds << "}"
             << (i + 1 == stage_count ? "\n" : ",\n");
    }
    json << "  ],\n";
    // Generate cutoff derived the same way as reference.py: n_steps is
    // min(N_FORCED_STEPS, continuation length). For probe generate runs the
    // continuation is the forced list (reference IDs or own greedy output).
    const size_t gen_steps = want_generate ? forced_steps.size()
                           : static_cast<size_t>(kProbeForcedSteps < kProbeMaxTokens ? kProbeForcedSteps
                                                                                      : kProbeMaxTokens);
    const std::string gen_cutoff =
        gen_steps > 0 ? "forced_step" + std::to_string(gen_steps - 1) : "last_pos_logits";
    json << "  \"cutoffs\": {\"frontend\": \"stacked\", \"encoder\": \"encoder_last\", "
            "\"projector\": \"projected\", \"prefill\": \"last_pos_logits\", "
            "\"generate\": \"" + gen_cutoff + "\"}\n";
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
        engine::debug::configure_logging({has_flag(argc, argv, "--log")});
        const auto start = std::chrono::steady_clock::now();
        const int result = run(argc, argv);
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(start));
        return result;
    } catch (const std::exception & error) {
        std::cerr << "granite_speech5_parity_probe failed: " << error.what() << "\n";
        return 1;
    }
}
