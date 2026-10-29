#pragma once

// Shared by the LFM3-Audio real-weight tests (test_lfm3_audio_{parity,asr,tts,s2s}).
//
// Arguments: --model <safetensors dir or GGUF> (default models/LFM3-Audio-2B), --fixtures <dir> (default
// models/LFM3-Audio-2B/fixtures, made by tools/community_models/lfm3_audio/make_fixtures.py),
// --backend cpu|best, --threads N, --weight-type f32|native (default f32). Skips with 125 without the model or
// the fixtures. Every expected value comes from the fixtures at run time; nothing derived from the
// checkpoint is compiled in.
//
// Tolerances (rel = ||got - want|| / ||want||, per stage):
// - F32 weights on the CPU: 1e-5, the codec waveform 1e-4; generated ids and codes identical. Encoder stages run on the
//   host's features 3e-5 (test_lfm3_audio_parity), the 418 s clip 1e-4.
// - F32 weights on a GPU (TF32 / half-tile matmuls): 5e-3, the waveform 1e-2; a greedy sequence may only diverge
//   at a near-tie.
// - Native bf16/f16: max(2e-2, 2x the stage's bf16-vs-fp32 noise floor from the fixture manifest).
// - q8_0: max(5e-2, 3x the floor). q4_k: numbers are reported, not checked.

#include "engine/community_models/lfm3_audio/debug.h"
#include "engine/community_models/lfm3_audio/session.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef ENGINE_REPO_ROOT
#define ENGINE_REPO_ROOT "."
#endif

namespace lfm3_model_test {

namespace lfm3 = engine::community_models::lfm3_audio;
namespace runtime = engine::runtime;
namespace json = engine::io::json;

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 125;
constexpr int kCodebooks = 12;
// A greedy flip on a GPU counts as a near-tie when the reference's top-2 logit gap is below this; GPU F32 logits
// are off by up to ~1e-2 absolute (TF32 / half tiles).
constexpr double kNearTieGap = 1e-2;
constexpr double kReportOnly = std::numeric_limits<double>::infinity();

enum class Profile { F32Cpu, F32Gpu, Native, Q8_0, Q4_K };

inline const char * profile_name(Profile profile) {
    switch (profile) {
        case Profile::F32Cpu: return "f32-cpu";
        case Profile::F32Gpu: return "f32-gpu";
        case Profile::Native: return "native";
        case Profile::Q8_0: return "q8_0";
        case Profile::Q4_K: return "q4_k";
    }
    return "?";
}

inline std::filesystem::path repo_path(const std::string & relative) {
    return std::filesystem::path(ENGINE_REPO_ROOT) / relative;
}

inline std::string arg_value(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

inline bool has_flag(int argc, char ** argv, const std::string & name) {
    return std::find(argv + 1, argv + argc, name) != argv + argc;
}

struct Context {
    std::filesystem::path model;
    std::filesystem::path fixtures;
    engine::core::BackendConfig backend;
    std::string weight_type;
    Profile profile = Profile::F32Cpu;
    json::Value manifest;
    // The tests call the loader and asset reader directly, which ignore request.model_spec_override.
    std::unique_ptr<engine::model_spec::ScopedSpecOverride> specs;
    std::unique_ptr<runtime::ILoadedVoiceModel> loaded;

    [[nodiscard]] bool strict_text() const { return profile != Profile::Q8_0 && profile != Profile::Q4_K; }
    [[nodiscard]] lfm3::Lfm3WeightOptions weight_options() const {
        lfm3::Lfm3WeightOptions out;
        out.type = weight_type == "f32" ? lfm3::Lfm3WeightType::F32 : lfm3::Lfm3WeightType::Native;
        return out;
    }

    // The bf16-vs-fp32 rel-RMS the fixture run measured for a stage, 0 when it has none.
    [[nodiscard]] double noise_floor(const std::string & key) const {
        const auto * floors = manifest.find("noise_floor");
        const auto * stage = floors != nullptr ? floors->find(key) : nullptr;
        const auto * rel = stage != nullptr ? stage->find("rel_rms") : nullptr;
        return rel != nullptr && rel->is_number() ? rel->as_number() : 0.0;
    }

    // `long_input`: the 418 s clip, where the fp32 reference itself is 1.15e-5 off an fp64 run, so 1e-5 cannot hold.
    [[nodiscard]] double tolerance(const std::string & floor_key, bool wave = false, bool long_input = false) const {
        switch (profile) {
            case Profile::F32Cpu: return wave || long_input ? 1e-4 : 1e-5;
            case Profile::F32Gpu: return wave ? 1e-2 : 5e-3;
            case Profile::Native: return std::max(2e-2, 2.0 * noise_floor(floor_key));
            case Profile::Q8_0: return std::max(5e-2, 3.0 * noise_floor(floor_key));
            case Profile::Q4_K: return kReportOnly;
        }
        return kReportOnly;
    }

    [[nodiscard]] std::string result_text(const std::string & key) const {
        return manifest.require("results").require(key).require("text").as_string();
    }

    [[nodiscard]] std::filesystem::path fixture(const std::string & name) const { return fixtures / name; }

    std::vector<float> f32(const std::string & name, std::vector<int64_t> * shape = nullptr) const {
        return lfm3::lfm3_read_npy_f32(fixtures / (name + ".npy"), shape);
    }
    std::vector<int32_t> i32(const std::string & name, std::vector<int64_t> * shape = nullptr) const {
        return lfm3::lfm3_read_npy_i32(fixtures / (name + ".npy"), shape);
    }

    runtime::ILoadedVoiceModel & model_handle() {
        if (loaded == nullptr) {
            runtime::ModelLoadRequest request;
            request.model_path = model;
            request.family_hint = "lfm3_audio";
            loaded = lfm3::make_lfm3_audio_loader()->load(request);
        }
        return *loaded;
    }

    template <typename Session>
    std::unique_ptr<Session> open(runtime::VoiceTaskKind kind, runtime::RunMode mode, std::unordered_map<std::string, std::string> extra = {}) {
        runtime::SessionOptions options;
        options.backend = backend;
        options.options = std::move(extra);
        options.options["lfm3_audio.weight_type"] = weight_type;
        if (kind == runtime::VoiceTaskKind::Asr) {
            options.options["lfm3_audio.vad_model_path"] = repo_path("assets/framework/models/silero_vad").string();
        }
        auto session = model_handle().create_task_session({kind, mode}, options);
        session->prepare({});
        auto * typed = dynamic_cast<Session *>(session.get());
        if (typed == nullptr) {
            throw std::runtime_error("unexpected LFM3-Audio session type");
        }
        session.release();
        return std::unique_ptr<Session>(typed);
    }
};

inline std::optional<Context> parse_context(int argc, char ** argv, const char * test, int & exit_code) {
    Context ctx;
    ctx.model = arg_value(argc, argv, "--model", repo_path("models/LFM3-Audio-2B").string());
    ctx.fixtures = arg_value(argc, argv, "--fixtures", repo_path("models/LFM3-Audio-2B/fixtures").string());
    const auto backend = arg_value(argc, argv, "--backend", "cpu");
    ctx.weight_type = arg_value(argc, argv, "--weight-type", "f32");
    const int threads = std::atoi(arg_value(argc, argv, "--threads", "8").c_str());
    if ((backend != "cpu" && backend != "best") || (ctx.weight_type != "f32" && ctx.weight_type != "native")) {
        std::cerr << "FAIL: --backend must be cpu or best, --weight-type f32 or native\n";
        exit_code = kExitFail;
        return std::nullopt;
    }

    if (!std::filesystem::exists(ctx.model) || !engine::io::is_existing_file(ctx.fixtures / "manifest.json")) {
        std::fprintf(stderr,
                     "SKIP: %s needs the LFM3-Audio model at '%s' and its fixtures at '%s' (--model, --fixtures).\n"
                     "      Fixtures: tools/community_models/lfm3_audio/make_fixtures.py\n",
                     test, ctx.model.string().c_str(), ctx.fixtures.string().c_str());
        exit_code = kExitSkip;
        return std::nullopt;
    }

    ctx.backend.type = backend == "cpu" ? engine::core::BackendType::Cpu : engine::core::BackendType::BestAvailable;
    ctx.backend.threads = threads > 0 ? threads : 1;
    std::string file = ctx.model.filename().string();
    std::transform(file.begin(), file.end(), file.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ctx.weight_type == "f32") {
        ctx.profile = backend == "cpu" ? Profile::F32Cpu : Profile::F32Gpu;
    } else if (file.find("q8_0") != std::string::npos) {
        ctx.profile = Profile::Q8_0;
    } else if (file.find("q4_k") != std::string::npos || file.find("q4_0") != std::string::npos) {  // any 4-bit: reported only
        ctx.profile = Profile::Q4_K;
    } else {
        ctx.profile = Profile::Native;
    }
    ctx.manifest = json::parse_file(ctx.fixtures / "manifest.json");
    ctx.specs = std::make_unique<engine::model_spec::ScopedSpecOverride>(repo_path("model_specs"), ctx.model);
    std::cout << test << ": model " << ctx.model.string() << ", backend " << backend << ", weights " << ctx.weight_type << ", profile "
              << profile_name(ctx.profile) << "\n";
    return ctx;
}

struct Diff {
    double rel = 0.0;
    double max_abs = 0.0;
};

inline Diff diff(const float * got, const float * want, size_t n) {
    double num = 0.0, den = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(got[i]) - want[i];
        num += d * d;
        den += static_cast<double>(want[i]) * want[i];
        max_abs = std::max(max_abs, std::fabs(d));
    }
    return {std::sqrt(num / std::max(den, 1e-30)), max_abs};
}

inline Diff diff(const std::vector<float> & got, const std::vector<float> & want) {
    return diff(got.data(), want.data(), std::min(got.size(), want.size()));
}

inline std::string sci(double value) {
    char out[32];
    std::snprintf(out, sizeof(out), "%.3e", value);
    return out;
}

class Checks {
public:
    void expect(bool ok, const std::string & label, const std::string & detail = "") {
        std::cout << (ok ? "PASS " : "FAIL ") << label << (detail.empty() ? "" : ": " + detail) << "\n";
        failures_ += ok ? 0 : 1;
    }

    void info(const std::string & label, const std::string & detail = "") {
        std::cout << "INFO " << label << (detail.empty() ? "" : ": " + detail) << "\n";
    }

    bool expect_rel(const Diff & d, double tolerance, const std::string & label, const std::string & extra = "") {
        const bool ok = d.rel <= tolerance;
        const std::string detail = "rel " + sci(d.rel) + " max_abs " + sci(d.max_abs) +
                                   (std::isinf(tolerance) ? std::string(" (report only)") : " (tol " + sci(tolerance) + ")") + extra;
        if (std::isinf(tolerance)) {
            info(label, detail);
        } else {
            expect(ok, label, detail);
        }
        return ok;
    }

    void expect_or_info(bool strict, bool ok, const std::string & label, const std::string & detail = "") {
        if (strict || ok) {
            expect(ok, label, detail);
        } else {
            info(label + " (report only, differs)", detail);
        }
    }

    [[nodiscard]] int failures() const { return failures_; }

    int finish(const char * test) const {
        std::cout << (failures_ == 0 ? "PASS" : "FAIL") << ": " << test << ", " << failures_ << " failed checks\n";
        return failures_ == 0 ? kExitPass : kExitFail;
    }

private:
    int failures_ = 0;
};

inline void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

inline void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

// Depth logits are dumped for 400 frames: the fixtures keep 40, and later frames give our own top-2 gaps.
class ScopedDumps {
public:
    explicit ScopedDumps(const std::string & name, const std::filesystem::path & orders = {})
        : dir_(std::filesystem::temp_directory_path() / ("lfm3_audio_model_test_" + name + "_" + std::to_string(std::random_device{}()))) {
        std::filesystem::remove_all(dir_);
        std::filesystem::create_directories(dir_);
        set_env("LFM3_AUDIO_DUMP_DIR", dir_.string());
        set_env("LFM3_AUDIO_DUMP_FRAMES", "400");
        if (!orders.empty()) {
            set_env("LFM3_AUDIO_MASKGIT_ORDERS", orders.string());
        }
    }
    ~ScopedDumps() {
        stop();
        std::error_code ignored;
        std::filesystem::remove_all(dir_, ignored);
    }
    ScopedDumps(const ScopedDumps &) = delete;
    ScopedDumps & operator=(const ScopedDumps &) = delete;

    void stop() {
        unset_env("LFM3_AUDIO_DUMP_DIR");
        unset_env("LFM3_AUDIO_DUMP_FRAMES");
        unset_env("LFM3_AUDIO_MASKGIT_ORDERS");
    }

    [[nodiscard]] bool has(const std::string & name) const { return std::filesystem::exists(dir_ / (name + ".npy")); }
    std::vector<float> f32(const std::string & name, std::vector<int64_t> * shape = nullptr) const {
        return lfm3::lfm3_read_npy_f32(dir_ / (name + ".npy"), shape);
    }
    std::vector<int32_t> i32(const std::string & name, std::vector<int64_t> * shape = nullptr) const {
        return lfm3::lfm3_read_npy_i32(dir_ / (name + ".npy"), shape);
    }

private:
    std::filesystem::path dir_;
};

inline runtime::AudioBuffer read_audio(const std::filesystem::path & path) {
    auto wav = engine::audio::read_wav_f32(path);
    return runtime::AudioBuffer{wav.sample_rate, wav.channels, std::move(wav.samples)};
}

inline runtime::TaskRequest audio_request(runtime::AudioBuffer audio, std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest out;
    out.audio_input = std::move(audio);
    out.options = std::move(options);
    return out;
}

inline runtime::TaskRequest text_request(const std::string & text, std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest out;
    out.text_input = runtime::Transcript{text, ""};
    out.options = std::move(options);
    return out;
}

inline std::string text_of(const runtime::TaskResult & result) {
    return result.text_output.has_value() ? result.text_output->text : std::string("<no text>");
}

inline std::vector<std::string> wer_words(const std::string & text) {
    std::vector<std::string> words;
    std::string word;
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        if (c >= 0x80 || std::isalnum(c) || c == '\'') {
            word.push_back(static_cast<char>(std::tolower(c)));
        } else if (!word.empty()) {
            words.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    return words;
}

inline double wer(const std::string & reference, const std::string & hypothesis) {
    const auto ref = wer_words(reference);
    const auto hyp = wer_words(hypothesis);
    std::vector<size_t> row(hyp.size() + 1);
    for (size_t j = 0; j <= hyp.size(); ++j) row[j] = j;
    for (size_t i = 1; i <= ref.size(); ++i) {
        size_t diagonal = row[0];
        row[0] = i;
        for (size_t j = 1; j <= hyp.size(); ++j) {
            const size_t up = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (ref[i - 1] == hyp[j - 1] ? 0 : 1)});
            diagonal = up;
        }
    }
    return static_cast<double>(row[hyp.size()]) / static_cast<double>(std::max<size_t>(ref.size(), 1));
}

// A text token, or a frame (token -1, 12 codes) as the generator's steps dump writes it.
struct Slot {
    int32_t token = -1;
    std::vector<int32_t> codes;
    bool operator==(const Slot & other) const { return token == other.token && codes == other.codes; }
};

inline std::vector<Slot> dumped_slots(const ScopedDumps & dumps) {
    std::vector<int64_t> shape;
    const auto rows = dumps.i32("steps", &shape);
    std::vector<Slot> out(static_cast<size_t>(shape.at(0)));
    for (size_t s = 0; s < out.size(); ++s) {
        const auto * row = rows.data() + s * (kCodebooks + 1);
        out[s].token = row[0];
        if (row[1] >= 0) out[s].codes.assign(row + 1, row + 1 + kCodebooks);
    }
    return out;
}

inline std::vector<Slot> fixture_slots(const Context & ctx, const std::string & tag) {
    const auto kind = ctx.i32("gen/" + tag + "_slot_kind");
    const auto token = ctx.i32("gen/" + tag + "_slot_token");
    const auto codes = ctx.i32("gen/" + tag + "_slot_codes");
    std::vector<Slot> out(kind.size());
    for (size_t s = 0; s < out.size(); ++s) {
        if (kind[s] == 1) {
            out[s].codes.assign(codes.begin() + static_cast<std::ptrdiff_t>(s * kCodebooks), codes.begin() + static_cast<std::ptrdiff_t>((s + 1) * kCodebooks));
        } else {
            out[s].token = token[s];
        }
    }
    return out;
}

// "T3 A5 ..." run-length pattern of the slot kinds.
inline std::string slot_pattern(const std::vector<Slot> & slots) {
    std::string out;
    for (size_t s = 0; s < slots.size();) {
        const bool frame = !slots[s].codes.empty();
        size_t e = s;
        while (e < slots.size() && slots[e].codes.empty() != frame) ++e;
        out += (out.empty() ? "" : " ") + std::string(frame ? "A" : "T") + std::to_string(e - s);
        s = e;
    }
    return out;
}

inline double top2_gap(const float * logits, size_t n) {
    float first = -std::numeric_limits<float>::infinity(), second = first;
    for (size_t i = 0; i < n; ++i) {
        if (logits[i] > first) {
            second = first;
            first = logits[i];
        } else if (logits[i] > second) {
            second = logits[i];
        }
    }
    return static_cast<double>(first) - second;
}

// Compares a greedy run with the fixture run `tag`: slots, MaskGIT orders and depth logits before the first difference.
// `drop_last` leaves out the fixture's trailing <|im_end|>, which TTS does not generate. Returns whether the slots are identical.
inline bool check_greedy_run(Checks & checks, const Context & ctx, const ScopedDumps & dumps, const std::string & tag, bool drop_last) {
    auto want = fixture_slots(ctx, tag);
    if (drop_last) want.pop_back();
    const auto got = dumped_slots(dumps);
    size_t first = 0;
    while (first < std::min(got.size(), want.size()) && got[first] == want[first]) ++first;
    const bool same = first == got.size() && got.size() == want.size();
    const std::string label = tag + " greedy slots";
    if (same) {
        checks.expect(true, label, "identical, " + std::to_string(got.size()) + " slots, " + slot_pattern(got));
    } else {
        std::string where = "first difference at slot " + std::to_string(first) + " of " + std::to_string(want.size()) + " (got " +
                            std::to_string(got.size()) + ", " + slot_pattern(got) + " vs " + slot_pattern(want) + ")";
        double gap = kReportOnly;
        if (first < want.size()) {
            const auto before = static_cast<size_t>(std::count_if(want.begin(), want.begin() + static_cast<std::ptrdiff_t>(first), [&](const Slot & s) {
                return s.codes.empty() == want[first].codes.empty();
            }));
            if (want[first].codes.empty()) {
                std::vector<int64_t> shape;
                const auto top = ctx.f32("gen/" + tag + "_text_top20_vals", &shape);
                if (before < static_cast<size_t>(shape[0])) gap = static_cast<double>(top[before * 20]) - top[before * 20 + 1];
                where += ", text step " + std::to_string(before);
            } else if (first < got.size() && !got[first].codes.empty()) {
                // The first codebook to differ in reveal order (later passes see it), on the raw picks (the reference's before EOS
                // normalization, ours the argmax of each dumped pass): an EOS frame is all end-of-audio codes whichever pick ended it.
                const auto order = ctx.i32("gen/" + tag + "_frame_order");
                const auto * row = order.data() + before * kCodebooks;
                std::vector<int64_t> shape;
                const auto logits = ctx.f32("gen/" + tag + "_frame_pass_logits", &shape);
                const size_t vocab = static_cast<size_t>(shape[3]);
                const auto raw = ctx.i32("gen/" + tag + "_frame_codes_raw");
                size_t at = 0;
                for (; at < kCodebooks; ++at) {
                    const auto name = "depth_logits_f" + std::to_string(before) + "_p" + std::to_string(at / 4);
                    if (!dumps.has(name)) break;  // MaskGIT stops after the pass that yields the EOS code
                    const auto mine = dumps.f32(name);
                    const float * r = mine.data() + static_cast<size_t>(row[at]) * vocab;
                    if (std::max_element(r, r + vocab) - r != raw[before * kCodebooks + static_cast<size_t>(row[at])]) break;
                }
                // Without the raw difference in the dumps, the normalized codes tell.
                if (at == kCodebooks || !dumps.has("depth_logits_f" + std::to_string(before) + "_p" + std::to_string(at / 4))) {
                    at = 0;
                    while (at < kCodebooks && got[first].codes[row[at]] == want[first].codes[row[at]]) ++at;
                    at = std::min<size_t>(at, kCodebooks - 1);
                }
                const auto k = static_cast<size_t>(row[at]);
                const size_t pass = at / 4;
                where += ", frame " + std::to_string(before) + " codebook " + std::to_string(k) + " pass " + std::to_string(pass);
                const bool has_ref = static_cast<int64_t>(before) < shape[0];
                const float * ref = has_ref ? logits.data() + ((before * 3 + pass) * kCodebooks + k) * vocab : nullptr;
                if (has_ref) gap = top2_gap(ref, vocab);
                const auto name = "depth_logits_f" + std::to_string(before) + "_p" + std::to_string(pass);
                if (dumps.has(name)) {
                    // The fixtures keep 40 frames of logits; past them only this run's own gap tells a near-tie.
                    const auto mine = dumps.f32(name);
                    const double ours = top2_gap(mine.data() + k * vocab, vocab);
                    where += ", our top-2 gap " + sci(ours) + (has_ref ? ", row max_abs vs reference " + sci(diff(mine.data() + k * vocab, ref, vocab).max_abs) : "");
                    if (!has_ref) gap = ours;
                }
            }
        }
        where += ", near-tie gap " + (std::isinf(gap) ? std::string("n/a") : sci(gap));
        if (ctx.profile == Profile::F32Cpu) {
            checks.expect(false, label, where);
        } else if (ctx.profile == Profile::F32Gpu) {
            checks.expect(gap < kNearTieGap, label + " diverge only at a near-tie (gap < " + sci(kNearTieGap) + ")", where);
        } else {
            checks.info(label + " (report only)", where);
        }
    }

    // The orders are injected for the fixture's frames; a longer run draws the rest.
    std::vector<int64_t> shape;
    const auto orders = dumps.i32("maskgit_orders", &shape);
    const auto reference = ctx.i32("gen/" + tag + "_frame_order");
    const size_t compared = std::min(orders.size(), reference.size());
    checks.expect(std::equal(orders.begin(), orders.begin() + static_cast<std::ptrdiff_t>(compared), reference.begin()),
                  tag + " MaskGIT orders are the injected ones", std::to_string(compared / kCodebooks) + " of " + std::to_string(shape.at(0)) + " frames");

    // Depth logits of the frames both runs reached with the same inputs.
    const auto frames_before = static_cast<size_t>(std::count_if(want.begin(), want.begin() + static_cast<std::ptrdiff_t>(first), [](const Slot & s) {
        return !s.codes.empty();
    }));
    const auto logits = ctx.f32("gen/" + tag + "_frame_pass_logits", &shape);
    const size_t vocab = static_cast<size_t>(shape[3]);
    double num = 0.0, den = 0.0, max_abs = 0.0;
    size_t passes = 0;
    for (size_t f = 0; f < std::min<size_t>(frames_before, static_cast<size_t>(shape[0])); ++f) {
        for (size_t p = 0; p < 3; ++p) {
            const auto name = "depth_logits_f" + std::to_string(f) + "_p" + std::to_string(p);
            if (!dumps.has(name)) continue;  // MaskGIT stops after the pass that yields the EOS code
            const auto mine = dumps.f32(name);
            const float * ref = logits.data() + (f * 3 + p) * kCodebooks * vocab;
            const auto d = diff(mine.data(), ref, kCodebooks * vocab);
            double ref_sq = 0.0;
            for (size_t i = 0; i < kCodebooks * vocab; ++i) ref_sq += static_cast<double>(ref[i]) * ref[i];
            num += d.rel * d.rel * ref_sq;
            den += ref_sq;
            max_abs = std::max(max_abs, d.max_abs);
            ++passes;
        }
    }
    if (passes > 0) {
        // Generation feeds the depth decoder this run's own backbone output, so only F32 is held to the stage tolerance.
        const bool f32 = ctx.profile == Profile::F32Cpu || ctx.profile == Profile::F32Gpu;
        checks.expect_rel({std::sqrt(num / std::max(den, 1e-30)), max_abs}, f32 ? ctx.tolerance("depth.logits") : kReportOnly,
                          tag + " depth logits in generation", ", " + std::to_string(passes) + " passes of " + std::to_string(frames_before) + " frames");
    } else if (frames_before > 0 && shape[0] > 0) {
        checks.expect(false, tag + " depth logits in generation", "no depth_logits dumps for " + std::to_string(frames_before) + " frames");
    }
    return same;
}

}  // namespace lfm3_model_test
