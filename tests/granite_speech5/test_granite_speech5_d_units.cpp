#include "engine/framework/debug/profiler.h"
// granite_speech5 D unit checks: session option validation, chunk planning,
// JP-aware join, generation budget math, loader contract, plus
// checkpoint-gated session runs (golden transcripts, A/B/A error recovery,
// cache isolation, VAD planning, native/f32 parity).
//
// Always-run (no checkpoint): option resolution, fixed planner + tail merge,
// join rules, budget edges, loader contract negatives, task-mismatch guard.
// Checkpoint-gated (GRANITE_SPEECH5_MODEL;
// skips loudly otherwise): loader can_load/load, golden en/ja transcripts,
// A/B/A recovery with a failed request between, different-length cache
// isolation, VAD planning on longer audio, native-vs-f32 transcript parity.

#include "engine/community_models/granite_speech5/decoder.h"
#include "engine/community_models/granite_speech5/session.h"
#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"
#include "test_assert.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace test = engine::test;
namespace g2b = engine::community_models::granite_speech5;
namespace runtime = engine::runtime;

class ScopedWorkingDirectory {
public:
    explicit ScopedWorkingDirectory(const std::filesystem::path & dir)
        : previous_(std::filesystem::current_path()) {
        std::filesystem::current_path(dir);
    }
    ~ScopedWorkingDirectory() {
        std::error_code ec;
        std::filesystem::current_path(previous_, ec);
    }

private:
    std::filesystem::path previous_;
};

#ifdef AUDIOCPP_SOURCE_DIR
std::filesystem::path source_dir() {
    return std::filesystem::path(AUDIOCPP_SOURCE_DIR);
}
#else
std::filesystem::path source_dir() {
    return std::filesystem::current_path();
}
#endif

std::optional<std::filesystem::path> find_model_dir() {
    if (const char * env = std::getenv("GRANITE_SPEECH5_MODEL")) {
        const std::filesystem::path candidate(env);
        if (std::filesystem::exists(candidate / "model.safetensors")) {
            return candidate;
        }
    }

    return std::nullopt;
}

runtime::TaskRequest request_with_options(
    std::unordered_map<std::string, std::string> options,
    const std::string & text_language = "") {
    runtime::TaskRequest request;
    request.options = std::move(options);
    if (!text_language.empty()) {
        request.text_input = runtime::Transcript{"", text_language};
    }
    return request;
}

void require_spans_eq(
    const std::vector<runtime::TimeSpan> & actual,
    const std::vector<std::pair<int64_t, int64_t>> & expected,
    const std::string & label) {
    test::require_eq(actual.size(), expected.size(), label + " span count");
    for (size_t i = 0; i < expected.size(); ++i) {
        test::require_eq(actual[i].start_sample, expected[i].first, label + " start");
        test::require_eq(actual[i].end_sample, expected[i].second, label + " end");
    }
}

void require_strings_eq(
    const std::vector<std::string> & actual,
    const std::vector<std::string> & expected,
    const std::string & label) {
    test::require_eq(actual.size(), expected.size(), label + " size");
    for (size_t i = 0; i < expected.size(); ++i) {
        test::require_eq(actual[i], expected[i], label + "[" + std::to_string(i) + "]");
    }
}

void test_language_resolution() {
    // Explicit option wins, case-insensitive + trimmed.
    auto resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"language", "EN"}}, "ja"));
    test::require_eq(resolved.language, "en", "explicit option wins");
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"language", "  Ja "}}));
    test::require_eq(resolved.language, "ja", "trim+lowercase");
    // text_input fallback.
    resolved = g2b::granite_speech5_resolve_request_options(request_with_options({}, "fr"));
    test::require_eq(resolved.language, "fr", "text_input fallback");
    // Blank = auto, from either source.
    resolved = g2b::granite_speech5_resolve_request_options(request_with_options({{"language", " "}}));
    test::require_eq(resolved.language, "", "blank option is auto");
    resolved = g2b::granite_speech5_resolve_request_options(request_with_options({}));
    test::require_eq(resolved.language, "", "missing is auto");
    // All supported codes normalize.
    for (const char * code : {"en", "fr", "de", "es", "pt", "ja"}) {
        resolved = g2b::granite_speech5_resolve_request_options(
            request_with_options({{"language", code}}));
        test::require_eq(resolved.language, code, std::string("code ") + code);
    }
    // Unsupported and non-ASCII reject loudly.
    for (const char * bad : {"xx", "eng", "jp", "en-US", "zh", "auto"}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_resolve_request_options(
                request_with_options({{"language", bad}}));
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, std::string("reject language ") + bad);
    }
    bool threw = false;
    try {
        (void) g2b::granite_speech5_resolve_request_options(
            request_with_options({{"language", "j\xC3\xA4"}}));
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "reject non-ASCII language");
}

void test_max_tokens_resolution() {
    auto resolved = g2b::granite_speech5_resolve_request_options(request_with_options({}));
    test::require_eq(resolved.max_tokens, 512, "max_tokens default");
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"max_tokens", "1"}}));
    test::require_eq(resolved.max_tokens, 1, "max_tokens min");
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"max_tokens", "16383"}}));
    test::require_eq(resolved.max_tokens, 16383, "max_tokens max");
    for (const char * bad : {"0", "-5", "16384", "abc", "1.5"}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_resolve_request_options(
                request_with_options({{"max_tokens", bad}}));
        } catch (const std::exception &) {
            threw = true;
        }
        test::require(threw, std::string("reject max_tokens ") + bad);
    }
}

void test_chunk_option_resolution() {
    auto resolved = g2b::granite_speech5_resolve_request_options(request_with_options({}));
    test::require(
        resolved.chunk_mode == engine::audio::AudioChunkMode::Auto, "chunk mode default auto");
    test::require_eq(resolved.chunk_duration_sec, 120.0F, "chunk duration default");
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"audio_chunk_mode", "fixed"}, {"audio_chunk_duration_sec", "4.5"}}));
    test::require(
        resolved.chunk_mode == engine::audio::AudioChunkMode::Fixed, "fixed mode");
    test::require_eq(resolved.chunk_duration_sec, 4.5F, "duration override");
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"audio_chunk_mode", "vad"}}));
    test::require(resolved.chunk_mode == engine::audio::AudioChunkMode::Vad, "vad mode");
    // none ignores the duration override entirely.
    resolved = g2b::granite_speech5_resolve_request_options(
        request_with_options({{"audio_chunk_mode", "none"}, {"audio_chunk_duration_sec", "999"}}));
    test::require(resolved.chunk_mode == engine::audio::AudioChunkMode::None, "none mode");
    test::require_eq(resolved.chunk_duration_sec, 120.0F, "none ignores duration");
    for (const char * bad_mode : {"turbo", "quiet_energy", "silence"}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_resolve_request_options(
                request_with_options({{"audio_chunk_mode", bad_mode}}));
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(threw, std::string("reject chunk mode ") + bad_mode);
    }
    for (const char * bad_dur : {"0", "-1", "121", "inf", "nan", "abc"}) {
        bool threw = false;
        try {
            (void) g2b::granite_speech5_resolve_request_options(
                request_with_options({{"audio_chunk_duration_sec", bad_dur}}));
        } catch (const std::exception &) {
            threw = true;
        }
        test::require(threw, std::string("reject duration ") + bad_dur);
    }
}

void test_fixed_planner_and_tail_merge() {
    constexpr int64_t kRate = 16000;
    constexpr int64_t kChunk = 120 * kRate;  // 120 s default cap
    // Whole short input, exact-cap input.
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            g2b::granite_speech5_plan_fixed_spans(56080, kChunk), kChunk),
        {{0, 56080}},
        "short whole");
    // Exact multiples stay cap-aligned.
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            g2b::granite_speech5_plan_fixed_spans(2 * kChunk, kChunk), kChunk),
        {{0, kChunk}, {kChunk, 2 * kChunk}},
        "exact multiple");
    // Fixed 4 s chunks over the 14.2 s librispeech clip: last span in-bounds.
    constexpr int64_t kFourSec = 4 * kRate;
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            g2b::granite_speech5_plan_fixed_spans(227200, kFourSec), kFourSec),
        {{0, 64000}, {64000, 128000}, {128000, 192000}, {192000, 227200}},
        "fixed 4 s over 14.2 s");
    // Tiny tail joins the previous span; grown span stays in-bounds.
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            g2b::granite_speech5_plan_fixed_spans(kChunk + 100, kChunk), kChunk),
        {{0, (kChunk + 100 + 1) / 2}, {(kChunk + 100 + 1) / 2, kChunk + 100}},
        "tiny tail rebalance");
    // Tail merge that overflows the cap rebalances the last two.
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            g2b::granite_speech5_plan_fixed_spans(2 * kChunk + 100, kChunk), kChunk),
        {{0, kChunk + 50}, {kChunk + 50, 2 * kChunk + 100}},
        "over-cap tail rebalances last two");
    // VAD tiny spans merge forward; a lone blip drops to no spans.
    require_spans_eq(
        g2b::granite_speech5_merge_tiny_tails(
            {{{1000, 1100}, {5000, 9000}}}, kChunk, true),
        {{1000, 9000}},
        "vad tiny merges forward");
    test::require(
        g2b::granite_speech5_merge_tiny_tails({{{1000, 1100}}}, kChunk, true).empty(),
        "vad lone blip drops");
    // Lone tiny whole input rejects; empty spans reject.
    for (bool vad_ok : {false, true}) {
        if (!vad_ok) {
            bool threw = false;
            try {
                (void) g2b::granite_speech5_merge_tiny_tails({{{0, 100}}}, kChunk, vad_ok);
            } catch (const std::runtime_error &) {
                threw = true;
            }
            test::require(threw, "lone tiny whole input rejects");
        }
    }
    bool threw = false;
    try {
        (void) g2b::granite_speech5_plan_fixed_spans(0, kChunk);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "planner rejects empty input");
    threw = false;
    try {
        (void) g2b::granite_speech5_merge_tiny_tails({{{5, 5}}}, kChunk, false);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "merge rejects empty span");
}

void test_join_rules() {
    std::string merged;
    g2b::granite_speech5_append_chunk_text(merged, "  Concord returned ");
    test::require_eq(merged, "Concord returned", "trim first");
    g2b::granite_speech5_append_chunk_text(merged, "to its place");
    test::require_eq(merged, "Concord returned to its place", "latin space");
    g2b::granite_speech5_append_chunk_text(merged, "amidst the tents,");
    test::require_eq(merged, "Concord returned to its place amidst the tents,", "punctuation");
    g2b::granite_speech5_append_chunk_text(merged, "");
    test::require_eq(merged, "Concord returned to its place amidst the tents,", "empty chunk");
    // Japanese neighbors concatenate without injected spaces.
    std::string jp;
    g2b::granite_speech5_append_chunk_text(jp, "日本語で書かれた");
    g2b::granite_speech5_append_chunk_text(jp, "本を読みます。");
    test::require_eq(jp, "日本語で書かれた本を読みます。", "jp concat");
    // Mixed-script boundaries concatenate; latin-latin keeps the space.
    std::string mixed;
    g2b::granite_speech5_append_chunk_text(mixed, "hello");
    g2b::granite_speech5_append_chunk_text(mixed, "世界");
    g2b::granite_speech5_append_chunk_text(mixed, "test");
    test::require_eq(mixed, "hello世界test", "mixed concat");
    std::string latin = "a";
    g2b::granite_speech5_append_chunk_text(latin, "b");
    test::require_eq(latin, "a b", "latin pair");
}

void test_budget_edges() {
    // Session enforcement: prompt_len + max_tokens <= 16384, both positive.
    g2b::granite_speech5_require_generation_budget(51, 512);
    g2b::granite_speech5_require_generation_budget(1, 16383);
    g2b::granite_speech5_require_generation_budget(270, 16384 - 270);
    for (const auto & [prompt_len, max_tokens] : std::vector<std::pair<int64_t, int64_t>>{
             {0, 512}, {51, 0}, {51, -1}, {270, 16384 - 270 + 1}, {16384, 1}}) {
        bool threw = false;
        try {
            g2b::granite_speech5_require_generation_budget(prompt_len, max_tokens);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        test::require(
            threw,
            "budget rejects prompt=" + std::to_string(prompt_len) +
                " max=" + std::to_string(max_tokens));
    }
}

void test_loader_contract() {
    const auto loader = g2b::make_granite_speech5_loader();
    test::require_eq(loader->family(), "granite_speech5", "family id");
    test::require(loader->family_aliases().empty(), "no aliases");
    const auto advertised = loader->advertised_capabilities();
    test::require_eq(advertised.supported_tasks.size(), 1U, "one task");
    test::require(
        advertised.supported_tasks[0].task == runtime::VoiceTaskKind::Asr, "asr task");
    test::require_eq(advertised.supported_tasks[0].modes.size(), 1U, "offline only");
    test::require(
        advertised.supported_tasks[0].modes[0] == runtime::RunMode::Offline, "offline mode");
    require_strings_eq(
        advertised.languages,
        {"en", "fr", "de", "es", "pt", "ja"},
        "languages");
    test::require(!advertised.supports_timestamps, "no word timings");

    runtime::ModelLoadRequest request;
    request.model_path = "/nonexistent/checkpoint";
    test::require(!loader->can_load(request), "missing dir does not load");
    request.family_hint = "granite5asr";
    test::require(!loader->can_load(request), "wrong hint does not load");
    request.family_hint = "granite_speech5";
    request.model_path = "/nonexistent/checkpoint.gguf";
    test::require(!loader->can_load(request), "missing gguf does not load");

    // Load fails closed: bad family hint, GGUF path, bad storage, bad dir.
    request.family_hint = "granite5asr";
    request.model_path = "/nonexistent/checkpoint";
    bool threw = false;
    try {
        (void) loader->load(request);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "load rejects wrong hint");
    request.family_hint = "granite_speech5";
    request.model_path = "/nonexistent/model.gguf";
    threw = false;
    try {
        (void) loader->load(request);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "load rejects missing gguf");
    request.model_path = "/nonexistent/checkpoint";
    request.options = {{"granite_speech5.weight_type", "q8_0"}};
    threw = false;
    try {
        (void) loader->load(request);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "load rejects unsupported storage");
    test::require(!loader->can_load(request), "bad dir does not load");
    threw = false;
    try {
        (void) loader->inspect(request);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "inspect rejects bad dir");

    // Task mismatch fails before any weight access (fake assets suffice).
    auto assets = std::make_shared<const g2b::GraniteSpeech5Assets>();
    g2b::GraniteSpeech5LoadedModel model(
        runtime::ModelMetadata{}, runtime::CapabilitySet{}, assets,
        engine::assets::TensorStorageType::Native);
    threw = false;
    try {
        (void) model.create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "vad task rejected");
    threw = false;
    try {
        (void) model.create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Asr, runtime::RunMode::Streaming},
            runtime::SessionOptions{});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "streaming rejected");
}

void test_native_weight_guard() {
    const auto loader = g2b::make_granite_speech5_loader();
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "g2b_native_weight_guard_test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);
    test::require(!ec, "fixture dir created");
    test::require(
        !g2b::granite_speech5_directory_has_native_weights(tmp), "empty dir has no native weights");
    test::require(
        !g2b::granite_speech5_directory_has_native_weights(tmp / "missing"),
        "missing dir has no native weights");
    {
        std::ofstream stub(tmp / "model.safetensors", std::ios::binary);
        test::require(static_cast<bool>(stub), "weight stub created");
    }
    test::require(
        g2b::granite_speech5_directory_has_native_weights(tmp), "safetensors stub detected");

    // A foreign native checkpoint directory holding a stray GGUF must not be
    // claimed, even though the GGUF embeds a valid g2b spec. The GGUF is
    // linked (not copied) from the E1 dist artifact; skip loudly when the
    // gated dist file is absent.
    const std::filesystem::path foreign = tmp / "foreign_native";
    std::filesystem::create_directories(foreign, ec);
    test::require(!ec, "foreign dir created");
    {
        std::ofstream config(foreign / "config.json", std::ios::binary);
        config << R"({"model_type": "granite_speech5_ctc"})";
    }
    {
        std::ofstream stub(foreign / "model.safetensors", std::ios::binary);
        test::require(static_cast<bool>(stub), "foreign weight stub created");
    }
    runtime::ModelLoadRequest foreign_request;
    foreign_request.model_path = foreign;
    test::require(!loader->can_load(foreign_request), "foreign native dir without gguf not claimed");

    const char * gguf_path = std::getenv("GRANITE_SPEECH5_GGUF");
    const std::filesystem::path dist_gguf = gguf_path ? gguf_path : "";
    if (gguf_path == nullptr) {
        std::cout << "NOTE: stray-GGUF claim check requires GRANITE_SPEECH5_GGUF\n";
    } else {
        std::filesystem::create_symlink(dist_gguf, foreign / "stray.gguf", ec);
        test::require(!ec, "stray gguf linked");
        test::require(
            g2b::granite_speech5_directory_has_native_weights(foreign),
            "foreign weights detected with stray gguf present");
        test::require(
            !loader->can_load(foreign_request), "foreign native dir with stray gguf not claimed");
    }
    std::filesystem::remove_all(tmp, ec);
}

// --- checkpoint-gated helpers ---

runtime::AudioBuffer read_mono_buffer(const std::filesystem::path & path) {
    const auto wav = engine::audio::read_wav_f32(path);
    runtime::AudioBuffer audio;
    audio.sample_rate = wav.sample_rate;
    audio.channels = wav.channels;
    audio.samples = wav.samples;
    return audio;
}

std::unique_ptr<runtime::IOfflineVoiceTaskSession> make_session(
    const std::filesystem::path & model_dir,
    std::unordered_map<std::string, std::string> session_options = {},
    std::unordered_map<std::string, std::string> load_options = {}) {
    runtime::ModelLoadRequest load_request;
    load_request.model_path = model_dir;
    load_request.family_hint = "granite_speech5";
    load_request.options = std::move(load_options);
    auto registry = runtime::make_default_registry();
    auto model = registry.load(load_request);
    runtime::SessionOptions options;
    options.backend.type = engine::core::BackendType::Cpu;
    options.backend.threads = 8;
    options.options = std::move(session_options);
    auto session = model->create_task_session(
        runtime::TaskSpec{runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline}, options);
    auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
    if (offline == nullptr) {
        throw std::runtime_error("session is not offline");
    }
    session.release();
    return std::unique_ptr<runtime::IOfflineVoiceTaskSession>(offline);
}

std::string transcribe_buffer(
    runtime::IOfflineVoiceTaskSession & session,
    const runtime::AudioBuffer & audio,
    std::unordered_map<std::string, std::string> options = {}) {
    runtime::TaskRequest request;
    request.audio_input = audio;
    request.options = std::move(options);
    runtime::SessionPreparationRequest preparation;
    session.prepare(preparation);
    auto result = session.run(request);
    if (!result.text_output.has_value()) {
        throw std::runtime_error("missing text_output");
    }
    return result.text_output->text;
}

void test_gated_flows(const std::filesystem::path & model_dir) {
    const auto librispeech = read_mono_buffer(
        source_dir() / "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
    const auto librispeech_long = read_mono_buffer(
        source_dir() / "assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0001.wav");
    const auto japanese = read_mono_buffer(
        source_dir() / "assets/asr_validation/common_voice_ja/common_voice_ja_20461197.wav");

    // Loader agreement: can_load + load + inspect on the native directory.
    const auto loader = g2b::make_granite_speech5_loader();
    runtime::ModelLoadRequest probe_request;
    probe_request.model_path = model_dir;
    test::require(loader->can_load(probe_request), "native dir auto-detect");
    probe_request.family_hint = "granite_speech5";
    test::require(loader->can_load(probe_request), "explicit family");
    // Hint-free registry routing lands here (this loader precedes granite5asr,
    // whose can_load also matches model_type granite_speech5).
    {
        runtime::ModelLoadRequest auto_request;
        auto_request.model_path = model_dir;
        auto auto_model = runtime::make_default_registry().load(auto_request);
        test::require_eq(auto_model->metadata().family, "granite_speech5", "registry auto-detect");
    }
    const auto inspection = loader->inspect(probe_request);
    test::require_eq(inspection.metadata.family, "granite_speech5", "inspect family");
    test::require(!inspection.discovered_weights.empty(), "inspect discovers weights");

    auto session = make_session(model_dir);

    // Golden transcripts (reference trace texts): short en + short ja, auto.
    const std::string english = transcribe_buffer(*session, librispeech);
    test::require_eq(
        english, "Concord returned to its place amidst the tents,", "librispeech golden");
    const std::string japanese_text = transcribe_buffer(*session, japanese, {{"language", "ja"}});
    test::require(!japanese_text.empty(), "japanese hint produces text");
    const std::string japanese_auto = transcribe_buffer(*session, japanese);
    test::require_eq(japanese_auto, "日本語で書かれた本を読みます。", "ja golden");

    // A/B/A: failed requests (unsupported language, over-budget max_tokens)
    // between valid ones leave the lease usable with identical output.
    bool threw = false;
    try {
        (void) transcribe_buffer(*session, librispeech, {{"language", "xx"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "unsupported language fails");
    test::require_eq(transcribe_buffer(*session, librispeech), english, "A/B/A language");
    threw = false;
    try {
        (void) transcribe_buffer(*session, librispeech, {{"max_tokens", "16383"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "over-budget max_tokens fails");
    test::require_eq(transcribe_buffer(*session, librispeech), english, "A/B/A budget");

    // Cache isolation: different lengths (and hint on/off) stay sane across
    // a failed request, with no position carryover.
    const std::string long_text = transcribe_buffer(*session, librispeech_long);
    test::require(!long_text.empty(), "longer clip transcribes");
    threw = false;
    try {
        (void) transcribe_buffer(*session, japanese, {{"audio_chunk_mode", "silence"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "unsupported mode fails");
    test::require_eq(transcribe_buffer(*session, japanese), japanese_auto, "isolation after failure");
    test::require_eq(transcribe_buffer(*session, librispeech), english, "isolation back to short");
    test::require(!transcribe_buffer(*session, librispeech_long).empty(), "long again");

    // Native vs F32 storage transcript parity on the ja clip.
    auto f32_session = make_session(
        model_dir, {}, {{"granite_speech5.weight_type", "f32"}});
    test::require_eq(
        transcribe_buffer(*f32_session, japanese), japanese_auto, "f32 storage parity");

    // VAD planning on the longer clip with the repo silero weights, plus the
    // explicit-vad requirement and the no-VAD fixed fallback.
    const std::filesystem::path silero =
        source_dir() / "assets/framework/models/silero_vad/silero_vad_16k.safetensors";
    if (std::filesystem::is_regular_file(silero)) {
        auto vad_session = make_session(
            model_dir, {{"granite_speech5.vad_model_path", silero.string()}});
        const std::string vad_text = transcribe_buffer(
            *vad_session,
            librispeech_long,
            {{"audio_chunk_mode", "vad"}, {"audio_chunk_duration_sec", "5"}});
        test::require(!vad_text.empty(), "vad chunking transcribes");
        const std::string auto_text = transcribe_buffer(
            *vad_session,
            librispeech_long,
            {{"audio_chunk_mode", "auto"}, {"audio_chunk_duration_sec", "5"}});
        test::require(!auto_text.empty(), "auto with vad transcribes");
    } else {
        std::cout << "NOTE: silero VAD file missing; skipping D VAD planning checks\n";
    }
    threw = false;
    try {
        (void) transcribe_buffer(
            *session, librispeech, {{"audio_chunk_mode", "vad"},
                                    {"granite_speech5.vad_model_path", "/nonexistent/vad.safetensors"}});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    test::require(threw, "explicit vad requires a VAD file");
    // Auto without a VAD file falls back to fixed chunking on long input.
    auto no_vad_session = make_session(
        model_dir, {{"granite_speech5.vad_model_path", "/nonexistent/vad.safetensors"}});
    test::require(
        !transcribe_buffer(
             *no_vad_session, librispeech_long,
             {{"audio_chunk_mode", "auto"}, {"audio_chunk_duration_sec", "5"}})
             .empty(),
        "auto falls back to fixed");
}

}  // namespace

int main(int argc, char ** argv) {
    engine::debug::configure_logging({argc > 1 && std::string(argv[1]) == "--log"});
    const auto started = std::chrono::steady_clock::now();
    try {
        test_language_resolution();
        test_max_tokens_resolution();
        test_chunk_option_resolution();
        test_fixed_planner_and_tail_merge();
        test_join_rules();
        test_budget_edges();
        test_loader_contract();
        test_native_weight_guard();
        if (const auto model_dir = find_model_dir()) {
            // Spec discovery and the default VAD path resolve against the
            // working directory in dev builds; run gated flows from the repo
            // root like the CLI/server verification does.
            ScopedWorkingDirectory repo_root(source_dir());
            test_gated_flows(*model_dir);
        } else {
            std::cout << "NOTE: D checkpoint parity skipped (set GRANITE_SPEECH5_MODEL)\n";
        }
    } catch (const std::exception & error) {
        std::cerr << "FAIL: granite_speech5 D unit checks: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS: granite_speech5 D unit checks\n";
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
}
