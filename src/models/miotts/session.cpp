#include "engine/models/miotts/session.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/wavlm_encoder.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"
#include "engine/framework/text/utf8.h"
#include "engine/models/miocodec/assets.h"
#include "engine/models/miocodec/audio_pipeline.h"
#include "engine/models/miocodec/components.h"
#include "engine/models/miocodec/weights.h"
#include "engine/models/miotts/causal_lm.h"
#include "engine/models/miotts/tokenizer.h"
#include "engine/models/qwen3_asr/loader.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <limits>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::miotts {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultLmWeightContextBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultLmPrefillGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultLmDecodeGraphArenaBytes = 1024ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecWeightContextBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultGlobalGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kDefaultWaveGraphArenaBytes = 512ull * 1024ull * 1024ull;
constexpr size_t kDefaultCodecConstantContextBytes = 256ull * 1024ull * 1024ull;
constexpr int64_t kMioTTSTokenRateHz = 25;
constexpr int64_t kMioCodecWaveHeadBins = 394;
constexpr double kBestOfNTokenRateHz = 25.0;
constexpr double kBestOfNSilenceRatioThreshold = 0.2;
constexpr double kBestOfNLongSilenceThresholdSec = 2.0;
constexpr double kBestOfNLengthWeight = 0.4;
constexpr double kBestOfNSilenceWeight = 0.4;
constexpr double kBestOfNRepeatWeight = 0.2;
constexpr double kBestOfNHeuristicWeight = 0.3;
constexpr int64_t kReferenceTextChunkCodepoints = 180;

std::shared_ptr<const MioTTSAssets> require_assets(std::shared_ptr<const MioTTSAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MioTTS session requires assets");
    }
    return assets;
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " supports only native, f32, f16, bf16, and q8_0");
}

std::filesystem::path default_codec_model_path(const std::filesystem::path & miotts_root) {
    const auto sibling = miotts_root.parent_path() / "MioCodec-25Hz-44.1kHz-v2";
    return sibling;
}

std::filesystem::path default_asr_model_path(const std::filesystem::path & miotts_root) {
    return miotts_root.parent_path() / "Qwen3-ASR-0.6B";
}

std::filesystem::path resolve_codec_model_path(const runtime::SessionOptions & options, const MioTTSAssets & assets) {
    if (const auto value = runtime::find_option(options.options, {"codec_model"})) {
        return std::filesystem::path(*value);
    }
    return default_codec_model_path(assets.resources.model_root());
}

std::filesystem::path resolve_best_of_n_asr_model_path(
    const runtime::SessionOptions & options,
    const MioTTSAssets & assets) {
    if (const auto value = runtime::find_option(
            options.options,
            {"best_of_n_asr_model"})) {
        return std::filesystem::path(*value);
    }
    return default_asr_model_path(assets.resources.model_root());
}

std::string normalized_best_of_n_language(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "ja" || value == "en" || value == "auto") {
        return value;
    }
    throw std::runtime_error("MioTTS best_of_n_language must be auto, en, or ja");
}

int clamp_best_of_n(int n, int max_n) {
    if (n <= 0) {
        throw std::runtime_error("MioTTS best_of_n must be positive");
    }
    if (max_n <= 0) {
        throw std::runtime_error("MioTTS best_of_n_max must be positive");
    }
    return std::min(n, max_n);
}

MioTTSGenerationOptions generation_options_from_request(
    const MioTTSConfig & config,
    const runtime::TaskRequest & request) {
    MioTTSGenerationOptions out;
    out.max_tokens = config.max_tokens;
    out.top_k = config.top_k;
    out.top_p = config.top_p;
    out.temperature = config.temperature;
    out.repetition_penalty = config.repetition_penalty;
    out.presence_penalty = config.presence_penalty;
    out.frequency_penalty = config.frequency_penalty;
    out.do_sample = config.do_sample;
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.max_tokens = *value;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"top_k"})) {
        out.top_k = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"top_p"})) {
        out.top_p = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"repetition_penalty"})) {
        out.repetition_penalty = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"presence_penalty"})) {
        out.presence_penalty = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"frequency_penalty"})) {
        out.frequency_penalty = *value;
    }
    out.seed = runtime::parse_u32_option(request.options, {"seed"})
        .value_or(runtime::random_u32_seed());
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        out.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (out.max_tokens <= 0) {
        throw std::runtime_error("MioTTS max_tokens must be positive");
    }
    if (out.temperature < 0.0F || out.temperature > 2.0F) {
        throw std::runtime_error("MioTTS temperature must be in [0, 2]");
    }
    if (out.top_p < 0.0F || out.top_p > 1.0F) {
        throw std::runtime_error("MioTTS top_p must be in [0, 1]");
    }
    if (out.repetition_penalty < 1.0F || out.repetition_penalty > 1.5F) {
        throw std::runtime_error("MioTTS repetition_penalty must be in [1, 1.5]");
    }
    if (out.presence_penalty < 0.0F || out.presence_penalty > 1.0F) {
        throw std::runtime_error("MioTTS presence_penalty must be in [0, 1]");
    }
    if (out.frequency_penalty < 0.0F || out.frequency_penalty > 1.0F) {
        throw std::runtime_error("MioTTS frequency_penalty must be in [0, 1]");
    }
    return out;
}

struct ResolvedBestOfN {
    bool enabled = false;
    int n = 1;
    std::string language = "auto";
};

ResolvedBestOfN best_of_n_from_request(
    const runtime::TaskRequest & request,
    bool session_enabled,
    int session_default,
    int session_max,
    const std::string & session_language) {
    ResolvedBestOfN out;
    out.enabled = session_enabled;
    out.n = clamp_best_of_n(session_default, session_max);
    out.language = normalized_best_of_n_language(session_language);
    if (const auto value = runtime::parse_int_option(request.options, {"best_of_n"})) {
        out.n = clamp_best_of_n(*value, session_max);
        out.enabled = out.n > 1;
    }
    if (const auto value = runtime::find_option(request.options, {"best_of_n_enabled"})) {
        out.enabled = runtime::parse_bool_option(*value, "best_of_n_enabled");
    }
    if (const auto value = runtime::find_option(request.options, {"best_of_n_language"})) {
        out.language = normalized_best_of_n_language(*value);
    }
    if (!out.enabled) {
        out.n = 1;
    }
    return out;
}

int64_t stft_frames_for_codec_tokens(
    const miocodec::MioCodecConfig & config,
    int64_t tokens) {
    if (tokens <= 0) {
        throw std::runtime_error("MioTTS generated no codec tokens");
    }
    if (config.sample_rate % kMioTTSTokenRateHz != 0) {
        throw std::runtime_error("MioTTS codec sample rate is not divisible by token rate");
    }
    const int upsample_factor = std::accumulate(
        config.wave_upsampler_factors.begin(),
        config.wave_upsampler_factors.end(),
        1,
        [](int lhs, int rhs) {
            return lhs * rhs;
        });
    const int64_t samples = tokens * (config.sample_rate / kMioTTSTokenRateHz);
    const int64_t divisor = static_cast<int64_t>(config.hop_length) * upsample_factor;
    if (samples % divisor != 0) {
        throw std::runtime_error("MioTTS codec token length does not align with wave decoder STFT frames");
    }
    return samples / divisor;
}

struct MioTTSBestOfNCandidate {
    MioTTSGeneratedTokens generated;
    std::vector<float> audio;
    std::string asr_text;
    double asr_error = 1.0;
    double repeat_penalty = 0.0;
    double length_penalty = 0.0;
    double silence_penalty = 0.0;
    double score = std::numeric_limits<double>::infinity();
};

bool is_ascii_alpha(unsigned char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
}

bool is_ascii_alnum_or_apostrophe(unsigned char ch) {
    return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '\'';
}

std::string lowercase_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::vector<uint32_t> utf8_codepoints(std::string_view text, std::string_view label) {
    std::vector<uint32_t> out;
    out.reserve(engine::text::utf8_codepoint_count(text, label));
    for (size_t pos = 0; pos < text.size();) {
        const auto ch = static_cast<unsigned char>(text[pos]);
        size_t width = 0;
        uint32_t codepoint = 0;
        if (ch <= 0x7FU) {
            width = 1;
            codepoint = ch;
        } else if ((ch & 0xE0U) == 0xC0U) {
            width = 2;
            codepoint = ch & 0x1FU;
        } else if ((ch & 0xF0U) == 0xE0U) {
            width = 3;
            codepoint = ch & 0x0FU;
        } else if ((ch & 0xF8U) == 0xF0U) {
            width = 4;
            codepoint = ch & 0x07U;
        } else {
            throw std::runtime_error(std::string(label) + " contains invalid UTF-8");
        }
        if (pos + width > text.size()) {
            throw std::runtime_error(std::string(label) + " contains truncated UTF-8");
        }
        for (size_t i = 1; i < width; ++i) {
            const auto cont = static_cast<unsigned char>(text[pos + i]);
            if (!engine::text::is_utf8_continuation(cont)) {
                throw std::runtime_error(std::string(label) + " contains invalid UTF-8 continuation byte");
            }
            codepoint = (codepoint << 6U) | (cont & 0x3FU);
        }
        out.push_back(codepoint);
        pos += width;
    }
    return out;
}

bool is_japanese_codepoint(uint32_t codepoint) {
    return (codepoint >= 0x3040U && codepoint <= 0x309FU) ||
        (codepoint >= 0x30A0U && codepoint <= 0x30FFU) ||
        (codepoint >= 0x4E00U && codepoint <= 0x9FFFU) ||
        (codepoint >= 0x3400U && codepoint <= 0x4DBFU) ||
        (codepoint >= 0xF900U && codepoint <= 0xFAFFU);
}

std::string detect_best_of_n_language(const std::string & text) {
    const auto codepoints = utf8_codepoints(text, "MioTTS best_of_n text");
    int64_t total = 0;
    int64_t ja_count = 0;
    int64_t en_count = 0;
    for (const uint32_t cp : codepoints) {
        if (cp == ' ' || cp == '\n' || cp == '\r' || cp == '\t') {
            continue;
        }
        ++total;
        if (is_japanese_codepoint(cp)) {
            ++ja_count;
        }
        if (cp <= 0x7FU && is_ascii_alpha(static_cast<unsigned char>(cp))) {
            ++en_count;
        }
    }
    if (total == 0) {
        return "auto";
    }
    if (static_cast<double>(ja_count) / static_cast<double>(total) >= 0.2) {
        return "ja";
    }
    if (static_cast<double>(en_count) / static_cast<double>(total) >= 0.5) {
        return "en";
    }
    return "auto";
}

std::string resolve_best_of_n_language(const std::string & text, const std::string & requested) {
    if (requested == "en" || requested == "ja") {
        return requested;
    }
    return detect_best_of_n_language(text);
}

double ngram_repeat_ratio(const std::vector<int32_t> & tokens, size_t n) {
    if (tokens.size() < n) {
        return 0.0;
    }
    const size_t total = tokens.size() - n + 1;
    std::unordered_map<std::string, int> counts;
    counts.reserve(total);
    for (size_t i = 0; i < total; ++i) {
        std::string key;
        key.reserve(n * 8);
        for (size_t j = 0; j < n; ++j) {
            key += std::to_string(tokens[i + j]);
            key.push_back(',');
        }
        counts[key] += 1;
    }
    return 1.0 - (static_cast<double>(counts.size()) / static_cast<double>(total));
}

double repeat_penalty(const std::vector<int32_t> & tokens) {
    return std::max({ngram_repeat_ratio(tokens, 2), ngram_repeat_ratio(tokens, 3), ngram_repeat_ratio(tokens, 4)});
}

int64_t approximate_phoneme_count(const std::string & text, const std::string & language) {
    if (language == "en") {
        int64_t count = 0;
        for (const unsigned char ch : text) {
            if (is_ascii_alpha(ch) || (ch >= '0' && ch <= '9')) {
                ++count;
            }
        }
        return std::max<int64_t>(count, 1);
    }
    return std::max<int64_t>(static_cast<int64_t>(engine::text::utf8_codepoint_count(text, "MioTTS best_of_n text")), 1);
}

double punctuation_bonus_seconds(const std::string & text) {
    double bonus = 0.0;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (ch == '.' && i + 2 < text.size() && text[i + 1] == '.' && text[i + 2] == '.') {
            bonus += 1.0;
            i += 2;
        } else if (ch == ',' || ch == ';' || ch == ':') {
            bonus += 0.20;
        } else if (ch == '.' || ch == '!' || ch == '?') {
            bonus += 0.40;
        } else if (ch == '-' && i + 1 < text.size() && text[i + 1] == '-') {
            bonus += 0.12;
            ++i;
        }
    }
    if (!text.empty()) {
        const char last = text.back();
        if ((last == '.' || last == '!' || last == '?') && bonus >= 0.40) {
            bonus -= 0.40;
        }
    }
    return std::min(10.0, bonus);
}

double length_penalty(const std::string & text, const std::vector<int32_t> & tokens, const std::string & language) {
    const double duration_sec = static_cast<double>(tokens.size()) / kBestOfNTokenRateHz;
    const int64_t phonemes = approximate_phoneme_count(text, language);
    double min_spp = 0.07;
    double max_spp = 0.18;
    if (language == "en") {
        min_spp = 0.06;
        max_spp = 0.12;
    } else if (language == "ja") {
        min_spp = 0.07;
        max_spp = 0.15;
    }
    const double bonus = punctuation_bonus_seconds(text);
    const double min_expected = static_cast<double>(phonemes) * min_spp + bonus;
    const double max_expected = static_cast<double>(phonemes) * max_spp + bonus;
    if (duration_sec <= 0.0 || min_expected <= 0.0) {
        return 0.0;
    }
    if (duration_sec < min_expected) {
        return (min_expected - duration_sec) / min_expected;
    }
    if (duration_sec > max_expected) {
        return (duration_sec - max_expected) / max_expected;
    }
    return 0.0;
}

std::pair<double, double> silence_stats(const std::vector<float> & audio, int sample_rate) {
    if (audio.empty()) {
        return {1.0, 0.0};
    }
    const size_t frame_size = std::max<size_t>(1, static_cast<size_t>(static_cast<double>(sample_rate) * 0.02));
    const size_t frames = audio.size() / frame_size;
    if (frames == 0) {
        double energy = 0.0;
        for (const float sample : audio) {
            energy += std::abs(static_cast<double>(sample));
        }
        energy /= static_cast<double>(audio.size());
        return {energy < 1e-4 ? 1.0 : 0.0, static_cast<double>(audio.size()) / static_cast<double>(sample_rate)};
    }
    size_t silent_frames = 0;
    size_t longest = 0;
    size_t current = 0;
    for (size_t frame = 0; frame < frames; ++frame) {
        double energy = 0.0;
        const size_t offset = frame * frame_size;
        for (size_t i = 0; i < frame_size; ++i) {
            energy += std::abs(static_cast<double>(audio[offset + i]));
        }
        energy /= static_cast<double>(frame_size);
        const bool silent = energy < 1e-4;
        if (silent) {
            ++silent_frames;
            ++current;
            longest = std::max(longest, current);
        } else {
            current = 0;
        }
    }
    return {
        static_cast<double>(silent_frames) / static_cast<double>(frames),
        static_cast<double>(longest * frame_size) / static_cast<double>(sample_rate),
    };
}

double silence_penalty(const std::vector<float> & audio, int sample_rate) {
    const auto [ratio, longest] = silence_stats(audio, sample_rate);
    double penalty = 0.0;
    if (ratio > kBestOfNSilenceRatioThreshold) {
        penalty += (ratio - kBestOfNSilenceRatioThreshold) / (1.0 - kBestOfNSilenceRatioThreshold);
    }
    if (longest > kBestOfNLongSilenceThresholdSec) {
        penalty += (longest - kBestOfNLongSilenceThresholdSec) / kBestOfNLongSilenceThresholdSec;
    }
    return penalty;
}

std::vector<std::string> normalize_for_wer(const std::string & text) {
    const auto lowered = lowercase_ascii(text);
    std::vector<std::string> words;
    std::string current;
    for (const unsigned char ch : lowered) {
        if (is_ascii_alnum_or_apostrophe(ch)) {
            current.push_back(static_cast<char>(ch));
        } else if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    }
    if (!current.empty()) {
        words.push_back(std::move(current));
    }
    return words;
}

std::vector<uint32_t> normalize_for_cer(const std::string & text) {
    std::vector<uint32_t> out;
    for (const uint32_t cp : utf8_codepoints(lowercase_ascii(text), "MioTTS best_of_n ASR text")) {
        if ((cp <= 0x7FU && std::isalnum(static_cast<unsigned char>(cp)) != 0) || is_japanese_codepoint(cp)) {
            out.push_back(cp);
        }
    }
    return out;
}

template <typename T>
int64_t edit_distance(const std::vector<T> & ref, const std::vector<T> & hyp) {
    if (ref.empty()) {
        return static_cast<int64_t>(hyp.size());
    }
    if (hyp.empty()) {
        return static_cast<int64_t>(ref.size());
    }
    std::vector<int64_t> dp(hyp.size() + 1);
    std::iota(dp.begin(), dp.end(), 0);
    for (size_t i = 0; i < ref.size(); ++i) {
        int64_t prev = dp[0];
        dp[0] = static_cast<int64_t>(i + 1);
        for (size_t j = 0; j < hyp.size(); ++j) {
            const int64_t temp = dp[j + 1];
            const int64_t cost = ref[i] == hyp[j] ? 0 : 1;
            dp[j + 1] = std::min({dp[j + 1] + 1, dp[j] + 1, prev + cost});
            prev = temp;
        }
    }
    return dp.back();
}

double asr_error(const std::string & reference, const std::string & hypothesis, const std::string & language) {
    if (language == "en") {
        const auto ref = normalize_for_wer(reference);
        const auto hyp = normalize_for_wer(hypothesis);
        if (ref.empty()) {
            return hyp.empty() ? 0.0 : 1.0;
        }
        return static_cast<double>(edit_distance(ref, hyp)) / static_cast<double>(ref.size());
    }
    const auto ref = normalize_for_cer(reference);
    const auto hyp = normalize_for_cer(hypothesis);
    if (ref.empty()) {
        return hyp.empty() ? 0.0 : 1.0;
    }
    return static_cast<double>(edit_distance(ref, hyp)) / static_cast<double>(ref.size());
}

}  // namespace

MioTTSSession::MioTTSSession(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MioTTSAssets> assets)
    : runtime::RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))) {
    if (task_.task != runtime::VoiceTaskKind::Tts) {
        throw std::runtime_error("MioTTS only supports VoiceTaskKind::Tts");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MioTTS currently supports offline sessions");
    }
    const auto lm_weight_type = option_weight_type(options, "miotts.weight_type", engine::assets::TensorStorageType::Native);
    validate_matmul_weight_storage(lm_weight_type, "miotts.weight_type");
    text_chunk_size_ = engine::text::parse_text_chunk_size_override(options.options)
        .value_or(kReferenceTextChunkCodepoints);
    if (const auto value = runtime::find_option(options.options, {"best_of_n_enabled"})) {
        best_of_n_enabled_ = runtime::parse_bool_option(*value, "best_of_n_enabled");
    }
    best_of_n_default_ = runtime::parse_int_option(options.options, {"best_of_n_default"})
        .value_or(1);
    best_of_n_max_ = runtime::parse_int_option(options.options, {"best_of_n_max"})
        .value_or(8);
    if (best_of_n_default_ <= 0) {
        throw std::runtime_error("MioTTS best_of_n_default must be positive");
    }
    if (best_of_n_max_ <= 0) {
        throw std::runtime_error("MioTTS best_of_n_max must be positive");
    }
    if (best_of_n_default_ > best_of_n_max_) {
        best_of_n_default_ = best_of_n_max_;
    }
    best_of_n_language_ = normalized_best_of_n_language(
        runtime::find_option(options.options, {"best_of_n_language"}).value_or("auto"));
    best_of_n_asr_model_path_ = resolve_best_of_n_asr_model_path(options, *assets_);
    tokenizer_ = std::make_unique<MioTTSTokenizer>(assets_);
    language_model_ = std::make_unique<MioTTSCausalLMRuntime>(
        assets_,
        execution_context(),
        runtime::parse_size_mb_option(options.options, {"miotts.prefill_graph_arena_mb"}, kDefaultLmPrefillGraphArenaBytes),
        runtime::parse_size_mb_option(options.options, {"miotts.decode_graph_arena_mb"}, kDefaultLmDecodeGraphArenaBytes),
        runtime::parse_size_mb_option(options.options, {"miotts.weight_context_mb"}, kDefaultLmWeightContextBytes),
        lm_weight_type);
    assets_->model_weights->release_storage();

    const auto codec_model_path = resolve_codec_model_path(options, *assets_);
    engine::debug::trace_log_scalar("miotts.codec_model_path", codec_model_path.string());
    codec_assets_ = miocodec::load_miocodec_assets(codec_model_path);
    codec_weights_ = miocodec::load_miocodec_weights(
        *codec_assets_->model_weights,
        execution_context(),
        runtime::parse_size_mb_option(options.options, {"miotts.codec_weight_context_mb"}, kDefaultCodecWeightContextBytes),
        codec_assets_->config);
    engine::modules::WavlmEncoderConfig wavlm_config;
    wavlm_config.output_hidden_layer = 9;
    wavlm_config.weight_storage_type = lm_weight_type;
    auto wavlm = engine::modules::WavlmEncoderComponent::load_from_tensor_source(
        *codec_assets_->wavlm_weights,
        options.backend,
        wavlm_config);
    const size_t constant_context_bytes = runtime::parse_size_mb_option(
        options.options,
        {"miotts.codec_constant_context_mb"},
        kDefaultCodecConstantContextBytes);
    global_encoder_ = std::make_unique<miocodec::MioCodecGlobalEncoderRuntime>(
        codec_assets_,
        codec_weights_,
        execution_context(),
        runtime::parse_size_mb_option(options.options, {"miotts.global_graph_arena_mb"}, kDefaultGlobalGraphArenaBytes));
    global_reference_ = std::make_unique<miocodec::MioCodecGlobalReferenceEncoder>(
        miocodec::MioCodecSslFeatureExtractor(
            codec_assets_,
            std::move(wavlm),
            codec_assets_->config.global_ssl_layers,
            false,
            "miotts.global_ssl"),
        *global_encoder_);
    wave_decoder_ = std::make_unique<miocodec::MioCodecWaveDecoderRuntime>(
        codec_assets_,
        codec_weights_,
        execution_context(),
        runtime::parse_size_mb_option(options.options, {"miotts.wave_graph_arena_mb"}, kDefaultWaveGraphArenaBytes),
        constant_context_bytes);
    waveform_reconstructor_ = std::make_unique<miocodec::MioCodecWaveformReconstructor>(
        codec_assets_,
        execution_context());
}

MioTTSSession::~MioTTSSession() = default;

std::string MioTTSSession::family() const {
    return "miotts";
}

runtime::VoiceTaskKind MioTTSSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MioTTSSession::run_mode() const {
    return task_.mode;
}

void MioTTSSession::prepare(const runtime::SessionPreparationRequest &) {
    mark_prepared();
}

runtime::IOfflineVoiceTaskSession & MioTTSSession::best_of_n_asr_session() {
    if (best_of_n_asr_session_ == nullptr) {
        engine::debug::trace_log_scalar("miotts.best_of_n.asr_model_path", best_of_n_asr_model_path_.string());
        best_of_n_asr_model_ = qwen3_asr::load_qwen3_asr_model(best_of_n_asr_model_path_);
        runtime::TaskSpec asr_task{runtime::VoiceTaskKind::Asr, runtime::RunMode::Offline};
        best_of_n_asr_session_ = best_of_n_asr_model_->create_task_session(asr_task, options());
    }
    auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(best_of_n_asr_session_.get());
    if (offline == nullptr) {
        throw std::runtime_error("MioTTS best-of-n ASR model did not create an offline ASR session");
    }
    if (!best_of_n_asr_prepared_) {
        runtime::SessionPreparationRequest prepare;
        prepare.audio = runtime::AudioPreparationContract{codec_assets_->config.sample_rate, 1, 0};
        best_of_n_asr_session_->prepare(prepare);
        best_of_n_asr_prepared_ = true;
    }
    return *offline;
}

runtime::TaskResult MioTTSSession::run(const runtime::TaskRequest & request) {
    const auto wall_start = Clock::now();
    require_prepared("MioTTS run");
    if (!request.text_input.has_value() || request.text_input->text.empty()) {
        throw std::runtime_error("MioTTS run() requires text_input");
    }
    const auto text_length = static_cast<int64_t>(
        engine::text::utf8_codepoint_count(request.text_input->text, "MioTTS text_input"));
    const auto text_chunk_size_override = engine::text::parse_text_chunk_size_override(request.options);
    engine::debug::trace_log_scalar("miotts.text_length", text_length);
    if (!request.voice.has_value() || !request.voice->speaker.has_value() ||
        !request.voice->speaker->audio.has_value()) {
        throw std::runtime_error("MioTTS run() requires voice speaker audio");
    }
    auto timing_start = Clock::now();
    const auto reference_audio = miocodec::prepare_miocodec_mono_audio(
        *request.voice->speaker->audio,
        codec_assets_->config.sample_rate);
    engine::debug::timing_log_scalar("miotts.audio.reference_prepare_ms", engine::debug::elapsed_ms(timing_start));
    const auto generation = generation_options_from_request(assets_->config, request);
    engine::debug::trace_log_scalar("miotts.sampling.max_tokens", generation.max_tokens);
    engine::debug::trace_log_scalar("miotts.sampling.temperature", generation.temperature);
    engine::debug::trace_log_scalar("miotts.sampling.top_p", generation.top_p);
    engine::debug::trace_log_scalar("miotts.sampling.repetition_penalty", generation.repetition_penalty);
    engine::debug::trace_log_scalar("miotts.sampling.presence_penalty", generation.presence_penalty);
    engine::debug::trace_log_scalar("miotts.sampling.frequency_penalty", generation.frequency_penalty);
    const auto best_of_n = best_of_n_from_request(
        request,
        best_of_n_enabled_,
        best_of_n_default_,
        best_of_n_max_,
        best_of_n_language_);
    const auto text_chunks = text_chunk_size_override.has_value()
        ? engine::text::split_text_chunks(request.text_input->text, *text_chunk_size_override)
        : engine::text::split_text_chunks(request.text_input->text, text_chunk_size_);
    engine::debug::trace_log_scalar("miotts.best_of_n.enabled", best_of_n.enabled);
    engine::debug::trace_log_scalar("miotts.best_of_n.n", best_of_n.n);
    engine::debug::trace_log_scalar(
        "miotts.text_chunk_size",
        text_chunk_size_override.value_or(text_chunk_size_));
    engine::debug::trace_log_scalar("miotts.text_chunk_count", static_cast<int64_t>(text_chunks.size()));
    for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
        const std::string prefix = "miotts.text_chunk_" + std::to_string(chunk_index);
        engine::debug::trace_log_scalar(prefix + ".text", text_chunks[chunk_index]);
        engine::debug::trace_log_scalar(
            prefix + ".length",
            static_cast<int64_t>(engine::text::utf8_codepoint_count(text_chunks[chunk_index], "MioTTS text chunk")));
    }
    timing_start = Clock::now();
    const auto global = global_reference_->embedding_for_reference(reference_audio);
    engine::debug::timing_log_scalar("miotts.global_reference_ms", engine::debug::elapsed_ms(timing_start));
    const engine::audio::STFTConfig stft_config{
        codec_assets_->config.n_fft,
        codec_assets_->config.hop_length,
        codec_assets_->config.n_fft,
        true,
        engine::audio::STFTPadMode::Reflect,
        engine::audio::STFTFamily::Kokoro,
    };
    const auto & stft_window = engine::audio::get_cached_stft_window(stft_config);
    double prompt_ms = 0.0;
    double lm_ms = 0.0;
    double wave_decoder_ms = 0.0;
    double istft_ms = 0.0;
    double best_of_n_asr_ms = 0.0;
    double best_of_n_score_ms = 0.0;
    int64_t total_prompt_tokens = 0;
    int64_t total_generated_tokens = 0;
    std::vector<float> output_samples;
    for (size_t chunk_index = 0; chunk_index < text_chunks.size(); ++chunk_index) {
        const std::string chunk_prefix = "miotts.chunk_" + std::to_string(chunk_index);
        const std::string scoring_language = resolve_best_of_n_language(text_chunks[chunk_index], best_of_n.language);
        engine::debug::trace_log_scalar(chunk_prefix + ".best_of_n.language", scoring_language);
        timing_start = Clock::now();
        const auto prompt = tokenizer_->build_prompt(text_chunks[chunk_index]);
        prompt_ms += engine::debug::elapsed_ms(timing_start);
        total_prompt_tokens += static_cast<int64_t>(prompt.input_ids.size());

        std::vector<MioTTSBestOfNCandidate> candidates;
        candidates.reserve(static_cast<size_t>(best_of_n.n));
        timing_start = Clock::now();
        for (int candidate_index = 0; candidate_index < best_of_n.n; ++candidate_index) {
            auto candidate_generation = generation;
            if (best_of_n.enabled && best_of_n.n > 1) {
                candidate_generation.seed = generation.seed +
                    static_cast<uint32_t>(chunk_index * static_cast<size_t>(best_of_n.n) + static_cast<size_t>(candidate_index));
            }
            engine::debug::trace_log_scalar(
                chunk_prefix + ".candidate_" + std::to_string(candidate_index) + ".seed",
                candidate_generation.seed);
            MioTTSBestOfNCandidate candidate;
            candidate.generated = language_model_->generate(prompt, candidate_generation);
            engine::debug::trace_log_scalar(
                chunk_prefix + ".candidate_" + std::to_string(candidate_index) + ".generated_tokens",
                static_cast<int64_t>(candidate.generated.codec_tokens.size()));
            if (!candidate.generated.codec_tokens.empty()) {
                candidates.push_back(std::move(candidate));
            }
        }
        lm_ms += engine::debug::elapsed_ms(timing_start);
        if (candidates.empty()) {
            throw std::runtime_error("MioTTS generated no usable speech-token candidates");
        }

        for (auto & candidate : candidates) {
            const int64_t stft_frames = stft_frames_for_codec_tokens(
                codec_assets_->config,
                static_cast<int64_t>(candidate.generated.codec_tokens.size()));
            timing_start = Clock::now();
            const auto head = wave_decoder_->decode_tokens(candidate.generated.codec_tokens, global, stft_frames);
            wave_decoder_ms += engine::debug::elapsed_ms(timing_start);
            if (head.bins != kMioCodecWaveHeadBins) {
                throw std::runtime_error("MioTTS wave head bin count mismatch");
            }
            timing_start = Clock::now();
            candidate.audio = waveform_reconstructor_->reconstruct(head, stft_window);
            istft_ms += engine::debug::elapsed_ms(timing_start);
        }

        size_t selected_index = 0;
        if (best_of_n.enabled && candidates.size() > 1) {
            const auto score_start = Clock::now();
            const double asr_ms_before_chunk = best_of_n_asr_ms;
            auto & asr = best_of_n_asr_session();
            double best_score = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < candidates.size(); ++i) {
                auto & candidate = candidates[i];
                candidate.repeat_penalty = repeat_penalty(candidate.generated.codec_tokens);
                candidate.length_penalty = length_penalty(text_chunks[chunk_index], candidate.generated.codec_tokens, scoring_language);
                candidate.silence_penalty = silence_penalty(candidate.audio, codec_assets_->config.sample_rate);
                runtime::TaskRequest asr_request;
                asr_request.audio_input = runtime::AudioBuffer{
                    codec_assets_->config.sample_rate,
                    1,
                    candidate.audio,
                };
                if (scoring_language == "en") {
                    asr_request.text_input = runtime::Transcript{"", "English"};
                } else if (scoring_language == "ja") {
                    asr_request.text_input = runtime::Transcript{"", "Japanese"};
                }
                const auto asr_start = Clock::now();
                const auto asr_result = asr.run(asr_request);
                best_of_n_asr_ms += engine::debug::elapsed_ms(asr_start);
                if (!asr_result.text_output.has_value()) {
                    throw std::runtime_error("MioTTS best-of-n ASR did not produce text_output");
                }
                candidate.asr_text = asr_result.text_output->text;
                candidate.asr_error = asr_error(text_chunks[chunk_index], candidate.asr_text, scoring_language);
                const double heuristic =
                    kBestOfNLengthWeight * candidate.length_penalty +
                    kBestOfNSilenceWeight * candidate.silence_penalty +
                    kBestOfNRepeatWeight * candidate.repeat_penalty;
                candidate.score = candidate.asr_error + kBestOfNHeuristicWeight * heuristic;
                const std::string prefix = chunk_prefix + ".candidate_" + std::to_string(i);
                engine::debug::trace_log_scalar(prefix + ".asr_text", candidate.asr_text);
                engine::debug::trace_log_scalar(prefix + ".asr_error", candidate.asr_error);
                engine::debug::trace_log_scalar(prefix + ".length_penalty", candidate.length_penalty);
                engine::debug::trace_log_scalar(prefix + ".silence_penalty", candidate.silence_penalty);
                engine::debug::trace_log_scalar(prefix + ".repeat_penalty", candidate.repeat_penalty);
                engine::debug::trace_log_scalar(prefix + ".score", candidate.score);
                if (candidate.score < best_score) {
                    best_score = candidate.score;
                    selected_index = i;
                }
            }
            best_of_n_score_ms += engine::debug::elapsed_ms(score_start) - (best_of_n_asr_ms - asr_ms_before_chunk);
        }
        const auto & selected = candidates[selected_index];
        engine::debug::trace_log_scalar(chunk_prefix + ".best_of_n.selected_index", static_cast<int64_t>(selected_index));
        engine::debug::trace_log_scalar(
            chunk_prefix + ".generated_tokens",
            static_cast<int64_t>(selected.generated.codec_tokens.size()));
        total_generated_tokens += static_cast<int64_t>(selected.generated.codec_tokens.size());
        output_samples.insert(output_samples.end(), selected.audio.begin(), selected.audio.end());
    }
    engine::debug::timing_log_scalar("miotts.prompt_ms", prompt_ms);
    engine::debug::timing_log_scalar("miotts.lm_ms", lm_ms);
    engine::debug::trace_log_scalar("miotts.prompt_tokens", total_prompt_tokens);
    engine::debug::timing_log_scalar("miotts.wave_decoder_ms", wave_decoder_ms);
    engine::debug::timing_log_scalar("miotts.istft_ms", istft_ms);
    engine::debug::timing_log_scalar("miotts.best_of_n.asr_ms", best_of_n_asr_ms);
    engine::debug::timing_log_scalar("miotts.best_of_n.score_ms", std::max(0.0, best_of_n_score_ms));
    engine::debug::trace_log_scalar("miotts.generated_tokens", total_generated_tokens);
    timing_start = Clock::now();
    runtime::TaskResult result;
    result.audio_output = runtime::AudioBuffer{
        codec_assets_->config.sample_rate,
        1,
        std::move(output_samples),
    };
    engine::debug::timing_log_scalar("miotts.audio_result_ms", engine::debug::elapsed_ms(timing_start));
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

}  // namespace engine::models::miotts
