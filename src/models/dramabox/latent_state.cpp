#include "engine/models/dramabox/latent_state.h"

#include "engine/models/dramabox/audio_vae.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace engine::models::dramabox {
namespace {

constexpr double kCharsPerSecond = 14.0;
constexpr int64_t kAudioLatentDownsampleFactor = 4;
constexpr int64_t kBaseShiftAnchor = 1024;
constexpr int64_t kMaxShiftAnchor = 4096;
constexpr double kMaxShift = 2.05;
constexpr double kBaseShift = 0.95;
constexpr double kTerminalSigma = 0.1;

double count_regex_seconds(const std::string & text, std::string_view pattern, double seconds) {
    const std::regex re(std::string(pattern), std::regex_constants::icase);
    return static_cast<double>(std::distance(
        std::sregex_iterator(text.begin(), text.end(), re),
        std::sregex_iterator())) * seconds;
}

std::vector<std::string> regex_captures(const std::string & text, std::string_view pattern) {
    const std::regex re{std::string(pattern)};
    std::vector<std::string> out;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
        if (it->size() > 1) {
            out.push_back((*it)[1].str());
        }
    }
    return out;
}

int64_t word_count(const std::string & text) {
    const std::regex word_re("\\S+");
    return static_cast<int64_t>(std::distance(
        std::sregex_iterator(text.begin(), text.end(), word_re),
        std::sregex_iterator()));
}

double contextual_laugh_duration(const std::string & text) {
    const std::vector<std::pair<std::string, double>> laugh_verbs = {
        {R"(\blaugh(?:s|ed|ing)?\b)", 1.5},
        {R"(\bcackl(?:e|es|ed|ing)\b)", 1.5},
        {R"(\bchuckl(?:e|es|ed|ing)\b)", 1.0},
        {R"(\bgiggl(?:e|es|ed|ing)\b)", 1.0},
        {R"(\bsnicker(?:s|ed|ing)?\b)", 0.8},
        {R"(\bcru?el laugh\b)", 1.5},
    };
    const std::regex short_mod(
        R"(^\s*(?:[a-z]+ly )?(?:briefly|shortly|once|quickly))",
        std::regex_constants::icase);
    const std::regex long_mod(
        R"(^\s*(?:[a-z]+ly )?(?:maniacally|heartily|uproariously|uncontrollably|hysterically|darkly|wickedly|evilly|loudly|long)|^\s*between phrases)",
        std::regex_constants::icase);
    double total = 0.0;
    for (const auto & [pattern, base] : laugh_verbs) {
        const std::regex re(pattern, std::regex_constants::icase);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), re); it != std::sregex_iterator(); ++it) {
            const auto end = static_cast<size_t>(it->position() + it->length());
            const std::string ctx = text.substr(end, std::min<size_t>(40, text.size() - end));
            if (std::regex_search(ctx, short_mod)) {
                total += base * 0.4;
            } else if (std::regex_search(ctx, long_mod)) {
                total += base * 1.2;
            } else {
                total += base;
            }
        }
    }
    auto quoted = regex_captures(text, R"qq("([^"]+)")qq");
    auto single = regex_captures(text, R"('((?:[^']|'(?![\s.,!?)\]]))+)')");
    quoted.insert(quoted.end(), single.begin(), single.end());
    const std::regex laugh_run(R"((?:h[ae]){3,}|(?:h[ae][ \-]?){3,})", std::regex_constants::icase);
    const std::regex syllable(R"(h[ae])", std::regex_constants::icase);
    for (const auto & q : quoted) {
        for (auto it = std::sregex_iterator(q.begin(), q.end(), laugh_run); it != std::sregex_iterator(); ++it) {
            const std::string run = it->str();
            const auto syllables = static_cast<int64_t>(std::distance(
                std::sregex_iterator(run.begin(), run.end(), syllable),
                std::sregex_iterator()));
            total += 0.2 * static_cast<double>(std::max<int64_t>(syllables - 2, 0));
        }
    }
    return total;
}

double nonverbal_duration(const std::string & text) {
    const std::vector<std::pair<std::string, double>> patterns = {
        {R"(\bsighs?\b)", 0.8}, {R"(\bshaky breath\b)", 1.0}, {R"(\bbreathing deeply\b)", 1.0},
        {R"(\bgasps?\b)", 0.5}, {R"(\bburps?\b)", 0.5}, {R"(\byawns?\b)", 1.0},
        {R"(\bpants?\b)", 0.8}, {R"(\bwheezes?\b)", 0.8}, {R"(\bcoughs?\b)", 0.8},
        {R"(\bsniffles?\b)", 0.5}, {R"(\bsnorts?\b)", 0.3}, {R"(\bgroans?\b)", 0.8},
        {R"(\blong pause\b)", 1.0}, {R"(\bpauses? briefly\b)", 0.3}, {R"(\bpauses?\b)", 0.5},
        {R"(\bsilence\b)", 1.0}, {R"(\blets? the .{1,20} hang\b)", 1.0},
        {R"(\blets? .{1,20} sink in\b)", 1.0}, {R"(\bslams?\b)", 0.5}, {R"(\bclaps?\b)", 0.3},
        {R"(\bdraws? (?:his|her|a) sword\b)", 0.5}, {R"(\btakes? a (?:drag|swig|sip|drink)\b)", 0.5},
        {R"(\bwhistles?\b)", 1.0}, {R"(\bhums?\b)", 0.8}, {R"(\bmutters?\b)", 1.5},
        {R"(\bmumbles?\b)", 1.0}, {R"(\bwhispers?\b)", 0.0},
        {R"(\bclears? (?:his|her) throat\b)", 0.5}, {R"(\bgulps?\b)", 0.5},
        {R"(\bswallows?\b)", 0.5}, {R"(\bvoice (?:breaks?|cracks?|trembles?|drops?|rises?)\b)", 0.5},
        {R"(\bsteadies? (?:him|her)self\b)", 1.0}, {R"(\bcatches? (?:his|her) breath\b)", 1.0},
        {R"(\bcomposes? (?:him|her)self\b)", 0.8}, {R"(\bdemeanor shifts?\b)", 0.5},
        {R"(\bsettles? in\b)", 0.5}, {R"(\bleans? in\b)", 0.3}, {R"(\bwipes? (?:his|her) eyes\b)", 0.5},
    };
    double extra = contextual_laugh_duration(text);
    for (const auto & [pattern, seconds] : patterns) {
        extra += count_regex_seconds(text, pattern, seconds);
    }
    return extra;
}

double round_one_decimal(double value) {
    return std::round(value * 10.0) / 10.0;
}

std::string spoken_text(const std::string & text) {
    auto quotes = regex_captures(text, R"qq("([^"]+)")qq");
    if (quotes.empty()) {
        auto single = regex_captures(text, R"('((?:[^']|'(?![\s.,!?)\]]))+)')");
        for (const auto & q : single) {
            if (word_count(q) > 3) {
                quotes.push_back(q);
            }
        }
    }
    if (!quotes.empty()) {
        std::string joined;
        for (const auto & q : quotes) {
            if (!joined.empty()) {
                joined.push_back(' ');
            }
            joined += q;
        }
        return joined;
    }
    const auto colon = text.find(':');
    if (colon != std::string::npos) {
        return text.substr(colon + 1);
    }
    return text;
}

std::pair<std::string, std::string> split_speaker_prefix(const std::string & prompt) {
    static const std::regex prefix_re(R"(^([^"']{3,}?)(,\s*)(?=["']))", std::regex::ECMAScript);
    std::smatch match;
    if (!std::regex_search(prompt, match, prefix_re)) {
        return {"", prompt};
    }
    return {match[1].str(), prompt.substr(static_cast<size_t>(match.position() + match.length()))};
}

std::vector<std::string> split_sentences_outside_quotes(const std::string & text) {
    std::vector<std::string> sentences;
    std::string buffer;
    bool in_double = false;
    bool in_single = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        buffer.push_back(ch);
        if (ch == '"' && !in_single) {
            const bool was_inside = in_double;
            in_double = !in_double;
            if (was_inside && buffer.size() >= 2 && (buffer[buffer.size() - 2] == '.' || buffer[buffer.size() - 2] == '!' || buffer[buffer.size() - 2] == '?') &&
                (i + 1 >= text.size() || std::isspace(static_cast<unsigned char>(text[i + 1])))) {
                const auto first = buffer.find_first_not_of(" \t\r\n");
                if (first != std::string::npos) {
                    const auto last = buffer.find_last_not_of(" \t\r\n");
                    sentences.push_back(buffer.substr(first, last - first + 1));
                }
                buffer.clear();
            }
        } else if (ch == '\'' && !in_double) {
            const char prev = i > 0 ? text[i - 1] : ' ';
            const char next = i + 1 < text.size() ? text[i + 1] : ' ';
            if (!(std::isalpha(static_cast<unsigned char>(prev)) && std::isalpha(static_cast<unsigned char>(next)))) {
                in_single = !in_single;
            }
        } else if ((ch == '.' || ch == '!' || ch == '?') && !in_double && !in_single) {
            size_t j = i + 1;
            while (j < text.size() && (text[j] == '.' || text[j] == '"' || text[j] == '\'' || text[j] == ')' || text[j] == ']')) {
                buffer.push_back(text[j]);
                if (text[j] == '"') {
                    in_double = !in_double;
                }
                ++j;
            }
            if (j >= text.size() || std::isspace(static_cast<unsigned char>(text[j]))) {
                const auto first = buffer.find_first_not_of(" \t\r\n");
                if (first != std::string::npos) {
                    const auto last = buffer.find_last_not_of(" \t\r\n");
                    sentences.push_back(buffer.substr(first, last - first + 1));
                }
                buffer.clear();
                i = j;
            }
        }
    }
    const auto first = buffer.find_first_not_of(" \t\r\n");
    if (first != std::string::npos) {
        const auto last = buffer.find_last_not_of(" \t\r\n");
        sentences.push_back(buffer.substr(first, last - first + 1));
    }
    return sentences;
}

std::string assemble_prompt_chunk(const std::string & prefix, const std::vector<std::string> & sentences) {
    std::string body;
    for (const auto & sentence : sentences) {
        if (sentence.empty()) {
            continue;
        }
        if (!body.empty()) {
            body.push_back(' ');
        }
        body += sentence;
    }
    if (prefix.empty()) {
        return body;
    }
    const auto first = body.find_first_not_of(" \t\r\n");
    if (first != std::string::npos && (body[first] == '"' || body[first] == '\'')) {
        return prefix + ", " + body;
    }
    return prefix + ". " + body;
}

}  // namespace

int64_t DramaBoxLatentShape::token_count() const noexcept {
    return frames;
}

int64_t DramaBoxLatentShape::latent_value_count() const noexcept {
    return batch * channels * frames * mel_bins;
}

double estimate_dramabox_speech_duration_seconds(const std::string & prompt) {
    const std::string spoken = spoken_text(prompt);
    double chars_per_sec = kCharsPerSecond;
    const auto text_len = static_cast<int64_t>(spoken.size());
    if (text_len < 40) {
        chars_per_sec *= 0.6;
    } else if (text_len < 80) {
        chars_per_sec *= 0.8;
    }
    double duration = static_cast<double>(text_len) / chars_per_sec;
    duration += static_cast<double>(std::count(spoken.begin(), spoken.end(), '.') +
                                    std::count(spoken.begin(), spoken.end(), '!') +
                                    std::count(spoken.begin(), spoken.end(), '?')) * 0.3;
    duration += nonverbal_duration(prompt);
    return std::max(3.0, round_one_decimal(duration + 2.0));
}

double estimate_dramabox_duration_seconds(const std::string & prompt, float duration_scale) {
    return std::max(3.0, round_one_decimal(estimate_dramabox_speech_duration_seconds(prompt) * duration_scale));
}

std::vector<DramaBoxPromptChunk> chunk_prompt_for_duration(
    const std::string & prompt,
    float max_duration_seconds,
    float target_duration_seconds,
    float duration_scale) {
    const auto estimate = [&](const std::string & text) {
        return estimate_dramabox_speech_duration_seconds(text) * static_cast<double>(duration_scale);
    };
    const double total = estimate(prompt);
    if (total <= static_cast<double>(max_duration_seconds)) {
        return {DramaBoxPromptChunk{prompt, total}};
    }
    const auto [prefix, body] = split_speaker_prefix(prompt);
    auto sentences = split_sentences_outside_quotes(body);
    if (sentences.empty()) {
        const std::regex word_re("\\S+");
        for (auto it = std::sregex_iterator(body.begin(), body.end(), word_re); it != std::sregex_iterator(); ++it) {
            sentences.push_back(it->str());
        }
    }
    std::vector<DramaBoxPromptChunk> chunks;
    std::vector<std::string> current;
    double current_duration = 0.0;
    for (const auto & sentence : sentences) {
        auto candidate_sentences = current;
        candidate_sentences.push_back(sentence);
        const auto candidate = assemble_prompt_chunk(prefix, candidate_sentences);
        const double candidate_duration = estimate(candidate);
        if (!current.empty() && candidate_duration > static_cast<double>(target_duration_seconds)) {
            const auto assembled = assemble_prompt_chunk(prefix, current);
            chunks.push_back(DramaBoxPromptChunk{assembled, estimate(assembled)});
            current = {sentence};
            current_duration = estimate(assemble_prompt_chunk(prefix, current));
        } else {
            current = std::move(candidate_sentences);
            current_duration = candidate_duration;
        }
        if (current.size() == 1 && current_duration > static_cast<double>(max_duration_seconds)) {
            const auto solo = assemble_prompt_chunk(prefix, current);
            chunks.push_back(DramaBoxPromptChunk{solo, current_duration});
            current.clear();
            current_duration = 0.0;
        }
    }
    if (!current.empty()) {
        const auto assembled = assemble_prompt_chunk(prefix, current);
        chunks.push_back(DramaBoxPromptChunk{assembled, estimate(assembled)});
    }
    return chunks;
}

int64_t dramabox_aligned_pixel_frames(double duration_seconds, float fps) {
    int64_t frames = static_cast<int64_t>(std::llround(duration_seconds * static_cast<double>(fps))) + 1;
    frames = ((frames - 1 + 4) / 8) * 8 + 1;
    return frames;
}

DramaBoxLatentShape dramabox_target_latent_shape(double duration_seconds, const DramaBoxConfig & config) {
    const int64_t pixel_frames = dramabox_aligned_pixel_frames(duration_seconds, config.fps);
    const double pixel_duration = static_cast<double>(pixel_frames) / static_cast<double>(config.fps);
    const double latents_per_second =
        static_cast<double>(config.audio_vae.sample_rate) /
        static_cast<double>(config.audio_vae.hop_length) /
        static_cast<double>(kAudioLatentDownsampleFactor);
    DramaBoxLatentShape shape;
    shape.batch = 1;
    shape.channels = config.audio_vae.latent_channels;
    shape.frames = static_cast<int64_t>(std::llround(pixel_duration * latents_per_second));
    shape.mel_bins = config.audio_vae.latent_mel_bins;
    return shape;
}

DramaBoxLatentState create_dramabox_initial_state(const DramaBoxLatentShape & shape, const DramaBoxConfig & config) {
    if (shape.batch <= 0 || shape.channels <= 0 || shape.frames <= 0 || shape.mel_bins <= 0) {
        throw std::runtime_error("DramaBox latent shape must be positive");
    }
    DramaBoxLatentState state;
    state.target_shape = shape;
    state.tokens = shape.token_count();
    state.latent.assign(static_cast<size_t>(shape.latent_value_count()), 0.0F);
    state.clean_latent = state.latent;
    state.denoise_mask.assign(static_cast<size_t>(shape.latent_value_count()), 1.0F);
    state.positions.assign(static_cast<size_t>(shape.batch * state.tokens * 2), 0.0F);
    for (int64_t b = 0; b < shape.batch; ++b) {
        for (int64_t t = 0; t < shape.frames; ++t) {
            double start_mel = static_cast<double>(t * kAudioLatentDownsampleFactor);
            double end_mel = static_cast<double>((t + 1) * kAudioLatentDownsampleFactor);
            if (config.audio_vae.causal) {
                start_mel = std::max(0.0, start_mel + 1.0 - static_cast<double>(kAudioLatentDownsampleFactor));
                end_mel = std::max(0.0, end_mel + 1.0 - static_cast<double>(kAudioLatentDownsampleFactor));
            }
            const size_t base = static_cast<size_t>((b * shape.frames + t) * 2);
            state.positions[base] =
                static_cast<float>(start_mel * static_cast<double>(config.audio_vae.hop_length) /
                                   static_cast<double>(config.audio_vae.sample_rate));
            state.positions[base + 1] =
                static_cast<float>(end_mel * static_cast<double>(config.audio_vae.hop_length) /
                                   static_cast<double>(config.audio_vae.sample_rate));
        }
    }
    return state;
}

std::vector<float> make_dramabox_ltx2_sigmas(int64_t steps, int64_t latent_tokens) {
    if (steps <= 0 || latent_tokens <= 0) {
        throw std::runtime_error("DramaBox sigma schedule requires positive steps and latent token count");
    }
    std::vector<float> sigmas(static_cast<size_t>(steps + 1));
    const double mm = (kMaxShift - kBaseShift) / static_cast<double>(kMaxShiftAnchor - kBaseShiftAnchor);
    const double b = kBaseShift - mm * static_cast<double>(kBaseShiftAnchor);
    const double sigma_shift = static_cast<double>(latent_tokens) * mm + b;
    const double exp_shift = std::exp(sigma_shift);
    for (int64_t i = 0; i <= steps; ++i) {
        const double lin = 1.0 - static_cast<double>(i) / static_cast<double>(steps);
        if (lin == 0.0) {
            sigmas[static_cast<size_t>(i)] = 0.0F;
        } else {
            sigmas[static_cast<size_t>(i)] = static_cast<float>(exp_shift / (exp_shift + (1.0 / lin - 1.0)));
        }
    }
    const double last_nonzero = sigmas[static_cast<size_t>(steps - 1)];
    const double scale_factor = (1.0 - last_nonzero) / (1.0 - kTerminalSigma);
    for (int64_t i = 0; i < steps; ++i) {
        sigmas[static_cast<size_t>(i)] =
            static_cast<float>(1.0 - ((1.0 - static_cast<double>(sigmas[static_cast<size_t>(i)])) / scale_factor));
    }
    return sigmas;
}

float auto_rescale_for_cfg(float cfg) {
    if (cfg <= 2.0F) {
        return 0.0F;
    }
    if (cfg <= 3.0F) {
        return 0.6F * (cfg - 2.0F);
    }
    if (cfg <= 4.0F) {
        return 0.6F + 0.2F * (cfg - 3.0F);
    }
    if (cfg <= 8.0F) {
        return 0.8F;
    }
    return std::min(1.0F, 0.8F + 0.1F * (cfg - 8.0F));
}

void append_reference_latents(
    DramaBoxLatentState & state,
    const DramaBoxEncodedReferenceLatents & ref,
    const DramaBoxConfig & config) {
    const int64_t target_tokens = state.tokens;
    const int64_t total_tokens = target_tokens + ref.tokens;
    std::vector<float> latent(static_cast<size_t>(total_tokens * 128), 0.0F);
    std::copy(state.latent.begin(), state.latent.end(), latent.begin());
    std::copy(ref.values.begin(), ref.values.end(), latent.begin() + static_cast<std::ptrdiff_t>(state.latent.size()));
    std::vector<float> clean = latent;
    std::vector<float> mask(static_cast<size_t>(total_tokens * 128), 1.0F);
    std::fill(mask.begin() + static_cast<std::ptrdiff_t>(target_tokens * 128), mask.end(), 0.0F);
    std::vector<float> positions(static_cast<size_t>(total_tokens * 2), 0.0F);
    std::copy(state.positions.begin(), state.positions.end(), positions.begin());
    for (int64_t t = 0; t < ref.tokens; ++t) {
        const int64_t latent_frame = t;
        int64_t start_mel = latent_frame * 4;
        if (config.audio_vae.causal) {
            start_mel = std::max<int64_t>(0, start_mel + 1 - 4);
        }
        int64_t end_mel = (latent_frame + 1) * 4;
        if (config.audio_vae.causal) {
            end_mel = std::max<int64_t>(0, end_mel + 1 - 4);
        }
        positions[static_cast<size_t>((target_tokens + t) * 2 + 0)] =
            static_cast<float>(start_mel * static_cast<double>(config.audio_vae.hop_length) /
                               static_cast<double>(config.audio_vae.sample_rate) + 0.5);
        positions[static_cast<size_t>((target_tokens + t) * 2 + 1)] =
            static_cast<float>(end_mel * static_cast<double>(config.audio_vae.hop_length) /
                               static_cast<double>(config.audio_vae.sample_rate) + 0.5);
    }
    state.latent = std::move(latent);
    state.clean_latent = std::move(clean);
    state.denoise_mask = std::move(mask);
    state.positions = std::move(positions);
    state.tokens = total_tokens;
}

void guided_prediction_from_velocity(
    const std::vector<float> & velocity,
    const DramaBoxLatentState & state,
    int64_t branch_count,
    float sigma,
    float cfg_scale,
    float spatio_temporal_guidance_scale,
    float guidance_rescale,
    bool cfg_enabled,
    bool stg_enabled,
    std::vector<float> & cond,
    std::vector<float> & pred) {
    const int64_t values = state.tokens * 128;
    if (static_cast<int64_t>(state.latent.size()) != values ||
        static_cast<int64_t>(velocity.size()) != branch_count * values) {
        throw std::runtime_error("DramaBox velocity shape mismatch");
    }
    cond.resize(static_cast<size_t>(values));
    pred.resize(static_cast<size_t>(values));
    for (int64_t branch = 0; branch < branch_count; ++branch) {
        for (int64_t i = 0; i < values; ++i) {
            const size_t branch_index = static_cast<size_t>(branch * values + i);
            const size_t value_index = static_cast<size_t>(i);
            const float denoised = state.latent[value_index] -
                velocity[branch_index] * sigma * state.denoise_mask[value_index];
            if (branch == 0) {
                cond[value_index] = denoised;
                pred[value_index] = denoised;
            } else if (cfg_enabled && branch == 1) {
                pred[value_index] += (cfg_scale - 1.0F) * (cond[value_index] - denoised);
            } else if (stg_enabled && branch == (cfg_enabled ? 2 : 1)) {
                pred[value_index] += spatio_temporal_guidance_scale * (cond[value_index] - denoised);
            }
        }
    }
    if (guidance_rescale == 0.0F || pred.empty()) {
        return;
    }
    const auto calc_std = [](const std::vector<float> & values) {
        if (values.size() <= 1) {
            return 0.0F;
        }
        double mean = 0.0;
        for (const float value : values) {
            mean += static_cast<double>(value);
        }
        mean /= static_cast<double>(values.size());
        double var = 0.0;
        for (const float value : values) {
            const double d = static_cast<double>(value) - mean;
            var += d * d;
        }
        var /= static_cast<double>(values.size() - 1);
        return static_cast<float>(std::sqrt(std::max(var, 0.0)));
    };
    const float pred_std = calc_std(pred);
    if (pred_std <= 0.0F) {
        return;
    }
    const float factor = guidance_rescale * (calc_std(cond) / pred_std) + (1.0F - guidance_rescale);
    for (float & value : pred) {
        value *= factor;
    }
}

void post_process_and_euler_step(
    DramaBoxLatentState & state,
    std::vector<float> & denoised,
    float sigma,
    float sigma_next) {
    auto & latent = state.latent;
    if (latent.size() != denoised.size()) {
        throw std::runtime_error("DramaBox Euler step shape mismatch");
    }
    const float dt = sigma_next - sigma;
    for (size_t i = 0; i < latent.size(); ++i) {
        denoised[i] = denoised[i] * state.denoise_mask[i] + state.clean_latent[i] * (1.0F - state.denoise_mask[i]);
        const float velocity = (latent[i] - denoised[i]) / sigma;
        latent[i] += velocity * dt;
    }
}

}  // namespace engine::models::dramabox
