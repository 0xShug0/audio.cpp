#include "engine/models/crisperwhisper/model.h"
#include "engine/framework/io/text.h"
#include "engine/framework/runtime/partial_text.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::crisperwhisper {

std::vector<CrisperWhisperWord> align_words(const CrisperWhisperAssets & assets,
    const CrisperWhisperDecodeResult & decoded, const std::string & language) {
    const auto frames = static_cast<size_t>(assets.encoder.n_audio_ctx);
    if (decoded.attention.size() != decoded.tokens.size() * frames) {
        throw std::runtime_error("CrisperWhisper alignment attention shape mismatch");
    }
    std::vector<std::vector<size_t>> groups;
    std::vector<size_t> current;
    const auto flush = [&]() {
        if (!current.empty()) {
            groups.push_back(std::move(current));
            current.clear();
        }
    };
    const bool unicode = language == "zh" || language == "ja" || language == "th" ||
                         language == "lo" || language == "my" || language == "yue";
    for (size_t i = 0; i < decoded.tokens.size(); ++i) {
        const auto id = decoded.tokens[i];
        const auto piece = assets.decode_text({id});
        if (piece.empty() || id == 220 || piece == " ") {
            flush();
            continue;
        }
        if (!piece.empty() && piece.front() == ' ') {
            flush();
        }
        current.push_back(i);
        if (unicode) {
            std::vector<int32_t> ids;
            for (const auto index : current) {
                ids.push_back(decoded.tokens[index]);
            }
            const auto text = assets.decode_text(ids);
            // Do not split a codepoint whose bytes span several BPE tokens.
            if (runtime::transcript_publishable_end(text) == text.size()) {
                flush();
            }
        }
    }
    flush();
    if (groups.empty()) {
        return {};
    }

    std::vector<float> token_logp(decoded.attention.size());
    for (size_t t = 0; t < decoded.tokens.size(); ++t) {
        double sum = 0.0;
        for (size_t f = 0; f < frames; ++f) {
            sum += std::pow(std::max(0.0, static_cast<double>(decoded.attention[t * frames + f])), 5.0);
        }
        sum = std::max(sum, 1e-8);
        for (size_t f = 0; f < frames; ++f) {
            const auto a = std::max(0.0, static_cast<double>(decoded.attention[t * frames + f]));
            token_logp[t * frames + f] = static_cast<float>(std::log(std::pow(a, 5.0) / sum + 1e-8));
        }
    }
    const auto mel_frames = static_cast<size_t>(decoded.features.frames);
    const auto bins = static_cast<size_t>(decoded.features.mel_bins);
    std::vector<float> energy(mel_frames, 0.0f);
    for (size_t f = 0; f < mel_frames; ++f) {
        for (size_t m = 0; m < bins; ++m) {
            energy[f] += decoded.features.values[m * mel_frames + f];
        }
        energy[f] /= static_cast<float>(bins);
    }
    auto sorted = energy;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double q) {
        const auto position = q * (sorted.size() - 1);
        const auto lo = static_cast<size_t>(position);
        return sorted[lo] + (sorted[std::min(lo + 1, sorted.size() - 1)] - sorted[lo]) * (position - lo);
    };
    const auto low = percentile(0.1), high = percentile(0.9);
    for (auto & value : energy) {
        value = static_cast<float>(std::clamp((value - low) / std::max(1e-6, high - low), 0.0, 1.0));
    }
    std::vector<float> blank(frames);
    for (size_t f = 0; f < frames; ++f) {
        const double position = static_cast<double>(f) * (mel_frames - 1) / (frames - 1);
        const auto lo = static_cast<size_t>(position);
        const auto value = energy[lo] + (energy[std::min(lo + 1, mel_frames - 1)] - energy[lo]) * (position - lo);
        const auto p = std::clamp(1.0 - value, 1e-4, 1.0 - 1e-4);
        blank[f] = static_cast<float>(std::log(std::clamp(p * p * p, 1e-4, 1.0) + 1e-6) - 3.0);
    }
    std::vector<float> emission(groups.size() * frames, -1e9f);
    std::vector<CrisperWhisperWord> words;
    for (size_t w = 0; w < groups.size(); ++w) {
        std::vector<int32_t> ids;
        for (size_t j = 0; j < groups[w].size(); ++j) {
            const auto t = groups[w][j];
            ids.push_back(decoded.tokens[t]);
            for (size_t f = 0; f < frames; ++f) {
                auto & e = emission[w * frames + f];
                const auto p = token_logp[t * frames + f];
                e = j == 0 ? p : std::max(e, p) + std::log1p(std::exp(-std::abs(e - p)));
            }
        }
        words.push_back({io::trim_ascii_whitespace(assets.decode_text(ids))});
    }
    // Word states alternate with virtual blanks. Ties stay in the current state.
    const auto states = 2 * groups.size() + 1;
    std::vector<float> previous(states, -1e9f), next(states);
    std::vector<uint8_t> back(states * frames, 0);
    previous[0] = blank[0];
    for (size_t f = 1; f < frames; ++f) {
        for (size_t s = 0; s < states; ++s) {
            const float advance = s > 0 ? previous[s - 1] : -1e9f;
            const bool take = advance > previous[s];
            back[s * frames + f] = take;
            next[s] = (take ? advance : previous[s]) +
                (s % 2 == 0 ? blank[f] : emission[(s / 2) * frames + f]);
        }
        previous.swap(next);
    }
    size_t state = 0;
    for (size_t s = 1; s < states; ++s) {
        if (previous[s] + static_cast<float>(s) * 1e-4f > previous[state] + static_cast<float>(state) * 1e-4f) {
            state = s;
        }
    }
    for (size_t f = frames; f-- > 0;) {
        if (state % 2 == 1) {
            auto & word = words[state / 2];
            word.start = static_cast<double>(f) * 0.02;
            if (word.end < 0) {
                word.end = word.start;
            }
        }
        if (f > 0 && back[state * frames + f]) {
            --state;
        }
    }
    CrisperWhisperWord * prior = nullptr;
    for (auto & word : words) {
        if (word.start < 0) {
            continue;
        }
        if (prior) {
            const double gap = word.start - prior->end;
            if (gap > 0 && gap <= 0.1) {
                prior->end += gap / 2;
                word.start = prior->end;
            }
        }
        prior = &word;
    }
    return words;
}

}  // namespace engine::models::crisperwhisper
