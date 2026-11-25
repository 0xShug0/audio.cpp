#include "engine/community_models/reverb/frontend.h"

#include <utf8proc.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace engine::community_models::reverb {
namespace {

std::vector<int32_t> codepoints(std::string_view text) {
    std::vector<int32_t> out;
    const auto * data = reinterpret_cast<const utf8proc_uint8_t *>(text.data());
    utf8proc_ssize_t offset = 0;
    const auto size = static_cast<utf8proc_ssize_t>(text.size());
    while (offset < size) {
        utf8proc_int32_t cp = 0;
        const auto read = utf8proc_iterate(data + offset, size - offset, &cp);
        if (read <= 0) {
            throw std::runtime_error("Reverb text is not valid UTF-8");
        }
        out.push_back(cp);
        offset += read;
    }
    return out;
}

void append_utf8(std::string & out, int32_t cp) {
    utf8proc_uint8_t buffer[4];
    const auto n = utf8proc_encode_char(cp, buffer);
    out.append(reinterpret_cast<const char *>(buffer), static_cast<size_t>(n));
}

// Python's str.isspace() for str.split().
bool is_space(int32_t cp) {
    if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F) || cp == 0x85) {
        return true;
    }
    const auto category = utf8proc_category(cp);
    return category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP;
}

double char_weight(int32_t cp) {
    switch (utf8proc_category(cp)) {
        case UTF8PROC_CATEGORY_ZS:
        case UTF8PROC_CATEGORY_ZL:
        case UTF8PROC_CATEGORY_ZP:
            return 0.2;
        case UTF8PROC_CATEGORY_MN:
        case UTF8PROC_CATEGORY_MC:
        case UTF8PROC_CATEGORY_ME:
            return 0.0;
        case UTF8PROC_CATEGORY_PC:
        case UTF8PROC_CATEGORY_PD:
        case UTF8PROC_CATEGORY_PS:
        case UTF8PROC_CATEGORY_PE:
        case UTF8PROC_CATEGORY_PI:
        case UTF8PROC_CATEGORY_PF:
        case UTF8PROC_CATEGORY_PO:
        case UTF8PROC_CATEGORY_SM:
        case UTF8PROC_CATEGORY_SC:
        case UTF8PROC_CATEGORY_SK:
        case UTF8PROC_CATEGORY_SO:
            return 0.5;
        case UTF8PROC_CATEGORY_ND:
        case UTF8PROC_CATEGORY_NL:
        case UTF8PROC_CATEGORY_NO:
            return 3.5;
        default:
            return 1.0;
    }
}

double text_weight(std::string_view text) {
    double total = 0.0;
    for (const int32_t cp : codepoints(text)) {
        total += char_weight(cp);
    }
    return total;
}

// 20 * log10(sqrt(mean of each whole window's squares) + 1e-10), computed as numpy does for the
// input's dtype (float32 for takes, float64 for the pace measure).
template <typename T>
std::vector<T> levels(const std::vector<T> & audio, size_t window) {
    const size_t count = window == 0 ? 0 : audio.size() / window;
    std::vector<T> out(count);
    for (size_t w = 0; w < count; ++w) {
        double sum = 0.0;
        for (size_t i = 0; i < window; ++i) {
            const T v = audio[w * window + i];
            sum += static_cast<double>(v * v);
        }
        const T mean = static_cast<T>(sum / static_cast<double>(window));
        out[w] = static_cast<T>(20) * std::log10(std::sqrt(mean) + static_cast<T>(1e-10));
    }
    return out;
}

constexpr double kSilenceDb = -50.0;

double bessel_i0(double x) {
    // Power series; converges quickly for the arguments a kaiser window with beta ~15 produces.
    double sum = 1.0;
    double term = 1.0;
    const double half = x / 2.0;
    for (int k = 1; k < 500; ++k) {
        term *= (half / k) * (half / k);
        sum += term;
        if (term < sum * 1e-17) {
            break;
        }
    }
    return sum;
}

}  // namespace

const std::vector<std::string> & supported_languages() {
    static const std::vector<std::string> languages = {"de", "en", "es", "fr", "it", "pl", "pt"};
    return languages;
}

bool is_supported_language(std::string_view language) {
    const auto & all = supported_languages();
    return std::find(all.begin(), all.end(), language) != all.end();
}

std::string prepare_text(std::string_view text) {
    std::string out;
    bool pending_space = false;
    for (const int32_t cp : codepoints(text)) {
        if (cp == '(' || cp == ')' || cp == '[' || cp == ']' || cp == '{' || cp == '}') {
            continue;
        }
        if (is_space(cp)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        append_utf8(out, cp);
    }
    return out;
}

int64_t round_half_even(double value) {
    return static_cast<int64_t>(std::nearbyint(value));
}

int64_t estimate_frames(std::string_view text, std::string_view reference_text, int64_t reference_frames) {
    const double reference_weight = text_weight(reference_text);
    if (reference_weight <= 0.0 || reference_frames <= 0) {
        throw std::runtime_error("Reverb needs a reference transcript and frames to estimate the target length");
    }
    double frames = text_weight(text) / (reference_weight / static_cast<double>(reference_frames));
    if (frames < 50.0) {
        frames = 50.0 * std::pow(frames / 50.0, 1.0 / 3.0);
    }
    return std::max<int64_t>(1, static_cast<int64_t>(frames));
}

std::vector<std::string> split_text(
    std::string_view text,
    std::string_view reference_text,
    int64_t reference_frames,
    int64_t longest,
    int64_t piece) {
    if (estimate_frames(text, reference_text, reference_frames) <= longest) {
        return {std::string(text)};
    }
    // re.split(r"(?<=[.!?])\s+", text): cut after a sentence end at each run of whitespace.
    std::vector<std::string> sentences;
    std::string current;
    const auto cps = codepoints(text);
    for (size_t i = 0; i < cps.size();) {
        if (is_space(cps[i]) && i > 0 && (cps[i - 1] == '.' || cps[i - 1] == '!' || cps[i - 1] == '?')) {
            sentences.push_back(current);
            current.clear();
            while (i < cps.size() && is_space(cps[i])) {
                ++i;
            }
            continue;
        }
        append_utf8(current, cps[i]);
        ++i;
    }
    sentences.push_back(current);
    std::vector<std::string> pieces;
    for (const auto & sentence : sentences) {
        if (!pieces.empty()) {
            const std::string joined = pieces.back() + " " + sentence;
            if (estimate_frames(joined, reference_text, reference_frames) <= piece) {
                pieces.back() = joined;
                continue;
            }
        }
        pieces.push_back(sentence);
    }
    return pieces;
}

std::vector<float> raise_quiet(std::vector<float> audio) {
    if (audio.empty()) {
        return audio;
    }
    double sum = 0.0;
    float peak = 0.0F;
    for (const float v : audio) {
        sum += static_cast<double>(v) * static_cast<double>(v);
        peak = std::max(peak, std::fabs(v));
    }
    const double rms = static_cast<double>(static_cast<float>(std::sqrt(sum / static_cast<double>(audio.size()))));
    if (rms == 0.0 || 20.0 * std::log10(rms) >= -40.0) {
        return audio;
    }
    const double gain = std::min(std::pow(10.0, -20.0 / 20.0) / rms, 0.99 / static_cast<double>(peak));
    for (float & v : audio) {
        v = static_cast<float>(static_cast<double>(v) * gain);
    }
    return audio;
}

double speech_length(const std::vector<float> & audio, int sample_rate) {
    const size_t window = static_cast<size_t>(sample_rate / 100);
    const size_t count = window == 0 ? 0 : audio.size() / window;
    const double seconds = static_cast<double>(audio.size()) / static_cast<double>(sample_rate);
    if (count == 0) {
        return seconds;
    }
    std::vector<double> x(audio.begin(), audio.begin() + static_cast<std::ptrdiff_t>(count * window));
    double sum = 0.0;
    for (const double v : x) {
        sum += v * v;
    }
    const double scale = 0.1 / std::max(std::sqrt(sum / static_cast<double>(x.size())), 1e-8);
    for (double & v : x) {
        v *= scale;
    }
    const auto level = levels(x, window);
    std::vector<int64_t> loud;
    for (size_t i = 0; i < level.size(); ++i) {
        if (level[i] >= kSilenceDb) {
            loud.push_back(static_cast<int64_t>(i));
        }
    }
    if (loud.empty()) {
        return seconds;
    }
    const int64_t first = loud.front();
    const int64_t last = loud.back();
    int64_t kept = static_cast<int64_t>(loud.size());
    for (size_t i = 1; i < loud.size(); ++i) {
        kept += std::min<int64_t>(loud[i] - loud[i - 1] - 1, 50);
    }
    const int64_t edges = std::min<int64_t>(first, 10) + std::min<int64_t>(static_cast<int64_t>(count) - 1 - last, 20);
    return static_cast<double>(kept + edges) / 100.0;
}

std::vector<float> trim_silence(const std::vector<float> & audio, int sample_rate, double keep) {
    const size_t window = static_cast<size_t>(sample_rate / 100);
    const auto level = levels(audio, window);
    std::vector<int64_t> loud;
    for (size_t i = 0; i < level.size(); ++i) {
        if (static_cast<double>(level[i]) >= kSilenceDb) {
            loud.push_back(static_cast<int64_t>(i));
        }
    }
    if (loud.empty()) {
        return audio;
    }
    const auto count = static_cast<int64_t>(level.size());
    // Runs of loud windows separated by over 200 ms of silence.
    std::vector<std::pair<int64_t, int64_t>> sounds;  // [first, last] window, inclusive
    sounds.push_back({loud[0], loud[0]});
    for (size_t i = 1; i < loud.size(); ++i) {
        if (loud[i] - loud[i - 1] > 20) {
            sounds.push_back({loud[i], loud[i]});
        } else {
            sounds.back().second = loud[i];
        }
    }
    const auto windows_in = [&](const std::pair<int64_t, int64_t> & sound) {
        return static_cast<int64_t>(std::count_if(loud.begin(), loud.end(),
            [&](int64_t w) { return w >= sound.first && w <= sound.second; }));
    };
    // A sound of at most 120 ms, alone at either edge, is a codec burst rather than speech.
    if (sounds.size() > 1 && windows_in(sounds.front()) <= 12) {
        sounds.erase(sounds.begin());
    }
    if (sounds.size() > 1 && windows_in(sounds.back()) <= 12) {
        sounds.pop_back();
    }
    const int64_t margin = round_half_even(keep * 100.0);
    const int64_t begin = std::max<int64_t>(sounds.front().first - margin, 0) * static_cast<int64_t>(window);
    const int64_t end = std::min<int64_t>(sounds.back().second + 1 + margin, count) * static_cast<int64_t>(window);
    return std::vector<float>(audio.begin() + begin, audio.begin() + end);
}

std::vector<float> resample_kaiser_best(const std::vector<float> & audio, int source_rate, int target_rate) {
    if (source_rate <= 0 || target_rate <= 0) {
        throw std::runtime_error("Reverb resample rates must be positive");
    }
    if (source_rate == target_rate || audio.empty()) {
        return audio;
    }
    constexpr int64_t width_taps = 64;
    constexpr double rolloff = 0.9475937167399596;
    constexpr double beta = 14.769656459379392;
    const int64_t gcd = std::gcd(static_cast<int64_t>(source_rate), static_cast<int64_t>(target_rate));
    const int64_t orig = source_rate / gcd;
    const int64_t next = target_rate / gcd;
    const double base_freq = static_cast<double>(std::min(orig, next)) * rolloff;
    const auto width = static_cast<int64_t>(std::ceil(static_cast<double>(width_taps * orig) / base_freq));
    const int64_t taps = 2 * width + orig;
    // float32 kernel, as torchaudio computes it for a float32 waveform.
    std::vector<float> kernel(static_cast<size_t>(next * taps));
    const float i0_beta = static_cast<float>(bessel_i0(beta));
    const float lowpass = static_cast<float>(width_taps);
    for (int64_t phase = 0; phase < next; ++phase) {
        const float offset = static_cast<float>(-phase) / static_cast<float>(next);
        for (int64_t k = 0; k < taps; ++k) {
            const float idx = static_cast<float>(k - width) / static_cast<float>(orig);
            float t = (offset + idx) * static_cast<float>(base_freq);
            t = std::clamp(t, -lowpass, lowpass);
            const float ratio = t / lowpass;
            const float window = static_cast<float>(bessel_i0(static_cast<double>(
                static_cast<float>(beta) * std::sqrt(1.0F - ratio * ratio)))) / i0_beta;
            const float tp = t * static_cast<float>(M_PI);
            const float sinc = tp == 0.0F ? 1.0F : std::sin(tp) / tp;
            kernel[static_cast<size_t>(phase * taps + k)] = sinc * (window * static_cast<float>(base_freq / orig));
        }
    }
    const auto length = static_cast<int64_t>(audio.size());
    std::vector<float> padded(static_cast<size_t>(length + 2 * width + orig), 0.0F);
    std::copy(audio.begin(), audio.end(), padded.begin() + width);
    const auto target_length = static_cast<int64_t>(
        std::ceil(static_cast<double>(next * length) / static_cast<double>(orig)));
    const int64_t positions = (static_cast<int64_t>(padded.size()) - taps) / orig + 1;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(positions * next));
    for (int64_t j = 0; j < positions; ++j) {
        const float * window = padded.data() + j * orig;
        for (int64_t phase = 0; phase < next; ++phase) {
            const float * kern = kernel.data() + phase * taps;
            float acc = 0.0F;
            for (int64_t k = 0; k < taps; ++k) {
                acc += kern[k] * window[k];
            }
            out.push_back(acc);
        }
    }
    out.resize(static_cast<size_t>(std::min<int64_t>(target_length, static_cast<int64_t>(out.size()))));
    return out;
}

}  // namespace engine::community_models::reverb
