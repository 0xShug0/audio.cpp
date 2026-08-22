#include "engine/models/controlfoley/conditioning.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/conditioners/cav_mae_st_conditioner_runtime.h"
#include "engine/framework/conditioners/clap_audio_conditioner_runtime.h"
#include "engine/framework/conditioners/musicgen_style_conditioner_runtime.h"
#include "engine/framework/conditioners/open_clip_conditioner_runtime.h"
#include "engine/framework/conditioners/synchformer_conditioner_runtime.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/json.h"
#include "engine/models/controlfoley/video_processing.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::controlfoley {
namespace {

constexpr int64_t kOpenClipContextLength = 77;
constexpr engine::assets::TensorStorageType kNativeWeightType = engine::assets::TensorStorageType::Native;
constexpr int32_t kOpenClipStartToken = 49406;
constexpr int32_t kOpenClipEndToken = 49407;

std::shared_ptr<const ControlFoleyAssets> require_assets(
    std::shared_ptr<const ControlFoleyAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("ControlFoley conditioner requires assets");
    }
    return assets;
}

std::string utf8_codepoint(int value) {
    std::string out;
    if (value < 0x80) {
        out.push_back(static_cast<char>(value));
    } else if (value < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (value >> 6)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (value >> 12)));
        out.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (value & 0x3F)));
    }
    return out;
}

std::array<std::string, 256> open_clip_byte_encoder() {
    std::vector<int> bytes;
    for (int value = '!'; value <= '~'; ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xA1; value <= 0xAC; ++value) {
        bytes.push_back(value);
    }
    for (int value = 0xAE; value <= 0xFF; ++value) {
        bytes.push_back(value);
    }
    std::vector<int> codes = bytes;
    int extra = 0;
    for (int value = 0; value < 256; ++value) {
        if (std::find(bytes.begin(), bytes.end(), value) == bytes.end()) {
            bytes.push_back(value);
            codes.push_back(256 + extra);
            ++extra;
        }
    }
    std::array<std::string, 256> out;
    for (size_t i = 0; i < bytes.size(); ++i) {
        out[static_cast<size_t>(bytes[i])] = utf8_codepoint(codes[i]);
    }
    return out;
}

std::string normalize_open_clip_text(std::string text) {
    std::string out;
    out.reserve(text.size());
    bool previous_space = true;
    for (unsigned char value : text) {
        if (std::isspace(value)) {
            if (!previous_space) {
                out.push_back(' ');
                previous_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(value)));
        previous_space = false;
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::pair<std::string, std::string> pair_key(const std::string & left, const std::string & right) {
    return {left, right};
}

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> & value) const noexcept {
        return std::hash<std::string>{}(value.first) ^ (std::hash<std::string>{}(value.second) << 1);
    }
};

class OpenClipTokenizer {
public:
    explicit OpenClipTokenizer(const ControlFoleyAssets & assets) {
        const auto vocab = assets.resources.parse_json("open_clip_vocab").as_object();
        encoder.reserve(vocab.size());
        for (const auto & [token, id] : vocab) {
            encoder.emplace(token, static_cast<int32_t>(id.as_i64()));
        }
        const auto merges_text = assets.resources.read_text("open_clip_merges");
        size_t start = 0;
        int rank = 0;
        while (start < merges_text.size()) {
            const size_t end = merges_text.find('\n', start);
            auto line = merges_text.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (!line.empty() && line[0] != '#') {
                const size_t split = line.find(' ');
                if (split == std::string::npos) {
                    throw std::runtime_error("ControlFoley OpenCLIP merge entry is invalid: " + line);
                }
                ranks.emplace(pair_key(line.substr(0, split), line.substr(split + 1)), rank++);
            }
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        byte_encoder = open_clip_byte_encoder();
    }

    std::vector<int32_t> encode_padded(const std::string & text) const {
        std::vector<int32_t> out;
        out.reserve(static_cast<size_t>(kOpenClipContextLength));
        out.push_back(kOpenClipStartToken);
        const auto pieces = tokenize(normalize_open_clip_text(text));
        out.insert(out.end(), pieces.begin(), pieces.end());
        if (static_cast<int64_t>(out.size()) >= kOpenClipContextLength) {
            out.resize(static_cast<size_t>(kOpenClipContextLength));
            out.back() = kOpenClipEndToken;
        } else {
            out.push_back(kOpenClipEndToken);
            out.resize(static_cast<size_t>(kOpenClipContextLength), 0);
        }
        return out;
    }

private:
    std::vector<int32_t> tokenize(const std::string & text) const {
        static const std::regex pattern(
            "<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\\sA-Za-z0-9]+",
            std::regex::ECMAScript);
        std::vector<int32_t> out;
        for (std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) {
            std::string encoded;
            const auto token = it->str();
            for (unsigned char value : token) {
                encoded += byte_encoder[static_cast<size_t>(value)];
            }
            for (const auto & piece : bpe(encoded)) {
                const auto found = encoder.find(piece);
                if (found == encoder.end()) {
                    throw std::runtime_error("ControlFoley OpenCLIP tokenizer missing BPE piece: " + piece);
                }
                out.push_back(found->second);
            }
        }
        return out;
    }

    std::vector<std::string> bpe(const std::string & token) const {
        const auto cached = bpe_cache.find(token);
        if (cached != bpe_cache.end()) {
            return cached->second;
        }
        std::vector<std::string> word;
        for (size_t i = 0; i < token.size();) {
            const unsigned char c = static_cast<unsigned char>(token[i]);
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) {
                len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                len = 3;
            }
            word.push_back(token.substr(i, len));
            i += len;
        }
        if (word.empty()) {
            return {};
        }
        word.back() += "</w>";
        while (word.size() > 1) {
            int best_rank = std::numeric_limits<int>::max();
            size_t best_index = word.size();
            for (size_t i = 0; i + 1 < word.size(); ++i) {
                const auto found = ranks.find(pair_key(word[i], word[i + 1]));
                if (found != ranks.end() && found->second < best_rank) {
                    best_rank = found->second;
                    best_index = i;
                }
            }
            if (best_index == word.size()) {
                break;
            }
            word[best_index] += word[best_index + 1];
            word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
        }
        bpe_cache.emplace(token, word);
        return word;
    }

    std::unordered_map<std::string, int32_t> encoder;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> ranks;
    std::array<std::string, 256> byte_encoder;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache;
};

ControlFoleyFlowConditionInput empty_condition_input(
    const ControlFoleyAssets & assets,
    const ControlFoleyTemporalShape & shape) {
    const auto & config = assets.config;
    ControlFoleyFlowConditionInput out;
    out.batch = 1;
    out.text = assets.flow_weights->require_f32("empty_string_feat", {config.text_seq_len, config.text_dim});
    const auto expand = [](const std::vector<float> & value, int64_t tokens, int64_t dim, const char * label) {
        if (tokens <= 0 || dim <= 0 || static_cast<int64_t>(value.size()) != dim) {
            throw std::runtime_error(std::string("ControlFoley empty ") + label + " feature shape mismatch");
        }
        std::vector<float> out(static_cast<size_t>(tokens * dim));
        for (int64_t token = 0; token < tokens; ++token) {
            std::copy(value.begin(), value.end(), out.begin() + static_cast<std::ptrdiff_t>(token * dim));
        }
        return out;
    };
    out.clip = expand(assets.flow_weights->require_f32("empty_clip_feat", {1, config.clip_dim}), shape.clip, config.clip_dim, "clip");
    out.visual = expand(assets.flow_weights->require_f32("empty_visual_feat", {1, config.visual_dim}), shape.visual, config.visual_dim, "visual");
    out.sync = expand(assets.flow_weights->require_f32("empty_sync_feat", {1, config.sync_dim}), shape.sync, config.sync_dim, "sync");
    out.audio = assets.flow_weights->require_f32("empty_audio_feat", {1, config.audio_dim});
    out.timbre = assets.flow_weights->require_f32("empty_timbre_feat", {1, config.timbre_dim});
    return out;
}

std::vector<float> mean_cav_patches(const engine::conditioners::CavMaeStVisualFeatures & features) {
    std::vector<float> out(static_cast<size_t>(features.batch * features.hidden), 0.0F);
    const float scale = 1.0F / static_cast<float>(features.patches);
    const int64_t outputs = features.batch * features.hidden;
    for (int64_t index = 0; index < outputs; ++index) {
        const int64_t batch = index / features.hidden;
        const int64_t hidden = index % features.hidden;
        float sum = 0.0F;
        for (int64_t patch = 0; patch < features.patches; ++patch) {
            sum += features.values[static_cast<size_t>((batch * features.patches + patch) * features.hidden + hidden)];
        }
        out[static_cast<size_t>(index)] = sum * scale;
    }
    return out;
}

std::vector<float> mean_timbre_tokens(const engine::conditioners::MusicGenStyleEmbedding & embedding) {
    if (embedding.batch != 1 || embedding.tokens <= 0) {
        throw std::runtime_error("ControlFoley MusicGen style embedding shape mismatch");
    }
    std::vector<float> out(static_cast<size_t>(embedding.features), 0.0F);
    const float scale = 1.0F / static_cast<float>(embedding.tokens);
    for (int64_t feature = 0; feature < embedding.features; ++feature) {
        float sum = 0.0F;
        for (int64_t token = 0; token < embedding.tokens; ++token) {
            sum += embedding.values[static_cast<size_t>(token * embedding.features + feature)];
        }
        out[static_cast<size_t>(feature)] = sum * scale;
    }
    return out;
}

}  // namespace

struct ControlFoleyConditionerRuntime::Impl {
    struct ShapeKey {
        int64_t latent = 0;
        int64_t clip = 0;
        int64_t visual = 0;
        int64_t sync = 0;

        bool operator==(const ShapeKey & other) const {
            return latent == other.latent &&
                   clip == other.clip &&
                   visual == other.visual &&
                   sync == other.sync;
        }
    };

    Impl(
        std::shared_ptr<const ControlFoleyAssets> assets_in,
        engine::core::ExecutionContext & execution_in)
        : assets(require_assets(std::move(assets_in))),
          execution(&execution_in),
          tokenizer(*assets) {
        engine::conditioners::OpenClipRuntimeOptions options;
        options.weight_storage_type = kNativeWeightType;
        options.load_text = true;
        options.load_image = true;
        auto image_config = engine::conditioners::OpenClipImageConfig{};
        image_config.image_size = 384;
        open_clip_runtime = std::make_unique<engine::conditioners::OpenClipConditionerRuntime>(
            assets->open_clip_weights,
            *execution,
            engine::conditioners::OpenClipTextConfig{},
            options,
            image_config);
    }

    ControlFoleyConditioningBatch build(
        const ControlFoleyConditioningRequest & request,
        const ControlFoleyTemporalShape & shape) {
        ControlFoleyConditioningBatch out;
        const auto & empty = empty_condition(shape);
        out.condition = empty;
        out.empty = empty;

        const bool has_text = request.text.has_value() && !request.text->empty();
        const bool has_negative_text = !request.negative_text.empty();
        double tokenize_ms = 0.0;
        double encode_ms = 0.0;
        double negative_tokenize_ms = 0.0;
        double negative_encode_ms = 0.0;

        const bool text_cached = has_text &&
            positive_text_cache_key.has_value() &&
            *positive_text_cache_key == *request.text;
        const bool negative_text_cached = has_negative_text &&
            negative_text_cache_key.has_value() &&
            *negative_text_cache_key == request.negative_text;
        if (has_text && has_negative_text && !text_cached && !negative_text_cached) {
            encode_ms = engine::debug::measure_ms([&]() {
                encode_open_clip_text_pair(*request.text, request.negative_text, tokenize_ms, negative_tokenize_ms);
            });
            out.condition.text = positive_text_cache_value;
            out.empty.text = negative_text_cache_value;
        } else {
            if (has_text) {
                encode_ms = engine::debug::measure_ms([&]() {
                    out.condition.text = encode_open_clip_text(
                        *request.text,
                        positive_text_cache_key,
                        positive_text_cache_value,
                        tokenize_ms);
                });
            }
            if (has_negative_text) {
                negative_encode_ms = engine::debug::measure_ms([&]() {
                    out.empty.text = encode_open_clip_text(
                        request.negative_text,
                        negative_text_cache_key,
                        negative_text_cache_value,
                        negative_tokenize_ms);
                });
            }
        }
        if (has_text) {
            engine::debug::timing_log_scalar("controlfoley.cond.text_tokenize_ms", tokenize_ms);
            engine::debug::timing_log_scalar("controlfoley.cond.text_encode_ms", encode_ms);
        }
        if (has_negative_text) {
            engine::debug::timing_log_scalar("controlfoley.cond.negative_text_tokenize_ms", negative_tokenize_ms);
            engine::debug::timing_log_scalar("controlfoley.cond.negative_text_encode_ms", negative_encode_ms);
        } else {
            engine::debug::timing_log_scalar("controlfoley.cond.negative_text_tokenize_ms", 0.0);
            engine::debug::timing_log_scalar("controlfoley.cond.negative_text_encode_ms", 0.0);
        }
        if (request.video.has_value()) {
            const double video_ms = engine::debug::measure_ms([&]() {
                apply_video(*request.video, request.mask_away_clip, shape, out.condition);
            });
            engine::debug::timing_log_scalar("controlfoley.cond.video_total_ms", video_ms);
        }
        if (request.audio.has_value()) {
            const double audio_ms = engine::debug::measure_ms([&]() {
                apply_audio(*request.audio, out.condition);
            });
            engine::debug::timing_log_scalar("controlfoley.cond.audio_total_ms", audio_ms);
        }
        return out;
    }

    void apply_video(
        const std::filesystem::path & path,
        bool mask_away_clip,
        const ControlFoleyTemporalShape & shape,
        ControlFoleyFlowConditionInput & condition) {
        const int64_t sync_frame_count = ((shape.sync / 8 - 1) * 8 + 16);
        const float decode_duration_sec = std::max({
            static_cast<float>(shape.clip) / 8.0F,
            static_cast<float>(shape.visual) / 4.0F,
            static_cast<float>(sync_frame_count) / 25.0F,
        });
        video::SourceVideo source_video;
        const double decode_ms = engine::debug::measure_ms([&]() {
            source_video = video::load_video_frames(path, static_cast<double>(decode_duration_sec));
        });
        engine::debug::timing_log_scalar("controlfoley.cond.video.decode_ms", decode_ms);

        if (!mask_away_clip) {
            video::ResizedFrameCache clip_cache(source_video, 384, false);
            std::vector<float> clip_frames;
            const double extract_ms = engine::debug::measure_ms([&]() {
                clip_frames = clip_cache.extract(
                    static_cast<float>(shape.clip) / 8.0F,
                    8,
                    {0.48145466F, 0.4578275F, 0.40821073F},
                    {0.26862954F, 0.26130258F, 0.27577711F},
                    "controlfoley.cond.video.clip_extract");
            });
            const double encode_ms = engine::debug::measure_ms([&]() {
                auto embedding = open_clip().encode_image(clip_frames, shape.clip, 384, 384);
                condition.clip = std::move(embedding.values);
            });
            engine::debug::timing_log_scalar("controlfoley.cond.video.clip_extract_ms", extract_ms);
            engine::debug::timing_log_scalar("controlfoley.cond.video.clip_encode_ms", encode_ms);
        }
        video::ResizedFrameCache image_cache(source_video, 224, true);
        std::vector<float> visual_frames;
        const double visual_extract_ms = engine::debug::measure_ms([&]() {
            visual_frames = image_cache.extract(
                static_cast<float>(shape.visual) / 4.0F,
                4,
                {0.4850F, 0.4560F, 0.4060F},
                {0.2290F, 0.2240F, 0.2250F},
                "controlfoley.cond.video.visual_extract");
        });
        engine::conditioners::CavMaeStVisualFeatures cav_features;
        const double cav_ms = engine::debug::measure_ms([&]() {
            cav_features = cav_mae().encode_visual(visual_frames, shape.visual, 224, 224);
        });
        const double cav_mean_ms = engine::debug::measure_ms([&]() {
            condition.visual = mean_cav_patches(cav_features);
        });
        engine::debug::timing_log_scalar("controlfoley.cond.video.visual_extract_ms", visual_extract_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.video.cav_mae_ms", cav_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.video.cav_mean_ms", cav_mean_ms);

        std::vector<float> sync_frames;
        const double sync_extract_ms = engine::debug::measure_ms([&]() {
            sync_frames = image_cache.extract(
                static_cast<float>(sync_frame_count) / 25.0F,
                25,
                {0.5F, 0.5F, 0.5F},
                {0.5F, 0.5F, 0.5F},
                "controlfoley.cond.video.sync_extract");
        });
        const double sync_encode_ms = engine::debug::measure_ms([&]() {
            auto sync = synchformer().encode_frames(sync_frames, sync_frame_count, 224, 224, 8);
            condition.sync = std::move(sync.values);
        });
        engine::debug::timing_log_scalar("controlfoley.cond.video.sync_extract_ms", sync_extract_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.video.sync_encode_ms", sync_encode_ms);
    }

    void apply_audio(
        const engine::runtime::AudioBuffer & audio,
        ControlFoleyFlowConditionInput & condition) {
        if (audio.sample_rate <= 0 || audio.channels <= 0 || audio.samples.empty()) {
            throw std::runtime_error("ControlFoley reference audio is empty");
        }
        std::vector<float> clap_waveform;
        const double clap_resample_ms = engine::debug::measure_ms([&]() {
            auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
            clap_waveform = engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, 16000);
        });
        const double clap_encode_ms = engine::debug::measure_ms([&]() {
            auto clap_embedding = clap().encode_audio(clap_waveform, 1, static_cast<int64_t>(clap_waveform.size()), 0);
            condition.audio = std::move(clap_embedding.values);
        });
        engine::debug::timing_log_scalar("controlfoley.cond.audio.clap_resample_ms", clap_resample_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.audio.clap_encode_ms", clap_encode_ms);

        std::vector<float> timbre_waveform;
        const double timbre_resample_ms = engine::debug::measure_ms([&]() {
            auto mono = engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
            timbre_waveform = engine::audio::resample_mono_torchaudio_sinc_hann(mono, audio.sample_rate, 32000);
        });
        const size_t min_samples = 2ull * 32000ull;
        const size_t max_samples = 4ull * 32000ull;
        const double timbre_resize_ms = engine::debug::measure_ms([&]() {
            if (timbre_waveform.size() < min_samples) {
                timbre_waveform.resize(min_samples, 0.0F);
            } else if (timbre_waveform.size() > max_samples) {
                timbre_waveform.resize(max_samples);
            }
        });
        std::vector<float> mert_waveform;
        const double timbre_mert_resample_ms = engine::debug::measure_ms([&]() {
            mert_waveform = engine::audio::resample_mono_torchaudio_sinc_hann(timbre_waveform, 32000, 24000);
        });
        engine::conditioners::MusicGenStyleEmbedding timbre_embedding;
        const double timbre_encode_ms = engine::debug::measure_ms([&]() {
            timbre_embedding = musicgen_style().encode_audio(mert_waveform, 1, static_cast<int64_t>(mert_waveform.size()), 0);
        });
        const double timbre_mean_ms = engine::debug::measure_ms([&]() {
            condition.timbre = mean_timbre_tokens(timbre_embedding);
        });
        engine::debug::timing_log_scalar("controlfoley.cond.audio.timbre_resample_ms", timbre_resample_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.audio.timbre_resize_ms", timbre_resize_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.audio.timbre_mert_resample_ms", timbre_mert_resample_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.audio.timbre_encode_ms", timbre_encode_ms);
        engine::debug::timing_log_scalar("controlfoley.cond.audio.timbre_mean_ms", timbre_mean_ms);
    }

    const ControlFoleyFlowConditionInput & empty_condition(
        const ControlFoleyTemporalShape & shape) {
        const ShapeKey key{shape.latent, shape.clip, shape.visual, shape.sync};
        if (empty_condition_cache_key.has_value() && *empty_condition_cache_key == key) {
            return empty_condition_cache_value;
        }
        empty_condition_cache_key = key;
        const double build_ms = engine::debug::measure_ms([&]() {
            empty_condition_cache_value = empty_condition_input(*assets, shape);
        });
        engine::debug::timing_log_scalar("controlfoley.cond.empty_condition_ms", build_ms);
        return empty_condition_cache_value;
    }

    const std::vector<float> & encode_open_clip_text(
        const std::string & text,
        std::optional<std::string> & cache_key,
        std::vector<float> & cache_value,
        double & tokenize_ms) {
        if (cache_key.has_value() && *cache_key == text) {
            tokenize_ms = 0.0;
            return cache_value;
        }
        std::vector<int32_t> tokens;
        tokenize_ms = engine::debug::measure_ms([&]() {
            tokens = tokenizer.encode_padded(text);
        });
        cache_key = text;
        cache_value = open_clip().encode_text(tokens, 1, kOpenClipContextLength).values;
        return cache_value;
    }

    void encode_open_clip_text_pair(
        const std::string & positive,
        const std::string & negative,
        double & positive_tokenize_ms,
        double & negative_tokenize_ms) {
        std::vector<int32_t> positive_tokens;
        std::vector<int32_t> negative_tokens;
        positive_tokenize_ms = engine::debug::measure_ms([&]() {
            positive_tokens = tokenizer.encode_padded(positive);
        });
        negative_tokenize_ms = engine::debug::measure_ms([&]() {
            negative_tokens = tokenizer.encode_padded(negative);
        });
        std::vector<int32_t> tokens;
        tokens.reserve(static_cast<size_t>(2 * kOpenClipContextLength));
        tokens.insert(tokens.end(), positive_tokens.begin(), positive_tokens.end());
        tokens.insert(tokens.end(), negative_tokens.begin(), negative_tokens.end());
        auto hidden = open_clip().encode_text(tokens, 2, kOpenClipContextLength);
        const int64_t branch_values = kOpenClipContextLength * hidden.hidden;
        if (hidden.batch != 2 ||
            hidden.tokens != kOpenClipContextLength ||
            hidden.hidden <= 0 ||
            static_cast<int64_t>(hidden.values.size()) != 2 * branch_values) {
            throw std::runtime_error("ControlFoley OpenCLIP CFG text batch output shape mismatch");
        }
        positive_text_cache_key = positive;
        negative_text_cache_key = negative;
        positive_text_cache_value.assign(hidden.values.begin(), hidden.values.begin() + branch_values);
        negative_text_cache_value.assign(hidden.values.begin() + branch_values, hidden.values.end());
    }

    engine::conditioners::OpenClipConditionerRuntime & open_clip() {
        if (open_clip_runtime == nullptr) {
            throw std::runtime_error("ControlFoley OpenCLIP runtime was not initialized");
        }
        return *open_clip_runtime;
    }

    engine::conditioners::CavMaeStConditionerRuntime & cav_mae() {
        if (cav_mae_runtime == nullptr) {
            engine::conditioners::CavMaeStRuntimeOptions options;
            options.weight_storage_type = kNativeWeightType;
            cav_mae_runtime = std::make_unique<engine::conditioners::CavMaeStConditionerRuntime>(
                assets->cav_mae_weights,
                *execution,
                engine::conditioners::CavMaeStVisualConfig{},
                options);
        }
        return *cav_mae_runtime;
    }

    engine::conditioners::SynchformerConditionerRuntime & synchformer() {
        if (synchformer_runtime == nullptr) {
            engine::conditioners::SynchformerRuntimeOptions options;
            options.weight_storage_type = kNativeWeightType;
            synchformer_runtime = std::make_unique<engine::conditioners::SynchformerConditionerRuntime>(
                assets->synchformer_weights,
                *execution,
                engine::conditioners::SynchformerConfig{},
                options);
        }
        return *synchformer_runtime;
    }

    engine::conditioners::ClapAudioConditionerRuntime & clap() {
        if (clap_runtime == nullptr) {
            engine::conditioners::ClapAudioConfig config;
            engine::conditioners::ClapAudioRuntimeOptions options;
            options.weight_storage_type = kNativeWeightType;
            clap_runtime = std::make_unique<engine::conditioners::ClapAudioConditionerRuntime>(
                assets->clap_weights,
                *execution,
                config,
                options);
        }
        return *clap_runtime;
    }

    engine::conditioners::MusicGenStyleConditionerRuntime & musicgen_style() {
        if (musicgen_style_runtime == nullptr) {
            engine::conditioners::MusicGenStyleRuntimeOptions options;
            options.weight_storage_type = kNativeWeightType;
            musicgen_style_runtime = std::make_unique<engine::conditioners::MusicGenStyleConditionerRuntime>(
                assets->mert_weights,
                assets->musicgen_style_weights,
                *execution,
                engine::conditioners::MusicGenStyleConfig::mert_default(),
                options);
        }
        return *musicgen_style_runtime;
    }

    std::shared_ptr<const ControlFoleyAssets> assets;
    engine::core::ExecutionContext * execution = nullptr;
    OpenClipTokenizer tokenizer;
    std::optional<ShapeKey> empty_condition_cache_key;
    ControlFoleyFlowConditionInput empty_condition_cache_value;
    std::optional<std::string> positive_text_cache_key;
    std::vector<float> positive_text_cache_value;
    std::optional<std::string> negative_text_cache_key;
    std::vector<float> negative_text_cache_value;
    std::unique_ptr<engine::conditioners::OpenClipConditionerRuntime> open_clip_runtime;
    std::unique_ptr<engine::conditioners::CavMaeStConditionerRuntime> cav_mae_runtime;
    std::unique_ptr<engine::conditioners::SynchformerConditionerRuntime> synchformer_runtime;
    std::unique_ptr<engine::conditioners::ClapAudioConditionerRuntime> clap_runtime;
    std::unique_ptr<engine::conditioners::MusicGenStyleConditionerRuntime> musicgen_style_runtime;
};

ControlFoleyConditionerRuntime::ControlFoleyConditionerRuntime(
    std::shared_ptr<const ControlFoleyAssets> assets,
    engine::core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(assets), execution)) {}

ControlFoleyConditionerRuntime::~ControlFoleyConditionerRuntime() = default;

ControlFoleyConditioningBatch ControlFoleyConditionerRuntime::build(
    const ControlFoleyConditioningRequest & request,
    const ControlFoleyTemporalShape & shape) const {
    return impl_->build(request, shape);
}

}  // namespace engine::models::controlfoley
