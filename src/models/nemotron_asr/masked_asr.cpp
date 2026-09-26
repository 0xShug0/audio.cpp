#include "engine/models/nemotron_asr/masked_asr.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace engine::models::nemotron_asr {
namespace {

constexpr float kNemoMaskValue = -16.6355F;  // SpeakerTaggedASR.mask_features mask_value
constexpr float kActiveThreshold = 0.5F;     // mask and cache-gating threshold (plan D10)
constexpr int64_t kGatingChunks = 2;         // cache_gating_buffer_size

}  // namespace

std::vector<int> active_speakers(
    const std::function<float(int64_t, int64_t)> & value, int64_t speakers, int64_t begin, int64_t end) {
    std::vector<int> out;
    for (int64_t speaker = 0; speaker < speakers; ++speaker) {
        float peak = 0.0F;
        for (int64_t group = begin; group < end; ++group) peak = std::max(peak, value(group, speaker));
        if (peak > kActiveThreshold) out.push_back(static_cast<int>(speaker));
    }
    return out;
}

std::vector<float> feature_mask(const std::vector<bool> & active_groups, int64_t length) {
    std::vector<float> mask;
    for (const bool active : active_groups) mask.insert(mask.end(), kSpeakerRowsPerFrame, active ? 1.0F : 0.0F);
    if (static_cast<int64_t>(mask.size()) > length) {
        mask.resize(static_cast<size_t>(length));
    } else {
        mask.insert(mask.begin(), static_cast<size_t>(length) - mask.size(), 0.0F);
    }
    return mask;
}

void apply_feature_mask(std::vector<float> & features, const std::vector<float> & mask, int64_t dim) {
    for (size_t t = 0; t < mask.size(); ++t) {
        for (int64_t f = 0; f < dim; ++f) {
            float & x = features[t * static_cast<size_t>(dim) + static_cast<size_t>(f)];
            if (x == 0.0F) {
                x = kNemoMaskValue;
            } else if (mask[t] == 0.0F) {
                x *= 0.0F;
            }
        }
    }
}

MelFeatureSource::MelFeatureSource(const NemotronFrontend & frontend, int64_t hop_length, int64_t n_fft)
    : frontend_(&frontend), hop_(hop_length), n_fft_(n_fft) {}

void MelFeatureSource::append(const float * samples, size_t count) {
    audio_.insert(audio_.end(), samples, samples + count);
    received_ += static_cast<int64_t>(count);
}

void MelFeatureSource::append(std::vector<float> && samples) {
    received_ += static_cast<int64_t>(samples.size());
    if (audio_.empty()) {
        audio_ = std::move(samples);
    } else {
        audio_.insert(audio_.end(), samples.begin(), samples.end());
    }
}

int64_t MelFeatureSource::available_frames(bool final) const noexcept {
    if (final) return received_ > 0 ? received_ / hop_ + 1 : 0;
    const int64_t half = n_fft_ / 2;
    return received_ >= half ? (received_ - half) / hop_ + 1 : 0;
}

std::vector<float> MelFeatureSource::frames(
    int64_t begin, int64_t end, bool final, const std::function<bool(int64_t)> * keep_group) const {
    if (begin < 0 || end <= begin || end > available_frames(final)) {
        throw std::runtime_error("Nemotron ASR masked feature range is not available");
    }
    if (keep_group != nullptr) return compute(begin, end, final, keep_group);
    // Frame values depend only on their own window, so one call over every available
    // frame gives the same values as per-chunk calls and runs the frontend once.
    // Frames available before the end of audio never touch the end padding, so any
    // cached range stays valid once the audio is final.
    const bool cached = begin >= cache_begin_ && end <= cache_end_;
    if (!cached) {
        cache_end_ = available_frames(final);
        cache_ = compute(begin, cache_end_, final, nullptr);
        cache_begin_ = begin;
    }
    const auto dim = static_cast<std::ptrdiff_t>(feature_dim_);
    return std::vector<float>(cache_.begin() + (begin - cache_begin_) * dim, cache_.begin() + (end - cache_begin_) * dim);
}

std::vector<float> MelFeatureSource::compute(
    int64_t begin, int64_t end, bool final, const std::function<bool(int64_t)> * keep_group) const {
    // Frames are computed from a centered slice that starts two hops early, so the
    // left padding and the slice-start pre-emphasis only touch discarded frames.
    const int64_t first = begin >= 2 ? begin - 2 : 0;
    const int64_t slice_begin = first * hop_;
    const int64_t slice_end = final ? received_ : (end - 1) * hop_ + n_fft_ / 2;
    if (slice_begin < base_) throw std::runtime_error("Nemotron ASR masked features were discarded");
    std::vector<float> slice(
        audio_.begin() + static_cast<std::ptrdiff_t>(slice_begin - base_),
        audio_.begin() + static_cast<std::ptrdiff_t>(slice_end - base_));
    if (keep_group != nullptr) {
        for (int64_t group = slice_begin / kSpeakerFrameSamples; group * kSpeakerFrameSamples < slice_end; ++group) {
            if ((*keep_group)(group)) continue;
            const int64_t from = std::max(group * kSpeakerFrameSamples, slice_begin) - slice_begin;
            const int64_t to = std::min((group + 1) * kSpeakerFrameSamples, slice_end) - slice_begin;
            std::fill(slice.begin() + static_cast<std::ptrdiff_t>(from), slice.begin() + static_cast<std::ptrdiff_t>(to), 0.0F);
        }
    }
    const auto features = frontend_->extract_waveform(slice, true);
    feature_dim_ = features.feature_dim;
    const int64_t valid = final ? received_ / hop_ : std::numeric_limits<int64_t>::max();
    std::vector<float> out(static_cast<size_t>((end - begin) * features.feature_dim), 0.0F);
    for (int64_t t = begin; t < end; ++t) {
        if (t >= valid) continue;  // NeMo zeroes the frames past floor(samples / hop)
        const int64_t row = t - first;
        if (row >= features.frames) throw std::runtime_error("Nemotron ASR masked feature slice is too short");
        std::copy_n(features.values.begin() + static_cast<std::ptrdiff_t>(row * features.feature_dim),
                    features.feature_dim,
                    out.begin() + static_cast<std::ptrdiff_t>((t - begin) * features.feature_dim));
    }
    return out;
}

void MelFeatureSource::discard_before(int64_t frame) {
    const int64_t keep_from = std::max<int64_t>(0, (frame - 2) * hop_);
    // Compact only once half the buffer is stale, so dropping stays amortized O(1).
    if (2 * (keep_from - base_) < static_cast<int64_t>(audio_.size())) return;
    const int64_t drop = std::min<int64_t>(keep_from - base_, static_cast<int64_t>(audio_.size()));
    audio_.erase(audio_.begin(), audio_.begin() + static_cast<std::ptrdiff_t>(drop));
    base_ += drop;
}

MaskedSpeakerStreams::MaskedSpeakerStreams(
    NemotronEncoderRuntime & encoder,
    NemotronDecoderRuntime & decoder,
    const NemotronFrontend & frontend,
    const SpeakerProbabilities & probabilities,
    const SpeakerTaggingOptions & options,
    int64_t prompt_id,
    int64_t lookahead_tokens,
    NemotronDecodeOptions decode_options)
    : encoder_(&encoder),
      decoder_(&decoder),
      probabilities_(&probabilities),
      audio_mask_(options.audio_mask),
      prompt_id_(prompt_id),
      lookahead_(lookahead_tokens),
      decode_options_(decode_options),
      features_(frontend),
      segments_(options.gap_sec, options.max_event_sec) {}

void MaskedSpeakerStreams::push_audio(const float * samples, size_t count) {
    features_.append(samples, count);
}

void MaskedSpeakerStreams::push_audio(std::vector<float> && samples) {
    features_.append(std::move(samples));
}

double MaskedSpeakerStreams::stream_time() const {
    return static_cast<double>(next_chunk_ * (lookahead_ + 1)) * kSpeakerFrameSeconds;
}

float MaskedSpeakerStreams::group_value(int64_t group, int64_t speaker) const {
    // NeMo's stream has one more 80 ms frame than the file when floor(samples / 160)
    // is a multiple of 8 (its padding frame); repeat the last frame for it.
    const int64_t index = std::min(group, probabilities_->frames - 1);
    return probabilities_->frames80[static_cast<size_t>(index * probabilities_->speakers + speaker)];
}

void MaskedSpeakerStreams::process(bool final) {
    const int64_t chunk_frames = kSpeakerRowsPerFrame * (lookahead_ + 1);
    while (!done_) {
        const int64_t start = next_chunk_ * chunk_frames;
        const int64_t available = features_.available_frames(final);
        int64_t new_frames = chunk_frames;
        if (final) {
            new_frames = std::min(chunk_frames, available - start);
            // CacheAwareStreamingAudioBuffer stops when a chunk is shorter than the
            // subsampling factor.
            if (new_frames < kSpeakerRowsPerFrame) {
                done_ = true;
                break;
            }
        } else if (start + chunk_frames > available) {
            break;
        }
        run_chunk(next_chunk_, new_frames, final);
        ++next_chunk_;
        features_.discard_before(start + new_frames - encoder_->pad_and_drop_cache_frames());
    }
}

void MaskedSpeakerStreams::run_chunk(int64_t chunk, int64_t new_frames, bool final) {
    const int64_t hop = lookahead_ + 1;
    const int64_t cache = encoder_->pad_and_drop_cache_frames();
    const int64_t length = cache + new_frames;
    const int64_t begin = chunk * kSpeakerRowsPerFrame * hop - cache;

    auto chunk_features = [&](const std::function<bool(int64_t)> * keep) {
        std::vector<float> values;
        if (begin < 0) {
            // First chunk: the pre-encode cache is zeros (pad_and_drop_preencoded).
            values.assign(static_cast<size_t>(cache * features_.feature_dim()), 0.0F);
            auto fresh = features_.frames(0, new_frames, final, keep);
            values.insert(values.end(), fresh.begin(), fresh.end());
        } else {
            values = features_.frames(begin, begin + length, final, keep);
        }
        return values;
    };
    const auto unmasked = audio_mask_ ? std::vector<float>{} : chunk_features(nullptr);
    const int64_t dim = features_.feature_dim();

    // The diarizer adds ceil(new / 8) frames per chunk; the mask is the last `hop`
    // frames of that stream and gating looks at the last two chunks (NeMo live path).
    stream_groups_ += (new_frames + kSpeakerRowsPerFrame - 1) / kSpeakerRowsPerFrame;
    const int64_t end = stream_groups_;
    const int64_t mask_begin = std::max<int64_t>(0, end - hop);
    const int64_t gate_begin = std::max<int64_t>(0, end - kGatingChunks * hop);

    const auto value = [this](int64_t group, int64_t speaker) { return group_value(group, speaker); };
    const auto active = active_speakers(value, probabilities_->speakers, gate_begin, end);
    if (active.empty()) return;

    const double offset = static_cast<double>(chunk * hop) * kSpeakerFrameSeconds;
    std::vector<NemotronFrontendFeatures> rows(active.size());
    std::vector<const NemotronFrontendFeatures *> inputs;
    std::vector<NemotronEncoderSpeakerState *> encoder_states;
    std::vector<Speaker *> states;
    for (size_t index = 0; index < active.size(); ++index) {
        const int speaker = active[index];
        std::vector<float> values;
        if (audio_mask_) {
            const std::function<bool(int64_t)> keep = [&](int64_t group) {
                return group_value(group, speaker) > kActiveThreshold;
            };
            values = chunk_features(&keep);
        } else {
            // The left padding always masks the pre-encode cache frames.
            std::vector<bool> groups;
            for (int64_t group = mask_begin; group < end; ++group) {
                groups.push_back(group_value(group, speaker) > kActiveThreshold);
            }
            values = unmasked;
            apply_feature_mask(values, feature_mask(groups, length), dim);
        }
        auto & state = speakers_[static_cast<size_t>(speaker)];
        if (!state.has_value()) {
            state.emplace();
            state->decoder = decoder_->make_stream_state(decode_options_);
        }
        auto & features = rows[index];
        features.values = std::move(values);
        features.frames = length;
        features.valid_frames = length;
        features.feature_dim = dim;
        states.push_back(&*state);
        inputs.push_back(&features);
        encoder_states.push_back(&state->encoder);
    }

    // Active speakers share one batched encoder step (plan D14).
    const auto encoded = encoder_->encode_pad_and_drop_batch(inputs, prompt_id_, lookahead_, encoder_states);
    for (size_t index = 0; index < active.size(); ++index) {
        auto & state = *states[index];
        decoder_->decode_stream_chunk(encoded[index], state.decoder);
        const std::string text = nemo_hypothesis_text(
            decoder_->stream_text(state.decoder, true), !decode_options_.keep_language_tags);
        segments_.update_hypothesis(
            active[index], text, decoder_->stream_token_frames(state.decoder), state.decoder.encoded_frames, offset);
    }
}

}  // namespace engine::models::nemotron_asr
