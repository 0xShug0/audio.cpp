#include "engine/models/seed_vc/content_features.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::seed_vc {
namespace {

constexpr int64_t kSeedVcSslSampleRate = 16000;
constexpr int64_t kSeedVcSslMaxChunkSeconds = 30;
constexpr int64_t kSeedVcSslOverlapSeconds = 5;
constexpr int64_t kSeedVcSslFrameStride = 320;
constexpr int64_t kSeedVcSslFramesPerSecond = 50;

std::vector<float> take_hidden_prefix(
    const std::vector<float> & hidden,
    int64_t batch,
    int64_t tokens,
    int64_t hidden_size,
    int64_t wanted_tokens) {
    if (wanted_tokens == tokens) {
        return hidden;
    }
    std::vector<float> out(static_cast<size_t>(batch * wanted_tokens * hidden_size));
    for (int64_t b = 0; b < batch; ++b) {
        const auto * src = hidden.data() + static_cast<size_t>(b * tokens * hidden_size);
        auto * dst = out.data() + static_cast<size_t>(b * wanted_tokens * hidden_size);
        std::copy(
            src,
            src + static_cast<size_t>(wanted_tokens * hidden_size),
            dst);
    }
    return out;
}

void append_indices(
    std::vector<int32_t> & dst,
    const std::vector<int32_t> & src,
    int64_t start_token,
    int64_t tokens) {
    if (start_token < 0 || start_token > tokens) {
        throw std::runtime_error("Seed-VC content feature append token range is invalid");
    }
    const auto begin = src.begin() + static_cast<std::ptrdiff_t>(start_token);
    const auto end = src.begin() + static_cast<std::ptrdiff_t>(tokens);
    dst.insert(dst.end(), begin, end);
}

}  // namespace

std::vector<float> seed_vc_wav2vec2_normalize_16k(const std::vector<float> & waveform) {
    if (waveform.empty()) {
        throw std::runtime_error("Seed-VC content extraction requires non-empty 16 kHz waveform");
    }
    double sum = 0.0;
    for (const float sample : waveform) {
        sum += static_cast<double>(sample);
    }
    const double mean = sum / static_cast<double>(waveform.size());
    double variance_sum = 0.0;
    for (const float sample : waveform) {
        const double centered = static_cast<double>(sample) - mean;
        variance_sum += centered * centered;
    }
    const double denom = std::sqrt(variance_sum / static_cast<double>(waveform.size()) + 1.0e-7);
    std::vector<float> out(waveform.size());
    for (size_t index = 0; index < waveform.size(); ++index) {
        out[index] = static_cast<float>((static_cast<double>(waveform[index]) - mean) / denom);
    }
    return out;
}

SeedVcContentFeatureExtractor::SeedVcContentFeatureExtractor(
    const engine::modules::Wav2Vec2EncoderRuntime * hubert,
    const SeedVcAstralQuantizer * wide_quantizer,
    const SeedVcAstralQuantizer * narrow_quantizer)
    : hubert_(hubert),
      wide_quantizer_(wide_quantizer),
      narrow_quantizer_(narrow_quantizer) {
    if (hubert_ == nullptr || wide_quantizer_ == nullptr || narrow_quantizer_ == nullptr) {
        throw std::runtime_error("Seed-VC content extractor requires HuBERT and ASTRAL quantizers");
    }
}

SeedVcContentFeatureOutput SeedVcContentFeatureExtractor::extract_chunk_16k_mono(
    const std::vector<float> & waveform,
    SeedVcContentFeatureKind kind) const {
    if (hubert_ == nullptr || wide_quantizer_ == nullptr || narrow_quantizer_ == nullptr) {
        throw std::runtime_error("Seed-VC content extractor is not initialized");
    }
    const auto normalized = seed_vc_wav2vec2_normalize_16k(waveform);
    auto hidden = hubert_->encode(normalized, 1, static_cast<int64_t>(normalized.size()));
    const int64_t feature_len = std::min<int64_t>(
        static_cast<int64_t>(normalized.size()) / kSeedVcSslFrameStride,
        hidden.tokens);
    if (feature_len <= 0) {
        throw std::runtime_error("Seed-VC content extraction produced no HuBERT frames");
    }
    auto hidden_prefix = take_hidden_prefix(
        hidden.hidden_states,
        hidden.batch,
        hidden.tokens,
        hidden.hidden_size,
        feature_len);
    const SeedVcAstralQuantizer & quantizer =
        kind == SeedVcContentFeatureKind::Wide ? *wide_quantizer_ : *narrow_quantizer_;
    const auto quantized = quantizer.run(hidden_prefix, hidden.batch, feature_len);
    SeedVcContentFeatureOutput out;
    out.indices = quantized.indices;
    out.batch = quantized.batch;
    out.tokens = quantized.tokens;
    out.codebook_size = quantized.codebook_size;
    return out;
}

SeedVcContentFeatureOutput SeedVcContentFeatureExtractor::extract_16k_mono(
    const std::vector<float> & waveform,
    SeedVcContentFeatureKind kind) const {
    const int64_t max_chunk_samples = kSeedVcSslSampleRate * kSeedVcSslMaxChunkSeconds;
    const int64_t overlap_samples = kSeedVcSslSampleRate * kSeedVcSslOverlapSeconds;
    if (static_cast<int64_t>(waveform.size()) <= max_chunk_samples) {
        return extract_chunk_16k_mono(waveform, kind);
    }

    std::vector<int32_t> combined;
    std::vector<float> buffer;
    int64_t traversed = 0;
    bool first = true;
    int64_t codebook_size = 0;
    while (traversed < static_cast<int64_t>(waveform.size())) {
        std::vector<float> chunk;
        if (first) {
            const int64_t chunk_end = std::min<int64_t>(max_chunk_samples, static_cast<int64_t>(waveform.size()));
            chunk.assign(waveform.begin(), waveform.begin() + static_cast<std::ptrdiff_t>(chunk_end));
        } else {
            chunk = buffer;
            const int64_t take = max_chunk_samples - overlap_samples;
            const int64_t chunk_end = std::min<int64_t>(traversed + take, static_cast<int64_t>(waveform.size()));
            chunk.insert(
                chunk.end(),
                waveform.begin() + static_cast<std::ptrdiff_t>(traversed),
                waveform.begin() + static_cast<std::ptrdiff_t>(chunk_end));
        }
        const auto chunk_features = extract_chunk_16k_mono(chunk, kind);
        codebook_size = chunk_features.codebook_size;
        const int64_t skip_tokens = first ? 0 : kSeedVcSslFramesPerSecond * kSeedVcSslOverlapSeconds;
        append_indices(
            combined,
            chunk_features.indices,
            std::min<int64_t>(skip_tokens, chunk_features.tokens),
            chunk_features.tokens);
        const int64_t buffer_start = std::max<int64_t>(0, static_cast<int64_t>(chunk.size()) - overlap_samples);
        buffer.assign(
            chunk.begin() + static_cast<std::ptrdiff_t>(buffer_start),
            chunk.end());
        traversed += first ? max_chunk_samples : static_cast<int64_t>(chunk.size()) - overlap_samples;
        first = false;
    }

    SeedVcContentFeatureOutput out;
    out.indices = std::move(combined);
    out.batch = 1;
    out.tokens = static_cast<int64_t>(out.indices.size());
    out.codebook_size = codebook_size;
    return out;
}

}  // namespace engine::models::seed_vc
