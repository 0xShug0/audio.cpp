#include "engine/community_models/sopro_tts/streaming.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace engine::community_models::sopro_tts {
namespace {

// Python's floor division, which the keep_end arithmetic relies on.
int64_t floor_div(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

// Columns [from, to) of a channel-major [rows, columns] matrix.
std::vector<float> columns(const std::vector<float> & matrix, int64_t rows, int64_t from, int64_t to) {
    const auto width = static_cast<int64_t>(matrix.size()) / rows;
    std::vector<float> out(static_cast<size_t>(rows * (to - from)));
    for (int64_t r = 0; r < rows; ++r) {
        std::copy(
            matrix.begin() + static_cast<ptrdiff_t>(r * width + from),
            matrix.begin() + static_cast<ptrdiff_t>(r * width + to),
            out.begin() + static_cast<ptrdiff_t>(r * (to - from)));
    }
    return out;
}

std::vector<float> normal_noise(int64_t count, std::mt19937_64 & rng) {
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> out(static_cast<size_t>(count));
    for (auto & value : out) {
        value = normal(rng);
    }
    return out;
}

}  // namespace

SoproPromptState build_prompt_state(
    const SoproAcousticRuntime & acoustic,
    const SoproReference & reference,
    int64_t steps,
    int64_t chunk_frames,
    int64_t hop_ratio,
    int64_t lookahead_tokens,
    std::mt19937_64 & rng) {
    const auto n_mels = static_cast<int64_t>(reference.mel.size()) / reference.mel_frames;
    const int64_t aligned = reference.mel_frames / hop_ratio;
    SoproPromptState out;
    out.prompt_frames = aligned * hop_ratio;
    out.semantic_tokens.assign(
        reference.semantic_tokens.begin(),
        reference.semantic_tokens.begin() + static_cast<ptrdiff_t>(std::min<int64_t>(
            aligned, static_cast<int64_t>(reference.semantic_tokens.size()))));
    out.mel = columns(reference.mel, n_mels, 0, out.prompt_frames);
    out.noise = normal_noise(n_mels * out.prompt_frames, rng);
    const int64_t final_tokens = aligned - lookahead_tokens;
    const int64_t frames = floor_div(std::min(out.prompt_frames, final_tokens * hop_ratio), chunk_frames) * chunk_frames;
    if (frames > 0) {
        SoproChunkedRequest request;
        request.semantic_tokens = out.semantic_tokens;
        request.cond_vec = reference.cond_vec;
        request.prompt_mel = out.mel;
        request.prompt_frames = out.prompt_frames;
        request.x0 = out.noise;
        request.steps = steps;
        request.chunk_frames = chunk_frames;
        acoustic.solve_chunked(out.state, request, frames);
    }
    return out;
}

SoproStreamSession::SoproStreamSession(
    const SoproAcousticRuntime & acoustic,
    const SoproVocosRuntime & vocoder,
    const SoproReference & reference,
    const SoproPromptState & prompt,
    const std::vector<float> & mel_mean,
    const std::vector<float> & mel_std,
    int64_t steps,
    int64_t chunk_frames,
    int64_t hop_ratio,
    int64_t lookahead_tokens,
    int64_t warmup_frames,
    std::mt19937_64 & rng)
    : acoustic_(acoustic),
      mel_mean_(mel_mean),
      mel_std_(mel_std),
      vocoder_(vocoder),
      rng_(rng),
      state_(prompt.state),
      n_mels_(vocoder.n_mels()),
      hop_ratio_(hop_ratio),
      finality_margin_(lookahead_tokens * hop_ratio),
      skip_samples_(warmup_frames * vocoder.hop_length()) {
    canvas_.semantic_tokens = prompt.semantic_tokens;
    canvas_.cond_vec = reference.cond_vec;
    canvas_.prompt_mel = prompt.mel;
    canvas_.prompt_frames = prompt.prompt_frames;
    canvas_.x0 = prompt.noise;
    canvas_.steps = steps;
    canvas_.chunk_frames = chunk_frames;
    const int64_t warmup = std::min(warmup_frames, prompt.prompt_frames);
    if (warmup > 0) {
        feed(columns(prompt.mel, n_mels_, prompt.prompt_frames - warmup, prompt.prompt_frames), false);
    }
}

void SoproStreamSession::extend_noise(int64_t frames) {
    const auto have = static_cast<int64_t>(canvas_.x0.size()) / n_mels_;
    if (have >= frames) {
        return;
    }
    const auto extra = normal_noise(n_mels_ * (frames - have), rng_);
    std::vector<float> grown(static_cast<size_t>(n_mels_ * frames));
    for (int64_t c = 0; c < n_mels_; ++c) {
        std::copy_n(canvas_.x0.begin() + static_cast<ptrdiff_t>(c * have), have,
                    grown.begin() + static_cast<ptrdiff_t>(c * frames));
        std::copy_n(extra.begin() + static_cast<ptrdiff_t>(c * (frames - have)), frames - have,
                    grown.begin() + static_cast<ptrdiff_t>(c * frames + have));
    }
    canvas_.x0 = std::move(grown);
}

std::vector<float> SoproStreamSession::render(int64_t keep_end) {
    const int64_t prompt_frames = canvas_.prompt_frames;
    extend_noise(prompt_frames + static_cast<int64_t>(tokens_.size()) * hop_ratio_);
    const int64_t cached = state_.cached;
    auto mel = acoustic_.solve_chunked(state_, canvas_, prompt_frames + keep_end);
    kept_ = keep_end;
    // The first render also finishes the prompt frames the prompt state left
    // unsolved; only generated frames leave.
    const int64_t frames = static_cast<int64_t>(mel.size()) / n_mels_;
    const int64_t skip = std::min(frames, std::max<int64_t>(0, prompt_frames - cached));
    return skip > 0 ? columns(mel, n_mels_, skip, frames) : mel;
}

std::vector<float> SoproStreamSession::feed(const std::vector<float> & mel, bool flush) {
    const int64_t frames = static_cast<int64_t>(mel.size()) / n_mels_;
    std::vector<float> denormalised(mel.size());
    for (int64_t c = 0; c < n_mels_; ++c) {
        for (int64_t t = 0; t < frames; ++t) {
            const auto index = static_cast<size_t>(c * frames + t);
            denormalised[index] = mel[index] * mel_std_[static_cast<size_t>(c)] + mel_mean_[static_cast<size_t>(c)];
        }
    }
    auto audio = vocoder_.push(denormalised, frames, flush);
    if (skip_samples_ > 0) {
        const auto drop = std::min<int64_t>(skip_samples_, static_cast<int64_t>(audio.size()));
        skip_samples_ -= drop;
        audio.erase(audio.begin(), audio.begin() + static_cast<ptrdiff_t>(drop));
    }
    return audio;
}

std::vector<float> SoproStreamSession::push(const std::vector<int32_t> & tokens) {
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
    canvas_.semantic_tokens.insert(canvas_.semantic_tokens.end(), tokens.begin(), tokens.end());
    const int64_t covered = static_cast<int64_t>(tokens_.size()) * hop_ratio_;
    const int64_t chunk = canvas_.chunk_frames;
    const int64_t keep_end =
        floor_div(canvas_.prompt_frames + covered - finality_margin_, chunk) * chunk - canvas_.prompt_frames;
    if (keep_end <= kept_) {
        return {};
    }
    return feed(render(keep_end), false);
}

std::vector<float> SoproStreamSession::finish() {
    const int64_t covered = static_cast<int64_t>(tokens_.size()) * hop_ratio_;
    return feed(covered > kept_ ? render(covered) : std::vector<float>{}, true);
}

}  // namespace engine::community_models::sopro_tts
