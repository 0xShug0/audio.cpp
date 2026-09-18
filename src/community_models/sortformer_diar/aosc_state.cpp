// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "engine/community_models/sortformer_diar/aosc_state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace engine::community_models::sortformer_diar;

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();
constexpr float kBirthSpeech = 0.30f;
constexpr float kBirthClean = 0.95f;
constexpr float kEstablishedQuiet = 0.02f;
constexpr float kBirthFading = 0.90f;
constexpr float kEstablishedFading = 0.15f;
constexpr int kBirthCleanFrames = 4;
constexpr int kBirthFadingFrames = 20;
constexpr int kBirthEpisodeGapFrames = 25;
constexpr int kBirthRevisionFrames = 128;

std::vector<int> topk_column(const std::vector<float> & scores, int n, int n_spk, int spk, int k) {
    if (n <= 0 || k <= 0) return {};
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    k = std::min(k, n);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](int a, int b) {
        const float sa = scores[static_cast<size_t>(a) * n_spk + spk];
        const float sb = scores[static_cast<size_t>(b) * n_spk + spk];
        return sa != sb ? sa > sb : a < b;
    });
    indices.resize(k);
    return indices;
}
}  // namespace

SortformerV2ChannelBirthGate::SortformerV2ChannelBirthGate(int n_spk) : n_spk_(n_spk) {
    if (n_spk_ <= 0) {
        throw std::invalid_argument("SortformerV2ChannelBirthGate: n_spk must be positive");
    }
    reset();
}

void SortformerV2ChannelBirthGate::reset() {
    frame_ = 0;
    established_.assign(n_spk_, false);
    clean_frames_.assign(n_spk_, 0);
    fading_frames_.assign(n_spk_, 0);
    last_win_.assign(n_spk_, std::numeric_limits<int64_t>::min() / 2);
    raw_ring_.clear();
}

bool SortformerV2ChannelBirthGate::observe(const float * probs) {
    int winner = 0;
    for (int s = 1; s < n_spk_; ++s) {
        if (probs[s] > probs[winner]) {
            winner = s;
        }
    }

    bool changed = false;
    if (probs[winner] >= kBirthSpeech && !established_[winner]) {
        if (frame_ - last_win_[winner] > kBirthEpisodeGapFrames) {
            clean_frames_[winner] = 0;
            fading_frames_[winner] = 0;
        }
        last_win_[winner] = frame_;
        float established_prob = 0.0f;
        for (int s = 0; s < n_spk_; ++s) {
            if (established_[s]) {
                established_prob = std::max(established_prob, probs[s]);
            }
        }
        if (probs[winner] >= kBirthClean && established_prob <= kEstablishedQuiet) {
            ++clean_frames_[winner];
        }
        if (probs[winner] >= kBirthFading && established_prob <= kEstablishedFading) {
            ++fading_frames_[winner];
        }
        if (clean_frames_[winner] >= kBirthCleanFrames || fading_frames_[winner] >= kBirthFadingFrames) {
            established_[winner] = true;
            changed = true;
        }
    }
    ++frame_;
    return changed;
}

void SortformerV2ChannelBirthGate::relabel(float * probs) const {
    int target = -1;
    for (int s = 0; s < n_spk_; ++s) {
        if (established_[s] && (target < 0 || probs[s] > probs[target])) {
            target = s;
        }
    }
    if (target < 0) {
        return;
    }
    for (int s = 0; s < n_spk_; ++s) {
        if (!established_[s] && probs[s] > 0.0f) {
            probs[target] = std::max(probs[target], probs[s]);
            probs[s] = 0.0f;
        }
    }
}

void SortformerV2ChannelBirthGate::push_raw(const float * probs) {
    raw_ring_.insert(raw_ring_.end(), probs, probs + n_spk_);
    const size_t capacity = static_cast<size_t>(kBirthRevisionFrames) * n_spk_;
    if (raw_ring_.size() > capacity) {
        raw_ring_.erase(raw_ring_.begin(), raw_ring_.begin() + n_spk_);
    }
}

void SortformerV2ChannelBirthGate::append(const std::vector<float> & raw, std::vector<float> & timeline) {
    if (raw.size() % static_cast<size_t>(n_spk_) != 0) {
        throw std::invalid_argument("SortformerV2ChannelBirthGate: incomplete probability frame");
    }
    bool changed = false;
    for (size_t i = 0; i < raw.size(); i += static_cast<size_t>(n_spk_)) {
        push_raw(raw.data() + i);
        changed |= observe(raw.data() + i);
    }

    const size_t old_size = timeline.size();
    timeline.insert(timeline.end(), raw.begin(), raw.end());
    for (size_t i = old_size; i < timeline.size(); i += static_cast<size_t>(n_spk_)) {
        relabel(timeline.data() + i);
    }
    if (!changed) {
        return;
    }
    const size_t window = std::min(raw_ring_.size(), timeline.size());
    const size_t ring_offset = raw_ring_.size() - window;
    const size_t timeline_offset = timeline.size() - window;
    std::copy(raw_ring_.begin() + ring_offset, raw_ring_.end(), timeline.begin() + timeline_offset);
    for (size_t i = timeline_offset; i < timeline.size(); i += static_cast<size_t>(n_spk_)) {
        relabel(timeline.data() + i);
    }
}

bool SortformerV2ChannelBirthGate::is_established(int speaker) const {
    return speaker >= 0 && speaker < n_spk_ && established_[speaker];
}

SortformerV2DiarGeometry SortformerV2DiarGeometry::from_model(const SortformerV2ModelConfig & config) {
    const auto & stream = config.streaming;
    return {
        static_cast<int>(stream.spkcache_len),
        static_cast<int>(stream.fifo_len),
        static_cast<int>(stream.chunk_len),
        static_cast<int>(stream.spkcache_update_period),
        static_cast<int>(stream.chunk_left_context),
        static_cast<int>(stream.chunk_right_context),
    };
}

SortformerV2DiarGeometry SortformerV2DiarGeometry::preset(const std::string & name) {
    if (name == "streaming") return {188, 0, 188, 188, 1, 1};
    if (name == "very_high_latency") return {188, 40, 340, 300, 1, 40};
    if (name == "high_latency") return {188, 124, 124, 124, 1, 1};
    if (name == "low_latency") return {188, 188, 6, 144, 1, 7};
    throw std::invalid_argument(
        "unknown diarizer geometry preset '" + name +
        "' (expected streaming | very_high_latency | high_latency | low_latency)");
}

SortformerV2DiarScoringConfig SortformerV2DiarScoringConfig::from_model(const SortformerV2ModelConfig & config) {
    const auto & stream = config.streaming;
    return {
        static_cast<int>(stream.spkcache_sil_frames_per_spk),
        stream.pred_score_threshold,
        stream.scores_boost_latest,
        stream.sil_threshold,
        stream.strong_boost_rate,
        stream.weak_boost_rate,
        stream.min_pos_scores_rate,
        static_cast<int>(stream.max_index),
    };
}

void SortformerV2DiarGeometry::validate(int n_spk, int sil_frames_per_spk, int pos_emb_max_len) const {
    auto fail = [](const std::string & message) { throw std::invalid_argument("diar geometry: " + message); };
    if (chunk_len < 1) fail("chunk_len must be >= 1");
    if (spkcache_update_period < 1) fail("spkcache_update_period must be >= 1");
    if (fifo_len < 0 || chunk_left_context < 0 || chunk_right_context < 0) {
        fail("fifo_len and contexts must be >= 0");
    }
    const int minimum_cache = (1 + sil_frames_per_spk) * n_spk;
    if (spkcache_len < minimum_cache) {
        fail("spkcache_len is smaller than the speaker/silence budget");
    }
    const int total = spkcache_len + fifo_len + chunk_left_context + chunk_len + chunk_right_context;
    if (total > pos_emb_max_len) {
        fail("state plus window exceeds the positional-encoding limit");
    }
}

SortformerV2AoscState::SortformerV2AoscState(
    const SortformerV2DiarGeometry & geometry,
    const SortformerV2DiarScoringConfig & scoring,
    int n_spk,
    int embedding_dim)
    : geometry_(geometry), scoring_(scoring), n_spk_(n_spk), embedding_dim_(embedding_dim) {
    if (n_spk_ <= 0 || embedding_dim_ <= 0) {
        throw std::invalid_argument("SortformerV2AoscState dimensions must be positive");
    }
    if (scoring_.max_index < n_spk_ ||
        !std::isfinite(scoring_.pred_score_threshold) ||
        !std::isfinite(scoring_.scores_boost_latest) ||
        !std::isfinite(scoring_.sil_threshold) ||
        !std::isfinite(scoring_.strong_boost_rate) ||
        !std::isfinite(scoring_.weak_boost_rate) ||
        !std::isfinite(scoring_.min_pos_scores_rate) ||
        scoring_.pred_score_threshold < 0.0f || scoring_.pred_score_threshold > 1.0f ||
        scoring_.sil_threshold < 0.0f || scoring_.sil_threshold > 1.0f ||
        scoring_.strong_boost_rate < 0.0f || scoring_.weak_boost_rate < 0.0f ||
        scoring_.min_pos_scores_rate < 0.0f) {
        throw std::invalid_argument("SortformerV2AoscState received invalid scoring configuration");
    }
    mean_sil_emb_.assign(static_cast<size_t>(embedding_dim_), 0.0f);
}

void SortformerV2AoscState::accumulate_silence(const float * embeddings, const float * probabilities, int frames) {
    int count = 0;
    std::vector<double> sum(static_cast<size_t>(embedding_dim_), 0.0);
    for (int f = 0; f < frames; ++f) {
        float activity = 0.0f;
        for (int s = 0; s < n_spk_; ++s) activity += probabilities[static_cast<size_t>(f) * n_spk_ + s];
        if (activity < scoring_.sil_threshold) {
            ++count;
            for (int d = 0; d < embedding_dim_; ++d) sum[static_cast<size_t>(d)] += embeddings[static_cast<size_t>(f) * embedding_dim_ + d];
        }
    }
    if (count == 0) return;
    const int64_t new_count = n_sil_frames_ + count;
    for (int d = 0; d < embedding_dim_; ++d) {
        mean_sil_emb_[static_cast<size_t>(d)] = static_cast<float>(
            (static_cast<double>(mean_sil_emb_[static_cast<size_t>(d)]) * n_sil_frames_ + sum[static_cast<size_t>(d)]) /
            static_cast<double>(new_count));
    }
    n_sil_frames_ = new_count;
}

std::vector<float> SortformerV2AoscState::update(
    const float * chunk_embeddings, int window_frames, const float * probabilities, int left_context, int right_context) {
    const int chunk_valid = window_frames - left_context - right_context;
    if (chunk_valid <= 0) return {};
    const int old_spk = spk_frames_;
    const int old_fifo = fifo_frames_;
    const float * fifo_probs = probabilities + static_cast<size_t>(old_spk) * n_spk_;
    const float * chunk_probs = probabilities + static_cast<size_t>(old_spk + old_fifo + left_context) * n_spk_;
    const float * chunk_valid_embeddings = chunk_embeddings + static_cast<size_t>(left_context) * embedding_dim_;
    std::vector<float> emitted(chunk_probs, chunk_probs + static_cast<size_t>(chunk_valid) * n_spk_);

    fifo_.insert(fifo_.end(), chunk_valid_embeddings, chunk_valid_embeddings + static_cast<size_t>(chunk_valid) * embedding_dim_);
    std::vector<float> fifo_probs_full(static_cast<size_t>(old_fifo + chunk_valid) * n_spk_);
    std::memcpy(fifo_probs_full.data(), fifo_probs, static_cast<size_t>(old_fifo) * n_spk_ * sizeof(float));
    std::memcpy(fifo_probs_full.data() + static_cast<size_t>(old_fifo) * n_spk_, chunk_probs, static_cast<size_t>(chunk_valid) * n_spk_ * sizeof(float));
    fifo_frames_ = old_fifo + chunk_valid;

    if (fifo_frames_ > geometry_.fifo_len) {
        int pop = geometry_.spkcache_update_period;
        pop = std::max(pop, chunk_valid - geometry_.fifo_len + old_fifo);
        pop = std::min(pop, fifo_frames_);
        const float * pop_embeddings = fifo_.data();
        const float * pop_probs = fifo_probs_full.data();
        accumulate_silence(pop_embeddings, pop_probs, pop);
        spkcache_.insert(spkcache_.end(), pop_embeddings, pop_embeddings + static_cast<size_t>(pop) * embedding_dim_);
        if (spkcache_preds_valid()) spkcache_preds_.insert(spkcache_preds_.end(), pop_probs, pop_probs + static_cast<size_t>(pop) * n_spk_);
        spk_frames_ += pop;
        if (spk_frames_ > geometry_.spkcache_len && !spkcache_preds_valid()) {
            spkcache_preds_.resize(static_cast<size_t>(spk_frames_) * n_spk_);
            std::memcpy(spkcache_preds_.data(), probabilities, static_cast<size_t>(old_spk) * n_spk_ * sizeof(float));
            std::memcpy(spkcache_preds_.data() + static_cast<size_t>(old_spk) * n_spk_, pop_probs, static_cast<size_t>(pop) * n_spk_ * sizeof(float));
        }
        fifo_.erase(fifo_.begin(), fifo_.begin() + static_cast<size_t>(pop) * embedding_dim_);
        fifo_frames_ -= pop;
        if (spk_frames_ > geometry_.spkcache_len) compress(spkcache_preds_);
    }
    return emitted;
}

void SortformerV2AoscState::compress(const std::vector<float> & cache_probs) {
    last_compression_input_preds_ = cache_probs;
    const int n = spk_frames_;
    const int capacity = geometry_.spkcache_len;
    const int per_speaker = capacity / n_spk_ - scoring_.sil_frames_per_spk;
    const int strong_k = static_cast<int>(std::floor(per_speaker * scoring_.strong_boost_rate));
    const int weak_k = static_cast<int>(std::floor(per_speaker * scoring_.weak_boost_rate));
    const int minimum_positive = static_cast<int>(std::floor(per_speaker * scoring_.min_pos_scores_rate));
    const float log_half = std::log(0.5f);
    std::vector<float> scores(static_cast<size_t>(n) * n_spk_);
    for (int f = 0; f < n; ++f) {
        float sum_log1p = 0.0f;
        for (int s = 0; s < n_spk_; ++s) {
            const float p = cache_probs[static_cast<size_t>(f) * n_spk_ + s];
            sum_log1p += std::log(std::max(1.0f - p, scoring_.pred_score_threshold));
        }
        for (int s = 0; s < n_spk_; ++s) {
            const float p = cache_probs[static_cast<size_t>(f) * n_spk_ + s];
            const float logp = std::log(std::max(p, scoring_.pred_score_threshold));
            const float log1p = std::log(std::max(1.0f - p, scoring_.pred_score_threshold));
            scores[static_cast<size_t>(f) * n_spk_ + s] = logp - log1p + sum_log1p - log_half;
        }
    }
    std::vector<int> positive_count(n_spk_, 0);
    for (int f = 0; f < n; ++f) {
        for (int s = 0; s < n_spk_; ++s) {
            const size_t index = static_cast<size_t>(f) * n_spk_ + s;
            if (!(cache_probs[index] > 0.5f)) scores[index] = kNegInf;
            if (scores[index] > 0.0f) ++positive_count[s];
        }
    }
    for (int s = 0; s < n_spk_; ++s) {
        if (positive_count[s] < minimum_positive) continue;
        for (int f = 0; f < n; ++f) {
            const size_t index = static_cast<size_t>(f) * n_spk_ + s;
            if (cache_probs[index] > 0.5f && !(scores[index] > 0.0f)) scores[index] = kNegInf;
        }
    }
    if (scoring_.scores_boost_latest > 0.0f) {
        for (int f = capacity; f < n; ++f) for (int s = 0; s < n_spk_; ++s) scores[static_cast<size_t>(f) * n_spk_ + s] += scoring_.scores_boost_latest;
    }
    for (int s = 0; s < n_spk_; ++s) for (int f : topk_column(scores, n, n_spk_, s, strong_k)) scores[static_cast<size_t>(f) * n_spk_ + s] -= 2.0f * log_half;
    for (int s = 0; s < n_spk_; ++s) for (int f : topk_column(scores, n, n_spk_, s, weak_k)) scores[static_cast<size_t>(f) * n_spk_ + s] -= log_half;

    const int padded_frames = n + scoring_.sil_frames_per_spk;
    std::vector<int64_t> flat(static_cast<size_t>(n_spk_) * padded_frames);
    std::iota(flat.begin(), flat.end(), 0);
    auto flat_score = [&](int64_t index) {
        const int f = static_cast<int>(index % padded_frames);
        if (f >= n) return kPosInf;
        const int s = static_cast<int>(index / padded_frames);
        return scores[static_cast<size_t>(f) * n_spk_ + s];
    };
    std::partial_sort(flat.begin(), flat.begin() + capacity, flat.end(), [&](int64_t a, int64_t b) {
        const float sa = flat_score(a), sb = flat_score(b);
        return sa != sb ? sa > sb : a < b;
    });
    std::vector<int64_t> picked(flat.begin(), flat.begin() + capacity);
    for (auto & index : picked) {
        if (flat_score(index) == kNegInf) index = static_cast<int64_t>(scoring_.max_index) * padded_frames + scoring_.max_index;
    }
    std::sort(picked.begin(), picked.end());
    last_compression_indices_ = picked;
    last_compression_frame_indices_.resize(static_cast<size_t>(capacity));
    last_compression_disabled_.resize(static_cast<size_t>(capacity));
    for (int j = 0; j < capacity; ++j) {
        const int64_t index = picked[static_cast<size_t>(j)];
        const int frame = static_cast<int>(index % padded_frames);
        const bool disabled = index >= static_cast<int64_t>(n_spk_) * padded_frames || frame >= n;
        last_compression_frame_indices_[static_cast<size_t>(j)] = disabled ? 0 : frame;
        last_compression_disabled_[static_cast<size_t>(j)] = disabled ? 1 : 0;
    }
    ++compression_count_;
    std::vector<float> new_cache(static_cast<size_t>(capacity) * embedding_dim_);
    std::vector<float> new_probs(static_cast<size_t>(capacity) * n_spk_, 0.0f);
    for (int j = 0; j < capacity; ++j) {
        const int64_t index = picked[static_cast<size_t>(j)];
        const int frame = static_cast<int>(index % padded_frames);
        const bool disabled = index >= static_cast<int64_t>(n_spk_) * padded_frames || frame >= n;
        if (disabled) {
            std::copy(mean_sil_emb_.begin(), mean_sil_emb_.end(), new_cache.begin() + static_cast<size_t>(j) * embedding_dim_);
        } else {
            std::copy_n(spkcache_.begin() + static_cast<size_t>(frame) * embedding_dim_, embedding_dim_, new_cache.begin() + static_cast<size_t>(j) * embedding_dim_);
            std::copy_n(cache_probs.begin() + static_cast<size_t>(frame) * n_spk_, n_spk_, new_probs.begin() + static_cast<size_t>(j) * n_spk_);
        }
    }
    last_compression_spkcache_preds_ = new_probs;
    spkcache_ = std::move(new_cache);
    spkcache_preds_ = std::move(new_probs);
    spk_frames_ = capacity;
}
