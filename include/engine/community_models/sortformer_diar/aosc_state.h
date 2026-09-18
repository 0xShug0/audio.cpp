// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "engine/community_models/sortformer_diar/assets.h"

#include <cstdint>
#include <string>
#include <vector>

namespace engine::community_models::sortformer_diar {

struct SortformerV2DiarGeometry {
    int spkcache_len = 188;
    int fifo_len = 0;
    int chunk_len = 188;
    int spkcache_update_period = 188;
    int chunk_left_context = 1;
    int chunk_right_context = 1;

    static SortformerV2DiarGeometry from_model(const SortformerV2ModelConfig & config);
    static SortformerV2DiarGeometry preset(const std::string & name);
    void validate(int n_spk, int sil_frames_per_spk, int pos_emb_max_len) const;
};

struct SortformerV2DiarScoringConfig {
    int sil_frames_per_spk = 3;
    float pred_score_threshold = 0.25f;
    float scores_boost_latest = 0.05f;
    float sil_threshold = 0.2f;
    float strong_boost_rate = 0.75f;
    float weak_boost_rate = 1.5f;
    float min_pos_scores_rate = 0.5f;
    int max_index = 99999;

    static SortformerV2DiarScoringConfig from_model(const SortformerV2ModelConfig & config);
};

class SortformerV2ChannelBirthGate {
public:
    explicit SortformerV2ChannelBirthGate(int n_spk);

    void reset();
    void append(const std::vector<float> & raw, std::vector<float> & timeline);
    bool is_established(int speaker) const;

private:
    bool observe(const float * probs);
    void relabel(float * probs) const;
    void push_raw(const float * probs);

    int n_spk_;
    int64_t frame_ = 0;
    std::vector<uint8_t> established_;
    std::vector<int> clean_frames_;
    std::vector<int> fading_frames_;
    std::vector<int64_t> last_win_;
    std::vector<float> raw_ring_;
};

class SortformerV2AoscState {
public:
    SortformerV2AoscState(
        const SortformerV2DiarGeometry & geometry,
        const SortformerV2DiarScoringConfig & scoring,
        int n_spk,
        int embedding_dim);

    std::vector<float> update(
        const float * chunk_embeddings,
        int window_frames,
        const float * probabilities,
        int left_context,
        int right_context);

    int spkcache_frames() const { return spk_frames_; }
    int fifo_frames() const { return fifo_frames_; }
    const std::vector<float> & spkcache() const { return spkcache_; }
    const std::vector<float> & fifo() const { return fifo_; }
    bool spkcache_preds_valid() const { return !spkcache_preds_.empty(); }
    const std::vector<float> & spkcache_preds() const { return spkcache_preds_; }
    const std::vector<float> & mean_silence_embedding() const { return mean_sil_emb_; }
    int64_t silence_frames() const { return n_sil_frames_; }
    const std::vector<int64_t> & last_compression_indices() const { return last_compression_indices_; }
    const std::vector<int> & last_compression_frame_indices() const { return last_compression_frame_indices_; }
    const std::vector<uint8_t> & last_compression_disabled() const { return last_compression_disabled_; }
    const std::vector<float> & last_compression_input_preds() const { return last_compression_input_preds_; }
    const std::vector<float> & last_compression_spkcache_preds() const { return last_compression_spkcache_preds_; }
    int compression_count() const { return compression_count_; }

private:
    void accumulate_silence(const float * embeddings, const float * probabilities, int frames);
    void compress(const std::vector<float> & cache_preds);

    SortformerV2DiarGeometry geometry_;
    SortformerV2DiarScoringConfig scoring_;
    int n_spk_;
    int embedding_dim_;
    std::vector<float> spkcache_;
    std::vector<float> spkcache_preds_;
    int spk_frames_ = 0;
    std::vector<float> fifo_;
    int fifo_frames_ = 0;
    std::vector<float> mean_sil_emb_;
    int64_t n_sil_frames_ = 0;
    std::vector<int64_t> last_compression_indices_;
    std::vector<int> last_compression_frame_indices_;
    std::vector<uint8_t> last_compression_disabled_;
    std::vector<float> last_compression_input_preds_;
    std::vector<float> last_compression_spkcache_preds_;
    int compression_count_ = 0;
};

}  // namespace engine::community_models::sortformer_diar
