#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/community_models/minimax_h3/assets.h"
#include "engine/community_models/minimax_h3/types.h"

#include "ggml-alloc.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace engine::models::minimax_h3 {

class MiniMaxH3DitWeightStore {
public:
    MiniMaxH3DitWeightStore(
        engine::core::ExecutionContext & execution_context,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        size_t weight_context_bytes);
    MiniMaxH3DitWeightStore(
        engine::core::ExecutionContext & execution_context,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        size_t weight_context_bytes,
        const std::vector<std::string> & required_names,
        const std::vector<std::string> & prefix_filters,
        bool load_adaln_table);

    const engine::core::TensorValue & require(std::string_view name) const;
    const engine::core::TensorValue * find(std::string_view name) const;

    engine::core::ExecutionContext & execution;
    std::vector<float> adaln_curve_table;

private:
    std::shared_ptr<const engine::assets::TensorSource> source_;
    engine::core::BackendWeightStore store_;
    std::unordered_map<std::string, engine::core::TensorValue> weights_;
};

struct DitGraphResult {
    std::vector<float> video;
    std::vector<float> audio;
    std::vector<float> hidden;
    std::vector<float> next_video;
    std::vector<float> next_audio;
};

class MiniMaxH3DitLayerwiseRuntime {
public:
    MiniMaxH3DitLayerwiseRuntime(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const engine::assets::TensorSource> tensor_source,
        const MiniMaxH3Config & cfg,
        const std::vector<float> & prompt,
        size_t weight_context_bytes,
        int64_t layer_batch,
        int64_t mlp_chunk_tokens);

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool split_audio_timestep,
        bool read_logits,
        bool read_next_rows,
        DitGraphResult & result);

    double input_upload_ms() const;
    double output_read_ms() const;
    const std::vector<float> & adaln_curve_table() const;

private:
    engine::core::ExecutionContext & execution_;
    std::shared_ptr<const engine::assets::TensorSource> tensor_source_;
    MiniMaxH3Config cfg_;
    std::vector<float> prompt_;
    size_t weight_context_bytes_ = 0;
    int64_t layer_batch_ = 1;
    int64_t mlp_chunk_tokens_ = 0;
    std::vector<float> adaln_curve_table_;
    double input_upload_ms_ = 0.0;
    double output_read_ms_ = 0.0;
};

class MiniMaxH3DitFirstBlockCacheRuntime {
public:
    MiniMaxH3DitFirstBlockCacheRuntime(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3GenerateRequest & request,
        const std::vector<float> & prompt);
    ~MiniMaxH3DitFirstBlockCacheRuntime();

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool split_audio_timestep,
        DitGraphResult & result);

    double input_upload_ms() const;
    double output_read_ms() const;
    int64_t full_steps() const;
    int64_t cached_steps() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MiniMaxH3DitGraph {
public:
    struct PackedSequenceLayout {
        int64_t text_len = 0;
        int64_t audio_start = 0;
        int64_t audio_rows = 0;
        int64_t video_start = 0;
        int64_t video_rows = 0;
        int64_t total = 0;
    };

    struct DitOutput {
        engine::core::TensorValue video_logits;
        engine::core::TensorValue audio_logits;
        engine::core::TensorValue hidden;
        engine::core::TensorValue next_video;
        engine::core::TensorValue next_audio;
    };

    MiniMaxH3DitGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const std::vector<float> & prompt,
        bool include_sampler_update = true);
    ~MiniMaxH3DitGraph();

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool split_audio_timestep,
        bool read_logits,
        bool read_hidden,
        bool read_next_rows,
        DitGraphResult & result);

    double input_upload_ms() const;
    double output_read_ms() const;
    const PackedSequenceLayout & layout() const;
    ggml_tensor * video_state_tensor() const;
    ggml_tensor * audio_state_tensor() const;
    ggml_tensor * video_logits_tensor() const;
    ggml_tensor * audio_logits_tensor() const;

private:
    MiniMaxH3DitWeightStore & weights_;
    const MiniMaxH3Config cfg_;
    const int64_t text_len_;
    const PackedSequenceLayout layout_;
    ggml_context * ctx_ = nullptr;
    ggml_context * input_ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    engine::core::HostGraphPlan plan_;
    engine::core::TensorValue prompt_t_;
    engine::core::TensorValue audio_t_;
    engine::core::TensorValue video_t_;
    engine::core::TensorValue audio_mask_t_;
    engine::core::TensorValue video_mask_t_;
    engine::core::TensorValue text_positions_t_;
    engine::core::TensorValue time_t_;
    engine::core::TensorValue sigma_delta_audio_t_;
    engine::core::TensorValue sigma_delta_video_t_;
    engine::core::TensorValue combined_t_;
    engine::core::TensorValue inverse_t_;
    engine::core::TensorValue cos_t_;
    engine::core::TensorValue sin_t_;
    DitOutput out_;
    bool include_sampler_update_ = true;
    std::vector<int32_t> combined_shared_timestep_;
    std::vector<int32_t> inverse_shared_timestep_;
    std::vector<int32_t> combined_split_timestep_;
    std::vector<int32_t> inverse_split_timestep_;
    bool indices_uploaded_ = false;
    bool indices_are_split_ = false;
    double input_upload_ms_ = 0.0;
    double output_read_ms_ = 0.0;
};

class MiniMaxH3DitFinalGraph {
public:
    MiniMaxH3DitFinalGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const MiniMaxH3DitGraph::PackedSequenceLayout & layout);
    ~MiniMaxH3DitFinalGraph();

    void run(
        const std::vector<float> & hidden,
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool split_audio_timestep,
        DitGraphResult & result);

    double input_upload_ms() const;
    double output_read_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class MiniMaxH3DitCfgGraph {
public:
    MiniMaxH3DitCfgGraph(
        MiniMaxH3DitWeightStore & weights,
        const MiniMaxH3Config & cfg,
        const std::vector<float> & positive_prompt,
        const std::vector<float> & negative_prompt,
        bool include_sampler_update = true);
    ~MiniMaxH3DitCfgGraph();

    void run(
        const std::vector<float> & audio_rows,
        const std::vector<float> & video_rows,
        const float * timestep_values,
        size_t timestep_count,
        float sigma_delta_audio,
        float sigma_delta_video,
        bool split_audio_timestep,
        float guidance_scale,
        bool read_logits,
        bool read_hidden,
        bool read_next_rows,
        DitGraphResult & result);

    double input_upload_ms() const;
    double output_read_ms() const;
    const MiniMaxH3DitGraph::PackedSequenceLayout & layout() const;
    ggml_tensor * video_state_tensor() const;
    ggml_tensor * audio_state_tensor() const;
    ggml_tensor * video_logits_tensor() const;
    ggml_tensor * audio_logits_tensor() const;

private:
    struct BranchState {
        int64_t text_len = 0;
        MiniMaxH3DitGraph::PackedSequenceLayout layout{};
        engine::core::TensorValue prompt;
        engine::core::TensorValue audio;
        engine::core::TensorValue video;
        engine::core::TensorValue audio_mask;
        engine::core::TensorValue video_mask;
        engine::core::TensorValue text_positions;
        engine::core::TensorValue combined;
        engine::core::TensorValue inverse;
        engine::core::TensorValue cos;
        engine::core::TensorValue sin;
        std::vector<int32_t> combined_shared_timestep;
        std::vector<int32_t> inverse_shared_timestep;
        std::vector<int32_t> combined_split_timestep;
        std::vector<int32_t> inverse_split_timestep;
    };

    MiniMaxH3DitWeightStore & weights_;
    const MiniMaxH3Config cfg_;
    BranchState positive_;
    BranchState negative_;
    ggml_context * ctx_ = nullptr;
    ggml_context * input_ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    engine::core::HostGraphPlan plan_;
    engine::core::TensorValue time_t_;
    engine::core::TensorValue guidance_scale_t_;
    engine::core::TensorValue sigma_delta_audio_t_;
    engine::core::TensorValue sigma_delta_video_t_;
    MiniMaxH3DitGraph::DitOutput out_;
    bool include_sampler_update_ = true;
    bool indices_uploaded_ = false;
    bool indices_are_split_ = false;
    double input_upload_ms_ = 0.0;
    double output_read_ms_ = 0.0;
};

}  // namespace engine::models::minimax_h3
