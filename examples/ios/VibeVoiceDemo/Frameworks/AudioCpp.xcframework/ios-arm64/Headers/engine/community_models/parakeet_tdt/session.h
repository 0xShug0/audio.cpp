#pragma once

#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/community_models/parakeet_tdt/assets.h"
#include "engine/community_models/parakeet_tdt/decoder.h"
#include "engine/community_models/parakeet_tdt/encoder.h"
#include "engine/community_models/parakeet_tdt/frontend.h"
#include "engine/community_models/parakeet_tdt/weights.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::community_models::parakeet_tdt {

std::shared_ptr<runtime::IVoiceModelLoader> make_parakeet_tdt_loader();

class ParakeetTDTSessionBase : public runtime::RuntimeSessionBase {
public:
    ParakeetTDTSessionBase(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);
    ~ParakeetTDTSessionBase() override;

protected:
    std::string family_impl() const;
    runtime::VoiceTaskKind task_kind_impl() const;
    runtime::RunMode run_mode_impl() const;
    ParakeetDecodeOptions decode_options_for_request(const runtime::TaskRequest & request) const;

    runtime::TaskSpec task_;
    std::shared_ptr<const ParakeetTDTAssets> assets_;
    std::shared_ptr<const engine::model_spec::ModelContract> contract_;
    std::shared_ptr<const ParakeetWeights> weights_;
    size_t weight_context_bytes_ = 3072ull * 1024ull * 1024ull;
    size_t encoder_graph_arena_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t decoder_graph_arena_bytes_ = 256ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType matmul_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    engine::assets::TensorStorageType conv_weight_storage_type_ = engine::assets::TensorStorageType::Native;
    bool encoder_flash_attention_ = false;
    ParakeetFrontend frontend_;
    std::unique_ptr<ParakeetEncoderRuntime> encoder_;
    std::unique_ptr<ParakeetDecoderRuntime> decoder_;
};

class ParakeetTDTOfflineSession final
    : public ParakeetTDTSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    ParakeetTDTOfflineSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    runtime::TaskResult run_long_form(
        const runtime::AudioBuffer & audio,
        const ParakeetDecodeOptions & options);

    std::string offline_mode_ = "full_context";
    int64_t center_samples_ = 0;
    int64_t left_context_samples_ = 0;
    int64_t right_context_samples_ = 0;
    int64_t auto_full_context_max_samples_ = 0;
};

class ParakeetTDTStreamingSession final
    : public ParakeetTDTSessionBase
    , public runtime::IStreamingVoiceTaskSession {
public:
    ParakeetTDTStreamingSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetTDTAssets> assets,
        std::shared_ptr<const engine::model_spec::ModelContract> contract);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    runtime::StreamEvent process_ready_windows(bool flush_tail);
    bool process_center_window(int64_t center_end_sample, bool flush_tail);
    ParakeetDecodedText merged_decode() const;

    int64_t center_samples_ = 0;
    int64_t left_context_samples_ = 0;
    int64_t right_context_samples_ = 0;
    int64_t next_center_start_sample_ = 0;
    int64_t decoded_frame_offset_ = 0;
    int64_t buffer_start_sample_ = 0;
    int64_t received_samples_ = 0;
    runtime::AudioBuffer streaming_audio_;
    ParakeetDecodeOptions streaming_decode_options_;
    std::vector<int32_t> token_ids_;
    std::vector<int32_t> token_frame_indices_;
    std::vector<int32_t> token_durations_;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
    bool finalized_ = false;
};

}  // namespace engine::community_models::parakeet_tdt
