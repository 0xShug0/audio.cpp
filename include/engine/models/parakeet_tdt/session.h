#pragma once

#include "engine/framework/decoders/tdt_decoder_algorithm.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/parakeet_tdt/assets.h"
#include "engine/models/parakeet_tdt/decoder.h"
#include "engine/models/parakeet_tdt/frontend.h"
#include "engine/models/parakeet_tdt/pre_encode.h"
#include "engine/models/parakeet_tdt/weights.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace engine::models::parakeet_tdt {

inline constexpr int64_t kParakeetLongformContext = 256;

class ParakeetTDTSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    ParakeetTDTSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const ParakeetAssets> assets);
    ~ParakeetTDTSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;

    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finalize() override;

private:
    enum class BufferedEncoderVariant {
        FullContext,
        LongContext,
    };

    struct FullContextEncoderGraph;
    struct SharedEncoderState {
        std::vector<float> projected_output_scratch;
        std::vector<float> hidden_output_scratch;
        std::vector<float> pos_emb_cache;
        std::vector<int32_t> pad_mask_cache;
        std::vector<int32_t> keep_mask_cache;
        int64_t cached_position_frames = -1;
        int64_t cached_mask_frames = -1;
        int64_t cached_mask_valid_frames = -1;
        int64_t cached_pos_frames = -1;
        int64_t position_cache_generation = 0;
        int64_t mask_cache_generation = 0;
        bool cached_position_long_context = false;
    };
    struct FullContextEncoderState {
        FullContextEncoderState();
        ~FullContextEncoderState();
        FullContextEncoderState(FullContextEncoderState &&) noexcept;
        FullContextEncoderState & operator=(FullContextEncoderState &&) noexcept;
        FullContextEncoderState(const FullContextEncoderState &) = delete;
        FullContextEncoderState & operator=(const FullContextEncoderState &) = delete;
        std::unique_ptr<FullContextEncoderGraph> graph;
        std::vector<float> attention_bias_cache;
        int64_t attention_cache_generation = 0;
        int64_t cached_attention_frames = -1;
        int64_t cached_attention_valid_frames = -1;
    };
    struct LongContextEncoderGraph;
    struct LongContextEncoderState {
        LongContextEncoderState();
        ~LongContextEncoderState();
        LongContextEncoderState(LongContextEncoderState &&) noexcept;
        LongContextEncoderState & operator=(LongContextEncoderState &&) noexcept;
        LongContextEncoderState(const LongContextEncoderState &) = delete;
        LongContextEncoderState & operator=(const LongContextEncoderState &) = delete;
        std::unique_ptr<LongContextEncoderGraph> graph;
    };
    struct StreamingChunkEncoderGraph;
    struct StreamingChunkEncoderState {
        StreamingChunkEncoderState();
        ~StreamingChunkEncoderState();
        StreamingChunkEncoderState(StreamingChunkEncoderState &&) noexcept;
        StreamingChunkEncoderState & operator=(StreamingChunkEncoderState &&) noexcept;
        StreamingChunkEncoderState(const StreamingChunkEncoderState &) = delete;
        StreamingChunkEncoderState & operator=(const StreamingChunkEncoderState &) = delete;
        std::unique_ptr<StreamingChunkEncoderGraph> graph;
        std::vector<float> projected_output_scratch;
        std::vector<float> hidden_output_scratch;
    };
    struct StreamingContextSizes {
        int64_t left_samples = 0;
        int64_t chunk_samples = 0;
        int64_t right_samples = 0;
        int64_t window_samples = 0;
        int64_t left_frames = 0;
        int64_t chunk_frames = 0;
        int64_t right_frames = 0;
        int64_t window_frames = 0;
        int64_t encoder_frame_to_samples = 0;
    };
    struct BufferedCapacityContract {
        int64_t audio_samples = 0;
        int64_t pre_encode_frames = 0;
    };
    struct FullContextCapacityTier {
        BufferedCapacityContract contract;
        ParakeetPreEncodeScratch pre_encode_scratch;
        FullContextEncoderState encoder;
    };
    struct LongContextCapacityTier {
        BufferedCapacityContract contract;
        ParakeetPreEncodeScratch pre_encode_scratch;
        LongContextEncoderState encoder;
    };
    struct BufferedDecodeTimings {
        double frontend_ms = 0.0;
        double pre_encode_ms = 0.0;
        double encoder_ms = 0.0;
        double decoder_ms = 0.0;
        int64_t buffered_capacity_frames = 0;
        int64_t buffered_valid_frames = 0;
    };
    struct BufferedChunkDecode {
        ParakeetDecodeResult decoded;
        BufferedDecodeTimings timings;
    };
    struct StreamingState {
        struct Timings {
            double frontend_ms = 0.0;
            double pre_encode_ms = 0.0;
            double encoder_ms = 0.0;
            double encoder_ensure_graph_ms = 0.0;
            double encoder_graph_build_ms = 0.0;
            double encoder_input_write_ms = 0.0;
            double encoder_pos_emb_ms = 0.0;
            double encoder_attention_mask_ms = 0.0;
            double encoder_keep_mask_ms = 0.0;
            double encoder_graph_compute_ms = 0.0;
            double encoder_hidden_read_ms = 0.0;
            double encoder_projected_read_ms = 0.0;
            double encoder_output_zero_ms = 0.0;
            double decoder_ms = 0.0;
            double timestamps_ms = 0.0;
            int64_t chunk_windows = 0;
            int64_t encoder_graph_rebuilds = 0;
            bool wall_started = false;
            std::chrono::steady_clock::time_point wall_start{};
        };
        StreamingContextSizes context;
        Timings timings;
        ParakeetStreamingDecoderState decoder_state;
        ParakeetDecodeResult merged_decoded;
        runtime::Transcript latest_text;
        std::vector<runtime::WordTimestamp> latest_word_timestamps;
        bool has_latest_result = false;
        int64_t next_chunk_start_sample = 0;
        int64_t decoded_frame_offset = 0;
        bool configured = false;
    };

    runtime::TaskResult run_one_shot_request(const runtime::AudioBuffer & audio);
    runtime::TaskResult flush_streaming_request();
    static decoders::TdtDecoderAlgorithm parse_decoder_algorithm_option(const runtime::SessionOptions & options);
    runtime::TaskResult transcribe_buffered_audio(
        const runtime::AudioBuffer & audio,
        BufferedEncoderVariant encoder_variant,
        ParakeetTraceContractMode trace_mode,
        bool emit_streaming_trace);
    BufferedChunkDecode decode_buffered_audio_chunk(
        const runtime::AudioBuffer & audio,
        const BufferedCapacityContract & contract,
        size_t contract_index,
        BufferedEncoderVariant encoder_variant,
        ParakeetTraceContractMode trace_mode,
        bool emit_streaming_trace);
    void ensure_full_context_encoder_graph(FullContextEncoderState & state, int64_t frames);
    void ensure_long_context_encoder_graph(LongContextEncoderState & state, int64_t frames);
    void ensure_streaming_chunk_encoder_graph(int64_t frames, int64_t left_context, int64_t right_context);
    void initialize_shared_encoder_cache(
        const ParakeetPreEncodeBatch & pre_encode,
        BufferedEncoderVariant encoder_variant,
        FullContextEncoderState * full_context_encoder);
    const std::vector<float> & run_offline_graph_encoder(
        const ParakeetPreEncodeBatch & pre_encode,
        FullContextEncoderState & state);
    const std::vector<float> & run_long_context_graph_encoder(
        const ParakeetPreEncodeBatch & pre_encode,
        LongContextEncoderState & state);
    const std::vector<float> & run_streaming_chunk_graph_encoder(
        const ParakeetPreEncodeBatch & pre_encode,
        int64_t left_context,
        int64_t chunk_frames,
        int64_t right_context);
    ParakeetDecodeResult run_shared_decoder(
        const std::vector<float> & encoder_projected,
        int64_t frames,
        decoders::TdtDecoderAlgorithm decoder_algorithm,
        ParakeetTraceContractMode trace_mode);
    static BufferedCapacityContract make_capacity_contract_for_target_frames(
        const ParakeetAssets & assets,
        int64_t target_frames);
    std::pair<const BufferedCapacityContract *, size_t> find_buffered_capacity_contract(
        int64_t capacity_audio_samples,
        BufferedEncoderVariant encoder_variant);
    runtime::GraphCapacityController & buffered_capacity_controller(BufferedEncoderVariant encoder_variant);
    const runtime::GraphCapacityController & buffered_capacity_controller(BufferedEncoderVariant encoder_variant) const;
    runtime::DiscreteGraphCapacityAdapter make_buffered_capacity_adapter(BufferedEncoderVariant encoder_variant);
    std::vector<int64_t> buffered_capacity_catalog(BufferedEncoderVariant encoder_variant) const;
    std::vector<int64_t> prepared_buffered_capacities(BufferedEncoderVariant encoder_variant) const;
    void prepare_buffered_capacity(BufferedEncoderVariant encoder_variant, int64_t capacity_audio_samples);
    void initialize_buffered_capacity_contracts();
    void prepare_buffered_tier(size_t contract_index, BufferedEncoderVariant encoder_variant);
    void prepare_streaming_resources();
    void ensure_streaming_configured();
    std::optional<runtime::StreamEvent> maybe_process_streaming_window(bool final_chunk);
    runtime::StreamEvent make_stream_event_from_merged_decode() const;
    runtime::TaskResult make_task_result_from_merged_decode() const;

    runtime::TaskSpec task_;
    runtime::SessionOptions options_;
    std::shared_ptr<const ParakeetAssets> assets_;
    runtime::GraphCapacityController full_context_capacity_controller_;
    runtime::GraphCapacityController long_context_capacity_controller_;
    decoders::TdtDecoderAlgorithm decoder_algorithm_ = decoders::TdtDecoderAlgorithm::GreedyDurationLoop;
    std::shared_ptr<const ParakeetTDTWeights> weights_;
    size_t weight_context_bytes_ = 1536ull * 1024ull * 1024ull;
    assets::TensorStorageType matmul_weight_storage_type_ = assets::TensorStorageType::Native;
    assets::TensorStorageType conv_weight_storage_type_ = assets::TensorStorageType::Native;
    std::unique_ptr<ParakeetSharedDecoder> shared_decoder_;
    std::vector<runtime::AudioBuffer> frontend_audio_batch_;
    ParakeetFrontendBatch frontend_batch_;
    ParakeetFrontendScratch frontend_scratch_;
    ParakeetPreEncodeBatch pre_encode_batch_;
    ParakeetPreEncodeScratch streaming_pre_encode_scratch_;
    std::vector<FullContextCapacityTier> full_context_tiers_;
    std::vector<LongContextCapacityTier> long_context_tiers_;
    size_t latest_full_context_tier_index_ = 0;
    size_t latest_long_context_tier_index_ = 0;
    size_t full_context_pre_encode_graphs_built_ = 0;
    size_t full_context_encoder_graphs_built_ = 0;
    size_t long_context_pre_encode_graphs_built_ = 0;
    size_t long_context_encoder_graphs_built_ = 0;
    SharedEncoderState shared_encoder_;
    StreamingChunkEncoderState streaming_chunk_encoder_;
    int stream_sample_rate_ = 0;
    int stream_channels_ = 0;
    int64_t next_stream_sample_ = 0;
    std::vector<float> stream_samples_;
    std::vector<int64_t> frontend_audio_lengths_override_;
    StreamingState streaming_state_;
};

}  // namespace engine::models::parakeet_tdt
