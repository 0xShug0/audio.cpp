#pragma once

#include "engine/framework/runtime/session_base.h"
#include "engine/models/vevo2/ar.h"
#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/components.h"
#include "engine/models/vevo2/fm.h"
#include "engine/models/vevo2/prompt_builder.h"
#include "engine/models/vevo2/vocoder.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
enum class TensorStorageType;
}

namespace engine::core {
enum class BackendType;
class BackendWeightStore;
class ExecutionContext;
}

namespace engine::models::vevo2 {

class Vevo2WhisperEmbeddingRuntime final {
public:
    Vevo2WhisperEmbeddingRuntime(
        const Vevo2Assets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Vevo2WhisperEmbeddingRuntime();

    Vevo2WhisperEmbeddingRuntime(const Vevo2WhisperEmbeddingRuntime &) = delete;
    Vevo2WhisperEmbeddingRuntime & operator=(const Vevo2WhisperEmbeddingRuntime &) = delete;

    std::vector<float> encode_log_mel(const std::vector<float> & log_mel);
    std::vector<float> extract_features(const runtime::AudioBuffer & audio, int64_t target_frames, size_t threads = 0);

private:
    struct Graph;

    void ensure_graph();

    engine::core::ExecutionContext & execution_context_;
    engine::core::BackendType backend_type_;
    size_t graph_context_bytes_ = 0;
    engine::modules::WhisperEmbeddingConfig config_;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<engine::core::BackendWeightStore> weight_store_;
    engine::modules::WhisperEmbeddingWeights weights_;
    std::unique_ptr<Graph> graph_;
};

class Vevo2Session final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession {
public:
    Vevo2Session(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const Vevo2Assets> assets);

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

private:
    struct AudioFeatureCacheEntry {
        uint64_t key = 0;
        int sample_rate = 0;
        int channels = 0;
        int64_t target_frames = 0;
        size_t samples = 0;
        std::vector<float> features;
    };

    struct AudioTokenCacheEntry {
        uint64_t key = 0;
        int sample_rate = 0;
        int channels = 0;
        int64_t feature_frames = 0;
        size_t samples = 0;
        Vevo2TokenSequence tokens;
    };

    Vevo2Request make_request(const runtime::TaskRequest & request) const;
    std::vector<float> cached_whisper_features(
        const runtime::AudioBuffer & audio,
        int64_t target_frames,
        size_t threads);
    Vevo2TokenSequence cached_content_style_tokens(
        const runtime::AudioBuffer & audio,
        const std::vector<float> & whisper_features,
        int64_t feature_frames);

    runtime::TaskSpec task_;
    std::shared_ptr<const Vevo2Assets> assets_;
    size_t ar_weight_context_bytes_ = 2ull * 1024ull * 1024ull * 1024ull;
    size_t ar_prefill_graph_context_bytes_ = 512ull * 1024ull * 1024ull;
    size_t ar_decode_graph_context_bytes_ = 512ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType ar_weight_storage_type_;
    engine::core::ExecutionContext reference_execution_context_;
    size_t whisper_weight_context_bytes_ = 256ull * 1024ull * 1024ull;
    size_t whisper_graph_context_bytes_ = 512ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType whisper_matmul_weight_storage_type_;
    engine::assets::TensorStorageType whisper_conv_weight_storage_type_;
    size_t tokenizer_weight_context_bytes_ = 768ull * 1024ull * 1024ull;
    size_t tokenizer_graph_context_bytes_ = 768ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType tokenizer_matmul_weight_storage_type_;
    engine::assets::TensorStorageType tokenizer_conv_weight_storage_type_;
    size_t fm_weight_context_bytes_ = 2ull * 1024ull * 1024ull * 1024ull;
    size_t fm_graph_context_bytes_ = 1024ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType fm_matmul_weight_storage_type_;
    engine::assets::TensorStorageType fm_conv_weight_storage_type_;
    size_t vocoder_weight_context_bytes_ = 1024ull * 1024ull * 1024ull;
    size_t vocoder_graph_context_bytes_ = 768ull * 1024ull * 1024ull;
    engine::assets::TensorStorageType vocoder_matmul_weight_storage_type_;
    engine::assets::TensorStorageType vocoder_conv_weight_storage_type_;
    Vevo2WhisperEmbeddingRuntime whisper_embedding_;
    Vevo2ProsodyTokenizerRuntime prosody_tokenizer_;
    Vevo2ContentStyleTokenizerRuntime content_style_tokenizer_;
    Vevo2AutoregressiveRuntime autoregressive_model_;
    Vevo2FlowMatchingRuntime flow_matching_model_;
    Vevo2VocoderRuntime vocoder_;
    Vevo2PromptBuilder prompt_builder_;
    std::vector<AudioFeatureCacheEntry> whisper_feature_cache_;
    std::vector<AudioTokenCacheEntry> content_style_token_cache_;
};

}  // namespace engine::models::vevo2
