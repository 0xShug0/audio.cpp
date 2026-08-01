#pragma once

#include "engine/models/vevo2/assets.h"
#include "engine/models/vevo2/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace engine::assets {
enum class TensorStorageType;
}

namespace engine::core {
class ExecutionContext;
}

namespace engine::models::vevo2 {

struct Vevo2CocoTokenizerWeights;
struct Vevo2CocoTokenizerGraph;

class Vevo2ProsodyTokenizerRuntime final {
public:
    Vevo2ProsodyTokenizerRuntime(
        const Vevo2Assets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Vevo2ProsodyTokenizerRuntime();

    Vevo2TokenSequence encode(
        const runtime::AudioBuffer & prosody_audio,
        const std::optional<runtime::AudioBuffer> & style_ref_audio,
        const Vevo2GenerationOptions & generation) const;
    Vevo2TokenSequence encode_chromagram_features(
        const std::vector<float> & chromagram_features,
        int64_t feature_frames) const;

private:
    Vevo2CocoTokenizerGraph & ensure_graph(int64_t feature_frames, bool uses_whisper) const;

    Vevo2CocoTokenizerConfig config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<const Vevo2CocoTokenizerWeights> weights_;
    mutable std::unique_ptr<Vevo2CocoTokenizerGraph> graph_;
    std::string name_;
};

class Vevo2ContentStyleTokenizerRuntime final {
public:
    Vevo2ContentStyleTokenizerRuntime(
        const Vevo2Assets & assets,
        engine::core::ExecutionContext & execution_context,
        size_t weight_context_bytes,
        size_t graph_context_bytes,
        engine::assets::TensorStorageType matmul_weight_storage_type,
        engine::assets::TensorStorageType conv_weight_storage_type);
    ~Vevo2ContentStyleTokenizerRuntime();

    Vevo2TokenSequence encode_style_reference(
        const std::optional<runtime::AudioBuffer> & style_ref_audio,
        const std::optional<std::vector<float>> & whisper_features,
        int64_t feature_frames,
        const Vevo2GenerationOptions & generation) const;
    Vevo2TokenSequence encode_timbre_reference(
        const runtime::AudioBuffer & timbre_ref_audio,
        const std::vector<float> & whisper_features,
        int64_t feature_frames) const;
    Vevo2TokenSequence encode_feature_frames(
        const std::vector<float> & whisper_features,
        const std::vector<float> & chromagram_features,
        int64_t feature_frames) const;
    Vevo2TokenSequence encode_shifted_reference(
        const runtime::AudioBuffer & audio,
        const std::vector<float> & whisper_features,
        int64_t feature_frames,
        int pitch_shift_steps) const;

private:
    Vevo2CocoTokenizerGraph & ensure_graph(int64_t feature_frames, bool uses_whisper) const;

    Vevo2CocoTokenizerConfig config_;
    engine::core::ExecutionContext & execution_context_;
    size_t graph_context_bytes_ = 0;
    std::shared_ptr<const engine::assets::TensorSource> weight_source_;
    std::shared_ptr<const Vevo2CocoTokenizerWeights> weights_;
    mutable std::unique_ptr<Vevo2CocoTokenizerGraph> graph_;
    std::vector<float> whisper_mean_;
    std::vector<float> whisper_std_;
    std::string name_;
};

}  // namespace engine::models::vevo2
