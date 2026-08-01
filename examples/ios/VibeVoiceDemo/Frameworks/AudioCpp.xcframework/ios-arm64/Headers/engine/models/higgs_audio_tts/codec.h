#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/higgs_audio_tts/assets.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::higgs_audio_tts {

class HiggsCodecDecodeGraph;
class HiggsCodecEncodeGraph;

struct HiggsCodecVectorQuantizerWeights {
    core::TensorValue codebook;
    modules::LinearWeights project_in;
    modules::LinearWeights project_out;
};

struct HiggsCodecResidualUnitWeights {
    modules::Snake1dWeights snake1;
    modules::Conv1dWeights conv1;
    modules::Snake1dWeights snake2;
    modules::Conv1dWeights conv2;
};

struct HiggsCodecDecoderBlockWeights {
    modules::Snake1dWeights snake;
    modules::ConvTranspose1dWeights conv_transpose;
    std::vector<HiggsCodecResidualUnitWeights> residual_units;
};

struct HiggsCodecEncoderBlockWeights {
    modules::Snake1dWeights snake;
    modules::Conv1dWeights conv;
    std::vector<HiggsCodecResidualUnitWeights> residual_units;
};

struct HiggsCodecSemanticResidualUnitWeights {
    modules::Conv1dWeights conv1;
    modules::Conv1dWeights conv2;
};

struct HiggsCodecSemanticEncoderBlockWeights {
    std::vector<HiggsCodecSemanticResidualUnitWeights> residual_units;
    modules::Conv1dWeights conv;
};

struct HiggsCodecWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, core::TensorValue> semantic_model;
    std::vector<HiggsCodecVectorQuantizerWeights> quantizers;
    modules::Conv1dWeights acoustic_encoder_input;
    std::vector<HiggsCodecEncoderBlockWeights> acoustic_encoder_blocks;
    modules::Snake1dWeights acoustic_encoder_output_snake;
    modules::Conv1dWeights acoustic_encoder_output;
    modules::Conv1dWeights semantic_encoder_input;
    std::vector<HiggsCodecSemanticEncoderBlockWeights> semantic_encoder_blocks;
    modules::LinearWeights codec_project;
    modules::LinearWeights acoustic_project;
    modules::Conv1dWeights acoustic_decoder_input;
    std::vector<HiggsCodecDecoderBlockWeights> acoustic_decoder_blocks;
    modules::Snake1dWeights acoustic_decoder_output_snake;
    modules::Conv1dWeights acoustic_decoder_output;
};

struct HiggsCodecDecodeOutput {
    int sample_rate = 24000;
    int channels = 1;
    int64_t samples = 0;
    std::vector<float> values;
};

struct HiggsCodecEncodeOutput {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    int64_t codebooks = 0;
};

class HiggsCodecRuntime {
public:
    HiggsCodecRuntime(
        std::shared_ptr<const HiggsAssets> assets,
        core::ExecutionContext & execution,
        size_t weight_context_bytes,
        size_t decode_graph_arena_bytes,
        size_t encode_graph_arena_bytes,
        assets::TensorStorageType weight_storage_type);
    ~HiggsCodecRuntime();

    const HiggsCodecWeights & weights() const noexcept;
    ggml_backend_t backend() const noexcept;
    core::BackendType backend_type() const noexcept;
    int threads() const noexcept;
    size_t decode_graph_arena_bytes() const noexcept;
    size_t encode_graph_arena_bytes() const noexcept;

    HiggsCodecEncodeOutput encode_reference(const runtime::AudioBuffer & audio) const;
    HiggsCodecDecodeOutput decode_codes(
        const std::vector<int32_t> & codes,
        int64_t frames,
        int64_t codebooks) const;
    void release_encode_graph();
    void release_runtime_graphs();

private:
    std::shared_ptr<const HiggsAssets> assets_;
    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    size_t decode_graph_arena_bytes_ = 0;
    size_t encode_graph_arena_bytes_ = 0;
    std::shared_ptr<const HiggsCodecWeights> weights_;
    mutable std::unique_ptr<HiggsCodecEncodeGraph> encode_graph_;
    mutable std::unique_ptr<HiggsCodecDecodeGraph> decode_graph_;
};

HiggsCodecWeights load_higgs_codec_decode_weights(
    const HiggsAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type);

}  // namespace engine::models::higgs_audio_tts
