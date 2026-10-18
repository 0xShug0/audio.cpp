#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/execution_context.h"

#include <memory>
#include <array>
#include <vector>

namespace engine::models::sam_audio {

struct DacVAEConfig {
    int64_t encoder_dim = 64;
    int64_t latent_dim = 1024;
    int64_t codebook_dim = 128;
    std::vector<int> encoder_rates{2, 8, 10, 12};
    std::vector<int> decoder_rates{12, 10, 8, 2};
    std::vector<int> watermark_rates{8, 5, 4, 2};
};

class DacVAEEncoder {
public:
    DacVAEEncoder(std::shared_ptr<const assets::TensorSource> source,
                 core::ExecutionContext & execution, DacVAEConfig config, bool memory_bounded = false);
    ~DacVAEEncoder();
    std::vector<float> encode(const std::vector<float> & audio);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class CodecTileStage { Main, WatermarkEncoder, EncoderLSTM, Bridge, DecoderLSTM, WatermarkDecoder };

class DacVAEDecoderModule {
public:
    using RecurrentState = std::array<core::TensorValue, 4>;
    DacVAEDecoderModule(std::shared_ptr<const assets::TensorSource> source,
                       core::ExecutionContext & execution, DacVAEConfig config);
    ~DacVAEDecoderModule();
    core::TensorValue project_latents(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue build_block(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                 size_t block) const;
    core::TensorValue embed_message(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                   const core::TensorValue & message_indices) const;
    core::TensorValue watermark_base(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_encode(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_decode(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_encode_convs(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_encode_output(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_decode_input(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_decode_convs(core::ModuleBuildContext & ctx, const core::TensorValue & input) const;
    core::TensorValue watermark_lstm(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                                    bool decoder, RecurrentState * state = nullptr) const;
    int64_t context_frames(CodecTileStage stage) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DacVAEDecoder {
public:
    DacVAEDecoder(std::shared_ptr<const assets::TensorSource> source,
                 core::ExecutionContext & execution, DacVAEConfig config, bool memory_bounded = false);
    ~DacVAEDecoder();
    std::vector<float> decode(const std::vector<float> & latents, int64_t batch, int64_t frames,
                              const std::vector<int32_t> & message_bits);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::sam_audio
