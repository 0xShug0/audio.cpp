#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <memory>
#include <vector>

namespace engine::codecs::s3gen {

struct S3FlowEncoderWeights {
    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };
    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
    };
    struct RelativeAttentionWeights {
        LinearWeights q;
        LinearWeights k;
        LinearWeights v;
        LinearWeights out;
        LinearWeights pos;
        engine::core::TensorValue pos_bias_u_tensor;
        engine::core::TensorValue pos_bias_v_tensor;
    };
    struct FeedForwardWeights {
        LinearWeights w1;
        LinearWeights w2;
    };
    struct EncoderLayerWeights {
        LayerNormWeights norm_mha;
        RelativeAttentionWeights attn;
        LayerNormWeights norm_ff;
        FeedForwardWeights ff;
    };
    LinearWeights speaker_affine;
    LinearWeights encoder_proj;
    LinearWeights embed_linear;
    LayerNormWeights embed_norm;
    Conv1dWeights prelook_conv1;
    Conv1dWeights prelook_conv2;
    std::vector<EncoderLayerWeights> encoders;
    Conv1dWeights up_layer_conv;
    LinearWeights up_embed_linear;
    LayerNormWeights up_embed_norm;
    std::vector<EncoderLayerWeights> up_encoders;
    LayerNormWeights after_norm;
    engine::core::TensorValue input_embedding_tensor;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

struct S3FlowDecoderWeights {
    struct LayerNormWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
    };
    struct LinearWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };
    struct Conv1dWeights {
        engine::core::TensorValue weight_tensor;
        engine::core::TensorValue bias_tensor;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        bool use_bias = false;
    };
    struct CausalBlockWeights {
        Conv1dWeights conv;
        LayerNormWeights norm;
    };
    struct ResnetBlockWeights {
        LinearWeights time_mlp;
        CausalBlockWeights block1;
        CausalBlockWeights block2;
        Conv1dWeights res_conv;
    };
    struct TransformerBlockWeights {
        LayerNormWeights norm1;
        LinearWeights attn_q;
        LinearWeights attn_k;
        LinearWeights attn_v;
        LinearWeights attn_out;
        LayerNormWeights norm3;
        LinearWeights ff_proj_in;
        LinearWeights ff_proj_out;
    };
    struct DownBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
        Conv1dWeights downsample;
    };
    struct MidBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
    };
    struct UpBlockWeights {
        ResnetBlockWeights resnet;
        std::vector<TransformerBlockWeights> transformers;
        Conv1dWeights upsample;
    };

    LinearWeights time_mlp_1;
    LinearWeights time_mlp_2;
    std::vector<DownBlockWeights> down_blocks;
    std::vector<MidBlockWeights> mid_blocks;
    std::vector<UpBlockWeights> up_blocks;
    CausalBlockWeights final_block;
    Conv1dWeights final_proj;
    // Meanflow-distilled decoders (Chatterbox Turbo) mix a second "r" (end-time) sinusoidal
    // embedding into the time embedding via this diagonal-init, no-bias linear layer before
    // feeding the UNet1D estimator; see decoder.py::get_intmeanflow_time_mixer upstream. Unset
    // (weight_tensor.tensor == nullptr) for the base Chatterbox 10-step CFG decoder.
    bool meanflow = false;
    LinearWeights time_embed_mixer;
    const engine::core::ExecutionContext * execution_context = nullptr;
    std::shared_ptr<engine::core::BackendWeightStore> store;
};

}  // namespace engine::codecs::s3gen

