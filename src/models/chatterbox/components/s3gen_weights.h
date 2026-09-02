#pragma once

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::chatterbox {

struct S3SpeakerEncoderWeights {
    struct BatchNorm1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
        bool affine = true;
    };

    struct BatchNorm2dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        std::vector<float> running_mean;
        std::vector<float> running_var;
    };

    struct Conv1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool use_bias = false;
    };

    struct Conv2dWeights {
        std::vector<float> weight;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel_h = 0;
        int64_t kernel_w = 0;
        int64_t stride_h = 1;
        int64_t stride_w = 1;
        int64_t padding_h = 0;
        int64_t padding_w = 0;
    };

    struct BasicResBlockWeights {
        Conv2dWeights conv1;
        BatchNorm2dWeights bn1;
        Conv2dWeights conv2;
        BatchNorm2dWeights bn2;
        bool use_shortcut = false;
        Conv2dWeights shortcut_conv;
        BatchNorm2dWeights shortcut_bn;
    };

    struct CAMLayerWeights {
        Conv1dWeights linear_local;
        Conv1dWeights linear1;
        Conv1dWeights linear2;
    };

    struct CAMDenseTDNNLayerWeights {
        BatchNorm1dWeights nonlinear1_bn;
        Conv1dWeights linear1;
        BatchNorm1dWeights nonlinear2_bn;
        CAMLayerWeights cam_layer;
    };

    struct CAMDenseTDNNBlockWeights {
        std::vector<CAMDenseTDNNLayerWeights> layers;
    };

    struct TransitLayerWeights {
        BatchNorm1dWeights nonlinear_bn;
        Conv1dWeights linear;
    };

    Conv2dWeights head_conv1;
    BatchNorm2dWeights head_bn1;
    std::vector<BasicResBlockWeights> head_layer1;
    std::vector<BasicResBlockWeights> head_layer2;
    Conv2dWeights head_conv2;
    BatchNorm2dWeights head_bn2;

    Conv1dWeights tdnn_linear;
    BatchNorm1dWeights tdnn_bn;
    std::vector<CAMDenseTDNNBlockWeights> blocks;
    std::vector<TransitLayerWeights> transits;
    BatchNorm1dWeights out_nonlinear_bn;
    Conv1dWeights dense_linear;
    BatchNorm1dWeights dense_bn;
};

struct S3TokenizerV2Weights {
    struct Conv1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
    };

    struct LayerNormWeights {
        std::vector<float> weight;
        std::vector<float> bias;
    };

    struct LinearWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct BlockWeights {
        LayerNormWeights attn_ln;
        LinearWeights attn_query;
        LinearWeights attn_key;
        LinearWeights attn_value;
        LinearWeights attn_qkv_packed;
        LinearWeights attn_out;
        std::vector<float> fsmn_weight;
        LayerNormWeights mlp_ln;
        LinearWeights mlp_fc1;
        LinearWeights mlp_fc2;
    };

    Conv1dWeights conv1;
    Conv1dWeights conv2;
    std::vector<BlockWeights> blocks;
    LinearWeights quantizer_project_down;
};

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

struct S3HiFTVocoderWeights {
    struct WeightNormConv1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        bool use_bias = false;
    };

    struct WeightNormConvTranspose1dWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        bool use_bias = false;
    };

    struct LinearWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct SnakeWeights {
        std::vector<float> alpha;
        bool alpha_logscale = false;
    };

    struct ResBlockWeights {
        std::vector<WeightNormConv1dWeights> convs1;
        std::vector<WeightNormConv1dWeights> convs2;
        std::vector<SnakeWeights> activations1;
        std::vector<SnakeWeights> activations2;
    };

    struct F0PredictorWeights {
        std::vector<WeightNormConv1dWeights> condnet;
        LinearWeights classifier;
    };

    F0PredictorWeights f0_predictor;
    WeightNormConv1dWeights conv_pre;
    std::vector<WeightNormConvTranspose1dWeights> ups;
    std::vector<S3SpeakerEncoderWeights::Conv1dWeights> source_downs;
    std::vector<ResBlockWeights> source_resblocks;
    std::vector<ResBlockWeights> resblocks;
    WeightNormConv1dWeights conv_post;
    LinearWeights source_linear;
    int64_t sampling_rate = 24000;
    int64_t harmonic_num = 8;
    float sine_amp = 0.1f;
    float noise_std = 0.003f;
    float voiced_threshold = 10.0f;
    float lrelu_slope = 0.1f;
    float audio_limit = 0.99f;
    int64_t istft_n_fft = 16;
    int64_t istft_hop = 4;
};

}  // namespace engine::models::chatterbox
