#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/io/json.h"

#include <ggml.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kokoro_ggml {
struct KokoroWeights {
    struct LinearWeights {
        engine::core::TensorValue weight;
        std::optional<engine::core::TensorValue> bias;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct HostAffineWeights {
        std::vector<float> weight;
        std::vector<float> bias;
        int64_t out_features = 0;
        int64_t in_features = 0;
        bool use_bias = false;
    };

    struct EmbeddingWeights {
        engine::core::TensorValue weight;
        int64_t num_embeddings = 0;
        int64_t embedding_dim = 0;
    };

    struct LayerNormWeights {
        engine::core::TensorValue weight;
        engine::core::TensorValue bias;
        int64_t channels = 0;
        float eps = 1.0e-5f;
    };

    struct WeightNormConv1dWeights {
        engine::core::TensorValue weight;
        std::optional<engine::core::TensorValue> bias;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool use_bias = false;
    };

    struct Conv1dWeights {
        engine::core::TensorValue weight;
        std::optional<engine::core::TensorValue> bias;
        int64_t out_channels = 0;
        int64_t in_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t dilation = 1;
        int64_t groups = 1;
        bool use_bias = false;
    };

    struct WeightNormConvTranspose1dWeights {
        engine::core::TensorValue weight;
        engine::core::TensorValue dense_weight;
        std::optional<engine::core::TensorValue> bias;
        std::shared_ptr<Conv1dWeights> phase_shuffle_conv;
        int64_t in_channels = 0;
        int64_t out_channels = 0;
        int64_t kernel = 0;
        int64_t stride = 1;
        int64_t padding = 0;
        int64_t output_padding = 0;
        int64_t groups = 1;
        bool use_bias = false;
    };

    struct LstmWeights {
        engine::core::TensorValue weight_ih_l0;
        engine::core::TensorValue weight_hh_l0;
        engine::core::TensorValue bias_ih_l0;
        engine::core::TensorValue bias_hh_l0;
        engine::core::TensorValue combined_bias_l0;
        engine::core::TensorValue weight_ih_l0_reverse;
        engine::core::TensorValue weight_hh_l0_reverse;
        engine::core::TensorValue bias_ih_l0_reverse;
        engine::core::TensorValue bias_hh_l0_reverse;
        engine::core::TensorValue combined_bias_l0_reverse;
        int64_t input_size = 0;
        int64_t hidden_size = 0;
    };

    struct AdaLayerNormWeights {
        LinearWeights fc;
        int64_t channels = 0;
        float eps = 1.0e-5f;
    };

    struct AdaIn1dWeights {
        LinearWeights fc;
        int64_t channels = 0;
        float eps = 1.0e-5f;
    };

    struct AlbertEmbeddingsWeights {
        EmbeddingWeights word_embeddings;
        EmbeddingWeights position_embeddings;
        EmbeddingWeights token_type_embeddings;
        LayerNormWeights layer_norm;
    };

    struct AlbertAttentionWeights {
        LinearWeights query;
        LinearWeights key;
        LinearWeights value;
        LinearWeights dense;
        LayerNormWeights layer_norm;
    };

    struct AlbertLayerWeights {
        AlbertAttentionWeights attention;
        LinearWeights ffn;
        LinearWeights ffn_output;
        LayerNormWeights full_layer_layer_norm;
    };

    struct AlbertWeights {
        AlbertEmbeddingsWeights embeddings;
        LinearWeights embedding_hidden_mapping_in;
        AlbertLayerWeights shared_layer;
        LinearWeights pooler;
        int64_t hidden_size = 768;
        int64_t embedding_size = 128;
        int64_t intermediate_size = 2048;
        int64_t max_position_embeddings = 512;
        int64_t num_hidden_layers = 12;
        int64_t num_attention_heads = 12;
        float layer_norm_eps = 1.0e-12f;
    };

    struct TextEncoderBlockWeights {
        WeightNormConv1dWeights conv;
        LayerNormWeights layer_norm;
    };

    struct TextEncoderWeights {
        EmbeddingWeights embedding;
        std::vector<TextEncoderBlockWeights> cnn;
        LstmWeights lstm;
    };

    struct DurationEncoderWeights {
        std::vector<LstmWeights> lstms;
        std::vector<AdaLayerNormWeights> ada_layer_norms;
    };

    struct AdainResBlock1dWeights {
        WeightNormConv1dWeights conv1;
        WeightNormConv1dWeights conv2;
        WeightNormConv1dWeights conv1x1;
        WeightNormConvTranspose1dWeights pool;
        AdaIn1dWeights norm1;
        AdaIn1dWeights norm2;
        bool learned_sc = false;
        bool use_pool = false;
        bool upsample = false;
    };

    struct GeneratorResBlockWeights {
        std::vector<WeightNormConv1dWeights> convs1;
        std::vector<WeightNormConv1dWeights> convs2;
        std::vector<AdaIn1dWeights> adain1;
        std::vector<AdaIn1dWeights> adain2;
        std::vector<engine::core::TensorValue> alpha1;
        std::vector<engine::core::TensorValue> alpha2;
    };

    struct GeneratorWeights {
        std::vector<WeightNormConvTranspose1dWeights> ups;
        std::vector<Conv1dWeights> noise_convs;
        std::vector<GeneratorResBlockWeights> noise_res;
        std::vector<GeneratorResBlockWeights> resblocks;
        WeightNormConv1dWeights conv_post;
        HostAffineWeights source_linear;
        int64_t harmonic_num = 8;
        int64_t sampling_rate = 24000;
        float sine_amp = 0.1f;
        float noise_std = 0.003f;
        float voiced_threshold = 10.0f;
        int64_t gen_istft_n_fft = 20;
        int64_t gen_istft_hop_size = 5;
    };

    struct ProsodyPredictorWeights {
        DurationEncoderWeights duration_encoder;
        LstmWeights lstm;
        LinearWeights duration_proj;
        LstmWeights shared;
        std::vector<AdainResBlock1dWeights> f0_blocks;
        std::vector<AdainResBlock1dWeights> n_blocks;
        Conv1dWeights f0_proj;
        Conv1dWeights n_proj;
    };

    struct DecoderWeights {
        AdainResBlock1dWeights encode;
        std::vector<AdainResBlock1dWeights> decode;
        WeightNormConv1dWeights f0_conv;
        WeightNormConv1dWeights n_conv;
        WeightNormConv1dWeights asr_res;
        GeneratorWeights generator;
    };

    AlbertWeights bert;
    LinearWeights bert_encoder;
    ProsodyPredictorWeights predictor;
    TextEncoderWeights text_encoder;
    DecoderWeights decoder;

    int64_t n_token = 178;
    int64_t hidden_dim = 512;
    int64_t style_dim = 128;
    int64_t n_layer = 3;
    int64_t max_dur = 50;
    int64_t n_mels = 80;
    int64_t context_length = 512;
    int64_t max_output_tokens = 512 * 50;
    float dropout = 0.2f;
    float lrelu_slope = 0.1f;
    float post_lrelu_slope = 0.01f;

    std::shared_ptr<engine::core::BackendWeightStore> store;
};
namespace g2p_en {
class EnglishG2P;
}
}

namespace engine::models::kokoro_tts {

struct KokoroVoicePack {
    std::string id;
    std::string language_code;
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<float> values;
};

struct KokoroAssets {
    std::filesystem::path model_root;
    engine::io::json::Value config;
    std::shared_ptr<const engine::assets::TensorSource> model_weights;
    int64_t context_length = 512;
    std::unordered_map<std::string, int32_t> vocab;
    std::unordered_map<std::string, KokoroVoicePack> voices;
    std::filesystem::path english_lexicon_dir;
    std::shared_ptr<const kokoro_ggml::g2p_en::EnglishG2P> english_g2p_us;
    std::shared_ptr<const kokoro_ggml::g2p_en::EnglishG2P> english_g2p_gb;
};

std::shared_ptr<const KokoroAssets> load_kokoro_assets(const std::filesystem::path & model_root);

std::shared_ptr<const kokoro_ggml::KokoroWeights> load_kokoro_backend_weights(
    const KokoroAssets & assets,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    engine::assets::TensorStorageType matmul_storage_type,
    engine::assets::TensorStorageType conv_storage_type,
    size_t weight_context_bytes);

}  // namespace engine::models::kokoro_tts
