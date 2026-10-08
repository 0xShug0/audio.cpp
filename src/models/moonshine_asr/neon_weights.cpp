#include "engine/models/moonshine_asr/neon_weights.h"
#include "engine/models/moonshine_asr/assets.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace engine::models::moonshine_asr {
namespace {
int dimension(int64_t value) {
    if (value <= 0 || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("Moonshine native dimension invalid");
    }
    return static_cast<int>(value);
}

} // namespace

MoonshineNeonDecoderWeights load_moonshine_neon_decoder_weights(const MoonshineAssets &asset,
                                                                assets::TensorStorageType storage) {
    const auto &source = *asset.source;
    const auto &config = asset.config.decoder;
    auto linear = [&](const std::string &prefix, int64_t input, int64_t output, bool bias) {
        MoonshineNeonLinear result{dimension(input), dimension(output), {}, {}};
        const auto name = prefix + ".weight";
        auto data = source.require_tensor(name, storage, {output, input});
        if (data.type == GGML_TYPE_Q8_0) {
            if (input % 32 || ggml_blck_size(data.type) != 32 || ggml_type_size(data.type) != 34) {
                throw std::runtime_error("Moonshine native Q8 block layout invalid");
            }
            neon::BlockInt8Weights quantized;
            const size_t blocks = static_cast<size_t>(input) * output / 32;
            if (data.bytes.size() != blocks * 34) {
                throw std::runtime_error("Moonshine native Q8 weight size invalid");
            }
            quantized.values.resize(blocks * 32);
            quantized.scales.resize(blocks);
            for (size_t block = 0; block < blocks; ++block) {
                ggml_fp16_t scale;
                std::memcpy(&scale, data.bytes.data() + block * 34, 2);
                quantized.scales[block] = ggml_fp16_to_fp32(scale);
                std::memcpy(quantized.values.data() + block * 32, data.bytes.data() + block * 34 + 2, 32);
            }
            result.weight = std::move(quantized);
        } else {
            result.weight = assets::tensor_data_to_f32(name, data);
        }
        if (bias) {
            result.bias = source.require_f32(prefix + ".bias", {output});
        }
        return result;
    };
    auto norm = [&](const std::string &prefix) { return source.require_f32(prefix + ".weight", {config.hidden_size}); };
    MoonshineNeonDecoderWeights result;
    result.token_embedding = linear("model.decoder.embed_tokens", config.hidden_size, config.vocab_size, false);
    const int64_t query_features = config.heads * config.head_dim;
    const int64_t kv_features = config.kv_heads * config.head_dim;
    for (int layer = 0; layer < config.layers; ++layer) {
        const auto prefix = "model.decoder.layers." + std::to_string(layer);
        MoonshineNeonDecoderLayer bound;
        bound.input_norm = norm(prefix + ".input_layernorm");
        bound.cross_norm = norm(prefix + ".post_attention_layernorm");
        bound.mlp_norm = norm(prefix + ".final_layernorm");
        bound.self_query =
            linear(prefix + ".self_attn.q_proj", config.hidden_size, query_features, config.attention_bias);
        bound.self_key = linear(prefix + ".self_attn.k_proj", config.hidden_size, kv_features, config.attention_bias);
        bound.self_value = linear(prefix + ".self_attn.v_proj", config.hidden_size, kv_features, config.attention_bias);
        bound.self_output = linear(prefix + ".self_attn.o_proj", query_features, config.hidden_size, false);
        bound.cross_query =
            linear(prefix + ".encoder_attn.q_proj", config.hidden_size, query_features, config.attention_bias);
        bound.cross_key =
            linear(prefix + ".encoder_attn.k_proj", config.hidden_size, kv_features, config.attention_bias);
        bound.cross_value =
            linear(prefix + ".encoder_attn.v_proj", config.hidden_size, kv_features, config.attention_bias);
        bound.cross_output = linear(prefix + ".encoder_attn.o_proj", query_features, config.hidden_size, false);
        bound.mlp_input = linear(prefix + ".mlp.fc1", config.hidden_size, config.intermediate_size * 2, true);
        bound.mlp_output = linear(prefix + ".mlp.fc2", config.intermediate_size, config.hidden_size, true);
        result.layers.push_back(std::move(bound));
    }
    result.norm = norm("model.decoder.norm");
    result.head = linear(source.has_tensor("proj_out.weight") ? "proj_out" : "model.proj_out", config.hidden_size,
                         config.vocab_size, false);
    return result;
}

} // namespace engine::models::moonshine_asr
