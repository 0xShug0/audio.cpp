#include "engine/models/mulacover/style_encoder.h"

#include "engine/framework/assets/resource_bundle.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/transformers/causal_decoder_runtime.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/tokenizers/qwen_bpe_bundle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace engine::models::mulacover {

class StyleEncoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source,
         const std::filesystem::path & tokenizer_directory,
         const core::BackendConfig & backend,
         assets::TensorStorageType storage_type)
        : execution_(backend),
          store_(execution_.backend(), execution_.backend_type(),
                 "mulacover.style_encoder.weights", 2 * 1024 * 1024) {
        if (!source) {
            throw std::invalid_argument("MuLaCover style encoder requires a tensor source");
        }
        assets::ResourceBundle bundle(tokenizer_directory);
        bundle.add_model_files({{"tokenizer_config", "tokenizer_config.json"},
                                {"tokenizer_json", "tokenizer.json"},
                                {"vocab", "vocab.json", false},
                                {"merges", "merges.txt", false}});
        tokenizer_ = tokenizers::load_qwen_bpe_tokenizer(bundle);
        eos_id_ = static_cast<int32_t>(tokenizers::require_added_token_id(bundle, "<|endoftext|>"));

        // The fixed Qwen/Qwen3-Embedding-0.6B config shipped with MuLaCover.
        modules::CausalDecoderRuntimeConfig config;
        config.trace_name = "mulacover.style_encoder";
        config.output_mode = modules::CausalDecoderOutputMode::Hidden;
        config.prefill_graph_arena_bytes = 64 * 1024 * 1024;
        config.decode_graph_arena_bytes = 64 * 1024 * 1024;
        config.decoder.logits_mode = modules::CausalDecoderLogitsMode::LastStep;
        auto & stack = config.decoder.stack;
        stack.hidden_size = 1024;
        stack.num_attention_heads = 16;
        stack.num_key_value_heads = 8;
        stack.head_dim = 128;
        stack.intermediate_size = 3072;
        stack.layers = 28;
        stack.rms_norm_eps = 1e-6f;
        stack.rope_theta = 1000000.0f;
        stack.use_qk_norm = true;

        modules::CausalDecoderRuntimeWeights weights;
        weights.token_embedding = store_.load_tensor(
            *source, "embed_tokens.weight", storage_type, {151669, 1024});
        weights.stack.layers.reserve(static_cast<size_t>(stack.layers));
        for (int64_t i = 0; i < stack.layers; ++i) {
            const std::string prefix = "layers." + std::to_string(i);
            modules::DecoderLayerWeights layer;
            layer.input_norm = modules::binding::norm_weight_from_source(
                store_, *source, prefix + ".input_layernorm", 1024);
            layer.self_attention.q_weight = modules::binding::linear_from_source(
                store_, *source, prefix + ".self_attn.q_proj", storage_type, 2048, 1024, false).weight;
            layer.self_attention.k_weight = modules::binding::linear_from_source(
                store_, *source, prefix + ".self_attn.k_proj", storage_type, 1024, 1024, false).weight;
            layer.self_attention.v_weight = modules::binding::linear_from_source(
                store_, *source, prefix + ".self_attn.v_proj", storage_type, 1024, 1024, false).weight;
            layer.self_attention.out_weight = modules::binding::linear_from_source(
                store_, *source, prefix + ".self_attn.o_proj", storage_type, 1024, 2048, false).weight;
            layer.q_norm = modules::binding::norm_weight_from_source(
                store_, *source, prefix + ".self_attn.q_norm", 128);
            layer.k_norm = modules::binding::norm_weight_from_source(
                store_, *source, prefix + ".self_attn.k_norm", 128);
            layer.post_norm = modules::binding::norm_weight_from_source(
                store_, *source, prefix + ".post_attention_layernorm", 1024);
            layer.mlp.gate_proj = modules::binding::linear_from_source(
                store_, *source, prefix + ".mlp.gate_proj", storage_type, 3072, 1024, false);
            layer.mlp.up_proj = modules::binding::linear_from_source(
                store_, *source, prefix + ".mlp.up_proj", storage_type, 3072, 1024, false);
            layer.mlp.down_proj = modules::binding::linear_from_source(
                store_, *source, prefix + ".mlp.down_proj", storage_type, 1024, 3072, false);
            weights.stack.layers.push_back(std::move(layer));
        }
        weights.final_norm = modules::binding::norm_weight_from_source(store_, *source, "norm", 1024);
        store_.upload();
        runtime_ = std::make_unique<modules::CausalDecoderRuntime>(execution_, config, std::move(weights));
    }

    std::vector<float> encode(const std::string & tags) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<float> embedding(1024, 0.0f);
        if (!tags.empty()) {
            auto ids = tokenizer_->encode(tags);
            // tokenizer.json TemplateProcessing appends EOS even when the text
            // already ends in that token. The BPE encoder does not apply it.
            ids.push_back(eos_id_);
            if (ids.size() > 8192) {
                throw std::invalid_argument("Style exceeds the Qwen limit of 8192 tokens");
            }
            embedding = runtime_->prefill_tokens(ids).hidden;
            if (embedding.size() != 1024) {
                throw std::runtime_error("The style encoder dimension does not match MuLaCover");
            }
            float norm_sq = 0.0f;
            for (float value : embedding) {
                norm_sq += value * value;
            }
            // torch.nn.functional.normalize's default epsilon is 1e-12.
            const float norm = std::max(std::sqrt(norm_sq), 1e-12f);
            for (float & value : embedding) {
                value /= norm;
            }
        }
        debug::timing_log_scalar("mulacover.style_encoder.wall_ms",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        return embedding;
    }

private:
    core::ExecutionContext execution_;
    core::BackendWeightStore store_;
    std::shared_ptr<tokenizers::LlamaBpeTokenizer> tokenizer_;
    int32_t eos_id_ = 0;
    std::unique_ptr<modules::CausalDecoderRuntime> runtime_;
};

StyleEncoder::StyleEncoder(std::shared_ptr<const assets::TensorSource> source,
                           const std::filesystem::path & tokenizer_directory,
                           const core::BackendConfig & backend,
                           assets::TensorStorageType storage_type)
    : impl_(std::make_unique<Impl>(std::move(source), tokenizer_directory, backend, storage_type)) {}

StyleEncoder::~StyleEncoder() = default;

std::vector<float> StyleEncoder::encode(const std::string & tags) {
    return impl_->encode(tags);
}

}  // namespace engine::models::mulacover
