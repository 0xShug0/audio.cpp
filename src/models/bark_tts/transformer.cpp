#include "engine/models/bark_tts/transformer.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/constant_tensor_cache.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <stdexcept>

namespace engine::models::bark_tts {
namespace {

namespace binding = engine::modules::binding;

struct LayerWeights {
    engine::modules::NormWeights norm1;
    engine::modules::AttentionWeights attention;
    engine::modules::NormWeights norm2;
    engine::modules::FeedForwardWeights mlp;
};

struct Weights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::vector<engine::core::TensorValue> embeddings;
    engine::core::TensorValue positions;
    std::vector<LayerWeights> layers;
    engine::modules::NormWeights final_norm;
    std::vector<engine::core::TensorValue> heads;
};

struct ContextDelete { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };

engine::modules::NormWeights load_norm(engine::core::BackendWeightStore & store,
                                      const engine::assets::TensorSource & source,
                                      const std::string & prefix, int64_t hidden, bool bias) {
    engine::modules::NormWeights out;
    out.weight = store.load_tensor(source, prefix + ".weight", engine::assets::TensorStorageType::F32, {hidden});
    if (bias) out.bias = store.load_tensor(source, prefix + ".bias", engine::assets::TensorStorageType::F32, {hidden});
    return out;
}

std::shared_ptr<const Weights> load_weights(const BarkAssets & assets,
                                            ggml_backend_t backend,
                                            engine::core::BackendType backend_type,
                                            const std::string & prefix,
                                            const BarkTransformerConfig & config,
                                            bool causal,
                                            engine::assets::TensorStorageType storage) {
    const auto & source = *assets.weights;
    auto out = std::make_shared<Weights>();
    out->store = std::make_shared<engine::core::BackendWeightStore>(backend, backend_type,
                                                                    "bark.transformer.weights", 768ULL * 1024ULL * 1024ULL);
    const int embeddings = causal ? 1 : 8;
    for (int i = 0; i < embeddings; ++i) {
        const std::string name = causal ? prefix + ".input_embeds_layer.weight" :
            prefix + ".input_embeds_layers." + std::to_string(i) + ".weight";
        out->embeddings.push_back(out->store->load_tensor(source, name, storage, {config.input_vocab, config.hidden}));
    }
    out->positions = out->store->load_tensor(source, prefix + ".position_embeds_layer.weight", storage,
                                             {config.block_size, config.hidden});
    for (int64_t i = 0; i < config.layers; ++i) {
        const std::string p = prefix + ".layers." + std::to_string(i);
        LayerWeights layer;
        layer.norm1 = load_norm(*out->store, source, p + ".layernorm_1", config.hidden, !causal || config.bias);
        layer.norm2 = load_norm(*out->store, source, p + ".layernorm_2", config.hidden, !causal || config.bias);
        layer.attention.qkv_weight = out->store->load_tensor(source, p + ".attn.att_proj.weight", storage,
                                                             {3 * config.hidden, config.hidden});
        if (config.bias) {
            layer.attention.qkv_bias = out->store->load_tensor(source, p + ".attn.att_proj.bias",
                engine::assets::TensorStorageType::F32, {3 * config.hidden});
        }
        layer.attention.out_weight = out->store->load_tensor(source, p + ".attn.out_proj.weight", storage,
                                                             {config.hidden, config.hidden});
        if (config.bias) {
            layer.attention.out_bias = out->store->load_tensor(source, p + ".attn.out_proj.bias",
                engine::assets::TensorStorageType::F32, {config.hidden});
        }
        layer.mlp.fc1_weight = out->store->load_tensor(source, p + ".mlp.in_proj.weight", storage,
                                                       {4 * config.hidden, config.hidden});
        layer.mlp.fc2_weight = out->store->load_tensor(source, p + ".mlp.out_proj.weight", storage,
                                                       {config.hidden, 4 * config.hidden});
        if (config.bias) {
            layer.mlp.fc1_bias = out->store->load_tensor(source, p + ".mlp.in_proj.bias",
                engine::assets::TensorStorageType::F32, {4 * config.hidden});
            layer.mlp.fc2_bias = out->store->load_tensor(source, p + ".mlp.out_proj.bias",
                engine::assets::TensorStorageType::F32, {config.hidden});
        }
        out->layers.push_back(std::move(layer));
    }
    out->final_norm = load_norm(*out->store, source, prefix + ".layernorm_final", config.hidden, !causal || config.bias);
    const int heads = causal ? 1 : 7;
    for (int i = 0; i < heads; ++i) {
        const std::string name = causal ? prefix + ".lm_head.weight" : prefix + ".lm_heads." + std::to_string(i) + ".weight";
        out->heads.push_back(out->store->load_tensor(source, name, storage, {config.output_vocab, config.hidden}));
    }
    out->store->upload();
    return out;
}

engine::core::TensorValue block(engine::core::ModuleBuildContext & ctx,
                                const engine::core::TensorValue & input,
                                const LayerWeights & weights,
                                const BarkTransformerConfig & config,
                                bool causal) {
    auto n1 = engine::modules::LayerNormModule({config.hidden, 1.0e-5F, true, !causal || config.bias})
                  .build(ctx, input, weights.norm1);
    auto attention = engine::modules::SelfAttentionModule({
        config.hidden, config.heads, config.bias, GGML_PREC_F32, GGML_PREC_F32,
        engine::modules::AttentionPrefixCacheLayout::SequenceHeads, true, causal, false, 0, 0, 0, false,
    }).build(ctx, n1, weights.attention);
    auto hidden = engine::modules::AddModule{}.build(ctx, input, attention);
    auto n2 = engine::modules::LayerNormModule({config.hidden, 1.0e-5F, true, !causal || config.bias})
                  .build(ctx, hidden, weights.norm2);
    auto ff = engine::modules::FeedForwardGeluModule({config.hidden, 4 * config.hidden,
        config.bias, engine::modules::GeluApproximation::ExactErf, GGML_PREC_F32}).build(ctx, n2, weights.mlp);
    return engine::modules::AddModule{}.build(ctx, hidden, ff);
}

}  // namespace

struct BarkTransformer::Impl {
    std::shared_ptr<const BarkAssets> assets;
    engine::core::ExecutionContext & execution;
    std::string prefix;
    BarkTransformerConfig config;
    bool causal;
    std::shared_ptr<const Weights> weights;

    std::vector<float> run(const std::vector<std::vector<int32_t>> & channels, int head, bool last_only) const {
        if (channels.empty() || channels.front().empty()) throw std::runtime_error("Bark transformer input is empty");
        const int64_t sequence = static_cast<int64_t>(channels.front().size());
        if (sequence > config.block_size) throw std::runtime_error("Bark transformer input exceeds block size");
        for (const auto & row : channels) if (static_cast<int64_t>(row.size()) != sequence)
            throw std::runtime_error("Bark embedding channel lengths differ");
        if (head < 0 || static_cast<size_t>(head) >= weights->heads.size()) throw std::runtime_error("Bark LM head index is invalid");
        std::unique_ptr<ggml_context, ContextDelete> context(ggml_init({128ULL * 1024ULL * 1024ULL, nullptr, true}));
        if (!context) throw std::runtime_error("failed to create Bark graph context");
        engine::core::ModuleBuildContext build{context.get(), "bark.transformer", execution.backend_type()};
        engine::core::TensorValue hidden;
        std::vector<ggml_tensor *> inputs;
        std::vector<ggml_tensor *> masks;
        std::vector<std::vector<int32_t>> safe_channels = channels;
        std::vector<std::vector<float>> mask_values(channels.size(), std::vector<float>(static_cast<size_t>(sequence), 1.0F));
        for (size_t channel = 0; channel < channels.size(); ++channel) {
            for (int64_t step = 0; step < sequence; ++step) {
                if (safe_channels[channel][static_cast<size_t>(step)] < 0) {
                    safe_channels[channel][static_cast<size_t>(step)] = 0;
                    mask_values[channel][static_cast<size_t>(step)] = 0.0F;
                }
            }
            auto * raw = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, sequence);
            inputs.push_back(raw);
            auto ids = engine::core::wrap_tensor(raw, engine::core::TensorShape::from_dims({sequence}), GGML_TYPE_I32);
            auto embedded = engine::modules::EmbeddingModule({config.input_vocab, config.hidden})
                                .build(build, ids, weights->embeddings.at(causal ? 0 : channel));
            embedded = engine::core::reshape_tensor(build, embedded,
                engine::core::TensorShape::from_dims({1, sequence, config.hidden}));
            auto * mask_raw = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, sequence);
            masks.push_back(mask_raw);
            auto mask = engine::core::wrap_tensor(mask_raw,
                engine::core::TensorShape::from_dims({1, sequence, 1}), GGML_TYPE_F32);
            embedded = engine::core::wrap_tensor(ggml_mul(context.get(), embedded.tensor, mask.tensor),
                                                   embedded.shape, GGML_TYPE_F32);
            hidden = hidden.valid() ? engine::modules::AddModule{}.build(build, hidden, embedded) : embedded;
        }
        std::vector<int32_t> positions(static_cast<size_t>(sequence));
        for (int32_t i = 0; i < sequence; ++i) positions[static_cast<size_t>(i)] = i;
        auto * pos_raw = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, sequence);
        auto pos_ids = engine::core::wrap_tensor(pos_raw, engine::core::TensorShape::from_dims({sequence}), GGML_TYPE_I32);
        auto pos = engine::modules::EmbeddingModule({config.block_size, config.hidden}).build(build, pos_ids, weights->positions);
        pos = engine::core::reshape_tensor(build, pos, engine::core::TensorShape::from_dims({1, sequence, config.hidden}));
        hidden = engine::modules::AddModule{}.build(build, hidden, pos);
        for (const auto & layer : weights->layers) hidden = block(build, hidden, layer, config, causal);
        hidden = engine::modules::LayerNormModule({config.hidden, 1.0e-5F, true, !causal || config.bias})
                     .build(build, hidden, weights->final_norm);
        if (last_only) hidden = engine::modules::SliceModule({1, sequence - 1, 1}).build(build, hidden);
        auto logits = engine::modules::LinearModule({config.hidden, config.output_vocab, false, GGML_PREC_F32})
                          .build(build, hidden, {weights->heads[static_cast<size_t>(head)], std::nullopt});
        ggml_set_output(logits.tensor);
        auto * graph = ggml_new_graph_custom(context.get(), 65536, false);
        ggml_build_forward_expand(graph, logits.tensor);
        auto * buffer = ggml_backend_alloc_ctx_tensors(context.get(), execution.backend());
        if (!buffer) throw std::runtime_error("failed to allocate Bark graph tensors");
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_backend_tensor_set(inputs[i], safe_channels[i].data(), 0, safe_channels[i].size() * sizeof(int32_t));
            ggml_backend_tensor_set(masks[i], mask_values[i].data(), 0, mask_values[i].size() * sizeof(float));
        }
        ggml_backend_tensor_set(pos_raw, positions.data(), 0, positions.size() * sizeof(int32_t));
        engine::core::set_backend_threads(execution.backend(), execution.config().threads);
        const auto status = engine::core::compute_backend_graph(execution.backend(), graph);
        ggml_backend_synchronize(execution.backend());
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Bark transformer graph compute failed");
        std::vector<float> out(static_cast<size_t>((last_only ? 1 : sequence) * config.output_vocab));
        ggml_backend_tensor_get(logits.tensor, out.data(), 0, out.size() * sizeof(float));
        engine::core::release_backend_graph_resources(execution.backend(), graph);
        ggml_backend_buffer_free(buffer);
        return out;
    }
};

BarkTransformer::BarkTransformer(std::shared_ptr<const BarkAssets> assets,
                                 engine::core::ExecutionContext & execution,
                                 std::string prefix,
                                 BarkTransformerConfig config,
                                 bool causal,
                                 engine::assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(Impl{assets, execution, std::move(prefix), config, causal, nullptr})) {
    impl_->weights = load_weights(*impl_->assets, execution.backend(), execution.backend_type(), impl_->prefix, config, causal, storage);
}

BarkTransformer::~BarkTransformer() = default;

std::vector<float> BarkTransformer::causal_logits(const std::vector<std::vector<int32_t>> & channels) const {
    if (!impl_->causal) throw std::runtime_error("fine Bark transformer cannot run causal logits");
    return impl_->run(channels, 0, true);
}

std::vector<float> BarkTransformer::fine_logits(const std::vector<std::vector<int32_t>> & codes, int codebook_idx) const {
    if (impl_->causal || codebook_idx < 1 || codebook_idx > 7) throw std::runtime_error("invalid Bark fine codebook");
    return impl_->run(std::vector<std::vector<int32_t>>(codes.begin(), codes.begin() + codebook_idx + 1), codebook_idx - 1, false);
}

}  // namespace engine::models::bark_tts
