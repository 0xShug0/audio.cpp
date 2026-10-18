#include "pipeline_internal.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/text_encoders/t5_base_encoder.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::liveavatar {
namespace {

constexpr size_t kTextEncoderWeightContextBytes = 1100ull * 1024ull * 1024ull;
constexpr size_t kTextEncoderGraphContextBytes = 256ull * 1024ull * 1024ull;
constexpr float kTextMaskNegInf = -1.0e9F;
constexpr int32_t kT5PadTokenId = 0;

engine::modules::T5BaseEncoderLayerWeights load_t5_layer(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    const engine::modules::T5BaseEncoderConfig & config,
    engine::assets::TensorStorageType storage_type) {
    engine::modules::T5BaseEncoderLayerWeights weights;
    weights.self_attention_layer_norm = store.load_f32_tensor(source, prefix + ".attn_norm.weight", {config.hidden_size});
    weights.ffn_layer_norm = store.load_f32_tensor(source, prefix + ".ffn_norm.weight", {config.hidden_size});
    weights.q_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".attn_q",
        storage_type,
        config.attention_heads * config.head_dim,
        config.hidden_size,
        false);
    weights.k_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".attn_k",
        storage_type,
        config.attention_heads * config.head_dim,
        config.hidden_size,
        false);
    weights.v_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".attn_v",
        storage_type,
        config.attention_heads * config.head_dim,
        config.hidden_size,
        false);
    weights.o_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".attn_o",
        storage_type,
        config.hidden_size,
        config.attention_heads * config.head_dim,
        false);
    weights.wi_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".ffn_up",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    weights.gate_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".ffn_gate",
        storage_type,
        config.intermediate_size,
        config.hidden_size,
        false);
    weights.wo_proj = engine::modules::binding::linear_from_source(
        store,
        source,
        prefix + ".ffn_down",
        storage_type,
        config.hidden_size,
        config.intermediate_size,
        false);
    weights.relative_attention_bias =
        store.load_f32_tensor(source, prefix + ".attn_rel_b.weight", {config.relative_attention_num_buckets, config.attention_heads});
    return weights;
}

struct LiveAvatarTextEncoderWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::modules::T5BaseEncoderWeights encoder;
};

LiveAvatarTextEncoderWeights load_text_encoder_weights(
    const engine::assets::TensorSource & source,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::modules::T5BaseEncoderConfig & config) {
    LiveAvatarTextEncoderWeights weights;
    weights.store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "liveavatar.text_encoder.weights",
        kTextEncoderWeightContextBytes);
    weights.encoder.embed_tokens =
        weights.store->load_tensor(source, "token_embd.weight", engine::assets::TensorStorageType::F16, {config.vocab_size, config.hidden_size});
    weights.encoder.layers.reserve(static_cast<size_t>(config.layers));
    for (int64_t layer = 0; layer < config.layers; ++layer) {
        weights.encoder.layers.push_back(load_t5_layer(
            *weights.store,
            source,
            "enc.blk." + std::to_string(layer),
            config,
            engine::assets::TensorStorageType::Native));
    }
    weights.encoder.final_layer_norm = weights.store->load_f32_tensor(source, "enc.output_norm.weight", {config.hidden_size});
    weights.store->upload();
    return weights;
}

std::vector<float> make_additive_attention_mask(int64_t tokens, int64_t valid_tokens, int64_t heads) {
    if (tokens <= 0 || valid_tokens <= 0 || valid_tokens > tokens || heads <= 0) {
        throw std::runtime_error("LiveAvatar text attention mask shape is invalid");
    }
    std::vector<float> mask(static_cast<size_t>(heads * tokens * tokens), 0.0F);
    for (int64_t head = 0; head < heads; ++head) {
        for (int64_t q = 0; q < tokens; ++q) {
            for (int64_t k = valid_tokens; k < tokens; ++k) {
                mask[static_cast<size_t>((head * tokens + q) * tokens + k)] = kTextMaskNegInf;
            }
        }
    }
    return mask;
}

std::vector<float> make_batched_additive_attention_mask(
    int64_t batch,
    int64_t tokens,
    const std::vector<int64_t> & valid_tokens,
    int64_t heads) {
    if (batch <= 0 || static_cast<int64_t>(valid_tokens.size()) != batch) {
        throw std::runtime_error("LiveAvatar batched text attention mask batch mismatch");
    }
    std::vector<float> out(static_cast<size_t>(batch * heads * tokens * tokens), 0.0F);
    const int64_t single = heads * tokens * tokens;
    for (int64_t b = 0; b < batch; ++b) {
        auto mask = make_additive_attention_mask(tokens, valid_tokens[static_cast<size_t>(b)], heads);
        std::copy(mask.begin(), mask.end(), out.begin() + static_cast<std::ptrdiff_t>(b * single));
    }
    return out;
}

class LiveAvatarTextEncoderGraph {
public:
    LiveAvatarTextEncoderGraph(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::modules::T5BaseEncoderConfig config,
        int64_t max_batch = 2)
        : execution_(execution),
          assets_(std::move(assets)),
          config_(std::move(config)),
          max_batch_(max_batch),
          weights_(load_text_encoder_weights(*assets_->text_encoder_weights, execution_.backend(), execution_.backend_type(), config_)) {
        if (execution_.backend() == nullptr) {
            throw std::runtime_error("LiveAvatar text encoder backend initialization failed");
        }
        if (max_batch_ <= 0) {
            throw std::runtime_error("LiveAvatar text encoder max batch must be positive");
        }
        assets_->text_encoder_weights->release_storage();
        build();
    }

    ~LiveAvatarTextEncoderGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    std::vector<float> encode(const std::vector<int32_t> & input_ids, int64_t valid_tokens) const {
        return encode_batch({input_ids}, {valid_tokens}).front();
    }

    std::vector<std::vector<float>> encode_batch(
        const std::vector<std::vector<int32_t>> & input_ids,
        const std::vector<int64_t> & valid_tokens) const {
        const int64_t batch = static_cast<int64_t>(input_ids.size());
        if (batch <= 0 || batch > max_batch_ || static_cast<int64_t>(valid_tokens.size()) != batch) {
            throw std::runtime_error("LiveAvatar text encoder batch size mismatch");
        }
        const int64_t tokens = assets_->config.text_len;
        std::vector<int32_t> padded(static_cast<size_t>(max_batch_ * tokens), kT5PadTokenId);
        std::vector<int64_t> padded_valid(static_cast<size_t>(max_batch_), 1);
        for (int64_t b = 0; b < batch; ++b) {
            if (static_cast<int64_t>(input_ids[static_cast<size_t>(b)].size()) != tokens) {
                throw std::runtime_error("LiveAvatar text encoder input length mismatch");
            }
            std::copy(
                input_ids[static_cast<size_t>(b)].begin(),
                input_ids[static_cast<size_t>(b)].end(),
                padded.begin() + static_cast<std::ptrdiff_t>(b * tokens));
            padded_valid[static_cast<size_t>(b)] = valid_tokens[static_cast<size_t>(b)];
        }
        engine::core::write_tensor_i32(input_ids_, padded);
        engine::core::write_tensor_f32(attention_mask_, make_batched_additive_attention_mask(
            max_batch_,
            tokens,
            padded_valid,
            config_.attention_heads));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.text_encoder");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar text encoder graph compute failed");
        }
        const auto full = engine::core::read_tensor_f32(output_);
        const int64_t row_values = tokens * config_.hidden_size;
        std::vector<std::vector<float>> out;
        out.reserve(static_cast<size_t>(batch));
        for (int64_t b = 0; b < batch; ++b) {
            auto row = std::vector<float>(
                full.begin() + static_cast<std::ptrdiff_t>(b * row_values),
                full.begin() + static_cast<std::ptrdiff_t>((b + 1) * row_values));
            const int64_t valid = valid_tokens[static_cast<size_t>(b)];
            std::fill(
                row.begin() + static_cast<std::ptrdiff_t>(valid * config_.hidden_size),
                row.end(),
                0.0F);
            out.push_back(std::move(row));
        }
        return out;
    }

private:
    void build() {
        ggml_init_params params{kTextEncoderGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar text encoder ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.text_encoder", execution_.backend_type()};
        const int64_t tokens = assets_->config.text_len;
        input_ids_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({max_batch_, tokens}));
        relative_buckets_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_I32,
            engine::core::TensorShape::from_dims({tokens, tokens}));
        attention_mask_ = engine::core::make_tensor(
            build_ctx,
            GGML_TYPE_F32,
            engine::core::TensorShape::from_dims({max_batch_, config_.attention_heads, tokens, tokens}));
        ggml_set_input(input_ids_.tensor);
        ggml_set_input(relative_buckets_.tensor);
        ggml_set_input(attention_mask_.tensor);
        const auto output = engine::modules::T5BaseEncoderModule(config_).build(
            build_ctx,
            input_ids_,
            relative_buckets_,
            attention_mask_,
            weights_.encoder);
        output_ = output.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 262144, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar text encoder backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
        engine::core::write_tensor_i32(
            relative_buckets_,
            engine::modules::t5_base_relative_position_buckets(
                tokens,
                tokens,
                config_.relative_attention_num_buckets,
                config_.relative_attention_max_distance));
    }

    engine::core::ExecutionContext & execution_;
    std::shared_ptr<const LiveAvatarAssets> assets_;
    engine::modules::T5BaseEncoderConfig config_;
    int64_t max_batch_ = 1;
    LiveAvatarTextEncoderWeights weights_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue input_ids_;
    engine::core::TensorValue relative_buckets_;
    engine::core::TensorValue attention_mask_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

}  // namespace

struct LiveAvatarTextEncoderRuntime::Data {
    Data(
        engine::core::ExecutionContext & execution,
        std::shared_ptr<const LiveAvatarAssets> assets,
        engine::modules::T5BaseEncoderConfig config)
        : graph(execution, std::move(assets), std::move(config)) {}

    LiveAvatarTextEncoderGraph graph;
};

LiveAvatarTextEncoderRuntime::LiveAvatarTextEncoderRuntime(
    engine::core::ExecutionContext & execution,
    std::shared_ptr<const LiveAvatarAssets> assets,
    engine::modules::T5BaseEncoderConfig config)
    : data_(std::make_unique<Data>(execution, std::move(assets), std::move(config))) {}

LiveAvatarTextEncoderRuntime::~LiveAvatarTextEncoderRuntime() = default;

std::vector<std::vector<float>> LiveAvatarTextEncoderRuntime::encode_batch(
    const std::vector<std::vector<int32_t>> & input_ids,
    const std::vector<int64_t> & token_counts) {
    return data_->graph.encode_batch(input_ids, token_counts);
}

}  // namespace engine::community_models::liveavatar
