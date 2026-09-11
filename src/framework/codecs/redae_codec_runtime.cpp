#include "engine/framework/codecs/redae_codec_runtime.h"

#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/transformers/qwen_causal_decode_runtime.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::codecs {
namespace {

namespace binding = engine::modules::binding;
namespace core = engine::core;
namespace modules = engine::modules;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr_t alloc) const noexcept {
        if (alloc != nullptr) {
            ggml_gallocr_free(alloc);
        }
    }
};

struct GraphMemory {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<ggml_context, GgmlContextDeleter> input_ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_gallocr_t>, GgmlGallocrDeleter> gallocr;
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    ~GraphMemory() {
        reset(nullptr);
    }

    void reset(ggml_backend_t backend) {
        if (graph != nullptr && backend != nullptr) {
            core::release_backend_graph_resources(backend, graph);
        }
        graph = nullptr;
        gallocr.reset();
        if (input_buffer != nullptr) {
            ggml_backend_buffer_free(input_buffer);
            input_buffer = nullptr;
        }
        input_ctx.reset();
        ctx.reset();
    }
};

modules::QwenCausalDecodeRuntimeConfig qwen_runtime_config(
    const std::string & trace,
    int64_t hidden,
    int64_t intermediate,
    int64_t layers,
    int64_t heads,
    int64_t kv_heads,
    int64_t head_dim,
    modules::QwenCausalDecoderLogitsMode hidden_mode,
    size_t prefill_arena,
    size_t decode_arena,
    core::BackendType backend_type,
    bool bf16_autocast = false,
    int64_t sliding_window = 0) {
    modules::QwenCausalDecodeRuntimeConfig out;
    out.trace_name = trace;
    out.prefill_graph_arena_bytes = prefill_arena;
    out.decode_graph_arena_bytes = decode_arena;
    out.decoder.stack.hidden_size = hidden;
    out.decoder.stack.num_attention_heads = heads;
    out.decoder.stack.num_key_value_heads = kv_heads;
    out.decoder.stack.head_dim = head_dim;
    out.decoder.stack.intermediate_size = intermediate;
    out.decoder.stack.layers = layers;
    out.decoder.stack.rms_norm_eps = 1.0e-6F;
    out.decoder.stack.rope_theta = 1000000.0F;
    out.decoder.stack.rope_type = GGML_ROPE_TYPE_NEOX;
    out.decoder.stack.use_qk_norm = true;
    out.decoder.stack.attention_precision = GGML_PREC_F32;
    out.decoder.stack.projection_precision = GGML_PREC_DEFAULT;
    out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.attention.static_mode = modules::QwenDecoderAttentionMode::FlashGroupedViewKV;
    out.decoder.stack.runtime.static_cache.update_mode = modules::QwenDecoderStaticCacheUpdateMode::DirectSetRows;
    out.decoder.stack.runtime.static_cache.set_rows_mode = modules::QwenDecoderStaticCacheSetRowsMode::BackendViewOptimized;
    out.sliding_window = sliding_window;
    if (bf16_autocast && backend_type != core::BackendType::Cpu && backend_type != core::BackendType::Vulkan &&
        backend_type != core::BackendType::Metal) {
        out.decoder.stack.activation_cast.enabled = true;
        out.decoder.stack.activation_cast.type = GGML_TYPE_BF16;
        out.decoder.stack.activation_cast.after_input_norm = true;
        out.decoder.stack.activation_cast.after_qkv_projection = true;
        out.decoder.stack.activation_cast.after_qk_norm = true;
        out.decoder.stack.activation_cast.after_rope = true;
        out.decoder.stack.activation_cast.after_static_cache_update = true;
        out.decoder.stack.activation_cast.after_attention = true;
        out.decoder.stack.activation_cast.after_attention_output = true;
        out.decoder.stack.activation_cast.after_residual = true;
        out.decoder.stack.activation_cast.after_ffn_norm = true;
        out.decoder.stack.activation_cast.after_mlp_projection = true;
        out.decoder.stack.activation_cast.after_mlp_silu = true;
        out.decoder.stack.activation_cast.after_mlp_mul = true;
        out.decoder.stack.activation_cast.after_output = true;
        out.decoder.static_cache_type = GGML_TYPE_BF16;
    }
    out.decoder.logits_mode = hidden_mode;
    out.output_mode = modules::QwenCausalDecodeOutputMode::Hidden;
    out.return_hidden = true;
    if (bf16_autocast && backend_type != core::BackendType::Cpu && backend_type != core::BackendType::Vulkan &&
        backend_type != core::BackendType::Metal) {
        out.readback_round_type = GGML_TYPE_BF16;
    }
    return out;
}

modules::QwenDecoderLayerWeights load_qwen_layer(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const modules::QwenCausalDecoderConfig & config,
    assets::TensorStorageType storage_type) {
    modules::QwenDecoderLayerWeights out;
    out.input_norm = binding::norm_weight_from_source(store, source, prefix + ".input_layernorm", config.stack.hidden_size);
    out.self_attention.q_weight = store.load_tensor(
        source,
        prefix + ".self_attn.q_proj.weight",
        storage_type,
        {config.stack.num_attention_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.k_weight = store.load_tensor(
        source,
        prefix + ".self_attn.k_proj.weight",
        storage_type,
        {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.v_weight = store.load_tensor(
        source,
        prefix + ".self_attn.v_proj.weight",
        storage_type,
        {config.stack.num_key_value_heads * config.stack.head_dim, config.stack.hidden_size});
    out.self_attention.out_weight = store.load_tensor(
        source,
        prefix + ".self_attn.o_proj.weight",
        storage_type,
        {config.stack.hidden_size, config.stack.num_attention_heads * config.stack.head_dim});
    out.q_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.q_norm", config.stack.head_dim);
    out.k_norm = binding::norm_weight_from_source(store, source, prefix + ".self_attn.k_norm", config.stack.head_dim);
    out.post_norm = binding::norm_weight_from_source(store, source, prefix + ".post_attention_layernorm", config.stack.hidden_size);
    out.mlp.gate_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.gate_proj",
        storage_type,
        config.stack.intermediate_size,
        config.stack.hidden_size,
        false);
    out.mlp.up_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.up_proj",
        storage_type,
        config.stack.intermediate_size,
        config.stack.hidden_size,
        false);
    out.mlp.down_proj = binding::linear_from_source(
        store,
        source,
        prefix + ".mlp.down_proj",
        storage_type,
        config.stack.hidden_size,
        config.stack.intermediate_size,
        false);
    return out;
}

modules::QwenCausalDecodeRuntimeWeights load_qwen_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    const modules::QwenCausalDecodeRuntimeConfig & runtime_config,
    int64_t vocab_size,
    assets::TensorStorageType storage_type) {
    const auto & config = runtime_config.decoder;
    modules::QwenCausalDecodeRuntimeWeights out;
    out.token_embedding = store.load_tensor(
        source,
        prefix + ".embed_tokens.weight",
        storage_type,
        {vocab_size, config.stack.hidden_size});
    out.stack.layers.reserve(static_cast<size_t>(config.stack.layers));
    for (int64_t layer = 0; layer < config.stack.layers; ++layer) {
        out.stack.layers.push_back(load_qwen_layer(
            store,
            source,
            prefix + ".layers." + std::to_string(layer),
            config,
            storage_type));
    }
    out.final_norm = binding::norm_weight_from_source(store, source, prefix + ".norm", config.stack.hidden_size);
    return out;
}

struct RedAeWeights {
    std::string trace_prefix;
    std::shared_ptr<core::BackendWeightStore> store;
    modules::LinearWeights enc_in0;
    modules::LinearWeights enc_in1;
    modules::QwenCausalDecodeRuntimeWeights encoder_qwen;
    core::TensorValue downsample_cls;
    modules::QwenCausalDecodeRuntimeWeights downsample_qwen;
    modules::LinearWeights enc_out;
    modules::LinearWeights dec_in;
    modules::QwenCausalDecodeRuntimeWeights decoder_qwen;
    modules::LinearWeights istft_head;
    core::TensorValue istft_window;
};

std::shared_ptr<RedAeWeights> load_redae_weights(
    const RedAeCodecSources & sources,
    const RedAeCodecConfig & c,
    const RedAeCodecWeightBinding & binding_config,
    core::ExecutionContext & execution,
    const RedAeCodecRuntimeOptions & options) {
    if (sources.encoder == nullptr || sources.decoder == nullptr) {
        throw std::runtime_error("RedAE codec runtime requires encoder and decoder tensor sources");
    }
    auto weights = std::make_shared<RedAeWeights>();
    weights->trace_prefix = binding_config.trace_prefix;
    weights->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(),
        execution.backend_type(),
        binding_config.weight_store_name,
        options.weight_context_bytes);
    const auto & encoder_source = *sources.encoder;
    const auto & decoder_source = *sources.decoder;
    weights->enc_in0 = binding::linear_from_source(
        *weights->store, encoder_source, binding_config.enc_in0, options.weight_storage_type, c.enc_hidden_size, c.audio_patch_size, true);
    weights->enc_in1 = binding::linear_from_source(
        *weights->store, encoder_source, binding_config.enc_in1, options.weight_storage_type, c.enc_hidden_size, c.enc_hidden_size, true);
    const auto encoder_config = qwen_runtime_config(
        binding_config.trace_prefix + ".encoder",
        c.enc_hidden_size,
        c.enc_intermediate_size,
        c.enc_layers,
        c.enc_heads,
        c.enc_kv_heads,
        c.enc_head_dim,
        modules::QwenCausalDecoderLogitsMode::AllSteps,
        options.graph_arena_bytes,
        options.graph_arena_bytes,
        execution.backend_type(),
        true,
        c.enc_sliding_window);
    auto encoder_load_config = encoder_config;
    encoder_load_config.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
    weights->encoder_qwen = load_qwen_weights(
        *weights->store, encoder_source, binding_config.encoder_qwen, encoder_load_config, binding_config.qwen_vocab_size, options.weight_storage_type);
    weights->downsample_cls = weights->store->load_f32_tensor(
        encoder_source, binding_config.downsample_cls, {1, 1, c.enc_hidden_size});
    const auto downsample_config = qwen_runtime_config(
        binding_config.trace_prefix + ".downsample",
        c.enc_hidden_size,
        c.enc_intermediate_size,
        c.enc_downsample_layers,
        c.enc_heads,
        c.enc_kv_heads,
        c.enc_head_dim,
        modules::QwenCausalDecoderLogitsMode::AllSteps,
        options.graph_arena_bytes,
        options.graph_arena_bytes,
        execution.backend_type(),
        true,
        0);
    auto downsample_load_config = downsample_config;
    downsample_load_config.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
    weights->downsample_qwen = load_qwen_weights(
        *weights->store, encoder_source, binding_config.downsample_qwen, downsample_load_config, binding_config.qwen_vocab_size, options.weight_storage_type);
    weights->enc_out = binding::linear_from_source(
        *weights->store, encoder_source, binding_config.enc_out, options.weight_storage_type, c.bottleneck_dim, c.enc_hidden_size, true);
    weights->dec_in = binding::linear_from_source(
        *weights->store,
        decoder_source,
        binding_config.dec_in,
        options.weight_storage_type,
        c.enc_extra_downsample_rate * c.dec_hidden_size,
        c.bottleneck_dim,
        true);
    const auto decoder_config = qwen_runtime_config(
        binding_config.trace_prefix + ".decoder",
        c.dec_hidden_size,
        c.dec_intermediate_size,
        c.dec_layers,
        c.dec_heads,
        c.dec_kv_heads,
        c.dec_head_dim,
        modules::QwenCausalDecoderLogitsMode::AllSteps,
        options.graph_arena_bytes,
        options.graph_arena_bytes,
        execution.backend_type(),
        false,
        c.dec_sliding_window);
    auto decoder_load_config = decoder_config;
    decoder_load_config.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
    weights->decoder_qwen = load_qwen_weights(
        *weights->store, decoder_source, binding_config.decoder_qwen, decoder_load_config, binding_config.qwen_vocab_size, options.weight_storage_type);
    weights->istft_head = binding::linear_from_source(
        *weights->store,
        decoder_source,
        binding_config.istft_head,
        options.weight_storage_type,
        c.audio_patch_size * 4 + 2,
        c.dec_hidden_size,
        true);
    weights->istft_window = weights->store->load_f32_tensor(
        decoder_source,
        binding_config.istft_window,
        {c.audio_patch_size * 4});
    weights->store->upload();
    return weights;
}

class RedAeInGraph {
public:
    RedAeInGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const RedAeWeights> weights,
        RedAeCodecConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~RedAeInGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> run(const std::vector<float> & audio_24k) {
        if (audio_24k.empty() || static_cast<int64_t>(audio_24k.size()) % config_.audio_patch_size != 0) {
            throw std::runtime_error("RedAE codec encoder input must align to audio patch size");
        }
        const int64_t frames = static_cast<int64_t>(audio_24k.size()) / config_.audio_patch_size;
        ensure(frames);
        std::vector<float> patches(static_cast<size_t>(frames * config_.audio_patch_size));
        for (int64_t t = 0; t < frames; ++t) {
            std::copy(
                audio_24k.begin() + static_cast<std::ptrdiff_t>(t * config_.audio_patch_size),
                audio_24k.begin() + static_cast<std::ptrdiff_t>((t + 1) * config_.audio_patch_size),
                patches.begin() + static_cast<std::ptrdiff_t>(t * config_.audio_patch_size));
        }
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, frames, config_.audio_patch_size})), patches);
        const std::string label = weights_->trace_prefix + ".encoder_in";
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, label.c_str()) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("RedAE codec encoder input graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        frames_ = 0;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    void ensure(int64_t frames) {
        if (mem_.graph != nullptr && frames_ == frames) {
            return;
        }
        mem_.reset(execution_.backend());
        ggml_init_params params{graph_arena_bytes_, nullptr, true};
        mem_.ctx.reset(ggml_init(params));
        ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
        mem_.input_ctx.reset(ggml_init(input_params));
        const std::string label = weights_->trace_prefix + ".encoder_in";
        const std::string input_label = label + ".inputs";
        core::ModuleBuildContext ctx{mem_.ctx.get(), label.c_str(), execution_.backend_type()};
        core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), input_label.c_str(), execution_.backend_type()};
        auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, config_.audio_patch_size}));
        input_ = x.tensor;
        ggml_set_input(input_);
        x = modules::LinearModule({config_.audio_patch_size, config_.enc_hidden_size, true}).build(ctx, x, weights_->enc_in0);
        if (execution_.backend_type() != core::BackendType::Cpu && execution_.backend_type() != core::BackendType::Vulkan &&
            execution_.backend_type() != core::BackendType::Metal) {
            x = core::wrap_tensor(
                ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_BF16), GGML_TYPE_F32),
                x.shape,
                GGML_TYPE_F32);
        }
        x = modules::LinearModule({config_.enc_hidden_size, config_.enc_hidden_size, true}).build(ctx, x, weights_->enc_in1);
        if (execution_.backend_type() != core::BackendType::Cpu && execution_.backend_type() != core::BackendType::Vulkan &&
            execution_.backend_type() != core::BackendType::Metal) {
            x = core::wrap_tensor(
                ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_BF16), GGML_TYPE_F32),
                x.shape,
                GGML_TYPE_F32);
        }
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 8192, false);
        ggml_build_forward_expand(mem_.graph, output_);
        mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
        mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
            !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
            !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
            mem_.reset(execution_.backend());
            throw std::runtime_error("failed to allocate RedAE codec encoder input graph");
        }
        frames_ = frames;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const RedAeWeights> weights_;
    RedAeCodecConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t frames_ = 0;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class RedAeOutGraph {
public:
    RedAeOutGraph(
        core::ExecutionContext & execution,
        std::shared_ptr<const RedAeWeights> weights,
        RedAeCodecConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          graph_arena_bytes_(graph_arena_bytes) {}

    ~RedAeOutGraph() {
        mem_.reset(execution_.backend());
    }

    std::vector<float> encode_out(const std::vector<float> & hidden, int64_t rows) {
        return run_linear(
            hidden,
            rows,
            config_.enc_hidden_size,
            config_.bottleneck_dim,
            weights_->enc_out,
            weights_->trace_prefix + ".encoder_out",
            true);
    }

    std::vector<float> decode_in(const std::vector<float> & latents, int64_t rows) {
        auto up = run_linear(
            latents,
            rows,
            config_.bottleneck_dim,
            config_.enc_extra_downsample_rate * config_.dec_hidden_size,
            weights_->dec_in,
            weights_->trace_prefix + ".decoder_in",
            false);
        std::vector<float> out(static_cast<size_t>(rows * config_.enc_extra_downsample_rate * config_.dec_hidden_size));
        std::copy(up.begin(), up.end(), out.begin());
        return out;
    }

    std::vector<float> istft_head(const std::vector<float> & hidden, int64_t rows) {
        return run_linear(
            hidden,
            rows,
            config_.dec_hidden_size,
            config_.audio_patch_size * 4 + 2,
            weights_->istft_head,
            weights_->trace_prefix + ".istft_head",
            false);
    }

    void release_graph() {
        mem_.reset(execution_.backend());
        rows_ = 0;
        in_features_ = 0;
        out_features_ = 0;
        label_.clear();
        bf16_autocast_ = false;
        input_ = nullptr;
        output_ = nullptr;
    }

private:
    std::vector<float> run_linear(
        const std::vector<float> & input,
        int64_t rows,
        int64_t in_features,
        int64_t out_features,
        const modules::LinearWeights & weights,
        const std::string & label,
        bool bf16_autocast) {
        if (static_cast<int64_t>(input.size()) != rows * in_features) {
            throw std::runtime_error(std::string(label) + " input size mismatch");
        }
        if (mem_.graph == nullptr || rows_ != rows || in_features_ != in_features || out_features_ != out_features ||
            label_ != label || bf16_autocast_ != bf16_autocast) {
            mem_.reset(execution_.backend());
            ggml_init_params params{graph_arena_bytes_, nullptr, true};
            mem_.ctx.reset(ggml_init(params));
            ggml_init_params input_params{8ull * 1024ull * 1024ull, nullptr, true};
            mem_.input_ctx.reset(ggml_init(input_params));
            core::ModuleBuildContext ctx{mem_.ctx.get(), label.c_str(), execution_.backend_type()};
            core::ModuleBuildContext input_ctx{mem_.input_ctx.get(), label.c_str(), execution_.backend_type()};
            auto x = core::make_tensor(input_ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, rows, in_features}));
            input_ = x.tensor;
            ggml_set_input(input_);
            x = modules::LinearModule({in_features, out_features, true}).build(ctx, x, weights);
            if (bf16_autocast && execution_.backend_type() != core::BackendType::Cpu &&
                execution_.backend_type() != core::BackendType::Vulkan &&
                execution_.backend_type() != core::BackendType::Metal) {
                x = core::wrap_tensor(
                    ggml_cast(ctx.ggml, ggml_cast(ctx.ggml, x.tensor, GGML_TYPE_BF16), GGML_TYPE_F32),
                    x.shape,
                    GGML_TYPE_F32);
            }
            output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
            ggml_set_output(output_);
            mem_.graph = ggml_new_graph_custom(mem_.ctx.get(), 8192, false);
            ggml_build_forward_expand(mem_.graph, output_);
            mem_.input_buffer = ggml_backend_alloc_ctx_tensors(mem_.input_ctx.get(), execution_.backend());
            mem_.gallocr.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
            if (mem_.input_buffer == nullptr || mem_.gallocr == nullptr ||
                !ggml_gallocr_reserve(mem_.gallocr.get(), mem_.graph) ||
                !ggml_gallocr_alloc_graph(mem_.gallocr.get(), mem_.graph)) {
                mem_.reset(execution_.backend());
                throw std::runtime_error(std::string("failed to allocate ") + label + " graph");
            }
            rows_ = rows;
            in_features_ = in_features;
            out_features_ = out_features;
            label_ = label;
            bf16_autocast_ = bf16_autocast;
        }
        core::write_tensor_f32(core::wrap_tensor(input_, core::TensorShape::from_dims({1, rows, in_features})), input);
        if (core::compute_backend_graph(execution_.backend(), mem_.graph, nullptr, label.c_str()) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label) + " graph compute failed");
        }
        return core::read_tensor_f32(output_);
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const RedAeWeights> weights_;
    RedAeCodecConfig config_;
    size_t graph_arena_bytes_;
    GraphMemory mem_;
    int64_t rows_ = 0;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    std::string label_;
    bool bf16_autocast_ = false;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};

class RedAeRuntime {
public:
    RedAeRuntime(
        core::ExecutionContext & execution,
        std::shared_ptr<const RedAeWeights> weights,
        RedAeCodecConfig config,
        size_t graph_arena_bytes)
        : execution_(execution),
          weights_(std::move(weights)),
          config_(config),
          encoder_in_(execution, weights_, config_, graph_arena_bytes),
          encoder_out_(execution, weights_, config_, graph_arena_bytes),
          encoder_qwen_(execution, encoder_config(graph_arena_bytes), weights_->encoder_qwen),
          downsample_qwen_(execution, downsample_config(graph_arena_bytes), weights_->downsample_qwen),
          decoder_qwen_(execution, decoder_config(graph_arena_bytes), weights_->decoder_qwen),
          istft_window_(core::read_tensor_f32(weights_->istft_window.tensor)) {}

    std::vector<float> encode(const std::vector<float> & audio_24k) {
        auto hidden = encoder_in_.run(audio_24k);
        const int64_t frames = static_cast<int64_t>(audio_24k.size()) / config_.audio_patch_size;
        auto encoder = encoder_qwen_.prefill_embeddings(hidden, frames);
        const int64_t down_frames = frames / config_.enc_extra_downsample_rate;
        std::vector<float> down_input(static_cast<size_t>(down_frames * 3 * config_.enc_hidden_size));
        const auto cls = core::read_tensor_f32(weights_->downsample_cls.tensor);
        for (int64_t i = 0; i < down_frames; ++i) {
            std::copy(
                encoder.hidden.begin() + static_cast<std::ptrdiff_t>((i * 2) * config_.enc_hidden_size),
                encoder.hidden.begin() + static_cast<std::ptrdiff_t>((i * 2 + 2) * config_.enc_hidden_size),
                down_input.begin() + static_cast<std::ptrdiff_t>(i * 3 * config_.enc_hidden_size));
            std::copy(
                cls.begin(),
                cls.end(),
                down_input.begin() + static_cast<std::ptrdiff_t>((i * 3 + 2) * config_.enc_hidden_size));
        }
        auto down = downsample_qwen_.prefill_embeddings_batched(down_input, down_frames, 3);
        std::vector<float> cls_hidden(static_cast<size_t>(down_frames * config_.enc_hidden_size));
        for (int64_t i = 0; i < down_frames; ++i) {
            std::copy(
                down.hidden.begin() + static_cast<std::ptrdiff_t>((i * 3 + 2) * config_.enc_hidden_size),
                down.hidden.begin() + static_cast<std::ptrdiff_t>((i * 3 + 3) * config_.enc_hidden_size),
                cls_hidden.begin() + static_cast<std::ptrdiff_t>(i * config_.enc_hidden_size));
        }
        return encoder_out_.encode_out(cls_hidden, down_frames);
    }

    runtime::AudioBuffer decode(const std::vector<float> & latents) {
        if (latents.empty() || static_cast<int64_t>(latents.size()) % config_.bottleneck_dim != 0) {
            throw std::runtime_error("RedAE codec decode latent size mismatch");
        }
        const int64_t latent_frames = static_cast<int64_t>(latents.size()) / config_.bottleneck_dim;
        auto decoder_in = encoder_out_.decode_in(latents, latent_frames);
        const int64_t qwen_frames = latent_frames * config_.enc_extra_downsample_rate;
        auto decoder = decoder_qwen_.prefill_embeddings(decoder_in, qwen_frames);
        auto spec = encoder_out_.istft_head(decoder.hidden, qwen_frames);
        runtime::AudioBuffer audio;
        audio.sample_rate = static_cast<int>(config_.sample_rate);
        audio.channels = 1;
        if (host_istft_ == nullptr || host_istft_frames_ != qwen_frames) {
            audio::HostLogMagnitudePhaseISTFTConfig cfg;
            cfg.frames = qwen_frames;
            cfg.n_fft = config_.audio_patch_size * 4;
            cfg.hop_length = config_.audio_patch_size;
            cfg.out_dim = config_.audio_patch_size * 4 + 2;
            cfg.threads = static_cast<size_t>(std::max(1, execution_.config().threads));
            host_istft_ = std::make_unique<audio::HostLogMagnitudePhaseISTFT>(cfg);
            host_istft_frames_ = qwen_frames;
        }
        const auto result = host_istft_->compute(spec, istft_window_);
        audio.samples = result.audio;
        return audio;
    }

    void release_graphs() {
        encoder_in_.release_graph();
        encoder_out_.release_graph();
        encoder_qwen_.release_runtime_graphs();
        downsample_qwen_.release_runtime_graphs();
        decoder_qwen_.release_runtime_graphs();
        host_istft_.reset();
        host_istft_frames_ = 0;
    }

private:
    modules::QwenCausalDecodeRuntimeConfig encoder_config(size_t graph_arena_bytes) const {
        auto out = qwen_runtime_config(
            weights_->trace_prefix + ".encoder",
            config_.enc_hidden_size,
            config_.enc_intermediate_size,
            config_.enc_layers,
            config_.enc_heads,
            config_.enc_kv_heads,
            config_.enc_head_dim,
            modules::QwenCausalDecoderLogitsMode::AllSteps,
            graph_arena_bytes,
            graph_arena_bytes,
            execution_.backend_type(),
            true,
            config_.enc_sliding_window);
        out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
        return out;
    }

    modules::QwenCausalDecodeRuntimeConfig downsample_config(size_t graph_arena_bytes) const {
        auto out = qwen_runtime_config(
            weights_->trace_prefix + ".downsample",
            config_.enc_hidden_size,
            config_.enc_intermediate_size,
            config_.enc_downsample_layers,
            config_.enc_heads,
            config_.enc_kv_heads,
            config_.enc_head_dim,
            modules::QwenCausalDecoderLogitsMode::AllSteps,
            graph_arena_bytes,
            graph_arena_bytes,
            execution_.backend_type(),
            true,
            0);
        out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
        return out;
    }

    modules::QwenCausalDecodeRuntimeConfig decoder_config(size_t graph_arena_bytes) const {
        auto out = qwen_runtime_config(
            weights_->trace_prefix + ".decoder",
            config_.dec_hidden_size,
            config_.dec_intermediate_size,
            config_.dec_layers,
            config_.dec_heads,
            config_.dec_kv_heads,
            config_.dec_head_dim,
            modules::QwenCausalDecoderLogitsMode::AllSteps,
            graph_arena_bytes,
            graph_arena_bytes,
            execution_.backend_type(),
            false,
            config_.dec_sliding_window);
        out.decoder.stack.runtime.attention.prefill_mode = modules::QwenDecoderAttentionMode::ManualRepeat;
        return out;
    }

    core::ExecutionContext & execution_;
    std::shared_ptr<const RedAeWeights> weights_;
    RedAeCodecConfig config_;
    RedAeInGraph encoder_in_;
    RedAeOutGraph encoder_out_;
    modules::QwenCausalDecodeRuntime encoder_qwen_;
    modules::QwenCausalDecodeRuntime downsample_qwen_;
    modules::QwenCausalDecodeRuntime decoder_qwen_;
    std::vector<float> istft_window_;
    std::unique_ptr<audio::HostLogMagnitudePhaseISTFT> host_istft_;
    int64_t host_istft_frames_ = 0;
};


}  // namespace

struct RedAeCodecRuntime::Impl {
    Impl(
        RedAeCodecSources sources,
        core::ExecutionContext & execution,
        RedAeCodecConfig config,
        RedAeCodecWeightBinding binding,
        RedAeCodecRuntimeOptions options)
        : runtime(
              execution,
              load_redae_weights(sources, config, binding, execution, options),
              config,
              options.graph_arena_bytes) {}

    RedAeRuntime runtime;
};

RedAeCodecRuntime::RedAeCodecRuntime(
    RedAeCodecSources sources,
    core::ExecutionContext & execution,
    RedAeCodecConfig config,
    RedAeCodecWeightBinding binding,
    RedAeCodecRuntimeOptions options)
    : impl_(std::make_unique<Impl>(
          std::move(sources),
          execution,
          std::move(config),
          std::move(binding),
          options)) {}

RedAeCodecRuntime::~RedAeCodecRuntime() = default;

std::vector<float> RedAeCodecRuntime::encode(const std::vector<float> & audio_24k) {
    return impl_->runtime.encode(audio_24k);
}

runtime::AudioBuffer RedAeCodecRuntime::decode(const std::vector<float> & latents) {
    return impl_->runtime.decode(latents);
}

void RedAeCodecRuntime::release_runtime_graphs() {
    impl_->runtime.release_graphs();
}

}  // namespace engine::codecs
