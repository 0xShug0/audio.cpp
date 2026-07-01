#include "engine/models/moss_tts_local/codec_decoder.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/moss_tts_local/codec_quantizer.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;

constexpr float kMaskedAttentionBias = std::numeric_limits<float>::lowest();
constexpr int64_t kCodeDim = 768;
constexpr int64_t kSamplesPerFrame = 3840;  // downsample_rate (per interleaved stream frame)
constexpr float kRopeTheta = 10000.0F;
constexpr float kLayerNormEps = 1.0e-5F;

// One decoder sub-transformer (ProjectedTransformer) description, mirroring the
// v2 codec config.json decoder_kwargs (indices decoder.0/2/4/6/8/10). context
// is the local-attention window in tokens at that stage's frame rate, computed
// as round(frame_rate * context_duration); the patch that follows doubles the
// frame rate (the last one expands by 240).
struct TransformerSpec {
    int64_t input_dim;
    int64_t output_dim;
    int64_t d_model;
    int64_t num_heads;
    int64_t num_layers;
    int64_t intermediate_size;
    int64_t context;
    int64_t patch_after;
};

constexpr TransformerSpec kDecoderSpecs[] = {
    {kCodeDim, 1280, 1280, 20, 32, 5120, 125, 2},
    {640, 768, 768, 12, 12, 3072, 250, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 768, 768, 12, 12, 3072, 400, 2},
    {384, 240, 768, 12, 12, 3072, 400, 240},
};

constexpr size_t kNumTransformers = sizeof(kDecoderSpecs) / sizeof(kDecoderSpecs[0]);

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct LayerWeights {
    core::TensorValue norm1_w;
    core::TensorValue norm1_b;
    core::TensorValue in_proj;   // fused qkv [3 * d_model, d_model]
    core::TensorValue out_proj;  // [d_model, d_model]
    core::TensorValue norm2_w;
    core::TensorValue norm2_b;
    core::TensorValue fc1;  // [intermediate_size, d_model]
    core::TensorValue fc2;  // [d_model, intermediate_size]
    core::TensorValue layer_scale1;  // [d_model]
    core::TensorValue layer_scale2;  // [d_model]
};

struct TransformerWeights {
    TransformerSpec spec;
    core::TensorValue input_proj;   // [d_model, input_dim]
    core::TensorValue output_proj;  // [output_dim, d_model]
    std::vector<LayerWeights> layers;
};

// Opens the codec safetensors shards and resolves a tensor to the shard that
// holds it (the codec ships model-0000N-of-00003.safetensors + an index).
class CodecShards {
public:
    explicit CodecShards(const std::filesystem::path & codec_dir) {
        for (const auto & entry : std::filesystem::directory_iterator(codec_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                sources_.push_back(assets::open_tensor_source(entry.path()));
            }
        }
        if (sources_.empty()) {
            throw std::runtime_error("MOSS codec has no safetensors shards: " + codec_dir.string());
        }
    }

    const assets::TensorSource & source_for(const std::string & name) const {
        for (const auto & source : sources_) {
            if (source->has_tensor(name)) {
                return *source;
            }
        }
        throw std::runtime_error("MOSS codec tensor not found: " + name);
    }

private:
    std::vector<std::shared_ptr<const assets::TensorSource>> sources_;
};

TransformerWeights load_transformer(
    core::BackendWeightStore & store,
    const CodecShards & shards,
    const TransformerSpec & spec,
    int64_t decoder_index) {
    const std::string prefix = "decoder." + std::to_string(decoder_index);
    const auto load = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_tensor(shards.source_for(name), name, assets::TensorStorageType::F32, shape);
    };
    const auto load_f32 = [&](const std::string & name, std::initializer_list<int64_t> shape) {
        return store.load_f32_tensor(shards.source_for(name), name, shape);
    };

    TransformerWeights weights;
    weights.spec = spec;
    weights.input_proj = load(prefix + ".input_proj.weight", {spec.d_model, spec.input_dim});
    weights.output_proj = load(prefix + ".output_proj.weight", {spec.output_dim, spec.d_model});
    weights.layers.reserve(static_cast<size_t>(spec.num_layers));
    for (int64_t layer = 0; layer < spec.num_layers; ++layer) {
        const std::string lp = prefix + ".transformer.layers." + std::to_string(layer);
        LayerWeights w;
        w.norm1_w = load_f32(lp + ".norm1.weight", {spec.d_model});
        w.norm1_b = load_f32(lp + ".norm1.bias", {spec.d_model});
        w.in_proj = load(lp + ".self_attn.in_proj.weight", {3 * spec.d_model, spec.d_model});
        w.out_proj = load(lp + ".self_attn.out_proj.weight", {spec.d_model, spec.d_model});
        w.norm2_w = load_f32(lp + ".norm2.weight", {spec.d_model});
        w.norm2_b = load_f32(lp + ".norm2.bias", {spec.d_model});
        w.fc1 = load(lp + ".ffn.0.weight", {spec.intermediate_size, spec.d_model});
        w.fc2 = load(lp + ".ffn.2.weight", {spec.d_model, spec.intermediate_size});
        w.layer_scale1 = load_f32(lp + ".layer_scale_1.scale", {spec.d_model});
        w.layer_scale2 = load_f32(lp + ".layer_scale_2.scale", {spec.d_model});
        weights.layers.push_back(std::move(w));
    }
    return weights;
}

core::TensorValue ensure_contiguous(core::ModuleBuildContext & ctx, const core::TensorValue & value) {
    return core::ensure_backend_addressable_layout(ctx, value);
}

core::TensorValue reshape_heads(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t heads,
    int64_t dim) {
    auto contiguous = ensure_contiguous(ctx, input);
    return core::reshape_tensor(
        ctx,
        contiguous,
        core::TensorShape::from_dims({input.shape.dims[0], input.shape.dims[1], heads, dim}));
}

// Slices a fused qkv projection [1, T, 3*d_model] into its q/k/v thirds, each
// [1, T, d_model]. The 3*d_model axis is laid out as [3, heads, head_dim].
core::TensorValue slice_projection(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & qkv,
    int64_t which,
    int64_t d_model,
    int64_t steps) {
    ggml_tensor * source = qkv.tensor;
    const size_t element_size = ggml_element_size(source);
    ggml_tensor * view = ggml_view_3d(
        ctx.ggml,
        source,
        d_model,
        steps,
        1,
        source->nb[1],
        source->nb[2],
        static_cast<size_t>(which * d_model) * element_size);
    return core::wrap_tensor(
        ggml_cont(ctx.ggml, view),
        core::TensorShape::from_dims({1, steps, d_model}),
        GGML_TYPE_F32);
}

core::TensorValue attention(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & q_heads,
    const core::TensorValue & k_heads,
    const core::TensorValue & v_heads,
    int64_t dim,
    const core::TensorValue & mask) {
    const modules::MatMulModule matmul;
    auto scores = matmul.build(
        ctx,
        q_heads,
        modules::TransposeModule({{0, 1, 3, 2}, k_heads.shape.rank}).build(ctx, k_heads));
    scores = core::ensure_backend_addressable_layout(ctx, scores);
    auto attn = core::wrap_tensor(
        ggml_soft_max_ext(
            ctx.ggml,
            scores.tensor,
            mask.tensor,
            1.0F / std::sqrt(static_cast<float>(dim)),
            0.0F),
        scores.shape,
        GGML_TYPE_F32);
    return matmul.build(ctx, attn, v_heads);
}

core::TensorValue transformer_layer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const LayerWeights & weights,
    const TransformerSpec & spec,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    int64_t steps) {
    const int64_t dim = spec.d_model / spec.num_heads;
    const modules::LayerNormModule norm({spec.d_model, kLayerNormEps, true, true});

    auto normed = norm.build(ctx, input, binding::norm_data(ctx, weights.norm1_w, weights.norm1_b));
    auto qkv = modules::LinearModule(binding::linear_config(spec.d_model, 3 * spec.d_model, false))
                   .build(ctx, normed, binding::linear_data(ctx, weights.in_proj));

    auto q = slice_projection(ctx, qkv, 0, spec.d_model, steps);
    auto k = slice_projection(ctx, qkv, 1, spec.d_model, steps);
    auto v = slice_projection(ctx, qkv, 2, spec.d_model, steps);

    q = reshape_heads(ctx, q, spec.num_heads, dim);
    k = reshape_heads(ctx, k, spec.num_heads, dim);
    v = reshape_heads(ctx, v, spec.num_heads, dim);
    q = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, q, positions);
    k = modules::RoPEModule({dim, GGML_ROPE_TYPE_NORMAL, kRopeTheta}).build(ctx, k, positions);

    auto q_heads = modules::TransposeModule({{0, 2, 1, 3}, q.shape.rank}).build(ctx, q);
    auto k_heads = modules::TransposeModule({{0, 2, 1, 3}, k.shape.rank}).build(ctx, k);
    auto v_heads = modules::TransposeModule({{0, 2, 1, 3}, v.shape.rank}).build(ctx, v);
    auto context = attention(ctx, q_heads, k_heads, v_heads, dim, mask);
    context = modules::TransposeModule({{0, 2, 1, 3}, context.shape.rank}).build(ctx, context);
    context = ensure_contiguous(ctx, context);
    context = core::reshape_tensor(ctx, context, core::TensorShape::from_dims({1, steps, spec.d_model}));
    auto attn_out = modules::LinearModule(binding::linear_config(spec.d_model, spec.d_model, false))
                        .build(ctx, context, binding::linear_data(ctx, weights.out_proj));
    attn_out = core::wrap_tensor(
        ggml_mul(ctx.ggml, attn_out.tensor, weights.layer_scale1.tensor), attn_out.shape, GGML_TYPE_F32);
    auto x = modules::AddModule{}.build(ctx, input, attn_out);

    auto ff_in = norm.build(ctx, x, binding::norm_data(ctx, weights.norm2_w, weights.norm2_b));
    auto ff = modules::LinearModule(binding::linear_config(spec.d_model, spec.intermediate_size, false))
                  .build(ctx, ff_in, binding::linear_data(ctx, weights.fc1));
    ff = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, ff);
    ff = modules::LinearModule(binding::linear_config(spec.intermediate_size, spec.d_model, false))
             .build(ctx, ff, binding::linear_data(ctx, weights.fc2));
    ff = core::wrap_tensor(
        ggml_mul(ctx.ggml, ff.tensor, weights.layer_scale2.tensor), ff.shape, GGML_TYPE_F32);
    return modules::AddModule{}.build(ctx, x, ff);
}

// ProjectedTransformer: input projection -> transformer stack -> output
// projection. Input/output are [1, steps, channels] (feature-last).
core::TensorValue run_transformer(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    const TransformerWeights & weights,
    const core::TensorValue & positions,
    const core::TensorValue & mask,
    int64_t steps) {
    const auto & spec = weights.spec;
    auto x = modules::LinearModule(binding::linear_config(spec.input_dim, spec.d_model, false))
                 .build(ctx, input, binding::linear_data(ctx, weights.input_proj));
    for (const auto & layer : weights.layers) {
        x = transformer_layer(ctx, x, layer, spec, positions, mask, steps);
    }
    return modules::LinearModule(binding::linear_config(spec.d_model, spec.output_dim, false))
        .build(ctx, x, binding::linear_data(ctx, weights.output_proj));
}

// PatchedPretransform (decode/upsample): [1, l, d*patch] -> [1, l*patch, d].
// Each frame is unpacked into `patch` consecutive frames along time, matching
// x.reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l * h).
core::TensorValue patch_upsample(
    core::ModuleBuildContext & ctx,
    const core::TensorValue & input,
    int64_t patch) {
    auto contiguous = ensure_contiguous(ctx, input);
    const int64_t length = contiguous.shape.dims[1];
    const int64_t packed = contiguous.shape.dims[2];
    const int64_t channels = packed / patch;
    ggml_tensor * reshaped = ggml_reshape_3d(ctx.ggml, contiguous.tensor, patch, channels, length);
    ggml_tensor * permuted = ggml_cont(ctx.ggml, ggml_permute(ctx.ggml, reshaped, 1, 0, 2, 3));
    ggml_tensor * unpacked = ggml_reshape_2d(ctx.ggml, permuted, channels, length * patch);
    return core::wrap_tensor(
        unpacked, core::TensorShape::from_dims({1, length * patch, channels}), GGML_TYPE_F32);
}

std::vector<float> causal_context_mask(int64_t steps, int64_t context) {
    std::vector<float> mask(static_cast<size_t>(steps * steps), kMaskedAttentionBias);
    for (int64_t query = 0; query < steps; ++query) {
        for (int64_t key = 0; key <= query; ++key) {
            if (query - key < context) {
                mask[static_cast<size_t>(query * steps + key)] = 0.0F;
            }
        }
    }
    return mask;
}

}  // namespace

struct MossCodecDecoder::Impl {
    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    size_t graph_arena_bytes = 0;
    std::unique_ptr<MossCodecDequantizer> dequantizer;
    std::unique_ptr<core::BackendWeightStore> store;
    std::vector<TransformerWeights> transformers;
};

MossCodecDecoder::MossCodecDecoder(
    const std::filesystem::path & codec_dir,
    core::ExecutionContext & execution_context,
    int64_t num_quantizers,
    size_t weight_context_bytes,
    size_t graph_arena_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->backend = execution_context.backend();
    if (impl_->backend == nullptr) {
        throw std::runtime_error("MOSS codec decoder backend is not initialized");
    }
    impl_->backend_type = execution_context.backend_type();
    impl_->graph_arena_bytes = graph_arena_bytes;
    impl_->dequantizer = std::make_unique<MossCodecDequantizer>(codec_dir, num_quantizers);

    CodecShards shards(codec_dir);
    impl_->store = std::make_unique<core::BackendWeightStore>(
        impl_->backend, impl_->backend_type, "moss_tts_local.codec.decoder", weight_context_bytes);
    impl_->transformers.reserve(kNumTransformers);
    for (size_t index = 0; index < kNumTransformers; ++index) {
        impl_->transformers.push_back(
            load_transformer(*impl_->store, shards, kDecoderSpecs[index], static_cast<int64_t>(2 * index)));
    }
    impl_->store->upload();
}

MossCodecDecoder::~MossCodecDecoder() = default;

int64_t MossCodecDecoder::sampling_rate() const noexcept {
    return 48000;
}

std::vector<std::vector<float>> MossCodecDecoder::decode(
    const std::vector<std::vector<int32_t>> & codes) const {
    const int64_t frames = codes.empty() ? 0 : static_cast<int64_t>(codes.front().size());
    if (frames <= 0) {
        throw std::runtime_error("MOSS codec decoder requires a non-empty code sequence");
    }

    // Codes -> continuous latent [code_dim, frames] (channel-major), transposed
    // into the feature-last [1, frames, code_dim] layout the decoder expects.
    const auto latent = impl_->dequantizer->decode(codes);
    std::vector<float> latent_input(static_cast<size_t>(frames * kCodeDim));
    for (int64_t channel = 0; channel < kCodeDim; ++channel) {
        for (int64_t step = 0; step < frames; ++step) {
            latent_input[static_cast<size_t>(step * kCodeDim + channel)] =
                latent[static_cast<size_t>(channel * frames + step)];
        }
    }

    ggml_init_params params{impl_->graph_arena_bytes, nullptr, true};
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_ctx(ggml_init(params));
    if (graph_ctx == nullptr) {
        throw std::runtime_error("failed to initialize MOSS codec decoder graph context");
    }
    core::ModuleBuildContext ctx{graph_ctx.get(), "moss_tts_local.codec.decode", impl_->backend_type};

    auto latent_tensor =
        core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, kCodeDim}));
    ggml_set_input(latent_tensor.tensor);

    struct StageInput {
        ggml_tensor * positions;
        std::vector<int32_t> position_host;
        ggml_tensor * mask;
        std::vector<float> mask_host;
    };
    std::vector<StageInput> stage_inputs;
    stage_inputs.reserve(impl_->transformers.size());

    auto hidden = latent_tensor;
    int64_t steps = frames;
    for (const auto & transformer : impl_->transformers) {
        auto positions = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({steps}));
        ggml_set_input(positions.tensor);
        auto mask =
            core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1, steps, steps}));
        ggml_set_input(mask.tensor);

        StageInput stage;
        stage.positions = positions.tensor;
        stage.position_host.resize(static_cast<size_t>(steps));
        for (int64_t i = 0; i < steps; ++i) {
            stage.position_host[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        }
        stage.mask = mask.tensor;
        stage.mask_host = causal_context_mask(steps, transformer.spec.context);
        stage_inputs.push_back(std::move(stage));

        hidden = run_transformer(ctx, hidden, transformer, positions, mask, steps);
        hidden = patch_upsample(ctx, hidden, transformer.spec.patch_after);
        steps *= transformer.spec.patch_after;
    }

    hidden = ensure_contiguous(ctx, hidden);
    ggml_set_output(hidden.tensor);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 131072, false);
    ggml_build_forward_expand(graph, hidden.tensor);

    ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) || !ggml_gallocr_alloc_graph(gallocr, graph)) {
        if (gallocr != nullptr) {
            ggml_gallocr_free(gallocr);
        }
        throw std::runtime_error("failed to allocate MOSS codec decoder forward graph");
    }

    ggml_backend_tensor_set(
        latent_tensor.tensor, latent_input.data(), 0, latent_input.size() * sizeof(float));
    for (const auto & stage : stage_inputs) {
        ggml_backend_tensor_set(
            stage.positions, stage.position_host.data(), 0, stage.position_host.size() * sizeof(int32_t));
        ggml_backend_tensor_set(
            stage.mask, stage.mask_host.data(), 0, stage.mask_host.size() * sizeof(float));
    }

    const ggml_status status = ggml_backend_graph_compute(impl_->backend, graph);
    ggml_backend_synchronize(impl_->backend);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_gallocr_free(gallocr);
        throw std::runtime_error("MOSS codec decoder forward graph compute failed");
    }

    const int64_t interleaved = steps;  // frames * 3840 * 2 (stereo interleaved)
    std::vector<float> flat(static_cast<size_t>(interleaved));
    ggml_backend_tensor_get(hidden.tensor, flat.data(), 0, flat.size() * sizeof(float));
    ggml_gallocr_free(gallocr);

    // De-interleave the jointly-processed stream back into left/right channels
    // (channel 0 = even samples, channel 1 = odd samples).
    const int64_t per_channel = frames * kSamplesPerFrame;
    std::vector<std::vector<float>> stereo(2, std::vector<float>(static_cast<size_t>(per_channel)));
    for (int64_t i = 0; i < per_channel; ++i) {
        stereo[0][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i)];
        stereo[1][static_cast<size_t>(i)] = flat[static_cast<size_t>(2 * i + 1)];
    }
    return stereo;
}

}  // namespace engine::models::moss_tts_local
