#include "engine/models/moss_soundeffect/dit.h"

#include "engine/framework/modules/transformers/dit_runtime.h"

namespace engine::models::moss_soundeffect {
namespace {
modules::DitRuntimeConfig runtime_config(core::ExecutionContext & execution, const Assets & assets,
    assets::TensorStorageType storage, core::AttentionPreference preference, bool optimized) {
    const auto & c = assets.dit;
    modules::DitRuntimeConfig out;
    out.label = "moss_soundeffect.dit";
    out.weight_type = storage;

    out.optimizations = {optimized, optimized, optimized};
    auto & m = out.model;
    m.input_channels = c.latent_dim; m.output_channels = c.latent_dim;
    m.hidden_size = c.dim; m.intermediate_size = c.ffn_dim; m.layers = c.layers;
    m.norm_eps = c.eps; m.context_dim = c.text_dim; m.time_features = c.freq_dim;
    m.modulation = modules::DitModulation::SharedTable;
    m.projection_type = modules::DitProjectionType::Bf16Autocast;
    m.context_activation = modules::DitEmbeddingActivation::Gelu;
    const auto lowering = core::resolve_flash_attention(execution.backend(), c.dim / c.heads, preference)
        ? modules::ScaledDotProductAttentionLowering::FlashPreserveViews : modules::ScaledDotProductAttentionLowering::Explicit;
    m.self_attention = {c.heads, c.heads, c.dim / c.heads, modules::DitQKNorm::Projection, c.eps,
        {c.dim / c.heads, GGML_ROPE_TYPE_NORMAL, 10000.f}, lowering};
    m.cross_attention = m.self_attention; m.cross_attention.rotary.dimensions = 0;
    return out;
}

modules::DitWeightNames weight_names() {
    modules::DitWeightNames w;
    w.input = {"patch_embedding.weight", "patch_embedding.bias"};
    w.output = {"head.head.weight", "head.head.bias"};
    w.time_embedding = {{"time_embedding.0.weight", "time_embedding.0.bias"}, {"time_embedding.2.weight", "time_embedding.2.bias"}};
    w.context_embedding = {{"text_embedding.0.weight", "text_embedding.0.bias"}, {"text_embedding.2.weight", "text_embedding.2.bias"}};
    w.time_modulation = {"time_projection.1.weight", "time_projection.1.bias"};
    w.final_modulation_table = "head.modulation";
    w.layer.modulation_table = "blocks.{layer}.modulation";
    w.layer.cross_norm = {"blocks.{layer}.norm3.weight", "blocks.{layer}.norm3.bias"};
    w.layer.self_attention = {
        {"blocks.{layer}.self_attn.q.weight", "blocks.{layer}.self_attn.q.bias"},
        {"blocks.{layer}.self_attn.k.weight", "blocks.{layer}.self_attn.k.bias"},
        {"blocks.{layer}.self_attn.v.weight", "blocks.{layer}.self_attn.v.bias"},
        {"blocks.{layer}.self_attn.o.weight", "blocks.{layer}.self_attn.o.bias"}, {}, {},
        {"blocks.{layer}.self_attn.norm_q.weight", ""}, {"blocks.{layer}.self_attn.norm_k.weight", ""}};
    w.layer.cross_attention = {
        {"blocks.{layer}.cross_attn.q.weight", "blocks.{layer}.cross_attn.q.bias"},
        {"blocks.{layer}.cross_attn.k.weight", "blocks.{layer}.cross_attn.k.bias"},
        {"blocks.{layer}.cross_attn.v.weight", "blocks.{layer}.cross_attn.v.bias"},
        {"blocks.{layer}.cross_attn.o.weight", "blocks.{layer}.cross_attn.o.bias"}, {}, {},
        {"blocks.{layer}.cross_attn.norm_q.weight", ""}, {"blocks.{layer}.cross_attn.norm_k.weight", ""}};
    w.layer.feed_forward_in = {"blocks.{layer}.ffn.0.weight", "blocks.{layer}.ffn.0.bias"};
    w.layer.feed_forward_out = {"blocks.{layer}.ffn.2.weight", "blocks.{layer}.ffn.2.bias"};
    return w;
}
} // namespace

struct DitRuntime::Impl {
    std::shared_ptr<const Assets> assets;
    modules::DitRuntime runtime;
    Impl(core::ExecutionContext & execution, std::shared_ptr<const Assets> a, assets::TensorStorageType storage,
        core::AttentionPreference attention, bool optimized)
        : assets(std::move(a)), runtime(execution, *assets->dit_weights,
            runtime_config(execution, *assets, storage, attention, optimized), weight_names()) {}
};

DitRuntime::DitRuntime(core::ExecutionContext & execution, std::shared_ptr<const Assets> assets,
    assets::TensorStorageType storage, core::AttentionPreference attention, bool optimized)
    : impl_(std::make_unique<Impl>(execution, std::move(assets), storage, attention, optimized)) {}
DitRuntime::~DitRuntime() = default;
void DitRuntime::prepare_context(const std::vector<float> & positive, const std::vector<float> & negative) {
    const auto & a = *impl_->assets;
    modules::DitConditioning conditions;
    conditions.frames = static_cast<int64_t>(a.max_duration_sec) * a.sample_rate / a.vae.hop_length;
    conditions.branches = 2; conditions.context_tokens = a.text_length;
    conditions.context = positive;
    conditions.context.insert(conditions.context.end(), negative.begin(), negative.end());
    impl_->runtime.prepare_conditioning(std::move(conditions));
}
std::vector<float> DitRuntime::predict(const std::vector<float> & latents, float timestep, bool negative) {
    return impl_->runtime.predict(latents, timestep, negative ? 1 : 0);
}
std::vector<float> DitRuntime::predict_cfg(const std::vector<float> & latents, float timestep) {
    return impl_->runtime.predict_cfg(latents, timestep);
}
std::vector<float> DitRuntime::sample(const std::vector<float> & noise, int steps, float guidance, float shift) {
    modules::DitSamplingRequest request;
    request.noise = noise; request.steps = steps; request.guidance = guidance;
    request.schedule_type = modules::DitSchedule::Shifted; request.sigma_shift = shift; request.timestep_scale = 1000.f;
    return impl_->runtime.sample(request);
}
} // namespace engine::models::moss_soundeffect
