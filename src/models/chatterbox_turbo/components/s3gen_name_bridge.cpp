#include "s3gen_name_bridge.h"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace engine::models::chatterbox_turbo {
namespace {

using Rule = std::pair<std::string_view, std::string_view>;

// Longest/most-specific prefixes first so a shorter rule never shadows a longer one
// (e.g. "flow.encoder_proj" must not be caught by a "flow.encoder." rule).
constexpr std::array<Rule, 14> kFlowPrefixRules = {{
    {"flow.encoder.encoders.", "fe.enc."},
    {"flow.encoder.up_encoders.", "fe.ue."},
    {"flow.encoder.up_layer.", "fe.ul."},
    {"flow.encoder.up_embed.", "fe.uemb."},
    {"flow.encoder.pre_lookahead_layer.", "fe.pla."},
    {"flow.encoder.embed.", "fe.embed."},
    {"flow.encoder.after_norm", "fe.an"},
    {"flow.decoder.estimator.down_blocks.", "fd.db."},
    {"flow.decoder.estimator.mid_blocks.", "fd.mb."},
    {"flow.decoder.estimator.up_blocks.", "fd.ub."},
    {"flow.decoder.estimator.final_block", "fd.fb"},
    {"flow.decoder.estimator.final_proj", "fd.fp"},
    {"flow.decoder.estimator.time_mlp.", "fd.tm."},
    {"flow.decoder.estimator.time_embed_mixer", "fd.tmx"},
    // flow.encoder_proj / flow.spk_embed_affine_layer / flow.input_embedding: unchanged.
}};

// Applied to whatever remains after a prefix rule above consumed the "flow.decoder.estimator.*"
// or "flow.encoder.{encoders,up_encoders}.N." portion of the name.
constexpr std::array<Rule, 15> kSegmentRules = {{
    {".self_attn.linear_q", ".sa.lq"},
    {".self_attn.linear_k", ".sa.lk"},
    {".self_attn.linear_v", ".sa.lv"},
    {".self_attn.linear_out", ".sa.lo"},
    {".self_attn.linear_pos", ".sa.lp"},
    {".self_attn.pos_bias_u", ".sa.pbu"},
    {".self_attn.pos_bias_v", ".sa.pbv"},
    {".norm_mha", ".nmha"},
    {".norm_ff", ".nff"},
    {".feed_forward.w_1", ".ff.w_1"},
    {".feed_forward.w_2", ".ff.w_2"},
    {".block1.block.0", ".b1.0"},
    {".block1.block.2", ".b1.2"},
    {".block2.block.0", ".b2.0"},
    {".block2.block.2", ".b2.2"},
}};

constexpr std::array<Rule, 6> kTransformerBlockRules = {{
    {".res_conv", ".rc"},
    {".attn1.to_q", ".attn1.q"},
    {".attn1.to_k", ".attn1.k"},
    {".attn1.to_v", ".attn1.v"},
    {".attn1.to_out.0", ".attn1.o"},
    {".ff.net.0.proj", ".ff.up"},
}};

constexpr std::array<Rule, 6> kVocoderPrefixRules = {{
    {"conv_pre", "cpre"},
    {"conv_post", "cpost"},
    {"resblocks.", "rb."},
    {"source_downs.", "sd."},
    {"source_resblocks.", "srb."},
    {"f0_predictor.condnet.", "f0.cn."},
}};

constexpr std::array<Rule, 6> kVocoderExactRules = {{
    {"f0_predictor.classifier", "f0.cls"},
    {"m_source.l_linear", "ms.ll"},
    {".convs1.", ".c1."},
    {".convs2.", ".c2."},
    {".activations1.", ".a1."},
    {".activations2.", ".a2."},
}};

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

std::string replace_all(std::string text, std::string_view search, std::string_view replace) {
    if (search.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(search, pos)) != std::string::npos) {
        text.replace(pos, search.size(), replace);
        pos += replace.size();
    }
    return text;
}

std::string translate_flow_name(std::string_view name) {
    std::string translated(name);
    for (const auto & [from, to] : kFlowPrefixRules) {
        if (starts_with(translated, from)) {
            translated = std::string(to) + translated.substr(from.size());
            break;
        }
    }
    for (const auto & [from, to] : kSegmentRules) {
        translated = replace_all(std::move(translated), from, to);
    }
    // ".ff.net.2" would also match the shorter ".ff.net.0.proj" search text's neighbourhood,
    // so handle it (and the rest of the transformer-block-only renames) after the segment pass.
    translated = replace_all(std::move(translated), ".ff.net.2", ".ff.down");
    for (const auto & [from, to] : kTransformerBlockRules) {
        translated = replace_all(std::move(translated), from, to);
    }
    return translated;
}

std::string translate_vocoder_name(std::string_view name) {
    std::string translated(name);
    for (const auto & [from, to] : kVocoderPrefixRules) {
        if (starts_with(translated, from)) {
            translated = std::string(to) + translated.substr(from.size());
            break;
        }
    }
    for (const auto & [from, to] : kVocoderExactRules) {
        if (translated == from || starts_with(translated, std::string(from) + ".")) {
            translated = std::string(to) + translated.substr(from.size());
            break;
        }
        translated = replace_all(translated, from, to);
    }
    // HiftVocoderConfig.tensor_prefix is left empty (see s3gen_turbo.cpp) so this translator
    // sees HiftVocoderComponent's plain requested names; the turbo GGUF's vocoder tensors all
    // live under the top-level "v." namespace, which is added here rather than via tensor_prefix.
    return "v." + translated;
}

class TranslatingTensorSource final : public engine::assets::TensorSource {
public:
    TranslatingTensorSource(
        std::shared_ptr<const engine::assets::TensorSource> delegate,
        std::string (*translate)(std::string_view))
        : delegate_(std::move(delegate)), translate_(translate) {}

    const std::filesystem::path & source_path() const noexcept override {
        return delegate_->source_path();
    }
    bool has_tensor(std::string_view name) const noexcept override {
        return delegate_->has_tensor(translate_(name));
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        return delegate_->require_metadata(translate_(name));
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        return delegate_->tensors();
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        return delegate_->require_tensor_data(translate_(name));
    }
    // Some turbo GGUF tensors squeeze a leading size-1 dimension the base loader code still
    // expects explicitly (e.g. the vocoder's f0 classifier: nn.Linear(512, 1).weight is stored
    // as ne=[512] rather than [1, 512]). Rather than special-case those, skip the delegate's
    // strict per-dimension shape check here and only verify the element count still matches.
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        auto values = delegate_->require_f32(translate_(name), std::nullopt);
        if (expected_shape.has_value()) {
            int64_t expected_count = 1;
            for (int64_t dim : *expected_shape) {
                expected_count *= dim;
            }
            if (static_cast<int64_t>(values.size()) != expected_count) {
                throw std::runtime_error("tensor element count mismatch for " + std::string(name));
            }
        }
        return values;
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        return delegate_->require_i64_scalar(translate_(name));
    }

private:
    std::shared_ptr<const engine::assets::TensorSource> delegate_;
    std::string (*translate_)(std::string_view);
};

}  // namespace

std::shared_ptr<const engine::assets::TensorSource> make_turbo_s3gen_flow_bridge(
    std::shared_ptr<const engine::assets::TensorSource> turbo_s3_source) {
    return std::make_shared<TranslatingTensorSource>(std::move(turbo_s3_source), &translate_flow_name);
}

std::shared_ptr<const engine::assets::TensorSource> make_turbo_vocoder_bridge(
    std::shared_ptr<const engine::assets::TensorSource> turbo_s3_source) {
    return std::make_shared<TranslatingTensorSource>(std::move(turbo_s3_source), &translate_vocoder_name);
}

}  // namespace engine::models::chatterbox_turbo
