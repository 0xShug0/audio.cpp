#include "engine/community_models/f5_tts/weights.h"

#include "engine/framework/assets/tensor_source.h"

#include <string>

namespace engine::models::f5_tts {
namespace {

// Strips the "ema_model.transformer." prefix from checkpoint tensor names so
// the framework tensor source resolves them by their torch module paths.
}  // namespace

F5DiTWeights load_dit_weights(
    const assets::TensorSource & source,
    ggml_backend_t backend,
    core::BackendType backend_type) {
    F5DiTWeights w;
    w.store = std::make_shared<core::BackendWeightStore>(
        backend, backend_type, "f5_tts.weights", 2ULL * 1024ULL * 1024ULL * 1024ULL);

    const auto tensor = [&](const std::string & n,
                            std::initializer_list<int64_t> shape) {
        return w.store->load_f32_tensor(source, n, shape);
    };
    const auto linear = [&](const std::string & n, int64_t out_f, int64_t in_f) {
        modules::LinearWeights lw;
        lw.weight = tensor(n + ".weight", {out_f, in_f});
        lw.bias = tensor(n + ".bias", {out_f});
        return lw;
    };

    constexpr int64_t kVocab = 2731;
    constexpr int64_t kTextDim = 512;
    constexpr int64_t kDim = 1024;
    constexpr int64_t kFF = 2048;
    constexpr int64_t kMel = 100;

    w.text_embedding = tensor("text_embed.text_embed.weight", {kVocab, kTextDim});
    w.input_proj = linear("input_embed.proj", kDim, kMel * 2 + kTextDim);
    w.cpe0.weight = tensor("input_embed.conv_pos_embed.conv1d.0.weight", {kDim, kDim / 16, 31});
    w.cpe0.bias = tensor("input_embed.conv_pos_embed.conv1d.0.bias", {kDim});
    w.cpe2.weight = tensor("input_embed.conv_pos_embed.conv1d.2.weight", {kDim, kDim / 16, 31});
    w.cpe2.bias = tensor("input_embed.conv_pos_embed.conv1d.2.bias", {kDim});
    w.time0 = linear("time_embed.time_mlp.0", kDim, 256);
    w.time2 = linear("time_embed.time_mlp.2", kDim, kDim);

    w.text_blocks.reserve(4);
    for (int i = 0; i < 4; ++i) {
        const std::string p = "text_embed.text_blocks." + std::to_string(i);
        F5TextConvNextWeights b;
        b.dwconv.weight = tensor(p + ".dwconv.weight", {kTextDim, 1, 7});
        b.dwconv.bias = tensor(p + ".dwconv.bias", {kTextDim});
        b.norm.weight = tensor(p + ".norm.weight", {kTextDim});
        b.norm.bias = tensor(p + ".norm.bias", {kTextDim});
        b.pw1 = linear(p + ".pwconv1", kDim, kTextDim);
        b.pw2 = linear(p + ".pwconv2", kTextDim, kDim);
        b.grn_gamma = source.require_f32(p + ".grn.gamma");
        b.grn_beta = source.require_f32(p + ".grn.beta");
        w.text_blocks.push_back(std::move(b));
    }
    w.blocks.reserve(22);
    for (int i = 0; i < 22; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i);
        F5BlockWeights b;
        b.attn_norm = linear(p + ".attn_norm.linear", 6 * kDim, kDim);
        b.to_q = linear(p + ".attn.to_q", kDim, kDim);
        b.to_k = linear(p + ".attn.to_k", kDim, kDim);
        b.to_v = linear(p + ".attn.to_v", kDim, kDim);
        b.to_out = linear(p + ".attn.to_out.0", kDim, kDim);
        b.ff0 = linear(p + ".ff.ff.0.0", kFF, kDim);
        b.ff2 = linear(p + ".ff.ff.2", kDim, kFF);
        w.blocks.push_back(std::move(b));
    }
    w.norm_out = linear("norm_out.linear", 2 * kDim, kDim);
    w.proj_out = linear("proj_out", kMel, kDim);
    w.store->upload();
    source.release_storage();
    return w;
}

}  // namespace engine::models::f5_tts
