#include "engine/models/kokoro_tts/neon_generator.h"
#include "engine/framework/neon/blocks.h"

#include <optional>
#include <cstring>
#include <stdexcept>

namespace kokoro_ggml {
namespace neon = engine::neon;
namespace {

template <typename Weights>
neon::ExecutionPlan::Value convolution(neon::ExecutionPlan &plan, neon::ExecutionPlan::Value input,
    const Weights &w) {
    if (w.groups != 1) throw std::invalid_argument("Kokoro generator convolution requires groups=1");
    return plan.add(neon::Conv1d({int(w.in_channels), int(w.out_channels), int(w.kernel), int(w.stride),
        int(w.dilation), int(w.padding), int(w.padding)}, engine::core::read_tensor_float(w.weight.tensor),
        w.use_bias ? engine::core::read_tensor_float(w.bias->tensor) : std::vector<float>{}), {input});
}

neon::ExecutionPlan::Value residual(neon::ExecutionPlan &plan, neon::ExecutionPlan::Value input,
    neon::ExecutionPlan::Value style, const KokoroWeights::GeneratorResBlockWeights &w) {
    const auto count = w.convs1.size();
    if (!count || w.convs2.size() != count || w.adain1.size() != count || w.adain2.size() != count ||
        w.alpha1.size() != count || w.alpha2.size() != count)
        throw std::invalid_argument("Kokoro NEON residual weights inconsistent");
    auto normalize = [&](neon::ExecutionPlan::Value x, const KokoroWeights::AdaIn1dWeights &norm) {
        const auto &fc = norm.fc;
        std::optional<neon::ExecutionPlan::Value> affine;
        auto bias = fc.use_bias ? engine::core::read_tensor_float(fc.bias->tensor) : std::vector<float>{};
        if (fc.weight.type == GGML_TYPE_Q8_0) {
            // Translate GGML storage at the model boundary; the library's
            // quantized operator keeps the original activation quantization.
            const auto raw = engine::core::read_tensor_bytes(fc.weight.tensor);
            const size_t blocks = size_t(fc.in_features) * fc.out_features / 32;
            if (fc.in_features % 32 || ggml_type_size(GGML_TYPE_Q8_0) != 34 || raw.size() != blocks * 34)
                throw std::invalid_argument("Kokoro Q8 style affine layout invalid");
            neon::BlockInt8Weights packed;
            packed.values.resize(blocks * 32);
            packed.scales.resize(blocks);
            for (size_t block = 0; block < blocks; ++block) {
                ggml_fp16_t scale;
                std::memcpy(&scale, raw.data() + block * 34, sizeof(scale));
                packed.scales[block] = ggml_fp16_to_fp32(scale);
                std::memcpy(packed.values.data() + block * 32, raw.data() + block * 34 + 2, 32);
            }
            affine = plan.add(neon::QuantizedLinear(int(fc.in_features), int(fc.out_features),
                std::move(packed), std::move(bias)), {style});
        } else {
            affine = plan.add(neon::Linear(int(fc.in_features), int(fc.out_features),
                engine::assets::tensor_data_to_f32("Kokoro style affine",
                    {fc.weight.shape, fc.weight.type, engine::core::read_tensor_bytes(fc.weight.tensor)}), std::move(bias)), {style});
        }
        return plan.add(neon::AdaptiveInstanceNorm(norm.eps), {x, *affine});
    };
    for (size_t layer = 0; layer < count; ++layer) {
        auto x = normalize(input, w.adain1[layer]);
        x = plan.add(neon::Snake(engine::core::read_tensor_float(w.alpha1[layer].tensor)), {x});
        x = convolution(plan, x, w.convs1[layer]);
        x = normalize(x, w.adain2[layer]);
        x = plan.add(neon::Snake(engine::core::read_tensor_float(w.alpha2[layer].tensor)), {x});
        x = convolution(plan, x, w.convs2[layer]);
        input = plan.add(neon::ScaleAdd({1.0F}), {input, x});
    }
    return input;
}

} // namespace

KokoroNeonGenerator::KokoroNeonGenerator(const KokoroWeights::GeneratorWeights &w, int style_channels)
    : style_channels_(style_channels) {
    if (!neon::available()) throw std::runtime_error("Kokoro NEON generator requires an AArch64 NEON CPU");
    if (w.ups.empty() || w.noise_convs.size() != w.ups.size() || w.noise_res.size() != w.ups.size() ||
        w.resblocks.size() != w.ups.size() * 3)
        throw std::invalid_argument("Kokoro NEON generator stages inconsistent");
    auto latent = plan_.input(), conditioning = plan_.input(), style = plan_.input();
    auto x = latent;
    for (size_t stage = 0; stage < w.ups.size(); ++stage) {
        auto source = convolution(plan_, conditioning, w.noise_convs[stage]);
        source = residual(plan_, source, style, w.noise_res[stage]);
        x = plan_.add(neon::Activation(neon::ActivationKind::LeakyReLU, 0.1F), {x});
        const auto &up = w.ups[stage];
        if (up.groups != 1) throw std::invalid_argument("Kokoro generator upsampling requires groups=1");
        x = neon::build_conv_transpose1d(plan_, x,
            {int(up.in_channels), int(up.out_channels), int(up.kernel), int(up.stride), int(up.padding), int(up.output_padding)},
            engine::core::read_tensor_float(up.weight.tensor),
            up.use_bias ? engine::core::read_tensor_float(up.bias->tensor) : std::vector<float>{});
        if (stage + 1 == w.ups.size()) x = plan_.add(neon::ReflectPad1d(1, 0), {x});
        x = plan_.add(neon::ScaleAdd({1.0F}), {x, source});
        std::optional<neon::ExecutionPlan::Value> sum;
        for (size_t k = 0; k < 3; ++k) {
            auto block = residual(plan_, x, style, w.resblocks[stage * 3 + k]);
            sum = sum ? plan_.add(neon::ScaleAdd({1.0F}), {*sum, block}) : block;
        }
        x = plan_.add(neon::Scale(1.0F / 3.0F), {*sum});
    }
    x = plan_.add(neon::Activation(neon::ActivationKind::LeakyReLU, 0.01F), {x});
    plan_.set_output(convolution(plan_, x, w.conv_post));
}

const neon::Tensor &KokoroNeonGenerator::run(const std::vector<float> &latent, int frames,
    const std::vector<float> &conditioning, int conditioning_frames, const std::vector<float> &style, int threads) {
    return plan_.run({neon::TensorView(latent.data(), latent.size(), {512, frames}),
        neon::TensorView(conditioning.data(), conditioning.size(), {22, conditioning_frames}),
        neon::TensorView(style.data(), style.size(), {style_channels_, 1})}, threads);
}

} // namespace kokoro_ggml
