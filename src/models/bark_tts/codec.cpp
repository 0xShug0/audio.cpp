#include "engine/models/bark_tts/codec.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/models/bark_tts/assets.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <array>
#include <stdexcept>

namespace engine::models::bark_tts {
namespace {

struct Conv { engine::modules::Conv1dWeights weights; int64_t in, out, kernel, dilation; };
struct Up { engine::modules::ConvTranspose1dWeights weights; int64_t in, out, kernel, stride; };
struct Residual { Conv first; Conv second; Conv shortcut; };
struct CodecWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::vector<engine::core::TensorValue> codebooks;
    Conv initial;
    engine::modules::LSTMStackWeights lstm;
    std::array<Up, 4> ups;
    std::array<Residual, 4> residuals;
    Conv final;
};
struct ContextDelete { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };

Conv load_conv(engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
               const std::string & prefix, int64_t in, int64_t out, int64_t kernel, int64_t dilation,
               engine::assets::TensorStorageType storage) {
    Conv result{{}, in, out, kernel, dilation};
    result.weights = engine::modules::binding::conv1d_from_source(store, source, prefix, storage,
                                                                  out, in, kernel, true);
    return result;
}

Up load_up(engine::core::BackendWeightStore & store, const engine::assets::TensorSource & source,
           const std::string & prefix, int64_t in, int64_t out, int64_t kernel, int64_t stride,
           engine::assets::TensorStorageType storage) {
    Up result{{}, in, out, kernel, stride};
    result.weights = engine::modules::binding::conv_transpose1d_from_source(store, source, prefix, storage,
                                                                            in, out, kernel, true);
    return result;
}

std::shared_ptr<const CodecWeights> load(const BarkAssets & assets, ggml_backend_t backend, engine::core::BackendType backend_type,
                                         engine::assets::TensorStorageType storage) {
    auto out = std::make_shared<CodecWeights>();
    out->store = std::make_shared<engine::core::BackendWeightStore>(backend, backend_type,
                                                                    "bark.codec.weights", 384ULL * 1024ULL * 1024ULL);
    const auto & source = *assets.weights;
    for (int i = 0; i < 8; ++i) out->codebooks.push_back(out->store->load_tensor(source,
        "codec_model.quantizer.layers." + std::to_string(i) + ".codebook.embed", storage, {1024, 128}));
    out->initial = load_conv(*out->store, source, "codec_model.decoder.layers.0.conv", 128, 512, 7, 1, storage);
    for (int i = 0; i < 2; ++i) {
        const std::string p = "codec_model.decoder.layers.1.lstm.";
        engine::modules::LSTMCellWeights cell;
        cell.weight_ih = out->store->load_tensor(source, p + "weight_ih_l" + std::to_string(i), storage, {2048, 512});
        cell.weight_hh = out->store->load_tensor(source, p + "weight_hh_l" + std::to_string(i), storage, {2048, 512});
        cell.bias_ih = out->store->load_tensor(source, p + "bias_ih_l" + std::to_string(i), engine::assets::TensorStorageType::F32, {2048});
        cell.bias_hh = out->store->load_tensor(source, p + "bias_hh_l" + std::to_string(i), engine::assets::TensorStorageType::F32, {2048});
        out->lstm.layers.push_back(std::move(cell));
    }
    const int layer[] = {3, 6, 9, 12};
    const int in[] = {512, 256, 128, 64};
    const int channel[] = {256, 128, 64, 32};
    const int ratio[] = {8, 5, 4, 2};
    const int residual_layer[] = {4, 7, 10, 13};
    for (int i = 0; i < 4; ++i) {
        const auto base = "codec_model.decoder.layers." + std::to_string(layer[i]) + ".conv";
        out->ups[static_cast<size_t>(i)] = load_up(*out->store, source, base, in[i], channel[i], ratio[i] * 2, ratio[i], storage);
        const auto r = "codec_model.decoder.layers." + std::to_string(residual_layer[i]);
        out->residuals[static_cast<size_t>(i)] = {
            load_conv(*out->store, source, r + ".block.1.conv", channel[i], channel[i] / 2, 3, 1, storage),
            load_conv(*out->store, source, r + ".block.3.conv", channel[i] / 2, channel[i], 1, 1, storage),
            load_conv(*out->store, source, r + ".shortcut.conv", channel[i], channel[i], 1, 1, storage),
        };
    }
    out->final = load_conv(*out->store, source, "codec_model.decoder.layers.15.conv", 32, 1, 7, 1, storage);
    out->store->upload();
    return out;
}

engine::core::TensorValue causal_conv(engine::core::ModuleBuildContext & ctx,
                                      const engine::core::TensorValue & input, const Conv & conv) {
    const int64_t effective = (conv.kernel - 1) * conv.dilation + 1;
    const auto contiguous = engine::core::ensure_backend_addressable_layout(ctx, input);
    auto padded = engine::modules::ReflectPad1dModule({effective - 1, 0}).build(ctx, contiguous);
    return engine::modules::Conv1dModule({conv.in, conv.out, conv.kernel, 1, 0,
                                          static_cast<int>(conv.dilation), true}).build(ctx, padded, conv.weights);
}

engine::core::TensorValue upsample(engine::core::ModuleBuildContext & ctx,
                                   const engine::core::TensorValue & input, const Up & up) {
    auto full = engine::modules::ConvTranspose1dModule({up.in, up.out, up.kernel,
        static_cast<int>(up.stride), 0, 1, true}).build(ctx, input, up.weights);
    const int64_t frames = full.shape.dims[2] - (up.kernel - up.stride);
    return engine::modules::SliceModule({2, 0, frames}).build(ctx, full);
}

engine::core::TensorValue elu(engine::core::ModuleBuildContext & ctx, const engine::core::TensorValue & input) {
    return engine::modules::EluModule{}.build(ctx, input);
}

engine::core::TensorValue residual(engine::core::ModuleBuildContext & ctx,
                                   const engine::core::TensorValue & input, const Residual & weights) {
    auto hidden = causal_conv(ctx, elu(ctx, input), weights.first);
    hidden = causal_conv(ctx, elu(ctx, hidden), weights.second);
    return engine::modules::AddModule{}.build(ctx, causal_conv(ctx, input, weights.shortcut), hidden);
}

}  // namespace

struct BarkCodecDecoder::Impl {
    std::shared_ptr<const BarkAssets> assets;
    engine::core::ExecutionContext & execution;
    std::shared_ptr<const CodecWeights> weights;
};

BarkCodecDecoder::BarkCodecDecoder(std::shared_ptr<const BarkAssets> assets,
                                   engine::core::ExecutionContext & execution,
                                   engine::assets::TensorStorageType storage)
    : impl_(std::make_unique<Impl>(Impl{assets, execution, load(*assets, execution.backend(), execution.backend_type(), storage)})) {}
BarkCodecDecoder::~BarkCodecDecoder() = default;

std::vector<float> BarkCodecDecoder::decode(const std::vector<std::vector<int32_t>> & codes) const {
    if (codes.size() != 8 || codes.front().empty()) throw std::runtime_error("Bark codec requires 8 non-empty codebooks");
    const int64_t frames = static_cast<int64_t>(codes.front().size());
    for (const auto & row : codes) if (static_cast<int64_t>(row.size()) != frames)
        throw std::runtime_error("Bark codec codebook lengths differ");
    std::unique_ptr<ggml_context, ContextDelete> context(ggml_init({256ULL * 1024ULL * 1024ULL, nullptr, true}));
    engine::core::ModuleBuildContext build{context.get(), "bark.codec", impl_->execution.backend_type()};
    engine::core::TensorValue quantized;
    std::vector<ggml_tensor *> inputs;
    for (size_t i = 0; i < 8; ++i) {
        auto * raw = ggml_new_tensor_1d(context.get(), GGML_TYPE_I32, frames);
        inputs.push_back(raw);
        auto ids = engine::core::wrap_tensor(raw, engine::core::TensorShape::from_dims({frames}), GGML_TYPE_I32);
        auto value = engine::modules::EmbeddingModule({1024, 128}).build(build, ids, impl_->weights->codebooks[i]);
        value = engine::core::reshape_tensor(build, value, engine::core::TensorShape::from_dims({1, frames, 128}));
        value = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, value);
        quantized = quantized.valid() ? engine::modules::AddModule{}.build(build, quantized, value) : value;
    }
    auto hidden = causal_conv(build, quantized, impl_->weights->initial);
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, hidden);
    hidden = engine::core::ensure_backend_addressable_layout(build, hidden);
    hidden = engine::core::reshape_tensor(build, hidden, engine::core::TensorShape::from_dims({frames, 512}));
    const auto lstm_residual = hidden;
    for (const auto & cell : impl_->weights->lstm.layers) {
        auto zero = engine::core::wrap_tensor(ggml_scale(context.get(), engine::modules::SliceModule({0, 0, 1}).build(build, hidden).tensor, 0.0F),
            engine::core::TensorShape::from_dims({1, 512}), GGML_TYPE_F32);
        hidden = engine::modules::LSTMSequenceModule({512, 512, false}).build(build, hidden, zero, zero, {cell}).sequence;
    }
    hidden = engine::modules::AddModule{}.build(build, hidden, lstm_residual);
    hidden = engine::core::reshape_tensor(build, hidden, engine::core::TensorShape::from_dims({1, frames, 512}));
    hidden = engine::modules::TransposeModule({{0, 2, 1, 3}, 3}).build(build, hidden);
    for (size_t i = 0; i < 4; ++i) hidden = residual(build, upsample(build, elu(build, hidden), impl_->weights->ups[i]), impl_->weights->residuals[i]);
    auto waveform = causal_conv(build, elu(build, hidden), impl_->weights->final);
    ggml_set_output(waveform.tensor);
    auto * graph = ggml_new_graph_custom(context.get(), 65536, false);
    ggml_build_forward_expand(graph, waveform.tensor);
    auto * buffer = ggml_backend_alloc_ctx_tensors(context.get(), impl_->execution.backend());
    if (!buffer) throw std::runtime_error("failed to allocate Bark codec graph");
    for (size_t i = 0; i < 8; ++i) ggml_backend_tensor_set(inputs[i], codes[i].data(), 0, codes[i].size() * sizeof(int32_t));
    engine::core::set_backend_threads(impl_->execution.backend(), impl_->execution.config().threads);
    const auto status = engine::core::compute_backend_graph(impl_->execution.backend(), graph);
    ggml_backend_synchronize(impl_->execution.backend());
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("Bark codec graph compute failed");
    std::vector<float> out(static_cast<size_t>(waveform.shape.dims[2]));
    ggml_backend_tensor_get(waveform.tensor, out.data(), 0, out.size() * sizeof(float));
    engine::core::release_backend_graph_resources(impl_->execution.backend(), graph);
    ggml_backend_buffer_free(buffer);
    return out;
}

}  // namespace engine::models::bark_tts
