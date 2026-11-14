#include "engine/community_models/zonos2/speaker_encoder.h"

#include "engine/framework/assets/gguf_metadata.h"
#include "engine/framework/audio/mel_spectrogram_frontend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/prepared_graph.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conditioning_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/errors.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cfloat>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::zonos2 {
namespace {

constexpr int64_t kMelBins = 128;
constexpr int64_t kNfft = 1024;
constexpr int64_t kHop = 256;
constexpr int64_t kChannels = 512;
constexpr int64_t kScale = 8;
constexpr int64_t kSeChannels = 128;
constexpr int64_t kMfaChannels = 1536;
constexpr int64_t kAttentionChannels = 128;
constexpr int kGraphNodes = 4096;

using core::TensorValue;

struct ConvWeights {
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t kernel = 0;
    int64_t dilation = 1;
    modules::Conv1dWeights weights;
};

struct SERes2NetWeights {
    ConvWeights tdnn1, tdnn2, se_conv1, se_conv2;
    std::vector<ConvWeights> res2net;
};

TensorValue shaped(ggml_tensor * x, int64_t channels, int64_t frames) {
    return core::wrap_tensor(x, core::TensorShape::from_dims({1, channels, frames}), GGML_TYPE_F32);
}

}  // namespace

class Zonos2SpeakerEncoder::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : source_(std::move(source)), execution_(execution) {
        load_weights();
        audio::MelSpectrogramFrontendConfig mel;
        mel.sample_rate = kZonos2SpeakerSampleRate;
        mel.n_fft = kNfft;
        mel.hop_length = kHop;
        mel.win_length = kNfft;
        mel.n_mels = kMelBins;
        mel.mel_fmin = 0.0f;
        mel.mel_fmax = 12000.0f;
        frontend_ = audio::get_cached_mel_spectrogram_frontend(mel);
    }

    std::vector<float> embed(const std::vector<float> & mono) const {
        if (mono.size() < static_cast<size_t>(kNfft)) {
            throw runtime::InvalidRequestError("zonos2: reference audio is shorter than 1024 samples at 24 kHz");
        }
        const auto mel = frontend_->extract_mono(mono, static_cast<size_t>(std::max(1, execution_.config().threads)));
        if (mel.mel_bins != kMelBins || mel.frames <= 0) throw std::runtime_error("zonos2: speaker frontend produced no frames");
        const int64_t frames = mel.frames;
        core::PreparedGraph prepared(execution_, core::PreparedGraphConfig{
            "zonos2.speaker_encoder",
            ggml_graph_overhead_custom(kGraphNodes, false) + kGraphNodes * ggml_tensor_overhead(), kGraphNodes,
            4 * ggml_tensor_overhead()});
        auto * input = ggml_new_tensor_3d(prepared.persistent_build_context().ggml, GGML_TYPE_F32, frames, kMelBins, 1);
        ggml_set_input(input);
        prepared.allocate_persistent();
        auto & ctx = prepared.build_context();
        auto * output = build(ctx, shaped(input, kMelBins, frames)).tensor;
        ggml_set_output(output);
        ggml_build_forward_expand(prepared.graph(), output);
        prepared.allocate();
        ggml_backend_tensor_set(input, mel.values.data(), 0, mel.values.size() * sizeof(float));
        prepared.compute();
        std::vector<float> embedding(static_cast<size_t>(embedding_dim_));
        ggml_backend_tensor_get(output, embedding.data(), 0, embedding.size() * sizeof(float));
        return embedding;
    }

private:
    // Conv weights are stored ne [IC, OC, K], which TensorSource reports as [K, OC, IC] ([OC, IC] for K = 1: GGUF drops
    // trailing unit dimensions); the convolutions want [OC, IC, K]. F16 values load losslessly as F32 so the
    // convolutions run in F32.
    ConvWeights load_conv(const std::string & prefix, int64_t out_channels, int64_t in_channels, int64_t kernel,
                          int64_t dilation = 1) {
        const auto & s = *source_;
        const std::string weight = prefix + ".weight";
        const std::string bias = prefix + ".bias";
        consumed_.insert(weight);
        consumed_.insert(bias);
        ConvWeights conv{in_channels, out_channels, kernel, dilation, {}};
        if (kernel == 1) {
            conv.weights.weight = store_->load_tensor_as_shape(s, weight, assets::TensorStorageType::F32, {out_channels, in_channels},
                                                               core::TensorShape::from_dims({out_channels, in_channels, 1}));
        } else {
            const auto values = s.require_f32(weight, {kernel, out_channels, in_channels});
            std::vector<float> permuted(values.size());
            for (int64_t k = 0; k < kernel; ++k) {
                for (int64_t oc = 0; oc < out_channels; ++oc) {
                    for (int64_t ic = 0; ic < in_channels; ++ic) {
                        permuted[static_cast<size_t>((oc * in_channels + ic) * kernel + k)] =
                            values[static_cast<size_t>((k * out_channels + oc) * in_channels + ic)];
                    }
                }
            }
            conv.weights.weight = store_->make_f32(core::TensorShape::from_dims({out_channels, in_channels, kernel}),
                                                   std::move(permuted));
        }
        conv.weights.bias = store_->load_f32_tensor(s, bias, {out_channels});
        return conv;
    }

    void load_weights() {
        const assets::GgufMetadata meta(source_->source_path());
        const auto kernels = meta.require_int_array("ecapa-tdnn.enc_kernel_sizes");
        const auto dilations = meta.require_int_array("ecapa-tdnn.enc_dilations");
        if (kernels.size() != 5 || dilations.size() != 5 || kernels[0] <= 0 || kernels[0] % 2 == 0) {
            throw std::runtime_error("zonos2: speaker encoder GGUF: unexpected enc_kernel_sizes / enc_dilations");
        }
        if (meta.require_int("ecapa-tdnn.res2net_scale") != kScale || meta.require_int("ecapa-tdnn.se_channels") != kSeChannels ||
            meta.require_int("ecapa-tdnn.attention_channels") != kAttentionChannels) {
            throw std::runtime_error("zonos2: speaker encoder GGUF: unexpected ECAPA layout");
        }
        embedding_dim_ = meta.require_int("ecapa-tdnn.enc_dim", 1, 65536);
        store_ = std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
                                                            "zonos2.speaker_encoder.weights", 4 * 1024 * 1024);
        block0_ = load_conv("blocks.0.conv", kChannels, kMelBins, kernels[0]);
        const int64_t width = kChannels / kScale;
        for (int block = 1; block <= 3; ++block) {
            const std::string p = "blocks." + std::to_string(block);
            const int64_t dilation = dilations[static_cast<size_t>(block)];
            SERes2NetWeights layer;
            layer.tdnn1 = load_conv(p + ".tdnn1.conv", kChannels, kChannels, 1);
            for (int64_t i = 0; i < kScale - 1; ++i) {
                layer.res2net.push_back(load_conv(p + ".res2net_block.blocks." + std::to_string(i) + ".conv", width, width,
                                                  kernels[static_cast<size_t>(block)], dilation));
            }
            layer.tdnn2 = load_conv(p + ".tdnn2.conv", kChannels, kChannels, 1);
            layer.se_conv1 = load_conv(p + ".se_block.conv1", kSeChannels, kChannels, 1);
            layer.se_conv2 = load_conv(p + ".se_block.conv2", kChannels, kSeChannels, 1);
            blocks_.push_back(std::move(layer));
        }
        mfa_ = load_conv("mfa.conv", kMfaChannels, kMfaChannels, 1);
        asp_tdnn_ = load_conv("asp.tdnn.conv", kAttentionChannels, 3 * kMfaChannels, 1);
        asp_conv_ = load_conv("asp.conv", kMfaChannels, kAttentionChannels, 1);
        fc_ = load_conv("fc", embedding_dim_, 2 * kMfaChannels, 1);
        // The in-graph mel front end of Zyphra's encoder; audio::MelSpectrogramFrontend computes the same features.
        consumed_.insert("mel_fb");
        consumed_.insert("mel_window");
        for (const auto & tensor : source_->tensors()) {
            if (!consumed_.count(tensor.name)) {
                throw std::runtime_error("zonos2: speaker encoder GGUF: unmapped tensor '" + tensor.name + "'");
            }
        }
        store_->upload();
        source_->release_storage();
    }

    // Reflect-padded 'same' convolution.
    TensorValue conv1d(core::ModuleBuildContext & ctx, TensorValue x, const ConvWeights & conv) const {
        const int64_t padding = conv.dilation * (conv.kernel - 1) / 2;
        if (padding > 0) x = modules::ReflectPad1dModule({padding, padding}).build(ctx, x);
        return modules::Conv1dModule({conv.in_channels, conv.out_channels, conv.kernel, 1, 0, static_cast<int>(conv.dilation), true})
            .build(ctx, x, conv.weights);
    }

    TensorValue tdnn(core::ModuleBuildContext & ctx, const TensorValue & x, const ConvWeights & conv) const {
        return modules::ReluModule{}.build(ctx, conv1d(ctx, x, conv));
    }

    TensorValue se_res2net(core::ModuleBuildContext & ctx, const TensorValue & input, const SERes2NetWeights & block) const {
        auto y = tdnn(ctx, input, block.tdnn1);
        const int64_t width = kChannels / kScale;
        TensorValue merged, previous;
        for (int64_t i = 0; i < kScale; ++i) {
            auto chunk = modules::SliceModule({1, i * width, width}).build(ctx, y);
            TensorValue out;
            if (i == 0) {
                out = chunk;
            } else if (i == 1) {
                out = tdnn(ctx, chunk, block.res2net[0]);
            } else {
                out = tdnn(ctx, modules::ResidualAddModule{}.build(ctx, chunk, previous), block.res2net[static_cast<size_t>(i - 1)]);
            }
            previous = out;
            merged = merged.valid() ? modules::ConcatModule({1}).build(ctx, merged, out) : out;
        }
        y = tdnn(ctx, merged, block.tdnn2);
        y = modules::SqueezeExcite1dModule({kChannels, kSeChannels, true})
                .build(ctx, y, {block.se_conv1.weights, block.se_conv2.weights});
        return modules::ResidualAddModule{}.build(ctx, y, input);
    }

    // std = sqrt(clamp(var, 1e-12)), as Zyphra's encoder; the pooled statistics see exactly the input frames.
    TensorValue attentive_statistics_pool(core::ModuleBuildContext & ctx, const TensorValue & x) const {
        auto * g = ctx.ggml;
        auto stddev = [&](const TensorValue & variance) {
            return core::wrap_tensor(ggml_sqrt(g, ggml_clamp(g, variance.tensor, 1e-12f, FLT_MAX)), variance.shape, GGML_TYPE_F32);
        };
        auto mean = modules::ReduceMeanModule({2}).build(ctx, x);
        auto mean_rep = modules::RepeatModule({x.shape}).build(ctx, mean);
        auto centered = core::wrap_tensor(ggml_sub(g, x.tensor, mean_rep.tensor), x.shape, GGML_TYPE_F32);
        auto variance = modules::ReduceMeanModule({2}).build(ctx, modules::MulModule{}.build(ctx, centered, centered));
        auto std_rep = modules::RepeatModule({x.shape}).build(ctx, stddev(variance));
        auto attention = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, x, mean_rep), std_rep);
        attention = modules::TanhModule{}.build(ctx, tdnn(ctx, attention, asp_tdnn_));
        auto weights = modules::SoftmaxModule{}.build(ctx, conv1d(ctx, attention, asp_conv_));
        auto mean_att = modules::ReduceSumModule({2}).build(ctx, modules::MulModule{}.build(ctx, x, weights));
        auto mean_att_rep = modules::RepeatModule({x.shape}).build(ctx, mean_att);
        auto diff = core::wrap_tensor(ggml_sub(g, x.tensor, mean_att_rep.tensor), x.shape, GGML_TYPE_F32);
        auto weighted_var = modules::MulModule{}.build(ctx, modules::MulModule{}.build(ctx, diff, diff), weights);
        auto var_att = modules::ReduceSumModule({2}).build(ctx, weighted_var);
        return modules::ConcatModule({1}).build(ctx, mean_att, stddev(var_att));
    }

    TensorValue build(core::ModuleBuildContext & ctx, const TensorValue & mel) const {
        auto x = tdnn(ctx, mel, block0_);
        std::vector<TensorValue> hidden;
        for (const auto & block : blocks_) {
            x = se_res2net(ctx, x, block);
            hidden.push_back(x);
        }
        x = modules::ConcatModule({1}).build(ctx, modules::ConcatModule({1}).build(ctx, hidden[0], hidden[1]), hidden[2]);
        x = tdnn(ctx, x, mfa_);
        x = attentive_statistics_pool(ctx, x);
        return conv1d(ctx, x, fc_);
    }

    std::shared_ptr<const assets::TensorSource> source_;
    core::ExecutionContext & execution_;
    std::unique_ptr<core::BackendWeightStore> store_;
    std::shared_ptr<const audio::MelSpectrogramFrontend> frontend_;
    std::set<std::string> consumed_;
    int64_t embedding_dim_ = 0;
    ConvWeights block0_, mfa_, asp_tdnn_, asp_conv_, fc_;
    std::vector<SERes2NetWeights> blocks_;
};

Zonos2SpeakerEncoder::Zonos2SpeakerEncoder(std::shared_ptr<const assets::TensorSource> spk, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(spk), execution)) {}

Zonos2SpeakerEncoder::~Zonos2SpeakerEncoder() = default;

std::vector<float> Zonos2SpeakerEncoder::embed(const std::vector<float> & mono_24k) const { return impl_->embed(mono_24k); }

}  // namespace engine::models::zonos2
