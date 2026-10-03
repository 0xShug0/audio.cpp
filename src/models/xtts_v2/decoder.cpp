#include "engine/models/xtts_v2/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::xtts_v2 {
namespace {
namespace binding = engine::modules::binding;
namespace modules = engine::modules;

struct ResBlockWeights {
    std::array<modules::Conv1dWeights, 3> first;
    std::array<modules::Conv1dWeights, 3> second;
    int64_t channels = 0;
    int64_t kernel = 0;
};
struct ContextDeleter { void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); } };

core::TensorValue resblock(core::ModuleBuildContext & ctx, core::TensorValue x, const ResBlockWeights & weights) {
    constexpr std::array<int, 3> dilation{1, 3, 5};
    for (size_t i = 0; i < 3; ++i) {
        auto y = modules::LeakyReluModule({0.1F}).build(ctx, x);
        y = modules::Conv1dModule({weights.channels, weights.channels, weights.kernel, 1,
            static_cast<int>((weights.kernel * dilation[i] - dilation[i]) / 2), dilation[i], true}).build(ctx, y, weights.first[i]);
        y = modules::LeakyReluModule({0.1F}).build(ctx, y);
        y = modules::Conv1dModule({weights.channels, weights.channels, weights.kernel, 1,
            static_cast<int>((weights.kernel - 1) / 2), 1, true}).build(ctx, y, weights.second[i]);
        x = modules::AddModule{}.build(ctx, x, y);
    }
    return x;
}

core::TensorValue upsample(core::ModuleBuildContext & ctx, const core::TensorValue & input,
                           int64_t in_channels, int64_t out_channels, int64_t kernel, int stride,
                           const modules::ConvTranspose1dWeights & weights) {
    const int padding = static_cast<int>((kernel - stride) / 2);
    auto full = modules::ConvTranspose1dModule({in_channels, out_channels, kernel, stride, 0, 1, true})
                    .build(ctx, input, weights);
    const int64_t cropped = (input.shape.dims[2] - 1) * stride - 2 * padding + kernel;
    return modules::SliceModule({2, padding, cropped}).build(ctx, full);
}

std::vector<float> interpolate_frame_major(const std::vector<float> & input, int64_t input_frames,
                                            int64_t output_frames, double scale) {
    std::vector<float> output(static_cast<size_t>(output_frames * 1024));
    for (int64_t t = 0; t < output_frames; ++t) {
        const double source = (static_cast<double>(t) + 0.5) / scale - 0.5;
        const int64_t left = std::max<int64_t>(0, std::min<int64_t>(input_frames - 1, static_cast<int64_t>(std::floor(source))));
        const int64_t right = std::max<int64_t>(0, std::min<int64_t>(input_frames - 1, left + 1));
        const float fraction = static_cast<float>(std::max(0.0, std::min(1.0, source - static_cast<double>(left))));
        for (int64_t c = 0; c < 1024; ++c) {
            const float a = input[static_cast<size_t>(left * 1024 + c)];
            const float b = input[static_cast<size_t>(right * 1024 + c)];
            output[static_cast<size_t>(t * 1024 + c)] = a + (b - a) * fraction;
        }
    }
    return output;
}
}  // namespace

struct XttsV2DecoderRuntime::Weights {
    std::shared_ptr<core::BackendWeightStore> store;
    modules::Conv1dWeights pre;
    modules::Conv1dWeights condition;
    std::array<modules::ConvTranspose1dWeights, 4> up;
    std::array<modules::Conv1dWeights, 4> stage_condition;
    std::array<ResBlockWeights, 12> residual;
    modules::Conv1dWeights post;
};

class XttsV2DecoderRuntime::Graph {
public:
    Graph(core::ExecutionContext & execution, std::shared_ptr<const Weights> weights, int64_t frames, size_t arena)
        : execution_(execution), weights_(std::move(weights)), frames_(frames) {
        const int64_t ar_interpolated = frames_ * 4;
        const int64_t interpolated = ar_interpolated * 24000 / 22050;
        ctx_.reset(ggml_init({arena, nullptr, true}));
        input_ctx_.reset(ggml_init({8U * 1024U * 1024U, nullptr, true}));
        if (!ctx_ || !input_ctx_) throw std::runtime_error("failed to initialize XTTS v2 decoder graph");
        core::ModuleBuildContext ctx{ctx_.get(), "xtts_v2.decoder", execution_.backend_type()};
        core::ModuleBuildContext ictx{input_ctx_.get(), "xtts_v2.decoder.inputs", execution_.backend_type()};
        latent_ = core::make_tensor(ictx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 1024, interpolated})).tensor;
        speaker_ = core::make_tensor(ictx, GGML_TYPE_F32, core::TensorShape::from_dims({1, 512, 1})).tensor;
        ggml_set_input(latent_); ggml_set_input(speaker_);
        auto x = core::wrap_tensor(latent_, core::TensorShape::from_dims({1, 1024, interpolated}), GGML_TYPE_F32);
        x = modules::Conv1dModule({1024, 512, 7, 1, 3, 1, true}).build(ctx, x, weights_->pre);
        auto speaker = core::wrap_tensor(speaker_, core::TensorShape::from_dims({1, 512, 1}), GGML_TYPE_F32);
        auto cond = modules::Conv1dModule({512, 512, 1, 1, 0, 1, true}).build(ctx, speaker, weights_->condition);
        x = modules::AddModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, cond));
        constexpr std::array<int, 4> strides{8, 8, 2, 2};
        constexpr std::array<int64_t, 4> kernels{16, 16, 4, 4};
        constexpr std::array<int64_t, 4> channels{256, 128, 64, 32};
        int64_t in_channels = 512;
        for (size_t stage = 0; stage < 4; ++stage) {
            x = modules::LeakyReluModule({0.1F}).build(ctx, x);
            x = upsample(ctx, x, in_channels, channels[stage], kernels[stage], strides[stage], weights_->up[stage]);
            auto stage_cond = modules::Conv1dModule({512, channels[stage], 1, 1, 0, 1, true}).build(ctx, speaker, weights_->stage_condition[stage]);
            x = modules::AddModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, stage_cond));
            auto sum = resblock(ctx, x, weights_->residual[stage * 3]);
            sum = modules::AddModule{}.build(ctx, sum, resblock(ctx, x, weights_->residual[stage * 3 + 1]));
            sum = modules::AddModule{}.build(ctx, sum, resblock(ctx, x, weights_->residual[stage * 3 + 2]));
            x = core::wrap_tensor(ggml_scale(ctx.ggml, sum.tensor, 1.0F / 3.0F), sum.shape, GGML_TYPE_F32);
            in_channels = channels[stage];
        }
        x = modules::LeakyReluModule({0.01F}).build(ctx, x);
        x = modules::Conv1dModule({32, 1, 7, 1, 3, 1, false}).build(ctx, x, weights_->post);
        x = modules::TanhModule{}.build(ctx, x);
        output_frames_ = x.shape.dims[2];
        output_ = core::ensure_backend_addressable_layout(ctx, x).tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 131072, false); ggml_build_forward_expand(graph_, output_);
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_ctx_.get(), execution_.backend());
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!input_buffer_ || !allocator_ || !ggml_gallocr_reserve(allocator_, graph_) || !ggml_gallocr_alloc_graph(allocator_, graph_))
            throw std::runtime_error("failed to allocate XTTS v2 decoder graph");
    }
    ~Graph() {
        if (graph_) core::release_backend_graph_resources(execution_.backend(), graph_);
        if (allocator_) ggml_gallocr_free(allocator_); if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }
    bool matches(int64_t frames) const noexcept { return frames == frames_; }
    std::vector<float> run(const std::vector<float> & latent, const std::vector<float> & speaker) {
        const int64_t interpolated = (frames_ * 4) * 24000 / 22050;
        if (static_cast<int64_t>(latent.size()) != interpolated * 1024 || speaker.size() != 512) throw std::runtime_error("XTTS v2 decoder input shape mismatch");
        ggml_backend_tensor_set(latent_, latent.data(), 0, latent.size() * sizeof(float));
        ggml_backend_tensor_set(speaker_, speaker.data(), 0, speaker.size() * sizeof(float));
        core::set_backend_threads(execution_.backend(), execution_.config().threads);
        if (core::compute_backend_graph(execution_.backend(), graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("XTTS v2 decoder compute failed");
        ggml_backend_synchronize(execution_.backend()); std::vector<float> output(static_cast<size_t>(output_frames_));
        ggml_backend_tensor_get(output_, output.data(), 0, output.size() * sizeof(float)); return output;
    }
private:
    core::ExecutionContext & execution_; std::shared_ptr<const Weights> weights_; int64_t frames_, output_frames_;
    std::unique_ptr<ggml_context, ContextDeleter> ctx_, input_ctx_; ggml_tensor * latent_ = nullptr, * speaker_ = nullptr, * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr; ggml_gallocr_t allocator_ = nullptr; ggml_backend_buffer_t input_buffer_ = nullptr;
};

XttsV2DecoderRuntime::XttsV2DecoderRuntime(const XttsV2Assets & assets, core::ExecutionContext & execution,
    size_t weight_context_bytes, size_t graph_context_bytes, assets::TensorStorageType type)
    : execution_(execution), graph_context_bytes_(graph_context_bytes) {
    auto out = std::make_shared<Weights>(); out->store = std::make_shared<core::BackendWeightStore>(
        execution.backend(), execution.backend_type(), "xtts_v2.decoder.weights", weight_context_bytes);
    const auto & source = *assets.decoder;
    out->pre = binding::conv1d_from_source(*out->store, source, "conv_pre", type, 512, 1024, 7, true);
    out->condition = binding::conv1d_from_source(*out->store, source, "cond_layer", type, 512, 512, 1, true);
    constexpr std::array<int64_t, 4> in_channels{512, 256, 128, 64};
    constexpr std::array<int64_t, 4> channels{256, 128, 64, 32};
    constexpr std::array<int64_t, 4> kernels{16, 16, 4, 4};
    for (size_t i = 0; i < 4; ++i) {
        out->up[i] = binding::conv_transpose1d_from_source(*out->store, source, "ups." + std::to_string(i), type, in_channels[i], channels[i], kernels[i], true);
        out->stage_condition[i] = binding::conv1d_from_source(*out->store, source, "conds." + std::to_string(i), type, channels[i], 512, 1, true);
    }
    constexpr std::array<int64_t, 3> rb_kernel{3, 7, 11}; constexpr std::array<int, 3> dilation{1, 3, 5};
    for (size_t stage = 0; stage < 4; ++stage) for (size_t branch = 0; branch < 3; ++branch) {
        const size_t index = stage * 3 + branch; auto & rb = out->residual[index]; rb.channels = channels[stage]; rb.kernel = rb_kernel[branch];
        for (size_t layer = 0; layer < 3; ++layer) {
            const std::string base = "resblocks." + std::to_string(index);
            rb.first[layer] = binding::conv1d_from_source(*out->store, source, base + ".convs1." + std::to_string(layer), type, rb.channels, rb.channels, rb.kernel, true);
            rb.second[layer] = binding::conv1d_from_source(*out->store, source, base + ".convs2." + std::to_string(layer), type, rb.channels, rb.channels, rb.kernel, true);
        }
    }
    out->post = binding::conv1d_from_source(*out->store, source, "conv_post", type, 1, 32, 7, false);
    out->store->upload(); weights_ = std::move(out);
}
XttsV2DecoderRuntime::~XttsV2DecoderRuntime() = default;
std::vector<float> XttsV2DecoderRuntime::decode(const std::vector<float> & latents, int64_t frames, const XttsV2SpeakerEmbedding & speaker) {
    if (frames <= 0) throw std::runtime_error("XTTS v2 decoder requires latent frames");
    if (!graph_ || !graph_->matches(frames)) graph_ = std::make_unique<Graph>(execution_, weights_, frames, graph_context_bytes_);
    auto stage1 = interpolate_frame_major(latents, frames, frames * 4, 4.0);
    const int64_t stage2_frames = (frames * 4) * 24000 / 22050;
    auto stage2 = interpolate_frame_major(stage1, frames * 4, stage2_frames, 24000.0 / 22050.0);
    // Convert [time, channels] to the channel-major [B,C,T] input expected by ggml.
    std::vector<float> channel_major(stage2.size());
    for (int64_t t = 0; t < stage2_frames; ++t) for (int64_t c = 0; c < 1024; ++c)
        channel_major[static_cast<size_t>(c * stage2_frames + t)] = stage2[static_cast<size_t>(t * 1024 + c)];
    return graph_->run(channel_major, speaker.values);
}

}  // namespace engine::models::xtts_v2
