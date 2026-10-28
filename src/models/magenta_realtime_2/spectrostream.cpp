#include "engine/models/magenta_realtime_2/spectrostream.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/fft.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/primitive_modules.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <complex>
#include <map>
#include <unordered_map>

namespace engine::models::magenta_realtime_2 {
namespace {

struct ContextDeleter {
    void operator()(ggml_context * context) const { if (context) ggml_free(context); }
};

struct CodecWeights {
    core::BackendWeightStore store;
    std::unordered_map<std::string, core::TensorValue> tensors;

    CodecWeights(core::ExecutionContext & execution, const MagentaAssets & assets)
        : store(execution.backend(), execution.backend_type(), "magenta_realtime_2.codec.weights", 256 * 1024) {
        for (const auto & tensor : assets.weights->tensors()) {
            if (tensor.name.rfind("codec.", 0) != 0) continue;
            auto shape = tensor.shape;
            if (tensor.name == "codec.quantizer.weight") shape = {12288, 256};
            tensors.emplace(tensor.name, shape == tensor.shape ? store.load_f32_tensor(*assets.weights, tensor.name, shape) :
                store.load_tensor_as_shape(*assets.weights, tensor.name, engine::assets::TensorStorageType::F32,
                                          tensor.shape, core::TensorShape::from_dims({12288, 256})));
        }
        store.upload();
    }
};

class CodecGraph {
public:
    CodecGraph(core::ExecutionContext & execution, const CodecWeights & weights, int64_t frames)
        : execution_(execution), weights_(weights), context_(ggml_init({4 * 1024 * 1024, nullptr, true})),
          input_context_(ggml_init({128 * 1024, nullptr, true})) {
        if (!context_ || !input_context_) throw std::runtime_error("Cannot allocate SpectroStream graph metadata");
        auto * c = context_.get();
        ctx_ = {c, "magenta_realtime_2.codec", execution.backend_type()};
        core::ModuleBuildContext inputs{input_context_.get(), "magenta_realtime_2.codec.inputs", execution.backend_type()};
        const auto input = core::make_tensor(inputs, GGML_TYPE_I32, core::TensorShape::from_dims({frames * 12}));
        input_ = input.tensor;
        ggml_set_input(input_);
        auto x = modules::CodebookLookupModule({12288, 256}).build(ctx_, input, weights_.tensors.at("codec.quantizer.weight"));
        x = modules::ReshapeModule({core::TensorShape::from_dims({frames, 12, 256})}).build(ctx_, x);
        x = modules::ReduceSumModule({1}).build(ctx_, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({frames, 256})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
        x = modules::TransposeModule({{1, 0, 2, 3}, 2}).build(ctx_, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({1, 256, frames, 1})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
        auto direct = conv(x, "input_layer.conv1x1_first");
        auto branch = conv(x, "input_layer.shortcut.conv1x1_b1");
        branch = conv(modules::EluModule().build(ctx_, branch), "input_layer.shortcut.conv1x1_b2");
        x = modules::AddModule().build(ctx_, direct, branch);
        x = modules::TransposeModule({{0, 3, 2, 1}, 4}).build(ctx_, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({frames, 5, 512})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
        x = modules::TransposeModule({{2, 0, 1, 3}, 3}).build(ctx_, x);
        x = modules::ReshapeModule({core::TensorShape::from_dims({1, 512, frames, 5})}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
        auto residual = x;
        x = conv(modules::EluModule().build(ctx_, x), "input_layers_residual_unit.conv2d_3x3_a");
        x = modules::AddModule().build(ctx_, residual, conv(modules::EluModule().build(ctx_, x), "input_layers_residual_unit.conv2d_3x3"));
        const int time_strides[] = {2, 2, 1, 1, 1, 1, 1};
        const int freq_strides[] = {1, 2, 2, 2, 3, 2, 2};
        for (int i = 0; i < 7; ++i) {
            const auto prefix = "decoder_" + std::to_string(i);
            residual = x;
            const auto shortcut = prefix + ".shortcut.conv1x1.weight";
            if (weights_.tensors.count("codec.decoder." + shortcut)) residual = conv(residual, prefix + ".shortcut.conv1x1");
            residual = modules::NearestUpsample2dModule({residual.shape.dims[2] * time_strides[i],
                residual.shape.dims[3] * freq_strides[i]}).build(ctx_, residual);
            const std::string transpose = prefix + ".conv2dtranspose_" + std::to_string(std::max(3, time_strides[i] * 2)) + "x" +
                                          std::to_string(std::max(3, freq_strides[i] * 2));
            x = conv(modules::EluModule().build(ctx_, x), transpose, time_strides[i], freq_strides[i]);
            x = conv(modules::EluModule().build(ctx_, x), prefix + ".conv2d_3x3");
            x = modules::AddModule().build(ctx_, residual, x);
            if (i == 0) x = modules::ReshapeModule({core::TensorShape::from_dims({2, x.shape.dims[1] / 2,
                x.shape.dims[2], x.shape.dims[3]})}).build(ctx_, x);
        }
        output_ = conv(modules::EluModule().build(ctx_, x), "output_layer.base_conv_last").tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(c, 8192, false);
        ggml_build_forward_expand(graph_, output_);
        for (const auto & state : states_) {
            ggml_set_output(state.second);
            ggml_build_forward_expand(graph_, state.second);
        }
        // Spectral decoder activations require the existing model-local F32 policy.
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            auto * node = ggml_graph_node(graph_, i);
            if (node->op != GGML_OP_MUL_MAT) continue;
            ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            if (execution.backend_type() == core::BackendType::Vulkan)
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_VULKAN_F32_INPUTS);
            // Metal's default matmul narrows operands to half; codec activations can overflow it.
            if (execution.backend_type() == core::BackendType::Metal)
                ggml_mul_mat_set_lowering(node, GGML_MUL_MAT_LOWERING_METAL_F32_INPUTS);
        }
        input_buffer_ = ggml_backend_alloc_ctx_tensors(input_context_.get(), execution.backend());
        if (!input_buffer_) throw std::runtime_error("Cannot allocate SpectroStream persistent inputs");
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend()));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("Cannot allocate SpectroStream graph");
        core::validate_backend_graph_supported(execution.backend(), graph_, "SpectroStream");
        host_plan_ = core::create_backend_graph_plan_if_host(execution.backend(), graph_);
        debug::timing_log_context_reservation("magenta_realtime_2.codec", c);
        debug::timing_log_scalar("magenta_realtime_2.codec.graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator_, 0) / (1024.0 * 1024.0));
        debug::timing_log_scalar("magenta_realtime_2.codec.state.buffer_mb", ggml_backend_buffer_get_size(input_buffer_) / (1024.0 * 1024.0));
    }

    ~CodecGraph() {
        core::free_backend_graph_plan(execution_.backend(), host_plan_);
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        if (allocator_) ggml_gallocr_free(allocator_);
        if (input_buffer_) ggml_backend_buffer_free(input_buffer_);
    }

    void reset() {
        for (const auto & state : states_) ggml_backend_tensor_memset(state.first, 0, 0, ggml_nbytes(state.first));
    }

    void import_state(const CodecGraph & source) {
        if (states_.size() != source.states_.size()) throw std::runtime_error("SpectroStream state layout mismatch");
        for (size_t i = 0; i < states_.size(); ++i) {
            if (ggml_nbytes(states_[i].first) != ggml_nbytes(source.states_[i].second))
                throw std::runtime_error("SpectroStream state capacity mismatch");
            ggml_backend_tensor_copy_async(execution_.backend(), execution_.backend(), source.states_[i].second, states_[i].first);
        }
    }

    std::vector<float> run(const std::vector<std::array<int32_t, 12>> & codes) {
        std::vector<int32_t> unique;
        unique.reserve(codes.size() * 12);
        for (const auto & frame : codes) {
            for (int i = 0; i < 12; ++i) {
                if (frame[static_cast<size_t>(i)] < 0 || frame[static_cast<size_t>(i)] >= 1024)
                    throw std::runtime_error("SpectroStream code is outside its codebook");
                unique.push_back(i * 1024 + frame[static_cast<size_t>(i)]);
            }
        }
        ggml_backend_tensor_set(input_, unique.data(), 0, unique.size() * sizeof(int32_t));
        if (core::compute_backend_graph(execution_.backend(), graph_, host_plan_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SpectroStream graph execution failed");
        return core::read_tensor_f32(output_);
    }

private:
    core::TensorValue conv(core::TensorValue x, const std::string & name, int time_stride = 1, int freq_stride = 1) {
        const auto & weight = weights_.tensors.at("codec.decoder." + name + ".weight");
        const auto & bias = weights_.tensors.at("codec.decoder." + name + ".bias");
        const int64_t kh = weight.shape.dims[2], kw = weight.shape.dims[3];
        const int64_t frames = x.shape.dims[2], history = (kh - 1 + time_stride - 1) / time_stride;
        if (history) {
            core::ModuleBuildContext inputs{input_context_.get(), "magenta_realtime_2.codec.state", execution_.backend_type()};
            const auto state = core::make_tensor(inputs, GGML_TYPE_F32,
                core::TensorShape::from_dims({x.shape.dims[0], x.shape.dims[1], history, x.shape.dims[3]}));
            ggml_set_input(state.tensor);
            x = modules::ConcatModule({2}).build(ctx_, state, x);
            auto next = modules::SliceModule({2, x.shape.dims[2] - history, history}).build(ctx_, x);
            next = modules::ReshapeModule({next.shape}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, next));
            states_.emplace_back(state.tensor, next.tensor);
        }
        for (const auto & upsample : {std::make_pair(freq_stride, false), std::make_pair(time_stride, true)}) {
            const int rate = upsample.first;
            const bool time = upsample.second;
            if (rate == 1) continue;
            if (time) {
                x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, x);
                x = modules::ReshapeModule({x.shape}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
            }
            const auto batch = x.shape.dims[0], channels = x.shape.dims[1];
            const auto second = x.shape.dims[2], first = x.shape.dims[3];
            x = modules::ReshapeModule({core::TensorShape::from_dims({batch * channels, second, first, 1})}).build(ctx_, x);
            x = modules::Pad2dModule({0, rate - 1, 0, 0}).build(ctx_, x);
            x = modules::ReshapeModule({core::TensorShape::from_dims({batch, channels, second, first * rate})}).build(ctx_, x);
            if (time) {
                x = modules::TransposeModule({{0, 1, 3, 2}, 4}).build(ctx_, x);
                x = modules::ReshapeModule({x.shape}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, x));
            }
        }
        const int left = freq_stride == 1 ? static_cast<int>((kw - 1) / 2) : static_cast<int>((kw + freq_stride - 1) / 2);
        const int right = static_cast<int>(kw - 1) - left;
        modules::Conv2dConfig config{weight.shape.dims[1], weight.shape.dims[0], kh, kw};
        auto output = modules::CausalConv2dModule({config, left, right, kh - 1, 0}).build(ctx_, x, {weight, bias});
        const int64_t length = frames * time_stride;
        output = modules::SliceModule({2, output.shape.dims[2] - length, length}).build(ctx_, output);
        return modules::ReshapeModule({output.shape}).build(ctx_, core::ensure_backend_addressable_layout(ctx_, output));
    }

    core::ExecutionContext & execution_;
    const CodecWeights & weights_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    std::unique_ptr<ggml_context, ContextDeleter> input_context_;
    ggml_backend_buffer_t input_buffer_ = nullptr;
    core::ModuleBuildContext ctx_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t host_plan_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> states_;
};

}  // namespace

class SpectroStreamRuntime::Impl {
public:
    Impl(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets)
        : execution(execution), assets(std::move(assets)), weights(execution, *this->assets),
          fft(audio::get_real_fft_plan(960)), spectrum(481 * 2), inverse(960 * 2) {
        audio::STFTConfig config{960, 480, 960, false, audio::STFTPadMode::Constant, audio::STFTFamily::PeriodicF32};
        window = audio::get_cached_stft_window(config);
        const auto forward = window;
        for (size_t i = 0; i < window.size(); ++i) {
            const float a = forward[i], b = forward[(i + 480) % 960];
            window[i] /= a * a + b * b;
        }
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const MagentaAssets> assets;
    CodecWeights weights;
    std::map<int64_t, std::unique_ptr<CodecGraph>> graphs;
    CodecGraph * previous = nullptr;
    std::shared_ptr<audio::RealFFTPlan> fft;
    std::vector<std::complex<float>> spectrum;
    std::vector<float> inverse;
    std::vector<float> window;
    std::array<float, 960> overlap{};
    int64_t spectral_frames = 0;
};

SpectroStreamRuntime::SpectroStreamRuntime(core::ExecutionContext & execution, std::shared_ptr<const MagentaAssets> assets)
    : impl_(std::make_unique<Impl>(execution, std::move(assets))) {}
SpectroStreamRuntime::~SpectroStreamRuntime() = default;

void SpectroStreamRuntime::reset() {
    impl_->previous = nullptr;
    impl_->spectral_frames = 0;
    impl_->overlap.fill(0);
}

std::vector<float> SpectroStreamRuntime::decode(const std::vector<std::array<int32_t, 12>> & codes) {
    const auto start = std::chrono::steady_clock::now();
    if (codes.empty() || codes.size() > 4) throw std::runtime_error("SpectroStream expects one to four frames");
    auto & p = *impl_;
    auto & graph = p.graphs[static_cast<int64_t>(codes.size())];
    if (!graph) graph = std::make_unique<CodecGraph>(p.execution, p.weights, static_cast<int64_t>(codes.size()));
    if (p.previous) graph->import_state(*p.previous);
    else graph->reset();
    const auto spectra = graph->run(codes);
    p.previous = graph.get();
    const size_t frames = codes.size() * 4;
    std::vector<float> result(frames * 480 * 2);
    for (size_t t = 0; t < frames; ++t) {
        for (size_t channel = 0; channel < 2; ++channel) {
            for (size_t bin = 0; bin < 480; ++bin) {
                const auto offset = channel * 2 * frames * 480 + t * 480 + bin;
                p.spectrum[channel * 481 + bin] = p.spectral_frames < 4 ? std::complex<float>{} :
                    std::complex<float>{spectra[offset], spectra[offset + frames * 480]};
            }
            p.spectrum[channel * 481 + 480] = {};
        }
        p.fft->inverse({2, 960}, {static_cast<std::ptrdiff_t>(481 * sizeof(std::complex<float>)), static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))},
                       {static_cast<std::ptrdiff_t>(960 * sizeof(float)), static_cast<std::ptrdiff_t>(sizeof(float))},
                       1, p.spectrum.data(), p.inverse.data(), 1.0F / 960, 1);
        for (size_t i = 0; i < 480; ++i) {
            for (size_t channel = 0; channel < 2; ++channel) {
                result[(t * 480 + i) * 2 + channel] = p.inverse[channel * 960 + i] * p.window[i] + p.overlap[i * 2 + channel];
                p.overlap[i * 2 + channel] = p.inverse[channel * 960 + i + 480] * p.window[i + 480];
            }
        }
        ++p.spectral_frames;
    }
    debug::timing_log_scalar("magenta_realtime_2.codec.decode_ms", debug::elapsed_ms(start));
    return result;
}

}  // namespace engine::models::magenta_realtime_2
