#include "engine/models/sam_audio/codec.h"

#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/graph_optimizer.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <array>
#include <numeric>
#include <unordered_map>

namespace engine::models::sam_audio {
namespace {
using core::TensorShape;
using core::TensorValue;
using Weights = std::unordered_map<std::string, TensorValue>;

// Evaluate convolution windows with full receptive-field halos. Only the valid
// interior is copied; the first/last windows retain the real model boundaries.
template<class Evaluate>
std::vector<float> evaluate_windows(const std::vector<float> & input, int64_t rows,
    int64_t input_frames, int64_t window, int64_t halo, int64_t output_rows,
    int64_t output_window, Evaluate evaluate) {
    const int64_t output_frames = input_frames * output_window / window;
    const int64_t step = window == input_frames ? window : window - 2 * halo;
    std::vector<float> output(output_rows * output_frames), tile(rows * window);
    for (int64_t begin = 0; begin < input_frames; begin += step) {
        const int64_t end = std::min(begin + step, input_frames);
        const int64_t start = std::clamp(begin - halo, int64_t{0}, input_frames - window);
        for (int64_t row = 0; row < rows; ++row)
            std::copy_n(input.data() + row * input_frames + start, window, tile.data() + row * window);
        const auto value = evaluate(tile);
        const int64_t offset = (begin - start) * output_window / window;
        const int64_t count = (end - begin) * output_window / window;
        for (int64_t row = 0; row < output_rows; ++row)
            std::copy_n(value.data() + row * output_window + offset, count,
                        output.data() + row * output_frames + begin * output_window / window);
    }
    return output;
}

class DacVAEEncoderGraph {
public:
    DacVAEEncoderGraph(core::ExecutionContext & execution, const DacVAEConfig & config,
                 const Weights & weights, int64_t samples) : backend_(execution.backend()) {
        constexpr size_t nodes = 8192;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio encoder context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.codec.encoder", execution.backend_type()};
        auto x = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, samples}));
        input_ = x.tensor;
        ggml_set_input(input_);
        const int64_t hop = std::accumulate(config.encoder_rates.begin(), config.encoder_rates.end(), int64_t{1}, std::multiplies<>());
        if (samples % hop) x = modules::ReflectPad1dModule({0, hop - samples % hop}).build(ctx, x);
        auto conv = [&](TensorValue input, const std::string & name, int stride = 1, int dilation = 1) {
            const auto & w = weights.at(name + ".weight");
            const int64_t kernel = w.shape.dims[2];
            return modules::Conv1dModule({w.shape.dims[1], w.shape.dims[0], kernel,
                stride, static_cast<int>((kernel - stride) * dilation / 2), dilation, true})
                .build(ctx, input, {w, weights.at(name + ".bias")});
        };
        auto snake = [&](TensorValue input, const std::string & name) {
            auto alpha = weights.at(name + ".alpha");
            alpha = core::reshape_tensor(ctx, alpha, TensorShape::from_dims({input.shape.dims[1]}));
            // DAC's Snake includes 1e-9 in the denominator; SnakeBeta matches it with beta=alpha.
            return modules::SnakeBeta1dModule({input.shape.dims[1], false}).build(ctx, input, {alpha, alpha});
        };
        const std::string root = "audio_codec.encoder.block.";
        x = conv(x, root + "0");
        for (size_t i = 0; i < config.encoder_rates.size(); ++i) {
            const std::string block = root + std::to_string(i + 1) + ".block.";
            int dilation = 1;
            for (int r = 0; r < 3; ++r, dilation *= 3) {
                const std::string unit = block + std::to_string(r) + ".block.";
                auto y = snake(x, unit + "0");
                y = conv(y, unit + "1", 1, dilation);
                y = snake(y, unit + "2");
                y = conv(y, unit + "3");
                x = modules::AddModule().build(ctx, x, y);
            }
            x = snake(x, block + "3");
            x = conv(x, block + "4", config.encoder_rates[i]);
        }
        x = snake(x, root + std::to_string(config.encoder_rates.size() + 1));
        x = conv(x, root + std::to_string(config.encoder_rates.size() + 2));
        x = conv(x, "audio_codec.quantizer.in_proj");
        x = modules::SliceModule({1, 0, config.codebook_dim}).build(ctx, x);
        x = core::ensure_backend_addressable_layout(ctx, x);
        output_ = x.tensor;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_);
        runtime::optimize_graph(*graph_, runtime::GraphOptimizationBackend::Gpu);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("SAM Audio encoder graph allocation failed");
        debug::timing_log_scalar("sam_audio.codec.encoder.graph_buffer_mb", ggml_gallocr_get_buffer_size(allocator_.get(), 0) / 1048576.0);
    }

    ~DacVAEEncoderGraph() { core::release_backend_graph_resources(backend_, graph_, true); }

    std::vector<float> run(const std::vector<float> & audio, bool log_timing = true) {
        const auto started = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(input_, audio.data(), 0, audio.size() * sizeof(float));
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM Audio encoder graph execution failed");
        std::vector<float> result(static_cast<size_t>(ggml_nelements(output_)));
        ggml_backend_tensor_get(output_, result.data(), 0, result.size() * sizeof(float));
        if (log_timing) debug::timing_log_scalar("sam_audio.codec.encoder.wall_ms", debug::elapsed_ms(started));
        return result;
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * output_ = nullptr;
};
}  // namespace

struct DacVAEEncoder::Impl {
    core::ExecutionContext & execution;
    DacVAEConfig config;
    core::BackendWeightStore store;
    Weights weights;
    std::unique_ptr<DacVAEEncoderGraph> graph;
    size_t samples = 0;
    bool memory_bounded = false;
    int64_t context_samples = 0;

    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution_, DacVAEConfig config_)
        : execution(execution_), config(std::move(config_)),
          store(execution.backend(), execution.backend_type(), "sam_audio.codec.encoder.weights", 4 * 1024 * 1024) {
        source = assets::make_weight_norm_folded_tensor_source(std::move(source), {"audio_codec.*"});
        for (const auto & tensor : source->tensors()) {
            if (tensor.name.rfind("audio_codec.encoder.", 0) == 0 ||
                tensor.name.rfind("audio_codec.quantizer.in_proj.", 0) == 0) {
                weights.emplace(tensor.name, store.load_tensor(*source, tensor.name,
                    assets::TensorStorageType::Native, tensor.shape));
            }
        }
        store.upload();
        int64_t span = weights.at("audio_codec.encoder.block.0.weight").shape.dims[2], jump = 1;
        for (size_t i = 0; i < config.encoder_rates.size(); ++i) {
            const auto block = "audio_codec.encoder.block." + std::to_string(i + 1) + ".block.";
            for (int r = 0, dilation = 1; r < 3; ++r, dilation *= 3) {
                const auto unit = block + std::to_string(r) + ".block.";
                span += ((weights.at(unit + "1.weight").shape.dims[2] - 1) * dilation +
                         weights.at(unit + "3.weight").shape.dims[2] - 1) * jump;
            }
            span += (weights.at(block + "4.weight").shape.dims[2] - 1) * jump;
            jump *= config.encoder_rates[i];
        }
        span += (weights.at("audio_codec.encoder.block." + std::to_string(config.encoder_rates.size() + 2) +
                            ".weight").shape.dims[2] - 1) * jump;
        span += (weights.at("audio_codec.quantizer.in_proj.weight").shape.dims[2] - 1) * jump;
        context_samples = ((span + jump - 1) / jump) * jump;
    }
};

DacVAEEncoder::DacVAEEncoder(std::shared_ptr<const assets::TensorSource> source,
                           core::ExecutionContext & execution, DacVAEConfig config, bool memory_bounded)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config))) {
    impl_->memory_bounded = memory_bounded;
}
DacVAEEncoder::~DacVAEEncoder() = default;

std::vector<float> DacVAEEncoder::encode(const std::vector<float> & audio) {
    if (impl_->memory_bounded) {
        const auto start = std::chrono::steady_clock::now();
        const int64_t hop = std::accumulate(impl_->config.encoder_rates.begin(), impl_->config.encoder_rates.end(),
                                            int64_t{1}, std::multiplies<>());
        const auto padded = audio::reflect_pad_samples(audio, 0, (hop - audio.size() % hop) % hop);
        const int64_t window = std::min<int64_t>(padded.size(), 128 * hop + 2 * impl_->context_samples);
        if (!impl_->graph || impl_->samples != static_cast<size_t>(window)) {
            impl_->graph.reset();
            impl_->graph = std::make_unique<DacVAEEncoderGraph>(impl_->execution, impl_->config, impl_->weights, window);
            impl_->samples = window;
        }
        auto result = evaluate_windows(padded, 1, padded.size(), window, impl_->context_samples,
            impl_->config.codebook_dim, window / hop,
            [&](const auto & tile) { return impl_->graph->run(tile, false); });
        debug::timing_log_scalar("sam_audio.codec.encoder.wall_ms", debug::elapsed_ms(start));
        return result;
    }
    if (!impl_->graph || impl_->samples != audio.size()) {
        impl_->graph.reset();
        impl_->graph = std::make_unique<DacVAEEncoderGraph>(impl_->execution, impl_->config, impl_->weights, audio.size());
        impl_->samples = audio.size();
    }
    return impl_->graph->run(audio);
}

struct DacVAEDecoderModule::Impl {
    DacVAEConfig config;
    core::BackendType backend_type;
    core::BackendWeightStore store;
    Weights weights;
    TensorValue zero_state;

    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution, DacVAEConfig config_)
        : config(std::move(config_)), backend_type(execution.backend_type()),
          store(execution.backend(), execution.backend_type(), "sam_audio.codec.decoder.weights", 4 * 1024 * 1024) {
        source = assets::make_weight_norm_folded_tensor_source(std::move(source), {"audio_codec.*"});
        for (const auto & tensor : source->tensors()) {
            if (tensor.name.rfind("audio_codec.decoder.", 0) == 0 ||
                tensor.name.rfind("audio_codec.quantizer.out_proj.", 0) == 0) {
                weights.emplace(tensor.name, store.load_tensor(*source, tensor.name,
                    assets::TensorStorageType::Native, tensor.shape));
            }
        }
        zero_state = store.make_f32(TensorShape::from_dims({1, 512}), std::vector<float>(512, 0.0f));
        store.upload();
    }

    TensorValue causal_conv(core::ModuleBuildContext & ctx, const TensorValue & input,
                            const std::string & name, int stride = 1) const {
        const auto & weight = weights.at(name + ".weight");
        modules::CausalConv1dConfig conv;
        conv.in_channels = weight.shape.dims[1];
        conv.out_channels = weight.shape.dims[0];
        conv.kernel_size = weight.shape.dims[2];
        conv.stride = stride;
        conv.padding_mode = modules::CausalConv1dPaddingMode::Explicit;
        conv.explicit_left = conv.kernel_size - stride;
        conv.explicit_right = (stride - input.shape.dims[2] % stride) % stride;
        const modules::CausalConv1dModule module(conv);
        const modules::Conv1dWeights conv_weights{weight, weights.at(name + ".bias")};
        if (backend_type != core::BackendType::Vulkan || input.shape.dims[0] == 1) {
            return module.build(ctx, input, conv_weights);
        }

        TensorValue output;
        for (int64_t batch_index = 0; batch_index < input.shape.dims[0]; ++batch_index) {
            auto batch_input = modules::SliceModule({0, batch_index, 1}).build(ctx, input);
            batch_input = core::ensure_backend_addressable_layout(ctx, batch_input);
            const auto batch_output = module.build(ctx, batch_input, conv_weights);
            output = output.valid()
                ? modules::ConcatModule({0}).build(ctx, output, batch_output)
                : batch_output;
        }
        return output;
    }

    TensorValue elu_residual(core::ModuleBuildContext & ctx, const TensorValue & input,
                             const std::string & name) const {
        auto x = causal_conv(ctx, modules::EluModule().build(ctx, input), name + ".block.1");
        x = causal_conv(ctx, modules::EluModule().build(ctx, x), name + ".block.3");
        return modules::AddModule().build(ctx, input, x);
    }

    TensorValue lstm(core::ModuleBuildContext & ctx, const TensorValue & input, const std::string & name,
                     DacVAEDecoderModule::RecurrentState * state = nullptr) const {
        auto x = modules::TransposeModule({{2, 0, 1, 3}, 3}).build(ctx, input);
        x = core::ensure_backend_addressable_layout(ctx, x);
        const auto residual = x;
        const auto zero = modules::RepeatModule({TensorShape::from_dims({input.shape.dims[0], 512})}).build(ctx, zero_state);
        for (int layer = 0; layer < 2; ++layer) {
            const auto suffix = "_l" + std::to_string(layer);
            modules::LSTMSequenceWeights w{{weights.at(name + ".weight_ih" + suffix),
                weights.at(name + ".weight_hh" + suffix), weights.at(name + ".bias_ih" + suffix),
                weights.at(name + ".bias_hh" + suffix)}};
            const auto result = modules::LSTMSequenceModule({512, 512, false, true, true}).build(
                ctx, x, state ? (*state)[2 * layer] : zero, state ? (*state)[2 * layer + 1] : zero, w);
            x = result.sequence;
            if (state) {
                (*state)[2 * layer] = result.hidden;
                (*state)[2 * layer + 1] = result.cell;
            }
        }
        x = modules::AddModule().build(ctx, x, residual);
        return modules::TransposeModule({{1, 2, 0, 3}, 3}).build(ctx, x);
    }
};

DacVAEDecoderModule::DacVAEDecoderModule(std::shared_ptr<const assets::TensorSource> source,
                                       core::ExecutionContext & execution, DacVAEConfig config)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config))) {}
DacVAEDecoderModule::~DacVAEDecoderModule() = default;

TensorValue DacVAEDecoderModule::project_latents(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    const auto & weights = impl_->weights;
    return modules::Conv1dModule({impl_->config.codebook_dim, impl_->config.latent_dim, 1, 1, 0, 1, true})
        .build(ctx, input, {weights.at("audio_codec.quantizer.out_proj.weight"),
                            weights.at("audio_codec.quantizer.out_proj.bias")});
}

TensorValue DacVAEDecoderModule::embed_message(core::ModuleBuildContext & ctx, const TensorValue & input,
                                             const TensorValue & message_indices) const {
    const auto & embedding = impl_->weights.at("audio_codec.decoder.wm_model.msg_processor.msg_processor.weight");
    auto message = modules::EmbeddingModule({embedding.shape.dims[0], embedding.shape.dims[1]})
        .build(ctx, message_indices, embedding);
    message = modules::ReduceSumModule({1}).build(ctx, message);
    message = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, message);
    return modules::AddModule().build(ctx, input, modules::RepeatModule({input.shape}).build(ctx, message));
}

TensorValue DacVAEDecoderModule::watermark_base(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    const auto & weights = impl_->weights;
    const std::string prefix = "audio_codec.decoder.wm_model.encoder_block.pre.";
    auto alpha = core::reshape_tensor(ctx, weights.at(prefix + "0.alpha"), TensorShape::from_dims({input.shape.dims[1]}));
    auto x = modules::SnakeBeta1dModule({input.shape.dims[1], false}).build(ctx, input, {alpha, alpha});
    x = modules::Conv1dModule({input.shape.dims[1], 1, 7, 1, 3, 1, true})
        .build(ctx, x, {weights.at(prefix + "1.weight"), weights.at(prefix + "1.bias")});
    return modules::TanhModule().build(ctx, x);
}

TensorValue DacVAEDecoderModule::watermark_encode(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    return watermark_encode_output(ctx, watermark_lstm(ctx, watermark_encode_convs(ctx, input), false));
}

TensorValue DacVAEDecoderModule::watermark_encode_convs(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    const std::string prefix = "audio_codec.decoder.wm_model.encoder_block.";
    auto x = impl_->causal_conv(ctx, input, prefix + "pre.3");
    for (size_t i = impl_->config.watermark_rates.size(); i > 0; --i) {
        const auto block = "audio_codec.decoder.model." + std::to_string(i) + ".block.";
        x = impl_->elu_residual(ctx, x, block + "7");
        x = impl_->causal_conv(ctx, modules::EluModule().build(ctx, x), block + "11", impl_->config.watermark_rates[i - 1]);
    }
    return x;
}

TensorValue DacVAEDecoderModule::watermark_encode_output(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    return impl_->causal_conv(ctx, modules::EluModule().build(ctx, input),
                             "audio_codec.decoder.wm_model.encoder_block.post.2");
}

TensorValue DacVAEDecoderModule::watermark_decode(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    return watermark_decode_convs(ctx, watermark_lstm(ctx, watermark_decode_input(ctx, input), true));
}

TensorValue DacVAEDecoderModule::watermark_decode_input(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    return impl_->causal_conv(ctx, input, "audio_codec.decoder.wm_model.decoder_block.pre.0");
}

TensorValue DacVAEDecoderModule::watermark_lstm(core::ModuleBuildContext & ctx, const TensorValue & input,
                                              bool decoder, RecurrentState * state) const {
    return impl_->lstm(ctx, input, decoder ? "audio_codec.decoder.wm_model.decoder_block.pre.1.lstm"
                                         : "audio_codec.decoder.wm_model.encoder_block.post.0.lstm", state);
}

TensorValue DacVAEDecoderModule::watermark_decode_convs(core::ModuleBuildContext & ctx, const TensorValue & input) const {
    const std::string prefix = "audio_codec.decoder.wm_model.decoder_block.";
    auto x = input;
    for (size_t i = 0; i < impl_->config.watermark_rates.size(); ++i) {
        const auto block = "audio_codec.decoder.model." + std::to_string(i + 1) + ".block.";
        const auto & weight = impl_->weights.at(block + "3.weight");
        const int stride = impl_->config.watermark_rates[i];
        const int64_t frames = x.shape.dims[2] * stride;
        x = modules::ConvTranspose1dModule({weight.shape.dims[0], weight.shape.dims[1], weight.shape.dims[2],
            stride, 0, 1, true}).build(ctx, modules::EluModule().build(ctx, x),
                {weight, impl_->weights.at(block + "3.bias")});
        x = modules::SliceModule({2, 0, frames}).build(ctx, x);
        x = impl_->elu_residual(ctx, x, block + "6");
    }
    return impl_->causal_conv(ctx, modules::EluModule().build(ctx, x), prefix + "post.1");
}

TensorValue DacVAEDecoderModule::build_block(core::ModuleBuildContext & ctx, const TensorValue & input,
                                           size_t block) const {
    const auto & weights = impl_->weights;
    auto conv = [&](const TensorValue & value, const std::string & name, int dilation = 1) {
        const auto & weight = weights.at(name + ".weight");
        const auto kernel = weight.shape.dims[2];
        return modules::Conv1dModule({weight.shape.dims[1], weight.shape.dims[0], kernel,
            1, static_cast<int>((kernel - 1) * dilation / 2), dilation, true})
            .build(ctx, value, {weight, weights.at(name + ".bias")});
    };
    auto snake = [&](const TensorValue & value, const std::string & name) {
        auto alpha = core::reshape_tensor(ctx, weights.at(name + ".alpha"),
                                         TensorShape::from_dims({value.shape.dims[1]}));
        return modules::SnakeBeta1dModule({value.shape.dims[1], false}).build(ctx, value, {alpha, alpha});
    };
    const std::string root = "audio_codec.decoder.model." + std::to_string(block);
    if (block == 0) return conv(input, root);
    const int stride = impl_->config.decoder_rates.at(block - 1);
    auto x = snake(input, root + ".block.0");
    const auto & weight = weights.at(root + ".block.1.weight");
    const int padding = (stride + 1) / 2;
    const int backend_padding = impl_->backend_type == core::BackendType::Cpu ? 0 : padding;
    x = modules::ConvTranspose1dModule({weight.shape.dims[0], weight.shape.dims[1], weight.shape.dims[2],
        stride, backend_padding, 1, true}).build(ctx, x, {weight, weights.at(root + ".block.1.bias")});
    if (backend_padding == 0 && padding != 0) {
        x = modules::SliceModule({2, padding, x.shape.dims[2] - 2 * padding}).build(ctx, x);
    }
    int dilation = 1;
    for (int unit : {4, 5, 8}) {
        const std::string name = root + ".block." + std::to_string(unit) + ".block.";
        auto y = conv(snake(x, name + "0"), name + "1", dilation);
        y = conv(snake(y, name + "2"), name + "3");
        x = modules::AddModule().build(ctx, x, y);
        dilation *= 3;
    }
    return x;
}

int64_t DacVAEDecoderModule::context_frames(CodecTileStage stage) const {
    // A conservative full receptive-field span, rounded to the downsampling
    // lattice where needed, is sufficient as a halo on either side.
    const auto kernel = [&](const std::string & name) { return impl_->weights.at(name + ".weight").shape.dims[2]; };
    const std::string root = "audio_codec.decoder.";
    if (stage == CodecTileStage::Bridge)
        return kernel(root + "wm_model.encoder_block.post.2") + kernel(root + "wm_model.decoder_block.pre.0") - 2;
    if (stage == CodecTileStage::WatermarkEncoder) {
        int64_t span = kernel(root + "wm_model.encoder_block.pre.3"), jump = 1;
        for (size_t i = impl_->config.watermark_rates.size(); i > 0; --i) {
            const auto block = root + "model." + std::to_string(i) + ".block.";
            span += (kernel(block + "7.block.1") + kernel(block + "7.block.3") - 2 + kernel(block + "11") - 1) * jump;
            jump *= impl_->config.watermark_rates[i - 1];
        }
        return ((span + jump - 1) / jump) * jump;
    }
    const bool main = stage == CodecTileStage::Main;
    const auto & rates = main ? impl_->config.decoder_rates : impl_->config.watermark_rates;
    int64_t span = kernel(root + (main ? "wm_model.encoder_block.pre.1" : "wm_model.decoder_block.post.1"));
    for (size_t i = rates.size(); i > 0; --i) {
        const auto block = root + "model." + std::to_string(i) + ".block.";
        if (main) {
            int dilation = 1;
            for (int unit : {4, 5, 8}) {
                const auto name = block + std::to_string(unit) + ".block.";
                span += (kernel(name + "1") - 1) * dilation + kernel(name + "3") - 1;
                dilation *= 3;
            }
        } else {
            span += kernel(block + "6.block.1") + kernel(block + "6.block.3") - 2;
        }
        span = (span + kernel(block + (main ? "1" : "3")) - 2 + rates[i - 1] - 1) / rates[i - 1] + 1;
    }
    return main ? span + kernel(root + "model.0") - 1 : span;
}

namespace {
class DacVAETileGraph {
public:
    DacVAETileGraph(core::ExecutionContext & execution, const DacVAEDecoderModule & module,
                   const DacVAEConfig & config, CodecTileStage stage, int64_t batch, int64_t frames)
        : backend_(execution.backend()), frames_(frames) {
        const bool recurrent = stage == CodecTileStage::EncoderLSTM || stage == CodecTileStage::DecoderLSTM;
        const size_t nodes = recurrent ? frames * batch * 128 + 8192 : 8192;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio codec tile context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.codec.decoder", execution.backend_type()};
        const int64_t channels = stage == CodecTileStage::Main ? config.codebook_dim :
                                 stage == CodecTileStage::WatermarkEncoder ? 1 : 512;
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, channels, frames}));
        ggml_set_input(input_.tensor);
        auto x = input_;
        if (stage == CodecTileStage::Main) {
            x = module.project_latents(ctx, x);
            for (size_t i = 0; i <= config.decoder_rates.size(); ++i) x = module.build_block(ctx, x, i);
            x = module.watermark_base(ctx, x);
        } else if (stage == CodecTileStage::WatermarkEncoder) {
            x = module.watermark_encode_convs(ctx, x);
        } else if (recurrent) {
            for (auto & state : state_in_) {
                state = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, 512}));
                ggml_set_input(state.tensor);
                ggml_set_output(state.tensor);
            }
            state_out_ = state_in_;
            x = module.watermark_lstm(ctx, x, stage == CodecTileStage::DecoderLSTM, &state_out_);
        } else if (stage == CodecTileStage::Bridge) {
            message_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, 16}));
            ggml_set_input(message_.tensor);
            x = module.watermark_decode_input(ctx, module.embed_message(ctx, module.watermark_encode_output(ctx, x), message_));
        } else {
            x = module.watermark_decode_convs(ctx, x);
        }
        output_ = core::ensure_backend_addressable_layout(ctx, x);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        for (auto & state : state_out_) {
            if (!state.valid()) continue;
            state = core::ensure_backend_addressable_layout(ctx, state);
            ggml_set_output(state.tensor);
            ggml_build_forward_expand(graph_, state.tensor);
        }
        auto optimization = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Gpu);
        // Preserve the original watermark stage's view/materialization policy.
        if (stage != CodecTileStage::Main) optimization.fold_identity_materializations = false;
        if (execution.backend_type() == core::BackendType::Vulkan || execution.backend_type() == core::BackendType::Cpu)
            optimization.fold_unary_broadcast_repeats = false;
        runtime::optimize_graph(*graph_, optimization);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_))
            throw std::runtime_error("SAM Audio codec tile allocation failed");
        debug::timing_log_scalar("sam_audio.codec.decoder.graph_buffer_mb", ggml_gallocr_get_buffer_size(allocator_.get(), 0) / 1048576.0);
    }

    ~DacVAETileGraph() { core::release_backend_graph_resources(backend_, graph_, true); }
    int64_t frames() const { return frames_; }
    int64_t output_frames() const { return output_.shape.dims[2]; }
    void reset_state() {
        for (const auto & state : state_in_)
            if (state.valid()) core::write_tensor_f32(state, std::vector<float>(ggml_nelements(state.tensor), 0.0f));
    }
    std::vector<float> run(const std::vector<float> & input, const std::vector<int32_t> & indices) {
        core::write_tensor_f32(input_, input);
        if (message_.valid()) core::write_tensor_i32(message_, indices);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("SAM Audio codec tile execution failed");
        auto result = core::read_tensor_f32(output_.tensor);
        for (size_t i = 0; i < state_in_.size(); ++i)
            if (state_in_[i].valid()) ggml_backend_tensor_copy(state_out_[i].tensor, state_in_[i].tensor);
        return result;
    }

private:
    ggml_backend_t backend_;
    int64_t frames_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    TensorValue input_, message_, output_;
    DacVAEDecoderModule::RecurrentState state_in_{}, state_out_{};
};

class DacVAEDecoderGraph {
public:
    DacVAEDecoderGraph(core::ExecutionContext & execution, const DacVAEDecoderModule & module,
                     const DacVAEConfig & config, int64_t batch, int64_t frames)
        : backend_(execution.backend()) {
        const int64_t samples = frames * std::accumulate(
            config.decoder_rates.begin(), config.decoder_rates.end(), int64_t{1}, std::multiplies<>());
        int64_t watermark_frames = samples;
        for (auto it = config.watermark_rates.rbegin(); it != config.watermark_rates.rend(); ++it)
            watermark_frames = (watermark_frames + *it - 1) / *it;
        const size_t nodes = static_cast<size_t>(watermark_frames * batch * 256 + 16384);
        context_.reset(ggml_init({
            nodes * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("SAM Audio decoder context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "sam_audio.codec.decoder", execution.backend_type()};
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({batch, config.codebook_dim, frames}));
        message_ = core::make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({batch, 16}));
        ggml_set_input(input_.tensor);
        ggml_set_input(message_.tensor);
        auto decoded = module.project_latents(ctx, input_);
        for (size_t block = 0; block <= config.decoder_rates.size(); ++block)
            decoded = module.build_block(ctx, decoded, block);
        base_ = module.watermark_base(ctx, decoded);
        ggml_set_output(base_.tensor);
        auto * graph = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph, base_.tensor);
        const int main_end = ggml_graph_n_nodes(graph);
        const auto encoded = module.watermark_encode(ctx, base_);
        ggml_set_output(encoded.tensor);
        ggml_build_forward_expand(graph, encoded.tensor);
        const int encoder_end = ggml_graph_n_nodes(graph);
        output_ = module.watermark_decode(ctx, module.embed_message(ctx, encoded, message_));
        ggml_set_output(output_.tensor);
        ggml_build_forward_expand(graph, output_.tensor);
        const int ends[] = {main_end, encoder_end, ggml_graph_n_nodes(graph)};
        ggml_tensor * outputs[] = {base_.tensor, encoded.tensor, output_.tensor};
        int begin = 0;
        // Keep the original stage optimizations and CUDA captures, but allocate their
        // connected graph once so scratch can be reused after each stage finishes.
        for (size_t i = 0; i < stages_.size(); ++i) {
            auto * stage = ggml_new_graph_custom(context_.get(), ends[i] + 1024, false);
            ggml_build_forward_expand(stage, outputs[i]);
            std::copy(ggml_graph_nodes(graph) + begin, ggml_graph_nodes(graph) + ends[i],
                      ggml_graph_nodes(stage));
            ggml_graph_set_n_nodes(stage, ends[i] - begin);
            stages_[i] = stage;
            begin = ends[i];
        }
        for (size_t i = 0; i < stages_.size(); ++i) {
            auto optimization = runtime::graph_optimization_options_for_backend(runtime::GraphOptimizationBackend::Gpu);
            if (i != 0) optimization.fold_identity_materializations = false;
            if (execution.backend_type() == core::BackendType::Vulkan ||
                execution.backend_type() == core::BackendType::Cpu) {
                optimization.fold_unary_broadcast_repeats = false;
            }
            runtime::optimize_graph(*stages_[i], optimization);
        }
        int count = 0;
        for (auto * stage : stages_) {
            const int size = ggml_graph_n_nodes(stage);
            std::copy_n(ggml_graph_nodes(stage), size, ggml_graph_nodes(graph) + count);
            count += size;
        }
        ggml_graph_set_n_nodes(graph, count);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph))
            throw std::runtime_error("SAM Audio decoder graph allocation failed");
        debug::timing_log_scalar("sam_audio.codec.decoder.graph_buffer_mb",
                                 ggml_gallocr_get_buffer_size(allocator_.get(), 0) / 1048576.0);
    }

    ~DacVAEDecoderGraph() {
        for (auto * stage : stages_)
            core::release_backend_graph_resources(backend_, stage, true);
    }

    std::vector<float> run(const std::vector<float> & latents, const std::vector<int32_t> & indices) {
        core::write_tensor_f32(input_, latents);
        core::write_tensor_i32(message_, indices);
        const char * keys[] = {"sam_audio.codec.decoder.main.wall_ms",
                              "sam_audio.codec.decoder.watermark_encoder.wall_ms",
                              "sam_audio.codec.decoder.watermark_decoder.wall_ms"};
        for (size_t i = 0; i < stages_.size(); ++i) {
            const auto start = std::chrono::steady_clock::now();
            if (core::compute_backend_graph(backend_, stages_[i]) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("SAM Audio decoder graph execution failed");
            ggml_backend_synchronize(backend_);
            debug::timing_log_scalar(keys[i], debug::elapsed_ms(start));
            // Return the base waveform without running the watermark stages.
            if (i == 0) return core::read_tensor_f32(base_.tensor);
        }
        auto decoded = core::read_tensor_f32(output_.tensor);
        const auto base = core::read_tensor_f32(base_.tensor);
        if (decoded.size() != base.size()) throw std::runtime_error("SAM Audio watermark output length mismatch");
        for (size_t i = 0; i < decoded.size(); ++i) decoded[i] = base[i] + 0.25f * decoded[i];
        return decoded;
    }

private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    std::array<ggml_cgraph *, 3> stages_{};
    TensorValue input_, message_, output_, base_;
};
}  // namespace

struct DacVAEDecoder::Impl {
    core::ExecutionContext & execution;
    DacVAEConfig config;
    DacVAEDecoderModule module;
    std::unique_ptr<DacVAEDecoderGraph> graph;
    std::array<std::unique_ptr<DacVAETileGraph>, 6> tiles;
    bool memory_bounded = false;
    int64_t batch = 0;
    int64_t frames = 0;

    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution_, DacVAEConfig config_)
        : execution(execution_), config(std::move(config_)), module(std::move(source), execution, config) {}
};

DacVAEDecoder::DacVAEDecoder(std::shared_ptr<const assets::TensorSource> source,
                           core::ExecutionContext & execution, DacVAEConfig config, bool memory_bounded)
    : impl_(std::make_unique<Impl>(std::move(source), execution, std::move(config))) {
    impl_->memory_bounded = memory_bounded;
}
DacVAEDecoder::~DacVAEDecoder() = default;

std::vector<float> DacVAEDecoder::decode(const std::vector<float> & latents, int64_t batch, int64_t frames,
                                       const std::vector<int32_t> & message_bits) {
    if (batch <= 0 || frames <= 0 || latents.size() != static_cast<size_t>(batch * frames * impl_->config.codebook_dim))
        throw std::runtime_error("SAM Audio decoder latent shape mismatch");
    if (message_bits.size() != static_cast<size_t>(batch * 16))
        throw std::runtime_error("SAM Audio watermark requires 16 message bits per audio stream");
    std::vector<int32_t> indices(message_bits.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        if (message_bits[i] != 0 && message_bits[i] != 1)
            throw std::runtime_error("SAM Audio watermark message values must be zero or one");
        indices[i] = 2 * static_cast<int32_t>(i % 16) + message_bits[i];
    }
    if (impl_->memory_bounded) {
        const auto start = std::chrono::steady_clock::now();
        const int64_t hop = std::accumulate(impl_->config.decoder_rates.begin(), impl_->config.decoder_rates.end(),
                                            int64_t{1}, std::multiplies<>());
        const int64_t wm_hop = std::accumulate(impl_->config.watermark_rates.begin(), impl_->config.watermark_rates.end(),
                                               int64_t{1}, std::multiplies<>());
        if (impl_->batch != batch) {
            for (auto & tile : impl_->tiles) tile.reset();
            impl_->batch = batch;
        }
        auto convolve = [&](CodecTileStage stage, const std::vector<float> & input,
                            int64_t channels, int64_t length, int64_t core, int64_t output_channels) {
            const int64_t halo = impl_->module.context_frames(stage);
            const int64_t window = std::min(length, core + 2 * halo);
            auto & graph = impl_->tiles[static_cast<size_t>(stage)];
            if (!graph || graph->frames() != window) {
                graph.reset();
                graph = std::make_unique<DacVAETileGraph>(impl_->execution, impl_->module, impl_->config, stage, batch, window);
            }
            return evaluate_windows(input, batch * channels, length, window, halo,
                batch * output_channels, graph->output_frames(),
                [&](const auto & tile) { return graph->run(tile, indices); });
        };
        auto recurrent = [&](CodecTileStage stage, const std::vector<float> & input, int64_t length) {
            constexpr int64_t window = 128;
            auto & graph = impl_->tiles[static_cast<size_t>(stage)];
            if (!graph) graph = std::make_unique<DacVAETileGraph>(
                impl_->execution, impl_->module, impl_->config, stage, batch, window);
            graph->reset_state();
            std::vector<float> result(input.size()), tile(batch * 512 * window);
            for (int64_t begin = 0; begin < length; begin += window) {
                const int64_t count = std::min(window, length - begin);
                std::fill(tile.begin(), tile.end(), 0.0f);
                for (int64_t row = 0; row < batch * 512; ++row)
                    std::copy_n(input.data() + row * length + begin, count, tile.data() + row * window);
                const auto output = graph->run(tile, indices);
                for (int64_t row = 0; row < batch * 512; ++row)
                    std::copy_n(output.data() + row * window, count, result.data() + row * length + begin);
            }
            return result;
        };
        auto base = convolve(CodecTileStage::Main, latents, impl_->config.codebook_dim, frames, 128, 1);
        // Preserve watermark code for future use, but return unwatermarked audio.
        debug::timing_log_scalar("sam_audio.codec.decoder.wall_ms", debug::elapsed_ms(start));
        return base;
        const int64_t samples = frames * hop, wm_frames = samples / wm_hop;
        auto x = convolve(CodecTileStage::WatermarkEncoder, base, 1, samples, 128 * hop, 512);
        x = recurrent(CodecTileStage::EncoderLSTM, x, wm_frames);
        x = convolve(CodecTileStage::Bridge, x, 512, wm_frames, 128, 512);
        x = recurrent(CodecTileStage::DecoderLSTM, x, wm_frames);
        x = convolve(CodecTileStage::WatermarkDecoder, x, 512, wm_frames, 128 * hop / wm_hop, 1);
        if (x.size() != base.size()) throw std::runtime_error("SAM Audio watermark output length mismatch");
        for (size_t i = 0; i < base.size(); ++i) base[i] += 0.25f * x[i];
        debug::timing_log_scalar("sam_audio.codec.decoder.wall_ms", debug::elapsed_ms(start));
        return base;
    }
    if (impl_->batch != batch || impl_->frames != frames) {
        impl_->graph.reset();
        impl_->batch = batch;
        impl_->frames = frames;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!impl_->graph) impl_->graph = std::make_unique<DacVAEDecoderGraph>(
        impl_->execution, impl_->module, impl_->config, batch, frames);
    auto decoded = impl_->graph->run(latents, indices);
    debug::timing_log_scalar("sam_audio.codec.decoder.wall_ms", debug::elapsed_ms(start));
    return decoded;
}

}  // namespace engine::models::sam_audio
