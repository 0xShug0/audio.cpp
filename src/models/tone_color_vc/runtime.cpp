#include "engine/models/tone_color_vc/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/resampling.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/vocoders/hifigan_vocoder.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/sampling/noise.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace engine::models::tone_color_vc {
namespace {

using core::TensorValue;
using core::TensorShape;

constexpr int kSampleRate = 22050;
constexpr size_t kContextBytes = 4 * 1024 * 1024;

struct ContextDeleter {
    void operator()(ggml_context * p) const { ggml_free(p); }
};
struct AllocatorDeleter {
    void operator()(ggml_gallocr_t p) const { ggml_gallocr_free(p); }
};

struct Weights {
    core::BackendWeightStore store;
    std::map<std::string, TensorValue> tensors;
    TensorValue zero_conditioning, zero_hidden, reverse_indices;

    Weights(const assets::TensorSource & source, core::ExecutionContext & execution)
        : store(execution.backend(), execution.backend_type(), "tone_color_vc.weights", kContextBytes) {
        for (const auto & tensor : source.tensors()) {
            if (tensor.name.rfind("dec.", 0) != 0) {
                tensors.emplace(tensor.name, store.load_tensor(
                    source, tensor.name, assets::TensorStorageType::Native, tensor.shape));
            }
        }
        zero_conditioning = store.make_f32(TensorShape::from_dims({1, 256, 1}), std::vector<float>(256, 0));
        zero_hidden = store.make_f32(TensorShape::from_dims({1, 128}), std::vector<float>(128, 0));
        std::vector<int32_t> indices(192);
        for (int i = 0; i < 192; ++i) indices[i] = 191 - i;
        reverse_indices = store.make_tensor(TensorShape::from_dims({192}), GGML_TYPE_I32,
            indices.data(), indices.size() * sizeof(int32_t));
        store.upload();
    }

    const TensorValue & at(const std::string & name) const { return tensors.at(name); }
};

class Graph {
public:
    Graph(core::ExecutionContext & execution, const Weights & weights)
        : execution_(execution), weights_(weights), context_(ggml_init({kContextBytes, nullptr, true})),
          ctx_{context_.get(), "tone_color_vc", execution.backend_type()} {
        if (!context_) throw std::runtime_error("Tone color converter graph context allocation failed");
    }
    ~Graph() { core::release_backend_graph_resources(execution_.backend(), graph_, true); }

    TensorValue input(std::initializer_list<int64_t> shape) {
        auto value = core::make_tensor(ctx_, GGML_TYPE_F32, TensorShape::from_dims(shape));
        ggml_set_input(value.tensor);
        return value;
    }

    TensorValue conv(TensorValue x, const std::string & prefix) {
        const auto & w = weights_.at(prefix + ".weight");
        const auto & d = w.shape.dims;
        return modules::Conv1dModule({d[1], d[0], d[2], 1, int(d[2] / 2), 1, true})
            .build(ctx_, x, {w, weights_.at(prefix + ".bias")});
    }

    TensorValue linear(TensorValue x, const std::string & prefix) {
        const auto & w = weights_.at(prefix + ".weight");
        return modules::LinearModule({w.shape.dims[1], w.shape.dims[0], true})
            .build(ctx_, x, {w, weights_.at(prefix + ".bias")});
    }

    TensorValue wavenet(TensorValue x, TensorValue conditioning, const std::string & prefix, int layers) {
        auto g = conv(conditioning, prefix + ".cond_layer");
        TensorValue output;
        for (int i = 0; i < layers; ++i) {
            auto h = conv(x, prefix + ".in_layers." + std::to_string(i));
            auto c = modules::SliceModule({1, i * 384, 384}).build(ctx_, g);
            c = modules::RepeatModule({h.shape}).build(ctx_, c);
            h = modules::AddModule().build(ctx_, h, c);
            auto a = modules::TanhModule().build(ctx_, modules::SliceModule({1, 0, 192}).build(ctx_, h));
            auto b = modules::SigmoidModule().build(ctx_, modules::SliceModule({1, 192, 192}).build(ctx_, h));
            h = conv(modules::MulModule().build(ctx_, a, b), prefix + ".res_skip_layers." + std::to_string(i));
            if (i + 1 < layers) {
                x = modules::AddModule().build(ctx_, x, modules::SliceModule({1, 0, 192}).build(ctx_, h));
                h = modules::SliceModule({1, 192, 192}).build(ctx_, h);
            }
            output = i == 0 ? h : modules::AddModule().build(ctx_, output, h);
        }
        return output;
    }

    TensorValue flow(TensorValue x, TensorValue conditioning, bool reverse) {
        // Flip reverses all channels, not just the two coupling partitions.
        for (int step = 0; step < 4; ++step) {
            const int layer = reverse ? 3 - step : step;
            if (reverse) x = channel_flip(x);
            auto a = modules::SliceModule({1, 0, 96}).build(ctx_, x);
            auto b = modules::SliceModule({1, 96, 96}).build(ctx_, x);
            const auto prefix = "flow.flows." + std::to_string(2 * layer);
            auto mean = conv(wavenet(conv(a, prefix + ".pre"), conditioning, prefix + ".enc", 4), prefix + ".post");
            b = reverse ? core::wrap_tensor(ggml_sub(ctx_.ggml, b.tensor, mean.tensor), b.shape)
                        : modules::AddModule().build(ctx_, mean, b);
            x = modules::ConcatModule({1}).build(ctx_, a, b);
            if (!reverse) x = channel_flip(x);
        }
        return x;
    }

    TensorValue channel_flip(TensorValue x) {
        auto rows = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x),
            TensorShape::from_dims({192, x.shape.dims[2]}));
        auto reversed = modules::EmbeddingModule({192, x.shape.dims[2]}).build(ctx_, weights_.reverse_indices, rows);
        return core::reshape_tensor(ctx_, reversed, x.shape);
    }

    void build_conversion(int64_t frames) {
        frames_ = frames;
        spectrum_ = input({1, 513, frames});
        source_embedding_ = input({1, 256, 1});
        target_embedding_ = input({1, 256, 1});
        noise_ = input({1, 192, frames});
        auto stats = conv(wavenet(conv(spectrum_, "enc_q.pre"), weights_.zero_conditioning, "enc_q.enc", 16), "enc_q.proj");
        auto mean = modules::SliceModule({1, 0, 192}).build(ctx_, stats);
        auto logs = modules::SliceModule({1, 192, 192}).build(ctx_, stats);
        auto scale = core::wrap_tensor(ggml_exp(ctx_.ggml, core::ensure_backend_addressable_layout(ctx_, logs).tensor), logs.shape);
        auto z = modules::AddModule().build(ctx_, mean, modules::MulModule().build(ctx_, noise_, scale));
        finish(flow(flow(z, source_embedding_, false), target_embedding_, true));
    }

    void build_reference(int64_t frames) {
        frames_ = frames;
        spectrum_ = input({1, 513, frames});
        auto x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx_, spectrum_);
        x = modules::LayerNormModule({513, 1e-5f, true, true}).build(ctx_, x,
            {weights_.at("ref_enc.layernorm.weight"), weights_.at("ref_enc.layernorm.bias")});
        x = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x), TensorShape::from_dims({1, 1, frames, 513}));
        for (int i = 0; i < 6; ++i) {
            const auto prefix = "ref_enc.convs." + std::to_string(i);
            const auto & w = weights_.at(prefix + ".weight");
            x = modules::Conv2dModule({w.shape.dims[1], w.shape.dims[0], 3, 3, 2, 2, 1, 1})
                .build(ctx_, x, {w, weights_.at(prefix + ".bias")});
            x = modules::ReluModule().build(ctx_, x);
        }
        const int64_t steps = x.shape.dims[2];
        const int64_t features = x.shape.dims[1] * x.shape.dims[3];
        x = modules::TransposeModule({{0, 2, 1, 3}, 4}).build(ctx_, x);
        x = core::reshape_tensor(ctx_, core::ensure_backend_addressable_layout(ctx_, x), TensorShape::from_dims({steps, features}));
        const auto & wi = weights_.at("ref_enc.gru.weight_ih_l0");
        const auto & wh = weights_.at("ref_enc.gru.weight_hh_l0");
        auto projected = modules::LinearModule({features, 384, true}).build(ctx_, x,
            {wi, weights_.at("ref_enc.gru.bias_ih_l0")});
        auto hidden = weights_.zero_hidden;
        for (int64_t t = 0; t < steps; ++t) {
            auto gate_input = modules::SliceModule({0, t, 1}).build(ctx_, projected);
            auto recurrent = modules::LinearModule({128, 384, true}).build(ctx_, hidden,
                {wh, weights_.at("ref_enc.gru.bias_hh_l0")});
            auto reset = modules::SigmoidModule().build(ctx_, modules::AddModule().build(ctx_,
                modules::SliceModule({1, 0, 128}).build(ctx_, gate_input),
                modules::SliceModule({1, 0, 128}).build(ctx_, recurrent)));
            auto update = modules::SigmoidModule().build(ctx_, modules::AddModule().build(ctx_,
                modules::SliceModule({1, 128, 128}).build(ctx_, gate_input),
                modules::SliceModule({1, 128, 128}).build(ctx_, recurrent)));
            auto candidate = modules::TanhModule().build(ctx_, modules::AddModule().build(ctx_,
                modules::SliceModule({1, 256, 128}).build(ctx_, gate_input),
                modules::MulModule().build(ctx_, reset, modules::SliceModule({1, 256, 128}).build(ctx_, recurrent))));
            auto diff = core::wrap_tensor(ggml_sub(ctx_.ggml, hidden.tensor, candidate.tensor), hidden.shape);
            hidden = modules::AddModule().build(ctx_, candidate, modules::MulModule().build(ctx_, update, diff));
        }
        finish(linear(hidden, "ref_enc.proj"));
    }

    void finish(TensorValue output) {
        output_ = core::ensure_backend_addressable_layout(ctx_, output);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(ctx_.ggml, 16384, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        auto optimization = runtime::graph_optimization_options_for_backend(
            execution_.backend_type() == core::BackendType::Cpu
                ? runtime::GraphOptimizationBackend::Cpu : runtime::GraphOptimizationBackend::Gpu);
        // Gallocr must see metadata views when sizing its allocation bookkeeping.
        optimization.elide_noop_nodes = false;
        optimization.elide_metadata_only_ops = false;
        runtime::optimize_graph(*graph_, optimization);
        core::validate_backend_graph_supported(execution_.backend(), graph_, "tone_color_vc");
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend())));
        if (!ggml_gallocr_alloc_graph(allocator_.get(), graph_)) throw std::runtime_error("Tone color converter graph allocation failed");
        core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    std::vector<float> run(const std::vector<float> & spectrum,
                           const std::vector<float> & source = {}, const std::vector<float> & target = {},
                           const std::vector<float> & noise = {}) {
        core::write_tensor_f32(spectrum_, spectrum);
        if (!source.empty()) {
            core::write_tensor_f32(source_embedding_, source);
            core::write_tensor_f32(target_embedding_, target);
            core::write_tensor_f32(noise_, noise);
        }
        if (core::compute_graph(execution_, graph_, plan_, "tone_color_vc") != GGML_STATUS_SUCCESS)
            throw std::runtime_error("Tone color converter graph compute failed");
        return core::read_tensor_f32(output_.tensor);
    }
    int64_t frames() const { return frames_; }

private:
    core::ExecutionContext & execution_;
    const Weights & weights_;
    std::unique_ptr<ggml_context, ContextDeleter> context_;
    core::ModuleBuildContext ctx_;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator_;
    core::HostGraphPlan plan_;
    ggml_cgraph * graph_ = nullptr;
    TensorValue spectrum_, source_embedding_, target_embedding_, noise_, output_;
    int64_t frames_ = 0;
};

std::vector<float> spectrum(const runtime::AudioBuffer & audio, size_t threads) {
    auto mono = audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels);
    if (audio.sample_rate != kSampleRate) {
        auto resampled = audio::try_resample_mono_soxr(mono, audio.sample_rate, kSampleRate, {});
        if (!resampled) throw std::runtime_error("Tone Color VC resampling requires libsoxr or 22050 Hz input");
        mono = std::move(*resampled);
    }
    if (mono.size() <= 384) throw std::runtime_error("Tone color converter requires more than 384 samples at 22050 Hz");
    std::vector<float> padded(mono.size() + 768);
    std::copy(mono.begin(), mono.end(), padded.begin() + 384);
    for (size_t i = 0; i < 384; ++i) {
        padded[i] = mono[384 - i];
        padded[384 + mono.size() + i] = mono[mono.size() - 2 - i];
    }
    const audio::STFTConfig config{1024, 256, 1024, false, audio::STFTPadMode::Reflect, audio::STFTFamily::Kokoro};
    auto complex = audio::STFT().compute_complex(padded, audio::get_cached_stft_window(config), 1, padded.size(), config, threads);
    std::vector<float> magnitude(complex.values.size() / 2);
    for (size_t i = 0; i < magnitude.size(); ++i) {
        const float re = complex.values[2 * i], im = complex.values[2 * i + 1];
        magnitude[i] = std::sqrt(re * re + im * im + 1e-6f);
    }
    return magnitude;
}

}  // namespace

struct ToneColorRuntime::State {
    core::ExecutionContext & execution;
    Weights weights;
    modules::HifiGanVocoderComponent decoder;
    std::unique_ptr<Graph> source_graph, reference_graph, conversion_graph;

    State(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & ctx, core::BackendConfig backend,
          const io::json::Value & model_config)
        : execution(ctx), weights(*source, ctx) {
        const auto & data = model_config.require("data");
        const auto & model = model_config.require("model");
        if (data.require("sampling_rate").as_i64() != kSampleRate ||
            data.require("filter_length").as_i64() != 1024 ||
            data.require("win_length").as_i64() != 1024 ||
            data.require("hop_length").as_i64() != 256 ||
            data.require("n_speakers").as_i64() != 0 ||
            model.require("inter_channels").as_i64() != 192 ||
            model.require("hidden_channels").as_i64() != 192 ||
            model.require("gin_channels").as_i64() != 256 ||
            !model.require("zero_g").as_bool()) {
            throw std::runtime_error("Tone Color VC requires the supported converter architecture");
        }
        modules::HifiGanVocoderConfig config;
        config.sampling_rate = kSampleRate;
        config.num_mels = 192;
        config.upsample_initial_channel = model.require("upsample_initial_channel").as_i64();
        for (const auto & value : model.require("upsample_rates").as_array()) config.upsample_rates.push_back(value.as_i64());
        for (const auto & value : model.require("upsample_kernel_sizes").as_array()) config.upsample_kernel_sizes.push_back(value.as_i64());
        for (const auto & value : model.require("resblock_kernel_sizes").as_array()) config.resblock_kernel_sizes.push_back(value.as_i64());
        for (const auto & values : model.require("resblock_dilation_sizes").as_array()) {
            std::vector<int64_t> dilations;
            for (const auto & value : values.as_array()) dilations.push_back(value.as_i64());
            config.resblock_dilation_sizes.push_back(std::move(dilations));
        }
        const auto resblock = model.require("resblock").as_string();
        if (resblock != "1" && resblock != "2") throw std::runtime_error("Unknown Tone Color VC residual block type");
        config.resblock_kind = resblock == "1" ? modules::HifiGanResBlockKind::PairedConv : modules::HifiGanResBlockKind::SingleConv;
        config.conv_post_use_bias = false;
        config.post_leaky_relu_slope = 0.01f;
        config.tensor_prefix = "dec";
        config.global_conditioning.channels = 256;
        config.lower_padded_conv_transpose_as_crop = ctx.backend_type() == core::BackendType::Cpu;
        decoder = modules::HifiGanVocoderComponent::load_from_tensor_source(source, backend, config);
    }
};

ToneColorRuntime::ToneColorRuntime(std::shared_ptr<const assets::TensorSource> source,
                                 core::ExecutionContext & execution, core::BackendConfig backend,
                                 const io::json::Value & config)
    : state_(std::make_unique<State>(std::move(source), execution, backend, config)) {}
ToneColorRuntime::~ToneColorRuntime() = default;

runtime::AudioBuffer ToneColorRuntime::convert(const runtime::AudioBuffer & source,
                                             const runtime::AudioBuffer & reference,
                                             float temperature, uint32_t seed) {
    auto & s = *state_;
    auto started = std::chrono::steady_clock::now();
    const auto source_spec = spectrum(source, s.execution.config().threads);
    const auto target_spec = spectrum(reference, s.execution.config().threads);
    debug::timing_log_scalar("tone_color_vc.frontend_ms", debug::elapsed_ms(started));
    const auto source_frames = static_cast<int64_t>(source_spec.size() / 513);
    const auto target_frames = static_cast<int64_t>(target_spec.size() / 513);
    const auto prepare_reference = [&](std::unique_ptr<Graph> & graph, int64_t frames) {
        if (!graph || graph->frames() != frames) {
            graph.reset();
            graph = std::make_unique<Graph>(s.execution, s.weights);
            graph->build_reference(frames);
        }
    };
    started = std::chrono::steady_clock::now();
    prepare_reference(s.source_graph, source_frames);
    prepare_reference(s.reference_graph, target_frames);
    auto src = s.source_graph->run(source_spec);
    auto tgt = s.reference_graph->run(target_spec);
    debug::timing_log_scalar("tone_color_vc.reference_encoder_ms", debug::elapsed_ms(started));
    started = std::chrono::steady_clock::now();
    if (!s.conversion_graph || s.conversion_graph->frames() != source_frames) {
        s.conversion_graph.reset();
        s.conversion_graph = std::make_unique<Graph>(s.execution, s.weights);
        s.conversion_graph->build_conversion(source_frames);
    }
    auto noise = temperature == 0.0f
        ? std::vector<float>(192 * source_frames, 0.0f)
        : sampling::generate_normal_noise(192 * source_frames, seed, temperature);
    auto latent = s.conversion_graph->run(source_spec, src, tgt, noise);
    debug::timing_log_scalar("tone_color_vc.flow_ms", debug::elapsed_ms(started));
    started = std::chrono::steady_clock::now();
    std::vector<float> zero(256, 0);
    modules::HifiGanVocoderRequest request;
    request.mel = &latent;
    request.frames = source_frames;
    request.conditioning = &zero;
    request.conditioning_frames = 1;
    auto decoded = s.decoder.synthesize(request);
    debug::timing_log_scalar("tone_color_vc.decoder_ms", debug::elapsed_ms(started));
    runtime::AudioBuffer output;
    output.sample_rate = kSampleRate;
    output.channels = 1;
    output.samples = std::move(decoded.waveform);
    return output;
}

}  // namespace engine::models::tone_color_vc
