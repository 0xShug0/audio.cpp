#include "engine/models/parakeet_tdt/frontend.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace engine::models::parakeet_tdt {

namespace {

constexpr size_t kFrontendGraphBytes = 32 * 1024 * 1024;
constexpr size_t kFrontendGraphNodes = 256;
constexpr float kLogZeroGuard = 1.0f / 16777216.0f;  // 2**-24
constexpr float kStddevGuard = 1e-5f;

engine::core::TensorValue repeat_like(
    engine::core::ModuleBuildContext & ctx,
    const engine::core::TensorValue & input,
    const engine::core::TensorShape & output_shape) {
    return engine::modules::RepeatModule({output_shape}).build(ctx, input);
}

std::vector<float> build_hann_window(int64_t win_length) {
    std::vector<float> window(static_cast<size_t>(win_length), 0.0f);
    if (win_length <= 1) {
        if (win_length == 1) {
            window[0] = 1.0f;
        }
        return window;
    }

    constexpr long double kPi = 3.14159265358979323846264338327950288L;
    for (int64_t i = 0; i < win_length; ++i) {
        window[static_cast<size_t>(i)] =
            0.5f - 0.5f * std::cos(2.0L * kPi * static_cast<long double>(i) / static_cast<long double>(win_length - 1));
    }
    return window;
}

}  // namespace

struct ParakeetFrontendGraph {
    int64_t batch = 0;
    int64_t frames = 0;
    int64_t freq_bins = 0;
    int64_t feature_size = 0;
    int64_t sample_rate = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    engine::core::TensorValue power_input = {};
    engine::core::TensorValue mel_weight = {};
    engine::core::TensorValue mask = {};
    engine::core::TensorValue mean_scale = {};
    engine::core::TensorValue variance_scale = {};
    engine::core::TensorValue log_guard = {};
    engine::core::TensorValue stddev_guard = {};
    engine::core::TensorValue output = {};

    ~ParakeetFrontendGraph() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
        }
    }
};

ParakeetFrontendScratch::ParakeetFrontendScratch() = default;
ParakeetFrontendScratch::~ParakeetFrontendScratch() = default;
ParakeetFrontendScratch::ParakeetFrontendScratch(ParakeetFrontendScratch &&) noexcept = default;
ParakeetFrontendScratch & ParakeetFrontendScratch::operator=(ParakeetFrontendScratch &&) noexcept = default;

namespace {

ParakeetFrontendGraph & ensure_frontend_graph(
    const ParakeetFeatureExtractorConfig & config,
    const engine::core::ExecutionContext & execution_context,
    int64_t batch,
    int64_t frames,
    int64_t freq_bins,
    ParakeetFrontendScratch & scratch) {
    const bool graph_mismatch =
        !scratch.graph ||
        scratch.graph->backend != execution_context.backend() ||
        scratch.graph->batch != batch ||
        scratch.graph->frames != frames ||
        scratch.graph->freq_bins != freq_bins ||
        scratch.graph->feature_size != config.feature_size ||
        scratch.graph->sample_rate != config.sample_rate;

    if (!graph_mismatch) {
        return *scratch.graph;
    }

    auto graph = std::make_unique<ParakeetFrontendGraph>();
    graph->batch = batch;
    graph->frames = frames;
    graph->freq_bins = freq_bins;
    graph->feature_size = config.feature_size;
    graph->sample_rate = config.sample_rate;
    graph->backend = execution_context.backend();

    ggml_init_params params = {};
    params.mem_size = kFrontendGraphBytes;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    graph->ggml = ggml_init(params);
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Failed to initialize Parakeet frontend ggml context");
    }

    engine::core::ModuleBuildContext ctx = {};
    ctx.ggml = graph->ggml;
    ctx.module_instance_name = "parakeet_tdt_frontend";

    graph->power_input = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({batch * frames, freq_bins}));
    graph->mel_weight = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({freq_bins, config.feature_size}));
    graph->mask = engine::core::make_tensor(
        ctx,
        GGML_TYPE_I32,
        engine::core::TensorShape::from_dims({batch, frames}));
    graph->mean_scale = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({batch, 1, 1}));
    graph->variance_scale = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({batch, 1, 1}));
    graph->log_guard = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 1, 1}));
    graph->stddev_guard = engine::core::make_tensor(
        ctx,
        GGML_TYPE_F32,
        engine::core::TensorShape::from_dims({1, 1, 1}));

    auto x = engine::modules::MatMulModule().build(ctx, graph->power_input, graph->mel_weight);
    x = engine::modules::ReshapeModule({
        engine::core::TensorShape::from_dims({batch, frames, config.feature_size}),
    }).build(ctx, x);
    x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);

    const auto log_guard_rep = repeat_like(ctx, graph->log_guard, x.shape);
    x = engine::core::wrap_tensor(ggml_add(ctx.ggml, x.tensor, log_guard_rep.tensor), x.shape, GGML_TYPE_F32);
    x = engine::core::wrap_tensor(ggml_log(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);

    auto mask_f32 = engine::core::wrap_tensor(ggml_cast(ctx.ggml, graph->mask.tensor, GGML_TYPE_F32), graph->mask.shape, GGML_TYPE_F32);
    mask_f32 = engine::modules::ReshapeModule({
        engine::core::TensorShape::from_dims({batch, 1, frames}),
    }).build(ctx, mask_f32);
    const auto mask_broadcast = repeat_like(ctx, mask_f32, x.shape);

    auto masked = engine::core::wrap_tensor(ggml_mul(ctx.ggml, x.tensor, mask_broadcast.tensor), x.shape, GGML_TYPE_F32);

    const engine::core::TensorShape reduced_shape = engine::core::TensorShape::from_dims({batch, config.feature_size, 1});
    auto mean = engine::core::wrap_tensor(ggml_mean(ctx.ggml, masked.tensor), reduced_shape, GGML_TYPE_F32);
    const auto mean_scale_rep = repeat_like(ctx, graph->mean_scale, reduced_shape);
    mean = engine::core::wrap_tensor(ggml_mul(ctx.ggml, mean.tensor, mean_scale_rep.tensor), reduced_shape, GGML_TYPE_F32);

    const auto mean_rep = repeat_like(ctx, mean, x.shape);
    auto centered = engine::core::wrap_tensor(ggml_sub(ctx.ggml, x.tensor, mean_rep.tensor), x.shape, GGML_TYPE_F32);
    centered = engine::core::wrap_tensor(ggml_mul(ctx.ggml, centered.tensor, mask_broadcast.tensor), x.shape, GGML_TYPE_F32);

    auto squared = engine::core::wrap_tensor(ggml_mul(ctx.ggml, centered.tensor, centered.tensor), x.shape, GGML_TYPE_F32);
    auto variance = engine::core::wrap_tensor(ggml_mean(ctx.ggml, squared.tensor), reduced_shape, GGML_TYPE_F32);
    const auto variance_scale_rep = repeat_like(ctx, graph->variance_scale, reduced_shape);
    variance = engine::core::wrap_tensor(ggml_mul(ctx.ggml, variance.tensor, variance_scale_rep.tensor), reduced_shape, GGML_TYPE_F32);

    auto stddev = engine::core::wrap_tensor(ggml_sqrt(ctx.ggml, variance.tensor), reduced_shape, GGML_TYPE_F32);
    const auto stddev_guard_rep = repeat_like(ctx, graph->stddev_guard, reduced_shape);
    stddev = engine::core::wrap_tensor(ggml_add(ctx.ggml, stddev.tensor, stddev_guard_rep.tensor), reduced_shape, GGML_TYPE_F32);

    const auto stddev_rep = repeat_like(ctx, stddev, x.shape);
    x = engine::core::wrap_tensor(ggml_div(ctx.ggml, centered.tensor, stddev_rep.tensor), x.shape, GGML_TYPE_F32);
    x = engine::modules::TransposeModule({{0, 2, 1}, 3}).build(ctx, x);
    x = engine::core::wrap_tensor(ggml_cont(ctx.ggml, x.tensor), x.shape, GGML_TYPE_F32);

    graph->output = x;
    graph->graph = ggml_new_graph_custom(graph->ggml, kFrontendGraphNodes, false);
    ggml_build_forward_expand(graph->graph, graph->output.tensor);
    graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, execution_context.backend());
    if (graph->buffer == nullptr) {
        throw std::runtime_error("Failed to allocate Parakeet frontend backend tensors");
    }

    const auto mel_filterbank = audio::MelFilterbank().build(
        audio::MelFilterbankConfig{
            config.sample_rate,
            config.n_fft,
            config.feature_size,
            0.0f,
            static_cast<float>(config.sample_rate) / 2.0f,
            true,
        });
    std::vector<float> mel_weight_values(static_cast<size_t>(freq_bins * config.feature_size), 0.0f);
    for (int64_t mel = 0; mel < config.feature_size; ++mel) {
        for (int64_t freq = 0; freq < freq_bins; ++freq) {
            mel_weight_values[static_cast<size_t>(freq * config.feature_size + mel)] =
                mel_filterbank.values[static_cast<size_t>(mel * freq_bins + freq)];
        }
    }
    engine::core::write_tensor_f32(graph->mel_weight, mel_weight_values);
    engine::core::write_tensor_f32(graph->log_guard, std::vector<float>{kLogZeroGuard});
    engine::core::write_tensor_f32(graph->stddev_guard, std::vector<float>{kStddevGuard});

    scratch.graph = std::move(graph);
    return *scratch.graph;
}

}  // namespace

void compute_parakeet_frontend(
    const std::vector<runtime::AudioBuffer> & audio,
    const std::vector<int64_t> * audio_lengths_override,
    const ParakeetFeatureExtractorConfig & config,
    const core::ExecutionContext & execution_context,
    ParakeetFrontendBatch & output,
    ParakeetFrontendScratch & scratch) {
    if (audio.empty()) {
        throw std::runtime_error("Parakeet frontend requires at least one audio buffer");
    }
    const int64_t batch = static_cast<int64_t>(audio.size());
    int64_t max_samples = 0;
    for (const auto & item : audio) {
        if (item.sample_rate != config.sample_rate) {
            throw std::runtime_error("Parakeet frontend sample_rate mismatch");
        }
        if (item.channels != 1) {
            throw std::runtime_error("Parakeet frontend only supports mono audio");
        }
        max_samples = std::max<int64_t>(max_samples, static_cast<int64_t>(item.samples.size()));
    }

    scratch.padded.assign(static_cast<size_t>(batch * max_samples), 0.0f);
    scratch.audio_lengths.assign(static_cast<size_t>(batch), 0);
    for (int64_t b = 0; b < batch; ++b) {
        const auto & samples = audio[static_cast<size_t>(b)].samples;
        int64_t logical_length = static_cast<int64_t>(samples.size());
        if (audio_lengths_override != nullptr) {
            if (audio_lengths_override->size() != static_cast<size_t>(batch)) {
                throw std::runtime_error("Parakeet frontend audio_lengths_override size mismatch");
            }
            logical_length = (*audio_lengths_override)[static_cast<size_t>(b)];
            if (logical_length < 0 || logical_length > static_cast<int64_t>(samples.size())) {
                throw std::runtime_error("Parakeet frontend audio length override out of range");
            }
        }
        scratch.audio_lengths[static_cast<size_t>(b)] = logical_length;
        for (size_t i = 0; i < samples.size(); ++i) {
            scratch.padded[static_cast<size_t>(b * max_samples + static_cast<int64_t>(i))] = samples[i];
        }
    }

    if (config.preemphasis != 0.0f) {
        for (int64_t b = 0; b < batch; ++b) {
            const int64_t valid = scratch.audio_lengths[static_cast<size_t>(b)];
            const size_t base = static_cast<size_t>(b * max_samples);
            audio::apply_preemphasis_in_place(
                scratch.padded.data() + base,
                static_cast<size_t>(valid),
                config.preemphasis);
        }
    }

    const audio::STFTConfig stft_config{
        config.n_fft,
        config.hop_length,
        config.win_length,
        true,
        audio::STFTPadMode::Constant,
    };
    const auto & window = [&]() -> const std::vector<float> & {
        static std::vector<float> cache;
        static int64_t cached_len = -1;
        if (cached_len != config.win_length) {
            cache = build_hann_window(config.win_length);
            cached_len = config.win_length;
        }
        return cache;
    }();

    audio::STFT stft;
    const auto complex = stft.compute_complex(
        scratch.padded,
        window,
        batch,
        max_samples,
        stft_config,
        static_cast<size_t>(std::max(1, execution_context.config().threads)));
    const int64_t freq_bins = complex.shape[1];
    const int64_t frames = complex.shape[2];

    scratch.power.resize(static_cast<size_t>(batch * frames * freq_bins));
#ifdef _OPENMP
    #pragma omp parallel for collapse(3) if(batch * freq_bins * frames >= 4096)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t t = 0; t < frames; ++t) {
            for (int64_t f = 0; f < freq_bins; ++f) {
                const size_t real_idx = static_cast<size_t>(((b * freq_bins + f) * frames + t) * 2);
                const float real = complex.values[real_idx];
                const float imag = complex.values[real_idx + 1];
                scratch.power[static_cast<size_t>((b * frames + t) * freq_bins + f)] = real * real + imag * imag;
            }
        }
    }

    std::vector<int64_t> feature_lengths_i64(static_cast<size_t>(batch), 0);
    output.attention_mask.resize(static_cast<size_t>(batch * frames));
    std::vector<int32_t> attention_mask_i32(static_cast<size_t>(batch * frames), 0);
    std::vector<float> mean_scale(static_cast<size_t>(batch), 0.0f);
    std::vector<float> variance_scale(static_cast<size_t>(batch), 0.0f);
    const int64_t pad = config.n_fft / 2;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t valid_frames = std::max<int64_t>(
            0,
            (scratch.audio_lengths[static_cast<size_t>(b)] + 2 * pad - config.n_fft) / config.hop_length);
        feature_lengths_i64[static_cast<size_t>(b)] = valid_frames;
        for (int64_t t = 0; t < frames; ++t) {
            const int32_t mask_value = t < valid_frames ? 1 : 0;
            attention_mask_i32[static_cast<size_t>(b * frames + t)] = mask_value;
            output.attention_mask[static_cast<size_t>(b * frames + t)] = mask_value;
        }
        if (valid_frames > 0) {
            mean_scale[static_cast<size_t>(b)] = static_cast<float>(frames) / static_cast<float>(valid_frames);
        }
        if (valid_frames > 1) {
            variance_scale[static_cast<size_t>(b)] = static_cast<float>(frames) / static_cast<float>(valid_frames - 1);
        }
    }

    auto & graph = ensure_frontend_graph(config, execution_context, batch, frames, freq_bins, scratch);
    core::write_tensor_f32(graph.power_input, scratch.power);
    core::write_tensor_i32(graph.mask, attention_mask_i32);
    core::write_tensor_f32(graph.mean_scale, mean_scale.data(), mean_scale.size());
    core::write_tensor_f32(graph.variance_scale, variance_scale.data(), variance_scale.size());

    core::set_backend_threads(execution_context.backend(), std::max(1, execution_context.config().threads));
    engine::core::compute_backend_graph(execution_context.backend(), graph.graph);
    core::read_tensor_f32_into(graph.output.tensor, output.features);

    output.feature_lengths.clear();
    output.feature_lengths.reserve(static_cast<size_t>(batch));
    for (const int64_t value : feature_lengths_i64) {
        output.feature_lengths.push_back(static_cast<int32_t>(value));
    }
    output.batch = batch;
    output.frames = frames;
    output.feature_dim = config.feature_size;
}

}  // namespace engine::models::parakeet_tdt
