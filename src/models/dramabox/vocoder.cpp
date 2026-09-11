#include "engine/models/dramabox/vocoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/vocoders/bigvgan_vocoder.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::dramabox {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kVocoderWeightContextBytes = 1536ull * 1024ull * 1024ull;
constexpr size_t kVocoderGraphContextBytes = 384ull * 1024ull * 1024ull;
constexpr size_t kVocoderGraphNodeCapacity = 65536;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> expand_channel_filter(const std::vector<float> & filter, int64_t channels) {
    if (filter.empty() || channels <= 0) {
        throw std::runtime_error("DramaBox vocoder filter shape mismatch");
    }
    std::vector<float> expanded(static_cast<size_t>(channels * static_cast<int64_t>(filter.size())), 0.0F);
    for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy(
            filter.begin(),
            filter.end(),
            expanded.begin() + static_cast<std::ptrdiff_t>(channel * static_cast<int64_t>(filter.size())));
    }
    return expanded;
}

float sinc(float x) {
    if (x == 0.0F) {
        return 1.0F;
    }
    constexpr float pi = 3.14159265358979323846F;
    return std::sin(pi * x) / (pi * x);
}

std::vector<float> make_hann_resampler_filter(int64_t ratio) {
    constexpr float rolloff = 0.99F;
    constexpr float lowpass_filter_width = 6.0F;
    const int64_t width = static_cast<int64_t>(std::ceil(lowpass_filter_width / rolloff));
    const int64_t kernel_size = 2 * width * ratio + 1;
    std::vector<float> filter(static_cast<size_t>(kernel_size), 0.0F);
    constexpr float pi = 3.14159265358979323846F;
    for (int64_t i = 0; i < kernel_size; ++i) {
        const float time_axis = (static_cast<float>(i) / static_cast<float>(ratio) - static_cast<float>(width)) * rolloff;
        const float clamped = std::max(-lowpass_filter_width, std::min(lowpass_filter_width, time_axis));
        const float window = std::pow(std::cos(clamped * pi / lowpass_filter_width / 2.0F), 2.0F);
        filter[static_cast<size_t>(i)] = sinc(time_axis) * window * rolloff / static_cast<float>(ratio);
    }
    return filter;
}

modules::BigVganVocoderConfig make_bigvgan_config(
    assets::TensorStorageType storage_type,
    int64_t sample_rate,
    int64_t initial_channel,
    const std::vector<int64_t> & upsample_rates,
    const std::vector<int64_t> & upsample_kernel_sizes) {
    return {
        sample_rate,
        128,
        1,
        1,
        1,
        initial_channel,
        true,
        upsample_rates,
        upsample_kernel_sizes,
        {3, 7, 11},
        storage_type,
    };
}

std::vector<float> convert_transpose_conv1d_weight(
    const std::vector<float> & weight,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel) {
    if (static_cast<int64_t>(weight.size()) != in_channels * out_channels * kernel) {
        throw std::runtime_error("DramaBox BigVGAN transposed-conv weight shape mismatch");
    }
    std::vector<float> converted(static_cast<size_t>(out_channels * in_channels * kernel), 0.0F);
    for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
            for (int64_t tap = 0; tap < kernel; ++tap) {
                const size_t src = static_cast<size_t>((in_channel * out_channels + out_channel) * kernel + tap);
                const size_t dst =
                    static_cast<size_t>((out_channel * in_channels + in_channel) * kernel + (kernel - 1 - tap));
                converted[dst] = weight[src];
            }
        }
    }
    return converted;
}

void convert_direct_bigvgan_upsample_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    modules::BigVganVocoderWeights & weights) {
    for (size_t index = 0; index < weights.ups.size(); ++index) {
        auto & up = weights.ups[index];
        const std::string weight_name = prefix + ".ups." + std::to_string(index) + ".weight";
        const auto weight = source.require_f32(weight_name, {up.in_channels, up.out_channels, up.kernel});
        up.conv1d_weight = store.make_from_f32(
            core::TensorShape::from_dims({up.out_channels, up.in_channels, up.kernel}),
            assets::TensorStorageType::F32,
            convert_transpose_conv1d_weight(weight, up.in_channels, up.out_channels, up.kernel));
        up.transpose_weight.reset();
    }
}

ggml_tensor * repeat_frame(ggml_context * ctx, ggml_tensor * x, int64_t frame, int64_t count) {
    ggml_tensor * view = ggml_view_2d(ctx, x, 1, x->ne[1], x->nb[1], static_cast<size_t>(frame) * x->nb[0]);
    return ggml_repeat_4d(ctx, view, count, x->ne[1], 1, 1);
}

ggml_tensor * replicate_pad(ggml_context * ctx, ggml_tensor * x, int64_t left, int64_t right) {
    ggml_tensor * out = x;
    if (left > 0) {
        out = ggml_concat(ctx, repeat_frame(ctx, x, 0, left), out, 0);
    }
    if (right > 0) {
        out = ggml_concat(ctx, out, repeat_frame(ctx, x, x->ne[0] - 1, right), 0);
    }
    return out;
}

ggml_tensor * zero_pad_right(ggml_context * ctx, ggml_tensor * x, int64_t right) {
    if (right <= 0) {
        return x;
    }
    ggml_tensor * zero = ggml_scale(ctx, repeat_frame(ctx, x, x->ne[0] - 1, right), 0.0F);
    return ggml_concat(ctx, x, zero, 0);
}

ggml_tensor * build_stereo_mel_from_wave(
    ggml_context * ctx,
    ggml_tensor * low_wave,
    const DramaBoxVocoderWeights & weights,
    int64_t padded_samples) {
    ggml_tensor * left = ggml_scale(ctx, repeat_frame(ctx, low_wave, 0, 432), 0.0F);
    ggml_tensor * x = ggml_concat(ctx, left, low_wave, 0);
    x = zero_pad_right(ctx, x, padded_samples - low_wave->ne[0]);
    ggml_tensor * x3 = ggml_reshape_3d(ctx, core::has_backend_addressable_layout(x) ? x : ggml_cont(ctx, x), x->ne[0], 1, x->ne[1]);
    ggml_tensor * spec = ggml_conv_1d_fast_1d_im2col(ctx, weights.stft_forward_basis.tensor, x3, 80, 0, 1);
    ggml_tensor * real = ggml_view_3d(ctx, spec, spec->ne[0], 257, spec->ne[2], spec->nb[1], spec->nb[2], 0);
    ggml_tensor * imag = ggml_view_3d(ctx, spec, spec->ne[0], 257, spec->ne[2], spec->nb[1], spec->nb[2], 257 * spec->nb[1]);
    ggml_tensor * mag = ggml_sqrt(
        ctx,
        ggml_add(
            ctx,
            ggml_sqr(ctx, core::has_backend_addressable_layout(real) ? real : ggml_cont(ctx, real)),
            ggml_sqr(ctx, core::has_backend_addressable_layout(imag) ? imag : ggml_cont(ctx, imag))));
    ggml_tensor * mag_for_mel = ggml_cont(ctx, ggml_transpose(ctx, mag));
    ggml_tensor * mel = ggml_mul_mat(ctx, weights.mel_basis.tensor, mag_for_mel);
    mel = ggml_cont(ctx, ggml_transpose(ctx, mel));
    mel = ggml_clamp(ctx, mel, 1.0e-5F, INFINITY);
    mel = ggml_log(ctx, mel);
    mel = core::has_backend_addressable_layout(mel) ? mel : ggml_cont(ctx, mel);
    return ggml_reshape_2d(ctx, mel, mel->ne[0], mel->ne[1] * mel->ne[2]);
}

ggml_tensor * build_resampler_skip(
    ggml_context * ctx,
    core::BackendType backend_type,
    ggml_tensor * low_wave,
    const DramaBoxVocoderWeights & weights,
    int64_t padded_samples) {
    const int64_t ratio = 3;
    const int64_t width = 7;
    const int64_t kernel_size = 2 * width * ratio + 1;
    const int64_t pad_left = 2 * width * ratio;
    const int64_t pad_right = kernel_size - ratio;
    ggml_tensor * x = zero_pad_right(ctx, low_wave, padded_samples - low_wave->ne[0]);
    x = replicate_pad(ctx, x, width, width);
    core::ModuleBuildContext build_ctx{ctx, "resampler_skip", backend_type};
    const auto input = core::wrap_tensor(
        x,
        core::TensorShape::from_dims({x->ne[1], x->ne[0]}),
        GGML_TYPE_F32);
    const auto output = modules::DepthwiseConvTranspose1dModule({
        x->ne[1],
        weights.resampler_filter.shape.dims[3],
        static_cast<int>(ratio),
        false,
    }).build(build_ctx, input, {weights.resampler_filter, std::nullopt});
    ggml_tensor * up = output.tensor;
    up = ggml_scale(ctx, up, static_cast<float>(ratio));
    return ggml_view_2d(
        ctx,
        up,
        up->ne[0] - pad_left - pad_right,
        up->ne[1],
        up->nb[1],
        static_cast<size_t>(pad_left) * up->nb[0]);
}

}  // namespace

DramaBoxVocoderWeights load_dramabox_vocoder_weights(
    const DramaBoxAssets & assets,
    ggml_backend_t backend,
    core::BackendType backend_type,
    size_t weight_context_bytes,
    assets::TensorStorageType weight_storage_type) {
    const auto & source = *assets.audio_weights;
    (void)weight_storage_type;
    constexpr auto vocoder_storage_type = assets::TensorStorageType::F32;
    DramaBoxVocoderWeights weights;
    weights.store = std::make_shared<core::BackendWeightStore>(
        backend,
        backend_type,
        "dramabox.vocoder.weights",
        weight_context_bytes == 0 ? kVocoderWeightContextBytes : weight_context_bytes);
    weights.vocoder = modules::load_direct_bigvgan_from_tensor_source(
        *weights.store,
        source,
        "vocoder.vocoder",
        make_bigvgan_config(
            vocoder_storage_type,
            assets.config.vocoder.input_sample_rate,
            1536,
            {5, 2, 2, 2, 2, 2},
            {11, 4, 4, 4, 4, 4}),
        modules::BigVganActivationLayout::GroupedByStage);
    weights.bwe = modules::load_direct_bigvgan_from_tensor_source(
        *weights.store,
        source,
        "vocoder.bwe_generator",
        make_bigvgan_config(
            vocoder_storage_type,
            assets.config.vocoder.output_sample_rate,
            assets.config.vocoder.bwe_initial_channel,
            assets.config.vocoder.bwe_upsample_rates,
            assets.config.vocoder.bwe_upsample_kernel_sizes),
        modules::BigVganActivationLayout::GroupedByStage);
    convert_direct_bigvgan_upsample_weights(*weights.store, source, "vocoder.vocoder", weights.vocoder);
    convert_direct_bigvgan_upsample_weights(*weights.store, source, "vocoder.bwe_generator", weights.bwe);
    weights.mel_basis = weights.store->load_tensor(
        source,
        "vocoder.mel_stft.mel_basis",
        assets::TensorStorageType::F32,
        {assets.config.vocoder.num_mels, assets.config.vocoder.n_fft / 2 + 1});
    weights.stft_forward_basis = weights.store->load_tensor(
        source,
        "vocoder.mel_stft.stft_fn.forward_basis",
        assets::TensorStorageType::F32,
        {assets.config.vocoder.n_fft + 2, 1, assets.config.vocoder.n_fft});
    weights.resampler_filter = weights.store->make_from_f32(
        core::TensorShape::from_dims({2, 1, 1, 43}),
        assets::TensorStorageType::F32,
        expand_channel_filter(make_hann_resampler_filter(3), 2));
    weights.store->upload();
    return weights;
}

class DramaBoxVocoderRuntime::VocoderGraph {
public:
    VocoderGraph(core::ExecutionContext & execution, const DramaBoxVocoderWeights & weights, int64_t mel_frames)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(weights),
          mel_frames_(mel_frames) {
        build();
    }

    ~VocoderGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t mel_frames) const noexcept {
        return mel_frames == mel_frames_;
    }

    void run(const std::vector<float> & mel) const {
        const auto input_start = Clock::now();
        core::write_tensor_f32(mel_value_, mel);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.vocoder16.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(nullptr);
    }

    void run_from_device(const ggml_tensor * mel) const {
        if (mel == nullptr ||
            mel->type != input_->type ||
            mel->ne[0] != input_->ne[0] ||
            mel->ne[1] != input_->ne[1] ||
            mel->ne[2] != input_->ne[2] ||
            mel->ne[3] != input_->ne[3]) {
            throw std::runtime_error("DramaBox vocoder device mel shape mismatch");
        }
        const auto input_start = Clock::now();
        ggml_backend_tensor_copy(mel, input_);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.vocoder16.device_input_copy_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(nullptr);
    }

    void compute(std::vector<float> * out) const {
        const auto compute_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.vocoder16") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox vocoder graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.vocoder16.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        if (out != nullptr) {
            const auto read_start = Clock::now();
            core::read_tensor_f32_into(output_, *out);
            debug::timing_log_scalar("dramabox.vocoder16.output_read_ms", debug::elapsed_ms(read_start, Clock::now()));
        }
    }

    int64_t samples() const noexcept {
        return output_->ne[0];
    }

    const ggml_tensor * output_tensor() const noexcept {
        return output_;
    }

private:
    void build() {
        const auto build_start = Clock::now();
        ggml_init_params params{kVocoderGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox vocoder ggml context initialization failed");
        }
        input_ = ggml_new_tensor_4d(ctx_.get(), GGML_TYPE_F32, 64, mel_frames_, 2, 1);
        mel_value_ = core::wrap_tensor(input_, core::TensorShape::from_dims({1, 2, mel_frames_, 64}), GGML_TYPE_F32);
        ggml_set_input(input_);
        mel_ = ggml_reshape_2d(ctx_.get(), ggml_cont(ctx_.get(), ggml_permute(ctx_.get(), input_, 1, 0, 2, 3)), mel_frames_, 128);
        output_ = modules::build_bigvgan_graph(
            ctx_.get(),
            backend_type_,
            weights_.vocoder,
            mel_,
            {true, false, modules::BigVganActivationLayout::GroupedByStage, true});
        output_ = core::has_backend_addressable_layout(output_) ? output_ : ggml_cont(ctx_.get(), output_);
        ggml_set_output(output_);
        const auto expand_start = Clock::now();
        graph_ = ggml_new_graph_custom(ctx_.get(), kVocoderGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        debug::timing_log_scalar("dramabox.vocoder16.graph.expand_ms", debug::elapsed_ms(expand_start, Clock::now()));
        debug::trace_log_scalar("dramabox.vocoder16.graph.nodes", static_cast<int64_t>(ggml_graph_n_nodes(graph_)));
        const auto alloc_start = Clock::now();
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox vocoder backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.vocoder16.graph.alloc_ms", debug::elapsed_ms(alloc_start, Clock::now()));
        debug::timing_log_scalar("dramabox.vocoder16.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    const DramaBoxVocoderWeights & weights_;
    int64_t mel_frames_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * mel_ = nullptr;
    core::TensorValue mel_value_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

class DramaBoxVocoderRuntime::BweGraph {
public:
    BweGraph(core::ExecutionContext & execution, const DramaBoxVocoderWeights & weights, int64_t low_samples)
        : backend_(execution.backend()),
          backend_type_(execution.backend_type()),
          threads_(std::max(1, execution.config().threads)),
          weights_(weights),
          low_samples_(low_samples) {
        build();
    }

    ~BweGraph() {
        if (backend_ != nullptr && graph_ != nullptr) {
            core::release_backend_graph_resources(backend_type_, backend_, graph_);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    bool matches(int64_t low_samples) const noexcept {
        return low_samples == low_samples_;
    }

    void run(const std::vector<float> & low_wave, std::vector<float> & out) const {
        const auto input_start = Clock::now();
        core::write_tensor_f32(input_value_, low_wave);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.bwe.input_upload_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(out);
    }

    void run_from_device(const ggml_tensor * low_wave, std::vector<float> & out) const {
        if (low_wave == nullptr ||
            low_wave->type != input_->type ||
            low_wave->ne[0] != input_->ne[0] ||
            low_wave->ne[1] != input_->ne[1] ||
            low_wave->ne[2] != input_->ne[2] ||
            low_wave->ne[3] != input_->ne[3]) {
            throw std::runtime_error("DramaBox BWE device waveform shape mismatch");
        }
        const auto input_start = Clock::now();
        ggml_backend_tensor_copy(low_wave, input_);
        core::set_backend_threads(backend_, threads_);
        debug::timing_log_scalar("dramabox.bwe.device_input_copy_ms", debug::elapsed_ms(input_start, Clock::now()));
        compute(out);
    }

    void compute(std::vector<float> & out) const {
        const auto compute_start = Clock::now();
        if (core::compute_backend_graph(backend_, graph_, nullptr, "dramabox.bwe") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DramaBox BWE graph compute failed");
        }
        ggml_backend_synchronize(backend_);
        debug::timing_log_scalar("dramabox.bwe.graph.compute_ms", debug::elapsed_ms(compute_start, Clock::now()));
        const auto read_start = Clock::now();
        core::read_tensor_f32_into(output_, out);
        debug::timing_log_scalar("dramabox.bwe.output_read_ms", debug::elapsed_ms(read_start, Clock::now()));
    }

private:
    void build() {
        const auto build_start = Clock::now();
        padded_samples_ = low_samples_;
        const int64_t remainder = padded_samples_ % 80;
        if (remainder != 0) {
            padded_samples_ += 80 - remainder;
        }
        output_samples_ = low_samples_ * 3;
        ggml_init_params params{kVocoderGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("DramaBox BWE ggml context initialization failed");
        }
        input_ = ggml_new_tensor_2d(ctx_.get(), GGML_TYPE_F32, low_samples_, 2);
        input_value_ = core::wrap_tensor(input_, core::TensorShape::from_dims({2, low_samples_}), GGML_TYPE_F32);
        ggml_set_input(input_);
        ggml_tensor * bwe_mel = build_stereo_mel_from_wave(ctx_.get(), input_, weights_, padded_samples_);
        ggml_tensor * residual = modules::build_bigvgan_graph(
            ctx_.get(),
            backend_type_,
            weights_.bwe,
            bwe_mel,
            {false, false, modules::BigVganActivationLayout::GroupedByStage, true});
        ggml_tensor * skip = build_resampler_skip(ctx_.get(), backend_type_, input_, weights_, padded_samples_);
        ggml_tensor * summed = ggml_clamp(ctx_.get(), ggml_add(ctx_.get(), residual, skip), -1.0F, 1.0F);
        output_ = ggml_view_2d(ctx_.get(), summed, output_samples_, summed->ne[1], summed->nb[1], 0);
        output_ = core::has_backend_addressable_layout(output_) ? output_ : ggml_cont(ctx_.get(), output_);
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), kVocoderGraphNodeCapacity, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("DramaBox BWE backend buffer allocation failed");
        }
        debug::timing_log_scalar("dramabox.bwe.graph.build_ms", debug::elapsed_ms(build_start, Clock::now()));
    }

    ggml_backend_t backend_ = nullptr;
    core::BackendType backend_type_ = core::BackendType::Cpu;
    int threads_ = 1;
    const DramaBoxVocoderWeights & weights_;
    int64_t low_samples_ = 0;
    int64_t padded_samples_ = 0;
    int64_t output_samples_ = 0;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_tensor * input_ = nullptr;
    core::TensorValue input_value_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
};

DramaBoxVocoderRuntime::DramaBoxVocoderRuntime(
    core::ExecutionContext & execution,
    std::shared_ptr<const DramaBoxAssets> assets,
    assets::TensorStorageType weight_storage_type)
    : execution_(&execution),
      assets_(std::move(assets)),
      weight_storage_type_(weight_storage_type) {
    if (execution_ == nullptr) {
        throw std::runtime_error("DramaBox vocoder runtime requires execution context");
    }
    if (assets_ == nullptr) {
        throw std::runtime_error("DramaBox vocoder runtime requires assets");
    }
}

DramaBoxVocoderRuntime::~DramaBoxVocoderRuntime() = default;

void DramaBoxVocoderRuntime::prepare(int64_t mel_frames) const {
    if (!weights_) {
        weights_ = std::make_unique<DramaBoxVocoderWeights>(load_dramabox_vocoder_weights(
            *assets_,
            execution_->backend(),
            execution_->backend_type(),
            0,
            weight_storage_type_));
    }
    if (!vocoder_graph_ || !vocoder_graph_->matches(mel_frames)) {
        vocoder_graph_.reset();
        vocoder_graph_ = std::make_unique<VocoderGraph>(*execution_, *weights_, mel_frames);
    }
    const int64_t low_samples = vocoder_graph_->samples();
    if (!bwe_graph_ || !bwe_graph_->matches(low_samples)) {
        bwe_graph_.reset();
        bwe_graph_ = std::make_unique<BweGraph>(*execution_, *weights_, low_samples);
    }
}

DramaBoxVocoderOutput DramaBoxVocoderRuntime::synthesize(const DramaBoxDecodedMel & mel) const {
    if (mel.batch != 1 || mel.channels != 2 || mel.mel_bins != 64 ||
        (mel.device_values == nullptr &&
         static_cast<int64_t>(mel.values.size()) != mel.batch * mel.channels * mel.frames * mel.mel_bins)) {
        throw std::runtime_error("DramaBox vocoder received invalid mel dimensions");
    }
    prepare(mel.frames);
    if (mel.device_values != nullptr) {
        vocoder_graph_->run_from_device(mel.device_values);
    } else {
        vocoder_graph_->run(mel.values);
    }
    DramaBoxVocoderOutput out;
    bwe_graph_->run_from_device(vocoder_graph_->output_tensor(), out.waveform);
    out.channels = 2;
    out.samples = static_cast<int64_t>(out.waveform.size()) / 2;
    out.sample_rate = assets_->config.vocoder.output_sample_rate;
    return out;
}

void DramaBoxVocoderRuntime::release_runtime_state() const {
    bwe_graph_.reset();
    vocoder_graph_.reset();
    weights_.reset();
}

}  // namespace engine::models::dramabox
