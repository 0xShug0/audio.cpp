#include "engine/community_models/kitten_tts/decoder.h"

#include "engine/community_models/kitten_tts/assets.h"

#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::models::kitten_tts {

namespace {

namespace audio = engine::audio;
namespace core = engine::core;
namespace modules = engine::modules;

constexpr float kPi = 3.14159265358979323846f;

using engine::debug::measure_ms;

void set_graph_output(ggml_tensor *tensor) {
    ggml_set_output(tensor);
    for (ggml_tensor *backing = tensor->view_src; backing != nullptr; backing = backing->view_src) {
        ggml_set_output(backing);
    }
}

struct TimeMaskInputs {
    ggml_tensor *keep = nullptr;
    ggml_tensor *norm = nullptr;
    int64_t frame_capacity = 0;
};

const TimeMaskInputs &add_time_mask_inputs(ggml_context *ctx, std::vector<TimeMaskInputs> &masks,
                                           int64_t frame_capacity) {
    if (frame_capacity <= 0) {
        throw std::runtime_error("kitten decoder mask capacity must be positive");
    }
    TimeMaskInputs mask = {};
    mask.keep = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frame_capacity, 1, 1);
    mask.norm = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frame_capacity, 1, 1);
    mask.frame_capacity = frame_capacity;
    ggml_set_input(mask.keep);
    ggml_set_input(mask.norm);
    masks.push_back(mask);
    return masks.back();
}

ggml_tensor *build_masked_adain_bct(core::ModuleBuildContext &build_ctx, ggml_tensor *x, const core::TensorValue &gamma,
                                    const core::TensorValue &beta, int64_t channels, float eps,
                                    std::vector<TimeMaskInputs> &masks) {
    const TimeMaskInputs &mask = add_time_mask_inputs(build_ctx.ggml, masks, x->ne[0]);
    const auto x_value =
        core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[2], channels, x->ne[0]}), GGML_TYPE_F32);
    const auto keep = modules::RepeatModule({x_value.shape})
                          .build(build_ctx, core::wrap_tensor(mask.keep, core::TensorShape::from_dims({1, 1, x->ne[0]}),
                                                              GGML_TYPE_F32));
    const auto norm = modules::RepeatModule({x_value.shape})
                          .build(build_ctx, core::wrap_tensor(mask.norm, core::TensorShape::from_dims({1, 1, x->ne[0]}),
                                                              GGML_TYPE_F32));
    const auto masked = modules::MulModule{}.build(build_ctx, x_value, norm);
    const auto mean = modules::ReduceMeanModule({2}).build(build_ctx, masked);
    const auto mean_rep = modules::RepeatModule({x_value.shape}).build(build_ctx, mean);
    const auto centered =
        core::wrap_tensor(ggml_sub(build_ctx.ggml, x_value.tensor, mean_rep.tensor), x_value.shape, GGML_TYPE_F32);
    const auto centered_for_variance = modules::MulModule{}.build(build_ctx, centered, norm);
    const auto squared = modules::MulModule{}.build(build_ctx, centered, centered_for_variance);
    const auto variance = modules::ReduceMeanModule({2}).build(build_ctx, squared);
    const auto stddev = modules::SqrtModule{}.build(
        build_ctx,
        core::wrap_tensor(ggml_scale_bias(build_ctx.ggml, variance.tensor, 1.0f, eps), variance.shape, GGML_TYPE_F32));
    const auto stddev_rep = modules::RepeatModule({x_value.shape}).build(build_ctx, stddev);
    auto normalized =
        core::wrap_tensor(ggml_div(build_ctx.ggml, centered.tensor, stddev_rep.tensor), x_value.shape, GGML_TYPE_F32);
    normalized = modules::MulModule{}.build(build_ctx, normalized, keep);
    core::validate_shape(gamma, core::TensorShape::from_dims({channels}), "kitten decoder adain gamma");
    core::validate_shape(beta, core::TensorShape::from_dims({channels}), "kitten decoder adain beta");
    const auto gamma_bc = core::reshape_tensor(build_ctx, gamma, core::TensorShape::from_dims({1, channels, 1}));
    const auto beta_bc = core::reshape_tensor(build_ctx, beta, core::TensorShape::from_dims({1, channels, 1}));
    const auto scaled = modules::MulModule{}.build(build_ctx, normalized, gamma_bc);
    const auto out = modules::AddModule{}.build(build_ctx, scaled, beta_bc);
    return modules::MulModule{}.build(build_ctx, out, keep).tensor;
}

void upload_time_masks(const std::vector<TimeMaskInputs> &masks, int64_t valid_base_frames,
                       int64_t base_frame_capacity) {
    if (valid_base_frames <= 0 || valid_base_frames > base_frame_capacity) {
        throw std::runtime_error("kitten decoder valid frame count exceeds prepared capacity");
    }
    for (const TimeMaskInputs &mask : masks) {
        if (mask.frame_capacity <= 0 || mask.frame_capacity < base_frame_capacity) {
            std::ostringstream message;
            message << "kitten decoder mask capacity is not aligned with graph capacity"
                    << " mask_frames=" << mask.frame_capacity << " valid_base_frames=" << valid_base_frames
                    << " base_frame_capacity=" << base_frame_capacity;
            throw std::runtime_error(message.str());
        }
        const int64_t scale = mask.frame_capacity / base_frame_capacity;
        const int64_t offset = mask.frame_capacity % base_frame_capacity;
        if (scale <= 0 || offset > 1) {
            std::ostringstream message;
            message << "kitten decoder mask capacity is not represented by the "
                       "generator length schedule"
                    << " mask_frames=" << mask.frame_capacity << " base_frame_capacity=" << base_frame_capacity
                    << " scale=" << scale << " offset=" << offset;
            throw std::runtime_error(message.str());
        }
        const int64_t valid_frames = valid_base_frames * scale + offset;
        if (valid_frames <= 0 || valid_frames > mask.frame_capacity) {
            throw std::runtime_error("kitten decoder mask valid frame count is invalid");
        }
        std::vector<float> keep(static_cast<size_t>(mask.frame_capacity), 0.0f);
        std::vector<float> norm(static_cast<size_t>(mask.frame_capacity), 0.0f);
        std::fill(keep.begin(), keep.begin() + valid_frames, 1.0f);
        const float norm_value = static_cast<float>(mask.frame_capacity) / static_cast<float>(valid_frames);
        std::fill(norm.begin(), norm.begin() + valid_frames, norm_value);
        core::write_tensor_f32(
            core::wrap_tensor(mask.keep, core::TensorShape::from_dims({1, 1, mask.frame_capacity}), GGML_TYPE_F32),
            keep);
        if (mask.norm != nullptr) {
            core::write_tensor_f32(
                core::wrap_tensor(mask.norm, core::TensorShape::from_dims({1, 1, mask.frame_capacity}), GGML_TYPE_F32),
                norm);
        }
    }
}

ggml_tensor *build_adaptive_instance_norm_bct_decoder(ggml_context *ctx, ggml_tensor *x,
                                                      const KittenWeights::AdaIn1dWeights &weights,
                                                      const core::TensorValue &style,
                                                      std::vector<TimeMaskInputs> &masks, bool use_time_masks) {
    const int64_t channels = x->ne[1];
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto affine = modules::LinearModule({weights.fc.in_features, weights.fc.out_features, weights.fc.use_bias})
                            .build(build_ctx, style,
                                   {
                                       weights.fc.weight,
                                       weights.fc.bias,
                                   });
    const auto scale_delta = modules::SliceModule({1, 0, channels}).build(build_ctx, affine);
    const auto shift = modules::SliceModule({1, channels, channels}).build(build_ctx, affine);
    const auto gamma_2d =
        core::wrap_tensor(ggml_scale_bias(ctx, scale_delta.tensor, 1.0f, 1.0f), scale_delta.shape, GGML_TYPE_F32);
    const auto gamma = core::wrap_tensor(ggml_reshape_1d(ctx, gamma_2d.tensor, channels),
                                         core::TensorShape::from_dims({channels}), GGML_TYPE_F32);
    const auto beta = core::wrap_tensor(ggml_reshape_1d(ctx, shift.tensor, channels),
                                        core::TensorShape::from_dims({channels}), GGML_TYPE_F32);
    if (use_time_masks) {
        return build_masked_adain_bct(build_ctx, x, gamma, beta, channels, weights.eps, masks);
    }
    const auto input =
        core::wrap_tensor(x, core::TensorShape::from_dims({x->ne[2], channels, x->ne[0]}), GGML_TYPE_F32);
    return modules::AdaptiveInstanceNorm1dModule({channels, weights.eps}).build(build_ctx, input, {gamma, beta}).tensor;
}

template <typename ConvWeightsT> modules::Conv1dWeights make_conv1d_weights(const ConvWeightsT &conv) {
    modules::Conv1dWeights weights = {};
    weights.weight = conv.weight;
    if (conv.use_bias) {
        weights.bias = conv.bias;
    }
    return weights;
}

modules::ConvTranspose1dWeights
make_conv_transpose1d_weights(const KittenWeights::WeightNormConvTranspose1dWeights &conv) {
    modules::ConvTranspose1dWeights weights = {};
    weights.weight = conv.groups == 1 ? conv.weight : conv.dense_weight;
    if (conv.use_bias) {
        weights.bias = conv.bias;
    }
    return weights;
}

template <typename ConvWeightsT>
ggml_tensor *build_decoder_conv1d_bct(ggml_context *ctx, ggml_tensor *input, const ConvWeightsT &conv) {
    if (conv.groups != 1) {
        throw std::runtime_error("kitten decoder conv1d requires groups == 1");
    }
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto input_bct = core::wrap_tensor(
        ggml_cont(ctx, input), core::TensorShape::from_dims({input->ne[2], input->ne[1], input->ne[0]}), GGML_TYPE_F32);
    return modules::Conv1dModule({conv.in_channels, conv.out_channels, conv.kernel, static_cast<int>(conv.stride),
                                  static_cast<int>(conv.padding), static_cast<int>(conv.dilation), conv.use_bias})
        .build(build_ctx, input_bct, make_conv1d_weights(conv))
        .tensor;
}

ggml_tensor *build_conv_transpose1d_bct_decoder(ggml_context *ctx, ggml_tensor *input,
                                                const KittenWeights::WeightNormConvTranspose1dWeights &conv) {
    if (conv.groups != 1) {
        throw std::runtime_error("kitten decoder conv_transpose1d requires groups == 1");
    }
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    const auto input_bct = core::wrap_tensor(
        ggml_cont(ctx, input), core::TensorShape::from_dims({input->ne[2], input->ne[1], input->ne[0]}), GGML_TYPE_F32);
    auto output_bct = modules::ConvTranspose1dModule({conv.in_channels, conv.out_channels, conv.kernel,
                                                      static_cast<int>(conv.stride), 0, 1, conv.use_bias})
                          .build(build_ctx, input_bct, make_conv_transpose1d_weights(conv));
    const int64_t cropped_len = (input->ne[0] - 1) * conv.stride - 2 * conv.padding + conv.kernel + conv.output_padding;
    const auto cropped = core::wrap_tensor(
        ggml_cont(ctx, ggml_view_3d(ctx, output_bct.tensor, cropped_len, conv.out_channels, 1, output_bct.tensor->nb[1],
                                    output_bct.tensor->nb[2], static_cast<size_t>(conv.padding) * sizeof(float))),
        core::TensorShape::from_dims({1, conv.out_channels, cropped_len}), GGML_TYPE_F32);
    return cropped.tensor;
}

struct DeterministicRng {
    uint64_t state = 0;
    bool has_spare = false;
    float spare = 0.0f;

    explicit DeterministicRng(uint64_t seed) : state(seed) {}

    uint32_t next_u32() {
        state = state * 6364136223846793005ULL + 1ULL;
        return static_cast<uint32_t>(state >> 32);
    }

    float uniform01() { return (static_cast<float>(next_u32()) + 0.5f) / 4294967296.0f; }

    float normal() {
        if (has_spare) {
            has_spare = false;
            return spare;
        }
        const float u1 = std::max(uniform01(), 1.0e-12f);
        const float u2 = uniform01();
        const float radius = std::sqrt(-2.0f * std::log(u1));
        const float theta = 2.0f * kPi * u2;
        spare = radius * std::sin(theta);
        has_spare = true;
        return radius * std::cos(theta);
    }
};

struct HarmonicConditioning {
    std::vector<float> features;
    int64_t feature_rows = 0;
    int64_t feature_cols = 0;
    int64_t valid_feature_cols = 0;
};

class SourceSignalGraphRuntime {
  public:
    SourceSignalGraphRuntime(const KittenWeights::GeneratorWeights &generator, ggml_backend_t backend, int n_threads)
        : generator_(&generator), backend_(backend), n_threads_(std::max(1, n_threads)) {}

    void run(const std::vector<float> &phase, const std::vector<float> &noise, const std::vector<float> &voiced,
             int64_t sample_count, int64_t harmonic_dims, std::vector<float> &output) {
        if (sample_count <= 0 || harmonic_dims <= 0) {
            throw std::runtime_error("Kitten source graph dimensions must be positive");
        }
        if (static_cast<int64_t>(phase.size()) != sample_count * harmonic_dims ||
            static_cast<int64_t>(noise.size()) != sample_count * harmonic_dims ||
            static_cast<int64_t>(voiced.size()) != sample_count) {
            throw std::runtime_error("Kitten source graph input shape mismatch");
        }
        prepare(sample_count, harmonic_dims);
        session_->run(phase, noise, voiced, output);
    }

  private:
    struct Session {
        const KittenWeights::GeneratorWeights *generator = nullptr;
        ggml_backend_t backend = nullptr;
        int n_threads = 1;
        int64_t sample_count = 0;
        int64_t harmonic_dims = 0;
        ggml_context *ctx = nullptr;
        ggml_tensor *phase_in = nullptr;
        ggml_tensor *noise_in = nullptr;
        ggml_tensor *voiced_in = nullptr;
        ggml_tensor *output = nullptr;
        ggml_cgraph *graph = nullptr;
        ggml_gallocr_t gallocr = nullptr;

        Session(const KittenWeights::GeneratorWeights &generator_in, ggml_backend_t backend_in, int n_threads_in,
                int64_t sample_count_in, int64_t harmonic_dims_in)
            : generator(&generator_in), backend(backend_in), n_threads(std::max(1, n_threads_in)),
              sample_count(sample_count_in), harmonic_dims(harmonic_dims_in) {
            if (generator->source_linear.weight.empty() ||
                static_cast<int64_t>(generator->source_linear.weight.size()) != harmonic_dims ||
                (generator->source_linear.use_bias && generator->source_linear.bias.empty())) {
                throw std::runtime_error("kitten decoder source affine weights are missing");
            }
            ggml_init_params params{
                /*.mem_size   =*/64ull * 1024ull * 1024ull,
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };
            ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to initialize ggml context for Kitten source graph");
            }
            try {
                phase_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sample_count, harmonic_dims);
                noise_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sample_count, harmonic_dims);
                voiced_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sample_count);
                ggml_set_input(phase_in);
                ggml_set_input(noise_in);
                ggml_set_input(voiced_in);

                core::ModuleBuildContext build_ctx = {};
                build_ctx.ggml = ctx;
                const auto source_shape = core::TensorShape::from_dims({sample_count});
                const auto voiced_value = core::wrap_tensor(voiced_in, source_shape, GGML_TYPE_F32);
                ggml_tensor *unvoiced = ggml_scale_bias(ctx, voiced_in, -1.0f, 1.0f);
                const auto noise_amp = modules::AddModule{}.build(
                    build_ctx,
                    core::wrap_tensor(ggml_scale(ctx, voiced_in, generator->noise_std), source_shape, GGML_TYPE_F32),
                    core::wrap_tensor(ggml_scale(ctx, unvoiced, generator->sine_amp / 3.0f), source_shape,
                                      GGML_TYPE_F32));
                const float source_bias = generator->source_linear.use_bias ? generator->source_linear.bias[0] : 0.0f;
                auto merged = core::wrap_tensor(
                    ggml_scale_bias(ctx, ggml_view_1d(ctx, phase_in, sample_count, 0), 0.0f, source_bias), source_shape,
                    GGML_TYPE_F32);
                for (int64_t h = 0; h < harmonic_dims; ++h) {
                    const size_t row_offset = static_cast<size_t>(h * sample_count) * sizeof(float);
                    ggml_tensor *phase_h = ggml_view_1d(ctx, phase_in, sample_count, row_offset);
                    ggml_tensor *noise_h = ggml_view_1d(ctx, noise_in, sample_count, row_offset);
                    const auto base_sine = core::wrap_tensor(
                        ggml_scale(ctx, ggml_sin(ctx, phase_h), generator->sine_amp), source_shape, GGML_TYPE_F32);
                    const auto voiced_sine = modules::MulModule{}.build(build_ctx, base_sine, voiced_value);
                    const auto noise_h_value = core::wrap_tensor(noise_h, source_shape, GGML_TYPE_F32);
                    const auto noisy_sine = modules::AddModule{}.build(
                        build_ctx, voiced_sine, modules::MulModule{}.build(build_ctx, noise_amp, noise_h_value));
                    merged = modules::AddModule{}.build(
                        build_ctx, merged,
                        core::wrap_tensor(
                            ggml_scale(ctx, noisy_sine.tensor, generator->source_linear.weight[static_cast<size_t>(h)]),
                            source_shape, GGML_TYPE_F32));
                }
                output = modules::TanhModule{}.build(build_ctx, merged).tensor;
                output = ggml_cont(ctx, output);
                set_graph_output(output);

                graph = ggml_new_graph_custom(ctx, 4096, false);
                ggml_build_forward_expand(graph, output);
                core::set_backend_threads(backend, n_threads);
                gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                if (gallocr == nullptr || !ggml_gallocr_reserve(gallocr, graph) ||
                    !ggml_gallocr_alloc_graph(gallocr, graph)) {
                    throw std::runtime_error("failed to allocate Kitten source graph tensors");
                }
            } catch (...) {
                core::release_backend_graph_resources(backend, graph);
                if (gallocr) {
                    ggml_gallocr_free(gallocr);
                }
                if (ctx) {
                    ggml_free(ctx);
                }
                ctx = nullptr;
                throw;
            }
        }

        ~Session() {
            core::release_backend_graph_resources(backend, graph);
            if (gallocr) {
                ggml_gallocr_free(gallocr);
            }
            if (ctx) {
                ggml_free(ctx);
            }
        }

        void run(const std::vector<float> &phase, const std::vector<float> &noise, const std::vector<float> &voiced,
                 std::vector<float> &out) {
            const double upload_ms = measure_ms([&]() {
                core::write_tensor_f32(core::wrap_tensor(phase_in,
                                                         core::TensorShape::from_dims({harmonic_dims, sample_count}),
                                                         GGML_TYPE_F32),
                                       phase);
                core::write_tensor_f32(core::wrap_tensor(noise_in,
                                                         core::TensorShape::from_dims({harmonic_dims, sample_count}),
                                                         GGML_TYPE_F32),
                                       noise);
                core::write_tensor_f32(
                    core::wrap_tensor(voiced_in, core::TensorShape::from_dims({1, sample_count}), GGML_TYPE_F32),
                    voiced);
            });
            engine::debug::timing_log_scalar("kitten.decoder.conditioning_source.graph.upload_ms", upload_ms);
            core::set_backend_threads(backend, n_threads);
            ggml_status status = GGML_STATUS_SUCCESS;
            const double compute_ms =
                measure_ms([&]() { status = engine::core::compute_backend_graph(backend, graph); });
            engine::debug::timing_log_scalar("kitten.decoder.conditioning_source.graph.compute_ms", compute_ms);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("kitten source graph compute failed: ") +
                                         ggml_status_to_string(status));
            }
            out.clear();
            const double read_ms = measure_ms([&]() { out = core::read_tensor_f32(output); });
            engine::debug::timing_log_scalar("kitten.decoder.conditioning_source.graph.read_ms", read_ms);
        }
    };

    void prepare(int64_t sample_count, int64_t harmonic_dims) {
        if (session_ && session_->sample_count == sample_count && session_->harmonic_dims == harmonic_dims) {
            return;
        }
        session_.reset();
        session_ = std::make_unique<Session>(*generator_, backend_, n_threads_, sample_count, harmonic_dims);
    }

    const KittenWeights::GeneratorWeights *generator_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    int n_threads_ = 1;
    std::unique_ptr<Session> session_;
};

class ConditioningGraphRuntime {
  public:
    ConditioningGraphRuntime(const KittenWeights::GeneratorWeights &generator, ggml_backend_t backend,
                             int64_t decoder_frame_capacity, int64_t conditioning_frame_capacity, int n_threads,
                             bool use_device_backend)
        : session_(std::make_unique<Session>(generator, backend, decoder_frame_capacity, conditioning_frame_capacity,
                                             std::max(1, n_threads), use_device_backend)) {}

    void prepare(int64_t decoder_frame_capacity, int64_t conditioning_frame_capacity) {
        session_->prepare(decoder_frame_capacity, conditioning_frame_capacity);
    }

    const HarmonicConditioning &run(const std::vector<float> &f0_curve, DeterministicRng &rng) {
        return session_->run(f0_curve, rng);
    }

  private:
    struct Session {
        const KittenWeights::GeneratorWeights *generator = nullptr;
        ggml_backend_t backend = nullptr;
        int64_t decoder_frame_capacity = 0;
        int64_t conditioning_frame_capacity = 0;
        int64_t upsample_scale = 300;
        int64_t conditioning_sample_capacity = 0;
        int64_t harmonic_dims = 0;
        int n_threads = 1;
        bool use_device_backend = false;
        audio::STFTConfig stft_config = {};
        std::vector<float> source_signal;
        std::vector<float> valid_phase_coarse;
        std::vector<float> valid_phase_upsampled;
        std::vector<float> voiced_mask;
        std::vector<float> harmonic_noise;
        std::vector<float> valid_source_signal;
        HarmonicConditioning conditioning;
        std::unique_ptr<SourceSignalGraphRuntime> source_graph;

        Session(const KittenWeights::GeneratorWeights &generator_in, ggml_backend_t backend_in,
                int64_t decoder_frame_capacity_in, int64_t conditioning_frame_capacity_in, int n_threads_in,
                bool use_device_backend_in)
            : generator(&generator_in), backend(backend_in), harmonic_dims(generator_in.harmonic_num + 1),
              n_threads(n_threads_in), use_device_backend(use_device_backend_in),
              stft_config{
                  generator_in.gen_istft_n_fft, generator_in.gen_istft_hop_size, generator_in.gen_istft_n_fft, true,
                  audio::STFTPadMode::Reflect,  audio::STFTFamily::Kokoro,
              } {
            if (use_device_backend) {
                source_graph = std::make_unique<SourceSignalGraphRuntime>(generator_in, backend, n_threads);
            }
            prepare(decoder_frame_capacity_in, conditioning_frame_capacity_in);
        }

        void prepare(int64_t decoder_frame_capacity_in, int64_t conditioning_frame_capacity_in) {
            if (decoder_frame_capacity_in <= 0 || conditioning_frame_capacity_in <= 0) {
                throw std::runtime_error("Kitten conditioning graph capacity must be positive");
            }
            if (decoder_frame_capacity == decoder_frame_capacity_in &&
                conditioning_frame_capacity == conditioning_frame_capacity_in) {
                return;
            }
            decoder_frame_capacity = decoder_frame_capacity_in;
            conditioning_frame_capacity = conditioning_frame_capacity_in;
            conditioning_sample_capacity = decoder_frame_capacity * upsample_scale;
            conditioning.feature_rows = (stft_config.n_fft / 2 + 1) * 2;
            conditioning.feature_cols = conditioning_frame_capacity;
            conditioning.valid_feature_cols = 0;
            source_signal.resize(static_cast<size_t>(conditioning_sample_capacity));
            conditioning.features.resize(static_cast<size_t>(conditioning.feature_rows * conditioning.feature_cols));
        }

        const HarmonicConditioning &run(const std::vector<float> &f0_curve, DeterministicRng &rng) {
            const int64_t valid_decoder_frames = static_cast<int64_t>(f0_curve.size());
            if (valid_decoder_frames <= 0 || valid_decoder_frames > decoder_frame_capacity) {
                throw std::runtime_error("Kitten conditioning graph input frame count "
                                         "exceeds prepared capacity");
            }
            const double source_ms =
                measure_ms([&]() { synthesize_source_signal(f0_curve, valid_decoder_frames, rng); });
            const double stft_ms = measure_ms([&]() { compute_conditioning_features(valid_decoder_frames); });
            engine::debug::timing_log_scalar("kitten.decoder.conditioning_source_ms", source_ms);
            engine::debug::timing_log_scalar("kitten.decoder.conditioning_stft_ms", stft_ms);
            return conditioning;
        }

        void synthesize_source_signal(const std::vector<float> &f0_curve, int64_t valid_decoder_frames,
                                      DeterministicRng &rng) {
            const int64_t valid_sample_count = valid_decoder_frames * upsample_scale;
            std::vector<float> initial_phase(static_cast<size_t>(harmonic_dims), 0.0f);
            for (int64_t h = 1; h < harmonic_dims; ++h) {
                initial_phase[static_cast<size_t>(h)] = rng.uniform01();
            }
            valid_phase_coarse.resize(static_cast<size_t>(harmonic_dims * valid_decoder_frames));
#ifdef _OPENMP
#pragma omp parallel for if (harmonic_dims >= 4)
#endif
            for (int64_t h = 0; h < harmonic_dims; ++h) {
                const float harmonic = static_cast<float>(h + 1);
                float accumulated_phase = initial_phase[static_cast<size_t>(h)];
                for (int64_t t = 0; t < valid_decoder_frames; ++t) {
                    accumulated_phase += std::fmod((f0_curve[static_cast<size_t>(t)] * harmonic) /
                                                       static_cast<float>(generator->sampling_rate),
                                                   1.0f);
                    const float phase = accumulated_phase * 2.0f * kPi;
                    valid_phase_coarse[static_cast<size_t>(h * valid_decoder_frames + t)] =
                        phase * static_cast<float>(upsample_scale);
                }
            }
            harmonic_noise.resize(static_cast<size_t>(harmonic_dims * valid_sample_count));
            for (float &value : harmonic_noise) {
                value = rng.normal();
            }
            if (generator->source_linear.weight.empty() ||
                (generator->source_linear.use_bias && generator->source_linear.bias.empty())) {
                throw std::runtime_error("kitten decoder source affine weights are missing");
            }
            valid_phase_upsampled.resize(static_cast<size_t>(harmonic_dims * valid_sample_count));
            voiced_mask.resize(static_cast<size_t>(valid_sample_count));
#ifdef _OPENMP
#pragma omp parallel for if (valid_sample_count >= 4096)
#endif
            for (int64_t t = 0; t < valid_sample_count; ++t) {
                const int64_t frame = std::min<int64_t>(t / upsample_scale, valid_decoder_frames - 1);
                const float uv_t = f0_curve[static_cast<size_t>(frame)] > generator->voiced_threshold ? 1.0f : 0.0f;
                voiced_mask[static_cast<size_t>(t)] = uv_t;
                const float src = (static_cast<float>(t) + 0.5f) / static_cast<float>(upsample_scale) - 0.5f;
                int64_t left = 0;
                int64_t right = 0;
                float frac = 0.0f;
                if (src >= static_cast<float>(valid_decoder_frames - 1)) {
                    left = valid_decoder_frames - 1;
                    right = valid_decoder_frames - 1;
                } else if (src > 0.0f) {
                    left = static_cast<int64_t>(std::floor(src));
                    right = left + 1;
                    frac = src - static_cast<float>(left);
                }
                for (int64_t h = 0; h < harmonic_dims; ++h) {
                    valid_phase_upsampled[static_cast<size_t>(h * valid_sample_count + t)] =
                        valid_phase_coarse[static_cast<size_t>(h * valid_decoder_frames + left)] * (1.0f - frac) +
                        valid_phase_coarse[static_cast<size_t>(h * valid_decoder_frames + right)] * frac;
                }
            }
            if (source_graph) {
                source_graph->run(valid_phase_upsampled, harmonic_noise, voiced_mask, valid_sample_count, harmonic_dims,
                                  source_signal);
                return;
            }
            const float source_bias = generator->source_linear.use_bias ? generator->source_linear.bias[0] : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for if (valid_sample_count >= 4096)
#endif
            for (int64_t t = 0; t < valid_sample_count; ++t) {
                float merged = source_bias;
                const float uv_t = voiced_mask[static_cast<size_t>(t)];
                const float noise_amp = uv_t * generator->noise_std + (1.0f - uv_t) * generator->sine_amp / 3.0f;
                for (int64_t h = 0; h < harmonic_dims; ++h) {
                    const float base_sine =
                        std::sin(valid_phase_upsampled[static_cast<size_t>(h * valid_sample_count + t)]) *
                        generator->sine_amp;
                    const float noisy_sine =
                        base_sine * uv_t + noise_amp * harmonic_noise[static_cast<size_t>(h * valid_sample_count + t)];
                    merged += generator->source_linear.weight[static_cast<size_t>(h)] * noisy_sine;
                }
                source_signal[static_cast<size_t>(t)] = std::tanh(merged);
            }
        }

        void compute_conditioning_features(int64_t valid_decoder_frames) {
            const int64_t valid_sample_count = valid_decoder_frames * upsample_scale;
            const auto &window = audio::get_cached_stft_window(stft_config);
            valid_source_signal.assign(source_signal.begin(), source_signal.begin() + valid_sample_count);
            const audio::AudioTensor complex = audio::STFT().compute_complex(
                valid_source_signal, window, 1, valid_sample_count, stft_config, static_cast<size_t>(n_threads));
            const int64_t bins = complex.shape[1];
            const int64_t frames = complex.shape[2];
            if (conditioning.feature_rows != bins * 2 || frames > conditioning.feature_cols) {
                throw std::runtime_error("kitten conditioning STFT output exceeds prepared capacity");
            }
            conditioning.valid_feature_cols = frames;
            std::fill(conditioning.features.begin(), conditioning.features.end(), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (bins * frames >= 4096)
#endif
            for (int64_t bin = 0; bin < bins; ++bin) {
                for (int64_t frame = 0; frame < frames; ++frame) {
                    const size_t base = static_cast<size_t>(((bin * frames) + frame) * 2);
                    const float re = complex.values[base];
                    const float im = complex.values[base + 1];
                    conditioning.features[static_cast<size_t>(bin * conditioning.feature_cols + frame)] =
                        std::sqrt(re * re + im * im);
                    conditioning.features[static_cast<size_t>((bin + bins) * conditioning.feature_cols + frame)] =
                        std::atan2(im, re);
                }
            }
        }
    };

    std::unique_ptr<Session> session_;
};

struct GeneratorGraphStage {
    const KittenWeights::Conv1dWeights *noise_conv = nullptr;
    const KittenWeights::GeneratorResBlockWeights *noise_res = nullptr;
    const KittenWeights::WeightNormConvTranspose1dWeights *up = nullptr;
    std::vector<const KittenWeights::GeneratorResBlockWeights *> resblocks;
    bool reflect_pad = false;
};

ggml_tensor *build_generator_resblock(ggml_context *ctx, ggml_tensor *x,
                                      const KittenWeights::GeneratorResBlockWeights &block,
                                      const core::TensorValue &style, std::vector<TimeMaskInputs> &time_masks,
                                      bool use_time_masks) {
    ggml_tensor *current = x;
    core::ModuleBuildContext build_ctx = {};
    build_ctx.ggml = ctx;
    for (size_t i = 0; i < block.convs1.size(); ++i) {
        ggml_tensor *xt =
            build_adaptive_instance_norm_bct_decoder(ctx, current, block.adain1[i], style, time_masks, use_time_masks);
        xt = modules::Snake1dModule({xt->ne[1]})
                 .build(build_ctx,
                        core::wrap_tensor(xt, core::TensorShape::from_dims({xt->ne[2], xt->ne[1], xt->ne[0]}),
                                          GGML_TYPE_F32),
                        {core::reshape_tensor(build_ctx, block.alpha1[i], core::TensorShape::from_dims({xt->ne[1]}))})
                 .tensor;
        xt = build_decoder_conv1d_bct(ctx, xt, block.convs1[i]);
        xt = build_adaptive_instance_norm_bct_decoder(ctx, xt, block.adain2[i], style, time_masks, use_time_masks);
        xt = modules::Snake1dModule({xt->ne[1]})
                 .build(build_ctx,
                        core::wrap_tensor(xt, core::TensorShape::from_dims({xt->ne[2], xt->ne[1], xt->ne[0]}),
                                          GGML_TYPE_F32),
                        {core::reshape_tensor(build_ctx, block.alpha2[i], core::TensorShape::from_dims({xt->ne[1]}))})
                 .tensor;
        xt = build_decoder_conv1d_bct(ctx, xt, block.convs2[i]);
        const auto xt_value =
            core::wrap_tensor(xt, core::TensorShape::from_dims({xt->ne[2], xt->ne[1], xt->ne[0]}), GGML_TYPE_F32);
        const auto current_value = core::wrap_tensor(
            current, core::TensorShape::from_dims({current->ne[2], current->ne[1], current->ne[0]}), GGML_TYPE_F32);
        current = modules::AddModule{}.build(build_ctx, xt_value, current_value).tensor;
    }
    return current;
}

struct GeneratorGraphSession {
    const KittenWeights::GeneratorWeights *weights = nullptr;
    int64_t decoder_frame_capacity = 0;
    int64_t conditioning_frame_capacity = 0;
    int64_t style_dim = 0;
    int n_threads = 1;
    ggml_context *ctx = nullptr;
    ggml_tensor *decoder_x_in = nullptr;
    ggml_tensor *conditioning_in = nullptr;
    ggml_tensor *style_in = nullptr;
    ggml_tensor *output = nullptr;
    std::vector<TimeMaskInputs> time_masks;
    ggml_cgraph *graph = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t *gallocr = nullptr;

    GeneratorGraphSession(const KittenWeights::GeneratorWeights &weights_in, ggml_backend_t backend_in,
                          const std::vector<GeneratorGraphStage> &stages, int64_t decoder_frame_capacity_in,
                          int64_t conditioning_frame_capacity_in, int64_t style_dim_in, int n_threads_in,
                          ggml_gallocr_t &gallocr_in)
        : weights(&weights_in), decoder_frame_capacity(decoder_frame_capacity_in),
          conditioning_frame_capacity(conditioning_frame_capacity_in), style_dim(style_dim_in), n_threads(n_threads_in),
          backend(backend_in), gallocr(&gallocr_in) {
        ggml_init_params params{
            /*.mem_size   =*/64ull * 1024ull * 1024ull,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("failed to initialize ggml context for Kitten generator graph");
        }

        try {
            const double define_ms = measure_ms([&]() {
                decoder_x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, decoder_frame_capacity, 512, 1);
                conditioning_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, conditioning_frame_capacity, 22, 1);
                style_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, style_dim, 1);
                ggml_set_input(decoder_x_in);
                ggml_set_input(conditioning_in);
                ggml_set_input(style_in);
                const auto style =
                    core::wrap_tensor(style_in, core::TensorShape::from_dims({1, style_dim}), GGML_TYPE_F32);

                ggml_tensor *current = decoder_x_in;
                for (size_t i = 0; i < stages.size(); ++i) {
                    const GeneratorGraphStage &stage = stages[i];
                    ggml_tensor *source = build_decoder_conv1d_bct(ctx, conditioning_in, *stage.noise_conv);
                    source = build_generator_resblock(ctx, source, *stage.noise_res, style, time_masks, false);

                    ggml_tensor *x = ggml_leaky_relu(ctx, current, 0.1f, false);
                    x = build_conv_transpose1d_bct_decoder(ctx, x, *stage.up);
                    core::ModuleBuildContext add_ctx = {};
                    add_ctx.ggml = ctx;
                    if (stage.reflect_pad) {
                        x = modules::ReflectPad1dModule({1, 0})
                                .build(add_ctx, core::wrap_tensor(
                                                    x, core::TensorShape::from_dims({x->ne[2], x->ne[1], x->ne[0]}),
                                                    GGML_TYPE_F32))
                                .tensor;
                    }
                    const auto x_value = core::wrap_tensor(
                        x, core::TensorShape::from_dims({x->ne[2], x->ne[1], x->ne[0]}), GGML_TYPE_F32);
                    const auto source_value = core::wrap_tensor(
                        source, core::TensorShape::from_dims({source->ne[2], source->ne[1], source->ne[0]}),
                        GGML_TYPE_F32);
                    x = modules::AddModule{}.build(add_ctx, x_value, source_value).tensor;

                    ggml_tensor *stage_sum = nullptr;
                    for (size_t j = 0; j < stage.resblocks.size(); ++j) {
                        ggml_tensor *block =
                            build_generator_resblock(ctx, x, *stage.resblocks[j], style, time_masks, false);
                        if (stage_sum == nullptr) {
                            stage_sum = block;
                        } else {
                            const auto stage_sum_value = core::wrap_tensor(
                                stage_sum,
                                core::TensorShape::from_dims({stage_sum->ne[2], stage_sum->ne[1], stage_sum->ne[0]}),
                                GGML_TYPE_F32);
                            const auto block_value = core::wrap_tensor(
                                block, core::TensorShape::from_dims({block->ne[2], block->ne[1], block->ne[0]}),
                                GGML_TYPE_F32);
                            stage_sum = modules::AddModule{}.build(add_ctx, stage_sum_value, block_value).tensor;
                        }
                    }
                    if (stage.resblocks.empty()) {
                        throw std::runtime_error("Kitten generator stage has no residual blocks");
                    }
                    current = ggml_scale(ctx, stage_sum, 1.0f / static_cast<float>(stage.resblocks.size()));
                }

                current = ggml_leaky_relu(ctx, current, 0.01f, false);
                output = build_decoder_conv1d_bct(ctx, current, weights->conv_post);
                output = ggml_cont(ctx, output);
                set_graph_output(output);

                graph = ggml_new_graph_custom(ctx, 65536, false);
                ggml_build_forward_expand(graph, output);
            });

            core::set_backend_threads(backend, n_threads);
            const double alloc_ms = measure_ms([&]() {
                if (*gallocr == nullptr) {
                    *gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
                }
                if (*gallocr != nullptr) {
                    if (!ggml_gallocr_reserve(*gallocr, graph) || !ggml_gallocr_alloc_graph(*gallocr, graph)) {
                        ggml_gallocr_free(*gallocr);
                        *gallocr = nullptr;
                    }
                }
            });
            if (*gallocr == nullptr) {
                throw std::runtime_error("failed to allocate Kitten generator graph tensors");
            }
            engine::debug::timing_log_scalar("kitten.graph.build.decoder_generator_define_ms", define_ms);
            const double materialize_ms = measure_ms([&]() {
                std::vector<float> decoder(static_cast<size_t>(512 * decoder_frame_capacity), 0.0f);
                std::vector<float> conditioning(static_cast<size_t>(22 * conditioning_frame_capacity), 0.0f);
                std::vector<float> style(static_cast<size_t>(style_dim), 0.0f);
                core::write_tensor_f32(core::wrap_tensor(decoder_x_in,
                                                         core::TensorShape::from_dims({1, 512, decoder_frame_capacity}),
                                                         GGML_TYPE_F32),
                                       decoder);
                core::write_tensor_f32(
                    core::wrap_tensor(conditioning_in,
                                      core::TensorShape::from_dims({1, 22, conditioning_frame_capacity}),
                                      GGML_TYPE_F32),
                    conditioning);
                core::write_tensor_f32(
                    core::wrap_tensor(style_in, core::TensorShape::from_dims({style_dim}), GGML_TYPE_F32), style);
                upload_time_masks(time_masks, decoder_frame_capacity, decoder_frame_capacity);
            });
            engine::debug::timing_log_scalar("kitten.graph.build.decoder_generator_alloc_ms", alloc_ms);
            engine::debug::timing_log_scalar("kitten.graph.build.decoder_generator_materialize_ms", materialize_ms);
        } catch (...) {
            core::release_backend_graph_resources(backend, graph);
            if (ctx) {
                ggml_free(ctx);
            }
            ctx = nullptr;
            throw;
        }
    }

    ~GeneratorGraphSession() {
        core::release_backend_graph_resources(backend, graph);
        if (ctx) {
            ggml_free(ctx);
        }
    }

    std::vector<float> run(const std::vector<float> *decoder_x, int64_t decoder_x_rows, int64_t decoder_x_cols,
                           const ggml_tensor *borrowed_decoder_x_tensor, const HarmonicConditioning &conditioning,
                           const std::vector<float> &style) {
        if (borrowed_decoder_x_tensor == nullptr) {
            if (decoder_x == nullptr || decoder_x_rows != 512 || decoder_x_cols <= 0 ||
                decoder_x_cols > decoder_frame_capacity ||
                static_cast<int64_t>(decoder_x->size()) != decoder_x_rows * decoder_x_cols) {
                throw std::runtime_error("Kitten generator decoder input shape does "
                                         "not match graph capacity");
            }
        } else if (borrowed_decoder_x_tensor->ne[1] != 512 ||
                   borrowed_decoder_x_tensor->ne[0] != decoder_frame_capacity ||
                   decoder_x_cols != decoder_frame_capacity) {
            throw std::runtime_error("Kitten generator decoder backend tensor shape "
                                     "does not match exact graph capacity");
        }

        if (conditioning.feature_rows != 22 || conditioning.feature_cols != conditioning_frame_capacity ||
            conditioning.valid_feature_cols != conditioning_frame_capacity) {
            throw std::runtime_error("Kitten generator conditioning shape does not "
                                     "match exact graph capacity");
        }

        double decoder_upload_ms = 0.0;
        if (borrowed_decoder_x_tensor != nullptr) {
            decoder_upload_ms =
                measure_ms([&]() { ggml_backend_tensor_copy(borrowed_decoder_x_tensor, decoder_x_in); });
        } else {
            const int64_t valid_decoder_frames = decoder_x_cols;
            std::vector<float> padded_decoder(static_cast<size_t>(512 * decoder_frame_capacity), 0.0f);
            for (int64_t channel = 0; channel < 512; ++channel) {
                std::memcpy(padded_decoder.data() + static_cast<ptrdiff_t>(channel * decoder_frame_capacity),
                            decoder_x->data() + static_cast<ptrdiff_t>(channel * valid_decoder_frames),
                            static_cast<size_t>(valid_decoder_frames) * sizeof(float));
            }
            decoder_upload_ms = measure_ms([&]() {
                core::write_tensor_f32(core::wrap_tensor(decoder_x_in,
                                                         core::TensorShape::from_dims({1, 512, decoder_frame_capacity}),
                                                         GGML_TYPE_F32),
                                       padded_decoder);
            });
        }
        engine::debug::timing_log_scalar("kitten.decoder.generator.decoder_upload_ms", decoder_upload_ms);
        const double conditioning_upload_ms = measure_ms([&]() {
            core::write_tensor_f32(core::wrap_tensor(conditioning_in,
                                                     core::TensorShape::from_dims({1, 22, conditioning_frame_capacity}),
                                                     GGML_TYPE_F32),
                                   conditioning.features);
        });
        engine::debug::timing_log_scalar("kitten.decoder.generator.conditioning_upload_ms", conditioning_upload_ms);
        const double style_upload_ms = measure_ms([&]() {
            core::write_tensor_f32(
                core::wrap_tensor(style_in, core::TensorShape::from_dims({style_dim}), GGML_TYPE_F32), style);
            upload_time_masks(time_masks, decoder_x_cols, decoder_frame_capacity);
        });
        engine::debug::timing_log_scalar("kitten.decoder.generator.style_upload_ms", style_upload_ms);
        core::set_backend_threads(backend, n_threads);
        ggml_status status = GGML_STATUS_SUCCESS;
        const double compute_ms = measure_ms([&]() { status = engine::core::compute_backend_graph(backend, graph); });
        engine::debug::timing_log_scalar("kitten.decoder.generator.graph.compute_ms", compute_ms);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("kitten generator graph compute failed: ") +
                                     ggml_status_to_string(status));
        }

        std::vector<float> out;
        const double output_read_ms = measure_ms([&]() { out = core::read_tensor_f32(output); });
        engine::debug::timing_log_scalar("kitten.decoder.generator.output_read_ms", output_read_ms);
        return out;
    }
};

class GeneratorGraphRuntime {
  public:
    GeneratorGraphRuntime(const KittenWeights::GeneratorWeights &generator, int64_t style_dim, ggml_backend_t backend,
                          int n_threads)
        : generator_(&generator), style_dim_(style_dim), backend_(backend), n_threads_(std::max(1, n_threads)) {
        if (generator.noise_convs.size() != generator.ups.size() ||
            generator.noise_res.size() != generator.ups.size() ||
            generator.resblocks.size() % generator.ups.size() != 0) {
            throw std::runtime_error("Kitten generator weight layout is inconsistent");
        }
        const size_t resblocks_per_stage = generator.resblocks.size() / generator.ups.size();
        stages_.reserve(generator.ups.size());
        for (size_t i = 0; i < generator.ups.size(); ++i) {
            GeneratorGraphStage stage;
            stage.noise_conv = &generator.noise_convs[i];
            stage.noise_res = &generator.noise_res[i];
            stage.up = &generator.ups[i];
            stage.resblocks.reserve(resblocks_per_stage);
            for (size_t j = 0; j < resblocks_per_stage; ++j) {
                stage.resblocks.push_back(&generator.resblocks[i * resblocks_per_stage + j]);
            }
            stage.reflect_pad = (i + 1 == generator.ups.size());
            stages_.push_back(stage);
        }
    }

    ~GeneratorGraphRuntime() {
        session_.reset();
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    void prepare(int64_t decoder_frame_capacity, int64_t conditioning_frame_capacity) {
        if (decoder_frame_capacity <= 0 || conditioning_frame_capacity <= 0) {
            throw std::runtime_error("Kitten generator graph capacity must be positive");
        }
        if (session_ != nullptr && session_->decoder_frame_capacity == decoder_frame_capacity &&
            session_->conditioning_frame_capacity == conditioning_frame_capacity) {
            return;
        }
        const double release_ms = measure_ms([&]() { session_.reset(); });
        double construct_ms = 0.0;
        construct_ms = measure_ms([&]() {
            session_ =
                std::make_unique<GeneratorGraphSession>(*generator_, backend_, stages_, decoder_frame_capacity,
                                                        conditioning_frame_capacity, style_dim_, n_threads_, gallocr_);
        });
        engine::debug::timing_log_scalar("kitten.graph.build.decoder_generator_release_ms", release_ms);
        engine::debug::timing_log_scalar("kitten.graph.build.decoder_generator_construct_ms", construct_ms);
    }

    std::vector<float> run(const std::vector<float> &decoder_x, int64_t decoder_x_rows, int64_t decoder_x_cols,
                           const ggml_tensor *borrowed_decoder_x_tensor, const HarmonicConditioning &conditioning,
                           const std::vector<float> &style) {
        if (session_ == nullptr) {
            throw std::runtime_error("Kitten generator graph was not prepared");
        }
        return session_->run(borrowed_decoder_x_tensor ? nullptr : &decoder_x, decoder_x_rows, decoder_x_cols,
                             borrowed_decoder_x_tensor, conditioning, style);
    }

  private:
    ggml_gallocr_t gallocr_ = nullptr;
    const KittenWeights::GeneratorWeights *generator_ = nullptr;
    int64_t style_dim_ = 0;
    ggml_backend_t backend_ = nullptr;
    int n_threads_ = 1;
    std::vector<GeneratorGraphStage> stages_;
    std::unique_ptr<GeneratorGraphSession> session_;
};

} // namespace

struct KittenDecoderRuntime::Impl {
    std::shared_ptr<const KittenWeights> weights;
    int n_threads = 1;
    bool use_device_backend = false;
    uint64_t rng_seed = 0;
    ggml_backend_t backend = nullptr;
    audio::STFTConfig inverse_stft_config;
    std::unique_ptr<ConditioningGraphRuntime> conditioning_graph;
    GeneratorGraphRuntime generator_graph;

    Impl(std::shared_ptr<const KittenWeights> weights_in, ggml_backend_t backend_in, int n_threads_in,
         bool use_device_backend_in, uint64_t rng_seed_in, KittenDecoderCapacityContract contract_in)
        : weights(std::move(weights_in)), n_threads(std::max(1, n_threads_in)),
          use_device_backend(use_device_backend_in), rng_seed(rng_seed_in), backend(backend_in),
          inverse_stft_config({
              weights->decoder.generator.gen_istft_n_fft,
              weights->decoder.generator.gen_istft_hop_size,
              weights->decoder.generator.gen_istft_n_fft,
              true,
              audio::STFTPadMode::Reflect,
              audio::STFTFamily::Kokoro,
          }),
          generator_graph(weights->decoder.generator, weights->style_dim, backend, n_threads) {
        prepare(contract_in);
    }

    void prepare(KittenDecoderCapacityContract contract) {
        if (!conditioning_graph) {
            conditioning_graph =
                std::make_unique<ConditioningGraphRuntime>(weights->decoder.generator, backend, contract.decoder_frames,
                                                           contract.conditioning_frames, n_threads, use_device_backend);
        } else {
            conditioning_graph->prepare(contract.decoder_frames, contract.conditioning_frames);
        }
        generator_graph.prepare(contract.decoder_frames, contract.conditioning_frames);
    }
};

KittenDecoderRuntime::KittenDecoderRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend,
                                           int n_threads, bool use_device_backend, uint64_t rng_seed,
                                           KittenDecoderCapacityContract contract)
    : impl_(std::make_unique<Impl>(std::move(weights), backend, n_threads, use_device_backend, rng_seed, contract)) {}

KittenDecoderRuntime::~KittenDecoderRuntime() = default;

void KittenDecoderRuntime::prepare(KittenDecoderCapacityContract contract) { impl_->prepare(contract); }

std::vector<float> KittenDecoderRuntime::decode(const PredictorOutputs &predictor, const std::vector<float> &ref_s) {
    if (static_cast<int64_t>(ref_s.size()) != 256) {
        throw std::runtime_error("Kitten decoder requires ref_s with 256 elements");
    }

    const std::vector<float> style_decoder(ref_s.begin(), ref_s.begin() + 128);
    DeterministicRng rng(impl_->rng_seed);
    double conditioning_ms = 0.0;
    double generator_ms = 0.0;
    double istft_ms = 0.0;
    const HarmonicConditioning *conditioning = nullptr;
    conditioning_ms = measure_ms([&]() { conditioning = &impl_->conditioning_graph->run(predictor.f0_curve, rng); });

    std::vector<float> generator_output;
    generator_ms = measure_ms([&]() {
        generator_output =
            impl_->generator_graph.run(predictor.decoder_x, predictor.decoder_x_rows, predictor.decoder_x_cols,
                                       predictor.borrowed_decoder_x_tensor, *conditioning, style_decoder);
    });
    const int64_t bins = impl_->inverse_stft_config.n_fft / 2 + 1;
    const int64_t generated_frames = static_cast<int64_t>(generator_output.size()) / (bins * 2);
    const int64_t frames = conditioning->valid_feature_cols;
    if (frames <= 0 || frames > generated_frames) {
        throw std::runtime_error("Kitten decoder valid spectrogram frame count "
                                 "exceeds generated output");
    }
    std::vector<float> waveform;
    istft_ms = measure_ms([&]() {
        std::vector<float> complex_spec(static_cast<size_t>(bins * frames * 2), 0.0f);
        for (int64_t f = 0; f < bins; ++f) {
            for (int64_t c = 0; c < frames; ++c) {
                const float mag = std::exp(generator_output[static_cast<size_t>(f * generated_frames + c)]);
                const float phase = std::sin(generator_output[static_cast<size_t>((f + bins) * generated_frames + c)]);
                const size_t base = static_cast<size_t>((f * frames + c) * 2);
                complex_spec[base] = mag * std::cos(phase);
                complex_spec[base + 1] = mag * std::sin(phase);
            }
        }
        const auto &window = audio::get_cached_stft_window(impl_->inverse_stft_config);
        const int64_t samples = impl_->inverse_stft_config.hop_length * (frames - 1);
        waveform =
            audio::ISTFT().compute(complex_spec, window, 1, bins, frames, samples, impl_->inverse_stft_config).values;
        const int64_t n_fft = impl_->inverse_stft_config.n_fft;
        const int64_t hop = impl_->inverse_stft_config.hop_length;
        const int64_t center_pad = n_fft / 2;
        std::vector<float> overlap_scale(static_cast<size_t>(samples + 2 * center_pad), 0.0f);
        for (int64_t frame = 0; frame < frames; ++frame) {
            const int64_t start = frame * hop;
            for (int64_t i = 0; i < n_fft; ++i) {
                const float value = window[static_cast<size_t>(i)];
                overlap_scale[static_cast<size_t>(start + i)] += value * value;
            }
        }
        for (int64_t i = 0; i < samples; ++i) {
            waveform[static_cast<size_t>(i)] *= overlap_scale[static_cast<size_t>(i + center_pad)];
        }
    });
    const int64_t valid_waveform_samples =
        impl_->inverse_stft_config.hop_length * std::max<int64_t>(conditioning->valid_feature_cols - 1, 0);
    if (valid_waveform_samples < 0 || valid_waveform_samples > static_cast<int64_t>(waveform.size())) {
        throw std::runtime_error("Kitten decoder valid waveform length exceeds generated output");
    }
    waveform.resize(static_cast<size_t>(valid_waveform_samples));
    engine::debug::timing_log_scalar("kitten.decoder.conditioning_ms", conditioning_ms);
    engine::debug::timing_log_scalar("kitten.decoder.generator_ms", generator_ms);
    engine::debug::timing_log_scalar("kitten.decoder.istft_ms", istft_ms);

    return waveform;
}

} // namespace engine::models::kitten_tts
