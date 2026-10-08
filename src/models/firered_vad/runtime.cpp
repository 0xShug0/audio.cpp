// SPDX-License-Identifier: Apache-2.0
// DFSMN and segment rules adapted from FireRedTeam/FireRedVAD.
// Copyright 2026 Xiaohongshu (Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu).
#include "engine/models/firered_vad/runtime.h"

#include "engine/framework/audio/kaldi_fbank.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace engine::models::firered_vad {
namespace {

constexpr int64_t kHop = 160;
constexpr int64_t kWindow = 400;
constexpr int64_t kHistory = 19;
constexpr int64_t kChannels = 128;
constexpr int64_t kLayers = 8;

struct ContextDeleter { void operator()(ggml_context * value) const { ggml_free(value); } };
struct AllocatorDeleter { void operator()(ggml_gallocr_t value) const { ggml_gallocr_free(value); } };
struct BackendDeleter { void operator()(ggml_backend_t value) const { ggml_backend_free(value); } };
struct SchedulerDeleter { void operator()(ggml_backend_sched_t value) const { ggml_backend_sched_free(value); } };

void validate_audio(int rate, int channels, const std::vector<float> & samples) {
    if (rate != 16000 || channels != 1) throw std::runtime_error("FireRed VAD requires 16000 Hz mono audio");
    if (!std::all_of(samples.begin(), samples.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("FireRed VAD requires finite audio samples");
}

void validate_options(const DetectionOptions & options) {
    if (!std::isfinite(options.threshold) || options.threshold < 0 || options.threshold > 1 ||
        options.smooth_window_frames < 1 || options.min_speech_frames < 1 ||
        options.max_speech_frames < 2 || options.min_silence_frames < 0 ||
        options.merge_silence_frames < 0 || options.extend_speech_frames < 0 ||
        options.pad_start_frames < 0 || options.chunk_frames < 1)
        throw std::runtime_error("Invalid FireRed VAD detection options");
}

enum class State { Silence, PossibleSpeech, Speech, PossibleSilence };

runtime::SpeechSegment make_segment(int64_t start, int64_t end, int64_t samples) {
    runtime::SpeechSegment result;
    result.span = {std::min(start, samples), std::min(end, samples)};
    return result;
}

}  // namespace

std::vector<runtime::SpeechSegment> segment_probabilities(
    const std::vector<float> & probabilities, int64_t samples, const DetectionOptions & options) {
    validate_options(options);
    std::vector<int> decisions(probabilities.size(), 0);
    double sum = 0;
    State state = State::Silence;
    int64_t speech_start = -1, silence_start = -1;
    for (size_t t = 0; t < probabilities.size(); ++t) {
        sum += probabilities[t];
        if (t >= static_cast<size_t>(options.smooth_window_frames)) sum -= probabilities[t - options.smooth_window_frames];
        const bool speech = sum / std::min<size_t>(t + 1, options.smooth_window_frames) >= options.threshold;
        switch (state) {
        case State::Silence:
            if (speech) { state = State::PossibleSpeech; speech_start = t; }
            break;
        case State::PossibleSpeech:
            if (!speech) { state = State::Silence; speech_start = -1; }
            else if (static_cast<int64_t>(t) - speech_start >= options.min_speech_frames) {
                state = State::Speech;
                std::fill(decisions.begin() + speech_start, decisions.begin() + t, 1);
            }
            break;
        case State::Speech:
            if (!speech) { state = State::PossibleSilence; silence_start = t; }
            break;
        case State::PossibleSilence:
            if (speech) { state = State::Speech; silence_start = -1; }
            else if (static_cast<int64_t>(t) - silence_start >= options.min_silence_frames) {
                state = State::Silence; speech_start = -1;
            }
            break;
        }
        decisions[t] = state == State::Speech || state == State::PossibleSilence;
    }
    auto fixed = decisions;
    for (size_t t = 1; t < decisions.size(); ++t)
        if (!decisions[t - 1] && decisions[t])
            std::fill(fixed.begin() + std::max<int64_t>(0, static_cast<int64_t>(t) - options.smooth_window_frames), fixed.begin() + t, 1);
    decisions = fixed;
    silence_start = -1;
    for (size_t t = 1; t < decisions.size(); ++t) {
        if (decisions[t - 1] && !decisions[t]) silence_start = t;
        else if (!decisions[t - 1] && decisions[t] && silence_start >= 0) {
            if (static_cast<int64_t>(t) - silence_start < options.merge_silence_frames)
                std::fill(fixed.begin() + silence_start, fixed.begin() + t, 1);
            silence_start = -1;
        }
    }
    decisions = fixed;
    for (size_t t = 0; t < decisions.size(); ++t) {
        if (!decisions[t]) continue;
        const auto begin = std::max<int64_t>(0, static_cast<int64_t>(t) - options.extend_speech_frames);
        const auto end = std::min<int64_t>(decisions.size(), t + static_cast<int64_t>(options.extend_speech_frames) + 1);
        std::fill(fixed.begin() + begin, fixed.begin() + end, 1);
    }
    // Upstream splits overlong segments at the lowest raw probability in the
    // second half of each maximum-duration window, leaving one silence frame.
    for (size_t t = 0; t < fixed.size();) {
        if (!fixed[t]) { ++t; continue; }
        size_t end = t;
        while (end < fixed.size() && fixed[end]) ++end;
        size_t start = t;
        while (end - start > static_cast<size_t>(options.max_speech_frames)) {
            const size_t lo = start + options.max_speech_frames / 2;
            const size_t hi = start + options.max_speech_frames;
            const size_t split = std::min_element(probabilities.begin() + lo, probabilities.begin() + hi) - probabilities.begin();
            fixed[split] = 0;
            start = split + 1;
        }
        t = end;
    }
    std::vector<runtime::SpeechSegment> result;
    for (size_t t = 0; t < fixed.size();) {
        if (!fixed[t]) { ++t; continue; }
        const size_t start = t;
        while (t < fixed.size() && fixed[t]) ++t;
        const int64_t end = t * kHop + (t == fixed.size() ? kWindow : 0);
        result.push_back(make_segment(start * kHop, end, samples));
    }
    return result;
}

class FireRedDfsmnRuntime::Impl {
public:
    Impl(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
        : execution(execution), store(execution.backend(), execution.backend_type(), "firered_vad.weights", 1024 * 1024) {
        causal = !source->has_tensor("dfsmn.fsmn1.lookahead_filter.weight");
        frontend.window_type = audio::KaldiFbankWindowType::Povey;
        frontend.lfr_m = frontend.lfr_n = 1;
        frontend.upscale_samples = true;
        frontend.apply_cmvn = true;
        frontend.sparse_filterbank = true;
        frontend.cmvn_shift = source->require_f32("frontend.cmvn_shift", {80});
        frontend.cmvn_scale = source->require_f32("frontend.cmvn_scale", {80});
        first = modules::binding::linear_from_source(store, *source, "dfsmn.fc1.0", assets::TensorStorageType::Native, 256, 80, true);
        second = modules::binding::linear_from_source(store, *source, "dfsmn.fc2.0", assets::TensorStorageType::Native, 128, 256, true);
        for (int layer = 0; layer < kLayers; ++layer) {
            const std::string block = "dfsmn.fsmns." + std::to_string(layer - 1);
            const std::string memory = layer == 0 ? "dfsmn.fsmn1" : block + ".fsmn";
            if (layer > 0) {
                blocks[layer].first = modules::binding::linear_from_source(store, *source, block + ".fc1.0", assets::TensorStorageType::Native, 256, 128, true);
                blocks[layer].second = modules::binding::linear_from_source(store, *source, block + ".fc2", assets::TensorStorageType::Native, 128, 256, false);
            }
            blocks[layer].back = modules::binding::depthwise_conv1d_from_source(store, *source, memory + ".lookback_filter", assets::TensorStorageType::Native, 128, 20, false);
            if (!causal) blocks[layer].ahead = modules::binding::depthwise_conv1d_from_source(store, *source, memory + ".lookahead_filter", assets::TensorStorageType::Native, 128, 20, false);
        }
        dnn = modules::binding::linear_from_source(store, *source, "dfsmn.dnns.0", assets::TensorStorageType::Native, 256, 128, true);
        classifier = modules::binding::linear_from_source(store, *source, "out", assets::TensorStorageType::Native, 1, 256, true);
        store.upload();
        if (execution.backend_type() == core::BackendType::Cpu) {
            blas.reset(ggml_backend_init_by_name("BLAS", nullptr));
            if (blas) core::set_backend_threads(blas.get(), execution.config().threads);
        }
        DetectionOptions defaults;
        if (causal) { defaults.threshold = 0.5; defaults.min_speech_frames = 8; }
        reset(defaults);
    }
    ~Impl() { clear_graph(); }

    void clear_graph() {
        if (graph) core::release_backend_graph_resources(execution.backend(), graph, true);
        scheduler.reset();
        plan.reset();
        allocator.reset();
        context.reset();
        graph = nullptr;
    }

    void build(int64_t frames) {
        if (graph && capacity == frames) return;
        clear_graph();
        capacity = frames;
        const auto started = std::chrono::steady_clock::now();
        context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
        if (!context) throw std::runtime_error("FireRed VAD graph context allocation failed");
        core::ModuleBuildContext ctx{context.get(), "firered_vad", execution.backend_type()};
        input = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, 80}));
        ggml_set_input(input.tensor);
        const auto linear = [&](core::TensorValue x, const modules::LinearWeights & weights, bool relu) {
            const auto in = weights.weight.shape.dims[1];
            const auto out = weights.weight.shape.dims[0];
            if (execution.backend_type() == core::BackendType::Cuda &&
                weights.weight.type == GGML_TYPE_F32 && frames > 3 && frames <= 16 && out > 16) {
                // W * X^T avoids the small-matrix TF32 lowering, which ignores
                // GGML_PREC_F32 and can change streaming VAD threshold crossings.
                auto transposed = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                auto matrix = core::reshape_tensor(ctx, weights.weight, core::TensorShape::from_dims({1, out, in}));
                x = modules::MatMulModule{}.build(ctx, matrix, transposed);
                x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
                if (weights.bias) {
                    auto bias = core::reshape_tensor(ctx, *weights.bias, core::TensorShape::from_dims({1, 1, out}));
                    x = modules::AddModule{}.build(ctx, x, modules::RepeatModule({x.shape}).build(ctx, bias));
                }
            } else {
                x = modules::LinearModule({in, out, weights.bias.has_value(), GGML_PREC_F32}).build(ctx, x, weights);
            }
            return relu ? modules::ReluModule{}.build(ctx, x) : x;
        };
        const auto cpu_memory = [&](core::TensorValue values, const modules::DepthwiseConv1dWeights & weights) {
            // Keep channels contiguous for SIMD. Fixed time blocks become batch
            // rows, which the CPU kernel distributes across any thread count.
            const int64_t output_frames = values.shape.dims[1] - kHistory;
            const int64_t block_frames = std::min<int64_t>(32, output_frames);
            const int64_t block_count = (output_frames + block_frames - 1) / block_frames;
            auto * data = core::ensure_backend_addressable_layout(ctx, values).tensor;
            // GGML's view constructor also checks the dense logical extent.
            const int64_t padded_frames = block_count * (block_frames + kHistory);
            if (padded_frames > values.shape.dims[1])
                data = ggml_pad(ctx.ggml, data, 0, padded_frames - values.shape.dims[1], 0, 0);
            // Adjacent blocks share the 19 input frames needed by the filter.
            data = ggml_view_4d(ctx.ggml, data, kChannels, block_frames + kHistory, 1, block_count,
                kChannels * sizeof(float), kChannels * (block_frames + kHistory) * sizeof(float),
                kChannels * block_frames * sizeof(float), 0);
            data = ggml_cont(ctx.ggml, data);
            data = ggml_permute(ctx.ggml, data, 2, 0, 1, 3);
            auto * kernel = weights.weight.tensor;
            if (kernel->type != GGML_TYPE_F32) kernel = ggml_cast(ctx.ggml, kernel, GGML_TYPE_F32);
            kernel = ggml_reshape_2d(ctx.ggml, kernel, kHistory + 1, kChannels);
            kernel = ggml_cont(ctx.ggml, ggml_transpose(ctx.ggml, kernel));
            kernel = ggml_reshape_4d(ctx.ggml, kernel, kChannels, 1, kHistory + 1, 1);
            kernel = ggml_permute(ctx.ggml, kernel, 3, 2, 0, 1);
            auto * result = ggml_conv_2d_dw_direct(ctx.ggml, kernel, data, 1, 1, 0, 0, 1, 1);
            result = ggml_permute(ctx.ggml, result, 1, 2, 0, 3);
            result = ggml_reshape_3d(ctx.ggml, result, kChannels, block_count * block_frames, 1);
            return modules::SliceModule({1, 0, output_frames}).build(ctx,
                core::wrap_tensor(result, core::TensorShape::from_dims({1, block_count * block_frames, kChannels})));
        };
        auto x = linear(linear(input, first, true), second, true);
        for (int layer = 0; layer < kLayers; ++layer) {
            const auto residual = x;
            if (layer > 0) x = linear(linear(x, blocks[layer].first, true), blocks[layer].second, false);
            const auto projected = x;
            if (execution.backend_type() == core::BackendType::Cpu) {
                core::TensorValue past;
                if (causal) {
                    cache_inputs[layer] = core::make_tensor(ctx, GGML_TYPE_F32,
                        core::TensorShape::from_dims({1, kHistory, kChannels}));
                    ggml_set_input(cache_inputs[layer].tensor);
                    past = modules::ConcatModule({1}).build(ctx, cache_inputs[layer], x);
                    cache_outputs[layer] = core::ensure_backend_addressable_layout(ctx,
                        modules::SliceModule({1, frames, kHistory}).build(ctx, past));
                    ggml_set_output(cache_outputs[layer].tensor);
                } else {
                    past = core::wrap_tensor(ggml_pad_ext(ctx.ggml,
                        core::ensure_backend_addressable_layout(ctx, x).tensor, 0, 0, kHistory, 0, 0, 0, 0, 0),
                        core::TensorShape::from_dims({1, frames + kHistory, kChannels}));
                }
                auto memory = cpu_memory(past, blocks[layer].back);
                if (!causal) {
                    auto future = core::wrap_tensor(ggml_pad(ctx.ggml,
                        core::ensure_backend_addressable_layout(ctx, x).tensor, 0, kHistory + 1, 0, 0),
                        core::TensorShape::from_dims({1, frames + kHistory + 1, kChannels}));
                    future = modules::SliceModule({1, 1, frames}).build(ctx, cpu_memory(future, blocks[layer].ahead));
                    memory = modules::AddModule{}.build(ctx, memory, future);
                }
                x = modules::AddModule{}.build(ctx, projected, memory);
                if (layer > 0) x = modules::ResidualAddModule{}.build(ctx, x, residual);
                continue;
            }
            x = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
            core::TensorValue past;
            if (causal) {
                cache_inputs[layer] = core::make_tensor(ctx, GGML_TYPE_F32,
                    core::TensorShape::from_dims({1, 128, kHistory}));
                ggml_set_input(cache_inputs[layer].tensor);
                past = modules::ConcatModule({2}).build(ctx, cache_inputs[layer], x);
                cache_outputs[layer] = core::ensure_backend_addressable_layout(ctx, modules::SliceModule({2, frames, kHistory}).build(ctx, past));
                ggml_set_output(cache_outputs[layer].tensor);
            } else {
                past = core::wrap_tensor(ggml_pad_ext(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor,
                    kHistory, 0, 0, 0, 0, 0, 0, 0), core::TensorShape::from_dims({1, 128, frames + kHistory}));
            }
            auto memory = modules::DepthwiseConv1dModule({128, 20, 1, 0, 1, false}).build(ctx, past, blocks[layer].back);
            if (!causal) {
                auto future = core::wrap_tensor(ggml_pad(ctx.ggml, core::ensure_backend_addressable_layout(ctx, x).tensor, 20, 0, 0, 0),
                    core::TensorShape::from_dims({1, 128, frames + 20}));
                future = modules::SliceModule({2, 1, frames + 19}).build(ctx, future);
                future = modules::DepthwiseConv1dModule({128, 20, 1, 0, 1, false}).build(ctx, future, blocks[layer].ahead);
                memory = modules::AddModule{}.build(ctx, memory, future);
            }
            memory = modules::TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, memory);
            x = modules::AddModule{}.build(ctx, projected, memory);
            if (layer > 0) x = modules::ResidualAddModule{}.build(ctx, x, residual);
        }
        output = modules::SigmoidModule{}.build(ctx, linear(linear(x, dnn, true), classifier, false));
        ggml_set_output(output.tensor);
        graph = ggml_new_graph_custom(context.get(), 2048, false);
        ggml_build_forward_expand(graph, output.tensor);
        if (causal) for (const auto & cache : cache_outputs) ggml_build_forward_expand(graph, cache.tensor);
        core::validate_backend_graph_supported(execution.backend(), graph, "firered_vad");
        if (blas) {
            ggml_backend_t backends[] = {blas.get(), execution.backend()};
            scheduler.reset(ggml_backend_sched_new(backends, nullptr, 2, ggml_graph_size(graph), false, false));
            if (!scheduler || !ggml_backend_sched_reserve(scheduler.get(), graph) ||
                !ggml_backend_sched_alloc_graph(scheduler.get(), graph))
                throw std::runtime_error("FireRed VAD scheduled graph allocation failed");
        }
        if (!scheduler) {
            allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph)) throw std::runtime_error("FireRed VAD graph allocation failed");
            core::prepare_host_graph_plan(execution, graph, plan);
        }
        debug::timing_log_scalar("firered_vad.graph_prepare_ms", debug::elapsed_ms(started));
    }

    std::vector<float> infer(const std::vector<float> & features, bool retain_state) {
        if (features.size() % 80) throw std::runtime_error("FireRed VAD features must have 80 bins per frame");
        if (features.empty()) return {};
        if (retain_state && !causal) throw std::runtime_error("FireRed VAD state requires the causal checkpoint");
        build(features.size() / 80);
        core::write_tensor_f32(input, features);
        if (causal) {
            const std::vector<float> zeros(kChannels * kHistory, 0.0f);
            for (int layer = 0; layer < kLayers; ++layer) core::write_tensor_f32(cache_inputs[layer], retain_state ? caches[layer] : zeros);
        }
        const auto started = std::chrono::steady_clock::now();
        const auto status = scheduler ? ggml_backend_sched_graph_compute(scheduler.get(), graph)
            : core::compute_graph(execution, graph, plan, "firered_vad");
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error("FireRed VAD graph execution failed");
        auto probabilities = core::read_tensor_f32(output.tensor);
        if (causal && retain_state) for (int layer = 0; layer < kLayers; ++layer) caches[layer] = core::read_tensor_f32(cache_outputs[layer].tensor);
        debug::timing_log_scalar("firered_vad.network_ms", debug::elapsed_ms(started));
        return probabilities;
    }

    runtime::StreamEvent consume(const std::vector<float> & probabilities) {
        runtime::StreamEvent event;
        for (float probability : probabilities) {
            ++frame_count;
            smooth.push_back(probability);
            smooth_sum += probability;
            if (smooth.size() > static_cast<size_t>(options.smooth_window_frames)) { smooth_sum -= smooth.front(); smooth.pop_front(); }
            const bool speech = smooth_sum / smooth.size() >= options.threshold;
            bool start = false, end = false;
            int64_t start_frame = -1;
            if (hit_max) { start = true; start_frame = frame_count; last_start = start_frame; hit_max = false; }
            switch (state) {
            case State::Silence:
                if (speech) { state = State::PossibleSpeech; ++speech_count; }
                else { ++silence_count; speech_count = 0; }
                break;
            case State::PossibleSpeech:
                if (speech) {
                    if (++speech_count >= options.min_speech_frames) {
                        state = State::Speech; start = true;
                        start_frame = std::max<int64_t>({1, frame_count - speech_count + 1 - std::max(options.smooth_window_frames, options.pad_start_frames), last_end + 1});
                        last_start = start_frame; silence_count = 0;
                    }
                } else { state = State::Silence; silence_count = 1; speech_count = 0; }
                break;
            case State::Speech:
                ++speech_count;
                if (speech) {
                    silence_count = 0;
                    if (speech_count >= options.max_speech_frames) { hit_max = true; speech_count = 0; end = true; }
                } else { state = State::PossibleSilence; ++silence_count; }
                break;
            case State::PossibleSilence:
                ++speech_count;
                if (speech) {
                    state = State::Speech; silence_count = 0;
                    if (speech_count >= options.max_speech_frames) { hit_max = true; speech_count = 0; end = true; }
                } else if (++silence_count >= options.min_silence_frames) { state = State::Silence; end = true; speech_count = 0; }
                break;
            }
            if (start) event.voice_activity.push_back({runtime::VoiceActivityEvent::Kind::SpeechStart, (start_frame - 1) * kHop, probability, std::nullopt});
            if (end) {
                auto segment = make_segment((last_start - 1) * kHop, (frame_count - 1) * kHop, total_samples);
                segments.push_back(segment);
                event.voice_activity.push_back({runtime::VoiceActivityEvent::Kind::SpeechEnd, segment.span.end_sample, probability, segment});
                last_end = frame_count; last_start = -1;
            }
        }
        return event;
    }

    void reset(const DetectionOptions & value) {
        validate_options(value);
        options = value;
        for (auto & cache : caches) cache.assign(kChannels * kHistory, 0.0f);
        pending.clear(); segments.clear(); smooth.clear(); smooth_sum = 0;
        total_samples = frame_count = speech_count = silence_count = 0;
        last_start = last_end = -1;
        hit_max = finished = false;
        state = State::Silence;
    }

    struct Block {
        modules::LinearWeights first, second;
        modules::DepthwiseConv1dWeights back, ahead;
    };
    core::ExecutionContext & execution;
    core::BackendWeightStore store;
    std::unique_ptr<ggml_backend, BackendDeleter> blas;
    std::unique_ptr<ggml_backend_sched, SchedulerDeleter> scheduler;
    audio::KaldiFbankOptions frontend;
    modules::LinearWeights first, second, dnn, classifier;
    std::array<Block, kLayers> blocks;
    bool causal;
    std::unique_ptr<ggml_context, ContextDeleter> context;
    std::unique_ptr<ggml_gallocr, AllocatorDeleter> allocator;
    ggml_cgraph * graph = nullptr;
    core::HostGraphPlan plan;
    int64_t capacity = 0;
    core::TensorValue input, output;
    std::array<core::TensorValue, kLayers> cache_inputs, cache_outputs;
    std::array<std::vector<float>, kLayers> caches;
    DetectionOptions options;
    std::vector<float> pending;
    std::vector<runtime::SpeechSegment> segments;
    std::deque<double> smooth;
    double smooth_sum = 0;
    int64_t total_samples = 0, frame_count = 0, speech_count = 0, silence_count = 0, last_start = -1, last_end = -1;
    bool hit_max = false, finished = false;
    State state = State::Silence;
};

FireRedDfsmnRuntime::FireRedDfsmnRuntime(std::shared_ptr<const assets::TensorSource> source, core::ExecutionContext & execution)
    : impl_(std::make_unique<Impl>(std::move(source), execution)) {}
FireRedDfsmnRuntime::~FireRedDfsmnRuntime() = default;
bool FireRedDfsmnRuntime::causal() const { return impl_->causal; }

std::vector<float> FireRedDfsmnRuntime::extract_features(const runtime::AudioBuffer & audio) const {
    validate_audio(audio.sample_rate, audio.channels, audio.samples);
    const auto started = std::chrono::steady_clock::now();
    auto result = audio::extract_kaldi_fbank(audio.samples, impl_->frontend);
    debug::timing_log_scalar("firered_vad.frontend_ms", debug::elapsed_ms(started));
    return result.values;
}

std::vector<float> FireRedDfsmnRuntime::infer_features(const std::vector<float> & features, bool retain_state) {
    return impl_->infer(features, retain_state);
}

runtime::TaskResult FireRedDfsmnRuntime::detect(const runtime::AudioBuffer & audio, const DetectionOptions & options) {
    validate_options(options);
    auto features = extract_features(audio);
    std::vector<float> probabilities;
    const size_t chunk = static_cast<size_t>(options.chunk_frames) * 80;
    for (size_t offset = 0; offset < features.size(); offset += chunk) {
        const size_t end = std::min(features.size(), offset + chunk);
        auto values = impl_->infer({features.begin() + offset, features.begin() + end}, false);
        probabilities.insert(probabilities.end(), values.begin(), values.end());
    }
    runtime::TaskResult result;
    if (causal()) {
        impl_->reset(options);
        impl_->total_samples = audio.samples.size();
        impl_->consume(probabilities);
        return finalize();
    }
    result.speech_segments = segment_probabilities(probabilities, audio.samples.size(), options);
    return result;
}

void FireRedDfsmnRuntime::reset(const DetectionOptions & options) { impl_->reset(options); }

runtime::StreamEvent FireRedDfsmnRuntime::process(const runtime::AudioChunk & chunk) {
    if (!causal()) throw std::runtime_error("FireRed VAD streaming requires the Stream-VAD checkpoint");
    if (impl_->finished) throw std::runtime_error("FireRed VAD stream has finished; reset before reuse");
    validate_audio(chunk.sample_rate, chunk.channels, chunk.samples);
    if (chunk.start_sample != impl_->total_samples) throw std::runtime_error("FireRed VAD streaming audio must be contiguous");
    impl_->total_samples += chunk.samples.size();
    impl_->pending.insert(impl_->pending.end(), chunk.samples.begin(), chunk.samples.end());
    const int64_t frames = impl_->pending.size() < kWindow ? 0 : 1 + (impl_->pending.size() - kWindow) / kHop;
    if (!frames) return {};
    const int64_t samples = (frames - 1) * kHop + kWindow;
    runtime::AudioBuffer audio{16000, 1, {impl_->pending.begin(), impl_->pending.begin() + samples}};
    auto features = extract_features(audio);
    impl_->pending.erase(impl_->pending.begin(), impl_->pending.begin() + frames * kHop);
    return impl_->consume(impl_->infer(features, true));
}

runtime::TaskResult FireRedDfsmnRuntime::finalize() {
    if (!impl_->finished && impl_->last_start >= 0) {
        impl_->segments.push_back(make_segment((impl_->last_start - 1) * kHop, std::max<int64_t>(0, impl_->frame_count - 1) * kHop, impl_->total_samples));
        impl_->last_start = -1;
    }
    impl_->finished = true;
    runtime::TaskResult result;
    result.speech_segments = impl_->segments;
    return result;
}

}  // namespace engine::models::firered_vad
