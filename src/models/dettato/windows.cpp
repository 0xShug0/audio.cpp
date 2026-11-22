#include "windows.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"

#include <algorithm>
#include <cmath>
#include <future>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

struct WindowTranscriber::Impl {
    const ExecutionContext &execution;
    HarmonicFrontend frontend;
    LaneTranscriber network;
    std::unique_ptr<PreparedGraph> average;
    std::array<TensorValue, 3> raw;
    std::array<TensorValue, 2> indices;
    TensorValue taper;
    // Full maps/hidden/weight, then subgroup maps/hidden/stem/weight.
    std::array<TensorValue, 7> banks;
    std::array<TensorValue, 5> results;
    std::array<std::shared_ptr<std::vector<ggml_fp16_t>>, 3> host_taps{
        std::make_shared<std::vector<ggml_fp16_t>>(), std::make_shared<std::vector<ggml_fp16_t>>(),
        std::make_shared<std::vector<ggml_fp16_t>>()};
    int64_t frames = 0;
    const int64_t hop_samples;
    bool ready = false;
    std::chrono::steady_clock::time_point window_start;

    Impl(const assets::TensorSource &source, const ExecutionContext &context, int window_hop_sec)
        : execution(context), frontend(source, "front", context.config().threads), network(source, context),
          hop_samples(window_hop_sec * 16000) {
        if (window_hop_sec != 2 && window_hop_sec != 4)
            throw std::invalid_argument("Dettato window_hop_sec must be 2 or 4");
        if (window_hop_sec == 4) host_taps[1] = host_taps[0];
    }

    void prepare(int64_t count) {
        if (average && count == frames)
            return;
        average.reset();
        frames = count;
        const int64_t windows = (frames + 99) / 100 + 2;
        const int64_t rows = (windows - 1) * 100 + 400;
        average = std::make_unique<PreparedGraph>(
            execution, PreparedGraphConfig{"dettato.window_average", 1024 * 1024, 1024, 1024 * 1024});
        auto &persistent = average->persistent_build_context();
        raw[0] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 6, 400, 88}));
        raw[1] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 368, 400, 96}));
        raw[2] = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({2, 368, 400, 88}));
        taper = make_tensor(persistent, GGML_TYPE_F32, TensorShape::from_dims({400, 1}));
        for (auto &value : indices) {
            value = make_tensor(persistent, GGML_TYPE_I32, TensorShape::from_dims({400}));
            ggml_set_input(value.tensor);
        }
        const std::array<int64_t, 7> widths{528, 88 * 368, 1, 528, 88 * 368, 88 * 368, 1};
        for (size_t i = 0; i < banks.size(); ++i) {
            if (hop_samples == 64000 && (i == 3 || i == 4 || i == 6)) {
                banks[i] = banks[i == 3 ? 0 : i == 4 ? 1 : 2];
                continue;
            }
            banks[i] = make_tensor(persistent, (i == 1 || i == 4 || i == 5) ? GGML_TYPE_F16 : GGML_TYPE_F32,
                                   TensorShape::from_dims({rows, widths[i]}));
        }
        const auto second = average->add_graph("odd_window");
        const auto finish = average->add_graph("finish");
        auto &ctx = average->build_context();
        for (int batch = 0; batch < 2; ++batch) {
            auto probabilities = SliceModule({0, batch, 1}).build(ctx, raw[0]);
            probabilities = SigmoidModule{}.build(ctx, probabilities);
            probabilities = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, probabilities);
            probabilities = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, probabilities),
                                           TensorShape::from_dims({400, 528}));
            probabilities = MulModule{true}.build(ctx, probabilities, taper);
            const auto weighted = [&](const TensorValue &input) {
                auto value = SliceModule({0, batch, 1}).build(ctx, input);
                value = SliceModule({3, 0, 88}).build(ctx, value);
                value = TransposeModule({{0, 2, 3, 1}, 4}).build(ctx, value);
                value = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, value),
                                       TensorShape::from_dims({400, 88 * 368}));
                value = MulModule{true}.build(ctx, value, taper);
                // The contribution rounds before addition to the half bank.
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), value.shape, GGML_TYPE_F16);
                return wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape);
            };
            const auto hidden = weighted(raw[2]);
            const auto graph = batch ? second : PreparedGraph::GraphId{0};
            const auto accumulate = [&](size_t index, const TensorValue &contribution) {
                auto previous = wrap_tensor(ggml_get_rows(ctx.ggml, banks[index].tensor, indices[batch].tensor),
                                            contribution.shape);
                const auto sum = AddModule{}.build(ctx, previous, contribution);
                auto *update = ggml_set_rows(ctx.ggml, banks[index].tensor, sum.tensor, indices[batch].tensor);
                ggml_set_output(update);
                ggml_build_forward_expand(average->graph(graph), update);
            };
            accumulate(0, probabilities);
            accumulate(1, hidden);
            accumulate(2, taper);
            if (batch == 0 || hop_samples == 64000) {
                if (hop_samples != 64000) {
                    accumulate(3, probabilities);
                    accumulate(4, hidden);
                    accumulate(6, taper);
                }
                accumulate(5, weighted(raw[1]));
            }
        }
        const std::array<size_t, 5> source{0, 3, 1, 4, 5};
        for (size_t i = 0; i < source.size(); ++i) {
            if (hop_samples == 64000 && (i == 1 || i == 3)) {
                results[i] = results[i - 1];
                continue;
            }
            auto value = SliceModule({0, 200, frames}).build(ctx, banks[source[i]]);
            value = ensure_backend_addressable_layout(ctx, value);
            const size_t weight_index = (i == 0 || i == 2) ? 2 : 6;
            auto weight = SliceModule({0, 200, frames}).build(ctx, banks[weight_index]);
            weight = wrap_tensor(ggml_clamp(ctx.ggml, weight.tensor, 1e-6f, INFINITY), weight.shape);
            if (i >= 2) {
                weight = wrap_tensor(ggml_cast(ctx.ggml, weight.tensor, GGML_TYPE_F16), weight.shape, GGML_TYPE_F16);
                weight = wrap_tensor(ggml_cast(ctx.ggml, weight.tensor, GGML_TYPE_F32), weight.shape);
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F32), value.shape);
            }
            value = wrap_tensor(ggml_div(ctx.ggml, value.tensor, weight.tensor), value.shape);
            if (i >= 2)
                value = wrap_tensor(ggml_cast(ctx.ggml, value.tensor, GGML_TYPE_F16), value.shape, GGML_TYPE_F16);
            results[i] = value;
            ggml_set_output(value.tensor);
            ggml_build_forward_expand(average->graph(finish), value.tensor);
        }
        average->allocate_persistent();
        average->allocate();
        std::vector<float> hann(400);
        for (size_t i = 0; i < hann.size(); ++i)
            hann[i] = static_cast<float>(0.5 - 0.5 * std::cos(2 * 3.14159265358979323846 * (i + 1) / 401));
        write_tensor_f32(taper, hann);
    }

    void run_device(const std::vector<float> &audio) {
        ready = false;
        if (audio.empty())
            throw std::runtime_error("Dettato needs nonempty audio");
        window_start = std::chrono::steady_clock::now();
        const auto samples = static_cast<int64_t>(audio.size());
        prepare((samples + 319) / 320);
        for (size_t i = 0; i < banks.size(); ++i)
            if (hop_samples != 64000 || (i != 3 && i != 4 && i != 6))
                ggml_backend_tensor_memset(banks[i].tensor, 0, 0, ggml_nbytes(banks[i].tensor));
        double frontend_ms = 0, averaging_ms = 0;
        const auto prepare_features = [&](int64_t first) {
            const auto frontend_start = std::chrono::steady_clock::now();
            std::vector<float> windows(256000, 0);
            for (int64_t batch = 0; batch < 2; ++batch) {
                const auto origin = first + batch * hop_samples - 64000;
                const auto lo = std::max<int64_t>(origin, 0), hi = std::min<int64_t>(origin + 128000, samples);
                if (hi > lo)
                    std::copy(audio.begin() + lo, audio.begin() + hi, windows.begin() + batch * 128000 + lo - origin);
            }
            auto features = frontend.extract(windows, 2);
            return std::make_pair(std::move(features), debug::elapsed_ms(frontend_start));
        };
        // Only one frontend job is outstanding. Its private next-window data
        // overlaps GPU execution; no graph or FFT workspace is used concurrently.
        const bool prefetch = execution.backend_type() != BackendType::Cpu;
        std::future<std::pair<HarmonicFeatures, double>> next_features;
        std::vector<int32_t> rows(400);
        for (int64_t first = 0; first < samples + 64000; first += 2 * hop_samples) {
            auto prepared = next_features.valid() ? next_features.get() : prepare_features(first);
            if (prefetch && first + 2 * hop_samples < samples + 64000)
                next_features = std::async(std::launch::async, prepare_features, first + 2 * hop_samples);
            frontend_ms += prepared.second;
            const auto & features = prepared.first;
            const auto outputs = network.run_device(features.values, 2, 800);
            const auto average_start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < raw.size(); ++i)
                ggml_backend_tensor_copy(outputs[i].tensor, raw[i].tensor);
            for (int64_t batch = 0; batch < 2 && first + batch * hop_samples < samples + 64000; ++batch) {
                for (size_t i = 0; i < rows.size(); ++i)
                    rows[i] = static_cast<int32_t>((first + batch * hop_samples) / 320 + i);
                write_tensor_i32(indices[batch], rows);
                average->compute(static_cast<size_t>(batch));
            }
            averaging_ms += debug::elapsed_ms(average_start);
        }
        const auto finish_start = std::chrono::steady_clock::now();
        average->compute(2);
        debug::timing_log_scalar("dettato.transcriber.average_ms", averaging_ms);
        debug::timing_log_scalar("dettato.transcriber.finish_ms", debug::elapsed_ms(finish_start));
        debug::timing_log_scalar("dettato.transcriber.frontend_ms", frontend_ms);
        ready = true;
    }

    WindowTranscription readback() {
        if (!ready) throw std::runtime_error("Dettato windows require a completed run_device() before readback");
        const auto readback_start = std::chrono::steady_clock::now();
        WindowTranscription out;
        out.frames = frames;
        out.hidden = host_taps[0];
        out.grouping_hidden = host_taps[1];
        out.grouping_stem = host_taps[2];
        for (size_t group = 0; group < 2; ++group) {
            const auto values = read_tensor_f32(results[group].tensor);
            auto &maps = group ? out.grouping_maps : out.maps;
            for (size_t head = 0; head < maps.size(); ++head) {
                maps[head].resize(static_cast<size_t>(frames * 88));
                for (int64_t t = 0; t < frames; ++t)
                    std::copy_n(values.begin() + t * 528 + head * 88, 88, maps[head].begin() + t * 88);
            }
        }
        const std::array<std::vector<ggml_fp16_t> *, 3> target{out.hidden.get(), out.grouping_hidden.get(), out.grouping_stem.get()};
        for (size_t i = 0; i < target.size(); ++i) {
            if (i == 1 && hop_samples == 64000) continue;
            target[i]->resize(static_cast<size_t>(frames * 88 * 368));
            ggml_backend_tensor_get(results[i + 2].tensor, target[i]->data(), 0,
                                    target[i]->size() * sizeof(ggml_fp16_t));
        }
        debug::timing_log_scalar("dettato.transcriber.readback_ms", debug::elapsed_ms(readback_start));
        debug::timing_log_scalar("dettato.transcriber.windows_ms", debug::elapsed_ms(window_start));
        return out;
    }
};

WindowTranscriber::WindowTranscriber(const assets::TensorSource &source, const core::ExecutionContext &execution, int window_hop_sec)
    : impl_(std::make_unique<Impl>(source, execution, window_hop_sec)) {}
WindowTranscriber::~WindowTranscriber() = default;
void WindowTranscriber::run_device(const std::vector<float> &audio16k) {
    impl_->run_device(audio16k);
}
WindowTranscription WindowTranscriber::readback() {
    return impl_->readback();
}
} // namespace engine::models::dettato::detail
