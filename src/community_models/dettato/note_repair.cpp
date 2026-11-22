#include "engine/community_models/dettato/note_repair.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/feed_forward_modules.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/community_models/dettato/transcriber.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>

namespace engine::community_models::dettato::detail {

namespace {
bool peak(const std::vector<float> &values, int64_t t, int lane, int64_t frames, float threshold) {
    const float value = values[static_cast<size_t>(t * 88 + lane)];
    const float left = t ? values[static_cast<size_t>((t - 1) * 88 + lane)] : -1;
    const float right = t + 1 < frames ? values[static_cast<size_t>((t + 1) * 88 + lane)] : -1;
    return value >= threshold && value > left && value >= right;
}

double refine(const std::vector<float> &values, int64_t t, int lane, int64_t frames) {
    const float value = values[static_cast<size_t>(t * 88 + lane)];
    const float left = t ? values[static_cast<size_t>((t - 1) * 88 + lane)] : value;
    const float right = t + 1 < frames ? values[static_cast<size_t>((t + 1) * 88 + lane)] : value;
    const float denominator = 2 * (value - std::min(left, right));
    return denominator <= 1e-6f ? 0 : std::clamp((right - left) / denominator, -0.5f, 0.5f);
}
} // namespace

std::vector<Note> decode_notes(const std::array<std::vector<float>, 6> &maps, int64_t frames, double hop,
                               const NoteThresholds &thresholds) {
    if (frames <= 0 || hop <= 0 || thresholds.low_frames <= 0)
        throw std::runtime_error("Invalid Dettato note-decoding shape or thresholds");
    for (const auto &values : maps)
        if (values.size() != static_cast<size_t>(frames * 88))
            throw std::runtime_error("Dettato probability-map shape mismatch");
    std::vector<Note> notes;
    for (int voice = 0; voice < 2; ++voice) {
        const auto &on = maps[voice * 3];
        const auto &fr = maps[voice * 3 + 1];
        const auto &off = maps[voice * 3 + 2];
        for (int lane = 0; lane < 88; ++lane) {
            std::vector<int64_t> starts, ends, lows;
            for (int64_t t = 0; t < frames; ++t) {
                if (peak(on, t, lane, frames, voice ? thresholds.second_onset : thresholds.onset))
                    starts.push_back(t);
                if (peak(off, t, lane, frames, thresholds.offset))
                    ends.push_back(t);
                bool low = true;
                for (int k = 0; k < thresholds.low_frames && t + k < frames; ++k)
                    low = low && fr[static_cast<size_t>((t + k) * 88 + lane)] < thresholds.frame;
                if (low)
                    lows.push_back(t);
            }
            if (thresholds.onset_nms > 0 && starts.size() > 1) {
                std::vector<size_t> order(starts.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                    return on[static_cast<size_t>(starts[a] * 88 + lane)] >
                           on[static_cast<size_t>(starts[b] * 88 + lane)];
                });
                std::vector<bool> keep(starts.size(), true);
                for (const auto i : order) {
                    if (!keep[i])
                        continue;
                    for (size_t j = 0; j < starts.size(); ++j)
                        if (i != j && std::abs(starts[i] - starts[j]) <= thresholds.onset_nms)
                            keep[j] = false;
                }
                size_t count = 0;
                for (size_t i = 0; i < starts.size(); ++i)
                    if (keep[i])
                        starts[count++] = starts[i];
                starts.resize(count);
            }
            for (size_t i = 0; i < starts.size(); ++i) {
                const auto t = starts[i], next = i + 1 < starts.size() ? starts[i + 1] : frames;
                const auto low = std::lower_bound(lows.begin(), lows.end(), t + 1);
                const auto end = std::lower_bound(ends.begin(), ends.end(), t + 2);
                const auto frame_end = low == lows.end() ? frames : *low;
                const auto offset_end = end == ends.end() ? frames : *end;
                const double onset = t + refine(on, t, lane, frames);
                double stop;
                if (offset_end < frames && offset_end <= std::min(frame_end, next))
                    stop = offset_end + refine(off, offset_end, lane, frames);
                else if (frame_end <= next)
                    stop = frame_end - 0.5;
                else
                    stop = next + (next < frames ? refine(on, next, lane, frames) : 0);
                const auto confidence_frame =
                    std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(onset)), 0, frames - 1);
                notes.push_back({onset * hop, std::max(stop * hop, onset * hop + 0.02), 21 + lane, 100,
                                 on[static_cast<size_t>(confidence_frame * 88 + lane)], voice + 1, false});
            }
        }
    }
    std::sort(notes.begin(), notes.end(), [](const Note &a, const Note &b) {
        return std::tie(a.onset, a.offset, a.pitch) < std::tie(b.onset, b.offset, b.pitch);
    });
    return notes;
}

using namespace core;
using namespace modules;

std::vector<float> extract_note_evidence(const WindowTranscription &transcription, const std::vector<Note> &notes,
                                         size_t threads) {
    const auto start = std::chrono::steady_clock::now();
    (void)threads;
    const auto frames = transcription.frames;
    if (frames <= 0 || transcription.hidden->size() != static_cast<size_t>(frames * 88 * 368))
        throw std::runtime_error("Dettato evidence requires matching time/key/channel states");
    std::vector<float> result(notes.size() * 769), at(notes.size() * 368), mean(at.size());
    std::vector<int64_t> starts(notes.size()), ends(notes.size());
    std::array<std::vector<size_t>, 88> lanes;
    const auto clip = [&](int64_t frame) { return std::clamp<int64_t>(frame, 0, frames - 1); };
    for (size_t i = 0; i < notes.size(); ++i) {
        const auto &note = notes[i];
        const int lane = note.pitch - 21;
        if (lane < 0 || lane >= 88 || (note.voice != 1 && note.voice != 2))
            throw std::runtime_error("Invalid Dettato evidence note key/voice");
        const auto a = clip(static_cast<int64_t>(std::nearbyint(note.onset / 0.02)));
        const auto b = std::clamp<int64_t>(static_cast<int64_t>(std::nearbyint(note.offset / 0.02)), a + 1, frames);
        starts[i] = a;
        ends[i] = b;
        lanes[static_cast<size_t>(lane)].push_back(i);
        const int own = note.voice == 1 ? 0 : 3, other = 3 - own;
        const auto get = [&](int head, int64_t t, int key) {
            return transcription.maps[head][static_cast<size_t>(clip(t) * 88 + key)];
        };
        auto *output = result.data() + i * 769;
        size_t k = 0;
        for (const int shift : {-2, -1, 0, 1, 2})
            output[k++] = get(own, a + shift, lane);
        for (int j = 0; j < 8; ++j)
            output[k++] = get(own + 1, a + j * (b - a) / 8, lane);
        for (const int shift : {1, 3, 6, 12})
            output[k++] = get(own + 1, b - 1 + shift, lane);
        float before = 0;
        for (int shift = 1; shift <= 5; ++shift)
            if (a - shift >= 0)
                before = std::max(before, get(own + 1, a - shift, lane));
        output[k++] = before;
        float peak = 0, position = 0;
        if (a + 1 < std::min(b + 12, frames)) {
            int64_t best = a + 1;
            peak = get(own + 2, best, lane);
            for (int64_t t = a + 2; t < std::min(b + 12, frames); ++t)
                if (get(own + 2, t, lane) > peak) {
                    best = t;
                    peak = get(own + 2, t, lane);
                }
            position = std::min(static_cast<float>(best - a) / (b - a), 2.0f);
        }
        output[k++] = peak;
        output[k++] = position;
        output[k++] = get(own + 2, b - 1, lane);
        output[k++] = std::max({get(other, a - 1, lane), get(other, a, lane), get(other, a + 1, lane)});
        output[k++] = get(other + 1, a, lane);
        float other_mean = 0;
        for (int j = 0; j < 4; ++j)
            other_mean += get(other + 1, a + j * (b - a) / 4, lane);
        output[k++] = other_mean / 4;
        output[k++] = note.voice == 2 ? 1.0f : 0.0f;
        for (const int shift : {-1, 1, -12, 12, 19, 24}) {
            const int key = lane + shift;
            output[k++] =
                (key < 0 || key >= 88) ? 0.0f : std::max({get(0, a - 1, key), get(0, a, key), get(0, a + 1, key)});
        }
    }
    // Keep exactly the F32 prefix endpoints that the notes read. The running
    // F64 sum and its order match torch.cumsum; a recording-sized prefix table
    // is unnecessary. Independent lanes can execute concurrently.
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
    for (int64_t lane = 0; lane < static_cast<int64_t>(lanes.size()); ++lane) {
        if (lanes[lane].empty())
            continue;
        std::vector<std::tuple<int64_t, size_t, bool>> endpoints;
        endpoints.reserve(lanes[lane].size() * 2);
        for (const auto i : lanes[lane]) {
            endpoints.emplace_back(starts[i], i, false);
            endpoints.emplace_back(ends[i], i, true);
        }
        std::sort(endpoints.begin(), endpoints.end());
        std::array<double, 368> accumulated{};
        std::array<float, 368> row;
        int64_t t = 0;
        for (const auto &endpoint : endpoints) {
            const auto frame = std::get<0>(endpoint), i = static_cast<int64_t>(std::get<1>(endpoint));
            while (t < frame) {
                ggml_fp16_to_fp32_row(transcription.hidden->data() + (t * 88 + lane) * 368, row.data(), 368);
                for (size_t c = 0; c < 368; ++c)
                    accumulated[c] += row[c];
                ++t;
            }
            if (!std::get<2>(endpoint)) {
                ggml_fp16_to_fp32_row(transcription.hidden->data() + (starts[i] * 88 + lane) * 368, at.data() + i * 368,
                                      368);
                for (size_t c = 0; c < 368; ++c)
                    mean[i * 368 + c] = -static_cast<float>(accumulated[c]);
            } else {
                for (size_t c = 0; c < 368; ++c)
                    mean[i * 368 + c] = (static_cast<float>(accumulated[c]) + mean[i * 368 + c]) /
                                        static_cast<float>(ends[i] - starts[i]);
            }
        }
    }
    for (size_t i = 0; i < notes.size(); ++i) {
        for (size_t part = 0; part < 2; ++part) {
            const auto *vector = (part ? mean : at).data() + i * 368;
            double sum = 0, square = 0;
            for (size_t c = 0; c < 368; ++c)
                sum += vector[c];
            const float center = static_cast<float>(sum / 368);
            for (size_t c = 0; c < 368; ++c)
                square += static_cast<double>(vector[c] - center) * (vector[c] - center);
            const float sd = static_cast<float>(std::sqrt(square / 368)) + 1e-6f;
            result[i * 769 + 31 + part] = std::log(sd);
            for (size_t c = 0; c < 368; ++c) {
                const float z = std::clamp((vector[c] - center) / sd, -3.96f, 3.96f);
                result[i * 769 + 33 + part * 368 + c] = std::nearbyint(z * 32) / 32;
            }
        }
        for (size_t c = 0; c < 33; ++c)
            result[i * 769 + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(result[i * 769 + c]));
    }
    debug::timing_log_scalar("dettato.evidence.features_ms", debug::elapsed_ms(start));
    return result;
}

NoteRepairInputs make_note_repair_inputs(const std::vector<Note> &notes, const std::vector<float> &evidence,
                                         float window_start) {
    const size_t count = notes.size();
    if (evidence.size() != count * 769)
        throw std::runtime_error("Dettato evidence rows must match the input notes");
    NoteRepairInputs result;
    result.evidence = evidence;
    result.instrument.assign(count, 0);
    result.pitch.resize(count);
    result.velocity_bin.resize(count);
    result.continuous.resize(count * 110);
    result.cosine.resize(count * 14);
    result.sine.resize(count * 14);
    result.onsets.resize(count);
    result.durations.resize(count);
    auto &on = result.onsets;
    auto &duration = result.durations;
    std::vector<float> off(count);
    constexpr float pi2 = 6.2831853071795864769f;
    std::array<float, 72> periods{}, strength{}, phase{};
    for (size_t i = 0; i < count; ++i) {
        result.pitch[i] = notes[i].pitch;
        result.velocity_bin[i] =
            static_cast<int32_t>(std::clamp(
                std::nearbyint(1 + 126 * (static_cast<double>(notes[i].confidence) - 0.25) / 0.75), 1.0, 127.0)) /
            8;
        const float a = static_cast<float>(std::nearbyint(notes[i].onset * 10000) / 10000);
        const float b = static_cast<float>(
            std::max(std::nearbyint(notes[i].offset * 10000) / 10000, static_cast<double>(a) + 0.001));
        on[i] = a - window_start;
        duration[i] = std::max(b - a, 0.0f);
        off[i] = on[i] + duration[i];
    }
    for (size_t k = 0; k < periods.size(); ++k) {
        periods[k] = std::exp(std::log(0.18f) + (std::log(1.5f) - std::log(0.18f)) * static_cast<float>(k) / 71);
        float real = 0, imag = 0;
        for (const auto value : on) {
            const float angle = pi2 * value / periods[k];
            real += std::cos(angle);
            imag += std::sin(angle);
        }
        strength[k] = std::sqrt(real * real + imag * imag) / static_cast<float>(std::max<size_t>(count, 1));
        phase[k] = std::atan2(imag, real);
    }
    auto remaining = strength;
    std::array<size_t, 2> selected;
    for (auto &best : selected) {
        best = static_cast<size_t>(std::max_element(remaining.begin(), remaining.end()) - remaining.begin());
        for (size_t k = 0; k < remaining.size(); ++k)
            if (std::abs(static_cast<int>(k) - static_cast<int>(best)) <= 3)
                remaining[k] = -1;
    }
    for (size_t i = 0; i < count; ++i) {
        auto *output = result.continuous.data() + i * 110;
        const float log_duration = std::log(std::max(duration[i], 1e-4f));
        const std::array<float, 3> values{on[i], log_duration, off[i]};
        for (size_t group = 0; group < values.size(); ++group) {
            const float lo = group == 1 ? 0.04f : 0.01f, hi = group == 1 ? 40.0f : 32.0f;
            for (int k = 0; k < 12; ++k) {
                const float wavelength =
                    std::exp(std::log(lo) + (std::log(hi) - std::log(lo)) * static_cast<float>(k) / 11);
                const float angle = pi2 * values[group] / wavelength;
                output[group * 24 + k] = std::sin(angle);
                output[group * 24 + 12 + k] = std::cos(angle);
            }
        }
        output[72] = (log_duration + 3) / 3;
        output[73] = std::min(duration[i], 8.0f) / 2;
        float previous = 10, previous_off = 0, next = 10, chord = 0, density = 0, near = 0;
        for (size_t j = 0; j < count; ++j) {
            const float dt = on[j] - on[i], distance = std::abs(dt);
            if (distance < 0.035f)
                ++chord;
            if (distance < 0.5f)
                ++density;
            if (i == j || notes[i].pitch != notes[j].pitch)
                continue;
            if (dt <= 0 && -dt < previous) {
                previous = -dt;
                previous_off = off[j];
            }
            if (dt > 0 && dt < next)
                next = dt;
            if (distance < 0.25f)
                ++near;
        }
        const std::array<float, 4> gaps{previous, previous < 10 ? on[i] - previous_off : 10, next,
                                        next < 10 ? off[i] - (on[i] + next) : -10};
        for (size_t k = 0; k < gaps.size(); ++k)
            output[74 + k] = std::copysign(std::log1p(std::abs(gaps[k]) * 20), gaps[k]);
        output[78] = std::log1p(chord);
        output[79] = std::log1p(density);
        output[80] = std::log1p(density);
        output[81] = std::log1p(near);
        size_t column = 82;
        for (const auto best : selected) {
            const float base = pi2 * on[i] / periods[best] - phase[best];
            for (const int subdivision : {1, 2, 3, 4, 6, 8}) {
                output[column++] = std::sin(subdivision * base);
                output[column++] = std::cos(subdivision * base);
            }
            output[column++] = strength[best];
            output[column++] = std::log(periods[best]);
        }
        for (int pair = 0; pair < 14; ++pair) {
            const bool time = pair < 10;
            const float lo = time ? 0.02f : 2.0f, hi = time ? 40.0f : 96.0f;
            const int k = time ? pair : pair - 10, n = time ? 10 : 4;
            const float wavelength =
                std::exp(std::log(lo) + (std::log(hi) - std::log(lo)) * static_cast<float>(k) / (n - 1));
            const float angle = (time ? on[i] : static_cast<float>(notes[i].pitch)) * (pi2 / wavelength);
            result.cosine[i * 14 + pair] = std::cos(angle);
            result.sine[i * 14 + pair] = std::sin(angle);
        }
    }
    return result;
}

NoteRepairNetwork::NoteRepairNetwork(NoteRepairWeights weights, const ExecutionContext &execution)
    : execution_(execution), weights_(std::move(weights)), graphs_(32) {}

TensorValue NoteRepairNetwork::linear(ModuleBuildContext &ctx, const TensorValue &x, const LinearWeights &weights) {
    return LinearModule(
               {weights.weight.shape.dims[1], weights.weight.shape.dims[0], weights.bias.has_value(), GGML_PREC_F32})
        .build(ctx, x, weights);
}

TensorValue NoteRepairNetwork::rms(ModuleBuildContext &ctx, const TensorValue &x, const NormWeights &weights) {
    return RMSNormModule({192, 1e-6f, true, false}).build(ctx, x, weights);
}

NoteRepairNetwork::GraphState &NoteRepairNetwork::prepare(int64_t notes) {
    if (notes <= 0 || notes > 4096)
        throw std::runtime_error("Dettato repair graph expects 1..4096 notes");
    if (auto *found = graphs_.find(notes))
        return **found;
    auto state = std::make_unique<GraphState>();
    auto &graph_ = state->graph;
    auto &inputs_ = state->inputs;
    auto &outputs_ = state->outputs;
    auto &probabilities_ = state->probabilities;
    graph_ =
        std::make_unique<PreparedGraph>(execution_, PreparedGraphConfig{"dettato.evidence", 2u * 1024 * 1024, 8192});
    auto &ctx = graph_->build_context();
    for (int i = 0; i < 3; ++i)
        inputs_[i] = make_tensor(ctx, GGML_TYPE_I32, TensorShape::from_dims({1, notes}));
    inputs_[3] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, 110}));
    inputs_[4] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, notes, 769}));
    for (int i = 5; i < 7; ++i)
        inputs_[i] = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, notes, 14}));
    for (const auto &value : inputs_)
        ggml_set_input(value.tensor);
    auto x = EmbeddingModule({130, 192}).build(ctx, inputs_[0], weights_.instrument_embedding);
    x = AddModule{}.build(ctx, x, EmbeddingModule({256, 192}).build(ctx, inputs_[1], weights_.pitch_embedding));
    x = AddModule{}.build(ctx, x, EmbeddingModule({16, 192}).build(ctx, inputs_[2], weights_.velocity_embedding));
    const auto continuous =
        linear(ctx, SiluModule{}.build(ctx, linear(ctx, inputs_[3], weights_.continuous[0])), weights_.continuous[1]);
    const auto evidence =
        linear(ctx, SiluModule{}.build(ctx, linear(ctx, inputs_[4], weights_.evidence[0])), weights_.evidence[1]);
    x = rms(ctx, AddModule{}.build(ctx, AddModule{}.build(ctx, x, continuous), evidence), weights_.input_norm);
    // Two continuous position axes rotate 14 interleaved pairs; four head
    // channels pass through. Existing pair modules express the exact math.
    std::array<TensorValue, 2> tables;
    for (int i = 0; i < 2; ++i) {
        auto table = RepeatModule({TensorShape::from_dims({1, 6, notes, 14})}).build(ctx, inputs_[5 + i]);
        tables[i] = reshape_tensor(ctx, table, TensorShape::from_dims({6, notes, 14, 1}));
    }
    const auto rotate = [&](TensorValue value) {
        const auto original = value;
        value = SliceModule({3, 0, 28}).build(ctx, value);
        value = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, value),
                               TensorShape::from_dims({6, notes, 14, 2}));
        value = SplitRoPEModule({2}).build(ctx, value, tables[0], tables[1]);
        value = reshape_tensor(ctx, value, TensorShape::from_dims({1, 6, notes, 28}));
        return ConcatModule({3}).build(ctx, value, SliceModule({3, 28, 4}).build(ctx, original));
    };
    for (const auto &block : weights_.blocks) {
        const auto normalized = rms(ctx, x, block.attention_norm);
        const auto q = linear(ctx, normalized, block.query);
        const auto kv = linear(ctx, normalized, block.key_value);
        std::array<TensorValue, 3> heads{q, SliceModule({2, 0, 192}).build(ctx, kv),
                                         SliceModule({2, 192, 192}).build(ctx, kv)};
        for (auto &head : heads) {
            head = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, head),
                                  TensorShape::from_dims({1, notes, 6, 32}));
            head = TransposeModule({{0, 2, 1, 3}, 4}).build(ctx, head);
        }
        heads[0] = rotate(heads[0]);
        heads[1] = rotate(heads[1]);
        auto attended =
            ScaledDotProductAttentionModule({32, ScaledDotProductAttentionLowering::Explicit, GGML_PREC_F32})
                .build(ctx, heads[0], heads[1], heads[2]);
        attended = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, attended),
                                  TensorShape::from_dims({1, notes, 192}));
        x = AddModule{}.build(ctx, x, linear(ctx, attended, block.output));
        GatedFeedForwardConfig cfg;
        cfg.hidden_size = 192;
        cfg.intermediate_size = 576;
        cfg.activation = GatedFeedForwardActivation::Silu;
        cfg.packed_gate_first = true;
        cfg.projection_precision = GGML_PREC_F32;
        x = AddModule{}.build(ctx, x, GatedFeedForwardModule(cfg).build(ctx, rms(ctx, x, block.ffn_norm), block.ffn));
    }
    x = rms(ctx, x, weights_.final_norm);
    for (size_t i = 0; i < weights_.heads.size(); ++i) {
        outputs_[i] = linear(ctx, x, weights_.heads[i]);
        ggml_set_output(outputs_[i].tensor);
        // Linear's logical output can be a reshape view. Retain its storage
        // while probability heads consume it and callers inspect raw logits.
        if (outputs_[i].tensor->view_src)
            ggml_set_output(outputs_[i].tensor->view_src);
        ggml_build_forward_expand(graph_->graph(), outputs_[i].tensor);
    }
    probabilities_[0] = SigmoidModule{}.build(ctx, outputs_[0]);
    probabilities_[1] = SoftmaxModule{}.build(ctx, outputs_[1]);
    probabilities_[2] = SoftmaxModule{}.build(ctx, outputs_[2]);
    for (size_t slot = 3; slot < 5; ++slot) {
        const auto gate = SigmoidModule{}.build(ctx, SliceModule({2, 0, 1}).build(ctx, outputs_[slot]));
        const auto pitch = SliceModule({2, 1, 49}).build(ctx, outputs_[slot]);
        const auto time = SoftmaxModule{}.build(ctx, SliceModule({2, 50, 401}).build(ctx, outputs_[slot]));
        const auto duration = SoftmaxModule{}.build(ctx, SliceModule({2, 451, 128}).build(ctx, outputs_[slot]));
        probabilities_[slot] = ConcatModule({2}).build(
            ctx, ConcatModule({2}).build(ctx, ConcatModule({2}).build(ctx, gate, pitch), time), duration);
    }
    for (const auto &probability : probabilities_) {
        ggml_set_output(probability.tensor);
        ggml_build_forward_expand(graph_->graph(), probability.tensor);
    }
    debug::timing_log_context_reservation("dettato.evidence.graph", ctx.ggml);
    graph_->allocate();
    // Exact shapes retain their own plans/captures. Bound retained workspace
    // to 32 MiB, or one oversized graph; never retain all dense-note graphs.
    constexpr size_t workspace_budget = 32u * 1024 * 1024;
    const size_t required = graph_->workspace_bytes();
    while (graphs_.size()) {
        size_t retained = 0;
        graphs_.for_each_value([&](const auto &entry) { retained += entry->graph->workspace_bytes(); });
        if (retained + required <= workspace_budget)
            break;
        graphs_.set_capacity(graphs_.size() - 1);
        graphs_.set_capacity(32);
    }
    auto *prepared = state.get();
    graphs_.put(notes, std::move(state));
    return *prepared;
}

std::array<std::vector<float>, 5> NoteRepairNetwork::run(const NoteRepairInputs &inputs, bool probabilities) {
    auto &state = prepare(static_cast<int64_t>(inputs.instrument.size()));
    const auto &graph_ = state.graph;
    const auto &inputs_ = state.inputs;
    const auto &outputs_ = state.outputs;
    const auto &probabilities_ = state.probabilities;
    write_tensor_i32(inputs_[0], inputs.instrument);
    write_tensor_i32(inputs_[1], inputs.pitch);
    write_tensor_i32(inputs_[2], inputs.velocity_bin);
    write_tensor_f32(inputs_[3], inputs.continuous);
    write_tensor_f32(inputs_[4], inputs.evidence);
    write_tensor_f32(inputs_[5], inputs.cosine);
    write_tensor_f32(inputs_[6], inputs.sine);
    const auto start = std::chrono::steady_clock::now();
    graph_->compute();
    std::array<std::vector<float>, 5> result;
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = read_tensor_f32((probabilities ? probabilities_ : outputs_)[i].tensor);
    debug::timing_log_scalar("dettato.evidence.forward_ms", debug::elapsed_ms(start));
    return result;
}

std::vector<Note> decode_repair_candidates(const NoteRepairInputs &inputs,
                                           const std::array<std::vector<float>, 5> &probabilities) {
    const size_t count = inputs.pitch.size();
    if (inputs.onsets.size() != count || inputs.durations.size() != count)
        throw std::runtime_error("Dettato repair requires prepared note times");
    const std::array<size_t, 5> widths{1, 201, 161, 579, 579};
    for (size_t i = 0; i < widths.size(); ++i)
        if (probabilities[i].size() != count * widths[i])
            throw std::runtime_error("Dettato repair probability shape mismatch");
    const auto box_argmax = [](const float *values, int size, int half) {
        float maximum = -1;
        int best = 0;
        for (int i = 0; i < size; ++i) {
            float sum = 0;
            for (int j = std::max(0, i - half); j < std::min(size, i + half + 1); ++j)
                sum += values[j];
            if (sum > maximum) {
                maximum = sum;
                best = i;
            }
        }
        return best;
    };
    static const std::array<float, 128> duration_centers = [] {
        std::array<float, 128> result;
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = static_cast<float>(std::exp(std::log(.01) + (std::log(16.) - std::log(.01)) * (i + .5) / 128));
        return result;
    }();
    static const std::array<std::vector<size_t>, 128> duration_columns = [] {
        std::array<std::vector<size_t>, 128> result;
        for (size_t j = 0; j < result.size(); ++j)
            for (size_t i = 0; i < duration_centers.size(); ++i)
                if (std::abs(duration_centers[j] - duration_centers[i]) <= std::max(.2f * duration_centers[i], .05f))
                    result[j].push_back(i);
        return result;
    }();
    std::vector<Note> result(count * 3);
    for (size_t i = 0; i < count; ++i) {
        const float on = inputs.onsets[i];
        const float duration = std::max(inputs.durations[i], 1e-4f);
        const int onset_bin = box_argmax(probabilities[1].data() + i * 201, 201, 10);
        const int ratio_bin = box_argmax(probabilities[2].data() + i * 161, 161, 3);
        const float corrected_on = on + (onset_bin * .005f - .5f);
        const float corrected_duration = duration * std::exp(ratio_bin * .05f - 4.0f);
        result[i] = {corrected_on, corrected_on + corrected_duration, inputs.pitch[i], 100, probabilities[0][i], 1,
                     false};
        for (size_t slot = 0; slot < 2; ++slot) {
            const auto *values = probabilities[slot + 3].data() + i * 579;
            const int pitch_bin = static_cast<int>(std::max_element(values + 1, values + 50) - (values + 1));
            const int time_bin = box_argmax(values + 50, 401, 5);
            float maximum = -1;
            size_t duration_bin = 0;
            for (size_t j = 0; j < duration_columns.size(); ++j) {
                float score = 0;
                for (const auto k : duration_columns[j])
                    score += values[451 + k];
                if (score > maximum) {
                    maximum = score;
                    duration_bin = j;
                }
            }
            const float start = on + (time_bin * .01f - 1.0f), end = start + duration_centers[duration_bin];
            result[count + i * 2 + slot] = {
                start, end, std::clamp(inputs.pitch[i] + pitch_bin - 24, 0, 127), 100, values[0], 1, true};
        }
    }
    return result;
}

RepairResult repair_notes(NoteRepairNetwork &network, const std::vector<Note> &notes,
                          const std::vector<float> &evidence, float keep_threshold, float insertion_threshold) {
    if (evidence.size() != notes.size() * 769)
        throw std::runtime_error("Repair evidence must align with decoded notes");
    if (notes.empty())
        return {};
    std::vector<size_t> order(notes.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<Note> input = notes;
    for (auto &note : input) {
        const double on = std::nearbyint(note.onset * 10000) / 10000;
        note.onset = static_cast<float>(on);
        note.offset = static_cast<float>(std::max(std::nearbyint(note.offset * 10000) / 10000, on + 0.001));
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return std::tie(input[a].onset, input[a].pitch) < std::tie(input[b].onset, input[b].pitch);
    });
    std::vector<Note> sorted;
    sorted.reserve(input.size());
    std::vector<float> aligned(evidence.size());
    for (size_t i = 0; i < order.size(); ++i) {
        sorted.push_back(input[order[i]]);
        std::copy_n(evidence.begin() + order[i] * 769, 769, aligned.begin() + i * 769);
    }
    const double end = std::max_element(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
                           return a.offset < b.offset;
                       })->offset;
    RepairResult out;
    std::vector<Note> insertions;
    // Refining a peak at frame zero can yield a negative onset. Such notes lie
    // outside every repair window; upstream preserves their original values.
    for (const auto &input_note : sorted) {
        if (input_note.onset >= 0)
            break;
        auto note = input_note;
        note.confidence = 1;
        out.notes.push_back(note);
    }
    for (float start = 0; start < end + 8; start += 8) {
        const auto begin = std::lower_bound(sorted.begin(), sorted.end(), start,
                                            [](const auto &note, float time) { return note.onset < time; });
        const auto finish = std::lower_bound(begin, sorted.end(), start + 8,
                                             [](const auto &note, float time) { return note.onset < time; });
        const auto count = static_cast<size_t>(finish - begin);
        if (count == 0)
            continue;
        if (count > 4096) {
            // Upstream's documented repairer cap leaves this window unchanged.
            debug::log_message(debug::LogLevel::Warning,
                               "Dettato repair window exceeds 4096 notes; preserving transcribed notes");
            for (auto it = begin; it != finish; ++it) {
                auto note = *it;
                note.confidence = 1;
                out.notes.push_back(note);
            }
            continue;
        }
        const auto first = static_cast<size_t>(begin - sorted.begin());
        const std::vector<Note> window(begin, finish);
        const std::vector<float> features(aligned.begin() + first * 769, aligned.begin() + (first + count) * 769);
        const auto prepared = make_note_repair_inputs(window, features, start);
        const auto candidates = decode_repair_candidates(prepared, network.run(prepared, true));
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto note = candidates[i];
            if (note.onset < 0 || note.onset >= 8)
                continue;
            note.onset += start;
            note.offset += start;
            if (i < count && note.confidence >= keep_threshold)
                out.notes.push_back(note);
            else if (i >= count && note.confidence >= insertion_threshold)
                insertions.push_back(note);
        }
    }
    // Original notes are never deduplicated; both voices may share a key.
    insertions.erase(std::remove_if(insertions.begin(), insertions.end(),
                                    [&](const Note &insertion) {
                                        return std::any_of(out.notes.begin(), out.notes.end(), [&](const Note &kept) {
                                            return insertion.pitch == kept.pitch &&
                                                   std::abs(insertion.onset - kept.onset) < 0.03;
                                        });
                                    }),
                     insertions.end());
    std::vector<size_t> insertion_order(insertions.size());
    std::iota(insertion_order.begin(), insertion_order.end(), 0);
    std::stable_sort(insertion_order.begin(), insertion_order.end(), [&](size_t a, size_t b) {
        return std::tie(insertions[a].pitch, insertions[a].onset) < std::tie(insertions[b].pitch, insertions[b].onset);
    });
    std::vector<bool> keep(insertions.size(), true);
    for (size_t i = 0; i + 1 < insertion_order.size(); ++i) {
        const auto a = insertion_order[i], b = insertion_order[i + 1];
        if (!keep[a] || insertions[a].pitch != insertions[b].pitch || insertions[b].onset - insertions[a].onset >= 0.03)
            continue;
        if (insertions[a].confidence >= insertions[b].confidence)
            keep[b] = false;
        else
            keep[a] = false;
    }
    out.dropped = notes.size() - out.notes.size();
    for (size_t i = 0; i < insertions.size(); ++i)
        if (keep[i]) {
            out.notes.push_back(insertions[i]);
            ++out.inserted;
        }
    for (auto &note : out.notes) {
        const auto original_onset = note.onset;
        note.onset = std::nearbyint(original_onset * 10000) / 10000;
        note.offset = std::nearbyint(std::max(note.offset, original_onset + 0.01) * 10000) / 10000;
    }
    return out;
}
} // namespace engine::community_models::dettato::detail
