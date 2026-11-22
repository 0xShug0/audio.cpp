#include "drums.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "gru.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/modules/weight_binding.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

DrumEmbedder::DrumEmbedder(const assets::TensorSource & source, std::string prefix,
    const io::json::Value & config, const ExecutionContext & execution, GraphWorkspace *workspace)
    : execution_(execution), workspace_(workspace), store_(execution.backend(), execution.backend_type(), "dettato.drums." + prefix + "weights", 1024 * 1024),
      stem_(config.require("stem").as_string()), activation_(config.require("act").as_string()),
      width_(config.require("w").as_i64()), band_hidden_(config.require("collapse_m").as_i64()) {
    for (const auto & metadata : source.tensors()) {
        if (metadata.name.rfind(prefix, 0) != 0) continue;
        const auto name = metadata.name.substr(prefix.size());
        if (name.rfind("stem.", 0) != 0 && name.rfind("collapse.", 0) != 0) continue;
        weights_.emplace(name, store_.load_tensor(source, metadata.name, assets::TensorStorageType::Native, metadata.shape));
        if (name.size() > 13 && name.compare(name.size() - 13, 13, ".running_mean") == 0) {
            const auto bn = name.substr(0, name.size() - 13);
            const auto affine = binding::batch_norm_eval_from_source(store_, source, prefix + bn, metadata.shape[0], 1e-5f);
            weights_.emplace(bn + ".scale", affine.scale);
            weights_.emplace(bn + ".offset", affine.bias);
        }
    }
    store_.upload();
}

void DrumEmbedder::prepare(const std::vector<int64_t> & shapes) {
    if (graph_ && graph_->graph() && shapes_ == shapes) return;
    graph_.reset(); shapes_ = shapes; ports_.clear(); ports_.resize(shapes.size());
    graph_ = std::make_unique<PreparedGraph>(execution_, PreparedGraphConfig{"dettato.drums.embed", 8 * 1024 * 1024, 8192, 1024 * 1024});
    for (size_t shape = 0; shape < shapes.size(); ++shape) {
        const auto frames = shapes[shape];
        auto & ports = ports_[shape];
        const auto id = shape == 0 ? 0 : graph_->add_graph("window_" + std::to_string(frames));
        auto & ctx = graph_->build_context();
        ports.input = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, 1, frames, 160}));
        ggml_set_input(ports.input.tensor);
        auto x = ScaleBiasModule({1.f / 3, 0}).build(ctx, ScaleBiasModule({1, 5}).build(ctx, ports.input));
        const auto activate = [&](const TensorValue & value) {
            if (activation_ == "silu") return SiluModule{}.build(ctx, value);
            if (activation_ == "relu") return ReluModule{}.build(ctx, value);
            if (activation_ == "gelu") return GeluModule{}.build(ctx, value);
            throw std::invalid_argument("Unknown Dettato drum stem activation: " + activation_);
        };
        std::istringstream tokens(stem_); std::string token;
        int index = 0;
        while (std::getline(tokens, token, ',')) {
            const auto prefix = "stem.net." + std::to_string(index);
            if (token[0] == 'c') {
                const auto weight = weights_.at(prefix + ".weight");
                Conv2dConfig conv;
                conv.in_channels = weight.shape.dims[1]; conv.out_channels = weight.shape.dims[0];
                conv.kernel_height = conv.kernel_width = 3; conv.padding_height = conv.padding_width = 1;
                x = Conv2dModule(conv).build(ctx, x, {weight, weights_.at(prefix + ".bias")});
                const auto bn = "stem.net." + std::to_string(index + 1);
                x = BatchNorm2dEvalModule({conv.out_channels}).build(ctx, x, {weights_.at(bn + ".scale"), weights_.at(bn + ".offset")});
                x = activate(x); index += 3;
            } else if (token == "r") {
                const auto residual = x;
                const auto channels = x.shape.dims[1];
                DepthwiseConv2dConfig dw;
                dw.channels = channels; dw.kernel_height = dw.kernel_width = 3; dw.padding_height = dw.padding_width = 1;
                x = DepthwiseConv2dModule(dw).build(ctx, x, {weights_.at(prefix + ".dw.weight"), weights_.at(prefix + ".dw.bias")});
                x = BatchNorm2dEvalModule({channels}).build(ctx, x, {weights_.at(prefix + ".bn1.scale"), weights_.at(prefix + ".bn1.offset")});
                x = activate(x);
                Conv2dConfig pointwise; pointwise.in_channels = pointwise.out_channels = channels; pointwise.kernel_height = pointwise.kernel_width = 1;
                x = Conv2dModule(pointwise).build(ctx, x, {weights_.at(prefix + ".pw1.weight"), weights_.at(prefix + ".pw1.bias")});
                x = BatchNorm2dEvalModule({channels}).build(ctx, x, {weights_.at(prefix + ".bn2.scale"), weights_.at(prefix + ".bn2.offset")});
                x = AddModule{}.build(ctx, residual, x); ++index;
            } else if (token[0] == 'p' && token.size() >= 3) {
                const int time = token[1] - '0', frequency = std::stoi(token.substr(2));
                auto shape = x.shape; shape.dims[2] /= time; shape.dims[3] /= frequency;
                x = wrap_tensor(ggml_pool_2d(ctx.ggml, x.tensor, GGML_OP_POOL_MAX, frequency, time, frequency, time, 0, 0), shape); ++index;
            } else throw std::invalid_argument("Unknown Dettato drum stem token: " + token);
        }
        const int64_t time = x.shape.dims[2], bands = x.shape.dims[3], channels = x.shape.dims[1];
        if (time != frames / 2) throw std::invalid_argument("Dettato drum stem must halve its time dimension");
        x = TransposeModule({{3, 0, 2, 1}, 4}).build(ctx, x);
        x = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({bands, time, channels}));
        ports.initial = make_tensor(graph_->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({time, band_hidden_}));
        ggml_set_input(ports.initial.tensor);
        std::array<GRUWeights, 2> parameters;
        for (int d = 0; d < 2; ++d) {
            const auto suffix = std::string(d ? "_l0_reverse" : "_l0");
            parameters[d] = {weights_.at("collapse.rnn.weight_ih" + suffix), weights_.at("collapse.rnn.weight_hh" + suffix),
                 weights_.at("collapse.rnn.bias_ih" + suffix), weights_.at("collapse.rnn.bias_hh" + suffix)};
        }
        const auto recurrent = BidirectionalGRU({channels, band_hidden_, ctx.backend_type == BackendType::Cuda ? GRURecurrenceLowering::DeviceCell : GRURecurrenceLowering::Unrolled})
            .build(ctx, x, ports.initial, ports.initial, parameters[0], parameters[1]);
        const auto hidden = ConcatModule({1}).build(ctx, recurrent.forward_hidden, recurrent.reverse_hidden);
        auto mean = ReduceMeanModule({0}).build(ctx, recurrent.sequence);
        mean = reshape_tensor(ctx, ensure_backend_addressable_layout(ctx, mean), TensorShape::from_dims({time, band_hidden_ * 2}));
        const auto collapsed = ConcatModule({1}).build(ctx, hidden, mean);
        ports.output = LinearModule({4 * band_hidden_, width_, true, GGML_PREC_F32}).build(ctx, collapsed,
            {weights_.at("collapse.proj.weight"), weights_.at("collapse.proj.bias")});
        ports.output = reshape_tensor(ctx, ports.output, TensorShape::from_dims({1, time, width_}));
        ggml_set_output(ports.output.tensor); ggml_build_forward_expand(graph_->graph(id), ports.output.tensor);
    }
    graph_->allocate_persistent();
    if (workspace_) workspace_->add(*graph_);
    else graph_->allocate();
    for (size_t i = 0; i < shapes.size(); ++i)
        write_tensor_f32(ports_[i].initial, std::vector<float>(static_cast<size_t>(shapes[i] / 2 * band_hidden_)));
}

void DrumEmbedder::prepare(int64_t frames, int64_t long_above) {
    constexpr int64_t chunk = 8000, margin = 200;
    const auto step = frames > long_above ? chunk : frames;
    std::vector<int64_t> shapes;
    for (int64_t start = 0; start < frames; start += step) {
        const auto begin = std::max<int64_t>(0, start - margin);
        const auto end = std::min(frames, start + step + margin);
        const auto size = end - begin;
        if (std::find(shapes.begin(), shapes.end(), size) == shapes.end()) shapes.push_back(size);
    }
    prepare(shapes);
}

std::vector<float> DrumEmbedder::run(const std::vector<float> & mel, int64_t long_above) {
    if (mel.size() % 160 || mel.size() < 320)
        throw std::invalid_argument("Dettato drum embedding expects at least two 160-band frames");
    const auto frames = static_cast<int64_t>(mel.size() / 160);
    // These are the upstream DrumNet/OrchestralAdapter windows, including the
    // different direct-path thresholds supplied by their callers.
    constexpr int64_t chunk = 8000, margin = 200;
    const auto step = frames > long_above ? chunk : frames;
    prepare(frames, long_above);
    std::vector<float> output;
    output.reserve(static_cast<size_t>(frames / 2 * width_));
    for (int64_t start = 0; start < frames; start += step) {
        const auto begin = std::max<int64_t>(0, start - margin);
        const auto end = std::min(frames, start + step + margin);
        const auto id = static_cast<size_t>(std::find(shapes_.begin(), shapes_.end(), end - begin) - shapes_.begin());
        window_.assign(mel.begin() + begin * 160, mel.begin() + end * 160);
        write_tensor_f32(ports_[id].input, window_);
        const auto phase = std::chrono::steady_clock::now(); graph_->compute(id);
        debug::timing_log_scalar("dettato.drums.embed_ms", debug::elapsed_ms(phase));
        const auto embeddings = read_tensor_f32(ports_[id].output.tensor);
        const auto offset = (start - begin) / 2 * width_;
        const auto count = (std::min(start + step, frames) - start) / 2 * width_;
        output.insert(output.end(), embeddings.begin() + offset, embeddings.begin() + offset + count);
    }
    return output;
}

Track decode_drum_hits(const DrumProbabilities & probabilities, const io::json::Value & config) {
    const auto & thresholds = config.require("thresholds");
    const auto & decode = config.require("decode");
    const bool orchestral = probabilities.router >= thresholds.require("router").as_number();
    const auto & onset = orchestral ? probabilities.orchestral : probabilities.kit;
    const auto & velocity = orchestral ? probabilities.velocity : probabilities.kit_velocity;
    if (onset.size() % 61 || onset.size() != velocity.size()) throw std::invalid_argument("Dettato drum onset/velocity shapes differ");
    const size_t frames = onset.size() / 61;
    const int gap = static_cast<int>(decode.require("peak_gap_frames").as_i64());
    const double repeat = decode.require("timpani_repeat_s").as_number();
    const double length = decode.require("hit_length_s").as_number();
    Track out; out.name = "Drums"; out.is_drum = true;
    const auto peaks = [&](const std::vector<float> & values, int lanes, float threshold, const auto & accept) {
        std::vector<int64_t> previous(lanes, -10);
        for (size_t t = 0; t < frames; ++t) for (int lane = 0; lane < lanes; ++lane) {
            const float p = values[t * lanes + lane];
            const float left = values[(t ? t - 1 : t) * lanes + lane];
            const float right = values[(t + 1 < frames ? t + 1 : t) * lanes + lane];
            if (p >= threshold && p >= left && p >= right && static_cast<int64_t>(t) - previous[lane] >= gap) {
                accept(t, lane); previous[lane] = static_cast<int64_t>(t);
            }
        }
    };
    const auto append = [&](size_t frame, int pitch, float value) {
        Note note;
        note.onset = std::nearbyint((frame / 50.) * 10000.) / 10000.;
        note.offset = std::nearbyint((frame / 50. + length) * 10000.) / 10000.;
        note.pitch = pitch;
        const double rounded = std::nearbyint(static_cast<double>(std::clamp(value, .05f, 1.f)) * 1000.) / 1000.;
        note.confidence = static_cast<float>(rounded);
        note.velocity = static_cast<int>(std::clamp(std::nearbyint(rounded * 127.), 1., 127.));
        out.notes.push_back(note);
    };
    const float threshold = thresholds.require(orchestral ? "orchestral_drum" : "drum_kit").as_f32();
    peaks(onset, 61, threshold, [&](size_t frame, int lane) { append(frame, lane + 27, velocity[frame * 61 + lane]); });
    if (orchestral) {
        if (probabilities.timpani.size() != frames * 13) throw std::invalid_argument("Dettato timpani shape differs from orchestral drums");
        const size_t drum_hits = out.notes.size();
        peaks(probabilities.timpani, 13, thresholds.require("timpani").as_f32(), [&](size_t frame, int lane) {
            const double onset = frame / 50.; const int pitch = lane + 41;
            for (size_t i = 0; i < drum_hits; ++i) if (out.notes[i].pitch == pitch && std::abs(out.notes[i].onset - onset) <= repeat) return;
            append(frame, pitch, std::clamp(probabilities.timpani[frame * 13 + lane], .3f, 1.f));
        });
        std::stable_sort(out.notes.begin(), out.notes.end(), [](const auto & a, const auto & b) { return a.onset < b.onset; });
    }
    return out;
}
} // namespace engine::models::dettato::detail
