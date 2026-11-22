#include "drums.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include <algorithm>
#include <numeric>

namespace engine::models::dettato::detail {
using namespace core;
using namespace modules;

struct DrumTranscriber::Impl {
    const ExecutionContext & execution;
    io::json::Value config;
    DrumEmbedder kit_embedder;
    DrumTrunk kit_trunk;
    std::vector<std::unique_ptr<OrchestralDrumExpert>> experts;
    std::unique_ptr<PreparedGraph> graph;
    std::array<TensorValue, 2> kit_inputs;
    std::vector<std::array<TensorValue, 5>> expert_inputs;
    std::array<TensorValue, 6> outputs;
    TensorValue divisor;
    int64_t frames = 0;

    Impl(const assets::TensorSource & source, const io::json::Value & cfg, const ExecutionContext & ctx)
        : execution(ctx), config(cfg),
          kit_embedder(source, "drum_kit.", cfg.require("drum_kit"), ctx),
          kit_trunk(source, "drum_kit.", cfg.require("drum_kit"), ctx) {
        const auto count = cfg.require("orchestral").require("n_models").as_i64();
        if (count <= 0) throw std::invalid_argument("Dettato drum routing requires orchestral experts");
        for (int64_t i = 0; i < count; ++i) experts.push_back(std::make_unique<OrchestralDrumExpert>(source, static_cast<size_t>(i), cfg, ctx));
    }
    void prepare(int64_t count) {
        if (graph && frames == count) return;
        graph.reset(); frames = count;
        graph = std::make_unique<PreparedGraph>(execution, PreparedGraphConfig{"dettato.drums.probabilities", 1024 * 1024, 1024, 64 * 1024});
        auto & ctx = graph->build_context();
        for (auto & input : kit_inputs) {
            input = make_tensor(ctx, GGML_TYPE_F32, TensorShape::from_dims({1, count, 61})); ggml_set_input(input.tensor);
        }
        outputs[0] = SigmoidModule{}.build(ctx, kit_inputs[0]); outputs[1] = SigmoidModule{}.build(ctx, kit_inputs[1]);
        expert_inputs.resize(experts.size()); std::array<TensorValue, 3> sums;
        TensorValue routers;
        for (size_t i = 0; i < experts.size(); ++i) {
            auto & inputs = expert_inputs[i];
            for (size_t j = 0; j < inputs.size(); ++j) {
                inputs[j] = make_tensor(ctx, GGML_TYPE_F32, j == 4 ? TensorShape::from_dims({1, 1})
                    : TensorShape::from_dims({1, count, j == 2 ? 13 : 61}));
                ggml_set_input(inputs[j].tensor);
            }
            // Preserve upstream's base + (corrected - base) rounding boundary.
            const auto difference = wrap_tensor(ggml_sub(ctx.ggml, inputs[1].tensor, inputs[0].tensor), inputs[0].shape);
            const std::array<TensorValue, 3> p{
                SigmoidModule{}.build(ctx, AddModule{}.build(ctx, inputs[0], difference)),
                SigmoidModule{}.build(ctx, inputs[2]), SigmoidModule{}.build(ctx, inputs[3])};
            for (size_t k = 0; k < sums.size(); ++k) sums[k] = i == 0 ? p[k] : AddModule{}.build(ctx, sums[k], p[k]);
            const auto router = SigmoidModule{}.build(ctx, inputs[4]);
            routers = i == 0 ? router : ConcatModule({1}).build(ctx, routers, router);
        }
        divisor = make_tensor(graph->persistent_build_context(), GGML_TYPE_F32, TensorShape::from_dims({1})); ggml_set_input(divisor.tensor);
        for (size_t i = 0; i < sums.size(); ++i) outputs[i + 2] = wrap_tensor(ggml_div(ctx.ggml, sums[i].tensor, divisor.tensor), sums[i].shape);
        outputs[5] = routers;
        for (const auto & output : outputs) { ggml_set_output(output.tensor); ggml_build_forward_expand(graph->graph(), output.tensor); }
        graph->allocate_persistent(); graph->allocate(); write_tensor_f32(divisor, {static_cast<float>(experts.size())});
    }
};

DrumTranscriber::DrumTranscriber(const assets::TensorSource & source, const io::json::Value & config,
    const ExecutionContext & execution) : impl_(std::make_unique<Impl>(source, config, execution)) {}
DrumTranscriber::~DrumTranscriber() = default;

DrumProbabilities DrumTranscriber::probabilities(const std::vector<float> & features) {
    if (features.size() % 280 || features.size() < 560) throw std::invalid_argument("Dettato drum network requires at least two 280-band frames");
    const auto start = std::chrono::steady_clock::now();
    auto & state = *impl_; const int64_t frames = static_cast<int64_t>(features.size() / 280) / 2 * 2;
    state.prepare(frames / 2);
    std::vector<float> mel(static_cast<size_t>(frames * 160));
    for (int64_t t = 0; t < frames; ++t) std::copy_n(features.data() + t * 280, 160, mel.data() + t * 160);
    const auto kit = state.kit_trunk.run(state.kit_embedder.run(mel, 9000));
    ggml_backend_tensor_copy(kit[1].tensor, state.kit_inputs[0].tensor); ggml_backend_tensor_copy(kit[2].tensor, state.kit_inputs[1].tensor);
    for (size_t i = 0; i < state.experts.size(); ++i) {
        const auto output = state.experts[i]->run(features);
        for (size_t j = 0; j < output.size(); ++j) ggml_backend_tensor_copy(output[j].tensor, state.expert_inputs[i][j].tensor);
    }
    state.graph->compute();
    const auto routers = read_tensor_f32(state.outputs[5].tensor);
    DrumProbabilities result;
    // Upstream converts each F32 router to a Python float before averaging.
    result.router = std::accumulate(routers.begin(), routers.end(), 0.) / routers.size();
    result.kit = read_tensor_f32(state.outputs[0].tensor); result.kit_velocity = read_tensor_f32(state.outputs[1].tensor);
    result.orchestral = read_tensor_f32(state.outputs[2].tensor); result.timpani = read_tensor_f32(state.outputs[3].tensor);
    result.velocity = read_tensor_f32(state.outputs[4].tensor);
    debug::timing_log_scalar("dettato.drums.network_ms", debug::elapsed_ms(start));
    return result;
}

DrumResult DrumTranscriber::transcribe_features(const std::vector<float> & features) {
    const auto start = std::chrono::steady_clock::now();
    DrumResult result;
    if (!features.empty()) {
        const auto p = probabilities(features);
        result.router = p.router;
        result.model = p.router < impl_->config.require("thresholds").require("router").as_number() ? "drum-kit" : "orchestral";
        result.track = decode_drum_hits(p, impl_->config);
    }
    debug::timing_log_scalar("dettato.drums.wall_ms", debug::elapsed_ms(start));
    return result;
}
} // namespace engine::models::dettato::detail
