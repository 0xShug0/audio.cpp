#include "engine/framework/modules/vocoders/vocos_vocoder.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/audio/istft_graph.h"
#include "engine/framework/modules/convnext_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/runtime/cache_slots.h"
#include "ggml-alloc.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace engine::modules {
namespace {
constexpr int kMel = 100, kFFT = 1024, kHop = 256;
struct Graph {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_context * inputs = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t arena = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * mel = nullptr;
    ggml_tensor * spec = nullptr;
    std::unique_ptr<audio::HostLogMagnitudePhaseISTFT> istft;
    ~Graph() {
        if (backend && graph) core::release_backend_graph_resources(backend, graph, true);
        if (arena) ggml_gallocr_free(arena);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (inputs) ggml_free(inputs);
    }
};
}
core::TensorValue build_vocos_backbone(core::ModuleBuildContext & ctx,
    const core::TensorValue & mel, const VocosBackboneWeights & weights) {
    const auto channels = weights.embed.weight.shape.at(0);
    const auto kernel = weights.embed.weight.shape.last_dim();
    auto x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, mel);
    x = Conv1dModule({mel.shape.last_dim(), channels, kernel, 1, int(kernel / 2), 1, true}).build(ctx, x, weights.embed);
    x = TransposeModule({{0, 2, 1, 3}, 3}).build(ctx, x);
    x = LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.input_norm);
    for (const auto & block : weights.blocks) x = build_convnext1d(ctx, x, block);
    x = LayerNormModule({channels, 1e-6f, true, true}).build(ctx, x, weights.final_norm);
    return LinearModule({channels, weights.head.weight.shape.at(0), true}).build(ctx, x, weights.head);
}
class VocosVocoder::Impl {
public:
    ggml_backend_t backend;
    int threads;
    core::BackendType type;
    core::BackendWeightStore store;
    VocosBackboneWeights weights;
    std::vector<float> window;
    runtime::CacheSlots<int64_t, std::unique_ptr<Graph>> graphs{1};

    Impl(const std::string & checkpoint, ggml_backend_t b, int n)
        : backend(b), threads(n > 0 ? n : 1), type(core::backend_type(b)),
          store(b, type, "vocos", 8ULL << 20) {
        auto source = assets::open_tensor_source(checkpoint);
        if (std::filesystem::path(checkpoint).extension() == ".gguf") {
            source = assets::make_prefixed_tensor_source(source, "vocos");
        }
        const auto f32 = [&](const std::string & name) {
            return store.load_f32_tensor(*source, name, source->require_metadata(name).shape);
        };
        const auto norm = [&](const std::string & name) -> NormWeights {
            return {f32(name + ".weight"), f32(name + ".bias")};
        };
        const auto linear = [&](const std::string & name) -> LinearWeights {
            // Keep matrix weights quantized; convolution and norm weights stay F32.
            const auto weight = name + ".weight";
            return {store.load_tensor(*source, weight, assets::TensorStorageType::Native,
                                     source->require_metadata(weight).shape),
                    f32(name + ".bias")};
        };
        weights.embed = {f32("backbone.embed.weight"), f32("backbone.embed.bias")};
        weights.input_norm = norm("backbone.norm");
        weights.final_norm = norm("backbone.final_layer_norm");
        weights.head = linear("head.out");
        for (int i = 0; source->has_tensor("backbone.convnext." + std::to_string(i) + ".dwconv.weight"); ++i) {
            const auto p = "backbone.convnext." + std::to_string(i);
            weights.blocks.push_back({{f32(p + ".dwconv.weight"), f32(p + ".dwconv.bias")},
                norm(p + ".norm"), linear(p + ".pwconv1"), linear(p + ".pwconv2"), f32(p + ".gamma")});
        }
        if (weights.blocks.empty()) throw std::runtime_error("Vocos has no ConvNeXt blocks");
        store.upload();
        source->release_storage();
        window.resize(kFFT);
        for (int i = 0; i < kFFT; ++i) window[i] = 0.5f * (1 - std::cos(2.0f * 3.14159265358979323846f * i / kFFT));
    }

    std::unique_ptr<Graph> build(int64_t frames) {
        auto g = std::make_unique<Graph>();
        g->backend = backend;
        g->ctx = ggml_init({16ULL << 20, nullptr, true});
        g->inputs = ggml_init({1ULL << 20, nullptr, true});
        if (!g->ctx || !g->inputs) throw std::runtime_error("Vocos context allocation failed");
        core::ModuleBuildContext ctx{g->ctx, "vocos", type}, io{g->inputs, "vocos.input", type};
        auto mel = core::make_tensor(io, GGML_TYPE_F32, core::TensorShape::from_dims({1, frames, kMel}));
        g->mel = mel.tensor;
        ggml_set_input(g->mel);
        auto x = build_vocos_backbone(ctx, mel, weights);
        g->spec = x.tensor;
        ggml_set_output(g->spec);
        g->graph = ggml_new_graph_custom(g->ctx, 8192, false);
        ggml_build_forward_expand(g->graph, g->spec);
        if (type != core::BackendType::Cpu) {
            for (int i = 0; i < ggml_graph_n_nodes(g->graph); ++i) {
                auto * node = ggml_graph_node(g->graph, i);
                if (node->op == GGML_OP_MUL_MAT) ggml_mul_mat_set_prec(node, GGML_PREC_F32);
            }
        }
        g->buffer = ggml_backend_alloc_ctx_tensors(g->inputs, backend);
        g->arena = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!g->buffer || !g->arena || !ggml_gallocr_alloc_graph(g->arena, g->graph))
            throw std::runtime_error("Vocos graph allocation failed");
        g->istft = std::make_unique<audio::HostLogMagnitudePhaseISTFT>(
            audio::HostLogMagnitudePhaseISTFTConfig{frames, kFFT, kHop, kFFT + 2, size_t(threads)});
        return g;
    }
    std::vector<float> decode(const std::vector<float> & mel) {
        if (mel.size() < 2 * kMel || mel.size() % kMel) throw std::invalid_argument("Vocos needs complete 100-bin mel frames");
        const int64_t frames = mel.size() / kMel;
        if (!graphs.find(frames)) {
            // Release the previous arena before allocating the next shape.
            graphs.clear();
            graphs.put(frames, build(frames));
        }
        auto & g = **graphs.find(frames);
        ggml_backend_tensor_set(g.mel, mel.data(), 0, mel.size() * sizeof(float));
        core::set_backend_threads(backend, threads);
        if (core::compute_backend_graph(backend, g.graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Vocos compute failed");
        ggml_backend_synchronize(backend);
        auto spec = core::read_tensor_f32(g.spec);
        auto audio = g.istft->compute(spec, window).audio;
        // Framework ISTFT uses same padding (T*hop). Vocos uses center=True:
        // trim a further half hop at each end to obtain (T-1)*hop samples.
        return {audio.begin() + kHop / 2, audio.end() - kHop / 2};
    }
};
VocosVocoder::VocosVocoder(const std::string & path, ggml_backend_t backend, int threads)
    : impl_(std::make_unique<Impl>(path, backend, threads)) {}
VocosVocoder::~VocosVocoder() = default;
std::vector<float> VocosVocoder::decode(const std::vector<float> & mel) { return impl_->decode(mel); }
size_t VocosVocoder::cached_graph_count() const { return impl_->graphs.size(); }
}
