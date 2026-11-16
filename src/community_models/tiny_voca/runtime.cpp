#include "engine/community_models/tiny_voca/runtime.h"

#include "engine/community_models/tiny_voca/source_filter.h"
#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/attention/scaled_dot_product_attention.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/positional_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/sampling/noise.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>

namespace engine::models::tiny_voca {
namespace {

namespace core = engine::core;
namespace modules = engine::modules;
using core::TensorShape;
using core::TensorValue;

constexpr float kNormEpsilon = 1.0e-5F;
// The attention scores of a block of query rows are rows x length values for
// each head. The rows go in blocks that hold at most this many values, so the
// memory the scores take stays the same however long the utterance is.
constexpr int64_t kScoreValues = 1 << 20;
constexpr int64_t kFewestScoreRows = 16;

// Query rows that go through attention together.
int64_t score_rows(int64_t length, int64_t heads) {
    return std::min(length, std::max(kFewestScoreRows, kScoreValues / (length * heads)));
}

// An upper bound of the nodes of a stack's graph: under a hundred for a block
// and under twenty-four for each further block of query rows.
size_t graph_nodes(int64_t layers, int64_t length, int64_t heads) {
    const int64_t rows = score_rows(length, heads);
    const int64_t row_blocks = (length + rows - 1) / rows;
    return static_cast<size_t>(64 + layers * (96 + 24 * row_blocks));
}

struct GgmlContextDeleter {
    void operator()(ggml_context * context) const noexcept {
        if (context != nullptr) {
            ggml_free(context);
        }
    }
};

// ---- weights ------------------------------------------------------------------

struct BackendWeights {
    std::shared_ptr<core::BackendWeightStore> store;
    std::unordered_map<std::string, TensorValue> tensors;

    const TensorValue & at(const std::string & name) const {
        const auto found = tensors.find(name);
        if (found == tensors.end()) {
            throw std::runtime_error("tiny-voca missing tensor: " + name);
        }
        return found->second;
    }
};

bool in_a_stack(const std::string & name) {
    return name.rfind("encoder.", 0) == 0 || name.rfind("prosody_stack.", 0) == 0 || name.rfind("acoustic_stack.", 0) == 0;
}

// The weights of the three transformer stacks, on the backend.
std::shared_ptr<const BackendWeights> load_stack_weights(const assets::TensorSource & source, ggml_backend_t backend, core::BackendType type) {
    auto out = std::make_shared<BackendWeights>();
    size_t count = 0;
    for (const auto & tensor : source.tensors()) {
        count += in_a_stack(tensor.name) ? 1 : 0;
    }
    // The store's context holds the descriptions of the tensors; their data goes to a backend buffer.
    out->store = std::make_shared<core::BackendWeightStore>(backend, type, "tiny_voca.weights", (count + 8) * ggml_tensor_overhead());
    for (const auto & tensor : source.tensors()) {
        if (!in_a_stack(tensor.name)) {
            continue;
        }
        if (assets::ggml_type_for_tensor_dtype(tensor.dtype) != GGML_TYPE_F32) {
            throw std::runtime_error("tiny-voca supports FP32 weights only: " + tensor.name);
        }
        out->tensors.emplace(
            tensor.name,
            out->store->load_tensor(source, tensor.name, assets::TensorStorageType::F32, tensor.shape));
    }
    out->store->upload();
    return out;
}

// ---- small layers on the host ---------------------------------------------------

struct Linear {
    int64_t inputs = 0;
    int64_t outputs = 0;
    std::vector<float> weight;          // outputs x inputs
    std::vector<float> bias;

    void apply(const float * in, float * out) const {
        for (int64_t o = 0; o < outputs; ++o) {
            const float * row = &weight[static_cast<size_t>(o * inputs)];
            float sum = 0.0F;
            for (int64_t i = 0; i < inputs; ++i) {
                sum += row[i] * in[i];
            }
            out[o] = sum + bias[static_cast<size_t>(o)];
        }
    }
};

Linear load_linear(const assets::TensorSource & source, const std::string & prefix, int64_t outputs, int64_t inputs) {
    Linear out;
    out.inputs = inputs;
    out.outputs = outputs;
    out.weight = source.require_f32(prefix + ".weight", {outputs, inputs});
    out.bias = source.require_f32(prefix + ".bias", {outputs});
    return out;
}

float gelu(float x) {
    return 0.5F * x * (1.0F + std::erf(x / 1.41421356237309504880F));
}

// The duration predictor: two convolutions over the tokens, each followed by
// GELU and a layer norm, then two values for each token.
struct Predictor {
    int64_t channels = 0;
    int64_t hidden = 0;
    std::vector<float> conv1_weight, conv1_bias, conv2_weight, conv2_bias;      // out x in x 3
    std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    Linear out;

    static void conv(const std::vector<float> & x, int64_t length, int64_t inputs, int64_t outputs, const std::vector<float> & weight,
                     const std::vector<float> & bias, std::vector<float> & y) {
        y.assign(static_cast<size_t>(length * outputs), 0.0F);
        for (int64_t t = 0; t < length; ++t) {
            float * row = &y[static_cast<size_t>(t * outputs)];
            for (int64_t o = 0; o < outputs; ++o) {
                float sum = bias[static_cast<size_t>(o)];
                for (int64_t k = 0; k < 3; ++k) {
                    const int64_t at = t + k - 1;
                    if (at < 0 || at >= length) {
                        continue;
                    }
                    const float * in = &x[static_cast<size_t>(at * inputs)];
                    const float * w = &weight[static_cast<size_t>((o * inputs) * 3 + k)];
                    for (int64_t i = 0; i < inputs; ++i) {
                        sum += w[i * 3] * in[i];
                    }
                }
                row[o] = sum;
            }
        }
    }

    static void gelu_norm(std::vector<float> & x, int64_t length, int64_t width, const std::vector<float> & weight, const std::vector<float> & bias) {
        for (int64_t t = 0; t < length; ++t) {
            float * row = &x[static_cast<size_t>(t * width)];
            double mean = 0.0;
            for (int64_t c = 0; c < width; ++c) {
                row[c] = gelu(row[c]);
                mean += row[c];
            }
            mean /= static_cast<double>(width);
            double variance = 0.0;
            for (int64_t c = 0; c < width; ++c) {
                variance += (row[c] - mean) * (row[c] - mean);
            }
            const double scale = 1.0 / std::sqrt(variance / static_cast<double>(width) + static_cast<double>(kNormEpsilon));
            for (int64_t c = 0; c < width; ++c) {
                row[c] = static_cast<float>((row[c] - mean) * scale) * weight[static_cast<size_t>(c)] + bias[static_cast<size_t>(c)];
            }
        }
    }

    // `x` holds `length` rows of `channels`; returns `length` rows of two values.
    std::vector<float> apply(const std::vector<float> & x, int64_t length) const {
        std::vector<float> a, b;
        conv(x, length, channels, hidden, conv1_weight, conv1_bias, a);
        gelu_norm(a, length, hidden, norm1_weight, norm1_bias);
        conv(a, length, hidden, hidden, conv2_weight, conv2_bias, b);
        gelu_norm(b, length, hidden, norm2_weight, norm2_bias);
        std::vector<float> result(static_cast<size_t>(length * 2));
        for (int64_t t = 0; t < length; ++t) {
            out.apply(&b[static_cast<size_t>(t * hidden)], &result[static_cast<size_t>(t * 2)]);
        }
        return result;
    }
};

// ---- a transformer stack as a graph --------------------------------------------

struct GraphResources {
    ~GraphResources() {
        core::free_backend_graph_plan(backend, plan);
        core::release_backend_graph_resources(backend, graph);
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
        if (io_buffer != nullptr) {
            ggml_backend_buffer_free(io_buffer);
        }
    }

    std::unique_ptr<ggml_context, GgmlContextDeleter> io_context;
    std::unique_ptr<ggml_context, GgmlContextDeleter> graph_context;
    ggml_backend_buffer_t io_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;
    ggml_cgraph * graph = nullptr;
};

struct StackGraph : GraphResources {
    int64_t length = 0;
    ggml_tensor * input = nullptr;          // [1, length, channels]
    ggml_tensor * positions = nullptr;      // [length]
    ggml_tensor * output = nullptr;         // [1, length, channels]
};

modules::LinearWeights linear_weights(const BackendWeights & weights, const std::string & prefix) {
    return {weights.at(prefix + ".weight"), weights.at(prefix + ".bias")};
}

modules::NormWeights norm_weights(const BackendWeights & weights, const std::string & prefix) {
    return {weights.at(prefix + ".weight"), weights.at(prefix + ".bias")};
}

// [1, length, channels] -> [1, length, heads, width]
TensorValue split_heads(core::ModuleBuildContext & ctx, const TensorValue & x, int64_t heads, int64_t width) {
    return core::reshape_tensor(
        ctx, core::ensure_backend_addressable_layout(ctx, x), TensorShape::from_dims({1, x.shape.dims[1], heads, width}));
}

// Attention of the query rows [first, first + count) over every key and value,
// as [1, count, channels]. More than `rows` rows are split in two and the
// halves joined, so no score tensor is larger than rows x length for each head.
TensorValue attend(
    core::ModuleBuildContext & ctx,
    const modules::ScaledDotProductAttentionModule & attention,
    const TensorValue & q,
    const TensorValue & k,
    const TensorValue & v,
    int64_t first,
    int64_t count,
    int64_t rows) {
    if (count > rows) {
        const int64_t left = ((count + rows - 1) / rows / 2) * rows;
        return modules::ConcatModule({1}).build(
            ctx, attend(ctx, attention, q, k, v, first, left, rows), attend(ctx, attention, q, k, v, first + left, count - left, rows));
    }
    const TensorValue part = count == q.shape.dims[2] ? q : modules::SliceModule({2, first, count}).build(ctx, q);
    const TensorValue out = attention.build(ctx, part, k, v);                               // [1, count, heads, width]
    return core::reshape_tensor(
        ctx, core::ensure_backend_addressable_layout(ctx, out), TensorShape::from_dims({1, count, q.shape.dims[1] * q.shape.dims[3]}));
}

// One block: self-attention with rotary positions, then a depthwise
// convolution and a feed-forward layer, each added to its input.
// `x` is [1, length, channels].
TensorValue block(
    core::ModuleBuildContext & ctx,
    const BackendWeights & weights,
    const TinyVocaConfig & config,
    const std::string & prefix,
    const TensorValue & x,
    const TensorValue & positions) {
    const int64_t channels = config.channels;
    const int64_t heads = config.heads;
    const int64_t width = channels / heads;
    const int64_t length = x.shape.dims[1];
    const modules::AddModule add;
    const modules::LayerNormModule norm({channels, kNormEpsilon, true, true});
    const modules::RoPEModule rotary({width, GGML_ROPE_TYPE_NORMAL, 10000.0F});
    const modules::TransposeModule heads_first({{0, 2, 1, 3}, 4});
    const modules::TransposeModule swap({{0, 2, 1}, 3});

    TensorValue h = norm.build(ctx, x, norm_weights(weights, prefix + ".norm1"));
    const TensorValue qkv =
        modules::LinearModule({channels, 3 * channels, true}).build(ctx, h, linear_weights(weights, prefix + ".attention.qkv"));
    auto part = [&](int64_t index) {
        return split_heads(ctx, modules::SliceModule({2, index * channels, channels}).build(ctx, qkv), heads, width);
    };
    const TensorValue q = heads_first.build(ctx, rotary.build(ctx, part(0), positions));
    const TensorValue k = heads_first.build(ctx, rotary.build(ctx, part(1), positions));
    const TensorValue v = heads_first.build(ctx, part(2));
    const TensorValue attention =
        attend(ctx, modules::ScaledDotProductAttentionModule({width}), q, k, v, 0, length, score_rows(length, heads));
    const TensorValue attended = add.build(
        ctx, x, modules::LinearModule({channels, channels, true}).build(ctx, attention, linear_weights(weights, prefix + ".attention.out")));

    h = norm.build(ctx, attended, norm_weights(weights, prefix + ".norm2"));
    // the convolution runs over [1, channels, length]
    h = modules::DepthwiseConv1dModule({channels, config.kernel, 1, static_cast<int>(config.kernel / 2), 1, true})
            .build(ctx, swap.build(ctx, h), {weights.at(prefix + ".depthwise.weight"), weights.at(prefix + ".depthwise.bias")});
    h = core::ensure_backend_addressable_layout(ctx, swap.build(ctx, h));
    h = modules::LinearModule({channels, config.feedforward, true}).build(ctx, h, linear_weights(weights, prefix + ".expand"));
    h = modules::GeluModule({modules::GeluApproximation::ExactErf}).build(ctx, h);
    h = modules::LinearModule({config.feedforward, channels, true}).build(ctx, h, linear_weights(weights, prefix + ".project"));
    return add.build(ctx, attended, h);
}

std::unique_ptr<StackGraph> build_stack_graph(
    const BackendWeights & weights,
    const TinyVocaConfig & config,
    ggml_backend_t backend,
    const std::string & prefix,
    int64_t layers,
    int64_t length) {
    auto out = std::make_unique<StackGraph>();
    out->backend = backend;
    out->length = length;
    // The contexts hold the descriptions of the tensors only; their data lives in backend buffers.
    const size_t nodes = graph_nodes(layers, length, config.heads);
    out->io_context.reset(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
    out->graph_context.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
    if (out->io_context == nullptr || out->graph_context == nullptr) {
        throw std::runtime_error("tiny-voca failed to create graph contexts");
    }
    core::ModuleBuildContext ctx{out->graph_context.get(), "tiny_voca.stack", core::backend_type(backend)};
    out->input = ggml_new_tensor_3d(out->io_context.get(), GGML_TYPE_F32, config.channels, length, 1);
    out->positions = ggml_new_tensor_1d(out->io_context.get(), GGML_TYPE_I32, length);
    ggml_set_input(out->input);
    ggml_set_input(out->positions);
    TensorValue x = core::wrap_tensor(out->input, TensorShape::from_dims({1, length, config.channels}));
    const TensorValue positions = core::wrap_tensor(out->positions, TensorShape::from_dims({length}), GGML_TYPE_I32);
    for (int64_t layer = 0; layer < layers; ++layer) {
        x = block(ctx, weights, config, prefix + ".blocks." + std::to_string(layer), x, positions);
    }
    x = modules::LayerNormModule({config.channels, kNormEpsilon, true, true}).build(ctx, x, norm_weights(weights, prefix + ".norm"));
    out->output = core::ensure_backend_addressable_layout(ctx, x).tensor;
    ggml_set_output(out->output);
    out->graph = ggml_new_graph_custom(ctx.ggml, nodes, false);
    ggml_build_forward_expand(out->graph, out->output);

    out->io_buffer = ggml_backend_alloc_ctx_tensors(out->io_context.get(), backend);
    if (out->io_buffer == nullptr) {
        throw std::runtime_error("tiny-voca failed to allocate graph inputs");
    }
    out->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (out->allocator == nullptr || !ggml_gallocr_reserve(out->allocator, out->graph) ||
        !ggml_gallocr_alloc_graph(out->allocator, out->graph)) {
        throw std::runtime_error("tiny-voca failed to allocate a backend graph");
    }
    core::validate_backend_graph_supported(backend, out->graph, "tiny-voca");
    out->plan = core::create_backend_graph_plan_if_host(backend, out->graph);
    std::vector<int32_t> steps(static_cast<size_t>(length));
    for (int64_t i = 0; i < length; ++i) {
        steps[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    ggml_backend_tensor_set(out->positions, steps.data(), 0, steps.size() * sizeof(int32_t));
    return out;
}

}  // namespace

struct TinyVocaRuntime::State {
    struct BackendOwner {
        ggml_backend_t value = nullptr;
        ~BackendOwner() {
            if (value != nullptr) {
                ggml_backend_free(value);
            }
        }
    };

    struct Stack {
        std::string prefix;
        int64_t layers = 0;
    };

    State(std::shared_ptr<const TinyVocaAssets> assets_in, core::BackendConfig backend_config)
        : assets(std::move(assets_in)) {
        if (assets == nullptr) {
            throw std::runtime_error("tiny-voca runtime requires assets");
        }
        const TinyVocaConfig & c = assets->config;
        const assets::TensorSource & source = *assets->weights;
        backend_config.threads = std::max(1, backend_config.threads);
        backend.value = core::init_backend(backend_config);
        core::set_backend_threads(backend.value, backend_config.threads);
        weights = load_stack_weights(source, backend.value, core::backend_type(backend.value));

        const auto voices = static_cast<int64_t>(c.voices.size());
        tokens = source.require_f32("tokens.weight", {c.vocab_size, c.channels});
        stress = source.require_f32("stress.weight", {3, c.channels});
        voice = source.require_f32("voices.weight", {voices, c.channels});
        frame_voice = source.require_f32("frame_voices.weight", {voices, c.channels});
        frame_position = load_linear(source, "frame_position", c.channels, 3);
        prosody_head = load_linear(source, "prosody_head", 3, c.channels);
        prosody_projection = load_linear(source, "prosody_projection", c.channels, 3);
        duration.channels = c.channels;
        duration.hidden = c.predictor_channels;
        duration.conv1_weight = source.require_f32("duration.conv1.weight", {c.predictor_channels, c.channels, 3});
        duration.conv1_bias = source.require_f32("duration.conv1.bias", {c.predictor_channels});
        duration.conv2_weight = source.require_f32("duration.conv2.weight", {c.predictor_channels, c.predictor_channels, 3});
        duration.conv2_bias = source.require_f32("duration.conv2.bias", {c.predictor_channels});
        duration.norm1_weight = source.require_f32("duration.norm1.weight", {c.predictor_channels});
        duration.norm1_bias = source.require_f32("duration.norm1.bias", {c.predictor_channels});
        duration.norm2_weight = source.require_f32("duration.norm2.weight", {c.predictor_channels});
        duration.norm2_bias = source.require_f32("duration.norm2.bias", {c.predictor_channels});
        duration.out = load_linear(source, "duration.out", 2, c.predictor_channels);
        const int64_t harmonic_outputs = c.cep_causal + c.cep_anti + c.hop / c.synth_hop;
        const int64_t noise_outputs = c.noise_cep + c.hop / c.noise_hop;
        vocoder = std::make_unique<TinyVocaSourceFilter>(
            c,
            source.require_f32("vocoder.harmonic.weight", {harmonic_outputs, c.channels}),
            source.require_f32("vocoder.harmonic.bias", {harmonic_outputs}),
            source.require_f32("vocoder.noise.weight", {noise_outputs, c.channels}),
            source.require_f32("vocoder.noise.bias", {noise_outputs}));
        source.release_storage();

        encoder.prefix = "encoder";
        encoder.layers = c.encoder_layers;
        prosody.prefix = "prosody_stack";
        prosody.layers = c.prosody_layers;
        acoustic.prefix = "acoustic_stack";
        acoustic.layers = c.acoustic_layers;

        // Which tokens are phonemes (they always sound) and which are vowels (always voiced).
        phone.assign(static_cast<size_t>(c.vocab_size), false);
        vowel.assign(static_cast<size_t>(c.vocab_size), false);
        static const char * const vowels[] = {
            "ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː", "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
            "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n\xCC\xA9",
        };
        // Symbols 0-4 are special tokens and 5-16 punctuation; every later one is a phoneme.
        for (size_t id = 17; id < c.symbols.size(); ++id) {
            phone[id] = true;
            for (const char * symbol : vowels) {
                if (c.symbols[id] == symbol) {
                    vowel[id] = true;
                }
            }
        }
    }

    // Runs one stack over `length` rows of `channels` values, in place.
    //
    // The graph is made for this length and freed, with its work buffer, before
    // this returns. Graphs are not kept from one utterance to the next: making
    // one takes under a millisecond (under one percent of running it), the same
    // number of frames hardly ever comes back, and the memory a kept graph
    // holds is what the next step needs.
    void run(const Stack & stack, std::vector<float> & values, int64_t length, const char * label) {
        const auto build_start = std::chrono::steady_clock::now();
        const std::unique_ptr<StackGraph> graph =
            build_stack_graph(*weights, assets->config, backend.value, stack.prefix, stack.layers, length);
        const std::string name = std::string("tiny_voca.") + label + "_graph";
        engine::debug::timing_log_scalar(name + ".build_ms", engine::debug::elapsed_ms(build_start));
        engine::debug::trace_log_scalar(
            name + ".work_mb", static_cast<double>(ggml_gallocr_get_buffer_size(graph->allocator, 0)) / (1024.0 * 1024.0));
        ggml_backend_tensor_set(graph->input, values.data(), 0, values.size() * sizeof(float));
        const auto status = core::compute_backend_graph(graph->backend, graph->graph, graph->plan, label);
        ggml_backend_synchronize(graph->backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("tiny-voca ") + label + " graph compute failed");
        }
        ggml_backend_tensor_get(graph->output, values.data(), 0, values.size() * sizeof(float));
    }

    std::shared_ptr<const TinyVocaAssets> assets;
    BackendOwner backend;
    std::shared_ptr<const BackendWeights> weights;
    std::vector<float> tokens, stress, voice, frame_voice;
    Linear frame_position, prosody_head, prosody_projection;
    Predictor duration;
    std::unique_ptr<TinyVocaSourceFilter> vocoder;
    Stack encoder, prosody, acoustic;
    std::vector<bool> phone, vowel;
};

TinyVocaRuntime::TinyVocaRuntime(std::shared_ptr<const TinyVocaAssets> assets, core::BackendConfig backend_config)
    : state_(std::make_unique<State>(std::move(assets), backend_config)) {}

TinyVocaRuntime::~TinyVocaRuntime() = default;

TinyVocaOutput TinyVocaRuntime::synthesize(
    const std::vector<int32_t> & ids,
    const std::vector<int32_t> & stress,
    const TinyVocaGenerationOptions & options,
    const std::vector<float> * noise) {
    State & s = *state_;
    const TinyVocaConfig & c = s.assets->config;
    const auto count = static_cast<int64_t>(ids.size());
    const int64_t channels = c.channels;
    if (count < 1 || count > c.max_tokens || stress.size() != ids.size()) {
        throw std::runtime_error("tiny-voca: an utterance needs between 1 and " + std::to_string(c.max_tokens) + " phoneme tokens");
    }
    if (options.voice < 0 || options.voice >= static_cast<int64_t>(c.voices.size())) {
        throw std::runtime_error("tiny-voca: unknown voice");
    }
    const auto total_start = std::chrono::steady_clock::now();

    // Phoneme, stress and voice embeddings, then the text encoder.
    std::vector<float> hidden(static_cast<size_t>(count * channels));
    const float * voice = &s.voice[static_cast<size_t>(options.voice * channels)];
    for (int64_t n = 0; n < count; ++n) {
        const int64_t id = ids[static_cast<size_t>(n)];
        const int64_t level = stress[static_cast<size_t>(n)];
        if (id < 0 || id >= c.vocab_size || level < 0 || level > 2) {
            throw std::runtime_error("tiny-voca: a phoneme id or stress value is out of range");
        }
        const float * token = &s.tokens[static_cast<size_t>(id * channels)];
        const float * mark = &s.stress[static_cast<size_t>(level * channels)];
        for (int64_t k = 0; k < channels; ++k) {
            hidden[static_cast<size_t>(n * channels + k)] = token[k] + mark[k] + voice[k];
        }
    }
    auto stage_start = std::chrono::steady_clock::now();
    s.run(s.encoder, hidden, count, "encoder");
    engine::debug::timing_log_scalar("tiny_voca.encoder_ms", engine::debug::elapsed_ms(stage_start));

    // Durations. Every phoneme sounds for one frame at least. A token that can
    // be a pause (an edge, a mark, a word boundary) lasts its predicted length
    // when the second value says there is a pause, and vanishes otherwise.
    const std::vector<float> predicted = s.duration.apply(hidden, count);
    TinyVocaOutput out;
    out.durations.assign(static_cast<size_t>(count), 0);
    int64_t frames = 0;
    const float longest = std::log1p(400.0F);
    for (int64_t n = 0; n < count; ++n) {
        const float value = std::min(std::max(predicted[static_cast<size_t>(2 * n)], 0.0F), longest);
        const auto length = std::max<int64_t>(static_cast<int64_t>(std::nearbyint(std::expm1(value) / options.speed)), 1);
        const bool present = s.phone[static_cast<size_t>(ids[static_cast<size_t>(n)])] || predicted[static_cast<size_t>(2 * n + 1)] > 0.0F;
        out.durations[static_cast<size_t>(n)] = present ? length : 0;
        frames += out.durations[static_cast<size_t>(n)];
    }
    if (frames < 1) {
        throw std::runtime_error("tiny-voca: the utterance has no frames");
    }
    if (frames > c.max_frames) {
        throw TinyVocaTooLong();
    }

    // Each token's state repeated for its frames, with the place of the frame inside the token.
    std::vector<float> expanded(static_cast<size_t>(frames * channels));
    std::vector<bool> vowel_frame(static_cast<size_t>(frames), false);
    const float * frame_voice = &s.frame_voice[static_cast<size_t>(options.voice * channels)];
    std::vector<float> place(static_cast<size_t>(channels));
    int64_t frame = 0;
    for (int64_t n = 0; n < count; ++n) {
        const int64_t owned = out.durations[static_cast<size_t>(n)];
        for (int64_t offset = 0; offset < owned; ++offset, ++frame) {
            const float relative = (static_cast<float>(offset) + 0.5F) / static_cast<float>(owned);
            const float features[3] = {relative, 1.0F - relative, std::log1p(static_cast<float>(owned)) / 4.0F};
            s.frame_position.apply(features, place.data());
            for (int64_t k = 0; k < channels; ++k) {
                expanded[static_cast<size_t>(frame * channels + k)] =
                    hidden[static_cast<size_t>(n * channels + k)] + place[static_cast<size_t>(k)] + frame_voice[k];
            }
            vowel_frame[static_cast<size_t>(frame)] = s.vowel[static_cast<size_t>(ids[static_cast<size_t>(n)])];
        }
    }

    // Pitch, voicing and energy of every frame.
    stage_start = std::chrono::steady_clock::now();
    s.run(s.prosody, expanded, frames, "prosody");
    engine::debug::timing_log_scalar("tiny_voca.prosody_ms", engine::debug::elapsed_ms(stage_start));
    std::vector<float> prosody(static_cast<size_t>(frames * 3));
    for (int64_t t = 0; t < frames; ++t) {
        s.prosody_head.apply(&expanded[static_cast<size_t>(t * channels)], &prosody[static_cast<size_t>(t * 3)]);
        // A vowel is voiced by definition, whatever the predictor says.
        if (vowel_frame[static_cast<size_t>(t)]) {
            prosody[static_cast<size_t>(t * 3 + 1)] = std::max(prosody[static_cast<size_t>(t * 3 + 1)], 2.0F);
        }
    }
    if (options.pitch_range != 1.0F) {
        // Pitch movement scaled around the mean of the voiced frames.
        float sum = 0.0F;
        float voiced = 0.0F;
        for (int64_t t = 0; t < frames; ++t) {
            if (prosody[static_cast<size_t>(t * 3 + 1)] > 0.0F) {
                sum += prosody[static_cast<size_t>(t * 3)];
                voiced += 1.0F;
            }
        }
        const float mean = sum / std::max(voiced, 1.0F);
        for (int64_t t = 0; t < frames; ++t) {
            prosody[static_cast<size_t>(t * 3)] = mean + (prosody[static_cast<size_t>(t * 3)] - mean) * options.pitch_range;
        }
    }
    std::vector<float> f0(static_cast<size_t>(frames));
    for (int64_t t = 0; t < frames; ++t) {
        const float pitch = prosody[static_cast<size_t>(t * 3)];
        const float condition[3] = {
            std::min(std::max(pitch, -2.0F), 1.5F),
            prosody[static_cast<size_t>(t * 3 + 1)] > 0.0F ? 1.0F : 0.0F,
            std::min(std::max(prosody[static_cast<size_t>(t * 3 + 2)], -4.5F), 3.0F),
        };
        s.prosody_projection.apply(condition, place.data());
        for (int64_t k = 0; k < channels; ++k) {
            expanded[static_cast<size_t>(t * channels + k)] += place[static_cast<size_t>(k)];
        }
        f0[static_cast<size_t>(t)] = std::min(std::max(c.pitch_reference_hz * std::exp(pitch + options.pitch_shift), c.f0_min), c.f0_max);
    }

    // The filter of every frame, then the waveform.
    stage_start = std::chrono::steady_clock::now();
    s.run(s.acoustic, expanded, frames, "acoustic");
    engine::debug::timing_log_scalar("tiny_voca.acoustic_ms", engine::debug::elapsed_ms(stage_start));
    stage_start = std::chrono::steady_clock::now();
    const auto samples = static_cast<size_t>(frames * c.hop);
    if (noise != nullptr) {
        out.samples = s.vocoder->synthesize(expanded, f0, *noise, frames);
    } else {
        out.samples = s.vocoder->synthesize(expanded, f0, sampling::generate_normal_noise(samples, options.seed), frames);
    }
    out.f0 = std::move(f0);
    engine::debug::timing_log_scalar("tiny_voca.waveform_ms", engine::debug::elapsed_ms(stage_start));
    engine::debug::trace_log_scalar("tiny_voca.token_count", count);
    engine::debug::trace_log_scalar("tiny_voca.frames", frames);
    engine::debug::timing_log_scalar("tiny_voca.synthesize_ms", engine::debug::elapsed_ms(total_start));
    return out;
}

}  // namespace engine::models::tiny_voca
