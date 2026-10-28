#include "engine/framework/conditioners/musiccoca_runtime.h"

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/audio/waveform_ops.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/tokenizers/sentencepiece.h"

#include <ggml-alloc.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace engine::conditioners {
namespace json = io::json;
namespace {

using Shape = std::vector<int64_t>;

core::TensorShape tensor_shape(const Shape & shape, bool squeeze = true) {
    core::TensorShape result;
    for (auto d : shape) {
        if (squeeze && d == 1) continue;
        if (result.rank == core::kMaxTensorRank) throw std::runtime_error("MusicCoCa tensor exceeds four non-unit axes");
        result.dims[result.rank++] = d;
    }
    if (result.rank == 0) result = core::TensorShape::from_dims({1});
    return result;
}

class MusicCoCaGraph {
public:
    MusicCoCaGraph(core::ExecutionContext & execution, const engine::assets::TensorSource & source,
                  const json::Value & definition, std::string name)
        : execution_(execution), name_(std::move(name)),
          weights_(execution.backend(), execution.backend_type(), name_ + ".weights", 1024 * 1024) {
        ctx_ = ggml_init({8 * 1024 * 1024, nullptr, true});
        if (!ctx_) throw std::runtime_error("Cannot allocate MusicCoCa graph metadata");
        for (const auto & value : definition.require("values").as_array()) {
            shapes_.push_back(json::number_array_as<int64_t>(value.require("shape")));
        }
        build_context_ = {ctx_, "musiccoca", execution.backend_type()};
        values_.resize(shapes_.size());
        for (const auto & [key, value] : definition.require("constants").as_object()) {
            const auto id = static_cast<size_t>(std::stoull(key));
            if (const auto * name_value = value.find("tensor")) {
                const auto & tensor_name = name_value->as_string();
                auto found = constants_.find(tensor_name);
                if (found == constants_.end()) {
                    const auto metadata = source.require_metadata(tensor_name);
                    found = constants_.emplace(tensor_name,
                        weights_.load_f32_tensor(source, tensor_name, metadata.shape)).first;
                }
                values_.at(id) = modules::ReshapeModule({tensor_shape(shapes_.at(id))}).build(build_context_, found->second);
            } else {
                std::vector<float> data;
                for (const auto & item : value.require("values").as_array()) {
                    data.push_back(item.is_bool() ? (item.as_bool() ? 1.0F : 0.0F) : item.as_f32());
                }
                values_.at(id) = weights_.make_f32(tensor_shape(shapes_.at(id)), std::move(data));
                if (json::require_bool(value, "integer")) inline_constants_[id] = dimensions_numeric(value.require("values"));
            }
        }
        one_ = weights_.make_f32(core::TensorShape::from_dims({1}), {1.0F});
        weights_.upload();
        for (const auto & id : definition.require("inputs").as_array()) {
            const size_t index = static_cast<size_t>(id.as_i64());
            const bool integer = json::require_bool(definition.require("values").as_array().at(index), "integer");
            auto input = core::make_tensor(build_context_, integer ? GGML_TYPE_I32 : GGML_TYPE_F32,
                                          tensor_shape(shapes_.at(index)));
            ggml_set_input(input.tensor);
            values_.at(index) = input;
            inputs_.push_back(input.tensor);
        }
        for (const auto & node : definition.require("nodes").as_array()) build(node);
        for (const auto & id : definition.require("outputs").as_array()) {
            output_ = values_.at(static_cast<size_t>(id.as_i64())).tensor;
        }
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_, 16384, false);
        ggml_build_forward_expand(graph_, output_);
        allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (!ggml_gallocr_alloc_graph(allocator_, graph_)) throw std::runtime_error("Cannot allocate MusicCoCa graph");
        core::validate_backend_graph_supported(execution_.backend(), graph_, name_.c_str());
        host_plan_ = core::create_backend_graph_plan_if_host(execution_.backend(), graph_);
        debug::timing_log_context_reservation("musiccoca." + name_, ctx_);
        debug::timing_log_scalar("musiccoca." + name_ + ".graph.buffer_mb", ggml_gallocr_get_buffer_size(allocator_, 0) / (1024.0 * 1024.0));
    }

    ~MusicCoCaGraph() {
        core::free_backend_graph_plan(execution_.backend(), host_plan_);
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        ggml_gallocr_free(allocator_);
        ggml_free(ctx_);
    }

    ggml_tensor * input(size_t index) const { return inputs_.at(index); }

    std::vector<float> run() {
        const auto start = std::chrono::steady_clock::now();
        if (core::compute_backend_graph(execution_.backend(), graph_, host_plan_, name_.c_str()) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("MusicCoCa execution failed: " + name_);
        auto result = core::read_tensor_f32(output_);
        debug::timing_log_scalar("musiccoca." + name_ + ".graph.compute_ms", debug::elapsed_ms(start));
        return result;
    }

private:
    static Shape dimensions_numeric(const json::Value & array) {
        Shape result;
        for (const auto & v : array.as_array()) result.push_back(v.is_bool() ? static_cast<int64_t>(v.as_bool()) : v.as_i64());
        return result;
    }

    core::TensorValue broadcast(size_t id, const Shape & target) {
        Shape shape(target.size(), 1);
        const auto & source = shapes_.at(id);
        if (source.size() > target.size()) throw std::runtime_error("MusicCoCa broadcast rank mismatch");
        std::copy(source.begin(), source.end(), shape.end() - source.size());
        Shape retained;
        for (size_t i = 0; i < target.size(); ++i) if (target[i] != 1) retained.push_back(shape[i]);
        return modules::ReshapeModule({tensor_shape(retained, false)}).build(build_context_, values_.at(id));
    }

    void build(const json::Value & node) {
        const auto op = json::require_string(node, "op");
        const size_t destination = static_cast<size_t>(json::require_i64(node, "output"));
        const auto ids = json::number_array_as<int64_t>(node.require("inputs"));
        const auto & shape = shapes_.at(destination);
        auto a = values_.at(static_cast<size_t>(ids.at(0)));
        core::TensorValue result;
        auto & ctx = build_context_;
        if (op == "RESHAPE") {
            result = modules::ReshapeModule({tensor_shape(shape)}).build(ctx, a);
        } else if (op == "ADD" || op == "MUL" || op == "SUB" || op == "GREATER_EQUAL") {
            a = modules::RepeatModule({tensor_shape(shape)}).build(ctx, broadcast(static_cast<size_t>(ids[0]), shape));
            const auto b = broadcast(static_cast<size_t>(ids[1]), shape);
            if (op == "ADD") result = modules::AddModule(false, true).build(ctx, a, b);
            else if (op == "MUL") {
                result = a.shape.rank == b.shape.rank && a.shape.dims == b.shape.dims
                    ? modules::MulModule().build(ctx, a, b)
                    : core::wrap_tensor(ggml_mul(ctx_, a.tensor, b.tensor), a.shape);
            } else if (op == "SUB") result = core::wrap_tensor(ggml_sub(ctx_, a.tensor, b.tensor), a.shape);
            else {
                const auto right = modules::RepeatModule({a.shape}).build(ctx, b);
                result = core::wrap_tensor(ggml_step(ctx_, ggml_sub(ctx_, right.tensor, a.tensor)), a.shape);
                result = modules::ScaleBiasModule({-1.0F, 1.0F}).build(ctx, result);
            }
        } else if (op == "SELECT_V2") {
            const auto target = tensor_shape(shape);
            const auto mask = modules::RepeatModule({target}).build(ctx, broadcast(static_cast<size_t>(ids[0]), shape));
            const auto yes = modules::RepeatModule({target}).build(ctx, broadcast(static_cast<size_t>(ids[1]), shape));
            const auto no = broadcast(static_cast<size_t>(ids[2]), shape);
            const auto complement = modules::ScaleBiasModule({-1.0F, 1.0F}).build(ctx, mask);
            const auto selected_no = core::wrap_tensor(ggml_mul(ctx_, complement.tensor, no.tensor), target);
            result = modules::AddModule().build(ctx, modules::MulModule().build(ctx, mask, yes), selected_no);
        } else if (op == "FULLY_CONNECTED") {
            const auto & weight_shape = shapes_.at(static_cast<size_t>(ids[1]));
            const int64_t in = weight_shape.at(1), out = weight_shape.at(0);
            const auto input = modules::ReshapeModule({core::TensorShape::from_dims({a.shape.num_elements() / in, in})}).build(ctx, a);
            modules::LinearWeights weights{values_.at(static_cast<size_t>(ids[1])), std::nullopt};
            if (ids.size() > 2 && ids[2] >= 0) weights.bias = values_.at(static_cast<size_t>(ids[2]));
            result = modules::LinearModule({in, out, weights.bias.has_value(), GGML_PREC_F32}).build(ctx, input, weights);
        } else if (op == "BATCH_MATMUL") {
            a = modules::ReshapeModule({tensor_shape(shapes_.at(static_cast<size_t>(ids[0])), false)}).build(ctx, a);
            auto b = modules::ReshapeModule({tensor_shape(shapes_.at(static_cast<size_t>(ids[1])), false)}).build(ctx, values_.at(static_cast<size_t>(ids[1])));
            if (json::require_bool(node, "adj_x")) {
                std::array<int, 4> axes{0, 1, 2, 3};
                std::swap(axes[a.shape.rank - 1], axes[a.shape.rank - 2]);
                a = modules::TransposeModule({axes, a.shape.rank}).build(ctx, a);
                a = modules::ReshapeModule({a.shape}).build(ctx, core::ensure_backend_addressable_layout(ctx, a));
            }
            if (json::require_bool(node, "adj_y")) {
                std::array<int, 4> axes{0, 1, 2, 3};
                std::swap(axes[b.shape.rank - 1], axes[b.shape.rank - 2]);
                b = modules::TransposeModule({axes, b.shape.rank}).build(ctx, b);
            }
            result = modules::MatMulModule().build(ctx, a, b);
            ggml_mul_mat_set_prec(result.tensor, GGML_PREC_F32);
        } else if (op == "TRANSPOSE") {
            const auto permutation = json::number_array_as<int64_t>(node.require("perm"));
            const auto & source = shapes_.at(static_cast<size_t>(ids[0]));
            std::vector<int64_t> retained;
            for (size_t i = 0; i < source.size(); ++i) if (source[i] != 1) retained.push_back(static_cast<int64_t>(i));
            std::array<int, 4> axes{0, 1, 2, 3};
            size_t index = 0;
            for (auto axis : permutation) {
                const auto found = std::find(retained.begin(), retained.end(), axis);
                if (found != retained.end()) axes[index++] = static_cast<int>(found - retained.begin());
            }
            result = modules::TransposeModule({axes, a.shape.rank}).build(ctx, a);
        } else if (op == "SUM") {
            auto original = shapes_.at(static_cast<size_t>(ids[0]));
            result = a;
            for (auto axis : inline_constants_.at(static_cast<size_t>(ids[1]))) {
                if (axis < 0) axis += static_cast<int64_t>(original.size());
                if (original.at(static_cast<size_t>(axis)) == 1) continue;
                const int squeezed_axis = static_cast<int>(std::count_if(original.begin(), original.begin() + axis,
                    [](int64_t size) { return size != 1; }));
                result = modules::ReduceSumModule({squeezed_axis}).build(ctx, result);
                original[static_cast<size_t>(axis)] = 1;
                result = modules::ReshapeModule({tensor_shape(original)}).build(ctx, core::ensure_backend_addressable_layout(ctx, result));
            }
        } else if (op == "SLICE") {
            const auto & starts = inline_constants_.at(static_cast<size_t>(ids[1]));
            const auto & lengths = inline_constants_.at(static_cast<size_t>(ids[2]));
            const auto & source = shapes_.at(static_cast<size_t>(ids[0]));
            result = a;
            int axis = 0;
            for (size_t i = 0; i < source.size(); ++i) {
                if (source[i] == 1) continue;
                const int64_t length = lengths[i] < 0 ? source[i] - starts[i] : lengths[i];
                if (starts[i] != 0 || length != source[i]) result = modules::SliceModule({axis, starts[i], length}).build(ctx, result);
                ++axis;
            }
        } else if (op == "CONCATENATION") {
            auto axis = json::require_i64(node, "axis");
            if (axis < 0) axis += static_cast<int64_t>(shape.size());
            const int squeezed_axis = static_cast<int>(std::count_if(shape.begin(), shape.begin() + axis,
                [](int64_t size) { return size != 1; }));
            result = broadcast(static_cast<size_t>(ids[0]), shape);
            for (size_t i = 1; i < ids.size(); ++i)
                result = modules::ConcatModule({squeezed_axis}).build(ctx, result, broadcast(static_cast<size_t>(ids[i]), shape));
        } else if (op == "EMBEDDING_LOOKUP") {
            const auto & weight = values_.at(static_cast<size_t>(ids[1]));
            result = modules::EmbeddingModule({weight.shape.dims[0], weight.shape.dims[1]}).build(ctx, a, weight);
        } else if (op == "SOFTMAX") {
            const float beta = json::require_f32(node, "beta");
            result = beta == 1.0F ? modules::SoftmaxModule().build(ctx, a)
                : core::wrap_tensor(ggml_soft_max_ext(ctx_, a.tensor, nullptr, beta, 0.0F), a.shape);
        } else if (op == "GELU") {
            result = modules::GeluModule({json::require_bool(node, "approximate") ?
                modules::GeluApproximation::Tanh : modules::GeluApproximation::ExactErf}).build(ctx, a);
        } else if (op == "TANH") {
            result = modules::TanhModule().build(ctx, a);
        } else if (op == "RSQRT") {
            auto scalar_shape = a.shape;
            std::fill_n(scalar_shape.dims.begin(), scalar_shape.rank, 1);
            const auto one = modules::ReshapeModule({scalar_shape}).build(ctx, one_);
            const auto numerator = modules::RepeatModule({a.shape}).build(ctx, one);
            const auto denominator = modules::SqrtModule().build(ctx, a);
            result = core::wrap_tensor(ggml_div(ctx_, numerator.tensor, denominator.tensor), a.shape);
        } else {
            throw std::runtime_error("Unsupported MusicCoCa operation: " + op);
        }
        if (const auto * activation = node.find("activation"); activation && activation->as_i64() != 0)
            throw std::runtime_error("Unsupported MusicCoCa fused activation");
        if (!ggml_is_contiguous(result.tensor)) result = core::ensure_backend_addressable_layout(ctx, result);
        values_.at(destination) = modules::ReshapeModule({tensor_shape(shape)}).build(ctx, result);
    }

    core::ExecutionContext & execution_;
    std::string name_;
    core::BackendWeightStore weights_;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    ggml_backend_graph_plan_t host_plan_ = nullptr;
    core::ModuleBuildContext build_context_;
    std::vector<Shape> shapes_;
    std::vector<core::TensorValue> values_;
    std::vector<ggml_tensor *> inputs_;
    ggml_tensor * output_ = nullptr;
    core::TensorValue one_;
    std::unordered_map<std::string, core::TensorValue> constants_;
    std::unordered_map<size_t, Shape> inline_constants_;
};

}  // namespace

class MusicCoCaRuntime::Impl {
public:
    Impl(core::ExecutionContext & execution, std::shared_ptr<const assets::TensorSource> weights,
         json::Value graphs, const std::filesystem::path & tokenizer)
        : execution(execution), source(std::move(weights)), definitions(std::move(graphs)),
          vocabulary(tokenizers::load_sentencepiece_model(tokenizer)) {}

    MusicCoCaGraph & graph(const std::string & name) {
        auto & result = graphs[name];
        if (!result) result = std::make_unique<MusicCoCaGraph>(execution, *source, definitions.require(name), name);
        return *result;
    }

    core::ExecutionContext & execution;
    std::shared_ptr<const engine::assets::TensorSource> source;
    json::Value definitions;
    std::vector<tokenizers::SentencePiecePiece> vocabulary;
    std::unordered_map<std::string, std::unique_ptr<MusicCoCaGraph>> graphs;
    std::unique_ptr<core::BackendWeightStore> quantizer_weights;
    ggml_context * quantizer_ctx = nullptr;
    ggml_cgraph * quantizer_graph = nullptr;
    ggml_gallocr_t quantizer_allocator = nullptr;
    ggml_backend_graph_plan_t quantizer_host_plan = nullptr;
    ggml_tensor * quantizer_input = nullptr;
    ggml_tensor * quantizer_output = nullptr;
    std::vector<float> frontend_window;
    audio::SparseMelFilterbank frontend_filters;

    ~Impl() {
        core::free_backend_graph_plan(execution.backend(), quantizer_host_plan);
        core::release_backend_graph_resources(execution.backend(), quantizer_graph, true);
        if (quantizer_allocator) ggml_gallocr_free(quantizer_allocator);
        if (quantizer_ctx) ggml_free(quantizer_ctx);
    }
};

MusicCoCaRuntime::MusicCoCaRuntime(core::ExecutionContext & execution,
                                 std::shared_ptr<const assets::TensorSource> weights,
                                 json::Value graphs, const std::filesystem::path & tokenizer)
    : impl_(std::make_unique<Impl>(execution, std::move(weights), std::move(graphs), tokenizer)) {}
MusicCoCaRuntime::~MusicCoCaRuntime() = default;

std::vector<float> MusicCoCaRuntime::embed_text(const std::string & text) {
    auto & graph = impl_->graph("text_encoder");
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    auto tokens = tokenizers::tokenize_sentencepiece(impl_->vocabulary, lower);
    tokens.resize(std::min<size_t>(127, tokens.size()));
    tokens.insert(tokens.begin(), 1);
    std::vector<float> padding(128, 1.0F);
    std::fill_n(padding.begin(), tokens.size(), 0.0F);
    tokens.resize(128, 0);
    for (size_t i = 0; i < 2; ++i) {
        auto * input = graph.input(i);
        if (input->type == GGML_TYPE_I32) ggml_backend_tensor_set(input, tokens.data(), 0, tokens.size() * sizeof(int32_t));
        else ggml_backend_tensor_set(input, padding.data(), 0, padding.size() * sizeof(float));
    }
    return graph.run();
}

std::vector<float> MusicCoCaRuntime::map_text(const std::vector<float> & embedding, const std::vector<float> & noise) {
    if (embedding.size() != 768 || noise.size() != 768) throw std::runtime_error("MusicCoCa mapper expects 768 values");
    auto & graph = impl_->graph("mapper");
    ggml_backend_tensor_set(graph.input(0), embedding.data(), 0, embedding.size() * sizeof(float));
    ggml_backend_tensor_set(graph.input(1), noise.data(), 0, noise.size() * sizeof(float));
    auto result = graph.run();
    double norm = 0;
    for (float v : result) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);
    for (auto & v : result) v = static_cast<float>(v / norm);
    return result;
}

std::vector<float> MusicCoCaRuntime::map_text(const std::vector<float> & embedding, uint32_t seed) {
    // NumPy RandomState uses MT19937, 53-bit uniforms, and polar Box-Muller.
    std::mt19937 rng(seed);
    auto uniform = [&]() {
        const auto a = rng() >> 5, b = rng() >> 6;
        return (static_cast<double>(a) * 67108864.0 + b) / 9007199254740992.0;
    };
    std::vector<float> noise(768);
    for (size_t i = 0; i < noise.size(); i += 2) {
        double x, y, radius;
        do {
            x = 2 * uniform() - 1;
            y = 2 * uniform() - 1;
            radius = x * x + y * y;
        } while (radius >= 1 || radius == 0);
        const double scale = std::sqrt(-2 * std::log(radius) / radius);
        noise[i] = static_cast<float>(y * scale);
        noise[i + 1] = static_cast<float>(x * scale);
    }
    return map_text(embedding, noise);
}

std::vector<float> MusicCoCaRuntime::embed_features(const std::vector<float> & features) {
    if (features.size() != 992 * 128) throw std::runtime_error("MusicCoCa expects 992 by 128 log-mel features");
    auto & graph = impl_->graph("music_encoder");
    ggml_backend_tensor_set(graph.input(0), features.data(), 0, features.size() * sizeof(float));
    return graph.run();
}

std::vector<float> MusicCoCaRuntime::preprocess_audio(const std::vector<float> & mono) {
    if (mono.empty() || mono.size() > 160000) throw std::runtime_error("MusicCoCa frontend expects at most ten seconds at 16 kHz");
    auto & p = *impl_;
    if (p.frontend_window.empty()) {
        p.frontend_window = p.source->require_f32("frontend.window", {400});
        p.frontend_window.resize(2048, 0);
        const audio::AudioTensor filterbank{p.source->require_f32("frontend.filters", {128, 1025}), {128, 1025}};
        p.frontend_filters = audio::MelFilterbank().prepare_sparse(filterbank);
    }
    auto samples = mono;
    samples.resize(160000, 0);
    audio::apply_preemphasis_in_place(samples, 0.97F);
    samples.resize(160608, 0);
    const audio::STFTConfig config{2048, 160, 2048, false, audio::STFTPadMode::Constant};
    auto magnitude = audio::STFT().compute_magnitude(samples, p.frontend_window, 1, 160608, config,
                                                    static_cast<size_t>(p.execution.config().threads));
    const auto mel = audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values, 1, 1025, 992, 992, p.frontend_filters);
    std::vector<float> result(992 * 128);
    for (size_t t = 0; t < 992; ++t) {
        for (size_t m = 0; m < 128; ++m) result[t * 128 + m] = std::log(mel.values[m * 992 + t] + 0.001F);
    }
    return result;
}

std::vector<float> MusicCoCaRuntime::embed_audio(const std::vector<float> & mono) {
    if (mono.empty()) throw std::runtime_error("MusicCoCa audio style input is empty");
    std::vector<float> result(768, 0);
    size_t count = 0;
    for (size_t start = 0; start < mono.size(); start += 160000) {
        auto end = std::min(mono.size(), start + 160000);
        const auto embedding = embed_features(preprocess_audio({mono.begin() + static_cast<std::ptrdiff_t>(start),
                                                               mono.begin() + static_cast<std::ptrdiff_t>(end)}));
        for (size_t i = 0; i < result.size(); ++i) result[i] += embedding[i];
        ++count;
    }
    for (auto & value : result) value /= static_cast<float>(count);
    return result;
}

std::array<int32_t, 12> MusicCoCaRuntime::quantize(const std::vector<float> & embedding) {
    if (embedding.size() != 768) throw std::runtime_error("MusicCoCa quantizer expects 768 values");
    auto & p = *impl_;
    if (!p.quantizer_graph) {
        p.quantizer_weights = std::make_unique<core::BackendWeightStore>(p.execution.backend(), p.execution.backend_type(),
                                                                       "musiccoca.quantizer.weights", 256 * 1024);
        p.quantizer_ctx = ggml_init({1024 * 1024, nullptr, true});
        if (!p.quantizer_ctx) throw std::runtime_error("Cannot allocate MusicCoCa quantizer metadata");
        auto * ctx = p.quantizer_ctx;
        core::ModuleBuildContext build{ctx, "musiccoca.quantizer", p.execution.backend_type()};
        auto residual = core::make_tensor(build, GGML_TYPE_F32, core::TensorShape::from_dims({1, 768}));
        p.quantizer_input = residual.tensor;
        ggml_set_input(p.quantizer_input);
        core::TensorValue codes;
        for (int i = 0; i < 12; ++i) {
            const auto name = "quantizer." + std::to_string(i);
            const auto weight = p.quantizer_weights->load_f32_tensor(*p.source, name + ".weight", {1024, 768});
            const auto bias = p.quantizer_weights->load_f32_tensor(*p.source, name + ".bias", {1024});
            auto score = modules::LinearModule({768, 1024, false, GGML_PREC_F32}).build(build, residual, {weight, std::nullopt});
            score = modules::ScaleBiasModule({2.0F, 0.0F}).build(build, score);
            score = core::wrap_tensor(ggml_sub(ctx, score.tensor, bias.tensor), score.shape);
            const auto code = core::wrap_tensor(ggml_argmax(ctx, score.tensor), core::TensorShape::from_dims({1}), GGML_TYPE_I32);
            const auto float_code = core::wrap_tensor(ggml_cast(ctx, code.tensor, GGML_TYPE_F32), code.shape);
            codes = codes.valid() ? modules::ConcatModule({0}).build(build, codes, float_code) : float_code;
            if (i != 11) {
                const auto centroid = modules::CodebookLookupModule({1024, 768}).build(build, code, weight);
                residual = core::wrap_tensor(ggml_sub(ctx, residual.tensor, centroid.tensor), residual.shape);
            }
        }
        p.quantizer_output = ggml_cast(ctx, codes.tensor, GGML_TYPE_I32);
        ggml_set_output(p.quantizer_output);
        p.quantizer_graph = ggml_new_graph_custom(ctx, 2048, false);
        ggml_build_forward_expand(p.quantizer_graph, p.quantizer_output);
        p.quantizer_weights->upload();
        p.quantizer_allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(p.execution.backend()));
        if (!ggml_gallocr_alloc_graph(p.quantizer_allocator, p.quantizer_graph))
            throw std::runtime_error("Cannot allocate MusicCoCa quantizer graph");
        p.quantizer_host_plan = core::create_backend_graph_plan_if_host(p.execution.backend(), p.quantizer_graph);
        debug::timing_log_scalar("musiccoca.quantizer.graph.buffer_mb", ggml_gallocr_get_buffer_size(p.quantizer_allocator, 0) / (1024.0 * 1024.0));
    }
    ggml_backend_tensor_set(p.quantizer_input, embedding.data(), 0, embedding.size() * sizeof(float));
    if (core::compute_backend_graph(p.execution.backend(), p.quantizer_graph, p.quantizer_host_plan) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("MusicCoCa quantizer failed");
    std::array<int32_t, 12> result{};
    ggml_backend_tensor_get(p.quantizer_output, result.data(), 0, sizeof(result));
    return result;
}

}  // namespace engine::conditioners
