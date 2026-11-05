// LFM3 bindings and shared framework layers against the host fp64 reference
// (lfm3_audio_reference.h): the causal conv in both lowerings, and gated
// attention on one-layer stacks with generated weights written as a
// safetensors file with HF names.
//
//   lfm3_audio_blocks_test [--backend cpu|cuda|metal|best] [--threads N]
//
// A requested backend that is not available exits 77 (skipped).

#include "../lfm2_audio/lfm2_audio_test_package.h"
#include "lfm3_audio_reference.h"
#include "test_assert.h"


#include "engine/community_models/lfm3_audio/assets.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/modules/streaming_conv_modules.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/safetensors.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace engine::community_models::lfm3_audio;
namespace core = engine::core;
namespace ref = lfm3_audio_test;
using LayerKind = engine::modules::LfmDecoderLayerKind;
using core::TensorShape;
using core::TensorValue;
using engine::test::require;
using lfm2_audio_test::Random;
using ref::report;

constexpr double kConvTolerance = 1e-6;
// Native mode: flash attention, whose GPU kernels round K/V to F16.
constexpr double kNativeTolerance = 5e-3;

using Entries = std::vector<engine::io::SafeTensorWriteEntry>;

void add_tensor(Entries & entries, const std::string & name, std::vector<int64_t> shape, const std::vector<float> & values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    entries.push_back({name, "F32", std::move(shape), std::move(bytes)});
}

// A stack's tensors under `prefix` with HF names: norms near 1, projections scaled by 1/sqrt(fan-in).
void add_stack(Entries & entries, Random & random, const std::string & prefix, const engine::modules::LfmDecoderConfig & config) {
    const int64_t d = config.hidden_size;
    const int64_t hd = config.head_dim;
    const bool lfm2 = false;
    auto norm = [&](const std::string & name, int64_t size) {
        auto values = random.uniform(static_cast<size_t>(size), 0.2f);
        for (auto & value : values) {
            value += 1.0f;
        }
        add_tensor(entries, name, {size}, values);
    };
    auto matrix = [&](const std::string & name, int64_t rows, int64_t cols) {
        add_tensor(entries, name, {rows, cols}, random.uniform(static_cast<size_t>(rows * cols), 1.0f / std::sqrt(static_cast<float>(cols))));
    };

    for (size_t layer = 0; layer < config.layers.size(); ++layer) {
        const std::string p = prefix + "layers." + std::to_string(layer) + ".";
        norm(p + "operator_norm.weight", d);
        norm(p + "ffn_norm.weight", d);
        if (config.sandwich_norm) {
            norm(p + "post_operator_norm.weight", d);
            norm(p + "post_ffn_norm.weight", d);
        }
        matrix(p + "feed_forward.w1.weight", config.intermediate_size, d);
        matrix(p + "feed_forward.w3.weight", config.intermediate_size, d);
        matrix(p + "feed_forward.w2.weight", d, config.intermediate_size);
        if (config.layers[layer] == LayerKind::ShortConv) {
            matrix(p + "conv.in_proj.weight", 3 * d, d);
            matrix(p + "conv.out_proj.weight", d, d);
            add_tensor(entries, p + "conv.conv.weight", {d, 1, config.conv_kernel_size},
                       random.uniform(static_cast<size_t>(d * config.conv_kernel_size), 0.5f));
        } else {
            const std::string a = p + "self_attn.";
            matrix(a + "q_proj.weight", config.num_attention_heads * hd, d);
            matrix(a + "k_proj.weight", config.num_key_value_heads * hd, d);
            matrix(a + "v_proj.weight", config.num_key_value_heads * hd, d);
            matrix(a + (lfm2 ? "out_proj.weight" : "o_proj.weight"), d, config.num_attention_heads * hd);
            norm(a + (lfm2 ? "q_layernorm.weight" : "q_norm.weight"), hd);
            norm(a + (lfm2 ? "k_layernorm.weight" : "k_norm.weight"), hd);
            if (config.attention_gate) {
                matrix(a + "gate_proj.weight", config.num_attention_heads, d);
            }
        }
    }
    norm(prefix + "embedding_norm.weight", d);
}

std::shared_ptr<const engine::assets::TensorSource> write_source(const std::string & name, const Entries & entries) {
    const auto dir = std::filesystem::temp_directory_path() / "lfm3_audio_blocks_test";
    std::filesystem::create_directories(dir);
    const auto path = dir / (name + ".safetensors");
    engine::io::write_safetensors_file(path, entries);
    return engine::assets::open_tensor_source(path);
}

struct Model {
    std::shared_ptr<const engine::assets::TensorSource> source;
    std::unique_ptr<Lfm3Weights> stores;
    engine::modules::LfmDecoderWeights weights;
    ref::RefStack reference;
};

const std::string kPrefix = "stack.";

Model make_model(core::ExecutionContext & execution, const std::string & name, const engine::modules::LfmDecoderConfig & config, uint64_t seed,
                 const Lfm3WeightOptions & options) {
    Entries entries;
    Random random(seed);
    add_stack(entries, random, kPrefix, config);

    Model model;
    model.source = write_source(name, entries);
    model.stores = std::make_unique<Lfm3Weights>(execution, "lfm3_audio_blocks_test", 1024 * ggml_tensor_overhead(), options);
    model.weights = load_lfm3_stack_weights(*model.stores, *model.source, kPrefix, config);
    model.stores->upload();
    model.reference = ref::read_ref_stack(*model.source, kPrefix, config);
    return model;
}

// One graph whose tensors all live in one buffer (no allocator reuse: the tests are tiny).
class Graph {
public:
    explicit Graph(core::ExecutionContext & execution) : execution_(execution) {
        constexpr size_t kNodes = 16384;
        ggml_init_params params{ggml_tensor_overhead() * kNodes + ggml_graph_overhead_custom(kNodes, false), nullptr, true};
        ctx_ = ggml_init(params);
        require(ctx_ != nullptr, "failed to initialize the graph context");
        graph_ = ggml_new_graph_custom(ctx_, kNodes, false);
        build_.ggml = ctx_;
        build_.module_instance_name = "lfm3_audio_blocks_test";
        build_.backend_type = execution.backend_type();
    }

    ~Graph() {
        core::release_backend_graph_resources(execution_.backend(), graph_, true);
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        ggml_free(ctx_);
    }

    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    core::ModuleBuildContext & ctx() { return build_; }

    TensorValue input(ggml_type type, const TensorShape & shape) {
        auto value = core::make_tensor(build_, type, shape);
        ggml_set_input(value.tensor);
        return value;
    }

    void allocate(const TensorValue & output) {
        ggml_set_output(output.tensor);
        ggml_build_forward_expand(graph_, output.tensor);
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, execution_.backend());
        require(buffer_ != nullptr, "failed to allocate the graph tensors");
    }

    void compute() {
        const auto status = ggml_backend_graph_compute(execution_.backend(), graph_);
        ggml_backend_synchronize(execution_.backend());
        require(status == GGML_STATUS_SUCCESS, "graph compute failed");
    }

private:
    core::ExecutionContext & execution_;
    ggml_context * ctx_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    core::ModuleBuildContext build_{};
};

void write_mask(const TensorValue & tensor, const std::vector<int32_t> & positions, int64_t kv, int64_t window) {
    std::vector<ggml_fp16_t> mask(static_cast<size_t>(positions.size() * kv), ggml_fp32_to_fp16(-INFINITY));
    for (size_t q = 0; q < positions.size(); ++q) {
        for (int64_t k = 0; k < kv; ++k) {
            if (positions[q] >= k && (!window || positions[q] - k < window)) mask[q * kv + k] = ggml_fp32_to_fp16(0.f);
        }
    }
    require(ggml_nbytes(tensor.tensor) == mask.size() * sizeof(ggml_fp16_t), "mask size mismatch");
    ggml_backend_tensor_set(tensor.tensor, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
}

// Both lowerings: the backend's own (ssm_conv on CPU/Metal) and CUDA's (shifted views, row sum for one step),
// forced through ctx.backend_type.
void test_causal_conv(core::ExecutionContext & execution) {
    const int64_t d = 64;
    const bool cuda = core::uses_ggml_cuda_or_hip_backend(execution.backend_type());
    for (const bool shifted : {false, true}) {
        if (shifted && cuda) {
            continue;  // the backend's own lowering already was the shifted views
        }
        for (const int64_t k : {3, 7}) {
        for (const int64_t steps : {1, 5, 300}) {
            Random random(static_cast<uint64_t>(k * 1000 + steps));
            const auto padded = random.uniform(static_cast<size_t>((steps + k - 1) * d), 1.0f);
            const auto kernel = random.uniform(static_cast<size_t>(k * d), 0.5f);

            Graph g(execution);
            if (shifted) {
                g.ctx().backend_type = core::BackendType::Cuda;
            }
            auto padded_in = g.input(GGML_TYPE_F32, TensorShape::from_dims({1, steps + k - 1, d}));
            auto kernel_in = g.input(GGML_TYPE_F32, TensorShape::from_dims({k, d}));
            auto native_kernel = g.input(GGML_TYPE_F32, TensorShape::from_dims({d, 1, k}));
            auto out = engine::modules::DepthwiseConv1dModule({d, k, 1, 0, 1, false})
                .build_time_major(g.ctx(), padded_in, {native_kernel, std::nullopt}, kernel_in);
            g.allocate(out);
            core::write_tensor_f32(padded_in, padded);
            core::write_tensor_f32(kernel_in, kernel);
            std::vector<float> native_values(kernel.size());
            for (int64_t tap = 0; tap < k; ++tap) {
                for (int64_t c = 0; c < d; ++c) native_values[static_cast<size_t>(c * k + tap)] = kernel[static_cast<size_t>(tap * d + c)];
            }
            core::write_tensor_f32(native_kernel, native_values);
            g.compute();

            const auto expected = ref::causal_conv(ref::to_values(padded), ref::to_values(kernel), d, k, steps);
            const char * lowering = !(shifted || cuda) ? "ssm_conv" : steps == 1 ? "row sum" : "shifted views";
            report(std::string("causal_conv ") + lowering + " K=" + std::to_string(k) + " S=" + std::to_string(steps),
                   ref::rel_error(core::read_tensor_f32(out.tensor), expected), kConvTolerance);
        }
        }
    }
}

engine::modules::LfmDecoderConfig attention_config(int64_t head_dim, LayerKind kind, bool causal) {
    engine::modules::LfmDecoderConfig config;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = head_dim;
    config.hidden_size = config.num_attention_heads * head_dim;
    config.intermediate_size = 32;
    config.layers = {kind};
    config.attention_gate = true;
    config.sandwich_norm = false;
    config.full_rope.theta = 1e6f;
    config.sliding_rope.theta = 1e4f;
    config.full_rope.rotary_dims = causal ? head_dim / 2 : -1;  // backbone full layers rotate half; the depth decoder all
    return config;
}

void test_gated_attention(core::ExecutionContext & execution) {
    struct Case {
        const char * name;
        LayerKind kind;
        bool causal;
    };
    const Case cases[] = {
        {"full zero-tail", LayerKind::FullAttention, true},
        {"sliding w4", LayerKind::SlidingAttention, true},
        {"bidirectional", LayerKind::FullAttention, false},
    };
    const int64_t steps = 37;
    for (const int64_t hd : {32, 64, 96, 128}) {
        for (const auto & c : cases) {
            const auto config = attention_config(hd, c.kind, c.causal);
            for (const auto & options : {ref::reference_mode(), ref::native_mode()}) {
                const auto model = make_model(execution, "attention", config, static_cast<uint64_t>(hd * 10 + (&c - cases)), options);
                Random random(77);
                const auto normed = random.uniform(static_cast<size_t>(steps * config.hidden_size), 1.0f);
                const auto positions = ref::positions_from(0, steps);

                Graph g(execution);
                engine::modules::LfmDecoderInputs in;
                auto x = g.input(GGML_TYPE_F32, TensorShape::from_dims({1, steps, config.hidden_size}));
                in.hidden = x;
                in.positions = g.input(GGML_TYPE_I32, TensorShape::from_dims({steps}));
                if (c.causal) {
                    auto mask = g.input(GGML_TYPE_F16, TensorShape::from_dims({steps, steps}));
                    (c.kind == LayerKind::FullAttention ? in.full_mask : in.sliding_mask) = mask;
                }
                auto out = engine::modules::build_lfm_decoder(g.ctx(), in, model.weights, lfm3_decoder_config(config, options)).hidden;
                g.allocate(out);
                core::write_tensor_f32(x, normed);
                core::write_tensor_i32(in.positions, positions);
                if (in.full_mask) {
                    write_mask(*in.full_mask, positions, steps, 0);
                }
                if (in.sliding_mask) {
                    write_mask(*in.sliding_mask, positions, steps, 4);
                }
                g.compute();

                const auto expected = ref::stack(config, model.reference, ref::to_values(normed), positions, 4, c.causal);
                report(std::string("decoder attention ") + c.name + " hd=" + std::to_string(hd) + " " + ref::mode_name(options),
                       ref::rel_error(core::read_tensor_f32(out.tensor), expected),
                       lfm3_reference_precision(options) ? ref::reference_tolerance(execution) : kNativeTolerance);
            }
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    int status = 0;
    const auto execution = ref::open_backend(argc, argv, status);
    if (!execution) {
        return status;
    }

    try {
        const auto start = std::chrono::steady_clock::now();
        test_causal_conv(*execution);
        test_gated_attention(*execution);
        std::printf("PASS: lfm3_audio blocks (%.1f s)\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
