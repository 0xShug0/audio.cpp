#include "engine/models/moss/moss_tts_local/generator.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/sampling/torch_random.h"
#include "engine/models/moss/shared/sampling.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::moss_tts_local {
namespace {

namespace modules = engine::modules;
namespace binding = engine::modules::binding;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

// Projects a hidden-state vector through a [rows, hidden] row-major weight matrix. The
// binary gate head is tiny, so this product is cheap to run on the host.
std::vector<float> project(
    const std::vector<float> & weight,
    const std::vector<float> & hidden,
    int64_t rows,
    int64_t hidden_size) {
    std::vector<float> logits(static_cast<size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        const float * weight_row = weight.data() + static_cast<size_t>(row) * hidden_size;
        double accumulator = 0.0;
        for (int64_t index = 0; index < hidden_size; ++index) {
            accumulator += static_cast<double>(weight_row[index]) * static_cast<double>(hidden[static_cast<size_t>(index)]);
        }
        logits[static_cast<size_t>(row)] = static_cast<float>(accumulator);
    }
    return logits;
}

}  // namespace

struct MossGenerator::ProjectionRuntime {
    struct Graph {
        ggml_tensor * input = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_cgraph * graph = nullptr;
        int64_t codebook_size = 0;
    };

    ProjectionRuntime(
        const MossTTSLocalAssets & assets,
        core::ExecutionContext & execution_context,
        int64_t hidden_size,
        size_t graph_arena_bytes,
        size_t weight_context_bytes)
        : backend(execution_context.backend()),
          backend_type(execution_context.backend_type()),
          threads(execution_context.config().threads),
          store(std::make_shared<core::BackendWeightStore>(
              backend,
              backend_type,
              "moss_tts_local.generator.projection.weights",
              weight_context_bytes)) {
        if (assets.model_weights == nullptr) {
            throw std::runtime_error("MOSS-TTS-Local projection runtime requires model weights");
        }
        const auto & source = *assets.model_weights;
        weights.reserve(static_cast<size_t>(assets.config.num_codebooks));
        for (int64_t codebook = 0; codebook < assets.config.num_codebooks; ++codebook) {
            const int64_t codebook_size = assets.config.audio_codebook_sizes.empty()
                ? assets.config.audio_vocab_size
                : assets.config.audio_codebook_sizes[static_cast<size_t>(codebook)];
            weights.push_back(store->load_f32_tensor(
                source,
                "audio_embeddings." + std::to_string(codebook) + ".weight",
                {codebook_size, hidden_size}));
        }
        store->upload();

        ggml_init_params params{graph_arena_bytes, nullptr, true};
        ctx.reset(ggml_init(params));
        if (ctx == nullptr) {
            throw std::runtime_error("failed to initialize MOSS-TTS-Local projection graph context");
        }
        core::ModuleBuildContext build_ctx{ctx.get(), "moss_tts_local.generator.projection", backend_type};
        graphs.reserve(weights.size());
        for (size_t index = 0; index < weights.size(); ++index) {
            const int64_t codebook_size = weights[index].shape.at(0);
            auto input = core::make_tensor(
                build_ctx,
                GGML_TYPE_F32,
                core::TensorShape::from_dims({hidden_size}));
            auto logits = modules::LinearModule(
                              binding::linear_config(hidden_size, codebook_size, false))
                              .build(build_ctx, input, {weights[index], std::nullopt});
            Graph graph;
            graph.input = input.tensor;
            graph.logits = logits.tensor;
            graph.codebook_size = codebook_size;
            graph.graph = ggml_new_graph_custom(ctx.get(), 8192, false);
            ggml_set_output(graph.logits);
            ggml_build_forward_expand(graph.graph, graph.logits);
            graphs.push_back(graph);
        }
        graph_buffer = ggml_backend_alloc_ctx_tensors(ctx.get(), backend);
        if (graph_buffer == nullptr) {
            throw std::runtime_error("failed to allocate MOSS-TTS-Local projection graphs");
        }
    }

    ~ProjectionRuntime() {
        for (const auto & graph : graphs) {
            core::release_backend_graph_resources(backend, graph.graph);
        }
        if (graph_buffer != nullptr) {
            ggml_backend_buffer_free(graph_buffer);
        }
    }

    std::vector<float> project(int64_t codebook, const std::vector<float> & hidden) const {
        if (codebook < 0 || static_cast<size_t>(codebook) >= graphs.size()) {
            throw std::runtime_error("MOSS-TTS-Local projection codebook is out of range");
        }
        const auto & graph = graphs[static_cast<size_t>(codebook)];
        ggml_backend_tensor_set(graph.input, hidden.data(), 0, hidden.size() * sizeof(float));
        core::set_backend_threads(backend, threads);
        const ggml_status status =
            core::compute_backend_graph(backend, graph.graph, nullptr, "MOSS-TTS-Local projection");
        ggml_backend_synchronize(backend);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("MOSS-TTS-Local projection graph compute failed");
        }
        std::vector<float> logits(static_cast<size_t>(graph.codebook_size));
        ggml_backend_tensor_get(graph.logits, logits.data(), 0, logits.size() * sizeof(float));
        return logits;
    }

    ggml_backend_t backend = nullptr;
    core::BackendType backend_type = core::BackendType::Cpu;
    int threads = 1;
    std::shared_ptr<core::BackendWeightStore> store;
    std::vector<core::TensorValue> weights;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::vector<Graph> graphs;
    ggml_backend_buffer_t graph_buffer = nullptr;
};

MossGenerator::MossGenerator(
    std::shared_ptr<const MossTTSLocalAssets> assets,
    core::ExecutionContext & execution_context,
    size_t projection_graph_arena_bytes,
    size_t projection_weight_context_bytes,
    const MossBackboneRuntime & backbone,
    const MossDepthTransformer & depth)
    : assets_(std::move(assets)),
      backbone_(backbone),
      depth_(depth),
      sampling_policy_(
          execution_context.backend_type() == core::BackendType::Cuda
              ? engine::sampling::resolve_torch_cuda_sampling_policy(
                    execution_context.backend_type(),
                    execution_context.config().device,
                    "moss_tts_local.generator.cuda_sampling_policy",
                    "MOSS-TTS-Local",
                    engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda)
              : engine::sampling::TorchCudaSamplingPolicy{}) {
    if (assets_ == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local generator requires assets");
    }
    if (assets_->model_weights == nullptr) {
        throw std::runtime_error("MOSS-TTS-Local generator requires model weights");
    }
    const auto & config = assets_->config;
    hidden_size_ = config.backbone.hidden_size;
    num_codebooks_ = config.num_codebooks;
    if (config.local_text_head_mode != "binary") {
        throw std::runtime_error("MOSS-TTS-Local generator only supports the binary local text head");
    }
    const auto & source = *assets_->model_weights;
    moss::AudioCodebookSpec codebooks;
    codebooks.hidden_size = hidden_size_;
    codebooks.num_codebooks = num_codebooks_;
    codebooks.audio_vocab_size = config.audio_vocab_size;
    codebooks.audio_codebook_sizes = config.audio_codebook_sizes;
    codebooks.audio_pad_token_id = config.audio_pad_token_id;
    audio_codebooks_ = std::make_unique<moss::AudioCodebookEmbeddings>(source, std::move(codebooks));
    local_text_head_ = source.require_f32("local_text_lm_head.weight", {2, hidden_size_});
    projection_ = std::make_unique<ProjectionRuntime>(
        *assets_,
        execution_context,
        hidden_size_,
        projection_graph_arena_bytes,
        projection_weight_context_bytes);
}

MossGenerator::~MossGenerator() = default;

std::vector<std::vector<int32_t>> MossGenerator::generate(
    const std::vector<int32_t> & text_tokens,
    const std::vector<int32_t> & audio_codes,
    const MossGenerationOptions & options) const {
    const auto & config = assets_->config;
    const int64_t hidden = hidden_size_;
    const int64_t n_vq = num_codebooks_;
    const int32_t assistant_slot = static_cast<int32_t>(config.audio_assistant_slot_token_id);

    if (text_tokens.empty()) {
        throw std::runtime_error("MOSS-TTS-Local generation requires a non-empty prompt");
    }
    if (static_cast<int64_t>(audio_codes.size()) != static_cast<int64_t>(text_tokens.size()) * n_vq) {
        throw std::runtime_error("MOSS-TTS-Local generation audio codes must be [seq, n_vq]");
    }

    const bool collect_timing = engine::debug::timing_log_enabled();
    const auto add_timing = [&](double & target, auto && fn) {
        if (collect_timing) {
            target += engine::debug::measure_ms(fn);
        } else {
            fn();
        }
    };
    const auto set_timing = [&](double & target, auto && fn) {
        if (collect_timing) {
            target = engine::debug::measure_ms(fn);
        } else {
            fn();
        }
    };

    std::vector<std::vector<int32_t>> generated_frames;
    generated_frames.reserve(static_cast<size_t>(options.max_new_frames));
    std::vector<std::vector<int32_t>> code_history(static_cast<size_t>(n_vq));
    const bool use_repetition_penalty = options.audio_repetition_penalty != 1.0F;
    std::vector<std::vector<uint8_t>> code_seen(static_cast<size_t>(n_vq));
    if (use_repetition_penalty) {
        for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
            const int64_t codebook_size = audio_codebooks_->codebook_size(codebook);
            code_seen[static_cast<size_t>(codebook)].assign(static_cast<size_t>(codebook_size), 0);
        }
    }
    std::mt19937 rng(options.seed);
    uint64_t sample_call_index = 0;
    double bias_ms = 0.0;
    double prefill_ms = 0.0;
    double depth_ms = 0.0;
    double gate_ms = 0.0;
    double projection_ms = 0.0;
    double sampling_ms = 0.0;
    double backbone_step_ms = 0.0;
    int64_t depth_calls = 0;
    int64_t projection_calls = 0;
    int64_t backbone_step_calls = 0;
    backbone_.reset_timing();
    depth_.reset_timing();

    // Prefill the whole prompt in a single batched forward (fills the cache in one pass
    // instead of one graph launch per token), then generate one row at a time from the cache.
    // The last prompt position's hidden state seeds the first generated frame.
    const int64_t prefix_len = static_cast<int64_t>(text_tokens.size());
    backbone_.begin_generation(prefix_len + options.max_new_frames);
    std::vector<float> prefix_bias(static_cast<size_t>(prefix_len * hidden), 0.0F);
    for (int64_t position = 0; position < prefix_len; ++position) {
        std::vector<float> row;
        add_timing(bias_ms, [&]() {
            row = audio_codebooks_->bias_for(audio_codes.data() + position * n_vq);
        });
        std::copy(row.begin(), row.end(), prefix_bias.begin() + position * hidden);
    }
    std::vector<float> last_hidden;
    set_timing(prefill_ms, [&]() {
        last_hidden = backbone_.prefill(text_tokens, prefix_bias);
    });

    for (int64_t frame = 0; frame < options.max_new_frames; ++frame) {
        // Seed the depth transformer with the backbone hidden state (local position 0).
        std::vector<float> local_embeds = last_hidden;
        local_embeds.reserve(static_cast<size_t>(n_vq * hidden));
        std::vector<float> local_hidden;
        add_timing(depth_ms, [&]() {
            local_hidden = depth_.forward(local_embeds, 1);
        });
        ++depth_calls;

        // Binary gate: continue with the assistant slot or stop on the audio-end token.
        std::vector<float> gate_logits;
        int32_t gate_index = 0;
        add_timing(gate_ms, [&]() {
            gate_logits = project(local_text_head_, local_hidden, 2, hidden);
            gate_index = options.do_sample
                ? moss::sample_index(
                      gate_logits,
                      options.text_top_k,
                      options.text_top_p,
                      options.text_temperature,
                      rng,
                      "MOSS-TTS-Local sampler",
                      &sampling_policy_,
                      options.seed,
                      sample_call_index++)
                : moss::argmax_index(gate_logits, "MOSS-TTS-Local sampler");
        });
        if (gate_index != 0) {
            break;
        }

        std::vector<int32_t> frame_codes(static_cast<size_t>(n_vq));
        for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
            const int64_t codebook_size = audio_codebooks_->codebook_size(codebook);
            std::vector<float> logits;
            add_timing(projection_ms, [&]() {
                logits = projection_->project(codebook, local_hidden);
            });
            ++projection_calls;
            if (static_cast<int64_t>(logits.size()) != codebook_size) {
                throw std::runtime_error("MOSS-TTS-Local projection returned an unexpected logits size");
            }
            int32_t code = 0;
            add_timing(sampling_ms, [&]() {
                moss::apply_repetition_penalty(
                    logits,
                    code_history[static_cast<size_t>(codebook)],
                    options.audio_repetition_penalty,
                    "MOSS-TTS-Local sampler");
                code = options.do_sample
                    ? moss::sample_index(
                          logits,
                          options.audio_top_k,
                          options.audio_top_p,
                          options.audio_temperature,
                          rng,
                          "MOSS-TTS-Local sampler",
                          &sampling_policy_,
                          options.seed,
                          sample_call_index++)
                    : moss::argmax_index(logits, "MOSS-TTS-Local sampler");
            });
            frame_codes[static_cast<size_t>(codebook)] = code;
            if (use_repetition_penalty) {
                auto & seen = code_seen[static_cast<size_t>(codebook)];
                if (code >= 0 && static_cast<size_t>(code) < seen.size() && seen[static_cast<size_t>(code)] == 0) {
                    seen[static_cast<size_t>(code)] = 1;
                    code_history[static_cast<size_t>(codebook)].push_back(code);
                }
            }

            if (codebook + 1 < n_vq) {
                const float * embedding = audio_codebooks_->embedding(codebook, code);
                local_embeds.insert(local_embeds.end(), embedding, embedding + hidden);
                add_timing(depth_ms, [&]() {
                    local_hidden = depth_.forward(local_embeds, codebook + 2);
                });
                ++depth_calls;
            }
        }
        generated_frames.push_back(frame_codes);

        // Append the emitted frame as the next decoder row (assistant slot + codes) and
        // advance the backbone cache; the returned hidden seeds the next frame.
        std::vector<float> frame_bias;
        add_timing(bias_ms, [&]() {
            frame_bias = audio_codebooks_->bias_for(frame_codes.data());
        });
        add_timing(backbone_step_ms, [&]() {
            backbone_.step_into(assistant_slot, frame_bias, last_hidden);
        });
        ++backbone_step_calls;
    }

    engine::debug::timing_log_scalar("moss_tts_local.generator.bias_ms", bias_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.prefill_ms", prefill_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.depth_ms", depth_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.gate_ms", gate_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.projection_ms", projection_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.sampling_ms", sampling_ms);
    engine::debug::timing_log_scalar("moss_tts_local.generator.backbone_step_ms", backbone_step_ms);
    engine::debug::trace_log_scalar("moss_tts_local.generator.depth_calls", depth_calls);
    engine::debug::trace_log_scalar("moss_tts_local.generator.projection_calls", projection_calls);
    engine::debug::trace_log_scalar("moss_tts_local.generator.backbone_step_calls", backbone_step_calls);
    backbone_.log_timing();
    depth_.log_timing();

    return generated_frames;
}

}  // namespace engine::models::moss_tts_local
