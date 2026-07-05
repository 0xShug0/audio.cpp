#include "engine/models/parakeet_tdt/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/decoders/tdt_types.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "engine/framework/modules/structural_modules.h"
#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include "ggml-backend.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>

namespace engine::models::parakeet_tdt {

namespace {

int32_t argmax_range(const std::vector<float> & values, size_t start, size_t end) {
    int32_t best = static_cast<int32_t>(start);
    float best_value = -std::numeric_limits<float>::infinity();
    for (size_t i = start; i < end; ++i) {
        if (values[i] > best_value) {
            best_value = values[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

struct GgmlDecoderPredictGraph {
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;

    core::TensorValue token_index;
    core::TensorValue h0_in;
    core::TensorValue c0_in;
    core::TensorValue h1_in;
    core::TensorValue c1_in;

    core::TensorValue h0_out;
    core::TensorValue c0_out;
    core::TensorValue h1_out;
    core::TensorValue c1_out;
    core::TensorValue decoder_output;

    ~GgmlDecoderPredictGraph() {
        if (plan != nullptr) {
            engine::core::free_backend_graph_plan(backend, plan);
            plan = nullptr;
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
    }
};

struct GgmlDecoderJointGraph {
    int64_t max_frames = 0;
    ggml_backend_t backend = nullptr;
    ggml_context * ggml = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_graph_plan_t plan = nullptr;

    core::TensorValue encoder_frames;
    core::TensorValue decoder_output;
    core::TensorValue logits;

    ~GgmlDecoderJointGraph() {
        if (plan != nullptr) {
            engine::core::free_backend_graph_plan(backend, plan);
            plan = nullptr;
        }
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ggml != nullptr) {
            ggml_free(ggml);
            ggml = nullptr;
        }
    }
};

class GgmlTdtBatchDecoder {
public:
    GgmlTdtBatchDecoder(const ParakeetAssets & assets, const ParakeetTDTWeights & weights)
        : hidden_(assets.model_config.decoder_hidden_size),
          vocab_(assets.model_config.vocab_size),
          logits_dim_(vocab_ + static_cast<int64_t>(assets.model_config.durations.size())),
          durations_(assets.model_config.durations),
          weights_(weights),
          zero_state_(static_cast<size_t>(hidden_), 0.0f) {}

    void reset_state() {
        if (predict_graph_ == nullptr || active_joint_graph_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder state reset before initialization");
        }
        core::write_tensor_f32(predict_graph_->h0_in, zero_state_);
        core::write_tensor_f32(predict_graph_->c0_in, zero_state_);
        core::write_tensor_f32(predict_graph_->h1_in, zero_state_);
        core::write_tensor_f32(predict_graph_->c1_in, zero_state_);
        core::write_tensor_f32(active_joint_graph_->decoder_output, zero_state_);
    }

    void load_streaming_state(const ParakeetStreamingDecoderState & state) {
        if (predict_graph_ == nullptr || active_joint_graph_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder state load before initialization");
        }
        if (!state.initialized) {
            throw std::runtime_error("GGML TDT decoder streaming state must be initialized");
        }
        if (static_cast<int64_t>(state.h0.size()) != hidden_ ||
            static_cast<int64_t>(state.c0.size()) != hidden_ ||
            static_cast<int64_t>(state.h1.size()) != hidden_ ||
            static_cast<int64_t>(state.c1.size()) != hidden_ ||
            static_cast<int64_t>(state.decoder_output.size()) != hidden_) {
            throw std::runtime_error("GGML TDT decoder streaming state shape mismatch");
        }
        core::write_tensor_f32(predict_graph_->h0_in, state.h0);
        core::write_tensor_f32(predict_graph_->c0_in, state.c0);
        core::write_tensor_f32(predict_graph_->h1_in, state.h1);
        core::write_tensor_f32(predict_graph_->c1_in, state.c1);
        core::write_tensor_f32(active_joint_graph_->decoder_output, state.decoder_output);
    }

    void save_streaming_state(ParakeetStreamingDecoderState & state) const {
        if (predict_graph_ == nullptr || active_joint_graph_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder state save before initialization");
        }
        core::read_tensor_f32_into(predict_graph_->h0_in.tensor, state.h0);
        core::read_tensor_f32_into(predict_graph_->c0_in.tensor, state.c0);
        core::read_tensor_f32_into(predict_graph_->h1_in.tensor, state.h1);
        core::read_tensor_f32_into(predict_graph_->c1_in.tensor, state.c1);
        core::read_tensor_f32_into(active_joint_graph_->decoder_output.tensor, state.decoder_output);
        state.initialized = true;
    }

    void ensure_initialized(core::ExecutionContext & execution_context, int64_t capacity_frames) {
        backend_ = execution_context.backend();
        backend_threads_ = std::max(1, execution_context.config().threads);
        if (predict_graph_ == nullptr) {
            predict_graph_ = build_predict_graph();
        }
        active_joint_graph_ = ensure_joint_graph(capacity_frames);
    }

    void predict_start(int32_t blank_id) {
        predict_token(blank_id);
    }

    void predict_token(int32_t token_id) {
        if (predict_graph_ == nullptr || active_joint_graph_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder was not initialized");
        }
        core::write_tensor_i32(predict_graph_->token_index, std::vector<int32_t>{token_id});
        compute_graph(*predict_graph_, 1);
        ggml_backend_tensor_copy(predict_graph_->h0_out.tensor, predict_graph_->h0_in.tensor);
        ggml_backend_tensor_copy(predict_graph_->c0_out.tensor, predict_graph_->c0_in.tensor);
        ggml_backend_tensor_copy(predict_graph_->h1_out.tensor, predict_graph_->h1_in.tensor);
        ggml_backend_tensor_copy(predict_graph_->c1_out.tensor, predict_graph_->c1_in.tensor);
        ggml_backend_tensor_copy(predict_graph_->decoder_output.tensor, active_joint_graph_->decoder_output.tensor);
    }

    void upload_encoder_frames(const std::vector<float> & encoder_projected) {
        if (active_joint_graph_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder joint graph is not initialized");
        }
        if (static_cast<int64_t>(encoder_projected.size()) != active_joint_graph_->max_frames * hidden_) {
            throw std::runtime_error("GGML TDT decoder encoder frame upload shape mismatch");
        }
        core::write_tensor_f32(active_joint_graph_->encoder_frames, encoder_projected);
    }

    const std::vector<float> & compute_joint_logits() {
        if (active_joint_graph_ == nullptr || backend_ == nullptr) {
            throw std::runtime_error("GGML TDT decoder joint graph is not initialized");
        }
        compute_graph(*active_joint_graph_, backend_threads_);
        core::read_tensor_f32_into(active_joint_graph_->logits.tensor, joint_logits_host_);
        return joint_logits_host_;
    }

    decoders::TdtJointStep joint_step_argmax(const std::vector<float> & logits, int64_t frame_idx) {
        const size_t row_base = static_cast<size_t>(frame_idx * logits_dim_);
        const int32_t label =
            argmax_range(logits, row_base, row_base + static_cast<size_t>(vocab_)) -
            static_cast<int32_t>(row_base);
        const int32_t duration_index = argmax_range(
            logits,
            row_base + static_cast<size_t>(vocab_),
            row_base + static_cast<size_t>(logits_dim_)) - static_cast<int32_t>(row_base + static_cast<size_t>(vocab_));
        return decoders::TdtJointStep{
            label,
            logits.at(row_base + static_cast<size_t>(label)),
            duration_index,
        };
    }

    decoders::TdtDecodeResult run_greedy_duration_loop(
        const std::vector<float> & encoder_projected,
        int64_t frames,
        int32_t blank_id,
        int64_t max_symbols_per_step) {
        return run_greedy_duration_loop_with_state(
            encoder_projected,
            frames,
            blank_id,
            max_symbols_per_step,
            true);
    }

    decoders::TdtDecodeResult run_greedy_duration_loop_with_state(
        const std::vector<float> & encoder_projected,
        int64_t frames,
        int32_t blank_id,
        int64_t max_symbols_per_step,
        bool reset_to_sos) {
        if (frames < 0 || hidden_ <= 0) {
            throw std::runtime_error("Invalid decoder frame count");
        }
        if (active_joint_graph_ == nullptr) {
            throw std::runtime_error("TDT decoder joint graph is not initialized");
        }
        if (frames > active_joint_graph_->max_frames) {
            throw std::runtime_error("TDT decoder valid frame count exceeds graph capacity");
        }
        if (static_cast<int64_t>(encoder_projected.size()) != active_joint_graph_->max_frames * hidden_) {
            throw std::runtime_error("TDT decoder encoder_projected shape mismatch");
        }
        if (max_symbols_per_step <= 0) {
            throw std::runtime_error("TDT decoder max_symbols_per_step must be positive");
        }
        if (durations_.empty()) {
            throw std::runtime_error("TDT decoder durations must not be empty");
        }
        if (blank_id < 0 || blank_id >= vocab_) {
            throw std::runtime_error("TDT decoder blank_id is out of range");
        }
        upload_encoder_frames(encoder_projected);
        if (reset_to_sos) {
            reset_state();
            predict_start(blank_id);
        }

        decoders::TdtDecodeResult result;
        result.token_ids.reserve(static_cast<size_t>(frames));
        result.token_timestamps.reserve(static_cast<size_t>(frames));
        result.token_durations.reserve(static_cast<size_t>(frames));

        int64_t time_idx = 0;
        int64_t last_label_time_idx = -1;
        int64_t labels_at_current_time_idx = 0;

        while (time_idx < frames) {
            const auto & logits = compute_joint_logits();

            bool emitted_label = false;
            int64_t local_idx = time_idx;
            while (local_idx < frames) {
                const size_t row_base = static_cast<size_t>(local_idx * logits_dim_);
                const int32_t label =
                    argmax_range(logits, row_base, row_base + static_cast<size_t>(vocab_)) -
                    static_cast<int32_t>(row_base);
                const int32_t duration_index = argmax_range(
                    logits,
                    row_base + static_cast<size_t>(vocab_),
                    row_base + static_cast<size_t>(logits_dim_)) -
                    static_cast<int32_t>(row_base + static_cast<size_t>(vocab_));
                int32_t duration = durations_.at(static_cast<size_t>(duration_index));

                if (label == blank_id) {
                    if (duration == 0) {
                        duration = 1;
                    }
                    time_idx += duration;
                    local_idx = time_idx;
                    continue;
                }

                result.token_ids.push_back(label);
                result.token_timestamps.push_back(static_cast<int32_t>(time_idx));
                result.token_durations.push_back(duration);
                predict_token(label);

                if (time_idx == last_label_time_idx) {
                    ++labels_at_current_time_idx;
                } else {
                    last_label_time_idx = time_idx;
                    labels_at_current_time_idx = 1;
                }

                time_idx += duration;
                if (labels_at_current_time_idx >= max_symbols_per_step && time_idx == last_label_time_idx) {
                    ++time_idx;
                }
                emitted_label = true;
                break;
            }

            if (!emitted_label && local_idx >= frames) {
                break;
            }
        }

        return result;
    }

    int64_t hidden_size() const noexcept {
        return hidden_;
    }

    int64_t logits_size() const noexcept {
        return logits_dim_;
    }

    const std::vector<float> & decoder_output() const {
        core::read_tensor_f32_into(predict_graph_->decoder_output.tensor, decoder_output_scratch_);
        return decoder_output_scratch_;
    }

    const std::vector<float> & logits() const noexcept {
        return joint_logits_host_;
    }

    size_t joint_graph_count() const noexcept {
        return joint_graphs_.size();
    }

    int64_t active_joint_frames() const noexcept {
        return active_joint_graph_ != nullptr ? active_joint_graph_->max_frames : 0;
    }

private:
    std::unique_ptr<GgmlDecoderPredictGraph> build_predict_graph() {
        auto graph = std::make_unique<GgmlDecoderPredictGraph>();
        graph->backend = backend_;
        ggml_init_params params = {};
        params.mem_size = 64 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph->ggml = ggml_init(params);
        if (graph->ggml == nullptr) {
            throw std::runtime_error("Failed to initialize GGML decoder predictor context");
        }

        core::ModuleBuildContext ctx = {};
        ctx.ggml = graph->ggml;
        ctx.module_instance_name = "parakeet_tdt_decoder_predict";

        graph->token_index = core::make_tensor(ctx, GGML_TYPE_I32, core::TensorShape::from_dims({1}));
        graph->h0_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));
        graph->c0_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));
        graph->h1_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));
        graph->c1_in = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));

        auto embedding = modules::EmbeddingModule({vocab_, hidden_}).build(ctx, graph->token_index, weights_.decoder.embedding_weight);
        auto lstm0 = modules::LSTMCellModule({hidden_, hidden_}).build(
            ctx,
            embedding,
            graph->h0_in,
            graph->c0_in,
            weights_.decoder.lstm0);
        auto lstm1 = modules::LSTMCellModule({hidden_, hidden_}).build(
            ctx,
            lstm0.hidden,
            graph->h1_in,
            graph->c1_in,
            weights_.decoder.lstm1);
        auto decoder_output = modules::LinearModule({hidden_, hidden_, true}).build(
            ctx,
            lstm1.hidden,
            weights_.decoder.decoder_projector);

        graph->h0_out = lstm0.hidden;
        graph->c0_out = lstm0.cell;
        graph->h1_out = lstm1.hidden;
        graph->c1_out = lstm1.cell;
        graph->decoder_output = decoder_output;

        graph->graph = ggml_new_graph_custom(graph->ggml, 4096, false);
        ggml_build_forward_expand(graph->graph, graph->decoder_output.tensor);
        graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, backend_);
        if (graph->buffer == nullptr) {
            throw std::runtime_error("Failed to allocate GGML decoder predictor buffer");
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            graph->plan = engine::core::create_backend_graph_plan_if_host(backend_, graph->graph);
            if (graph->plan == nullptr) {
                throw std::runtime_error("Failed to create GGML decoder predictor plan");
            }
        }

        return graph;
    }

    std::unique_ptr<GgmlDecoderJointGraph> build_joint_graph(int64_t frames) {
        auto graph = std::make_unique<GgmlDecoderJointGraph>();
        graph->max_frames = frames;
        graph->backend = backend_;
        ggml_init_params params = {};
        params.mem_size = 64 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        graph->ggml = ggml_init(params);
        if (graph->ggml == nullptr) {
            throw std::runtime_error("Failed to initialize GGML decoder joint context");
        }

        core::ModuleBuildContext ctx = {};
        ctx.ggml = graph->ggml;
        ctx.module_instance_name = "parakeet_tdt_decoder_joint";

        graph->encoder_frames = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({frames, hidden_}));
        graph->decoder_output = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, hidden_}));

        auto repeated_decoder = modules::RepeatModule({core::TensorShape::from_dims({frames, hidden_})}).build(ctx, graph->decoder_output);
        auto fused = modules::ReluModule().build(ctx, modules::AddModule().build(ctx, graph->encoder_frames, repeated_decoder));
        graph->logits = modules::LinearModule({hidden_, logits_dim_, true}).build(
            ctx,
            fused,
            weights_.joint.head);

        graph->graph = ggml_new_graph_custom(graph->ggml, 4096, false);
        ggml_build_forward_expand(graph->graph, graph->logits.tensor);
        graph->buffer = ggml_backend_alloc_ctx_tensors(graph->ggml, backend_);
        if (graph->buffer == nullptr) {
            throw std::runtime_error("Failed to allocate GGML decoder joint buffer");
        }
        if (engine::core::uses_host_graph_plan(backend_)) {
            graph->plan = engine::core::create_backend_graph_plan_if_host(backend_, graph->graph);
            if (graph->plan == nullptr) {
                throw std::runtime_error("Failed to create GGML decoder joint plan");
            }
        }

        return graph;
    }

    GgmlDecoderJointGraph * ensure_joint_graph(int64_t capacity_frames) {
        for (auto & graph : joint_graphs_) {
            if (graph->backend == backend_ && graph->max_frames == capacity_frames) {
                return graph.get();
            }
        }
        joint_graphs_.push_back(build_joint_graph(capacity_frames));
        return joint_graphs_.back().get();
    }

    template <typename GraphT>
    void compute_graph(GraphT & graph, int threads) {
        if (engine::core::uses_host_graph_plan(backend_)) {
            core::set_backend_threads(backend_, std::max(1, threads));
        }
        engine::core::compute_backend_graph(backend_, graph.graph, graph.plan);
    }

    ggml_backend_t backend_ = nullptr;
    int backend_threads_ = 1;
    int64_t hidden_ = 0;
    int64_t vocab_ = 0;
    int64_t logits_dim_ = 0;
    std::vector<int32_t> durations_;
    const ParakeetTDTWeights & weights_;
    std::vector<float> zero_state_;
    mutable std::vector<float> decoder_output_scratch_;
    std::vector<float> joint_logits_host_;
    std::unique_ptr<GgmlDecoderPredictGraph> predict_graph_;
    std::vector<std::unique_ptr<GgmlDecoderJointGraph>> joint_graphs_;
    GgmlDecoderJointGraph * active_joint_graph_ = nullptr;
};

}  // namespace

struct ParakeetSharedDecoder::Impl {
    Impl(const ParakeetAssets & assets_in, const ParakeetTDTWeights & weights_in)
        : assets(assets_in),
          ggml_core(assets_in, weights_in) {}

    ParakeetDecodeResult run(
        core::ExecutionContext & execution_context,
        const std::vector<float> & encoder_projected,
        int64_t frames,
        decoders::TdtDecoderAlgorithm algorithm) {
        if (algorithm != decoders::TdtDecoderAlgorithm::GreedyDurationLoop) {
            throw std::runtime_error(
                "ParakeetSharedDecoder only supports decoder_algorithm='greedy_duration_loop'");
        }
        if (frames < 0) {
            throw std::runtime_error("Decoder valid frame count must be non-negative");
        }
        if (static_cast<int64_t>(encoder_projected.size()) % ggml_core.hidden_size() != 0) {
            throw std::runtime_error("Decoder encoder_projected capacity is not divisible by hidden size");
        }
        const int64_t capacity_frames = static_cast<int64_t>(encoder_projected.size()) / ggml_core.hidden_size();
        if (frames > capacity_frames) {
            throw std::runtime_error("Decoder valid frame count exceeds encoder_projected capacity");
        }
        ggml_core.ensure_initialized(execution_context, capacity_frames);
        const decoders::TdtDecodeResult decoded = ggml_core.run_greedy_duration_loop(
            encoder_projected,
            frames,
            static_cast<int32_t>(assets.model_config.blank_token_id),
            assets.model_config.max_symbols_per_step);

        ParakeetDecodeResult result;
        result.token_ids = decoded.token_ids;
        result.token_timestamps = decoded.token_timestamps;
        result.token_durations = decoded.token_durations;
        result.text = assets.tokenizer->decode_ids(result.token_ids, assets.model_config.blank_token_id);

        return result;
    }

    ParakeetDecodeResult run_streaming_chunk(
        core::ExecutionContext & execution_context,
        const std::vector<float> & encoder_projected,
        int64_t frames,
        decoders::TdtDecoderAlgorithm algorithm,
        ParakeetStreamingDecoderState & state) {
        if (algorithm != decoders::TdtDecoderAlgorithm::GreedyDurationLoop) {
            throw std::runtime_error(
                "ParakeetSharedDecoder only supports decoder_algorithm='greedy_duration_loop'");
        }
        if (frames < 0) {
            throw std::runtime_error("Decoder valid frame count must be non-negative");
        }
        if (static_cast<int64_t>(encoder_projected.size()) % ggml_core.hidden_size() != 0) {
            throw std::runtime_error("Decoder encoder_projected capacity is not divisible by hidden size");
        }
        const int64_t capacity_frames = static_cast<int64_t>(encoder_projected.size()) / ggml_core.hidden_size();
        if (frames > capacity_frames) {
            throw std::runtime_error("Decoder valid frame count exceeds encoder_projected capacity");
        }
        ggml_core.ensure_initialized(execution_context, capacity_frames);
        constexpr bool kResumeStreamingState = true;
        if (kResumeStreamingState && state.initialized) {
            ggml_core.load_streaming_state(state);
        }
        const decoders::TdtDecodeResult decoded = ggml_core.run_greedy_duration_loop_with_state(
            encoder_projected,
            frames,
            static_cast<int32_t>(assets.model_config.blank_token_id),
            assets.model_config.max_symbols_per_step,
            !(kResumeStreamingState && state.initialized));
        ggml_core.save_streaming_state(state);

        ParakeetDecodeResult result;
        result.token_ids = decoded.token_ids;
        result.token_timestamps = decoded.token_timestamps;
        result.token_durations = decoded.token_durations;
        result.text = assets.tokenizer->decode_ids(result.token_ids, assets.model_config.blank_token_id);
        return result;
    }

    const ParakeetAssets & assets;
    GgmlTdtBatchDecoder ggml_core;
};

ParakeetSharedDecoder::ParakeetSharedDecoder(const ParakeetAssets & assets, const ParakeetTDTWeights & weights)
    : impl_(std::make_unique<Impl>(assets, weights)) {}

ParakeetSharedDecoder::~ParakeetSharedDecoder() = default;

ParakeetSharedDecoder::ParakeetSharedDecoder(ParakeetSharedDecoder &&) noexcept = default;

ParakeetSharedDecoder & ParakeetSharedDecoder::operator=(ParakeetSharedDecoder &&) noexcept = default;

size_t ParakeetSharedDecoder::joint_graph_count() const {
    return impl_->ggml_core.joint_graph_count();
}

int64_t ParakeetSharedDecoder::active_joint_frames() const {
    return impl_->ggml_core.active_joint_frames();
}

ParakeetDecodeResult ParakeetSharedDecoder::run(
    core::ExecutionContext & execution_context,
    const std::vector<float> & encoder_projected,
    int64_t frames,
    decoders::TdtDecoderAlgorithm algorithm) {
    return impl_->run(execution_context, encoder_projected, frames, algorithm);
}

ParakeetDecodeResult ParakeetSharedDecoder::run_streaming_chunk(
    core::ExecutionContext & execution_context,
    const std::vector<float> & encoder_projected,
    int64_t frames,
    decoders::TdtDecoderAlgorithm algorithm,
    ParakeetStreamingDecoderState & state) {
    return impl_->run_streaming_chunk(execution_context, encoder_projected, frames, algorithm, state);
}

void ParakeetSharedDecoder::prepare(core::ExecutionContext & execution_context, int64_t capacity_frames) {
    impl_->ggml_core.ensure_initialized(execution_context, capacity_frames);
}

}  // namespace engine::models::parakeet_tdt
