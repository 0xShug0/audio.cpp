#include "engine/models/nemotron_asr/decoder.h"

#include "engine/framework/core/backend.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/primitive_modules.h"
#include "engine/framework/modules/recurrent_modules.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::nemotron_asr {
namespace {

using Clock = std::chrono::steady_clock;

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

int32_t argmax_index(const std::vector<float> & values) {
    if (values.empty()) {
        throw std::runtime_error("Nemotron ASR decoder cannot select from empty logits");
    }
    return static_cast<int32_t>(
        std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::string replace_all(std::string text, const std::string & needle, const std::string & replacement) {
    if (needle.empty()) {
        return text;
    }
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
    return text;
}

std::string decode_timestamp_token(const NemotronAssets & assets, int32_t token_id, bool & emitted_text) {
    const auto & vocab = assets.tokenizer->id_to_token();
    if (token_id < 0 || token_id >= static_cast<int32_t>(vocab.size())) {
        return {};
    }
    std::string chunk = vocab[static_cast<size_t>(token_id)];
    if (!assets.metaspace_replacement.empty()) {
        chunk = replace_all(chunk, assets.metaspace_replacement, " ");
    }
    if (!emitted_text && assets.trim_leading_space) {
        while (!chunk.empty() && chunk.front() == ' ') {
            chunk.erase(chunk.begin());
        }
    }
    if (!chunk.empty()) {
        emitted_text = true;
    }
    return chunk;
}

std::vector<runtime::WordTimestamp> build_token_timestamps(
    const NemotronAssets & assets,
    const std::vector<int32_t> & token_ids,
    const std::vector<int32_t> & durations) {
    std::vector<runtime::WordTimestamp> out;
    out.reserve(token_ids.size());
    const int64_t samples_per_frame =
        assets.config.frontend.hop_length * assets.config.encoder.subsampling_factor;
    int64_t frame_index = 0;
    bool emitted_text = false;
    const size_t count = std::min(token_ids.size(), durations.size());
    for (size_t i = 0; i < count; ++i) {
        const int32_t token_id = token_ids[i];
        const int64_t token_frame = frame_index;
        frame_index += std::max<int32_t>(durations[i], 0);
        if (token_id == static_cast<int32_t>(assets.config.blank_token_id) ||
            token_id == static_cast<int32_t>(assets.config.pad_token_id)) {
            continue;
        }
        if (token_id >= 0 &&
            token_id < static_cast<int32_t>(assets.special_token_ids.size()) &&
            assets.special_token_ids[static_cast<size_t>(token_id)] != 0) {
            continue;
        }
        std::string chunk = decode_timestamp_token(assets, token_id, emitted_text);
        if (chunk.empty()) {
            continue;
        }
        runtime::WordTimestamp timestamp;
        timestamp.span.start_sample = token_frame * samples_per_frame;
        timestamp.span.end_sample = (token_frame + 1) * samples_per_frame;
        timestamp.word = std::move(chunk);
        timestamp.confidence = 0.0f;
        out.push_back(std::move(timestamp));
    }
    return out;
}

}  // namespace

struct NemotronDecoderRuntime::Graph {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue token_id;
    engine::core::TensorValue encoder_frame;
    std::vector<engine::core::TensorValue> hidden_in;
    std::vector<engine::core::TensorValue> cell_in;
    std::vector<engine::core::TensorValue> hidden_out;
    std::vector<engine::core::TensorValue> cell_out;
    engine::core::TensorValue decoder_cache_out;
    engine::core::TensorValue logits;

    ~Graph() {
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
    }
};

struct NemotronDecoderRuntime::JointGraph {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ggml;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    engine::core::HostGraphPlan host_plan;
    engine::core::TensorValue encoder_frame;
    engine::core::TensorValue decoder_cache_in;
    engine::core::TensorValue logits;

    ~JointGraph() {
        if (allocator != nullptr) {
            ggml_gallocr_free(allocator);
        }
    }
};

NemotronDecoderRuntime::NemotronDecoderRuntime(
    std::shared_ptr<const NemotronAssets> assets,
    std::shared_ptr<const NemotronWeights> weights,
    engine::core::ExecutionContext & execution_context,
    size_t graph_arena_bytes)
    : assets_(std::move(assets)),
      weights_(std::move(weights)),
      execution_context_(&execution_context),
      graph_arena_bytes_(graph_arena_bytes) {
    if (assets_ == nullptr || weights_ == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder requires assets and weights");
    }
}

NemotronDecoderRuntime::~NemotronDecoderRuntime() = default;

void NemotronDecoderRuntime::prepare() {
    ensure_graph();
    ensure_joint_graph();
}

void NemotronDecoderRuntime::ensure_graph() {
    if (graph_ != nullptr) {
        debug::timing_log_scalar("nemotron_asr.decoder.graph_rebuild_ms", 0.0);
        debug::trace_log_scalar("nemotron_asr.decoder.graph_cache_hit", true);
        return;
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config;
    auto graph = std::make_unique<Graph>();
    struct ggml_init_params params {
        graph_arena_bytes_, nullptr, true
    };
    graph->ggml.reset(ggml_init(params));
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder failed to create ggml context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml.get(), "nemotron_asr.decoder", execution_context_->backend_type()};
    graph->token_id = engine::core::make_tensor(ctx, GGML_TYPE_I32, engine::core::TensorShape::from_dims({1}));
    graph->encoder_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size}));
    graph->hidden_in.reserve(static_cast<size_t>(config.decoder_layers));
    graph->cell_in.reserve(static_cast<size_t>(config.decoder_layers));
    graph->hidden_out.reserve(static_cast<size_t>(config.decoder_layers));
    graph->cell_out.reserve(static_cast<size_t>(config.decoder_layers));
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        graph->hidden_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size})));
        graph->cell_in.push_back(engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size})));
    }

    ggml_set_input(graph->token_id.tensor);
    ggml_set_input(graph->encoder_frame.tensor);
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        ggml_set_input(graph->hidden_in[static_cast<size_t>(layer)].tensor);
        ggml_set_input(graph->cell_in[static_cast<size_t>(layer)].tensor);
    }

    auto hidden = engine::modules::EmbeddingModule({config.vocab_size, config.decoder_hidden_size})
                      .build(ctx, graph->token_id, weights_->decoder.embedding);
    hidden = engine::core::reshape_tensor(ctx, hidden, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size}));
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        const auto outputs = engine::modules::LSTMCellModule({config.decoder_hidden_size, config.decoder_hidden_size})
                                 .build(
                                     ctx,
                                     hidden,
                                     graph->hidden_in[static_cast<size_t>(layer)],
                                     graph->cell_in[static_cast<size_t>(layer)],
                                     weights_->decoder.lstm_layers[static_cast<size_t>(layer)]);
        graph->hidden_out.push_back(outputs.hidden);
        graph->cell_out.push_back(outputs.cell);
        hidden = graph->hidden_out.back();
    }
    auto projected = engine::modules::LinearModule({config.decoder_hidden_size, config.decoder_hidden_size, true})
                         .build(ctx, hidden, weights_->decoder.decoder_projector);
    graph->decoder_cache_out = projected;
    auto joint = engine::modules::AddModule().build(ctx, graph->encoder_frame, graph->decoder_cache_out);
    joint = engine::modules::ReluModule().build(ctx, joint);
    graph->logits = engine::modules::LinearModule({config.decoder_hidden_size, config.vocab_size, true})
                        .build(ctx, joint, weights_->decoder.joint_head);

    ggml_set_output(graph->logits.tensor);
    ggml_set_output(graph->decoder_cache_out.tensor);
    for (size_t layer = 0; layer < graph->hidden_out.size(); ++layer) {
        ggml_set_output(graph->hidden_out[layer].tensor);
        ggml_set_output(graph->cell_out[layer].tensor);
    }

    graph->graph = ggml_new_graph(graph->ggml.get());
    ggml_build_forward_expand(graph->graph, graph->logits.tensor);
    ggml_build_forward_expand(graph->graph, graph->decoder_cache_out.tensor);
    for (size_t layer = 0; layer < graph->hidden_out.size(); ++layer) {
        ggml_build_forward_expand(graph->graph, graph->hidden_out[layer].tensor);
        ggml_build_forward_expand(graph->graph, graph->cell_out[layer].tensor);
    }
    graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (graph->allocator == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder failed to create graph allocator");
    }
    if (!ggml_gallocr_alloc_graph(graph->allocator, graph->graph)) {
        throw std::runtime_error("Nemotron ASR decoder graph allocation failed");
    }
    engine::core::validate_backend_graph_supported(execution_context_->backend(), graph->graph, "Nemotron ASR decoder");
    engine::core::prepare_host_graph_plan(*execution_context_, graph->graph, graph->host_plan);

    logits_scratch_.assign(static_cast<size_t>(config.vocab_size), 0.0f);
    hidden_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    cell_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    decoder_cache_scratch_.assign(static_cast<size_t>(config.decoder_hidden_size), 0.0f);
    hidden_read_scratch_.assign(static_cast<size_t>(config.decoder_hidden_size), 0.0f);
    cell_read_scratch_.assign(static_cast<size_t>(config.decoder_hidden_size), 0.0f);

    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("nemotron_asr.decoder.graph_build_ms", build_ms);
    debug::timing_log_scalar("nemotron_asr.decoder.graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("nemotron_asr.decoder.graph_cache_hit", false);
    graph_ = std::move(graph);
}

void NemotronDecoderRuntime::ensure_joint_graph() {
    if (joint_graph_ != nullptr) {
        debug::timing_log_scalar("nemotron_asr.decoder.joint_graph_rebuild_ms", 0.0);
        debug::trace_log_scalar("nemotron_asr.decoder.joint_graph_cache_hit", true);
        return;
    }

    const auto build_start = Clock::now();
    const auto & config = assets_->config;
    auto graph = std::make_unique<JointGraph>();
    struct ggml_init_params params {
        graph_arena_bytes_, nullptr, true
    };
    graph->ggml.reset(ggml_init(params));
    if (graph->ggml == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder failed to create joint ggml context");
    }

    engine::core::ModuleBuildContext ctx{graph->ggml.get(), "nemotron_asr.decoder_joint", execution_context_->backend_type()};
    graph->encoder_frame = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size}));
    graph->decoder_cache_in = engine::core::make_tensor(ctx, GGML_TYPE_F32, engine::core::TensorShape::from_dims({1, config.decoder_hidden_size}));
    ggml_set_input(graph->encoder_frame.tensor);
    ggml_set_input(graph->decoder_cache_in.tensor);

    auto joint = engine::modules::AddModule().build(ctx, graph->encoder_frame, graph->decoder_cache_in);
    joint = engine::modules::ReluModule().build(ctx, joint);
    graph->logits = engine::modules::LinearModule({config.decoder_hidden_size, config.vocab_size, true})
                        .build(ctx, joint, weights_->decoder.joint_head);
    ggml_set_output(graph->logits.tensor);

    graph->graph = ggml_new_graph(graph->ggml.get());
    ggml_build_forward_expand(graph->graph, graph->logits.tensor);
    graph->allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_context_->backend()));
    if (graph->allocator == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder failed to create joint graph allocator");
    }
    if (!ggml_gallocr_alloc_graph(graph->allocator, graph->graph)) {
        throw std::runtime_error("Nemotron ASR decoder joint graph allocation failed");
    }
    engine::core::validate_backend_graph_supported(execution_context_->backend(), graph->graph, "Nemotron ASR decoder joint");
    engine::core::prepare_host_graph_plan(*execution_context_, graph->graph, graph->host_plan);

    const double build_ms = engine::debug::elapsed_ms(build_start, Clock::now());
    debug::timing_log_scalar("nemotron_asr.decoder.joint_graph_build_ms", build_ms);
    debug::timing_log_scalar("nemotron_asr.decoder.joint_graph_rebuild_ms", build_ms);
    debug::trace_log_scalar("nemotron_asr.decoder.joint_graph_cache_hit", false);
    joint_graph_ = std::move(graph);
}

int32_t NemotronDecoderRuntime::run_joint_step(const float * encoder_frame) {
    if (joint_graph_ == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder joint graph is not prepared");
    }
    auto & graph = *joint_graph_;
    const auto & config = assets_->config;
    engine::core::write_tensor_f32(graph.encoder_frame, encoder_frame, static_cast<size_t>(config.decoder_hidden_size));
    engine::core::write_tensor_f32(graph.decoder_cache_in, decoder_cache_scratch_);

    const auto status = engine::core::compute_graph(*execution_context_, graph.graph, graph.host_plan, "Nemotron ASR decoder joint step");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR decoder joint graph compute failed");
    }

    engine::core::read_tensor_f32_into(graph.logits.tensor, logits_scratch_);
    return argmax_index(logits_scratch_);
}

int32_t NemotronDecoderRuntime::run_step(
    int32_t input_token,
    const float * encoder_frame,
    bool decoder_cache_initialized) {
    if (graph_ == nullptr) {
        throw std::runtime_error("Nemotron ASR decoder graph is not prepared");
    }
    auto & graph = *graph_;
    const auto & config = assets_->config;
    if (input_token < 0 || input_token >= config.vocab_size) {
        throw std::runtime_error("Nemotron ASR decoder token is out of range");
    }

    const bool update_predictor =
        !decoder_cache_initialized || input_token != static_cast<int32_t>(config.blank_token_id);
    if (!update_predictor) {
        if (joint_graph_ == nullptr) {
            ensure_joint_graph();
        }
        return run_joint_step(encoder_frame);
    }

    engine::core::write_tensor_i32(graph.token_id, &input_token, 1);
    engine::core::write_tensor_f32(graph.encoder_frame, encoder_frame, static_cast<size_t>(config.decoder_hidden_size));
    for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
        const size_t offset = static_cast<size_t>(layer * config.decoder_hidden_size);
        engine::core::write_tensor_f32(
            graph.hidden_in[static_cast<size_t>(layer)],
            hidden_scratch_.data() + offset,
            static_cast<size_t>(config.decoder_hidden_size));
        engine::core::write_tensor_f32(
            graph.cell_in[static_cast<size_t>(layer)],
            cell_scratch_.data() + offset,
            static_cast<size_t>(config.decoder_hidden_size));
    }

    const auto status = engine::core::compute_graph(*execution_context_, graph.graph, graph.host_plan, "Nemotron ASR decoder step");
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("Nemotron ASR decoder graph compute failed");
    }

    engine::core::read_tensor_f32_into(graph.logits.tensor, logits_scratch_);
    if (update_predictor) {
        engine::core::read_tensor_f32_into(graph.decoder_cache_out.tensor, decoder_cache_scratch_);
        for (int64_t layer = 0; layer < config.decoder_layers; ++layer) {
            const size_t offset = static_cast<size_t>(layer * config.decoder_hidden_size);
            engine::core::read_tensor_f32_into(graph.hidden_out[static_cast<size_t>(layer)].tensor, hidden_read_scratch_);
            std::copy(hidden_read_scratch_.begin(), hidden_read_scratch_.end(), hidden_scratch_.begin() + static_cast<std::ptrdiff_t>(offset));
            engine::core::read_tensor_f32_into(graph.cell_out[static_cast<size_t>(layer)].tensor, cell_read_scratch_);
            std::copy(cell_read_scratch_.begin(), cell_read_scratch_.end(), cell_scratch_.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }
    return argmax_index(logits_scratch_);
}

std::string NemotronDecoderRuntime::decode_text(const std::vector<int32_t> & token_ids, bool keep_language_tags) const {
    std::vector<int32_t> filtered;
    filtered.reserve(token_ids.size());
    for (const int32_t id : token_ids) {
        if (id == static_cast<int32_t>(assets_->config.blank_token_id) ||
            id == static_cast<int32_t>(assets_->config.pad_token_id)) {
            continue;
        }
        if (!keep_language_tags &&
            id >= 0 &&
            id < static_cast<int32_t>(assets_->special_token_ids.size()) &&
            assets_->special_token_ids[static_cast<size_t>(id)] != 0) {
            continue;
        }
        filtered.push_back(id);
    }
    return assets_->tokenizer->decode_ids(filtered);
}

NemotronDecodedText NemotronDecoderRuntime::decode(
    const NemotronEncodedAudio & encoded,
    const NemotronDecodeOptions & options) {
    if (encoded.valid_frames <= 0 || encoded.hidden_size != assets_->config.decoder_hidden_size) {
        throw std::runtime_error("Nemotron ASR decoder requires encoded audio frames");
    }
    const auto wall_start = Clock::now();
    ensure_graph();
    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto & config = assets_->config;
    const int64_t max_tokens = options.max_tokens > 0
        ? options.max_tokens
        : (encoded.valid_frames * config.max_symbols_per_step + 1);
    hidden_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    cell_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    decoder_cache_scratch_.assign(static_cast<size_t>(config.decoder_hidden_size), 0.0f);

    NemotronDecodedText out;
    out.token_ids.reserve(static_cast<size_t>(std::min<int64_t>(max_tokens + 1, 4096)));
    out.durations.reserve(out.token_ids.capacity());
    out.token_ids.push_back(static_cast<int32_t>(config.blank_token_id));
    out.durations.push_back(0);

    int64_t frame_index = 0;
    int64_t symbols_at_frame = 0;
    int32_t input_token = static_cast<int32_t>(config.blank_token_id);
    bool decoder_cache_initialized = false;
    while (frame_index < encoded.valid_frames && static_cast<int64_t>(out.token_ids.size()) - 1 < max_tokens) {
        const float * frame = encoded.values.data() + static_cast<std::ptrdiff_t>(frame_index * encoded.hidden_size);
        const int32_t token = run_step(input_token, frame, decoder_cache_initialized);
        decoder_cache_initialized = true;
        out.token_ids.push_back(token);
        const bool blank = token == static_cast<int32_t>(config.blank_token_id);
        if (!blank) {
            ++symbols_at_frame;
        }
        const bool force_advance = symbols_at_frame >= config.max_symbols_per_step;
        if (blank || force_advance) {
            ++frame_index;
            symbols_at_frame = 0;
            out.durations.push_back(1);
        } else {
            out.durations.push_back(0);
        }
        input_token = token;
    }
    out.text = decode_text(out.token_ids, options.keep_language_tags);
    out.token_timestamps = build_token_timestamps(*assets_, out.token_ids, out.durations);
    debug::timing_log_scalar("nemotron_asr.decoder_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.decoder.tokens", out.token_ids.size());
    debug::trace_log_scalar("nemotron_asr.decoder.encoded_valid_frames", encoded.valid_frames);
    return out;
}

NemotronDecodedText NemotronDecoderRuntime::decode_streaming(
    const NemotronDecodeOptions & options,
    const std::function<bool(NemotronEncodedAudio &)> & next_chunk,
    const NemotronTextDeltaCallback & on_text_delta) {
    if (!next_chunk) {
        throw std::runtime_error("Nemotron ASR streaming decoder requires a chunk producer");
    }
    const auto wall_start = Clock::now();
    ensure_graph();
    engine::core::set_backend_threads(execution_context_->backend(), execution_context_->config().threads);
    const auto & config = assets_->config;
    const int64_t max_tokens = options.max_tokens > 0
        ? options.max_tokens
        : (std::numeric_limits<int64_t>::max() / 4);
    hidden_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    cell_scratch_.assign(static_cast<size_t>(config.decoder_layers * config.decoder_hidden_size), 0.0f);
    decoder_cache_scratch_.assign(static_cast<size_t>(config.decoder_hidden_size), 0.0f);

    std::vector<float> encoded_values;
    int64_t encoded_valid_frames = 0;
    int64_t encoded_hidden_size = 0;
    bool stream_exhausted = false;
    auto append_next_chunk = [&]() -> bool {
        NemotronEncodedAudio chunk;
        if (!next_chunk(chunk)) {
            stream_exhausted = true;
            return false;
        }
        if (chunk.valid_frames <= 0 || chunk.hidden_size != config.decoder_hidden_size) {
            throw std::runtime_error("Nemotron ASR streaming decoder received invalid encoded chunk");
        }
        if (encoded_hidden_size == 0) {
            encoded_hidden_size = chunk.hidden_size;
        } else if (encoded_hidden_size != chunk.hidden_size) {
            throw std::runtime_error("Nemotron ASR streaming decoder chunk hidden size mismatch");
        }
        encoded_values.insert(
            encoded_values.end(),
            chunk.values.begin(),
            chunk.values.begin() + static_cast<std::ptrdiff_t>(chunk.valid_frames * chunk.hidden_size));
        encoded_valid_frames += chunk.valid_frames;
        return true;
    };
    if (!append_next_chunk()) {
        throw std::runtime_error("Nemotron ASR streaming decoder received no encoded chunks");
    }

    NemotronDecodedText out;
    out.token_ids.reserve(4096);
    out.durations.reserve(out.token_ids.capacity());
    out.token_ids.push_back(static_cast<int32_t>(config.blank_token_id));
    out.durations.push_back(0);
    std::string emitted_text;

    int64_t frame_index = 0;
    int64_t symbols_at_frame = 0;
    int32_t input_token = static_cast<int32_t>(config.blank_token_id);
    bool decoder_cache_initialized = false;
    while (static_cast<int64_t>(out.token_ids.size()) - 1 < max_tokens) {
        while (frame_index >= encoded_valid_frames && !stream_exhausted) {
            append_next_chunk();
        }
        if (frame_index >= encoded_valid_frames) {
            break;
        }

        const float * frame = encoded_values.data() + static_cast<std::ptrdiff_t>(frame_index * encoded_hidden_size);
        const int32_t token = run_step(input_token, frame, decoder_cache_initialized);
        decoder_cache_initialized = true;
        out.token_ids.push_back(token);
        const bool blank = token == static_cast<int32_t>(config.blank_token_id);
        if (!blank) {
            ++symbols_at_frame;
        }
        const bool force_advance = symbols_at_frame >= config.max_symbols_per_step;
        if (blank || force_advance) {
            ++frame_index;
            symbols_at_frame = 0;
            out.durations.push_back(1);
        } else {
            out.durations.push_back(0);
        }
        if (on_text_delta && !blank) {
            const auto current_text = decode_text(out.token_ids, options.keep_language_tags);
            if (current_text.size() > emitted_text.size() &&
                current_text.compare(0, emitted_text.size(), emitted_text) == 0) {
                on_text_delta(current_text.substr(emitted_text.size()));
                emitted_text = current_text;
            } else if (current_text != emitted_text) {
                on_text_delta(current_text);
                emitted_text = current_text;
            }
        }
        input_token = token;
    }
    out.text = decode_text(out.token_ids, options.keep_language_tags);
    out.token_timestamps = build_token_timestamps(*assets_, out.token_ids, out.durations);
    debug::timing_log_scalar("nemotron_asr.decoder_ms", engine::debug::elapsed_ms(wall_start, Clock::now()));
    debug::trace_log_scalar("nemotron_asr.decoder.tokens", out.token_ids.size());
    debug::trace_log_scalar("nemotron_asr.decoder.encoded_valid_frames", encoded_valid_frames);
    return out;
}

}  // namespace engine::models::nemotron_asr
