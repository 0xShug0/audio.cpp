#include "engine/models/muscriptor/session.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::muscriptor {
namespace {

using Clock = std::chrono::steady_clock;

constexpr const char * kFamily = "muscriptor";

enum class MuScriptorOutputFormat {
    Json,
    Midi,
};

std::shared_ptr<const MuScriptorAssets> require_assets(std::shared_ptr<const MuScriptorAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("MuScriptor session requires assets");
    }
    return assets;
}

std::shared_ptr<const engine::model_spec::ModelContract> require_contract(
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    if (contract == nullptr) {
        throw std::runtime_error("MuScriptor session requires a model contract");
    }
    return contract;
}

MuScriptorPerfMode perf_mode_from_options(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"muscriptor.perf_mode"})) {
        if (*value == "off") {
            return MuScriptorPerfMode::Exact;
        }
        if (*value == "flash_attention") {
            return MuScriptorPerfMode::FlashAttention;
        }
        throw std::runtime_error("Invalid muscriptor.perf_mode: " + *value);
    }
    return MuScriptorPerfMode::FlashAttention;
}

MuScriptorDecoderOptions decoder_options_from_session_options(const runtime::SessionOptions & options) {
    MuScriptorDecoderOptions out;
    if (const auto value = runtime::find_option(options.options, {"muscriptor.weight_type"})) {
        out.weight_type = assets::parse_tensor_storage_type(*value);
    }
    out.perf_mode = perf_mode_from_options(options);
    out.weight_context_bytes =
        runtime::parse_size_mb_option(options.options, {"muscriptor.weight_context_mb"}, out.weight_context_bytes);
    out.condition_graph_arena_bytes =
        runtime::parse_size_mb_option(options.options, {"muscriptor.conditioning_graph_arena_mb"}, out.condition_graph_arena_bytes);
    out.prefill_graph_arena_bytes =
        runtime::parse_size_mb_option(options.options, {"muscriptor.decoder_prefill_graph_arena_mb"}, out.prefill_graph_arena_bytes);
    out.decode_graph_arena_bytes =
        runtime::parse_size_mb_option(options.options, {"muscriptor.decoder_decode_graph_arena_mb"}, out.decode_graph_arena_bytes);
    return out;
}

MuScriptorOutputFormat output_format_from_request(const runtime::TaskRequest & request) {
    const std::string value = runtime::find_option(request.options, {"output_format"}).value_or("midi");
    if (value == "midi") {
        return MuScriptorOutputFormat::Midi;
    }
    if (value == "json") {
        return MuScriptorOutputFormat::Json;
    }
    throw std::runtime_error("MuScriptor output_format must be json or midi");
}

std::unique_ptr<runtime::IVoiceTaskSession> create_muscriptor_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options,
    std::shared_ptr<const MuScriptorAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract) {
    return std::make_unique<MuScriptorSession>(
        task,
        options,
        std::move(assets),
        std::move(contract));
}

}  // namespace

MuScriptorSession::MuScriptorSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const MuScriptorAssets> assets,
    std::shared_ptr<const engine::model_spec::ModelContract> contract)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      assets_(require_assets(std::move(assets))),
      contract_(require_contract(std::move(contract))),
      frontend_(assets_, execution_context().config().threads),
      decoder_(
          assets_,
          execution_context(),
          decoder_options_from_session_options(RuntimeSessionBase::options())) {
    if (task_.task != runtime::VoiceTaskKind::Midi && task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("MuScriptor supports VoiceTaskKind::Midi");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("MuScriptor supports offline and streaming sessions");
    }
    runtime::validate_spec_backed_session_options(RuntimeSessionBase::options(), *contract_, kFamily, "MuScriptor");
}

MuScriptorSession::~MuScriptorSession() = default;

std::string MuScriptorSession::family() const {
    return kFamily;
}

runtime::VoiceTaskKind MuScriptorSession::task_kind() const {
    return task_.task;
}

runtime::RunMode MuScriptorSession::run_mode() const {
    return task_.mode;
}

void MuScriptorSession::prepare(const runtime::SessionPreparationRequest & request) {
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MuScriptor");
    mark_prepared();
}

runtime::TaskResult MuScriptorSession::run(const runtime::TaskRequest & request) {
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("MuScriptor run() requires offline mode");
    }
    return transcribe(request, nullptr);
}

runtime::StreamingPolicy MuScriptorSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_seconds = 5.0;
    return policy;
}

void MuScriptorSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("MuScriptor start_stream()");
    reset();
    streaming_request_ = request;
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MuScriptor streaming requires audio_input format");
    }
    streaming_audio_.sample_rate = request.audio_input->sample_rate;
    streaming_audio_.channels = request.audio_input->channels;
    stream_started_ = true;
}

void MuScriptorSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_sink_ = std::move(sink);
}

runtime::TaskResult MuScriptorSession::finish_stream() {
    return finalize();
}

void MuScriptorSession::reset() {
    streaming_request_ = {};
    streaming_audio_ = {};
    stream_started_ = false;
}

runtime::StreamEvent MuScriptorSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    if (!stream_started_) {
        throw std::runtime_error("MuScriptor process_audio_chunk() requires start_stream()");
    }
    if (chunk.sample_rate != streaming_audio_.sample_rate || chunk.channels != streaming_audio_.channels) {
        throw std::runtime_error("MuScriptor streaming audio format changed mid-stream");
    }
    streaming_audio_.samples.insert(streaming_audio_.samples.end(), chunk.samples.begin(), chunk.samples.end());
    return {};
}

runtime::TaskResult MuScriptorSession::finalize() {
    if (!stream_started_) {
        throw std::runtime_error("MuScriptor finalize() requires start_stream()");
    }
    streaming_request_.audio_input = streaming_audio_;
    auto result = transcribe(streaming_request_, stream_sink_);
    runtime::StreamEvent done;
    done.is_final = true;
    if (result.artifact_output.has_value()) {
        done.output_artifacts.push_back(*result.artifact_output);
    }
    if (stream_sink_) {
        stream_sink_(done);
    }
    reset();
    return result;
}

void MuScriptorSession::emit_stream_events(
    const std::vector<MuScriptorEvent> & decoded,
    size_t & published_events,
    size_t completed_chunks,
    size_t total_chunks,
    const runtime::StreamEventCallback & stream_sink) const {
    if (!stream_sink) {
        return;
    }
    while (published_events < decoded.size()) {
        const auto & event = decoded[published_events];
        runtime::StreamEvent stream_event;
        stream_event.output_artifacts.push_back(runtime::make_text_artifact(
            runtime::ArtifactKind::Custom,
            "event-" + std::to_string(published_events),
            muscriptor_event_to_json(event),
            {
                {"mime", "application/json"},
                {"format", "muscriptor-event-json"},
                {"event_type", event.kind == MuScriptorEvent::Kind::Start ? "start" : "end"},
                {"extension", "json"},
            }));
        stream_sink(stream_event);
        ++published_events;
    }
    runtime::StreamEvent progress;
    progress.output_artifacts.push_back(runtime::make_text_artifact(
        runtime::ArtifactKind::Custom,
        "progress-" + std::to_string(completed_chunks),
        "{\"type\":\"progress\",\"completed\":" + std::to_string(completed_chunks) +
            ",\"total\":" + std::to_string(total_chunks) + "}",
        {
            {"mime", "application/json"},
            {"format", "muscriptor-event-json"},
            {"event_type", "progress"},
            {"extension", "json"},
        }));
    stream_sink(progress);
}

runtime::TaskResult MuScriptorSession::transcribe(
    const runtime::TaskRequest & request,
    const runtime::StreamEventCallback & stream_sink) {
    require_prepared("MuScriptor run()");
    runtime::validate_spec_backed_request_options(request.options, *contract_, "MuScriptor");
    const auto total_start = Clock::now();
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("MuScriptor run() requires audio_input");
    }
    MuScriptorGenerationOptions generation;
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        generation.use_sampling = runtime::parse_bool_option(*value, "do_sample");
    }
    generation.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(1.0F);
    generation.guidance_scale = runtime::parse_finite_float_option(request.options, {"guidance_scale"}).value_or(1.0F);
    generation.batch_size = runtime::parse_positive_i64_option(request.options, {"batch_size"}, 1);
    generation.num_beams = runtime::parse_positive_i64_option(request.options, {"num_beams"}, 1);
    if (const auto value = runtime::find_option(request.options, {"prelude_forcing"})) {
        generation.prelude_forcing = runtime::parse_bool_option(*value, "prelude_forcing");
    }
    generation.seed = static_cast<uint64_t>(runtime::parse_i64_option(request.options, {"seed"}).value_or(0));
    if (generation.prelude_forcing && generation.batch_size > 1) {
        throw std::runtime_error("MuScriptor batch_size > 1 requires prelude_forcing=false");
    }
    if (generation.temperature < 0.0F) {
        throw std::runtime_error("MuScriptor temperature must be non-negative");
    }
    const int64_t max_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"}, 2000);
    const std::string instruments = runtime::find_option(request.options, {"instruments"}).value_or("");
    const auto forbidden_tokens = instruments.empty() ? std::vector<int32_t>{} : tokenizer_.forbidden_token_ids(instruments);
    const auto instrument_ids = tokenizer_.condition_instrument_ids(instruments);
    const auto chunks = frontend_.extract_chunks(*request.audio_input);
    std::vector<MuScriptorGeneratedChunk> generated;
    generated.reserve(chunks.size());
    size_t published_events = 0;
    const bool use_cfg = generation.guidance_scale != 1.0F;
    for (size_t batch_start = 0; batch_start < chunks.size(); batch_start += static_cast<size_t>(generation.batch_size)) {
        const size_t batch_end = std::min(chunks.size(), batch_start + static_cast<size_t>(generation.batch_size));
        const int64_t batch = static_cast<int64_t>(batch_end - batch_start);
        const int64_t frames = static_cast<int64_t>(chunks[batch_start].mask.size());
        const int64_t instrument_steps = static_cast<int64_t>(instrument_ids.size());
        const int64_t condition_batch = use_cfg ? batch * 2 : batch;
        std::vector<float> log_mel(static_cast<size_t>(condition_batch * frames * assets_->config.n_mels), 0.0F);
        std::vector<int32_t> mel_mask(static_cast<size_t>(condition_batch * frames), 0);
        std::vector<int32_t> batch_instrument_ids(static_cast<size_t>(condition_batch * instrument_steps), 1);
        std::vector<int32_t> dataset_ids(static_cast<size_t>(condition_batch), 1);
        std::vector<std::vector<int32_t>> prompts;
        prompts.reserve(static_cast<size_t>(batch));
        for (int64_t row = 0; row < batch; ++row) {
            const size_t i = batch_start + static_cast<size_t>(row);
            if (static_cast<int64_t>(chunks[i].mask.size()) != frames) {
                throw std::runtime_error("MuScriptor batched chunks must have equal frame count");
            }
            std::copy(
                chunks[i].log_mel.begin(),
                chunks[i].log_mel.end(),
                log_mel.begin() + static_cast<ptrdiff_t>(row * frames * assets_->config.n_mels));
            std::copy(
                chunks[i].mask.begin(),
                chunks[i].mask.end(),
                mel_mask.begin() + static_cast<ptrdiff_t>(row * frames));
            std::copy(
                instrument_ids.begin(),
                instrument_ids.end(),
                batch_instrument_ids.begin() + static_cast<ptrdiff_t>(row * instrument_steps));
            std::vector<int32_t> prompt;
            if (generation.prelude_forcing && i > 0) {
                prompt = tokenizer_.tie_section_token_ids(tokenizer_.open_note_keys(generated));
            }
            prompts.push_back(std::move(prompt));
        }
        const auto conditioning = decoder_.condition_batch(log_mel, mel_mask, batch_instrument_ids, dataset_ids, condition_batch);
        auto batch_generated = decoder_.generate_batch(
            conditioning,
            prompts,
            max_tokens,
            tokenizer_.eos_id(),
            forbidden_tokens,
            generation);
        for (size_t j = 0; j < batch_generated.size(); ++j) {
            generated.push_back(std::move(batch_generated[j]));
        }
        if (stream_sink) {
            emit_stream_events(
                tokenizer_.decode_chunks(generated),
                published_events,
                std::min(batch_end, chunks.size()),
                chunks.size(),
                stream_sink);
        }
    }
    const auto decode_events_start = Clock::now();
    const auto decoded = tokenizer_.decode_chunks(generated);
    const std::string events_json = muscriptor_events_to_json(decoded);
    engine::debug::timing_log_scalar("muscriptor.decode_events_ms", engine::debug::elapsed_ms(decode_events_start, Clock::now()));
    runtime::TaskResult result;
    result.text_output = runtime::Transcript{events_json, "midi-json"};
    switch (output_format_from_request(request)) {
    case MuScriptorOutputFormat::Json:
        result.artifact_output = runtime::make_text_artifact(
            runtime::ArtifactKind::Custom,
            "result",
            events_json,
            {
                {"mime", "application/json"},
                {"format", "muscriptor-event-json"},
                {"extension", "json"},
            });
        break;
    case MuScriptorOutputFormat::Midi:
        result.artifact_output = runtime::make_voice_artifact(
            runtime::ArtifactKind::Midi,
            "result",
            muscriptor_events_to_midi_bytes(decoded),
            {
                {"mime", "audio/midi"},
                {"format", "midi"},
                {"extension", "mid"},
            });
        break;
    }
    engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(total_start, Clock::now()));
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> make_muscriptor_loader() {
    runtime::SpecBackedVoiceModelConfig<MuScriptorAssets> config;
    config.family = kFamily;
    config.load_assets = load_muscriptor_assets;
    config.create_session = create_muscriptor_session;
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::muscriptor
