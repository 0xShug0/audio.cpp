#include "engine/models/qwen3_asr/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/qwen3_forced_aligner/session.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace engine::models::qwen3_asr {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kTimestampFixedChunkContextSeconds = 1.0;

std::shared_ptr<const Qwen3ASRAssets> require_assets(std::shared_ptr<const Qwen3ASRAssets> assets) {
    if (assets == nullptr) {
        throw std::runtime_error("Qwen3 ASR session requires assets");
    }
    return assets;
}

void validate_matmul_weight_storage(engine::assets::TensorStorageType storage_type, const char * option_name) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16 ||
        storage_type == engine::assets::TensorStorageType::BF16 ||
        storage_type == engine::assets::TensorStorageType::Q8_0) {
        return;
    }
    throw std::runtime_error(std::string(option_name) + " currently supports only native, f32, f16, bf16, and q8_0");
}

void validate_audio_encoder_weight_storage(engine::assets::TensorStorageType storage_type) {
    if (storage_type == engine::assets::TensorStorageType::Native ||
        storage_type == engine::assets::TensorStorageType::F32 ||
        storage_type == engine::assets::TensorStorageType::F16) {
        return;
    }
    throw std::runtime_error("qwen3_asr.audio_encoder_weight_type currently supports only native, f32, and f16");
}

engine::assets::TensorStorageType option_weight_type(
    const runtime::SessionOptions & options,
    const char * key,
    engine::assets::TensorStorageType default_value) {
    const auto it = options.options.find(key);
    if (it == options.options.end()) {
        return default_value;
    }
    return engine::assets::parse_tensor_storage_type(it->second);
}

std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

int64_t audio_frame_count(const runtime::AudioBuffer & audio) {
    if (audio.channels <= 0) {
        throw std::runtime_error("Qwen3 ASR audio chunking requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("Qwen3 ASR audio samples must be divisible by channel count");
    }
    return static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
}

bool request_return_timestamps(const runtime::TaskRequest & request) {
    if (const auto value = runtime::find_option(request.options, {"return_timestamps"})) {
        return runtime::parse_bool_option(*value, "return_timestamps");
    }
    return false;
}

}  // namespace

Qwen3ASRSession::Qwen3ASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const Qwen3ASRAssets> assets)
    : RuntimeSessionBase(options),
      task_(task),
      assets_(require_assets(std::move(assets))),
      audio_encoder_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.audio_encoder_graph_arena_mb"}, 128ull * 1024ull * 1024ull)),
      thinker_prefill_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_prefill_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_decode_graph_arena_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_decode_graph_arena_mb"}, 256ull * 1024ull * 1024ull)),
      thinker_weight_context_bytes_(runtime::parse_size_mb_option(options.options, {"qwen3_asr.thinker_weight_context_mb"}, 64ull * 1024ull * 1024ull)),
      audio_encoder_weight_storage_type_(option_weight_type(options, "qwen3_asr.audio_encoder_weight_type", engine::assets::TensorStorageType::Native)),
      thinker_weight_storage_type_(option_weight_type(
          options,
          "qwen3_asr.thinker_weight_type",
          option_weight_type(options, "qwen3_asr.weight_type", engine::assets::TensorStorageType::Native))),
      tokenizer_(assets_),
      frontend_(assets_),
      audio_encoder_(assets_, execution_context(), audio_encoder_graph_arena_bytes_, audio_encoder_weight_storage_type_),
      thinker_(
          assets_,
          execution_context(),
          thinker_prefill_graph_arena_bytes_,
          thinker_decode_graph_arena_bytes_,
          thinker_weight_context_bytes_,
          thinker_weight_storage_type_),
      prompt_builder_(tokenizer_),
      postprocessor_(tokenizer_),
      vad_model_path_(runtime::find_option(options.options, {"qwen3_asr.vad_model_path"}).value_or(default_vad_model_path().string())) {
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Qwen3 ASR only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline) {
        throw std::runtime_error("Qwen3 ASR currently supports offline sessions");
    }
    validate_audio_encoder_weight_storage(audio_encoder_weight_storage_type_);
    validate_matmul_weight_storage(thinker_weight_storage_type_, "qwen3_asr.thinker_weight_type");
    for (const auto & [key, value] : options.options) {
        (void) value;
        if (key.rfind("qwen3_asr.", 0) == 0 &&
            key != "qwen3_asr.audio_encoder_graph_arena_mb" &&
            key != "qwen3_asr.thinker_prefill_graph_arena_mb" &&
            key != "qwen3_asr.thinker_decode_graph_arena_mb" &&
            key != "qwen3_asr.thinker_weight_context_mb" &&
            key != "qwen3_asr.audio_encoder_weight_type" &&
            key != "qwen3_asr.thinker_weight_type" &&
            key != "qwen3_asr.weight_type" &&
            key != "qwen3_asr.forced_aligner_model_path" &&
            key != "qwen3_asr.aligner_model_path" &&
            key != "qwen3_asr.vad_model_path") {
            throw std::runtime_error("unknown Qwen3 ASR session option: " + key);
        }
    }
    if (const auto aligner_path = runtime::find_option(
            options.options,
            {"qwen3_asr.forced_aligner_model_path", "qwen3_asr.aligner_model_path"})) {
        runtime::SessionOptions aligner_options;
        aligner_options.backend = options.backend;
        for (const auto & [key, value] : options.options) {
            if (key.rfind("qwen3_forced_aligner.", 0) == 0) {
                aligner_options.options.emplace(key, value);
            }
        }
        forced_aligner_session_ = std::make_unique<engine::models::qwen3_forced_aligner::Qwen3ForcedAlignerSession>(
            runtime::TaskSpec{runtime::VoiceTaskKind::Alignment, runtime::RunMode::Offline},
            aligner_options,
            load_qwen3_asr_assets(std::filesystem::path(*aligner_path)));
    }
    assets_->model_weights->release_storage();
}

Qwen3ASRSession::~Qwen3ASRSession() = default;

std::string Qwen3ASRSession::family() const {
    return "qwen3_asr";
}

runtime::VoiceTaskKind Qwen3ASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode Qwen3ASRSession::run_mode() const {
    return task_.mode;
}

void Qwen3ASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Qwen3 ASR prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult Qwen3ASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Qwen3 ASR run()");
    const auto chunks = audio_chunk_plan(request);
    if (chunks.empty()) {
        const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
        if (request.audio_input.has_value() &&
            (mode == engine::audio::AudioChunkMode::Vad ||
             (mode == engine::audio::AudioChunkMode::Auto && request_return_timestamps(request)))) {
            runtime::TaskResult empty;
            empty.text_output = runtime::Transcript{"", request.text_input.has_value() ? request.text_input->language : ""};
            return empty;
        }
        return run_single(make_request(request));
    }
    const auto & audio = *request.audio_input;
    if (chunks.size() <= 1) {
        auto item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunks.front().source_span);
        auto item = run_single(make_request(item_request));
        std::vector<runtime::WordTimestamp> merged_words;
        engine::audio::append_chunk_word_timestamps(
            merged_words,
            item.word_timestamps,
            chunks.front().source_span,
            chunks.front().keep_span);
        item.word_timestamps = std::move(merged_words);
        return item;
    }
    runtime::TaskResult merged;
    std::ostringstream text;
    for (const auto & chunk : chunks) {
        runtime::TaskRequest item_request = request;
        item_request.audio_input = engine::audio::slice_audio_buffer(audio, chunk.source_span);
        auto item = run_single(make_request(item_request));
        if (item.text_output.has_value() && !item.text_output->text.empty()) {
            if (text.tellp() > 0) {
                text << ' ';
            }
            text << item.text_output->text;
            if (!merged.text_output.has_value()) {
                merged.text_output = runtime::Transcript{"", item.text_output->language};
            } else if (merged.text_output->language.empty()) {
                merged.text_output->language = item.text_output->language;
            }
        }
        engine::audio::append_chunk_word_timestamps(
            merged.word_timestamps,
            item.word_timestamps,
            chunk.source_span,
            chunk.keep_span);
    }
    if (merged.text_output.has_value()) {
        if (request_return_timestamps(request) && !merged.word_timestamps.empty()) {
            std::ostringstream word_text;
            for (const auto & word : merged.word_timestamps) {
                if (word_text.tellp() > 0) {
                    word_text << ' ';
                }
                word_text << word.word;
            }
            merged.text_output->text = word_text.str();
        } else {
            merged.text_output->text = text.str();
        }
    }
    return merged;
}

runtime::TaskResult Qwen3ASRSession::run_single(const Qwen3ASRRequest & asr_request) {
    const auto wall_start = Clock::now();
    if (asr_request.generation.return_timestamps && forced_aligner_session_ == nullptr) {
        throw std::runtime_error(
            "Qwen3 ASR timestamp output requires --session-option "
            "qwen3_asr.forced_aligner_model_path=<path-to-Qwen3-ForcedAligner-0.6B>");
    }
    const auto frontend_start = Clock::now();
    const auto features = frontend_.extract(asr_request.audio);
    const auto frontend_end = Clock::now();
    const auto prompt_start = Clock::now();
    const auto prompt = prompt_builder_.build(asr_request, features.encoder_tokens);
    const auto prompt_end = Clock::now();
    const auto encoder_start = Clock::now();
    const auto audio_embeddings = audio_encoder_.encode(features);
    const auto encoder_end = Clock::now();
    const auto thinker_start = Clock::now();
    const auto tokens = thinker_.generate(prompt, audio_embeddings, asr_request.generation);
    const auto thinker_end = Clock::now();
    const auto postprocess_start = Clock::now();
    const auto decoded = postprocessor_.decode(tokens, asr_request);
    const auto postprocess_end = Clock::now();

    runtime::TaskResult result;
    result.text_output = runtime::Transcript{decoded.text, decoded.language};
    result.word_timestamps = decoded.word_timestamps;
    if (asr_request.generation.return_timestamps) {
        if (!decoded.text.empty()) {
            if (decoded.language.empty()) {
                throw std::runtime_error("Qwen3 ASR timestamp output requires a requested or detected language");
            }
            runtime::TaskRequest align_request;
            align_request.audio_input = asr_request.audio;
            align_request.text_input = runtime::Transcript{decoded.text, decoded.language};
            align_request.options["audio_chunk_mode"] = "none";
            forced_aligner_session_->prepare(runtime::build_preparation_request(align_request));
            auto aligned = forced_aligner_session_->run(align_request);
            result.word_timestamps = std::move(aligned.word_timestamps);
        }
    }
    const auto wall_end = Clock::now();
    debug::timing_log_scalar("qwen3_asr.frontend_ms", engine::debug::elapsed_ms(frontend_start, frontend_end));
    debug::timing_log_scalar("qwen3_asr.prompt_ms", engine::debug::elapsed_ms(prompt_start, prompt_end));
    debug::timing_log_scalar("qwen3_asr.audio_encoder_ms", engine::debug::elapsed_ms(encoder_start, encoder_end));
    debug::timing_log_scalar("qwen3_asr.thinker_ms", engine::debug::elapsed_ms(thinker_start, thinker_end));
    debug::timing_log_scalar("qwen3_asr.postprocess_ms", engine::debug::elapsed_ms(postprocess_start, postprocess_end));
    debug::trace_log_scalar("qwen3_asr.prompt_tokens", prompt.input_ids.size());
    debug::trace_log_scalar("qwen3_asr.audio_frames", features.frames);
    debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(wall_start, wall_end));
    return result;
}

std::vector<Qwen3ASRSession::AudioChunkPlan> Qwen3ASRSession::audio_chunk_plan(const runtime::TaskRequest & request) {
    if (!request.audio_input.has_value()) {
        return {};
    }
    const auto mode = engine::audio::parse_audio_chunk_mode(request.options);
    if (mode == engine::audio::AudioChunkMode::None) {
        return {};
    }
    if (mode == engine::audio::AudioChunkMode::QuietEnergy) {
        throw std::runtime_error("Qwen3 ASR does not support audio_chunk_mode=quiet_energy");
    }
    const bool return_timestamps = request_return_timestamps(request);
    const auto & audio = *request.audio_input;
    const int64_t frames = audio_frame_count(audio);
    if (mode == engine::audio::AudioChunkMode::Vad ||
        (mode == engine::audio::AudioChunkMode::Auto && return_timestamps)) {
        const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options).value_or(15.0F);
        if (!(seconds > 0.0F)) {
            throw std::runtime_error("Qwen3 ASR audio_chunk_seconds must be positive");
        }
        const auto options = engine::audio::VadAudioChunkOptions{
            static_cast<int64_t>(std::llround(static_cast<double>(seconds) * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.5 * static_cast<double>(audio.sample_rate))),
            static_cast<int64_t>(std::llround(0.25 * static_cast<double>(audio.sample_rate))),
        };
        if (options.max_chunk_samples <= 0) {
            throw std::runtime_error("Qwen3 ASR audio_chunk_seconds produced an empty chunk");
        }
        const auto spans = engine::audio::plan_vad_audio_chunks(audio, vad_session(), options);
        std::vector<AudioChunkPlan> plan;
        plan.reserve(spans.size());
        for (const auto & span : spans) {
            plan.push_back(AudioChunkPlan{span, span});
        }
        return plan;
    }
    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(request.options).value_or(return_timestamps ? 15.0F : 30.0F);
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("Qwen3 ASR audio_chunk_seconds must be positive");
    }
    const int64_t samples = static_cast<int64_t>(
        std::llround(static_cast<double>(seconds) * static_cast<double>(request.audio_input->sample_rate)));
    if (samples <= 0) {
        throw std::runtime_error("Qwen3 ASR audio_chunk_seconds produced an empty chunk");
    }
    const auto chunks = engine::audio::plan_audio_chunks(
        frames,
        engine::audio::AudioChunkSpec{
            samples,
            samples,
            engine::audio::AudioChunkPadMode::Zero,
            engine::audio::AudioChunkTailAlignment::Start,
            0,
        });
    const int64_t context_samples = return_timestamps
        ? std::min<int64_t>(
            samples / 2,
            static_cast<int64_t>(std::llround(kTimestampFixedChunkContextSeconds * static_cast<double>(audio.sample_rate))))
        : 0;
    std::vector<AudioChunkPlan> plan;
    plan.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        const runtime::TimeSpan keep_span{
            chunk.output_start_sample,
            chunk.output_start_sample + chunk.valid_samples,
        };
        const runtime::TimeSpan source_span{
            std::max<int64_t>(0, keep_span.start_sample - context_samples),
            std::min<int64_t>(frames, keep_span.end_sample + context_samples),
        };
        plan.push_back(AudioChunkPlan{source_span, keep_span});
    }
    return plan;
}

runtime::IOfflineVoiceTaskSession & Qwen3ASRSession::vad_session() {
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Qwen3 ASR internal VAD session does not support offline execution");
        }
        session.release();
        vad_session_.reset(offline);
    }
    return *vad_session_;
}

Qwen3ASRRequest Qwen3ASRSession::make_request(const runtime::TaskRequest & request) const {
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Qwen3 ASR run() requires audio_input");
    }
    Qwen3ASRRequest out;
    out.audio = *request.audio_input;
    out.generation.max_new_tokens = assets_->config.max_new_tokens;
    if (request.text_input.has_value()) {
        out.context = request.text_input->text;
        out.language = request.text_input->language;
    }
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.generation.max_new_tokens = *value;
        if (out.generation.max_new_tokens <= 0) {
            throw std::runtime_error("Qwen3 ASR max_tokens must be positive");
        }
    }
    if (const auto value = runtime::find_option(request.options, {"return_timestamps"})) {
        out.generation.return_timestamps = runtime::parse_bool_option(*value, "return_timestamps");
    }
    return out;
}

}  // namespace engine::models::qwen3_asr
