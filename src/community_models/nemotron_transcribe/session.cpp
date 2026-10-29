#include "engine/community_models/nemotron_transcribe/session.h"

#include "engine/community_models/nemotron_transcribe/assets.h"
#include "engine/community_models/nemotron_transcribe/tokenizer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace engine::models::nemotron_transcribe {
namespace {
namespace json = engine::io::json;
constexpr const char * kFamily = "nemotron_transcribe";
constexpr const char * kTranscriptionPrompt =
    "Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, "
    "measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate.";
constexpr const char * kSpeakerPrompt =
    "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
    "Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as "
    "<spk:0>, <spk:1>, <spk:2>, etc.; assign one stable tag per speaker and keep speaker identities "
    "consistent. Output only the speaker-tagged transcript.";

bool flag(const std::unordered_map<std::string, std::string> & options, const char * key) {
    const auto value = runtime::find_option(options, {key});
    return value && runtime::parse_bool_option(*value, key);
}

int bounded_option(const runtime::SessionOptions & options, const char * key, int fallback, int minimum, int maximum) {
    const auto value = runtime::parse_i64_option(options.options, {key}).value_or(fallback);
    if (value < minimum || value > maximum)
        throw std::runtime_error(std::string(key) + " is outside the supported range");
    return static_cast<int>(value);
}

std::string language_name(const std::string & value) {
    static const std::map<std::string, std::string> names = {
        {"en", "English"}, {"zh", "Chinese"}, {"ja", "Japanese"}, {"es", "Spanish"},
        {"fr", "French"}, {"de", "German"}, {"it", "Italian"}, {"pt", "Portuguese"},
        {"ko", "Korean"}, {"ru", "Russian"}, {"ar", "Arabic"}, {"hi", "Hindi"},
        {"nl", "Dutch"}, {"pl", "Polish"}, {"uk", "Ukrainian"}, {"vi", "Vietnamese"},
        {"th", "Thai"}, {"id", "Indonesian"}, {"tr", "Turkish"}, {"fa", "Persian"}};
    const auto found = names.find(value);
    return found == names.end() ? value : found->second;
}

runtime::TimeSpan span(double start, double end, const runtime::AudioBuffer & audio) {
    const auto count = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    return {std::clamp<int64_t>(std::llround(start * audio.sample_rate), 0, count),
            std::clamp<int64_t>(std::llround(end * audio.sample_rate), 0, count)};
}

const char * termination_name(TerminationReason reason) {
    switch (reason) {
        case TerminationReason::StopToken: return "end_of_turn";
        case TerminationReason::TokenLimit: return "token_limit";
        case TerminationReason::Cancelled: return "cancelled";
    }
    return "unknown";
}

std::string trim_response(const std::string & text) {
    const auto first = text.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) return {};
    const auto last = text.find_last_not_of(" \t\r\n\f\v");
    return text.substr(first, last - first + 1);
}

class TranscribeSession final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession {
public:
    TranscribeSession(runtime::TaskSpec task, runtime::SessionOptions options,
        std::shared_ptr<const TranscribeAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          tokenizer_(assets_->resources.require_file("tokenizer_json"), assets_->resources.require_file("tokenizer_config")),
          frontend_(assets_->frontend, assets_->mel_filterbank, assets_->frontend_window) {
        runtime::validate_spec_backed_session_options(options, *contract_, kFamily, "Nemotron Transcribe");
        if (task.task != runtime::VoiceTaskKind::Asr || task.mode != runtime::RunMode::Offline)
            throw std::runtime_error("Nemotron Transcribe supports offline audio understanding through the ASR task");
        auto encoder_config = assets_->encoder;
        encoder_config.online_inference_length = bounded_option(options, "nemotron_transcribe.encoder_window_size",
            encoder_config.online_inference_length, 1, 5000);
        encoder_config.chunk_left_context = bounded_option(options, "nemotron_transcribe.encoder_left_context",
            encoder_config.chunk_left_context, 0, 5000);
        encoder_config.chunk_right_context = bounded_option(options, "nemotron_transcribe.encoder_right_context",
            encoder_config.chunk_right_context, 0, 5000);
        auto & cache = encoder_config.speaker_cache;
        cache.cache_frames = bounded_option(options, "nemotron_transcribe.speaker_cache_size", cache.cache_frames, 8, 5000);
        cache.fifo_frames = bounded_option(options, "nemotron_transcribe.speaker_fifo_size", cache.fifo_frames, 0, 5000);
        cache.update_period = bounded_option(options, "nemotron_transcribe.speaker_update_period", cache.update_period, 1, 5000);
        if (const auto value = runtime::find_option(options.options, {"nemotron_transcribe.flash_attention"}))
            encoder_config.use_flash_attention = runtime::parse_bool_option(*value, "nemotron_transcribe.flash_attention");
        ctc_options_.head_window = encoder_config.online_inference_length;
        ctc_options_.head_left_context = encoder_config.chunk_left_context;
        ctc_options_.head_right_context = encoder_config.chunk_right_context;
        ctc_options_.special_tokens = tokenizer_.alignment_special_tokens();
        auto decoder_config = assets_->decoder;
        decoder_config.max_context = runtime::parse_positive_i64_option(options.options,
            {"nemotron_transcribe.max_context"}, decoder_config.max_context);
        if (decoder_config.max_context > assets_->maximum_context)
            throw std::runtime_error("Nemotron Transcribe max_context exceeds the backbone context limit");
        max_context_ = decoder_config.max_context;
        decoder_config.prefill_batch_size = bounded_option(options, "nemotron_transcribe.prefill_batch_size",
            static_cast<int>(decoder_config.prefill_batch_size), 1, 2048);
        const auto storage = assets::parse_tensor_storage_type(
            runtime::find_option(options.options, {"nemotron_transcribe.weight_type"}).value_or("native"));
        encoder_ = std::make_unique<ParallelExpertEncoder>(execution_context(), *assets_->encoder_weights, encoder_config, storage);
        decoder_ = std::make_unique<NemotronHDecoderRuntime>(assets_->decoder_weights, execution_context(), decoder_config, storage);
        ctc_ = std::make_unique<CtcTimestampHead>(assets_->encoder_weights, execution_context(),
            assets_->ctc, assets_->alignment_tokenizer);
        assets_->decoder_weights->release_storage();
        assets_->encoder_weights->release_storage();
    }

    std::string family() const override { return kFamily; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        if (!request.audio || request.audio->sample_rate != assets_->frontend.sample_rate || request.audio->channels != 1)
            throw runtime::InvalidRequestError("Nemotron Transcribe requires 16 kHz mono audio");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("Nemotron Transcribe run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "Nemotron Transcribe");
        if (!request.audio_input || request.audio_input->sample_rate != assets_->frontend.sample_rate ||
            request.audio_input->channels != 1 || request.audio_input->samples.empty())
            throw runtime::InvalidRequestError("Nemotron Transcribe requires nonempty 16 kHz mono audio");
        const auto & audio = *request.audio_input;
        if (static_cast<double>(audio.samples.size()) / audio.sample_rate > assets_->encoder.maximum_audio_seconds)
            throw runtime::InvalidRequestError("Nemotron Transcribe audio exceeds one hour");
        const auto mode = runtime::find_option(request.options, {"template_name"}).value_or("default");
        if (mode != "default" && mode != "translation" && mode != "audio_qa" && mode != "language_id")
            throw runtime::InvalidRequestError("unknown Nemotron Transcribe template_name");
        const bool diarization = flag(request.options, "diarization");
        const bool word_times = flag(request.options, "return_timestamps") || diarization;
        if (word_times && mode != "default")
            throw runtime::InvalidRequestError("CTC timestamps and speaker labels require a source-language transcript");
        auto instruction = runtime::find_option(request.options, {"instruct"}).value_or(
            request.text_input ? request.text_input->text : "");
        const auto language = runtime::find_option(request.options, {"language"}).value_or("auto");
        if (instruction.empty()) {
            if (mode == "audio_qa") throw runtime::InvalidRequestError("audio_qa requires instruct or text_input");
            if (mode == "translation") {
                const auto target = runtime::find_option(request.options, {"target_language"}).value_or("English");
                if (target.empty() || target == "auto") throw runtime::InvalidRequestError("translation requires a target language");
                instruction = "Translate the audio into " + language_name(target) + ".";
            } else if (mode == "language_id") {
                instruction = "Identify the language spoken in this audio. Output only the language name.";
            } else {
                instruction = diarization ? kSpeakerPrompt : kTranscriptionPrompt;
                if (!language.empty() && language != "auto") instruction += " The speech is in " + language_name(language) + ".";
            }
        }
        const auto system = runtime::find_option(request.options, {"system_prompt"}).value_or("");
        const auto prompt = tokenizer_.build_prompt(instruction, system);
        GenerationOptions generation;
        const auto duration_budget = std::max<int64_t>(24,
            static_cast<int64_t>(std::ceil(6.0 * audio.samples.size() / audio.sample_rate - 1e-9)) + 24);
        generation.max_new_tokens = runtime::parse_positive_i64_option(request.options, {"max_tokens"},
            assets_->max_new_tokens > 0 ? assets_->max_new_tokens : duration_budget);
        generation.do_sample = flag(request.options, "do_sample");
        generation.temperature = runtime::parse_finite_float_option(request.options, {"temperature"}).value_or(1.0F);
        generation.top_k = runtime::parse_i64_option(request.options, {"top_k"}).value_or(0);
        generation.top_p = runtime::parse_finite_float_option(request.options, {"top_p"}).value_or(1.0F);
        generation.repetition_penalty = runtime::parse_positive_finite_float_option(request.options, {"repetition_penalty"}).value_or(1.0F);
        generation.seed = runtime::parse_u64_option(request.options, {"seed"}).value_or(0);
        generation.stop_token_ids = {TextTokenizer::end_of_turn_id};
        if (is_near_silent(audio.samples)) {
            runtime::TaskResult empty;
            empty.text_output = runtime::Transcript{"", ""};
            json::Value::Object detail{{"termination", json::Value::make_string("near_silent")},
                {"generated_tokens", json::Value::make_number(0)}};
            if (word_times) {
                detail.emplace("alignment_status", json::Value::make_string("unavailable"));
                detail.emplace("alignment_failure_reason", json::Value::make_string("empty_transcript"));
            }
            empty.custom_schema_output = runtime::CustomSchemaOutput{
                "nemotron_transcribe.result.v1", json::Value::make_object(std::move(detail))};
            return empty;
        }
        const auto start = std::chrono::steady_clock::now();
        const auto features = frontend_.extract(audio, execution_context().config().threads);
        const auto audio_frames = (features.frames + assets_->encoder.asr.subsampling_factor - 1) /
            assets_->encoder.asr.subsampling_factor;
        const auto prompt_tokens = static_cast<int64_t>(prompt.before_audio.size() + prompt.after_audio.size()) + audio_frames;
        if (prompt_tokens >= max_context_ || generation.max_new_tokens > max_context_ - prompt_tokens)
            throw runtime::CapacityError("Nemotron Transcribe prompt, audio and output budget exceed max_context");
        const auto frontend_ms = debug::elapsed_ms(start);
        auto stage = std::chrono::steady_clock::now();
        const auto encoded = encoder_->encode(features);
        const auto encoder_ms = debug::elapsed_ms(stage);
        stage = std::chrono::steady_clock::now();
        const auto generated = decoder_->generate(prompt.before_audio, encoded.audio_embeddings,
            encoded.frames, prompt.after_audio, generation);
        const auto decoder_ms = debug::elapsed_ms(stage);
        const auto raw_transcript = tokenizer_.decode(generated.token_ids);
        const auto transcript = trim_response(raw_transcript);
        runtime::TaskResult result;
        result.text_output = runtime::Transcript{transcript, ""};
        json::Value::Object report{
            {"template_name", json::Value::make_string(mode)},
            {"termination", json::Value::make_string(termination_name(generated.termination))},
            {"stop_token_id", json::Value::make_number(generated.stop_token_id)},
            {"prompt_tokens", json::Value::make_number(static_cast<double>(generated.prompt_tokens))},
            {"generated_tokens", json::Value::make_number(static_cast<double>(generated.token_ids.size()))},
            {"audio_frames", json::Value::make_number(static_cast<double>(encoded.frames))},
            {"frontend_ms", json::Value::make_number(frontend_ms)},
            {"encoder_ms", json::Value::make_number(encoder_ms)},
            {"decoder_ms", json::Value::make_number(decoder_ms)}};
        if (word_times) {
            stage = std::chrono::steady_clock::now();
            const CtcSpeakerEvidence speakers{encoded.speaker_probs80.data(), static_cast<int>(encoded.frames),
                encoded.speaker_count, encoded.speaker_probs10.data(), static_cast<int>(encoded.native_speaker_frames),
                encoded.duration_seconds, 0.0};
            CtcTimestampResult aligned;
            try {
                const auto evidence = ctc_->prepare(encoded.asr_states.data(), static_cast<int>(encoded.frames), ctc_options_);
                aligned = ctc_->timestamps(evidence, raw_transcript, speakers, ctc_options_);
            } catch (const std::exception & error) {
                // A timestamp failure must not discard a successful transcript.
                aligned.failure_reason = "ctc_head_error";
                aligned.error = error.what();
                aligned.alignment.diarization = timestamps::diarization_segments(encoded.speaker_probs10.data(),
                    static_cast<int>(encoded.native_speaker_frames), encoded.speaker_count,
                    static_cast<double>(assets_->frontend.hop_length) / assets_->frontend.sample_rate,
                    encoded.duration_seconds);
            }
            report.emplace("alignment_status", json::Value::make_string(aligned.status));
            report.emplace("alignment_failure_reason", json::Value::make_string(aligned.failure_reason));
            report.emplace("alignment_error", json::Value::make_string(aligned.error));
            report.emplace("alignment_ms", json::Value::make_number(debug::elapsed_ms(stage)));
            json::Value::Array words;
            for (const auto & row : aligned.alignment.words) {
                const auto & word = aligned.targets.words.at(static_cast<size_t>(row.word));
                const std::string label = row.column < 0 ? "unknown" : "speaker_" + std::to_string(row.column);
                const std::string display = word.display.empty() ? word.word : word.display;
                result.word_timestamps.push_back({span(row.start, row.end, audio), word.word, 0.0F});
                words.push_back(json::Value::make_object({
                    {"word", json::Value::make_string(word.word)}, {"display", json::Value::make_string(display)},
                    {"joins_previous", json::Value::make_bool(word.joins_previous)}, {"start", json::Value::make_number(row.start)},
                    {"end", json::Value::make_number(row.end)}, {"speaker_id", json::Value::make_string(label)},
                    {"transcript_speaker_tag", json::Value::make_string(timestamps::speaker_label(row.tag))},
                    {"speaker_column", json::Value::make_number(row.column)},
                    {"speaker_confidence", row.speaker_confidence < 0 ? json::Value::make_null() :
                        json::Value::make_number(row.speaker_confidence)}}));
            }
            report.emplace("words", json::Value::make_array(std::move(words)));
            json::Value::Array failed;
            for (const auto & stream : aligned.alignment.failed_streams)
                failed.push_back(json::Value::make_object({{"tag", json::Value::make_number(stream.tag)},
                    {"reason", json::Value::make_string(stream.reason)}}));
            report.emplace("failed_streams", json::Value::make_array(std::move(failed)));
            if (diarization) {
                for (const auto & segment : aligned.alignment.diarization)
                    result.speaker_turns.push_back({span(segment.start, segment.end, audio),
                        "speaker_" + std::to_string(segment.speaker), 0.0F, ""});
            }
        }
        report.emplace("wall_ms", json::Value::make_number(debug::elapsed_ms(start)));
        result.custom_schema_output = runtime::CustomSchemaOutput{
            "nemotron_transcribe.result.v1", json::Value::make_object(std::move(report))};
        return result;
    }
private:
    runtime::TaskSpec task_;
    std::shared_ptr<const TranscribeAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    TextTokenizer tokenizer_;
    Frontend frontend_;
    CtcOptions ctc_options_;
    int64_t max_context_ = 0;
    std::unique_ptr<ParallelExpertEncoder> encoder_;
    std::unique_ptr<NemotronHDecoderRuntime> decoder_;
    std::unique_ptr<CtcTimestampHead> ctc_;
};
}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_nemotron_transcribe_loader() {
    runtime::SpecBackedVoiceModelConfig<TranscribeAssets> config;
    config.family = kFamily;
    config.load_assets = load_transcribe_assets;
    config.model_resources = [](const std::filesystem::path & path, const auto &) {
        const auto assets = load_transcribe_assets(path);
        runtime::ModelResourcePlan plan;
        // The host already prices the primary decoder. Account separately for
        // its external speech/CTC sibling, using the same native-storage
        // overhead factor as the host's ordinary ggml weight estimate.
        const auto & file = assets->encoder_weights->source_path();
        const auto bytes = static_cast<size_t>(std::filesystem::file_size(file));
        plan.resident_bytes = bytes + bytes / 2;
        plan.engines.push_back({file, plan.resident_bytes});
        return plan;
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
        std::shared_ptr<const TranscribeAssets> assets, std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<TranscribeSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}
}  // namespace engine::models::nemotron_transcribe
