#include "engine/community_models/fish_drama3/session.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>

namespace engine::models::fish_drama3 {
namespace {
namespace json = engine::io::json;

json::Value parse_conditioning_json(std::string_view text) {
    try {
        return json::parse(text);
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
}

codecs::FishDacCodes parse_codes(const json::Value & value, int64_t codebook_size) {
    codecs::FishDacCodes codes;
    const auto & books = value.as_array();
    codes.codebooks = static_cast<int64_t>(books.size());
    if (books.size() != 10) throw std::runtime_error("Drama 3 vq_codes needs 10 codebooks");
    codes.frames = static_cast<int64_t>(books.front().as_array().size());
    for (const auto & book : books) {
        if (static_cast<int64_t>(book.as_array().size()) != codes.frames) throw std::runtime_error("Drama 3 vq_codes rows must have equal length");
        for (const auto & code : book.as_array()) {
            const int64_t integer = code.as_i64();
            if (integer < 0 || integer >= codebook_size || code.as_number() != static_cast<double>(integer)) {
                throw std::runtime_error("Drama 3 vq_codes requires integers in the codebook range");
            }
            codes.codes.push_back(static_cast<int32_t>(integer));
        }
    }
    validate_drama3_codes(codes, codebook_size);
    return codes;
}

runtime::AudioBuffer read_reference(const std::string & path) {
    auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
    return {wav.sample_rate, wav.channels, std::move(wav.samples)};
}

json::Value take_json(const Drama3Take & take) {
    json::Value::Array books;
    for (int64_t book = 0; book < 10; ++book) {
        json::Value::Array frames;
        for (int64_t frame = 0; frame < take.codes.frames; ++frame) {
            frames.push_back(json::Value::make_number(take.codes.codes[book * take.codes.frames + frame]));
        }
        books.push_back(json::Value::make_array(std::move(frames)));
    }
    return json::Value::make_object({{"text", json::Value::make_string(take.text)},
                                    {"vq_codes", json::Value::make_array(std::move(books))}});
}

codecs::FishDacCodes join_codes(const std::vector<Drama3Take> & takes) {
    codecs::FishDacCodes merged;
    merged.codebooks = 10;
    for (const auto & take : takes) merged.frames += take.codes.frames;
    merged.codes.reserve(static_cast<size_t>(merged.frames * 10));
    for (int64_t book = 0; book < 10; ++book) {
        for (const auto & take : takes) {
            const auto begin = take.codes.codes.begin() + book * take.codes.frames;
            merged.codes.insert(merged.codes.end(), begin, begin + take.codes.frames);
        }
    }
    return merged;
}

}  // namespace

Drama3Session::Drama3Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
                             std::shared_ptr<const Drama3Assets> assets)
    : runtime::RuntimeSessionBase(options), task_(task), assets_(std::move(assets)) {
    if (!assets_ || (task.task != runtime::VoiceTaskKind::Tts && task.task != runtime::VoiceTaskKind::VoiceCloning)) {
        throw std::runtime_error("Fish Drama 3 supports TTS and voice cloning");
    }
    generator_ = std::make_unique<Drama3Generator>(assets_, execution_context(), options,
        task_.mode == runtime::RunMode::Streaming);
}
Drama3Session::~Drama3Session() = default;
std::string Drama3Session::family() const { return "fish_drama3"; }
runtime::VoiceTaskKind Drama3Session::task_kind() const { return task_.task; }
runtime::RunMode Drama3Session::run_mode() const { return task_.mode; }

void Drama3Session::prepare(const runtime::SessionPreparationRequest & request) {
    defaults_ = {};
    defaults_.text_input = request.text;
    defaults_.voice = request.voice;
    defaults_.options = request.options;
    mark_prepared();
}

Drama3Take Drama3Session::conditioning_take(const json::Value & value) {
    Drama3Take take;
    std::optional<std::string> audio_path;
    try {
        take.text = json::require_string(value, "text");
        validate_drama3_text(take.text, "conditioning text");
        if (const auto * codes = value.find("vq_codes")) {
            take.codes = parse_codes(*codes, assets_->tower.codebook_size);
        } else if (const auto * audio = value.find("audio_path")) audio_path = audio->as_string();
        else if (const auto * audio = value.find("audio")) audio_path = audio->as_string();
        else throw runtime::InvalidRequestError("Drama 3 conditioning requires vq_codes, audio_path, or audio");
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
    // Only parsing above translates client errors. I/O and codec failures retain
    // their original classification.
    if (audio_path) take.codes = generator_->encode_reference(read_reference(*audio_path));
    return take;
}

runtime::TaskResult Drama3Session::run(const runtime::TaskRequest & request) { return synthesize(request, false); }
runtime::TaskResult Drama3Session::generate_stream(const runtime::TaskRequest & request) { return synthesize(request, true); }

runtime::TaskResult Drama3Session::synthesize(const runtime::TaskRequest & request, bool streaming) {
    require_prepared("Fish Drama 3 generation");
    auto effective = defaults_;
    if (request.text_input) effective.text_input = request.text_input;
    if (request.voice) effective.voice = request.voice;
    for (const auto & option : request.options) effective.options[option.first] = option.second;
    effective.input_artifacts = request.input_artifacts;
    if (!effective.text_input) throw runtime::InvalidRequestError("Drama 3 requires input text");
    Drama3Sampling sampling;
    std::optional<uint32_t> seed;
    try {
        sampling.temperature = runtime::parse_finite_float_option(effective.options, {"temperature"}).value_or(1.0F);
        sampling.top_p = runtime::parse_finite_float_option(effective.options, {"top_p"}).value_or(0.9F);
        sampling.top_k = runtime::parse_int_option(effective.options, {"top_k"}).value_or(4097);
        sampling.max_frames = runtime::parse_positive_i64_option(effective.options, {"max_tokens", "max_new_tokens"}, 1024);
        seed = runtime::parse_u32_option(effective.options, {"seed"});
    } catch (const std::runtime_error & error) {
        throw runtime::InvalidRequestError(error.what());
    } catch (const std::logic_error & error) {
        throw runtime::InvalidRequestError(error.what());
    }
    sampling.seed = seed.value_or(runtime::random_u32_seed());
    std::vector<Drama3Take> references;
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->cached_voice_id &&
        !effective.voice->speaker->audio) {
        throw runtime::InvalidRequestError("Drama 3 requires reference audio or a reference descriptor with VQ codes");
    }
    if (effective.voice && effective.voice->speaker && effective.voice->speaker->audio) {
        const auto transcript = runtime::find_option(effective.options, {"reference_text"});
        if (!transcript) throw runtime::InvalidRequestError("Drama 3 reference audio requires reference_text");
        references.push_back({*transcript, generator_->encode_reference(*effective.voice->speaker->audio)});
    }
    if (const auto ordered = runtime::find_option(effective.options, {"multi_reference_cond", "references"})) {
        const auto descriptors = parse_conditioning_json(*ordered);
        if (!descriptors.is_array()) throw runtime::InvalidRequestError("Drama 3 references must be a JSON array");
        for (const auto & value : descriptors.as_array()) references.push_back(conditioning_take(value));
    }
    if (task_.task == runtime::VoiceTaskKind::VoiceCloning && references.empty()) {
        throw runtime::InvalidRequestError("Drama 3 cloning requires at least one reference");
    }
    if (const auto prepare = runtime::find_option(effective.options, {"encode_reference_only"});
        prepare && runtime::parse_bool_option(*prepare, "encode_reference_only")) {
        if (references.empty()) throw runtime::InvalidRequestError("Reference preparation requires reference audio or codes");
        if (task_.mode == runtime::RunMode::Streaming) generator_->warm_streaming_codec(references.front().codes);
        runtime::TaskResult result;
        for (size_t index = 0; index < references.size(); ++index) {
            result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
                "drama3_reference_" + std::to_string(index), json::stringify(take_json(references[index])),
                {{"family", "fish_drama3"}, {"schema", "drama3_reference_v1"}}));
        }
        return result;
    }
    std::vector<Drama3Take> history;
    for (const auto & artifact : effective.input_artifacts) {
        if (artifact.kind != runtime::ArtifactKind::AcousticTokens) continue;
        const auto artifact_family = artifact.meta.find("family");
        if (artifact_family != artifact.meta.end() && artifact_family->second != "fish_drama3") {
            throw runtime::InvalidRequestError("Drama 3 continuation cannot consume acoustic tokens from another model family");
        }
        const std::string payload(reinterpret_cast<const char *>(artifact.payload.data()), artifact.payload.size());
        history.push_back(conditioning_take(parse_conditioning_json(payload)));
    }
    const auto takes = runtime::find_option(effective.options, {"previous_takes"});
    const auto takes_file = runtime::find_option(effective.options, {"previous_takes_file"});
    if (takes && takes_file) throw runtime::InvalidRequestError("choose previous_takes or previous_takes_file");
    if (takes || takes_file) {
        const auto descriptors = parse_conditioning_json(takes_file ? engine::io::read_text_file(*takes_file) : *takes);
        const auto append_take = [&](const json::Value & value) {
            if (value.is_string()) {
                auto generated = generator_->generate(value.as_string(), references, history, sampling, false);
                history.push_back({value.as_string(), std::move(generated.codes)});
            } else history.push_back(conditioning_take(value));
        };
        if (descriptors.is_array()) {
            for (const auto & value : descriptors.as_array()) append_take(value);
        } else {
            append_take(descriptors);
        }
    }
    const int64_t chunk_size = runtime::parse_positive_i64_option(effective.options, {"text_chunk_size"}, 128);
    const auto mode = engine::text::parse_text_chunk_mode_override(effective.options).value_or(engine::text::TextChunkMode::TagAware);
    const auto chunks = runtime::chunk_text_request(effective, chunk_size, mode);
    runtime::TaskResult result;
    runtime::AudioBuffer merged;
    std::vector<Drama3Take> generated_takes;
    const auto started = std::chrono::steady_clock::now();
    bool limited = false;
    for (size_t index = 0; index < chunks.size(); ++index) {
        auto chunk_sampling = sampling;
        chunk_sampling.seed += static_cast<uint32_t>(index);
        auto context = history;
        if (!generated_takes.empty()) context.push_back(generated_takes.back());
        std::function<void(const runtime::AudioBuffer &)> callback;
        if (streaming && stream_event_sink()) {
            callback = [&](const runtime::AudioBuffer & audio) {
                runtime::StreamEvent event;
                event.audio_output = audio;
                stream_event_sink()(event);
            };
        }
        const auto & text = chunks[index].text_input->text;
        auto generated = generator_->generate(text, references, context, chunk_sampling, streaming, callback, index > 0);
        runtime::append_audio_buffer(merged, generated.audio);
        limited = limited || generated.reached_limit;
        generated_takes.push_back({text, std::move(generated.codes)});
    }
    Drama3Take take{effective.text_input->text, join_codes(generated_takes)};
    result.output_artifacts.push_back(runtime::make_text_artifact(runtime::ArtifactKind::AcousticTokens,
        "drama3_take", json::stringify(take_json(take)), {{"family", "fish_drama3"}, {"schema", "drama3_take_v1"}}));
    result.audio_output = std::move(merged);
    const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    engine::debug::timing_log_scalar("session.wall_ms", wall_ms);
    result.custom_schema_output = runtime::CustomSchemaOutput{"drama3_generation_v1", json::Value::make_object({
        {"frames", json::Value::make_number(take.codes.frames)},
        {"reached_limit", json::Value::make_bool(limited)},
        {"wall_ms", json::Value::make_number(wall_ms)}})};
    return result;
}

}  // namespace engine::models::fish_drama3
