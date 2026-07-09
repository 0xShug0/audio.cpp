#include "runtime.h"

#include "multipart.h"

#include "../cli/request.h"
#include "../streaming/streaming.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace minitts::server {
namespace {

using engine::io::json::Value;

using Clock = std::chrono::steady_clock;

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

std::filesystem::path resolve_path(const std::filesystem::path & base, const std::filesystem::path & path) {
    return path.is_absolute() ? path : base / path;
}

// Minimal application/x-www-form-urlencoded query string lookup, e.g.
// query_param("model=pocket-tts&foo=bar", "model") -> "pocket-tts".
std::string query_param(const std::string & query, const std::string & key) {
    size_t pos = 0;
    while (pos < query.size()) {
        const size_t amp = query.find('&', pos);
        const std::string pair = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        const auto eq = pair.find('=');
        const std::string name = pair.substr(0, eq);
        if (name == key) {
            return eq == std::string::npos ? "" : pair.substr(eq + 1);
        }
        if (amp == std::string::npos) {
            break;
        }
        pos = amp + 1;
    }
    return {};
}

const char * backend_name(engine::core::BackendType type) {
    switch (type) {
        case engine::core::BackendType::Cpu:
            return "cpu";
        case engine::core::BackendType::Cuda:
            return "cuda";
        case engine::core::BackendType::Vulkan:
            return "vulkan";
        case engine::core::BackendType::Metal:
            return "metal";
        case engine::core::BackendType::BestAvailable:
            return "best";
    }
    return "unknown";
}

std::unordered_map<std::string, std::string> options_from_object(const Value * value) {
    return minitts::cli::json_options_map(value);
}

void add_option_from_json(
    std::unordered_map<std::string, std::string> & options,
    const Value & object,
    const std::string & field,
    const std::string & option_key) {
    const auto * value = object.find(field);
    if (value != nullptr && !value->is_null()) {
        options[option_key] = minitts::cli::json_option_string(*value);
    }
}

std::vector<uint8_t> encode_pcm16_wav(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    const uint16_t channels = static_cast<uint16_t>(audio.channels);
    const uint16_t bits_per_sample = 16;
    const uint32_t data_bytes = static_cast<uint32_t>(audio.samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_bytes;
    const uint32_t byte_rate = static_cast<uint32_t>(audio.sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;

    std::vector<uint8_t> out;
    out.reserve(44 + data_bytes);
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    auto append_u16 = [&](uint16_t value) { append_bytes(&value, sizeof(value)); };
    auto append_u32 = [&](uint32_t value) { append_bytes(&value, sizeof(value)); };

    out.insert(out.end(), {'R', 'I', 'F', 'F'});
    append_u32(riff_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(16);
    append_u16(1);
    append_u16(channels);
    append_u32(static_cast<uint32_t>(audio.sample_rate));
    append_u32(byte_rate);
    append_u16(block_align);
    append_u16(bits_per_sample);
    out.insert(out.end(), {'d', 'a', 't', 'a'});
    append_u32(data_bytes);
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::vector<uint8_t> encode_pcm16_samples(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("audio output sample rate must be positive");
    }
    if (audio.channels <= 0) {
        throw std::runtime_error("audio output channel count must be positive");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("audio output sample count must be divisible by channel count");
    }

    std::vector<uint8_t> out;
    out.reserve(audio.samples.size() * sizeof(int16_t));
    auto append_bytes = [&](const void * data, size_t size) {
        const auto * bytes = static_cast<const uint8_t *>(data);
        out.insert(out.end(), bytes, bytes + size);
    };
    for (float sample : audio.samples) {
        sample = std::max(-1.0F, std::min(1.0F, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(sample * 32767.0F));
        append_bytes(&pcm, sizeof(pcm));
    }
    return out;
}

std::string base64_encode(const uint8_t * data, size_t size) {
    constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((size + 2) / 3) * 4);
    for (size_t i = 0; i < size; i += 3) {
        const uint32_t b0 = data[i];
        const uint32_t b1 = i + 1 < size ? data[i + 1] : 0;
        const uint32_t b2 = i + 2 < size ? data[i + 2] : 0;
        const uint32_t chunk = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(chunk >> 18) & 0x3f]);
        out.push_back(kAlphabet[(chunk >> 12) & 0x3f]);
        out.push_back(i + 1 < size ? kAlphabet[(chunk >> 6) & 0x3f] : '=');
        out.push_back(i + 2 < size ? kAlphabet[chunk & 0x3f] : '=');
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t> & bytes) {
    return base64_encode(bytes.data(), bytes.size());
}

void write_sse(HttpStreamWriter & writer, const std::string & json) {
    writer.write("data: " + json + "\n\n");
}

void write_sse_done(HttpStreamWriter & writer) {
    writer.write("data: [DONE]\n\n");
}

bool bool_field(const Value & object, const std::string & key, bool default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    if (value->is_bool()) {
        return value->as_bool();
    }
    if (value->is_string()) {
        const auto str = value->as_string();
        if (str == "true" || str == "1") {
            return true;
        }
        if (str == "false" || str == "0") {
            return false;
        }
    }
    throw std::runtime_error(key + " must be a boolean");
}

HttpResponse sse_response(std::function<void(HttpStreamWriter &)> stream) {
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/event-stream; charset=utf-8";
    response.headers.emplace("X-Accel-Buffering", "no");
    response.stream_body = std::move(stream);
    return response;
}

HttpResponse chunked_audio_response(std::function<void(HttpStreamWriter &)> stream) {
    HttpResponse response;
    response.status = 200;
    response.content_type = "application/octet-stream";
    response.stream_body = std::move(stream);
    return response;
}

// Multipart file uploads arrive as in-memory bytes, but the WAV decoder only reads from disk,
// so uploaded audio is spooled to a uniquely named temp file before decoding.
std::filesystem::path write_temp_upload(const std::string & filename, const std::string & data) {
    std::filesystem::path ext = std::filesystem::path(filename).extension();
    if (ext.empty()) {
        ext = ".wav";
    }
    static std::atomic<uint64_t> counter{0};
    std::ostringstream name;
    name << "audiocpp_upload_" << Clock::now().time_since_epoch().count() << "_" << counter.fetch_add(1)
         << ext.string();
    const auto path = std::filesystem::temp_directory_path() / name.str();
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to create temp file for upload: " + path.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) {
        throw std::runtime_error("failed to write temp file for upload: " + path.string());
    }
    return path;
}

struct TempFileGuard {
    std::filesystem::path path;
    ~TempFileGuard() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

double elapsed_ms(Clock::time_point started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

double audio_duration_ms(const engine::runtime::AudioBuffer & audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return 1000.0 * static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

double audio_rtf(double wall_ms, double duration_ms) {
    return duration_ms > 0.0 ? wall_ms / duration_ms : 0.0;
}

std::string timing_json(double wall_ms) {
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms << "}";
    return out.str();
}

std::string timing_json(double wall_ms, const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    std::ostringstream out;
    out << "{\"wall_ms\":" << wall_ms
        << ",\"audio_duration_ms\":" << duration_ms
        << ",\"rtf\":" << audio_rtf(wall_ms, duration_ms) << "}";
    return out.str();
}

std::string ttft_timing_json(double ttft_ms) {
    std::ostringstream out;
    out << "{\"ttft_ms\":" << ttft_ms << "}";
    return out.str();
}

bool stream_event_has_output(const engine::runtime::StreamEvent & event) {
    return (event.partial_text.has_value() && !event.partial_text->text.empty()) ||
        event.audio_output.has_value() ||
        !event.named_audio_outputs.empty();
}

bool task_result_has_output(const engine::runtime::TaskResult & result) {
    return result.text_output.has_value() ||
        result.audio_output.has_value() ||
        !result.named_audio_outputs.empty();
}

double require_ttft_ms(const std::optional<double> & ttft_ms) {
    if (!ttft_ms.has_value()) {
        throw std::runtime_error("streaming response produced no TTFT event");
    }
    return *ttft_ms;
}

std::unordered_map<std::string, std::string> timing_headers(
    double wall_ms,
    const engine::runtime::AudioBuffer & audio) {
    const double duration_ms = audio_duration_ms(audio);
    return {
        {"X-AudioCPP-Wall-Ms", std::to_string(wall_ms)},
        {"X-AudioCPP-Audio-Duration-Ms", std::to_string(duration_ms)},
        {"X-AudioCPP-RTF", std::to_string(audio_rtf(wall_ms, duration_ms))},
    };
}

std::string task_result_json_with_timing(
    const engine::runtime::TaskResult & result,
    const std::string & timing) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const std::string & name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << json_quote(name) << ":";
    };

    if (result.text_output.has_value()) {
        field("text");
        out << json_quote(result.text_output->text);
        if (!result.text_output->language.empty()) {
            field("language");
            out << json_quote(result.text_output->language);
        }
    }
    if (result.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*result.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
        field("sample_rate");
        out << result.audio_output->sample_rate;
        field("channels");
        out << result.audio_output->channels;
    }
    if (!result.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < result.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(result.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(result.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"sample_rate\":" << result.named_audio_outputs[i].audio.sample_rate
                << ",\"channels\":" << result.named_audio_outputs[i].audio.channels
                << "}";
        }
        out << "]";
    }
    if (!result.speech_segments.empty()) {
        field("segments");
        out << "[";
        for (size_t i = 0; i < result.speech_segments.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & segment = result.speech_segments[i];
            out << "{\"start_sample\":" << segment.span.start_sample
                << ",\"end_sample\":" << segment.span.end_sample
                << ",\"confidence\":" << segment.confidence << "}";
        }
        out << "]";
    }
    if (!result.speaker_turns.empty()) {
        field("speaker_turns");
        out << "[";
        for (size_t i = 0; i < result.speaker_turns.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & turn = result.speaker_turns[i];
            out << "{\"start_sample\":" << turn.span.start_sample
                << ",\"end_sample\":" << turn.span.end_sample
                << ",\"speaker_id\":" << json_quote(turn.speaker_id)
                << ",\"confidence\":" << turn.confidence << "}";
        }
        out << "]";
    }
    if (!result.word_timestamps.empty()) {
        field("words");
        out << "[";
        for (size_t i = 0; i < result.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto & word = result.word_timestamps[i];
            out << "{\"word\":" << json_quote(word.word)
                << ",\"start_sample\":" << word.span.start_sample
                << ",\"end_sample\":" << word.span.end_sample
                << ",\"confidence\":" << word.confidence << "}";
        }
        out << "]";
    }
    field("timing");
    out << timing;
    out << "}";
    return out.str();
}

std::string task_result_json(const engine::runtime::TaskResult & result, double wall_ms) {
    if (result.audio_output.has_value()) {
        return task_result_json_with_timing(result, timing_json(wall_ms, *result.audio_output));
    }
    if (result.named_audio_outputs.size() == 1) {
        return task_result_json_with_timing(result, timing_json(wall_ms, result.named_audio_outputs.front().audio));
    }
    return task_result_json_with_timing(result, timing_json(wall_ms));
}

std::string streaming_task_result_json(
    const engine::runtime::TaskResult & result,
    const std::optional<double> & ttft_ms) {
    return task_result_json_with_timing(result, ttft_timing_json(require_ttft_ms(ttft_ms)));
}

std::string stream_event_json(const engine::runtime::StreamEvent & event) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    auto field = [&](const char * name) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << name << "\":";
    };
    if (event.partial_text.has_value()) {
        field("partial_text");
        out << "{\"text\":" << json_quote(event.partial_text->text)
            << ",\"language\":" << json_quote(event.partial_text->language)
            << "}";
    }
    if (event.audio_output.has_value()) {
        const auto wav = encode_pcm16_wav(*event.audio_output);
        field("audio");
        out << json_quote(base64_encode(wav));
    }
    if (!event.named_audio_outputs.empty()) {
        field("named_audio_outputs");
        out << "[";
        for (size_t i = 0; i < event.named_audio_outputs.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            const auto wav = encode_pcm16_wav(event.named_audio_outputs[i].audio);
            out << "{\"id\":" << json_quote(event.named_audio_outputs[i].id)
                << ",\"audio\":" << json_quote(base64_encode(wav))
                << ",\"format\":\"wav\"}";
        }
        out << "]";
    }
    if (!event.word_timestamps.empty()) {
        field("word_timestamps");
        out << "[";
        for (size_t i = 0; i < event.word_timestamps.size(); ++i) {
            if (i != 0) {
                out << ",";
            }
            out << "{\"start_sample\":" << event.word_timestamps[i].span.start_sample
                << ",\"end_sample\":" << event.word_timestamps[i].span.end_sample
                << ",\"word\":" << json_quote(event.word_timestamps[i].word)
                << ",\"confidence\":" << event.word_timestamps[i].confidence
                << "}";
        }
        out << "]";
    }
    field("is_final");
    out << (event.is_final ? "true" : "false");
    out << "}";
    return out.str();
}

const engine::runtime::AudioBuffer & select_audio_output(const engine::runtime::TaskResult & result) {
    if (result.audio_output.has_value()) {
        return *result.audio_output;
    }
    if (result.named_audio_outputs.size() == 1) {
        return result.named_audio_outputs.front().audio;
    }
    throw std::runtime_error("model result did not contain exactly one audio output");
}

engine::runtime::TaskRequest build_openai_transcription_request(const Value & body, const std::filesystem::path & base_dir) {
    const auto * audio = body.find("audio");
    if (audio == nullptr) {
        audio = body.find("audio_path");
    }
    if (audio == nullptr) {
        audio = body.find("file");
    }
    if (audio == nullptr || !audio->is_string()) {
        throw std::runtime_error("transcription request requires audio, audio_path, or file path");
    }

    engine::runtime::TaskRequest request;
    request.audio_input = minitts::cli::read_audio_buffer(resolve_path(base_dir, audio->as_string()));
    request.options = options_from_object(body.find("options"));
    if (const auto * value = body.find("language")) {
        request.options["language"] = value->as_string();
    }
    return request;
}

}  // namespace

ServerState::ServerState(ServerConfig config, std::filesystem::path request_base)
    : config_(std::move(config)),
      request_base_(std::move(request_base)) {
    if (config_.backend != engine::core::BackendType::Cuda) {
        std::cerr
            << "audio.cpp is optimized for CUDA. The "
            << backend_name(config_.backend)
            << " server backend is intended for portability and testing, but performance and model coverage may be lower than CUDA.\n";
    }
    load_models();
}

HttpResponse ServerState::handle(const HttpRequest & request) {
    if (request.method == "GET" && request.path == "/health") {
        return json_response(
            "{\"status\":\"ok\",\"backend\":\"" +
            std::string(backend_name(config_.backend)) +
            "\",\"models\":" +
            std::to_string(models_.size()) +
            "}");
    }
    if (request.method == "GET" && request.path == "/v1/models") {
        return json_response(models_json());
    }
    if (request.method == "GET" && request.path == "/v1/audio/voices") {
        return handle_voices(request);
    }
    if (request.method == "POST" && request.path == "/v1/audio/speech") {
        return handle_speech(request.body);
    }
    if (request.method == "POST" && request.path == "/v1/audio/transcriptions") {
        return handle_transcription(request);
    }
    if (request.method == "POST" && request.path == "/v1/tasks/run") {
        return handle_generic_run(request.body);
    }
    if (request.method == "POST" && request.path == "/v1/tasks/stream") {
        return handle_generic_stream(request.body);
    }
    return error_response(404, "unknown endpoint: " + request.path, "not_found");
}

void ServerState::load_models() {
    for (auto & config : config_.models) {
        auto loaded = std::make_unique<LoadedModel>();
        loaded->config = std::move(config);
        loaded->task = engine::runtime::TaskSpec{
            engine::runtime::parse_voice_task_kind(loaded->config.task),
            engine::runtime::parse_run_mode(loaded->config.mode),
        };
        if (!model_index_.emplace(loaded->config.id, models_.size()).second) {
            throw std::runtime_error("duplicate server model id: " + loaded->config.id);
        }
        load_voice_presets(*loaded);
        if (!loaded->config.lazy) {
            ensure_model_loaded_locked(*loaded);
        }
        models_.push_back(std::move(loaded));
    }
}

ServerState::LoadedModel::RuntimeVoicePreset ServerState::load_runtime_voice_preset(
    const ServerModelConfig::VoicePreset & preset) const {
    LoadedModel::RuntimeVoicePreset out;
    out.voice_id = preset.voice_id;
    out.reference_text = preset.reference_text;
    if (preset.voice_ref.has_value()) {
        out.audio = minitts::cli::read_audio_buffer(*preset.voice_ref);
    }
    return out;
}

void ServerState::load_voice_presets(LoadedModel & model) const {
    for (const auto & [name, preset] : model.config.voice_presets) {
        auto [it, inserted] = model.voice_presets.emplace(name, load_runtime_voice_preset(preset));
        if (!inserted) {
            throw std::runtime_error("duplicate runtime voice preset for model " + model.config.id + ": " + name);
        }
        (void) it;
    }
    if (model.config.default_voice_preset_id.has_value()) {
        const auto it = model.voice_presets.find(*model.config.default_voice_preset_id);
        if (it == model.voice_presets.end()) {
            throw std::runtime_error(
                "default_voice_preset for model " + model.config.id +
                " was not loaded: " +
                *model.config.default_voice_preset_id);
        }
        model.default_voice_preset = it->second;
    } else if (model.config.default_voice_preset.has_value()) {
        model.default_voice_preset = load_runtime_voice_preset(*model.config.default_voice_preset);
    }
}

void ServerState::ensure_model_loaded_locked(LoadedModel & model) {
    if (model.session != nullptr) {
        return;
    }
    auto registry = engine::runtime::make_default_registry();

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model.config.path;
    load_request.family_hint = model.config.family;
    load_request.config_id = model.config.config_id;
    load_request.weight_id = model.config.weight_id;
    load_request.options = model.config.load_options;

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = config_.backend;
    session_options.backend.device = config_.device;
    session_options.backend.threads = config_.threads;
    session_options.options = model.config.session_options;

    auto loaded_model = registry.load(load_request);
    auto session = loaded_model->create_task_session(model.task, session_options);
    auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
    auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
    if (model.task.mode == engine::runtime::RunMode::Offline && offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    if (model.task.mode == engine::runtime::RunMode::Streaming && streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    model.model = std::move(loaded_model);
    model.session = std::move(session);
    model.offline = offline;
    model.streaming = streaming;
}

ServerState::LoadedModel & ServerState::require_model(const Value & body) {
    const std::string id = engine::io::json::require_string(body, "model");
    const auto it = model_index_.find(id);
    if (it == model_index_.end()) {
        throw std::runtime_error("unknown model id: " + id);
    }
    return *models_.at(it->second);
}

const ServerState::LoadedModel::RuntimeVoicePreset * ServerState::select_voice_preset(
    const LoadedModel & model,
    const Value & body,
    bool & voice_field_is_preset) const {
    voice_field_is_preset = false;
    if (const auto * value = body.find("voice")) {
        const auto it = model.voice_presets.find(value->as_string());
        if (it != model.voice_presets.end()) {
            voice_field_is_preset = true;
            return &it->second;
        }
        return nullptr;
    }
    if (body.find("voice_ref") != nullptr) {
        return nullptr;
    }
    return model.default_voice_preset.has_value() ? &*model.default_voice_preset : nullptr;
}

engine::runtime::TaskRequest ServerState::build_speech_request(const LoadedModel & model, const Value & body) const {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{
        engine::io::json::require_string(body, "input"),
        engine::io::json::optional_string(body, "language", ""),
    };

    request.options = options_from_object(body.find("options"));
    add_option_from_json(request.options, body, "seed", "seed");
    add_option_from_json(request.options, body, "temperature", "temperature");
    add_option_from_json(request.options, body, "top_k", "top_k");
    add_option_from_json(request.options, body, "top_p", "top_p");
    add_option_from_json(request.options, body, "max_tokens", "max_tokens");
    add_option_from_json(request.options, body, "max_steps", "max_steps");
    add_option_from_json(request.options, body, "repetition_penalty", "repetition_penalty");
    add_option_from_json(request.options, body, "guidance_scale", "guidance_scale");
    add_option_from_json(request.options, body, "num_inference_steps", "num_inference_steps");
    if (const auto * value = body.find("instructions")) {
        request.options["instruct"] = value->as_string();
    }

    bool voice_field_is_preset = false;
    const auto * preset = select_voice_preset(model, body, voice_field_is_preset);

    engine::runtime::VoiceCondition voice;
    bool has_voice = false;
    if (preset != nullptr) {
        if (preset->voice_id.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
            voice.speaker->cached_voice_id = *preset->voice_id;
            has_voice = true;
        }
        if (preset->audio.has_value()) {
            if (!voice.speaker.has_value()) {
                voice.speaker = engine::runtime::VoiceReference{};
            }
            voice.speaker->audio = *preset->audio;
            has_voice = true;
        }
        if (preset->reference_text.has_value() && request.options.find("reference_text") == request.options.end()) {
            request.options["reference_text"] = *preset->reference_text;
        }
    }
    if (const auto * value = body.find("voice"); value != nullptr && !voice_field_is_preset) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        voice.speaker->cached_voice_id = value->as_string();
        has_voice = true;
    }
    if (const auto * value = body.find("voice_ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        voice.speaker->audio = minitts::cli::read_audio_buffer(resolve_path(request_base_, value->as_string()));
        has_voice = true;
    }
    if (const auto * value = body.find("reference_text")) {
        request.options["reference_text"] = value->as_string();
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }
    return request;
}

struct ServerState::TimedTaskResult {
    engine::runtime::TaskResult result;
    double wall_ms = 0.0;
    std::optional<double> ttft_ms;
};

ServerState::TimedTaskResult ServerState::run_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request) {
    std::lock_guard<std::mutex> lock(model.mutex);
    ensure_model_loaded_locked(model);
    if (model.offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    auto result = model.offline->run(request);
    return TimedTaskResult{std::move(result), elapsed_ms(started), std::nullopt};
}

ServerState::TimedTaskResult ServerState::run_streaming_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink) {
    std::lock_guard<std::mutex> lock(model.mutex);
    ensure_model_loaded_locked(model);
    if (model.streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    TimedTaskResult timed_result;
    auto result = minitts::app::run_streaming_task(
        *model.streaming,
        request,
        [&](const engine::runtime::StreamEvent & event) {
            if (!timed_result.ttft_ms.has_value() && stream_event_has_output(event)) {
                timed_result.ttft_ms = elapsed_ms(started);
            }
            if (event_sink) {
                event_sink(event);
            }
        });
    timed_result.result = std::move(result);
    timed_result.wall_ms = elapsed_ms(started);
    if (!timed_result.ttft_ms.has_value() && task_result_has_output(timed_result.result)) {
        timed_result.ttft_ms = timed_result.wall_ms;
    }
    return timed_result;
}

HttpResponse ServerState::handle_speech(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto request = build_speech_request(model, body);
    if (body.find("stream_format") != nullptr || bool_field(body, "stream", false)) {
        return handle_speech_stream(model, request, body);
    }
    const auto timed_result = model.task.mode == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request)
        : run_model(model, request);
    const auto & audio = select_audio_output(timed_result.result);
    const auto wav = encode_pcm16_wav(audio);
    const auto response_format = engine::io::json::optional_string(body, "response_format", "wav");
    if (response_format == "json" || response_format == "b64_json") {
        return json_response(
            "{\"audio\":" + json_quote(base64_encode(wav)) +
            ",\"format\":\"wav\",\"timing\":" + timing_json(timed_result.wall_ms, audio) + "}");
    }
    HttpResponse response;
    response.status = 200;
    response.content_type = "audio/wav";
    response.body = std::string(reinterpret_cast<const char *>(wav.data()), wav.size());
    response.headers = timing_headers(timed_result.wall_ms, audio);
    return response;
}

HttpResponse ServerState::handle_speech_stream(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const Value & body) {
    if (model.task.mode != engine::runtime::RunMode::Streaming) {
        throw std::runtime_error("speech streaming requires a model configured with mode=streaming");
    }
    const auto stream_format = engine::io::json::optional_string(body, "stream_format", "sse");
    const auto response_format = engine::io::json::optional_string(body, "response_format", "pcm");
    if (response_format != "pcm") {
        throw std::runtime_error("streaming speech currently supports response_format=pcm");
    }
    if (stream_format != "sse" && stream_format != "audio") {
        throw std::runtime_error("streaming speech stream_format must be sse or audio");
    }

    LoadedModel * model_ptr = &model;
    auto stream_body = [this, model_ptr, request](HttpStreamWriter & writer) {
        bool wrote_audio = false;
        const auto timed_result = run_streaming_model(
            *model_ptr,
            request,
            [&](const engine::runtime::StreamEvent & event) {
                std::vector<engine::runtime::AudioBuffer> buffers;
                if (event.audio_output.has_value()) {
                    buffers.push_back(*event.audio_output);
                }
                for (const auto & named : event.named_audio_outputs) {
                    buffers.push_back(named.audio);
                }
                for (const auto & audio : buffers) {
                    const auto pcm = encode_pcm16_samples(audio);
                    write_sse(
                        writer,
                        "{\"type\":\"speech.audio.delta\",\"audio\":" +
                            json_quote(base64_encode(pcm)) +
                            "}");
                    wrote_audio = true;
                }
            });
        if (!wrote_audio) {
            throw std::runtime_error("streaming speech model produced no audio delta events");
        }
        write_sse(
            writer,
            "{\"type\":\"speech.audio.done\",\"timing\":" +
                ttft_timing_json(require_ttft_ms(timed_result.ttft_ms)) +
                "}");
        write_sse_done(writer);
    };
    if (stream_format == "sse") {
        return sse_response(std::move(stream_body));
    }
    return chunked_audio_response([this, model_ptr, request](HttpStreamWriter & writer) {
        bool wrote_audio = false;
        (void)run_streaming_model(
            *model_ptr,
            request,
            [&](const engine::runtime::StreamEvent & event) {
                if (event.audio_output.has_value()) {
                    const auto pcm = encode_pcm16_samples(*event.audio_output);
                    writer.write(std::string(reinterpret_cast<const char *>(pcm.data()), pcm.size()));
                    wrote_audio = true;
                }
                for (const auto & named : event.named_audio_outputs) {
                    const auto pcm = encode_pcm16_samples(named.audio);
                    writer.write(std::string(reinterpret_cast<const char *>(pcm.data()), pcm.size()));
                    wrote_audio = true;
                }
            });
        if (!wrote_audio) {
            throw std::runtime_error("streaming speech model produced no audio delta events");
        }
    });
}

HttpResponse ServerState::handle_transcription(const HttpRequest & request) {
    std::string content_type;
    if (const auto it = request.headers.find("content-type"); it != request.headers.end()) {
        content_type = it->second;
    }
    if (const auto boundary = extract_multipart_boundary(content_type)) {
        return handle_transcription_multipart(request.body, *boundary);
    }
    return handle_transcription_json(request.body);
}

HttpResponse ServerState::handle_transcription_json(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto request = build_openai_transcription_request(body, request_base_);
    if (bool_field(body, "stream", false)) {
        return run_transcription_stream(model, request);
    }
    return run_transcription(model, request);
}

// Accepts the same multipart/form-data shape OpenAI's Whisper API (and clients built against it,
// e.g. Open WebUI) send: a "file" part with the audio bytes, plus "model" and optional "language"
// fields. audio.cpp's native JSON request only takes a server-local path, so the uploaded bytes are
// spooled to a temp file and routed through the existing JSON request builder.
HttpResponse ServerState::handle_transcription_multipart(const std::string & body_text, const std::string & boundary) {
    const auto parts = parse_multipart_body(body_text, boundary);

    const MultipartPart * file_part = nullptr;
    std::string model_id;
    std::string language;
    bool stream = false;
    for (const auto & part : parts) {
        if (part.name == "file") {
            file_part = &part;
        } else if (part.name == "model") {
            model_id = part.data;
        } else if (part.name == "language") {
            language = part.data;
        } else if (part.name == "stream") {
            if (part.data == "true" || part.data == "True" || part.data == "1") {
                stream = true;
            } else if (part.data == "false" || part.data == "False" || part.data == "0") {
                stream = false;
            } else {
                throw std::runtime_error("multipart transcription stream field must be true or false");
            }
        }
    }
    if (file_part == nullptr || file_part->data.empty()) {
        throw std::runtime_error("multipart transcription request requires a non-empty 'file' field");
    }
    if (model_id.empty()) {
        throw std::runtime_error("multipart transcription request requires a 'model' field");
    }

    const TempFileGuard guard{write_temp_upload(file_part->filename, file_part->data)};

    engine::io::json::Value::Object fields;
    fields.emplace("model", engine::io::json::Value::make_string(model_id));
    fields.emplace("audio", engine::io::json::Value::make_string(guard.path.string()));
    if (!language.empty()) {
        fields.emplace("language", engine::io::json::Value::make_string(language));
    }
    const auto body = engine::io::json::Value::make_object(std::move(fields));

    auto & model = require_model(body);
    const auto request = build_openai_transcription_request(body, request_base_);
    if (stream) {
        return run_transcription_stream(model, request);
    }
    return run_transcription(model, request);
}

HttpResponse ServerState::run_transcription(LoadedModel & model, const engine::runtime::TaskRequest & request) {
    const auto timed_result = model.task.mode == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request)
        : run_model(model, request);
    const auto & result = timed_result.result;
    if (!result.text_output.has_value()) {
        throw std::runtime_error("model result did not contain transcript text");
    }
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("transcription timing requires audio_input");
    }
    return json_response(
        "{\"text\":" + json_quote(result.text_output->text) +
        ",\"timing\":" + timing_json(timed_result.wall_ms, *request.audio_input) + "}");
}

HttpResponse ServerState::run_transcription_stream(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request) {
    if (model.task.mode != engine::runtime::RunMode::Streaming) {
        throw std::runtime_error("transcription stream=true requires a model configured with mode=streaming");
    }
    LoadedModel * model_ptr = &model;
    return sse_response([this, model_ptr, request](HttpStreamWriter & writer) {
        const auto timed_result = run_streaming_model(
            *model_ptr,
            request,
            [&](const engine::runtime::StreamEvent & event) {
                if (!event.partial_text.has_value() || event.partial_text->text.empty()) {
                    return;
                }
                write_sse(
                    writer,
                    "{\"type\":\"transcript.text.delta\",\"delta\":" +
                        json_quote(event.partial_text->text) +
                        "}");
            });
        if (!timed_result.result.text_output.has_value()) {
            throw std::runtime_error("streaming transcription result did not contain transcript text");
        }
        write_sse(
            writer,
            "{\"type\":\"transcript.text.done\",\"text\":" +
                json_quote(timed_result.result.text_output->text) +
                ",\"timing\":" +
                ttft_timing_json(require_ttft_ms(timed_result.ttft_ms)) +
                "}");
        write_sse_done(writer);
    });
}

HttpResponse ServerState::handle_generic_run(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto * request_json = body.find("request");
    const auto request = minitts::cli::build_request_from_json(
        request_json != nullptr ? *request_json : body,
        request_base_);
    const auto timed_result = model.task.mode == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request)
        : run_model(model, request);
    return json_response(task_result_json(timed_result.result, timed_result.wall_ms));
}

HttpResponse ServerState::handle_generic_stream(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto * request_json = body.find("request");
    const auto request = minitts::cli::build_request_from_json(
        request_json != nullptr ? *request_json : body,
        request_base_);
    std::vector<engine::runtime::StreamEvent> events;
    const auto timed_result = run_streaming_model(
        model,
        request,
        [&](const engine::runtime::StreamEvent & event) {
            events.push_back(event);
        });
    std::ostringstream out;
    out << "{\"events\":[";
    for (size_t i = 0; i < events.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << stream_event_json(events[i]);
    }
    out << "],\"result\":" << streaming_task_result_json(timed_result.result, timed_result.ttft_ms) << "}";
    return json_response(out.str());
}

// Cached-voice discovery for the "voice"/cached_voice_id request field. Families that
// support voice presets (e.g. pocket_tts, see assets.cpp: model_root/embeddings/<id>.safetensors)
// keep them under an "embeddings" directory next to the model weights; other families simply
// have no such directory and report no voices. Used by clients (llama-swap's playground, and
// potentially Open WebUI) that call GET /v1/audio/voices?model=<id> to populate a voice picker
// instead of guessing generic names like "alloy"/"nova".
HttpResponse ServerState::handle_voices(const HttpRequest & request) const {
    const std::string model_id = query_param(request.query, "model");
    std::vector<std::string> voices;

    const auto it = model_index_.find(model_id);
    if (it != model_index_.end()) {
        for (const auto & [name, preset] : models_.at(it->second)->voice_presets) {
            (void) preset;
            voices.push_back(name);
        }
        const auto embeddings_dir = models_.at(it->second)->config.path / "embeddings";
        std::error_code ec;
        if (std::filesystem::is_directory(embeddings_dir, ec)) {
            for (const auto & entry : std::filesystem::directory_iterator(embeddings_dir, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                    voices.push_back(entry.path().stem().string());
                }
            }
        }
    }
    std::sort(voices.begin(), voices.end());
    voices.erase(std::unique(voices.begin(), voices.end()), voices.end());

    std::ostringstream out;
    out << "{\"voices\":[";
    for (size_t i = 0; i < voices.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << json_quote(voices[i]);
    }
    out << "]}";
    return json_response(out.str());
}

std::string ServerState::models_json() const {
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < models_.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & model = *models_[i];
        out << "{\"id\":" << json_quote(model.config.id)
            << ",\"object\":\"model\""
            << ",\"owned_by\":\"engine\""
            << ",\"family\":" << json_quote(model.config.family)
            << ",\"task\":" << json_quote(engine::runtime::to_string(model.task.task))
            << ",\"mode\":" << json_quote(engine::runtime::to_string(model.task.mode))
            << "}";
    }
    out << "]}";
    return out.str();
}

}  // namespace minitts::server
