#include "runtime.h"

#include "base64.h"
#include "multipart.h"
#include "ui_assets.h"

#include "../cli/request.h"
#include "../streaming/pcm_source.h"
#include "../streaming/streaming.h"

#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/metadata.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace minitts::server {
namespace {

using engine::io::json::Value;

using Clock = std::chrono::steady_clock;

#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
void materialize_embedded_demo_voices(const std::filesystem::path & voice_dir) {
    std::filesystem::create_directories(voice_dir);
    for (const auto & voice : embedded_demo_voices()) {
        std::ofstream wav(
            voice_dir / (std::string(voice.name) + ".wav"),
            std::ios::binary | std::ios::trunc);
        if (!wav) {
            throw std::runtime_error("failed to create embedded demo voice: " + std::string(voice.name));
        }
        wav.write(voice.wav_bytes.data(), static_cast<std::streamsize>(voice.wav_bytes.size()));
        if (!wav) {
            throw std::runtime_error("failed to write embedded demo voice: " + std::string(voice.name));
        }
    }
    const auto prompt_text = embedded_demo_voice_prompt_text();
    std::ofstream prompts(voice_dir / "prompt_text", std::ios::binary | std::ios::trunc);
    if (!prompts) {
        throw std::runtime_error("failed to create embedded demo voice prompt_text");
    }
    prompts.write(prompt_text.data(), static_cast<std::streamsize>(prompt_text.size()));
    if (!prompts) {
        throw std::runtime_error("failed to write embedded demo voice prompt_text");
    }
}
#endif

// Per-request override for the busy timeout. Absent means "use the model's
// configured ceiling"; a value is clamped to that ceiling by resolve_busy_timeout_ms
// so a client can shorten its own wait but never weaken the guard.
std::optional<int> parse_busy_timeout_override(const Value & body) {
    const auto * value = body.find("busy_timeout_ms");
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto requested = engine::io::json::optional_i32(body, "busy_timeout_ms", 0);
    if (requested < 0) {
        throw std::runtime_error("busy_timeout_ms must be >= 0 (0 means no client-side bound)");
    }
    return requested;
}

bool model_accepts_request_option(std::string_view family, std::string_view option) {
    const auto contract = engine::model_spec::model_contract(family);
    if (!contract.has_value()) {
        return true;
    }
    return contract->request_option_keys.find(std::string(option)) != contract->request_option_keys.end();
}

std::string json_quote(std::string_view value) {
    return engine::io::json::stringify_string(value);
}

std::filesystem::path resolve_path(const std::filesystem::path & base, const std::filesystem::path & path) {
    return path.is_absolute() ? path : base / path;
}

// Looks up `<voice_name>` in `<voice_dir>/prompt_text`, a mapping file with one
// `<basename>|<transcript>` line per built-in voice (same format the webui uses).
std::optional<std::string> load_voice_library_text(
    const std::filesystem::path & voice_dir, const std::string & voice_name) {
    std::ifstream f(voice_dir / "prompt_text");
    std::string line;
    while (std::getline(f, line)) {
        const auto sep = line.find('|');
        if (sep == std::string::npos) continue;
        std::string name = line.substr(0, sep);
        // trim trailing whitespace from name
        while (!name.empty() && std::isspace(static_cast<unsigned char>(name.back()))) {
            name.pop_back();
        }
        if (name == voice_name) {
            return line.substr(sep + 1);
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> resolve_voice_library_wav(
    const std::filesystem::path & voice_dir,
    const std::string & voice_name) {
    const std::filesystem::path name_path(voice_name);
    if (voice_name.empty() ||
        name_path.has_root_name() ||
        name_path.has_root_directory() ||
        name_path.has_parent_path() ||
        name_path.filename().string() != voice_name ||
        voice_name == "." ||
        voice_name == "..") {
        return std::nullopt;
    }
    const auto wav = voice_dir / (voice_name + ".wav");
    std::error_code ec;
    if (!std::filesystem::is_regular_file(wav, ec)) {
        return std::nullopt;
    }
    return wav;
}

std::unordered_map<std::string, std::string> options_from_object(const Value * value);

std::string safe_upload_name(std::string value) {
    value = std::filesystem::path(value).filename().string();
    for (char & ch : value) {
        const auto uch = static_cast<unsigned char>(ch);
        if (!(std::isalnum(uch) || ch == '.' || ch == '-' || ch == '_')) {
            ch = '_';
        }
    }
    if (value.empty() || value == "." || value == "..") {
        value = "audio.wav";
    }
    return value;
}

ServerModelConfig model_config_from_json(
    const Value & body,
    const std::filesystem::path & request_base,
    bool lazy) {
    ServerModelConfig model;
    model.id = engine::io::json::require_string(body, "id");
    model.path = resolve_path(request_base, engine::io::json::require_string(body, "path"));
    model.family = engine::io::json::require_string(body, "family");
    model.task = engine::io::json::optional_string(body, "task", model.task);
    model.mode = engine::io::json::optional_string(body, "mode", model.mode);
    model.lazy = lazy;
    if (const auto * value = body.find("model_spec_override")) {
        model.model_spec_override = resolve_path(request_base, value->as_string());
    }
    if (const auto * value = body.find("config")) {
        model.config_id = value->as_string();
    }
    if (const auto * value = body.find("weight")) {
        model.weight_id = value->as_string();
    }
    model.load_options = options_from_object(body.find("load_options"));
    model.session_options = options_from_object(body.find("session_options"));
    return model;
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

int hex_value(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

std::string url_decode_query_value(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '+') {
            out.push_back(' ');
        } else if (value[i] == '%' && i + 2 < value.size()) {
            const int hi = hex_value(value[i + 1]);
            const int lo = hex_value(value[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
            } else {
                out.push_back(value[i]);
            }
        } else {
            out.push_back(value[i]);
        }
    }
    return out;
}

std::string decoded_query_param(const std::string & query, const std::string & key) {
    return url_decode_query_value(query_param(query, key));
}

const char * backend_name(engine::core::BackendType type) {
    switch (type) {
        case engine::core::BackendType::Cpu:
            return "cpu";
        case engine::core::BackendType::Cuda:
            return "cuda";
        case engine::core::BackendType::Hip:
            return "hip";
        case engine::core::BackendType::Vulkan:
            return "vulkan";
        case engine::core::BackendType::Metal:
            return "metal";
        case engine::core::BackendType::BestAvailable:
            return "best";
    }
    return "unknown";
}

HttpResponse ui_service_worker_retirement_response() {
    HttpResponse response;
    response.status = 200;
    response.content_type = "application/javascript; charset=utf-8";
    response.body = R"JS(
// Retire service workers left by applications that previously used this origin.
self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(cacheNames.map((name) => caches.delete(name)));
    await self.clients.claim();
    const windows = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    await self.registration.unregister();
    await Promise.all(windows.map((client) => client.navigate(client.url)));
  })());
});
)JS";
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0";
    response.headers["Clear-Site-Data"] = "\"cache\"";
    response.headers["Expires"] = "0";
    response.headers["Pragma"] = "no-cache";
    response.headers["Service-Worker-Allowed"] = "/";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
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

bool is_wav_upload_filename(const std::string & filename) {
    std::string ext = std::filesystem::path(filename).extension().string();
    for (char & ch : ext) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return ext.empty() || ext == ".wav";
}

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string_view trim_ascii(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1);
    }
    return value;
}

bool is_json_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    if (it == request.headers.end()) {
        return false;
    }
    const std::string content_type = lower_ascii(it->second);
    const auto media_type = trim_ascii(std::string_view(content_type).substr(0, content_type.find(';')));
    return media_type == "application/json" ||
        (media_type.size() > 5 && media_type.substr(media_type.size() - 5) == "+json");
}

std::string request_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    return it == request.headers.end() ? "" : it->second;
}

void log_request_body_if_enabled(const ServerConfig & config, const HttpRequest & request) {
    if (!config.log_request_body || !engine::debug::log_enabled()) {
        return;
    }
    if (request.method != "POST") {
        return;
    }
    if (!request.body.empty() && is_json_content_type(request)) {
        engine::debug::log_message("[REQUEST_BODY] " + request.method + " " + request.path);
        engine::debug::log_message(request.body);
        return;
    }
    if (request.body.empty() && request.body_stream == nullptr) {
        return;
    }
    std::ostringstream out;
    out << "[REQUEST_BODY_SKIPPED] " << request.method << " " << request.path
        << " content_type=" << json_quote(request_content_type(request));
    if (request.body_stream != nullptr) {
        out << " body=stream";
    } else {
        out << " body_bytes=" << request.body.size();
    }
    if (!request.query.empty()) {
        out << " query=" << json_quote(request.query);
    }
    engine::debug::log_message(out.str());
}

void log_multipart_request_summary_if_enabled(
    const ServerConfig & config,
    const std::vector<MultipartPart> & parts) {
    if (!config.log_request_body || !engine::debug::log_enabled()) {
        return;
    }
    for (const auto & part : parts) {
        if (part.filename.empty()) {
            continue;
        }
        std::ostringstream out;
        out << "[REQUEST_BODY_SKIPPED] multipart_file"
            << " field=" << json_quote(part.name)
            << " filename=" << json_quote(part.filename)
            << " bytes=" << part.data.size();
        engine::debug::log_message(out.str());
    }
}

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

std::string live_speech_timing_json(double request_start_to_first_audio_ms, double input_end_ms) {
    const double input_end_to_first_audio_ms = request_start_to_first_audio_ms - input_end_ms;
    const bool first_audio_before_input_end = input_end_to_first_audio_ms < 0.0;
    const double overlap_ms = first_audio_before_input_end ? -input_end_to_first_audio_ms : 0.0;
    std::ostringstream out;
    out << "{\"ttft_ms\":";
    if (first_audio_before_input_end) {
        out << "null";
    } else {
        out << input_end_to_first_audio_ms;
    }
    out << ",\"first_audio_before_input_end\":" << (first_audio_before_input_end ? "true" : "false")
        << ",\"request_start_to_first_audio_ms\":" << request_start_to_first_audio_ms
        << ",\"input_end_ms\":" << input_end_ms
        << ",\"overlap_ms\":" << overlap_ms << "}";
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
        !result.named_audio_outputs.empty() ||
        result.artifact_output.has_value() ||
        !result.output_artifacts.empty();
}

const char * artifact_kind_name(engine::runtime::ArtifactKind kind) {
    using engine::runtime::ArtifactKind;
    switch (kind) {
    case ArtifactKind::SpeakerEmbedding: return "speaker_embedding";
    case ArtifactKind::StyleEmbedding: return "style_embedding";
    case ArtifactKind::PromptEmbedding: return "prompt_embedding";
    case ArtifactKind::AcousticTokens: return "acoustic_tokens";
    case ArtifactKind::Midi: return "midi";
    case ArtifactKind::TranscriptAlignment: return "transcript_alignment";
    case ArtifactKind::DiarizationState: return "diarization_state";
    case ArtifactKind::VadState: return "vad_state";
    case ArtifactKind::Custom: return "custom";
    }
    return "custom";
}

double require_ttft_ms(const std::optional<double> & ttft_ms) {
    if (!ttft_ms.has_value()) {
        throw std::runtime_error("streaming response produced no TTFT event");
    }
    return *ttft_ms;
}

double require_input_end_ms(const std::optional<double> & input_end_ms) {
    if (!input_end_ms.has_value()) {
        throw std::runtime_error("live streaming response did not observe input end");
    }
    return *input_end_ms;
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
    if (result.artifact_output.has_value() || !result.output_artifacts.empty()) {
        field("artifacts");
        out << "[";
        bool first_artifact = true;
        auto write_artifact = [&](const engine::runtime::VoiceArtifact & artifact) {
            if (!first_artifact) out << ",";
            first_artifact = false;
            out << "{\"id\":" << json_quote(artifact.id)
                << ",\"kind\":" << json_quote(artifact_kind_name(artifact.kind))
                << ",\"payload\":" << json_quote(base64_encode(artifact.payload))
                << ",\"meta\":{";
            bool first_meta = true;
            for (const auto & [key, value] : artifact.meta) {
                if (!first_meta) out << ",";
                first_meta = false;
                out << json_quote(key) << ":" << json_quote(value);
            }
            out << "}}";
        };
        if (result.artifact_output.has_value()) write_artifact(*result.artifact_output);
        for (const auto & artifact : result.output_artifacts) write_artifact(artifact);
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
                << ",\"confidence\":" << segment.confidence;
            if (!segment.text.empty()) {
                out << ",\"text\":" << json_quote(segment.text);
            }
            out << "}";
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
                << ",\"confidence\":" << turn.confidence;
            if (!turn.text.empty()) {
                out << ",\"text\":" << json_quote(turn.text);
            }
            out << "}";
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

engine::runtime::TaskRequest build_openai_transcription_request(
    const Value & body,
    const std::filesystem::path & base_dir,
    const std::string * uploaded_audio_bytes = nullptr) {
    const auto * audio = body.find("audio");
    if (audio == nullptr) {
        audio = body.find("audio_path");
    }
    if (audio == nullptr) {
        audio = body.find("file");
    }
    if (uploaded_audio_bytes == nullptr && (audio == nullptr || !audio->is_string())) {
        throw std::runtime_error("transcription request requires audio, audio_path, or file path");
    }

    engine::runtime::TaskRequest request;
    if (uploaded_audio_bytes == nullptr) {
        request.audio_input = minitts::cli::read_audio_buffer(resolve_path(base_dir, audio->as_string()));
    } else {
        request.audio_input = minitts::cli::read_audio_buffer(std::string_view(*uploaded_audio_bytes));
    }
    request.options = options_from_object(body.find("options"));
    std::string language;
    if (const auto * value = body.find("language")) {
        language = value->as_string();
        request.options["language"] = language;
    }
    std::string context;
    if (const auto * value = body.find("text")) {
        context = value->as_string();
    }
    if (!language.empty() || !context.empty()) {
        request.text_input = engine::runtime::Transcript{std::move(context), std::move(language)};
    }
    return request;
}

template <typename Predicate>
std::optional<std::filesystem::path> find_ancestor(
    std::filesystem::path start,
    Predicate predicate) {
    std::error_code ec;
    start = std::filesystem::absolute(std::move(start), ec).lexically_normal();
    if (ec) {
        return std::nullopt;
    }
    for (;;) {
        if (predicate(start)) {
            return start;
        }
        const auto parent = start.parent_path();
        if (parent.empty() || parent == start) {
            return std::nullopt;
        }
        start = parent;
    }
}

template <typename Predicate>
std::optional<std::filesystem::path> find_from_roots(
    const std::filesystem::path & request_base,
    const std::filesystem::path & resource_anchor,
    Predicate predicate) {
    if (auto found = find_ancestor(request_base, predicate)) {
        return found;
    }
    if (!resource_anchor.empty()) {
        return find_ancestor(resource_anchor, predicate);
    }
    return std::nullopt;
}

}  // namespace

ServerState::ServerState(
    ServerConfig config,
    std::filesystem::path request_base,
    std::filesystem::path ui_resource_anchor)
    : config_(std::move(config)),
      request_base_(std::filesystem::absolute(std::move(request_base)).lexically_normal()) {
#if !defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
    (void) ui_resource_anchor;
#endif
    if (config_.backend != engine::core::BackendType::Cuda) {
        std::cerr
            << "audio.cpp is optimized for CUDA. The "
            << backend_name(config_.backend)
            << " server backend is intended for portability and testing, but performance and model coverage may be lower than CUDA.\n";
    }
    if (config_.ui_enabled || config_.ui_management) {
        upload_root_ = std::filesystem::temp_directory_path() /
            ("audiocpp-ui-" + std::to_string(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()));
        std::filesystem::create_directories(upload_root_);
    }
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
    if (config_.ui_management) {
        repository_root_ = find_from_roots(
            request_base_,
            ui_resource_anchor,
            [](const std::filesystem::path & root) {
                return std::filesystem::is_directory(root / "model_specs");
            }).value_or(request_base_);
        const auto binary_directory = ui_resource_anchor.empty()
            ? request_base_
            : std::filesystem::absolute(ui_resource_anchor).lexically_normal();
        default_models_root_ = (binary_directory / "models").lexically_normal();
        models_root_ = default_models_root_;
        request_base_ = binary_directory;
        std::filesystem::create_directories(models_root_);
        std::cerr
            << "native WebUI model root: " << models_root_ << "\n"
            << "native WebUI package resources: " << repository_root_ << "\n";
        if (!config_.voice_dir.has_value()) {
            const auto embedded_voices = upload_root_ / "demo_voices";
            materialize_embedded_demo_voices(embedded_voices);
            config_.voice_dir = embedded_voices;
        }
        model_installer_ = std::make_unique<ModelInstaller>(
            repository_root_,
            models_root_);
    }
#endif
    load_models();
}

ServerState::~ServerState() {
    if (!upload_root_.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(upload_root_, ec);
    }
}

HttpResponse ServerState::handle(const HttpRequest & request) {
  HttpResponse response;
  const std::string allowed_origin = get_allowed_origin(request);
  try {
    log_request_body_if_enabled(config_, request);
    if (request.method == "OPTIONS" && (!allowed_origin.empty() || config_.ui_enabled)) {
        response.status = 204;
        response.content_type = "text/plain";
        response.headers["Access-Control-Allow-Headers"] = "*";
        response.headers["Access-Control-Allow-Methods"] = "GET, POST";
    }
    else if (request.method == "GET" && (request.path == "/" || request.path == "/index.html")) {
        response = handle_ui_asset();
    }
    else if (
        request.method == "GET" && config_.ui_enabled &&
        (request.path == "/sw.js" || request.path == "/service-worker.js")) {
        response = ui_service_worker_retirement_response();
    }
    else if (request.method == "GET" && request.path == "/favicon.ico") {
        response.status = 204;
        response.content_type = "image/x-icon";
    }
    else if (request.method == "GET" && request.path == "/health") {
        size_t model_count = 0;
        {
            std::lock_guard<std::mutex> lock(models_mutex_);
            model_count = models_.size();
        }
        response = json_response(
            "{\"status\":\"ok\",\"backend\":\"" +
            std::string(backend_name(config_.backend)) +
            "\",\"models\":" +
            std::to_string(model_count) +
            ",\"ui\":" + (config_.ui_enabled ? "true" : "false") +
            ",\"ui_management\":" + (config_.ui_management ? "true" : "false") +
            "}");
    }
    else if (request.method == "GET" && request.path == "/v1/models") {
        response = json_response(models_json());
    }
    else if (request.method == "GET" && request.path == "/v1/audio/voices") {
        response = handle_voices(request);
    }
    else if (request.method == "POST" && request.path == "/v1/models/load") {
        response = handle_model_load(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/models/unload") {
        response = handle_model_unload(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/path-status") {
        response = handle_path_status(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/upload") {
        response = handle_ui_upload(request);
    }
#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
    else if (request.method == "GET" && request.path == "/v1/ui/models-root") {
        response = handle_models_root_get();
    }
    else if (request.method == "POST" && request.path == "/v1/ui/models-root") {
        response = handle_models_root_set(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/browse-directories") {
        response = handle_directory_browser(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/models/install") {
        response = handle_model_install(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/models/install/stop") {
        response = handle_model_install_stop(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/models/clean-partial") {
        response = handle_model_clean_partial(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/ui/models/delete") {
        response = handle_model_remove(request.body);
    }
    else if (request.method == "GET" && request.path == "/v1/ui/models/install-status") {
        response = handle_model_install_status(request);
    }
    else if (request.method == "GET" && request.path == "/v1/ui/models/package-sizes") {
        response = handle_model_package_sizes();
    }
#endif
    else if (request.method == "POST" && request.path == "/v1/audio/speech") {
        response = handle_speech(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/audio/speech/live") {
        response = handle_speech_live(request);
    }
    else if (request.method == "POST" && request.path == "/v1/audio/transcriptions") {
        response = handle_transcription(request);
    }
    // Separate path rather than a flag on the endpoint above: the input transport
    // differs (raw chunked PCM vs a complete upload), so keeping it distinct leaves
    // every existing transcription client untouched.
    else if (request.method == "POST" && request.path == "/v1/audio/transcriptions/live") {
        response = handle_transcription_live(request);
    }
    else if (request.method == "POST" && request.path == "/v1/tasks/run") {
        response = handle_generic_run(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/tasks/stream") {
        response = handle_generic_stream(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/tasks/unload_models") {
        response = handle_unload_models(request.body);
    }
    else if (request.method == "POST" && request.path == "/v1/tasks/unload_all_models") {
        response = handle_unload_all_models();
    }
    else {
        response = error_response(404, "unknown endpoint: " + request.path, "not_found");
    }
  } catch (const engine::runtime::CapacityError & ex) {
    // The request is too big for the device, which is the caller's to fix --
    // reporting it as 500 sends them looking for a server fault that is not
    // there. Checked before ServerBusyError only because both are
    // runtime_error; the two conditions are disjoint.
    response = error_response(400, ex.what(), "invalid_request_error");
  } catch (const ServerBusyError & ex) {
    // Non-streaming requests surface the busy state as 503 before any response is
    // sent. (Streaming requests acquire the lock inside the stream body, after
    // headers are sent, so there it becomes a stream error event instead.)
    response = error_response(503, ex.what(), "server_busy");
  }
  if (!allowed_origin.empty()) {
      response.headers["Access-Control-Allow-Origin"] = allowed_origin;
  }
  return response;
}

void ServerState::load_models() {
    for (auto & config : config_.models) {
        auto loaded = make_model(std::move(config));
        if (!model_index_.emplace(loaded->config.id, models_.size()).second) {
            throw std::runtime_error("duplicate server model id: " + loaded->config.id);
        }
        if (!loaded->config.lazy) {
            ensure_model_loaded_locked(*loaded);
        }
        models_.push_back(std::move(loaded));
    }
}

std::unique_ptr<ServerState::LoadedModel> ServerState::make_model(ServerModelConfig config) {
    auto loaded = std::make_unique<LoadedModel>();
    loaded->config = std::move(config);
    loaded->task = engine::runtime::TaskSpec{
        engine::runtime::parse_voice_task_kind(loaded->config.task),
        engine::runtime::parse_run_mode(loaded->config.mode),
    };
    load_voice_presets(*loaded);
    return loaded;
}

HttpResponse ServerState::handle_model_load(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "dynamic model management is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    auto requested = model_config_from_json(body, request_base_, false);
    requested.path = resolve_ui_model_path(engine::io::json::require_string(body, "path"));

    LoadedModel * existing = nullptr;
    {
        std::lock_guard<std::mutex> state_lock(models_mutex_);
        const auto found = model_index_.find(requested.id);
        if (found != model_index_.end()) {
            existing = models_.at(found->second).get();
        }
    }

    if (existing != nullptr) {
        BusyGuard::Lock run_lock = acquire_model_run(*existing, std::nullopt);
        std::unique_lock<std::shared_mutex> metadata_lock(existing->metadata_mutex);
        const bool changed =
            existing->config.path != requested.path ||
            existing->config.family != requested.family ||
            existing->config.task != requested.task ||
            existing->config.mode != requested.mode ||
            existing->config.load_options != requested.load_options ||
            existing->config.session_options != requested.session_options ||
            existing->config.model_spec_override != requested.model_spec_override;
        if (changed) {
            existing->streaming = nullptr;
            existing->offline = nullptr;
            existing->session.reset();
            existing->model.reset();
            existing->loaded.store(false);
            existing->voice_presets.clear();
            existing->default_voice_preset.reset();
            existing->config = std::move(requested);
            existing->task = engine::runtime::TaskSpec{
                engine::runtime::parse_voice_task_kind(existing->config.task),
                engine::runtime::parse_run_mode(existing->config.mode),
            };
            load_voice_presets(*existing);
        }
        ensure_model_loaded_locked(*existing);
        return json_response(
            "{\"id\":" + json_quote(existing->config.id) +
            ",\"loaded\":true,\"reconfigured\":" + (changed ? "true" : "false") + "}");
    }

    auto loaded = make_model(std::move(requested));
    ensure_model_loaded_locked(*loaded);
    const std::string id = loaded->config.id;
    {
        std::lock_guard<std::mutex> state_lock(models_mutex_);
        if (model_index_.find(id) != model_index_.end()) {
            throw std::runtime_error("model id was registered concurrently: " + id);
        }
        model_index_.emplace(id, models_.size());
        models_.push_back(std::move(loaded));
    }
    return json_response("{\"id\":" + json_quote(id) + ",\"loaded\":true,\"reconfigured\":false}");
}

std::filesystem::path ServerState::resolve_ui_model_path(const std::filesystem::path & path) const {
    if (path.is_absolute()) {
        return path;
    }

    const auto relative = path.lexically_normal();
    const auto generic = relative.generic_string();
    if (!repository_root_.empty() &&
        (generic == "assets/framework" || generic.rfind("assets/framework/", 0) == 0)) {
        return (repository_root_ / relative).lexically_normal();
    }
    return (request_base_ / relative).lexically_normal();
}

HttpResponse ServerState::handle_model_unload(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "dynamic model management is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const std::string id = engine::io::json::require_string(body, "id");
    LoadedModel * model = nullptr;
    {
        std::lock_guard<std::mutex> state_lock(models_mutex_);
        const auto found = model_index_.find(id);
        if (found == model_index_.end()) {
            return error_response(404, "unknown model id: " + id, "not_found");
        }
        model = models_.at(found->second).get();
    }
    BusyGuard::Lock run_lock = acquire_model_run(*model, std::nullopt);
    model->streaming = nullptr;
    model->offline = nullptr;
    model->session.reset();
    model->model.reset();
    model->loaded.store(false);
    return json_response("{\"id\":" + json_quote(id) + ",\"loaded\":false}");
}

HttpResponse ServerState::handle_path_status(const std::string & body_text) const {
    if (!config_.ui_management) {
        return error_response(403, "UI path inspection is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto path = resolve_ui_model_path(engine::io::json::require_string(body, "path"));
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    const bool directory = exists && std::filesystem::is_directory(path, ec);
    const bool regular_file = exists && std::filesystem::is_regular_file(path, ec);
    return json_response(
        "{\"path\":" + json_quote(path.string()) +
        ",\"exists\":" + (exists ? "true" : "false") +
        ",\"directory\":" + (directory ? "true" : "false") +
        ",\"file\":" + (regular_file ? "true" : "false") + "}");
}

HttpResponse ServerState::handle_ui_upload(const HttpRequest & request) {
    if (!config_.ui_enabled && !config_.ui_management) {
        return error_response(403, "UI uploads are disabled", "forbidden");
    }
    if (request.body.empty()) {
        return error_response(400, "upload body is empty", "invalid_request_error");
    }
    constexpr size_t kMaxUploadBytes = size_t{2} * 1024 * 1024 * 1024;
    if (request.body.size() > kMaxUploadBytes) {
        return error_response(413, "upload exceeds the 2 GiB limit", "invalid_request_error");
    }
    std::string filename = "audio.wav";
    if (const auto it = request.headers.find("x-audiocpp-filename"); it != request.headers.end()) {
        filename = safe_upload_name(it->second);
    }
    const auto id = next_upload_id_.fetch_add(1);
    const auto path = upload_root_ / (std::to_string(id) + "-" + filename);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("could not create temporary upload: " + path.string());
    }
    out.write(request.body.data(), static_cast<std::streamsize>(request.body.size()));
    if (!out) {
        throw std::runtime_error("could not write temporary upload: " + path.string());
    }
    return json_response(
        "{\"path\":" + json_quote(path.string()) +
        ",\"bytes\":" + std::to_string(request.body.size()) + "}");
}

#if defined(AUDIOCPP_HAS_NATIVE_MODEL_MANAGER)
HttpResponse ServerState::handle_model_install(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const std::string package_id = engine::io::json::require_string(body, "id");
    const std::string source_directory =
        engine::io::json::optional_string(body, "source_directory", "");
    const std::string source_file =
        engine::io::json::optional_string(body, "source_file", "");
    const std::string output_file =
        engine::io::json::optional_string(body, "output_file", "");
    const std::string variant =
        engine::io::json::optional_string(body, "variant", "");
    const bool overwrite =
        engine::io::json::optional_bool(body, "overwrite", false);
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    return json_response(model_installer_->start(
        package_id,
        source_file,
        output_file,
        source_directory,
        variant,
        overwrite));
}

HttpResponse ServerState::handle_model_remove(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "UI model removal is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const std::string package_id = engine::io::json::require_string(body, "id");
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model removal is disabled", "forbidden");
    }
    return json_response(model_installer_->remove(package_id));
}

HttpResponse ServerState::handle_model_install_stop(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const std::string package_id = engine::io::json::require_string(body, "id");
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    return json_response(model_installer_->stop(package_id));
}

HttpResponse ServerState::handle_model_clean_partial(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "UI model cleanup is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const std::string package_id = engine::io::json::require_string(body, "id");
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model cleanup is disabled", "forbidden");
    }
    return json_response(model_installer_->clean_partial(package_id));
}

HttpResponse ServerState::handle_model_install_status(const HttpRequest & request) const {
    if (!config_.ui_management) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    return json_response(model_installer_->status(query_param(request.query, "id")));
}

HttpResponse ServerState::handle_model_package_sizes() {
    if (!config_.ui_management) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model installation is disabled", "forbidden");
    }
    return json_response(model_installer_->package_sizes());
}

HttpResponse ServerState::handle_models_root_get() const {
    if (!config_.ui_management) {
        return error_response(403, "UI model management is disabled", "forbidden");
    }
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model management is disabled", "forbidden");
    }
    return json_response(
        "{\"models_root\":" + json_quote(models_root_.string()) +
        ",\"default_models_root\":" + json_quote(default_models_root_.string()) +
        ",\"is_default\":" + (models_root_ == default_models_root_ ? "true" : "false") + "}");
}

HttpResponse ServerState::handle_models_root_set(const std::string & body_text) {
    if (!config_.ui_management) {
        return error_response(403, "UI model management is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto requested_text = engine::io::json::optional_string(body, "path", "");
    auto requested = requested_text.empty()
        ? default_models_root_
        : std::filesystem::path(requested_text);
    if (requested.is_relative()) {
        requested = request_base_ / requested;
    }
    requested = std::filesystem::absolute(requested).lexically_normal();
    std::lock_guard<std::mutex> lock(model_installer_mutex_);
    if (!model_installer_) {
        return error_response(403, "UI model management is disabled", "forbidden");
    }
    if (requested == models_root_) {
        return json_response(
            "{\"models_root\":" + json_quote(models_root_.string()) +
            ",\"default_models_root\":" + json_quote(default_models_root_.string()) +
            ",\"is_default\":" + (models_root_ == default_models_root_ ? "true" : "false") + "}");
    }
    if (model_installer_->has_active_jobs()) {
        return error_response(
            409,
            "cannot change the models folder while a package download is running",
            "model_install_active");
    }

    std::error_code error;
    if (std::filesystem::exists(requested, error) && !std::filesystem::is_directory(requested, error)) {
        return error_response(400, "models folder path is not a directory: " + requested.string(), "invalid_request_error");
    }
    std::filesystem::create_directories(requested, error);
    if (error) {
        return error_response(
            400,
            "could not create models folder '" + requested.string() + "': " + error.message(),
            "invalid_request_error");
    }

    models_root_ = requested;
    model_installer_ = std::make_unique<ModelInstaller>(repository_root_, models_root_);
    std::cerr << "native WebUI model root changed to: " << models_root_ << "\n";
    return json_response(
        "{\"models_root\":" + json_quote(models_root_.string()) +
        ",\"default_models_root\":" + json_quote(default_models_root_.string()) +
        ",\"is_default\":" + (models_root_ == default_models_root_ ? "true" : "false") + "}");
}

HttpResponse ServerState::handle_directory_browser(const std::string & body_text) const {
    if (!config_.ui_management) {
        return error_response(403, "UI directory browsing is disabled", "forbidden");
    }
    const auto body = engine::io::json::parse(body_text);
    const auto requested_text = engine::io::json::optional_string(body, "path", "");

    std::filesystem::path current;
    {
        std::lock_guard<std::mutex> lock(model_installer_mutex_);
        current = requested_text.empty() ? models_root_ : std::filesystem::path(requested_text);
    }
    if (current.is_relative()) {
        current = request_base_ / current;
    }
    current = std::filesystem::absolute(current).lexically_normal();

    std::error_code error;
    if (!std::filesystem::is_directory(current, error)) {
        return error_response(
            400,
            "folder is not accessible: " + current.string(),
            "invalid_request_error");
    }

    std::vector<std::filesystem::path> roots;
#ifdef _WIN32
    for (char drive = 'A'; drive <= 'Z'; ++drive) {
        const auto root = std::filesystem::path(std::string(1, drive) + ":\\");
        error.clear();
        if (std::filesystem::is_directory(root, error)) {
            roots.push_back(root);
        }
    }
#else
    roots.emplace_back("/");
#endif

    struct DirectoryEntry {
        std::string name;
        std::filesystem::path path;
    };
    std::vector<DirectoryEntry> directories;
    error.clear();
    std::filesystem::directory_iterator iterator(
        current,
        std::filesystem::directory_options::skip_permission_denied,
        error);
    const std::filesystem::directory_iterator end;
    while (!error && iterator != end) {
        std::error_code entry_error;
        if (iterator->is_directory(entry_error)) {
            directories.push_back(DirectoryEntry{
                iterator->path().filename().string(),
                iterator->path().lexically_normal(),
            });
        }
        iterator.increment(error);
    }
    std::sort(directories.begin(), directories.end(), [](const auto & left, const auto & right) {
        return left.name < right.name;
    });

    auto parent = current.parent_path();
    if (parent == current) {
        parent.clear();
    }
    std::string response =
        "{\"current\":" + json_quote(current.string()) +
        ",\"parent\":" + json_quote(parent.string()) +
        ",\"roots\":[";
    for (size_t index = 0; index < roots.size(); ++index) {
        if (index > 0) response += ",";
        response += json_quote(roots[index].string());
    }
    response += "],\"directories\":[";
    for (size_t index = 0; index < directories.size(); ++index) {
        if (index > 0) response += ",";
        response += "{\"name\":" + json_quote(directories[index].name) +
            ",\"path\":" + json_quote(directories[index].path.string()) + "}";
    }
    return json_response(response + "]}");
}
#endif

HttpResponse ServerState::handle_ui_asset() const {
    if (!config_.ui_enabled) {
        return error_response(404, "WebUI is disabled", "not_found");
    }
    HttpResponse response;
    response.status = 200;
    response.content_type = "text/html; charset=utf-8";
    const auto html = embedded_ui_html();
    response.body.assign(html.data(), html.size());
    // The WebUI shares the server origin (usually localhost:8080) with any app
    // that previously occupied that port. Never let an old shell survive a
    // server upgrade, and ask the browser to discard only cached resources.
    // Deliberately omit the Clear-Site-Data "storage" directive: saved voices,
    // the selected models folder, and UI preferences live in local storage.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0";
    response.headers["Clear-Site-Data"] = "\"cache\"";
    response.headers["Expires"] = "0";
    response.headers["Pragma"] = "no-cache";
    response.headers["Content-Security-Policy"] =
        "default-src 'self' 'unsafe-inline' blob: data:; connect-src 'self'; media-src 'self' blob: data:";
    response.headers["X-Content-Type-Options"] = "nosniff";
    return response;
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
    load_request.model_spec_override = model.config.model_spec_override.has_value()
        ? model.config.model_spec_override
        : config_.model_spec_override;
    load_request.family_hint = model.config.family;
    load_request.config_id = model.config.config_id;
    load_request.weight_id = model.config.weight_id;
    load_request.options = model.config.load_options;

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = config_.backend;
    session_options.backend.device = config_.device;
    session_options.backend.threads = config_.threads;
    session_options.options = model.config.session_options;

    engine::debug::trace_log_scalar("server.model.id", model.config.id);
    engine::debug::trace_log_scalar("server.model.path", model.config.path.string());
    engine::debug::trace_log_scalar("server.model.family", model.config.family);
    engine::debug::trace_log_scalar(
        "server.model.task",
        std::string_view(engine::runtime::to_string(model.task.task)));
    engine::debug::trace_log_scalar(
        "server.model.mode",
        std::string_view(engine::runtime::to_string(model.task.mode)));
    engine::debug::trace_log_scalar("server.model.backend", std::string_view(backend_name(session_options.backend.type)));
    engine::debug::trace_log_scalar("server.model.device", int64_t{session_options.backend.device});
    engine::debug::trace_log_scalar("server.model.threads", int64_t{session_options.backend.threads});
    engine::debug::trace_log_scalar(
        "server.model.session_option_count",
        static_cast<int64_t>(session_options.options.size()));
    for (const auto & [key, value] : session_options.options) {
        engine::debug::trace_log_scalar("server.model.session_options." + key, value);
    }

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
    model.loaded.store(true);
}

LiveIngestLimits ServerState::live_ingest_limits(const HttpRequest & request) const {
    const std::string model_id = query_param(request.query, "model");
    if (model_id.empty()) {
        return config_.live_ingest;
    }
    const auto it = model_index_.find(model_id);
    if (it == model_index_.end()) {
        return config_.live_ingest;
    }
    return resolve_live_ingest_limits(config_.live_ingest, models_.at(it->second)->config.live_ingest);
}

ServerState::LoadedModel & ServerState::require_model(const Value & body) {
    const std::string id = engine::io::json::require_string(body, "model");
    std::lock_guard<std::mutex> state_lock(models_mutex_);
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
    std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
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
        request.options["instruction"] = value->as_string();
    }

    bool voice_field_is_preset = false;
    const auto * preset = select_voice_preset(model, body, voice_field_is_preset);
    const bool can_inject_reference_text =
        model_accepts_request_option(model.config.family, "reference_text");

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
        if (can_inject_reference_text &&
            preset->reference_text.has_value() &&
            request.options.find("reference_text") == request.options.end()) {
            request.options["reference_text"] = *preset->reference_text;
        }
    }
    const bool has_explicit_voice_ref = body.find("voice_ref") != nullptr;
    if (const auto * value = body.find("voice"); value != nullptr && !voice_field_is_preset) {
        // Voice library: "voice" may name a wav in the configured voice_dir. When it
        // does, that audio becomes the cloning reference and the transcript from
        // prompt_text is injected unless the request already sets reference_text.
        bool voice_library_resolved = false;
        if (config_.voice_dir.has_value() && !has_explicit_voice_ref) {
            const std::string voice_name = value->as_string();
            if (const auto wav = resolve_voice_library_wav(*config_.voice_dir, voice_name)) {
                voice.speaker = engine::runtime::VoiceReference{};
                voice.speaker->audio = minitts::cli::read_audio_buffer(*wav);
                has_voice = true;
                voice_library_resolved = true;
                if (can_inject_reference_text && request.options.find("reference_text") == request.options.end()) {
                    auto text = load_voice_library_text(*config_.voice_dir, voice_name);
                    if (text.has_value()) {
                        request.options["reference_text"] = *text;
                    }
                }
            }
        }
        // Names that do not resolve to a wav keep the cached_voice_id behavior.
        if (!voice_library_resolved) {
            if (!voice.speaker.has_value()) {
                voice.speaker = engine::runtime::VoiceReference{};
            }
            voice.speaker->cached_voice_id = value->as_string();
            has_voice = true;
        }
    }
    if (const auto * value = body.find("voice_ref")) {
        if (!voice.speaker.has_value()) {
            voice.speaker = engine::runtime::VoiceReference{};
        }
        if (value->is_string()) {
            voice.speaker->audio = minitts::cli::read_audio_buffer(resolve_path(request_base_, value->as_string()));
        } else if (value->is_object()) {
            const auto & type = engine::io::json::require_string(*value, "type");
            if (type == "path") {
                voice.speaker->audio = minitts::cli::read_audio_buffer(
                    resolve_path(request_base_, engine::io::json::require_string(*value, "path")));
            } else if (type == "base64") {
                // Bound the inline reference audio so a huge base64 payload cannot
                // blow up host RAM through decode + f32 expansion (~3x its size).
                constexpr size_t kMaxVoiceRefBytes = size_t{5} * 1024 * 1024;
                // 4/3 expansion plus slack for a data URI prefix and whitespace.
                constexpr size_t kMaxVoiceRefB64Length = ((kMaxVoiceRefBytes + 2) / 3) * 4 + 4096;
                const auto & data = engine::io::json::require_string(*value, "data");
                if (data.size() > kMaxVoiceRefB64Length) {
                    throw std::runtime_error("voice_ref base64 data exceeds the 5 MiB limit");
                }
                const auto bytes = base64_decode(data);
                if (bytes.empty()) {
                    throw std::runtime_error("voice_ref base64 data decoded to an empty payload");
                }
                if (bytes.size() > kMaxVoiceRefBytes) {
                    throw std::runtime_error("voice_ref base64 data exceeds the 5 MiB limit");
                }
                voice.speaker->audio = minitts::cli::read_audio_buffer(
                    std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
            } else {
                throw std::runtime_error("voice_ref type must be \"path\" or \"base64\"");
            }
        } else {
            throw std::runtime_error("voice_ref must be a path string or an object with type \"path\" or \"base64\"");
        }
        has_voice = true;
    }
    if (const auto * value = body.find("reference_text")) {
        request.options["reference_text"] = value->as_string();
    }
    if (has_voice) {
        request.voice = std::move(voice);
    }
    return apply_default_request_options(model, std::move(request));
}

engine::runtime::TaskRequest ServerState::apply_default_request_options(
    const LoadedModel & model,
    engine::runtime::TaskRequest request) const {
    if (model.config.default_request_options.empty()) {
        return request;
    }
    auto options = model.config.default_request_options;
    for (auto & [key, value] : request.options) {
        options[key] = std::move(value);
    }
    request.options = std::move(options);
    return request;
}

struct ServerState::TimedTaskResult {
    engine::runtime::TaskResult result;
    double wall_ms = 0.0;
    std::optional<double> ttft_ms;
};

engine::runtime::RunMode ServerState::model_run_mode(const LoadedModel & model) const {
    std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
    return model.task.mode;
}

BusyGuard::Lock ServerState::acquire_model_run(
    LoadedModel & model,
    std::optional<int> request_timeout_ms) {
    int timeout_ms = 0;
    std::string model_id;
    {
        std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
        timeout_ms = resolve_busy_timeout_ms(
            model.config.busy_timeout_ms.value_or(config_.busy_timeout_ms),
            request_timeout_ms);
        model_id = model.config.id;
    }
    return model.busy.acquire(timeout_ms, model_id);
}

ServerState::TimedTaskResult ServerState::run_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    std::optional<int> busy_timeout_ms) {
    BusyGuard::Lock lock = acquire_model_run(model, busy_timeout_ms);
    ensure_model_loaded_locked(model);
    if (model.offline == nullptr) {
        throw std::runtime_error("configured model does not provide offline execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    auto result = model.offline->run(request);
    return TimedTaskResult{std::move(result), elapsed_ms(started), std::nullopt};
}

// `audio` selects where the samples come from: null means request.audio_input, as
// every caller did before live ingest existed; non-null pulls them from a stream
// as they arrive. Everything else — locking, preparation, timing — is identical,
// so both entry points share this body rather than drifting apart.
ServerState::TimedTaskResult ServerState::run_streaming_model_impl(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const minitts::app::AudioChunkStream * audio,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
    std::optional<int> busy_timeout_ms) {
    BusyGuard::Lock lock = acquire_model_run(model, busy_timeout_ms);
    ensure_model_loaded_locked(model);
    if (model.streaming == nullptr) {
        throw std::runtime_error("configured model does not provide streaming execution: " + model.config.id);
    }
    const auto started = Clock::now();
    model.session->prepare(engine::runtime::build_preparation_request(request));
    TimedTaskResult timed_result;
    const auto sink = [&](const engine::runtime::StreamEvent & event) {
        if (!timed_result.ttft_ms.has_value() && stream_event_has_output(event)) {
            timed_result.ttft_ms = elapsed_ms(started);
        }
        if (event_sink) {
            event_sink(event);
        }
    };
    auto result = audio != nullptr
        ? minitts::app::run_streaming_task(*model.streaming, request, sink, *audio)
        : minitts::app::run_streaming_task(*model.streaming, request, sink);
    timed_result.result = std::move(result);
    timed_result.wall_ms = elapsed_ms(started);
    if (!timed_result.ttft_ms.has_value() && task_result_has_output(timed_result.result)) {
        timed_result.ttft_ms = timed_result.wall_ms;
    }
    return timed_result;
}

ServerState::TimedTaskResult ServerState::run_streaming_model(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
    std::optional<int> busy_timeout_ms) {
    return run_streaming_model_impl(model, request, nullptr, event_sink, busy_timeout_ms);
}

ServerState::TimedTaskResult ServerState::run_streaming_model_from(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    const minitts::app::AudioChunkStream & audio,
    const std::function<void(const engine::runtime::StreamEvent &)> & event_sink,
    std::optional<int> busy_timeout_ms) {
    return run_streaming_model_impl(model, request, &audio, event_sink, busy_timeout_ms);
}

HttpResponse ServerState::handle_speech(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto request = build_speech_request(model, body);
    if (body.find("stream_format") != nullptr || bool_field(body, "stream", false)) {
        return handle_speech_stream(model, request, body);
    }
    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    const auto timed_result = model_run_mode(model) == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request, {}, busy_timeout_ms)
        : run_model(model, request, busy_timeout_ms);
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
    if (model_run_mode(model) != engine::runtime::RunMode::Streaming) {
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

    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    LoadedModel * model_ptr = &model;
    auto stream_body = [this, model_ptr, request, busy_timeout_ms](HttpStreamWriter & writer) {
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
            },
            busy_timeout_ms);
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
    return chunked_audio_response([this, model_ptr, request, busy_timeout_ms](HttpStreamWriter & writer) {
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
            },
            busy_timeout_ms);
        if (!wrote_audio) {
            throw std::runtime_error("streaming speech model produced no audio delta events");
        }
    });
}

HttpResponse ServerState::handle_speech_live(const HttpRequest & request) {
    if (request.body_stream == nullptr) {
        return error_response(
            400,
            "live speech requires an incrementally delivered body (Transfer-Encoding: chunked)",
            "invalid_request_error");
    }

    LoadedModel * model_ptr = nullptr;
    int sample_rate = 16000;
    int channels = 1;
    std::optional<int> busy_timeout_ms;
    minitts::app::PcmSampleFormat sample_format = minitts::app::PcmSampleFormat::S16LE;
    std::string stream_format = "sse";
    engine::runtime::TaskRequest task_request;
    try {
        const std::string model_id = query_param(request.query, "model");
        if (model_id.empty()) {
            throw std::runtime_error("live speech requires a 'model' query parameter");
        }
        const std::string input = decoded_query_param(request.query, "input");
        if (input.empty()) {
            throw std::runtime_error("live speech requires an 'input' query parameter");
        }

        engine::io::json::Value::Object fields;
        fields.emplace("model", engine::io::json::Value::make_string(model_id));
        fields.emplace("input", engine::io::json::Value::make_string(input));
        const std::string language = decoded_query_param(request.query, "language");
        if (!language.empty()) {
            fields.emplace("language", engine::io::json::Value::make_string(language));
        }
        const std::string voice = decoded_query_param(request.query, "voice");
        if (!voice.empty()) {
            fields.emplace("voice", engine::io::json::Value::make_string(voice));
        }
        const auto body = engine::io::json::Value::make_object(std::move(fields));
        auto & model = require_model(body);
        if (model.task.mode != engine::runtime::RunMode::Streaming) {
            throw std::runtime_error(
                "live speech requires a model configured with mode=streaming: " +
                model.config.id);
        }
        model_ptr = &model;

        const auto parse_bounded_int = [&](const char * key, int fallback, int minimum, int maximum) {
            const std::string raw = query_param(request.query, key);
            if (raw.empty()) {
                return fallback;
            }
            long long value = 0;
            size_t consumed = 0;
            try {
                value = std::stoll(raw, &consumed);
            } catch (const std::exception &) {
                throw std::runtime_error(
                    std::string("live speech ") + key + " must be an integer");
            }
            if (consumed != raw.size()) {
                throw std::runtime_error(
                    std::string("live speech ") + key + " must be an integer");
            }
            if (value < minimum || value > maximum) {
                throw std::runtime_error(
                    std::string("live speech ") + key + " must be between " +
                    std::to_string(minimum) + " and " + std::to_string(maximum));
            }
            return static_cast<int>(value);
        };
        sample_rate = parse_bounded_int("sample_rate", 16000, 1000, 384'000);
        channels = parse_bounded_int("channels", 1, 1, 16);
        if (!query_param(request.query, "busy_timeout_ms").empty()) {
            busy_timeout_ms = parse_bounded_int(
                "busy_timeout_ms", 0, 0, std::numeric_limits<int>::max());
        }
        const std::string sample_format_name = query_param(request.query, "sample_format");
        sample_format = minitts::app::parse_pcm_sample_format(
            sample_format_name.empty() ? "s16le" : sample_format_name);
        const std::string response_format = query_param(request.query, "response_format").empty()
            ? "pcm"
            : query_param(request.query, "response_format");
        if (response_format != "pcm") {
            throw std::runtime_error("live speech currently supports response_format=pcm");
        }
        stream_format = query_param(request.query, "stream_format").empty()
            ? "sse"
            : query_param(request.query, "stream_format");
        if (stream_format != "sse" && stream_format != "audio") {
            throw std::runtime_error("live speech stream_format must be sse or audio");
        }

        task_request = build_speech_request(model, body);
        engine::runtime::AudioBuffer audio_contract;
        audio_contract.sample_rate = sample_rate;
        audio_contract.channels = channels;
        task_request.audio_input = std::move(audio_contract);
        const auto add_query_option = [&](const char * key, const char * option_key) {
            const std::string value = decoded_query_param(request.query, key);
            if (!value.empty()) {
                task_request.options[option_key] = value;
            }
        };
        add_query_option("seed", "seed");
        add_query_option("temperature", "temperature");
        add_query_option("top_k", "top_k");
        add_query_option("top_p", "top_p");
        add_query_option("max_tokens", "max_tokens");
        add_query_option("max_steps", "max_steps");
        add_query_option("repetition_penalty", "repetition_penalty");
        add_query_option("guidance_scale", "guidance_scale");
        add_query_option("num_inference_steps", "num_inference_steps");
        add_query_option("instructions", "system_prompt");
        add_query_option("reference_text", "reference_text");
        add_query_option("voice_id", "voice_id");
        add_query_option("system_prompt", "system_prompt");
        add_query_option("text_temperature", "text_temperature");
        add_query_option("text_top_k", "text_top_k");
        add_query_option("do_sample", "do_sample");
    } catch (const std::runtime_error & ex) {
        return error_response(400, ex.what(), "invalid_request_error");
    }

    std::istream * pcm_input = request.body_stream;
    if (stream_format == "audio") {
        return chunked_audio_response([this, model_ptr, task_request, pcm_input, sample_rate, channels, sample_format, busy_timeout_ms](
                                          HttpStreamWriter & writer) {
            const minitts::app::AudioStreamFormat format{sample_rate, channels};
            const auto audio = minitts::app::make_pcm_chunk_stream(*pcm_input, format, sample_format);
            bool wrote_audio = false;
            (void)run_streaming_model_from(
                *model_ptr,
                task_request,
                audio,
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
                },
                busy_timeout_ms);
            if (!wrote_audio) {
                throw std::runtime_error("live speech model produced no audio delta events");
            }
        });
    }
    return sse_response(
        [this, model_ptr, task_request, pcm_input, sample_rate, channels, sample_format, busy_timeout_ms](
            HttpStreamWriter & writer) {
            const minitts::app::AudioStreamFormat format{sample_rate, channels};
            const auto audio = minitts::app::make_pcm_chunk_stream(*pcm_input, format, sample_format);
            const auto request_started = Clock::now();
            std::optional<double> first_audio_ms;
            std::optional<double> input_end_ms;
            minitts::app::AudioChunkStream timed_audio;
            timed_audio.format = audio.format;
            timed_audio.read = [
                                   reader = audio.read,
                                   request_started,
                                   &input_end_ms
                               ](int64_t max_samples, std::vector<float> & samples) mutable {
                const bool has_more = reader(max_samples, samples);
                if (!has_more && !input_end_ms.has_value()) {
                    input_end_ms = elapsed_ms(request_started);
                }
                return has_more;
            };
            bool wrote_audio = false;
            const auto timed_result = run_streaming_model_from(
                *model_ptr,
                task_request,
                timed_audio,
                [&](const engine::runtime::StreamEvent & event) {
                    if (event.audio_output.has_value()) {
                        if (!first_audio_ms.has_value()) {
                            first_audio_ms = elapsed_ms(request_started);
                        }
                        const auto pcm = encode_pcm16_samples(*event.audio_output);
                        write_sse(
                            writer,
                            "{\"type\":\"speech.audio.delta\",\"audio\":" +
                                json_quote(base64_encode(pcm)) +
                                "}");
                        wrote_audio = true;
                    }
                    for (const auto & named : event.named_audio_outputs) {
                        if (!first_audio_ms.has_value()) {
                            first_audio_ms = elapsed_ms(request_started);
                        }
                        const auto pcm = encode_pcm16_samples(named.audio);
                        write_sse(
                            writer,
                            "{\"type\":\"speech.audio.delta\",\"audio\":" +
                                json_quote(base64_encode(pcm)) +
                                "}");
                        wrote_audio = true;
                    }
                },
                busy_timeout_ms);
            if (!wrote_audio) {
                throw std::runtime_error("live speech model produced no audio delta events");
            }
            write_sse(
                writer,
                "{\"type\":\"speech.audio.done\",\"timing\":" +
                    live_speech_timing_json(
                        require_ttft_ms(first_audio_ms),
                        require_input_end_ms(input_end_ms)) +
                    "}");
            write_sse_done(writer);
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
    const auto request = apply_default_request_options(
        model,
        build_openai_transcription_request(body, request_base_));
    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    if (bool_field(body, "stream", false)) {
        return run_transcription_stream(model, request, busy_timeout_ms);
    }
    return run_transcription(model, request, busy_timeout_ms);
}

// Accepts the same multipart/form-data shape OpenAI's Whisper API (and clients built against it,
// e.g. Open WebUI) send: a "file" part with the audio bytes, plus "model" and optional "language"
// fields. audio.cpp's native JSON request only takes a server-local path, so the uploaded bytes are
// spooled to a temp file and routed through the existing JSON request builder.
HttpResponse ServerState::handle_transcription_multipart(const std::string & body_text, const std::string & boundary) {
    const auto parts = parse_multipart_body(body_text, boundary);
    log_multipart_request_summary_if_enabled(config_, parts);

    const MultipartPart * file_part = nullptr;
    std::string model_id;
    std::string language;
    std::string prompt;
    std::optional<int> busy_timeout_ms;
    bool stream = false;
    for (const auto & part : parts) {
        if (part.name == "file") {
            file_part = &part;
        } else if (part.name == "model") {
            model_id = part.data;
        } else if (part.name == "language") {
            language = part.data;
        } else if (part.name == "prompt" || part.name == "text") {
            // Recognition-context biasing. "prompt" is the OpenAI
            // transcription field name; "text" matches the JSON request
            // builder, which already forwards it to request.text_input.
            prompt = part.data;
        } else if (part.name == "busy_timeout_ms") {
            try {
                busy_timeout_ms = std::stoi(part.data);
            } catch (const std::exception &) {
                throw std::runtime_error("multipart busy_timeout_ms field must be an integer");
            }
            if (*busy_timeout_ms < 0) {
                throw std::runtime_error("busy_timeout_ms must be >= 0 (0 means no client-side bound)");
            }
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
    if (!is_wav_upload_filename(file_part->filename)) {
        return error_response(
            400,
            "only WAV audio uploads are currently supported for transcription; MP3 support is planned",
            "invalid_request_error");
    }

    engine::io::json::Value::Object fields;
    fields.emplace("model", engine::io::json::Value::make_string(model_id));
    if (!language.empty()) {
        fields.emplace("language", engine::io::json::Value::make_string(language));
    }
    if (!prompt.empty()) {
        fields.emplace("text", engine::io::json::Value::make_string(prompt));
    }
    const auto body = engine::io::json::Value::make_object(std::move(fields));

    auto & model = require_model(body);
    const auto request = apply_default_request_options(
        model,
        build_openai_transcription_request(body, request_base_, &file_part->data));
    if (stream) {
        return run_transcription_stream(model, request, busy_timeout_ms);
    }
    return run_transcription(model, request, busy_timeout_ms);
}

HttpResponse ServerState::run_transcription(
    LoadedModel & model,
    const engine::runtime::TaskRequest & request,
    std::optional<int> busy_timeout_ms) {
    const auto timed_result = model_run_mode(model) == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request, {}, busy_timeout_ms)
        : run_model(model, request, busy_timeout_ms);
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
    const engine::runtime::TaskRequest & request,
    std::optional<int> busy_timeout_ms) {
    if (model_run_mode(model) != engine::runtime::RunMode::Streaming) {
        throw std::runtime_error("transcription stream=true requires a model configured with mode=streaming");
    }
    LoadedModel * model_ptr = &model;
    return sse_response([this, model_ptr, request, busy_timeout_ms](HttpStreamWriter & writer) {
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
            },
            busy_timeout_ms);
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

// Live PCM ingest. The client streams raw interleaved samples in a chunked request
// body while transcript deltas stream back as SSE on the same connection, so
// partials track capture instead of waiting for a finished upload. Same event shape
// as the file-backed `stream=true` path, so a client can share one SSE reader.
//
// Deliberately a single request rather than open/append/finish session endpoints:
// the busy lock is held for the length of a run, so a session spread across
// separate requests would pin the model while idling between a client's appends —
// and wedge it outright if that client vanished. One request bounds the lock by the
// lifetime of the connection.
HttpResponse ServerState::handle_transcription_live(const HttpRequest & request) {
    if (request.body_stream == nullptr) {
        return error_response(
            400,
            "live transcription requires an incrementally delivered body (Transfer-Encoding: chunked)",
            "invalid_request_error");
    }

    // Everything from here to the end of parameter parsing validates client input, so
    // a failure is the caller's fault and belongs in a 4xx. Left to propagate, these
    // would reach the generic handler and be reported as 500, which tells a client
    // to retry an identical request that cannot ever succeed.
    LoadedModel * model_ptr = nullptr;
    int sample_rate = 16000;
    int channels = 1;
    std::optional<int> busy_timeout_ms;
    minitts::app::PcmSampleFormat sample_format = minitts::app::PcmSampleFormat::S16LE;
    engine::runtime::TaskRequest task_request;
    try {
        const std::string model_id = query_param(request.query, "model");
        if (model_id.empty()) {
            throw std::runtime_error("live transcription requires a 'model' query parameter");
        }
        // There is no request body to carry JSON — it is all audio — so the parameters
        // arrive as query params and are re-shaped into the object require_model expects.
        engine::io::json::Value::Object fields;
        fields.emplace("model", engine::io::json::Value::make_string(model_id));
        const auto body = engine::io::json::Value::make_object(std::move(fields));
        auto & model = require_model(body);
        if (model.task.mode != engine::runtime::RunMode::Streaming) {
            throw std::runtime_error(
                "live transcription requires a model configured with mode=streaming: " +
                model.config.id);
        }
        model_ptr = &model;

        // These two are not merely descriptive: the streaming policy multiplies them
        // into a per-chunk sample count, which sizes a buffer allocated after the model
        // lock has been taken. Left unbounded, a request naming an absurd rate or
        // channel count overflows that multiplication or asks for a multi-terabyte
        // allocation while holding the lock, so both are range-checked at the edge.
        const auto parse_bounded_int = [&](const char * key, int fallback, int minimum, int maximum) {
            const std::string raw = query_param(request.query, key);
            if (raw.empty()) {
                return fallback;
            }
            // Parsed to the end of the string on purpose: stoi stops at the first
            // non-digit, so "16000junk" would silently become 16000. A misdeclared
            // format produces a confident, wrong transcript, so reject it instead.
            long long value = 0;
            size_t consumed = 0;
            try {
                value = std::stoll(raw, &consumed);
            } catch (const std::exception &) {
                throw std::runtime_error(
                    std::string("live transcription ") + key + " must be an integer");
            }
            if (consumed != raw.size()) {
                throw std::runtime_error(
                    std::string("live transcription ") + key + " must be an integer");
            }
            if (value < minimum || value > maximum) {
                throw std::runtime_error(
                    std::string("live transcription ") + key + " must be between " +
                    std::to_string(minimum) + " and " + std::to_string(maximum));
            }
            return static_cast<int>(value);
        };
        sample_rate = parse_bounded_int("sample_rate", 16000, 1000, 384'000);
        channels = parse_bounded_int("channels", 1, 1, 16);
        // The other routes take this in their JSON body; this one has no body to put
        // it in, so it arrives as a query param. Same meaning either way: a request
        // may shorten its own wait for the model lock but never lengthen it past the
        // configured ceiling — resolve_busy_timeout_ms clamps it.
        if (!query_param(request.query, "busy_timeout_ms").empty()) {
            busy_timeout_ms = parse_bounded_int(
                "busy_timeout_ms", 0, 0, std::numeric_limits<int>::max());
        }
        const std::string sample_format_name = query_param(request.query, "sample_format");
        sample_format = minitts::app::parse_pcm_sample_format(
            sample_format_name.empty() ? "s16le" : sample_format_name);

        // Format contract with no samples: prepare() takes the rate and channel count
        // from here, and the samples themselves arrive from the body. Same shape the
        // CLI's stdin path builds (app/cli/main.cpp:472-482).
        engine::runtime::AudioBuffer audio_contract;
        audio_contract.sample_rate = sample_rate;
        audio_contract.channels = channels;
        task_request.audio_input = std::move(audio_contract);
        const std::string language = query_param(request.query, "language");
        if (!language.empty()) {
            task_request.options["language"] = language;
            task_request.text_input = engine::runtime::Transcript{std::string(), language};
        }
        task_request = apply_default_request_options(model, std::move(task_request));
    } catch (const std::runtime_error & ex) {
        // Deliberately runtime_error and not exception: every rejection above is
        // thrown as one, while a genuine server fault inside this block is not.
        // std::bad_alloc derives from std::exception directly and std::out_of_range
        // from std::logic_error, so both keep propagating to the 500 path instead of
        // being mislabelled as the caller's mistake.
        return error_response(400, ex.what(), "invalid_request_error");
    }

    std::istream * pcm_input = request.body_stream;
    return sse_response(
        [this, model_ptr, task_request, pcm_input, sample_rate, channels, sample_format, busy_timeout_ms](
            HttpStreamWriter & writer) {
            const minitts::app::AudioStreamFormat format{sample_rate, channels};
            const auto audio = minitts::app::make_pcm_chunk_stream(*pcm_input, format, sample_format);
            const auto timed_result = run_streaming_model_from(
                *model_ptr,
                task_request,
                audio,
                [&](const engine::runtime::StreamEvent & event) {
                    if (!event.partial_text.has_value() || event.partial_text->text.empty()) {
                        return;
                    }
                    write_sse(
                        writer,
                        "{\"type\":\"transcript.text.delta\",\"delta\":" +
                            json_quote(event.partial_text->text) +
                            "}");
                },
                busy_timeout_ms);
            if (!timed_result.result.text_output.has_value()) {
                throw std::runtime_error("live transcription result did not contain transcript text");
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
    const auto request = apply_default_request_options(
        model,
        minitts::cli::build_request_from_json(
            request_json != nullptr ? *request_json : body,
            request_base_));
    const auto busy_timeout_ms = parse_busy_timeout_override(body);
    const auto timed_result = model_run_mode(model) == engine::runtime::RunMode::Streaming
        ? run_streaming_model(model, request, {}, busy_timeout_ms)
        : run_model(model, request, busy_timeout_ms);
    return json_response(task_result_json(timed_result.result, timed_result.wall_ms));
}

HttpResponse ServerState::handle_generic_stream(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    auto & model = require_model(body);
    const auto * request_json = body.find("request");
    const auto request = apply_default_request_options(
        model,
        minitts::cli::build_request_from_json(
            request_json != nullptr ? *request_json : body,
            request_base_));
    std::vector<engine::runtime::StreamEvent> events;
    const auto timed_result = run_streaming_model(
        model,
        request,
        [&](const engine::runtime::StreamEvent & event) {
            events.push_back(event);
        },
        parse_busy_timeout_override(body));
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
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    const std::string model_id = query_param(request.query, "model");
    std::vector<std::string> voices;

    size_t model_idx = SIZE_MAX;
    if (!model_id.empty()) {
        const auto it = model_index_.find(model_id);
        if (it != model_index_.end()) {
            model_idx = it->second;
        }
    } else if (models_.size() == 1) {
        model_idx = 0;
    }
    if (model_idx != SIZE_MAX) {
        const auto & model = *models_.at(model_idx);
        std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
        for (const auto & [name, preset] : model.voice_presets) {
            (void) preset;
            voices.push_back(name);
        }
        const auto embeddings_dir = model.config.path / "embeddings";
        std::error_code ec;
        if (std::filesystem::is_directory(embeddings_dir, ec)) {
            for (const auto & entry : std::filesystem::directory_iterator(embeddings_dir, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                    voices.push_back(entry.path().stem().string());
                }
            }
        }
    }
    if (config_.voice_dir.has_value()) {
        std::error_code ec;
        if (std::filesystem::is_directory(*config_.voice_dir, ec)) {
            for (const auto & entry : std::filesystem::directory_iterator(*config_.voice_dir, ec)) {
                if (entry.is_regular_file() && entry.path().extension() == ".wav") {
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
    std::lock_guard<std::mutex> state_lock(models_mutex_);
    std::ostringstream out;
    out << "{\"object\":\"list\",\"data\":[";
    for (size_t i = 0; i < models_.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        const auto & model = *models_[i];
        std::shared_lock<std::shared_mutex> metadata_lock(model.metadata_mutex);
        out << "{\"id\":" << json_quote(model.config.id)
            << ",\"object\":\"model\""
            << ",\"owned_by\":\"engine\""
            << ",\"family\":" << json_quote(model.config.family)
            << ",\"task\":" << json_quote(engine::runtime::to_string(model.task.task))
            << ",\"mode\":" << json_quote(engine::runtime::to_string(model.task.mode))
            << ",\"loaded\":" << (model.loaded.load() ? "true" : "false")
            << ",\"path\":" << json_quote(model.config.path.string())
            << "}";
    }
    out << "]}";
    return out.str();
}

std::string ServerState::get_allowed_origin(const HttpRequest & request) const {
    // TODO: Handle lists of specific origins.
    if (config_.cors_origins == "*") {
        if (const auto it = request.headers.find("origin"); it != request.headers.end()) {
            return it->second;
        }
    }
    return "";
}

void ServerState::LoadedModel::unload() {
	offline = nullptr;
    streaming = nullptr;
    session.reset();
    model.reset();
}

HttpResponse ServerState::handle_unload_models(const std::string & body_text) {
    const auto body = engine::io::json::parse(body_text);
    const auto * ids = body.find("model_ids");
    if (ids == nullptr || !ids->is_array()) {
        return error_response(400, "request requires a 'model_ids' string array", "invalid_request_error");
    }

    std::vector<std::string> unloaded;
    std::vector<std::string> not_found;

    for (const auto & id_val : ids->as_array()) {
        if (!id_val.is_string()) {
            return error_response(400, "each element of 'model_ids' must be a string", "invalid_request_error");
        }
        const std::string id = id_val.as_string();
        const auto it = model_index_.find(id);
        if (it == model_index_.end()) {
            not_found.push_back(id);
            continue;
        }
        LoadedModel & model = *models_.at(it->second);
        // Only unload if the model is currently loaded in memory. Acquire the busy
        // lock for the duration of the unload so no inference starts mid-operation.
        if (model.session != nullptr) {
            [[maybe_unused]] BusyGuard::Lock lock = model.busy.acquire(0, model.config.id);
            model.unload();
            unloaded.push_back(id);
        }
    }

    std::ostringstream out;
    out << "{\"unloaded\":[";
    for (size_t i = 0; i < unloaded.size(); ++i) {
        if (i != 0) { out << ","; }
        out << json_quote(unloaded[i]);
    }
    out << "],\"not_found\":";
    out << "[";
    for (size_t i = 0; i < not_found.size(); ++i) {
        if (i != 0) { out << ","; }
        out << json_quote(not_found[i]);
    }
    out << "]}";
    return json_response(out.str());
}

HttpResponse ServerState::handle_unload_all_models() {
    std::vector<std::string> unloaded;

    for (auto & model : models_) {
        if (model->session != nullptr) {
            [[maybe_unused]] BusyGuard::Lock lock = model->busy.acquire(0, model->config.id);
            model->unload();
            unloaded.push_back(model->config.id);
        }
    }

    std::ostringstream out;
    out << "{\"unloaded\":[";
    for (size_t i = 0; i < unloaded.size(); ++i) {
        if (i != 0) { out << ","; }
        out << json_quote(unloaded[i]);
    }
    out << "]}";
    return json_response(out.str());
}

}  // namespace minitts::server
