#include "../frontend.h"
#include "../multipart.h"

#include "common.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_ENCODING
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_THREADING
#define MA_NO_WAV
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace minitts::server {
namespace {

using engine::io::json::Value;

enum class EncodedAudioFormat {
    Unknown,
    Mp3,
    Flac,
};

std::string lower_extension(const std::filesystem::path & path) {
    return frontends::lower_ascii(path.extension().string());
}

EncodedAudioFormat format_from_extension(std::string_view ext) {
    if (ext == ".mp3" || ext == ".mpa" || ext == ".mpeg") {
        return EncodedAudioFormat::Mp3;
    }
    if (ext == ".flac") {
        return EncodedAudioFormat::Flac;
    }
    return EncodedAudioFormat::Unknown;
}

EncodedAudioFormat sniff_format(std::string_view data) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(data.data());
    if (data.size() >= 4 && bytes[0] == 'f' && bytes[1] == 'L' && bytes[2] == 'a' && bytes[3] == 'C') {
        return EncodedAudioFormat::Flac;
    }
    if (data.size() >= 3 && bytes[0] == 'I' && bytes[1] == 'D' && bytes[2] == '3') {
        return EncodedAudioFormat::Mp3;
    }
    if (data.size() >= 2 && bytes[0] == 0xff && (bytes[1] & 0xe0u) == 0xe0u) {
        return EncodedAudioFormat::Mp3;
    }
    return EncodedAudioFormat::Unknown;
}

EncodedAudioFormat sniff_format(const std::vector<uint8_t> & data) {
    const auto * bytes = reinterpret_cast<const char *>(data.data());
    return sniff_format(std::string_view(bytes, data.size()));
}

ma_encoding_format miniaudio_encoding_format(EncodedAudioFormat format) {
    switch (format) {
        case EncodedAudioFormat::Mp3:
            return ma_encoding_format_mp3;
        case EncodedAudioFormat::Flac:
            return ma_encoding_format_flac;
        case EncodedAudioFormat::Unknown:
            break;
    }
    throw std::runtime_error("unsupported encoded audio input format");
}

std::string format_name(EncodedAudioFormat format) {
    switch (format) {
        case EncodedAudioFormat::Mp3:
            return "MP3";
        case EncodedAudioFormat::Flac:
            return "FLAC";
        case EncodedAudioFormat::Unknown:
            break;
    }
    return "audio";
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open encoded audio input: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to size encoded audio input: " + path.string());
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty()) {
        input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!input) {
            throw std::runtime_error("failed to read encoded audio input: " + path.string());
        }
    }
    return data;
}

class MiniaudioDecoder {
public:
    MiniaudioDecoder(std::string_view label, std::string_view data, EncodedAudioFormat format) {
        auto config = ma_decoder_config_init(ma_format_f32, 0, 0);
        config.encodingFormat = miniaudio_encoding_format(format);
        const auto result = ma_decoder_init_memory(data.data(), data.size(), &config, &decoder_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error("failed to decode " + format_name(format) + " input: " + std::string(label));
        }
        initialized_ = true;
    }

    MiniaudioDecoder(const MiniaudioDecoder &) = delete;
    MiniaudioDecoder & operator=(const MiniaudioDecoder &) = delete;

    ~MiniaudioDecoder() {
        if (initialized_) {
            ma_decoder_uninit(&decoder_);
        }
    }

    ma_decoder * get() { return &decoder_; }

private:
    ma_decoder decoder_{};
    bool initialized_ = false;
};

engine::audio::WavData decode_encoded_audio(
    std::string_view label,
    std::string_view data,
    EncodedAudioFormat format) {
    if (data.empty()) {
        throw std::runtime_error("empty " + format_name(format) + " input: " + std::string(label));
    }

    MiniaudioDecoder decoder(label, data, format);

    ma_format sample_format{};
    ma_uint32 channels = 0;
    ma_uint32 sample_rate = 0;
    if (ma_decoder_get_data_format(decoder.get(), &sample_format, &channels, &sample_rate, nullptr, 0) != MA_SUCCESS) {
        throw std::runtime_error("failed to read " + format_name(format) + " input format: " + std::string(label));
    }
    if (sample_format != ma_format_f32 || channels == 0 || sample_rate == 0) {
        throw std::runtime_error("invalid decoded " + format_name(format) + " input format: " + std::string(label));
    }

    std::vector<float> samples;
    std::vector<float> frames(4096u * channels);
    for (;;) {
        ma_uint64 frames_read = 0;
        const auto result = ma_decoder_read_pcm_frames(decoder.get(), frames.data(), 4096, &frames_read);
        if (frames_read > 0) {
            samples.insert(
                samples.end(),
                frames.begin(),
                frames.begin() + static_cast<std::ptrdiff_t>(frames_read * channels));
        }
        if (result == MA_AT_END) {
            break;
        }
        if (result != MA_SUCCESS) {
            throw std::runtime_error("failed while decoding " + format_name(format) + " input: " + std::string(label));
        }
        if (frames_read == 0) {
            break;
        }
    }
    if (samples.empty()) {
        throw std::runtime_error("decoded " + format_name(format) + " input is empty: " + std::string(label));
    }

    engine::audio::WavData audio;
    audio.sample_rate = static_cast<int>(sample_rate);
    audio.channels = static_cast<int>(channels);
    audio.samples = std::move(samples);
    return audio;
}

engine::audio::WavData decode_encoded_audio(
    std::string_view label,
    const std::vector<uint8_t> & data,
    EncodedAudioFormat format) {
    const auto * bytes = reinterpret_cast<const char *>(data.data());
    return decode_encoded_audio(label, std::string_view(bytes, data.size()), format);
}

std::filesystem::path write_temp_wav(
    ServerFrontendContext & context,
    std::string_view filename,
    const engine::audio::WavData & audio) {
    std::string wav_name = std::filesystem::path(std::string(filename)).stem().string();
    if (wav_name.empty()) {
        wav_name = "audio";
    }
    wav_name += ".wav";
    const auto path = context.make_frontend_temp_path(wav_name);
    engine::audio::write_pcm16_wav(path, audio.sample_rate, audio.channels, audio.samples);
    return path;
}

std::optional<std::filesystem::path> convert_path_if_encoded_audio(
    ServerFrontendContext & context,
    const std::filesystem::path & request_path) {
    const auto path = context.resolve_request_path(request_path);
    const auto ext = lower_extension(path);
    if (ext == ".wav") {
        return std::nullopt;
    }
    auto format = format_from_extension(ext);
    if (format == EncodedAudioFormat::Unknown && !ext.empty()) {
        return std::nullopt;
    }

    const auto bytes = read_file_bytes(path);
    if (format == EncodedAudioFormat::Unknown) {
        format = sniff_format(bytes);
        if (format == EncodedAudioFormat::Unknown) {
            return std::nullopt;
        }
    }
    const auto decoded = decode_encoded_audio(path.string(), bytes, format);
    return write_temp_wav(context, path.filename().string(), decoded);
}

EncodedAudioFormat upload_format(const MultipartPart & part) {
    const auto ext = lower_extension(part.filename);
    if (ext == ".wav") {
        return EncodedAudioFormat::Unknown;
    }
    const auto format = format_from_extension(ext);
    if (format != EncodedAudioFormat::Unknown) {
        return format;
    }
    return ext.empty() ? sniff_format(part.data) : EncodedAudioFormat::Unknown;
}

bool rewrite_json_transcription(ServerFrontendContext & context, ServerFrontendRequest & request) {
    if (!frontends::is_json_request(request.request)) {
        return false;
    }

    const auto body = engine::io::json::parse(request.request.body);
    if (!body.is_object()) {
        return false;
    }

    Value::Object fields = body.as_object();
    const char * audio_key = nullptr;
    for (const char * key : {"audio", "audio_path", "file"}) {
        const auto it = fields.find(key);
        if (it != fields.end()) {
            if (!it->second.is_string()) {
                return false;
            }
            audio_key = key;
            break;
        }
    }
    if (audio_key == nullptr) {
        return false;
    }

    const auto converted = convert_path_if_encoded_audio(context, fields.at(audio_key).as_string());
    if (!converted.has_value()) {
        return false;
    }

    fields[audio_key] = Value::make_string(converted->string());
    request.request = frontends::json_forward_request(request.request, std::move(fields));
    return true;
}

bool rewrite_multipart_transcription(ServerFrontendContext & context, ServerFrontendRequest & request) {
    std::string content_type;
    if (const auto it = request.request.headers.find("content-type"); it != request.request.headers.end()) {
        content_type = it->second;
    }
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] audio_decode.multipart.enter content_type=" + content_type +
            " body_bytes=" + std::to_string(request.request.body.size()));
    }
    const auto boundary = extract_multipart_boundary(content_type);
    if (!boundary.has_value()) {
        if (engine::debug::log_enabled()) {
            engine::debug::log_message("[MP3_FRONTEND_DEBUG] audio_decode.multipart.no_boundary");
        }
        return false;
    }

    const auto parts = parse_multipart_body(request.request.body, *boundary);
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] audio_decode.multipart.parts count=" + std::to_string(parts.size()) +
            " boundary_bytes=" + std::to_string(boundary->size()));
    }
    const MultipartPart * file_part = nullptr;
    Value::Object fields;
    for (const auto & part : parts) {
        if (engine::debug::log_enabled()) {
            engine::debug::log_message(
                "[MP3_FRONTEND_DEBUG] audio_decode.multipart.part name=" + part.name +
                " filename=" + (part.filename.empty() ? std::string("<none>") : part.filename) +
                " bytes=" + std::to_string(part.data.size()));
        }
        if (part.name == "file") {
            file_part = &part;
        } else if (part.name == "model") {
            fields["model"] = Value::make_string(part.data);
        } else if (part.name == "language") {
            fields["language"] = Value::make_string(part.data);
        } else if (part.name == "prompt" || part.name == "text") {
            fields["text"] = Value::make_string(part.data);
        } else if (part.name == "busy_timeout_ms") {
            try {
                fields["busy_timeout_ms"] = Value::make_number(std::stoi(part.data));
            } catch (const std::exception &) {
                throw std::runtime_error("multipart busy_timeout_ms field must be an integer");
            }
        } else if (part.name == "stream") {
            if (part.data == "true" || part.data == "True" || part.data == "1") {
                fields["stream"] = Value::make_bool(true);
            } else if (part.data == "false" || part.data == "False" || part.data == "0") {
                fields["stream"] = Value::make_bool(false);
            } else {
                throw std::runtime_error("multipart transcription stream field must be true or false");
            }
        }
    }
    if (file_part == nullptr) {
        if (engine::debug::log_enabled()) {
            engine::debug::log_message("[MP3_FRONTEND_DEBUG] audio_decode.multipart.no_file_part");
        }
        return false;
    }

    const auto format = upload_format(*file_part);
    if (format == EncodedAudioFormat::Unknown) {
        if (engine::debug::log_enabled()) {
            engine::debug::log_message(
                "[MP3_FRONTEND_DEBUG] audio_decode.multipart.unknown_format filename=" +
                (file_part->filename.empty() ? std::string("<none>") : file_part->filename) +
                " bytes=" + std::to_string(file_part->data.size()));
        }
        return false;
    }
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] audio_decode.multipart.format " + format_name(format));
    }
    const auto model = fields.find("model");
    if (model == fields.end() || !model->second.is_string() || model->second.as_string().empty()) {
        throw std::runtime_error("multipart transcription request requires a 'model' field");
    }
    if (file_part->data.empty()) {
        throw std::runtime_error("multipart transcription request requires a non-empty 'file' field");
    }

    const auto decoded = decode_encoded_audio(
        file_part->filename.empty() ? std::string_view("multipart upload") : std::string_view(file_part->filename),
        file_part->data,
        format);
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] audio_decode.decoded sample_rate=" + std::to_string(decoded.sample_rate) +
            " channels=" + std::to_string(decoded.channels) +
            " samples=" + std::to_string(decoded.samples.size()));
    }
    const auto wav_path = write_temp_wav(context, file_part->filename, decoded);
    if (engine::debug::log_enabled()) {
        engine::debug::log_message("[MP3_FRONTEND_DEBUG] audio_decode.temp_wav " + wav_path.string());
    }
    fields["file"] = Value::make_string(wav_path.string());

    request.request = frontends::json_forward_request(request.request, std::move(fields));
    if (engine::debug::log_enabled()) {
        engine::debug::log_message(
            "[MP3_FRONTEND_DEBUG] audio_decode.forward_json body_bytes=" +
            std::to_string(request.request.body.size()));
    }
    return true;
}

class AudioDecodeModule final : public ServerFrontendModule {
public:
    std::string_view name() const override { return "audio_decode"; }

    std::optional<FrontendPreContract> pre_contract() const override {
        return FrontendPreContract{
            "POST",
            "/v1/audio/transcriptions",
            frontend_contracts::client_encoded_audio_request,
            frontend_contracts::core_wav_audio_request,
        };
    }

    void pre_process(ServerFrontendContext & context, ServerFrontendRequest & request) override {
        if (request.request.method != "POST" || request.request.path != "/v1/audio/transcriptions") {
            return;
        }
        if (rewrite_json_transcription(context, request)) {
            return;
        }
        (void) rewrite_multipart_transcription(context, request);
    }
};

std::unique_ptr<ServerFrontendModule> make_audio_decode_module() {
    return std::make_unique<AudioDecodeModule>();
}

} // namespace

void register_audio_decode_module(ServerFrontendRegistry & registry) {
    registry.add(make_audio_decode_module);
}

} // namespace minitts::server
