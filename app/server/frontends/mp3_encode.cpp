#include "../frontend.h"

#include "common.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/io/json.h"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#include <lame/lame.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace minitts::server {
namespace {

using engine::io::json::Value;

bool is_speech_request(const HttpRequest & request) {
    return request.method == "POST" && request.path == "/v1/audio/speech";
}

bool core_speech_format(std::string_view response_format) {
    return response_format == "wav" ||
           response_format == "json" ||
           response_format == "b64_json";
}

std::string speech_response_format(const HttpRequest & request) {
    if (!is_speech_request(request) || !frontends::is_json_request(request)) {
        return {};
    }
    const auto body = engine::io::json::parse(request.body);
    if (!body.is_object()) {
        return {};
    }
    return frontends::lower_ascii(engine::io::json::optional_string(body, "response_format", "wav"));
}

class LameEncoder {
public:
    LameEncoder() : handle_(lame_init()) {
        if (handle_ == nullptr) {
            throw std::runtime_error("failed to initialize MP3 encoder");
        }
    }

    LameEncoder(const LameEncoder &) = delete;
    LameEncoder & operator=(const LameEncoder &) = delete;

    ~LameEncoder() {
        if (handle_ != nullptr) {
            lame_close(handle_);
        }
    }

    lame_t get() const { return handle_; }

private:
    lame_t handle_ = nullptr;
};

std::string encode_mp3(const engine::audio::WavData & audio) {
    if (audio.sample_rate <= 0) {
        throw std::runtime_error("MP3 output sample rate must be positive");
    }
    if (audio.channels != 1 && audio.channels != 2) {
        throw std::runtime_error("MP3 output supports mono or stereo audio");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("MP3 output sample count must be divisible by channel count");
    }

    const auto frame_count = static_cast<int>(audio.samples.size() / static_cast<size_t>(audio.channels));
    LameEncoder encoder;
    if (lame_set_in_samplerate(encoder.get(), audio.sample_rate) < 0 ||
        lame_set_num_channels(encoder.get(), audio.channels) < 0 ||
        lame_set_VBR(encoder.get(), vbr_default) < 0 ||
        lame_set_quality(encoder.get(), 2) < 0 ||
        lame_init_params(encoder.get()) < 0) {
        throw std::runtime_error("failed to configure MP3 encoder");
    }

    std::vector<unsigned char> encoded(static_cast<size_t>(1.25 * frame_count) + 7200);
    int written = 0;
    if (audio.channels == 1) {
        written = lame_encode_buffer_ieee_float(
            encoder.get(),
            audio.samples.data(),
            nullptr,
            frame_count,
            encoded.data(),
            static_cast<int>(encoded.size()));
    } else {
        written = lame_encode_buffer_interleaved_ieee_float(
            encoder.get(),
            audio.samples.data(),
            frame_count,
            encoded.data(),
            static_cast<int>(encoded.size()));
    }
    if (written < 0) {
        throw std::runtime_error("failed to encode MP3 output");
    }

    int offset = written;
    const int flushed = lame_encode_flush(
        encoder.get(),
        encoded.data() + offset,
        static_cast<int>(encoded.size()) - offset);
    if (flushed < 0) {
        throw std::runtime_error("failed to flush MP3 output");
    }
    offset += flushed;
    return std::string(reinterpret_cast<const char *>(encoded.data()), static_cast<size_t>(offset));
}

class Mp3EncodeModule final : public ServerFrontendModule {
public:
    std::string_view name() const override { return "mp3_encode"; }

    std::optional<FrontendPreContract> pre_contract() const override {
        return FrontendPreContract{
            "POST",
            "/v1/audio/speech",
            frontend_contracts::client_mp3_speech_request,
            frontend_contracts::core_wav_speech_request,
        };
    }

    std::optional<FrontendPostContract> post_contract() const override {
        return FrontendPostContract{
            "POST",
            "/v1/audio/speech",
            frontend_contracts::core_wav_speech_response,
            frontend_contracts::client_mp3_speech_response,
        };
    }

    void pre_process(ServerFrontendContext & context, ServerFrontendRequest & request) override {
        (void) context;
        if (!is_speech_request(request.request) || !frontends::is_json_request(request.request)) {
            return;
        }

        const auto body = engine::io::json::parse(request.request.body);
        if (!body.is_object()) {
            return;
        }

        auto fields = body.as_object();
        const auto requested_format =
            frontends::lower_ascii(engine::io::json::optional_string(body, "response_format", "wav"));
        if (core_speech_format(requested_format)) {
            return;
        }
        if (requested_format != "mp3") {
            request.response = error_response(
                400,
                "server frontend supports speech response_format=mp3; unsupported response_format=" +
                    requested_format,
                "invalid_request_error");
            return;
        }
        if (body.find("stream_format") != nullptr || frontends::bool_field(body, "stream", false)) {
            request.response = error_response(
                400,
                "server frontend supports response_format=mp3 only for non-streaming speech",
                "invalid_request_error");
            return;
        }

        fields["response_format"] = Value::make_string("wav");
        request.request = frontends::json_forward_request(request.request, std::move(fields));
    }

    void post_process(ServerFrontendContext & context, ServerFrontendResponse & response) override {
        (void) context;
        if (speech_response_format(response.original_request) != "mp3") {
            return;
        }
        if (response.response.status != 200) {
            return;
        }
        if (frontends::media_type_from_header_value(response.response.content_type) != "audio/wav") {
            throw std::runtime_error("speech MP3 frontend expected core audio/wav response");
        }

        const auto wav = engine::audio::read_wav_f32(
            std::string_view(response.response.body.data(), response.response.body.size()));
        response.response.content_type = "audio/mpeg";
        response.response.body = encode_mp3(wav);
    }
};

std::unique_ptr<ServerFrontendModule> make_mp3_encode_module() {
    return std::make_unique<Mp3EncodeModule>();
}

} // namespace

void register_mp3_encode_module(ServerFrontendRegistry & registry) {
    registry.add(make_mp3_encode_module);
}

} // namespace minitts::server
