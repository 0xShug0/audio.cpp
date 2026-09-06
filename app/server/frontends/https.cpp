#include "../frontend.h"

#include "engine/framework/io/json.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace minitts::server {
namespace {

std::string lower_ascii(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::string raw_query_from_target(const httplib::Request & request) {
    const auto marker = request.target.find('?');
    if (marker == std::string::npos) {
        return {};
    }
    return request.target.substr(marker + 1);
}

HttpRequest to_http_request(const httplib::Request & request) {
    HttpRequest out;
    out.method = request.method;
    out.path = request.path;
    out.query = raw_query_from_target(request);
    out.body = request.body;
    for (const auto & header : request.headers) {
        out.headers[lower_ascii(header.first)] = header.second;
    }
    return out;
}

class HttplibStreamWriter final : public HttpStreamWriter {
public:
    explicit HttplibStreamWriter(httplib::DataSink & sink)
        : sink_(sink) {}

    void write(std::string_view data) override {
        if (data.empty()) {
            return;
        }
        if (!sink_.write(data.data(), data.size())) {
            throw std::runtime_error("HTTPS client disconnected while writing streamed response");
        }
    }

private:
    httplib::DataSink & sink_;
};

void copy_response_headers(const HttpResponse & source, httplib::Response & target) {
    for (const auto & [key, value] : source.headers) {
        const auto normalized = lower_ascii(key);
        if (normalized == "content-length" ||
            normalized == "transfer-encoding" ||
            normalized == "content-type") {
            continue;
        }
        target.set_header(key, value);
    }
}

std::string stream_error_event(const std::exception & ex) {
    return
        "data: {\"type\":\"error\",\"error\":{\"message\":" +
        engine::io::json::stringify_string(ex.what()) +
        "}}\n\n";
}

void send_response(HttpResponse response, httplib::Response & target) {
    target.status = response.status;
    copy_response_headers(response, target);
    if (!response.stream_body) {
        target.set_content(response.body, response.content_type);
        return;
    }

    auto stream_body = std::make_shared<std::function<void(HttpStreamWriter &)>>(
        std::move(response.stream_body));
    auto sent = std::make_shared<std::atomic_bool>(false);
    const auto content_type = response.content_type;
    target.set_chunked_content_provider(
        content_type,
        [stream_body, sent, content_type](size_t, httplib::DataSink & sink) {
            if (sent->exchange(true)) {
                return false;
            }
            HttplibStreamWriter writer(sink);
            try {
                (*stream_body)(writer);
            } catch (const std::exception & ex) {
                if (content_type.rfind("text/event-stream", 0) == 0) {
                    writer.write(stream_error_event(ex));
                } else {
                    std::cerr << "audiocpp_server HTTPS streaming response failed: " << ex.what() << "\n";
                }
            }
            sink.done();
            return true;
        });
}

void handle_request(
    const httplib::Request & request,
    httplib::Response & response,
    IHttpHandler & handler,
    uint64_t max_request_body_bytes) {
    if (request.body.size() > max_request_body_bytes) {
        send_response(
            error_response(413, "request body exceeds max_request_body_bytes", "request_too_large"),
            response);
        return;
    }
    send_response(handler.handle(to_http_request(request)), response);
}

} // namespace

void serve_frontend_https(
    const std::string & host,
    int port,
    IHttpHandler & handler,
    ShutdownRequested shutdown_requested,
    uint64_t max_request_body_bytes,
    const ServerFrontendHttpsConfig & config) {
    httplib::SSLServer server(
        config.cert_file.string().c_str(),
        config.key_file.string().c_str());
    if (!server.is_valid()) {
        throw std::runtime_error("could not initialize HTTPS frontend server from certificate/key files");
    }
    server.set_payload_max_length(static_cast<size_t>(max_request_body_bytes));
    server.set_idle_interval(std::chrono::milliseconds(250));

    const auto route = [&handler, max_request_body_bytes](const httplib::Request & request, httplib::Response & response) {
        try {
            handle_request(request, response, handler, max_request_body_bytes);
        } catch (const std::exception & ex) {
            send_response(error_response(500, ex.what(), "server_error"), response);
        }
    };
    server.Get(R"(.*)", route);
    server.Post(R"(.*)", route);
    server.Put(R"(.*)", route);
    server.Patch(R"(.*)", route);
    server.Delete(R"(.*)", route);
    server.Options(R"(.*)", route);

    if (!server.bind_to_port(host, port)) {
        throw std::runtime_error("could not bind HTTPS frontend server on " + host + ":" + std::to_string(port));
    }

    std::thread server_thread([&server] {
        if (!server.listen_after_bind()) {
            std::cerr << "audiocpp_server HTTPS frontend listener stopped before accepting requests\n";
        }
    });
    std::cout << "audiocpp_server HTTPS frontend listening on https://" << host << ":" << port << "\n";
    while (!shutdown_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    server.stop();
    if (server_thread.joinable()) {
        server_thread.join();
    }
    std::cout << "audiocpp_server stopped\n";
}

} // namespace minitts::server
