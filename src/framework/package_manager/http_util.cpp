#include "http_util.h"

#include "httplib.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace engine::package_manager::detail {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string getenv_text(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

std::string huggingface_token() {
    std::string token = getenv_text("HF_TOKEN");
    if (token.empty()) {
        token = getenv_text("HUGGING_FACE_HUB_TOKEN");
    }
    if (!token.empty()) {
        return token;
    }
#ifdef _WIN32
    const auto home = getenv_text("USERPROFILE");
#else
    const auto home = getenv_text("HOME");
#endif
    if (!home.empty()) {
        std::ifstream input(std::filesystem::path(home) / ".cache" / "huggingface" / "token");
        if (input) {
            std::getline(input, token);
        }
    }
    return token;
}

namespace {

bool unreserved(unsigned char ch) {
    return std::isalnum(ch) != 0 || ch == '-' || ch == '_' || ch == '.' || ch == '~';
}

}  // namespace

std::string url_encode(std::string_view value, bool keep_slash) {
    constexpr char hex[] = "0123456789ABCDEF";
    std::string out;
    for (const unsigned char ch : value) {
        if (unreserved(ch) || (keep_slash && ch == '/')) {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('%');
            out.push_back(hex[ch >> 4]);
            out.push_back(hex[ch & 0xf]);
        }
    }
    return out;
}

std::string huggingface_base_url() {
    auto base = getenv_text("AUDIOCPP_HF_BASE_URL");
    if (base.empty()) base = "https://huggingface.co";
    while (!base.empty() && base.back() == '/') base.pop_back();
    return base;
}

HttpUrl parse_http_url(const std::string & url) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid model URL: no scheme");
    }

    HttpUrl result;
    result.scheme = lower(url.substr(0, scheme_end));
    if (result.scheme != "http" && result.scheme != "https") {
        throw std::runtime_error("unsupported model URL scheme: " + result.scheme);
    }

    const auto authority_start = scheme_end + 3;
    const auto path_start = url.find('/', authority_start);
    auto authority = url.substr(authority_start,
        path_start == std::string::npos ? std::string::npos : path_start - authority_start);
    result.path = path_start == std::string::npos ? "/" : url.substr(path_start);
    if (authority.empty()) throw std::runtime_error("invalid model URL: no host");

    std::string port;
    if (authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string::npos) throw std::runtime_error("invalid IPv6 model URL");
        result.host = authority.substr(1, close - 1);
        if (close + 1 < authority.size()) {
            if (authority[close + 1] != ':') throw std::runtime_error("invalid model URL authority");
            port = authority.substr(close + 2);
        }
    } else {
        const auto colon = authority.rfind(':');
        if (colon != std::string::npos && authority.find(':') == colon) {
            result.host = authority.substr(0, colon);
            port = authority.substr(colon + 1);
        } else {
            result.host = authority;
        }
    }
    if (result.host.empty()) throw std::runtime_error("invalid model URL: no host");

    try {
        result.port = port.empty() ? (result.scheme == "https" ? 443 : 80) : std::stoi(port);
    } catch (...) {
        throw std::runtime_error("invalid model URL port");
    }
    if (result.port <= 0 || result.port > 65535) throw std::runtime_error("invalid model URL port");
    return result;
}

std::string http_origin(const HttpUrl & url) {
    const auto host = url.host.find(':') == std::string::npos ? url.host : "[" + url.host + "]";
    return url.scheme + "://" + host + ":" + std::to_string(url.port);
}

void configure_client(httplib::Client & client, long read_timeout_seconds) {
    client.set_follow_location(true);
    client.set_connection_timeout(60, 0);
    client.set_read_timeout(read_timeout_seconds, 0);
    client.set_write_timeout(60, 0);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    client.enable_server_certificate_verification(true);
#endif
}

}  // namespace engine::package_manager::detail
