#pragma once

// Internal helpers shared by the native package manager's HTTP transports.
// Not installed and not part of the public package-manager API.

#include <stdexcept>
#include <string>
#include <string_view>

namespace httplib {
class Client;
}

namespace engine::package_manager::detail {

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("download cancelled by user") {}
};

struct HttpUrl {
    std::string scheme;
    std::string host;
    int port = 0;
    std::string path;
};

std::string lower(std::string value);
std::string getenv_text(const char * name);
std::string huggingface_token();
std::string url_encode(std::string_view value, bool keep_slash);
std::string huggingface_base_url();

HttpUrl parse_http_url(const std::string & url);
std::string http_origin(const HttpUrl & url);
void configure_client(httplib::Client & client, long read_timeout_seconds);

}  // namespace engine::package_manager::detail
