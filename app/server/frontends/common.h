#pragma once

#include "../http.h"

#include "engine/framework/io/json.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace minitts::server::frontends {

inline std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

inline std::string media_type_from_header_value(std::string value) {
    value = lower_ascii(std::move(value));
    const auto semicolon = value.find(';');
    if (semicolon != std::string::npos) {
        value.resize(semicolon);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.pop_back();
    }
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    return value.substr(start);
}

inline std::string media_type_from_content_type(const HttpRequest & request) {
    const auto it = request.headers.find("content-type");
    return it == request.headers.end() ? std::string() : media_type_from_header_value(it->second);
}

inline bool is_json_request(const HttpRequest & request) {
    const auto media_type = media_type_from_content_type(request);
    return media_type == "application/json" ||
           (media_type.size() > 5 && media_type.substr(media_type.size() - 5) == "+json");
}

inline HttpRequest json_forward_request(
    const HttpRequest & request,
    engine::io::json::Value::Object fields) {
    HttpRequest forwarded = request;
    forwarded.headers["content-type"] = "application/json";
    forwarded.body = engine::io::json::stringify(engine::io::json::Value::make_object(std::move(fields)));
    forwarded.headers["content-length"] = std::to_string(forwarded.body.size());
    return forwarded;
}

inline bool bool_field(const engine::io::json::Value & object, const std::string & key, bool default_value) {
    const auto * value = object.find(key);
    if (value == nullptr || value->is_null()) {
        return default_value;
    }
    if (value->is_bool()) {
        return value->as_bool();
    }
    if (value->is_string()) {
        const auto & str = value->as_string();
        if (str == "true" || str == "1") {
            return true;
        }
        if (str == "false" || str == "0") {
            return false;
        }
    }
    throw std::runtime_error("field '" + key + "' must be a boolean");
}

} // namespace minitts::server::frontends
