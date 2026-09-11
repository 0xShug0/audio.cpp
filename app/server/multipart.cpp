#include "multipart.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace minitts::server {
namespace {

std::string lower_ascii(std::string value) {
    for (char & ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::string trim(std::string value) {
    auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) { return !is_space(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) { return !is_space(ch); }).base(), value.end());
    return value;
}

// Extracts a `key="value"` or `key=value` parameter from a header value such as
// `form-data; name="file"; filename="clip.wav"`.
std::string extract_header_param(const std::string & header_value, const std::string & key) {
    const std::string needle = key + "=";
    const std::string lowered = lower_ascii(header_value);
    size_t pos = 0;
    while (true) {
        pos = lowered.find(needle, pos);
        if (pos == std::string::npos) {
            return {};
        }
        // Make sure we matched a whole parameter name, not a suffix (e.g. "filename" vs "name").
        if (pos == 0 || header_value[pos - 1] == ' ' || header_value[pos - 1] == ';') {
            break;
        }
        pos += needle.size();
    }
    size_t start = pos + needle.size();
    if (start < header_value.size() && header_value[start] == '"') {
        const size_t end = header_value.find('"', start + 1);
        if (end == std::string::npos) {
            return header_value.substr(start + 1);
        }
        return header_value.substr(start + 1, end - start - 1);
    }
    size_t end = header_value.find(';', start);
    if (end == std::string::npos) {
        end = header_value.size();
    }
    return trim(header_value.substr(start, end - start));
}

std::string strip_trailing_newline(std::string value) {
    if (value.size() >= 2 && value.compare(value.size() - 2, 2, "\r\n") == 0) {
        value.resize(value.size() - 2);
    } else if (!value.empty() && value.back() == '\n') {
        value.resize(value.size() - 1);
    }
    return value;
}

MultipartPart parse_part(const std::string & segment) {
    MultipartPart part;
    size_t header_end = segment.find("\r\n\r\n");
    size_t separator_len = 4;
    if (header_end == std::string::npos) {
        header_end = segment.find("\n\n");
        separator_len = 2;
    }
    if (header_end == std::string::npos) {
        return part;
    }

    part.data = segment.substr(header_end + separator_len);

    std::istringstream header_stream(segment.substr(0, header_end));
    std::string line;
    while (std::getline(header_stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = lower_ascii(trim(line.substr(0, colon)));
        const std::string value = trim(line.substr(colon + 1));
        if (key == "content-disposition") {
            part.name = extract_header_param(value, "name");
            part.filename = extract_header_param(value, "filename");
        }
    }
    return part;
}

}  // namespace

std::optional<std::string> extract_multipart_boundary(const std::string & content_type) {
    const std::string lowered = lower_ascii(content_type);
    if (lowered.rfind("multipart/", 0) != 0) {
        return std::nullopt;
    }
    const std::string key = "boundary=";
    const auto pos = lowered.find(key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    std::string value = content_type.substr(pos + key.size());
    const auto semicolon = value.find(';');
    if (semicolon != std::string::npos) {
        value = value.substr(0, semicolon);
    }
    value = trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        value = value.substr(1, value.size() - 2);
    }
    if (value.empty()) {
        return std::nullopt;
    }
    return value;
}

std::vector<MultipartPart> parse_multipart_body(const std::string & body, const std::string & boundary) {
    std::vector<MultipartPart> parts;
    const std::string delimiter = "--" + boundary;

    size_t pos = body.find(delimiter);
    if (pos == std::string::npos) {
        return parts;
    }
    pos += delimiter.size();

    while (true) {
        if (body.compare(pos, 2, "--") == 0) {
            break;
        }
        if (body.compare(pos, 2, "\r\n") == 0) {
            pos += 2;
        } else if (pos < body.size() && body[pos] == '\n') {
            pos += 1;
        }

        const size_t next_delim = body.find(delimiter, pos);
        if (next_delim == std::string::npos) {
            break;
        }

        const std::string segment = strip_trailing_newline(body.substr(pos, next_delim - pos));
        parts.push_back(parse_part(segment));
        pos = next_delim + delimiter.size();
    }

    return parts;
}

}  // namespace minitts::server
