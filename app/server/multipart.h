#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace minitts::server {

// One part of a multipart/form-data body, e.g. a form field or an uploaded file.
struct MultipartPart {
    std::string name;      // Content-Disposition "name" parameter.
    std::string filename;  // Content-Disposition "filename" parameter; empty for plain fields.
    std::string data;      // Raw part payload.
};

// Returns the boundary token from a "Content-Type: multipart/form-data; boundary=..." header,
// or std::nullopt if the header does not describe a multipart body.
std::optional<std::string> extract_multipart_boundary(const std::string & content_type);

// Splits a multipart/form-data body into its constituent parts using the given boundary.
std::vector<MultipartPart> parse_multipart_body(const std::string & body, const std::string & boundary);

}  // namespace minitts::server
