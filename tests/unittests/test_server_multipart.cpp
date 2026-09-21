#include "multipart.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using minitts::server::extract_multipart_boundary;
using minitts::server::parse_multipart_body;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_extract_multipart_boundary() {
    require(
        extract_multipart_boundary("multipart/form-data; boundary=WebKitBoundary123") == std::string("WebKitBoundary123"),
        "unquoted boundary");
    require(
        extract_multipart_boundary("multipart/form-data; boundary=\"abc def\"") == std::string("abc def"),
        "quoted boundary");
    require(
        extract_multipart_boundary("multipart/form-data; charset=utf-8; boundary=xyz") == std::string("xyz"),
        "boundary after other parameters");
    require(
        !extract_multipart_boundary("application/json").has_value(),
        "non-multipart content-type yields no boundary");
    require(
        !extract_multipart_boundary("multipart/form-data").has_value(),
        "multipart content-type without boundary param yields no boundary");
}

void test_parse_multipart_body_fields_and_file() {
    const std::string boundary = "BOUNDARY";
    const std::string body =
        "--BOUNDARY\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n"
        "\r\n"
        "qwen3-asr\r\n"
        "--BOUNDARY\r\n"
        "Content-Disposition: form-data; name=\"language\"\r\n"
        "\r\n"
        "en\r\n"
        "--BOUNDARY\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n"
        "Content-Type: audio/wav\r\n"
        "\r\n"
        "RIFF....WAVEfmt \r\n"
        "--BOUNDARY--\r\n";

    const auto parts = parse_multipart_body(body, boundary);
    require(parts.size() == 3, "expected three parts");

    require(parts[0].name == "model", "first part name");
    require(parts[0].filename.empty(), "first part has no filename");
    require(parts[0].data == "qwen3-asr", "first part data");

    require(parts[1].name == "language", "second part name");
    require(parts[1].data == "en", "second part data");

    require(parts[2].name == "file", "third part name");
    require(parts[2].filename == "clip.wav", "third part filename");
    require(parts[2].data == "RIFF....WAVEfmt ", "third part data (binary-ish, no trailing CRLF)");
}

void test_parse_multipart_body_binary_with_embedded_nul() {
    const std::string boundary = "BOUNDARY";
    std::string payload;
    payload.push_back('\x00');
    payload.push_back('\x01');
    payload.push_back('\xff');
    payload += "middle";
    payload.push_back('\x00');

    std::string body = "--BOUNDARY\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"blob.bin\"\r\n";
    body += "\r\n";
    body += payload;
    body += "\r\n--BOUNDARY--\r\n";

    const auto parts = parse_multipart_body(body, boundary);
    require(parts.size() == 1, "expected a single part");
    require(parts[0].filename == "blob.bin", "binary part filename");
    require(parts[0].data.size() == payload.size(), "binary part data length preserved, including embedded NUL bytes");
    require(parts[0].data == payload, "binary part data is byte-exact");
}

void test_parse_multipart_body_only_accepts_delimiter_lines() {
    const std::string boundary = "BINARY_BOUNDARY";
    const std::string inline_marker = "--" + boundary;

    std::string payload;
    payload.push_back('\x00');
    payload.push_back('\x80');
    payload.push_back('\xff');
    payload += "prefix" + inline_marker + "-owned";
    payload.push_back('\r');
    payload += "bare-cr\nowned-newline\n";
    payload += "x" + inline_marker + "--inline-closing";
    for (int i = 0; i < 2000; ++i) {
        payload += "inline-" + inline_marker + (i % 2 == 0 ? "--" : "") + "-marker";
    }
    payload.push_back('\x01');

    // The preamble and filename both contain boundary-looking bytes. Neither is
    // a delimiter because the token does not begin a line in either location.
    std::string body = "preamble " + inline_marker + " bytes\r\n";
    body += "still preamble\r\n";
    body += inline_marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"inline-";
    body += inline_marker + "--name.bin\"\r\n";
    body += "Content-Type: application/octet-stream\r\n\r\n";
    body += payload;
    body += "\r\n" + inline_marker + "\r\n";
    body += "Content-Disposition: form-data; name=\"later\"\n\n";
    body += "kept-after-inline-markers\n";
    body += inline_marker + "--\n";

    const auto parts = parse_multipart_body(body, boundary);
    require(parts.size() == 2, "inline boundary-looking bytes must not create extra parts");
    require(parts[0].name == "file", "binary part name");
    require(parts[0].filename == "inline-" + inline_marker + "--name.bin", "filename preserves inline marker");
    require(parts[0].data == payload, "binary payload with inline markers is byte-exact");
    require(parts[1].name == "later", "later legitimate part remains reachable");
    require(parts[1].data == "kept-after-inline-markers", "later part data");
}

void test_parse_multipart_body_no_boundary_match() {
    require(parse_multipart_body("not a multipart body", "BOUNDARY").empty(), "no matching boundary yields no parts");
}

}  // namespace

int main() {
    try {
        test_extract_multipart_boundary();
        test_parse_multipart_body_fields_and_file();
        test_parse_multipart_body_binary_with_embedded_nul();
        test_parse_multipart_body_only_accepts_delimiter_lines();
        test_parse_multipart_body_no_boundary_match();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "server_multipart_test passed\n";
    return 0;
}
