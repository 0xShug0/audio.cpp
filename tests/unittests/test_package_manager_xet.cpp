// Offline tests for the native Hugging Face Xet download path. A local
// cpp-httplib fixture plays Hub, CAS, and xorb storage with xorbs built here
// from known bytes, so every reconstruction can be compared exactly.

#include "engine/framework/package_manager/manager.h"
#include "engine/framework/io/json.h"
#include "httplib.h"
#include "lz4frame.h"
#include "xet.h"

#include <openssl/evp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace xet = engine::package_manager::detail;
using Bytes = std::vector<uint8_t>;

namespace {

int failures = 0;

void check(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename F>
std::string error_of(F && f) {
    try {
        f();
    } catch (const std::exception & error) {
        return error.what();
    }
    return {};
}

void set_env(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

std::string hex(const unsigned char * data, size_t size) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out;
    for (size_t i = 0; i < size; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 0xf]);
    }
    return out;
}

std::string sha256(const Bytes & data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int length = 0;
    EVP_Digest(data.data(), data.size(), digest, &length, EVP_sha256(), nullptr);
    return hex(digest, length);
}

std::string fake_hash(int seed) {
    std::string out;
    for (int i = 0; i < 64; ++i) out.push_back("0123456789abcdef"[(seed * 7 + i * 13) % 16]);
    return out;
}

Bytes read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

Bytes bg4_group(const Bytes & data) {
    Bytes out;
    for (size_t g = 0; g < 4; ++g) {
        for (size_t i = g; i < data.size(); i += 4) out.push_back(data[i]);
    }
    return out;
}

Bytes lz4_frame(const Bytes & data) {
    Bytes out(LZ4F_compressFrameBound(data.size(), nullptr));
    const auto size = LZ4F_compressFrame(out.data(), out.size(), data.data(), data.size(), nullptr);
    if (LZ4F_isError(size)) throw std::runtime_error("LZ4F_compressFrame failed");
    out.resize(size);
    return out;
}

void put_le24(Bytes & out, size_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xff));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xff));
}

Bytes serialize_chunk(const Bytes & data, uint8_t scheme, uint8_t version = 0) {
    Bytes payload = scheme == 0 ? data : scheme == 1 ? lz4_frame(data) : lz4_frame(bg4_group(data));
    Bytes out{version};
    put_le24(out, payload.size());
    out.push_back(scheme);
    put_le24(out, data.size());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Somewhat compressible deterministic content.
Bytes make_data(size_t size, uint32_t seed) {
    std::mt19937 random(seed);
    Bytes out(size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<uint8_t>((random() % 7 == 0) ? random() : (i * 31 / 7) & 0x3f);
    }
    return out;
}

struct Xorb {
    std::string hash;
    std::vector<Bytes> chunks;          // decompressed chunk contents
    Bytes serialized;
    std::vector<uint64_t> chunk_offsets;  // serialized byte offset of each chunk (+ end)
};

Xorb make_xorb(const std::string & hash, const std::vector<std::pair<size_t, uint8_t>> & layout, uint32_t seed) {
    Xorb xorb;
    xorb.hash = hash;
    for (const auto & [size, scheme] : layout) {
        xorb.chunks.push_back(make_data(size, seed++));
        xorb.chunk_offsets.push_back(xorb.serialized.size());
        const auto chunk = serialize_chunk(xorb.chunks.back(), scheme);
        xorb.serialized.insert(xorb.serialized.end(), chunk.begin(), chunk.end());
    }
    xorb.chunk_offsets.push_back(xorb.serialized.size());
    return xorb;
}

struct TermSpec {
    size_t xorb;
    uint32_t start;
    uint32_t end;
};

// ---------------------------------------------------------------------------
// Fixture: Hub + CAS + xorb host on one local server.

struct Fixture {
    std::vector<Xorb> xorbs;
    std::vector<TermSpec> terms;
    Bytes file;
    std::string file_hash = fake_hash(99);
    std::string file_sha256;
    std::string served_sha256;  // guarded by mutex; tests may swap in a wrong value

    httplib::Server server;
    std::thread thread;
    std::string base;

    std::mutex mutex;
    int resolve_requests = 0;
    int token_requests = 0;
    int recipe_requests = 0;
    int xorb_requests = 0;
    int cdn_requests = 0;
    std::vector<std::string> leaks;

    std::atomic<int> v2_status{200};
    std::atomic<int> token_status{200};
    std::atomic<int> xorb_fail_status{0};   // next xorb request returns this once
    std::atomic<bool> xorb_ignore_range{false};
    std::atomic<bool> xorb_slow{false};
    std::atomic<bool> skew_lengths{false};  // move one byte between the first two terms
    // Streams xorb 0 slowly and rejects xorb 1 once xorb 0 is partly written,
    // so a refresh pass redoes a transfer that already reported progress.
    std::atomic<bool> refresh_mid_transfer{false};
    std::atomic<uint64_t> slow_streamed{0};

    Fixture() {
        // Mixed compression, BG4 with lengths % 4 == 1, 2, 3, and multi-chunk xorbs.
        xorbs.push_back(make_xorb(fake_hash(1), {{65536, 1}, {40001, 2}, {12345, 0}, {65539, 2},
                                                  {30002, 1}, {131072, 2}, {777, 0}}, 10));
        xorbs.push_back(make_xorb(fake_hash(2), {{50003, 2}, {65536, 0}, {9999, 1}, {65536, 2}}, 50));
        // Terms reuse chunks (dedup) and leave a gap in xorb 0 so its fetch
        // needs a multipart response.
        terms = {{0, 0, 2}, {1, 0, 2}, {0, 1, 2}, {0, 4, 7}, {1, 2, 4}, {0, 0, 1}};
        for (const auto & term : terms) {
            for (uint32_t c = term.start; c < term.end; ++c) {
                const auto & chunk = xorbs[term.xorb].chunks[c];
                file.insert(file.end(), chunk.begin(), chunk.end());
            }
        }
        file_sha256 = sha256(file);
        served_sha256 = file_sha256;
        routes();
        const int port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) throw std::runtime_error("fixture bind failed");
        base = "http://127.0.0.1:" + std::to_string(port);
        thread = std::thread([this] { server.listen_after_bind(); });
        while (!server.is_running()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ~Fixture() {
        server.stop();
        if (thread.joinable()) thread.join();
    }

    void note_leak(const httplib::Request & request, const std::string & where) {
        const auto auth = request.get_header_value("Authorization");
        std::lock_guard<std::mutex> lock(mutex);
        if (where == "hub" && auth.find("cas-secret") != std::string::npos) leaks.push_back("CAS token sent to Hub");
        if (where == "cas" && auth.find("hf-secret") != std::string::npos) leaks.push_back("HF token sent to CAS");
        if (where == "xorb" && !auth.empty()) leaks.push_back("credential sent to xorb host");
    }

    uint64_t term_length(const TermSpec & term) const {
        uint64_t length = 0;
        for (uint32_t c = term.start; c < term.end; ++c) length += xorbs[term.xorb].chunks[c].size();
        return length;
    }

    std::string recipe_json(uint64_t first, uint64_t last) const {
        std::ostringstream terms_json;
        std::map<size_t, std::set<uint32_t>> needed;
        uint64_t position = 0;
        bool any = false;
        uint64_t offset = 0;
        size_t index = 0;
        for (const auto & term : terms) {
            int64_t skew = 0;
            if (skew_lengths.load()) skew = index == 0 ? -1 : index == 1 ? 1 : 0;
            ++index;
            const uint64_t start = position;
            const uint64_t end = position + term_length(term);
            position = end;
            if (end <= first || start > last) continue;
            if (!any) offset = first - start;
            terms_json << (any ? "," : "") << R"({"hash":")" << xorbs[term.xorb].hash
                       << R"(","unpacked_length":)" << term_length(term) + skew << R"(,"range":{"start":)"
                       << term.start << R"(,"end":)" << term.end << "}}";
            any = true;
            for (uint32_t c = term.start; c < term.end; ++c) needed[term.xorb].insert(c);
        }
        std::ostringstream xorbs_json;
        bool first_xorb = true;
        for (const auto & [index, chunks] : needed) {
            const auto & xorb = xorbs[index];
            std::vector<std::pair<uint32_t, uint32_t>> runs;
            for (const auto chunk : chunks) {
                if (!runs.empty() && runs.back().second == chunk) ++runs.back().second;
                else runs.emplace_back(chunk, chunk + 1);
            }
            std::ostringstream ranges;
            std::string signed_range = "bytes=";
            for (size_t i = 0; i < runs.size(); ++i) {
                const auto byte_start = xorb.chunk_offsets[runs[i].first];
                const auto byte_end = xorb.chunk_offsets[runs[i].second] - 1;
                ranges << (i ? "," : "") << R"({"chunks":{"start":)" << runs[i].first << R"(,"end":)"
                       << runs[i].second << R"(},"bytes":{"start":)" << byte_start << R"(,"end":)" << byte_end << "}}";
                signed_range += (i ? "," : "") + std::to_string(byte_start) + "-" + std::to_string(byte_end);
            }
            const auto expires = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + 3600;
            xorbs_json << (first_xorb ? "" : ",") << '"' << xorb.hash << R"(":[{"url":")" << base << "/xorb/"
                       << xorb.hash << "?X-Xet-Signed-Range=" << httplib::encode_uri_component(signed_range)
                       << "&Expires=" << expires << R"(","ranges":[)" << ranges.str() << "]}]";
            first_xorb = false;
        }
        return R"({"offset_into_first_range":)" + std::to_string(offset) + R"(,"terms":[)" + terms_json.str() +
            R"(],"xorbs":{)" + xorbs_json.str() + "}}";
    }

    void routes() {
        server.Get(R"(/org/repo/resolve/main/(.*))", [this](const httplib::Request & request, httplib::Response & response) {
            note_leak(request, "hub");
            std::string sha;
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++resolve_requests;
                sha = served_sha256;
            }
            response.status = 302;
            response.set_header("Location", "/cdn/" + request.matches[1].str());
            response.set_header("X-Linked-Size", std::to_string(file.size()));
            response.set_header("X-Linked-ETag", "\"" + sha + "\"");
            response.set_header("X-Repo-Commit", "commit123");
            response.set_header("X-Xet-Hash", file_hash);
            response.set_header("Link", "<" + base + "/api/models/org/repo/xet-read-token/commit123>; rel=\"xet-auth\", <" +
                base + "/cas/v1/reconstructions/" + file_hash + ">; rel=\"xet-reconstruction-info\"");
        });
        server.Get("/api/models/org/repo/xet-read-token/commit123", [this](const httplib::Request & request, httplib::Response & response) {
            note_leak(request, "hub");
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++token_requests;
            }
            if (request.get_header_value("Authorization") != "Bearer hf-secret") {
                response.status = 401;
                return;
            }
            response.status = token_status.load();
            const auto exp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + 3600;
            response.set_content(R"({"accessToken":"cas-secret","exp":)" + std::to_string(exp) +
                R"(,"casUrl":")" + base + R"(/cas"})", "application/json");
        });
        server.Get(R"(/cas/v2/reconstructions/([0-9a-f]+))", [this](const httplib::Request & request, httplib::Response & response) {
            note_leak(request, "cas");
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++recipe_requests;
            }
            if (request.get_header_value("Authorization") != "Bearer cas-secret") {
                response.status = 401;
                return;
            }
            if (v2_status.load() != 200) {
                response.status = v2_status.load();
                return;
            }
            if (request.matches[1].str() != file_hash) {
                response.status = 404;
                return;
            }
            uint64_t first = 0;
            uint64_t last = file.size() - 1;
            const auto range = request.get_header_value("Range");
            if (!range.empty()) {
                const auto dash = range.find('-');
                first = std::stoull(range.substr(6, dash - 6));
                last = std::min<uint64_t>(std::stoull(range.substr(dash + 1)), file.size() - 1);
            }
            // The CAS Range names bytes of the reconstructed file, not of this
            // JSON body; keep httplib from applying it to the response.
            const_cast<httplib::Request &>(request).ranges.clear();
            response.status = 200;
            response.set_content(recipe_json(first, last), "application/json");
        });
        server.Get(R"(/xorb/([0-9a-f]+))", [this](const httplib::Request & request, httplib::Response & response) {
            note_leak(request, "xorb");
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++xorb_requests;
            }
            if (const int status = xorb_fail_status.exchange(0); status != 0) {
                response.status = status;
                return;
            }
            // The signed URL authorizes exactly one Range header value.
            if (request.get_header_value("Range") != request.get_param_value("X-Xet-Signed-Range")) {
                response.status = 403;
                return;
            }
            const auto found = std::find_if(xorbs.begin(), xorbs.end(),
                [&](const Xorb & xorb) { return xorb.hash == request.matches[1].str(); });
            if (found == xorbs.end()) {
                response.status = 404;
                return;
            }
            if (xorb_slow.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (refresh_mid_transfer.load()) {
                if (found != xorbs.begin()) {
                    for (int i = 0; i < 1000 && slow_streamed.load() < 120000; ++i) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                    refresh_mid_transfer = false;
                    response.status = 403;
                    return;
                }
                const auto body = std::make_shared<std::string>(found->serialized.begin(), found->serialized.end());
                response.set_content_provider(body->size(), "application/octet-stream",
                    [this, body](size_t offset, size_t length, httplib::DataSink & sink) {
                        const size_t piece = std::min<size_t>(length, 4096);
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                        slow_streamed += piece;
                        return sink.write(body->data() + offset, piece);
                    });
                return;
            }
            if (xorb_ignore_range.load()) response.status = 200;  // ignore Range: whole object, 200
            // Otherwise httplib answers 206 itself, multipart for several ranges.
            response.set_content(std::string(found->serialized.begin(), found->serialized.end()),
                "application/octet-stream");
        });
        server.Get(R"(/cdn/(.*))", [this](const httplib::Request &, httplib::Response & response) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                ++cdn_requests;
            }
            response.status = 200;
            response.set_content(std::string(file.begin(), file.end()), "application/octet-stream");
        });
    }

    xet::XetFileRequest request() const {
        xet::XetFileRequest out;
        out.repo = "org/repo";
        out.revision = "commit123";
        out.xet_hash = file_hash;
        out.auth_url = base + "/api/models/org/repo/xet-read-token/commit123";
        out.expected_sha256 = file_sha256;
        out.size = file.size();
        return out;
    }

    void reset_counters() {
        std::lock_guard<std::mutex> lock(mutex);
        resolve_requests = token_requests = recipe_requests = xorb_requests = cdn_requests = 0;
    }
};

xet::XetOptions fast_options() {
    xet::XetOptions options;
    options.concurrency = 4;
    options.initial_backoff_ms = 1;
    return options;
}

// ---------------------------------------------------------------------------
// Unit tests

void test_bg4() {
    for (size_t size = 0; size < 64; ++size) {
        const auto data = make_data(size, static_cast<uint32_t>(size));
        const auto grouped = bg4_group(data);
        Bytes out(size);
        xet::bg4_ungroup(grouped.data(), grouped.size(), out.data());
        check(out == data, "bg4 round trip for size " + std::to_string(size));
    }
    const Bytes grouped{'A', 'E', 'B', 'C', 'D'};  // input ABCDE -> groups AE, B, C, D
    Bytes out(5);
    xet::bg4_ungroup(grouped.data(), grouped.size(), out.data());
    check(std::string(out.begin(), out.end()) == "ABCDE", "bg4 remainder goes to the first groups");
}

struct DecodedChunks {
    std::vector<uint32_t> indices;
    Bytes data;
};

DecodedChunks decode(const Bytes & stream, uint32_t first, uint32_t count, size_t piece) {
    DecodedChunks out;
    xet::XetChunkDecoder decoder;
    decoder.begin(first, count, [&](uint32_t index, const uint8_t * data, size_t size) {
        out.indices.push_back(index);
        out.data.insert(out.data.end(), data, data + size);
    });
    std::mt19937 random(7);
    for (size_t offset = 0; offset < stream.size();) {
        const size_t take = std::min(stream.size() - offset, piece == 0 ? 1 + random() % 9000 : piece);
        decoder.feed(stream.data() + offset, take);
        offset += take;
    }
    decoder.end();
    return out;
}

void test_chunk_decoder() {
    const auto xorb = make_xorb(fake_hash(3), {{1001, 0}, {70000, 1}, {65537, 2}, {3, 2}, {131072, 1}}, 3);
    Bytes expected;
    for (const auto & chunk : xorb.chunks) expected.insert(expected.end(), chunk.begin(), chunk.end());
    for (const size_t piece : {size_t{1}, size_t{0}, xorb.serialized.size()}) {
        const auto decoded = decode(xorb.serialized, 10, 5, piece);
        check(decoded.data == expected, "chunk decoder output with piece size " + std::to_string(piece));
        check(decoded.indices == std::vector<uint32_t>({10, 11, 12, 13, 14}), "chunk decoder indices");
    }

    const auto bad = [&](Bytes stream, uint32_t count, const std::string & label) {
        const auto message = error_of([&] { decode(stream, 0, count, 0); });
        check(!message.empty(), "chunk decoder rejects " + label);
    };
    const auto one = serialize_chunk(make_data(100, 1), 1);
    bad(serialize_chunk(make_data(100, 1), 1, 1), 1, "unknown header version");
    auto scheme = one;
    scheme[4] = 3;
    bad(scheme, 1, "unknown compression scheme");
    auto size = one;
    size[5] = 99;  // declared uncompressed size 99 instead of 100
    bad(size, 1, "wrong uncompressed size");
    bad(Bytes(one.begin(), one.end() - 1), 1, "truncated chunk");
    bad(one, 2, "missing chunk");
    auto extra = one;
    extra.insert(extra.end(), one.begin(), one.end());
    bad(extra, 1, "extra chunk");
    auto corrupt = one;
    corrupt[12] ^= 0xff;
    bad(corrupt, 1, "corrupt LZ4 frame");
    Bytes huge{0};
    put_le24(huge, 10);
    huge.push_back(0);
    put_le24(huge, xet::kXetMaxChunkBytes + 1);
    bad(huge, 1, "oversized chunk");
}

struct ParsedParts {
    std::map<size_t, std::string> parts;
    std::vector<size_t> order;
};

ParsedParts parse_parts(const std::vector<xet::XetByteRange> & requested, const std::string & type,
                        const std::string & content_range, const std::string & body, size_t piece) {
    ParsedParts out;
    xet::XetRangeResponseParser::Callbacks callbacks;
    callbacks.begin = [&](size_t range) { out.order.push_back(range); };
    callbacks.data = [&](size_t range, const uint8_t * data, size_t size) {
        out.parts[range].append(reinterpret_cast<const char *>(data), size);
    };
    xet::XetRangeResponseParser parser(requested, type, content_range, callbacks);
    for (size_t offset = 0; offset < body.size(); offset += piece) {
        const auto take = std::min(piece, body.size() - offset);
        parser.feed(reinterpret_cast<const uint8_t *>(body.data()) + offset, take);
    }
    parser.finish();
    return out;
}

void test_range_parser() {
    const std::vector<xet::XetByteRange> requested{{0, 4}, {10, 12}, {20, 29}};
    // Parts out of order, quoted boundary, transport padding, preamble and epilogue.
    const std::string body =
        "preamble\r\n--b0und\r\nContent-Type: application/octet-stream\r\nContent-Range: bytes 10-12/100\r\n\r\n"
        "KLM\r\n--b0und \r\ncontent-range: bytes 20-29/100\r\n\r\nUVWXYZ0123\r\n--b0und\r\n"
        "Content-Range: bytes 0-4/*\r\n\r\nABCDE\r\n--b0und--\r\nepilogue";
    for (const size_t piece : {size_t{1}, size_t{3}, body.size()}) {
        const auto parsed = parse_parts(requested, "multipart/byteranges; boundary=\"b0und\"", "", body, piece);
        check(parsed.parts.at(0) == "ABCDE" && parsed.parts.at(1) == "KLM" && parsed.parts.at(2) == "UVWXYZ0123",
            "multipart parts map to requested ranges");
        check(parsed.order == std::vector<size_t>({1, 2, 0}), "multipart part order is preserved");
    }
    const auto single = parse_parts({{5, 9}}, "application/octet-stream", "bytes 5-9/50", "HELLO", 2);
    check(single.parts.at(0) == "HELLO", "single range body");

    const auto rejects = [&](const std::vector<xet::XetByteRange> & ranges, const std::string & type,
                             const std::string & content_range, const std::string & data, const std::string & label) {
        check(!error_of([&] { parse_parts(ranges, type, content_range, data, 1); }).empty(),
            "range parser rejects " + label);
    };
    const std::string mp = "multipart/byteranges; boundary=b";
    rejects({{5, 9}}, "application/octet-stream", "bytes 5-8/50", "HELL", "wrong single Content-Range");
    rejects({{5, 9}}, "application/octet-stream", "bytes 5-9/50", "HELLO!", "body longer than range");
    rejects({{5, 9}}, "application/octet-stream", "bytes 5-9/50", "HELL", "body shorter than range");
    rejects({{0, 1}, {4, 5}}, "application/octet-stream", "bytes 0-1/9", "ab", "non-multipart for many ranges");
    rejects({{0, 1}, {4, 5}}, mp, "", "--b\r\nContent-Range: bytes 0-1/9\r\n\r\nab\r\n--b--", "missing part");
    rejects({{0, 1}}, mp, "", "--b\r\nContent-Range: bytes 0-2/9\r\n\r\nabc\r\n--b--", "unrequested part");
    rejects({{0, 1}}, mp, "", "--b\r\n\r\nab\r\n--b--", "part without Content-Range");
    rejects({{0, 1}}, mp, "", "--b\r\nContent-Range: bytes 0-1/9\r\n\r\nab\r\n--x--", "bad delimiter");
    rejects({{0, 1}}, mp, "", "--b\r\nContent-Range: bytes 0-1/9\r\n\r\nab", "missing close delimiter");
    rejects({{0, 1}, {0, 1}}, mp, "",
        "--b\r\nContent-Range: bytes 0-1/9\r\n\r\nab\r\n--b\r\nContent-Range: bytes 0-1/9\r\n\r\nab\r\n--b\r\n"
        "Content-Range: bytes 0-1/9\r\n\r\nab\r\n--b--", "duplicate part");
    rejects({{0, 1}}, "multipart/byteranges", "", "", "missing boundary");
}

// ---------------------------------------------------------------------------
// Transfer tests against the fixture

void test_direct_transfer(Fixture & fixture, const fs::path & root) {
    const auto output = root / "direct.bin";
    uint64_t last_progress = 0;
    bool monotonic = true;
    xet::xet_download(fixture.request(), output, fast_options(), nullptr, [&](uint64_t bytes) {
        monotonic = monotonic && bytes >= last_progress;
        last_progress = bytes;
    });
    check(read_file(output) == fixture.file, "full Xet reconstruction matches");
    check(last_progress == fixture.file.size() && monotonic, "progress reports reconstructed bytes");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.recipe_requests == 1, "one reconstruction query for a small file");
        check(fixture.xorb_requests == 2, "each xorb fetched once despite repeated chunks");
    }

    // Ranged reconstruction batches that start inside terms and end inside chunks.
    for (const uint64_t segment : {uint64_t{4096}, uint64_t{100003}, uint64_t{200000}}) {
        auto options = fast_options();
        options.segment_bytes = segment;
        options.concurrency = 3;
        fixture.reset_counters();
        xet::xet_download(fixture.request(), output, options, nullptr, nullptr);
        check(read_file(output) == fixture.file, "segmented reconstruction matches, segment " + std::to_string(segment));
        std::lock_guard<std::mutex> lock(fixture.mutex);
        const auto expected = static_cast<int>((fixture.file.size() + segment - 1) / segment);
        check(fixture.recipe_requests == expected, "one reconstruction query per segment");
    }

    // A rejected signed URL triggers a recipe refresh.
    fixture.reset_counters();
    fixture.xorb_fail_status = 403;
    xet::xet_download(fixture.request(), output, fast_options(), nullptr, nullptr);
    check(read_file(output) == fixture.file, "reconstruction after signed URL refresh");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.recipe_requests == 2, "403 on a xorb URL re-queries the reconstruction");
    }

    // A refresh that redoes a partly finished transfer counts its bytes once.
    fixture.reset_counters();
    fixture.slow_streamed = 0;
    fixture.refresh_mid_transfer = true;
    uint64_t highest_progress = 0;
    last_progress = 0;
    monotonic = true;
    xet::xet_download(fixture.request(), output, fast_options(), nullptr, [&](uint64_t bytes) {
        monotonic = monotonic && bytes >= last_progress;
        last_progress = bytes;
        highest_progress = std::max(highest_progress, bytes);
    });
    check(read_file(output) == fixture.file, "reconstruction after mid-transfer refresh");
    check(fixture.slow_streamed.load() >= 120000, "refresh interrupted a partly written transfer");
    check(highest_progress == fixture.file.size() && last_progress == fixture.file.size() && monotonic,
        "progress is not counted twice across a refresh");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.recipe_requests == 2, "mid-transfer 403 re-queries the reconstruction");
    }

    // Transient errors are retried on the same URL.
    fixture.reset_counters();
    fixture.xorb_fail_status = 503;
    xet::xet_download(fixture.request(), output, fast_options(), nullptr, nullptr);
    check(read_file(output) == fixture.file, "reconstruction after transient xorb failure");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.recipe_requests == 1 && fixture.xorb_requests == 3, "503 retries without re-query");
    }

    fixture.xorb_ignore_range = true;
    check(!error_of([&] { xet::xet_download(fixture.request(), output, fast_options(), nullptr, nullptr); }).empty(),
        "200 instead of 206 is rejected");
    fixture.xorb_ignore_range = false;

    auto wrong = fixture.request();
    wrong.expected_sha256 = fake_hash(5);
    bool integrity = false;
    try {
        xet::xet_download(wrong, output, fast_options(), nullptr, nullptr);
    } catch (const xet::XetIntegrityError &) {
        integrity = true;
    } catch (...) {
    }
    check(integrity, "SHA-256 mismatch is an integrity error");

    fixture.skew_lengths = true;
    {
        auto options = fast_options();
        options.verify_sha256 = false;
        const auto message = error_of([&] { xet::xet_download(fixture.request(), output, options, nullptr, nullptr); });
        check(message.find("unpacked_length") != std::string::npos,
            "term unpacked_length mismatch is rejected: " + message);
    }
    fixture.skew_lengths = false;

    auto unverified = wrong;
    auto options = fast_options();
    options.verify_sha256 = false;
    check(error_of([&] { xet::xet_download(unverified, output, options, nullptr, nullptr); }).empty(),
        "verification can be disabled");

    fixture.v2_status = 501;
    bool unavailable = false;
    try {
        xet::xet_download(fixture.request(), output, fast_options(), nullptr, nullptr);
    } catch (const xet::XetUnavailable &) {
        unavailable = true;
    } catch (...) {
    }
    check(unavailable, "missing v2 endpoint reports XetUnavailable");
    fixture.v2_status = 200;

    std::lock_guard<std::mutex> lock(fixture.mutex);
    check(fixture.leaks.empty(), "no credential leaks: " + (fixture.leaks.empty() ? "" : fixture.leaks.front()));
}

void write_spec(const fs::path & root) {
    fs::create_directories(root / "model_specs");
    std::ofstream(root / "model_specs" / "demo.json") << R"({
      "family":"demo","display_name":"Demo","category":"tts",
      "status":"supported","tasks":["tts"],"modes":["offline"],
      "capabilities":{},"runtime":{},
      "package_defaults":{"download":{"kind":"huggingface_snapshot","repo":"org/repo"}},
      "packages":[{"id":"demo","display_name":"Demo","format":"gguf",
        "precision":"f32","target_directory":"Demo","files":["model.gguf"]}]
    })";
}

bool no_staging(const fs::path & models) {
    for (const auto & entry : fs::directory_iterator(models)) {
        if (entry.path().filename().string().rfind(".Demo.", 0) == 0) return false;
    }
    return true;
}

void test_package_manager(Fixture & fixture, const fs::path & root) {
    write_spec(root);
    const auto models = root / "models";
    const auto installed = models / "Demo" / "model.gguf";
    engine::package_manager::PackageManager manager(root, models);

    fixture.reset_counters();
    std::string last_message;
    manager.install("demo", true, nullptr, [&](const engine::package_manager::PackageProgress & progress) {
        last_message = progress.message;
    });
    check(read_file(installed) == fixture.file, "package installs through Xet");
    check(last_message.find("Downloaded model.gguf") != std::string::npos, "install progress completes");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.cdn_requests == 0, "Xet install does not use the HTTP route");
        check(fixture.token_requests == 1 && fixture.xorb_requests == 2, "Xet install requests");
    }
    const auto manifest = engine::io::json::parse_file(models / "Demo" / ".audiocpp-package-demo.json");
    check(manifest.require("files").require("model.gguf").require("etag").as_string() == fixture.file_hash,
        "manifest keeps the Xet etag");

    // Disabled by environment -> HTTP.
    set_env("AUDIOCPP_HF_XET", "0");
    fixture.reset_counters();
    manager.install("demo", true, nullptr, nullptr);
    unset_env("AUDIOCPP_HF_XET");
    {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.cdn_requests == 1 && fixture.token_requests == 0, "AUDIOCPP_HF_XET=0 uses HTTP");
    }

    // Unavailable Xet endpoints fall back to HTTP with the same result.
    for (const int status : {404, 501}) {
        fixture.v2_status = status;
        fixture.reset_counters();
        manager.install("demo", true, nullptr, [&](const auto & progress) { last_message = progress.message; });
        check(read_file(installed) == fixture.file, "fallback install after v2 HTTP " + std::to_string(status));
        std::lock_guard<std::mutex> lock(fixture.mutex);
        check(fixture.cdn_requests == 1, "fallback uses the HTTP route");
    }
    fixture.v2_status = 200;
    fixture.token_status = 403;
    fixture.reset_counters();
    manager.install("demo", true, nullptr, nullptr);
    fixture.token_status = 200;
    check(read_file(installed) == fixture.file, "fallback install after token failure");

    // A protocol violation from the xorb host also falls back.
    fixture.xorb_ignore_range = true;
    fixture.reset_counters();
    manager.install("demo", true, nullptr, nullptr);
    fixture.xorb_ignore_range = false;
    check(read_file(installed) == fixture.file, "fallback install after xorb protocol error");

    // Integrity failures are reported, not silently retried, and nothing is published.
    const auto set_served_sha = [&](const std::string & value) {
        std::lock_guard<std::mutex> lock(fixture.mutex);
        fixture.served_sha256 = value;
    };
    const auto original_sha = fixture.file_sha256;
    set_served_sha(fake_hash(6));
    fs::remove(installed);
    const auto message = error_of([&] { manager.install("demo", false, nullptr, nullptr); });
    set_served_sha(original_sha);
    check(message.find("SHA-256 mismatch") != std::string::npos, "integrity error surfaces: " + message);
    check(!fs::exists(installed) && no_staging(models), "failed install publishes nothing");

    // Cancellation mid-transfer.
    fixture.xorb_slow = true;
    auto cancelled = std::make_shared<std::atomic_bool>(false);
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cancelled->store(true);
    });
    const auto cancel_message = error_of([&] { manager.install("demo", false, cancelled, nullptr); });
    canceller.join();
    fixture.xorb_slow = false;
    check(cancel_message.find("cancelled") != std::string::npos, "cancellation surfaces: " + cancel_message);
    check(!fs::exists(installed) && no_staging(models), "cancelled install publishes nothing");

    std::lock_guard<std::mutex> lock(fixture.mutex);
    check(fixture.leaks.empty(), "no credential leaks in installs");
}

}  // namespace

int main() {
    const auto root = fs::temp_directory_path() / ("audiocpp-xet-test-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        test_bg4();
        test_chunk_decoder();
        test_range_parser();

        fs::create_directories(root);
        set_env("HF_TOKEN", "hf-secret");
        Fixture fixture;
        set_env("AUDIOCPP_HF_BASE_URL", fixture.base);
        test_direct_transfer(fixture, root);
        test_package_manager(fixture, root);
    } catch (const std::exception & error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    std::error_code error;
    fs::remove_all(root, error);
    if (failures == 0) std::cout << "package_manager_xet_test passed\n";
    return failures == 0 ? 0 : 1;
}
