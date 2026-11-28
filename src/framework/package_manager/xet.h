#pragma once

// Download-only Hugging Face Xet client used by the native package manager.
// Internal header; not installed and not part of the public API.
//
// Protocol reference: https://huggingface.co/docs/xet/index
//   1. GET {hub}/api/models/{repo}/xet-read-token/{revision} -> CAS token + URL
//   2. GET {cas}/v2/reconstructions/{xet_hash}                -> ordered terms
//      plus signed xorb byte ranges
//   3. GET each signed URL with its exact Range set (206, often multipart)
//   4. Decode chunks (none / LZ4 frame / ByteGrouping4 + LZ4 frame) and write
//      them at their term offsets in the staged output file.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine::package_manager::detail {

// Xet could not be used for this file (no token, no v2 endpoint, unsupported
// recipe layout, transport failure, ...). Nothing was published; callers may
// fall back to the regular HTTP download.
struct XetUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A complete Xet reconstruction did not match the Hub's size or SHA-256.
struct XetIntegrityError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Malformed or inconsistent data from CAS or the xorb host.
struct XetProtocolError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// True for a 64-character lowercase hexadecimal Xet hash string.
bool is_xet_hash(const std::string & value);

struct XetFileRequest {
    std::string repo;             // "namespace/name" (model repositories)
    std::string revision;         // commit hash when known, otherwise the spec revision
    std::string xet_hash;         // X-Xet-Hash from the un-followed /resolve response
    std::string auth_url;         // optional rel="xet-auth" Link from the same response
    std::string expected_sha256;  // optional X-Linked-ETag (lowercase hex)
    uint64_t size = 0;            // X-Linked-Size
};

struct XetOptions {
    size_t concurrency = 16;               // parallel signed xorb fetches
    uint64_t segment_bytes = 10ull << 30;  // ranged reconstruction batch size
    bool verify_sha256 = true;
    int max_attempts = 5;                  // per request, for transient failures
    int max_refreshes = 3;                 // re-queries after signed URL expiry
    int initial_backoff_ms = 250;
};

// AUDIOCPP_HF_XET=0 disables Xet transfers.
bool xet_enabled_from_env();
// AUDIOCPP_HF_XET_CONCURRENCY and AUDIOCPP_HF_XET_VERIFY.
XetOptions xet_options_from_env();

// Reconstructs one Xet-backed file into `destination` (created/truncated).
// `progress` receives the number of reconstructed bytes written so far.
// Throws Cancelled, XetIntegrityError, or another exception (including
// XetUnavailable) when the transfer could not be completed.
void xet_download(
    const XetFileRequest & request,
    const std::filesystem::path & destination,
    const XetOptions & options,
    const std::shared_ptr<std::atomic_bool> & cancelled,
    const std::function<void(uint64_t)> & progress);

// ---------------------------------------------------------------------------
// Building blocks, exposed for unit tests.

// Maximum decompressed chunk size allowed by the xorb specification.
constexpr size_t kXetMaxChunkBytes = 128 * 1024;

// Inverse of the ByteGrouping4 transform: `grouped` holds four byte groups
// (input bytes i % 4 == 0, 1, 2, 3); the first size % 4 groups carry one
// extra byte. Writes `size` bytes to `out`.
void bg4_ungroup(const uint8_t * grouped, size_t size, uint8_t * out);

struct XetByteRange {
    uint64_t start = 0;  // inclusive
    uint64_t end = 0;    // inclusive
};

// Splits a 206 response body into the requested byte ranges. Accepts a raw
// body (single requested range) or multipart/byteranges in any part order.
// Every part's Content-Range must equal exactly one requested range.
class XetRangeResponseParser {
public:
    struct Callbacks {
        std::function<void(size_t range)> begin;
        std::function<void(size_t range, const uint8_t * data, size_t size)> data;
        std::function<void(size_t range)> end;
    };

    XetRangeResponseParser(
        std::vector<XetByteRange> requested,
        const std::string & content_type,
        const std::string & content_range,
        Callbacks callbacks);

    void feed(const uint8_t * data, size_t size);
    void finish();

private:
    enum class State { single, preamble, delimiter_tail, part_headers, body, after_body, epilogue };

    void begin_part(const std::string & headers);

    std::vector<XetByteRange> requested_;
    std::vector<bool> seen_;
    Callbacks callbacks_;
    State state_ = State::single;
    std::string delimiter_;
    std::string buffer_;
    size_t current_ = 0;
    uint64_t remaining_ = 0;
};

// Decodes a contiguous run of serialized xorb chunks.
class XetChunkDecoder {
public:
    using Sink = std::function<void(uint32_t chunk_index, const uint8_t * data, size_t size)>;

    XetChunkDecoder();
    ~XetChunkDecoder();
    XetChunkDecoder(const XetChunkDecoder &) = delete;
    XetChunkDecoder & operator=(const XetChunkDecoder &) = delete;

    void begin(uint32_t first_chunk, uint32_t chunk_count, Sink sink);
    void feed(const uint8_t * data, size_t size);
    void end();

private:
    void decode_payload();
    void lz4_frame_decode(std::vector<uint8_t> & out);

    void * lz4_ = nullptr;
    Sink sink_;
    uint32_t next_chunk_ = 0;
    uint32_t remaining_chunks_ = 0;
    uint8_t header_[8] = {};
    size_t header_fill_ = 0;
    bool in_payload_ = false;
    uint8_t scheme_ = 0;
    uint32_t compressed_size_ = 0;
    uint32_t uncompressed_size_ = 0;
    std::vector<uint8_t> payload_;
    std::vector<uint8_t> scratch_;
    std::vector<uint8_t> output_;
};

}  // namespace engine::package_manager::detail
