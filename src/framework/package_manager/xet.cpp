#include "xet.h"

#include "http_util.h"

#include "engine/framework/io/json.h"

#include "httplib.h"
#include "lz4frame.h"

#include <openssl/evp.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

namespace engine::package_manager::detail {
namespace {

namespace json = engine::io::json;

constexpr const char * kUserAgent = "audio.cpp native model manager/1.0";
constexpr size_t kMaxJsonBytes = 256u << 20;
constexpr size_t kMaxPartHeaderBytes = 16 * 1024;
constexpr size_t kMaxTokenChars = 64000;
// LZ4 frames are never much larger than their input; allow generous framing
// overhead but keep the allocation bounded.
constexpr size_t kMaxCompressedChunkBytes = 2 * kXetMaxChunkBytes + 4096;
constexpr uint64_t kMaxChunkIndex = 1u << 24;

// The signed xorb URL stopped authorizing (usually expired). Re-query the
// reconstruction to get fresh URLs.
struct RefreshNeeded : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Connection errors, 429 and 5xx; retried with backoff.
struct TransientError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

bool is_cancelled(const std::shared_ptr<std::atomic_bool> & cancelled) {
    return cancelled && cancelled->load();
}

int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void sleep_with_cancel(int milliseconds, const std::shared_ptr<std::atomic_bool> & cancelled,
                       const std::atomic_bool * abort = nullptr) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(milliseconds);
    while (std::chrono::steady_clock::now() < deadline) {
        if (is_cancelled(cancelled)) throw Cancelled();
        if (abort != nullptr && abort->load()) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

int backoff_ms(const XetOptions & options, int attempt) {
    const int64_t value = static_cast<int64_t>(options.initial_backoff_ms) << std::min(attempt, 5);
    return static_cast<int>(std::min<int64_t>(value, 8000));
}

bool transient_status(int status) {
    return status == 429 || status == 500 || status == 502 || status == 503 || status == 504;
}

std::unique_ptr<httplib::Client> make_client(const HttpUrl & url, long read_timeout_seconds) {
    auto client = std::make_unique<httplib::Client>(http_origin(url));
    configure_client(*client, read_timeout_seconds);
    // Xet endpoints and signed xorb URLs must be used exactly as given: no
    // redirects (which could forward credentials) and no path re-encoding
    // (which would invalidate signatures).
    client->set_follow_location(false);
    client->set_path_encode(false);
    client->set_keep_alive(true);
    return client;
}

struct TextResponse {
    int status = 0;
    std::string body;
};

// GET with a bounded in-memory body. Throws TransientError on transport
// failures and Cancelled on cancellation.
TextResponse get_text(const std::string & url, const httplib::Headers & headers,
                      const std::shared_ptr<std::atomic_bool> & cancelled) {
    const auto parsed = parse_http_url(url);
    auto client = make_client(parsed, 120);
    TextResponse out;
    bool too_large = false;
    auto response = client->Get(parsed.path, headers,
        [&](const char * data, size_t size) {
            if (is_cancelled(cancelled)) return false;
            if (out.body.size() + size > kMaxJsonBytes) {
                too_large = true;
                return false;
            }
            out.body.append(data, size);
            return true;
        });
    if (!response) {
        if (is_cancelled(cancelled)) throw Cancelled();
        if (too_large) throw XetProtocolError("Xet metadata response is too large");
        throw TransientError("Xet metadata request failed: " + httplib::to_string(response.error()));
    }
    out.status = response->status;
    return out;
}

uint64_t require_u64(const json::Value & object, const char * key) {
    const auto & value = object.require(key);
    if (!value.is_number()) throw XetProtocolError(std::string("Xet field is not a number: ") + key);
    const double number = value.as_number();
    if (!(number >= 0.0) || number > 9007199254740991.0 || std::floor(number) != number) {
        throw XetProtocolError(std::string("Xet field is out of range: ") + key);
    }
    return static_cast<uint64_t>(number);
}

struct ChunkRange {
    uint32_t start = 0;
    uint32_t end = 0;  // exclusive
};

ChunkRange require_chunk_range(const json::Value & object, const char * key) {
    const auto & value = object.require(key);
    const auto start = require_u64(value, "start");
    const auto end = require_u64(value, "end");
    if (start >= end || end > kMaxChunkIndex) throw XetProtocolError("invalid Xet chunk range");
    return {static_cast<uint32_t>(start), static_cast<uint32_t>(end)};
}

// ---------------------------------------------------------------------------
// CAS token

struct CasToken {
    std::string access_token;
    std::string cas_url;
    int64_t exp = 0;
};

class TokenProvider {
public:
    TokenProvider(XetFileRequest request, std::shared_ptr<std::atomic_bool> cancelled, XetOptions options)
        : request_(std::move(request)), cancelled_(std::move(cancelled)), options_(options) {}

    CasToken get(bool force_refresh) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Refresh 30 seconds before expiry, as the Xet auth spec recommends.
        if (force_refresh || token_.access_token.empty() || now_unix() + 30 >= token_.exp) {
            token_ = fetch();
        }
        return token_;
    }

private:
    std::string token_url() const {
        const auto base = huggingface_base_url();
        // The commit-pinned rel="xet-auth" link may carry the Hub token only
        // when it points at the configured Hub origin.
        if (!request_.auth_url.empty()) {
            try {
                if (http_origin(parse_http_url(request_.auth_url)) == http_origin(parse_http_url(base))) {
                    return request_.auth_url;
                }
            } catch (const std::exception &) {
            }
        }
        return base + "/api/models/" + request_.repo + "/xet-read-token/" +
            url_encode(request_.revision, false);
    }

    CasToken fetch() const {
        const auto url = token_url();
        httplib::Headers headers{{"User-Agent", kUserAgent}};
        const auto hub_token = huggingface_token();
        if (!hub_token.empty()) headers.emplace("Authorization", "Bearer " + hub_token);

        TextResponse response;
        for (int attempt = 0;; ++attempt) {
            try {
                response = get_text(url, headers, cancelled_);
                if (!transient_status(response.status)) break;
            } catch (const TransientError &) {
                if (attempt + 1 >= options_.max_attempts) throw;
            }
            if (attempt + 1 >= options_.max_attempts) break;
            sleep_with_cancel(backoff_ms(options_, attempt), cancelled_);
        }
        if (response.status != 200) {
            throw XetUnavailable("Xet read token is unavailable (HTTP " + std::to_string(response.status) + ")");
        }

        CasToken token;
        try {
            const auto root = json::parse(response.body);
            token.access_token = json::require_string(root, "accessToken");
            token.cas_url = json::require_string(root, "casUrl");
            token.exp = static_cast<int64_t>(require_u64(root, "exp"));
        } catch (const XetProtocolError &) {
            throw;
        } catch (const std::exception & error) {
            throw XetProtocolError(std::string("malformed Xet read token response: ") + error.what());
        }
        if (token.access_token.empty() || token.access_token.size() > kMaxTokenChars ||
            token.cas_url.size() > kMaxTokenChars) {
            throw XetProtocolError("malformed Xet read token response");
        }
        while (!token.cas_url.empty() && token.cas_url.back() == '/') token.cas_url.pop_back();
        const auto cas = parse_http_url(token.cas_url);
        const auto hub = parse_http_url(huggingface_base_url());
        if (hub.scheme == "https" && cas.scheme != "https") {
            throw XetProtocolError("Xet CAS endpoint must use HTTPS");
        }
        return token;
    }

    XetFileRequest request_;
    std::shared_ptr<std::atomic_bool> cancelled_;
    XetOptions options_;
    std::mutex mutex_;
    CasToken token_;
};

// ---------------------------------------------------------------------------
// Reconstruction recipe (v2)

struct Term {
    std::string xorb;
    ChunkRange chunks;
    uint64_t unpacked_length = 0;
};

struct RangeDescriptor {
    ChunkRange chunks;
    XetByteRange bytes;
};

struct Fetch {
    std::string xorb;
    std::string url;
    std::vector<RangeDescriptor> ranges;
    uint64_t total_bytes = 0;
};

struct Recipe {
    uint64_t offset_into_first_range = 0;
    std::vector<Term> terms;
    std::vector<Fetch> fetches;
};

Recipe parse_recipe(const std::string & body) {
    Recipe recipe;
    try {
        const auto root = json::parse(body);
        recipe.offset_into_first_range = require_u64(root, "offset_into_first_range");
        for (const auto & item : root.require("terms").as_array()) {
            Term term;
            term.xorb = json::require_string(item, "hash");
            term.chunks = require_chunk_range(item, "range");
            term.unpacked_length = require_u64(item, "unpacked_length");
            if (!is_xet_hash(term.xorb)) throw XetProtocolError("invalid xorb hash in reconstruction");
            const uint64_t chunk_count = term.chunks.end - term.chunks.start;
            if (term.unpacked_length == 0 || term.unpacked_length > chunk_count * kXetMaxChunkBytes) {
                throw XetProtocolError("invalid term unpacked_length in reconstruction");
            }
            recipe.terms.push_back(std::move(term));
        }
        for (const auto & [hash, entries] : root.require("xorbs").as_object()) {
            if (!is_xet_hash(hash)) throw XetProtocolError("invalid xorb hash in reconstruction");
            for (const auto & entry : entries.as_array()) {
                Fetch fetch;
                fetch.xorb = hash;
                fetch.url = json::require_string(entry, "url");
                (void) parse_http_url(fetch.url);
                for (const auto & range : entry.require("ranges").as_array()) {
                    RangeDescriptor descriptor;
                    descriptor.chunks = require_chunk_range(range, "chunks");
                    const auto & bytes = range.require("bytes");
                    descriptor.bytes.start = require_u64(bytes, "start");
                    descriptor.bytes.end = require_u64(bytes, "end");
                    if (descriptor.bytes.end < descriptor.bytes.start) {
                        throw XetProtocolError("invalid xorb byte range in reconstruction");
                    }
                    const uint64_t length = descriptor.bytes.end - descriptor.bytes.start + 1;
                    const uint64_t chunks = descriptor.chunks.end - descriptor.chunks.start;
                    if (length < chunks * 8 || length > chunks * (8 + kMaxCompressedChunkBytes)) {
                        throw XetProtocolError("xorb byte range does not match its chunk count");
                    }
                    fetch.total_bytes += length;
                    fetch.ranges.push_back(descriptor);
                }
                if (fetch.ranges.empty()) throw XetProtocolError("xorb fetch entry has no ranges");
                recipe.fetches.push_back(std::move(fetch));
            }
        }
    } catch (const XetProtocolError &) {
        throw;
    } catch (const std::exception & error) {
        throw XetProtocolError(std::string("malformed Xet reconstruction: ") + error.what());
    }
    if (recipe.terms.empty()) throw XetProtocolError("Xet reconstruction has no terms");
    return recipe;
}

Recipe fetch_recipe(TokenProvider & tokens, const std::string & xet_hash, uint64_t first, uint64_t last,
                    const XetOptions & options, const std::shared_ptr<std::atomic_bool> & cancelled) {
    bool refreshed = false;
    TextResponse response;
    for (int attempt = 0;; ++attempt) {
        const auto token = tokens.get(false);
        const httplib::Headers headers{
            {"User-Agent", kUserAgent},
            {"Authorization", "Bearer " + token.access_token},
            {"Range", "bytes=" + std::to_string(first) + "-" + std::to_string(last)},
        };
        bool transient = false;
        try {
            response = get_text(token.cas_url + "/v2/reconstructions/" + xet_hash, headers, cancelled);
            transient = transient_status(response.status);
        } catch (const TransientError &) {
            transient = true;
            response.status = 0;
        }
        if (response.status == 401 && !refreshed) {
            tokens.get(true);
            refreshed = true;
            continue;
        }
        if (!transient) break;
        if (attempt + 1 >= options.max_attempts) {
            throw XetUnavailable("Xet reconstruction request kept failing (HTTP " +
                std::to_string(response.status) + ")");
        }
        sleep_with_cancel(backoff_ms(options, attempt), cancelled);
    }
    if (response.status == 404 || response.status == 501) {
        throw XetUnavailable("Xet v2 reconstruction is unavailable (HTTP " + std::to_string(response.status) + ")");
    }
    if (response.status != 200) {
        throw XetUnavailable("Xet reconstruction failed (HTTP " + std::to_string(response.status) + ")");
    }
    return parse_recipe(response.body);
}

// ---------------------------------------------------------------------------
// Transfer planning

struct Placement {
    size_t term = 0;
    uint32_t first_chunk = 0;
    uint32_t end_chunk = 0;
    uint64_t file_offset = 0;  // file position of the term's first byte
    uint64_t unpacked_length = 0;
    uint64_t cursor = 0;       // bytes of the term decoded so far
};

struct FetchJob {
    const Fetch * fetch = nullptr;
    std::vector<std::vector<Placement>> placements;  // per range descriptor
    uint64_t first_offset = std::numeric_limits<uint64_t>::max();  // earliest file byte it writes
};

// Builds the fetch jobs for the terms of one segment that are not done yet.
// Terms outside the segment are marked done; `term_starts` receives each
// term's first file byte, clipped to the segment.
std::vector<FetchJob> plan_jobs(const Recipe & recipe, uint64_t segment_start, uint64_t segment_end,
                                std::vector<char> & done, std::vector<uint64_t> & term_starts) {
    std::unordered_map<std::string, std::vector<std::pair<size_t, size_t>>> by_xorb;
    for (size_t f = 0; f < recipe.fetches.size(); ++f) {
        for (size_t r = 0; r < recipe.fetches[f].ranges.size(); ++r) {
            by_xorb[recipe.fetches[f].xorb].emplace_back(f, r);
        }
    }
    std::vector<FetchJob> jobs(recipe.fetches.size());
    for (size_t f = 0; f < jobs.size(); ++f) {
        jobs[f].fetch = &recipe.fetches[f];
        jobs[f].placements.resize(recipe.fetches[f].ranges.size());
    }

    if (recipe.offset_into_first_range >= recipe.terms.front().unpacked_length ||
        recipe.offset_into_first_range > segment_start) {
        throw XetProtocolError("invalid offset_into_first_range in Xet reconstruction");
    }
    uint64_t position = segment_start - recipe.offset_into_first_range;
    uint64_t covered = 0;
    term_starts.assign(recipe.terms.size(), segment_end);
    for (size_t t = 0; t < recipe.terms.size(); ++t) {
        const auto & term = recipe.terms[t];
        const uint64_t term_start = position;
        const uint64_t term_end = position + term.unpacked_length;
        position = term_end;
        const uint64_t lo = std::max(term_start, segment_start);
        const uint64_t hi = std::min(term_end, segment_end);
        if (lo >= hi) {
            done[t] = 1;
            continue;
        }
        term_starts[t] = lo;
        covered += hi - lo;
        if (done[t]) continue;

        const auto found = by_xorb.find(term.xorb);
        bool placed = false;
        if (found != by_xorb.end()) {
            for (const auto & [f, r] : found->second) {
                const auto & chunks = recipe.fetches[f].ranges[r].chunks;
                if (chunks.start <= term.chunks.start && term.chunks.end <= chunks.end) {
                    Placement placement;
                    placement.term = t;
                    placement.first_chunk = term.chunks.start;
                    placement.end_chunk = term.chunks.end;
                    placement.file_offset = term_start;
                    placement.unpacked_length = term.unpacked_length;
                    jobs[f].placements[r].push_back(placement);
                    jobs[f].first_offset = std::min(jobs[f].first_offset, lo);
                    placed = true;
                    break;
                }
            }
        }
        if (!placed) throw XetUnavailable("Xet reconstruction term is not covered by a single fetch range");
    }
    if (covered != segment_end - segment_start) {
        throw XetProtocolError("Xet reconstruction does not cover the requested byte range");
    }

    std::vector<FetchJob> active;
    for (auto & job : jobs) {
        const bool needed = std::any_of(job.placements.begin(), job.placements.end(),
            [](const auto & list) { return !list.empty(); });
        if (needed) active.push_back(std::move(job));
    }
    // Fetch in file order so the verified prefix of the output (hashed while
    // the transfer runs) keeps advancing.
    std::stable_sort(active.begin(), active.end(), [](const FetchJob & a, const FetchJob & b) {
        return a.first_offset < b.first_offset;
    });
    return active;
}

int64_t signed_url_expiry(const std::string & url) {
    const auto query = url.find('?');
    if (query == std::string::npos) return 0;
    size_t position = query + 1;
    while (position < url.size()) {
        const auto amp = url.find('&', position);
        const auto item = url.substr(position, amp == std::string::npos ? std::string::npos : amp - position);
        if (item.rfind("Expires=", 0) == 0) {
            try {
                return std::stoll(item.substr(8));
            } catch (...) {
                return 0;
            }
        }
        if (amp == std::string::npos) break;
        position = amp + 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// SHA-256 of the output, computed on a background thread over the prefix of
// the file whose terms are complete, so verification overlaps the transfer.

class PrefixHasher {
public:
    PrefixHasher(std::filesystem::path path, std::shared_ptr<std::atomic_bool> cancelled)
        : path_(std::move(path)), cancelled_(std::move(cancelled)), context_(EVP_MD_CTX_new(), &EVP_MD_CTX_free) {
        if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1) {
            throw std::runtime_error("could not initialize SHA-256");
        }
        thread_ = std::thread([this] { run(); });
    }

    ~PrefixHasher() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    PrefixHasher(const PrefixHasher &) = delete;
    PrefixHasher & operator=(const PrefixHasher &) = delete;

    // Bytes [0, ready) of the file are final.
    void advance(uint64_t ready) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ready > ready_) ready_ = ready;
        }
        wake_.notify_all();
    }

    std::string finish(uint64_t size) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ready_ = std::max(ready_, size);
            target_ = size;
            finishing_ = true;
        }
        wake_.notify_all();
        thread_.join();
        if (error_) std::rethrow_exception(error_);
        if (is_cancelled(cancelled_)) throw Cancelled();
        if (hashed_ != size) throw std::runtime_error("SHA-256 verification did not cover the whole file");
        unsigned char digest[EVP_MAX_MD_SIZE];
        unsigned int length = 0;
        EVP_DigestFinal_ex(context_.get(), digest, &length);
        constexpr char hex[] = "0123456789abcdef";
        std::string out;
        for (unsigned int i = 0; i < length; ++i) {
            out.push_back(hex[digest[i] >> 4]);
            out.push_back(hex[digest[i] & 0xf]);
        }
        return out;
    }

private:
    void run() {
        try {
            std::ifstream input;
            // Unbuffered: never read ahead into bytes that are not final yet.
            input.rdbuf()->pubsetbuf(nullptr, 0);
            input.open(path_, std::ios::binary);
            if (!input) throw std::runtime_error("could not read " + path_.string());
            std::vector<char> buffer(1 << 20);
            for (;;) {
                uint64_t ready = 0;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    wake_.wait(lock, [&] {
                        return stop_ || ready_ > hashed_ || (finishing_ && hashed_ >= target_);
                    });
                    if (stop_ || (finishing_ && hashed_ >= target_)) return;
                    ready = ready_;
                }
                while (hashed_ < ready) {
                    if (is_cancelled(cancelled_)) return;
                    const auto count = static_cast<size_t>(std::min<uint64_t>(buffer.size(), ready - hashed_));
                    input.seekg(static_cast<std::streamoff>(hashed_));
                    input.read(buffer.data(), static_cast<std::streamsize>(count));
                    if (static_cast<size_t>(input.gcount()) != count) {
                        throw std::runtime_error("could not read " + path_.string());
                    }
                    EVP_DigestUpdate(context_.get(), buffer.data(), count);
                    hashed_ += count;
                }
            }
        } catch (...) {
            error_ = std::current_exception();
        }
    }

    std::filesystem::path path_;
    std::shared_ptr<std::atomic_bool> cancelled_;
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
    std::mutex mutex_;
    std::condition_variable wake_;
    uint64_t ready_ = 0;
    uint64_t target_ = 0;
    bool finishing_ = false;
    bool stop_ = false;
    uint64_t hashed_ = 0;  // owned by the hashing thread until join
    std::exception_ptr error_;
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Parallel transfer

struct TransferState {
    const XetOptions * options = nullptr;
    std::shared_ptr<std::atomic_bool> cancelled;
    std::filesystem::path destination;
    uint64_t segment_start = 0;
    uint64_t segment_end = 0;
    std::vector<char> * done = nullptr;
    const std::vector<uint64_t> * term_starts = nullptr;
    size_t * prefix_term = nullptr;
    PrefixHasher * hasher = nullptr;

    std::atomic_bool abort{false};
    std::atomic_size_t next{0};
    std::mutex mutex;
    std::exception_ptr error;
    bool refresh_needed = false;

    // Progress counts each file byte once, however often it is rewritten by
    // retries or refresh passes: a term contributes the furthest point any
    // attempt has reached, and the segment total never moves backwards.
    const std::function<void(uint64_t)> * progress = nullptr;
    std::mutex progress_mutex;
    uint64_t progress_base = 0;                        // bytes of earlier segments
    std::vector<uint64_t> * term_progress = nullptr;   // per term, furthest byte written
    uint64_t * segment_progress = nullptr;             // sum of term_progress
    uint64_t * segment_reported = nullptr;             // highest segment_progress reported

    void fail(std::exception_ptr exception) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error) error = std::move(exception);
        abort = true;
    }

    // Called with `mutex` held after a job's terms are marked done.
    void advance_prefix_locked() {
        while (*prefix_term < done->size() && (*done)[*prefix_term]) ++*prefix_term;
        if (hasher != nullptr) {
            hasher->advance(*prefix_term == done->size() ? segment_end : (*term_starts)[*prefix_term]);
        }
    }

    void request_refresh() {
        std::lock_guard<std::mutex> lock(mutex);
        refresh_needed = true;
        abort = true;
    }

    // `reached` is how many bytes of `term` (from its clipped start) are written.
    void note_progress(size_t term, uint64_t reached) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        auto & previous = (*term_progress)[term];
        if (reached <= previous) return;
        *segment_progress += reached - previous;
        previous = reached;
        if (*segment_progress <= *segment_reported) return;
        *segment_reported = *segment_progress;
        if (progress != nullptr && *progress) (*progress)(progress_base + *segment_reported);
    }
};

class Worker {
public:
    explicit Worker(TransferState & state) : state_(state) {
        output_.open(state.destination, std::ios::binary | std::ios::in | std::ios::out);
        if (!output_) throw std::runtime_error("could not open " + state.destination.string());
    }

    void run() {
        for (;;) {
            if (state_.abort.load()) return;
            if (is_cancelled(state_.cancelled)) {
                state_.fail(std::make_exception_ptr(Cancelled()));
                return;
            }
            const auto index = state_.next.fetch_add(1);
            if (index >= jobs_->size()) return;
            try {
                run_job((*jobs_)[index]);
            } catch (const RefreshNeeded &) {
                state_.request_refresh();
                return;
            } catch (...) {
                state_.fail(std::current_exception());
                return;
            }
        }
    }

    std::vector<FetchJob> * jobs_ = nullptr;

private:
    httplib::Client & client_for(const HttpUrl & url) {
        auto & client = clients_[http_origin(url)];
        if (!client) client = make_client(url, 300);
        return *client;
    }

    void write_clipped(const Placement & placement, uint64_t position, const uint8_t * data, size_t size) {
        const uint64_t lo = std::max(position, state_.segment_start);
        const uint64_t hi = std::min(position + size, state_.segment_end);
        if (lo >= hi) return;
        output_.seekp(static_cast<std::streamoff>(lo));
        output_.write(reinterpret_cast<const char *>(data + (lo - position)),
            static_cast<std::streamsize>(hi - lo));
        if (!output_) throw std::runtime_error("could not write downloaded model file");
        state_.note_progress(placement.term, hi - (*state_.term_starts)[placement.term]);
    }

    void run_job(FetchJob & job) {
        for (int attempt = 0;; ++attempt) {
            try {
                attempt_job(job);
                break;
            } catch (const TransientError & error) {
                if (attempt + 1 >= state_.options->max_attempts) {
                    throw XetUnavailable(std::string("xorb transfer kept failing: ") + error.what());
                }
                sleep_with_cancel(backoff_ms(*state_.options, attempt), state_.cancelled, &state_.abort);
                if (state_.abort.load()) return;
            }
        }
        if (state_.abort.load() && !is_cancelled(state_.cancelled)) {
            // Another worker failed; this job may or may not have finished.
            // Leave its terms pending so a refresh pass redoes them.
            return;
        }
        output_.flush();
        if (!output_) throw std::runtime_error("could not write downloaded model file");
        std::lock_guard<std::mutex> lock(state_.mutex);
        for (const auto & list : job.placements) {
            for (const auto & placement : list) (*state_.done)[placement.term] = 1;
        }
        state_.advance_prefix_locked();
    }

    void attempt_job(FetchJob & job) {
        const auto & fetch = *job.fetch;
        const auto expires = signed_url_expiry(fetch.url);
        if (expires != 0 && now_unix() + 5 >= expires) throw RefreshNeeded("signed xorb URL expired");

        std::vector<XetByteRange> requested;
        std::string range_header = "bytes=";
        for (size_t i = 0; i < fetch.ranges.size(); ++i) {
            if (i != 0) range_header += ",";
            range_header += std::to_string(fetch.ranges[i].bytes.start) + "-" +
                std::to_string(fetch.ranges[i].bytes.end);
            requested.push_back(fetch.ranges[i].bytes);
        }
        // Signed URLs carry their own authorization: never send Hub or CAS
        // credentials to the storage host.
        const httplib::Headers headers{{"User-Agent", kUserAgent}, {"Range", range_header}};

        size_t active_range = 0;
        const auto chunk_sink = [&](uint32_t chunk, const uint8_t * data, size_t size) {
            for (auto & placement : job.placements[active_range]) {
                if (chunk < placement.first_chunk || chunk >= placement.end_chunk) continue;
                if (chunk == placement.first_chunk) placement.cursor = 0;
                if (placement.cursor + size > placement.unpacked_length) {
                    throw XetProtocolError("Xet term is longer than its unpacked_length");
                }
                write_clipped(placement, placement.file_offset + placement.cursor, data, size);
                placement.cursor += size;
                if (chunk + 1 == placement.end_chunk && placement.cursor != placement.unpacked_length) {
                    throw XetProtocolError("Xet term length does not match its unpacked_length");
                }
            }
        };
        XetRangeResponseParser::Callbacks callbacks;
        callbacks.begin = [&](size_t range) {
            active_range = range;
            const auto & chunks = fetch.ranges[range].chunks;
            decoder_.begin(chunks.start, chunks.end - chunks.start, chunk_sink);
        };
        callbacks.data = [&](size_t, const uint8_t * data, size_t size) { decoder_.feed(data, size); };
        callbacks.end = [&](size_t) { decoder_.end(); };

        const auto url = parse_http_url(fetch.url);
        auto & client = client_for(url);
        int status = 0;
        std::string data_error;
        std::unique_ptr<XetRangeResponseParser> parser;
        auto response = client.Get(url.path, headers,
            [&](const httplib::Response & head) {
                status = head.status;
                if (head.status != 206) return false;
                try {
                    parser = std::make_unique<XetRangeResponseParser>(requested,
                        head.get_header_value("Content-Type"), head.get_header_value("Content-Range"), callbacks);
                } catch (const std::exception & error) {
                    data_error = error.what();
                    return false;
                }
                return true;
            },
            [&](const char * data, size_t size) {
                if (state_.abort.load() || is_cancelled(state_.cancelled)) return false;
                try {
                    parser->feed(reinterpret_cast<const uint8_t *>(data), size);
                } catch (const std::exception & error) {
                    data_error = error.what();
                    return false;
                }
                return true;
            });

        if (is_cancelled(state_.cancelled)) throw Cancelled();
        if (!data_error.empty()) throw XetProtocolError(data_error);
        if (state_.abort.load()) return;
        if (status != 0 && status != 206) {
            if (status == 401 || status == 403) throw RefreshNeeded("signed xorb URL was rejected");
            if (transient_status(status)) throw TransientError("xorb host returned HTTP " + std::to_string(status));
            throw XetProtocolError("unexpected xorb response status HTTP " + std::to_string(status));
        }
        if (!response) {
            // Drop a possibly broken keep-alive connection before retrying.
            clients_.erase(http_origin(url));
            throw TransientError("xorb request failed: " + httplib::to_string(response.error()));
        }
        try {
            parser->finish();
        } catch (const std::exception & error) {
            throw TransientError(std::string("incomplete xorb response: ") + error.what());
        }
    }

    TransferState & state_;
    std::fstream output_;
    std::map<std::string, std::unique_ptr<httplib::Client>> clients_;
    XetChunkDecoder decoder_;
};

// Returns true when every job finished, false when signed URLs need a refresh.
bool run_jobs(std::vector<FetchJob> & jobs, TransferState & state) {
    const size_t worker_count = std::max<size_t>(1, std::min(state.options->concurrency, jobs.size()));
    std::vector<std::unique_ptr<Worker>> workers;
    workers.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers.push_back(std::make_unique<Worker>(state));
        workers.back()->jobs_ = &jobs;
    }
    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (auto & worker : workers) threads.emplace_back([&worker] { worker->run(); });
    for (auto & thread : threads) thread.join();
    workers.clear();  // closes and flushes every output stream
    if (state.error) std::rethrow_exception(state.error);
    if (is_cancelled(state.cancelled)) throw Cancelled();
    return !state.refresh_needed;
}

uint32_t read_le24(const uint8_t * data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
        (static_cast<uint32_t>(data[2]) << 16);
}

bool parse_content_range(const std::string & value, XetByteRange & out) {
    // "bytes <start>-<end>/<total|*>"
    auto text = lower(value);
    const auto first = text.find_first_not_of(" \t");
    if (first == std::string::npos || text.compare(first, 6, "bytes ") != 0) return false;
    text = text.substr(first + 6);
    const auto dash = text.find('-');
    const auto slash = text.find('/');
    if (dash == std::string::npos || slash == std::string::npos || dash > slash) return false;
    try {
        size_t used = 0;
        const auto start_text = text.substr(0, dash);
        const auto end_text = text.substr(dash + 1, slash - dash - 1);
        if (start_text.empty() || end_text.empty()) return false;
        out.start = std::stoull(start_text, &used);
        if (used != start_text.size()) return false;
        out.end = std::stoull(end_text, &used);
        if (used != end_text.size()) return false;
    } catch (...) {
        return false;
    }
    return out.start <= out.end;
}

std::string header_parameter(const std::string & header, const std::string & name) {
    const auto lowered = lower(header);
    size_t position = 0;
    while ((position = lowered.find(name + "=", position)) != std::string::npos) {
        const bool boundary_ok = position == 0 || lowered[position - 1] == ';' || lowered[position - 1] == ' ' ||
            lowered[position - 1] == '\t';
        if (!boundary_ok) {
            position += name.size();
            continue;
        }
        auto value = header.substr(position + name.size() + 1);
        if (!value.empty() && value.front() == '"') {
            const auto close = value.find('"', 1);
            return close == std::string::npos ? std::string{} : value.substr(1, close - 1);
        }
        const auto end = value.find_first_of("; \t");
        return end == std::string::npos ? value : value.substr(0, end);
    }
    return {};
}

}  // namespace

bool is_xet_hash(const std::string & value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char ch) {
        return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
    });
}

bool xet_enabled_from_env() {
    const auto value = lower(getenv_text("AUDIOCPP_HF_XET"));
    return !(value == "0" || value == "false" || value == "off" || value == "no");
}

XetOptions xet_options_from_env() {
    XetOptions options;
    const auto concurrency = getenv_text("AUDIOCPP_HF_XET_CONCURRENCY");
    if (!concurrency.empty()) {
        try {
            const auto value = std::stoul(concurrency);
            if (value >= 1 && value <= 256) options.concurrency = value;
        } catch (...) {
        }
    }
    const auto verify = lower(getenv_text("AUDIOCPP_HF_XET_VERIFY"));
    if (verify == "0" || verify == "false" || verify == "off" || verify == "no") options.verify_sha256 = false;
    return options;
}

void bg4_ungroup(const uint8_t * grouped, size_t size, uint8_t * out) {
    const size_t quarter = size / 4;
    const size_t remainder = size % 4;
    size_t offsets[4];
    offsets[0] = 0;
    for (size_t g = 1; g < 4; ++g) offsets[g] = offsets[g - 1] + quarter + (g - 1 < remainder ? 1 : 0);
    const uint8_t * g0 = grouped + offsets[0];
    const uint8_t * g1 = grouped + offsets[1];
    const uint8_t * g2 = grouped + offsets[2];
    const uint8_t * g3 = grouped + offsets[3];
    for (size_t i = 0; i < quarter; ++i) {
        out[4 * i + 0] = g0[i];
        out[4 * i + 1] = g1[i];
        out[4 * i + 2] = g2[i];
        out[4 * i + 3] = g3[i];
    }
    for (size_t g = 0; g < remainder; ++g) out[4 * quarter + g] = grouped[offsets[g] + quarter];
}

// ---------------------------------------------------------------------------
// XetRangeResponseParser

XetRangeResponseParser::XetRangeResponseParser(
    std::vector<XetByteRange> requested,
    const std::string & content_type,
    const std::string & content_range,
    Callbacks callbacks)
    : requested_(std::move(requested)), seen_(requested_.size(), false), callbacks_(std::move(callbacks)) {
    if (requested_.empty()) throw XetProtocolError("no byte ranges requested");
    const bool multipart = lower(content_type).find("multipart/byteranges") != std::string::npos;
    if (!multipart) {
        if (requested_.size() != 1) {
            throw XetProtocolError("expected a multipart/byteranges response for " +
                std::to_string(requested_.size()) + " ranges");
        }
        XetByteRange range;
        if (!parse_content_range(content_range, range) || range.start != requested_[0].start ||
            range.end != requested_[0].end) {
            throw XetProtocolError("xorb response Content-Range does not match the requested range");
        }
        state_ = State::single;
        current_ = 0;
        remaining_ = range.end - range.start + 1;
        seen_[0] = true;
        if (callbacks_.begin) callbacks_.begin(0);
        return;
    }
    const auto boundary = header_parameter(content_type, "boundary");
    if (boundary.empty() || boundary.size() > 200) throw XetProtocolError("invalid multipart boundary");
    delimiter_ = "--" + boundary;
    state_ = State::preamble;
}

void XetRangeResponseParser::begin_part(const std::string & headers) {
    XetByteRange range;
    bool found = false;
    size_t position = 0;
    while (position < headers.size()) {
        auto end = headers.find("\r\n", position);
        if (end == std::string::npos) end = headers.size();
        const auto line = headers.substr(position, end - position);
        position = end + 2;
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (lower(line.substr(0, colon)) == "content-range") {
            if (found || !parse_content_range(line.substr(colon + 1), range)) {
                throw XetProtocolError("invalid multipart Content-Range");
            }
            found = true;
        }
    }
    if (!found) throw XetProtocolError("multipart part has no Content-Range");
    for (size_t i = 0; i < requested_.size(); ++i) {
        if (!seen_[i] && requested_[i].start == range.start && requested_[i].end == range.end) {
            seen_[i] = true;
            current_ = i;
            remaining_ = range.end - range.start + 1;
            state_ = State::body;
            if (callbacks_.begin) callbacks_.begin(i);
            return;
        }
    }
    throw XetProtocolError("multipart part does not match any requested range");
}

void XetRangeResponseParser::feed(const uint8_t * data, size_t size) {
    while (size > 0) {
        switch (state_) {
        case State::single: {
            if (size > remaining_) throw XetProtocolError("xorb response is longer than the requested range");
            if (callbacks_.data) callbacks_.data(0, data, size);
            remaining_ -= size;
            return;
        }
        case State::body: {
            const size_t take = static_cast<size_t>(std::min<uint64_t>(remaining_, size));
            if (callbacks_.data) callbacks_.data(current_, data, take);
            data += take;
            size -= take;
            remaining_ -= take;
            if (remaining_ == 0) {
                if (callbacks_.end) callbacks_.end(current_);
                buffer_.clear();
                state_ = State::after_body;
            }
            break;
        }
        case State::epilogue:
            return;
        default: {
            const char ch = static_cast<char>(*data++);
            --size;
            buffer_.push_back(ch);
            if (buffer_.size() > kMaxPartHeaderBytes) throw XetProtocolError("multipart framing is too large");
            if (state_ == State::preamble) {
                if (buffer_.size() >= delimiter_.size() &&
                    buffer_.compare(buffer_.size() - delimiter_.size(), delimiter_.size(), delimiter_) == 0) {
                    buffer_.clear();
                    state_ = State::delimiter_tail;
                }
            } else if (state_ == State::after_body) {
                const std::string expected = "\r\n" + delimiter_;
                if (expected[buffer_.size() - 1] != ch) throw XetProtocolError("invalid multipart delimiter");
                if (buffer_.size() == expected.size()) {
                    buffer_.clear();
                    state_ = State::delimiter_tail;
                }
            } else if (state_ == State::delimiter_tail) {
                // Optional transport padding, then CRLF (next part) or "--" (close).
                if (buffer_ == " " || buffer_ == "\t") {
                    buffer_.clear();
                } else if (buffer_.size() == 2) {
                    if (buffer_ == "\r\n") {
                        state_ = State::part_headers;
                    } else if (buffer_ == "--") {
                        state_ = State::epilogue;
                    } else {
                        throw XetProtocolError("invalid multipart delimiter");
                    }
                    buffer_.clear();
                }
            } else if (state_ == State::part_headers) {
                if (buffer_.size() >= 4 && buffer_.compare(buffer_.size() - 4, 4, "\r\n\r\n") == 0) {
                    const auto headers = buffer_.substr(0, buffer_.size() - 4);
                    buffer_.clear();
                    begin_part(headers);
                }
            }
            break;
        }
        }
    }
}

void XetRangeResponseParser::finish() {
    if (state_ == State::single) {
        if (remaining_ != 0) throw XetProtocolError("xorb response is shorter than the requested range");
        if (callbacks_.end) callbacks_.end(0);
        state_ = State::epilogue;
        return;
    }
    if (state_ != State::epilogue) throw XetProtocolError("multipart response ended early");
    if (!std::all_of(seen_.begin(), seen_.end(), [](bool seen) { return seen; })) {
        throw XetProtocolError("multipart response is missing requested ranges");
    }
}

// ---------------------------------------------------------------------------
// XetChunkDecoder

XetChunkDecoder::XetChunkDecoder() {
    LZ4F_dctx * context = nullptr;
    if (LZ4F_isError(LZ4F_createDecompressionContext(&context, LZ4F_VERSION))) {
        throw std::runtime_error("could not create LZ4 decompression context");
    }
    lz4_ = context;
}

XetChunkDecoder::~XetChunkDecoder() {
    LZ4F_freeDecompressionContext(static_cast<LZ4F_dctx *>(lz4_));
}

void XetChunkDecoder::begin(uint32_t first_chunk, uint32_t chunk_count, Sink sink) {
    sink_ = std::move(sink);
    next_chunk_ = first_chunk;
    remaining_chunks_ = chunk_count;
    header_fill_ = 0;
    in_payload_ = false;
    payload_.clear();
}

void XetChunkDecoder::feed(const uint8_t * data, size_t size) {
    while (size > 0) {
        if (!in_payload_) {
            const size_t take = std::min(size, sizeof(header_) - header_fill_);
            std::memcpy(header_ + header_fill_, data, take);
            header_fill_ += take;
            data += take;
            size -= take;
            if (header_fill_ < sizeof(header_)) return;
            if (remaining_chunks_ == 0) throw XetProtocolError("xorb range has more data than its chunk count");
            if (header_[0] != 0) {
                throw XetProtocolError("unsupported xorb chunk header version " + std::to_string(header_[0]));
            }
            compressed_size_ = read_le24(header_ + 1);
            scheme_ = header_[4];
            uncompressed_size_ = read_le24(header_ + 5);
            if (uncompressed_size_ == 0 || uncompressed_size_ > kXetMaxChunkBytes || compressed_size_ == 0 ||
                compressed_size_ > kMaxCompressedChunkBytes) {
                throw XetProtocolError("invalid xorb chunk sizes");
            }
            if (scheme_ > 2) {
                throw XetProtocolError("unsupported xorb chunk compression scheme " + std::to_string(scheme_));
            }
            if (scheme_ == 0 && compressed_size_ != uncompressed_size_) {
                throw XetProtocolError("uncompressed xorb chunk has mismatched sizes");
            }
            payload_.clear();
            payload_.reserve(compressed_size_);
            in_payload_ = true;
        }
        const size_t take = std::min<size_t>(size, compressed_size_ - payload_.size());
        payload_.insert(payload_.end(), data, data + take);
        data += take;
        size -= take;
        if (payload_.size() == compressed_size_) {
            decode_payload();
            in_payload_ = false;
            header_fill_ = 0;
            ++next_chunk_;
            --remaining_chunks_;
        }
    }
}

void XetChunkDecoder::end() {
    if (in_payload_ || header_fill_ != 0) throw XetProtocolError("xorb range ends inside a chunk");
    if (remaining_chunks_ != 0) throw XetProtocolError("xorb range has fewer chunks than described");
}

void XetChunkDecoder::lz4_frame_decode(std::vector<uint8_t> & out) {
    auto * context = static_cast<LZ4F_dctx *>(lz4_);
    LZ4F_resetDecompressionContext(context);
    // One spare byte detects frames that decode to more than the header says.
    out.resize(uncompressed_size_ + 1);
    size_t source = 0;
    size_t target = 0;
    for (;;) {
        size_t source_size = payload_.size() - source;
        size_t target_size = out.size() - target;
        const size_t hint = LZ4F_decompress(context, out.data() + target, &target_size,
            payload_.data() + source, &source_size, nullptr);
        if (LZ4F_isError(hint)) {
            throw XetProtocolError(std::string("invalid LZ4 frame in xorb chunk: ") + LZ4F_getErrorName(hint));
        }
        source += source_size;
        target += target_size;
        if (hint == 0) break;
        if (source == payload_.size() || (source_size == 0 && target_size == 0)) {
            throw XetProtocolError("truncated LZ4 frame in xorb chunk");
        }
    }
    if (source != payload_.size()) throw XetProtocolError("trailing bytes after LZ4 frame in xorb chunk");
    if (target != uncompressed_size_) throw XetProtocolError("LZ4 xorb chunk has the wrong decompressed size");
    out.resize(target);
}

void XetChunkDecoder::decode_payload() {
    switch (scheme_) {
    case 0:
        sink_(next_chunk_, payload_.data(), payload_.size());
        return;
    case 1:
        lz4_frame_decode(output_);
        sink_(next_chunk_, output_.data(), output_.size());
        return;
    case 2:
        lz4_frame_decode(scratch_);
        output_.resize(scratch_.size());
        bg4_ungroup(scratch_.data(), scratch_.size(), output_.data());
        sink_(next_chunk_, output_.data(), output_.size());
        return;
    default:
        throw XetProtocolError("unsupported xorb chunk compression scheme");
    }
}

// ---------------------------------------------------------------------------

void xet_download(
    const XetFileRequest & request,
    const std::filesystem::path & destination,
    const XetOptions & options,
    const std::shared_ptr<std::atomic_bool> & cancelled,
    const std::function<void(uint64_t)> & progress) {
    if (!is_xet_hash(request.xet_hash)) throw XetUnavailable("file has no valid Xet hash");
    if (request.size == 0) throw XetUnavailable("Xet transfer needs a known, non-empty file size");
    if (options.segment_bytes == 0) throw std::invalid_argument("Xet segment size must be positive");

    std::filesystem::create_directories(destination.parent_path());
    {
        std::ofstream create(destination, std::ios::binary | std::ios::trunc);
        if (!create) throw std::runtime_error("could not create " + destination.string());
    }
    std::filesystem::resize_file(destination, request.size);

    TokenProvider tokens(request, cancelled, options);
    std::unique_ptr<PrefixHasher> hasher;
    if (options.verify_sha256 && is_xet_hash(request.expected_sha256)) {
        hasher = std::make_unique<PrefixHasher>(destination, cancelled);
    }
    for (uint64_t segment_start = 0; segment_start < request.size;) {
        const uint64_t segment_end = std::min(request.size, segment_start + options.segment_bytes);
        std::vector<char> done;
        std::vector<uint64_t> term_starts;
        size_t prefix_term = 0;
        std::vector<std::string> signature;
        std::vector<uint64_t> term_progress;
        uint64_t segment_progress = 0;
        uint64_t segment_reported = 0;
        for (int refresh = 0;; ++refresh) {
            if (refresh > options.max_refreshes) {
                throw XetUnavailable("signed xorb URLs kept expiring or failing authorization");
            }
            if (is_cancelled(cancelled)) throw Cancelled();
            const auto recipe = fetch_recipe(tokens, request.xet_hash, segment_start, segment_end - 1,
                options, cancelled);
            std::vector<std::string> current;
            current.reserve(recipe.terms.size());
            for (const auto & term : recipe.terms) {
                current.push_back(term.xorb + ":" + std::to_string(term.chunks.start) + ":" +
                    std::to_string(term.chunks.end));
            }
            // Completed terms survive a refresh only if the recipe is unchanged.
            if (current != signature) {
                signature = std::move(current);
                done.assign(recipe.terms.size(), 0);
                prefix_term = 0;
                // Term indices changed meaning; bytes already reported stay
                // reported, so progress pauses until the redo catches up.
                term_progress.assign(recipe.terms.size(), 0);
                segment_progress = 0;
            }
            auto jobs = plan_jobs(recipe, segment_start, segment_end, done, term_starts);
            if (jobs.empty()) break;

            TransferState state;
            state.options = &options;
            state.cancelled = cancelled;
            state.destination = destination;
            state.segment_start = segment_start;
            state.segment_end = segment_end;
            state.done = &done;
            state.term_starts = &term_starts;
            state.prefix_term = &prefix_term;
            state.hasher = hasher.get();
            state.progress = &progress;
            state.progress_base = segment_start;
            state.term_progress = &term_progress;
            state.segment_progress = &segment_progress;
            state.segment_reported = &segment_reported;
            if (run_jobs(jobs, state)) break;
        }
        if (hasher) hasher->advance(segment_end);
        segment_start = segment_end;
    }

    const auto size = std::filesystem::file_size(destination);
    if (size != request.size) {
        throw XetIntegrityError("Xet reconstruction size mismatch: expected " +
            std::to_string(request.size) + ", got " + std::to_string(size));
    }
    if (hasher) {
        const auto digest = hasher->finish(request.size);
        if (digest != request.expected_sha256) {
            throw XetIntegrityError("Xet reconstruction SHA-256 mismatch: expected " +
                request.expected_sha256 + ", got " + digest);
        }
    }
}

}  // namespace engine::package_manager::detail
