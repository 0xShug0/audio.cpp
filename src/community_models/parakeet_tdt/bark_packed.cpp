#include "engine/community_models/parakeet_tdt/bark_packed.h"
#include "engine/community_models/parakeet_tdt/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "third_party/xz/xz.h"
#include <ggml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace engine::community_models::parakeet_tdt {
namespace {
namespace json = engine::io::json;
using Bytes = std::vector<uint8_t>;

[[noreturn]] void invalid(const std::string & detail) {
    throw std::runtime_error("invalid Bark packed checkpoint: " + detail);
}

size_t size_field(const json::Value & object, const std::string & key, size_t limit) {
    const auto value = object.require(key).as_number();
    if (!std::isfinite(value) || value < 0 || std::floor(value) != value || value > static_cast<double>(limit))
        invalid("out-of-range " + key);
    return static_cast<size_t>(value);
}

uint32_t u32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint16_t u16(const uint8_t * p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint64_t u64(const uint8_t * p) { return uint64_t(u32(p)) | uint64_t(u32(p + 4)) << 32; }

// constriction 0.5.0's DefaultAnsCoder (u32 words/u64 state, 24-bit
// probabilities), and Categorical(perfect=False). Keep its two FP64
// normalization passes and symbol order: changing either changes the stream.
class Ans {
public:
    explicit Ans(const Bytes & bytes) : bytes_(bytes), words_(bytes.size() / 4) {
        if (bytes.size() % 4) invalid("unaligned ANS stream");
        if (words_) {
            state_ = pop();
            if (!state_) invalid("zero final ANS word");
            if (words_) state_ = (state_ << 32) | pop();
        }
    }
    static std::vector<uint32_t> cdf(const uint8_t * counts, size_t n) {
        double total = 0;
        for (size_t i = 0; i < n; ++i) total += u32(counts + i * 4);
        if (!total) invalid("empty ANS probability model");
        std::vector<double> probabilities(n);
        double normalization = 0;
        for (size_t i = 0; i < n; ++i) {
            probabilities[i] = double(u32(counts + i * 4)) / total;
            normalization += probabilities[i];
        }
        const double scale = double((1u << 24) - n) / normalization;
        double cumulative = 0;
        std::vector<uint32_t> result(n + 1);
        for (size_t i = 0; i < n; ++i) {
            result[i] = static_cast<uint32_t>(cumulative * scale) + static_cast<uint32_t>(i);
            cumulative += probabilities[i];
        }
        result[n] = 1u << 24;
        for (size_t i = 0; i < n; ++i)
            if (result[i + 1] <= result[i]) invalid("invalid ANS CDF");
        return result;
    }
    size_t decode(const std::vector<uint32_t> & cdf) {
        const uint32_t q = uint32_t(state_) & ((1u << 24) - 1);
        const size_t symbol = std::upper_bound(cdf.begin(), cdf.end(), q) - cdf.begin() - 1;
        state_ = (state_ >> 24) * (cdf[symbol + 1] - cdf[symbol]) + q - cdf[symbol];
        if (state_ < (uint64_t(1) << 32) && words_) state_ = (state_ << 32) | pop();
        return symbol;
    }
private:
    uint32_t pop() { return u32(bytes_.data() + --words_ * 4); }
    const Bytes & bytes_;
    size_t words_;
    uint64_t state_ = 0;
};

struct Blob { size_t offset = 0, size = 0; };
struct Record {
    assets::TensorMetadata metadata;
    std::string kind;
    size_t count = 1, levels = 0;
    Blob main, side, model;
    bool compressed = false;
};

Bytes slice(const Bytes & storage, Blob blob) {
    return Bytes(storage.begin() + blob.offset, storage.begin() + blob.offset + blob.size);
}

Bytes unpack(const Bytes & storage, Blob blob, bool compressed, size_t expected) {
    if (!compressed) {
        if (blob.size != expected) invalid("wrong side blob size");
        return slice(storage, blob);
    }
    static std::once_flag crc_initialized;
    std::call_once(crc_initialized, [] { xz_crc32_init(); xz_crc64_init(); });
    std::unique_ptr<xz_dec, decltype(&xz_dec_end)> decoder(xz_dec_init(XZ_SINGLE, 0), xz_dec_end);
    if (!decoder) throw std::bad_alloc();
    Bytes output(expected);
    xz_buf buffer{};
    buffer.in = storage.data() + blob.offset;
    buffer.in_size = blob.size;
    buffer.out = output.data();
    buffer.out_size = output.size();
    if (xz_dec_run(decoder.get(), &buffer) != XZ_STREAM_END || buffer.in_pos != blob.size || buffer.out_pos != expected)
        invalid("corrupt or oversized XZ blob");
    return output;
}

class BarkSource final : public assets::TensorSource {
public:
    explicit BarkSource(std::filesystem::path path) : path_(std::move(path)) {
        const auto storage = read();
        if (storage->size() < 8) invalid("truncated header");
        const uint64_t header_size = u64(storage->data());
        if (!header_size || header_size > 4 * 1024 * 1024 || header_size > storage->size() - 8)
            invalid("header size");
        const auto header = json::parse(std::string_view(reinterpret_cast<const char *>(storage->data() + 8), size_t(header_size)));
        header_bytes_.assign(storage->begin(), storage->begin() + 8 + size_t(header_size));
        file_size_ = storage->size();
        if (json::require_string(header, "format") != "bark-levels-v3") invalid("unsupported format");
        config_ = json::parse(json::require_string(header, "config"));
        size_t offset = 8 + size_t(header_size);
        const auto & index = header.require("index").as_array();
        if (index.empty() || index.size() > 1024) invalid("tensor inventory");
        auto blob = [&](const json::Value & e, const char * field) {
            const size_t bytes = size_field(e, field, 256 * 1024 * 1024);
            if (bytes > storage->size() - offset) invalid("truncated blob");
            Blob result{offset, bytes};
            offset += bytes;
            return result;
        };
        for (const auto & entry : index) {
            Record r;
            r.metadata.name = json::require_string(entry, "n");
            if (r.metadata.name.empty() || r.metadata.name.size() > 512) invalid("tensor name");
            const auto & shape = entry.require("shape").as_array();
            if (shape.size() > 4) invalid("tensor rank");
            for (const auto & dimension : shape) {
                const auto value = dimension.as_number();
                if (!std::isfinite(value) || value < 1 || value > 64 * 1024 * 1024 || value != std::floor(value)) invalid("shape");
                const auto d = static_cast<size_t>(value);
                if (r.count > 64 * 1024 * 1024 / d) invalid("oversized tensor");
                r.count *= d;
                r.metadata.shape.push_back(static_cast<int64_t>(d));
            }
            r.kind = json::require_string(entry, "k");
            if (r.kind == "raw") {
                const auto dtype = json::require_string(entry, "dtype");
                r.metadata.dtype = dtype == "torch.float32" ? "F16" : dtype == "torch.int64" ? "I64" : dtype == "torch.int32" ? "I32" : "";
                if (r.metadata.dtype.empty()) invalid("raw dtype");
                r.compressed = json::optional_bool(entry, "zr", false);
            } else {
                if (shape.size() < 2) invalid("ANS tensor rank");
                if (r.kind == "levels2") {
                    r.levels = size_field(entry, "K", 3);
                    if (!r.levels) invalid("level count");
                    r.side = blob(entry, "lev");
                    r.compressed = json::optional_bool(entry, "zl", false);
                    r.metadata.dtype = "F16";
                } else if (r.kind == "ansint5" || r.kind == "ansint6") {
                    r.side = blob(entry, "s");
                    r.compressed = json::optional_bool(entry, "zs", false);
                    r.metadata.dtype = "F32";
                } else invalid("unknown tensor codec");
                r.model = blob(entry, "meta");
            }
            r.main = blob(entry, "b");
            if (projection(r.metadata.name) || gamma(r.metadata.name)) r.metadata.dtype = "F32";
            const auto name = r.metadata.name;
            if (!records_.emplace(name, std::move(r)).second) invalid("duplicate tensor");
        }
        if (offset != storage->size()) invalid("trailing bytes");
        storage_ = storage;
    }

    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override { return records_.find(std::string(name)) != records_.end(); }
    assets::TensorMetadata require_metadata(std::string_view name) const override { return record(name).metadata; }
    std::vector<assets::TensorMetadata> tensors() const override {
        std::vector<assets::TensorMetadata> result;
        for (const auto & item : records_) result.push_back(item.second.metadata);
        std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.name < b.name; });
        return result;
    }
    void release_storage() const override { std::lock_guard<std::mutex> lock(mutex_); storage_.reset(); }
    std::vector<float> require_f32(std::string_view name, const std::optional<std::vector<int64_t>> & shape) const override {
        const auto & r = record(name);
        if (shape && *shape != r.metadata.shape) invalid("shape mismatch: " + std::string(name));
        const auto storage = snapshot();
        auto values = decode(r, *storage);
        if (projection(r.metadata.name)) for (auto & value : values) value /= 32.f;
        if (gamma(r.metadata.name)) {
            auto variance_name = r.metadata.name.substr(0, r.metadata.name.size() - 6) + "running_var";
            const auto & variance_record = record(variance_name);
            if (variance_record.metadata.shape != r.metadata.shape) invalid("batch norm variance shape");
            const auto variance = decode(variance_record, *storage);
            for (size_t i = 0; i < values.size(); ++i) {
                const double v = variance[i];
                if (v < 0) invalid("negative batch norm variance");
                values[i] = static_cast<float>(double(values[i]) * std::sqrt((std::max(v, 1e-5) + 1e-5) / (v + 1e-5)));
            }
        }
        return values;
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name, const std::optional<std::vector<int64_t>> & shape) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, shape);
    }
    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto & r = record(name);
        assets::RawTensorData result{r.metadata, {}};
        if (r.metadata.dtype == "I64" || r.metadata.dtype == "I32") {
            const auto storage = snapshot();
            const auto bytes = unpack(*storage, r.main, r.compressed, r.count * (r.metadata.dtype == "I64" ? 8 : 4));
            result.bytes.resize(bytes.size());
            std::memcpy(result.bytes.data(), bytes.data(), bytes.size());
            return result;
        }
        const auto values = require_f32(name, std::nullopt);
        result.bytes.resize(values.size() * (r.metadata.dtype == "F16" ? 2 : 4));
        if (r.metadata.dtype == "F16") {
            for (size_t i = 0; i < values.size(); ++i) {
                const auto half = ggml_fp32_to_fp16(values[i]);
                std::memcpy(result.bytes.data() + i * 2, &half, 2);
            }
        } else std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
        return result;
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        const auto & r = record(name);
        if (r.count != 1 || (r.metadata.dtype != "I64" && r.metadata.dtype != "I32")) invalid("not an integer scalar");
        const auto storage = snapshot();
        const auto bytes = unpack(*storage, r.main, r.compressed, r.metadata.dtype == "I64" ? 8 : 4);
        return r.metadata.dtype == "I64" ? static_cast<int64_t>(u64(bytes.data())) : static_cast<int32_t>(u32(bytes.data()));
    }
    const json::Value & config() const { return config_; }

private:
    static bool projection(const std::string & n) { return n == "encoder.subsampling.linear.weight" || n == "encoder.subsampling.linear.bias"; }
    static bool gamma(const std::string & n) { const std::string suffix = ".conv.norm.weight"; return n.size() >= suffix.size() && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0; }
    const Record & record(std::string_view name) const {
        const auto it = records_.find(std::string(name));
        if (it == records_.end()) invalid("missing tensor: " + std::string(name));
        return it->second;
    }
    std::shared_ptr<const Bytes> read() const {
        std::ifstream file(path_, std::ios::binary | std::ios::ate);
        if (!file) invalid("cannot open file");
        const auto length = file.tellg();
        if (length < 8 || length > 256 * 1024 * 1024) invalid("file size");
        auto storage = std::make_shared<Bytes>(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char *>(storage->data()), static_cast<std::streamsize>(length))) invalid("read failed");
        return storage;
    }
    std::shared_ptr<const Bytes> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!storage_) {
            const auto storage = read();
            if (storage->size() != file_size_ || !std::equal(header_bytes_.begin(), header_bytes_.end(), storage->begin()))
                invalid("checkpoint changed after loading");
            storage_ = storage;
        }
        return storage_;
    }
    static std::vector<float> decode(const Record & r, const Bytes & storage) {
        std::vector<float> result(r.count);
        if (r.kind == "raw") {
            if (r.metadata.dtype == "I64" || r.metadata.dtype == "I32") invalid("integer tensor requested as floating point");
            const auto bytes = unpack(storage, r.main, r.compressed, r.count * 2);
            for (size_t i = 0; i < r.count; ++i) result[i] = ggml_fp16_to_fp32(u16(bytes.data() + 2 * i));
        } else {
            const size_t rows = static_cast<size_t>(r.metadata.shape[0]), cols = r.count / rows;
            auto stream = slice(storage, r.main);
            Ans ans(stream);
            if (r.kind == "levels2") {
                const auto levels = unpack(storage, r.side, r.compressed, rows * r.levels * 2);
                const size_t packed = (cols + 1) / 2, symbols = 2 * r.levels + 1;
                if (r.model.size != packed + 16 * symbols * 4) invalid("level metadata size");
                const auto * model = storage.data() + r.model.offset;
                for (size_t group = 0; group < 16; ++group) {
                    const auto cdf = Ans::cdf(model + packed + group * symbols * 4, symbols);
                    for (size_t col = 0; col < cols; ++col) {
                        const auto cls = (col % 2 == 0) ? model[col / 2] >> 4 : model[col / 2] & 15;
                        if (cls != group) continue;
                        for (size_t row = 0; row < rows; ++row) {
                            const size_t code = ans.decode(cdf);
                            float value = code ? ggml_fp16_to_fp32(u16(levels.data() + (row * r.levels + (code - 1) % r.levels) * 2)) : 0.f;
                            result[row * cols + col] = code > r.levels ? -value : value;
                        }
                    }
                }
            } else {
                const size_t symbols = r.kind == "ansint5" ? 31 : 63;
                if (r.model.size != symbols * 4) invalid("integer metadata size");
                const auto cdf = Ans::cdf(storage.data() + r.model.offset, symbols);
                const auto scales = unpack(storage, r.side, r.compressed, rows * 2);
                for (size_t row = 0; row < rows; ++row) {
                    const float scale = ggml_fp16_to_fp32(u16(scales.data() + row * 2));
                    for (size_t col = 0; col < cols; ++col)
                        result[row * cols + col] = float(int(ans.decode(cdf)) - int(symbols / 2)) * scale;
                }
            }
        }
        for (const auto value : result) if (!std::isfinite(value)) invalid("nonfinite weight");
        return result;
    }
    std::filesystem::path path_;
    json::Value config_;
    Bytes header_bytes_;
    size_t file_size_ = 0;
    std::unordered_map<std::string, Record> records_;
    mutable std::mutex mutex_;
    mutable std::shared_ptr<const Bytes> storage_;
};
} // namespace

std::shared_ptr<const assets::TensorSource> open_bark_packed_source(const std::filesystem::path & path) {
    return std::make_shared<BarkSource>(path);
}

namespace {
std::filesystem::path package_root(const std::filesystem::path & path) {
    return path.extension() == ".bark" ? path.parent_path() : path;
}
std::filesystem::path checkpoint_path(const std::filesystem::path & path) {
    return path.extension() == ".bark" ? path : path / "model.bark";
}
std::filesystem::path spec_path(const runtime::ModelLoadRequest & request) {
    return request.model_spec_override.value_or(package_root(request.model_path) / "model_spec.json");
}

class PackedLoader final : public runtime::IVoiceModelLoader {
public:
    explicit PackedLoader(std::shared_ptr<runtime::IVoiceModelLoader> delegate) : delegate_(std::move(delegate)) {}
    std::string family() const override { return delegate_->family(); }
    std::vector<std::string> family_aliases() const override { return delegate_->family_aliases(); }
    runtime::CapabilitySet advertised_capabilities() const override { return delegate_->advertised_capabilities(); }
    std::string advertised_instructions_policy() const override { return delegate_->advertised_instructions_policy(); }
    std::vector<std::string> advertised_api_endpoints() const override { return delegate_->advertised_api_endpoints(); }
    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (!is_bark_packed_package(request.model_path)) return delegate_->can_load(request);
        if (request.family_hint && *request.family_hint != family()) {
            const auto aliases = family_aliases();
            if (std::find(aliases.begin(), aliases.end(), *request.family_hint) == aliases.end()) return false;
        }
        try {
            const engine::model_spec::ScopedSpecOverride scope(spec_path(request), request.model_path);
            (void)runtime::require_model_contract(family());
            return true;
        } catch (const std::exception &) { return false; }
    }
    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (!is_bark_packed_package(request.model_path)) return delegate_->inspect(request);
        const engine::model_spec::ScopedSpecOverride scope(spec_path(request), request.model_path);
        const auto contract = runtime::require_model_contract(family());
        runtime::ModelInspection result;
        result.metadata = contract->metadata;
        result.capabilities = contract->capabilities;
        result.cli = contract->cli;
        result.model_root = package_root(request.model_path);
        result.discovered_weights.push_back({"weights", checkpoint_path(request.model_path)});
        result.discovered_configs.push_back({"config", result.model_root / "config.json"});
        result.discovered_configs.push_back({"processor_config", result.model_root / "processor_config.json"});
        result.discovered_configs.push_back({"tokenizer_json", result.model_root / "tokenizer.json"});
        return result;
    }
    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        if (!is_bark_packed_package(request.model_path)) return delegate_->load(request);
        const engine::model_spec::ScopedSpecOverride scope(spec_path(request), request.model_path);
        return delegate_->load(request);
    }
    runtime::ModelResourcePlan model_resources(const std::filesystem::path & path,
        const std::unordered_map<std::string, std::string> & options) const override {
        if (!is_bark_packed_package(path)) return delegate_->model_resources(path, options);
        // The entropy-coded download size is not its runtime residency. A
        // conservative FP32 weight budget prevents pricing it as a 163 MB model.
        runtime::ModelResourcePlan result;
        result.resident_bytes = 2508229144ULL;
        return result;
    }
    runtime::PackageLoadTarget resolve_package_load(const assets::ModelPackage & package) const override {
        return delegate_->resolve_package_load(package);
    }
private:
    std::shared_ptr<runtime::IVoiceModelLoader> delegate_;
};
} // namespace

bool is_bark_packed_package(const std::filesystem::path & path) {
    std::error_code error;
    return std::filesystem::is_regular_file(checkpoint_path(path), error) &&
        std::filesystem::is_regular_file(package_root(path) / "model_spec.json", error);
}

std::shared_ptr<const ParakeetTDTAssets> load_bark_packed_assets(const std::filesystem::path & path) {
    const auto root = package_root(path);
    auto source = std::make_shared<BarkSource>(checkpoint_path(path));
    assets::ResourceBundle resources(root);
    resources.add_model_file("config", "config.json");
    resources.add_model_file("processor_config", "processor_config.json");
    resources.add_model_file("tokenizer_json", "tokenizer.json");
    auto result = make_parakeet_assets(std::move(resources), source);
    if (result->config.variant != "underdog_bark3_standard" && result->config.variant != "underdog_bark3_plus")
        invalid("compressed loading requires Standard or Plus metadata");
    // Package preparation adds only the variant tag. Every publisher-supplied
    // config field must still agree with the container that supplies the weights.
    const auto package_config = json::parse_file(root / "config.json");
    for (const auto & item : source->config().as_object()) {
        // JSON objects are unordered maps; compare subtrees recursively rather
        // than relying on member iteration order.
        const auto equal = [](const auto & self, const json::Value & a, const json::Value & b) -> bool {
            if (a.kind() != b.kind()) return false;
            if (a.is_object()) {
                if (a.as_object().size() != b.as_object().size()) return false;
                for (const auto & value : a.as_object()) {
                    const auto * other = b.find(value.first);
                    if (!other || !self(self, value.second, *other)) return false;
                }
                return true;
            }
            if (a.is_array()) {
                if (a.as_array().size() != b.as_array().size()) return false;
                for (size_t i = 0; i < a.as_array().size(); ++i)
                    if (!self(self, a.as_array()[i], b.as_array()[i])) return false;
                return true;
            }
            return json::stringify(a) == json::stringify(b);
        };
        const auto * value = package_config.find(item.first);
        if (!value || !equal(equal, item.second, *value)) invalid("package config differs from container: " + item.first);
    }
    const auto & config = result->config;
    if (config.encoder.layers != 24 || config.encoder.hidden_size != 1024 || config.encoder.heads != 8 ||
        config.encoder.intermediate_size != 4096 || config.vocab_size != 8193 || config.decoder_hidden_size != 640 ||
        config.blank_token_id != 8192 || config.pad_token_id != 2 || config.frontend.feature_size != 128 ||
        config.encoder.conv_kernel != 9 || config.durations != std::vector<int32_t>({0, 1, 2, 3, 4}) ||
        json::require_bool(source->config().require("encoder_config"), "scale_input") || source->tensors().size() != 723)
        invalid("unsupported Bark architecture");
    return result;
}

std::shared_ptr<runtime::IVoiceModelLoader> wrap_bark_packed_loader(std::shared_ptr<runtime::IVoiceModelLoader> loader) {
    return std::make_shared<PackedLoader>(std::move(loader));
}

} // namespace engine::community_models::parakeet_tdt
