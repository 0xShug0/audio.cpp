#include "engine/models/kokoro_tts/package.h"
#include "engine/framework/io/binary.h"
#include <gguf.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>

namespace engine::models::kokoro_tts {
namespace {
using namespace engine::assets;
class GgufSource final : public TensorSource {
    std::filesystem::path path_;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf_{nullptr, gguf_free};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors_{nullptr, ggml_free};
    std::map<std::string, std::string> names_;
public:
    explicit GgufSource(const std::filesystem::path & path) : path_(path) {
        ggml_context * tensors = nullptr;
        gguf_.reset(gguf_init_from_file(path.string().c_str(), {true, &tensors}));
        tensors_.reset(tensors);
        if (!gguf_ || !tensors_) throw std::runtime_error("Invalid Kokoro GGUF: " + path.string());
        const auto architecture = gguf_find_key(gguf_.get(), "general.architecture");
        if (architecture < 0 || gguf_get_kv_type(gguf_.get(), architecture) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(gguf_.get(), architecture)) != "kokoro_tts")
            throw std::runtime_error("Expected a kokoro_tts GGUF architecture");
        const auto names = gguf_find_key(gguf_.get(), "kokoro.tensor_names");
        if (names < 0 || gguf_get_kv_type(gguf_.get(), names) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(gguf_.get(), names) != GGUF_TYPE_STRING ||
            gguf_get_arr_n(gguf_.get(), names) != static_cast<size_t>(gguf_get_n_tensors(gguf_.get())))
            throw std::runtime_error("Missing Kokoro tensor name mapping");
        for (int64_t i = 0; i < gguf_get_n_tensors(gguf_.get()); ++i)
            if (!names_.emplace(gguf_get_arr_str(gguf_.get(), names, i), gguf_get_tensor_name(gguf_.get(), i)).second)
                throw std::runtime_error("Duplicate Kokoro tensor mapping");
        for (auto * t = ggml_get_first_tensor(tensors_.get()); t; t = ggml_get_next_tensor(tensors_.get(), t)) {
            if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 &&
                t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_Q8_0)
                throw std::runtime_error("Unsupported Kokoro GGUF tensor type: " + std::string(t->name));
        }
    }
    const gguf_context * metadata() const { return gguf_.get(); }
    const std::filesystem::path & source_path() const noexcept override { return path_; }
    bool has_tensor(std::string_view name) const noexcept override {
        return names_.find(std::string(name)) != names_.end();
    }
    TensorMetadata require_metadata(std::string_view name) const override {
        auto found = names_.find(std::string(name));
        auto * t = found == names_.end() ? nullptr : ggml_get_tensor(tensors_.get(), found->second.c_str());
        if (!t) throw std::runtime_error("Missing Kokoro tensor: " + std::string(name));
        TensorMetadata m{std::string(name), ggml_type_name(t->type), {}};
        // GGUF dimensions use the reverse of the source checkpoint order.
        for (int i = ggml_n_dims(t) - 1; i >= 0; --i) m.shape.push_back(t->ne[i]);
        // GGUF elides trailing singleton dimensions; retain the source shape metadata.
        auto key = "kokoro.tensor_shape." + std::string(name);
        auto id = gguf_find_key(gguf_.get(), key.c_str());
        if (id >= 0) {
            if (gguf_get_kv_type(gguf_.get(), id) != GGUF_TYPE_ARRAY ||
                gguf_get_arr_type(gguf_.get(), id) != GGUF_TYPE_INT64)
                throw std::runtime_error("Invalid Kokoro tensor shape metadata");
            const auto n = gguf_get_arr_n(gguf_.get(), id);
            const auto * dims = static_cast<const int64_t *>(gguf_get_arr_data(gguf_.get(), id));
            if (!n || n > 4) throw std::runtime_error("Invalid Kokoro tensor rank");
            m.shape.assign(dims, dims + n);
            for (size_t i = 0; i < n; ++i)
                if (dims[i] <= 0 || dims[i] != t->ne[n - i - 1])
                    throw std::runtime_error("Inconsistent Kokoro tensor shape");
        }
        return m;
    }
    std::vector<TensorMetadata> tensors() const override {
        std::vector<TensorMetadata> out;
        for (const auto & item : names_) out.push_back(require_metadata(item.first));
        return out;
    }
    RawTensorData require_tensor_data(std::string_view name) const override {
        RawTensorData out{require_metadata(name), {}};
        const auto id = gguf_find_tensor(gguf_.get(), names_.at(std::string(name)).c_str());
        out.bytes.resize(gguf_get_tensor_size(gguf_.get(), id));
        std::ifstream in(path_, std::ios::binary);
        in.seekg(gguf_get_data_offset(gguf_.get()) + gguf_get_tensor_offset(gguf_.get(), id));
        if (!in.read(reinterpret_cast<char *>(out.bytes.data()), out.bytes.size()))
            throw std::runtime_error("Truncated Kokoro GGUF tensor: " + std::string(name));
        return out;
    }
    std::vector<float> require_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected = std::nullopt) const override {
        auto raw = require_tensor_data(name);
        if (expected && *expected != raw.metadata.shape) throw std::runtime_error("Kokoro tensor shape mismatch");
        core::TensorShape shape;
        shape.rank = static_cast<int>(raw.metadata.shape.size());
        for (size_t i = 0; i < raw.metadata.shape.size(); ++i) shape.dims[i] = raw.metadata.shape[i];
        return tensor_data_to_f32(name, {shape,
            ggml_type_for_tensor_storage(tensor_storage_type_for_dtype(raw.metadata.dtype)), std::move(raw.bytes)});
    }
    std::optional<std::vector<float>> optional_f32(std::string_view name,
        const std::optional<std::vector<int64_t>> & expected = std::nullopt) const override {
        if (!has_tensor(name)) return std::nullopt;
        return require_f32(name, expected);
    }
    int64_t require_i64_scalar(std::string_view) const override {
        throw std::runtime_error("Kokoro GGUF does not contain integer scalars");
    }
};

void extract_resources(const gguf_context * g, const std::filesystem::path & root) {
    auto names = gguf_find_key(g, "audiocpp.embedded_files.names");
    auto offsets = gguf_find_key(g, "audiocpp.embedded_files.offsets");
    auto data = gguf_find_key(g, "audiocpp.embedded_files.data");
    if (names < 0 || offsets < 0 || data < 0 ||
        gguf_get_kv_type(g, names) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, names) != GGUF_TYPE_STRING ||
        gguf_get_kv_type(g, offsets) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, offsets) != GGUF_TYPE_UINT64 ||
        gguf_get_kv_type(g, data) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, data) != GGUF_TYPE_UINT8 ||
        gguf_get_arr_n(g, offsets) != gguf_get_arr_n(g, names) + 1)
        throw std::runtime_error("Kokoro GGUF is missing valid embedded resources");
    const auto * begin = static_cast<const char *>(gguf_get_arr_data(g, data));
    const auto * off = static_cast<const uint64_t *>(gguf_get_arr_data(g, offsets));
    const auto size = gguf_get_arr_n(g, data);
    std::set<std::filesystem::path> seen;
    for (size_t i = 0; i < gguf_get_arr_n(g, names); ++i) {
        std::string name = gguf_get_arr_str(g, names, i);
        std::replace(name.begin(), name.end(), '\\', '/');
        const auto relative = std::filesystem::u8path(name).lexically_normal();
        if (name.empty() || name.find(':') != std::string::npos || relative.has_root_path() ||
            relative.empty() || *relative.begin() == ".." || !seen.insert(relative).second ||
            off[i] > off[i + 1] || off[i + 1] > size)
            throw std::runtime_error("Unsafe Kokoro embedded resource");
        auto path = root / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::binary);
        if (!out.write(begin + off[i], off[i + 1] - off[i]))
            throw std::runtime_error("Cannot extract Kokoro resource: " + name);
    }
}
}

bool is_kokoro_gguf(const std::filesystem::path & path) noexcept {
    try {
        gguf_init_params params{true, nullptr};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
            gguf_init_from_file(path.string().c_str(), params), gguf_free);
        if (!metadata) return false;
        const auto architecture = gguf_find_key(metadata.get(), "general.architecture");
        return architecture >= 0 && gguf_get_kv_type(metadata.get(), architecture) == GGUF_TYPE_STRING &&
            std::string(gguf_get_val_str(metadata.get(), architecture)) == "kokoro_tts";
    } catch (...) {
        return false;
    }
}

KokoroPackage::~KokoroPackage() {
    if (temporary) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
}
std::shared_ptr<KokoroPackage> open_kokoro_package(const std::filesystem::path & path) {
    auto package = std::make_shared<KokoroPackage>();
    if (std::filesystem::is_directory(path)) {
        package->root = std::filesystem::canonical(path);
        package->weights = assets::open_tensor_source(package->root / "kokoro-v1_0.safetensors");
        return package;
    }
    auto source = std::make_shared<GgufSource>(std::filesystem::canonical(path));
    std::random_device rng;
    for (int tries = 0; tries < 20; ++tries) {
        package->root = std::filesystem::temp_directory_path() /
            ("audiocpp-kokoro-" + std::to_string(rng()) + "-" + std::to_string(rng()));
        if (std::filesystem::create_directory(package->root)) { package->temporary = true; break; }
    }
    if (!package->temporary) throw std::runtime_error("Cannot create Kokoro resource directory");
    extract_resources(source->metadata(), package->root);
    package->weights = source;
    return package;
}
}
