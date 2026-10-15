#include "engine/community_models/fish_drama3/assets.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
namespace assets = engine::assets;

template <typename T>
std::vector<unsigned char> bytes_for(const std::vector<T> & values) {
    std::vector<unsigned char> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template <typename F>
void require_failure(F action) {
    try { action(); } catch (const std::exception &) { return; }
    throw std::runtime_error("invalid codec source was accepted");
}

struct FixtureDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("audiocpp_drama3_codec_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    FixtureDirectory() { std::filesystem::create_directories(path); }
    ~FixtureDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
}  // namespace

int main(int argc, char ** argv) {
    try {
        FixtureDirectory directory;
        const std::string prefix = "decoder.transformer.layers.0.attention.";
        const std::string packed_name = prefix + "wqkv.weight";
        const char * projections[] = {"q_proj", "k_proj", "v_proj"};
        constexpr size_t width = 8;
        constexpr size_t count = width * width;
        std::vector<float> values(3 * count);
        for (size_t i = 0; i < values.size(); ++i) values[i] = (static_cast<float>(i) - 91) / 128;
        std::vector<ggml_fp16_t> half(values.size());
        std::vector<ggml_bf16_t> bfloat(values.size());
        ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
        ggml_fp32_to_bf16_row(values.data(), bfloat.data(), values.size());
        for (const auto & fixture : std::vector<std::pair<std::string, std::vector<unsigned char>>>{
            {"F32", bytes_for(values)}, {"F16", bytes_for(half)}, {"BF16", bytes_for(bfloat)}}) {
            const auto packed_path = directory.path / (fixture.first + "-packed.safetensors");
            const auto split_path = directory.path / (fixture.first + "-split.safetensors");
            engine::io::write_safetensors_file(packed_path, {{packed_name, fixture.first, {3 * width, width}, fixture.second}});
            std::vector<engine::io::SafeTensorWriteEntry> entries;
            const auto byte_count = fixture.second.size() / 3;
            for (size_t part = 0; part < 3; ++part) {
                const auto begin = fixture.second.begin() + part * byte_count;
                entries.push_back({prefix + projections[part] + ".weight", fixture.first, {width, width}, {begin, begin + byte_count}});
            }
            engine::io::write_safetensors_file(split_path, entries);
            const auto original = assets::open_tensor_source(packed_path);
            const auto adapted = engine::models::fish_drama3::make_drama3_codec_source(original);
            const auto split = assets::open_tensor_source(split_path);
            if (adapted->source_path() != packed_path || adapted->has_tensor("missing") || adapted->optional_f32("missing"))
                throw std::runtime_error("codec source delegation failed");
            if (adapted->require_tensor_data(packed_name).bytes != original->require_tensor_data(packed_name).bytes)
                throw std::runtime_error("packed source changed");
            for (const auto * projection : projections) {
                const auto name = prefix + projection + ".weight";
                const auto metadata = adapted->require_metadata(name);
                if (!adapted->has_tensor(name) || metadata.shape != std::vector<int64_t>{width, width} ||
                    metadata.dtype != split->require_metadata(name).dtype || metadata.name != name ||
                    adapted->require_tensor(name, assets::TensorStorageType::Native, {width, width}).bytes !=
                        split->require_tensor(name, assets::TensorStorageType::Native, {width, width}).bytes ||
                    adapted->require_f32(name) != split->require_f32(name)) {
                    throw std::runtime_error("codec slice differs from an independent split checkpoint");
                }
                require_failure([&] { (void) adapted->require_f32(name, std::vector<int64_t>{1, width}); });
            }
            const auto passthrough = engine::models::fish_drama3::make_drama3_codec_source(split);
            if (passthrough->tensors().size() != split->tensors().size()) throw std::runtime_error("split checkpoint changed");
            for (const auto & entry : entries) {
                if (passthrough->require_tensor_data(entry.name).bytes != split->require_tensor_data(entry.name).bytes)
                    throw std::runtime_error("split checkpoint bytes changed");
            }
            entries.push_back({packed_name, fixture.first, {3 * width, width}, fixture.second});
            const auto both_path = directory.path / (fixture.first + "-both.safetensors");
            engine::io::write_safetensors_file(both_path, entries);
            const auto both = engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(both_path));
            if (both->tensors().size() != entries.size()) throw std::runtime_error("complete split weights were shadowed");
            entries.erase(entries.begin() + 1, entries.begin() + 3);
            const auto partial_path = directory.path / (fixture.first + "-partial.safetensors");
            engine::io::write_safetensors_file(partial_path, entries);
            require_failure([&] { (void) engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(partial_path)); });
        }
        const auto invalid_path = directory.path / "invalid-shape.safetensors";
        engine::io::write_safetensors_file(invalid_path, {{packed_name, "F32", {3, width, width}, bytes_for(values)}});
        require_failure([&] { (void) engine::models::fish_drama3::make_drama3_codec_source(assets::open_tensor_source(invalid_path)); });
        require_failure([] { (void) engine::models::fish_drama3::make_drama3_codec_source(nullptr); });
        std::cout << "F32/F16/BF16 packed bytes and decoded values match split checkpoints; split passthrough and invalid inputs passed\n";
        // Optional real-package check needs no GPU and never loads backbone weights.
        for (int argument = 1; argument < argc; ++argument) {
            const auto model = engine::models::fish_drama3::load_drama3_assets(argv[argument]);
            const auto original = model->resources.open_tensor_source("codec_weights");
            const std::string suffix = ".attention.wqkv.weight";
            size_t checked = 0;
            for (const auto & tensor : original->tensors()) {
                if (tensor.name.size() < suffix.size() ||
                    tensor.name.compare(tensor.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
                const auto packed = original->require_f32(tensor.name);
                const auto part_size = packed.size() / 3;
                const auto layer = tensor.name.substr(0, tensor.name.size() - suffix.size()) + ".attention.";
                for (size_t part = 0; part < 3; ++part) {
                    const auto name = layer + projections[part] + ".weight";
                    const auto values = model->codec_weights->require_f32(name);
                    if (values.size() != part_size || !std::equal(values.begin(), values.end(), packed.begin() + part * part_size))
                        throw std::runtime_error("real codec slice differs from the original packed checkpoint");
                    ++checked;
                }
            }
            if (checked == 0) throw std::runtime_error("real checkpoint has no packed codec weights");
            std::cout << "real_package=" << argv[argument] << " exact_QKV_slices=" << checked << '\n';
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
