#include "engine/framework/codecs/fsq_audio_codec_runtime.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/io/safetensors.h"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
struct Fixture {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("audiocpp_fsq_reuse_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::vector<engine::io::SafeTensorWriteEntry> entries;
    Fixture() { std::filesystem::create_directory(root); }
    ~Fixture() { std::error_code error; std::filesystem::remove_all(root, error); }
    void tensor(const std::string & name, std::vector<int64_t> shape, bool norm = false, bool bias = false) {
        size_t size = 1;
        for (const auto dim : shape) size *= static_cast<size_t>(dim);
        std::vector<float> values(size);
        for (size_t i = 0; i < size; ++i) values[i] = norm ? 1.0F : bias ? 0.0F : 0.025F * std::sin(float(i) * 0.73F);
        std::vector<unsigned char> bytes(size * sizeof(float));
        std::memcpy(bytes.data(), values.data(), bytes.size());
        entries.push_back({name, "F32", std::move(shape), std::move(bytes)});
    }
    void linear(const std::string & name, int out, int in, bool bias) {
        tensor(name + ".weight", {out, in});
        if (bias) tensor(name + ".bias", {out}, false, true);
    }
    void norm(const std::string & name, int hidden, bool bias) {
        tensor(name + ".weight", {hidden}, true);
        if (bias) tensor(name + ".bias", {hidden}, false, true);
    }
    void conv(const std::string & name, int hidden, int kernel) {
        tensor(name + ".weight", {hidden, hidden, kernel});
        tensor(name + ".bias", {hidden}, false, true);
    }
};

void run() {
    using namespace engine;
    Fixture fixture;
    codecs::FsqAudioCodecConfig config;
    config.hidden_size = 32; config.intermediate_size = 64; config.layers = 2;
    config.attention_heads = 4; config.kv_heads = 4; config.head_dim = 8;
    config.quantization_dim = 8; config.quantization_levels = {8, 8, 8, 8};
    config.prior_blocks = 1; config.post_blocks = 1; config.hop_length = 16;
    config.deterministic_overlap_add = true;
    fixture.linear("quantizer.project_out", 8, 4, true);
    fixture.linear("acoustic_decoder.fc", 32, 8, true);
    fixture.conv("acoustic_decoder.embed", 32, 7);
    for (const std::string prefix : {"acoustic_decoder.prior_net.0", "acoustic_decoder.post_net.0"}) {
        fixture.norm(prefix + ".norm1", 32, true); fixture.norm(prefix + ".norm2", 32, true);
        fixture.conv(prefix + ".conv1", 32, 3); fixture.conv(prefix + ".conv2", 32, 3);
    }
    for (int layer = 0; layer < 2; ++layer) {
        const auto prefix = "acoustic_decoder.layers." + std::to_string(layer);
        fixture.norm(prefix + ".input_layernorm", 32, false);
        fixture.norm(prefix + ".post_attention_layernorm", 32, false);
        for (const std::string projection : {"q_proj", "k_proj", "v_proj", "o_proj"})
            fixture.linear(prefix + ".self_attn." + projection, 32, 32, false);
        fixture.linear(prefix + ".mlp.fc1", 64, 32, false);
        fixture.linear(prefix + ".mlp.fc2", 32, 64, false);
    }
    fixture.norm("acoustic_decoder.norm", 32, true);
    fixture.linear("acoustic_decoder.head.linear", 66, 32, true);
    const auto path = fixture.root / "weights.safetensors";
    io::write_safetensors_file(path, fixture.entries);
    const auto source = assets::open_tensor_source(path);
    core::BackendConfig backend; backend.type = core::BackendType::Cuda; backend.device = 0;
    core::ExecutionContext first_execution(backend), second_execution(backend);
    constexpr size_t arena = 8 * 1024 * 1024;
    codecs::FsqAudioCodecDecoderRuntime first(config, source, first_execution, arena, arena,
        assets::TensorStorageType::F32, assets::TensorStorageType::F32);
    codecs::FsqAudioCodecDecoderRuntime second(config, source, second_execution, arena, arena,
        assets::TensorStorageType::F32, assets::TensorStorageType::F32);
    std::vector<int32_t> codes(37), other(37);
    for (size_t i = 0; i < codes.size(); ++i) { codes[i] = int(i * 79 % 4096); other[i] = int(i * 151 % 4096); }
    const auto reference = first.decode_head(codes).values;
    auto repeat = [&](codecs::FsqAudioCodecDecoderRuntime & runtime) {
        for (int i = 0; i < 16; ++i) {
            runtime.decode_head(other);
            const auto values = runtime.decode_head(codes).values;
            if (values.size() != reference.size() ||
                std::memcmp(values.data(), reference.data(), values.size() * sizeof(float)) != 0)
                throw std::runtime_error("FSQ codec changed across graph reuse or independent session");
        }
    };
    auto a = std::async(std::launch::async, [&] { repeat(first); });
    auto b = std::async(std::launch::async, [&] { repeat(second); });
    a.get(); b.get();
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try { run(); std::cout << "fsq_audio_codec_reuse_test: ok\n"; return 0; }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
