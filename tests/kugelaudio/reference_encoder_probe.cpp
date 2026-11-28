#include "engine/models/kugelaudio/reference_encoder.h"
#include "engine/framework/audio/wav_reader.h"
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 5) {
        std::cerr << "usage: kugelaudio_reference_encoder_probe encoder.safetensors reference.wav cpu|cuda|vulkan output-prefix\n";
        return 2;
    }
    try {
        namespace kugel = engine::models::kugelaudio;
        engine::core::BackendConfig backend;
        const std::string type = argv[3];
        if (type == "cuda") backend.type = engine::core::BackendType::Cuda;
        else if (type == "vulkan") backend.type = engine::core::BackendType::Vulkan;
        else if (type != "cpu") throw std::runtime_error("unsupported probe backend");
        backend.threads = 8;
        engine::core::ExecutionContext execution(backend);
        auto source = engine::assets::open_tensor_source(argv[1]);
        auto wav = engine::audio::read_wav_f32(std::filesystem::path(argv[2]));
        auto input = kugel::prepare_reference_audio({wav.sample_rate, wav.channels, std::move(wav.samples)});
        const auto begin = std::chrono::steady_clock::now();
        kugel::ReferenceEncoder encoder(*source, execution, {});
        auto values = encoder.encode(input);
        auto write = [](const std::filesystem::path & path, const std::vector<float> & data) {
            std::ofstream f(path, std::ios::binary);
            if (!f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float)))
                throw std::runtime_error("cannot write probe output");
        };
        write(std::string(argv[4]) + ".input.f32", input);
        write(std::string(argv[4]) + ".mean.f32", values);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        std::ofstream(std::string(argv[4]) + ".json") << "{\"samples\":" << input.size()
            << ",\"frames\":" << values.size() / 64 << ",\"dim\":64,\"seconds\":" << seconds << "}";
        std::cout << "frames=" << values.size() / 64 << " seconds=" << seconds << '\n';
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
