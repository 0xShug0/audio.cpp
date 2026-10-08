#include "engine/community_models/whistle_asr/frontend.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void near(float actual, float expected) {
    if (std::abs(actual - expected) > 0.02f) {
        throw std::runtime_error("Whistle mel features diverge from the reference frontend");
    }
}

std::vector<float> read_f32(const char * path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamoff>(stream.tellg());
    if (!stream || bytes < 0 || bytes % sizeof(float) != 0) {
        throw std::runtime_error(std::string("Invalid float input: ") + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error(std::string("Cannot read float input: ") + path);
    }
    return values;
}

}  // namespace

int main(int argc, char ** argv) {
    using engine::community_models::whistle_asr::WhistleFrontend;
    std::vector<float> filterbank(257 * 80);
    for (size_t mel = 0; mel < 80; ++mel) {
        filterbank[(mel + 1) * 80 + mel] = 1.0f;
    }
    WhistleFrontend frontend(std::move(filterbank));
    std::vector<float> audio(1600);
    const double pi = std::acos(-1.0);
    for (size_t index = 0; index < audio.size(); ++index) {
        audio[index] = static_cast<float>(
            0.2 * std::sin(2.0 * pi * 440.0 * index / 16000.0) +
            0.05 * std::sin(2.0 * pi * 880.0 * index / 16000.0));
    }
    const auto mel = frontend.extract(audio);
    if (mel.frames != 10 || mel.values.size() != 800) {
        throw std::runtime_error("Whistle frontend frame count is wrong");
    }
    near(mel.values[13], -2.845809f);
    near(mel.values[27], -2.845455f);
    near(mel.values[80 + 13], 0.294407f);
    near(mel.values[80 + 27], 0.278846f);
    near(mel.values[5 * 80 + 13], 0.322108f);
    near(mel.values[9 * 80 + 27], 0.281397f);

    bool rejected = false;
    try {
        (void)frontend.extract(std::vector<float>(480001, 0.0f));
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Whistle accepted more than 30 seconds");
    }
    audio[0] = std::numeric_limits<float>::quiet_NaN();
    rejected = false;
    try {
        (void)frontend.extract(audio);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("Whistle accepted a non-finite audio sample");
    }
    if (argc == 3) {
        WhistleFrontend official(read_f32(argv[1]));
        const auto actual = official.extract(read_f32(argv[2]));
        if (actual.frames != 1407) {
            throw std::runtime_error("Whistle official sample frame count is wrong");
        }
        near(actual.values[5], -1.370733f);
        near(actual.values[20], -1.617518f);
        near(actual.values[80 + 5], -1.242171f);
        near(actual.values[50 * 80 + 10], -0.468752f);
        near(actual.values[500 * 80 + 20], -0.946105f);
        near(actual.values[1406 * 80 + 20], -1.161947f);
    } else if (argc != 1) {
        throw std::invalid_argument("Pass no arguments or <mel-filterbank.f32> <mono16k-audio.f32>");
    }
    std::cout << "Whistle frontend matches independent reference features\n";
}
