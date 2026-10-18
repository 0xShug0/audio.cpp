#pragma once
#include <filesystem>
#include <memory>
#include <string>
namespace engine::models::kokoro_tts {
class MultilingualG2P {
public:
    explicit MultilingualG2P(const std::filesystem::path & root);
    ~MultilingualG2P();
    std::string phonemize(const std::string & text, const std::string & language) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
