#pragma once
#include <filesystem>
#include <cstdint>
#include <vector>

namespace engine::models::soulx_flashhead {
std::vector<float> load_portrait(const std::filesystem::path & path, int64_t height, int64_t width);
}
