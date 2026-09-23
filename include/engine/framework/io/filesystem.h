#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::io {

/// A path from a UTF-8 string.
///
/// On Windows a narrow string handed to std::filesystem::path is decoded with
/// the active code page, not UTF-8, so "Thiền Tâm Đức.wav" names a file that is
/// not there and the caller is told the file could not be opened. Everywhere
/// else this is the identity.
std::filesystem::path path_from_utf8(std::string_view value);

bool is_existing_directory(const std::filesystem::path & path);
bool is_existing_file(const std::filesystem::path & path);

std::filesystem::path require_directory(const std::filesystem::path & path, std::string_view role);
std::filesystem::path require_file(const std::filesystem::path & path, std::string_view role);

std::optional<std::filesystem::path> find_first_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<std::filesystem::path> collect_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::string read_text_file(const std::filesystem::path & path);

}  // namespace engine::io
