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
/// the process code page, which is UTF-8 only when the executable asks for it.
/// A DLL runs under its host's code page, so "Thiền Tâm Đức.wav" would name a
/// file that is not there. Bytes that are not valid UTF-8 are decoded with the
/// process code page instead, as before. Everywhere else this is the identity.
std::filesystem::path path_from_utf8(std::string_view value);

/// A path as UTF-8, for ggml (which opens files by UTF-8 name) and for
/// messages. path::string() uses the process code page on Windows, so it
/// mangles or throws on a name that page cannot hold.
std::string path_to_utf8(const std::filesystem::path & path);

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
