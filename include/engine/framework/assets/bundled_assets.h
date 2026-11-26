#pragma once

#include <filesystem>

namespace engine::assets {

// Resolves the path of an asset that ships with audio.cpp, such as the default Silero VAD
// (assets/framework/models/silero_vad), without depending on the working directory.
// Returns the first of these that exists:
//
//   1. `path` as given: relative to the working directory, as before;
//   2. `path` relative to the directory of the module that contains audio.cpp (the C API
//      library, or an executable that links audio.cpp statically), then relative to each
//      of up to three parent directories, so build/bin/Release/ finds the source tree's
//      assets/ and an application can ship assets/ beside or above the library.
//
// A path with a root (absolute, or drive- or root-relative) is returned unchanged, and so
// is a path found nowhere: the caller's error then names the path as given.
std::filesystem::path resolve_bundled_asset(const std::filesystem::path & path);

}  // namespace engine::assets
