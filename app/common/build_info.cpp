#include "build_info.h"

#include "build_info_config.h"

#include <ostream>
#include <string>

namespace minitts::app {
namespace {

std::string build_type() {
    std::string value = AUDIOCPP_BUILD_TYPE_STRING;
    if (value.empty()) {
        value = "unknown";
    }
    return value;
}

std::string compiler_name() {
    const std::string id = AUDIOCPP_COMPILER_ID_STRING;
    const std::string version = AUDIOCPP_COMPILER_VERSION_STRING;
    if (id == "GNU") {
        return "gcc " + version;
    }
    if (id == "Clang") {
        return "clang " + version;
    }
    if (id == "AppleClang") {
        return "apple-clang " + version;
    }
    if (id == "MSVC") {
        return "msvc " + version;
    }
    return id.empty() ? "unknown" : id + " " + version;
}

}  // namespace

void print_build_info(std::ostream & out) {
    out << "audio.cpp " << AUDIOCPP_VERSION_STRING << "\n"
        << "git: " << AUDIOCPP_GIT_SHA_STRING << " " << AUDIOCPP_GIT_DATE_STRING << "\n"
        << "build: " << build_type() << ", " << compiler_name() << ", "
        << AUDIOCPP_SYSTEM_NAME_STRING << " " << AUDIOCPP_SYSTEM_PROCESSOR_STRING << "\n"
        << "backends: " << AUDIOCPP_BUILD_BACKENDS_STRING << "\n";
}

void print_build_info_summary(std::ostream & out) {
    out << "audio.cpp " << AUDIOCPP_VERSION_STRING
        << " (git " << AUDIOCPP_GIT_SHA_STRING << ", " << build_type() << ")"
        << " [backends: " << AUDIOCPP_BUILD_BACKENDS_STRING << "]\n";
}

}  // namespace minitts::app
