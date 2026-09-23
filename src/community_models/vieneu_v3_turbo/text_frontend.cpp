#include "engine/community_models/vieneu_v3_turbo/text_frontend.h"

#include "engine/framework/io/dynamic_library.h"

#include <stdexcept>

namespace engine::models::vieneu_v3_turbo {
namespace {

/// The ABI this build was written against; the library reports its own.
constexpr int kExpectedAbiVersion = 1;

}  // namespace

struct TextFrontend::Impl {
    io::DynamicLibraryHandle library = nullptr;
    void * handle = nullptr;
    std::filesystem::path library_path;
    std::filesystem::path dictionary_path;

    int (*abi_version)() = nullptr;
    const char * (*last_error)() = nullptr;
    void * (*open)(const char *) = nullptr;
    void (*close)(void *) = nullptr;
    void (*string_free)(char *) = nullptr;
    char * (*phonemize)(const void *, const char *, int) = nullptr;
    char * (*normalize)(const void *, const char *, int) = nullptr;

    ~Impl() {
        if (handle != nullptr && close != nullptr) {
            close(handle);
        }
        io::close_dynamic_library(library);
    }

    /// The detail sea-g2p left for the last failing call on this thread.
    std::string error() const {
        const char * message = last_error != nullptr ? last_error() : nullptr;
        return message != nullptr ? std::string(message) : std::string("unknown error");
    }

    template <typename Fn>
    void bind(Fn & slot, const char * name) {
        slot = reinterpret_cast<Fn>(io::dynamic_library_symbol(library, name));
        if (slot == nullptr) {
            throw std::runtime_error(
                std::string("sea-g2p library is missing ") + name +
                "; build it with --no-default-features --features capi");
        }
    }

    std::string run(char * (*call)(const void *, const char *, int), const std::string & text, bool flag) const {
        char * result = call(handle, text.c_str(), flag ? 1 : 0);
        if (result == nullptr) {
            throw std::runtime_error("sea-g2p failed: " + error());
        }
        std::string out(result);
        string_free(result);
        return out;
    }
};

TextFrontend::TextFrontend(const std::filesystem::path & library, const std::filesystem::path & dictionary)
    : impl_(std::make_unique<Impl>()) {
    if (library.empty()) {
        impl_->library = io::open_dynamic_library({
#ifdef _WIN32
            "sea_g2p_rs.dll", "sea_g2p.dll",
#elif defined(__APPLE__)
            "libsea_g2p_rs.dylib", "libsea_g2p.dylib",
#else
            "libsea_g2p_rs.so", "libsea_g2p.so",
#endif
        });
    } else {
        impl_->library = io::open_dynamic_library(library.string());
    }
    if (impl_->library == nullptr) {
        throw std::runtime_error(
            "could not load the sea-g2p library" +
            (library.empty() ? std::string(" by name; set vieneu_v3_turbo.g2p_library")
                             : std::string(" at ") + library.string()));
    }
    impl_->library_path = library;
    impl_->bind(impl_->abi_version, "sea_g2p_abi_version");
    impl_->bind(impl_->last_error, "sea_g2p_last_error");
    impl_->bind(impl_->open, "sea_g2p_open");
    impl_->bind(impl_->close, "sea_g2p_close");
    impl_->bind(impl_->string_free, "sea_g2p_string_free");
    impl_->bind(impl_->phonemize, "sea_g2p_phonemize");
    impl_->bind(impl_->normalize, "sea_g2p_normalize");
    // Refuse a library that speaks a different ABI rather than call into it.
    if (const int reported = impl_->abi_version(); reported != kExpectedAbiVersion) {
        throw std::runtime_error(
            "sea-g2p ABI " + std::to_string(reported) + " does not match the expected " +
            std::to_string(kExpectedAbiVersion));
    }
    if (dictionary.empty()) {
        throw std::runtime_error("sea-g2p needs its dictionary; set vieneu_v3_turbo.g2p_dict to sea_g2p.bin");
    }
    impl_->handle = impl_->open(dictionary.string().c_str());
    if (impl_->handle == nullptr) {
        throw std::runtime_error("sea-g2p could not open " + dictionary.string() + ": " + impl_->error());
    }
    impl_->dictionary_path = dictionary;
}

TextFrontend::~TextFrontend() = default;

std::string TextFrontend::phonemize(const std::string & text, bool punctuation_norm) const {
    return impl_->run(impl_->phonemize, text, punctuation_norm);
}

std::string TextFrontend::normalize(const std::string & text, bool punctuation_norm) const {
    return impl_->run(impl_->normalize, text, punctuation_norm);
}

const std::filesystem::path & TextFrontend::library_path() const noexcept {
    return impl_->library_path;
}

const std::filesystem::path & TextFrontend::dictionary_path() const noexcept {
    return impl_->dictionary_path;
}

}  // namespace engine::models::vieneu_v3_turbo
