#include "engine/framework/audio/espeak_phonemizer.h"
#include "engine/framework/audio/espeak_data.h"
#include "engine/framework/io/dynamic_library.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#ifdef AUDIOCPP_STATIC_ESPEAK
#include <espeak-ng/speak_lib.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#elif !defined(_WIN32)
#include <unistd.h>
#endif
#endif

namespace engine::audio {
namespace {
#ifdef AUDIOCPP_STATIC_ESPEAK
std::filesystem::path executable_directory() {
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length && length < buffer.size()) return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> buffer(size);
    if (_NSGetExecutablePath(buffer.data(), &size) == 0)
        return std::filesystem::weakly_canonical(buffer.data()).parent_path();
#else
    std::vector<char> buffer(4096);
    for (;;) {
        const auto size = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (size < 0) break;
        if (static_cast<size_t>(size) < buffer.size())
            return std::filesystem::path(std::string(buffer.data(), size)).parent_path();
        buffer.resize(buffer.size() * 2);
    }
#endif
    throw std::runtime_error("Cannot locate executable for eSpeak-ng data; specify the model's espeak_data_path option");
}
#endif
struct Runtime {
    io::DynamicLibraryHandle library = nullptr;
    int (*initialize)(int, int, const char *, int) = nullptr;
    int (*voice)(const char *) = nullptr;
    const char * (*phonemes)(const void **, int, int) = nullptr;
    int (*terminate)() = nullptr;
    bool initialized = false;
    std::filesystem::path library_path, data_path;

    ~Runtime() {
        if (initialized) terminate();
        io::close_dynamic_library(library);
    }
    void open(const std::filesystem::path & path, const std::filesystem::path & data) {
        library_path = path;
        data_path = data;
#ifdef AUDIOCPP_STATIC_ESPEAK
        if (path.empty()) {
            initialize = [](int output, int size, const char * root, int options) {
                return espeak_Initialize(static_cast<espeak_AUDIO_OUTPUT>(output), size, root, options);
            };
            voice = [](const char * name) { return static_cast<int>(espeak_SetVoiceByName(name)); };
            phonemes = espeak_TextToPhonemes;
            terminate = [] { return static_cast<int>(espeak_Terminate()); };
        } else {
#endif
        if (!path.empty()) {
#ifdef _WIN32
            library = LoadLibraryW(path.c_str());
#else
            library = io::open_dynamic_library(path.string());
#endif
        } else {
            library = io::open_dynamic_library({
#ifdef _WIN32
                "espeak-ng.dll", "libespeak-ng.dll",
#elif defined(__APPLE__)
                "libespeak-ng.dylib", "libespeak-ng.1.dylib",
#else
                "libespeak-ng.so.1", "libespeak-ng.so",
#endif
            });
        }
        if (!library) throw std::runtime_error("Could not load eSpeak-ng; install the shared library or provide its path");
        initialize = reinterpret_cast<decltype(initialize)>(io::dynamic_library_symbol(library, "espeak_Initialize"));
        voice = reinterpret_cast<decltype(voice)>(io::dynamic_library_symbol(library, "espeak_SetVoiceByName"));
        phonemes = reinterpret_cast<decltype(phonemes)>(io::dynamic_library_symbol(library, "espeak_TextToPhonemes"));
        terminate = reinterpret_cast<decltype(terminate)>(io::dynamic_library_symbol(library, "espeak_Terminate"));
        if (!initialize || !voice || !phonemes || !terminate)
            throw std::runtime_error("eSpeak-ng is missing required symbols");
#ifdef AUDIOCPP_STATIC_ESPEAK
        }
#endif
        // eSpeak appends espeak-ng-data to this path. DONT_EXIT (0x8000)
        // prevents a missing data installation from exiting the host process.
        const auto parent = data.empty() ? std::string() : data.parent_path().u8string();
        initialized = true; // Also clean up a partially initialized library on failure.
        if (initialize(2, 0, parent.empty() ? nullptr : parent.c_str(), 0x8000) <= 0)
            throw std::runtime_error("eSpeak-ng failed to initialize; check its data directory");
    }
};

// eSpeak's translator and output buffer are process-global, not per frontend.
// All configuration changes and copying of output must share the same lock.
struct Service {
    std::mutex mutex;
    std::unique_ptr<Runtime> runtime;
};
Service & service() { static Service instance; return instance; }
}

EspeakPhonemizer::EspeakPhonemizer(std::filesystem::path library,
                                 std::filesystem::path data,
                                 std::vector<std::string> voices)
    : library_(library.empty() ? library : std::filesystem::absolute(library).lexically_normal()),
      data_(data.empty() ? data : std::filesystem::absolute(data).lexically_normal()),
      voices_(std::move(voices)) {
#ifdef AUDIOCPP_STATIC_ESPEAK
    if (library_.empty() && data_.empty()) {
        const auto root = executable_directory();
        data_ = std::filesystem::is_regular_file(root / "espeak-ng-data.bin")
            ? root / "espeak-ng-data.bin"
            : std::filesystem::is_regular_file(root / "espeak-ng-data.gguf")
                ? root / "espeak-ng-data.gguf" : root / "espeak-ng-data";
    }
#endif
    if (data_.extension() == ".bin" || data_.extension() == ".gguf")
        data_ = materialize_espeak_data(data_);
    if (!data_.empty() && data_.filename().empty()) data_ = data_.parent_path();
    if (!library_.empty() && !std::filesystem::is_regular_file(library_))
        throw std::runtime_error("eSpeak-ng library does not exist: " + library_.string());
    if (!data_.empty() && (data_.filename() != "espeak-ng-data" ||
                          !std::filesystem::is_regular_file(data_ / "phontab")))
        throw std::runtime_error("Expected an espeak-ng-data directory containing phontab: " + data_.string());
    if (voices_.empty()) throw std::invalid_argument("eSpeak-ng requires at least one voice candidate");
    for (const auto & voice : voices_)
        if (voice.empty()) throw std::invalid_argument("eSpeak-ng voice candidates must not be empty");
    phonemize("", 2); // Preserve eager validation without retaining a voice globally.
}

std::string EspeakPhonemizer::phonemize(const std::string & text, int mode,
                                       const std::string & separator) const {
    auto & state = service();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.runtime || state.runtime->library_path != library_ || state.runtime->data_path != data_) {
        state.runtime.reset();
        auto runtime = std::make_unique<Runtime>();
        runtime->open(library_, data_);
        state.runtime = std::move(runtime);
    }
    auto & runtime = *state.runtime;
    bool selected = false;
    for (const auto & voice : voices_) {
        if (runtime.voice(voice.c_str()) == 0) { selected = true; break; }
    }
    if (!selected) throw std::runtime_error("eSpeak-ng has no voice matching '" + voices_.front() + "'");
    std::string out;
    const void * cursor = text.c_str();
    while (cursor && *static_cast<const char *>(cursor)) {
        const void * previous = cursor;
        const char * clause = runtime.phonemes(&cursor, 1, mode);
        if (!clause) break;
        if (!out.empty()) out += separator;
        out += clause;
        if (cursor == previous) throw std::runtime_error("eSpeak-ng did not advance the input cursor");
    }
    return out;
}
} // namespace engine::audio
