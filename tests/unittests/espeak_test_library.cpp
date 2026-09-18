// Deliberately process-global test double: no eSpeak installation required.
#include <string>
#include <thread>
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
static std::string voice, output;
static bool initialized = false;
EXPORT int espeak_Initialize(int, int, const char *, int options) {
    if (initialized || !(options & 0x8000)) return -1;
    initialized = true;
    return 22050;
}
EXPORT int espeak_Terminate() { initialized = false; return 0; }
EXPORT int espeak_SetVoiceByName(const char * name) {
    if (!initialized || std::string(name) == "missing") return 2;
    voice = name;
    std::this_thread::yield();
    return 0;
}
#ifndef ESPEAK_TEST_MISSING_SYMBOL
EXPORT const char * espeak_TextToPhonemes(const void ** cursor, int encoding, int mode) {
    if (!initialized || encoding != 1) return nullptr;
    const std::string text = static_cast<const char *>(*cursor);
    if (text != "stuck") {
        const auto split = text.find('|');
        *cursor = split == std::string::npos ? nullptr : static_cast<const char *>(*cursor) + split + 1;
        output = voice + ":" + std::to_string(mode) + ":" + text.substr(0, split);
    } else output = "stuck";
    std::this_thread::yield();
    return output.c_str();
}
#endif
