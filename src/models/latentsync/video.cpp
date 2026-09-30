#include "engine/models/latentsync/video.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <cerrno>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace engine::models::latentsync {
namespace {

class VideoPipe {
public:
    VideoPipe(const std::filesystem::path & path, double fps) {
#ifdef _WIN32
        SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE read = nullptr, write = nullptr;
        if (!CreatePipe(&read, &write, &attributes, 0)) throw std::runtime_error("Cannot create LatentSync video pipe");
        SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0);
        // Windows argv quoting: double backslashes before quotes and before the closing quote.
        std::wstring quoted = L"\"";
        size_t slashes = 0;
        for (wchar_t c : path.wstring()) {
            if (c == L'\\') { ++slashes; continue; }
            quoted.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
            quoted.push_back(c);
            slashes = 0;
        }
        quoted.append(slashes * 2, L'\\');
        quoted += L'"';
        std::wstring command = L"ffmpeg -nostdin -loglevel error -i " + quoted + L" -vf fps=" + std::to_wstring(fps) +
            L" -threads 1 -f image2pipe -vcodec ppm -";
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdOutput = write;
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        PROCESS_INFORMATION process{};
        const BOOL started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
        CloseHandle(write);
        if (!started) { CloseHandle(read); throw std::runtime_error("Cannot launch FFmpeg; install it and add it to PATH for LatentSync video input"); }
        process_ = process.hProcess;
        CloseHandle(process.hThread);
        const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(read), _O_BINARY | _O_RDONLY);
        stream_ = fd < 0 ? nullptr : _fdopen(fd, "rb");
        if (!stream_) {
            if (fd >= 0) _close(fd); else CloseHandle(read);
            WaitForSingleObject(process_, INFINITE); CloseHandle(process_); process_ = nullptr;
            throw std::runtime_error("Cannot open LatentSync video pipe");
        }
#else
        int pipe_fds[2];
        if (pipe(pipe_fds) != 0) throw std::runtime_error("Cannot create LatentSync video pipe");
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, pipe_fds[0]);
        posix_spawn_file_actions_addclose(&actions, pipe_fds[1]);
        std::string filename = path.string(), filter = "fps=" + std::to_string(fps);
        std::array<std::string, 14> arguments{"ffmpeg", "-nostdin", "-loglevel", "error", "-i", filename,
            "-vf", filter, "-threads", "1", "-f", "image2pipe", "-vcodec", "ppm"};
        std::vector<char *> argv;
        for (auto & argument : arguments) argv.push_back(argument.data());
        char output[] = "-";
        argv.push_back(output); argv.push_back(nullptr);
        const int status = posix_spawnp(&process_, "ffmpeg", &actions, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        close(pipe_fds[1]);
        if (status != 0) { close(pipe_fds[0]); process_ = -1; throw std::runtime_error("Cannot launch FFmpeg; install it and add it to PATH for LatentSync video input"); }
        stream_ = fdopen(pipe_fds[0], "rb");
        if (!stream_) { close(pipe_fds[0]); finish(); throw std::runtime_error("Cannot open LatentSync video pipe"); }
#endif
    }
    ~VideoPipe() { finish(); }
    FILE * stream() const { return stream_; }
    int finish() {
        if (stream_) { fclose(stream_); stream_ = nullptr; }
#ifdef _WIN32
        if (!process_) return 0;
        WaitForSingleObject(process_, INFINITE);
        DWORD code = 1; GetExitCodeProcess(process_, &code);
        CloseHandle(process_); process_ = nullptr;
        return static_cast<int>(code);
#else
        if (process_ < 0) return 0;
        int status = 0;
        pid_t result;
        do { result = waitpid(process_, &status, 0); } while (result < 0 && errno == EINTR);
        process_ = -1;
        return result >= 0 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    }
private:
    FILE * stream_ = nullptr;
#ifdef _WIN32
    HANDLE process_ = nullptr;
#else
    pid_t process_ = -1;
#endif
};

}  // namespace

LatentSyncVideo load_latentsync_video(const std::filesystem::path & path, double fps) {
    if (!std::filesystem::is_regular_file(path) || !std::isfinite(fps) || fps <= 0) throw std::runtime_error("Invalid LatentSync video path or FPS");
    VideoPipe pipe(path, fps);
    LatentSyncVideo video;
    while (true) {
        const int first = fgetc(pipe.stream());
        if (first == EOF) break;
        int width = 0, height = 0, maximum = 0;
        if (first != 'P' || fgetc(pipe.stream()) != '6' || fscanf(pipe.stream(), "%d %d %d", &width, &height, &maximum) != 3 ||
            width <= 0 || height <= 0 || maximum != 255 || fgetc(pipe.stream()) != '\n') {
            throw std::runtime_error("Invalid RGB frame from FFmpeg");
        }
        if (video.frames.empty()) { video.width = width; video.height = height; }
        else if (video.width != width || video.height != height) throw std::runtime_error("LatentSync video resolution changes between frames");
        std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3);
        if (fread(pixels.data(), 1, pixels.size(), pipe.stream()) != pixels.size()) throw std::runtime_error("Incomplete RGB frame from FFmpeg");
        video.frames.push_back(std::move(pixels));
    }
    if (pipe.finish() != 0) throw std::runtime_error("FFmpeg video decoding failed");
    if (video.frames.empty()) throw std::runtime_error("LatentSync video contains no frames");
    return video;
}

}  // namespace engine::models::latentsync
