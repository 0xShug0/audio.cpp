#pragma once

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace minitts::server {

// Owns only the opt-in AF_UNIX listener and its filesystem socket. Existing
// entries are never removed on startup. HTTP connections use the usual transport.
class UnixSocketListener {
public:
    explicit UnixSocketListener(std::string path) : path_(std::move(path)) {
#ifdef _WIN32
        throw std::runtime_error("unix: server listeners are supported on POSIX platforms only");
#else
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        if (path_.empty() || path_.find('\0') != std::string::npos) {
            throw std::runtime_error("unix: server host requires a nonempty socket path without NUL bytes");
        }
        if (path_.size() >= sizeof(address.sun_path)) {
            throw std::runtime_error("Unix socket path exceeds " +
                std::to_string(sizeof(address.sun_path) - 1) + " bytes: " + path_);
        }
        std::memcpy(address.sun_path, path_.c_str(), path_.size() + 1);
        socket_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (socket_ < 0) {
            throw std::runtime_error("could not create Unix listen socket: " + std::string(std::strerror(errno)));
        }
        const auto length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path_.size() + 1);
        if (::bind(socket_, reinterpret_cast<const sockaddr *>(&address), length) != 0) {
            const int error = errno;
            ::close(socket_);
            socket_ = -1;
            throw std::runtime_error("could not bind Unix socket " + path_ + ": " + std::strerror(error) +
                "; existing paths are not removed automatically");
        }
        // Retain filesystem identity so shutdown cannot remove a replacement file.
        if (::lstat(path_.c_str(), &identity_) != 0) {
            const int error = errno;
            ::close(socket_);
            socket_ = -1;
            throw std::runtime_error("could not stat bound Unix socket " + path_ + ": " + std::strerror(error));
        }
        if (::listen(socket_, 16) != 0) {
            const int error = errno;
            cleanup();
            throw std::runtime_error("could not listen on Unix socket " + path_ + ": " + std::strerror(error));
        }
#endif
    }

    UnixSocketListener(const UnixSocketListener &) = delete;
    UnixSocketListener & operator=(const UnixSocketListener &) = delete;
    ~UnixSocketListener() { cleanup(); }

    int get() const noexcept { return socket_; }

private:
    void cleanup() noexcept {
#ifndef _WIN32
        if (socket_ < 0) {
            return;
        }
        ::close(socket_);
        socket_ = -1;
        struct stat current{};
        if (::lstat(path_.c_str(), &current) == 0 &&
            current.st_dev == identity_.st_dev && current.st_ino == identity_.st_ino) {
            ::unlink(path_.c_str());
        }
#endif
    }

    std::string path_;
    int socket_ = -1;
#ifndef _WIN32
    struct stat identity_{};
#endif
};

} // namespace minitts::server
