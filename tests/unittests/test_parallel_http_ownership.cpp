#include "parallel_http.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket = SOCKET;
constexpr Socket invalid = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
using Socket = int;
constexpr Socket invalid = -1;
#endif
using namespace std::chrono_literals;
namespace srv = minitts::server;
namespace {
std::atomic<bool> stopping{false};
bool stop() { return stopping.load(); }
void require(bool value, const char * message) { if (!value) { throw std::runtime_error(message); } }
void close_socket(Socket socket) {
#ifdef _WIN32
    closesocket(socket);
#else
    close(socket);
#endif
}
struct Client {
    Socket socket = invalid;
    explicit Client(int port) {
        sockaddr_in address{}; address.sin_family = AF_INET; address.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        for (int attempt = 0; attempt < 200; ++attempt) {
            socket = ::socket(AF_INET, SOCK_STREAM, 0);
            require(socket != invalid, "socket failed");
            if (connect(socket, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) { return; }
            close_socket(socket); socket = invalid; std::this_thread::sleep_for(5ms);
        }
        throw std::runtime_error("listener did not start");
    }
    ~Client() { if (socket != invalid) { close_socket(socket); } }
    void send_text(const std::string & text) {
        size_t offset = 0;
        while (offset < text.size()) {
            const auto written = send(socket, text.data() + offset, static_cast<int>(text.size() - offset), 0);
            require(written > 0, "send failed"); offset += written;
        }
    }
};
struct Handler final : srv::IHttpHandler {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
    std::atomic<bool> completed{false};
    bool deferred;
    bool live;
    explicit Handler(bool deferred, bool live = false) : deferred(deferred), live(live) {}
    void hold() {
        std::unique_lock<std::mutex> lock(mutex); entered = true; cv.notify_all();
        cv.wait_for(lock, 5s, [&] { return released; }); completed = true;
    }
    srv::HttpResponse handle(const srv::HttpRequest & request) override {
        if (live) {
            {
                std::lock_guard<std::mutex> lock(mutex); entered = true; cv.notify_all();
            }
            try { char byte; request.body_stream->read(&byte, 1); }
            catch (const std::exception &) { completed = true; }
            return srv::json_response("{}");
        }
        if (!deferred) { hold(); return srv::json_response("{}"); }
        auto response = srv::json_response("{}");
        response.stream_body = [this](srv::HttpStreamWriter & writer) { hold(); writer.write("done"); };
        return response;
    }
};
void shutdown_drains_workers(bool deferred) {
    stopping = false;
    Handler handler(deferred);
    const int port = deferred ? 18118 : 18117;
    auto listener = std::async(std::launch::async, [&] {
        srv::serve_parallel_http("127.0.0.1", port, handler, stop, 1024);
    });
    struct Cleanup {
        Handler & handler;
        ~Cleanup() {
            stopping = true;
            std::lock_guard<std::mutex> lock(handler.mutex);
            handler.released = true; handler.cv.notify_all();
        }
    } cleanup{handler};
    Client stalled(port); stalled.send_text("POST / HTTP/1.1\r\nHost:");
    Client active(port); active.send_text("POST / HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n");
    bool entered;
    {
        std::unique_lock<std::mutex> lock(handler.mutex);
        entered = handler.cv.wait_for(lock, 2s, [&] { return handler.entered; });
    }
    stopping = true;
    // A partial header must be interrupted, rather than preventing the join.
    auto disconnected = std::async(std::launch::async, [&] { char byte; return recv(stalled.socket, &byte, 1, 0); });
    const bool socket_released = disconnected.wait_for(1s) == std::future_status::ready;
    const bool runtime_retained = listener.wait_for(0ms) == std::future_status::timeout;
    {
        std::lock_guard<std::mutex> lock(handler.mutex); handler.released = true; handler.cv.notify_all();
    }
    listener.get();
    const auto received = disconnected.get();
    require(entered && socket_released && received <= 0, "shutdown did not interrupt stalled socket I/O");
    require(runtime_retained && handler.completed, "listener returned before handler/deferred callback finished");
}
void shutdown_interrupts_live_ingest() {
    stopping = false; Handler handler(false, true);
    auto listener = std::async(std::launch::async, [&] {
        srv::serve_parallel_http("127.0.0.1", 18119, handler, stop, 1024);
    });
    Client client(18119);
    client.send_text("POST /v1/audio/transcriptions/live HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\n\r\n");
    bool entered;
    {
        std::unique_lock<std::mutex> lock(handler.mutex);
        entered = handler.cv.wait_for(lock, 2s, [&] { return handler.entered; });
    }
    stopping = true;
    const bool drained = listener.wait_for(2s) == std::future_status::ready;
    listener.get();
    require(entered && drained && handler.completed, "shutdown retained a live body until its normal idle deadline");
}
void shutdown_interrupts_stalled_response() {
    struct Sender final : srv::IHttpHandler {
        std::promise<void> entered;
        std::atomic<bool> interrupted{false};
        srv::HttpResponse handle(const srv::HttpRequest &) override {
            auto response = srv::json_response("{}");
            response.stream_body = [this](srv::HttpStreamWriter & writer) {
                entered.set_value();
                const std::string block(1024 * 1024, 'x');
                try { for (int i = 0; i < 1000; ++i) { writer.write(block); } }
                catch (const std::exception &) { interrupted = true; throw; }
            };
            return response;
        }
    } handler;
    stopping = false;
    auto entered = handler.entered.get_future();
    auto listener = std::async(std::launch::async, [&] {
        srv::serve_parallel_http("127.0.0.1", 18120, handler, stop, 1024);
    });
    Client client(18120);
    const int small_buffer = 1024;
#ifdef _WIN32
    setsockopt(client.socket, SOL_SOCKET, SO_RCVBUF, reinterpret_cast<const char *>(&small_buffer), sizeof(small_buffer));
#else
    setsockopt(client.socket, SOL_SOCKET, SO_RCVBUF, &small_buffer, sizeof(small_buffer));
#endif
    client.send_text("POST / HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n");
    const bool started = entered.wait_for(2s) == std::future_status::ready;
    stopping = true;
    const bool drained = listener.wait_for(2s) == std::future_status::ready;
    listener.get();
    require(started && drained && handler.interrupted, "shutdown retained a callback blocked by a non-reading client");
}
} // namespace
int main() {
#ifdef _WIN32
    WSADATA data; WSAStartup(MAKEWORD(2, 2), &data);
#endif
#ifdef SIGPIPE
    std::signal(SIGPIPE, SIG_IGN);
#endif
    try {
        shutdown_drains_workers(false); shutdown_drains_workers(true);
        shutdown_interrupts_live_ingest();
        shutdown_interrupts_stalled_response();
        std::cout << "PASS parallel HTTP owns active/deferred workers and interrupts stalled clients\n";
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
#ifdef _WIN32
    WSACleanup();
#endif
}
