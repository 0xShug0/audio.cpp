#include "server_state_fixture.h"
#include "engine/framework/runtime/errors.h"

#include <iostream>
#include <exception>
#include <stdexcept>


namespace {

void test_capacity_error() {
    // Test that CapacityError gets converted to a HTTP 400
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw rt::CapacityError("not enough capacity");
        }
    );
    f.add("test-capacity");
    const auto response = f.post("/v1/tasks/run",
        R"({"model":"test-capacity", "text":"hello"})");
    require(response.status == 400, "Expected HTTP 400, got " + std::to_string(response.status) + ": " + response.body);
    require(response.body.find("not enough capacity") != std::string::npos, "HTTP response did not contain the expected error: " + response.body);
    require(response.body.find("invalid_request_error") != std::string::npos, "HTTP response did not identify invalid_request_error: " + response.body);
}

void test_request_validation_error() {
    // Test that RequestValidationError gets converted to a HTTP 400
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw rt::RequestValidationError("steps must be positive");
        }
    );
    f.add("test-validation");
    const auto response = f.post("/v1/tasks/run",
        R"({"model":"test-validation", "text":"hello"})");
    require(response.status == 400, "Expected HTTP 400, got " + std::to_string(response.status) + ": " + response.body);
    require(response.body.find("steps must be positive") != std::string::npos, "HTTP response did not contain the expected error: " + response.body);
    require(response.body.find("invalid_request_error") != std::string::npos, "HTTP response did not identify invalid_request_error: " + response.body);
}

void test_insufficient_memory_error() {
    // Test that InsufficientMemoryError gets converted to a HTTP 503
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw srv::InsufficientMemoryError("VRAM oom");
        }
    );
    f.add("test-memory");
    const auto response = f.post("/v1/tasks/run",
        R"({"model":"test-memory", "text":"hello"})");
    require(response.status == 503, "Expected HTTP 503, got " + std::to_string(response.status) + ": " + response.body);
    require(response.body.find("VRAM oom") != std::string::npos, "HTTP response did not contain the expected error: " + response.body);
    require(response.body.find("insufficient_memory") != std::string::npos, "HTTP response did not identify insufficient_memory: " + response.body);
}

void test_server_busy_error() {
    // Test that ServerBusyError gets converted to a HTTP 503
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw srv::ServerBusyError("busy");
        }
    );
    f.add("test-busy");
    const auto response = f.post("/v1/tasks/run",
        R"({"model":"test-busy", "text":"hello"})");
    require(response.status == 503, "Expected HTTP 503, got " + std::to_string(response.status) + ": " + response.body);
    require(response.body.find("busy") != std::string::npos, "HTTP response did not contain the expected error: " + response.body);
    require(response.body.find("server_busy") != std::string::npos, "HTTP response did identify server_busy: " + response.body);
}

// Need a distinct exception type to ServerState::handle_request's
// std::exception catch block - do not want to reuse any existing exception types
// or any future exception types (so if more narrow catch's get added, this test
// will still test the std::exception catch block), but we also need to attach
// information to ensure it's actually the exception we threw.
class TestException final : public std::exception {
public:
    explicit TestException(const char * message) : message_(message) { }
    const char * what() const noexcept { return message_.c_str(); }

private:
    std::string message_;
};

void test_server_exception_cors() {
    // Test that other exception types convert to an HTTP 500 when from an allowed origin
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw TestException("internal error");
        }
    );
    f.add("test-cors1");
    const auto response = f.post("/v1/tasks/run",
        R"({"model":"test-cors1", "text":"hello"})", "http://allowed-origin");
    require(response.status == 500, "Expected HTTP 500, got " + std::to_string(response.status) + ": " + response.body);
    require(response.body.find("internal error") != std::string::npos, "HTTP response did not contain the expected error: " + response.body);
    require(response.body.find("server_error") != std::string::npos, "HTTP response did not identify server_error: " + response.body);
    require(response.headers.at("Access-Control-Allow-Origin") == "http://allowed-origin", "Expected Access-Control-Allow-Origin header");
}

void test_server_exception_no_cors() {
    // Test that the server throws the exception when not from an allowed origin
    Fixture f(
        [](const rt::TaskRequest &) -> rt::TaskResult {
            throw TestException("rethrow");
        },
        ""
    );
    f.add("test-cors2");
    try {
        const auto response = f.post("/v1/tasks/run",
            R"({"model":"test-cors2", "text":"hello"})");
    } catch (const TestException & ex) {
        require(std::string(ex.what()) == "rethrow", "caught exception was not the exception thrown");
        return;
    }
    // The exception should have been caught, if it didn't, it got swallowed somewhere internally.
    throw std::runtime_error("CORS exception did not result in an uncaught exception");
}

} // namespace

int main() {
    try {
        engine::io::json::enable_serialized_json_parsing();
        test_capacity_error();
        test_insufficient_memory_error();
        test_request_validation_error();
        test_server_busy_error();
        test_server_exception_cors();
        test_server_exception_no_cors();

        std::cout << "PASS ServerState request exception handling\n";
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "FAIL: " << ex.what() << '\n';
        return 1;
    }
}
