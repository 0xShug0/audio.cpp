// Shared RKNN plumbing (src/framework/rknn/engine.cpp): the part that is testable without
// a board, i.e. the engine-image reader and the driver-status formatter. The context calls
// need librknnrt and a compiled engine, so they are exercised on the board, not here.
#include "engine/framework/rknn/engine.h"

#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

void write_file(const std::filesystem::path & path, const std::string & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
}

void test_read_engine_round_trip(const std::filesystem::path & dir) {
    const auto path = dir / "engine.rknn";
    write_file(path, std::string("RKNN\x01\x00\x00\x00", 8));
    const auto data = engine::rknn::read_engine(path.string(), "Test stage RKNN");
    require_eq(data.size(), size_t{8}, "round-trip size");
    require_eq(std::string(data.begin(), data.end()), std::string("RKNN\x01\x00\x00\x00", 8),
               "round-trip bytes");
}

void test_read_engine_missing(const std::filesystem::path & dir) {
    try {
        (void)engine::rknn::read_engine((dir / "absent.rknn").string(), "Test stage RKNN");
        require(false, "a missing engine must throw");
    } catch (const std::runtime_error & e) {
        const std::string message = e.what();
        require(message.find("Test stage RKNN: cannot open model ") == 0,
                "the message names the stage: " + message);
        require(message.find("absent.rknn") != std::string::npos,
                "the message names the path: " + message);
    }
}

void test_read_engine_empty(const std::filesystem::path & dir) {
    const auto path = dir / "empty.rknn";
    write_file(path, "");
    try {
        (void)engine::rknn::read_engine(path.string(), "Test stage RKNN");
        require(false, "an empty engine must throw");
    } catch (const std::runtime_error & e) {
        require(std::string(e.what()).find("is empty") != std::string::npos,
                std::string("the message says why: ") + e.what());
    }
}

void test_status() {
    require_eq(
        engine::rknn::status("Piper TTS decoder RKNN", "rknn_run", -5),
        std::string("Piper TTS decoder RKNN: rknn_run failed with code -5"),
        "status format");
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "audio.cpp_rknn_engine_test";
    std::filesystem::create_directories(dir);
    test_read_engine_round_trip(dir);
    test_read_engine_missing(dir);
    test_read_engine_empty(dir);
    test_status();
    std::filesystem::remove_all(dir);
    return 0;
}
