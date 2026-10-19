// The board side of the RKNN path: everything in src/framework/rknn/engine.cpp that needs
// librknnrt and an engine compiled for the build host's SoC. The plan reader and the status
// formatter live in rknn_engine_test; this one is registered only in an RKNN-enabled build and
// self-skips (77) when no engine was supplied, so CI runs it as `Not Run` and a board runs it
// for real:
//
//   cmake --build build --target rknn_sidecar_test
//   ctest --test-dir build -R '^rknn_sidecar_test$' --output-on-failure \
//       # with -DAUDIOCPP_RKNN_TEST_ENGINE=/path/to/this-board.engine.rknn at configure
//
// The engine is a configure-time path (the repo ships no compiled engines): point it at any
// engine built for this SoC, e.g. the sidecar artefacts a board deployment keeps beside the
// weights.
#include "engine/framework/rknn/engine.h"

#include "test_assert.h"

#include <filesystem>
#include <string>

int main(int argc, char ** argv) {
    const std::filesystem::path engine_path = argc > 1 ? argv[1] : std::string();
    std::error_code ec;
    if (engine_path.empty() || !std::filesystem::exists(engine_path, ec)) {
        // No engine for this host: the driver path this test covers cannot run here.
        return 77;
    }

    // Residency pricing: the model-info-only init and RKNN_QUERY_MEM_SIZE must say a real
    // number for a real engine -- 0 is the "unknown" answer, and an engine the board's own
    // driver reports as 0 MiB resident would make every ladder accounting meaningless.
    const auto resident = engine::rknn::engine_memory_mib(engine_path.string(), "Sidecar test RKNN");
    engine::test::require(
        resident > 0,
        "the driver prices its own engine, got " + std::to_string(resident) + " MiB");

    // The same init caches the driver's self-report, which every later failure quotes.
    const auto report = engine::rknn::driver_report();
    engine::test::require(
        report.find("api ") != std::string::npos && report.find("driver ") != std::string::npos,
        "the driver report carries both versions: " + report);

    // And the full init path builds a real context for the same image.
    engine::rknn::Engine npu;
    npu.load(engine_path.string(), "Sidecar test RKNN");
    engine::test::require(npu.loaded(), "the driver built a context for its engine");
    engine::test::require(
        npu.input_count() > 0 && npu.output_count() > 0,
        "the context reports its graph's inputs and outputs");
    npu.release();
    return 0;
}
