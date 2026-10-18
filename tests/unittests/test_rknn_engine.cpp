// Shared RKNN plumbing (src/framework/rknn/engine.cpp): the part that is testable without
// a board, i.e. the engine-image reader, the driver-status formatter, the aperture budget
// arithmetic and the artifact-path resolver. The context calls need librknnrt and a
// compiled engine, so they are exercised on the board, not here.
#include "engine/framework/rknn/engine.h"

#include "test_assert.h"

#include <algorithm>
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

// The aperture accounting a ladder of engines spends, on the numbers CosyVoice3's flow estimator
// measured on RK3588 (5 MiB per compiled frame, a 2800 MiB resident budget) and the trace that
// board stand served: rungs 110/200/250/400/500/900, requests at 118/158/202/192/220/228 frames.
// The board measured the per-engine residency the driver reports (786 MiB at S=110, 1129 at 200,
// 1426 at 250, 2615 at 400, 3752 at 500, 1039 at 900) against a 3900 MiB aperture, and the sets that
// initialize together in one process are the ones the accounting admits -- so a budget that evicts a
// short rung it should keep, or admits a set the driver refuses, shows up here.
void test_aperture_budget_ladder() {
    const std::vector<int64_t> rungs = {110, 200, 250, 400, 500, 900};
    const std::vector<int64_t> cost = {786, 1129, 1426, 2615, 3752, 1039};
    engine::rknn::ApertureBudget budget(3900);
    const auto cost_of = [&rungs, &cost](int64_t frames) {
        for (size_t i = 0; i < rungs.size(); ++i) {
            if (rungs[i] == frames) {
                return cost[i];
            }
        }
        return int64_t{0};
    };
    std::vector<bool> loaded(rungs.size(), false);
    int loads = 0;
    for (const int64_t frames : {118, 158, 202, 192, 220, 228}) {
        for (size_t i = 0; i < rungs.size(); ++i) {
            if (rungs[i] < frames) {
                continue;
            }
            if (!loaded[i]) {
                if (!budget.fits(cost_of(rungs[i]))) {
                    std::fill(loaded.begin(), loaded.end(), false);
                    budget.clear();
                }
                loaded[i] = true;
                budget.reserve(cost_of(rungs[i]));
                ++loads;
            }
            break;
        }
    }
    require_eq(loads, 2, "the served ladder trace loads two engines");
    require_eq(budget.resident(), int64_t{2555}, "the resident set is the two short rungs");
    require(!budget.fits(3752), "S=500 does not fit beside 200+250");
    budget.clear();
    require(budget.fits(1129 + 2615), "200+400 fits the aperture, as it does on the board");
    require(!budget.fits(1426 + 2615), "250+400 does not, as it does not on the board");
    require(budget.fits(1039), "S=900's own cost fits the aperture; it fails at init on its own ceiling");
}

void test_resolve_engine_path(const std::filesystem::path & dir) {
    const auto engine_path = dir / "flow_estimator_seq200.rknn";
    write_file(engine_path, std::string("RKNN\x01\x00\x00\x00", 8));

    // What an operator writes in `exec_plan` has to work in both spellings: the board-absolute
    // path it has always taken, and the bare file name of an engine installed next to the weights.
    require_eq(engine::rknn::resolve_engine_path("flow_estimator_seq200.rknn", dir),
               engine_path.string(),
               "a name that exists under the model root resolves to it");
    require_eq(engine::rknn::resolve_engine_path(engine_path.string(), dir),
               engine_path.string(),
               "an absolute path is used as given");
    require_eq(engine::rknn::resolve_engine_path("absent.rknn", dir),
               std::string("absent.rknn"),
               "a name with no engine under the root is returned unchanged");
    require_eq(engine::rknn::resolve_engine_path("flow_estimator_seq200.rknn", std::filesystem::path{}),
               std::string("flow_estimator_seq200.rknn"),
               "an empty root resolves nothing");
    require_eq(engine::rknn::resolve_engine_path(std::string(), dir),
               std::string(),
               "an unset artifact stays unset");
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "audio.cpp_rknn_engine_test";
    std::filesystem::create_directories(dir);
    test_read_engine_round_trip(dir);
    test_read_engine_missing(dir);
    test_read_engine_empty(dir);
    test_status();
    test_aperture_budget_ladder();
    test_resolve_engine_path(dir);
    std::filesystem::remove_all(dir);
    return 0;
}
