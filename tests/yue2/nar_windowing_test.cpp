#include "../../src/models/yue2/nar_windowing.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using engine::models::yue2::detail::blend_nar_chunk;
using engine::models::yue2::detail::nar_chunk_ranges;
using Ranges = std::vector<std::pair<int64_t, int64_t>>;

void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

bool close(float a, float b) {
    return std::fabs(a - b) < 1.0e-6F;
}

void test_unwindowed_chunks_match_the_context_split() {
    // (24576 - 100 - 3) / 2 = 12236
    check(nar_chunk_ranges(30000, 100, 24576, 0, 0) ==
              Ranges{{0, 12236}, {12236, 24472}, {24472, 30000}},
          "unwindowed split changed");
    check(nar_chunk_ranges(100, 100, 24576, 0, 64) == Ranges{{0, 100}}, "overlap applies without a window");
}

void test_windows_overlap_and_cover_every_frame() {
    check(nar_chunk_ranges(1716, 718, 24576, 1024, 64) == Ranges{{0, 1024}, {960, 1716}}, "two-window split changed");
    check(nar_chunk_ranges(1024, 718, 24576, 1024, 64) == Ranges{{0, 1024}}, "exact fit needs one window");
    const auto ranges = nar_chunk_ranges(5000, 718, 24576, 1024, 64);
    check(ranges.front().first == 0 && ranges.back().second == 5000, "windows do not cover the song");
    for (size_t i = 1; i < ranges.size(); ++i) {
        check(ranges[i].first == ranges[i - 1].second - 64, "windows do not overlap by the requested frames");
    }
}

void test_context_still_caps_a_large_window() {
    check(nar_chunk_ranges(3000, 2000, 4000, 4096, 0) == Ranges{{0, 998}, {998, 1996}, {1996, 2994}, {2994, 3000}},
          "context no longer caps the window");
}

void test_invalid_shapes_throw() {
    for (const auto & args : std::vector<std::array<int64_t, 5>>{
             {0, 100, 24576, 1024, 64}, {100, 24576, 24576, 1024, 64}, {100, 100, 24576, 64, 64}, {100, 100, 24576, 1024, -1}}) {
        bool threw = false;
        try {
            (void) nar_chunk_ranges(args[0], args[1], args[2], args[3], args[4]);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "invalid NAR window was accepted");
    }
}

void test_blend_crossfades_only_the_shared_frames() {
    const int64_t dim = 2;
    std::vector<float> out(12, 0.0F);
    blend_nar_chunk(out, std::vector<float>(8, 1.0F), 0, 4, 0, dim);
    for (int i = 0; i < 8; ++i) check(close(out[i], 1.0F), "first chunk must be written unchanged");
    for (int i = 8; i < 12; ++i) check(close(out[i], 0.0F), "first chunk wrote past its end");

    blend_nar_chunk(out, std::vector<float>(8, 3.0F), 2, 6, 4, dim);
    const float expected[] = {1.0F, 1.0F, 1.0F, 1.0F, 5.0F / 3.0F, 5.0F / 3.0F, 7.0F / 3.0F, 7.0F / 3.0F, 3.0F, 3.0F, 3.0F, 3.0F};
    for (int i = 0; i < 12; ++i) check(close(out[i], expected[i]), "overlap is not a linear crossfade");
}
}  // namespace

int main() {
    try {
        test_unwindowed_chunks_match_the_context_split();
        test_windows_overlap_and_cover_every_frame();
        test_context_still_caps_a_large_window();
        test_invalid_shapes_throw();
        test_blend_crossfades_only_the_shared_frames();
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
