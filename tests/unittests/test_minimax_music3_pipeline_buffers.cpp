#include "engine/community_models/minimax_music3/condition_encoder.h"
#include "engine/community_models/minimax_music3/pipeline.h"
#include "test_assert.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using engine::models::minimax_music3::detail::append_cropped_interleaved_audio;
using engine::models::minimax_music3::detail::project_frame_hiddens;

void require_vector_eq(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    const char * label) {
    engine::test::require_eq(actual.size(), expected.size(), std::string(label) + " size");
    for (size_t index = 0; index < actual.size(); ++index) {
        engine::test::require_eq(
            actual[index],
            expected[index],
            std::string(label) + "[" + std::to_string(index) + "]");
    }
}

void test_projects_an_offset_frame_window_without_repacking() {
    const std::vector<float> all_frames{
        0.0F, 0.0F, 0.0F,
        4.0F, 4.0F, 4.0F,
        10.0F, 20.0F, 30.0F,
        14.0F, 24.0F, 34.0F,
        100.0F, 200.0F, 300.0F,
        104.0F, 204.0F, 304.0F,
    };
    const std::vector<float> layer_weights{0.25F, 0.75F};
    const float * window = all_frames.data() + 6;

    const auto projected = project_frame_hiddens(
        window,
        12,
        2,
        2,
        3,
        layer_weights);

    require_vector_eq(projected, {13.0F, 103.0F, 23.0F, 203.0F, 33.0F, 303.0F}, "projected");
}

void test_projection_rejects_a_truncated_window() {
    const std::vector<float> values(11, 0.0F);
    bool rejected = false;
    try {
        (void)project_frame_hiddens(values.data(), values.size(), 2, 2, 3, {0.25F, 0.75F});
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    engine::test::require(rejected, "truncated frame-hidden window was accepted");
}

void test_appends_only_the_uncropped_stereo_frames() {
    engine::runtime::AudioBuffer destination{44100, 2, {-1.0F, -10.0F}};
    const engine::runtime::AudioBuffer chunk{
        44100,
        2,
        {1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F},
    };

    append_cropped_interleaved_audio(destination, chunk, 1, 1);

    require_vector_eq(
        destination.samples,
        {-1.0F, -10.0F, 2.0F, 20.0F, 3.0F, 30.0F},
        "cropped append");
}

void test_append_rejects_mismatched_output_format() {
    engine::runtime::AudioBuffer destination{48000, 2, {}};
    const engine::runtime::AudioBuffer chunk{44100, 2, {1.0F, 2.0F}};
    bool rejected = false;
    try {
        append_cropped_interleaved_audio(destination, chunk, 0, 0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    engine::test::require(rejected, "mismatched chunk format was accepted");
}

}  // namespace

int main() {
    try {
        test_projects_an_offset_frame_window_without_repacking();
        test_projection_rejects_a_truncated_window();
        test_appends_only_the_uncropped_stereo_frames();
        test_append_rejects_mismatched_output_format();
        std::cout << "minimax_music3_pipeline_buffers_test: ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "minimax_music3_pipeline_buffers_test: " << error.what() << '\n';
        return 1;
    }
}
