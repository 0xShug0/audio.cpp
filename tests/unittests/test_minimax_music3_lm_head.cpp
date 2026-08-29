#include "engine/community_models/minimax_music3/global_lm.h"
#include "test_assert.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using engine::models::minimax_music3::MiniMaxMusic3LmHeadLayout;
using engine::models::minimax_music3::classify_minimax_music3_lm_head_shape;
using engine::models::minimax_music3::minimax_music3_lm_head_output_size;

void test_full_vocab_layout() {
    const auto layout = classify_minimax_music3_lm_head_shape({200000, 4096}, 200000, 4096);
    engine::test::require(layout == MiniMaxMusic3LmHeadLayout::FullVocab, "full-vocab layout");
    engine::test::require_eq(minimax_music3_lm_head_output_size(layout, 200000), int64_t{200000}, "full rows");
}

void test_semantic_compact_layout() {
    const auto layout = classify_minimax_music3_lm_head_shape({16385, 4096}, 200000, 4096);
    engine::test::require(
        layout == MiniMaxMusic3LmHeadLayout::SemanticCompactV1,
        "semantic compact layout");
    engine::test::require_eq(minimax_music3_lm_head_output_size(layout, 200000), int64_t{16385}, "compact rows");
}

void test_rejects_ambiguous_or_wrong_shapes() {
    for (const auto & shape : std::vector<std::vector<int64_t>>{
             {16389, 4096},
             {16385, 2048},
             {200000, 2048},
             {16385},
         }) {
        bool rejected = false;
        try {
            (void)classify_minimax_music3_lm_head_shape(shape, 200000, 4096);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        engine::test::require(rejected, "invalid MiniMax lm_head shape was accepted");
    }
}

}  // namespace

int main() {
    try {
        test_full_vocab_layout();
        test_semantic_compact_layout();
        test_rejects_ambiguous_or_wrong_shapes();
        std::cout << "minimax_music3_lm_head_test: ok\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "minimax_music3_lm_head_test: " << error.what() << '\n';
        return 1;
    }
}
