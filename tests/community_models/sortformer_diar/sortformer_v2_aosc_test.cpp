#include "engine/community_models/sortformer_diar/aosc_state.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace engine::community_models::sortformer_diar;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool near(float a, float b) {
    return std::fabs(a - b) < 1.0e-6f;
}

void test_channel_birth_gate() {
    SortformerV2ChannelBirthGate gate(4);
    std::vector<float> timeline;
    gate.append(
        {0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f,
         0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f, 0.01f},
        timeline);
    require(gate.is_established(0), "speaker 0 was not established");

    std::vector<float> redraw;
    for (int i = 0; i < 20; ++i) redraw.insert(redraw.end(), {0.4f, 0.01f, 0.01f, 0.8f});
    const size_t redraw_offset = timeline.size();
    gate.append(redraw, timeline);
    require(!gate.is_established(3), "transient speaker channel was established");
    for (size_t i = redraw_offset; i < timeline.size(); i += 4) {
        require(near(timeline[i], 0.8f) && near(timeline[i + 3], 0.0f), "transient channel was not relabeled");
    }

    gate.append({0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f}, timeline);
    const size_t handoff_offset = timeline.size() - 8;
    gate.append({0.01f, 0.99f, 0.01f, 0.01f, 0.01f, 0.99f, 0.01f, 0.01f}, timeline);
    require(gate.is_established(1), "speaker 1 handoff was not established");
    for (size_t i = handoff_offset; i < timeline.size(); i += 4) require(near(timeline[i + 1], 0.99f), "handoff was relabeled incorrectly");
}

void test_aosc_fifo_and_compression() {
    SortformerV2DiarGeometry geometry{8, 2, 3, 3, 0, 0};
    SortformerV2DiarScoringConfig scoring;
    scoring.sil_frames_per_spk = 1;
    geometry.validate(4, scoring.sil_frames_per_spk, 64);
    SortformerV2AoscState state(geometry, scoring, 4, 2);

    std::vector<float> embeddings(3 * 2, 1.0f);
    std::vector<float> probabilities(3 * 4, 0.1f);
    probabilities[0] = 0.9f;
    probabilities[5] = 0.9f;
    probabilities[10] = 0.9f;
    auto emitted = state.update(embeddings.data(), 3, probabilities.data(), 0, 0);
    require(emitted.size() == 3 * 4, "first AOSC emission size mismatch");
    require(state.fifo_frames() == 0 && state.spkcache_frames() == 3, "first FIFO transfer mismatch");

    std::vector<float> second_embeddings(3 * 2, 2.0f);
    std::vector<float> second_probabilities((3 + 3) * 4, 0.1f);
    auto second = state.update(second_embeddings.data(), 3, second_probabilities.data(), 0, 0);
    require(second.size() == 3 * 4, "second AOSC emission size mismatch");
    require(state.spkcache_frames() == 6 && state.fifo_frames() == 0, "second FIFO transfer mismatch");

    std::vector<float> third_embeddings(3 * 2, 3.0f);
    std::vector<float> third_probabilities((6 + 3) * 4, 0.1f);
    auto third = state.update(third_embeddings.data(), 3, third_probabilities.data(), 0, 0);
    require(third.size() == 3 * 4, "third AOSC emission size mismatch");
    require(state.spkcache_frames() == 8, "AOSC compression did not enforce cache capacity");
    require(state.spkcache_preds_valid(), "AOSC cache predictions were not seeded");
    require(state.spkcache().size() == 8 * 2 && state.spkcache_preds().size() == 8 * 4, "compressed cache shape mismatch");
}

}  // namespace

int main() {
    try {
        test_channel_birth_gate();
        test_aosc_fifo_and_compression();
        std::cout << "sortformer_v2_aosc_test: PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "sortformer_v2_aosc_test: " << error.what() << '\n';
        return 1;
    }
}
