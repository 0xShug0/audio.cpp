#include "engine/framework/decoders/tdt_decoder_runner.h"
#include "test_assert.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

class ScriptedTdtCore final : public engine::decoders::TdtDecoderCore {
public:
    explicit ScriptedTdtCore(std::vector<engine::decoders::TdtJointStep> steps)
        : steps_(std::move(steps)) {}

    void reset_state() override {
        ++reset_calls;
        predicted_tokens.clear();
        next_step_ = 0;
    }

    void predict_start(int32_t blank_id) override {
        ++start_calls;
        start_token = blank_id;
    }

    void predict_token(int32_t token_id) override {
        predicted_tokens.push_back(token_id);
    }

    engine::decoders::TdtJointStep joint_step_argmax(const float*) override {
        if (next_step_ >= steps_.size()) {
            throw std::runtime_error("scripted decoder exhausted");
        }
        return steps_[next_step_++];
    }

    engine::decoders::TdtPredictorStateSnapshot snapshot_state() const override {
        return {};
    }

    void restore_state(const engine::decoders::TdtPredictorStateSnapshot&) override {}

    int reset_calls = 0;
    int start_calls = 0;
    int32_t start_token = -1;
    std::vector<int32_t> predicted_tokens;

private:
    std::vector<engine::decoders::TdtJointStep> steps_;
    size_t next_step_ = 0;
};

void require_vector_eq(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected,
    const std::string& label) {
    engine::test::require_eq(actual.size(), expected.size(), label + " size");
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(actual[i], expected[i], label + "[" + std::to_string(i) + "]");
    }
}

}  // namespace

int main() {
    try {
        constexpr int32_t blank = 99;
        ScriptedTdtCore core({
            {blank, 0.f, 0},  // Blank + duration 0 advances one frame.
            {blank, 0.f, 2},  // Blank + duration 2 advances from frame 1 to 3.
            {10, 0.f, 0},     // Two labels can be emitted at frame 3.
            {10, 0.f, 0},     // Symbol cap then forces progress to frame 4.
            {12, 0.f, 1},     // Nonblank + duration 1 advances to frame 5.
            {blank, 0.f, 2},  // Advance to end of the seven-frame input.
        });
        const std::vector<float> encoder_frames(7, 0.f);
        const auto result = engine::decoders::run_tdt_decoder(
            engine::decoders::TdtDecoderAlgorithm::GreedyDurationLoop,
            core,
            encoder_frames,
            7,
            1,
            blank,
            {0, 1, 2},
            2);

        require_vector_eq(result.token_ids, {10, 10, 12}, "token ids");
        require_vector_eq(result.token_timestamps, {3, 3, 4}, "emission frames");
        require_vector_eq(result.token_durations, {0, 0, 1}, "token durations");
        require_vector_eq(core.predicted_tokens, {10, 10, 12}, "predictor tokens");
        engine::test::require_eq(core.reset_calls, 1, "reset calls");
        engine::test::require_eq(core.start_calls, 1, "start calls");
        engine::test::require_eq(core.start_token, blank, "start token");

        std::cout << "tdt_decoder_duration_loop_test passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "tdt_decoder_duration_loop_test failed: " << ex.what() << '\n';
        return 1;
    }
}
