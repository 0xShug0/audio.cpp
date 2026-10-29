#include "engine/community_models/nemotron_transcribe/encoder.h"
#include "engine/community_models/nemotron_transcribe/frontend.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace model = engine::models::nemotron_transcribe;

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        require(model::is_near_silent({}), "empty native sample sequence");
        require(model::is_near_silent(std::vector<float>(16000, 0.0F)), "digital silence");
        require(model::is_near_silent(std::vector<float>(16000, 1e-6F)), "low energy noise floor");
        require(!model::is_near_silent(std::vector<float>(16000, 1e-4F)), "RMS protects quiet delivery");
        auto click = std::vector<float>(16000, 0.0F);
        click[8000] = 0.001F;
        require(!model::is_near_silent(click), "peak at threshold must pass through");
        click[8000] = -0.001F;
        require(!model::is_near_silent(click), "negative peak at threshold must pass through");
        click[8000] = std::numeric_limits<float>::quiet_NaN();
        require(!model::is_near_silent(click), "non-finite audio must reach frontend validation");
        click[8000] = std::numeric_limits<float>::infinity();
        require(!model::is_near_silent(click), "infinite audio must reach frontend validation");
        model::EncoderConfig config;
        const auto windows = model::plan_encoder_windows(360000, config);
        int64_t frames = 0;
        for (const auto & window : windows) {
            require(window.core_frames > 0 && window.core_frames <= 500, "bounded native encoder windows");
            frames += window.core_frames;
        }
        require(windows.size() == 90 && frames == 45000, "one-hour geometry must cover every frame exactly once");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Nemotron Transcribe silence and window policy checks passed\n";
}
