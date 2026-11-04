#include "engine/community_models/nemotron_transcribe/encoder.h"
#include "engine/community_models/nemotron_transcribe/frontend.h"
#include "engine/community_models/nemotron_transcribe/assets.h"
#include "engine/community_models/nemotron_transcribe/session.h"
#include "engine/community_models/nemotron_transcribe/repetition_guard.h"
#include "engine/framework/runtime/errors.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace model = engine::models::nemotron_transcribe;

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

template <typename F>
void invalid_audio(F action) {
    try { action(); }
    catch (const engine::runtime::InvalidRequestError &) { return; }
    throw std::runtime_error("malformed input must produce a client error");
}

void repetition_guard_boundaries() {
    // Thresholds and the one-token commit convention match NVIDIA's LoopGuard.
    for (int period = 1; period <= 32; ++period) {
        model::RepetitionGuard guard;
        const int need = std::max(period >= 4 && period <= 12 ? 15 : 32, period * 4);
        const int check = 15 + ((need - 15 + 3) / 4) * 4;
        for (int i = 0; i <= check; ++i)
            require(guard.push(100 + i % period) == (i == check), "periodic loop boundary");
    }
    model::RepetitionGuard phrase;
    for (int i = 0; i < 18; ++i) require(!phrase.push(100 + i % 6), "three phrase repetitions are retained");
    model::RepetitionGuard replay;
    for (int i = 0; i <= 546; ++i)
        require(replay.push(1000 + i % 300) == (i == 546), "long replay boundary matches upstream");
    require(replay.push(7777), "guard stays fired");
    replay.reset();
    for (int i = 0; i < 1000; ++i) require(!replay.push(i), "new output has no history from previous generation");
    // A long phrase repeated only twice must not be mistaken for sustained replay.
    model::RepetitionGuard partial;
    for (int i = 0; i < 400; ++i) require(!partial.push(1000 + i % 300), "partial replay is retained");
}

int main() {
    try {
        repetition_guard_boundaries();
        for (const auto & entry : std::vector<std::pair<std::string, std::string>>{
            {"zh", "Chinese"}, {"zh-CN", "Chinese"}, {"ZH_cn", "Chinese"}, {" zh ", "Chinese"},
            {"en-US", "English"}, {"pt_BR", "Portuguese"}, {"ja-JP", "Japanese"},
            {"sv-SE", "Swedish"}, {"fil-PH", "Filipino"}, {"tl", "Filipino"},
            {"nb-NO", "Norwegian"}, {"no", "Norwegian"}, {"fa", "Persian"}}) {
            require(model::language_name(entry.first) == entry.second, "language tags must resolve to source-language names");
        }
        for (const auto & value : {"", "auto", "Chinese", "Mandarin", "und", "zz-ZZ", "zh-Hans-CN", "-CN", "  Custom name  "})
            require(model::language_name(value) == value, "custom or unknown language names must remain unchanged");
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

        engine::runtime::AudioBuffer mono{16000, 1, std::vector<float>(160, 0.1F)};
        require(!model::normalize_input_audio(mono), "matched input must not allocate or resample");
        engine::runtime::AudioBuffer stereo{48000, 2, std::vector<float>(960)};
        for (size_t i = 0; i < 480; ++i) {
            stereo.samples[2 * i] = 0.1F;
            stereo.samples[2 * i + 1] = 0.2F;
        }
        require(std::abs(model::validate_input_audio(stereo) - 0.01) < 1e-12,
            "duration uses per-channel frames in the original rate");
        const auto converted = model::normalize_input_audio(stereo);
        require(converted && converted->channels == 1 && converted->sample_rate == 16000 &&
            converted->samples.size() == 160, "48 kHz stereo must become 16 kHz mono");
        require(std::isfinite(converted->samples[80]) && std::abs(converted->samples[80] - 0.15F) < 0.01F,
            "channel averaging must preserve the signal scale");
        invalid_audio([&] { model::validate_input_audio(stereo, 0.009); });
        invalid_audio([&] { model::validate_input_audio({0, 1, {0.1F}}); });
        invalid_audio([&] { model::validate_input_audio({16000, 0, {0.1F}}); });
        invalid_audio([&] { model::validate_input_audio({16000, 2, {0.1F}}); });
        invalid_audio([&] { model::normalize_input_audio({16000, 1, {std::numeric_limits<float>::infinity()}}); });

        using Storage = engine::assets::TensorStorageType;
        const std::vector<engine::assets::TensorMetadata> experts{
            {"blk.1.ffn_up_exps.weight", "IQ4_NL", {128, 1856, 2688}},
            {"blk.1.ffn_down_exps.weight", "IQ4_NL", {128, 2688, 1856}}};
        const auto native = model::estimate_transcribe_weight_bytes(experts, Storage::Native);
        require(model::estimate_transcribe_weight_bytes(experts, Storage::Native, 8) == native / 16,
            "offload residency must price eight slots rather than 128 experts");
        const auto expanded = model::estimate_transcribe_weight_bytes(experts, Storage::F32);
        require(expanded == size_t(2) * 128 * 1856 * 2688 * 4 && expanded > 6 * native,
            "F32 residency must reflect expanded routed expert banks");
        require(model::estimate_transcribe_weight_bytes({{"odd", "F16", {8, 17}}}, Storage::Q8_0) == 8 * 17 * 4,
            "unaligned Q8 rows must price the framework's F32 fallback");
        require(model::estimate_transcribe_weight_bytes({{"ctc.test", "F16", {8, 32}}}, Storage::F16) == 8 * 32 * 4,
            "CTC parameters stay F32");
        bool rejected = false;
        try { model::transcribe_weight_type("q4_k"); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "unadvertised session storage must be rejected");
        rejected = false;
        try { model::estimate_transcribe_weight_bytes({{"huge", "F32", {
            std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max()}}}, Storage::F32); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "resource estimate overflow must fail before allocation");
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Nemotron Transcribe silence and window policy checks passed\n";
}
