#include "execution.h"

#include <chrono>
#include <stdexcept>

namespace minitts::app {
namespace {

engine::runtime::AudioBuffer concat_audio_outputs(
    const std::vector<AppRequestResult> & results,
    std::vector<AudioChapter> & chapters) {
    engine::runtime::AudioBuffer merged;
    for (const auto & item : results) {
        if (!item.result.audio_output.has_value()) {
            continue;
        }
        const auto & audio = *item.result.audio_output;
        if (merged.sample_rate == 0) {
            merged.sample_rate = audio.sample_rate;
            merged.channels = audio.channels;
        }
        if (audio.sample_rate != merged.sample_rate || audio.channels != merged.channels) {
            throw std::runtime_error("cannot merge batch audio outputs with different sample rates or channel counts");
        }
        const int64_t start = static_cast<int64_t>(merged.samples.size() / static_cast<size_t>(merged.channels));
        merged.samples.insert(merged.samples.end(), audio.samples.begin(), audio.samples.end());
        const int64_t end = static_cast<int64_t>(merged.samples.size() / static_cast<size_t>(merged.channels));
        chapters.push_back(AudioChapter{item.id, start, end});
    }
    if (merged.sample_rate == 0) {
        throw std::runtime_error("batch audio merge requested but no request produced a primary audio output");
    }
    return merged;
}

}  // namespace

AudioMergeMode parse_audio_merge_mode(const std::string & value) {
    if (value == "none") {
        return AudioMergeMode::None;
    }
    if (value == "concat") {
        return AudioMergeMode::Concat;
    }
    throw std::runtime_error("--batch-merge-audio must be none or concat");
}

AppBatchResult run_offline_batch(
    engine::runtime::IVoiceTaskSession & session,
    engine::runtime::IOfflineVoiceTaskSession & offline,
    const AppBatchRequest & batch,
    AudioMergeMode audio_merge_mode,
    const std::function<void(size_t, const AppRequestResult &)> & on_result) {
    if (batch.requests.empty()) {
        throw std::runtime_error("offline batch requires at least one request");
    }
    using Clock = std::chrono::steady_clock;
    const auto to_ms = [](Clock::duration d) {
        return std::chrono::duration<double, std::milli>(d).count();
    };
    const auto session_start = Clock::now();

    const auto prepare_start = Clock::now();
    session.prepare(engine::runtime::build_preparation_request(batch.requests.front().request));
    AppBatchResult out;
    out.prepare_ms = to_ms(Clock::now() - prepare_start);
    out.results.reserve(batch.requests.size());
    for (const auto & item : batch.requests) {
        const auto run_start = Clock::now();
        auto result = offline.run(item.request);
        const double wall_ms = to_ms(Clock::now() - run_start);
        out.results.push_back(AppRequestResult{
            item.id,
            std::move(result),
            wall_ms,
        });
        if (on_result) {
            on_result(out.results.size() - 1, out.results.back());
        }
    }
    out.session_wall_ms = to_ms(Clock::now() - session_start);
    if (audio_merge_mode == AudioMergeMode::Concat) {
        out.merged_audio = concat_audio_outputs(out.results, out.chapters);
    }
    return out;
}

}  // namespace minitts::app
