#pragma once

#include "execution.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace minitts::app {

struct AudioMetricsInfo {
    int sample_rate = 0;
    int channels = 0;
    size_t samples = 0;
};

struct FileOutputPolicy {
    std::optional<std::filesystem::path> audio_out;
    std::optional<std::filesystem::path> output_dir;
    std::optional<std::filesystem::path> segments_base;
    std::optional<std::filesystem::path> turns_base;
    std::optional<std::filesystem::path> words_base;
    std::optional<std::filesystem::path> batch_manifest_out;
};

std::string safe_output_name(const std::string & value);
std::string word_timestamps_to_json(const std::vector<engine::runtime::WordTimestamp> & words);
void emit_task_result(
    const engine::runtime::TaskResult & result,
    const std::optional<std::filesystem::path> & audio_out,
    const std::optional<std::filesystem::path> & audio_out_dir,
    const std::optional<std::filesystem::path> & artifact_out_dir,
    const std::optional<std::filesystem::path> & segments_out,
    const std::optional<std::filesystem::path> & turns_out,
    const std::optional<std::filesystem::path> & words_out);
void emit_task_metrics(
    const engine::runtime::TaskResult & result,
    const std::optional<AudioMetricsInfo> & input_audio,
    double wall_ms,
    const std::string & prefix = "metrics");
void emit_batch_item_result(
    size_t index,
    const AppRequestResult & item,
    const FileOutputPolicy & policy);
void emit_batch_summary(
    const AppBatchResult & batch,
    const FileOutputPolicy & policy);
void emit_batch_result(
    const AppBatchResult & batch,
    const FileOutputPolicy & policy);

}  // namespace minitts::app
