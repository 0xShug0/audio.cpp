#include "../../app/workflow/file_sink.h"

#include "engine/framework/io/json.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "failed to open test output " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::string multiline_text() {
    return "He said \"hello\".\nPath: C:\\audio\r\nSpeaker\ttext: \xc3\xa9";
}

void test_word_timestamp_json_escapes_control_characters() {
    engine::runtime::WordTimestamp word;
    word.span.start_sample = 1;
    word.span.end_sample = 2;
    word.word = multiline_text();

    const std::string json = minitts::app::word_timestamps_to_json({word});
    const auto parsed = engine::io::json::parse(json);
    require(parsed.as_array()[0].require("word").as_string() == word.word, "word text must round-trip exactly");

    require(json.find("\\\"hello\\\"") != std::string::npos, "quotes must be escaped");
    require(json.find("C:\\\\audio") != std::string::npos, "backslashes must be escaped");
    require(json.find("\\n") != std::string::npos, "newline must be escaped");
    require(json.find("\\r") != std::string::npos, "carriage return must be escaped");
    require(json.find("\\t") != std::string::npos, "tab must be escaped");
    require(json.find("\xc3\xa9") != std::string::npos, "ordinary UTF-8 must be retained");
    require(json.find("start_sample\":1") != std::string::npos, "word start number schema");
    require(json.find("end_sample\":2") != std::string::npos, "word end number schema");
}

void test_emit_task_result_serializes_all_file_outputs() {
    const auto root = std::filesystem::temp_directory_path() /
        ("audio_cpp_file_sink_json_test_" + std::to_string(std::random_device{}()));
    // Refuse a collision instead of deleting or reusing another test's files.
    require(std::filesystem::create_directory(root), "could not create isolated output directory");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } cleanup{root};
    const auto artifact_dir = root / "artifacts";
    const auto segments_path = root / "segments.json";
    const auto turns_path = root / "turns.json";
    const auto words_path = root / "words.json";

    const std::string value = multiline_text();
    engine::runtime::TaskResult result;
    result.speech_segments.push_back({{10, 20}, 0.75f, value});
    result.speaker_turns.push_back({{30, 40}, value, 0.5f, value});
    result.word_timestamps.push_back({{50, 60}, value, 0.25f});
    engine::runtime::VoiceArtifact artifact;
    artifact.kind = engine::runtime::ArtifactKind::Custom;
    artifact.id = value;
    artifact.payload = {std::byte{0x01}, std::byte{0x02}, std::byte{0x03}};
    artifact.meta.emplace("label\"key", value);
    result.output_artifacts.push_back(std::move(artifact));

    minitts::app::emit_task_result(
        result,
        std::nullopt,
        std::nullopt,
        artifact_dir,
        segments_path,
        turns_path,
        words_path);

    const auto segments = engine::io::json::parse(read_file(segments_path));
    const auto turns = engine::io::json::parse(read_file(turns_path));
    const auto words = engine::io::json::parse(read_file(words_path));
    require(segments.is_array() && segments.as_array().size() == 1, "speech segment output schema");
    require(turns.is_array() && turns.as_array().size() == 1, "speaker turn output schema");
    require(words.is_array() && words.as_array().size() == 1, "word output schema");
    require(segments.as_array()[0].require("start_sample").as_i64() == 10, "segment start number");
    require(turns.as_array()[0].require("end_sample").as_i64() == 40, "turn end number");
    require(words.as_array()[0].require("start_sample").as_i64() == 50, "word start number");

    require(segments.as_array()[0].require("text").as_string() == value, "segment text round-trip");
    require(turns.as_array()[0].require("text").as_string() == value, "speaker text round-trip");
    require(words.as_array()[0].require("word").as_string() == value, "word output round-trip");

    const auto artifact_path = artifact_dir / (minitts::app::safe_output_name(value) + ".json");
    const std::string artifact_json = read_file(artifact_path);
    const auto artifact_value = engine::io::json::parse(artifact_json);
    require(artifact_value.require("bytes").as_i64() == 3, "artifact byte count schema");
    require(artifact_json.find("label\\\"key") != std::string::npos, "artifact metadata key escape");

    require(artifact_value.require("meta").require("label\"key").as_string() == value,
            "artifact metadata text round-trip");

    minitts::app::AppBatchResult batch;
    minitts::app::AppRequestResult item;
    item.id = value;
    batch.results.push_back(std::move(item));
    batch.chapters.push_back({"chapter\"1", 7, 9});
    const auto manifest_path = root / "batch.json";
    minitts::app::FileOutputPolicy policy;
    policy.batch_manifest_out = manifest_path;
    minitts::app::emit_batch_result(batch, policy);
    const std::string manifest_json = read_file(manifest_path);
    const auto manifest = engine::io::json::parse(manifest_json);
    require(manifest.require("requests").as_array().size() == 1, "batch request schema");
    require(manifest.require("chapters").as_array()[0].require("start_sample").as_i64() == 7,
            "batch chapter start number");
    require(manifest.require("requests").as_array()[0].require("id").as_string() == value,
            "batch request text round-trip");
    require(manifest_json.find("chapter\\\"1") != std::string::npos, "chapter id quote escape");

}

}  // namespace

int main() {
    try {
        test_word_timestamp_json_escapes_control_characters();
        test_emit_task_result_serializes_all_file_outputs();
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "file_sink_json_test passed\n";
    return 0;
}
