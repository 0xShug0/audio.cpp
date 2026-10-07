#include "../../app/cli/args.h"
#include "../../app/cli/request.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"

#include "test_assert.h"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

std::string option_from_json_number(const std::string & number) {
    const auto root = engine::io::json::parse("{\"value\":" + number + "}");
    const auto * value = root.find("value");
    engine::test::require(value != nullptr, "json value field parsed");
    return minitts::cli::json_option_string(*value);
}

uint64_t parse_seed(const std::string & value) {
    const std::unordered_map<std::string, std::string> options{{"seed", value}};
    const auto parsed = engine::runtime::parse_u64_option(options, {"seed"});
    if (!parsed.has_value()) {
        throw std::runtime_error("seed option was not parsed");
    }
    return *parsed;
}

void require_seed_rejected(const std::string & value) {
    try {
        (void) parse_seed(value);
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("unsafe seed option was parsed unexpectedly: " + value);
}

void test_large_safe_integer_numbers_stay_decimal() {
    const std::string one_quadrillion = option_from_json_number("1000000000000000");
    engine::test::require_eq(one_quadrillion, "1000000000000000", "1e15 option string");
    engine::test::require_eq(parse_seed(one_quadrillion), uint64_t{1000000000000000}, "1e15 seed");

    const std::string nearby = option_from_json_number("1000000000000001");
    engine::test::require_eq(nearby, "1000000000000001", "1e15+1 option string");
    engine::test::require_eq(parse_seed(nearby), uint64_t{1000000000000001}, "1e15+1 seed");

    const std::string negative = option_from_json_number("-1000000000000000");
    engine::test::require_eq(negative, "-1000000000000000", "negative 1e15 option string");
}

void test_float_numbers_keep_json_formatting() {
    engine::test::require_eq(option_from_json_number("0.7"), "0.7", "float option string");
    engine::test::require_eq(option_from_json_number("1.25"), "1.25", "fractional option string");
}

void test_unsafe_integer_numbers_are_not_silently_rounded() {
    const std::string boundary = option_from_json_number("9007199254740992");
    engine::test::require(
        boundary.find('e') != std::string::npos || boundary.find('E') != std::string::npos,
        "2^53 boundary keeps scientific formatting");
    require_seed_rejected(boundary);

    const std::string rounded = option_from_json_number("9007199254740993");
    engine::test::require(
        rounded.find('e') != std::string::npos || rounded.find('E') != std::string::npos,
        "2^53+1 does not become a rounded decimal option");
    require_seed_rejected(rounded);
}

void test_audio_only_language_is_request_option() {
    const char * argv[] = {"audiocpp_cli", "--task", "asr", "--family", "qwen3_asr", "--language", "en"};
    const auto request = minitts::cli::build_request_from_cli(7, const_cast<char **>(argv));

    engine::test::require(!request.text_input.has_value(), "audio-only language does not synthesize text input");
    engine::test::require_eq(request.options.at("language"), std::string("en"), "audio-only language request option");
}

void test_out_format_selects_wav_sample_format() {
    const char * no_flag[] = {"audiocpp_cli", "--task", "gen"};
    const auto defaults = minitts::cli::wav_write_options_from_cli(3, const_cast<char **>(no_flag));
    engine::test::require(
        defaults.format == engine::audio::WavSampleFormat::Pcm16,
        "no --out-format keeps pcm16");
    engine::test::require(
        defaults.peak_policy == engine::audio::WavPeakPolicy::HardClip,
        "no --out-format keeps hard clip");

    const char * float_flag[] = {"audiocpp_cli", "--out", "a.wav", "--out-format", "float32"};
    engine::test::require(
        minitts::cli::wav_write_options_from_cli(5, const_cast<char **>(float_flag)).format ==
            engine::audio::WavSampleFormat::Float32,
        "--out-format float32");
    engine::test::require(
        minitts::cli::parse_wav_sample_format("pcm24") == engine::audio::WavSampleFormat::Pcm24,
        "--out-format pcm24");

    try {
        (void) minitts::cli::parse_wav_sample_format("wav");
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("unknown --out-format value was accepted");
}

engine::runtime::TaskRequest request_with_artifacts(
    const std::string & artifacts,
    const std::filesystem::path & base_dir) {
    return minitts::cli::build_request_from_json(
        engine::io::json::parse("{\"text\":\"hi\",\"artifacts\":" + artifacts + "}"), base_dir);
}

void require_artifacts_rejected(
    const std::string & artifacts,
    const std::filesystem::path & base_dir,
    const std::string & message) {
    try {
        (void) request_with_artifacts(artifacts, base_dir);
    } catch (const engine::runtime::InvalidRequestError & error) {
        engine::test::require(
            std::string(error.what()).find(message) != std::string::npos,
            "artifacts " + artifacts + " failed with '" + error.what() + "', expected '" + message + "'");
        return;
    }
    throw std::runtime_error("artifacts were accepted: " + artifacts);
}

std::string artifact_text(const engine::runtime::VoiceArtifact & artifact) {
    std::string text;
    for (const std::byte byte : artifact.payload) {
        text.push_back(static_cast<char>(byte));
    }
    return text;
}

void test_request_json_artifacts() {
    static_assert(std::is_base_of_v<std::runtime_error, engine::runtime::InvalidRequestError>,
                  "the CLI reports artifact errors like any other bad input");
    const auto dir = std::filesystem::temp_directory_path() /
        ("audiocpp-request-artifacts-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct RemoveDir {
        std::filesystem::path path;
        ~RemoveDir() {
            std::error_code ignored;
            std::filesystem::remove_all(path, ignored);
        }
    } remove_dir{dir};
    std::filesystem::create_directories(dir / "turns");
    std::ofstream(dir / "turns" / "reply.bin", std::ios::binary) << std::string("\x01\x00\xff\x7f", 4);
    std::ofstream(dir / "empty.bin", std::ios::binary).close();

    // Inline and file payloads, in order, ids repeated, meta values as strings.
    const auto request = request_with_artifacts(
        "[{\"id\":\"turn.question\",\"kind\":\"custom\",\"payload\":\"aGVsbG8=\","
        "\"meta\":{\"mime\":\"audio/wav\",\"steps\":12,\"ended\":true}},"
        "{\"id\":\"turn.reply\",\"kind\":\"acoustic_tokens\",\"path\":\"turns/reply.bin\"},"
        "{\"id\":\"turn.question\",\"kind\":\"custom\",\"payload\":\"data:audio/wav;base64,d29y\\nbGQ=\",\"meta\":null},"
        "{\"id\":\"empty\",\"kind\":\"midi\",\"path\":" +
            engine::io::json::stringify_string((dir / "empty.bin").string()) + "},"
        "{\"id\":\"none\",\"kind\":\"vad_state\",\"payload\":\"\",\"path\":null}]",
        dir);
    const auto & artifacts = request.input_artifacts;
    engine::test::require_eq(artifacts.size(), size_t{5}, "request artifact count");
    engine::test::require(
        artifacts[0].id == "turn.question" && artifacts[0].kind == engine::runtime::ArtifactKind::Custom &&
            artifact_text(artifacts[0]) == "hello",
        "inline artifact");
    engine::test::require(
        artifacts[0].meta.size() == 3 && artifacts[0].meta.at("mime") == "audio/wav" &&
            artifacts[0].meta.at("steps") == "12" && artifacts[0].meta.at("ended") == "true",
        "artifact meta values");
    engine::test::require(
        artifacts[1].id == "turn.reply" && artifacts[1].kind == engine::runtime::ArtifactKind::AcousticTokens &&
            artifact_text(artifacts[1]) == std::string("\x01\x00\xff\x7f", 4) && artifacts[1].meta.empty(),
        "file artifact, path relative to the request file");
    engine::test::require(
        artifacts[2].id == "turn.question" && artifact_text(artifacts[2]) == "world" && artifacts[2].meta.empty(),
        "data URI artifact with a line break");
    engine::test::require(
        artifacts[3].kind == engine::runtime::ArtifactKind::Midi && artifacts[3].payload.empty() &&
            artifacts[4].kind == engine::runtime::ArtifactKind::VadState && artifacts[4].payload.empty(),
        "empty payloads");

    // The kind names the server writes in responses.
    const std::vector<std::pair<std::string, engine::runtime::ArtifactKind>> kinds{
        {"speaker_embedding", engine::runtime::ArtifactKind::SpeakerEmbedding},
        {"style_embedding", engine::runtime::ArtifactKind::StyleEmbedding},
        {"prompt_embedding", engine::runtime::ArtifactKind::PromptEmbedding},
        {"acoustic_tokens", engine::runtime::ArtifactKind::AcousticTokens},
        {"midi", engine::runtime::ArtifactKind::Midi},
        {"transcript_alignment", engine::runtime::ArtifactKind::TranscriptAlignment},
        {"diarization_state", engine::runtime::ArtifactKind::DiarizationState},
        {"vad_state", engine::runtime::ArtifactKind::VadState},
        {"custom", engine::runtime::ArtifactKind::Custom},
    };
    for (const auto & [name, kind] : kinds) {
        const auto parsed = request_with_artifacts("[{\"id\":\"k\",\"kind\":\"" + name + "\",\"payload\":\"\"}]", dir);
        engine::test::require(parsed.input_artifacts.front().kind == kind, "artifact kind " + name);
    }
    engine::test::require(request_with_artifacts("null", dir).input_artifacts.empty(), "null artifacts");
    engine::test::require(
        minitts::cli::build_request_from_json(engine::io::json::parse("{\"text\":\"hi\"}"), dir).input_artifacts.empty(),
        "no artifacts field");

    require_artifacts_rejected("{}", dir, "artifacts must be an array of artifact objects");
    require_artifacts_rejected("[1]", dir, "artifacts[0] must be an object with id, kind, and payload or path");
    require_artifacts_rejected("[{\"kind\":\"custom\",\"payload\":\"\"}]", dir, "artifacts[0]: id must be a non-empty string");
    require_artifacts_rejected("[{\"id\":\"\",\"kind\":\"custom\",\"payload\":\"\"}]", dir, "artifacts[0]: id must be a non-empty string");
    require_artifacts_rejected("[{\"id\":7,\"kind\":\"custom\",\"payload\":\"\"}]", dir, "artifacts[0]: id must be a non-empty string");
    require_artifacts_rejected("[{\"id\":\"x\",\"payload\":\"\"}]", dir, "artifacts[0] (x): kind must be a string");
    require_artifacts_rejected(
        "[{\"id\":\"ok\",\"kind\":\"custom\",\"payload\":\"\"},{\"id\":\"x\",\"kind\":\"tokens\",\"payload\":\"\"}]", dir,
        "artifacts[1] (x): unknown kind 'tokens', expected one of speaker_embedding, style_embedding,");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\"}]", dir, "artifacts[0] (x): give exactly one of payload");
    require_artifacts_rejected(
        "[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":\"\",\"path\":\"turns/reply.bin\"}]", dir,
        "artifacts[0] (x): give exactly one of payload");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":5}]", dir, "artifacts[0] (x): payload must be a base64 string");
    require_artifacts_rejected(
        "[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":\"aGVsbG8*\"}]", dir,
        "artifacts[0] (x): payload is not valid base64 (malformed base64 payload: invalid character)");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\",\"path\":\"\"}]", dir, "artifacts[0] (x): path must be a non-empty string");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\",\"path\":\"missing.bin\"}]", dir, "artifacts[0] (x): path is not a regular file");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\",\"path\":\"turns\"}]", dir, "artifacts[0] (x): path is not a regular file");
    require_artifacts_rejected("[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":\"\",\"meta\":[]}]", dir, "artifacts[0] (x): meta must be an object");
    require_artifacts_rejected(
        "[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":\"\",\"meta\":{\"k\":{}}}]", dir,
        "artifacts[0] (x): meta.k must be a string, number or boolean");
    require_artifacts_rejected(
        "[{\"id\":\"x\",\"kind\":\"custom\",\"payload_hex\":\"00\"}]", dir,
        "artifacts[0]: unknown field 'payload_hex', an artifact has id, kind, payload or path, and meta");

#if !defined(_WIN32)
    // The payload limit counts every artifact of the request. The file is
    // sparse and too big to be read, so nothing near the limit is written.
    std::ofstream(dir / "limit.bin", std::ios::binary).close();
    std::filesystem::resize_file(dir / "limit.bin", minitts::cli::kMaxRequestArtifactBytes - 4);
    require_artifacts_rejected(
        "[{\"id\":\"x\",\"kind\":\"custom\",\"payload\":\"aGVsbG8=\"},"
        "{\"id\":\"y\",\"kind\":\"custom\",\"path\":\"limit.bin\"}]",
        dir, "artifacts[1] (y): the request's artifact payloads total more than 2147483648 bytes");
#endif
}

}  // namespace

int main() {
    test_large_safe_integer_numbers_stay_decimal();
    test_float_numbers_keep_json_formatting();
    test_unsafe_integer_numbers_are_not_silently_rounded();
    test_audio_only_language_is_request_option();
    test_out_format_selects_wav_sample_format();
    test_request_json_artifacts();
    std::cout << "cli_request_options_test passed\n";
    return 0;
}
