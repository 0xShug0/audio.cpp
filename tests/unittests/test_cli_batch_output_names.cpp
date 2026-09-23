#include "../../app/cli/batch.h"
#include "../../app/workflow/file_sink.h"

#include "test_assert.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Owns a freshly created directory and removes only that directory on exit.
class ScratchDir {
public:
    ScratchDir() {
        std::random_device rd;
        std::mt19937_64 rng((static_cast<uint64_t>(rd()) << 32) ^ rd());
        const auto base = fs::temp_directory_path();
        for (int attempt = 0; attempt < 64; ++attempt) {
            const auto candidate = base / ("audiocpp_cli_batch_output_names_" + std::to_string(rng()));
            std::error_code ec;
            if (fs::create_directory(candidate, ec) && !ec) {
                root_ = candidate;
                return;
            }
        }
        throw std::runtime_error("failed to create a fresh scratch directory under " + base.string());
    }

    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    ScratchDir(const ScratchDir &) = delete;
    ScratchDir & operator=(const ScratchDir &) = delete;

    fs::path make_subdir(const std::string & name) const {
        const auto path = root_ / name;
        if (!fs::create_directory(path)) {
            throw std::runtime_error("scratch subdirectory already exists: " + path.string());
        }
        return path;
    }

    const fs::path & root() const { return root_; }

private:
    fs::path root_;
};

void write_file(const fs::path & path, const std::string & content) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write test fixture: " + path.string());
    }
    out << content;
}

// Minimal valid 16 kHz mono PCM16 WAV with a handful of samples.
void write_test_wav(const fs::path & path) {
    const int16_t samples[] = {0, 1000, -1000, 500, -500, 0};
    const uint32_t data_bytes = static_cast<uint32_t>(sizeof(samples));
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write wav fixture: " + path.string());
    }
    const auto u16 = [&](uint16_t v) {
        out.put(static_cast<char>(v & 0xff));
        out.put(static_cast<char>((v >> 8) & 0xff));
    };
    const auto u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            out.put(static_cast<char>((v >> (8 * i)) & 0xff));
        }
    };
    out.write("RIFF", 4);
    u32(36 + data_bytes);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    u32(16);
    u16(1);
    u16(1);
    u32(16000);
    u32(16000 * 2);
    u16(2);
    u16(16);
    out.write("data", 4);
    u32(data_bytes);
    for (const int16_t sample : samples) {
        u16(static_cast<uint16_t>(sample));
    }
}

minitts::app::AppBatchRequest build_batch(
    const std::vector<std::string> & args,
    const engine::runtime::TaskRequest & base_request = {},
    const std::string & audio_role = "audio") {
    std::vector<std::string> storage;
    storage.reserve(args.size() + 1);
    storage.emplace_back("audiocpp_cli");
    storage.insert(storage.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for (auto & arg : storage) {
        argv.push_back(arg.data());
    }
    return minitts::cli::build_batch_request_from_cli(
        static_cast<int>(argv.size()), argv.data(), base_request, audio_role);
}

void require_collision(
    const std::vector<std::string> & args,
    const std::string & output_name,
    const std::string & label,
    const std::string & audio_role = "audio") {
    try {
        (void) build_batch(args, {}, audio_role);
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        engine::test::require(
            message.find("output name '" + output_name + "'") != std::string::npos,
            label + ": collision error names the output '" + output_name + "': " + message);
        return;
    }
    throw std::runtime_error(label + ": colliding batch output names were accepted");
}

std::vector<std::string> ids_of(const minitts::app::AppBatchRequest & batch) {
    std::vector<std::string> ids;
    for (const auto & item : batch.requests) {
        ids.push_back(item.id);
    }
    return ids;
}

void require_ids(
    const minitts::app::AppBatchRequest & batch,
    const std::vector<std::string> & expected,
    const std::string & label) {
    const auto actual = ids_of(batch);
    engine::test::require_eq(actual.size(), expected.size(), label + " request count");
    for (size_t i = 0; i < expected.size(); ++i) {
        engine::test::require_eq(actual[i], expected[i], label + " id[" + std::to_string(i) + "]");
    }
}

void test_sanitizer_preconditions() {
    // The collisions below are only meaningful if the real sanitizer merges these ids.
    engine::test::require_eq(
        minitts::app::safe_output_name("a b"), minitts::app::safe_output_name("a_b"), "'a b' vs 'a_b'");
    engine::test::require_eq(
        minitts::app::safe_output_name("take 1"), minitts::app::safe_output_name("take_1"), "'take 1' vs 'take_1'");
    engine::test::require_eq(
        minitts::app::safe_output_name("x.y"), minitts::app::safe_output_name("x y"), "'x.y' vs 'x y'");
}

void test_text_dir_duplicate_stems_rejected(const ScratchDir & scratch) {
    const auto dir = scratch.make_subdir("text_dup_stem");
    write_file(dir / "intro.txt", "Plain intro.\n");
    write_file(dir / "intro.md", "Markdown intro.\n");
    require_collision({"--batch-text-dir", dir.string()}, "intro", "text dir intro.txt/intro.md");
}

void test_text_dir_sanitized_names_rejected(const ScratchDir & scratch) {
    const auto dir = scratch.make_subdir("text_sanitized");
    write_file(dir / "a b.txt", "first\n");
    write_file(dir / "a_b.txt", "second\n");
    require_collision({"--batch-text-dir", dir.string()}, "a_b", "text dir 'a b'/'a_b'");
}

void test_audio_dir_sanitized_names_rejected(const ScratchDir & scratch) {
    const auto dir = scratch.make_subdir("audio_sanitized");
    write_test_wav(dir / "take 1.wav");
    write_test_wav(dir / "take_1.wav");
    require_collision({"--batch-audio-dir", dir.string()}, "take_1", "audio dir (audio role)", "audio");
    require_collision({"--batch-audio-dir", dir.string()}, "take_1", "audio dir (source_audio role)", "source_audio");
}

void test_request_sequence_duplicate_ids_rejected(const ScratchDir & scratch) {
    const auto exact = scratch.root() / "sequence_duplicate.json";
    write_file(exact, R"([{"id":"dup","text":"one"},{"id":"dup","text":"two"}])");
    require_collision({"--request-sequence", exact.string()}, "dup", "sequence exact duplicate ids");

    const auto sanitized = scratch.root() / "sequence_sanitized.json";
    write_file(sanitized, R"({"requests":[{"id":"x y","text":"one"},{"id":"x.y","text":"two"}]})");
    require_collision({"--request-sequence", sanitized.string()}, "x_y", "sequence sanitized ids");
}

void test_request_sequence_explicit_vs_fallback_rejected(const ScratchDir & scratch) {
    // The second item has no id and falls back to request_1, which the first item claimed explicitly.
    const auto path = scratch.root() / "sequence_fallback.json";
    write_file(path, R"([{"id":"request_1","text":"explicit"},{"text":"fallback"}])");
    require_collision({"--request-sequence", path.string()}, "request_1", "sequence explicit vs fallback id");
}

void test_unique_text_dir_preserved(const ScratchDir & scratch) {
    const auto dir = scratch.make_subdir("text_unique");
    write_file(dir / "alpha.txt", "Alpha   line\none.\n");
    write_file(dir / "beta.md", "Beta text.\n");
    write_file(dir / "gamma.json", R"({"text":"Gamma text."})");
    write_file(dir / "ignored.csv", "not,a,batch,input\n");

    engine::runtime::TaskRequest base;
    base.options["speed"] = "1.1";
    const auto batch = build_batch({"--batch-text-dir", dir.string(), "--language", "en"}, base);
    require_ids(batch, {"alpha", "beta", "gamma"}, "unique text dir");
    const std::vector<std::string> texts = {"Alpha line one.", "Beta text.", "Gamma text."};
    for (size_t i = 0; i < texts.size(); ++i) {
        const auto & request = batch.requests[i].request;
        engine::test::require(request.text_input.has_value(), "unique text dir text input " + std::to_string(i));
        engine::test::require_eq(request.text_input->text, texts[i], "unique text dir text " + std::to_string(i));
        engine::test::require_eq(request.text_input->language, std::string("en"), "unique text dir language");
        engine::test::require_eq(request.options.at("speed"), std::string("1.1"), "unique text dir base option");
    }
}

void test_unique_text_file_preserved(const ScratchDir & scratch) {
    const auto path = scratch.root() / "lines.txt";
    write_file(path, "first line\n\nthird line\r\n");
    const auto batch = build_batch({"--batch-text-file", path.string()});
    require_ids(batch, {"line_1", "line_3"}, "unique text file");
    engine::test::require_eq(batch.requests[1].request.text_input->text, std::string("third line"), "text file line 3");
}

void test_unique_audio_dir_preserved(const ScratchDir & scratch) {
    const auto dir = scratch.make_subdir("audio_unique");
    write_test_wav(dir / "one.wav");
    write_test_wav(dir / "two.wav");

    const auto audio_batch = build_batch({"--batch-audio-dir", dir.string()}, {}, "audio");
    require_ids(audio_batch, {"one", "two"}, "unique audio dir (audio role)");
    for (const auto & item : audio_batch.requests) {
        engine::test::require(item.request.audio_input.has_value(), "audio role loads wav for " + item.id);
    }

    const auto source_batch = build_batch({"--batch-audio-dir", dir.string()}, {}, "source_audio");
    require_ids(source_batch, {"one", "two"}, "unique audio dir (source_audio role)");
    engine::test::require_eq(
        source_batch.requests[0].request.options.at("source_audio"),
        (dir / "one.wav").string(),
        "source_audio option path");
}

void test_unique_request_sequence_preserved(const ScratchDir & scratch) {
    const auto path = scratch.root() / "sequence_unique.json";
    write_file(
        path,
        R"([{"id":"first","text":"a","options":{"speed":"1.2"}},{"text":"b"},{"id":"third","text":"c"}])");
    const auto batch = build_batch({"--request-sequence", path.string()});
    require_ids(batch, {"first", "request_1", "third"}, "unique request sequence");
    engine::test::require_eq(batch.requests[0].request.options.at("speed"), std::string("1.2"), "sequence option");
    engine::test::require_eq(batch.requests[2].request.text_input->text, std::string("c"), "sequence text order");
}

void test_existing_validation_preserved(const ScratchDir & scratch) {
    const auto path = scratch.root() / "sequence_empty.json";
    write_file(path, "[]");
    try {
        (void) build_batch({"--request-sequence", path.string()});
    } catch (const std::runtime_error & error) {
        const std::string message = error.what();
        engine::test::require(
            message.find("at least one request") != std::string::npos,
            "empty request sequence keeps its original error: " + message);
        return;
    }
    throw std::runtime_error("empty request sequence was accepted");
}

}  // namespace

int main() {
    try {
        ScratchDir scratch;
        const std::vector<std::pair<std::string, std::function<void()>>> cases = {
            {"sanitizer preconditions", [&] { test_sanitizer_preconditions(); }},
            {"text directory duplicate stems", [&] { test_text_dir_duplicate_stems_rejected(scratch); }},
            {"text directory sanitized names", [&] { test_text_dir_sanitized_names_rejected(scratch); }},
            {"audio directory sanitized names", [&] { test_audio_dir_sanitized_names_rejected(scratch); }},
            {"request sequence duplicate ids", [&] { test_request_sequence_duplicate_ids_rejected(scratch); }},
            {"request sequence fallback ids", [&] { test_request_sequence_explicit_vs_fallback_rejected(scratch); }},
            {"unique text directory", [&] { test_unique_text_dir_preserved(scratch); }},
            {"unique text file", [&] { test_unique_text_file_preserved(scratch); }},
            {"unique audio directory", [&] { test_unique_audio_dir_preserved(scratch); }},
            {"unique request sequence", [&] { test_unique_request_sequence_preserved(scratch); }},
            {"existing validation", [&] { test_existing_validation_preserved(scratch); }},
        };
        size_t failures = 0;
        for (const auto & item : cases) {
            try {
                item.second();
                std::cout << "PASS: " << item.first << "\n";
            } catch (const std::exception & error) {
                ++failures;
                std::cerr << "FAIL: " << item.first << ": " << error.what() << "\n";
            }
        }
        std::cout << (cases.size() - failures) << " passed, " << failures << " failed\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << "cli_batch_output_names_test setup failed: " << error.what() << "\n";
        return 1;
    }
}
