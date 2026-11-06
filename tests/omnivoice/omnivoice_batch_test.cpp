// OmniVoice native batch test: run() one request at a time vs run_batch()
// with the same requests, for several batch sizes.
//
// For every batch size B the first B texts (cycled) are used, request i gets
// seed + i. It reports, per batch size:
//   - wall time of B sequential run() calls and of one run_batch() call,
//   - per-request parity between the two (sample count, max abs diff),
//   - optionally, exact comparison of the run() output against raw float32
//     files saved by an earlier build (--compare-dir), to prove that the
//     single-request path did not change. Any mismatch or missing baseline
//     makes the program exit with status 1.
//
// The run() vs run_batch() lines are informational: with f16 weights batched
// kernels can round differently. omnivoice_batch_parity_test asserts the
// exact guarantees.
//
// Example:
//   build/bin/omnivoice_batch_test --model models/OmniVoice \
//     --clone-audio tests/omnivoice/assets/ref_audio_02.wav \
//     --reference-text-file tests/omnivoice/assets/ref_text_02.txt \
//     --text-file tests/omnivoice/assets/clone_batch_prompts.txt \
//     --batch-sizes 1,2,4,8 --out-dir tests/omnivoice/outputs/batch_test
//
// --timing-log <file> writes the engine [TIMING] lines (generator compute,
// readback, scoring, decode, ...) of every call.

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace rt = engine::runtime;
using Clock = std::chrono::steady_clock;

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
}

std::string arg_or(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    return find_arg(argc, argv, name).value_or(fallback);
}

std::vector<std::string> all_args(int argc, char ** argv, const std::string & name) {
    std::vector<std::string> values;
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            values.emplace_back(argv[i + 1]);
        }
    }
    return values;
}

std::string read_text_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream out;
    out << input.rdbuf();
    std::string text = out.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

std::vector<std::string> read_lines(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

std::vector<int> parse_int_list(const std::string & value) {
    std::vector<int> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }
    return out;
}

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

const rt::AudioBuffer & audio_of(const rt::TaskResult & result) {
    if (!result.audio_output.has_value()) {
        throw std::runtime_error("OmniVoice result has no audio output");
    }
    return *result.audio_output;
}

void save_raw(const std::filesystem::path & path, const std::vector<float> & samples) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(float)));
}

std::optional<std::vector<float>> load_raw(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return std::nullopt;
    }
    const auto bytes = static_cast<size_t>(input.tellg());
    std::vector<float> samples(bytes / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char *>(samples.data()), static_cast<std::streamsize>(samples.size() * sizeof(float)));
    return samples;
}

struct Diff {
    bool same_length = false;
    bool identical = false;
    float max_abs = 0.0F;
};

Diff compare(const std::vector<float> & a, const std::vector<float> & b) {
    Diff diff;
    diff.same_length = a.size() == b.size();
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        diff.max_abs = std::max(diff.max_abs, std::fabs(a[i] - b[i]));
    }
    diff.identical = diff.same_length && diff.max_abs == 0.0F;
    return diff;
}

std::string describe(const Diff & diff, size_t a_size, size_t b_size) {
    std::ostringstream out;
    if (diff.identical) {
        out << "identical";
    } else {
        out << "max |diff| " << std::setprecision(4) << diff.max_abs;
        if (!diff.same_length) {
            out << ", samples " << a_size << " vs " << b_size;
        }
    }
    return out.str();
}

void save_outputs(
    const std::filesystem::path & out_dir,
    const std::string & prefix,
    const std::vector<rt::TaskResult> & results) {
    for (size_t i = 0; i < results.size(); ++i) {
        const auto & audio = audio_of(results[i]);
        const std::string stem = prefix + "_" + std::to_string(i);
        engine::audio::write_wav(out_dir / (stem + ".wav"), audio.sample_rate, audio.channels, audio.samples);
        save_raw(out_dir / (stem + ".f32"), audio.samples);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto model_path = find_arg(argc, argv, "--model");
        const auto clone_audio = find_arg(argc, argv, "--clone-audio");
        const auto reference_text_file = find_arg(argc, argv, "--reference-text-file");
        const auto text_file = find_arg(argc, argv, "--text-file");
        if (!model_path || !text_file) {
            std::cerr
                << "usage: omnivoice_batch_test --model <dir> --text-file <txt> [--clone-audio <wav> "
                   "--reference-text-file <txt>] [--language <lang>] [--batch-sizes 1,2,4,8] [--repeats 3] "
                   "[--warmup 1] [--mode both|sequential|batch] [--seed 42] [--num-step 32] "
                   "[--backend cuda] [--device 0] [--threads 4] [--session-option key=value]... "
                   "[--request-option key=value]... "
                   "[--out-dir <dir>] [--compare-dir <dir>] [--timing-log <file>]\n";
            return 2;
        }
        const std::string language = arg_or(argc, argv, "--language", "");
        const auto batch_sizes = parse_int_list(arg_or(argc, argv, "--batch-sizes", "1,2,4,8"));
        const int repeats = std::max(1, std::stoi(arg_or(argc, argv, "--repeats", "3")));
        const int warmup = std::max(0, std::stoi(arg_or(argc, argv, "--warmup", "1")));
        const std::string mode = arg_or(argc, argv, "--mode", "both");
        const uint32_t seed = static_cast<uint32_t>(std::stoul(arg_or(argc, argv, "--seed", "42")));
        const std::string num_step = arg_or(argc, argv, "--num-step", "32");
        const std::filesystem::path out_dir = arg_or(argc, argv, "--out-dir", "tests/omnivoice/outputs/batch_test");
        const auto compare_dir = find_arg(argc, argv, "--compare-dir");
        const bool run_sequential = mode == "both" || mode == "sequential";
        const bool run_batched = mode == "both" || mode == "batch";
        if (!run_sequential && !run_batched) {
            throw std::runtime_error("--mode must be both, sequential, or batch");
        }
        if (compare_dir && !run_sequential) {
            throw std::runtime_error("--compare-dir compares run() outputs and needs --mode both or sequential");
        }
        std::filesystem::create_directories(out_dir);
        if (const auto timing_log = find_arg(argc, argv, "--timing-log")) {
            // Per-call engine timings ([TIMING] lines), e.g. omnivoice.generator.compute_ms.
            engine::debug::configure_logging({true, *timing_log});
        }

        const auto texts = read_lines(*text_file);
        if (texts.empty()) {
            throw std::runtime_error("--text-file has no non-empty lines");
        }
        std::optional<rt::AudioBuffer> reference_audio;
        std::string reference_text;
        if (clone_audio) {
            const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*clone_audio));
            reference_audio = rt::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
            if (reference_text_file) {
                reference_text = read_text_file(*reference_text_file);
            }
        }

        rt::ModelLoadRequest load_request;
        load_request.model_path = *model_path;
        load_request.family_hint = "omnivoice";
        auto registry = rt::make_default_registry();
        auto model = registry.load(load_request);

        rt::SessionOptions session_options;
        session_options.backend.type = parse_backend(arg_or(argc, argv, "--backend", "cuda"));
        session_options.backend.device = std::stoi(arg_or(argc, argv, "--device", "0"));
        session_options.backend.threads = std::stoi(arg_or(argc, argv, "--threads", "4"));
        for (const auto & option : all_args(argc, argv, "--session-option")) {
            const auto eq = option.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("--session-option expects key=value: " + option);
            }
            session_options.options[option.substr(0, eq)] = option.substr(eq + 1);
        }
        std::vector<std::pair<std::string, std::string>> request_options;
        for (const auto & option : all_args(argc, argv, "--request-option")) {
            const auto eq = option.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("--request-option expects key=value: " + option);
            }
            request_options.emplace_back(option.substr(0, eq), option.substr(eq + 1));
        }
        auto session_base = model->create_task_session(
            rt::TaskSpec{rt::VoiceTaskKind::Tts, rt::RunMode::Offline},
            session_options);
        auto * offline = dynamic_cast<rt::IOfflineVoiceTaskSession *>(session_base.get());
        auto * batched = dynamic_cast<rt::IBatchedOfflineVoiceTaskSession *>(session_base.get());
        if (offline == nullptr) {
            throw std::runtime_error("OmniVoice session is not an offline session");
        }
        if (run_batched && batched == nullptr) {
            std::cout << "session has no native run_batch(); running sequential only\n";
        }

        const auto make_request = [&](size_t index) {
            rt::TaskRequest request;
            request.text_input = rt::Transcript{texts[index % texts.size()], language};
            if (reference_audio) {
                request.voice = rt::VoiceCondition{};
                request.voice->speaker = rt::VoiceReference{};
                request.voice->speaker->audio = *reference_audio;
                request.options["reference_text"] = reference_text;
            }
            request.options["num_inference_steps"] = num_step;
            request.options["seed"] = std::to_string(seed + static_cast<uint32_t>(index));
            for (const auto & [key, value] : request_options) {
                request.options[key] = value;
            }
            return request;
        };

        session_base->prepare(rt::build_preparation_request(make_request(0)));
        for (int i = 0; i < warmup; ++i) {
            (void) offline->run(make_request(0));
        }

        int compare_failures = 0;
        std::cout << std::fixed << std::setprecision(3);
        for (const int batch_size : batch_sizes) {
            if (batch_size <= 0) {
                continue;
            }
            std::vector<rt::TaskRequest> requests;
            for (int i = 0; i < batch_size; ++i) {
                requests.push_back(make_request(static_cast<size_t>(i)));
            }
            const std::string tag = "b" + std::to_string(batch_size);
            std::cout << "\n=== batch size " << batch_size << " ===\n";

            std::vector<rt::TaskResult> sequential_results;
            if (run_sequential) {
                std::vector<double> times;
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    std::vector<rt::TaskResult> results;
                    const auto start = Clock::now();
                    for (const auto & request : requests) {
                        results.push_back(offline->run(request));
                    }
                    times.push_back(elapsed_ms(start, Clock::now()));
                    if (repeat > 0) {
                        for (size_t i = 0; i < results.size(); ++i) {
                            const auto diff = compare(audio_of(results[i]).samples, audio_of(sequential_results[i]).samples);
                            if (!diff.identical) {
                                std::cout << "  WARNING sequential repeat " << repeat << " request " << i
                                          << " differs from repeat 0: "
                                          << describe(diff, audio_of(results[i]).samples.size(),
                                                      audio_of(sequential_results[i]).samples.size())
                                          << "\n";
                            }
                        }
                    } else {
                        sequential_results = std::move(results);
                    }
                }
                std::cout << "sequential run() x" << batch_size << ": " << median(times) / 1000.0 << " s\n";
                save_outputs(out_dir, "sequential_" + tag, sequential_results);
                if (compare_dir) {
                    for (size_t i = 0; i < sequential_results.size(); ++i) {
                        const auto path = std::filesystem::path(*compare_dir) /
                            ("sequential_" + tag + "_" + std::to_string(i) + ".f32");
                        const auto baseline = load_raw(path);
                        if (!baseline) {
                            std::cout << "  FAIL baseline " << path.string() << " missing\n";
                            ++compare_failures;
                            continue;
                        }
                        const auto & samples = audio_of(sequential_results[i]).samples;
                        const auto diff = compare(samples, *baseline);
                        std::cout << "  " << (diff.identical ? "PASS" : "FAIL") << " request " << i
                                  << " run() vs baseline: " << describe(diff, samples.size(), baseline->size()) << "\n";
                        if (!diff.identical) {
                            ++compare_failures;
                        }
                    }
                }
            }

            if (run_batched && batched != nullptr) {
                std::vector<double> times;
                std::vector<rt::TaskResult> batch_results;
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    const auto start = Clock::now();
                    auto results = batched->run_batch(requests);
                    times.push_back(elapsed_ms(start, Clock::now()));
                    if (results.size() != requests.size()) {
                        throw std::runtime_error("run_batch returned the wrong number of results");
                    }
                    if (repeat == 0) {
                        batch_results = std::move(results);
                    }
                }
                std::cout << "run_batch() of " << batch_size << ": " << median(times) / 1000.0 << " s\n";
                save_outputs(out_dir, "batch_" + tag, batch_results);
                for (size_t i = 0; i < batch_results.size(); ++i) {
                    const auto & samples = audio_of(batch_results[i]).samples;
                    const int sample_rate = audio_of(batch_results[i]).sample_rate;
                    std::cout << "  request " << i << ": audio " << static_cast<double>(samples.size()) / sample_rate
                              << " s";
                    if (!sequential_results.empty()) {
                        const auto & reference = audio_of(sequential_results[i]).samples;
                        std::cout << ", vs run(): " << describe(compare(samples, reference), samples.size(), reference.size());
                    }
                    std::cout << "\n";
                }
            }
        }
        if (compare_failures > 0) {
            std::cout << "\n" << compare_failures << " --compare-dir check(s) failed\n";
            return 1;
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "omnivoice_batch_test failed: " << e.what() << "\n";
        return 1;
    }
}
