// OmniVoice batching and fast-path parity test. Exits non-zero when any
// check fails.
//
// The requests differ in length (several attention segments), seed, step
// count and guidance scale; one samples with class_temperature > 0. With f32
// generator weights:
//   - batch_of_one: run_batch({r}) equals run(r) bit for bit, for every
//     request.
//   - batched_vs_run: run_batch() of all requests equals run() of each greedy
//     request bit for bit, in request order, on CUDA. Other backends can pick
//     different kernels for the batched shapes, so there it requires the same
//     length and finite audio and reports the difference.
//   - batched_repeat: a second run_batch() gives the same audio for every
//     request, sampled included (each request keeps its own seeded RNG).
//   - device_scoring_off: device_scoring=false equals the default for every
//     request.
//   - packed_projections_off: packed_projections=false equals the default for
//     every greedy request.
// Batching and packed projections change GEMM shapes, so f32 logits can
// differ in the last bits. Greedy decoding absorbs that; sampled decoding can
// draw a different token, so sampled requests are only compared within one
// layout.
//
// Usage:
//   omnivoice_batch_parity_test --model models/OmniVoice \
//     --clone-audio tests/omnivoice/assets/ref_audio_02.wav \
//     --reference-text-file tests/omnivoice/assets/ref_text_02.txt \
//     [--backend cuda|vulkan|cpu] [--device 0] [--num-step 8]

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace rt = engine::runtime;

std::optional<std::string> find_arg(int argc, char ** argv, const std::string & name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::string(argv[i + 1]);
        }
    }
    return std::nullopt;
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
    throw std::runtime_error("unsupported backend: " + value);
}

const std::vector<float> & samples_of(const rt::TaskResult & result) {
    if (!result.audio_output.has_value() || result.audio_output->samples.empty()) {
        throw std::runtime_error("OmniVoice result has no audio");
    }
    return result.audio_output->samples;
}

// Requests of different lengths, so the batch has several attention segments
// (odd and even token counts), with per-request seeds, step counts, guidance
// scales and one sampled request.
struct RequestSpec {
    std::string text;
    std::unordered_map<std::string, std::string> options;
    bool sampled = false;
};

std::vector<RequestSpec> request_specs(const std::string & num_step) {
    return {
        {"The afternoon train is running on time.", {{"seed", "17"}}},
        {"Please remember to bring an umbrella tomorrow, the forecast says heavy rain will start before noon.",
         {{"seed", "18"}}},
        {"Thank you for calling.", {{"seed", "19"}, {"guidance_scale", "3"}}},
        {"Our new library opens next Monday with longer hours on weekends.",
         {{"seed", "20"}, {"class_temperature", "0.8"}},
         true},
        {"Good morning, everyone.", {{"seed", "21"}, {"num_inference_steps", std::to_string(std::stoi(num_step) + 2)}}},
    };
}

class Runner {
public:
    Runner(int argc, char ** argv) {
        const auto model_path = find_arg(argc, argv, "--model");
        const auto clone_audio = find_arg(argc, argv, "--clone-audio");
        const auto reference_text_file = find_arg(argc, argv, "--reference-text-file");
        if (!model_path || !clone_audio || !reference_text_file) {
            throw std::invalid_argument(
                "usage: omnivoice_batch_parity_test --model <dir> --clone-audio <wav> --reference-text-file <txt> "
                "[--backend cuda|vulkan|cpu] [--device 0] [--num-step 8]");
        }
        backend_ = parse_backend(find_arg(argc, argv, "--backend").value_or("cuda"));
        device_ = std::stoi(find_arg(argc, argv, "--device").value_or("0"));
        num_step_ = find_arg(argc, argv, "--num-step").value_or("8");
        const auto wav = engine::audio::read_wav_f32(std::filesystem::path(*clone_audio));
        reference_audio_ = rt::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
        reference_text_ = read_text_file(*reference_text_file);

        rt::ModelLoadRequest load_request;
        load_request.model_path = *model_path;
        load_request.family_hint = "omnivoice";
        registry_ = std::make_unique<rt::ModelRegistry>(rt::make_default_registry());
        model_ = registry_->load(load_request);
    }

    std::vector<rt::TaskRequest> requests() const {
        std::vector<rt::TaskRequest> out;
        for (const auto & spec : request_specs(num_step_)) {
            rt::TaskRequest request;
            request.text_input = rt::Transcript{spec.text, ""};
            request.voice = rt::VoiceCondition{};
            request.voice->speaker = rt::VoiceReference{};
            request.voice->speaker->audio = reference_audio_;
            request.options["reference_text"] = reference_text_;
            request.options["num_inference_steps"] = num_step_;
            for (const auto & [key, value] : spec.options) {
                request.options[key] = value;
            }
            out.push_back(std::move(request));
        }
        return out;
    }

    // CUDA's f32 kernels give a row the same result whatever the batch, so
    // batched greedy audio is bit-identical there. Other backends (Vulkan)
    // pick kernels by row count and can round batched rows differently.
    bool exact_batching() const { return backend_ == engine::core::BackendType::Cuda; }

    std::vector<bool> sampled() const {
        std::vector<bool> out;
        for (const auto & spec : request_specs(num_step_)) {
            out.push_back(spec.sampled);
        }
        return out;
    }

    // Runs `body` with a fresh offline session using f32 generator weights plus
    // `options`; the session is released before returning.
    void with_session(
        const std::unordered_map<std::string, std::string> & options,
        const std::function<void(rt::IOfflineVoiceTaskSession &, rt::IBatchedOfflineVoiceTaskSession &)> & body) {
        rt::SessionOptions session_options;
        session_options.backend.type = backend_;
        session_options.backend.device = device_;
        session_options.options["omnivoice.generator_weight_type"] = "f32";
        for (const auto & [key, value] : options) {
            session_options.options[key] = value;
        }
        auto session = model_->create_task_session(rt::TaskSpec{rt::VoiceTaskKind::Tts, rt::RunMode::Offline}, session_options);
        auto * offline = dynamic_cast<rt::IOfflineVoiceTaskSession *>(session.get());
        auto * batched = dynamic_cast<rt::IBatchedOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr || batched == nullptr) {
            throw std::runtime_error("OmniVoice session must implement offline run() and run_batch()");
        }
        session->prepare(rt::build_preparation_request(requests().front()));
        body(*offline, *batched);
    }

private:
    engine::core::BackendType backend_ = engine::core::BackendType::Cuda;
    int device_ = 0;
    std::string num_step_;
    rt::AudioBuffer reference_audio_;
    std::string reference_text_;
    std::unique_ptr<rt::ModelRegistry> registry_;
    std::unique_ptr<rt::ILoadedVoiceModel> model_;
};

int failures = 0;

// exact=true requires bit-identical audio. exact=false requires the same
// length and finite samples and only reports the difference.
void check_identical(
    const std::string & check,
    size_t request,
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    bool exact = true) {
    float max_abs = 0.0F;
    bool finite = true;
    const size_t n = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < n; ++i) {
        max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
        finite = finite && std::isfinite(actual[i]);
    }
    const bool same_length = actual.size() == expected.size();
    const bool ok = same_length && finite && (!exact || max_abs == 0.0F);
    std::cout << (ok ? "PASS " : "FAIL ") << check << " request " << request;
    if (!ok || max_abs != 0.0F) {
        std::cout << ": samples " << actual.size() << " vs " << expected.size() << ", max |diff| " << max_abs;
        if (!finite) {
            std::cout << ", non-finite samples";
        }
    }
    if (!ok) {
        ++failures;
    }
    std::cout << "\n";
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        Runner runner(argc, argv);
        const auto requests = runner.requests();
        const auto sampled = runner.sampled();

        const auto run_batch = [&](rt::IBatchedOfflineVoiceTaskSession & batched,
                                   const std::vector<rt::TaskRequest> & batch) {
            auto results = batched.run_batch(batch);
            if (results.size() != batch.size()) {
                throw std::runtime_error("run_batch() returned " + std::to_string(results.size()) + " results for " +
                                         std::to_string(batch.size()) + " requests");
            }
            return results;
        };

        // Reference: run() of every request on the default session.
        std::vector<std::vector<float>> expected;
        runner.with_session({}, [&](rt::IOfflineVoiceTaskSession & offline, rt::IBatchedOfflineVoiceTaskSession & batched) {
            for (const auto & request : requests) {
                expected.push_back(samples_of(offline.run(request)));
            }
            for (size_t i = 0; i < requests.size(); ++i) {
                const auto one = run_batch(batched, {requests[i]});
                check_identical("batch_of_one", i, samples_of(one.front()), expected[i]);
            }
            const auto first = run_batch(batched, requests);
            for (size_t i = 0; i < requests.size(); ++i) {
                if (!sampled[i]) {
                    check_identical("batched_vs_run", i, samples_of(first[i]), expected[i], runner.exact_batching());
                }
            }
            const auto second = run_batch(batched, requests);
            for (size_t i = 0; i < requests.size(); ++i) {
                check_identical("batched_repeat", i, samples_of(second[i]), samples_of(first[i]));
            }
        });

        struct Switch {
            std::string check;
            std::string option;
            bool sampled_too = false;
        };
        const std::vector<Switch> switches = {
            {"device_scoring_off", "omnivoice.device_scoring", true},
            {"packed_projections_off", "omnivoice.packed_projections", false},
        };
        for (const auto & entry : switches) {
            runner.with_session({{entry.option, "false"}}, [&](rt::IOfflineVoiceTaskSession & offline, rt::IBatchedOfflineVoiceTaskSession &) {
                for (size_t i = 0; i < requests.size(); ++i) {
                    if (entry.sampled_too || !sampled[i]) {
                        check_identical(entry.check, i, samples_of(offline.run(requests[i])), expected[i]);
                    }
                }
            });
        }

        if (failures > 0) {
            std::cout << failures << " check(s) failed\n";
            return 1;
        }
        std::cout << "all checks passed\n";
        return 0;
    } catch (const std::invalid_argument & e) {
        std::cerr << e.what() << "\n";
        return 2;
    } catch (const std::exception & e) {
        std::cerr << "omnivoice_batch_parity_test failed: " << e.what() << "\n";
        return 1;
    }
}
