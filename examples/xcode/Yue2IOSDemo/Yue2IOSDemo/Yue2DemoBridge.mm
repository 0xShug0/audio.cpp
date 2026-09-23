#import "Yue2DemoBridge.h"

#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

NSString * const Yue2DemoErrorDomain = @"Yue2DemoErrorDomain";
using Clock = std::chrono::steady_clock;

std::string to_cpp(NSString *value) {
    return value == nil ? std::string() : std::string(value.UTF8String);
}

NSError *make_error(const std::exception &ex) {
    return [NSError errorWithDomain:Yue2DemoErrorDomain
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithUTF8String:ex.what()]}];
}

engine::core::BackendType parse_backend(NSString *value) {
    const auto backend = to_cpp(value);
    if (backend == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (backend == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (backend == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + backend);
}

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

void write_audio_buffer(const engine::runtime::AudioBuffer &audio, NSString *path) {
    if (audio.samples.empty()) {
        throw std::runtime_error("Yue2 did not produce audio");
    }
    const std::filesystem::path output(to_cpp(path));
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    engine::audio::write_pcm16_wav(output, audio.sample_rate, audio.channels, audio.samples);
}

double audio_duration_seconds(const engine::runtime::AudioBuffer &audio) {
    if (audio.sample_rate <= 0 || audio.channels <= 0) {
        return 0.0;
    }
    return static_cast<double>(audio.samples.size()) /
        static_cast<double>(audio.sample_rate * audio.channels);
}

}  // namespace

@implementation Yue2GenerationResult
@end

@interface Yue2DemoBridge () {
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> _model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> _session;
    engine::runtime::IOfflineVoiceTaskSession *_offline;
}
@end

@implementation Yue2DemoBridge

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                                   backend:(NSString *)backend
                                    device:(int)device
                                   threads:(int)threads
                                     error:(NSError **)error {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    try {
        const auto root = std::filesystem::path(to_cpp(modelPath));
        engine::runtime::ModelLoadRequest loadRequest;
        loadRequest.model_path = root;
        loadRequest.family_hint = "yue2";

        engine::runtime::TaskSpec taskSpec;
        taskSpec.mode = engine::runtime::RunMode::Offline;
        taskSpec.task = engine::runtime::VoiceTaskKind::AudioGeneration;

        engine::runtime::SessionOptions sessionOptions;
        sessionOptions.backend.type = parse_backend(backend);
        sessionOptions.backend.device = device;
        sessionOptions.backend.threads = threads;
        sessionOptions.options["yue2.ios_mode"] = "true";
        constexpr const char *kIOSMainGguf = "yue2-3b-ios-q4_0.gguf";
        constexpr const char *kVaeGguf = "yue2-vae-f16.gguf";
        if (!std::filesystem::exists(root / kIOSMainGguf)) {
            throw std::runtime_error("Yue2 iOS demo requires bundled yue2-3b-ios-q4_0.gguf");
        }
        if (!std::filesystem::exists(root / kVaeGguf)) {
            throw std::runtime_error("Yue2 iOS demo requires bundled yue2-vae-f16.gguf");
        }
        sessionOptions.options["yue2.model_gguf"] = kIOSMainGguf;
        sessionOptions.options["yue2.vae_gguf"] = kVaeGguf;
        sessionOptions.options["yue2.model_weight_context_mb"] = "3072";
        sessionOptions.options["yue2.vae_weight_context_mb"] = "512";

        auto registry = engine::runtime::make_default_registry();
        _model = registry.load(loadRequest);
        _session = _model->create_task_session(taskSpec, sessionOptions);
        _offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(_session.get());
        if (_offline == nullptr) {
            throw std::runtime_error("Yue2 session does not support offline execution");
        }
    } catch (const std::exception &ex) {
        if (error != nil) {
            *error = make_error(ex);
        }
        return nil;
    }

    return self;
}

- (nullable Yue2GenerationResult *)generateWithStyle:(NSString *)style
                                             lyrics:(NSString *)lyrics
                                                abc:(NSString *)abc
                                            cotMode:(NSString *)cotMode
                                         outputPath:(NSString *)outputPath
                                               seed:(uint64_t)seed
                                     inferenceSteps:(int64_t)inferenceSteps
                                  semanticMaxTokens:(int64_t)semanticMaxTokens
                                              error:(NSError **)error {
    try {
        engine::runtime::TaskRequest request;
        request.text_input = engine::runtime::Transcript{to_cpp(lyrics), ""};
        request.options["style"] = to_cpp(style);
        request.options["lyrics"] = to_cpp(lyrics);
        const auto abcText = to_cpp(abc);
        const auto cot = to_cpp(cotMode);
        request.options["cot"] = cot;
        if (!abcText.empty() && cot != "off") {
            request.options["abc"] = abcText;
        }
        request.options["seed"] = std::to_string(seed);
        request.options["num_inference_steps"] = std::to_string(inferenceSteps);
        request.options["semantic_max_tokens"] = std::to_string(semanticMaxTokens);

        _session->prepare(engine::runtime::build_preparation_request(request));
        const auto start = Clock::now();
        const auto result = _offline->run(request);
        const auto end = Clock::now();
        if (!result.audio_output.has_value()) {
            throw std::runtime_error("Yue2 did not return audio");
        }

        write_audio_buffer(*result.audio_output, outputPath);

        Yue2GenerationResult *summary = [Yue2GenerationResult new];
        summary.outputPath = outputPath;
        summary.generationSeconds = elapsed_seconds(start, end);
        summary.audioSeconds = audio_duration_seconds(*result.audio_output);
        summary.rtf = summary.audioSeconds > 0.0 ? summary.generationSeconds / summary.audioSeconds : 0.0;
        return summary;
    } catch (const std::exception &ex) {
        if (error != nil) {
            *error = make_error(ex);
        }
        return nil;
    }
}

@end
