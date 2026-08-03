#import "VibeVoiceEngine.h"

#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr NSString *kVibeVoiceErrorDomain = @"VibeVoiceDemo.Engine";

void quietGgmlLogCallback(enum ggml_log_level level, const char *text, void *userData) {
    (void)userData;
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
        std::fflush(stderr);
    }
}

void installQuietGgmlLogging() {
    static std::once_flag once;
    std::call_once(once, [] {
        ggml_log_set(quietGgmlLogCallback, nullptr);
    });
}

NSError *makeError(NSString *message) {
    return [NSError errorWithDomain:static_cast<NSString *>(kVibeVoiceErrorDomain)
                               code:1
                           userInfo:@{NSLocalizedDescriptionKey: message}];
}

NSError *makeError(const std::exception &error) {
    return makeError([NSString stringWithUTF8String:error.what()]);
}

std::string pathString(NSURL *url) {
    const char *path = url.path.UTF8String;
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("empty file path");
    }
    return std::string(path);
}

std::string joinURLPaths(NSArray<NSURL *> *urls) {
    std::ostringstream joined;
    for (NSUInteger index = 0; index < urls.count; ++index) {
        if (index > 0) {
            joined << ",";
        }
        joined << pathString(urls[index]);
    }
    return joined.str();
}

std::string randomSeedString() {
    uint32_t seed = 0;
    arc4random_buf(&seed, sizeof(seed));
    return std::to_string(seed);
}

}  // namespace

@implementation VibeVoiceGenerationResult

- (instancetype)initWithWavURL:(NSURL *)wavURL
             generationSeconds:(double)generationSeconds
                  audioSeconds:(double)audioSeconds
                           rtf:(double)rtf {
    self = [super init];
    if (self) {
        _wavURL = wavURL;
        _generationSeconds = generationSeconds;
        _audioSeconds = audioSeconds;
        _rtf = rtf;
    }
    return self;
}

@end

@implementation VibeVoiceEngine {
    dispatch_queue_t _queue;
    std::mutex _sessionMutex;
    std::string _loadedModelPath;
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> _model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> _sessionBase;
}

+ (instancetype)sharedEngine {
    static VibeVoiceEngine *engine = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        engine = [[VibeVoiceEngine alloc] init];
    });
    return engine;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        installQuietGgmlLogging();
        _queue = dispatch_queue_create("audio.cpp.vibevoice.generation", DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)generateWithText:(NSString *)text
                modelURL:(NSURL *)modelURL
          voiceStateURLs:(NSArray<NSURL *> *)voiceStateURLs
          inferenceSteps:(NSInteger)inferenceSteps
              completion:(VibeVoiceGenerationCompletion)completion {
    NSString *textCopy = [text copy];
    NSURL *modelCopy = [modelURL copy];
    NSArray<NSURL *> *statesCopy = [voiceStateURLs copy];
    NSInteger stepsCopy = inferenceSteps;
    VibeVoiceGenerationCompletion completionCopy = [completion copy];

    dispatch_async(_queue, ^{
        @autoreleasepool {
            NSError *error = nil;
            VibeVoiceGenerationResult *result = nil;
            try {
                result = [self generateSynchronouslyWithText:textCopy
                                                    modelURL:modelCopy
                                              voiceStateURLs:statesCopy
                                              inferenceSteps:stepsCopy];
            } catch (const std::exception &ex) {
                error = makeError(ex);
            } catch (...) {
                error = makeError(@"unknown C++ exception");
            }

            dispatch_async(dispatch_get_main_queue(), ^{
                completionCopy(result, error);
            });
        }
    });
}

- (void)preloadWithModelURL:(NSURL *)modelURL
             voiceStateURLs:(NSArray<NSURL *> *)voiceStateURLs
                 completion:(void (^)(NSError *_Nullable error))completion {
    NSURL *modelCopy = [modelURL copy];
    NSArray<NSURL *> *statesCopy = [voiceStateURLs copy];
    void (^completionCopy)(NSError *_Nullable) = [completion copy];

    dispatch_async(_queue, ^{
        @autoreleasepool {
            NSError *error = nil;
            try {
                const std::string modelPath = pathString(modelCopy);
                if (statesCopy.count == 0) {
                    throw std::runtime_error("at least one voice state is required");
                }
                engine::runtime::TaskRequest request;
                request.text_input = engine::runtime::Transcript{"Speaker 1: preload", ""};
                request.options["voice_state_files"] = joinURLPaths(statesCopy);
                {
                    std::lock_guard<std::mutex> lock(_sessionMutex);
                    auto *session = [self lockedSessionForModelPath:modelPath];
                    session->prepare(engine::runtime::build_preparation_request(request));
                }
            } catch (const std::exception &ex) {
                error = makeError(ex);
            } catch (...) {
                error = makeError(@"unknown C++ exception");
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                completionCopy(error);
            });
        }
    });
}

- (VibeVoiceGenerationResult *)generateSynchronouslyWithText:(NSString *)text
                                                    modelURL:(NSURL *)modelURL
                                              voiceStateURLs:(NSArray<NSURL *> *)voiceStateURLs
                                              inferenceSteps:(NSInteger)inferenceSteps {
    if (text.length == 0) {
        throw std::runtime_error("text is empty");
    }
    if (voiceStateURLs.count == 0) {
        throw std::runtime_error("at least one voice state is required");
    }
    if (inferenceSteps <= 0) {
        throw std::runtime_error("inference steps must be positive");
    }

    const std::string modelPath = pathString(modelURL);

    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{std::string(text.UTF8String), ""};
    request.options["voice_state_files"] = joinURLPaths(voiceStateURLs);
    request.options["num_inference_steps"] = std::to_string(static_cast<int64_t>(inferenceSteps));
    request.options["guidance_scale"] = "1.3";
    request.options["seed"] = randomSeedString();

    engine::runtime::TaskResult taskResult;
    const auto start = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(_sessionMutex);
        engine::runtime::IOfflineVoiceTaskSession *session = [self lockedSessionForModelPath:modelPath];
        NSLog(@"VibeVoice: prepare");
        engine::runtime::SessionPreparationRequest preparation =
            engine::runtime::build_preparation_request(request);
        session->prepare(preparation);
        NSLog(@"VibeVoice: run");
        taskResult = session->run(request);
        NSLog(@"VibeVoice: run finished");
    }
    const auto end = std::chrono::steady_clock::now();

    if (!taskResult.audio_output.has_value() || taskResult.audio_output->samples.empty()) {
        throw std::runtime_error("VibeVoice produced no audio");
    }

    const auto &audio = *taskResult.audio_output;
    const int sampleRate = std::max(audio.sample_rate, 1);
    const int channels = std::max(audio.channels, 1);
    const double generationSeconds =
        std::chrono::duration<double>(end - start).count();
    const double audioSeconds =
        static_cast<double>(audio.samples.size()) /
        static_cast<double>(sampleRate * channels);
    const double rtf = audioSeconds > 0.0 ? generationSeconds / audioSeconds : 0.0;

    NSString *fileName = [NSString stringWithFormat:@"vibevoice-%@.wav", NSUUID.UUID.UUIDString];
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:fileName];
    engine::audio::write_pcm16_wav(
        std::filesystem::path(path.UTF8String),
        audio.sample_rate,
        audio.channels,
        audio.samples);

    NSURL *wavURL = [NSURL fileURLWithPath:path];
    return [[VibeVoiceGenerationResult alloc] initWithWavURL:wavURL
                                           generationSeconds:generationSeconds
                                                audioSeconds:audioSeconds
                                                         rtf:rtf];
}

- (engine::runtime::IOfflineVoiceTaskSession *)lockedSessionForModelPath:(const std::string &)modelPath {
    if (_sessionBase != nullptr && _loadedModelPath == modelPath) {
        auto *session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(_sessionBase.get());
        if (session == nullptr) {
            throw std::runtime_error("cached VibeVoice session is not an offline TTS session");
        }
        return session;
    }

    engine::runtime::ModelLoadRequest loadRequest;
    loadRequest.model_path = std::filesystem::path(modelPath);
    loadRequest.family_hint = std::string("vibevoice");

    NSLog(@"VibeVoice: loading model from %@", [NSString stringWithUTF8String:modelPath.c_str()]);
    auto registry = engine::runtime::make_default_registry();
    _model = registry.load(loadRequest);

    engine::runtime::SessionOptions options;
    options.backend.type = engine::core::BackendType::Metal;
    options.backend.device = 0;
    options.backend.threads = static_cast<int>(std::max<NSInteger>(
        1,
        NSProcessInfo.processInfo.processorCount - 2));

    NSLog(@"VibeVoice: creating Metal session");
    _sessionBase = _model->create_task_session(
        {engine::runtime::VoiceTaskKind::Tts, engine::runtime::RunMode::Offline},
        options);
    auto *session = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(_sessionBase.get());
    if (session == nullptr) {
        throw std::runtime_error("VibeVoice did not create an offline TTS session");
    }
    _loadedModelPath = modelPath;
    return session;
}

@end
