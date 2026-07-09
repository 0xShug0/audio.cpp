#include "args.h"
#include "batch.h"
#include "request.h"
#include "../streaming/streaming.h"
#include "../workflow/execution.h"
#include "../workflow/file_sink.h"
#include "../workflow/pipeline.h"
#include "../workflow/workflow.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/registry.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void print_task_list_help() {
    std::cout
        << "audiocpp_cli --task <task> --family <family> --model <path> --backend <backend> [options]\n"
        << "  Global:\n"
        << "    --task vad|asr|diar|sep|gen|tts|clon|vc|s2s|align|vdes|spk|svc\n"
        << "    --family <name>\n"
        << "    --model <path>\n"
        << "    --backend cpu|cuda|vulkan|metal|best\n"
        << "    --mode offline|streaming  default offline\n"
        << "    --device <n>\n"
        << "    --threads <n>  Backend and OpenMP worker threads, default 4\n"
        << "    --registry-config <path>\n"
        << "    --config <id>\n"
        << "    --weight <id>\n"
        << "    --log  Stream framework progress and timing logs to stdout\n"
        << "    --log-file <path>  Stream framework progress and timing logs to a file\n"
        << "    --load-option key=value\n"
        << "    --session-option key=value\n"
        << "    --request-option key=value\n"
        << "  Batch:\n"
        << "    --request-sequence <json>  Run JSON requests in one offline session\n"
        << "    --batch-text-file <txt>  Run one offline request per non-empty line\n"
        << "    --batch-text-dir <dir>  Run one offline request per .txt, .md, or .json file\n"
        << "    --batch-audio-dir <dir>  Run one offline request per .wav file\n"
        << "    --batch-audio-role audio|voice_ref|source_audio|target_voice|prosody_ref|style_ref\n"
        << "    --batch-merge-audio none|concat\n"
        << "    --batch-manifest-out <json>\n"
        << "    Shared CLI inputs and options are defaults for --batch-text-file, --batch-text-dir, and --batch-audio-dir\n"
        << "  Pipelines:\n"
        << "    --pipeline <json>  Run a JSON app workflow instead of a raw task\n"
        << "    --list-pipelines\n"
        << "    --workflow-input key=value  Override a top-level workflow input\n"
        << "    --audio-converter <cmd>  Converter executable for workflow media import, default ffmpeg\n"
        << "  Task routing and media roles:\n"
        << "    --task-route <name>  User-facing route, e.g. text2music, zero_shot_tts, style_preserved_vc\n"
        << "    --source-audio <wav>  Source audio path for models that take path-based source refs\n"
        << "    --target-voice <wav>  Target/timbre voice path for path-based voice conversion\n"
        << "    --prosody-ref <wav>\n"
        << "    --style-ref <wav>\n"
        << "    --target-text <text>\n"
        << "    --style-ref-text <text>\n"
        << "    --use-prosody-code true|false\n"
        << "    --predict-target-prosody true|false\n"
        << "    --use-pitch-shift true|false\n"
        << "    --source-shift-steps <n>\n"
        << "    --prosody-shift-steps <n>\n"
        << "    --style-shift-steps <n>\n"
        << "    --target-duration-seconds <float>\n"
        << "    --reference-duration-seconds <float>\n"
        << "    --lyrics <text>\n"
        << "    --track-name <name>\n"
        << "    --speaker <name>\n"
        << "    --duration-seconds <float>\n"
        << "    --repaint-start <float>\n"
        << "    --repaint-end <float>\n"
        << "    --repaint-mode <name>\n"
        << "    --repaint-strength <float>\n"
        << "  Common generation:\n"
        << "    --seed <n>\n"
        << "    --max-tokens <n>\n"
        << "    --max-steps <n>\n"
        << "    --temperature <float>\n"
        << "    --top-k <n>\n"
        << "    --top-p <float>\n"
        << "    --repetition-penalty <float>\n"
        << "    --do-sample true|false\n"
        << "    --num-beams <n>\n"
        << "    --guidance-scale <float>\n"
        << "    --num-inference-steps <n>\n"
        << "    --text-chunk-size <chars>\n"
        << "  Inputs:\n"
        << "    --audio <wav>\n"
        << "    --text <text>\n"
        << "    --max-text-length <chars>  Maximum input text length for models that enforce it\n"
        << "    --language <code>\n"
        << "    --voice-id <id>\n"
        << "    --voice-ref <wav>\n"
        << "    --reference-text <text>  Reference transcript for voice-clone models such as Qwen3 TTS\n"
        << "    --instruct <text>  Voice-design instruction for models such as Qwen3 TTS\n"
        << "    --style-language <code>\n"
        << "    --emotion <name>\n"
        << "    --speaking-rate <float>\n"
        << "    --pitch-shift <float>\n"
        << "    --energy-scale <float>\n"
        << "    --style-tag key=value\n"
        << "  Outputs:\n"
        << "    --out <wav>\n"
        << "    --out-dir <dir>  Write named multi-audio outputs or batch request outputs\n"
        << "    --text-out <txt>\n"
        << "    --segments-out <json>\n"
        << "    --vad-chunks-out <json>  Write offline VAD-based chunk windows\n"
        << "    --vad-chunk-max-seconds <float>  Maximum VAD chunk length, default 45\n"
        << "    --vad-chunk-merge-gap-seconds <float>  Merge nearby VAD spans, default 0.5\n"
        << "    --vad-chunk-padding-seconds <float>  Pad each VAD span before chunking, default 0.25\n"
        << "    --turns-out <json>\n"
        << "    --words-out <json>\n"
        << "    --voice-state-out <safetensors>  Export PocketTTS voice state from --voice-ref\n"
        << "  Streaming:\n"
        << "    --mode streaming uses the selected model's default streaming policy\n"
        << "  Utility:\n"
        << "    --inspect\n"
        << "    --list-loaders\n"
        << "\n"
        << "  Tasks:\n"
        << "    vad    voice activity detection\n"
        << "    asr    automatic speech recognition\n"
        << "    diar   speaker diarization\n"
        << "    sep    source separation\n"
        << "    gen    text/audio conditioned music or sound generation\n"
        << "    tts    text to speech\n"
        << "    clon   voice cloning\n"
        << "    vc     voice conversion\n"
        << "    s2s    speech to speech\n"
        << "    align  forced alignment\n"
        << "    vdes   voice design\n"
        << "    spk    speaker embedding/recognition\n"
        << "    svc    singing voice conversion\n";
}

void print_option_group(const char * title, const std::vector<engine::runtime::CliOptionInfo> & options) {
    if (options.empty()) {
        return;
    }
    std::cout << "  " << title << ":\n";
    for (const auto & option : options) {
        std::cout << "    " << option.name;
        if (!option.value_name.empty()) {
            std::cout << " <" << option.value_name << ">";
        }
        if (!option.description.empty()) {
            std::cout << "  " << option.description;
        }
        std::cout << "\n";
    }
}

bool model_supports_task(const engine::runtime::ModelInspection & inspection, engine::runtime::VoiceTaskKind task) {
    return std::any_of(
        inspection.capabilities.supported_tasks.begin(),
        inspection.capabilities.supported_tasks.end(),
        [&](const engine::runtime::TaskCapability & capability) {
            return capability.task == task;
        });
}

void print_model_common_options(const engine::runtime::ModelInspection & inspection) {
    std::cout << "  Available input options:\n";
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::AudioGeneration)) {
        std::cout
            << "    --text <text>\n"
            << "    --audio <wav>\n"
            << "    --batch-text-file <txt>\n"
            << "    --batch-text-dir <dir>\n"
            << "    --language <code>\n"
            << "    --duration-seconds <float>\n"
            << "    --guidance-scale <float>\n"
            << "    --num-inference-steps <n>\n"
            << "    --seed <n>\n"
            << "    --max-tokens <n>\n"
            << "    --temperature <float>\n"
            << "    --top-k <n>\n"
            << "    --top-p <float>\n"
            << "    --do-sample true|false\n";
    }
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::Tts) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::VoiceDesign) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::VoiceCloning)) {
        std::cout
            << "    --text <text>\n"
            << "    --batch-text-file <txt>\n"
            << "    --batch-text-dir <dir>\n"
            << "    --language <code>\n"
            << "    --voice-ref <wav>\n"
            << "    --voice-id <id>\n"
            << "    --reference-text <text>\n"
            << "    --instruct <text>\n"
            << "    --speaker <name>\n"
            << "    --seed <n>\n"
            << "    --max-tokens <n>\n"
            << "    --temperature <float>\n"
            << "    --top-k <n>\n"
            << "    --top-p <float>\n"
            << "    --repetition-penalty <float>\n"
            << "    --do-sample true|false\n";
    }
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::VoiceConversion) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::SpeechToSpeech) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::Svc)) {
        std::cout
            << "    --audio <wav>\n"
            << "    --batch-audio-dir <dir>\n"
            << "    --voice-ref <wav>\n"
            << "    --task-route <name>\n"
            << "    --source-audio <wav>\n"
            << "    --target-voice <wav>\n"
            << "    --prosody-ref <wav>\n"
            << "    --style-ref <wav>\n"
            << "    --target-text <text>\n"
            << "    --style-ref-text <text>\n"
            << "    --use-prosody-code true|false\n"
            << "    --predict-target-prosody true|false\n"
            << "    --use-pitch-shift true|false\n"
            << "    --source-shift-steps <n>\n"
            << "    --prosody-shift-steps <n>\n"
            << "    --style-shift-steps <n>\n"
            << "    --target-duration-seconds <float>\n"
            << "    --reference-duration-seconds <float>\n"
            << "    --num-inference-steps <n>\n"
            << "    --text-chunk-size <chars>\n"
            << "    --seed <n>\n";
    }
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::Asr) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::Vad) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::Diarization) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::SourceSeparation) ||
        model_supports_task(inspection, engine::runtime::VoiceTaskKind::SpeakerRecognition)) {
        std::cout
            << "    --audio <wav>\n"
            << "    --batch-audio-dir <dir>\n";
    }
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::Asr)) {
        std::cout
            << "    --audio-chunk-seconds <float>\n"
            << "    --audio-chunk-mode auto|fixed|vad|none\n";
    }
    if (model_supports_task(inspection, engine::runtime::VoiceTaskKind::Alignment)) {
        std::cout
            << "    --audio <wav>\n"
            << "    --batch-audio-dir <dir>\n"
            << "    --text <text>\n"
            << "    --language <code>\n"
            << "    --text-chunk-size <chars>\n"
            << "    --audio-chunk-seconds <float>\n"
            << "    --audio-chunk-mode auto|fixed|none\n";
    }
}

void print_model_help(const engine::runtime::ModelInspection & inspection) {
    std::cout
        << "family=" << inspection.metadata.family << "\n"
        << "variant=" << inspection.metadata.variant << "\n"
        << "model_root=" << inspection.model_root.string() << "\n"
        << "description=" << inspection.metadata.description << "\n"
        << "  Supported tasks:\n";
    for (const auto & capability : inspection.capabilities.supported_tasks) {
        std::cout << "    --task " << engine::runtime::to_string(capability.task) << " --mode ";
        for (size_t i = 0; i < capability.modes.size(); ++i) {
            if (i != 0) {
                std::cout << "|";
            }
            std::cout << engine::runtime::to_string(capability.modes[i]);
        }
        std::cout << "\n";
    }
    print_model_common_options(inspection);
    print_option_group("Model request options", inspection.cli.request_options);
    print_option_group("Model session options", inspection.cli.session_options);
    print_option_group("Model load options", inspection.cli.load_options);
    std::cout << "  Common output options:\n"
              << "    --out <wav>\n"
              << "    --out-dir <dir>\n"
              << "    --text-out <txt>\n"
              << "    --segments-out <json>\n"
              << "    --vad-chunks-out <json>\n"
              << "    --vad-chunk-max-seconds <float>\n"
              << "    --vad-chunk-merge-gap-seconds <float>\n"
              << "    --vad-chunk-padding-seconds <float>\n"
              << "    --turns-out <json>\n"
              << "    --words-out <json>\n";
}

void write_text_output(
    const engine::runtime::TaskResult & result,
    const std::filesystem::path & path,
    const std::string & label) {
    if (!result.text_output.has_value()) {
        throw std::runtime_error("--text-out was requested but the task result has no text output");
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open text output: " + path.string());
    }
    output << result.text_output->text << "\n";
    std::cout << label << "=" << path.string() << "\n";
}

void print_task_help(const engine::runtime::ModelRegistry & registry, const std::string & task_name) {
    const auto task = engine::runtime::parse_voice_task_kind(task_name);
    std::cout
        << "task=" << engine::runtime::to_string(task) << "\n"
        << "  Select a model to see model-owned options:\n"
        << "    audiocpp_cli --task " << engine::runtime::to_string(task)
            << " --family <family> --model <path> --backend <backend> --help\n"
        << "  Registered families:\n";
    for (const auto & family : registry.families()) {
        std::cout << "    " << family << "\n";
    }
}

void print_inspection(const engine::runtime::ModelInspection & inspection) {
    std::cout << "family=" << inspection.metadata.family << "\n";
    std::cout << "variant=" << inspection.metadata.variant << "\n";
    std::cout << "model_root=" << inspection.model_root.string() << "\n";
    std::cout << "supported_tasks=" << inspection.capabilities.supported_tasks.size() << "\n";
    for (const auto & capability : inspection.capabilities.supported_tasks) {
        std::cout << "task=" << engine::runtime::to_string(capability.task) << " modes=";
        for (size_t i = 0; i < capability.modes.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << engine::runtime::to_string(capability.modes[i]);
        }
        std::cout << "\n";
    }
    std::cout << "supports_speaker_reference=" << (inspection.capabilities.supports_speaker_reference ? "true" : "false") << "\n";
    std::cout << "supports_style_condition=" << (inspection.capabilities.supports_style_condition ? "true" : "false") << "\n";
    std::cout << "supports_timestamps=" << (inspection.capabilities.supports_timestamps ? "true" : "false") << "\n";
    std::cout << "languages=";
    for (size_t i = 0; i < inspection.capabilities.languages.size(); ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << inspection.capabilities.languages[i];
    }
    std::cout << "\n";
    std::cout << "configs=" << inspection.discovered_configs.size() << "\n";
    for (const auto & config : inspection.discovered_configs) {
        std::cout << "config=" << config.id << ":" << config.path.string() << "\n";
    }
    std::cout << "weights=" << inspection.discovered_weights.size() << "\n";
    for (const auto & weight : inspection.discovered_weights) {
        std::cout << "weight=" << weight.id << ":" << weight.path.string() << "\n";
    }
}

bool has_vad_chunk_option(int argc, char ** argv) {
    return minitts::cli::optional_path_arg(argc, argv, "--vad-chunks-out").has_value() ||
        minitts::cli::find_arg(argc, argv, "--vad-chunk-max-seconds").has_value() ||
        minitts::cli::find_arg(argc, argv, "--vad-chunk-merge-gap-seconds").has_value() ||
        minitts::cli::find_arg(argc, argv, "--vad-chunk-padding-seconds").has_value();
}

int64_t seconds_to_samples(float seconds, int sample_rate, const std::string & name) {
    if (seconds < 0.0F) {
        throw std::runtime_error(name + " must be non-negative");
    }
    return static_cast<int64_t>(std::llround(static_cast<double>(seconds) * sample_rate));
}

engine::audio::VadAudioChunkOptions vad_chunk_options_from_cli(
    int argc,
    char ** argv,
    int sample_rate) {
    if (sample_rate <= 0) {
        throw std::runtime_error("VAD chunk planning requires a positive audio sample rate");
    }
    const float max_seconds = minitts::cli::parse_optional_float_arg(argc, argv, "--vad-chunk-max-seconds").value_or(45.0F);
    const float merge_gap_seconds = minitts::cli::parse_optional_float_arg(argc, argv, "--vad-chunk-merge-gap-seconds").value_or(0.5F);
    const float padding_seconds = minitts::cli::parse_optional_float_arg(argc, argv, "--vad-chunk-padding-seconds").value_or(0.25F);
    auto options = engine::audio::VadAudioChunkOptions{
        seconds_to_samples(max_seconds, sample_rate, "--vad-chunk-max-seconds"),
        seconds_to_samples(merge_gap_seconds, sample_rate, "--vad-chunk-merge-gap-seconds"),
        seconds_to_samples(padding_seconds, sample_rate, "--vad-chunk-padding-seconds"),
    };
    if (options.max_chunk_samples <= 0) {
        throw std::runtime_error("--vad-chunk-max-seconds must be positive");
    }
    return options;
}

std::string vad_chunks_to_json(const std::vector<engine::runtime::TimeSpan> & chunks) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "{\"index\":" << i
            << ",\"start_sample\":" << chunks[i].start_sample
            << ",\"end_sample\":" << chunks[i].end_sample
            << "}";
    }
    out << "]";
    return out.str();
}

void write_vad_chunks_output(
    const engine::runtime::TaskResult & result,
    const engine::runtime::AudioBuffer & audio,
    const std::filesystem::path & path,
    const engine::audio::VadAudioChunkOptions & options) {
    if (audio.channels <= 0) {
        throw std::runtime_error("VAD chunk planning requires positive audio channels");
    }
    if (audio.samples.size() % static_cast<size_t>(audio.channels) != 0) {
        throw std::runtime_error("VAD chunk planning requires audio samples divisible by channel count");
    }
    const int64_t audio_frames = static_cast<int64_t>(audio.samples.size() / static_cast<size_t>(audio.channels));
    const auto chunks = engine::audio::plan_vad_audio_chunks(result.speech_segments, audio_frames, options);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream(path) << vad_chunks_to_json(chunks);
    std::cout << "vad_chunks_out=" << path.string() << "\n";
}

void run_streaming(
    int argc,
    char ** argv,
    engine::runtime::IStreamingVoiceTaskSession & streaming,
    engine::runtime::IVoiceTaskSession & session,
    engine::runtime::TaskRequest & request) {
    const auto out_dir = minitts::cli::optional_path_arg(argc, argv, "--out-dir");
    const auto result = minitts::app::run_streaming_task(
        streaming,
        request,
        [&](const engine::runtime::StreamEvent & event) {
            if (event.partial_text.has_value()) {
                std::cout << "partial_text=" << event.partial_text->text << "\n";
            }
            engine::runtime::TaskResult event_result;
            event_result.audio_output = event.audio_output;
            event_result.named_audio_outputs = event.named_audio_outputs;
            event_result.speaker_turns = event.speaker_turns;
            event_result.word_timestamps = event.word_timestamps;
            event_result.output_artifacts = event.output_artifacts;
            minitts::app::emit_task_result(
                event_result,
                std::nullopt,
                out_dir,
                out_dir,
                std::nullopt,
                std::nullopt,
                std::nullopt);
            for (const auto & activity : event.voice_activity) {
                std::cout << "event=";
                switch (activity.kind) {
                case engine::runtime::VoiceActivityEvent::Kind::SpeechStart:
                    std::cout << "speech_start";
                    break;
                case engine::runtime::VoiceActivityEvent::Kind::SpeechEnd:
                    std::cout << "speech_end";
                    break;
                case engine::runtime::VoiceActivityEvent::Kind::SpeechSegment:
                    std::cout << "speech_segment";
                    break;
                }
                std::cout << " sample=" << activity.sample << " probability=" << activity.probability << "\n";
            }
        });
    std::cout << "family=" << session.family() << "\n";
    std::cout << "task=" << engine::runtime::to_string(session.task_kind()) << "\n";
    std::cout << "mode=" << engine::runtime::to_string(session.run_mode()) << "\n";
    minitts::app::emit_task_result(
        result,
        minitts::cli::optional_path_arg(argc, argv, "--out"),
        std::nullopt,
        minitts::cli::optional_path_arg(argc, argv, "--out-dir"),
        minitts::cli::optional_path_arg(argc, argv, "--segments-out"),
        minitts::cli::optional_path_arg(argc, argv, "--turns-out"),
        minitts::cli::optional_path_arg(argc, argv, "--words-out"));
    if (const auto text_out = minitts::cli::optional_path_arg(argc, argv, "--text-out")) {
        write_text_output(result, *text_out, "text_out");
    }
}

}  // namespace

#ifdef _WIN32
namespace {

std::string wide_arg_to_utf8(const wchar_t * arg) {
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, arg, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 0) {
        throw std::runtime_error("failed to convert Windows command-line argument to UTF-8");
    }
    std::vector<char> buffer(static_cast<size_t>(size), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, arg, -1, buffer.data(), size, nullptr, nullptr);
    if (written != size) {
        throw std::runtime_error("failed to convert Windows command-line argument to UTF-8");
    }
    return std::string(buffer.data());
}

std::vector<std::string> wide_args_to_utf8(int argc, wchar_t ** wargv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.push_back(wide_arg_to_utf8(wargv[i]));
    }
    return args;
}

}  // namespace
#endif

int audiocpp_cli_main(int argc, char ** argv) {
    try {
        using namespace minitts::cli;

        const auto log_file = find_arg(argc, argv, "--log-file");
        engine::debug::configure_logging(engine::debug::LoggingConfig{
            has_arg(argc, argv, "--log") || log_file.has_value(),
            log_file,
        });

        const auto registry_config = find_arg(argc, argv, "--registry-config");
        auto registry = engine::runtime::make_default_registry(
            registry_config ? std::optional<std::filesystem::path>(std::filesystem::path(*registry_config)) : std::nullopt);
        auto pipeline_registry = minitts::app::make_default_pipeline_registry();
        const bool help_requested = has_arg(argc, argv, "--help");
        if (has_arg(argc, argv, "--list-pipelines")) {
            const auto ids = pipeline_registry.ids();
            std::cout << "registered_pipelines=" << ids.size() << "\n";
            for (const auto & id : ids) {
                std::cout << id << "\n";
            }
            return 0;
        }
        if (has_arg(argc, argv, "--list-loaders")) {
            const auto families = registry.families();
            std::cout << "registered_loaders=" << registry.size() << "\n";
            for (const auto & family : families) {
                std::cout << family << "\n";
            }
            return 0;
        }

        const auto model_arg = find_arg(argc, argv, "--model");
        const auto pipeline_arg = find_arg(argc, argv, "--pipeline");
        if (pipeline_arg.has_value()) {
            const int threads = parse_int_arg(argc, argv, "--threads", 4);
            if (threads <= 0) {
                throw std::runtime_error("--threads must be positive");
            }
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            engine::core::BackendConfig backend;
            backend.type = parse_backend(find_arg(argc, argv, "--backend").value_or("cpu"));
            backend.device = parse_int_arg(argc, argv, "--device", 0);
            backend.threads = threads;
            minitts::app::run_json_workflow(
                registry,
                minitts::app::WorkflowRunOptions{
                    std::filesystem::path(*pipeline_arg),
                    optional_path_arg(argc, argv, "--out-dir").value_or(std::filesystem::path("workflow_outputs")),
                    backend,
                    optional_path_arg(argc, argv, "--out"),
                    collect_key_value_args(argc, argv, "--load-option"),
                    collect_key_value_args(argc, argv, "--session-option"),
                    collect_key_value_args(argc, argv, "--workflow-input"),
                    find_arg(argc, argv, "--audio-converter").value_or("ffmpeg"),
                });
            return 0;
        }
        if (help_requested && !model_arg.has_value()) {
            if (const auto task = find_arg(argc, argv, "--task")) {
                print_task_help(registry, *task);
            } else {
                print_task_list_help();
            }
            return 0;
        }
        if (!model_arg) {
            throw std::runtime_error("missing required --model argument");
        }

        engine::runtime::ModelLoadRequest load_request;
        load_request.model_path = std::filesystem::path(*model_arg);
        if (const auto family = find_arg(argc, argv, "--family")) {
            load_request.family_hint = *family;
        }
        if (const auto config = find_arg(argc, argv, "--config")) {
            load_request.config_id = *config;
        }
        if (const auto weight = find_arg(argc, argv, "--weight")) {
            load_request.weight_id = *weight;
        }
        load_request.options = collect_key_value_args(argc, argv, "--load-option");

        if (help_requested) {
            print_model_help(registry.inspect(load_request));
            return 0;
        }

        if (has_arg(argc, argv, "--inspect")) {
            print_inspection(registry.inspect(load_request));
            return 0;
        }

        const auto task_name = find_arg(argc, argv, "--task");
        if (!task_name.has_value()) {
            throw std::runtime_error("missing required --task argument");
        }
        const auto mode_name = find_arg(argc, argv, "--mode").value_or("offline");
        const engine::runtime::TaskSpec task_spec{
            engine::runtime::parse_voice_task_kind(*task_name),
            engine::runtime::parse_run_mode(mode_name),
        };

        engine::runtime::SessionOptions session_options;
        session_options.backend.type = parse_backend(find_arg(argc, argv, "--backend").value_or("cpu"));
        session_options.backend.device = parse_int_arg(argc, argv, "--device", 0);
        const int threads = parse_int_arg(argc, argv, "--threads", 4);
        if (threads <= 0) {
            throw std::runtime_error("--threads must be positive");
        }
        session_options.backend.threads = threads;
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        session_options.options = collect_key_value_args(argc, argv, "--session-option");
        auto model = registry.load(load_request);
        auto session = model->create_task_session(task_spec, session_options);
        const auto voice_state_out = optional_path_arg(argc, argv, "--voice-state-out");
        const auto text_out = optional_path_arg(argc, argv, "--text-out");
        const auto words_out = optional_path_arg(argc, argv, "--words-out");
        const auto vad_chunks_out = optional_path_arg(argc, argv, "--vad-chunks-out");

        if (has_batch_input(argc, argv)) {
            if (task_spec.mode != engine::runtime::RunMode::Offline) {
                throw std::runtime_error("batch inputs require offline mode");
            }
            if (voice_state_out.has_value()) {
                throw std::runtime_error("--voice-state-out is not supported with batch inputs");
            }
            if (has_vad_chunk_option(argc, argv)) {
                throw std::runtime_error("VAD chunk output options are not supported with batch inputs");
            }
            const auto merge_mode = minitts::app::parse_audio_merge_mode(
                find_arg(argc, argv, "--batch-merge-audio").value_or("none"));
            if (optional_path_arg(argc, argv, "--out").has_value() &&
                merge_mode == minitts::app::AudioMergeMode::None) {
                throw std::runtime_error("batch --out requires --batch-merge-audio concat");
            }
            auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
            if (offline == nullptr) {
                throw std::runtime_error("selected task session does not support offline execution");
            }
            engine::runtime::TaskRequest base_request =
                optional_path_arg(argc, argv, "--request-sequence").has_value()
                    ? engine::runtime::TaskRequest{}
                    : build_request_from_cli(argc, argv);
            if (words_out.has_value()) {
                base_request.options["return_timestamps"] = "true";
            }
            auto batch_request = build_batch_request_from_cli(argc, argv, base_request);
            if (words_out.has_value()) {
                for (auto & item : batch_request.requests) {
                    item.request.options["return_timestamps"] = "true";
                }
            }
            const minitts::app::FileOutputPolicy output_policy{
                optional_path_arg(argc, argv, "--out"),
                optional_path_arg(argc, argv, "--out-dir"),
                optional_path_arg(argc, argv, "--segments-out"),
                optional_path_arg(argc, argv, "--turns-out"),
                words_out,
                optional_path_arg(argc, argv, "--batch-manifest-out"),
            };
            std::cout << "family=" << session->family() << "\n";
            std::cout << "task=" << engine::runtime::to_string(session->task_kind()) << "\n";
            std::cout << "mode=" << engine::runtime::to_string(session->run_mode()) << "\n";
            const auto batch_result = minitts::app::run_offline_batch(
                *session,
                *offline,
                batch_request,
                merge_mode,
                [&](size_t index, const minitts::app::AppRequestResult & item) {
                    minitts::app::emit_batch_item_result(index, item, output_policy);
                    if (text_out.has_value()) {
                        const auto request_id = minitts::app::safe_output_name(item.id);
                        const auto path = text_out->parent_path() /
                                          (text_out->stem().string() + "_" + request_id + text_out->extension().string());
                        write_text_output(item.result, path, "text_out[" + request_id + "]");
                    }
                });
            minitts::app::emit_batch_summary(batch_result, output_policy);
            return 0;
        }

        auto request = build_request_from_cli(argc, argv);
        if (has_vad_chunk_option(argc, argv)) {
            if (task_spec.mode != engine::runtime::RunMode::Offline ||
                task_spec.task != engine::runtime::VoiceTaskKind::Vad) {
                throw std::runtime_error("VAD chunk output options require offline --task vad");
            }
            if (!vad_chunks_out.has_value()) {
                throw std::runtime_error("VAD chunk options require --vad-chunks-out");
            }
            if (!request.audio_input.has_value()) {
                throw std::runtime_error("VAD chunk output requires --audio");
            }
        }
        if (words_out.has_value()) {
            request.options["return_timestamps"] = "true";
        }
        if (voice_state_out.has_value()) {
            if (session->family() != "pocket_tts") {
                throw std::runtime_error("--voice-state-out is only supported by PocketTTS");
            }
            if (task_spec.task != engine::runtime::VoiceTaskKind::Tts) {
                throw std::runtime_error("--voice-state-out requires --task tts");
            }
            request.options["pocket_tts.export_voice_state_path"] = voice_state_out->string();
        }
        session->prepare(engine::runtime::build_preparation_request(request));
        if (voice_state_out.has_value()) {
            std::cout << "family=" << session->family() << "\n";
            std::cout << "voice_state_out=" << voice_state_out->string() << "\n";
            return 0;
        }

        if (task_spec.mode == engine::runtime::RunMode::Offline) {
            auto * offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(session.get());
            if (offline == nullptr) {
                throw std::runtime_error("selected task session does not support offline execution");
            }
            const auto result = offline->run(request);
            std::cout << "family=" << session->family() << "\n";
            std::cout << "task=" << engine::runtime::to_string(session->task_kind()) << "\n";
            std::cout << "mode=" << engine::runtime::to_string(session->run_mode()) << "\n";
            minitts::app::emit_task_result(
                result,
                optional_path_arg(argc, argv, "--out"),
                optional_path_arg(argc, argv, "--out-dir"),
                optional_path_arg(argc, argv, "--out-dir"),
                optional_path_arg(argc, argv, "--segments-out"),
                optional_path_arg(argc, argv, "--turns-out"),
                words_out);
            if (vad_chunks_out.has_value()) {
                write_vad_chunks_output(
                    result,
                    *request.audio_input,
                    *vad_chunks_out,
                    vad_chunk_options_from_cli(argc, argv, request.audio_input->sample_rate));
            }
            if (text_out.has_value()) {
                write_text_output(result, *text_out, "text_out");
            }
            return 0;
        }

        auto * streaming = dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(session.get());
        if (streaming == nullptr) {
            throw std::runtime_error("selected task session does not support streaming execution");
        }
        run_streaming(argc, argv, *streaming, *session, request);
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_cli failed: " << ex.what() << "\n";
        return 1;
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t ** wargv) {
    try {
        auto utf8_args = wide_args_to_utf8(argc, wargv);
        std::vector<char *> argv;
        argv.reserve(utf8_args.size() + 1);
        for (auto & arg : utf8_args) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);
        return audiocpp_cli_main(argc, argv.data());
    } catch (const std::exception & ex) {
        std::cerr << "audiocpp_cli failed: " << ex.what() << "\n";
        return 1;
    }
}
#else
int main(int argc, char ** argv) {
    return audiocpp_cli_main(argc, argv);
}
#endif
