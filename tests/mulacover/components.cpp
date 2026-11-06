// Manual component parity driver; deliberately not registered with CTest.
// Usage: mulacover_components MODE --reference FILE.safetensors
//   --backend cpu|cuda|vulkan --log FILE [mode options below]
// style: --source QWEN_CHECKPOINT --tokenizer DIR --tags TEXT
// prompt: --source MODEL.gguf --lyrics FILE --tags TEXT [--weight-type f32]
// frontend: [--input WAV] --bpm BPM
// transcriber: --source CHECKPOINT [--input WAV]
// chord: --source FOLD0 --source1 FOLD1 ... --source4 FOLD4 --vocabulary FILE
// midi: --melody FILE --chord FILE [--drums FILE]
// conditioning: --source CHECKPOINT --first-layer N --train-tag true|false --cfg true|false
// Common: --metadata JSON, --atol FLOAT (1e-4), --rtol FLOAT (1e-3).
// Metadata may supply tags, bpm, sample_rate; explicit tags/bpm take precedence.
// Fixture contract (logical row-major shapes, floating expected tensors):
// style: style [1024]. frontend: waveform [samples] or [frames,channels],
//   sample_rate I64 scalar (or metadata), cqt [frames,252], tuning scalar.
// transcriber: same waveform inputs; notes [n,5], in native output order,
//   columns onset_seconds,pitch,duration_seconds,program,is_drum.
// chord: cqt [frames,252], chord_times [n,2] in seconds; metadata chord_labels [n].
// midi: pianoroll [steps,256], drum_pianoroll [steps,128], chord [steps,12].
// conditioning: same symbolic tensors and style [1024]; for N=0..7 require
//   layerN.style_scale, layerN.style_shift, layerN.key, layerN.value,
//   layerN.gate_key and layerN.gate_value.
// Prepared tensor shapes match the runtime's logical shapes, including CFG batch.
// No generation loop or per-token probes are used.
#include "engine/models/mulacover/style_encoder.h"
#include "engine/models/mulacover/frontend.h"
#include "engine/models/mulacover/transcriber.h"
#include "engine/models/mulacover/chord.h"
#include "engine/models/mulacover/symbolic.h"
#include "engine/models/mulacover/conditioning.h"
#include "engine/models/mulacover/assets.h"
#include "engine/models/mulacover/ar.h"
#include "engine/models/mulacover/prompt.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/filesystem.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// One comparison for all numerical outputs. Shape and every element must pass;
// cosine is diagnostic only (also handles empty and all-zero tensors).
void compare(const engine::assets::TensorSource & reference, const std::string & name,
             const std::vector<float> & actual, const std::vector<int64_t> & shape,
             double atol, double rtol) {
    const auto expected = reference.require_f32(name, shape);
    if (actual.size() != expected.size()) throw std::runtime_error(name + " size mismatch");
    double max_abs = 0, squared_error = 0, dot = 0, aa = 0, bb = 0;
    size_t failed = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error(name + " non-finite value");
        const double error = std::abs(a - b);
        max_abs = std::max(max_abs, error);
        squared_error += error * error;
        dot += a * b; aa += a * a; bb += b * b;
        failed += error > atol + rtol * std::abs(b);
    }
    const double cosine = aa > 0 && bb > 0 ? dot / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    const double rmse = actual.empty() ? 0 : std::sqrt(squared_error / actual.size());
    std::ostringstream metrics;
    metrics << std::setprecision(12) << "validation.tensor=" << name << " max_abs_error=" << max_abs
            << " rmse=" << rmse << " cosine=" << cosine << " failed_elements=" << failed;
    engine::debug::log_message(metrics.str());
    if (failed) throw std::runtime_error(name + " parity failed");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto started = std::chrono::steady_clock::now();
    bool logging = false;
    try {
        if (argc < 2) throw std::runtime_error("Usage: mulacover_components style|frontend|transcriber|chord|midi|conditioning --reference FILE --backend cpu|cuda|vulkan --log FILE [mode options; see source header]");
        const std::string mode = argv[1];
        const std::vector<std::string> modes{"style", "prompt", "frontend", "transcriber", "chord", "midi", "conditioning"};
        if (std::find(modes.begin(), modes.end(), mode) == modes.end()) throw std::runtime_error("Unknown mode: " + mode);
        const std::vector<std::string> flags{"--reference", "--backend", "--log", "--source", "--source1", "--source2", "--source3", "--source4", "--tokenizer", "--tags", "--lyrics", "--input", "--bpm", "--metadata", "--atol", "--rtol", "--vocabulary", "--melody", "--chord", "--drums", "--first-layer", "--train-tag", "--cfg", "--weight-type"};
        std::map<std::string, std::string> args;
        for (int i = 2; i < argc; i += 2) {
            const std::string flag = argv[i];
            if (std::find(flags.begin(), flags.end(), flag) == flags.end() || i + 1 == argc)
                throw std::runtime_error("Unknown option or missing value: " + flag);
            if (!args.emplace(flag, argv[i + 1]).second) throw std::runtime_error("Duplicate option: " + flag);
        }
        const auto required = [&](const std::string & flag) -> const std::string & {
            const auto it = args.find(flag);
            if (it == args.end() || (it->second.empty() && flag != "--tags")) throw std::runtime_error("Required option: " + flag);
            return it->second;
        };
        engine::debug::configure_logging({true, required("--log")});
        logging = true;
        engine::core::BackendConfig backend;
        const auto & backend_name = required("--backend");
        if (backend_name == "cpu") backend.type = engine::core::BackendType::Cpu;
        else if (backend_name == "cuda") backend.type = engine::core::BackendType::Cuda;
        else if (backend_name == "vulkan") backend.type = engine::core::BackendType::Vulkan;
        else throw std::runtime_error("Unknown backend: " + backend_name);
        backend.threads = 8;
        const double atol = args.count("--atol") ? std::stod(args.at("--atol")) : 1e-4;
        const double rtol = args.count("--rtol") ? std::stod(args.at("--rtol")) : 1e-3;
        if (!std::isfinite(atol) || !std::isfinite(rtol) || atol < 0 || rtol < 0) throw std::runtime_error("Invalid tolerances");
        const auto reference = engine::assets::open_tensor_source(required("--reference"));
        const auto metadata = args.count("--metadata") ? engine::io::json::parse_file(args.at("--metadata")) : engine::io::json::Value::make_object({});
        namespace mc = engine::models::mulacover;
        engine::debug::log_message("validation.mode=" + mode + " validation.backend=" + backend_name);
        const auto storage = engine::assets::parse_tensor_storage_type(
            args.count("--weight-type") ? args.at("--weight-type") : "native");
        if (mode == "style") {
            const std::string tags = args.count("--tags") ? args.at("--tags") : metadata.require("tags").as_string();
            mc::StyleEncoder encoder(engine::assets::open_tensor_source(required("--source")), required("--tokenizer"), backend,
                storage);
            const auto embedding = encoder.encode(tags);
            compare(*reference, "style", embedding, {1024}, atol, rtol);
            double norm = 0;
            for (float v : embedding) norm += static_cast<double>(v) * v;
            const double expected_norm = tags.empty() ? 0 : 1;
            if (std::abs(std::sqrt(norm) - expected_norm) > 1e-4) throw std::runtime_error("Style normalization invariant failed");
        } else if (mode == "prompt") {
            const auto assets = mc::load_mulacover_assets(required("--source"));
            engine::core::ExecutionContext execution(backend);
            mc::AutoregressiveRuntime runtime(execution, *assets, storage);
            mc::PromptEncoder encoder(execution, runtime.prompt_weights(), assets->mula_weights,
                assets->resources.require_file("tokenizer_json"), assets->generation_config);
            const auto shape = reference->require_metadata("prompt_embeddings").shape;
            if (shape.size() != 3 || (shape[0] != 1 && shape[0] != 2))
                throw std::runtime_error("prompt_embeddings must be [1|2,steps,3072]");
            const auto encoded = encoder.encode(engine::io::read_text_file(required("--lyrics")),
                required("--tags"), reference->require_f32("style", {1024}), shape[0] == 2 ? 1.5F : 1.0F);
            compare(*reference, "prompt_embeddings", encoded.values,
                {encoded.batch_size, encoded.steps, encoded.dims}, atol, rtol);
        } else if (mode == "frontend" || mode == "transcriber") {
            engine::audio::WavData audio;
            if (args.count("--input")) {
                audio = engine::audio::read_wav_f32(std::filesystem::path(args.at("--input")));
            } else {
                const auto shape = reference->require_metadata("waveform").shape;
                if (shape.size() != 1 && shape.size() != 2) throw std::runtime_error("waveform must be [samples] or [frames,channels]");
                audio.samples = reference->require_f32("waveform");
                audio.channels = shape.size() == 1 ? 1 : static_cast<int>(shape[1]);
                audio.sample_rate = static_cast<int>(metadata.find("sample_rate") ? metadata.require("sample_rate").as_i64() : reference->require_i64_scalar("sample_rate"));
            }
            if (mode == "frontend") {
                const double bpm = args.count("--bpm") ? std::stod(args.at("--bpm")) : metadata.require("bpm").as_number();
                mc::ChordFrontend frontend;
                const auto & output = frontend.forward(audio.samples, audio.sample_rate, audio.channels, bpm);
                compare(*reference, "cqt", output.cqt, {output.frames, 252}, atol, rtol);
                compare(*reference, "tuning", {static_cast<float>(output.tuning)}, {}, atol, rtol);
                if (reference->has_tensor("bpm")) {
                    compare(*reference, "bpm", {static_cast<float>(output.bpm)}, {}, atol, rtol);
                }
            } else {
                mc::Transcriber transcriber(engine::assets::open_tensor_source(required("--source")), backend);
                const auto notes = transcriber.transcribe(engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels), audio.sample_rate);
                std::vector<float> values;
                for (const auto & note : notes) {
                    values.insert(values.end(), {static_cast<float>(note.onset_seconds), static_cast<float>(note.pitch), static_cast<float>(note.duration), static_cast<float>(note.program), note.is_drum ? 1.f : 0.f});
                }
                compare(*reference, "notes", values, {static_cast<int64_t>(notes.size()), 5}, atol, rtol);
                const auto expected = reference->require_f32("notes");
                for (size_t i = 0; i < notes.size(); ++i)
                    for (size_t col : {size_t(1), size_t(3), size_t(4)})
                        if (values[i * 5 + col] != expected[i * 5 + col]) throw std::runtime_error("Note pitch/program/is_drum mismatch");
            }
        } else if (mode == "chord") {
            std::array<std::shared_ptr<const engine::assets::TensorSource>, 5> sources;
            for (size_t i = 0; i < sources.size(); ++i) sources[i] = engine::assets::open_tensor_source(required(i == 0 ? "--source" : "--source" + std::to_string(i)));
            const auto cqt_shape = reference->require_metadata("cqt").shape;
            if (cqt_shape.size() != 2 || cqt_shape[1] != 252) throw std::runtime_error("cqt must be [frames,252]");
            engine::core::ExecutionContext execution(backend);
            mc::ChordNet model(execution, sources, required("--vocabulary"));
            const auto events = model.forward(reference->require_f32("cqt"));
            const auto & labels = metadata.require("chord_labels").as_array();
            if (events.size() != labels.size()) throw std::runtime_error("Chord event count mismatch");
            std::vector<float> times;
            for (size_t i = 0; i < events.size(); ++i) {
                if (events[i].label != labels[i].as_string()) throw std::runtime_error("Chord label mismatch at " + std::to_string(i));
                times.insert(times.end(), {static_cast<float>(events[i].onset_seconds), static_cast<float>(events[i].offset_seconds)});
            }
            compare(*reference, "chord_times", times, {static_cast<int64_t>(events.size()), 2}, atol, rtol);
        } else if (mode == "midi") {
            const auto symbolic = mc::load_symbolic_midi(required("--melody"), required("--chord"), args.count("--drums") ? std::optional<std::filesystem::path>(args.at("--drums")) : std::nullopt);
            compare(*reference, "pianoroll", symbolic.pianoroll, {symbolic.length, 256}, 0, 0);
            compare(*reference, "drum_pianoroll", symbolic.drum_pianoroll, {symbolic.length, 128}, 0, 0);
            compare(*reference, "chord", symbolic.chord, {symbolic.length, 12}, 0, 0);
        } else {
            const auto boolean = [&](const std::string & flag) {
                const auto & value = required(flag);
                if (value != "true" && value != "false") throw std::runtime_error(flag + " must be true|false");
                return value == "true";
            };
            const auto shape = reference->require_metadata("pianoroll").shape;
            if (shape.size() != 2 || shape[1] != 256) throw std::runtime_error("pianoroll must be [steps,256]");
            mc::SymbolicInputs symbolic;
            symbolic.length = shape[0];
            symbolic.pianoroll = reference->require_f32("pianoroll", {symbolic.length, 256});
            symbolic.drum_pianoroll = reference->require_f32("drum_pianoroll", {symbolic.length, 128});
            symbolic.chord = reference->require_f32("chord", {symbolic.length, 12});
            const auto style = reference->require_f32("style", {1024});
            const bool cfg = boolean("--cfg");
            engine::core::ExecutionContext execution(backend);
            mc::ConditioningRuntime runtime(execution, engine::assets::open_tensor_source(required("--source")), std::stoll(required("--first-layer")), 1024, boolean("--train-tag"), storage,
                storage == engine::assets::TensorStorageType::F32 ? GGML_TYPE_F32 : GGML_TYPE_F16);
            if (!runtime.prepare(symbolic, style, cfg)) throw std::runtime_error("Initial conditioning prepare must create storage");
            const auto revision = runtime.revision();
            const auto * key = runtime.prepared().layers[0].attention.key.tensor;
            if (runtime.prepare(symbolic, style, cfg) || runtime.revision() != revision || runtime.prepared().layers[0].attention.key.tensor != key || runtime.context_length() != symbolic.length || runtime.batch_size() != (cfg ? 2 : 1))
                throw std::runtime_error("Conditioning cache/shape invariant failed");
            for (size_t i = 0; i < 8; ++i) {
                const auto & layer = runtime.prepared().layers[i];
                const std::array<engine::core::TensorValue, 6> tensors{layer.style_scale, layer.style_shift, layer.attention.key, layer.attention.value, layer.gate.key, layer.gate.value};
                const std::array<std::string, 6> names{"style_scale", "style_shift", "key", "value", "gate_key", "gate_value"};
                for (size_t j = 0; j < tensors.size(); ++j) {
                    const auto name = "layer" + std::to_string(i) + "." + names[j];
                    const auto & tensor = tensors[j];
                    if (!tensor.valid() || tensor.tensor->op != GGML_OP_NONE || tensor.tensor->view_src) throw std::runtime_error(name + " must be a detached tensor");
                    const std::vector<int64_t> dims(tensor.shape.dims.begin(), tensor.shape.dims.begin() + tensor.shape.rank);
                    compare(*reference, name, engine::core::read_tensor_f32(tensor.tensor), dims, atol, rtol);
                }
            }
        }
        engine::debug::log_message("validation.status=passed");
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return 0;
    } catch (const std::exception & error) {
        if (logging) {
            engine::debug::log_message(engine::debug::LogLevel::Error, "validation", error.what());
            engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
