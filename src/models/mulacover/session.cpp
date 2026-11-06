#include "engine/models/mulacover/session.h"

#include "engine/models/mulacover/assets.h"
#include "engine/models/mulacover/chord.h"
#include "engine/models/mulacover/frontend.h"
#include "engine/models/mulacover/prompt.h"
#include "engine/models/mulacover/style_encoder.h"
#include "engine/models/mulacover/symbolic.h"
#include "engine/models/mulacover/transcriber.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"
#include "engine/framework/codecs/heartcodec_runtime.h"
#include "engine/models/mulacover/ar.h"

#include <chrono>
#include <array>
#include <cmath>
#include <map>
#include <string_view>
#include <utility>

namespace engine::models::mulacover {
namespace {

using Clock = std::chrono::steady_clock;

class MuLaCoverSession final : public runtime::RuntimeSessionBase,
                              public runtime::IOfflineVoiceTaskSession {
public:
    MuLaCoverSession(runtime::TaskSpec task, runtime::SessionOptions options,
                    std::shared_ptr<const MuLaCoverAssets> assets,
                    std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)),
          contract_(std::move(contract)) {
        runtime::validate_spec_backed_session_options(options, *contract_, "mulacover", "MuLaCover");
        if (task.task != runtime::VoiceTaskKind::AudioGeneration || task.mode != runtime::RunMode::Offline) {
            throw runtime::InvalidRequestError("MuLaCover requires an offline music session");
        }
        mula_ = std::make_unique<AutoregressiveRuntime>(execution_context(), *assets_);
        prompt_ = std::make_unique<PromptEncoder>(execution_context(), mula_->prompt_weights(), assets_->mula_weights,
            assets_->resources.require_file("tokenizer_json"), assets_->generation_config);
        codec_ = std::make_unique<codecs::HeartCodecRuntime>(assets_->codec_weights,
            execution_context(), assets_->codec_config,
            codecs::HeartCodecRuntimeOptions{8ull << 20, 32ull << 20, 8ull << 20, 32ull << 20});
    }

    std::string family() const override { return "mulacover"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }
    void prepare(const runtime::SessionPreparationRequest &) override { mark_prepared(); }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("MuLaCover run");
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays,
                                                     *contract_, "MuLaCover");
        const auto start = Clock::now();
        const auto melody = runtime::find_option(request.options, {"melody_file"});
        const auto chord = runtime::find_option(request.options, {"chord_file"});
        const auto drums = runtime::find_option(request.options, {"drum_file"});
        if (request.audio_input && (melody || chord || drums)) {
            throw runtime::InvalidRequestError("MuLaCover audio input and MIDI file options are mutually exclusive");
        }
        if (!request.audio_input && (!melody || !chord)) {
            throw runtime::InvalidRequestError("MuLaCover requires reference audio or melody_file and chord_file");
        }
        if (request.voice || !request.input_artifacts.empty()) {
            throw runtime::InvalidRequestError("MuLaCover does not consume voices or input artifacts");
        }
        const auto lyrics = runtime::find_option(request.options, {"lyrics"}).value_or(
            request.text_input ? request.text_input->text : "");
        const auto tags = PromptEncoder::normalize_tags(
            runtime::find_option(request.options, {"tags"}).value_or(""));
        GenerationOptions generation;
        const auto duration_seconds = runtime::parse_positive_finite_float_option(
            request.options, {"duration_sec"}).value_or(300.0F);
        generation.temperature = runtime::parse_positive_finite_float_option(
            request.options, {"temperature"}).value_or(1.0F);
        generation.guidance_scale = runtime::parse_positive_finite_float_option(
            request.options, {"guidance_scale"}).value_or(1.5F);
        generation.top_k = runtime::parse_positive_i64_option(request.options, {"top_k"}, 250);
        codecs::HeartCodecDecodeOptions codec_options;
        codec_options.num_inference_steps = runtime::parse_positive_i64_option(
            request.options, {"num_inference_steps"}, 10);
        codec_options.guidance_scale = runtime::parse_positive_finite_float_option(
            request.options, {"codec_guidance_scale"}).value_or(1.25F);
        const double frame_budget = std::floor(static_cast<double>(duration_seconds) * 12.5);
        if (frame_budget < 1 || frame_budget > assets_->mula_config.max_seq_len) {
            throw runtime::CapacityError("MuLaCover duration exceeds the model context");
        }
        const auto frames_limit = static_cast<int64_t>(frame_budget);
        const auto seed = runtime::parse_u32_option(request.options, {"seed"}).value_or(1234);
        const double bpm = runtime::parse_finite_float_option(request.options, {"bpm"}).value_or(0.0F);
        if (bpm < 0) throw runtime::InvalidRequestError("MuLaCover bpm must be zero (automatic) or positive");
        auto symbolic = request.audio_input ? transcribe_reference(*request.audio_input, bpm) :
            load_symbolic_midi(*melody, *chord,
                drums ? std::optional<std::filesystem::path>(*drums) : std::nullopt);
        std::vector<float> style(static_cast<size_t>(assets_->mula_config.qwen_dim), 0.0F);
        if (!tags.empty()) {
            if (!style_) {
                style_ = std::make_unique<StyleEncoder>(assets_->qwen_weights,
                    assets_->resources.require_file("qwen_tokenizer").parent_path(), options().backend);
            }
            style = style_->encode(tags);
        }
        auto prompt = prompt_->encode(lyrics, tags, style, generation.guidance_scale);
        if (prompt.steps + frames_limit > assets_->mula_config.max_seq_len) {
            throw runtime::CapacityError("MuLaCover prompt and requested audio exceed the 8192-token context");
        }
        const int64_t expected_batch = generation.guidance_scale > 1.0F ? 2 : 1;
        mula_->prepare(symbolic, style, expected_batch == 2);
        const auto ar_start = Clock::now();
        auto frames = mula_->generate(prompt, generation, seed, frames_limit);
        if (frames.frames == 0) {
            throw std::runtime_error("MuLaCover emitted EOS before generating any audio");
        }
        const auto codec_start = Clock::now();
        const auto codec_rng_stride = sampling::torch_cuda_tensor_iterator_offset_blocks(
            static_cast<uint64_t>(static_cast<int64_t>(codec_options.chunk_duration_seconds * 25.0F) *
                assets_->codec_config.out_channels), frames.rng_policy);
        auto decoded = codec_->detokenize_codes(frames.codes, frames.frames, frames.codebooks,
            codec_options, seed, frames.rng_offset_blocks, codec_rng_stride);
        codec_->release_graph_workspaces();
        runtime::TaskResult result;
        result.audio_output = runtime::AudioBuffer{assets_->codec_config.sample_rate,
            static_cast<int>(decoded.channels), std::move(decoded.values)};
        debug::timing_log_scalar("mulacover.prepare_ms", debug::elapsed_ms(start, ar_start));
        debug::timing_log_scalar("mulacover.ar_ms", debug::elapsed_ms(ar_start, codec_start));
        debug::timing_log_scalar("mulacover.codec_ms", debug::elapsed_ms(codec_start, Clock::now()));
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(start, Clock::now()));
        return result;
    }

private:
    SymbolicInputs transcribe_reference(const runtime::AudioBuffer & input_audio, double bpm) {
        const auto mono = audio::mixdown_interleaved_to_mono_average(input_audio.samples, input_audio.channels);
        if (!chord_frontend_) chord_frontend_ = std::make_unique<ChordFrontend>();
        const auto & features = chord_frontend_->forward(mono, input_audio.sample_rate, 1, bpm);
        if (!transcriber_) transcriber_ = std::make_unique<Transcriber>(
            assets_->transcriber_weights, options().backend);
        if (!chord_net_) chord_net_ = std::make_unique<ChordNet>(execution_context(),
            assets_->chord_weights, assets_->resources.require_file("chord_vocabulary"));
        const auto notes = transcriber_->transcribe(mono, input_audio.sample_rate);
        const auto chord_grid = chord_net_->forward(features.cqt, features.bpm);
        std::vector<SymbolicNote> melody, drums, chords;
        for (const auto & note : notes) {
            if (!note.is_drum && note.program != 100) continue;
            const auto onset = static_cast<int64_t>(std::floor(note.onset_seconds * features.bpm / 15.0 + 0.5));
            const auto end = static_cast<int64_t>(std::floor(
                (note.onset_seconds + note.duration) * features.bpm / 15.0 + 0.5));
            (note.is_drum ? drums : melody).push_back({onset, note.pitch, end - onset});
        }
        // symbolic.py _chord_pitches: exact matches first, then the ORIGINAL
        // insertion order of quality prefixes (maj before maj7, min before min7).
        static const std::map<std::string, int> roots{
            {"C",0}, {"C#",1}, {"Db",1}, {"D",2}, {"D#",3}, {"Eb",3}, {"E",4},
            {"F",5}, {"F#",6}, {"Gb",6}, {"G",7}, {"G#",8}, {"Ab",8},
            {"A",9}, {"A#",10}, {"Bb",10}, {"B",11}};
        static const std::array<std::pair<std::string_view, std::vector<int>>, 10> qualities{{
            {"maj",{0,4,7}}, {"min",{0,3,7}}, {"dim",{0,3,6}}, {"aug",{0,4,8}},
            {"7",{0,4,7,10}}, {"maj7",{0,4,7,11}}, {"min7",{0,3,7,10}},
            {"sus2",{0,2,7}}, {"sus4",{0,5,7}}, {"5",{0,7}}}};
        for (const auto & event : chord_grid) {
            if (event.offset <= event.onset) continue;
            const auto colon = event.label.find(':');
            if (colon == std::string::npos) continue;
            const auto root = roots.find(event.label.substr(0, colon));
            if (root == roots.end()) continue;
            const std::string_view quality(event.label.data() + colon + 1, event.label.size() - colon - 1);
            const std::vector<int> * intervals = nullptr;
            for (const auto & candidate : qualities) if (quality == candidate.first) {
                intervals = &candidate.second;
                break;
            }
            if (!intervals) for (const auto & candidate : qualities) {
                if (quality.substr(0, candidate.first.size()) == candidate.first) {
                    intervals = &candidate.second;
                    break;
                }
            }
            if (intervals) for (int interval : *intervals) chords.push_back({
                event.onset * 4, (root->second + interval) % 12, (event.offset - event.onset) * 4});
        }
        return build_symbolic_inputs(melody, chords, drums);
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const MuLaCoverAssets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    std::unique_ptr<AutoregressiveRuntime> mula_;
    std::unique_ptr<PromptEncoder> prompt_;
    std::unique_ptr<StyleEncoder> style_;
    std::unique_ptr<codecs::HeartCodecRuntime> codec_;
    std::unique_ptr<Transcriber> transcriber_;
    std::unique_ptr<ChordFrontend> chord_frontend_;
    std::unique_ptr<ChordNet> chord_net_;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_mulacover_loader() {
    runtime::SpecBackedVoiceModelConfig<MuLaCoverAssets> config;
    config.family = "mulacover";
    config.load_assets = load_mulacover_assets;
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const MuLaCoverAssets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        return std::make_unique<MuLaCoverSession>(task, options, std::move(assets), std::move(contract));
    };
    return runtime::make_spec_backed_voice_loader(std::move(config));
}

}  // namespace engine::models::mulacover
