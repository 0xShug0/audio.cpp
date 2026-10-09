#include "engine/models/sherpa_kws/session.h"

#include "engine/models/sherpa_kws/assets.h"
#include "engine/models/sherpa_kws/encoder.h"
#include "engine/models/sherpa_kws/frontend.h"
#include "engine/models/sherpa_kws/keyword_decoder.h"
#include "engine/models/sherpa_kws/zipformer.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/io/filesystem.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/framework/runtime/spec_backed_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace engine::models::sherpa_kws {
namespace {

using Assets = SherpaKwsAssets;

runtime::AudioBuffer with_tail_padding(const runtime::AudioBuffer & audio) {
    runtime::AudioBuffer result = audio;
    if (result.sample_rate > 0 && result.channels > 0) {
        result.samples.resize(result.samples.size() +
                                  static_cast<size_t>(std::llround(0.66 * result.sample_rate)) *
                                      static_cast<size_t>(result.channels),
                              0.0F);
    }
    return result;
}

std::vector<std::string> rows(std::string value) {
    std::replace(value.begin(), value.end(), '/', '\n');
    std::vector<std::string> result;
    std::istringstream input(value);
    std::string line;
    while (std::getline(input, line)) {
        const auto first = line.find_first_not_of(" \t\r");
        const auto last = line.find_last_not_of(" \t\r");
        if (first != std::string::npos) result.push_back(line.substr(first, last - first + 1));
    }
    return result;
}

std::vector<KeywordDefinition> parse_keywords(
    const std::string & value, const Assets & assets, float default_score, float default_threshold) {
    std::unordered_map<std::string, int32_t> token_ids;
    for (size_t i = 0; i < assets.tokens.size(); ++i) token_ids.emplace(assets.tokens[i], static_cast<int32_t>(i));
    std::vector<KeywordDefinition> result;
    for (const auto & line : rows(value)) {
        KeywordDefinition keyword;
        keyword.score = default_score;
        keyword.threshold = default_threshold;
        std::istringstream words(line);
        std::string word;
        while (words >> word) {
            if (const auto it = token_ids.find(word); it != token_ids.end()) keyword.tokens.push_back(it->second);
            else if (!word.empty() && word.front() == ':') keyword.score = std::stof(word.substr(1));
            else if (!word.empty() && word.front() == '#') keyword.threshold = std::stof(word.substr(1));
            else if (!word.empty() && word.front() == '@') keyword.phrase = word.substr(1);
            else throw std::runtime_error("unknown sherpa KWS keyword token: " + word);
        }
        if (keyword.phrase.empty()) {
            for (size_t i = 0; i < keyword.tokens.size(); ++i) {
                if (i != 0) keyword.phrase.push_back(' ');
                keyword.phrase += assets.tokens[static_cast<size_t>(keyword.tokens[i])];
            }
        }
        result.push_back(std::move(keyword));
    }
    return result;
}

class Session final : public runtime::RuntimeSessionBase,
                      public runtime::IOfflineVoiceTaskSession,
                      public runtime::IStreamingVoiceTaskSession {
public:
    Session(runtime::TaskSpec task, const runtime::SessionOptions & options,
            std::shared_ptr<const Assets> assets,
            std::shared_ptr<const model_spec::ModelContract> contract)
        : RuntimeSessionBase(options), task_(task), assets_(std::move(assets)), contract_(std::move(contract)),
          decoder_(assets_), subsampling_(assets_, execution_context()), zipformer_(assets_, execution_context()),
          chunk_(static_cast<size_t>(assets_->config.chunk_size * assets_->config.feature_dim), 0.0F) {
        if (task_.task != runtime::VoiceTaskKind::WakeWord) {
            throw std::runtime_error("sherpa KWS supports the wake-word task");
        }
    }

    std::string family() const override { return "sherpa_kws"; }
    runtime::VoiceTaskKind task_kind() const override { return task_.task; }
    runtime::RunMode run_mode() const override { return task_.mode; }

    void prepare(const runtime::SessionPreparationRequest & request) override {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "sherpa KWS");
        mark_prepared();
    }

    runtime::TaskResult run(const runtime::TaskRequest & request) override {
        require_prepared("sherpa KWS run");
        if (!request.audio_input) throw std::runtime_error("sherpa KWS requires audio_input");
        configure(request);
        reset_runtime();
        const auto started = std::chrono::steady_clock::now();
        audio_ = *request.audio_input;
        process(true);
        runtime::TaskResult result;
        result.speech_segments = detections_;
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return result;
    }

    runtime::StreamingPolicy streaming_policy() const override {
        return {runtime::StreamingInputKind::AudioChunks, runtime::StreamingOutputKind::PullEvents, 16000, 1.0};
    }

    void start_stream(const runtime::TaskRequest & request) override {
        require_prepared("sherpa KWS start_stream");
        configure(request);
        reset_runtime();
        stream_started_ = true;
    }

    void reset() override { reset_runtime(); }

    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override {
        require_prepared("sherpa KWS process_audio_chunk");
        if (!stream_started_) throw std::runtime_error("sherpa KWS stream was not started");
        if (chunk.sample_rate != 16000 || chunk.channels != 1 || chunk.start_sample != received_samples_) {
            throw std::runtime_error("sherpa KWS streaming requires contiguous 16000 Hz mono chunks");
        }
        const auto started = std::chrono::steady_clock::now();
        const size_t before = detections_.size();
        audio_.samples.insert(audio_.samples.end(), chunk.samples.begin(), chunk.samples.end());
        received_samples_ += static_cast<int64_t>(chunk.samples.size());
        process(false);
        runtime::StreamEvent event;
        for (size_t i = before; i < detections_.size(); ++i) {
            runtime::VoiceActivityEvent item;
            item.kind = runtime::VoiceActivityEvent::Kind::SpeechSegment;
            item.sample = detections_[i].span.end_sample;
            item.probability = detections_[i].confidence;
            item.segment = detections_[i];
            event.voice_activity.push_back(std::move(item));
        }
        debug::timing_log_scalar("session.wall_ms", debug::elapsed_ms(started));
        return event;
    }

    runtime::TaskResult finish_stream() override {
        if (!stream_started_) throw std::runtime_error("sherpa KWS stream was not started");
        process(true);
        stream_started_ = false;
        runtime::TaskResult result;
        result.speech_segments = detections_;
        return result;
    }

    runtime::TaskResult finalize() override { return finish_stream(); }

private:
    void configure(const runtime::TaskRequest & request) {
        runtime::validate_spec_backed_request_options(request.options, request.option_arrays, *contract_, "sherpa KWS");
        const float score = runtime::parse_finite_float_option(request.options, {"keywords_score"}).value_or(1.0F);
        const float threshold = runtime::parse_finite_float_option(request.options, {"keywords_threshold"}).value_or(0.25F);
        const int paths = static_cast<int>(runtime::parse_i64_option(request.options, {"max_active_paths"}).value_or(4));
        const int blanks = static_cast<int>(runtime::parse_i64_option(request.options, {"num_trailing_blanks"}).value_or(1));
        const std::string text = runtime::find_option(request.options, {"keywords"})
                                     .value_or(io::read_text_file(assets_->resources.require_file("keywords")));
        decoder_.configure(parse_keywords(text, *assets_, score, threshold), paths, blanks);
    }

    void reset_runtime() {
        decoder_.reset();
        subsampling_.reset();
        zipformer_.reset();
        audio_ = {16000, 1, {}};
        detections_.clear();
        processed_features_ = 0;
        received_samples_ = 0;
        stream_started_ = false;
    }

    void process(bool final) {
        if (audio_.samples.empty()) return;
        const auto input = final ? with_tail_padding(audio_) : audio_;
        const auto features = compute_sherpa_fbank(input);
        const int64_t chunk_size = assets_->config.chunk_size;
        const int64_t shift = assets_->config.chunk_shift;
        const int64_t dim = assets_->config.feature_dim;
        while (processed_features_ < features.frames) {
            if (!final && processed_features_ + chunk_size + 1 > features.frames) break;
            std::fill(chunk_.begin(), chunk_.end(), 0.0F);
            const int64_t available = std::min(chunk_size, features.frames - processed_features_);
            std::copy_n(features.values.data() + processed_features_ * dim, available * dim, chunk_.data());
            const auto embedded = subsampling_.encode_subsampled_chunk(chunk_);
            const auto encoded = zipformer_.encode_chunk(embedded.values);
            const int64_t consumed = std::min(shift, features.frames - processed_features_);
            const int64_t valid = std::min<int64_t>(encoded.frames, (consumed + 3) / 4);
            for (const auto & detection : decoder_.append(encoded.values, valid, encoded.channels)) {
                runtime::SpeechSegment segment;
                segment.text = detection.phrase;
                segment.confidence = detection.confidence;
                segment.span.start_sample = detection.start_frame * 640;
                segment.span.end_sample = (detection.end_frame + 1) * 640;
                detections_.push_back(std::move(segment));
            }
            processed_features_ += shift;
        }
    }

    runtime::TaskSpec task_;
    std::shared_ptr<const Assets> assets_;
    std::shared_ptr<const model_spec::ModelContract> contract_;
    KeywordDecoder decoder_;
    SherpaEncoderRuntime subsampling_;
    SherpaZipformerRuntime zipformer_;
    std::vector<float> chunk_;
    runtime::AudioBuffer audio_{16000, 1, {}};
    std::vector<runtime::SpeechSegment> detections_;
    int64_t processed_features_ = 0;
    int64_t received_samples_ = 0;
    bool stream_started_ = false;
};

}  // namespace

std::shared_ptr<runtime::IVoiceModelLoader> make_sherpa_kws_loader() {
    runtime::SpecBackedVoiceModelConfig<Assets> config;
    config.family = "sherpa_kws";
    config.load_assets = [](const std::filesystem::path & path) {
        return load_sherpa_kws_assets(path);
    };
    config.create_session = [](const runtime::TaskSpec & task, const runtime::SessionOptions & options,
                               std::shared_ptr<const Assets> assets,
                               std::shared_ptr<const model_spec::ModelContract> contract) {
        runtime::validate_spec_backed_session_options(options, *contract, "sherpa_kws", "sherpa KWS");
        return std::make_unique<Session>(task, options, std::move(assets), std::move(contract));
    };
    return std::make_shared<runtime::SpecBackedVoiceModelLoader<Assets>>(std::move(config));
}

}  // namespace engine::models::sherpa_kws
