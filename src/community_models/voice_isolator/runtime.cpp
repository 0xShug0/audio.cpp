#include "engine/community_models/voice_isolator/runtime.h"

#include "engine/community_models/voice_isolator/backbone.h"
#include "engine/community_models/voice_isolator/config.h"
#include "engine/community_models/voice_isolator/network.h"
#include "engine/community_models/voice_isolator/spectral.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::voice_isolator {


struct ModelState {
    std::unique_ptr<Network> network;
    Options options;
    int64_t lookahead = 0;
    // Offline renders share one backbone and its graph.
    mutable std::mutex offline_mutex;
    mutable std::unique_ptr<Backbone> offline;
};

namespace {

// Python's floor division, for sample counts before the first frame.
int64_t floor_div(int64_t a, int64_t b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

// The isolated voice of one whole window: the clip plus the synthesized change.
std::vector<float> render_window(const ModelState & model, Backbone & backbone,
    const std::vector<float> & samples) {
    const auto & config = model.network->config();
    const auto & layout = model.network->layout();
    int64_t frames = 0;
    const auto spectra = clip_spectra(config, samples, frames);
    std::vector<float> packed(static_cast<size_t>(layout.bands() * frames * layout.encoder_inputs()));
    pack_bands(layout, spectra.data(), frames, packed.data());
    std::vector<float> raw(static_cast<size_t>(layout.bands() * frames * layout.decoder_outputs()));
    backbone.run(packed.data(), frames, raw.data());
    std::vector<float> change(spectra.size());
    for (int64_t frame = 0; frame < frames; ++frame) {
        spectral_change(layout, config.correction_bound, raw.data(), frames, frame, spectra.data(), frames,
            frame, change.data());
    }
    auto output = clip_waveform(config, change, frames, static_cast<int64_t>(samples.size()));
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] += samples[i];
    }
    return output;
}

std::vector<float> render_offline(const ModelState & model, const std::vector<float> & samples) {
    const auto & config = model.network->config();
    const double chunk_seconds = model.options.chunk_seconds > 0.0 ? model.options.chunk_seconds : config.chunk_seconds;
    const double overlap_seconds = model.options.overlap_seconds > 0.0 ? model.options.overlap_seconds : config.overlap_seconds;
    const auto chunk = static_cast<int64_t>(chunk_seconds * static_cast<double>(config.sample_rate));
    if (chunk < config.hop_length || overlap_seconds < 0.0) {
        throw std::runtime_error("Voice Isolator render window is shorter than one hop");
    }
    const int64_t overlap = std::min(static_cast<int64_t>(overlap_seconds * static_cast<double>(config.sample_rate)), chunk / 2);
    const auto n = static_cast<int64_t>(samples.size());
    std::lock_guard<std::mutex> lock(model.offline_mutex);
    if (!model.offline) {
        model.offline = model.network->offline_backbone();
    }
    if (n <= chunk) {
        return render_window(model, *model.offline, samples);
    }
    std::vector<float> output(static_cast<size_t>(n), 0.0F);
    std::vector<float> weights(static_cast<size_t>(n), 0.0F);
    for (const auto & window : plan_render_windows(n, chunk, overlap)) {
        const std::vector<float> piece(samples.begin() + window.start, samples.begin() + window.start + chunk);
        const audio::AudioChunkSpan span{0, window.start, chunk, window.start, 0};
        audio::overlap_add_planar_chunk(output, weights, render_window(model, *model.offline, piece), 1, n, span,
            window.weights(), audio::AudioChunkCounterMode::SharedAcrossLanes);
    }
    audio::normalize_overlap_added_planar(output, weights, 1, n, audio::AudioChunkCounterMode::SharedAcrossLanes);
    return output;
}

}  // namespace

// The stream: frames are computed as soon as their analysis window is complete,
// synthesized once the frame look-ahead frames later has been computed, and
// samples are released once every frame touching them is synthesized.
struct StreamingSession::State {
    const Config & config;
    const BandLayout & layout;
    int64_t lookahead = 0;
    int64_t block_frames = 1;
    std::unique_ptr<Backbone> backbone;
    StreamSynthesizer synthesizer;
    std::vector<float> audio;
    int64_t audio_base = 0;
    int64_t frames = 0;
    int64_t emitted = 0;
    int64_t released = 0;
    // Spectra [bins][2] and decoder output [bands][outputs] of frames emitted onward.
    std::deque<std::vector<float>> spectra;
    std::deque<std::vector<float>> raw;
    bool closed = false;

    explicit State(const ModelState & model)
        : config(model.network->config()),
          layout(model.network->layout()),
          lookahead(model.lookahead),
          block_frames(model.options.block_frames),
          backbone(model.network->streaming_backbone(model.lookahead, model.options.block_frames)),
          synthesizer(config) {}

    void compute(int64_t first, int64_t count) {
        const auto block = stream_spectra(config, audio.data(), audio_base, static_cast<int64_t>(audio.size()),
            first, count);
        std::vector<float> packed(static_cast<size_t>(layout.bands() * count * layout.encoder_inputs()));
        pack_bands(layout, block.data(), count, packed.data());
        std::vector<float> outputs(static_cast<size_t>(layout.bands() * count * layout.decoder_outputs()));
        backbone->run(packed.data(), count, outputs.data());
        const int64_t bins = config.bins();
        const int64_t width = layout.decoder_outputs();
        for (int64_t t = 0; t < count; ++t) {
            std::vector<float> frame(static_cast<size_t>(bins * 2));
            for (int64_t bin = 0; bin < bins; ++bin) {
                frame[bin * 2] = block[(bin * count + t) * 2];
                frame[bin * 2 + 1] = block[(bin * count + t) * 2 + 1];
            }
            spectra.push_back(std::move(frame));
            std::vector<float> heads(static_cast<size_t>(layout.bands() * width));
            for (int64_t band = 0; band < layout.bands(); ++band) {
                std::copy_n(outputs.begin() + (band * count + t) * width, width, heads.begin() + band * width);
            }
            raw.push_back(std::move(heads));
        }
    }

    // Synthesizes frame `emitted` from the decoder output of frame `source`.
    void synthesize(int64_t source) {
        std::vector<float> change(static_cast<size_t>(config.bins() * 2));
        spectral_change(layout, config.correction_bound, raw[static_cast<size_t>(source - emitted)].data(), 1, 0,
            spectra.front().data(), 1, 0, change.data());
        synthesizer.add(emitted, 1, change.data());
        ++emitted;
        spectra.pop_front();
        raw.pop_front();
    }

    void advance(int64_t limit) {
        while (frames < limit) {
            const int64_t count = std::min(block_frames, limit - frames);
            compute(frames, count);
            frames += count;
        }
        while (emitted + lookahead < frames) {
            synthesize(emitted + lookahead);
        }
    }

    std::vector<float> release(int64_t upto) {
        const int64_t total = audio_base + static_cast<int64_t>(audio.size());
        const int64_t start = released;
        const int64_t stop = std::min(upto, total);
        if (stop <= start) {
            return {};
        }
        std::vector<float> out(static_cast<size_t>(stop - start));
        synthesizer.take(start, stop, out.data());
        for (int64_t s = start; s < stop; ++s) {
            out[static_cast<size_t>(s - start)] += audio[static_cast<size_t>(s - audio_base)];
        }
        released = stop;
        return out;
    }

    // Drops the samples no frame or release still needs.
    void trim() {
        const int64_t needed_from = std::min(released, std::max<int64_t>(0, frames * config.hop_length - config.n_fft / 2));
        const int64_t drop = needed_from - audio_base;
        if (drop > 0) {
            audio.erase(audio.begin(), audio.begin() + drop);
            audio_base = needed_from;
        }
    }

    std::vector<float> push(const float * samples, size_t count) {
        if (closed) {
            throw std::runtime_error("The Voice Isolator stream was flushed; reset() starts a new one");
        }
        audio.insert(audio.end(), samples, samples + count);
        const int64_t total = audio_base + static_cast<int64_t>(audio.size());
        advance(std::max<int64_t>(0, floor_div(total - config.n_fft / 2, config.hop_length) + 1));
        auto out = release(emitted * config.hop_length - config.n_fft / 2);
        trim();
        return out;
    }

    std::vector<float> flush() {
        if (closed) {
            return {};
        }
        closed = true;
        const int64_t total = audio_base + static_cast<int64_t>(audio.size());
        advance(total / config.hop_length + 1);
        // The last computed frame serves every frame beyond it.
        while (emitted < frames) {
            synthesize(std::min(emitted + lookahead, frames - 1));
        }
        return release(total);
    }

    void reset() {
        backbone->reset();
        synthesizer.reset();
        audio.clear();
        spectra.clear();
        raw.clear();
        audio_base = frames = emitted = released = 0;
        closed = false;
    }
};

Model Model::load_from_safetensors(
    const std::filesystem::path & checkpoint_path,
    core::ExecutionContext & execution,
    const Options & options) {
    const auto index = io::load_safetensors_index(checkpoint_path);
    const auto config_entry = index.metadata.find("config");
    if (config_entry == index.metadata.end()) {
        throw std::runtime_error("Voice Isolator checkpoint has no 'config' metadata: " + checkpoint_path.string());
    }
    return load(config_entry->second, *assets::open_tensor_source(checkpoint_path), execution, options);
}

Model Model::load(
    std::string_view config_json,
    const assets::TensorSource & source,
    core::ExecutionContext & execution,
    const Options & options) {
    const auto config = parse_config(config_json);
    if (options.block_frames < 1) {
        throw std::runtime_error("Voice Isolator block_frames must be at least 1");
    }
    auto state = std::make_shared<ModelState>();
    state->options = options;
    state->lookahead = options.lookahead_frames < 0 ? config.lookahead_frames : options.lookahead_frames;
    if (config.causal) {
        const auto choices = config.streaming_lookaheads();
        if (std::find(choices.begin(), choices.end(), state->lookahead) == choices.end()) {
            throw std::runtime_error("Voice Isolator look-ahead " + std::to_string(state->lookahead) +
                " frames was not trained for this checkpoint");
        }
    }
    state->network = std::make_unique<Network>(config, source, execution,
        NetworkOptions{options.weight_storage, options.attention_cache_storage});
    return Model(std::move(state));
}

Model::Model() = default;
Model::~Model() = default;
Model::Model(Model &&) noexcept = default;
Model & Model::operator=(Model &&) noexcept = default;
Model::Model(std::shared_ptr<ModelState> state) : state_(std::move(state)) {}

bool Model::streams() const {
    if (!state_) {
        throw std::runtime_error("Voice Isolator model is not initialized");
    }
    return state_->network->config().causal;
}

int64_t Model::lookahead_frames() const {
    return streams() ? state_->lookahead : 0;
}

Waveform Model::isolate_mono_48k(const std::vector<float> & waveform) const {
    if (waveform.empty()) {
        throw std::runtime_error("The audio is empty: there is nothing to isolate");
    }
    Waveform output;
    output.sample_rate = static_cast<int>(state_->network->config().sample_rate);
    if (!streams()) {
        output.samples = render_offline(*state_, waveform);
        return output;
    }
    // A causal checkpoint runs the stream over the clip, in half-second pieces.
    StreamingSession session(state_);
    const size_t piece = static_cast<size_t>(output.sample_rate / 2);
    output.samples.reserve(waveform.size());
    for (size_t start = 0; start < waveform.size(); start += piece) {
        const auto part = session.push(waveform.data() + start, std::min(piece, waveform.size() - start));
        output.samples.insert(output.samples.end(), part.begin(), part.end());
    }
    const auto tail = session.flush();
    output.samples.insert(output.samples.end(), tail.begin(), tail.end());
    return output;
}

std::unique_ptr<StreamingSession> Model::create_streaming_session() const {
    if (!streams()) {
        throw std::runtime_error("This Voice Isolator checkpoint has no streaming mode: it was trained offline only");
    }
    return std::make_unique<StreamingSession>(state_);
}

StreamingSession::StreamingSession(std::shared_ptr<const ModelState> model)
    : model_(std::move(model)), state_(std::make_unique<State>(*model_)) {}

StreamingSession::~StreamingSession() = default;
StreamingSession::StreamingSession(StreamingSession &&) noexcept = default;
StreamingSession & StreamingSession::operator=(StreamingSession &&) noexcept = default;

std::vector<float> StreamingSession::push(const float * samples, size_t count) {
    return state_->push(samples, count);
}

std::vector<float> StreamingSession::push(const std::vector<float> & samples) {
    return state_->push(samples.data(), samples.size());
}

std::vector<float> StreamingSession::flush() {
    return state_->flush();
}

void StreamingSession::reset() {
    state_->reset();
}

int64_t StreamingSession::latency_samples() const {
    return state_->config.n_fft + state_->config.hop_length * state_->lookahead;
}

}  // namespace engine::models::voice_isolator
