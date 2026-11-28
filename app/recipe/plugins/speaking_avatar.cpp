#include "../recipe.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/streaming_resampler.h"
#include "../streaming_tempo.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/text/chunking.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <thread>

namespace minitts::app {
namespace {
using namespace engine::runtime;

// Two seconds of PCM bound producer lead. Cancellation releases either blocked side.
class AudioPipe {
public:
    explicit AudioPipe(int rate) : limit_(static_cast<size_t>(rate) * 2) {}
    void push(std::vector<float> samples) {
        size_t offset = 0;
        while (offset < samples.size()) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || buffered_ < limit_; });
            if (stopped_) throw std::runtime_error("streaming avatar audio was cancelled");
            const size_t count = std::min(limit_ - buffered_, samples.size() - offset);
            chunks_.emplace_back(samples.begin() + offset, samples.begin() + offset + count);
            buffered_ += count;
            offset += count;
            ready_.notify_all();
        }
    }
    bool read(int64_t maximum, std::vector<float> & samples) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopped_ || error_ || done_ || buffered_; });
        if (error_) std::rethrow_exception(error_);
        if (stopped_) throw std::runtime_error("streaming avatar audio was cancelled");
        if (!buffered_) return false;
        const auto count = std::min(static_cast<size_t>(maximum), chunks_.front().size() - offset_);
        samples.assign(chunks_.front().begin() + offset_, chunks_.front().begin() + offset_ + count);
        offset_ += count;
        buffered_ -= count;
        if (offset_ == chunks_.front().size()) { chunks_.pop_front(); offset_ = 0; }
        ready_.notify_all();
        return true;
    }
    void finish(std::exception_ptr error = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true; error_ = error; ready_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true; ready_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::vector<float>> chunks_;
    size_t limit_, buffered_ = 0, offset_ = 0;
    bool done_ = false, stopped_ = false;
    std::exception_ptr error_;
};

// Bound the text lead as well as PCM: conversation, synthesis and animation run independently.
class TextPipe {
public:
    void push(std::string text) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopped_ || chunks_.size() < 2; });
        if (stopped_) throw std::runtime_error("streaming conversation was cancelled");
        chunks_.push_back(std::move(text));
        ready_.notify_all();
    }
    bool read(std::string & text) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopped_ || error_ || done_ || !chunks_.empty(); });
        if (error_) std::rethrow_exception(error_);
        if (stopped_) throw std::runtime_error("streaming conversation was cancelled");
        if (chunks_.empty()) return false;
        text = std::move(chunks_.front()); chunks_.pop_front(); ready_.notify_all();
        return true;
    }
    void finish(std::exception_ptr error = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        done_ = true; error_ = error; ready_.notify_all();
    }
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true; ready_.notify_all();
    }
private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> chunks_;
    bool done_ = false, stopped_ = false;
    std::exception_ptr error_;
};

class SpeakingAvatarRecipe final : public IRecipe {
public:
    explicit SpeakingAvatarRecipe(bool conversation = false) : conversation_(conversation) {}
    std::string id() const override { return conversation_ ? "voice-chat-avatar" : "speaking-avatar"; }
    std::string description() const override { return "Stream conversation text, selected-voice speech and avatar animation"; }
    std::vector<RecipeModelRole> roles() const override {
        std::vector<RecipeModelRole> result{{"tts", {VoiceTaskKind::Tts, RunMode::Streaming}},
                {"avatar", {VoiceTaskKind::AudioGeneration, RunMode::Streaming}}};
        if (conversation_) result.push_back({"conversation", {VoiceTaskKind::SpeechToSpeech, RunMode::Streaming}});
        return result;
    }
    TaskResult run(IRecipeModelExecutor & executor, const TaskRequest & request,
        const RecipeEventSink & sink) const override {
        if (conversation_ ? !request.audio_input : (!request.text_input || request.text_input->text.empty()))
            throw InvalidRequestError(id() + (conversation_ ? " requires audio" : " requires text"));
        const int rate = parse_int_option(request.options, {"avatar_sample_rate"}).value_or(16000);
        if (rate <= 0) throw InvalidRequestError("avatar_sample_rate must be positive");
        const float speed = parse_finite_float_option(request.options, {"speaking_speed"}).value_or(1.0F);
        if (speed < 0.5F || speed > 2.0F) throw InvalidRequestError("speaking_speed must be in [0.5, 2]");
        const auto started = std::chrono::steady_clock::now();
        AudioPipe pipe(rate);
        TextPipe texts;
        std::mutex output_mutex;
        std::atomic<bool> tts_done{false}, conversation_done{!conversation_};
        bool first_video = true, first_audio = true;
        const auto publish = [&](const StreamEvent & event) {
            std::lock_guard<std::mutex> lock(output_mutex);
            if (sink) sink(event);
        };
        TaskResult reply;
        std::thread conversation;
        if (conversation_) conversation = std::thread([&] {
            try {
                TaskRequest question;
                question.audio_input = request.audio_input;
                question.text_input = request.text_input;
                question.input_artifacts = request.input_artifacts;
                question.options = options_for_recipe_role(request.options, "conversation", {"conversation", "tts", "avatar"}, false);
                std::string pending, streamed;
                reply = executor.stream("conversation", {VoiceTaskKind::SpeechToSpeech, RunMode::Streaming}, question,
                    [&](const StreamEvent & event) {
                        if (!event.partial_text || event.partial_text->text.empty()) return;
                        StreamEvent text;
                        text.partial_text = event.partial_text;
                        publish(text);
                        if (streamed.empty()) engine::debug::timing_log_scalar("recipe.conversation.first_text_ms", engine::debug::elapsed_ms(started));
                        streamed += text.partial_text->text;
                        pending += text.partial_text->text;
                        // Keep the unfinished tail. The framework splitter respects words, tags and UTF-8.
                        const auto chunks = engine::text::split_text_chunks(pending, 80, engine::text::TextChunkMode::TagAware);
                        size_t consumed = 0;
                        for (size_t i = 0; i + 1 < chunks.size(); ++i) {
                            consumed = pending.find(chunks[i], consumed) + chunks[i].size();
                            texts.push(chunks[i]);
                        }
                        pending.erase(0, consumed);
                    });
                if (!reply.text_output || reply.text_output->text != streamed)
                    throw std::runtime_error("conversation streamed text differs from its final reply");
                conversation_done = true;
                engine::debug::timing_log_scalar("recipe.conversation.wall_ms", engine::debug::elapsed_ms(started));
                for (auto & chunk : engine::text::split_text_chunks(pending, 80, engine::text::TextChunkMode::TagAware))
                    if (!chunk.empty()) texts.push(std::move(chunk));
                texts.finish();
            } catch (...) { texts.finish(std::current_exception()); }
        });
        TaskRequest synthesis;
        synthesis.voice = request.voice;
        synthesis.options = options_for_recipe_role(request.options, "tts", {"conversation", "tts", "avatar"}, true);
        synthesis.options.erase("avatar_sample_rate");
        synthesis.options.erase("speaking_speed");
        TaskResult speech;
        std::exception_ptr producer_error;
        std::thread producer([&] {
            try {
                int source_rate = 0;
                std::unique_ptr<engine::audio::StreamingSoxrResampler> resampler;
                std::unique_ptr<minitts::app::StreamingSpeechTempo> tempo;
                const auto emit_audio = [&](AudioBuffer audio) {
                    if (audio.samples.empty()) return;
                    if (first_audio) {
                        first_audio = false;
                        engine::debug::timing_log_scalar("recipe.tts.first_audio_ms", engine::debug::elapsed_ms(started));
                        engine::debug::trace_log_scalar("recipe.tts.first_audio_before_conversation_end", !conversation_done.load());
                    }
                    if (tempo) {
                        if (!speech.audio_output) speech.audio_output = AudioBuffer{source_rate, 1, {}};
                        append_audio_buffer(*speech.audio_output, audio);
                    }
                    StreamEvent pcm;
                    pcm.audio_output = std::move(audio);
                    publish(pcm);
                    pipe.push(resampler->process(engine::audio::mixdown_interleaved_to_mono_average(
                        pcm.audio_output->samples, pcm.audio_output->channels)));
                };
                const auto synthesize = [&](const Transcript & text) {
                    synthesis.text_input = text;
                    const auto part = executor.stream("tts", {VoiceTaskKind::Tts, RunMode::Streaming}, synthesis,
                        [&](const StreamEvent & event) {
                            if (!event.audio_output || event.audio_output->samples.empty()) return;
                            const auto & audio = *event.audio_output;
                            if (!resampler) {
                                source_rate = audio.sample_rate;
                                resampler = std::make_unique<engine::audio::StreamingSoxrResampler>(source_rate, rate);
                                if (speed != 1.0F) tempo = std::make_unique<minitts::app::StreamingSpeechTempo>(source_rate, speed);
                            }
                            if (source_rate != audio.sample_rate) throw std::runtime_error("TTS changed sample rate within a stream");
                            if (tempo) emit_audio(AudioBuffer{source_rate, 1, tempo->process(
                                engine::audio::mixdown_interleaved_to_mono_average(audio.samples, audio.channels))});
                            else emit_audio(audio);
                        });
                    if (!part.audio_output) throw std::runtime_error("TTS returned no audio");
                    if (!tempo) {
                        if (!speech.audio_output) speech.audio_output = AudioBuffer{part.audio_output->sample_rate, part.audio_output->channels, {}};
                        append_audio_buffer(*speech.audio_output, *part.audio_output);
                    }
                };
                if (conversation_) {
                    std::string text;
                    while (texts.read(text)) synthesize(Transcript{std::move(text), {}});
                } else synthesize(*request.text_input);
                if (!resampler) throw std::runtime_error("TTS returned no streamed audio");
                if (tempo) emit_audio(AudioBuffer{source_rate, 1, tempo->process({}, true)});
                tts_done = true;
                engine::debug::timing_log_scalar("recipe.tts.wall_ms", engine::debug::elapsed_ms(started));
                pipe.push(resampler->process({}, true));
                pipe.finish();
            } catch (...) {
                producer_error = std::current_exception(); texts.cancel(); pipe.finish(producer_error);
            }
        });
        TaskResult video;
        try {
            bool exhausted = false;
            AudioChunkStream source{{rate, 1}, [&](int64_t maximum, std::vector<float> & samples) {
                const bool more = pipe.read(maximum, samples);
                exhausted = !more;
                return more;
            }};
            TaskRequest animation;
            animation.audio_input = AudioBuffer{rate, 1, {}};
            animation.options = options_for_recipe_role(request.options, "avatar", {"conversation", "tts", "avatar"}, false);
            video = executor.stream("avatar", {VoiceTaskKind::AudioGeneration, RunMode::Streaming}, animation,
                [&](const StreamEvent & event) {
                    if (first_video && !event.output_artifacts.empty()) {
                        first_video = false;
                        engine::debug::timing_log_scalar("recipe.avatar.first_video_ms", engine::debug::elapsed_ms(started));
                        engine::debug::trace_log_scalar("recipe.avatar.first_video_before_tts_end", !tts_done.load());
                        engine::debug::trace_log_scalar("recipe.avatar.first_video_before_conversation_end", !conversation_done.load());
                    }
                    publish(event);
                }, &source);
            if (!exhausted) throw std::runtime_error("avatar did not consume the complete audio stream");
        } catch (...) {
            pipe.cancel(); texts.cancel(); producer.join();
            if (conversation.joinable()) conversation.join();
            throw;
        }
        producer.join();
        if (conversation.joinable()) conversation.join();
        if (producer_error) std::rethrow_exception(producer_error);
        video.audio_output = std::move(speech.audio_output);
        video.text_output = conversation_ ? std::move(reply.text_output) : request.text_input;
        video.output_artifacts.insert(video.output_artifacts.end(),
            std::make_move_iterator(reply.output_artifacts.begin()), std::make_move_iterator(reply.output_artifacts.end()));
        engine::debug::timing_log_scalar("session.wall_ms", engine::debug::elapsed_ms(started));
        return video;
    }
private:
    bool conversation_;
};
const RecipePluginRegistration registration([] { return std::make_unique<SpeakingAvatarRecipe>(); });
const RecipePluginRegistration conversation_registration([] { return std::make_unique<SpeakingAvatarRecipe>(true); });
} // namespace
} // namespace minitts::app
