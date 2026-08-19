#include "engine/community_models/moss_voicegen/delay_decoder.h"

#include "engine/models/moss/shared/sampling.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace engine::models::moss_voicegen {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

void forbid(std::vector<float> & logits, int64_t token_id) {
    if (token_id >= 0 && token_id < static_cast<int64_t>(logits.size())) {
        logits[static_cast<size_t>(token_id)] = kNegativeInfinity;
    }
}

void apply_temperature(std::vector<float> & logits, float temperature) {
    if (temperature == 1.0F || temperature <= 0.0F) {
        return;
    }
    for (float & value : logits) {
        value /= temperature;
    }
}

}  // namespace

MossVoiceGenDelayDecoder::MossVoiceGenDelayDecoder(
    MossVoiceGenConfig config,
    MossVoiceGenSamplingOptions sampling,
    uint32_t seed,
    MossVoiceGenLengthBounds bounds)
    : config_(std::move(config)),
      sampling_(sampling),
      bounds_(bounds),
      seed_(seed),
      rng_(seed) {
    if (config_.num_codebooks <= 0) {
        throw std::runtime_error("MOSS-VoiceGenerator delay decoder requires a positive codebook count");
    }
}

int32_t MossVoiceGenDelayDecoder::sample_text(std::vector<float> & logits) {
    if (!sampling_.do_sample) {
        return engine::models::moss::argmax_index(logits, "moss_voicegen.text");
    }
    return engine::models::moss::sample_index(
        logits,
        sampling_.text_top_k,
        sampling_.text_top_p,
        1.0F,  // the temperature is already folded into the logits
        rng_,
        "moss_voicegen.text",
        nullptr,
        seed_,
        sample_call_index_++);
}

int32_t MossVoiceGenDelayDecoder::sample_code(std::vector<float> & logits, int64_t codebook) {
    if (sampling_.audio_repetition_penalty != 1.0F) {
        // The reference penalises against every earlier row of this codebook, prompt rows
        // included. Those are all pad, and pad is masked to -inf just below, so restricting
        // this to the generated history gives the same result.
        std::vector<int32_t> previous;
        previous.reserve(history_.size());
        for (const auto & row : history_) {
            previous.push_back(row.codes[static_cast<size_t>(codebook)]);
        }
        engine::models::moss::apply_repetition_penalty(
            logits, previous, sampling_.audio_repetition_penalty, "moss_voicegen.audio");
    }
    forbid(logits, config_.audio_pad_code);
    if (!sampling_.do_sample) {
        return engine::models::moss::argmax_index(logits, "moss_voicegen.audio");
    }
    return engine::models::moss::sample_index(
        logits,
        sampling_.audio_top_k,
        sampling_.audio_top_p,
        1.0F,
        rng_,
        "moss_voicegen.audio",
        nullptr,
        seed_,
        sample_call_index_++);
}

MossVoiceGenDelayRow MossVoiceGenDelayDecoder::step(MossVoiceGenStepLogits & logits) {
    const int64_t n_vq = config_.num_codebooks;
    const bool delaying = delayed_length_ != kNotDelaying;

    // Text side. While the flush window is open the text token is dictated, not sampled:
    // one delay slot per remaining codebook, then the audio-end marker.
    // audio_length_ counts audio rows, and the first of them is the audio-start row that
    // carries no codes, so the frames emitted so far are one fewer.
    const int64_t frames_so_far = audio_length_ > 0 ? audio_length_ - 1 : 0;
    const bool below_floor =
        in_audio_ && bounds_.min_frames > 0 && frames_so_far < bounds_.min_frames;
    const bool at_ceiling =
        in_audio_ && bounds_.max_frames > 0 && frames_so_far >= bounds_.max_frames;

    int32_t next_text = static_cast<int32_t>(config_.pad_token_id);
    bool sample_text_token = false;
    bool audio_ended = false;
    if (!stopped_) {
        if (!delaying && at_ceiling) {
            // Out of budget: open the flush window instead of sampling another frame.
            next_text = static_cast<int32_t>(config_.audio_assistant_delay_slot_token_id);
        } else if (delaying && delayed_length_ < n_vq) {
            next_text = static_cast<int32_t>(config_.audio_assistant_delay_slot_token_id);
        } else if (delaying && delayed_length_ == n_vq) {
            next_text = static_cast<int32_t>(config_.audio_end_token_id);
            audio_ended = true;
        } else {
            sample_text_token = true;
        }
    }
    if (audio_ended) {
        in_audio_ = false;
    }

    if (sample_text_token) {
        apply_temperature(logits.text, sampling_.text_temperature);
        if (in_audio_) {
            // Mid-utterance the only legal continuations are "another audio frame" and
            // "start the flush", so everything else is masked out.
            for (size_t token = 0; token < logits.text.size(); ++token) {
                const auto id = static_cast<int64_t>(token);
                if (id != config_.audio_assistant_gen_slot_token_id
                    && id != config_.audio_assistant_delay_slot_token_id) {
                    logits.text[token] = kNegativeInfinity;
                }
            }
        } else {
            forbid(logits.text, config_.pad_token_id);
            forbid(logits.text, config_.audio_assistant_gen_slot_token_id);
            forbid(logits.text, config_.audio_assistant_delay_slot_token_id);
            forbid(logits.text, config_.audio_end_token_id);
        }
        if (step_index_ == 0) {
            // Nothing has been generated yet, so a flush cannot start here.
            forbid(logits.text, config_.audio_assistant_delay_slot_token_id);
        }
        if (step_index_ <= n_vq) {
            // Too early to end the turn: the first codebooks have not even been reached.
            forbid(logits.text, config_.im_end_token_id);
        }
        if (below_floor) {
            // Too little audio to be the whole utterance, so neither retire the codebooks
            // nor end the turn; the only remaining choice is another frame.
            forbid(logits.text, config_.audio_assistant_delay_slot_token_id);
            forbid(logits.text, config_.im_end_token_id);
        }
        next_text = sample_text(logits.text);
    }

    if (next_text == static_cast<int32_t>(config_.audio_start_token_id)) {
        in_audio_ = true;
    }
    if (next_text == static_cast<int32_t>(config_.im_end_token_id)) {
        stopped_ = true;
    }

    // Audio side. Codebook i is live once the audio has been running longer than its delay
    // and until the flush window has retired it.
    MossVoiceGenDelayRow row;
    row.text_token = next_text;
    row.codes.assign(static_cast<size_t>(n_vq), static_cast<int32_t>(config_.audio_pad_code));
    if (static_cast<int64_t>(logits.audio.size()) != n_vq) {
        throw std::runtime_error("MOSS-VoiceGenerator delay decoder expects one audio head per codebook");
    }
    for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
        const bool started = audio_length_ > codebook;
        const bool retired = delaying && codebook <= delayed_length_ - 1;
        if (!started || retired) {
            continue;
        }
        auto & codebook_logits = logits.audio[static_cast<size_t>(codebook)];
        apply_temperature(codebook_logits, sampling_.audio_temperature);
        row.codes[static_cast<size_t>(codebook)] = sample_code(codebook_logits, codebook);
    }

    // State updates, in the reference's order.
    if (next_text == static_cast<int32_t>(config_.audio_start_token_id)
        || next_text == static_cast<int32_t>(config_.audio_assistant_gen_slot_token_id)
        || next_text == static_cast<int32_t>(config_.audio_assistant_delay_slot_token_id)) {
        ++audio_length_;
    }
    if (next_text == static_cast<int32_t>(config_.audio_end_token_id)) {
        audio_length_ = 0;
    }
    if (!delaying && next_text == static_cast<int32_t>(config_.audio_assistant_delay_slot_token_id)) {
        delayed_length_ = 0;
    }
    if (delayed_length_ != kNotDelaying) {
        ++delayed_length_;
        if (delayed_length_ > n_vq) {
            delayed_length_ = kNotDelaying;
        }
    }

    ++step_index_;
    history_.push_back(row);
    return row;
}

std::vector<int32_t> MossVoiceGenDelayDecoder::extract_audio_codes(
    int64_t & codebooks_out,
    int64_t & frames_out) const {
    const int64_t n_vq = config_.num_codebooks;
    // Keep the rows that carry codes. The audio-start row is deliberately excluded: it
    // opens the audio segment but its own codes are all pad, because audio_length is still
    // zero while it is being emitted and no codebook is live yet. Counting it would shift
    // every codebook by one row and make the highest codebook read a pad code from the
    // final flush row.
    std::vector<const MossVoiceGenDelayRow *> audio_rows;
    for (const auto & row : history_) {
        const bool carries_codes =
            row.text_token == static_cast<int32_t>(config_.audio_assistant_gen_slot_token_id)
            || row.text_token == static_cast<int32_t>(config_.audio_assistant_delay_slot_token_id);
        if (carries_codes) {
            audio_rows.push_back(&row);
        }
    }
    const auto delayed_frames = static_cast<int64_t>(audio_rows.size());
    int64_t frames = delayed_frames - n_vq + 1;

    // The row count only bounds the frames; it does not guarantee they are all complete.
    // If generation is cut short — the step ceiling, or a turn that ends before the flush
    // window has run its course — the trailing rows still hold pad for the higher
    // codebooks. Reading those as codes puts a pad value into the codec, which rejects it
    // as out of range. Keep only the frames where every codebook carries a real code.
    const auto pad = static_cast<int32_t>(config_.audio_pad_code);
    int64_t complete = 0;
    while (complete < frames) {
        bool full = true;
        for (int64_t codebook = 0; codebook < n_vq && full; ++codebook) {
            full = audio_rows[static_cast<size_t>(codebook + complete)]->codes[static_cast<size_t>(codebook)] != pad;
        }
        if (!full) {
            break;
        }
        ++complete;
    }
    frames = complete;

    if (frames <= 0) {
        codebooks_out = n_vq;
        frames_out = 0;
        return {};
    }

    // De-delay: codebook i's frame f sits in delayed row i + f. Mirrors
    // MossTTSDelayProcessor.apply_de_delay_pattern.
    std::vector<int32_t> codes(static_cast<size_t>(n_vq * frames), 0);
    for (int64_t codebook = 0; codebook < n_vq; ++codebook) {
        for (int64_t frame = 0; frame < frames; ++frame) {
            codes[static_cast<size_t>(codebook * frames + frame)] =
                audio_rows[static_cast<size_t>(codebook + frame)]->codes[static_cast<size_t>(codebook)];
        }
    }
    codebooks_out = n_vq;
    frames_out = frames;
    return codes;
}

}  // namespace engine::models::moss_voicegen
