#include "engine/community_models/zonos3/prompt.h"

#include "engine/framework/runtime/errors.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace engine::models::zonos3 {
namespace {

[[noreturn]] void reject(const std::string & message) {
    throw runtime::InvalidRequestError("zonos3: " + message);
}

void check_buckets(const Zonos3Hparams & hp, const std::vector<int32_t> & buckets) {
    if (buckets.size() != static_cast<size_t>(hp.n_cond)) {
        reject("expected " + std::to_string(hp.n_cond) + " condition buckets");
    }
    for (int i = 0; i < hp.n_cond; ++i) {
        const int32_t count = hp.cond_bucket_counts[static_cast<size_t>(i)];
        if (buckets[static_cast<size_t>(i)] < 0 || buckets[static_cast<size_t>(i)] > count) {
            reject("condition bucket " + std::to_string(buckets[static_cast<size_t>(i)]) + " for " +
                   hp.cond_names[static_cast<size_t>(i)] + " outside 0.." + std::to_string(count - 1) + " (" +
                   std::to_string(count) + " = unknown)");
        }
    }
}

}  // namespace

Zonos3Prompt build_zonos3_prompt(const Zonos3Hparams & hp, const Zonos3Tokenizer & tokenizer, const std::string & text,
                                 const Zonos3PromptOptions & options) {
    const int ncb = hp.n_codebooks;
    const size_t width = static_cast<size_t>(ncb) + 1;
    if (options.speaker_embedding && (options.reference_codes || options.reference_frames)) {
        reject("a speaker embedding and a reference prefix are mutually exclusive");
    }
    if (options.reference_frames < 0 || options.reference_frames > kZonos3ReferenceMaxFrames ||
        (options.reference_frames > 0) != (options.reference_codes != nullptr)) {
        reject("invalid reference-code prefix");
    }
    if (options.silence_prefill_frames < 0 || options.silence_prefill_frames > kZonos3SilencePrefillMax) {
        reject("silence prefill must be 0.." + std::to_string(kZonos3SilencePrefillMax) + " frames");
    }
    if (options.speaker_embedding) {
        for (int i = 0; i < hp.speaker_dim; ++i) {
            if (!std::isfinite(options.speaker_embedding[i])) reject("speaker embedding contains a non-finite value");
        }
    }
    std::vector<int32_t> buckets = options.condition_buckets;
    if (buckets.empty()) buckets = hp.cond_bucket_counts;
    check_buckets(hp, buckets);
    const auto text_ids = tokenizer.text_ids(text);
    const size_t reference_rows = options.reference_frames ? static_cast<size_t>(options.reference_frames + ncb - 1) : 0;
    const size_t speaker_rows = options.speaker_embedding ? 1 : reference_rows;
    const size_t prefill = static_cast<size_t>(options.silence_prefill_frames);
    const size_t rows = speaker_rows + static_cast<size_t>(hp.n_cond) + text_ids.size() + prefill;
    if (rows > static_cast<size_t>(hp.n_ctx_train)) {
        reject("prompt exceeds the " + std::to_string(hp.n_ctx_train) + "-position context");
    }
    for (size_t i = 0; i < static_cast<size_t>(options.reference_frames) * static_cast<size_t>(ncb); ++i) {
        if (options.reference_codes[i] < 0 || options.reference_codes[i] >= hp.codebook_size) {
            reject("reference contains a nonordinary audio code");
        }
    }
    Zonos3Prompt out;
    out.ids.assign(rows * width, hp.audio_pad_id);
    for (size_t t = 0; t < rows; ++t) out.ids[t * width + static_cast<size_t>(ncb)] = hp.text_vocab;
    for (size_t t = 0; t < reference_rows; ++t) {
        for (int cb = 0; cb < ncb; ++cb) {
            const int frame = static_cast<int>(t) - cb;
            if (frame >= 0 && frame < options.reference_frames) {
                out.ids[t * width + static_cast<size_t>(cb)] =
                    options.reference_codes[static_cast<size_t>(frame) * static_cast<size_t>(ncb) + static_cast<size_t>(cb)];
            }
        }
    }
    const size_t text_start = speaker_rows + static_cast<size_t>(hp.n_cond);
    for (size_t t = 0; t < text_ids.size(); ++t) out.ids[(text_start + t) * width + static_cast<size_t>(ncb)] = text_ids[t];
    const size_t silence_start = text_start + text_ids.size();
    for (size_t k = 0; k < prefill; ++k) {
        for (int cb = 0; cb < ncb; ++cb) {
            if (static_cast<int>(k) >= cb) {
                out.ids[(silence_start + k) * width + static_cast<size_t>(cb)] = hp.silence_frame[static_cast<size_t>(cb)];
            }
        }
    }
    out.n_rows = static_cast<int>(rows);
    out.speaker_rows = static_cast<int>(speaker_rows);
    out.prefill_frames = static_cast<int>(prefill);
    out.cond.condition_row = static_cast<int>(speaker_rows);
    out.cond.buckets = std::move(buckets);
    if (options.speaker_embedding) {
        out.cond.speaker_row = 0;
        out.cond.speaker.assign(options.speaker_embedding, options.speaker_embedding + hp.speaker_dim);
    }
    zonos3_check_conditioning(hp, out);
    return out;
}

Zonos3Prompt build_zonos3_cfg_twin(const Zonos3Hparams & hp, const Zonos3Prompt & prompt) {
    if (prompt.speaker_rows <= 0) {
        throw runtime::InvalidRequestError("zonos3 guidance_scale needs a speaker source");
    }
    const size_t width = static_cast<size_t>(hp.n_codebooks) + 1;
    if (prompt.speaker_rows >= prompt.n_rows || prompt.ids.size() != static_cast<size_t>(prompt.n_rows) * width) {
        throw std::runtime_error("zonos3: inconsistent prompt for speaker CFG");
    }
    Zonos3Prompt twin;
    twin.ids.assign(prompt.ids.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(prompt.speaker_rows) * width),
                    prompt.ids.end());
    twin.n_rows = prompt.n_rows - prompt.speaker_rows;
    twin.speaker_rows = 0;
    twin.prefill_frames = prompt.prefill_frames;
    twin.cond.speaker_row = -1;
    twin.cond.condition_row = prompt.cond.condition_row - prompt.speaker_rows;
    twin.cond.buckets = hp.cond_bucket_counts;
    zonos3_check_conditioning(hp, twin);
    return twin;
}

int zonos3_condition_bucket(const Zonos3Hparams & hp, int index, double value) {
    if (index < 0 || index >= static_cast<int>(hp.cond_edges.size()) || !std::isfinite(value)) return -1;
    const auto & edges = hp.cond_edges[static_cast<size_t>(index)];
    int bucket = 0;
    while (bucket < static_cast<int>(edges.size()) && static_cast<double>(edges[static_cast<size_t>(bucket)]) <= value) {
        ++bucket;
    }
    return bucket;
}

std::vector<int32_t> zonos3_prefill_codes(const Zonos3Hparams & hp, const Zonos3Prompt & prompt) {
    const size_t ncb = static_cast<size_t>(hp.n_codebooks);
    const size_t width = ncb + 1;
    std::vector<int32_t> codes;
    codes.reserve(static_cast<size_t>(prompt.prefill_frames) * ncb);
    for (int k = 0; k < prompt.prefill_frames; ++k) {
        const auto row = prompt.ids.begin() + static_cast<std::ptrdiff_t>(
            static_cast<size_t>(prompt.n_rows - prompt.prefill_frames + k) * width);
        codes.insert(codes.end(), row, row + static_cast<std::ptrdiff_t>(ncb));
    }
    return codes;
}

void zonos3_check_conditioning(const Zonos3Hparams & hp, const Zonos3Prompt & prompt) {
    const size_t width = static_cast<size_t>(hp.n_codebooks) + 1;
    const auto & c = prompt.cond;
    auto is_pad_row = [&](int row) {
        const auto * ids = prompt.ids.data() + static_cast<size_t>(row) * width;
        for (int cb = 0; cb < hp.n_codebooks; ++cb) {
            if (ids[cb] != hp.audio_pad_id) return false;
        }
        return ids[hp.n_codebooks] == hp.text_vocab;
    };
    auto require_pad_row = [&](int row) {
        if (row < 0 || row >= prompt.n_rows || !is_pad_row(row)) {
            throw std::runtime_error("zonos3: conditioning row " + std::to_string(row) + " is not an all-pad row");
        }
    };
    if (c.speaker_row >= 0) {
        if (c.speaker.size() != static_cast<size_t>(hp.speaker_dim)) {
            reject("expected a " + std::to_string(hp.speaker_dim) + "-value speaker embedding");
        }
        for (const float v : c.speaker) {
            if (!std::isfinite(v)) reject("speaker embedding contains a non-finite value");
        }
        require_pad_row(c.speaker_row);
    }
    if (c.condition_row >= 0) {
        check_buckets(hp, c.buckets);
        for (int i = 0; i < hp.n_cond; ++i) require_pad_row(c.condition_row + i);
    }
    for (int r = 0; r < prompt.n_rows; ++r) {
        if (!is_pad_row(r)) continue;
        const bool speaker = c.speaker_row >= 0 && r == c.speaker_row;
        const bool condition = c.condition_row >= 0 && r >= c.condition_row && r < c.condition_row + hp.n_cond;
        if (!speaker && !condition) {
            throw std::runtime_error("zonos3: all-pad row " + std::to_string(r) + " has no speaker/condition embedding");
        }
    }
}

void zonos3_cfg_mix(const float * cond, const float * uncond, float scale, size_t n, float * out) {
    for (size_t i = 0; i < n; ++i) out[i] = uncond[i] + scale * (cond[i] - uncond[i]);
}

}  // namespace engine::models::zonos3
