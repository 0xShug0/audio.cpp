#include "engine/models/heartmula/generator.h"

#include "engine/framework/sampling/torch_random.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::heartmula {
namespace {

using TorchCudaSamplingPolicy = engine::sampling::TorchCudaSamplingPolicy;

struct TopKSamplerScratch {
    std::vector<float> scores;
    std::vector<float> threshold_values;
};

size_t prompt_flat(size_t batch, size_t row, size_t lane, size_t steps, size_t lanes) {
    return (batch * steps + row) * lanes + lane;
}

int32_t sample_topk_row(
    const float * logits,
    int64_t vocab_size,
    int64_t topk,
    float temperature,
    uint64_t seed,
    uint64_t call_index,
    const TorchCudaSamplingPolicy & policy,
    TopKSamplerScratch & scratch) {
    if (logits == nullptr || vocab_size <= 0) {
        throw std::runtime_error("HeartMuLa sampler requires logits");
    }
    if (!(temperature > 0.0F)) {
        throw std::runtime_error("HeartMuLa sampler temperature must be positive");
    }
    if (topk <= 0 || topk > vocab_size) {
        throw std::runtime_error("HeartMuLa sampler topk is out of range");
    }
    scratch.scores.resize(static_cast<size_t>(vocab_size));
    for (int64_t i = 0; i < vocab_size; ++i) {
        scratch.scores[static_cast<size_t>(i)] = logits[i] / temperature;
    }
    if (topk < vocab_size) {
        scratch.threshold_values = scratch.scores;
        auto nth = scratch.threshold_values.begin() + static_cast<std::ptrdiff_t>(topk - 1);
        std::nth_element(scratch.threshold_values.begin(), nth, scratch.threshold_values.end(), std::greater<float>());
        const float threshold = *nth;
        for (float & score : scratch.scores) {
            if (score < threshold) {
                score = -std::numeric_limits<float>::infinity();
            }
        }
    }
    float max_score = -std::numeric_limits<float>::infinity();
    for (const float score : scratch.scores) {
        if (std::isfinite(score)) {
            max_score = std::max(max_score, score);
        }
    }
    if (!std::isfinite(max_score)) {
        throw std::runtime_error("HeartMuLa sampler kept no finite logits");
    }
    double total = 0.0;
    for (const float score : scratch.scores) {
        if (std::isfinite(score)) {
            total += std::exp(static_cast<double>(score - max_score));
        }
    }
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::runtime_error("HeartMuLa sampler probability mass is invalid");
    }
    double best_rank = -std::numeric_limits<double>::infinity();
    int32_t best = -1;
    for (int64_t i = 0; i < vocab_size; ++i) {
        const float score = scratch.scores[static_cast<size_t>(i)];
        if (!std::isfinite(score)) {
            continue;
        }
        const double prob = std::exp(static_cast<double>(score - max_score)) / total;
        const float exponential = engine::sampling::torch_cuda_tensor_iterator_exponential_element(
            seed,
            static_cast<uint64_t>(vocab_size),
            static_cast<uint64_t>(i),
            call_index,
            policy.multiprocessor_count,
            policy.max_threads_per_multiprocessor);
        const double rank = prob / static_cast<double>(exponential);
        if (rank > best_rank) {
            best_rank = rank;
            best = static_cast<int32_t>(i);
        }
    }
    if (best < 0) {
        throw std::runtime_error("HeartMuLa sampler failed to select a token");
    }
    return best;
}

std::vector<int32_t> sample_with_cfg(
    const std::vector<float> & logits,
    int64_t batch,
    int64_t vocab_size,
    float cfg_scale,
    int64_t topk,
    float temperature,
    uint64_t seed,
    uint64_t & sample_call_index,
    const TorchCudaSamplingPolicy & policy,
    TopKSamplerScratch & scratch) {
    if (batch <= 0 || vocab_size <= 0 || static_cast<int64_t>(logits.size()) != batch * vocab_size) {
        throw std::runtime_error("HeartMuLa sampler logits shape mismatch");
    }
    const bool use_cfg = cfg_scale > 1.0F && batch > 1 && batch % 2 == 0;
    const int64_t rows = use_cfg ? batch / 2 : batch;
    std::vector<int32_t> out(static_cast<size_t>(batch), 0);
    std::vector<float> guided(static_cast<size_t>(vocab_size));
    for (int64_t row = 0; row < rows; ++row) {
        const float * source = logits.data() + static_cast<size_t>(row * vocab_size);
        if (use_cfg) {
            const float * uncond = logits.data() + static_cast<size_t>((row + rows) * vocab_size);
            for (int64_t v = 0; v < vocab_size; ++v) {
                guided[static_cast<size_t>(v)] = uncond[v] + (source[v] - uncond[v]) * cfg_scale;
            }
            source = guided.data();
        }
        const int32_t token = sample_topk_row(
            source,
            vocab_size,
            topk,
            temperature,
            seed,
            sample_call_index++,
            policy,
            scratch);
        out[static_cast<size_t>(row)] = token;
        if (use_cfg) {
            out[static_cast<size_t>(row + rows)] = token;
        }
    }
    return out;
}

HeartMuLaFrameEmbeddingInputs prompt_embedding_inputs(
    const HeartMuLaPromptEncoding & encoding,
    const HeartMuLaConfig & config,
    bool use_cfg) {
    HeartMuLaFrameEmbeddingInputs inputs;
    inputs.batch_size = encoding.batch_size;
    inputs.steps = encoding.prompt_len;
    const size_t batch = static_cast<size_t>(encoding.batch_size);
    const size_t steps = static_cast<size_t>(encoding.prompt_len);
    const size_t lanes = static_cast<size_t>(encoding.parallel_number);
    const size_t codebooks = static_cast<size_t>(config.audio_num_codebooks);
    inputs.audio_token_ids.assign(batch * steps * codebooks, 0);
    inputs.text_token_ids.assign(batch * steps, 0);
    inputs.audio_mask.assign(batch * steps * codebooks, 0.0F);
    inputs.text_cond_mask.assign(batch * steps, 0.0F);
    inputs.text_uncond_mask.assign(batch * steps, 0.0F);
    const size_t actual_batch = use_cfg ? batch / 2 : batch;
    for (size_t b = 0; b < batch; ++b) {
        const bool uncond = use_cfg && b >= actual_batch;
        for (size_t t = 0; t < steps; ++t) {
            for (size_t q = 0; q < codebooks; ++q) {
                const size_t src = prompt_flat(b, t, q, steps, lanes);
                const size_t dst = (b * steps + t) * codebooks + q;
                inputs.audio_token_ids[dst] = static_cast<int32_t>(
                    encoding.tokens[src] + static_cast<int64_t>(q) * config.audio_vocab_size);
                inputs.audio_mask[dst] = encoding.tokens_mask[src] != 0U ? 1.0F : 0.0F;
            }
            const size_t text_src = prompt_flat(b, t, lanes - 1, steps, lanes);
            inputs.text_token_ids[b * steps + t] = static_cast<int32_t>(encoding.tokens[text_src]);
            const float text_mask = encoding.tokens_mask[text_src] != 0U ? 1.0F : 0.0F;
            inputs.text_cond_mask[b * steps + t] = uncond ? 0.0F : text_mask;
            inputs.text_uncond_mask[b * steps + t] = uncond ? text_mask : 0.0F;
        }
    }
    inputs.apply_muq = true;
    inputs.muq_row = encoding.muq_idx.empty() ? 0 : encoding.muq_idx.front();
    inputs.muq_embed = encoding.muq_embed;
    inputs.muq_cond_mask.assign(batch, 1.0F);
    inputs.muq_uncond_mask.assign(batch, 0.0F);
    if (use_cfg) {
        for (size_t b = actual_batch; b < batch; ++b) {
            inputs.muq_cond_mask[b] = 0.0F;
            inputs.muq_uncond_mask[b] = 1.0F;
        }
    }
    return inputs;
}

HeartMuLaFrameEmbeddingInputs audio_frame_embedding_inputs(
    const std::vector<int32_t> & frame_tokens,
    int64_t batch,
    int64_t codebook,
    const HeartMuLaConfig & config) {
    if (static_cast<int64_t>(frame_tokens.size()) != batch) {
        throw std::runtime_error("HeartMuLa audio embedding token batch mismatch");
    }
    HeartMuLaFrameEmbeddingInputs inputs;
    inputs.batch_size = batch;
    inputs.steps = 1;
    const size_t codebooks = static_cast<size_t>(config.audio_num_codebooks);
    inputs.audio_token_ids.assign(static_cast<size_t>(batch) * codebooks, 0);
    inputs.audio_mask.assign(static_cast<size_t>(batch) * codebooks, 0.0F);
    inputs.text_token_ids.assign(static_cast<size_t>(batch), 0);
    inputs.text_cond_mask.assign(static_cast<size_t>(batch), 0.0F);
    inputs.text_uncond_mask.assign(static_cast<size_t>(batch), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t q = 0; q < config.audio_num_codebooks; ++q) {
            const size_t index = static_cast<size_t>(b * config.audio_num_codebooks + q);
            const int32_t raw = q == codebook ? frame_tokens[static_cast<size_t>(b)] : 0;
            inputs.audio_token_ids[index] = raw + static_cast<int32_t>(q * config.audio_vocab_size);
            inputs.audio_mask[index] = q == codebook ? 1.0F : 0.0F;
        }
    }
    return inputs;
}

HeartMuLaFrameEmbeddingInputs next_frame_embedding_inputs(
    const std::vector<int32_t> & frame_tokens,
    int64_t batch,
    const HeartMuLaConfig & config) {
    if (static_cast<int64_t>(frame_tokens.size()) != batch * config.audio_num_codebooks) {
        throw std::runtime_error("HeartMuLa next-frame token shape mismatch");
    }
    HeartMuLaFrameEmbeddingInputs inputs;
    inputs.batch_size = batch;
    inputs.steps = 1;
    const size_t count = static_cast<size_t>(batch * config.audio_num_codebooks);
    inputs.audio_token_ids.assign(count, 0);
    inputs.audio_mask.assign(count, 1.0F);
    inputs.text_token_ids.assign(static_cast<size_t>(batch), 0);
    inputs.text_cond_mask.assign(static_cast<size_t>(batch), 0.0F);
    inputs.text_uncond_mask.assign(static_cast<size_t>(batch), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t q = 0; q < config.audio_num_codebooks; ++q) {
            const size_t index = static_cast<size_t>(b * config.audio_num_codebooks + q);
            inputs.audio_token_ids[index] =
                frame_tokens[index] + static_cast<int32_t>(q * config.audio_vocab_size);
        }
    }
    return inputs;
}

std::vector<float> decoder_prefill_input(
    const HeartMuLaBackboneHidden & last_hidden,
    const HeartMuLaMergedEmbeddings & c0_embedding) {
    if (last_hidden.dims <= 0 || c0_embedding.dims != last_hidden.dims || c0_embedding.steps != 1) {
        throw std::runtime_error("HeartMuLa decoder prefill embedding shape mismatch");
    }
    const int64_t batch = c0_embedding.batch_size;
    std::vector<float> out(static_cast<size_t>(batch * 2 * last_hidden.dims), 0.0F);
    for (int64_t b = 0; b < batch; ++b) {
        const size_t hidden_src = static_cast<size_t>(b * last_hidden.dims);
        const size_t c0_src = static_cast<size_t>(b * last_hidden.dims);
        const size_t dst0 = static_cast<size_t>((b * 2) * last_hidden.dims);
        const size_t dst1 = static_cast<size_t>((b * 2 + 1) * last_hidden.dims);
        std::copy(
            last_hidden.values.begin() + static_cast<std::ptrdiff_t>(hidden_src),
            last_hidden.values.begin() + static_cast<std::ptrdiff_t>(hidden_src + last_hidden.dims),
            out.begin() + static_cast<std::ptrdiff_t>(dst0));
        std::copy(
            c0_embedding.values.begin() + static_cast<std::ptrdiff_t>(c0_src),
            c0_embedding.values.begin() + static_cast<std::ptrdiff_t>(c0_src + last_hidden.dims),
            out.begin() + static_cast<std::ptrdiff_t>(dst1));
    }
    return out;
}

std::vector<int32_t> generate_frame_from_backbone(
    const HeartMuLaBackboneResult & backbone_result,
    const HeartMuLaWeightsRuntime & mula,
    int64_t batch,
    const HeartMuLaGenerationOptions & options,
    uint64_t seed,
    uint64_t & sample_call_index,
    const TorchCudaSamplingPolicy & policy,
    TopKSamplerScratch & scratch) {
    const auto & config = mula.assets().mula_config;
    auto c0_sample = sample_with_cfg(
        backbone_result.logits.values,
        batch,
        backbone_result.logits.vocab_size,
        options.guidance_scale,
        options.top_k,
        options.temperature,
        seed,
        sample_call_index,
        policy,
        scratch);
    std::vector<int32_t> frame(static_cast<size_t>(batch * config.audio_num_codebooks), 0);
    for (int64_t b = 0; b < batch; ++b) {
        frame[static_cast<size_t>(b * config.audio_num_codebooks)] = c0_sample[static_cast<size_t>(b)];
    }
    auto c0_embedding = mula.merge_frame_embeddings(audio_frame_embedding_inputs(c0_sample, batch, 0, config));
    const auto decoder_input = decoder_prefill_input(backbone_result.last_hidden, c0_embedding);
    HeartMuLaDecoderCachedState decoder_state;
    auto decoder = mula.decoder_prefill_embeddings(
        decoder_input,
        batch,
        2,
        0);
    mula.reset_decoder_cached_state(decoder_state, std::move(decoder.state));
    for (int64_t codebook = 1; codebook < config.audio_num_codebooks; ++codebook) {
        const auto sample = sample_with_cfg(
            decoder.result.logits.values,
            batch,
            decoder.result.logits.vocab_size,
            options.guidance_scale,
            options.top_k,
            options.temperature,
            seed,
            sample_call_index,
            policy,
            scratch);
        for (int64_t b = 0; b < batch; ++b) {
            frame[static_cast<size_t>(b * config.audio_num_codebooks + codebook)] = sample[static_cast<size_t>(b)];
        }
        if (codebook + 1 >= config.audio_num_codebooks) {
            break;
        }
        auto ci_embedding = mula.merge_frame_embeddings(audio_frame_embedding_inputs(sample, batch, codebook, config));
        decoder.result = mula.decoder_cached_step(
            ci_embedding.values,
            batch,
            codebook,
            decoder_state,
            config.audio_num_codebooks);
    }
    return frame;
}

void append_first_branch_frame(
    HeartMuLaGeneratedFrames & out,
    const std::vector<int32_t> & frame,
    const HeartMuLaConfig & config) {
    if (static_cast<int64_t>(frame.size()) < config.audio_num_codebooks) {
        throw std::runtime_error("HeartMuLa generated frame is too small");
    }
    out.codes.insert(out.codes.end(), frame.begin(), frame.begin() + static_cast<std::ptrdiff_t>(config.audio_num_codebooks));
    ++out.frames;
}

bool first_branch_has_eos(const std::vector<int32_t> & frame, const HeartMuLaAssets & assets) {
    for (int64_t q = 0; q < assets.mula_config.audio_num_codebooks; ++q) {
        if (frame[static_cast<size_t>(q)] >= assets.generation_config.audio_eos_id) {
            return true;
        }
    }
    return false;
}

}  // namespace

HeartMuLaGeneratedFrames generate_heartmula_frames(
    const HeartMuLaPromptRequest & request,
    const HeartMuLaTextTokenizer & tokenizer,
    const HeartMuLaWeightsRuntime & mula,
    uint64_t seed) {
    const auto & assets = mula.assets();
    const auto & config = assets.mula_config;
    const auto encoding = tokenizer.encode_prompt(request);
    const bool use_cfg = request.options.guidance_scale > 1.0F && encoding.batch_size > 1 && encoding.batch_size % 2 == 0;
    auto prompt_embeddings = mula.merge_frame_embeddings(prompt_embedding_inputs(encoding, config, use_cfg));
    auto backbone = mula.backbone_prefill_embeddings(
        prompt_embeddings.values,
        prompt_embeddings.batch_size,
        prompt_embeddings.steps);
    const int64_t max_audio_frames = static_cast<int64_t>(request.options.duration_seconds * 1000.0F) / 80;
    HeartMuLaBackboneCachedState backbone_state;
    mula.reset_backbone_cached_state(backbone_state, std::move(backbone.state));

    const TorchCudaSamplingPolicy policy = engine::sampling::resolve_torch_cuda_sampling_policy(
        mula.backend_type(),
        mula.device(),
        "heartmula.cuda_sampling_policy",
        "HeartMuLa",
        engine::sampling::TorchCudaSamplingPolicyFailureMode::StrictCuda);
    TopKSamplerScratch scratch;
    uint64_t sample_call_index = 0;
    HeartMuLaGeneratedFrames out;
    out.codebooks = config.audio_num_codebooks;
    auto current_frame = generate_frame_from_backbone(
        backbone.result,
        mula,
        encoding.batch_size,
        request.options,
        seed,
        sample_call_index,
        policy,
        scratch);
    append_first_branch_frame(out, current_frame, config);

    for (int64_t frame_index = 0; frame_index < max_audio_frames; ++frame_index) {
        auto next_embeddings = mula.merge_frame_embeddings(next_frame_embedding_inputs(current_frame, encoding.batch_size, config));
        auto next_backbone = mula.backbone_cached_step(
            next_embeddings.values,
            encoding.batch_size,
            backbone_state,
            encoding.prompt_len + max_audio_frames + 1);
        current_frame = generate_frame_from_backbone(
            next_backbone,
            mula,
            encoding.batch_size,
            request.options,
            seed,
            sample_call_index,
            policy,
            scratch);
        if (first_branch_has_eos(current_frame, assets)) {
            break;
        }
        append_first_branch_frame(out, current_frame, config);
    }
    const uint64_t ar_offset_blocks = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(
        static_cast<uint64_t>(config.audio_vocab_size),
        policy);
    const uint64_t codec_flow_elements = static_cast<uint64_t>(
        static_cast<int64_t>(request.options.codec_duration * 25.0F) * assets.codec_config.out_channels);
    out.codec_randn_philox_offset = sample_call_index * ar_offset_blocks;
    out.codec_randn_call_offset_blocks = engine::sampling::torch_cuda_tensor_iterator_offset_blocks(codec_flow_elements, policy);
    return out;
}

}  // namespace engine::models::heartmula
