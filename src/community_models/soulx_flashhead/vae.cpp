#include "engine/community_models/soulx_flashhead/vae.h"
#include "graph.h"
#include "wan_vae.h"
#include "engine/framework/codecs/ltx_video_vae_runtime.h"
#include "engine/framework/sampling/torch_random.h"
#include <tuple>

namespace engine::community_models::soulx_flashhead {
struct VideoVae::State {
    core::ExecutionContext & execution;
    const Assets & assets;
    std::unique_ptr<Weights> weights;
    std::unique_ptr<codecs::LtxVideoVaeRuntime> ltx;
    std::map<std::tuple<bool, int64_t, int64_t, int64_t>, std::vector<float>> host_outputs;
    sampling::TorchCudaSamplingPolicy posterior_policy;
    struct Graphs {
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> arena{nullptr, ggml_gallocr_free};
        std::unique_ptr<Graph> reference, motion, decode;
    };
    std::map<std::pair<int64_t, int64_t>, Graphs> graphs;
    State(core::ExecutionContext & execution, const Assets & assets, bool cuda_cudnn)
        : execution(execution), assets(assets), posterior_policy(sampling::resolve_torch_cuda_sampling_policy(
              execution.backend_type(), execution.config().device, "soulx_flashhead", "SoulX-FlashHead")) {
        if (assets.config.variant == "lite") {
            codecs::LtxVideoVaeRuntimeConfig config;
#if defined(GGML_CUDA_CUDNN_CONV3D)
            config.cuda_cudnn = cuda_cudnn;
            config.cuda_f16_activations = config.cuda_cudnn;
#else
            if (cuda_cudnn)
                throw std::runtime_error("SoulX-FlashHead VAE cuDNN support is not compiled");
#endif
            ltx = std::make_unique<codecs::LtxVideoVaeRuntime>(execution, assets.vae, config);
        } else {
            weights = std::make_unique<Weights>(execution, assets.vae, "soulx_flashhead.vae.weights");
        }
    }
    TensorValue compute(bool encode, const std::vector<float> & values, const TensorValue * device,
                        int64_t frames, int64_t height, int64_t width) {
        const auto & config = assets.config;
        const int64_t pixel_height = encode ? height : height * config.stride[1];
        const int64_t pixel_width = encode ? width : width * config.stride[2];
        const int64_t latent_frames = 32 / config.stride[0] + 1;
        if (frames != (encode ? 33 : latent_frames) && !(encode && frames == config.stride[0] + 1))
            throw std::runtime_error("FlashHead VAE expects a reference, motion, or full latent chunk");
        if (ltx) {
            ltx->prepare(pixel_height, pixel_width, {33, config.stride[0] + 1}, {latent_frames});
            if (!encode)
                return ltx->decode(values, frames, pixel_height, pixel_width);
            const auto count = static_cast<size_t>(128 * ((frames - 1) / 8 + 1) * (height / 32) * (width / 32));
            auto noise = sampling::generate_torch_cuda_tensor_iterator_randn(
                count, posterior_seed, posterior_offset, posterior_policy, sampling::TorchRandnPrecision::BFloat16);
            posterior_offset += sampling::torch_cuda_tensor_iterator_offset_blocks(count, posterior_policy);
            return device ? ltx->encode(*device, noise, frames, height, width)
                          : ltx->encode(values, noise, frames, height, width);
        }
        const auto key = std::make_pair(pixel_height, pixel_width);
        auto found = graphs.find(key);
        if (found == graphs.end()) {
            Graphs group;
            group.arena.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution.backend())));
            if (!group.arena)
                throw std::runtime_error("Cannot create FlashHead VAE allocator");
            group.reference = std::make_unique<WanGraph>(execution, *weights, true, 33, pixel_height, pixel_width,
                                                         group.arena.get());
            group.motion = std::make_unique<WanGraph>(execution, *weights, true, config.stride[0] + 1, pixel_height,
                                                      pixel_width, group.arena.get());
            group.decode = std::make_unique<WanGraph>(execution, *weights, false, latent_frames,
                                                      pixel_height / config.stride[1],
                                                      pixel_width / config.stride[2], group.arena.get());
            // Bind all cached graphs together before capturing any backend graph.
            // Growing a shared allocator later would invalidate earlier tensor addresses.
            const size_t nodes = ggml_graph_size(group.reference->graph) + ggml_graph_size(group.motion->graph) +
                                 ggml_graph_size(group.decode->graph);
            std::unique_ptr<ggml_context, decltype(&ggml_free)> plan_ctx(
                ggml_init({ggml_graph_overhead_custom(nodes, false), nullptr, true}), ggml_free);
            if (!plan_ctx)
                throw std::runtime_error("Cannot create FlashHead VAE allocation plan");
            auto * plan = ggml_new_graph_custom(plan_ctx.get(), nodes, false);
            ggml_build_forward_expand(plan, group.reference->output.tensor);
            ggml_build_forward_expand(plan, group.motion->output.tensor);
            ggml_build_forward_expand(plan, group.decode->output.tensor);
            if (!ggml_gallocr_alloc_graph(group.arena.get(), plan))
                throw std::runtime_error("Cannot allocate FlashHead VAE graphs");
            debug::timing_log_scalar("soulx_flashhead.vae.buffer_mb",
                                     ggml_gallocr_get_buffer_size(group.arena.get(), 0) / 1048576.);
            found = graphs.emplace(key, std::move(group)).first;
        }
        auto & group = found->second;
        auto & graph = encode ? (frames == 33 ? group.reference : group.motion) : group.decode;
        return device ? graph->run_device({}, {{"input", *device}})
                      : graph->run_device({{"input", &values}});
    }
    const std::vector<float> & run(bool encode, const std::vector<float> & values,
                                  int64_t frames, int64_t height, int64_t width) {
        const auto output = compute(encode, values, nullptr, frames, height, width);
        auto & host_output = host_outputs[{encode, frames, height, width}];
        core::read_tensor_float_into(output.tensor, host_output);
        return host_output;
    }

    uint64_t posterior_offset = 0;
    uint64_t posterior_seed = 42;
};
VideoVae::VideoVae(core::ExecutionContext & execution, const Assets & assets, bool cuda_cudnn)
    : state_(std::make_unique<State>(execution, assets, cuda_cudnn)) {}
VideoVae::~VideoVae() = default;
void VideoVae::reset(uint64_t seed) {
    state_->posterior_offset = 0;
    state_->posterior_seed = seed;
}
std::vector<float> VideoVae::encode(const std::vector<float> & pixels, int64_t frames, int64_t height, int64_t width) {
    return state_->run(true, pixels, frames, height, width);
}
const std::vector<float> & VideoVae::decode(const std::vector<float> & latents, int64_t frames, int64_t height,
                                            int64_t width) {
    return state_->run(false, latents, frames, height, width);
}
TensorValue VideoVae::decode_device(const std::vector<float> & latents, int64_t frames,
                                   int64_t height, int64_t width) {
    return state_->compute(false, latents, nullptr, frames, height, width);
}
std::vector<float> VideoVae::encode_device(const TensorValue & pixels, int64_t frames,
                                         int64_t height, int64_t width) {
    const auto output = state_->compute(true, {}, &pixels, frames, height, width);
    std::vector<float> result;
    core::read_tensor_float_into(output.tensor, result);
    return result;
}
} // namespace engine::community_models::soulx_flashhead
