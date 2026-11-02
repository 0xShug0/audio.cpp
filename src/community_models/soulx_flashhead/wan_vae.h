#pragma once
#include "graph.h"
#include "engine/framework/codecs/wan_video_vae_runtime.h"

namespace engine::community_models::soulx_flashhead {
class WanGraph final : public Graph {
  public:
    WanGraph(core::ExecutionContext & execution, Weights & weights, bool encode, int64_t frames, int64_t height,
             int64_t width, ggml_gallocr_t arena)
        : Graph(execution, encode ? "soulx_flashhead.wan.encode" : "soulx_flashhead.wan.decode", arena),
          weights(weights) {
        codecs::WanVideoVAEArchitectureConfig config;
        config.use_cuda_fast_lowerings = execution.backend_type() == core::BackendType::Cuda;
        codecs::WanVideoVAEEncoderWeights ew;
        codecs::WanVideoVAEDecoderWeights dw;
        const auto scale = codecs::WanVideoVAELatentScaleModule::s2v_14b_config();
        auto inv = scale.std;
        for (auto & v : inv)
            v = 1 / v;
        ew.latent_mean = constant("mean", {16}, scale.mean);
        ew.latent_std = constant("inv_std", {16}, inv);
        dw.latent_mean = ew.latent_mean;
        dw.latent_std = ew.latent_std;
        if (encode) {
            ew.conv_in = conv3("encoder.conv1");
            const int dims[] = {96, 96, 192, 384, 384};
            int layer = 0;
            ew.stages.resize(4);
            for (int stage = 0; stage < 4; ++stage) {
                for (int block = 0; block < 2; ++block) {
                    codecs::WanVideoVAEStageBlockWeights b;
                    b.residual = residual("encoder.downsamples." + std::to_string(layer++),
                                          block ? dims[stage + 1] : dims[stage], dims[stage + 1]);
                    ew.stages[static_cast<size_t>(stage)].blocks.push_back(std::move(b));
                }
                if (stage < 3)
                    ew.stages[static_cast<size_t>(stage)].resample =
                        resample("encoder.downsamples." + std::to_string(layer++), stage > 0);
            }
            ew.middle1 = residual("encoder.middle.0", 384, 384);
            ew.middle_attention = attention("encoder.middle.1");
            ew.middle2 = residual("encoder.middle.2", 384, 384);
            ew.head_norm = rms("encoder.head.0.gamma");
            ew.head = conv3("encoder.head.2");
            ew.quant_conv = conv3("conv1");
        } else {
            dw.quant_conv = conv3("conv2");
            dw.conv_in = conv3("decoder.conv1");
            dw.middle1 = residual("decoder.middle.0", 384, 384);
            dw.middle_attention = attention("decoder.middle.1");
            dw.middle2 = residual("decoder.middle.2", 384, 384);
            const int dims[] = {384, 384, 384, 192, 96};
            int layer = 0;
            dw.stages.resize(4);
            for (int stage = 0; stage < 4; ++stage) {
                for (int block = 0; block < 3; ++block) {
                    codecs::WanVideoVAEStageBlockWeights b;
                    b.residual =
                        residual("decoder.upsamples." + std::to_string(layer++),
                                 block ? dims[stage + 1] : (stage ? dims[stage] / 2 : dims[stage]), dims[stage + 1]);
                    dw.stages[static_cast<size_t>(stage)].blocks.push_back(std::move(b));
                }
                if (stage < 3)
                    dw.stages[static_cast<size_t>(stage)].resample =
                        resample("decoder.upsamples." + std::to_string(layer++), stage < 2);
            }
            dw.head_norm = rms("decoder.head.0.gamma");
            dw.head = conv3("decoder.head.2");
        }
        const auto x = input("input", {encode ? 3 : 16, frames, height, width});
        std::vector<TensorValue> previous;
        TensorValue result;
        // The official Wan VAE initializes from one frame, then consumes groups
        // of four pixels frames (encoder) or one latent frame (decoder).
        for (int64_t begin = 0; begin < frames;) {
            const int64_t count = encode && begin ? std::min<int64_t>(4, frames - begin) : 1;
            auto slice = modules::SliceModule({1, begin, count}).build(ctx, x);
            // A one-frame slice still has gaps between channels in the parent video.
            slice = core::wrap_tensor(ggml_cont(ctx.ggml, slice.tensor), slice.shape);
            std::vector<TensorValue> next;
            codecs::WanVideoVAECacheBuildState cache;
            cache.input_caches = previous.empty() ? nullptr : &previous;
            cache.output_caches = &next;
            slice = encode ? codecs::WanVideoVAEEncoderModule(config).build_cached(ctx, slice, ew, cache)
                           : codecs::WanVideoVAEDecoderModule(config).build_cached(ctx, slice, dw, cache);
            result = result.valid() ? modules::ConcatModule({1}).build(ctx, result, slice) : slice;
            previous = std::move(next);
            begin += count;
        }
        finish(core::ensure_backend_addressable_layout(ctx, result));
    }

  private:
    Weights & weights;
    modules::Conv3dWeights conv3(const std::string & prefix) {
        return {weights.get(prefix + ".weight"), weights.get(prefix + ".bias")};
    }
    modules::Conv2dWeights conv2(const std::string & prefix) {
        return {weights.get(prefix + ".weight"), weights.get(prefix + ".bias")};
    }
    codecs::WanVideoRMSNorm3dWeights rms(const std::string & name) {
        auto x = weights.get(name);
        return {modules::ReshapeModule({TensorShape::from_dims({x.shape.num_elements()})}).build(ctx, x)};
    }
    codecs::WanVideoVAEResidualBlockWeights residual(const std::string & prefix, int64_t in, int64_t out) {
        codecs::WanVideoVAEResidualBlockWeights w;
        w.norm1 = rms(prefix + ".residual.0.gamma");
        w.norm2 = rms(prefix + ".residual.3.gamma");
        w.conv1 = conv3(prefix + ".residual.2");
        w.conv2 = conv3(prefix + ".residual.6");
        if (in != out)
            w.shortcut = conv3(prefix + ".shortcut");
        return w;
    }
    codecs::WanVideoVAEAttentionBlockWeights attention(const std::string & prefix) {
        codecs::WanVideoVAEAttentionBlockWeights w;
        w.norm = rms(prefix + ".norm.gamma");
        w.qkv = conv2(prefix + ".to_qkv");
        w.proj = conv2(prefix + ".proj");
        return w;
    }
    codecs::WanVideoVAEResampleWeights resample(const std::string & prefix, bool temporal) {
        codecs::WanVideoVAEResampleWeights w;
        w.spatial = conv2(prefix + ".resample.1");
        if (temporal)
            w.time = conv3(prefix + ".time_conv");
        return w;
    }
};
} // namespace engine::community_models::soulx_flashhead
