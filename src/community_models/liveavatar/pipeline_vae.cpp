#include "pipeline_internal.h"

#include "engine/framework/codecs/wan_video_vae_runtime.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/modules/weight_binding.h"

#include <ggml-alloc.h>
#include <ggml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::community_models::liveavatar {
namespace {

constexpr size_t kVAEGraphContextBytes = 1024ull * 1024ull * 1024ull;
constexpr int64_t kVAEDecodeSpatialTileSize = 192;
constexpr int64_t kVAEDecodeSpatialTileOverlap = 64;
constexpr int64_t kVAEDecodeSpatialTileThreshold = 512;
constexpr int64_t kVAEDecodeSpatialScale = 8;

struct LiveAvatarVideoTensor4D {
    int64_t channels = 3;
    int64_t frames = 0;
    int64_t height = 0;
    int64_t width = 0;
    std::vector<float> values;

    float & at(int64_t c, int64_t t, int64_t y, int64_t x) {
        return values[static_cast<size_t>(((c * frames + t) * height + y) * width + x)];
    }

    const float & at(int64_t c, int64_t t, int64_t y, int64_t x) const {
        return values[static_cast<size_t>(((c * frames + t) * height + y) * width + x)];
    }
};

std::pair<std::vector<int64_t>, std::vector<int64_t>> split_spatial_tiles(
    int64_t input_len,
    int64_t tile_size,
    int64_t min_overlap,
    int64_t ratio) {
    if (input_len <= 0 || tile_size <= 0 || min_overlap < 0 || ratio <= 0 || tile_size <= min_overlap) {
        throw std::runtime_error("LiveAvatar VAE tile configuration is invalid");
    }
    if (tile_size >= input_len) {
        return {{0}, {input_len}};
    }
    int64_t count = (input_len + tile_size - min_overlap - 1) / (tile_size - min_overlap);
    while (tile_size * count - min_overlap * (count - 1) < input_len) {
        ++count;
    }
    std::vector<int64_t> overlaps(static_cast<size_t>(count - 1), min_overlap);
    const int64_t remaining = tile_size * count - min_overlap * (count - 1) - input_len;
    for (int64_t i = 0; i < remaining / ratio; ++i) {
        overlaps[static_cast<size_t>(i % (count - 1))] += ratio;
    }
    std::vector<int64_t> starts{0};
    for (int64_t i = 0; i + 1 < count; ++i) {
        starts.push_back(starts.back() + tile_size - overlaps[static_cast<size_t>(i)]);
    }
    return {starts, overlaps};
}

void blend_height_from_previous(const LiveAvatarVideoTensor4D & previous, LiveAvatarVideoTensor4D & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.height, current.height});
    for (int64_t y = 0; y < extent; ++y) {
        const float weight = static_cast<float>(y) / static_cast<float>(extent);
        const int64_t prev_y = previous.height - extent + y;
        for (int64_t c = 0; c < current.channels; ++c) {
            for (int64_t t = 0; t < current.frames; ++t) {
                for (int64_t x = 0; x < current.width; ++x) {
                    current.at(c, t, y, x) =
                        previous.at(c, t, prev_y, x) * (1.0F - weight) + current.at(c, t, y, x) * weight;
                }
            }
        }
    }
}

void blend_width_from_previous(const LiveAvatarVideoTensor4D & previous, LiveAvatarVideoTensor4D & current, int64_t extent) {
    extent = std::min<int64_t>({extent, previous.width, current.width});
    for (int64_t x = 0; x < extent; ++x) {
        const float weight = static_cast<float>(x) / static_cast<float>(extent);
        const int64_t prev_x = previous.width - extent + x;
        for (int64_t c = 0; c < current.channels; ++c) {
            for (int64_t t = 0; t < current.frames; ++t) {
                for (int64_t y = 0; y < current.height; ++y) {
                    current.at(c, t, y, x) =
                        previous.at(c, t, y, prev_x) * (1.0F - weight) + current.at(c, t, y, x) * weight;
                }
            }
        }
    }
}


engine::core::TensorValue load_squeezed_f32(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t elements) {
    const auto metadata = source.require_metadata(name);
    int64_t actual_elements = 1;
    for (const int64_t dim : metadata.shape) {
        if (dim <= 0) {
            throw std::runtime_error("LiveAvatar tensor cannot be squeezed to expected shape: " + name);
        }
        actual_elements *= dim;
    }
    if (actual_elements != elements) {
        throw std::runtime_error("LiveAvatar tensor cannot be squeezed to expected shape: " + name);
    }
    return store.make_f32(
        engine::core::TensorShape::from_dims({elements}),
        source.require_f32(name, metadata.shape));
}

engine::codecs::WanVideoRMSNorm3dWeights load_vae_rms(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & name,
    int64_t channels) {
    engine::codecs::WanVideoRMSNorm3dWeights weights;
    weights.gamma = load_squeezed_f32(store, source, name, channels);
    return weights;
}

engine::modules::Conv3dWeights load_vae_conv3d(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t out_channels,
    int64_t in_channels,
    int64_t kernel_depth,
    int64_t kernel_height,
    int64_t kernel_width,
    bool use_bias) {
    return engine::modules::binding::conv3d_from_source(
        store,
        source,
        prefix,
        engine::assets::TensorStorageType::F32,
        out_channels,
        in_channels,
        kernel_depth,
        kernel_height,
        kernel_width,
        use_bias);
}

engine::codecs::WanVideoVAEResidualBlockWeights load_vae_residual(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t in_channels,
    int64_t out_channels) {
    engine::codecs::WanVideoVAEResidualBlockWeights weights;
    weights.norm1 = load_vae_rms(store, source, prefix + ".residual.0.gamma", in_channels);
    weights.conv1 = load_vae_conv3d(store, source, prefix + ".residual.2", out_channels, in_channels, 3, 3, 3, true);
    weights.norm2 = load_vae_rms(store, source, prefix + ".residual.3.gamma", out_channels);
    weights.conv2 = load_vae_conv3d(store, source, prefix + ".residual.6", out_channels, out_channels, 3, 3, 3, true);
    if (in_channels != out_channels) {
        weights.shortcut = load_vae_conv3d(store, source, prefix + ".shortcut", out_channels, in_channels, 1, 1, 1, true);
    }
    return weights;
}

engine::codecs::WanVideoVAEAttentionBlockWeights load_vae_attention(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels) {
    engine::codecs::WanVideoVAEAttentionBlockWeights weights;
    weights.norm = load_vae_rms(store, source, prefix + ".norm.gamma", channels);
    weights.qkv = engine::modules::binding::conv2d_from_source(
        store, source, prefix + ".to_qkv", engine::assets::TensorStorageType::F32, 3 * channels, channels, 1, 1, true);
    weights.proj = engine::modules::binding::conv2d_from_source(
        store, source, prefix + ".proj", engine::assets::TensorStorageType::F32, channels, channels, 1, 1, true);
    return weights;
}

engine::codecs::WanVideoVAEResampleWeights load_vae_resample(
    engine::core::BackendWeightStore & store,
    const engine::assets::TensorSource & source,
    const std::string & prefix,
    int64_t channels,
    bool upsample,
    bool temporal) {
    engine::codecs::WanVideoVAEResampleWeights weights;
    weights.spatial = engine::modules::binding::conv2d_from_source(
        store,
        source,
        prefix + ".resample.1",
        engine::assets::TensorStorageType::F32,
        upsample ? channels / 2 : channels,
        channels,
        3,
        3,
        true);
    if (temporal) {
        weights.time = load_vae_conv3d(
            store,
            source,
            prefix + ".time_conv",
            upsample ? channels * 2 : channels,
            channels,
            3,
            1,
            1,
            true);
    }
    return weights;
}

std::vector<float> inverse_std(const std::vector<float> & std_values) {
    std::vector<float> out;
    out.reserve(std_values.size());
    for (const float value : std_values) {
        if (value == 0.0F) {
            throw std::runtime_error("LiveAvatar VAE std contains zero");
        }
        out.push_back(1.0F / value);
    }
    return out;
}

struct LiveAvatarVAEWeights {
    std::shared_ptr<engine::core::BackendWeightStore> store;
    engine::codecs::WanVideoVAEEncoderWeights encoder;
    engine::codecs::WanVideoVAEDecoderWeights decoder;
};

LiveAvatarVAEWeights load_vae_weights(
    const engine::assets::TensorSource & source,
    ggml_backend_t backend,
    engine::core::BackendType backend_type,
    const engine::codecs::WanVideoVAEArchitectureConfig & arch) {
    LiveAvatarVAEWeights weights;
    weights.store = std::make_shared<engine::core::BackendWeightStore>(
        backend,
        backend_type,
        "liveavatar.vae.weights",
        384ull * 1024ull * 1024ull);

    const engine::codecs::WanVideoVAEConfig scale = engine::codecs::WanVideoVAELatentScaleModule::s2v_14b_config();
    const auto inv_std = inverse_std(scale.std);
    weights.encoder.latent_mean = weights.store->make_f32(engine::core::TensorShape::from_dims({arch.latent_channels}), scale.mean);
    weights.encoder.latent_std = weights.store->make_f32(engine::core::TensorShape::from_dims({arch.latent_channels}), inv_std);
    weights.decoder.latent_mean = weights.encoder.latent_mean;
    weights.decoder.latent_std = weights.encoder.latent_std;

    weights.encoder.conv_in = load_vae_conv3d(*weights.store, source, "encoder.conv1", 96, 3, 3, 3, 3, true);
    const std::vector<int64_t> encoder_dims{96, 96, 192, 384, 384};
    const std::vector<int64_t> encoder_resamples{2, 5, 8};
    weights.encoder.stages.resize(4);
    int64_t encoder_layer = 0;
    for (size_t stage = 0; stage < weights.encoder.stages.size(); ++stage) {
        int64_t in_channels = encoder_dims[stage];
        const int64_t out_channels = encoder_dims[stage + 1];
        for (int64_t block = 0; block < arch.num_res_blocks; ++block) {
            engine::codecs::WanVideoVAEStageBlockWeights block_weights;
            block_weights.residual = load_vae_residual(
                *weights.store,
                source,
                "encoder.downsamples." + std::to_string(encoder_layer++),
                in_channels,
                out_channels);
            weights.encoder.stages[stage].blocks.push_back(std::move(block_weights));
            in_channels = out_channels;
        }
        if (stage + 1 != weights.encoder.stages.size()) {
            weights.encoder.stages[stage].resample = load_vae_resample(
                *weights.store,
                source,
                "encoder.downsamples." + std::to_string(encoder_resamples[stage]),
                out_channels,
                false,
                arch.temporal_downsample[stage]);
            ++encoder_layer;
        }
    }
    weights.encoder.middle1 = load_vae_residual(*weights.store, source, "encoder.middle.0", 384, 384);
    weights.encoder.middle_attention = load_vae_attention(*weights.store, source, "encoder.middle.1", 384);
    weights.encoder.middle2 = load_vae_residual(*weights.store, source, "encoder.middle.2", 384, 384);
    weights.encoder.head_norm = load_vae_rms(*weights.store, source, "encoder.head.0.gamma", 384);
    weights.encoder.head = load_vae_conv3d(
        *weights.store, source, "encoder.head.2", 2 * arch.latent_channels, 384, 3, 3, 3, true);
    weights.encoder.quant_conv = load_vae_conv3d(
        *weights.store,
        source,
        "conv1",
        2 * arch.latent_channels,
        2 * arch.latent_channels,
        1,
        1,
        1,
        true);

    weights.decoder.quant_conv = load_vae_conv3d(
        *weights.store,
        source,
        "conv2",
        arch.latent_channels,
        arch.latent_channels,
        1,
        1,
        1,
        true);
    weights.decoder.conv_in = load_vae_conv3d(
        *weights.store, source, "decoder.conv1", 384, arch.latent_channels, 3, 3, 3, true);
    weights.decoder.middle1 = load_vae_residual(*weights.store, source, "decoder.middle.0", 384, 384);
    weights.decoder.middle_attention = load_vae_attention(*weights.store, source, "decoder.middle.1", 384);
    weights.decoder.middle2 = load_vae_residual(*weights.store, source, "decoder.middle.2", 384, 384);
    const std::vector<int64_t> decoder_dims{384, 384, 384, 192, 96};
    const std::vector<int64_t> decoder_resamples{3, 7, 11};
    weights.decoder.stages.resize(4);
    int64_t decoder_layer = 0;
    for (size_t stage = 0; stage < weights.decoder.stages.size(); ++stage) {
        int64_t in_channels = decoder_dims[stage];
        if (stage == 1 || stage == 2 || stage == 3) {
            in_channels /= 2;
        }
        const int64_t out_channels = decoder_dims[stage + 1];
        for (int64_t block = 0; block < arch.num_res_blocks + 1; ++block) {
            engine::codecs::WanVideoVAEStageBlockWeights block_weights;
            block_weights.residual = load_vae_residual(
                *weights.store,
                source,
                "decoder.upsamples." + std::to_string(decoder_layer++),
                in_channels,
                out_channels);
            weights.decoder.stages[stage].blocks.push_back(std::move(block_weights));
            in_channels = out_channels;
        }
        if (stage + 1 != weights.decoder.stages.size()) {
            weights.decoder.stages[stage].resample = load_vae_resample(
                *weights.store,
                source,
                "decoder.upsamples." + std::to_string(decoder_resamples[stage]),
                out_channels,
                true,
                arch.temporal_downsample[arch.temporal_downsample.size() - 1 - stage]);
            ++decoder_layer;
        }
    }
    weights.decoder.head_norm = load_vae_rms(*weights.store, source, "decoder.head.0.gamma", 96);
    weights.decoder.head = load_vae_conv3d(*weights.store, source, "decoder.head.2", 3, 96, 3, 3, 3, true);
    weights.store->upload();
    return weights;
}


enum class LiveAvatarVAEGraphMode {
    Encode,
    Decode,
};

class LiveAvatarVAEGraph {
public:
    LiveAvatarVAEGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarVAEWeights & weights,
        engine::codecs::WanVideoVAEArchitectureConfig config,
        LiveAvatarVAEGraphMode mode,
        engine::core::TensorShape input_shape)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          mode_(mode),
          input_shape_(input_shape) {
        build();
    }

    ~LiveAvatarVAEGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarVAEGraph(const LiveAvatarVAEGraph &) = delete;
    LiveAvatarVAEGraph & operator=(const LiveAvatarVAEGraph &) = delete;

    bool matches(
        LiveAvatarVAEGraphMode mode,
        const engine::core::TensorShape & shape,
        const engine::codecs::WanVideoVAEArchitectureConfig & config) const noexcept {
        if (mode_ != mode || input_shape_.rank != shape.rank) {
            return false;
        }
        if (config_.scale_encoded_latents != config.scale_encoded_latents ||
            config_.scale_decoded_latents != config.scale_decoded_latents ||
            config_.cache_raw_single_frame_inputs != config.cache_raw_single_frame_inputs ||
            config_.use_cuda_fast_lowerings != config.use_cuda_fast_lowerings ||
            config_.use_cuda_large_shape_lowerings != config.use_cuda_large_shape_lowerings ||
            config_.use_cuda_tile_f16_accum_output_lowering != config.use_cuda_tile_f16_accum_output_lowering) {
            return false;
        }
        for (size_t i = 0; i < shape.rank; ++i) {
            if (input_shape_.dims[i] != shape.dims[i]) {
                return false;
            }
        }
        return true;
    }

    const engine::core::TensorShape & output_shape() const noexcept {
        return output_shape_;
    }

    std::vector<float> run(const std::vector<float> & values) const {
        if (static_cast<int64_t>(values.size()) != input_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar VAE input payload size mismatch");
        }
        engine::core::write_tensor_f32(input_, values);
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.vae");
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar VAE graph compute failed");
        }
        return engine::core::read_tensor_f32(output_);
    }

private:
    void build() {
        ggml_init_params params{kVAEGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar VAE ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.vae", execution_.backend_type()};
        input_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, input_shape_);
        ggml_set_input(input_.tensor);
        const engine::core::TensorValue output =
            mode_ == LiveAvatarVAEGraphMode::Encode
                ? engine::codecs::WanVideoVAEEncoderModule(config_).build_initial(build_ctx, input_, weights_->encoder)
                : engine::codecs::WanVideoVAEDecoderModule(config_).build_initial(build_ctx, input_, weights_->decoder);
        output_ = output.tensor;
        output_shape_ = output.shape;
        ggml_set_output(output_);
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, output_);
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar VAE backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarVAEWeights * weights_ = nullptr;
    engine::codecs::WanVideoVAEArchitectureConfig config_;
    LiveAvatarVAEGraphMode mode_ = LiveAvatarVAEGraphMode::Encode;
    engine::core::TensorShape input_shape_;
    engine::core::TensorShape output_shape_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue input_;
    ggml_tensor * output_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

struct LiveAvatarVAEEncodeRun {
    std::vector<float> output;
    engine::core::TensorShape output_shape;
    std::vector<std::vector<float>> caches;
    std::vector<engine::core::TensorShape> cache_shapes;
};


class LiveAvatarVAEResidentCacheBank {
public:
    LiveAvatarVAEResidentCacheBank(
        engine::core::ExecutionContext & execution,
        std::vector<engine::core::TensorShape> shapes,
        std::vector<ggml_type> types)
        : execution_(execution),
          shapes_(std::move(shapes)),
          types_(std::move(types)) {
        if (shapes_.size() != types_.size()) {
            throw std::runtime_error("LiveAvatar VAE resident cache shape/type count mismatch");
        }
        ggml_init_params params{
            ggml_tensor_overhead() * (shapes_.size() + 8),
            nullptr,
            true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar VAE resident cache context initialization failed");
        }
        engine::core::ModuleBuildContext ctx{ctx_.get(), "liveavatar.vae_encoder_cache", execution_.backend_type()};
        values_.reserve(shapes_.size());
        for (size_t i = 0; i < shapes_.size(); ++i) {
            values_.push_back(engine::core::make_tensor(ctx, types_[i], shapes_[i]));
        }
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_.get(), execution_.backend());
        if (buffer_ == nullptr) {
            throw std::runtime_error("LiveAvatar VAE resident cache backend buffer allocation failed");
        }
    }

    ~LiveAvatarVAEResidentCacheBank() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
    }

    LiveAvatarVAEResidentCacheBank(const LiveAvatarVAEResidentCacheBank &) = delete;
    LiveAvatarVAEResidentCacheBank & operator=(const LiveAvatarVAEResidentCacheBank &) = delete;

    const std::vector<engine::core::TensorValue> & values() const noexcept {
        return values_;
    }

    const std::vector<engine::core::TensorShape> & shapes() const noexcept {
        return shapes_;
    }

    const std::vector<ggml_type> & types() const noexcept {
        return types_;
    }

private:
    engine::core::ExecutionContext & execution_;
    std::vector<engine::core::TensorShape> shapes_;
    std::vector<ggml_type> types_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<engine::core::TensorValue> values_;
};


class LiveAvatarCachedVAEEncodeGraph {
public:
    LiveAvatarCachedVAEEncodeGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarVAEWeights & weights,
        engine::codecs::WanVideoVAEArchitectureConfig config,
        engine::core::TensorShape input_shape,
        const LiveAvatarVAEResidentCacheBank * input_cache_bank,
        std::unique_ptr<LiveAvatarVAEResidentCacheBank> * output_cache_bank,
        bool emit_cache_outputs,
        bool cache_f16)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          input_shape_(input_shape),
          input_cache_bank_(input_cache_bank),
          output_cache_bank_(output_cache_bank),
          emit_cache_outputs_(emit_cache_outputs),
          cache_f16_(cache_f16) {
        if (input_cache_bank_ != nullptr) {
            cache_inputs_ = input_cache_bank_->values();
        }
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_graph_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarCachedVAEEncodeGraph() {
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
    }

    LiveAvatarCachedVAEEncodeGraph(const LiveAvatarCachedVAEEncodeGraph &) = delete;
    LiveAvatarCachedVAEEncodeGraph & operator=(const LiveAvatarCachedVAEEncodeGraph &) = delete;

    bool matches(
        const engine::core::TensorShape & input_shape,
        const std::vector<engine::core::TensorShape> & cache_shapes,
        const std::vector<ggml_type> & cache_types,
        const LiveAvatarVAEResidentCacheBank * input_cache_bank,
        const std::unique_ptr<LiveAvatarVAEResidentCacheBank> * output_cache_bank,
        bool emit_cache_outputs) const noexcept {
        if (!same_shape(input_shape_, input_shape) || cache_inputs_.size() != cache_shapes.size() ||
            input_cache_bank_ != input_cache_bank || output_cache_bank_ != output_cache_bank ||
            emit_cache_outputs_ != emit_cache_outputs) {
            return false;
        }
        for (size_t i = 0; i < cache_shapes.size(); ++i) {
            if (!same_shape(cache_inputs_[i].shape, cache_shapes[i]) || cache_inputs_[i].type != cache_types[i]) {
                return false;
            }
        }
        return true;
    }

    const std::vector<engine::core::TensorShape> & output_cache_shapes() const noexcept {
        return output_cache_shapes_;
    }

    const std::vector<ggml_type> & output_cache_types() const noexcept {
        return output_cache_types_;
    }

    const engine::core::TensorShape & output_shape() const noexcept {
        return output_shape_;
    }

    void copy_cache_outputs_to(LiveAvatarVAEResidentCacheBank & bank) const {
        if (bank.values().size() != cache_outputs_.size()) {
            throw std::runtime_error("LiveAvatar cached VAE encoder resident cache count mismatch");
        }
        for (size_t i = 0; i < cache_outputs_.size(); ++i) {
            const auto & dst = bank.values()[i];
            if (!same_shape(dst.shape, output_cache_shapes_[i]) || dst.type != output_cache_types_[i]) {
                throw std::runtime_error("LiveAvatar cached VAE encoder resident cache shape mismatch");
            }
            ggml_backend_tensor_copy(cache_outputs_[i], dst.tensor);
        }
    }

    LiveAvatarVAEEncodeRun run(const std::vector<float> & values) const {
        if (static_cast<int64_t>(values.size()) != input_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar cached VAE encoder input payload size mismatch");
        }
        engine::debug::trace_log_scalar("liveavatar.vae_encoder.input_shape", input_shape_.to_string());
        engine::debug::trace_log_scalar("liveavatar.vae_encoder.output_shape", output_shape_.to_string());
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(input_, values);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder.input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.vae_encoder");
        engine::debug::timing_log_scalar("liveavatar.vae_encoder.compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar cached VAE encoder graph compute failed");
        }
        LiveAvatarVAEEncodeRun out;
        const auto read_start = Clock::now();
        out.output = engine::core::read_tensor_f32(output_);
        out.output_shape = output_shape_;
        out.cache_shapes = output_cache_shapes_;
        engine::debug::timing_log_scalar("liveavatar.vae_encoder.output_read_ms", engine::debug::elapsed_ms(read_start));
        return out;
    }

private:
    void build() {
        ggml_init_params params{kVAEGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar cached VAE encoder ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.vae_encoder", execution_.backend_type()};
        input_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, input_shape_);
        ggml_set_input(input_.tensor);
        for (auto & cache_input : cache_inputs_) {
            ggml_set_input(cache_input.tensor);
        }
        std::vector<engine::core::TensorValue> cache_outputs;
        engine::codecs::WanVideoVAECacheBuildState cache_state;
        cache_state.input_caches = cache_inputs_.empty() ? nullptr : &cache_inputs_;
        cache_state.output_caches = (emit_cache_outputs_ || output_cache_bank_ != nullptr) ? &cache_outputs : nullptr;
        cache_state.raw_single_frame_inputs = config_.cache_raw_single_frame_inputs;
        cache_state.addressable_output_caches = output_cache_bank_ == nullptr;
        const auto output = engine::codecs::WanVideoVAEEncoderModule(config_).build_cached(
            build_ctx,
            input_,
            weights_->encoder,
            cache_state);
        if (!cache_inputs_.empty() && cache_state.cursor != cache_inputs_.size()) {
            throw std::runtime_error("LiveAvatar cached VAE encoder consumed an unexpected cache count");
        }
        output_ = output.tensor;
        output_shape_ = output.shape;
        ggml_set_output(output_);
        if (output_cache_bank_ != nullptr) {
            std::vector<engine::core::TensorShape> cache_shapes;
            std::vector<ggml_type> cache_types;
            cache_shapes.reserve(cache_outputs.size());
            cache_types.reserve(cache_outputs.size());
            for (const auto & cache_output : cache_outputs) {
                cache_shapes.push_back(cache_output.shape);
                cache_types.push_back(cache_f16_ ? GGML_TYPE_F16 : cache_output.type);
            }
            *output_cache_bank_ = std::make_unique<LiveAvatarVAEResidentCacheBank>(
                execution_,
                std::move(cache_shapes),
                std::move(cache_types));
            if ((*output_cache_bank_)->values().size() != cache_outputs.size()) {
                throw std::runtime_error("LiveAvatar cached VAE encoder output cache count mismatch");
            }
            output_cache_shapes_ = (*output_cache_bank_)->shapes();
            output_cache_types_ = (*output_cache_bank_)->types();
            for (size_t i = 0; i < cache_outputs.size(); ++i) {
                const auto & dst = (*output_cache_bank_)->values()[i];
                if (!same_shape(dst.shape, cache_outputs[i].shape)) {
                    throw std::runtime_error("LiveAvatar cached VAE encoder output cache shape mismatch");
                }
                auto * copied = ggml_cpy(ctx_.get(), cache_outputs[i].tensor, dst.tensor);
                ggml_set_output(copied);
                cache_copy_outputs_.push_back(copied);
            }
        } else if (emit_cache_outputs_) {
            cache_outputs_.reserve(cache_outputs.size());
            output_cache_shapes_.reserve(cache_outputs.size());
            for (const auto & cache_output : cache_outputs) {
                const auto cache_tensor = engine::core::make_tensor(
                    build_ctx,
                    cache_f16_ ? GGML_TYPE_F16 : cache_output.type,
                    cache_output.shape);
                auto * copied = ggml_cpy(ctx_.get(), cache_output.tensor, cache_tensor.tensor);
                cache_outputs_.push_back(copied);
                output_cache_shapes_.push_back(cache_tensor.shape);
                output_cache_types_.push_back(cache_tensor.type);
                ggml_set_output(copied);
            }
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        for (auto * cache_output : cache_outputs_) {
            ggml_build_forward_expand(graph_, cache_output);
        }
        for (auto * cache_copy_output : cache_copy_outputs_) {
            ggml_build_forward_expand(graph_, cache_copy_output);
        }
        ggml_build_forward_expand(graph_, output_);
        engine::debug::trace_log_scalar("liveavatar.vae_encoder.reserve_input_shape", input_shape_.to_string());
        engine::debug::trace_log_scalar("liveavatar.vae_encoder.reserve_output_shape", output_shape_.to_string());
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar cached VAE encoder backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarVAEWeights * weights_ = nullptr;
    engine::codecs::WanVideoVAEArchitectureConfig config_;
    engine::core::TensorShape input_shape_;
    const LiveAvatarVAEResidentCacheBank * input_cache_bank_ = nullptr;
    std::unique_ptr<LiveAvatarVAEResidentCacheBank> * output_cache_bank_ = nullptr;
    bool emit_cache_outputs_ = false;
    bool cache_f16_ = false;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue input_;
    std::vector<engine::core::TensorValue> cache_inputs_;
    ggml_tensor * output_ = nullptr;
    engine::core::TensorShape output_shape_;
    std::vector<ggml_tensor *> cache_outputs_;
    std::vector<ggml_tensor *> cache_copy_outputs_;
    std::vector<engine::core::TensorShape> output_cache_shapes_;
    std::vector<ggml_type> output_cache_types_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};

class LiveAvatarCachedVAEDecodeGraph {
public:
    LiveAvatarCachedVAEDecodeGraph(
        engine::core::ExecutionContext & execution,
        const LiveAvatarVAEWeights & weights,
        engine::codecs::WanVideoVAEArchitectureConfig config,
        engine::core::TensorShape input_shape,
        std::vector<engine::core::TensorShape> cache_shapes)
        : execution_(execution),
          weights_(&weights),
          config_(std::move(config)),
          input_shape_(input_shape),
          cache_shapes_(std::move(cache_shapes)) {
        const auto build_start = Clock::now();
        build();
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_graph_build_ms", engine::debug::elapsed_ms(build_start));
    }

    ~LiveAvatarCachedVAEDecodeGraph() {
        const auto release_start = Clock::now();
        if (execution_.backend() != nullptr && graph_ != nullptr) {
            engine::core::release_backend_graph_resources(execution_.backend_type(), execution_.backend(), graph_, true);
        }
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
        }
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_graph_release_ms", engine::debug::elapsed_ms(release_start));
    }

    LiveAvatarCachedVAEDecodeGraph(const LiveAvatarCachedVAEDecodeGraph &) = delete;
    LiveAvatarCachedVAEDecodeGraph & operator=(const LiveAvatarCachedVAEDecodeGraph &) = delete;

    bool matches(
        const engine::core::TensorShape & input_shape,
        const std::vector<engine::core::TensorShape> & cache_shapes) const noexcept {
        if (!same_shape(input_shape_, input_shape) || cache_shapes_.size() != cache_shapes.size()) {
            return false;
        }
        for (size_t i = 0; i < cache_shapes.size(); ++i) {
            if (!same_shape(cache_shapes_[i], cache_shapes[i])) {
                return false;
            }
        }
        return true;
    }

    const std::vector<engine::core::TensorShape> & output_cache_shapes() const noexcept {
        return output_cache_shapes_;
    }

    const engine::core::TensorShape & output_shape() const noexcept {
        return output_shape_;
    }

    void copy_cache_outputs_to_inputs_from(const LiveAvatarCachedVAEDecodeGraph & source) const {
        if (cache_inputs_.size() != source.cache_outputs_.size()) {
            throw std::runtime_error("LiveAvatar cached VAE decoder resident cache count mismatch");
        }
        for (size_t i = 0; i < cache_inputs_.size(); ++i) {
            if (!same_shape(cache_inputs_[i].shape, source.output_cache_shapes_[i])) {
                throw std::runtime_error("LiveAvatar cached VAE decoder resident cache shape mismatch");
            }
            ggml_backend_tensor_copy(source.cache_outputs_[i], cache_inputs_[i].tensor);
        }
    }

    LiveAvatarVAEEncodeRun run(const std::vector<float> & values) const {
        if (static_cast<int64_t>(values.size()) != input_shape_.num_elements()) {
            throw std::runtime_error("LiveAvatar cached VAE decoder input payload size mismatch");
        }
        const auto write_start = Clock::now();
        engine::core::write_tensor_f32(input_, values);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_input_write_ms", engine::debug::elapsed_ms(write_start));
        engine::core::set_backend_threads(execution_.backend(), std::max(1, execution_.config().threads));
        const auto compute_start = Clock::now();
        const ggml_status status = engine::core::compute_graph(execution_, graph_, plan_, "liveavatar.vae_decoder");
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_compute_ms", engine::debug::elapsed_ms(compute_start));
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("LiveAvatar cached VAE decoder graph compute failed");
        }
        LiveAvatarVAEEncodeRun out;
        const auto read_start = Clock::now();
        out.output = engine::core::read_tensor_f32(output_);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_output_read_ms", engine::debug::elapsed_ms(read_start));
        out.output_shape = output_shape_;
        out.cache_shapes = output_cache_shapes_;
        return out;
    }

private:
    void build() {
        ggml_init_params params{kVAEGraphContextBytes, nullptr, true};
        ctx_.reset(ggml_init(params));
        if (ctx_ == nullptr) {
            throw std::runtime_error("LiveAvatar cached VAE decoder ggml context initialization failed");
        }
        engine::core::ModuleBuildContext build_ctx{ctx_.get(), "liveavatar.vae_decoder", execution_.backend_type()};
        input_ = engine::core::make_tensor(build_ctx, GGML_TYPE_F32, input_shape_);
        ggml_set_input(input_.tensor);
        cache_inputs_.reserve(cache_shapes_.size());
        for (const auto & shape : cache_shapes_) {
            cache_inputs_.push_back(engine::core::make_tensor(build_ctx, GGML_TYPE_F32, shape));
            ggml_set_input(cache_inputs_.back().tensor);
        }
        std::vector<engine::core::TensorValue> cache_outputs;
        engine::codecs::WanVideoVAECacheBuildState cache_state;
        cache_state.input_caches = cache_inputs_.empty() ? nullptr : &cache_inputs_;
        cache_state.output_caches = &cache_outputs;
        cache_state.raw_single_frame_inputs = config_.cache_raw_single_frame_inputs;
        const auto output = engine::codecs::WanVideoVAEDecoderModule(config_).build_cached(
            build_ctx,
            input_,
            weights_->decoder,
            cache_state);
        if (!cache_inputs_.empty() && cache_state.cursor != cache_inputs_.size()) {
            throw std::runtime_error("LiveAvatar cached VAE decoder consumed an unexpected cache count");
        }
        output_ = output.tensor;
        output_shape_ = output.shape;
        ggml_set_output(output_);
        cache_outputs_.reserve(cache_outputs.size());
        output_cache_shapes_.reserve(cache_outputs.size());
        for (const auto & cache_output : cache_outputs) {
            cache_outputs_.push_back(cache_output.tensor);
            output_cache_shapes_.push_back(cache_output.shape);
            ggml_set_output(cache_output.tensor);
        }
        graph_ = ggml_new_graph_custom(ctx_.get(), 524288, false);
        ggml_build_forward_expand(graph_, output_);
        for (auto * cache_output : cache_outputs_) {
            ggml_build_forward_expand(graph_, cache_output);
        }
        gallocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_.backend()));
        if (gallocr_ == nullptr ||
            !ggml_gallocr_reserve(gallocr_, graph_) ||
            !ggml_gallocr_alloc_graph(gallocr_, graph_)) {
            throw std::runtime_error("LiveAvatar cached VAE decoder backend buffer allocation failed");
        }
        engine::core::prepare_host_graph_plan(execution_, graph_, plan_);
    }

    engine::core::ExecutionContext & execution_;
    const LiveAvatarVAEWeights * weights_ = nullptr;
    engine::codecs::WanVideoVAEArchitectureConfig config_;
    engine::core::TensorShape input_shape_;
    std::vector<engine::core::TensorShape> cache_shapes_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx_;
    engine::core::TensorValue input_;
    std::vector<engine::core::TensorValue> cache_inputs_;
    ggml_tensor * output_ = nullptr;
    engine::core::TensorShape output_shape_;
    std::vector<ggml_tensor *> cache_outputs_;
    std::vector<engine::core::TensorShape> output_cache_shapes_;
    ggml_cgraph * graph_ = nullptr;
    ggml_gallocr_t gallocr_ = nullptr;
    mutable engine::core::HostGraphPlan plan_;
};



}  // namespace

struct LiveAvatarVAERuntime::Data {
    explicit Data(std::shared_ptr<const LiveAvatarAssets> assets)
        : assets_(require_assets(std::move(assets))) {
        config.scale_encoded_latents = false;
        config.scale_decoded_latents = false;
        config.cache_raw_single_frame_inputs = true;
        config.use_cuda_fast_lowerings = true;
    }

    LiveAvatarVAEWeights & ensure_weights(engine::core::ExecutionContext & execution) {
        if (!weights.has_value()) {
            weights = load_vae_weights(*assets_->vae_weights, execution.backend(), execution.backend_type(), config);
            assets_->vae_weights->release_storage();
        }
        return *weights;
    }

    void release_graphs() {
        active_graph.reset();
    }

    void release(engine::core::ExecutionContext & execution) {
        release_graphs();
        weights.reset();
        engine::core::trim_backend_pools(execution.backend());
    }

    LiveAvatarVAEGraph & graph(
        engine::core::ExecutionContext & execution,
        LiveAvatarVAEGraphMode mode,
        const engine::core::TensorShape & input_shape,
        const engine::codecs::WanVideoVAEArchitectureConfig & graph_config) {
        if (active_graph != nullptr && active_graph->matches(mode, input_shape, graph_config)) {
            return *active_graph;
        }
        active_graph = std::make_unique<LiveAvatarVAEGraph>(
            execution,
            ensure_weights(execution),
            graph_config,
            mode,
            input_shape);
        return *active_graph;
    }

    std::vector<float> encode_raw(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        engine::core::TensorShape * output_shape) {
        auto raw_config = config;
        raw_config.scale_encoded_latents = false;
        auto raw_graph = LiveAvatarVAEGraph(
            execution,
            ensure_weights(execution),
            raw_config,
            LiveAvatarVAEGraphMode::Encode,
            engine::core::TensorShape::from_dims({channels, frames, height, width}));
        auto out = raw_graph.run(values);
        if (output_shape != nullptr) {
            *output_shape = raw_graph.output_shape();
        }
        return out;
    }

    std::vector<float> decode_cached_untiled(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        const engine::codecs::WanVideoVAEArchitectureConfig & run_config,
        engine::core::TensorShape * output_shape) {
        if (channels != assets_->config.latent_channels || frames <= 0 || height <= 0 || width <= 0 ||
            static_cast<int64_t>(values.size()) != channels * frames * height * width) {
            throw std::runtime_error("LiveAvatar cached VAE decode input shape mismatch");
        }
        std::vector<std::vector<float>> output_chunks;
        std::vector<engine::core::TensorShape> output_chunk_shapes;
        engine::core::TensorShape out_shape;
        int64_t out_frames = 0;
        double graph_run_ms = 0.0;
        double weight_ensure_ms = 0.0;
        double graph_ctor_ms = 0.0;
        double graph_reset_ms = 0.0;
        double cache_copy_ms = 0.0;
        double append_ms = 0.0;
        const auto first_weight_start = Clock::now();
        const auto & first_weights = ensure_weights(execution);
        weight_ensure_ms += engine::debug::elapsed_ms(first_weight_start);
        const auto first_ctor_start = Clock::now();
        auto first_graph = std::make_unique<LiveAvatarCachedVAEDecodeGraph>(
            execution,
            first_weights,
            run_config,
            engine::core::TensorShape::from_dims({channels, 1, height, width}),
            std::vector<engine::core::TensorShape>{});
        graph_ctor_ms += engine::debug::elapsed_ms(first_ctor_start);
        std::unique_ptr<LiveAvatarCachedVAEDecodeGraph> cached_graph_a;
        std::unique_ptr<LiveAvatarCachedVAEDecodeGraph> cached_graph_b;
        LiveAvatarCachedVAEDecodeGraph * previous_graph = first_graph.get();
        for (int64_t frame = 0; frame < frames;) {
            const int64_t chunk_frames = frame == 0 ? 1 : std::min<int64_t>(2, frames - frame);
            LiveAvatarCachedVAEDecodeGraph * current_graph = first_graph.get();
            if (frame > 0) {
                const auto input_shape = engine::core::TensorShape::from_dims({channels, chunk_frames, height, width});
                const auto & cache_shapes = previous_graph->output_cache_shapes();
                std::unique_ptr<LiveAvatarCachedVAEDecodeGraph> * graph_slot = nullptr;
                for (auto * candidate : {&cached_graph_a, &cached_graph_b}) {
                    if (candidate->get() != previous_graph &&
                        *candidate != nullptr &&
                        (*candidate)->matches(input_shape, cache_shapes)) {
                        graph_slot = candidate;
                        break;
                    }
                }
                if (graph_slot == nullptr) {
                    graph_slot = cached_graph_a.get() == previous_graph ? &cached_graph_b : &cached_graph_a;
                    const auto reset_start = Clock::now();
                    graph_slot->reset();
                    graph_reset_ms += engine::debug::elapsed_ms(reset_start);
                    const auto weight_start = Clock::now();
                    const auto & decode_weights = ensure_weights(execution);
                    weight_ensure_ms += engine::debug::elapsed_ms(weight_start);
                    const auto ctor_start = Clock::now();
                    *graph_slot = std::make_unique<LiveAvatarCachedVAEDecodeGraph>(
                        execution,
                        decode_weights,
                        run_config,
                        input_shape,
                        cache_shapes);
                    graph_ctor_ms += engine::debug::elapsed_ms(ctor_start);
                }
                current_graph = graph_slot->get();
                const auto cache_copy_start = Clock::now();
                current_graph->copy_cache_outputs_to_inputs_from(*previous_graph);
                cache_copy_ms += engine::debug::elapsed_ms(cache_copy_start);
                if (previous_graph == first_graph.get()) {
                    const auto reset_start = Clock::now();
                    first_graph.reset();
                    graph_reset_ms += engine::debug::elapsed_ms(reset_start);
                }
            }
            const auto run_start = Clock::now();
            auto run = current_graph->run(slice_video_time(values, channels, frames, height, width, frame, chunk_frames));
            graph_run_ms += engine::debug::elapsed_ms(run_start);
            if (run.output_shape.rank != 4 || run.output_shape.dims[1] <= 0) {
                throw std::runtime_error("LiveAvatar cached VAE decode produced invalid output shape");
            }
            if (output_chunks.empty()) {
                out_shape = run.output_shape;
            } else {
                if (run.output_shape.dims[0] != out_shape.dims[0] ||
                    run.output_shape.dims[2] != out_shape.dims[2] ||
                    run.output_shape.dims[3] != out_shape.dims[3]) {
                    throw std::runtime_error("LiveAvatar cached VAE decode output shape changed across chunks");
                }
            }
            out_frames += run.output_shape.dims[1];
            output_chunk_shapes.push_back(run.output_shape);
            output_chunks.push_back(std::move(run.output));
            previous_graph = current_graph;
            frame += chunk_frames;
        }
        out_shape.dims[1] = out_frames;
        std::vector<float> out(static_cast<size_t>(out_shape.num_elements()));
        const auto append_start = Clock::now();
        int64_t out_frame = 0;
        for (size_t i = 0; i < output_chunks.size(); ++i) {
            copy_video_time(
                out,
                out_shape.dims[0],
                out_shape.dims[1],
                out_shape.dims[2],
                out_shape.dims[3],
                out_frame,
                output_chunks[i],
                output_chunk_shapes[i].dims[1]);
            out_frame += output_chunk_shapes[i].dims[1];
        }
        append_ms += engine::debug::elapsed_ms(append_start);
        if (output_shape != nullptr) {
            *output_shape = out_shape;
        }
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_graph_run_ms", graph_run_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_weight_ensure_ms", weight_ensure_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_graph_ctor_ms", graph_ctor_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_graph_reset_ms", graph_reset_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_cache_copy_ms", cache_copy_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_decoder_append_ms", append_ms);
        return out;
    }

    std::vector<float> decode_spatial_tiled(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t tile_size,
        const engine::codecs::WanVideoVAEArchitectureConfig & run_config,
        engine::core::TensorShape * output_shape) {
        if (tile_size <= 0) {
            tile_size = kVAEDecodeSpatialTileSize;
        }
        if (tile_size <= kVAEDecodeSpatialTileOverlap || tile_size % kVAEDecodeSpatialScale != 0) {
            throw std::runtime_error("LiveAvatar VAE decoder tile size must be a multiple of 8 and larger than overlap");
        }
        const int64_t decoded_height = height * kVAEDecodeSpatialScale;
        const int64_t decoded_width = width * kVAEDecodeSpatialScale;
        const auto [y_starts, y_overlaps] = split_spatial_tiles(
            decoded_height,
            tile_size,
            kVAEDecodeSpatialTileOverlap,
            kVAEDecodeSpatialScale);
        const auto [x_starts, x_overlaps] = split_spatial_tiles(
            decoded_width,
            tile_size,
            kVAEDecodeSpatialTileOverlap,
            kVAEDecodeSpatialScale);
        engine::debug::trace_log_scalar("liveavatar.vae_decoder.tile_size", tile_size);
        engine::debug::trace_log_scalar("liveavatar.vae_decoder.tile_rows", static_cast<int64_t>(y_starts.size()));
        engine::debug::trace_log_scalar("liveavatar.vae_decoder.tile_cols", static_cast<int64_t>(x_starts.size()));

        LiveAvatarVideoTensor4D out;
        out.height = decoded_height;
        out.width = decoded_width;
        std::vector<LiveAvatarVideoTensor4D> previous_raw_row;
        previous_raw_row.reserve(x_starts.size());
        int64_t y_out = 0;
        for (size_t yi = 0; yi < y_starts.size(); ++yi) {
            std::vector<LiveAvatarVideoTensor4D> current_raw_row;
            current_raw_row.reserve(x_starts.size());
            int64_t x_out = 0;
            for (size_t xi = 0; xi < x_starts.size(); ++xi) {
                const int64_t latent_y = y_starts[yi] / kVAEDecodeSpatialScale;
                const int64_t latent_x = x_starts[xi] / kVAEDecodeSpatialScale;
                const int64_t tile_decoded_height =
                    std::min<int64_t>(tile_size, decoded_height - y_starts[yi]);
                const int64_t tile_decoded_width =
                    std::min<int64_t>(tile_size, decoded_width - x_starts[xi]);
                const int64_t tile_latent_height = tile_decoded_height / kVAEDecodeSpatialScale;
                const int64_t tile_latent_width = tile_decoded_width / kVAEDecodeSpatialScale;
                engine::core::TensorShape tile_shape;
                auto raw_values = decode_cached_untiled(
                    execution,
                    slice_video_spatial(
                        values,
                        channels,
                        frames,
                        height,
                        width,
                        latent_y,
                        tile_latent_height,
                        latent_x,
                        tile_latent_width),
                    channels,
                    frames,
                    tile_latent_height,
                    tile_latent_width,
                    run_config,
                    &tile_shape);
                if (tile_shape.rank != 4 || tile_shape.dims[0] != 3 ||
                    tile_shape.dims[2] != tile_decoded_height ||
                    tile_shape.dims[3] != tile_decoded_width) {
                    throw std::runtime_error("LiveAvatar tiled VAE decode output shape mismatch");
                }
                LiveAvatarVideoTensor4D raw;
                raw.channels = tile_shape.dims[0];
                raw.frames = tile_shape.dims[1];
                raw.height = tile_shape.dims[2];
                raw.width = tile_shape.dims[3];
                raw.values = std::move(raw_values);
                if (out.frames == 0) {
                    out.channels = raw.channels;
                    out.frames = raw.frames;
                    out.values.resize(static_cast<size_t>(out.channels * out.frames * out.height * out.width));
                } else if (raw.frames != out.frames || raw.channels != out.channels) {
                    throw std::runtime_error("LiveAvatar tiled VAE decode frame count changed across tiles");
                }

                auto tile = raw;
                if (yi > 0) {
                    blend_height_from_previous(previous_raw_row[xi], tile, y_overlaps[yi - 1]);
                }
                if (xi > 0) {
                    blend_width_from_previous(current_raw_row[xi - 1], tile, x_overlaps[xi - 1]);
                }
                const int64_t keep_h = yi + 1 < y_starts.size() ? tile.height - y_overlaps[yi] : tile.height;
                const int64_t keep_w = xi + 1 < x_starts.size() ? tile.width - x_overlaps[xi] : tile.width;
                for (int64_t c = 0; c < tile.channels; ++c) {
                    for (int64_t t = 0; t < tile.frames; ++t) {
                        for (int64_t y = 0; y < keep_h; ++y) {
                            for (int64_t x = 0; x < keep_w; ++x) {
                                out.at(c, t, y_out + y, x_out + x) = tile.at(c, t, y, x);
                            }
                        }
                    }
                }
                x_out += keep_w;
                current_raw_row.push_back(std::move(raw));
            }
            y_out += yi + 1 < y_starts.size() ? current_raw_row.front().height - y_overlaps[yi] : current_raw_row.front().height;
            previous_raw_row = std::move(current_raw_row);
        }
        if (output_shape != nullptr) {
            *output_shape = engine::core::TensorShape::from_dims({out.channels, out.frames, out.height, out.width});
        }
        return std::move(out.values);
    }

    std::vector<float> decode(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t tile_size,
        engine::core::TensorShape * output_shape) {
        if (channels != assets_->config.latent_channels || frames <= 0 || height <= 0 || width <= 0 ||
            static_cast<int64_t>(values.size()) != channels * frames * height * width) {
            throw std::runtime_error("LiveAvatar VAE decode input shape mismatch");
        }
        if (height * kVAEDecodeSpatialScale > kVAEDecodeSpatialTileThreshold ||
            width * kVAEDecodeSpatialScale > kVAEDecodeSpatialTileThreshold) {
            auto run_config = config;
            if (tile_size > 0) {
                run_config.use_cuda_large_shape_lowerings = true;
                run_config.use_cuda_tile_f16_accum_output_lowering = true;
            }
            return decode_spatial_tiled(execution, values, channels, frames, height, width, tile_size, run_config, output_shape);
        }
        return decode_cached_untiled(execution, values, channels, frames, height, width, config, output_shape);
    }

    std::vector<float> encode_cached_untiled(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t chunk_size,
        bool cache_f16,
        const engine::codecs::WanVideoVAEArchitectureConfig & run_config,
        engine::core::TensorShape * output_shape) {
        if (channels != 3 || frames <= 0 || height <= 0 || width <= 0 ||
            static_cast<int64_t>(values.size()) != channels * frames * height * width) {
            throw std::runtime_error("LiveAvatar cached VAE encode input shape mismatch");
        }
        if (chunk_size <= 0) {
            throw std::runtime_error("LiveAvatar cached VAE encode chunk size must be positive");
        }
        double graph_run_ms = 0.0;
        double slice_ms = 0.0;
        double append_ms = 0.0;
        double cache_update_ms = 0.0;

        const auto first_slice_start = Clock::now();
        auto first_values = slice_video_frames(values, channels, frames, height, width, 0, 1);
        slice_ms += engine::debug::elapsed_ms(first_slice_start);
        const auto first_weight_start = Clock::now();
        const auto & first_weights = ensure_weights(execution);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_weight_ensure_ms", engine::debug::elapsed_ms(first_weight_start));
        const auto first_ctor_start = Clock::now();
        auto first_graph = std::make_unique<LiveAvatarCachedVAEEncodeGraph>(
            execution,
            first_weights,
            run_config,
            engine::core::TensorShape::from_dims({channels, 1, height, width}),
            nullptr,
            nullptr,
            frames > 1,
            cache_f16);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_graph_ctor_ms", engine::debug::elapsed_ms(first_ctor_start));
        const auto first_run_start = Clock::now();
        auto first = first_graph->run(first_values);
        graph_run_ms += engine::debug::elapsed_ms(first_run_start);
        std::vector<float> out = std::move(first.output);
        engine::core::TensorShape out_shape = first.output_shape;
        int64_t out_frames = out_shape.dims[1];
        std::unique_ptr<LiveAvatarVAEResidentCacheBank> cache_bank_a;
        std::unique_ptr<LiveAvatarVAEResidentCacheBank> cache_bank_b;
        LiveAvatarVAEResidentCacheBank * previous_cache_bank = nullptr;
        double weight_ensure_ms = 0.0;
        double graph_ctor_ms = 0.0;
        double graph_reset_ms = 0.0;
        double cache_copy_ms = 0.0;
        if (frames > 1) {
            const auto cache_copy_start = Clock::now();
            cache_bank_a = std::make_unique<LiveAvatarVAEResidentCacheBank>(
                execution,
                first_graph->output_cache_shapes(),
                first_graph->output_cache_types());
            first_graph->copy_cache_outputs_to(*cache_bank_a);
            previous_cache_bank = cache_bank_a.get();
            cache_copy_ms += engine::debug::elapsed_ms(cache_copy_start);
            const auto reset_start = Clock::now();
            first_graph.reset();
            graph_reset_ms += engine::debug::elapsed_ms(reset_start);
            engine::core::trim_backend_pools(execution.backend());
        }
        for (int64_t start = 1; start < frames; start += chunk_size) {
            const int64_t chunk_frames = std::min<int64_t>(chunk_size, frames - start);
            const bool emit_cache_outputs = start + chunk_frames < frames;
            const auto slice_start = Clock::now();
            auto chunk_values = slice_video_frames(values, channels, frames, height, width, start, chunk_frames);
            slice_ms += engine::debug::elapsed_ms(slice_start);
            const auto input_shape = engine::core::TensorShape::from_dims({channels, chunk_frames, height, width});
            if (previous_cache_bank == nullptr) {
                throw std::runtime_error("LiveAvatar cached VAE encoder missing resident cache bank");
            }
            std::unique_ptr<LiveAvatarVAEResidentCacheBank> * output_cache_slot = nullptr;
            if (emit_cache_outputs) {
                output_cache_slot = previous_cache_bank == cache_bank_a.get() ? &cache_bank_b : &cache_bank_a;
                const auto reset_start = Clock::now();
                output_cache_slot->reset();
                graph_reset_ms += engine::debug::elapsed_ms(reset_start);
            }
            const auto weight_start = Clock::now();
            const auto & encode_weights = ensure_weights(execution);
            weight_ensure_ms += engine::debug::elapsed_ms(weight_start);
            const auto ctor_start = Clock::now();
            LiveAvatarCachedVAEEncodeGraph graph(
                execution,
                encode_weights,
                run_config,
                input_shape,
                previous_cache_bank,
                output_cache_slot,
                false,
                cache_f16);
            graph_ctor_ms += engine::debug::elapsed_ms(ctor_start);
            const auto run_start = Clock::now();
            auto run = graph.run(chunk_values);
            graph_run_ms += engine::debug::elapsed_ms(run_start);
            if (run.output_shape.dims[0] != out_shape.dims[0] ||
                run.output_shape.dims[2] != out_shape.dims[2] ||
                run.output_shape.dims[3] != out_shape.dims[3]) {
                throw std::runtime_error("LiveAvatar cached VAE encode output shape changed across chunks");
            }
            const auto append_start = Clock::now();
            append_video_time(
                out,
                out_shape.dims[0],
                out_frames,
                run.output_shape.dims[1],
                out_shape.dims[2],
                out_shape.dims[3],
                run.output);
            append_ms += engine::debug::elapsed_ms(append_start);
            out_frames += run.output_shape.dims[1];
            out_shape.dims[1] = out_frames;
            if (emit_cache_outputs) {
                previous_cache_bank = output_cache_slot->get();
            }
        }
        if (output_shape != nullptr) {
            *output_shape = out_shape;
        }
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_graph_run_ms", graph_run_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_slice_ms", slice_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_append_ms", append_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_cache_update_ms", cache_update_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_weight_ensure_ms", weight_ensure_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_graph_ctor_ms", graph_ctor_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_graph_reset_ms", graph_reset_ms);
        engine::debug::timing_log_scalar("liveavatar.vae_encoder_cache_copy_ms", cache_copy_ms);
        return out;
    }

    std::vector<float> encode_cached(
        engine::core::ExecutionContext & execution,
        const std::vector<float> & values,
        int64_t channels,
        int64_t frames,
        int64_t height,
        int64_t width,
        int64_t chunk_size,
        bool cache_f16,
        engine::core::TensorShape * output_shape) {
        if (channels != 3 || frames <= 0 || height <= 0 || width <= 0 ||
            static_cast<int64_t>(values.size()) != channels * frames * height * width) {
            throw std::runtime_error("LiveAvatar cached VAE encode input shape mismatch");
        }
        auto run_config = config;
        if (cache_f16) {
            run_config.use_cuda_large_shape_lowerings = true;
            run_config.use_cuda_tile_f16_accum_output_lowering = true;
        }
        return encode_cached_untiled(execution, values, channels, frames, height, width, chunk_size, cache_f16, run_config, output_shape);
    }

    std::shared_ptr<const LiveAvatarAssets> assets_;
    engine::codecs::WanVideoVAEArchitectureConfig config;
    std::optional<LiveAvatarVAEWeights> weights;
    std::unique_ptr<LiveAvatarVAEGraph> active_graph;
};

LiveAvatarVAERuntime::LiveAvatarVAERuntime(std::shared_ptr<const LiveAvatarAssets> assets)
    : data_(std::make_unique<Data>(std::move(assets))) {}

LiveAvatarVAERuntime::~LiveAvatarVAERuntime() = default;

void LiveAvatarVAERuntime::release(engine::core::ExecutionContext & execution) {
    data_->release(execution);
}

std::vector<float> LiveAvatarVAERuntime::encode_raw(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    engine::core::TensorShape * output_shape) {
    return data_->encode_raw(execution, values, channels, frames, height, width, output_shape);
}

std::vector<float> LiveAvatarVAERuntime::encode_cached(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t chunk_size,
    bool cache_f16,
    engine::core::TensorShape * output_shape) {
    return data_->encode_cached(execution, values, channels, frames, height, width, chunk_size, cache_f16, output_shape);
}

std::vector<float> LiveAvatarVAERuntime::decode(
    engine::core::ExecutionContext & execution,
    const std::vector<float> & values,
    int64_t channels,
    int64_t frames,
    int64_t height,
    int64_t width,
    int64_t tile_size,
    engine::core::TensorShape * output_shape) {
    return data_->decode(execution, values, channels, frames, height, width, tile_size, output_shape);
}

LiveAvatarVAEProbeResult run_liveavatar_vae_encode_probe(
    const LiveAvatarAssets & assets,
    engine::core::ExecutionContext & execution,
    const LiveAvatarVAEProbeRequest & request) {
    if (request.channels != 3 || request.frames <= 0 || request.height <= 0 || request.width <= 0 ||
        static_cast<int64_t>(request.input.size()) != request.channels * request.frames * request.height * request.width) {
        throw std::runtime_error("LiveAvatar VAE probe input shape mismatch");
    }

    auto config = engine::codecs::WanVideoVAEArchitectureConfig{};
    config.scale_encoded_latents = request.scale_encoded_latents;
    auto weights = load_vae_weights(*assets.vae_weights, execution.backend(), execution.backend_type(), config);
    assets.vae_weights->release_storage();

    LiveAvatarVAEGraph graph(
        execution,
        weights,
        config,
        LiveAvatarVAEGraphMode::Encode,
        engine::core::TensorShape::from_dims({
            request.channels,
            request.frames,
            request.height,
            request.width,
        }));

    LiveAvatarVAEProbeResult result;
    result.output = graph.run(request.input);
    result.output_shape = graph.output_shape();
    return result;
}


}  // namespace engine::community_models::liveavatar
