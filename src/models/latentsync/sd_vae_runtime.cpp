#include "engine/models/latentsync/sd_vae_runtime.h"
#include "engine/framework/debug/trace.h"

#include <ggml-alloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace engine::models::latentsync {
namespace {
constexpr float kLatentScale = 0.18215F;

class LatentSyncSdVaeGraph {
public:
    LatentSyncSdVaeGraph(core::ExecutionContext & execution, const SdVaeTowerWeights & weights,
        int64_t resolution, bool encoder) : backend_(execution.backend()) {
        constexpr size_t nodes = 4096;
        context_.reset(ggml_init({nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(nodes, false), nullptr, true}));
        if (!context_) throw std::runtime_error("LatentSync VAE graph context allocation failed");
        core::ModuleBuildContext ctx{context_.get(), "latentsync.vae", execution.backend_type()};
        const int64_t side = encoder ? resolution : resolution / 8;
        input_ = core::make_tensor(ctx, GGML_TYPE_F32, core::TensorShape::from_dims({1, encoder ? 3 : 4, side, side}));
        ggml_set_input(input_.tensor);
        const LatentSyncSdVae vae({});
        output_ = encoder ? vae.build_encoder(ctx, input_, weights) : vae.build_decoder(ctx, input_, weights);
        ggml_set_output(output_.tensor);
        graph_ = ggml_new_graph_custom(context_.get(), nodes, false);
        ggml_build_forward_expand(graph_, output_.tensor);
        allocator_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_)));
        if (!allocator_ || !ggml_gallocr_alloc_graph(allocator_.get(), graph_)) throw std::runtime_error("LatentSync VAE graph allocation failed");
        debug::timing_log_scalar(encoder ? "latentsync.vae_encoder.graph_buffer_mb" : "latentsync.vae_decoder.graph_buffer_mb",
            ggml_gallocr_get_buffer_size(allocator_.get(), 0) / 1048576.0);
    }
    ~LatentSyncSdVaeGraph() { core::release_backend_graph_resources(backend_, graph_, true); }
    std::vector<float> run(const std::vector<float> & input) {
        core::write_tensor_f32(input_, input);
        if (core::compute_backend_graph(backend_, graph_) != GGML_STATUS_SUCCESS) throw std::runtime_error("LatentSync VAE execution failed");
        return core::read_tensor_f32(output_.tensor);
    }
private:
    ggml_backend_t backend_;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph_ = nullptr;
    core::TensorValue input_, output_;
};
}

struct LatentSyncSdVaeRuntime::Impl {
    core::ExecutionContext & execution;
    int64_t resolution;
    core::BackendWeightStore store;
    SdVaeTowerWeights encoder_weights, decoder_weights;
    std::unique_ptr<LatentSyncSdVaeGraph> encoder, decoder;
    Impl(core::ExecutionContext & execution_in, const assets::TensorSource & source,
        assets::TensorStorageType type, int64_t resolution_in)
        : execution(execution_in), resolution(resolution_in),
          store(execution.backend(), execution.backend_type(), "latentsync.vae.weights", 4 * 1024 * 1024) {
        if (resolution <= 0 || resolution % 8) throw std::runtime_error("LatentSync VAE resolution must be a positive multiple of 8");
        encoder_weights = load_sd_vae_weights(store, source, {}, true, type);
        decoder_weights = load_sd_vae_weights(store, source, {}, false, type);
        store.upload();
    }
};

LatentSyncSdVaeRuntime::LatentSyncSdVaeRuntime(core::ExecutionContext & execution,
    const assets::TensorSource & source, assets::TensorStorageType weight_type, int64_t resolution)
    : impl_(std::make_unique<Impl>(execution, source, weight_type, resolution)) {}
LatentSyncSdVaeRuntime::~LatentSyncSdVaeRuntime() = default;

std::vector<float> LatentSyncSdVaeRuntime::encode_moments(const std::vector<float> & image) {
    const auto start = std::chrono::steady_clock::now();
    if (!impl_->encoder) impl_->encoder = std::make_unique<LatentSyncSdVaeGraph>(impl_->execution, impl_->encoder_weights, impl_->resolution, true);
    auto output = impl_->encoder->run(image);
    debug::timing_log_scalar("latentsync.vae_encoder.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

std::vector<float> LatentSyncSdVaeRuntime::encode(const std::vector<float> & image, const std::vector<float> & posterior_noise) {
    const auto moments = encode_moments(image);
    const size_t count = moments.size() / 2;
    if (posterior_noise.size() != count) throw std::runtime_error("LatentSync VAE posterior noise size mismatch");
    std::vector<float> latent(count);
    for (size_t i = 0; i < count; ++i) {
        const float stddev = std::exp(0.5F * std::clamp(moments[i + count], -30.0F, 20.0F));
        latent[i] = (moments[i] + stddev * posterior_noise[i]) * kLatentScale;
    }
    return latent;
}

std::vector<float> LatentSyncSdVaeRuntime::decode(const std::vector<float> & scaled_latent) {
    const auto start = std::chrono::steady_clock::now();
    if (!impl_->decoder) impl_->decoder = std::make_unique<LatentSyncSdVaeGraph>(impl_->execution, impl_->decoder_weights, impl_->resolution, false);
    auto latent = scaled_latent;
    for (auto & value : latent) value /= kLatentScale;
    auto output = impl_->decoder->run(latent);
    debug::timing_log_scalar("latentsync.vae_decoder.wall_ms",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return output;
}

}  // namespace engine::models::latentsync
