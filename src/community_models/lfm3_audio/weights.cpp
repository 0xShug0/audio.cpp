#include "engine/community_models/lfm3_audio/weights.h"

#include <ggml-cpp.h>

#include "engine/framework/core/backend.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace engine::community_models::lfm3_audio {

namespace {

ggml_context_ptr probe_context(size_t tensors) {
    ggml_context_ptr ctx(ggml_init({tensors * ggml_tensor_overhead(), nullptr, true}));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to initialize the LFM3-Audio op probe context");
    }
    return ctx;
}

// Probes matmul and get_rows, as embed_tokens is both the embedding table and the tied head.
bool backend_runs_bf16(ggml_backend_t backend) {
    const auto ctx = probe_context(8);
    auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, 256, 32);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 256, 4);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 4);
    return ggml_backend_supports_op(backend, ggml_mul_mat(ctx.get(), weight, x)) &&
           ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), weight, ids));
}

// ggml's CPU BF16 dot product has SIMD paths for x86, RISC-V and POWER only (ggml-cpu/vec.cpp), so on an arm64
// CPU BF16 matmuls are scalar: about 20x slower than F16 on an M3 Ultra.
bool cpu_bf16_is_scalar(ggml_backend_t backend) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
#else
    (void)backend;
    return false;
#endif
}

}  // namespace

Lfm3Weights::Lfm3Weights(core::ExecutionContext & execution, std::string name, size_t context_bytes, Lfm3WeightOptions options)
    : options_(options), execution_(execution), name_(std::move(name)), context_bytes_(context_bytes),
      plain_(execution.backend(), execution.backend_type(), name_, context_bytes) {
    if (options.cpu_repack) extra_buffer_types_ = core::cpu_extra_buffer_types(execution.backend());
}

core::TensorValue Lfm3Weights::load_f32_tensor(const assets::TensorSource & source, const std::string & name,
                                             const std::vector<int64_t> & shape) {
    return plain_.load_f32_tensor(source, name, shape);
}

core::TensorValue Lfm3Weights::load_tensor(const assets::TensorSource & source, const std::string & name,
                                         assets::TensorStorageType requested, const std::vector<int64_t> & shape) {
    const auto metadata = source.require_metadata(name);
    const auto type = assets::ggml_type_for_tensor_dtype(metadata.dtype);
    const auto storage = requested == assets::TensorStorageType::Native
        ? lfm3_storage_type(execution_.backend(), type, options_) : requested;
    auto stored_shape = metadata.shape;
    while (stored_shape.size() > shape.size() && stored_shape.back() == 1) stored_shape.pop_back();
    if (shape.size() == 2 && stored_shape != metadata.shape && stored_shape == shape) {
        return plain_.load_tensor_as_shape(source, name, storage, metadata.shape, core::TensorShape::from_dims({shape.at(0), shape.at(1)}));
    }
    const auto buffer_type = storage == assets::TensorStorageType::Native && shape.size() == 2
        ? matmul_buffer_type(type, shape[0], shape[1]) : nullptr;
    if (!buffer_type) return plain_.load_tensor(source, name, storage, shape);

    auto found = std::find_if(extra_.begin(), extra_.end(), [&](const Extra & extra) { return extra.type == buffer_type; });
    if (found == extra_.end()) {
        extra_.push_back({buffer_type, std::make_unique<core::BackendWeightStore>(execution_.backend(), execution_.backend_type(),
            name_ + "." + ggml_backend_buft_name(buffer_type), context_bytes_, buffer_type)});
        found = std::prev(extra_.end());
    }
    auto value = found->store->load_tensor(source, name, storage, shape);
    if (value.tensor->type != type) throw std::logic_error(name_ + " " + name + " changed type while preparing its CPU projection");
    ggml_set_name(value.tensor, name.c_str());
    return value;
}

ggml_backend_buffer_type_t Lfm3Weights::matmul_buffer_type(ggml_type type, int64_t rows, int64_t cols) const {
    for (auto * buffer_type : extra_buffer_types_) {
        if (core::buffer_type_supports_matmul_weight(execution_.backend(), buffer_type, type, cols, rows)) return buffer_type;
    }
    return nullptr;
}

void Lfm3Weights::upload() {
    plain_.upload();
    for (auto & extra : extra_) extra.store->upload();
}

assets::TensorStorageType lfm3_storage_type(ggml_backend_t backend, ggml_type stored, const Lfm3WeightOptions & options) {
    switch (options.type) {
        case Lfm3WeightType::F32:
            return assets::TensorStorageType::F32;
        case Lfm3WeightType::F16:
            return assets::TensorStorageType::F16;
        case Lfm3WeightType::Native:
            break;
    }

    if (stored == GGML_TYPE_BF16 && (cpu_bf16_is_scalar(backend) || !backend_runs_bf16(backend))) {
        return assets::TensorStorageType::F16;
    }
    return assets::TensorStorageType::Native;
}

bool lfm3_reference_precision(const Lfm3WeightOptions & options) {
    return options.type == Lfm3WeightType::F32;
}

bool lfm3_backend_gathers(ggml_backend_t backend, ggml_type type) {
    const auto ctx = probe_context(4);
    auto * table = ggml_new_tensor_2d(ctx.get(), type, ggml_blck_size(type), 1);
    auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    return ggml_backend_supports_op(backend, ggml_get_rows(ctx.get(), table, ids));
}

modules::LfmDecoderConfig lfm3_decoder_config(modules::LfmDecoderConfig config, const Lfm3WeightOptions & options) {
    config.projection_precision = kLfm3MatmulPrecision;
    config.attention_precision = GGML_PREC_F32;
    config.conv_lowering = modules::LfmDecoderConvLowering::OrderedTaps;
    config.preserve_mul_input_layout = true;
    const auto hd = config.head_dim;
    config.attention_lowering = !lfm3_reference_precision(options) && (hd == 64 || hd == 96 || hd == 128)
        ? modules::GroupedQueryAttentionLowering::FlashGroupedViewKV : modules::GroupedQueryAttentionLowering::ManualRepeat;
    return config;
}

}  // namespace engine::community_models::lfm3_audio
