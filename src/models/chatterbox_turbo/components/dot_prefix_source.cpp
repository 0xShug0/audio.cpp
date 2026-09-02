#include "dot_prefix_source.h"

#include <utility>

namespace engine::models::chatterbox_turbo {

namespace {

class DotPrefixedTensorSource final : public engine::assets::TensorSource {
public:
    DotPrefixedTensorSource(std::shared_ptr<const engine::assets::TensorSource> delegate, std::string prefix)
        : delegate_(std::move(delegate)), prefix_(std::move(prefix) + ".") {}

    const std::filesystem::path & source_path() const noexcept override {
        return delegate_->source_path();
    }
    bool has_tensor(std::string_view name) const noexcept override {
        return delegate_->has_tensor(prefix_ + std::string(name));
    }
    engine::assets::TensorMetadata require_metadata(std::string_view name) const override {
        return delegate_->require_metadata(prefix_ + std::string(name));
    }
    std::vector<engine::assets::TensorMetadata> tensors() const override {
        return delegate_->tensors();
    }
    engine::assets::RawTensorData require_tensor_data(std::string_view name) const override {
        return delegate_->require_tensor_data(prefix_ + std::string(name));
    }
    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return delegate_->require_f32(prefix_ + std::string(name), expected_shape);
    }
    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        return delegate_->optional_f32(prefix_ + std::string(name), expected_shape);
    }
    int64_t require_i64_scalar(std::string_view name) const override {
        return delegate_->require_i64_scalar(prefix_ + std::string(name));
    }

private:
    std::shared_ptr<const engine::assets::TensorSource> delegate_;
    std::string prefix_;
};

}  // namespace

std::shared_ptr<const engine::assets::TensorSource> make_dot_prefixed_tensor_source(
    std::shared_ptr<const engine::assets::TensorSource> source,
    std::string prefix) {
    return std::make_shared<DotPrefixedTensorSource>(std::move(source), std::move(prefix));
}

}  // namespace engine::models::chatterbox_turbo
