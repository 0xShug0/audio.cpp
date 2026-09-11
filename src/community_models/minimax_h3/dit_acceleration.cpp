#include "engine/community_models/minimax_h3/dit_acceleration.h"

#include "engine/framework/debug/profiler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine::models::minimax_h3 {
namespace {

void cholesky_decompose(std::vector<float> & a, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col <= row; ++col) {
            float sum = a[static_cast<size_t>(row * n + col)];
            for (int k = 0; k < col; ++k) {
                sum -= a[static_cast<size_t>(row * n + k)] * a[static_cast<size_t>(col * n + k)];
            }
            if (row == col) {
                if (!(sum > 0.0F) || !std::isfinite(sum)) {
                    throw std::runtime_error("MiniMax-H3 spectrum ridge system is not positive definite");
                }
                a[static_cast<size_t>(row * n + col)] = std::sqrt(sum);
            } else {
                a[static_cast<size_t>(row * n + col)] = sum / a[static_cast<size_t>(col * n + col)];
            }
        }
        for (int col = row + 1; col < n; ++col) {
            a[static_cast<size_t>(row * n + col)] = 0.0F;
        }
    }
}

}  // namespace

MiniMaxH3SpectrumForecaster::MiniMaxH3SpectrumForecaster(
    const MiniMaxH3GenerateRequest & request,
    int64_t steps,
    size_t feature_size)
    : feature_size_(feature_size),
      steps_(steps),
      warmup_steps_(request.spectrum_warmup_steps),
      current_window_(request.spectrum_initial_window),
      flex_window_(request.spectrum_flex_window),
      degree_(static_cast<int>(request.spectrum_degree)),
      history_size_(static_cast<int>(request.spectrum_history_size)),
      ridge_lambda_(request.spectrum_ridge_lambda) {
    if (feature_size_ == 0 || steps_ <= 0) {
        throw std::runtime_error("MiniMax-H3 spectrum requires non-empty feature state and positive steps");
    }
    if (warmup_steps_ < 1 || degree_ < 0 || history_size_ <= 0 ||
        !(current_window_ >= 1.0F) || !(flex_window_ >= 0.0F) ||
        !(ridge_lambda_ >= 0.0F) || !std::isfinite(ridge_lambda_)) {
        throw std::runtime_error("MiniMax-H3 spectrum configuration is invalid");
    }
    coordinates_.reserve(static_cast<size_t>(history_size_));
    features_.reserve(static_cast<size_t>(history_size_) * feature_size_);
}

bool MiniMaxH3SpectrumForecaster::should_run_full(int64_t step) {
    if (step < warmup_steps_ || step + 1 >= steps_ || coordinates_.empty()) {
        cached_steps_in_window_ = 0;
        ++full_steps_;
        return true;
    }
    const int64_t window = std::max<int64_t>(1, static_cast<int64_t>(std::floor(current_window_)));
    const bool actual = ((cached_steps_in_window_ + 1) % window) == 0;
    if (actual) {
        cached_steps_in_window_ = 0;
        current_window_ += flex_window_;
        ++full_steps_;
    } else {
        ++cached_steps_in_window_;
        ++forecast_steps_;
    }
    return actual;
}

void MiniMaxH3SpectrumForecaster::update(float coordinate, const std::vector<float> & feature) {
    if (feature.size() != feature_size_) {
        throw std::runtime_error("MiniMax-H3 spectrum update shape mismatch");
    }
    if (!std::isfinite(coordinate)) {
        throw std::runtime_error("MiniMax-H3 spectrum coordinate is not finite");
    }
    if (coordinates_.size() == static_cast<size_t>(history_size_)) {
        coordinates_.erase(coordinates_.begin());
        features_.erase(features_.begin(), features_.begin() + static_cast<std::ptrdiff_t>(feature_size_));
    }
    coordinates_.push_back(std::clamp(coordinate, -1.0F, 1.0F));
    features_.insert(features_.end(), feature.begin(), feature.end());
    factorization_valid_ = false;
}

int MiniMaxH3SpectrumForecaster::basis_size() const {
    return degree_ + 1;
}

void MiniMaxH3SpectrumForecaster::build_basis(float coordinate, float * out) const {
    const int p = basis_size();
    const float x = std::clamp(coordinate, -1.0F, 1.0F);
    out[0] = 1.0F;
    if (p == 1) {
        return;
    }
    out[1] = x;
    for (int i = 2; i < p; ++i) {
        out[i] = 2.0F * x * out[i - 1] - out[i - 2];
    }
}

void MiniMaxH3SpectrumForecaster::fit_if_needed() {
    if (factorization_valid_) {
        return;
    }
    if (coordinates_.empty()) {
        throw std::runtime_error("MiniMax-H3 spectrum predict requires history");
    }
    const int p = basis_size();
    const size_t rows = coordinates_.size();
    design_.assign(rows * static_cast<size_t>(p), 0.0F);
    for (size_t row = 0; row < rows; ++row) {
        build_basis(coordinates_[row], design_.data() + row * static_cast<size_t>(p));
    }
    std::vector<float> xtx(static_cast<size_t>(p * p), 0.0F);
    for (int a = 0; a < p; ++a) {
        for (int b = 0; b < p; ++b) {
            float sum = 0.0F;
            for (size_t row = 0; row < rows; ++row) {
                sum += design_[row * static_cast<size_t>(p) + static_cast<size_t>(a)] *
                       design_[row * static_cast<size_t>(p) + static_cast<size_t>(b)];
            }
            xtx[static_cast<size_t>(a * p + b)] = sum + (a == b ? ridge_lambda_ : 0.0F);
        }
    }
    cholesky_decompose(xtx, p);
    cholesky_ = std::move(xtx);
    factorization_valid_ = true;
}

std::vector<float> MiniMaxH3SpectrumForecaster::spectral_weights(float coordinate) {
    fit_if_needed();
    const int p = basis_size();
    const size_t rows = coordinates_.size();
    std::vector<float> basis(static_cast<size_t>(p));
    build_basis(coordinate, basis.data());
    std::vector<float> y(static_cast<size_t>(p));
    for (int row = 0; row < p; ++row) {
        float sum = basis[static_cast<size_t>(row)];
        for (int col = 0; col < row; ++col) {
            sum -= cholesky_[static_cast<size_t>(row * p + col)] * y[static_cast<size_t>(col)];
        }
        y[static_cast<size_t>(row)] = sum / cholesky_[static_cast<size_t>(row * p + row)];
    }
    std::vector<float> solved(static_cast<size_t>(p));
    for (int row = p - 1; row >= 0; --row) {
        float sum = y[static_cast<size_t>(row)];
        for (int col = row + 1; col < p; ++col) {
            sum -= cholesky_[static_cast<size_t>(col * p + row)] * solved[static_cast<size_t>(col)];
        }
        solved[static_cast<size_t>(row)] = sum / cholesky_[static_cast<size_t>(row * p + row)];
    }
    std::vector<float> weights(rows, 0.0F);
    for (size_t row = 0; row < rows; ++row) {
        const float * x = design_.data() + row * static_cast<size_t>(p);
        float sum = 0.0F;
        for (int col = 0; col < p; ++col) {
            sum += x[col] * solved[static_cast<size_t>(col)];
        }
        weights[row] = sum;
    }
    return weights;
}

std::vector<float> MiniMaxH3SpectrumForecaster::linear_weights(float coordinate) const {
    std::vector<float> weights(coordinates_.size(), 0.0F);
    if (weights.empty()) {
        throw std::runtime_error("MiniMax-H3 spectrum linear prediction requires history");
    }
    if (weights.size() == 1) {
        weights.back() = 1.0F;
        return weights;
    }
    const float previous = coordinates_[coordinates_.size() - 2];
    const float latest = coordinates_.back();
    const float spacing = latest - previous;
    if (std::abs(spacing) <= 1.0e-12F) {
        weights.back() = 1.0F;
        return weights;
    }
    const float ratio = (coordinate - latest) / spacing;
    weights[weights.size() - 2] = -ratio;
    weights.back() = 1.0F + ratio;
    return weights;
}

std::vector<float> MiniMaxH3SpectrumForecaster::combined_weights(float coordinate, float blend_weight) {
    const float blend = std::clamp(blend_weight, 0.0F, 1.0F);
    auto linear = linear_weights(coordinate);
    if (blend <= 1.0e-12F) {
        return linear;
    }
    auto spectral = spectral_weights(coordinate);
    if (blend >= 1.0F - 1.0e-12F) {
        return spectral;
    }
    for (size_t i = 0; i < linear.size(); ++i) {
        linear[i] = blend * spectral[i] + (1.0F - blend) * linear[i];
    }
    return linear;
}

void MiniMaxH3SpectrumForecaster::predict(
    float coordinate,
    int64_t row_width,
    int64_t audio_start,
    int64_t audio_rows,
    int64_t video_start,
    int64_t video_rows,
    std::vector<float> & out) {
    if (!std::isfinite(coordinate)) {
        throw std::runtime_error("MiniMax-H3 spectrum coordinate is not finite");
    }
    if (row_width <= 0 || audio_start < 0 || audio_rows < 0 || video_start < 0 || video_rows < 0) {
        throw std::runtime_error("MiniMax-H3 spectrum row segmentation is invalid");
    }
    const int64_t total_rows = static_cast<int64_t>(feature_size_ / static_cast<size_t>(row_width));
    if (feature_size_ % static_cast<size_t>(row_width) != 0 ||
        audio_start + audio_rows > total_rows ||
        video_start + video_rows > total_rows) {
        throw std::runtime_error("MiniMax-H3 spectrum row segmentation does not match feature shape");
    }
    constexpr float kVideoBlendWeight = 0.0F;
    constexpr float kAudioBlendWeight = 0.0F;
    const auto video_weights = combined_weights(coordinate, kVideoBlendWeight);
    const auto audio_weights = combined_weights(coordinate, kAudioBlendWeight);
    const size_t latest = coordinates_.size() - 1;
    out.resize(feature_size_);
    const float * latest_feature = features_.data() + latest * feature_size_;
    std::copy(latest_feature, latest_feature + feature_size_, out.data());
    auto apply_rows = [&](int64_t start, int64_t rows, const std::vector<float> & weights) {
        const size_t offset = static_cast<size_t>(start * row_width);
        const size_t count = static_cast<size_t>(rows * row_width);
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(offset), out.begin() + static_cast<std::ptrdiff_t>(offset + count), 0.0F);
        for (size_t row = 0; row < weights.size(); ++row) {
            const float scale = weights[row];
            const float * feature = features_.data() + row * feature_size_ + offset;
            float * dst = out.data() + offset;
            for (size_t i = 0; i < count; ++i) {
                dst[i] += scale * feature[i];
            }
        }
    };
    if (audio_rows > 0) {
        apply_rows(audio_start, audio_rows, audio_weights);
    }
    if (video_rows > 0) {
        apply_rows(video_start, video_rows, video_weights);
    }
}

int64_t MiniMaxH3SpectrumForecaster::full_steps() const {
    return full_steps_;
}

int64_t MiniMaxH3SpectrumForecaster::forecast_steps() const {
    return forecast_steps_;
}

}  // namespace engine::models::minimax_h3
