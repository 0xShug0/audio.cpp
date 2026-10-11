#pragma once

#include "engine/framework/neon/operators.h"
#include "worker_team.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace engine::neon::detail {

bool available() noexcept;

struct PackedConvWeights {
    int input_channels = 0;
    int output_channels = 0;
    int kernel_size = 0;
    int padded_output_channels = 0;
    std::vector<float> values;
    std::vector<float> bias;
};

PackedConvWeights pack_conv_weights(const float *weights, const float *bias, int input_channels, int output_channels,
                                    int kernel_size);

struct Workspace {
    int max_kernel_size = 1;
    std::vector<float> packed_input;
    std::vector<float> norm_mean;
    std::vector<float> norm_inverse_std;
    size_t panel_rows = 0;
    std::vector<float> panels;
    size_t attention_rows = 0;
    std::vector<float> attention;
    std::vector<int8_t> quantized_input;
    std::vector<float> quantized_scales;

    void prepare(int frames, int input_channels, int kernel_size);
    void reserve_panels(int rows, int workers);
    void reserve_attention(int rows, int workers);
};

// Four output rows per tile. Each 16-byte vector stores four input values
// from each output row; scales are four contiguous floats per 32-value block.
struct PackedInt8Rows {
    std::vector<int8_t> values;
    std::vector<float> scales;
};
PackedInt8Rows pack_int8_rows(const BlockInt8Weights &weights, int input_features, int output_features);

void multiply_panel(const float *weights, const float *panel, const float *bias, float *output, int rows,
                    int output_channels, int frames, int frame_offset, int frame_count, int block_begin,
                    int block_end) noexcept;

void grow(std::vector<float> &buffer, size_t size);
void split_range(int count, int worker, int workers, int &begin, int &end);

void linear(const float *weight, const float *bias, const float *input, float *output, int input_features,
            int output_features, int frames, bool input_frame_major, bool output_frame_major,
            WorkerContext &worker) noexcept;

void quantized_linear(const PackedInt8Rows &weight, const float *bias, const float *input, float *output,
                      int input_features, int output_features, int frames, bool input_frame_major,
                      bool output_frame_major, Workspace &workspace, WorkerContext &worker) noexcept;

void rotary_embedding(const float *input, float *output, int channels, int frames, bool frame_major, int head_dimension,
                      const std::vector<float> &frequencies, int position, WorkerContext &worker) noexcept;
void attention(const float *query, const float *key, const float *value, const float *current_key,
               const float *current_value, float *output, int query_frames, int key_frames, int heads,
               int key_value_heads, int dimension, int key_capacity, bool key_head_major, bool query_frame_major,
               bool output_frame_major, Workspace &workspace, WorkerContext &worker) noexcept;

// Allocation-free arithmetic kernels. WorkerContext carries partitioning and
// any internal packing barrier; the executor synchronizes between operators.
void convolve_1d(const PackedConvWeights &weights, const float *input, float *output, int output_row_stride, int frames,
                 Workspace &workspace, WorkerContext &worker) noexcept;
void convolve_1d_strided(const PackedConvWeights &weights, const float *input, int input_frames, float *output,
                         int stride, int dilation, int padding_left, int output_frames, Workspace &workspace,
                         WorkerContext &worker) noexcept;
void normalize_channels(const float *input, const float *weight, const float *bias, float *output, int channels,
                        int frames, float epsilon, Workspace &workspace, WorkerContext &worker) noexcept;
void convolve_depthwise_1d(const float *input, const float *weight, const float *bias, float *output, int channels,
                           int frames, int kernel_size, WorkerContext &worker) noexcept;
void apply_gelu(float *values, size_t begin, size_t end) noexcept;
void apply_silu(float *values, size_t begin, size_t end) noexcept;
void apply_leaky_relu_clamp(float *values, size_t begin, size_t end, float negative_slope, float minimum) noexcept;
void pool_max_2d(const float *input, float *output, int channels, int height, int width, int kernel_height,
                 int kernel_width, int stride_height, int stride_width, WorkerContext &worker) noexcept;

void convolve_2d(const PackedConvWeights &weights, const Conv2dConfig &config, const float *input, int height,
                 int width, float *output, Workspace &workspace, WorkerContext &worker) noexcept;

} // namespace engine::neon::detail
