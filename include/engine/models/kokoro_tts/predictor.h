#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

namespace kokoro_ggml {

struct KokoroWeights;

struct KokoroPredictorGraphConfig {
    size_t duration_graph_bytes = 384ull * 1024ull * 1024ull;
    size_t text_graph_bytes = 256ull * 1024ull * 1024ull;
    size_t tail_graph_bytes = 640ull * 1024ull * 1024ull;
    int graph_node_capacity = 131072;
};

struct PredictorOutputs {
    std::vector<int32_t> durations;
    std::vector<float> f0_curve;
    std::vector<float> decoder_x;
    int64_t decoder_x_rows = 0;
    int64_t decoder_x_cols = 0;
    const ggml_tensor * decoder_x_tensor = nullptr;
    bool decoder_x_on_backend = false;
};

class KokoroPredictorRuntime {
public:
    KokoroPredictorRuntime(
        std::shared_ptr<const KokoroWeights> weights,
        ggml_backend_t backend,
        int n_threads,
        bool use_device_backend,
        int64_t plbert_fixed_token_capacity = 0,
        int64_t pre_tail_token_capacity = 0,
        KokoroPredictorGraphConfig graph_config = {});
    ~KokoroPredictorRuntime();

    KokoroPredictorRuntime(const KokoroPredictorRuntime &) = delete;
    KokoroPredictorRuntime & operator=(const KokoroPredictorRuntime &) = delete;

    PredictorOutputs predict(
        const std::vector<int32_t> & input_ids,
        const std::vector<float> & ref_s,
        float speed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro_ggml
