#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend *ggml_backend_t;

namespace engine::models::kitten_tts {

struct KittenWeights;

struct KittenPredictorGraphConfig {
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
    const ggml_tensor *borrowed_decoder_x_tensor = nullptr;
    bool decoder_x_on_backend = false;
};

class KittenPredictorRuntime {
  public:
    KittenPredictorRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend, int n_threads,
                           bool use_device_backend, int64_t plbert_fixed_token_capacity = 0,
                           int64_t pre_tail_token_capacity = 0, KittenPredictorGraphConfig graph_config = {});
    ~KittenPredictorRuntime();

    KittenPredictorRuntime(const KittenPredictorRuntime &) = delete;
    KittenPredictorRuntime &operator=(const KittenPredictorRuntime &) = delete;

    PredictorOutputs predict(const std::vector<int32_t> &input_ids, const std::vector<float> &ref_s, float speed);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::kitten_tts
