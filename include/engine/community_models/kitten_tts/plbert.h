#pragma once

#include <cstdint>
#include <memory>
#include <vector>

typedef struct ggml_backend *ggml_backend_t;

namespace engine::models::kitten_tts {

struct KittenWeights;

int64_t kitten_plbert_output_dim(std::shared_ptr<const KittenWeights> weights, bool project_hidden);

class KittenPlbertRuntime {
  public:
    KittenPlbertRuntime(std::shared_ptr<const KittenWeights> weights, ggml_backend_t backend, int n_threads,
                        bool use_device_backend, int64_t fixed_token_capacity = 0);
    ~KittenPlbertRuntime();

    KittenPlbertRuntime(const KittenPlbertRuntime &) = delete;
    KittenPlbertRuntime &operator=(const KittenPlbertRuntime &) = delete;

    void prepare(bool project_hidden = true);
    std::vector<float> encode(const std::vector<int32_t> &input_ids, bool project_hidden = true);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::kitten_tts
