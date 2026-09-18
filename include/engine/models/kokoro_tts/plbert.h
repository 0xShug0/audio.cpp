#pragma once

#include <cstdint>
#include <memory>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace kokoro_ggml {

struct KokoroWeights;

int64_t kokoro_plbert_output_dim(std::shared_ptr<const KokoroWeights> weights, bool project_hidden);

class KokoroPlbertRuntime {
public:
    KokoroPlbertRuntime(
        std::shared_ptr<const KokoroWeights> weights,
        ggml_backend_t backend,
        int n_threads,
        bool use_device_backend,
        int64_t fixed_token_capacity = 0);
    ~KokoroPlbertRuntime();

    KokoroPlbertRuntime(const KokoroPlbertRuntime &) = delete;
    KokoroPlbertRuntime & operator=(const KokoroPlbertRuntime &) = delete;

    std::vector<float> encode(const std::vector<int32_t> & input_ids, bool project_hidden = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kokoro_ggml
