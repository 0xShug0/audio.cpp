#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::assets {
class TensorSource;
}

namespace engine::models::rvc {

struct RvcRetrievalIndex {
    int64_t dim = 0;
    int64_t nlist = 0;
    std::vector<float> centroids;
    std::vector<float> vectors;
    std::vector<int64_t> list_offsets;
    std::vector<int64_t> list_lengths;
};

RvcRetrievalIndex load_rvc_retrieval_index(
    const std::shared_ptr<const engine::assets::TensorSource> & source,
    int64_t dim,
    const std::string & label);
RvcRetrievalIndex load_rvc_retrieval_index(const std::filesystem::path & path, int64_t dim);

}  // namespace engine::models::rvc
