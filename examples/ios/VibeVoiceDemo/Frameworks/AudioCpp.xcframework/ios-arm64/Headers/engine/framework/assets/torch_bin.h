#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::assets {

// Opens a PyTorch `.bin` checkpoint (a ZIP archive wrapping a pickled state dict) as a
// tensor source. Only the restricted subset produced by torch.save of a flat
// OrderedDict[str, Tensor] is supported, and the pickle is walked with a whitelist of
// globals so no adapter code is ever executed. Tensor storages must be uncompressed.
std::shared_ptr<const TensorSource> open_torch_bin_tensor_source(const std::filesystem::path & path);

}  // namespace engine::assets
