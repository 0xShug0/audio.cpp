#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {
void register_im2col_dispatch(DispatchRegistryBuilder & registry);
}
