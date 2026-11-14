#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {
void register_softmax_dispatch(DispatchRegistryBuilder & registry);
}
