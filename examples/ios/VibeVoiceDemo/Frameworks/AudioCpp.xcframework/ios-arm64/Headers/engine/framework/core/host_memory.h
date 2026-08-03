#pragma once

#include <cstddef>

namespace engine::core {

// Physical memory the host still has free, or 0 when it cannot be determined
// (no portable query on this platform). Callers must treat 0 as "unknown" and
// fall back to whatever they would have done anyway, never as "no memory".
size_t available_host_memory_bytes();

}  // namespace engine::core
