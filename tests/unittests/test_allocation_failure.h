#pragma once

#include <cstddef>

namespace engine::test {

// Fail one ordinary C++ allocation on this thread. Used only by test binaries;
// no production fault-injection hook or allocator setting is required.
class AllocationFailure {
public:
    explicit AllocationFailure(size_t successful_allocations) noexcept;
    ~AllocationFailure();
    AllocationFailure(const AllocationFailure &) = delete;
    AllocationFailure & operator=(const AllocationFailure &) = delete;
    void disable() noexcept;
    static bool triggered() noexcept;
    // Successful allocations made while armed and not subsequently deleted.
    // Tracking survives disable(), so cleanup and retry can be checked.
    static size_t outstanding() noexcept;
};

} // namespace engine::test
