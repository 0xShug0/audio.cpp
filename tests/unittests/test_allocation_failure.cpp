#include "test_allocation_failure.h"

#include <cassert>
#include <cstdlib>
#include <new>

namespace {
thread_local std::ptrdiff_t remaining = -1;
thread_local bool failed = false;
thread_local void * tracked[64]{};
thread_local size_t count = 0;

void forget(void * pointer) noexcept {
    for (size_t i = 0; i < count; ++i) {
        if (tracked[i] == pointer) {
            tracked[i] = tracked[--count];
            break;
        }
    }
}
}

engine::test::AllocationFailure::AllocationFailure(size_t successful_allocations) noexcept {
    assert(remaining == -1 && count == 0);
    failed = false;
    remaining = static_cast<std::ptrdiff_t>(successful_allocations);
}
engine::test::AllocationFailure::~AllocationFailure() { disable(); }
void engine::test::AllocationFailure::disable() noexcept { remaining = -1; }
bool engine::test::AllocationFailure::triggered() noexcept { return failed; }
size_t engine::test::AllocationFailure::outstanding() noexcept { return count; }

void * operator new(size_t bytes) {
    const bool armed = remaining >= 0;
    if (armed && remaining-- == 0) {
        remaining = -1;
        failed = true;
        throw std::bad_alloc();
    }
    void * pointer = std::malloc(bytes ? bytes : 1);
    if (!pointer) throw std::bad_alloc();
    if (armed) {
        assert(count < sizeof(tracked) / sizeof(tracked[0]));
        tracked[count++] = pointer;
    }
    return pointer;
}
void operator delete(void * pointer) noexcept {
    forget(pointer);
    std::free(pointer);
}
void operator delete(void * pointer, size_t) noexcept { ::operator delete(pointer); }
void * operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete[](void * pointer) noexcept { ::operator delete(pointer); }
void operator delete[](void * pointer, size_t) noexcept { ::operator delete(pointer); }
