// Unit tests for TransformerKVCache ring_mode: slot mapping, wrap advance,
// chronological export, wrap-placement import, and non-ring regression.
#include "engine/framework/runtime/kv_cache.h"

#include "engine/framework/core/backend.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace core = engine::core;
namespace runtime = engine::runtime;

int failures = 0;

void check(bool condition, const std::string & message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << "\n";
    }
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct BackendDeleter {
    void operator()(ggml_backend_t backend) const noexcept {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

// 1 layer of F32 key/value tensors with room for (steps * elems) floats.
struct TestTensors {
    std::unique_ptr<ggml_context, GgmlContextDeleter> ctx;
    std::unique_ptr<std::remove_pointer_t<ggml_backend_buffer_t>, void (*)(ggml_backend_buffer_t)> buffer{
        nullptr, ggml_backend_buffer_free};
    core::TensorValue keys;
    core::TensorValue values;
};

ggml_backend_t cpu_backend() {
    static std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, BackendDeleter> backend(
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), BackendDeleter{});
    if (backend == nullptr) {
        throw std::runtime_error("test CPU backend init failed");
    }
    return backend.get();
}

TestTensors make_tensors(int64_t steps, int64_t elems) {
    TestTensors out;
    ggml_init_params params{1024ull * 1024ull, nullptr, true};
    out.ctx.reset(ggml_init(params));
    if (out.ctx == nullptr) {
        throw std::runtime_error("test context init failed");
    }
    const int64_t count = steps * elems;
    ggml_tensor * k = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, count);
    ggml_tensor * v = ggml_new_tensor_1d(out.ctx.get(), GGML_TYPE_F32, count);
    out.buffer.reset(ggml_backend_alloc_ctx_tensors(out.ctx.get(), cpu_backend()));
    if (out.buffer == nullptr) {
        throw std::runtime_error("test buffer alloc failed");
    }
    out.keys = core::TensorValue{k, core::TensorShape::from_dims({count}), GGML_TYPE_F32};
    out.values = core::TensorValue{v, core::TensorShape::from_dims({count}), GGML_TYPE_F32};
    return out;
}

runtime::TransformerKVCache make_cache(
    TestTensors & tensors, int64_t steps, int64_t elems, runtime::TransformerKVCacheOptions options) {
    return runtime::TransformerKVCache(
        steps, elems, {tensors.keys}, {tensors.values}, options);
}

// Row r filled with float(position): positions are directly readable.
std::vector<float> tagged_row(int64_t position, int64_t elems) {
    return std::vector<float>(static_cast<size_t>(elems), static_cast<float>(position));
}

runtime::KVLayerState tagged_layer(const std::vector<int64_t> & positions, int64_t elems) {
    runtime::KVLayerState layer;
    layer.valid_steps = static_cast<int64_t>(positions.size());
    for (const int64_t pos : positions) {
        const auto row = tagged_row(pos, elems);
        layer.key.insert(layer.key.end(), row.begin(), row.end());
        layer.value.insert(layer.value.end(), row.begin(), row.end());
    }
    return layer;
}

int64_t row_position(const std::vector<float> & flat, int64_t row, int64_t elems) {
    return static_cast<int64_t>(flat[static_cast<size_t>(row * elems)]);
}

void test_slot_mapping() {
    runtime::TransformerKVCacheOptions off;
    {
        TestTensors t = make_tensors(8, 4);
        runtime::TransformerKVCache cache = make_cache(t, 8, 4, off);
        check(cache.slot_for_position(0) == 0, "identity slot 0 when ring off");
        check(cache.slot_for_position(100) == 100, "identity slot 100 when ring off");
    }
    runtime::TransformerKVCacheOptions ring;
    ring.ring_mode = true;
    ring.ring_pinned_steps = 2;
    {
        TestTensors t = make_tensors(8, 4);
        runtime::TransformerKVCache cache = make_cache(t, 8, 4, ring);
        check(cache.slot_for_position(0) == 0, "pinned slot 0");
        check(cache.slot_for_position(1) == 1, "pinned slot 1");
        check(cache.slot_for_position(2) == 2, "first ring slot");
        check(cache.slot_for_position(7) == 7, "last ring slot first lap");
        check(cache.slot_for_position(8) == 2, "wrap to slot 2");
        check(cache.slot_for_position(9) == 3, "wrap to slot 3");
        check(cache.slot_for_position(14) == 2, "second wrap to slot 2");
        bool threw = false;
        try {
            cache.slot_for_position(-1);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "negative position throws");
    }
    std::cout << "slot mapping done\n";
}

void test_option_validation() {
    auto expect_throw = [](runtime::TransformerKVCacheOptions options, const std::string & what) {
        bool threw = false;
        try {
            TestTensors t = make_tensors(8, 4);
            runtime::TransformerKVCache cache = make_cache(t, 8, 4, options);
            (void)cache;
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, what);
    };
    runtime::TransformerKVCacheOptions zero_cap;
    zero_cap.ring_mode = true;
    // NOTE: cache_steps is clamped to >= 0 in the ctor; ring requires positive.
    {
        bool threw = false;
        try {
            TestTensors t = make_tensors(0, 4);
            runtime::TransformerKVCache cache = make_cache(t, 0, 4, zero_cap);
            (void)cache;
        } catch (const std::runtime_error &) {
            threw = true;
        }
        check(threw, "ring with zero capacity throws");
    }
    runtime::TransformerKVCacheOptions bad_pin;
    bad_pin.ring_mode = true;
    bad_pin.ring_pinned_steps = 8;
    expect_throw(bad_pin, "pinned == capacity throws");
    bad_pin.ring_pinned_steps = -1;
    expect_throw(bad_pin, "negative pinned throws");
    std::cout << "option validation done\n";
}

void test_wrap_round_trip() {
    constexpr int64_t steps = 8;
    constexpr int64_t elems = 4;
    runtime::TransformerKVCacheOptions ring;
    ring.ring_mode = true;
    ring.ring_pinned_steps = 2;
    TestTensors t = make_tensors(steps, elems);
    runtime::TransformerKVCache cache = make_cache(t, steps, elems, ring);
    // Chronological rows for positions [0,1] + [6..11] (2,3,4,5 evicted).
    runtime::TransformerKVState state;
    state.current_end = 12;
    state.layers.push_back(tagged_layer({0, 1, 6, 7, 8, 9, 10, 11}, elems));
    cache.import_state(state);
    check(cache.valid_steps() == 8, "import keeps valid count");
    check(cache.current_end() == 12, "import keeps absolute end");
    // Physical placement: prompt at 0,1; tail 6..11 wraps into 4,5,6,7,2,3.
    const std::vector<float> raw = core::read_tensor_f32(t.keys.tensor);
    const int64_t expected_slots[8] = {0, 1, 6, 7, 8, 9, 10, 11};
    check(row_position(raw, 0, elems) == 0, "slot 0 holds pinned 0");
    check(row_position(raw, 1, elems) == 1, "slot 1 holds pinned 1");
    check(row_position(raw, 2, elems) == 8, "slot 2 holds wrapped 8");
    check(row_position(raw, 3, elems) == 9, "slot 3 holds wrapped 9");
    check(row_position(raw, 4, elems) == 10, "slot 4 holds wrapped 10");
    check(row_position(raw, 5, elems) == 11, "slot 5 holds wrapped 11");
    check(row_position(raw, 6, elems) == 6, "slot 6 holds 6");
    check(row_position(raw, 7, elems) == 7, "slot 7 holds 7");
    // Chronological export restores input order.
    const runtime::TransformerKVState back = cache.export_state();
    check(back.current_end == 12, "export keeps absolute end");
    check(back.layers.front().valid_steps == 8, "export keeps valid count");
    for (int row = 0; row < 8; ++row) {
        check(row_position(back.layers.front().key, row, elems) == expected_slots[row],
              "export row " + std::to_string(row) + " chronological");
    }
    std::cout << "wrap round-trip done\n";
}

void test_wrap_advance() {
    constexpr int64_t steps = 8;
    constexpr int64_t elems = 4;
    runtime::TransformerKVCacheOptions ring;
    ring.ring_mode = true;
    ring.ring_pinned_steps = 2;
    TestTensors t = make_tensors(steps, elems);
    runtime::TransformerKVCache cache = make_cache(t, steps, elems, ring);
    runtime::TransformerKVState state;
    state.current_end = 2;
    state.layers.push_back(tagged_layer({0, 1}, elems));
    cache.import_state(state);
    cache.advance_after_direct_append(6);
    check(cache.valid_steps() == 8, "advance fills to capacity");
    check(cache.current_end() == 8, "advance moves end");
    cache.advance_after_direct_append(4);
    check(cache.valid_steps() == 8, "advance past capacity stays capped");
    check(cache.current_end() == 12, "advance past capacity moves end");
    // Non-ring still throws on overflow.
    runtime::TransformerKVCacheOptions off;
    TestTensors t2 = make_tensors(steps, elems);
    runtime::TransformerKVCache plain = make_cache(t2, steps, elems, off);
    plain.import_state(state);
    bool threw = false;
    try {
        plain.advance_after_direct_append(7);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "non-ring advance overflow still throws");
    std::cout << "wrap advance done\n";
}

void test_non_ring_regression() {
    constexpr int64_t steps = 8;
    constexpr int64_t elems = 4;
    runtime::TransformerKVCacheOptions off;
    TestTensors t = make_tensors(steps, elems);
    runtime::TransformerKVCache cache = make_cache(t, steps, elems, off);
    runtime::TransformerKVState state;
    state.current_end = 5;
    state.layers.push_back(tagged_layer({10, 11, 12, 13, 14}, elems));
    cache.import_state(state);
    const runtime::TransformerKVState back = cache.export_state();
    check(back.current_end == 5, "dense export keeps end");
    for (int row = 0; row < 5; ++row) {
        check(row_position(back.layers.front().key, row, elems) == 10 + row, "dense export row order");
    }
    cache.retain_prefix(3);
    check(cache.valid_steps() == 3, "retain_prefix keeps count");
    check(cache.current_end() == 3, "retain_prefix rewinds end");
    std::cout << "non-ring regression done\n";
}

}  // namespace

int main() {
    try {
        test_slot_mapping();
        test_option_validation();
        test_wrap_round_trip();
        test_wrap_advance();
        test_non_ring_regression();
    } catch (const std::exception & e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "transformer_kv_ring: all checks passed\n";
    return 0;
}
