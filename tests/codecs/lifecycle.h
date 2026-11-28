#pragma once

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <type_traits>

// Compare every exercised input with a fresh runtime, not with a model cache.
template <class Factory, class Encode, class Input, class Release = std::nullptr_t>
void check_runtime_lifecycle(Factory make, Encode encode, const Input & a, const Input & b, const Input & c,
                             Release release = nullptr) {
    const auto fresh_a = encode(*make(), a);
    const auto fresh_b = encode(*make(), b);
    const auto fresh_c = encode(*make(), c);
    if (fresh_a.empty() || fresh_b.empty() || fresh_c.empty()) throw std::runtime_error("empty fresh reference");
    bool passed = true;
    const auto check = [&](const char * phase, const auto & actual, const auto & expected) {
        const bool equal = actual == expected;
        size_t different = 0, nonfinite = 0;
        double error = 0, dot = 0, aa = 0, bb = 0;
        for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
            const double x = actual[i], y = expected[i];
            different += actual[i] != expected[i];
            if (!std::isfinite(x) || !std::isfinite(y)) { ++nonfinite; continue; }
            error = std::max(error, std::abs(x - y));
            dot += x * y; aa += x * x; bb += y * y;
        }
        std::cout << std::setprecision(12) << phase << " exact=" << equal << " values=" << actual.size()
                  << " different=" << different << " nonfinite=" << nonfinite << " max_abs=" << error;
        if constexpr (std::is_floating_point_v<typename std::decay_t<decltype(actual)>::value_type>) {
            if (aa > 0 && bb > 0) std::cout << " cosine=" << dot / std::sqrt(aa * bb);
        }
        std::cout << std::endl;
        passed = passed && equal;
    };
    {
        auto runtime = make();
        check("first_A", encode(*runtime, a), fresh_a);
        check("repeat_A", encode(*runtime, a), fresh_a);
        check("same_shape_B", encode(*runtime, b), fresh_b);
        check("return_A", encode(*runtime, a), fresh_a);
        check("changed_shape_C", encode(*runtime, c), fresh_c);
        check("shape_return_A", encode(*runtime, a), fresh_a);
        if constexpr (!std::is_same_v<Release, std::nullptr_t>) {
            release(*runtime);
            check("release_reprepare_A", encode(*runtime, a), fresh_a);
        }
    }
    check("destroy_recreate_A", encode(*make(), a), fresh_a);
    if (!passed) throw std::runtime_error("framework runtime lifecycle output changed");
}
