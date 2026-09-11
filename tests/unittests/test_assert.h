#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace engine::test {

inline void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T, typename U>
void require_eq(const T & actual, const U & expected, const std::string & label) {
    if (!(actual == expected)) {
        std::ostringstream oss;
        oss << label << " mismatch: expected=" << expected << " actual=" << actual;
        throw std::runtime_error(oss.str());
    }
}

inline void require_close(float actual, float expected, float tolerance, const std::string & label) {
    const float diff = std::fabs(actual - expected);
    if (diff > tolerance) {
        std::ostringstream oss;
        oss << label << " mismatch: expected=" << expected << " actual=" << actual
            << " diff=" << diff << " tolerance=" << tolerance;
        throw std::runtime_error(oss.str());
    }
}

}  // namespace engine::test
