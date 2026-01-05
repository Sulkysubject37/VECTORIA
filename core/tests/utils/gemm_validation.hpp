#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

namespace vectoria {
namespace test {

// Deterministic Pseudo-Random Number Generator
// We avoid std::rand() to ensure cross-platform identical sequences.
class DeterministicRNG {
public:
    explicit DeterministicRNG(uint32_t seed = 0x12345678) : state_(seed) {}

    // Returns float in [0.0, 1.0]
    float next_float() {
        state_ = state_ * 1664525 + 1013904223;
        return static_cast<float>(state_) / static_cast<float>(std::numeric_limits<uint32_t>::max());
    }

    void fill(float* data, size_t count, float scale = 1.0f) {
        for (size_t i = 0; i < count; ++i) {
            data[i] = (next_float() - 0.5f) * 2.0f * scale; // Range [-scale, scale]
        }
    }

private:
    uint32_t state_;
};

// Comparison Utilities
struct ValidationResult {
    bool match;
    float max_diff;
    size_t error_count;
    size_t first_error_idx;
};

inline ValidationResult compare_matrices(
    const float* ref, const float* target, 
    size_t count, float epsilon = 1e-5f
) {
    ValidationResult result = {true, 0.0f, 0, 0};
    bool first = true;

    for (size_t i = 0; i < count; ++i) {
        float diff = std::abs(ref[i] - target[i]);
        if (diff > result.max_diff) {
            result.max_diff = diff;
        }

        if (diff > epsilon) {
            result.match = false;
            result.error_count++;
            if (first) {
                result.first_error_idx = i;
                first = false;
            }
        }
    }
    return result;
}

inline void print_mismatch(
    const float* ref, const float* target, 
    size_t rows, size_t cols, 
    const ValidationResult& res,
    const char* name
) {
    std::cerr << "[FAIL] " << name << " Mismatch!" << std::endl;
    std::cerr << "       Max Diff: " << res.max_diff << std::endl;
    std::cerr << "       Errors:   " << res.error_count << "/" << (rows * cols) << std::endl;
    
    size_t idx = res.first_error_idx;
    size_t r = idx / cols;
    size_t c = idx % cols;
    
    std::cerr << "       First at [" << r << "," << c << "]: " 
              << "Ref=" << ref[idx] << " vs Target=" << target[idx] << std::endl;
}

} // namespace test
} // namespace vectoria
