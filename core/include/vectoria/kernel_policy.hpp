#pragma once

#include <cstdint>

namespace vectoria {

/**
 * Kernel Selection Policy.
 * Defines which implementation is used for operations.
 */
enum class KernelPolicy : uint8_t {
    /**
     * Reference Implementation (C++).
     * Guaranteed correct, deterministic, bit-exact.
     * Slow.
     */
    Reference = 0,

    /**
     * Architecture-Specific Assembly (AVX/Neon).
     * Fast.
     * Must be explicitly enabled and validated.
     */
    SIMD = 1
};

} // namespace vectoria
