#pragma once

#include <cstdint>

namespace vectoria {

/**
 * Execution Mode.
 * Defines the contract under which the engine operates.
 */
enum class ExecutionMode : uint8_t {
    /**
     * Research Mode (Default).
     * - Allows experimental kernels.
     * - Allows fallback to reference if SIMD fails.
     * - Allows composed ops to execute node-by-node.
     * - Prioritizes correctness and inspectability over deployment constraints.
     */
    Research = 0,

    /**
     * Deployment Mode.
     * - Strict validation of op support for target backend (e.g., CoreML).
     * - Rejects ops not supported by the deployment contract.
     * - Rejects shapes or parameters that cannot be lowered safely.
     * - Enforces semantic equivalence guarantees.
     */
    Deployment = 1
};

} // namespace vectoria
