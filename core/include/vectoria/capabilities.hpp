#pragma once

#include <string>
#include <vector>

namespace vectoria {
namespace capabilities {

/**
 * Supported Architectures.
 */
enum class Architecture {
    Unknown,
    X86_64,
    ARM64
};

/**
 * Capability information for the current build/host.
 */
struct SystemCapabilities {
    Architecture arch;
    std::string arch_name;
    bool simd_compiled;
    bool simd_supported_on_host;
    std::vector<std::string> available_kernels;
};

/**
 * Returns the capabilities of the current execution environment.
 */
SystemCapabilities get_system_capabilities();

} // namespace capabilities
} // namespace vectoria
