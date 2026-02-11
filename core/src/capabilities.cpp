#include "vectoria/capabilities.hpp"

namespace vectoria {
namespace capabilities {

SystemCapabilities get_system_capabilities() {
    SystemCapabilities caps;
    
#if defined(__x86_64__)
    caps.arch = Architecture::X86_64;
    caps.arch_name = "x86_64";
#elif defined(__aarch64__)
    caps.arch = Architecture::ARM64;
    caps.arch_name = "ARM64";
#else
    caps.arch = Architecture::Unknown;
    caps.arch_name = "Unknown";
#endif

#ifdef VECTORIA_USE_ASM
    caps.simd_compiled = true;
#else
    caps.simd_compiled = false;
#endif

    caps.simd_supported_on_host = caps.simd_compiled;

    caps.available_kernels.push_back("Reference");
    if (caps.simd_compiled) {
        if (caps.arch == Architecture::ARM64) {
            caps.available_kernels.push_back("NEON");
        } else if (caps.arch == Architecture::X86_64) {
            caps.available_kernels.push_back("AVX2");
        }
    }

    return caps;
}

} // namespace capabilities
} // namespace vectoria
