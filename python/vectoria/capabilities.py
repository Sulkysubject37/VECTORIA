from enum import Enum
from dataclasses import dataclass
import ctypes
from .runtime import _lib

class Architecture(Enum):
    UNKNOWN = 0
    X86_64 = 1
    ARM64 = 2

@dataclass
class SystemCapabilities:
    arch: Architecture
    arch_name: str
    simd_compiled: bool
    simd_supported: bool

def get_system_capabilities() -> SystemCapabilities:
    if not _lib:
        return SystemCapabilities(Architecture.UNKNOWN, "Unknown", False, False)
    
    c_arch = ctypes.c_int()
    c_compiled = ctypes.c_int()
    c_supported = ctypes.c_int()
    c_name = ctypes.create_string_buffer(64)
    
    _lib.vectoria_get_capabilities(
        ctypes.byref(c_arch),
        ctypes.byref(c_compiled),
        ctypes.byref(c_supported),
        c_name, 64
    )
    
    return SystemCapabilities(
        Architecture(c_arch.value),
        c_name.value.decode('utf-8'),
        c_compiled.value == 1,
        c_supported.value == 1
    )
