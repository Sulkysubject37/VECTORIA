#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Forward declare NEON kernels
extern "C" {
    VectoriaStatus add_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus mul_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus sub_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus div_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus relu_f32_neon(const float* in, float* out, size_t count);
}

void verify(const std::vector<float>& ref, const std::vector<float>& simd, const char* name) {
    if (ref.size() != simd.size()) {
        std::cerr << name << " Size Mismatch!" << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - simd[i]) > 1e-5f) {
            std::cerr << name << " Mismatch at " << i << ": Ref=" << ref[i] << " SIMD=" << simd[i] << std::endl;
            exit(1);
        }
    }
    std::cout << name << " PASSED" << std::endl;
}

int main() {
    std::cout << "Validating SIMD Elementwise Kernels..." << std::endl;
    
    size_t count = 1024 + 7; // Test tail handling
    std::vector<float> a(count), b(count), ref(count), simd(count);
    
    // Init
    for(size_t i=0; i<count; ++i) {
        a[i] = (float)(i % 100) + 1.0f;
        b[i] = (float)((i+1) % 50) + 0.5f;
    }
    
#if defined(__aarch64__)
    // ADD
    vectoria::kernels::reference::add_f32(a.data(), b.data(), ref.data(), count);
    add_f32_neon(a.data(), b.data(), simd.data(), count);
    verify(ref, simd, "ADD");
    
    // MUL
    vectoria::kernels::reference::mul_f32(a.data(), b.data(), ref.data(), count);
    mul_f32_neon(a.data(), b.data(), simd.data(), count);
    verify(ref, simd, "MUL");
    
    // SUB
    vectoria::kernels::reference::sub_f32(a.data(), b.data(), ref.data(), count, count);
    sub_f32_neon(a.data(), b.data(), simd.data(), count);
    verify(ref, simd, "SUB");
    
    // DIV
    vectoria::kernels::reference::div_f32(a.data(), b.data(), ref.data(), count, count);
    div_f32_neon(a.data(), b.data(), simd.data(), count);
    verify(ref, simd, "DIV");
    
    // RELU
    std::vector<float> r_in = a;
    for(size_t i=0; i<count; i+=2) r_in[i] *= -1.0f; // Mix pos/neg
    vectoria::kernels::reference::relu_f32(r_in.data(), ref.data(), count);
    relu_f32_neon(r_in.data(), simd.data(), count);
    verify(ref, simd, "RELU");
#else
    std::cout << "Skipping NEON validation on non-ARM64 host." << std::endl;
#endif

    return 0;
}
