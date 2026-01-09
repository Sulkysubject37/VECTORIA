#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

extern "C" {
    VectoriaStatus add_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus add_f32_avx2(const float* a, const float* b, float* out, size_t count);
}

void bench_add(size_t count) {
    std::vector<float> a(count, 1.0f);
    std::vector<float> b(count, 2.0f);
    std::vector<float> out(count);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        vectoria::kernels::reference::add_f32(a.data(), b.data(), out.data(), count);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ref_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
    
    double simd_time = 0.0;
#if defined(__aarch64__)
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        add_f32_neon(a.data(), b.data(), out.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
#elif defined(__x86_64__)
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        add_f32_avx2(a.data(), b.data(), out.data(), count);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
#endif

    std::cout << "ADD [N=" << count << "]: Ref=" << ref_time << "us | SIMD=" << simd_time << "us | Speedup=" << (ref_time/simd_time) << "x" << std::endl;
}

int main() {
    bench_add(1024);
    bench_add(1024*1024);
    return 0;
}
