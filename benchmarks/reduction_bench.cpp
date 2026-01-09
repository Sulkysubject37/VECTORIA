#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

extern "C" {
    VectoriaStatus reduce_sum_f32_neon(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_sum_f32_avx2(const float* in, float* out, size_t outer, size_t inner);
}

void bench_reduce(size_t outer, size_t inner) {
    std::vector<float> in(outer * inner, 1.0f);
    std::vector<float> out(outer);
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        vectoria::kernels::reference::reduce_sum_f32(in.data(), out.data(), outer, inner);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ref_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
    
    double simd_time = 0.0;
#if defined(__aarch64__)
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        reduce_sum_f32_neon(in.data(), out.data(), outer, inner);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
#elif defined(__x86_64__)
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; ++i) {
        reduce_sum_f32_avx2(in.data(), out.data(), outer, inner);
    }
    end = std::chrono::high_resolution_clock::now();
    simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 100.0;
#endif

    std::cout << "ReduceSum [" << outer << "x" << inner << "]: Ref=" << ref_time << "us | SIMD=" << simd_time << "us | Speedup=" << (ref_time/simd_time) << "x" << std::endl;
}

int main() {
    bench_reduce(1, 1024*1024);
    bench_reduce(1024, 1024);
    return 0;
}
