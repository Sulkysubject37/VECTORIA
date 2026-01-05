#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "utils/gemm_validation.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>

using namespace vectoria;

// Simple Adler-32 like checksum for floats
uint32_t checksum(const float* data, size_t count) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < count; ++i) {
        // Quantize to avoid tiny FP variations affecting checksum?
        // No, we want to detect variations.
        // But FP associativity means we might differ slightly.
        // Let's print the sum and mean.
        // Checksum is brittle for FP.
        // Let's return raw bytes? No.
        
        // We will just calculate Sum of Absolute Values.
        // And Sum.
        return 0; // Placeholder
    }
    return 0;
}

double sum_abs(const float* data, size_t count) {
    double sum = 0;
    for(size_t i=0; i<count; ++i) sum += std::abs(data[i]);
    return sum;
}

double sum_val(const float* data, size_t count) {
    double sum = 0;
    for(size_t i=0; i<count; ++i) sum += data[i];
    return sum;
}

void run_consistency_check() {
    std::cout << "Running Cross-Arch Consistency Check..." << std::endl;
    
    size_t size = 128;
    ir::Graph graph;
    graph.nodes.push_back({ {0}, ir::InputNode{"A", {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32} });
    graph.nodes.push_back({ {1}, ir::InputNode{"B", {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32} });
    graph.nodes.push_back({ {2}, ir::OpNode{ir::OpType::MatMul, {{0}, {1}}, {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32} });
    graph.outputs = {{2}};

    // Use Reference
    EngineConfig ref_cfg;
    ref_cfg.policy = KernelPolicy::Reference;
    Engine ref_engine(graph, ref_cfg);
    ref_engine.compile();
    
    test::DeterministicRNG rng(42); // Fixed Seed
    float* a = (float*)ref_engine.get_buffer(0);
    float* b = (float*)ref_engine.get_buffer(1);
    rng.fill(a, size*size);
    rng.fill(b, size*size);
    
    ref_engine.execute();
    float* c_ref = (float*)ref_engine.get_buffer(2);
    double s_ref = sum_val(c_ref, size*size);
    
    std::cout << "Reference Sum: " << std::setprecision(10) << s_ref << std::endl;

#ifdef VECTORIA_USE_ASM
    EngineConfig simd_cfg;
    simd_cfg.policy = KernelPolicy::SIMD;
    Engine simd_engine(graph, simd_cfg);
    simd_engine.compile();
    
    // Copy inputs
    float* a_s = (float*)simd_engine.get_buffer(0);
    float* b_s = (float*)simd_engine.get_buffer(1);
    for(size_t i=0; i<size*size; ++i) { a_s[i] = a[i]; b_s[i] = b[i]; }
    
    simd_engine.execute();
    float* c_simd = (float*)simd_engine.get_buffer(2);
    double s_simd = sum_val(c_simd, size*size);
    
    std::cout << "SIMD Sum:      " << std::setprecision(10) << s_simd << std::endl;
    
    double diff = std::abs(s_ref - s_simd);
    std::cout << "Difference:    " << diff << std::endl;
    
    if (diff > 1e-3) { // 128x128 accumulators can drift.
        std::cerr << "FAIL: Divergence too large." << std::endl;
        exit(1);
    }
    std::cout << "PASSED" << std::endl;
#else
    std::cout << "SIMD Skipped." << std::endl;
#endif
}

int main() {
    run_consistency_check();
    return 0;
}
