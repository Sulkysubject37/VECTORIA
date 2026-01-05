#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "vectoria/memory.hpp"
#include "vectoria/kernel_policy.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace vectoria;

double measure_engine(Engine& engine, int iterations) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        engine.execute();
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        engine.execute();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

void run_bench(size_t size, int iterations) {
    std::cout << "Benchmarking GEMM " << size << "x" << size << " (" << iterations << " iterations)..." << std::endl;

    // 1. Setup Graph
    ir::Graph graph;
    ir::InputNode input_a{"A", {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32};
    graph.nodes.push_back({ {0}, input_a });

    ir::InputNode input_b{"B", {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32};
    graph.nodes.push_back({ {1}, input_b });

    ir::OpNode matmul{ir::OpType::MatMul, {{0}, {1}}, {{static_cast<int64_t>(size), static_cast<int64_t>(size)}}, ir::DataType::Float32};
    graph.nodes.push_back({ {2}, matmul });
    
    graph.outputs = {{2}};

    double flops = 2.0 * size * size * size * iterations;

    // Reference Benchmark
    EngineConfig ref_config;
    ref_config.policy = KernelPolicy::Reference;
    Engine ref_engine(graph, ref_config);
    ref_engine.compile(); // Allocates memory

    double ref_ms = measure_engine(ref_engine, iterations);
    double ref_gflops = flops / (ref_ms / 1000.0) / 1e9;

    std::cout << "  [Ref ] Time: " << std::fixed << std::setprecision(2) << ref_ms << " ms | " 
              << ref_gflops << " GFLOPS" << std::endl;

    // SIMD Benchmark (if available)
#ifdef VECTORIA_USE_ASM
    EngineConfig simd_config;
    simd_config.policy = KernelPolicy::SIMD;
    Engine simd_engine(graph, simd_config);
    simd_engine.compile();

    try {
        double simd_ms = measure_engine(simd_engine, iterations);
        double simd_gflops = flops / (simd_ms / 1000.0) / 1e9;

        std::cout << "  [SIMD] Time: " << std::fixed << std::setprecision(2) << simd_ms << " ms | " 
                  << simd_gflops << " GFLOPS" << std::endl;
        
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << (ref_ms / simd_ms) << "x" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  [SIMD] Skipped (Error: " << e.what() << ")" << std::endl;
    }
#else
    std::cout << "  [SIMD] Skipped (Build flag disabled)" << std::endl;
#endif

    std::cout << "----------------------------------------" << std::endl;
}

int main() {
    // Fixed sizes for regression testing
    run_bench(64, 1000);
    run_bench(256, 100);
    run_bench(512, 20);
    // run_bench(1024, 5); // Optional larger size
    
    return 0;
}
