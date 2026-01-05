#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "vectoria/memory.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

using namespace vectoria;

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

    // 2. Engine
    Engine engine(graph);
    engine.compile();

    // 3. Warmup
    for (int i = 0; i < 5; ++i) {
        engine.execute();
    }

    // 4. Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        engine.execute();
    }
    auto end = std::chrono::high_resolution_clock::now();

    // 5. Report
    std::chrono::duration<double, std::milli> duration = end - start;
    double avg_ms = duration.count() / iterations;
    
    // GFLOPS = 2 * M * N * K / (time_sec * 1e9)
    double flops = 2.0 * size * size * size;
    double gflops = (flops * iterations) / (duration.count() / 1000.0) / 1e9;

    std::cout << "  Avg Time: " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << gflops << " GFLOPS (Reference Impl)" << std::endl;
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
