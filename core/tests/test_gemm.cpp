#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace vectoria;

void test_2x2_gemm() {
    std::cout << "Running 2x2 GEMM Test..." << std::endl;
    
    // Graph Construction
    // A (2x2) * B (2x2) = C (2x2)
    ir::Graph graph;
    
    // Node 0: Input A [2, 2]
    ir::InputNode input_a{"A", {{2, 2}}, ir::DataType::Float32};
    graph.nodes.push_back({ {0}, input_a });

    // Node 1: Input B [2, 2]
    ir::InputNode input_b{"B", {{2, 2}}, ir::DataType::Float32};
    graph.nodes.push_back({ {1}, input_b });

    // Node 2: MatMul(0, 1) -> [2, 2]
    ir::OpNode matmul{ir::OpType::MatMul, {{0}, {1}}, {{2, 2}}, ir::DataType::Float32};
    graph.nodes.push_back({ {2}, matmul });

    graph.outputs = {{2}};

    // Engine Setup
    Engine engine(graph);
    engine.compile();

    // Data Setup
    // A = [[1, 2], [3, 4]]
    // B = [[0.5, 1.0], [1.5, 2.0]]
    // C = [[1*0.5 + 2*1.5, 1*1.0 + 2*2.0], [3*0.5 + 4*1.5, 3*1.0 + 4*2.0]]
    // C = [[3.5, 5.0], [7.5, 11.0]]

    float* a_ptr = static_cast<float*>(engine.get_buffer(0));
    float* b_ptr = static_cast<float*>(engine.get_buffer(1));
    float* c_ptr = static_cast<float*>(engine.get_buffer(2));

    assert(a_ptr != nullptr);
    assert(b_ptr != nullptr);
    assert(c_ptr != nullptr);

    // Row-major initialization
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f;
    a_ptr[2] = 3.0f; a_ptr[3] = 4.0f;

    b_ptr[0] = 0.5f; b_ptr[1] = 1.0f;
    b_ptr[2] = 1.5f; b_ptr[3] = 2.0f;

    // Execute
    engine.execute();

    // Validation
    const float EPSILON = 1e-5f;
    float expected[4] = {3.5f, 5.0f, 7.5f, 11.0f};

    for (int i = 0; i < 4; ++i) {
        float val = c_ptr[i];
        float exp = expected[i];
        if (std::abs(val - exp) > EPSILON) {
            std::cerr << "Mismatch at index " << i << ": Got " << val << ", Expected " << exp << std::endl;
            exit(1);
        }
    }

    std::cout << "2x2 GEMM Test Passed." << std::endl;
}

int main() {
    try {
        test_2x2_gemm();
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
