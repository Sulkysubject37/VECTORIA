#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "utils/gemm_validation.hpp"
#include <iostream>
#include <cassert>

using namespace vectoria;

void test_gemm_bias_relu() {
    std::cout << "Testing GEMM -> BiasAdd -> ReLU..." << std::endl;

    // Graph: Input(X) -> MatMul(W) -> BiasAdd(B) -> ReLU -> Out
    ir::Graph graph;
    
    // Node 0: X [1, 4]
    graph.nodes.push_back({ {0}, ir::InputNode{"X", {{1, 4}}, ir::DataType::Float32} });
    
    // Node 1: W [4, 4]
    graph.nodes.push_back({ {1}, ir::ParameterNode{"W", {{4, 4}}, ir::DataType::Float32, 0} });
    
    // Node 2: MatMul(0, 1) -> [1, 4]
    graph.nodes.push_back({ {2}, ir::OpNode{ir::OpType::MatMul, {{0}, {1}}, {{1, 4}}, ir::DataType::Float32} });
    
    // Node 3: Bias [1, 4] (Parameter)
    graph.nodes.push_back({ {3}, ir::ParameterNode{"B", {{1, 4}}, ir::DataType::Float32, 0} });
    
    // Node 4: BiasAdd(2, 3) -> [1, 4]
    graph.nodes.push_back({ {4}, ir::OpNode{ir::OpType::BiasAdd, {{2}, {3}}, {{1, 4}}, ir::DataType::Float32} });
    
    // Node 5: ReLU(4) -> [1, 4]
    graph.nodes.push_back({ {5}, ir::OpNode{ir::OpType::Relu, {{4}}, {{1, 4}}, ir::DataType::Float32} });
    
    graph.outputs = {{5}};

    Engine engine(graph);
    engine.compile();

    // Data
    float* x_ptr = (float*)engine.get_buffer(0);
    float* w_ptr = (float*)engine.get_buffer(1);
    float* b_ptr = (float*)engine.get_buffer(3);
    
    // X = [1, 1, 1, 1]
    for(int i=0; i<4; ++i) x_ptr[i] = 1.0f;
    
    // W = Identity
    for(int i=0; i<16; ++i) w_ptr[i] = 0.0f;
    w_ptr[0] = 1.0f; w_ptr[5] = 1.0f; w_ptr[10] = 1.0f; w_ptr[15] = 1.0f;
    
    // B = [-2, -0.5, 0, 2]
    b_ptr[0] = -2.0f; b_ptr[1] = -0.5f; b_ptr[2] = 0.0f; b_ptr[3] = 2.0f;
    
    // MatMul Result = [1, 1, 1, 1]
    // BiasAdd Result = [-1, 0.5, 1, 3]
    // ReLU Result = [0, 0.5, 1, 3]
    
    engine.execute();
    
    float* out_ptr = (float*)engine.get_buffer(5);
    float expected[4] = {0.0f, 0.5f, 1.0f, 3.0f};
    
    for(int i=0; i<4; ++i) {
        if(std::abs(out_ptr[i] - expected[i]) > 1e-5f) {
            std::cerr << "Mismatch at " << i << ": Got " << out_ptr[i] << " Expected " << expected[i] << std::endl;
            exit(1);
        }
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_gemm_bias_relu();
    return 0;
}
