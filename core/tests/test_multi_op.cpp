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

void test_elementwise_and_reduction() {
    std::cout << "Testing Add -> Mul -> ReduceSum..." << std::endl;
    
    // Graph:
    // A [2, 3] = [[1, 2, 3], [4, 5, 6]]
    // B [2, 3] = [[1, 1, 1], [2, 2, 2]]
    // C = Add(A, B) -> [[2, 3, 4], [6, 7, 8]]
    // D [2, 3] = [[2, 2, 2], [0.5, 0.5, 0.5]]
    // E = Mul(C, D) -> [[4, 6, 8], [3, 3.5, 4]]
    // F = ReduceSum(E) -> [18, 10.5]
    
    ir::Graph graph;
    
    // Nodes
    auto add_in = [&](const char* name, std::vector<int64_t> shape) {
        size_t id = graph.nodes.size();
        graph.nodes.push_back({ {id}, ir::InputNode{name, {shape}, ir::DataType::Float32} });
        return id;
    };
    
    auto add_op = [&](ir::OpType type, std::vector<size_t> inputs, std::vector<int64_t> out_shape) {
        size_t id = graph.nodes.size();
        std::vector<ir::NodeId> ins;
        for(auto i : inputs) ins.push_back({i});
        graph.nodes.push_back({ {id}, ir::OpNode{type, ins, {out_shape}, ir::DataType::Float32} });
        return id;
    };
    
    size_t a = add_in("A", {2, 3});
    size_t b = add_in("B", {2, 3});
    size_t d = add_in("D", {2, 3});
    
    size_t c = add_op(ir::OpType::Add, {a, b}, {2, 3});
    size_t e = add_op(ir::OpType::Mul, {c, d}, {2, 3});
    size_t f = add_op(ir::OpType::ReduceSum, {e}, {2});
    
    graph.outputs = {{f}};
    
    Engine engine(graph);
    engine.compile();
    
    float* a_ptr = (float*)engine.get_buffer(a);
    float* b_ptr = (float*)engine.get_buffer(b);
    float* d_ptr = (float*)engine.get_buffer(d);
    
    // Init data
    a_ptr[0]=1; a_ptr[1]=2; a_ptr[2]=3;
    a_ptr[3]=4; a_ptr[4]=5; a_ptr[5]=6;
    
    b_ptr[0]=1; b_ptr[1]=1; b_ptr[2]=1;
    b_ptr[3]=2; b_ptr[4]=2; b_ptr[5]=2;
    
    d_ptr[0]=2; d_ptr[1]=2; d_ptr[2]=2;
    d_ptr[3]=0.5; d_ptr[4]=0.5; d_ptr[5]=0.5;
    
    engine.execute();
    
    float* out_ptr = (float*)engine.get_buffer(f);
    
    if (std::abs(out_ptr[0] - 18.0f) > 1e-5f) {
        std::cerr << "Mismatch at 0: " << out_ptr[0] << " != 18.0" << std::endl;
        exit(1);
    }
    if (std::abs(out_ptr[1] - 10.5f) > 1e-5f) {
        std::cerr << "Mismatch at 1: " << out_ptr[1] << " != 10.5" << std::endl;
        exit(1);
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_gemm_bias_relu();
    test_elementwise_and_reduction();
    return 0;
}
