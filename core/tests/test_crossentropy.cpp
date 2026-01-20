#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace vectoria;

void test_crossentropy_simple() {
    std::cout << "Testing CrossEntropy Simple..." << std::endl;
    ir::Graph g;
    
    // [2, 3]
    ir::InputNode logits_node;
    logits_node.name = "Logits";
    logits_node.shape.dims = {2, 3};
    logits_node.dtype = ir::DataType::Float32;
    size_t logits_id = g.nodes.size();
    g.nodes.push_back({ {logits_id}, logits_node });
    
    // [2, 3]
    ir::InputNode target_node;
    target_node.name = "Target";
    target_node.shape.dims = {2, 3};
    target_node.dtype = ir::DataType::Float32;
    size_t target_id = g.nodes.size();
    g.nodes.push_back({ {target_id}, target_node });
    
    int out_id = graph::add_crossentropy_composed(g, static_cast<int>(logits_id), static_cast<int>(target_id));
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* l_buf = static_cast<float*>(e.get_buffer(logits_id));
    float* t_buf = static_cast<float*>(e.get_buffer(target_id));
    
    // Row 0: Perfect prediction
    // Logits: [100, 0, 0] -> Softmax approx [1, 0, 0]
    // Target: [1, 0, 0]
    // Loss: -1 * log(1) = 0
    l_buf[0] = 100; l_buf[1] = 0; l_buf[2] = 0;
    t_buf[0] = 1; t_buf[1] = 0; t_buf[2] = 0;
    
    // Row 1: Uniform
    // Logits: [0, 0, 0] -> Softmax [1/3, 1/3, 1/3]
    // Target: [1, 0, 0]
    // Loss: -log(1/3) = log(3) approx 1.0986
    l_buf[3] = 0; l_buf[4] = 0; l_buf[5] = 0;
    t_buf[3] = 1; t_buf[4] = 0; t_buf[5] = 0;
    
    e.execute();
    
    float* out_buf = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    // Row 0
    if (std::abs(out_buf[0]) > 1e-4f) {
        std::cerr << "Row 0 failed: " << out_buf[0] << " expected 0.0" << std::endl;
        exit(1);
    }
    
    // Row 1
    float expected = std::log(3.0f);
    if (std::abs(out_buf[1] - expected) > 1e-4f) {
        std::cerr << "Row 1 failed: " << out_buf[1] << " expected " << expected << std::endl;
        exit(1);
    }
    
    std::cout << "CrossEntropy Simple Passed." << std::endl;
}

int main() {
    test_crossentropy_simple();
    return 0;
}
