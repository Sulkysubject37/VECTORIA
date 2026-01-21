#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph/concatenation.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

using namespace vectoria;

void test_concatenation_simple() {
    std::cout << "Testing Concatenation Simple (Axis 0)..." << std::endl;
    ir::Graph g;
    
    // X1 [2, 2]
    ir::InputNode in1;
    in1.name = "X1";
    in1.shape.dims = {2, 2};
    in1.dtype = ir::DataType::Float32;
    size_t id1 = g.nodes.size();
    g.nodes.push_back({ {id1}, in1 });
    
    // X2 [3, 2]
    ir::InputNode in2;
    in2.name = "X2";
    in2.shape.dims = {3, 2};
    in2.dtype = ir::DataType::Float32;
    size_t id2 = g.nodes.size();
    g.nodes.push_back({ {id2}, in2 });
    
    // Concat(X1, X2, axis=0) -> [5, 2]
    int out_id = graph::add_concat(g, {static_cast<int>(id1), static_cast<int>(id2)}, 0);
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* b1 = static_cast<float*>(e.get_buffer(id1));
    float* b2 = static_cast<float*>(e.get_buffer(id2));
    
    for(int i=0; i<4; ++i) b1[i] = static_cast<float>(i+1);
    for(int i=0; i<6; ++i) b2[i] = static_cast<float>(i+10);
    
    e.execute();
    
    float* out = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    // Expected: [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
    std::vector<float> expected = {1, 2, 3, 4, 10, 11, 12, 13, 14, 15};
    for(int i=0; i<10; ++i) {
        if (std::abs(out[i] - expected[i]) > 1e-5f) {
            std::cerr << "Concat failed at index " << i << ": " << out[i] << " != " << expected[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "PASSED" << std::endl;
}

void test_concatenation_axis1() {
    std::cout << "Testing Concatenation Axis 1..." << std::endl;
    ir::Graph g;
    
    // X1 [2, 2]
    ir::InputNode in1;
    in1.name = "X1";
    in1.shape.dims = {2, 2};
    in1.dtype = ir::DataType::Float32;
    size_t id1 = g.nodes.size();
    g.nodes.push_back({ {id1}, in1 });
    
    // X2 [2, 1]
    ir::InputNode in2;
    in2.name = "X2";
    in2.shape.dims = {2, 1};
    in2.dtype = ir::DataType::Float32;
    size_t id2 = g.nodes.size();
    g.nodes.push_back({ {id2}, in2 });
    
    // Concat(X1, X2, axis=1) -> [2, 3]
    int out_id = graph::add_concat(g, {static_cast<int>(id1), static_cast<int>(id2)}, 1);
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* b1 = static_cast<float*>(e.get_buffer(id1));
    float* b2 = static_cast<float*>(e.get_buffer(id2));
    
    b1[0] = 1; b1[1] = 2; b1[2] = 3; b1[3] = 4;
    b2[0] = 5; b2[1] = 6;
    
    e.execute();
    
    float* out = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    // Linear order:
    // Row 0: [1, 2, 5]
    // Row 1: [3, 4, 6]
    // Flat: [1, 2, 5, 3, 4, 6]
    std::vector<float> expected = {1, 2, 5, 3, 4, 6};
    for(int i=0; i<6; ++i) {
        if (std::abs(out[i] - expected[i]) > 1e-5f) {
            std::cerr << "Concat failed at index " << i << ": " << out[i] << " != " << expected[i] << std::endl;
            exit(1);
        }
    }
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_concatenation_simple();
    test_concatenation_axis1();
    return 0;
}
