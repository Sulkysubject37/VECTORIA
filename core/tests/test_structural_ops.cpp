#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/c_api.h" // For C API helpers if needed, but direct C++ API is cleaner
#include "vectoria/graph/transpose.hpp"
#include "vectoria/graph/reshape.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

using namespace vectoria;

void test_transpose_reshape() {
    std::cout << "Testing Structural Ops..." << std::endl;
    ir::Graph g;
    
    // Input [2, 3]
    ir::InputNode in_node;
    in_node.name = "Input";
    in_node.shape.dims = {2, 3};
    in_node.dtype = ir::DataType::Float32;
    size_t in_id = g.nodes.size();
    g.nodes.push_back({ {in_id}, in_node });
    
    // 1. Transpose [2, 3] -> [3, 2] (perm 1, 0)
    int t_id = graph::add_transpose(g, static_cast<int>(in_id), {1, 0});
    
    // 2. Reshape [3, 2] -> [6]
    int r_id = graph::add_reshape(g, t_id, {6});
    
    // 3. Reshape back to [2, 3] -> (Wait, reshape just reinterprets linear memory)
    // Transpose output is [3, 2].
    // Input:
    // [1, 2, 3]
    // [4, 5, 6]
    //
    // Transpose(1,0):
    // [1, 4]
    // [2, 5]
    // [3, 6]
    //
    // Reshape([6]):
    // [1, 4, 2, 5, 3, 6]
    
    g.outputs.push_back({static_cast<size_t>(r_id)});
    
    Engine e(g);
    e.compile();
    
    float* in_buf = static_cast<float*>(e.get_buffer(in_id));
    for(int i=0; i<6; ++i) in_buf[i] = static_cast<float>(i+1);
    
    e.execute();
    
    float* out_buf = static_cast<float*>(e.get_buffer(static_cast<size_t>(r_id)));
    
    std::vector<float> expected = {1, 4, 2, 5, 3, 6};
    
    for(int i=0; i<6; ++i) {
        if (std::abs(out_buf[i] - expected[i]) > 1e-5f) {
            std::cerr << "Structural Ops failed at index " << i << ": " << out_buf[i] << " expected " << expected[i] << std::endl;
            exit(1);
        }
    }
    
    std::cout << "Structural Ops Passed." << std::endl;
}

int main() {
    test_transpose_reshape();
    return 0;
}
