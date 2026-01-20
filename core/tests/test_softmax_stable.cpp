#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace vectoria;

void test_softmax_stable() {
    std::cout << "Testing Stable Softmax..." << std::endl;
    ir::Graph g;
    
    // Input [2, 3]
    ir::InputNode in_node;
    in_node.name = "Input";
    in_node.shape.dims = {2, 3};
    in_node.dtype = ir::DataType::Float32;
    size_t in_id = g.nodes.size();
    g.nodes.push_back({ {in_id}, in_node });
    
    // Stable Softmax
    int out_id = graph::add_softmax_stable_composed(g, static_cast<int>(in_id));
    
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* in_buf = static_cast<float*>(e.get_buffer(in_id));
    
    // Row 0: 0, 0, 0 -> exp: 1, 1, 1 -> sum: 3 -> soft: 1/3, 1/3, 1/3
    // Row 1: 1000, 1000, 1000 -> shifted: 0, 0, 0 -> same result
    std::vector<float> in_data = {0, 0, 0, 1000, 1000, 1000};
    for(size_t i=0; i<6; ++i) in_buf[i] = in_data[i];
    
    e.execute();
    
    float* out_buf = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    float expected = 1.0f / 3.0f;
    
    for(int r=0; r<2; ++r) {
        float row_sum = 0;
        for(int c=0; c<3; ++c) {
            float val = out_buf[r*3 + c];
            if (std::abs(val - expected) > 1e-4f) {
                std::cerr << "Row " << r << " Col " << c << " failed: " << val << " expected " << expected << std::endl;
                exit(1);
            }
            row_sum += val;
        }
        
        // Sum should be 1.0
        if (std::abs(row_sum - 1.0f) > 1e-4f) {
             std::cerr << "Row " << r << " Probability Sum failed: " << row_sum << std::endl;
             exit(1);
        }
    }
    
    std::cout << "Stable Softmax Passed." << std::endl;
}

int main() {
    test_softmax_stable();
    return 0;
}
