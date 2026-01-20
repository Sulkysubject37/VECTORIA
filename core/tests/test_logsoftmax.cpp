#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace vectoria;

void test_logsoftmax_simple() {
    std::cout << "Testing LogSoftmax Simple..." << std::endl;
    ir::Graph g;
    
    // Input [2, 3]
    ir::InputNode in_node;
    in_node.name = "Input";
    in_node.shape.dims = {2, 3};
    in_node.dtype = ir::DataType::Float32;
    size_t in_id = g.nodes.size();
    g.nodes.push_back({ {in_id}, in_node });
    
    // LogSoftmax
    int out_id = graph::add_logsoftmax_composed(g, static_cast<int>(in_id));
    
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* in_buf = static_cast<float*>(e.get_buffer(in_id));
    
    // Row 0: 0, 0, 0 -> exp: 1, 1, 1 -> sum: 3 -> log_sum: 1.0986 -> out: -1.0986
    // Row 1: 1000, 1000, 1000 -> shifted: 0, 0, 0 -> same result (stability check)
    std::vector<float> in_data = {0, 0, 0, 1000, 1000, 1000};
    for(size_t i=0; i<6; ++i) in_buf[i] = in_data[i];
    
    e.execute();
    
    float* out_buf = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    float expected = -std::log(3.0f); // approx -1.098612
    
    for(int r=0; r<2; ++r) {
        float row_sum_exp = 0;
        for(int c=0; c<3; ++c) {
            float val = out_buf[r*3 + c];
            if (std::abs(val - expected) > 1e-4f) {
                std::cerr << "Row " << r << " Col " << c << " failed: " << val << " expected " << expected << std::endl;
                exit(1);
            }
            row_sum_exp += std::exp(val);
        }
        
        // Sum of exp(log_softmax) should be 1.0
        if (std::abs(row_sum_exp - 1.0f) > 1e-4f) {
             std::cerr << "Row " << r << " Probability Sum failed: " << row_sum_exp << std::endl;
             exit(1);
        }
    }
    
    std::cout << "LogSoftmax Simple Passed." << std::endl;
}

int main() {
    test_logsoftmax_simple();
    return 0;
}
