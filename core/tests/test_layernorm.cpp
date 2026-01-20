#include "vectoria/ir.hpp"
#include "vectoria/engine.hpp"
#include "vectoria/graph_ops.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace vectoria;

void test_layernorm_simple() {
    std::cout << "Testing LayerNorm Simple..." << std::endl;
    ir::Graph g;
    
    // Input [2, 3]
    ir::InputNode in_node;
    in_node.name = "Input";
    in_node.shape.dims = {2, 3};
    in_node.dtype = ir::DataType::Float32;
    size_t in_id = g.nodes.size();
    g.nodes.push_back({ {in_id}, in_node });
    
    // Gamma [3] (ones)
    ir::ParameterNode gamma_node;
    gamma_node.name = "Gamma";
    gamma_node.shape.dims = {3};
    gamma_node.dtype = ir::DataType::Float32;
    size_t gamma_id = g.nodes.size();
    g.nodes.push_back({ {gamma_id}, gamma_node });

    // Beta [3] (zeros)
    ir::ParameterNode beta_node;
    beta_node.name = "Beta";
    beta_node.shape.dims = {3};
    beta_node.dtype = ir::DataType::Float32;
    size_t beta_id = g.nodes.size();
    g.nodes.push_back({ {beta_id}, beta_node });

    // LayerNorm
    int out_id = graph::add_layernorm_composed(g, static_cast<int>(in_id), static_cast<int>(gamma_id), static_cast<int>(beta_id));
    
    g.outputs.push_back({static_cast<size_t>(out_id)});
    
    Engine e(g);
    e.compile();
    
    float* in_buf = static_cast<float*>(e.get_buffer(in_id));
    float* gamma_buf = static_cast<float*>(e.get_buffer(gamma_id));
    float* beta_buf = static_cast<float*>(e.get_buffer(beta_id));
    
    // Set data
    // Row 0: 1, 2, 3 -> Mean 2, Var 0.666...
    // Row 1: 10, 20, 30 -> Mean 20, Var 66.666...
    std::vector<float> in_data = {1, 2, 3, 10, 20, 30};
    for(size_t i=0; i<6; ++i) in_buf[i] = in_data[i];
    
    for(size_t i=0; i<3; ++i) {
        gamma_buf[i] = 1.0f;
        beta_buf[i] = 0.0f;
    }
    
    e.execute();
    
    float* out_buf = static_cast<float*>(e.get_buffer(static_cast<size_t>(out_id)));
    
    // Expected:
    // Row 0: 1, 2, 3. Mean=2. Var=(1+0+1)/3 = 0.666666. Std=sqrt(0.66666 + 1e-5) ~= 0.8165
    // (1-2)/0.8165 = -1.2247
    // (2-2)/0.8165 = 0
    // (3-2)/0.8165 = 1.2247
    
    // Let's verify sum is approx 0 and std is approx 1 per row.
    for(int r=0; r<2; ++r) {
        float sum = 0;
        float sq_sum = 0;
        for(int c=0; c<3; ++c) {
            float val = out_buf[r*3 + c];
            sum += val;
            sq_sum += val*val;
            // Print for debug
            // std::cout << val << " ";
        }
        // std::cout << std::endl;
        
        float mean = sum / 3.0f;
        float var = sq_sum / 3.0f - mean*mean;
        
        if (std::abs(mean) > 1e-4f) {
            std::cerr << "Row " << r << " Mean failed: " << mean << std::endl;
            exit(1);
        }
        // Var should be approx 1.0 (if gamma=1)
        // Wait, population variance of result should be 1?
        // y = (x-u)/s. var(y) = var(x)/s^2.
        // s = sqrt(var(x) + eps).
        // If eps is small, var(y) ~= 1.
        if (std::abs(var - 1.0f) > 1e-3f) { // Looser tolerance for epsilon effect?
             std::cerr << "Row " << r << " Var failed: " << var << std::endl;
             exit(1);
        }
    }
    
    std::cout << "LayerNorm Simple Passed." << std::endl;
}

int main() {
    test_layernorm_simple();
    return 0;
}
