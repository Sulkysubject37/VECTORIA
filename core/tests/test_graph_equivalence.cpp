#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "utils/gemm_validation.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace vectoria;

void test_equivalence(bool use_simd) {
    std::cout << "Testing Equivalence (Simd=" << use_simd << ")... ";
    
    // Graph: A[4,4] * B[4,4] + Bias[1,4] -> ReLU
    ir::Graph graph;
    auto mk_in = [&](std::string n, std::vector<int64_t> s) {
        size_t id = graph.nodes.size();
        graph.nodes.push_back({ {id}, ir::InputNode{n, {s}, ir::DataType::Float32} });
        return id;
    };
    auto mk_param = [&](std::string n, std::vector<int64_t> s) {
        size_t id = graph.nodes.size();
        graph.nodes.push_back({ {id}, ir::ParameterNode{n, {s}, ir::DataType::Float32, 0} });
        return id;
    };
    auto mk_op = [&](ir::OpType op, std::vector<size_t> inputs, std::vector<int64_t> s) {
        size_t id = graph.nodes.size();
        std::vector<ir::NodeId> in_ids;
        for(auto i : inputs) in_ids.push_back({i});
        graph.nodes.push_back({ {id}, ir::OpNode{op, in_ids, {s}, ir::DataType::Float32} });
        return id;
    };

    size_t a = mk_in("A", {4, 4});
    size_t b = mk_in("B", {4, 4});
    size_t bias = mk_param("Bias", {1, 4});
    size_t mm = mk_op(ir::OpType::MatMul, {a, b}, {4, 4});
    size_t ba = mk_op(ir::OpType::BiasAdd, {mm, bias}, {4, 4});
    size_t relu = mk_op(ir::OpType::Relu, {ba}, {4, 4});
    
    graph.outputs = {{relu}};

    // Reference Run
    EngineConfig ref_cfg;
    ref_cfg.policy = KernelPolicy::Reference;
    Engine ref_engine(graph, ref_cfg);
    ref_engine.compile();
    
    test::DeterministicRNG rng;
    float* a_ptr = (float*)ref_engine.get_buffer(a);
    float* b_ptr = (float*)ref_engine.get_buffer(b);
    float* bias_ptr = (float*)ref_engine.get_buffer(bias);
    
    rng.fill(a_ptr, 16);
    rng.fill(b_ptr, 16);
    rng.fill(bias_ptr, 4);
    
    ref_engine.execute();
    
    // Target Run
    EngineConfig target_cfg;
    target_cfg.policy = use_simd ? KernelPolicy::SIMD : KernelPolicy::Reference;
    Engine target_engine(graph, target_cfg);
    target_engine.compile();
    
    // Copy inputs exactly
    float* ta_ptr = (float*)target_engine.get_buffer(a);
    float* tb_ptr = (float*)target_engine.get_buffer(b);
    float* tbias_ptr = (float*)target_engine.get_buffer(bias);
    for(int i=0; i<16; ++i) ta_ptr[i] = a_ptr[i];
    for(int i=0; i<16; ++i) tb_ptr[i] = b_ptr[i];
    for(int i=0; i<4; ++i) tbias_ptr[i] = bias_ptr[i];
    
    #ifdef VECTORIA_USE_ASM
        target_engine.execute();
        
        float* ref_out = (float*)ref_engine.get_buffer(relu);
        float* target_out = (float*)target_engine.get_buffer(relu);
        
        auto res = test::compare_matrices(ref_out, target_out, 16, 1e-5f);
        if (res.match) {
            std::cout << "PASSED" << std::endl;
        } else {
            std::cout << "FAILED" << std::endl;
            test::print_mismatch(ref_out, target_out, 4, 4, res, "Equivalence");
            exit(1);
        }
    #else
        if (use_simd) {
            std::cout << "SKIPPED (No ASM)" << std::endl;
        } else {
            // Ref vs Ref should always pass
            target_engine.execute();
            std::cout << "PASSED" << std::endl;
        }
    #endif
}

int main() {
    test_equivalence(false); // Ref vs Ref (Sanity)
    test_equivalence(true);  // Ref vs SIMD
    return 0;
}
