#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include "utils/gemm_validation.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace vectoria;

void stress_test_determinism() {
    std::cout << "Running Determinism Stress Test..." << std::endl;
    
    // Complex Graph: A*B + Bias -> ReLU -> BiasAdd -> ReLU
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

    size_t a = mk_in("A", {8, 8});
    size_t b = mk_in("B", {8, 8});
    size_t bias1 = mk_param("Bias1", {1, 8});
    size_t mm = mk_op(ir::OpType::MatMul, {a, b}, {8, 8});
    size_t ba1 = mk_op(ir::OpType::BiasAdd, {mm, bias1}, {8, 8});
    size_t relu1 = mk_op(ir::OpType::Relu, {ba1}, {8, 8});
    size_t bias2 = mk_param("Bias2", {1, 8});
    size_t ba2 = mk_op(ir::OpType::BiasAdd, {relu1, bias2}, {8, 8});
    size_t relu2 = mk_op(ir::OpType::Relu, {ba2}, {8, 8});
    
    graph.outputs = {{relu2}};

    EngineConfig config;
#ifdef VECTORIA_USE_ASM
    config.policy = KernelPolicy::SIMD;
#else
    config.policy = KernelPolicy::Reference;
#endif

    Engine engine(graph, config);
    engine.compile();
    
    test::DeterministicRNG rng(12345);
    float* a_ptr = (float*)engine.get_buffer(a);
    float* b_ptr = (float*)engine.get_buffer(b);
    float* b1_ptr = (float*)engine.get_buffer(bias1);
    float* b2_ptr = (float*)engine.get_buffer(bias2);
    
    rng.fill(a_ptr, 64);
    rng.fill(b_ptr, 64);
    rng.fill(b1_ptr, 8);
    rng.fill(b2_ptr, 8);
    
    engine.execute();
    float* out_ptr = (float*)engine.get_buffer(relu2);
    std::vector<float> golden(out_ptr, out_ptr + 64);
    
    auto total_initial_events = engine.get_tracer().get_events().size();
    engine.execute();
    auto events_per_execute = engine.get_tracer().get_events().size() - total_initial_events;

    for (int i = 0; i < 50; ++i) {
        auto before_count = engine.get_tracer().get_events().size();
        engine.execute();
        auto after_count = engine.get_tracer().get_events().size();
        
        for (size_t j = 0; j < 64; ++j) {
            if (out_ptr[j] != golden[j]) {
                std::cerr << "Determinism FAIL at iteration " << i << " index " << j << std::endl;
                exit(1);
            }
        }
        
        if ((after_count - before_count) != events_per_execute) {
            std::cerr << "Trace inconsistency at iteration " << i << " (expected " << events_per_execute << " new events, got " << (after_count - before_count) << ")" << std::endl;
            exit(1);
        }
    }

    std::cout << "PASSED (50 iterations)" << std::endl;
}

int main() {
    stress_test_determinism();
    return 0;
}
