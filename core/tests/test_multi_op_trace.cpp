#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include <iostream>
#include <cassert>

using namespace vectoria;

void test_trace_content() {
    ir::Graph graph;
    graph.nodes.push_back({ {0}, ir::InputNode{"X", {{1, 4}}, ir::DataType::Float32} });
    graph.nodes.push_back({ {1}, ir::ParameterNode{"W", {{4, 4}}, ir::DataType::Float32, 0} });
    graph.nodes.push_back({ {2}, ir::OpNode{ir::OpType::MatMul, {{0}, {1}}, {{1, 4}}, ir::DataType::Float32} });
    graph.outputs = {{2}};

    Engine engine(graph);
    engine.compile();
    engine.execute();

    const auto& events = engine.get_tracer().get_events();
    bool found_dispatch = false;
    for (const auto& ev : events) {
        if (ev.type == trace::EventType::KernelDispatch && ev.node_id == 2) {
            std::cout << "Found Dispatch Event: " << ev.details << std::endl;
            assert(ev.details.find("Inputs: [0, 1]") != std::string::npos);
            found_dispatch = true;
        }
    }
    assert(found_dispatch);
    std::cout << "Trace content validation passed!" << std::endl;
}

int main() {
    test_trace_content();
    return 0;
}
