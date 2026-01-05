#include "vectoria/engine.hpp"
#include "vectoria/ir.hpp"
#include <cassert>
#include <iostream>

using namespace vectoria;

int main() {
    ir::Graph graph;
    
    // Construct a simple graph: Input(0) -> Op(1)
    ir::InputNode input{"X", { {1, 10} }, ir::DataType::Float32};
    graph.nodes.push_back({ {0}, input });
    
    ir::OpNode op{ir::OpType::Relu, {{0}}, { {1, 10} }, ir::DataType::Float32};
    graph.nodes.push_back({ {1}, op });
    
    graph.outputs = {{1}};

    Engine engine(graph);
    
    std::cout << "Validating graph..." << std::endl;
    assert(engine.validate());
    
    std::cout << "Compiling engine..." << std::endl;
    engine.compile();
    
    std::cout << "Executing schedule..." << std::endl;
    engine.execute();
    
    const auto& schedule = engine.get_schedule();
    assert(schedule.size() == 2);
    assert(schedule[0] == 0);
    assert(schedule[1] == 1);
    
    std::cout << "Engine skeleton test passed!" << std::endl;
    
    return 0;
}