#include "vectoria/ir.hpp"
#include "vectoria/lowering/coreml.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>

namespace fs = std::filesystem;

void test_simple_export() {
    std::cout << "Testing CoreML Export (Structural)..." << std::endl;
    
    vectoria::ir::Graph graph;
    size_t id = 0;
    
    // Input X
    graph.nodes.push_back({ {id++}, vectoria::ir::InputNode{"X", {{2, 2}}, vectoria::ir::DataType::Float32} });
    // Input Y
    graph.nodes.push_back({ {id++}, vectoria::ir::InputNode{"Y", {{2, 2}}, vectoria::ir::DataType::Float32} });
    
    // Add(X, Y)
    graph.nodes.push_back({ {id++}, vectoria::ir::OpNode{
        vectoria::ir::OpType::Add, 
        {{0}, {1}}, 
        {{2, 2}}, 
        vectoria::ir::DataType::Float32
    }});
    
    graph.outputs = {{2}};
    
    std::string out_path = "test_model.mlpackage";
    if (fs::exists(out_path)) fs::remove_all(out_path);
    
    vectoria::lowering::export_to_coreml(graph, out_path);
    
    if (!fs::exists(out_path + "/Data/com.apple.CoreML/model.mil")) {
        std::cerr << "Failed to generate model.mil" << std::endl;
        exit(1);
    }
    
    std::cout << "PASSED" << std::endl;
    fs::remove_all(out_path);
}

int main() {
    test_simple_export();
    return 0;
}
