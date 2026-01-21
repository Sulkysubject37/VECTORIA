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

void test_structural_export() {
    std::cout << "Testing CoreML Export (Transpose/Reshape)..." << std::endl;
    
    vectoria::ir::Graph graph;
    size_t id = 0;
    
    // Input X [2, 3]
    graph.nodes.push_back({ {id++}, vectoria::ir::InputNode{"X", {{2, 3}}, vectoria::ir::DataType::Float32} });
    
    // Transpose [2, 3] -> [3, 2] (perm 1, 0)
    vectoria::ir::OpNode t_node;
    t_node.op = vectoria::ir::OpType::Transpose;
    t_node.inputs = {{0}};
    t_node.output_shape.dims = {3, 2};
    t_node.output_dtype = vectoria::ir::DataType::Float32;
    t_node.int_params = {1, 0};
    graph.nodes.push_back({ {id++}, t_node });
    
    // Reshape [3, 2] -> [6]
    vectoria::ir::OpNode r_node;
    r_node.op = vectoria::ir::OpType::Reshape;
    r_node.inputs = {{1}};
    r_node.output_shape.dims = {6};
    r_node.output_dtype = vectoria::ir::DataType::Float32;
    graph.nodes.push_back({ {id++}, r_node });
    
    graph.outputs = {{2}};
    
    std::string out_path = "test_structural.mlpackage";
    if (fs::exists(out_path)) fs::remove_all(out_path);
    
    vectoria::lowering::export_to_coreml(graph, out_path);
    
    std::ifstream mil_file(out_path + "/Data/com.apple.CoreML/model.mil");
    std::string content((std::istreambuf_iterator<char>(mil_file)), std::istreambuf_iterator<char>());
    
    if (content.find("transpose") == std::string::npos) {
        std::cerr << "MIL output missing transpose" << std::endl;
        exit(1);
    }
    if (content.find("reshape") == std::string::npos) {
        std::cerr << "MIL output missing reshape" << std::endl;
        exit(1);
    }
    
    std::cout << "PASSED" << std::endl;
    fs::remove_all(out_path);
}

int main() {
    test_simple_export();
    test_structural_export();
    return 0;
}
