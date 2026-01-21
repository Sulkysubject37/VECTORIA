#include "vectoria/lowering/coreml.hpp"
#include "vectoria/lowering/validation.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>

namespace fs = std::filesystem;

namespace vectoria {
namespace lowering {

std::string dtype_to_mil(ir::DataType dt) {
    switch(dt) {
        case ir::DataType::Float32: return "fp32";
        case ir::DataType::Float16: return "fp16";
        case ir::DataType::Int32:   return "int32";
        default: return "fp32";
    }
}

std::string shape_to_mil(const ir::TensorShape& shape) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.dims.size(); ++i) {
        ss << shape.dims[i];
        if (i < shape.dims.size() - 1) ss << ", ";
    }
    ss << ")";
    return ss.str();
}

void export_to_coreml(const ir::Graph& graph, const std::string& output_path) {
    // Validate first
    validate_for_deployment(graph);

    fs::path package_path(output_path);
    fs::path data_path = package_path / "Data";
    fs::path mil_path = data_path / "com.apple.CoreML";
    
    fs::create_directories(mil_path);
    
    std::ofstream mil_file(mil_path / "model.mil");
    if (!mil_file.is_open()) {
        throw std::runtime_error("Failed to open model.mil for writing");
    }

    mil_file << "graph main(\n";
    
    // Inputs
    bool first = true;
    for (const auto& node : graph.nodes) {
        if (auto* input = std::get_if<ir::InputNode>(&node.data)) {
            if (!first) mil_file << ",\n";
            mil_file << "    " << input->name << ": tensor<" 
                     << dtype_to_mil(input->dtype) << ", " 
                     << shape_to_mil(input->shape) << ">";
            first = false;
        }
    }
    mil_file << ") {\n";
    
    // Ops
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];
        std::string node_name = "n" + std::to_string(i); // Internal name
        
        if (std::get_if<ir::InputNode>(&node.data)) continue; // Handled
        
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            std::vector<std::string> inputs;
            for(auto inp : op->inputs) {
                // If input is InputNode, use its name. Else use n{id}.
                const auto& src = graph.nodes[inp.index];
                if (auto* in_node = std::get_if<ir::InputNode>(&src.data)) {
                    inputs.push_back(in_node->name);
                } else {
                    inputs.push_back("n" + std::to_string(inp.index));
                }
            }

            mil_file << "  " << node_name << " = ";
            
            switch (op->op) {
                case ir::OpType::Add:
                    mil_file << "add(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                case ir::OpType::Mul:
                    mil_file << "mul(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                case ir::OpType::Sub:
                    mil_file << "sub(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                case ir::OpType::Div:
                    mil_file << "real_div(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                case ir::OpType::Relu:
                    mil_file << "relu(x=" << inputs[0] << ");\n";
                    break;
                case ir::OpType::MatMul:
                    // CoreML linear/matmul expects specific args
                    // Simple matmul: x, y
                    mil_file << "matmul(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                case ir::OpType::ReduceSum:
                    mil_file << "reduce_sum(x=" << inputs[0] << ", axes=[-1], keep_dims=false);\n";
                    break;
                case ir::OpType::ReduceMax:
                    mil_file << "reduce_max(x=" << inputs[0] << ", axes=[-1], keep_dims=false);\n";
                    break;
                case ir::OpType::Exp:
                    mil_file << "exp(x=" << inputs[0] << ");\n";
                    break;
                case ir::OpType::Sqrt:
                    mil_file << "sqrt(x=" << inputs[0] << ");\n";
                    break;
                case ir::OpType::Log:
                    mil_file << "log(x=" << inputs[0] << ");\n";
                    break;
                case ir::OpType::Transpose:
                    {
                        mil_file << "transpose(x=" << inputs[0] << ", perm=[";
                        for (size_t p = 0; p < op->int_params.size(); ++p) {
                            mil_file << op->int_params[p];
                            if (p < op->int_params.size() - 1) mil_file << ", ";
                        }
                        mil_file << "]);\n";
                    }
                    break;
                case ir::OpType::Reshape:
                    {
                        mil_file << "reshape(x=" << inputs[0] << ", shape=[";
                        for (size_t s = 0; s < op->output_shape.dims.size(); ++s) {
                            mil_file << op->output_shape.dims[s];
                            if (s < op->output_shape.dims.size() - 1) mil_file << ", ";
                        }
                        mil_file << "]);\n";
                    }
                    break;
                case ir::OpType::BiasAdd:
                    // Map to add
                    mil_file << "add(x=" << inputs[0] << ", y=" << inputs[1] << ");\n";
                    break;
                default:
                    throw std::runtime_error("Unsupported op for CoreML export");
            }
        }
    }
    
    // Outputs
    // We assume the last node or explicitly marked outputs.
    // For now, let's return the last computed node(s).
    // Vectoria graph.outputs stores IDs.
    
    if (!graph.outputs.empty()) {
        mil_file << "  return (";
        for (size_t i = 0; i < graph.outputs.size(); ++i) {
            size_t oid = graph.outputs[i].index;
            mil_file << "n" << oid; // Use internal name
            if (i < graph.outputs.size() - 1) mil_file << ", ";
        }
        mil_file << ");\n";
    }
    
    mil_file << "}\n";
    mil_file.close();
}

} // namespace lowering
} // namespace vectoria
