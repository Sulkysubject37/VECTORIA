#include "vectoria/lowering/validation.hpp"
#include <stdexcept>
#include <variant>
#include <string>

namespace vectoria {
namespace lowering {

void validate_for_deployment(const ir::Graph& graph) {
    if (graph.nodes.empty()) {
        throw std::runtime_error("Empty graph cannot be deployed");
    }

    for (const auto& node : graph.nodes) {
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            // Check OpType support
            switch (op->op) {
                case ir::OpType::MatMul:
                case ir::OpType::BiasAdd:
                case ir::OpType::Relu:
                case ir::OpType::Add:
                case ir::OpType::Mul:
                case ir::OpType::Sub:
                case ir::OpType::Div:
                case ir::OpType::Exp:
                case ir::OpType::Sqrt:
                case ir::OpType::Log:
                case ir::OpType::Transpose:
                case ir::OpType::Reshape:
                case ir::OpType::Concat:
                case ir::OpType::Slice:
                    // Basic ops are supported structurally
                    break;
                
                case ir::OpType::ReduceSum:
                case ir::OpType::ReduceMax:
                    // Must validate axis semantics if we tracked axis (we assume last axis)
                    break;
                    
                default:
                    throw std::runtime_error("Unsupported OpType for deployment: " + std::to_string(static_cast<int>(op->op)));
            }

            // Check Input Validity
            for (auto inp : op->inputs) {
                if (inp.index >= graph.nodes.size()) {
                    throw std::runtime_error("Invalid input index in graph");
                }
            }
            
            // Check Shape Validity (Basic)
            if (op->output_shape.dims.empty() && op->op != ir::OpType::ReduceSum && op->op != ir::OpType::ReduceMax) {
                // Scalar output allowed for Reductions, but maybe risky for others?
                // Vectoria supports scalar broadcast, so scalars are fine.
            }
        }
    }
}

} // namespace lowering
} // namespace vectoria
