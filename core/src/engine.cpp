#include "vectoria/engine.hpp"
#include "vectoria/kernels.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <numeric>

namespace vectoria {

Engine::Engine(const ir::Graph& graph) : graph_(graph) {}

bool Engine::validate() const {
    // Basic validation: ensure all OpNode inputs refer to existing nodes
    // and that there are no obvious cycles (though topo-sort will catch them too).
    for (const auto& node : graph_.nodes) {
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            for (const auto& input_id : op->inputs) {
                if (input_id.index >= graph_.nodes.size()) {
                    return false;
                }
                // In a strictly ordered DAG, inputs must have lower indices 
                // if we assume construction order.
                if (input_id.index >= node.id.index) {
                    return false;
                }
            }
        }
    }
    return true;
}

size_t Engine::calculate_size_bytes(const ir::TensorShape& shape, ir::DataType dtype) const {
    if (shape.dims.empty()) return 0;
    size_t elements = 1;
    for (auto dim : shape.dims) {
        elements *= dim;
    }
    
    size_t bpe = 0;
    switch (dtype) {
        case ir::DataType::Float32: bpe = 4; break;
        case ir::DataType::Float16: bpe = 2; break;
        case ir::DataType::Int32:   bpe = 4; break;
        case ir::DataType::Int8:    bpe = 1; break;
    }
    return elements * bpe;
}

void* Engine::get_buffer(size_t node_idx) const {
    if (node_idx >= node_buffers_.size()) return nullptr;
    return node_buffers_[node_idx];
}

void Engine::compile() {
    if (!validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    schedule_.clear();
    // For now, we assume the IR construction order is a valid topological sort.
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        schedule_.push_back(i);
    }
    
    // Allocate memory
    arena_.reset();
    node_buffers_.resize(graph_.nodes.size());

    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        const auto& node = graph_.nodes[i];
        
        ir::TensorShape shape;
        ir::DataType dtype;

        if (auto* input = std::get_if<ir::InputNode>(&node.data)) {
            shape = input->shape;
            dtype = input->dtype;
        } else if (auto* param = std::get_if<ir::ParameterNode>(&node.data)) {
            shape = param->shape;
            dtype = param->dtype;
        } else if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            shape = op->output_shape;
            dtype = op->output_dtype;
        } else {
             // Should not happen if variant is exhaustive
             continue; 
        }

        size_t size = calculate_size_bytes(shape, dtype);
        // Align to 64 bytes for AVX-512 future proofing
        // 64 byte alignment is safe for all standard types
        node_buffers_[i] = arena_.allocate(size, 64);
    }

    compiled_ = true;
}

void Engine::execute() {
    if (!compiled_) {
        throw std::runtime_error("Engine must be compiled before execution");
    }

    // Helper to get shape from node
    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph_.nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    for (size_t node_idx : schedule_) {
        const auto& node = graph_.nodes[node_idx];
        
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            if (op->op == ir::OpType::MatMul) {
                if (op->inputs.size() != 2) throw std::runtime_error("MatMul requires 2 inputs");
                
                size_t input_a_idx = op->inputs[0].index;
                size_t input_b_idx = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[input_a_idx]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[input_b_idx]);
                float* c_ptr = static_cast<float*>(node_buffers_[node_idx]);

                ir::TensorShape shape_a = get_shape(input_a_idx);
                ir::TensorShape shape_b = get_shape(input_b_idx);
                // ir::TensorShape shape_c = op->output_shape; // Unused for now

                // Assume 2D shapes for now
                if (shape_a.dims.size() != 2 || shape_b.dims.size() != 2) {
                     throw std::runtime_error("MatMul supports only 2D tensors for now");
                }

                size_t m = shape_a.dims[0];
                size_t k = shape_a.dims[1];
                size_t n = shape_b.dims[1];
                
                // Validate K match
                if (shape_b.dims[0] != static_cast<int64_t>(k)) {
                     throw std::runtime_error("MatMul dimension mismatch");
                }

                // Explicitly dispatch to Reference Kernel
                kernels::reference::gemm_f32(
                    a_ptr, b_ptr, c_ptr,
                    m, n, k,
                    k, n, n, // lda=K, ldb=N, ldc=N (Row Major)
                    1.0f, 0.0f
                );
            }
            // Add other ops (Relu, etc.) later
        }
    }
}

} // namespace vectoria
