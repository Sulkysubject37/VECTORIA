#include "vectoria/engine.hpp"
#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <numeric>

namespace vectoria {

Engine::Engine(const ir::Graph& graph, EngineConfig config) 
    : graph_(graph), config_(config) {}

bool Engine::validate() const {
    for (const auto& node : graph_.nodes) {
        if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            for (const auto& input_id : op->inputs) {
                if (input_id.index >= graph_.nodes.size()) {
                    return false;
                }
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
    tracer_.clear();
    tracer_.log(trace::EventType::GraphCompilation, -1, "Start");

    if (!validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    schedule_.clear();
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
        schedule_.push_back(i);
    }
    
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
             continue; 
        }

        size_t size = calculate_size_bytes(shape, dtype);
        node_buffers_[i] = arena_.allocate(size, 64);
        tracer_.log(trace::EventType::MemoryAllocation, i, std::to_string(size) + " bytes");
    }

    compiled_ = true;
    tracer_.log(trace::EventType::GraphCompilation, -1, "End");
}

void Engine::execute() {
    if (!compiled_) {
        throw std::runtime_error("Engine must be compiled before execution");
    }

    auto get_shape = [&](size_t idx) -> ir::TensorShape {
        const auto& n = graph_.nodes[idx];
        if (auto* i = std::get_if<ir::InputNode>(&n.data)) return i->shape;
        if (auto* p = std::get_if<ir::ParameterNode>(&n.data)) return p->shape;
        if (auto* o = std::get_if<ir::OpNode>(&n.data)) return o->output_shape;
        return {};
    };

    for (size_t node_idx : schedule_) {
        const auto& node = graph_.nodes[node_idx];
        tracer_.log(trace::EventType::NodeExecutionStart, node_idx);
        
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

                if (shape_a.dims.size() != 2 || shape_b.dims.size() != 2) {
                     throw std::runtime_error("MatMul supports only 2D tensors for now");
                }

                size_t m = shape_a.dims[0];
                size_t k = shape_a.dims[1];
                size_t n = shape_b.dims[1];
                
                if (shape_b.dims[0] != static_cast<int64_t>(k)) {
                     throw std::runtime_error("MatMul dimension mismatch");
                }

                bool executed = false;

                if (config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    VectoriaStatus status = gemm_f32_neon(
                        a_ptr, b_ptr, c_ptr, m, n, k, k, n, n, 1.0f, 0.0f
                    );
                    if (status != VECTORIA_SUCCESS) throw std::runtime_error("ASM kernel failed");
                    executed = true;
    #elif defined(__x86_64__)
                    VectoriaStatus status = gemm_f32_avx2(
                        a_ptr, b_ptr, c_ptr, m, n, k, k, n, n, 1.0f, 0.0f
                    );
                    if (status != VECTORIA_SUCCESS) throw std::runtime_error("ASM kernel failed");
                    executed = true;
    #else
                    throw std::runtime_error("SIMD policy requested but architecture not supported");
    #endif
#else
                    throw std::runtime_error("SIMD policy requested but VECTORIA_USE_ASM not defined");
#endif
                }

                if (!executed) {
                    kernels::reference::gemm_f32(
                        a_ptr, b_ptr, c_ptr,
                        m, n, k,
                        k, n, n,
                        1.0f, 0.0f
                    );
                }
                
                std::string mode = executed ? "SIMD" : "Reference";
                #if defined(__aarch64__)
                    if (executed) mode += " [ARM64]";
                #elif defined(__x86_64__)
                    if (executed) mode += " [x86_64]";
                #endif
                mode += " | Inputs: [" + std::to_string(input_a_idx) + ", " + std::to_string(input_b_idx) + "]";
                
                tracer_.log(trace::EventType::KernelDispatch, node_idx, mode);
            }
            else if (op->op == ir::OpType::BiasAdd) {
                if (op->inputs.size() != 2) throw std::runtime_error("BiasAdd requires 2 inputs");
                size_t input_idx = op->inputs[0].index;
                size_t bias_idx = op->inputs[1].index;
                
                const float* in_ptr = static_cast<const float*>(node_buffers_[input_idx]);
                const float* bias_ptr = static_cast<const float*>(node_buffers_[bias_idx]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape shape_in = get_shape(input_idx);
                // Assume 2D [M, N]
                size_t m = shape_in.dims[0];
                size_t n = shape_in.dims[1];
                
                // Dispatch (Reference only for now)
                kernels::reference::bias_add_f32(in_ptr, bias_ptr, out_ptr, m, n);
                
                std::string mode = "Reference | Inputs: [" + std::to_string(input_idx) + ", " + std::to_string(bias_idx) + "]";
                tracer_.log(trace::EventType::KernelDispatch, node_idx, mode);
            }
            else if (op->op == ir::OpType::Relu) {
                if (op->inputs.size() != 1) throw std::runtime_error("Relu requires 1 input");
                size_t input_idx = op->inputs[0].index;
                
                const float* in_ptr = static_cast<const float*>(node_buffers_[input_idx]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape shape_in = get_shape(input_idx);
                size_t count = 1;
                for(auto d : shape_in.dims) count *= d;
                
                // Dispatch
                kernels::reference::relu_f32(in_ptr, out_ptr, count);
                
                std::string mode = "Reference | Inputs: [" + std::to_string(input_idx) + "]";
                tracer_.log(trace::EventType::KernelDispatch, node_idx, mode);
            }
            else if (op->op == ir::OpType::Add) {
                if (op->inputs.size() != 2) throw std::runtime_error("Add requires 2 inputs");
                size_t idx_a = op->inputs[0].index;
                size_t idx_b = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[idx_a]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[idx_b]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape s = get_shape(idx_a);
                size_t count = 1;
                for(auto d : s.dims) count *= d;
                
                kernels::reference::add_f32(a_ptr, b_ptr, out_ptr, count);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [" + std::to_string(idx_a) + ", " + std::to_string(idx_b) + "]");
            }
            else if (op->op == ir::OpType::Mul) {
                if (op->inputs.size() != 2) throw std::runtime_error("Mul requires 2 inputs");
                size_t idx_a = op->inputs[0].index;
                size_t idx_b = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[idx_a]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[idx_b]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape s = get_shape(idx_a);
                size_t count = 1;
                for(auto d : s.dims) count *= d;
                
                kernels::reference::mul_f32(a_ptr, b_ptr, out_ptr, count);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [" + std::to_string(idx_a) + ", " + std::to_string(idx_b) + "]");
            }
            else if (op->op == ir::OpType::ReduceSum) {
                if (op->inputs.size() != 1) throw std::runtime_error("ReduceSum requires 1 input");
                size_t idx_in = op->inputs[0].index;
                
                const float* in_ptr = static_cast<const float*>(node_buffers_[idx_in]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape s = get_shape(idx_in);
                if (s.dims.empty()) throw std::runtime_error("ReduceSum input must have at least 1 dim");
                
                size_t inner = s.dims.back();
                size_t outer = 1;
                for(size_t i=0; i<s.dims.size()-1; ++i) outer *= s.dims[i];
                
                kernels::reference::reduce_sum_f32(in_ptr, out_ptr, outer, inner);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [" + std::to_string(idx_in) + "]");
            }
        }
        tracer_.log(trace::EventType::NodeExecutionEnd, node_idx);
    }
}

} // namespace vectoria