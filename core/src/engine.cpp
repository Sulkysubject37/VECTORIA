#include "vectoria/engine.hpp"
#include "vectoria/kernels.hpp"
#include "vectoria/kernel_abi.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <numeric>
#include <cstring>

extern "C" {
#if defined(__aarch64__)
    VectoriaStatus add_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus mul_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus sub_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus div_f32_neon(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus relu_f32_neon(const float* in, float* out, size_t count);
    VectoriaStatus reduce_sum_f32_neon(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_max_f32_neon(const float* in, float* out, size_t outer, size_t inner);
#elif defined(__x86_64__)
    VectoriaStatus add_f32_avx2(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus mul_f32_avx2(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus sub_f32_avx2(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus div_f32_avx2(const float* a, const float* b, float* out, size_t count);
    VectoriaStatus relu_f32_avx2(const float* in, float* out, size_t count);
    VectoriaStatus reduce_sum_f32_avx2(const float* in, float* out, size_t outer, size_t inner);
    VectoriaStatus reduce_max_f32_avx2(const float* in, float* out, size_t outer, size_t inner);
#endif
}

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
    // Scalar has empty dims but 1 element
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
    std::string mode_str = (config_.mode == ExecutionMode::Deployment) ? "Deployment" : "Research";
    tracer_.log(trace::EventType::GraphCompilation, -1, "Start | Mode: " + mode_str);

    if (!validate()) {
        throw std::runtime_error("Graph validation failed");
    }

    if (config_.mode == ExecutionMode::Deployment) {
        // Strict check: Only supported ops allowed
        for (const auto& node : graph_.nodes) {
            if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
                // List of supported ops for CoreML lowering
                bool supported = false;
                switch (op->op) {
                    case ir::OpType::MatMul:
                    case ir::OpType::BiasAdd:
                    case ir::OpType::Relu:
                    case ir::OpType::Add:
                    case ir::OpType::Mul:
                    case ir::OpType::Sub:
                    case ir::OpType::Div:
                    case ir::OpType::ReduceSum:
                    case ir::OpType::ReduceMax:
                    case ir::OpType::Sqrt:
                    case ir::OpType::Log:
                        supported = true;
                        break;
                    case ir::OpType::Exp:
                        // Exp is supported in CoreML but Phase 6 spec didn't explicitly list it?
                        // "Supported Op Set: MatMul, Add, Sub, Mul, Div, ReduceSum, ReduceMax, ReLU, Softmax"
                        // Softmax uses Exp internally.
                        // If graph has Exp, can we export it?
                        // Phase 6 spec says "If any other op appears -> export MUST fail".
                        // BUT Softmax is composed of Exp.
                        // If Softmax is composed, the IR contains Exp.
                        // So Exp MUST be supported for Softmax to work.
                        // I will allow Exp.
                        supported = true;
                        break;
                    default:
                        supported = false;
                }
                
                if (!supported) {
                    tracer_.log(trace::EventType::GraphCompilation, node.id.index, "Unsupported Op for Deployment");
                    throw std::runtime_error("Op not supported in Deployment Mode: " + std::to_string(static_cast<int>(op->op)));
                }
            }
        }
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
        } else if (auto* c = std::get_if<ir::ConstantNode>(&node.data)) {
            shape = c->shape;
            dtype = c->dtype;
        } else if (auto* op = std::get_if<ir::OpNode>(&node.data)) {
            shape = op->output_shape;
            dtype = op->output_dtype;
        } else {
             continue; 
        }

        size_t size = calculate_size_bytes(shape, dtype);
        node_buffers_[i] = arena_.allocate(size, 64);
        tracer_.log(trace::EventType::MemoryAllocation, i, std::to_string(size) + " bytes");

        if (auto* c = std::get_if<ir::ConstantNode>(&node.data)) {
            // Initialize constant memory immediately
            if (dtype == ir::DataType::Float32 && !c->data_f32.empty()) {
                // Verify size match (basic check)
                if (c->data_f32.size() * sizeof(float) <= size) {
                     std::memcpy(node_buffers_[i], c->data_f32.data(), c->data_f32.size() * sizeof(float));
                }
            }
        }
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
        if (auto* c = std::get_if<ir::ConstantNode>(&n.data)) return c->shape;
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
                
                bool executed = false;
                if (config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (relu_f32_neon(in_ptr, out_ptr, count) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (relu_f32_avx2(in_ptr, out_ptr, count) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    kernels::reference::relu_f32(in_ptr, out_ptr, count);
                }
                
                std::string mode = executed ? "SIMD" : "Reference";
                #if defined(__aarch64__)
                    if (executed) mode += " [ARM64]";
                #elif defined(__x86_64__)
                    if (executed) mode += " [x86_64]";
                #endif
                mode += " | Inputs: [" + std::to_string(input_idx) + "]";
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
                size_t count_a = 1; for(auto d : get_shape(idx_a).dims) count_a *= d;
                size_t count_b = 1; for(auto d : get_shape(idx_b).dims) count_b *= d;
                
                bool executed = false;
                if (count_a == count_b && config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (add_f32_neon(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (add_f32_avx2(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    if (count_a == count_b) {
                        kernels::reference::add_f32(a_ptr, b_ptr, out_ptr, count_a);
                    } else {
                        // Broadcast logic matching Sub/Div
                        // Supports Col-vector broadcast (A[i,j] + B[i])
                        // Or Scalar broadcast (B[0]) if outer=1
                        
                        // Check divisibility
                        if (count_b == 0) throw std::runtime_error("Add broadcast div by zero");
                        if (count_a % count_b != 0) throw std::runtime_error("Add broadcast shape mismatch");

                        size_t outer = count_b;
                        size_t inner = count_a / count_b;
                        kernels::reference::add_broadcast_f32(a_ptr, b_ptr, out_ptr, outer, inner);
                    }
                }
                
                std::string mode = executed ? "SIMD" : "Reference";
                #if defined(__aarch64__)
                    if (executed) mode += " [ARM64]";
                #elif defined(__x86_64__)
                    if (executed) mode += " [x86_64]";
                #endif
                mode += " | Inputs: [" + std::to_string(idx_a) + ", " + std::to_string(idx_b) + "]";
                tracer_.log(trace::EventType::KernelDispatch, node_idx, mode);
            }
            else if (op->op == ir::OpType::Mul) {
                if (op->inputs.size() != 2) throw std::runtime_error("Mul requires 2 inputs");
                size_t idx_a = op->inputs[0].index;
                size_t idx_b = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[idx_a]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[idx_b]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape shape_a = get_shape(idx_a);
                ir::TensorShape shape_b = get_shape(idx_b);
                
                size_t count_a = 1; for(auto d : shape_a.dims) count_a *= d;
                size_t count_b = 1; for(auto d : shape_b.dims) count_b *= d;
                
                bool executed = false;
                if (count_a == count_b && config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (mul_f32_neon(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (mul_f32_avx2(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    if (count_a == count_b) {
                        kernels::reference::mul_f32(a_ptr, b_ptr, out_ptr, count_a);
                    } else {
                         // Attempt Broadcast: A [Outer, Inner] * B [Inner]
                         if (shape_a.dims.empty() || shape_b.dims.empty()) throw std::runtime_error("Mul broadcast requires rank >= 1");
                         
                         size_t inner_a = shape_a.dims.back();
                         size_t outer_a = count_a / inner_a;
                         
                         // We support B being [Inner] (rank 1) or [1, Inner] (rank 2) but effectively matching last dim
                         // Check total elements of B equals inner dimension of A
                         if (count_b == inner_a) {
                             kernels::reference::mul_broadcast_f32(a_ptr, b_ptr, out_ptr, outer_a, inner_a);
                         } else {
                             throw std::runtime_error("Mul broadcast shape mismatch: Expected B size " + std::to_string(inner_a) + " but got " + std::to_string(count_b));
                         }
                    }
                }
                tracer_.log(trace::EventType::KernelDispatch, node_idx, (executed ? "SIMD" : "Reference") + std::string(" | Inputs: [...]"));
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
                
                bool executed = false;
                if (config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (reduce_sum_f32_neon(in_ptr, out_ptr, outer, inner) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (reduce_sum_f32_avx2(in_ptr, out_ptr, outer, inner) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    kernels::reference::reduce_sum_f32(in_ptr, out_ptr, outer, inner);
                }
                tracer_.log(trace::EventType::KernelDispatch, node_idx, (executed ? "SIMD" : "Reference") + std::string(" | Inputs: [...]"));
            }
            else if (op->op == ir::OpType::ReduceMax) {
                if (op->inputs.size() != 1) throw std::runtime_error("ReduceMax requires 1 input");
                size_t idx_in = op->inputs[0].index;
                
                const float* in_ptr = static_cast<const float*>(node_buffers_[idx_in]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape s = get_shape(idx_in);
                if (s.dims.empty()) throw std::runtime_error("ReduceMax input must have at least 1 dim");
                
                size_t inner = s.dims.back();
                size_t outer = 1;
                for(size_t i=0; i<s.dims.size()-1; ++i) outer *= s.dims[i];
                
                bool executed = false;
                if (config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (reduce_max_f32_neon(in_ptr, out_ptr, outer, inner) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (reduce_max_f32_avx2(in_ptr, out_ptr, outer, inner) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    kernels::reference::reduce_max_f32(in_ptr, out_ptr, outer, inner);
                }
                tracer_.log(trace::EventType::KernelDispatch, node_idx, (executed ? "SIMD" : "Reference") + std::string(" | Inputs: [...]"));
            }
            else if (op->op == ir::OpType::Exp) {
                // Exp remains reference only
                size_t idx_in = op->inputs[0].index;
                const float* in_ptr = static_cast<const float*>(node_buffers_[idx_in]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                ir::TensorShape s = get_shape(idx_in);
                size_t count = 1; for(auto d : s.dims) count *= d;
                kernels::reference::exp_f32(in_ptr, out_ptr, count);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [...]");
            }
            else if (op->op == ir::OpType::Sqrt) {
                if (op->inputs.size() != 1) throw std::runtime_error("Sqrt requires 1 input");
                size_t idx_in = op->inputs[0].index;
                const float* in_ptr = static_cast<const float*>(node_buffers_[idx_in]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                ir::TensorShape s = get_shape(idx_in);
                size_t count = 1; for(auto d : s.dims) count *= d;
                kernels::reference::sqrt_f32(in_ptr, out_ptr, count);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [...]");
            }
            else if (op->op == ir::OpType::Log) {
                if (op->inputs.size() != 1) throw std::runtime_error("Log requires 1 input");
                size_t idx_in = op->inputs[0].index;
                const float* in_ptr = static_cast<const float*>(node_buffers_[idx_in]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                ir::TensorShape s = get_shape(idx_in);
                size_t count = 1; for(auto d : s.dims) count *= d;
                kernels::reference::log_f32(in_ptr, out_ptr, count);
                tracer_.log(trace::EventType::KernelDispatch, node_idx, "Reference | Inputs: [...]");
            }
            else if (op->op == ir::OpType::Sub) {
                if (op->inputs.size() != 2) throw std::runtime_error("Sub requires 2 inputs");
                size_t idx_a = op->inputs[0].index;
                size_t idx_b = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[idx_a]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[idx_b]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape shape_a = get_shape(idx_a);
                ir::TensorShape shape_b = get_shape(idx_b);
                
                size_t count_a = 1; for(auto d : shape_a.dims) count_a *= d;
                size_t count_b = 1; for(auto d : shape_b.dims) count_b *= d;
                
                bool executed = false;
                if (count_a == count_b && config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (sub_f32_neon(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (sub_f32_avx2(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    if (count_a == count_b) {
                        kernels::reference::sub_f32(a_ptr, b_ptr, out_ptr, count_a, count_b);
                    } else {
                        size_t outer = count_b;
                        size_t inner = count_a / count_b;
                        kernels::reference::sub_broadcast_f32(a_ptr, b_ptr, out_ptr, outer, inner);
                    }
                }
                tracer_.log(trace::EventType::KernelDispatch, node_idx, (executed ? "SIMD" : "Reference") + std::string(" | Inputs: [...]"));
            }
            else if (op->op == ir::OpType::Div) {
                if (op->inputs.size() != 2) throw std::runtime_error("Div requires 2 inputs");
                size_t idx_a = op->inputs[0].index;
                size_t idx_b = op->inputs[1].index;
                
                const float* a_ptr = static_cast<const float*>(node_buffers_[idx_a]);
                const float* b_ptr = static_cast<const float*>(node_buffers_[idx_b]);
                float* out_ptr = static_cast<float*>(node_buffers_[node_idx]);
                
                ir::TensorShape shape_a = get_shape(idx_a);
                ir::TensorShape shape_b = get_shape(idx_b);
                
                size_t count_a = 1; for(auto d : shape_a.dims) count_a *= d;
                size_t count_b = 1; for(auto d : shape_b.dims) count_b *= d;
                
                bool executed = false;
                if (count_a == count_b && config_.policy == KernelPolicy::SIMD) {
#ifdef VECTORIA_USE_ASM
    #if defined(__aarch64__)
                    if (div_f32_neon(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #elif defined(__x86_64__)
                    if (div_f32_avx2(a_ptr, b_ptr, out_ptr, count_a) == VECTORIA_SUCCESS) executed = true;
    #endif
#endif
                }

                if (!executed) {
                    if (count_a == count_b) {
                        kernels::reference::div_f32(a_ptr, b_ptr, out_ptr, count_a, count_b);
                    } else {
                        size_t outer = count_b;
                        size_t inner = count_a / count_b;
                        kernels::reference::div_broadcast_f32(a_ptr, b_ptr, out_ptr, outer, inner);
                    }
                }
                tracer_.log(trace::EventType::KernelDispatch, node_idx, (executed ? "SIMD" : "Reference") + std::string(" | Inputs: [...]"));
            }
        }
        tracer_.log(trace::EventType::NodeExecutionEnd, node_idx);
    }
}

} // namespace vectoria